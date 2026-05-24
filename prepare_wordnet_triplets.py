"""
prepare_wordnet_triplets.py — extract subsumption + meronymy triplets
from WordNet (Miller 1995) for LoRA fine-tuning.

Output JSONL schema, one record per line:
  {
    "anchor_synset":   "dog.n.01",
    "anchor_text":     "dog. a domesticated mammal …",
    "positive_synset": "canine.n.02",
    "positive_text":   "canine. any of various fissiped …",
    "negative_synset": "cat.n.01" | null,
    "negative_text":   "cat. a small carnivorous mammal …" | null,
    "instruction_type": "broader" | "narrower"
  }

Triplet sources:
  - hypernyms (forward subsumption)  -> instruction=broader
  - hyponyms  (inverse subsumption)  -> instruction=narrower
  - meronyms  (forward part-of)       -> instruction=broader
  - holonyms  (inverse part-of)       -> instruction=narrower

Hard negatives: cohyponyms of the anchor (siblings under the same
hypernym, anchor itself excluded). Falls back to a random WordNet
noun synset if no cohyponym exists. Hard negative is optional —
MultipleNegativesRankingLoss has in-batch negatives anyway, but the
explicit hard negative tightens the contrast on tough cases.

Anchor / positive / negative TEXT representation: "<label>. <definition>"
per the 2026-05-04 methodology decision (option b — middle ground
between bare labels and pseudo-Turtle, matches rdfs:comment register).

Synset split for train/val: 95/5 disjoint over noun synsets — NOT a
random row split, so train and val share no overlapping anchor synset.
A triplet is dropped when its anchor and positive end up on opposite
sides of the synset split (cross-split leakage).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger("prepare_wordnet_triplets")


def _label_for(synset) -> str:
    """First lemma, underscores to spaces. Stable across NLTK versions."""
    lemmas = synset.lemmas()
    if not lemmas:
        return synset.name().split(".")[0].replace("_", " ")
    return lemmas[0].name().replace("_", " ")


def _text_for(synset) -> str:
    """Anchor / positive / negative representation: "<label>. <definition>".
    Definition is 1-2 sentences in WordNet. Empty definition collapses to
    just the label, kept as fallback so the embedder never sees an empty
    string.
    """
    label = _label_for(synset)
    defn = (synset.definition() or "").strip()
    return f"{label}. {defn}" if defn else label


def _cohyponym_or_random(anchor, all_noun_synsets, rng) -> object | None:
    """Sample a sibling under one of the anchor's hypernyms; fall back to
    a random noun synset when the anchor is a root (no hypernyms) or all
    siblings are the anchor itself.
    """
    parents = anchor.hypernyms()
    if parents:
        rng.shuffle(parents)
        for parent in parents:
            siblings = [s for s in parent.hyponyms() if s != anchor]
            if siblings:
                return rng.choice(siblings)
    # Root or singleton-sibling case.
    if all_noun_synsets:
        for _ in range(8):
            cand = rng.choice(all_noun_synsets)
            if cand != anchor:
                return cand
    return None


def _emit_triplets(anchor, positives, instruction_type, rng, all_noun_synsets):
    """Yield triplet dicts for one (anchor, [positives], instruction_type)
    grouping. Each positive becomes one triplet with its own hard
    negative (cohyponym sample).
    """
    for pos in positives:
        if pos == anchor:
            continue
        neg = _cohyponym_or_random(anchor, all_noun_synsets, rng)
        # Avoid degenerate negatives that match the positive.
        attempts = 0
        while neg is not None and neg == pos and attempts < 5:
            neg = _cohyponym_or_random(anchor, all_noun_synsets, rng)
            attempts += 1
        yield {
            "anchor_synset":   anchor.name(),
            "anchor_text":     _text_for(anchor),
            "positive_synset": pos.name(),
            "positive_text":   _text_for(pos),
            "negative_synset": neg.name() if neg is not None else None,
            "negative_text":   _text_for(neg) if neg is not None else None,
            "instruction_type": instruction_type,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Build WordNet triplets for LoRA training.")
    p.add_argument("--output-path", default="data/wordnet_triplets.jsonl",
                   help="Output JSONL file (parent dir auto-created).")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Fraction of synsets reserved for validation. "
                        "Synset-disjoint split; cross-split triplets dropped.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-synsets", type=int, default=None,
                   help="Optional cap on noun-synset count for smoke testing.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    # NLTK setup — download wordnet if missing.
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        sys.exit("nltk not installed. Run: conda run -n melt-olala python -m pip install nltk")

    try:
        wn.synsets("dog")
    except LookupError:
        logger.info("Downloading WordNet (one-off)…")
        nltk.download("wordnet")
        # Reload after download.
        from nltk.corpus import wordnet as wn  # noqa: F811

    rng = random.Random(args.seed)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Synset universe (nouns only). ──────────────────────────────────
    all_noun_synsets = list(wn.all_synsets(pos="n"))
    if args.limit_synsets is not None:
        rng.shuffle(all_noun_synsets)
        all_noun_synsets = all_noun_synsets[: args.limit_synsets]
    logger.info("Noun synsets: %d", len(all_noun_synsets))

    # Synset-disjoint split.
    shuffled = list(all_noun_synsets)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.val_fraction))
    val_set = set(s.name() for s in shuffled[:n_val])
    train_set = set(s.name() for s in shuffled[n_val:])
    logger.info("Split: train=%d val=%d (val_fraction=%.3f)",
                len(train_set), len(val_set), args.val_fraction)

    # ── Triplet emission. ──────────────────────────────────────────────
    counts: Counter = Counter()
    n_with_neg = 0
    n_without_neg = 0
    n_cross_split_dropped = 0
    examples: dict[str, dict] = {}

    train_path = out_path.with_suffix(".train.jsonl")
    val_path   = out_path.with_suffix(".val.jsonl")

    with train_path.open("w") as f_train, val_path.open("w") as f_val:
        for anchor in all_noun_synsets:
            anchor_name = anchor.name()
            anchor_split = "train" if anchor_name in train_set else "val"

            buckets = [
                ("broader",  anchor.hypernyms(),                   "hypernyms"),
                ("narrower", anchor.hyponyms(),                    "hyponyms"),
                ("broader",  anchor.part_meronyms() + anchor.member_meronyms() + anchor.substance_meronyms(),  "meronyms"),
                ("narrower", anchor.part_holonyms() + anchor.member_holonyms() + anchor.substance_holonyms(),   "holonyms"),
            ]
            for instr, positives, source_label in buckets:
                for trip in _emit_triplets(anchor, positives, instr, rng, all_noun_synsets):
                    pos_split = "train" if trip["positive_synset"] in train_set else "val"
                    if anchor_split != pos_split:
                        n_cross_split_dropped += 1
                        continue
                    if trip["negative_synset"] is not None:
                        n_with_neg += 1
                    else:
                        n_without_neg += 1
                    counts[(instr, source_label)] += 1
                    bucket_key = f"{source_label}/{instr}"
                    if bucket_key not in examples:
                        examples[bucket_key] = trip
                    target = f_train if anchor_split == "train" else f_val
                    target.write(json.dumps(trip, ensure_ascii=False) + "\n")

    # Single combined file too — convenience for downstream code that
    # doesn't need the split (e.g. exploratory greps).
    with out_path.open("w") as f_all:
        for src in (train_path, val_path):
            f_all.write(src.read_text())

    # ── Statistics. ────────────────────────────────────────────────────
    total = sum(counts.values())
    logger.info("Triplets emitted: %d", total)
    for (instr, source_label), n in sorted(counts.items()):
        logger.info("  %s (%s): %d", instr, source_label, n)
    by_instr = Counter()
    for (instr, _src), n in counts.items():
        by_instr[instr] += n
    logger.info("By instruction-type: broader=%d narrower=%d", by_instr["broader"], by_instr["narrower"])
    logger.info("With hard negative: %d   without: %d   (%.1f%% have neg)",
                n_with_neg, n_without_neg,
                100 * n_with_neg / total if total else 0.0)
    logger.info("Cross-split triplets dropped: %d", n_cross_split_dropped)
    logger.info("Files: train=%s val=%s combined=%s", train_path, val_path, out_path)

    print()
    print("Example triplet per category:")
    for k, t in sorted(examples.items()):
        print(f"  [{k}]")
        print(f"    anchor:   {t['anchor_text'][:120]}")
        print(f"    positive: {t['positive_text'][:120]}")
        if t["negative_text"]:
            print(f"    negative: {t['negative_text'][:120]}")
        print(f"    instruction_type: {t['instruction_type']}")


if __name__ == "__main__":
    main()
