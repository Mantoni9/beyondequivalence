#!/usr/bin/env python3
"""e17_verify.py — Verification second-pass runner (E17).

Skeptical binary verification over an ALREADY-asserted set: for each kept
'<'/'>'/'=' assertion in a matrix cell's predictions.tsv, ask the served model
whether the claimed relation actually holds (P(yes) vs P(no), first-token
logprob). Additive: does not touch Stage-1/Stage-2 or the matrix cells.

Needs a running vLLM server (VLLM_BASE_URL). Output:
  e17_verify_<model>_<dataset>[<tag>].tsv  (source, target, rel, p_yes)

Usage (inside the serve job, after the server is healthy):
  python scripts/e17_verify.py --model <name> --model-path <hf/id> \
      --dataset g1-web --assertions <predictions.tsv> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdflib import URIRef                       # noqa: E402
from RDFGraphWrapper import RDFGraphWrapper      # noqa: E402
from LLMOpenAI import LLMOpenAI                  # noqa: E402
from prompt import get_reranking_prompt          # noqa: E402
from tracks.zenodo_loader import load_subdataset  # noqa: E402

CLAIMS = {
    "<": "the source entity is a MORE SPECIFIC kind of the target entity (source is a subclass of the target).",
    ">": "the source entity is a MORE GENERAL kind of the target entity (source is a superclass of the target).",
    "=": "the source and target entities denote the SAME concept (they are equivalent).",
}
# reverse relation for direction symmetrization (a false '<' often reads yes both ways)
REV = {"<": ">", ">": "<", "=": None}


def parse_verdict(text: str) -> float:
    """p_yes from a reasoning generation ending in 'VERDICT: Yes|No'."""
    import re
    t = (text or "").lower()
    m = re.findall(r"verdict\s*[:\-]?\s*(yes|no)", t)
    if m:
        return 1.0 if m[-1] == "yes" else 0.0
    if "\nyes" in t or t.strip().endswith("yes"):
        return 1.0
    if "\nno" in t or t.strip().endswith("no"):
        return 0.0
    return 0.5


def dataset_paths(dataset: str, repo: Path):
    if dataset == "vdi-ebay":
        return (repo / "goldstandard_ebay" / "vdi" / "vdi_karosserie_source_pos.owl",
                repo / "goldstandard_ebay" / "ebay_kfz_target.owl")
    s, t, _r = load_subdataset(dataset)
    return Path(s), Path(t)


def read_assertions(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("kept") == "True" and r.get("predicted_relation") in CLAIMS:
                out.append((r["source_uri"], r["target_uri"], r["predicted_relation"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--assertions", required=True)
    ap.add_argument("--out", default="results/e17")
    ap.add_argument("--tag", default="")
    ap.add_argument("--description", default="description_path_context")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-mode", choices=("firsttoken", "reasoning"), default="firsttoken",
                    help="firsttoken: P(yes) logprob (non-reasoning). reasoning: generate CoT, parse VERDICT (gpt-oss).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--reverse", action="store_true", help="also score the reverse-relation claim (symmetrization)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        sys.exit("VLLM_BASE_URL not set — serve the model first")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # local vLLM ignores the value

    asserts = read_assertions(Path(args.assertions))
    if args.limit:
        asserts = asserts[:args.limit]
    print(f"[e17] {args.model}/{args.dataset}: {len(asserts)} assertions to verify", flush=True)

    spath, tpath = dataset_paths(args.dataset, repo)
    kg_s = RDFGraphWrapper(str(spath)); kg_t = RDFGraphWrapper(str(tpath))
    desc_s = getattr(kg_s, args.description); desc_t = getattr(kg_t, args.description)
    tmpl = get_reranking_prompt("d_subs_verify_cot" if args.verify_mode == "reasoning" else "d_subs_verify")
    llm = LLMOpenAI(model_name=args.model_path, base_url=base_url,
                    max_concurrency=args.max_concurrency)

    # cache descriptions (many assertions reuse the same source/target)
    dcache: dict[tuple[str, str], str] = {}

    def dtext(kg_desc, uri):
        k = (id(kg_desc), uri)
        if k not in dcache:
            try:
                r = kg_desc(URIRef(uri)); dcache[k] = r if isinstance(r, str) else str(r)
            except Exception:
                dcache[k] = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        return dcache[k]

    def score(prompts):
        if args.verify_mode == "firsttoken":
            return llm.get_confidence_first_token(prompts)
        res = llm.get_text_completion_with_logprobs(
            prompts, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p)
        return [parse_verdict(r.get("text", "")) for r in res]

    rows, t0 = [], time.time()
    for start in range(0, len(asserts), args.batch_size):
        chunk = asserts[start:start + args.batch_size]
        fwd = [tmpl.format(source_url=s, target_url=t, source_kg=dtext(desc_s, s),
                           target_kg=dtext(desc_t, t), claim=CLAIMS[rel]) for (s, t, rel) in chunk]
        p_fwd = score(fwd)
        p_rev = [None] * len(chunk)
        if args.reverse:
            ridx = [i for i, (_, _, rel) in enumerate(chunk) if REV[rel] is not None]
            rprompts = [tmpl.format(source_url=chunk[i][0], target_url=chunk[i][1],
                                    source_kg=dtext(desc_s, chunk[i][0]), target_kg=dtext(desc_t, chunk[i][1]),
                                    claim=CLAIMS[REV[chunk[i][2]]]) for i in ridx]
            rscores = score(rprompts) if rprompts else []
            for i, p in zip(ridx, rscores):
                p_rev[i] = p
        for (s, t, rel), pf, pr in zip(chunk, p_fwd, p_rev):
            row = {"source_uri": s, "target_uri": t, "rel": rel, "p_yes": round(pf, 6)}
            if args.reverse:
                row["p_yes_rev"] = round(pr, 6) if pr is not None else ""
            rows.append(row)
        done = start + len(chunk)
        if done % (args.batch_size * 4) == 0 or done == len(asserts):
            print(f"[e17]   {done}/{len(asserts)}  ({(time.time()-t0):.0f}s)", flush=True)

    tag = args.tag or ""
    fp = out / f"e17_verify_{args.model}_{args.dataset}{tag}.tsv"
    fields = ["source_uri", "target_uri", "rel", "p_yes"] + (["p_yes_rev"] if args.reverse else [])
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    kept = sum(1 for r in rows if r["p_yes"] > 0.5)
    print(f"[e17] wrote {fp}  ({len(rows)} rows, YES-rate={kept/max(1,len(rows)):.3f})", flush=True)


if __name__ == "__main__":
    main()
