"""
finetune_lora.py — LoRA fine-tuning of Qwen3-Embedding-8B or
llama-embed-nemotron-8b on WordNet subsumption triplets.

Trainings-Sample-Format (per-row, per the 2026-05-04 methodology):
  anchor:   "Instruct: <broader|narrower instruction>\nQuery: <label>. <definition>"
  positive: "<label>. <definition>"
  negative: "<label>. <definition>" | omitted when null
  label:    "broader" | "narrower"   (used by BatchSamplers.GROUP_BY_LABEL
                                      to homogenise batches by instruction
                                      type; otherwise the in-batch negatives
                                      mix instruction-types and the loss
                                      smears the direction signal.)

Loss: MultipleNegativesRankingLoss (in-batch negatives + optional explicit
hard negative when present).

PEFT-LoRA stack on top of the loaded SentenceTransformer:
  - r=16, alpha=32, dropout=0.1, bias="none"
  - target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
  - The base model + any pretrained adapters stay frozen; only the new
    LoRA adapter's parameters are trainable. Verified via the
    `print_trainable_parameters()` call right after add_adapter().

Smoke mode: --smoke-triplets N caps the train set to N rows and forces
1 epoch + Cosine LR + tighter eval interval. Default smoke = 5000 rows.

Resume / output:
  output-dir/                  — final adapter weights (model.save_pretrained)
  output-dir/training_args.bin — Trainer artefacts
  output-dir/checkpoint-*/     — periodic checkpoints (last one)
  results/lora_smoke_<model>_<TS>.json — smoke validation summary

W&B group: lora_subsumption_<TS>_<SHA>.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

logger = logging.getLogger("finetune_lora")


# Single source of truth for the two B-pin instructions used at inference.
# Trainings-side anchors prefix exactly these strings, so train and inference
# share the same prompt prefix verbatim.
BROADER_INSTRUCTION = (
    "Given a category from a hierarchical taxonomy, retrieve broader categories "
    "that subsume it (the parent or ancestor concepts)"
)
NARROWER_INSTRUCTION = (
    "Given a category from a hierarchical taxonomy, retrieve narrower categories "
    "that it subsumes (the child or descendant concepts)"
)
INSTRUCT_PREFIX = "Instruct: {instruction}\nQuery: "

MODEL_ALIASES = {
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "nemo":  "nvidia/llama-embed-nemotron-8b",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_dataset(rows: list[dict], have_hard_negatives: bool):
    """Convert WordNet triplets to a HF datasets.Dataset whose columns
    match Sentence-Transformers' expected schema for
    MultipleNegativesRankingLoss with optional hard negatives:
      - anchor, positive  (always)
      - negative          (only when have_hard_negatives=True)
      - label             (instruction_type, drives GROUP_BY_LABEL batching)

    Cross-instruction-type triplet leakage is impossible here because each
    row carries its own label and the BatchSampler groups by it.
    """
    from datasets import Dataset
    out: dict[str, list] = {"anchor": [], "positive": [], "label": []}
    if have_hard_negatives:
        out["negative"] = []
    for r in rows:
        instr_text = (BROADER_INSTRUCTION if r["instruction_type"] == "broader"
                      else NARROWER_INSTRUCTION)
        anchor_with_prefix = INSTRUCT_PREFIX.format(instruction=instr_text) + r["anchor_text"]
        out["anchor"].append(anchor_with_prefix)
        out["positive"].append(r["positive_text"])
        out["label"].append(r["instruction_type"])
        if have_hard_negatives:
            # Use a positive-shaped fallback when hard negative missing.
            # Only triggers when have_hard_negatives=False overall sample
            # detection said most rows had a negative — should be rare.
            neg = r.get("negative_text") or r["positive_text"]
            out["negative"].append(neg)
    return Dataset.from_dict(out)


def _smoke_inference_check(model, output_dir: Path, log_path: Path) -> dict:
    """Verify the LoRA adapter actually changes embeddings vs. base.
    Loads a fresh base model (no adapter), embeds 10 ad-hoc test strings,
    compares against the trained model. If cosine distances are
    indistinguishable, the adapter has no effect — flag as RED.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    from prompt import get_loader_kwargs

    test_texts = [
        f"{INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION)}dog. a domesticated mammal kept as a pet",
        f"{INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION)}car. a wheeled motor vehicle",
        f"{INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION)}animal. a living organism that feeds on organic matter",
        f"{INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION)}vehicle. a conveyance used to transport people or cargo",
        f"{INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION)}piano. a keyboard instrument with strings",
        f"{INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION)}musical_instrument. a device producing sound",
        f"{INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION)}rose. a woody perennial flowering plant",
        f"{INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION)}flower. the reproductive structure of plants",
        "dog. a domesticated mammal kept as a pet",
        "vehicle. a conveyance used to transport people or cargo",
    ]

    emb_trained = model.encode(test_texts, convert_to_tensor=True, show_progress_bar=False)
    emb_trained = st_util.normalize_embeddings(emb_trained)

    base = SentenceTransformer(
        model.model_card_data.base_model if hasattr(model, "model_card_data") else MODEL_ALIASES.get(
            getattr(model, "_smoke_model_alias", "qwen3"), "Qwen/Qwen3-Embedding-8B"
        ),
        trust_remote_code=True,
        **get_loader_kwargs(MODEL_ALIASES.get(
            getattr(model, "_smoke_model_alias", "qwen3"), "Qwen/Qwen3-Embedding-8B"
        )),
    )
    emb_base = base.encode(test_texts, convert_to_tensor=True, show_progress_bar=False)
    emb_base = st_util.normalize_embeddings(emb_base)

    # Per-row cosine between trained and base for the SAME text. If the
    # adapter has any effect, this is < 1.0.
    per_row_cos = (emb_trained * emb_base).sum(dim=-1).cpu().tolist()
    mean_cos = sum(per_row_cos) / len(per_row_cos)
    max_cos  = max(per_row_cos)
    min_cos  = min(per_row_cos)

    summary = {
        "n_test_texts": len(test_texts),
        "trained_vs_base_cosine_mean": mean_cos,
        "trained_vs_base_cosine_min":  min_cos,
        "trained_vs_base_cosine_max":  max_cos,
        "adapter_has_effect": max_cos < 0.9999,  # any departure from identity
    }
    log_path.write_text(json.dumps(summary, indent=2))
    logger.info("Smoke inference check: %s", summary)
    if not summary["adapter_has_effect"]:
        logger.error("ADAPTER HAS NO EFFECT — trained embeddings match base "
                     "to within 1e-4. Investigate before continuing.")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tune an embedding model on WordNet triplets.")
    p.add_argument("--model", required=True, choices=list(MODEL_ALIASES),
                   help="Model alias: qwen3 | nemo")
    p.add_argument("--triplets-path", default="data/wordnet_triplets.jsonl")
    p.add_argument("--output-dir", default=None,
                   help="Default: lora_adapters/<model>_subsumption_lora")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-triplets", type=int, default=None,
                   help="When set, cap the train set to this many rows and "
                        "force epochs=1 + tighter eval. Use 5000 for the smoke.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-project", default="beyondequivalence-lora-finetune")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    _set_seeds(args.seed)

    model_id = MODEL_ALIASES[args.model]
    sha = _git_sha()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir is None:
        suffix = "_smoke" if args.smoke_triplets else ""
        args.output_dir = f"lora_adapters/{args.model}_subsumption_lora{suffix}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_group is None:
        args.wandb_group = f"lora_subsumption_{ts}_{sha}"

    logger.info("Model: %s (%s)", args.model, model_id)
    logger.info("SHA: %s  TS: %s", sha, ts)
    logger.info("Output: %s", output_dir)
    logger.info("Smoke mode: %s", args.smoke_triplets if args.smoke_triplets else "off")

    # ── Load datasets. ─────────────────────────────────────────────────
    triplets_path = Path(args.triplets_path)
    train_path = triplets_path.with_suffix(".train.jsonl")
    val_path   = triplets_path.with_suffix(".val.jsonl")
    if not train_path.is_file() or not val_path.is_file():
        sys.exit(f"Train/val splits not found. Run prepare_wordnet_triplets.py "
                 f"first to generate {train_path} and {val_path}.")
    train_rows = _load_jsonl(train_path)
    val_rows   = _load_jsonl(val_path)
    logger.info("Loaded train=%d val=%d", len(train_rows), len(val_rows))

    if args.smoke_triplets is not None:
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)
        train_rows = train_rows[: args.smoke_triplets]
        # Cap val proportionally too — keeps smoke cheap.
        val_rows = val_rows[: max(200, args.smoke_triplets // 25)]
        args.epochs = 1
        logger.info("Smoke cap applied: train=%d val=%d epochs=1",
                    len(train_rows), len(val_rows))

    # Detect whether enough rows have hard negatives to use them. We use
    # a single decision for the whole set so the column schema stays
    # consistent.
    have_hard_neg_train = sum(1 for r in train_rows if r.get("negative_text"))
    use_hard_neg = (have_hard_neg_train / max(1, len(train_rows))) >= 0.8
    logger.info("Hard negatives: %d/%d train rows have one. use_hard_neg=%s",
                have_hard_neg_train, len(train_rows), use_hard_neg)

    train_ds = _build_dataset(train_rows, use_hard_neg)
    val_ds   = _build_dataset(val_rows,   use_hard_neg)
    logger.info("Datasets built. columns=%s", train_ds.column_names)

    # ── Load model + add LoRA adapter. ─────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from prompt import get_loader_kwargs
    loader_kwargs = get_loader_kwargs(model_id)
    logger.info("Loading SentenceTransformer model='%s' loader_kwargs=%s",
                model_id, loader_kwargs)
    model = SentenceTransformer(model_id, trust_remote_code=True, **loader_kwargs)
    model.max_seq_length = args.max_seq_length
    setattr(model, "_smoke_model_alias", args.model)

    from peft import LoraConfig, get_peft_model, PeftModel
    peft_config = LoraConfig(
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    logger.info("PEFT config: r=%d alpha=%d dropout=%.2f bias=none target=q/k/v/o",
                args.lora_r, args.lora_alpha, args.lora_dropout)

    # Wrap the inner HF transformer with PEFT — NOT the SentenceTransformer
    # top-level object. SentenceTransformer.add_adapter is a stub that
    # silently no-ops (smoke job 242696, sentence-transformers 5.4.1 +
    # peft 0.19.1: peft_config keys before/after were both [], the smoke
    # assert correctly aborted). The canonical SBERT-PEFT pattern is to
    # wrap model[0].auto_model and plug it back into the SBERT module.
    inner = model[0].auto_model
    if isinstance(inner, PeftModel):
        sys.exit("Inner HF model is already a PeftModel — did the script "
                 "run twice without reloading? Aborting to avoid stacked "
                 "adapters.")
    peft_inner = get_peft_model(inner, peft_config)
    model[0].auto_model = peft_inner
    # Also publish peft_config on the SBERT object so external sanity
    # checks (e.g. inference-time loaders) can detect adapter presence
    # without reaching into model[0].
    model.peft_config = peft_inner.peft_config
    logger.info("Wrapped model[0].auto_model with get_peft_model; "
                "peft_config keys=%s", list(peft_inner.peft_config.keys()))

    # PEFT's own helper — primary trainable-param report.
    peft_inner.print_trainable_parameters()

    # Second wachhund — independent count over the whole SBERT graph in
    # case PEFT's helper doesn't see the SBERT pooling/normalize layers.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    ratio = 100 * n_trainable / max(1, n_total)
    logger.info("Trainable params (full SBERT graph): %d / %d  (%.4f%%)",
                n_trainable, n_total, ratio)
    if ratio >= 5.0:
        logger.warning("Trainable-param ratio >=5%% over the full SBERT "
                       "graph — base may not be frozen. Investigate.")
    if not list(peft_inner.peft_config.keys()):
        sys.exit("PEFT adapter not registered after get_peft_model. "
                 "Investigate transformers / peft compatibility.")

    if args.gradient_checkpointing:
        try:
            model[0].auto_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing: ON")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing: %s", e)

    # ── Training. ──────────────────────────────────────────────────────
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import (
        SentenceTransformerTrainingArguments, BatchSamplers,
    )
    from sentence_transformers import SentenceTransformerTrainer

    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_GROUP", args.wandb_group)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        # GROUP_BY_LABEL groups by `label` column so each batch is a single
        # instruction_type — keeps the in-batch negatives semantically
        # sharp (no broader vs. narrower mixing leaking the direction tag).
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        eval_strategy="epoch" if not args.smoke_triplets else "steps",
        eval_steps=200 if args.smoke_triplets else None,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        report_to=["wandb"] if args.wandb else [],
        run_name=f"lora_{args.model}_{sha}{'_smoke' if args.smoke_triplets else ''}",
        seed=args.seed,
    )

    loss_fn = MultipleNegativesRankingLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss_fn,
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    t_elapsed = time.perf_counter() - t0
    logger.info("Training finished in %.1fs (%.1fmin). Final metrics: %s",
                t_elapsed, t_elapsed / 60, train_result.metrics)

    # ── Save final adapter weights. ────────────────────────────────────
    # Save the inner PEFT model only — that writes adapter_config.json +
    # adapter_model.safetensors (~50-200 MB), NOT the 16-GB base. The
    # inference path reloads via PeftModel.from_pretrained(inner_base, path).
    model[0].auto_model.save_pretrained(str(output_dir))
    logger.info("Saved adapter weights to %s", output_dir)

    # ── Smoke-mode adapter-effect verification. ────────────────────────
    if args.smoke_triplets:
        Path("results").mkdir(parents=True, exist_ok=True)
        smoke_log = Path("results") / f"lora_smoke_{args.model}_{ts}.json"
        try:
            summary = _smoke_inference_check(model, output_dir, smoke_log)
        except Exception:
            logger.exception("Smoke inference check failed.")
            sys.exit(1)
        if not summary["adapter_has_effect"]:
            sys.exit("Smoke FAIL: adapter did not change embeddings vs. base.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
