"""
lora_inference_sanity.py — pre-Eval check that a LoRA adapter actually
changes embeddings on the inference path.

Symmetric to the training-side _smoke_inference_check in finetune_lora.py.
Loads the base SentenceTransformer, embeds 10 fixed test strings, then
attaches the adapter via PeftModel.from_pretrained (the same path the
ablation sweep uses) and re-embeds. If max(per-row cosine) is
indistinguishable from 1.0 across the 10 probes, the adapter has no
effect and the eval would silently produce baseline numbers — abort.

Usage:
    python lora_inference_sanity.py --model qwen3 --adapter <path>
    python lora_inference_sanity.py --model nemo  --adapter <path>

Exit codes:
    0 = adapter has measurable effect; eval may proceed.
    1 = no measurable effect; abort.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("lora_inference_sanity")


MODEL_ALIASES = {
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "nemo":  "nvidia/llama-embed-nemotron-8b",
}

BROADER_INSTRUCTION = (
    "Given a category from a hierarchical taxonomy, retrieve broader categories "
    "that subsume it (the parent or ancestor concepts)"
)
NARROWER_INSTRUCTION = (
    "Given a category from a hierarchical taxonomy, retrieve narrower categories "
    "that it subsumes (the child or descendant concepts)"
)
INSTRUCT_PREFIX = "Instruct: {instruction}\nQuery: "

TEST_TEXTS = [
    INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION) + "dog. a domesticated mammal kept as a pet",
    INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION) + "car. a wheeled motor vehicle",
    INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION) + "animal. a living organism that feeds on organic matter",
    INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION) + "vehicle. a conveyance used to transport people or cargo",
    INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION) + "piano. a keyboard instrument with strings",
    INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION) + "musical_instrument. a device producing sound",
    INSTRUCT_PREFIX.format(instruction=BROADER_INSTRUCTION) + "rose. a woody perennial flowering plant",
    INSTRUCT_PREFIX.format(instruction=NARROWER_INSTRUCTION) + "flower. the reproductive structure of plants",
    "dog. a domesticated mammal kept as a pet",
    "vehicle. a conveyance used to transport people or cargo",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Verify a LoRA adapter changes embeddings on inference.")
    p.add_argument("--model", required=True, choices=list(MODEL_ALIASES))
    p.add_argument("--adapter", required=True,
                   help="Path to the PEFT adapter dir (output of finetune_lora.py).")
    p.add_argument("--threshold", type=float, default=0.9999,
                   help="If max(per-row trained-vs-base cosine) >= threshold, "
                        "the adapter is considered ineffective. Default 0.9999.")
    p.add_argument("--report-path", default=None,
                   help="Optional JSON dump of the per-row cosine values.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        sys.exit(f"Adapter dir not found: {adapter_path}")

    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    from peft import PeftModel
    from prompt import get_loader_kwargs

    model_id = MODEL_ALIASES[args.model]
    loader_kwargs = get_loader_kwargs(model_id)

    logger.info("Loading base model: %s", model_id)
    base = SentenceTransformer(model_id, trust_remote_code=True, **loader_kwargs)
    emb_base = base.encode(TEST_TEXTS, convert_to_tensor=True, show_progress_bar=False)
    emb_base = st_util.normalize_embeddings(emb_base)
    logger.info("Base embeddings: shape=%s", tuple(emb_base.shape))

    logger.info("Loading trained model + attaching adapter: %s", adapter_path)
    trained = SentenceTransformer(model_id, trust_remote_code=True, **loader_kwargs)
    inner = trained[0].auto_model
    trained[0].auto_model = PeftModel.from_pretrained(inner, str(adapter_path))
    logger.info("PeftModel attached: peft_config keys=%s",
                list(trained[0].auto_model.peft_config.keys()))

    emb_trained = trained.encode(TEST_TEXTS, convert_to_tensor=True, show_progress_bar=False)
    emb_trained = st_util.normalize_embeddings(emb_trained)
    logger.info("Trained embeddings: shape=%s", tuple(emb_trained.shape))

    per_row_cos = (emb_trained * emb_base).sum(dim=-1).cpu().tolist()
    mean_cos = sum(per_row_cos) / len(per_row_cos)
    max_cos  = max(per_row_cos)
    min_cos  = min(per_row_cos)
    has_effect = max_cos < args.threshold

    summary = {
        "model": args.model,
        "model_id": model_id,
        "adapter": str(adapter_path),
        "n_test_texts": len(TEST_TEXTS),
        "per_row_cosine_trained_vs_base": per_row_cos,
        "trained_vs_base_cosine_mean": mean_cos,
        "trained_vs_base_cosine_min":  min_cos,
        "trained_vs_base_cosine_max":  max_cos,
        "threshold": args.threshold,
        "adapter_has_effect": has_effect,
    }

    if args.report_path:
        Path(args.report_path).write_text(json.dumps(summary, indent=2))
        logger.info("Wrote report: %s", args.report_path)

    print(json.dumps(summary, indent=2))
    if not has_effect:
        sys.exit(f"FAIL: adapter has no measurable effect "
                 f"(max cosine vs. base = {max_cos:.6f}, threshold {args.threshold}).")
    print(f"PASS: adapter changes embeddings (max cos = {max_cos:.6f}).")


if __name__ == "__main__":
    main()
