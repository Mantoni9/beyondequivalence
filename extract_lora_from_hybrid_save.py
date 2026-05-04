"""
extract_lora_from_hybrid_save.py — recover a PEFT-loadable adapter dir
from a SentenceTransformer checkpoint that contains LoRA tensors with
PEFT-internal names but no adapter_config.json next to them.

Background (smoke 2026-05-04):
  finetune_lora.py called model[0].auto_model.save_pretrained() on a
  PeftModel-wrapped HF AutoModel. Because the inner model was loaded
  with trust_remote_code=True (Qwen3Model / LlamaBidirectionalModel
  custom code), PEFT's save_pretrained fell into a hybrid path that
  serialises BOTH base_layer.weight AND lora_A/lora_B tensors into a
  single model.safetensors, without writing the adapter_config.json
  marker that signals to PEFT how to load them later.

  Inspecting model.safetensors keys confirms the tensors are present:
    layers.<N>.self_attn.{q,k,v,o}_proj.lora_A.default.weight
    layers.<N>.self_attn.{q,k,v,o}_proj.lora_B.default.weight
    layers.<N>.self_attn.{q,k,v,o}_proj.base_layer.weight

This script extracts the LoRA tensors only, writes them as a clean
PEFT adapter dir (adapter_model.safetensors + adapter_config.json),
and verifies the count matches the expected
4 (q/k/v/o) x 2 (A/B) x num_layers — 288 for Qwen3 (36L) / 256 for Nemo (32L).

Output is loadable via:
    PeftModel.from_pretrained(base_inner_model, output_dir)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger("extract_lora")

EXPECTED_LAYERS_PER_MODEL = {
    "Qwen/Qwen3-Embedding-8B":            36,
    "nvidia/llama-embed-nemotron-8b":     32,
}

LORA_KEY_PATTERN = re.compile(
    r"\.(?P<proj>q_proj|k_proj|v_proj|o_proj)\.(?P<side>lora_A|lora_B)\."
    r"(?P<adapter>[^.]+)\.weight$"
)


def main() -> None:
    p = argparse.ArgumentParser(description="Recover a PEFT adapter from a hybrid SBERT save.")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Path to the checkpoint dir holding model.safetensors "
                        "with hybrid base+lora keys.")
    p.add_argument("--output-dir", required=True,
                   help="Destination dir — will hold adapter_model.safetensors "
                        "+ adapter_config.json after extraction.")
    p.add_argument("--base-model-name", required=True,
                   help="HF model id of the base; goes into adapter_config.json's "
                        "base_model_name_or_path.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.10)
    p.add_argument("--target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])
    p.add_argument("--adapter-name", default="default",
                   help="Adapter name to look for (matches the .lora_A.<name>.weight "
                        "naming convention in the hybrid save).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    ckpt = Path(args.checkpoint_dir)
    out  = Path(args.output_dir)
    if not ckpt.is_dir():
        sys.exit(f"checkpoint-dir not found: {ckpt}")
    out.mkdir(parents=True, exist_ok=True)

    # The hybrid save may be sharded into multiple .safetensors shards
    # (model-00001-of-N.safetensors). Walk all of them.
    shard_paths = sorted(ckpt.glob("*.safetensors"))
    if not shard_paths:
        sys.exit(f"No .safetensors files in {ckpt}")
    logger.info("Found %d safetensors shard(s): %s",
                len(shard_paths), [p.name for p in shard_paths])

    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        sys.exit("safetensors not installed. Run: "
                 "conda run -n melt-olala python -m pip install safetensors")

    # Walk every shard and pull out only the lora_A / lora_B tensors that
    # match the adapter_name and a target proj.
    target_projs = set(args.target_modules)
    extracted: dict[str, "torch.Tensor"] = {}
    per_proj: dict[str, int] = {}
    layer_indices: set[int] = set()
    layer_idx_re = re.compile(r"\.layers\.(\d+)\.")

    for shard in shard_paths:
        logger.info("Loading shard: %s", shard.name)
        state = load_file(str(shard))
        for k, v in state.items():
            m = LORA_KEY_PATTERN.search(k)
            if not m:
                continue
            if m.group("adapter") != args.adapter_name:
                continue
            if m.group("proj") not in target_projs:
                continue
            # Rewrite the key into the canonical PEFT format expected by
            # PeftModel.from_pretrained:
            #   base_model.model.<rest>.<proj>.<side>.weight
            # Hybrid save key example:
            #   model.layers.5.self_attn.q_proj.lora_A.default.weight
            # PEFT canonical key:
            #   base_model.model.model.layers.5.self_attn.q_proj.lora_A.default.weight
            # PEFT prepends "base_model.model." regardless of inner prefix.
            new_key = f"base_model.model.{k}"
            extracted[new_key] = v
            per_proj[m.group("proj")] = per_proj.get(m.group("proj"), 0) + 1
            li = layer_idx_re.search(k)
            if li:
                layer_indices.add(int(li.group(1)))

    # Sanity: count should be 4 (q/k/v/o) x 2 (A/B) x num_layers.
    expected_layers = EXPECTED_LAYERS_PER_MODEL.get(args.base_model_name)
    n_extracted = len(extracted)
    expected_count = (
        len(target_projs) * 2 * expected_layers
        if expected_layers is not None else None
    )
    logger.info("Extracted %d LoRA tensors. Per-proj counts: %s",
                n_extracted, per_proj)
    logger.info("Layer indices observed: %d distinct (min=%s, max=%s)",
                len(layer_indices),
                min(layer_indices) if layer_indices else None,
                max(layer_indices) if layer_indices else None)
    if expected_count is not None:
        if n_extracted != expected_count:
            logger.error("MISMATCH: extracted=%d, expected=%d (%d projs x 2 sides x %d layers)",
                         n_extracted, expected_count, len(target_projs), expected_layers)
            sys.exit(1)
        # Each proj should contribute exactly 2 * num_layers tensors.
        for proj in target_projs:
            n = per_proj.get(proj, 0)
            if n != 2 * expected_layers:
                logger.error("Per-proj mismatch for %s: got=%d expected=%d",
                             proj, n, 2 * expected_layers)
                sys.exit(1)
    elif n_extracted == 0:
        sys.exit("No LoRA tensors matched the pattern — verify the hybrid save layout.")

    # Write adapter_model.safetensors with metadata={"format": "pt"}.
    save_file(extracted, str(out / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    logger.info("Wrote %s/adapter_model.safetensors (%d tensors)",
                out, n_extracted)

    # Write adapter_config.json — minimum field set PEFT needs to reload.
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": args.base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": args.lora_r,
        "revision": None,
        "target_modules": list(target_projs),
        "task_type": "FEATURE_EXTRACTION",
    }
    (out / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    logger.info("Wrote %s/adapter_config.json", out)

    # Print a one-liner summary.
    print()
    print(f"OK. Adapter recovered at {out}/")
    print(f"   adapter_config.json + adapter_model.safetensors")
    print(f"   tensors: {n_extracted}  (expected {expected_count if expected_count else 'n/a'})")
    print(f"   per-proj: {per_proj}")


if __name__ == "__main__":
    main()
