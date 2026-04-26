# Model Substitution Log

Append-only record of embedding-model swaps in the BeyondEquivalence retrieval
study. Each entry is dated and reproducible from the git history. New entries
go on top.

---

## 2026-04-26 — NV-Embed-v2 → llama-embed-nemotron-8b

**Direction:** the proposal-named `nvidia/NV-Embed-v2` is replaced by its direct
maintained successor `nvidia/llama-embed-nemotron-8b` for all subsumption-study
runs that previously specified NV-Embed-v2.

### Why

1. **Tool-chain incompatibility, no clean fix.** NV-Embed-v2's released custom
   code (`modeling_nvembed.py`, frozen since summer 2024 with a hard pin
   `transformers==4.42.4`) does not load on our cluster's `transformers==5.5.4`
   stack. The failure is the standard `_finalize_model_loading` AttributeError:

       AttributeError: 'NVEmbedModel' object has no attribute 'all_tied_weights_keys'

   The `all_tied_weights_keys` property was introduced in `transformers==5.0.0`
   (verified via [#43883](https://github.com/huggingface/transformers/issues/43883),
   [#43646](https://github.com/huggingface/transformers/issues/43646)) and the
   new loader path requires it on every `PreTrainedModel` subclass. NV-Embed
   inherits from `PreTrainedModel` but never defines the attribute.

2. **Downgrading the stack is not a single-variable change.** Nvidia's pin block
   for NV-Embed-v2 is a fully-frozen toolchain (`transformers==4.42.4`,
   `torch==2.2.0`, `flash-attn==2.2.0`, `sentence-transformers==2.7.0` with a
   manual L353 source patch). Adopting it would cascade into:
   - vLLM 0.19.1 dependency conflict (its `transformers` pin is incompatible
     with 4.42.4 — vLLM would have to be pinned down or split into a second
     conda env)
   - flash-attn 2.2.0 has no CUDA-13 wheel; source build against the cluster's
     CUDA 13 toolkit is required and historically fragile
   - `LLMHuggingFace.py` Llama-3.3 fallback uses transformers-5.x APIs (`dtype=`,
     not `torch_dtype=`) and would need a backwards-compat patch
   This is a multi-day toolchain migration with second-order risk to a working
   Stage-2 LLM pipeline.

3. **NV-Embed-v2 is unmaintained.** Last commit on `huggingface.co/nvidia/NV-Embed-v2`
   is from August 2024. No upstream fix is forthcoming.

4. **The successor is methodologically equivalent.** `llama-embed-nemotron-8b`
   (released October 2025, 7.5B parameters) is from the same NVIDIA NeMo
   Retriever group, fine-tuned from Llama-3.1-8B with bidirectional attention,
   MMTEB state-of-the-art at release. Same general architecture class
   (decoder-LLM with bidirectional fine-tuning), same parameter scale, same
   instruction-aware embedding paradigm.

5. **The successor has an upstream transformers-5.x compatibility fix.** The
   1B variant `nvidia/llama-nemotron-embed-1b-v2` has a merged PR (Discussion
   #16, February 2026) introducing introspection-based API detection that
   spans `transformers` 4.44 → 5.0+. The 8B variant inherits this pattern. If
   the 8B model card pin (`transformers==4.51.0`, `flash-attn==2.6.3`) turns
   out to be a soft suggestion rather than a hard requirement (likely, given
   the 1B precedent), no toolchain change is needed at all.

### Performance reference

- **MMTEB Retrieval (English):** llama-embed-nemotron-8b is documented as
  state-of-the-art at release; NV-Embed-v2 ≈ 62.0; e5-mistral-7b-instruct ≈ 56.9.
  See model card for the up-to-date number.
- **For our task (concept-alignment on STROMA/TaSeR):** the gap is expected
  smaller than on MTEB IR because Anatomy/groceries/etc. are short
  cross-ontology label-and-description matches, not classical IR. We will
  measure rather than assume.

### License notes

- llama-embed-nemotron-8b: "Non-commercial / research use only" + Llama-3.1
  Community License, gated. HF token with Llama-3 access required.
- This is a tighter license than NV-Embed-v2 (CC-BY-NC-4.0) but compatible with
  the master-thesis research context.

### Configuration deltas

The runner registers a new alias and the model family:

- `MODEL_ALIASES["llama-embed-nemotron-8b"]` → `nvidia/llama-embed-nemotron-8b`
- `_FAMILY_INFERENCE_RULES`: matches on `llama-embed-nemotron` /
  `llama-nemotron-embed` (both spellings to survive Nvidia's repo-naming drift)
- `_FAMILY_FORMATTERS["llama-embed-nemotron"]`: same `_wrap_instruct_query`
  formatter as NV-Embed / e5-mistral (i.e. `Instruct: {task}\nQuery: {text}`)
- `_FAMILY_LOADER_KWARGS["llama-embed-nemotron"]`:
  - `model_kwargs={"attn_implementation": "eager", "torch_dtype": "bfloat16"}`
  - `tokenizer_kwargs={"padding_side": "left"}`

These are passed into `SentenceTransformer(...)` via the generic
`get_loader_kwargs(family)` indirection, so other families remain unaffected
(empty dict → sentence-transformers defaults).

### What was NOT changed

- `LLMHuggingFace.py` (Llama-3.3-70B-Instruct fallback path) — untouched.
- `LLMOpenAI.py` (vLLM client) — untouched.
- `environment.yml` — no `transformers` / `torch` / `flash-attn` pin change.
- vLLM server-side configuration on DWS — untouched.

### Backup plan if `llama-embed-nemotron-8b` also hits a transformers-5.x bug

The 1B-variant fix (`nvidia/llama-nemotron-embed-1b-v2`) is the documented
reference. Its `llama_bidirectional_model.py` uses introspection-based
attribute detection to span the 4.44 → 5.0 API gap. If the 8B variant
exhibits a similar `all_tied_weights_keys` error on first load, port the
detection pattern from the 1B `llama_bidirectional_model.py` into the
8B copy in the HF cache as a one-time post-download patch. Document the
patch as a separate dated entry in this file when applied.

### Reproducibility

- Subsumption runner CLI: `--model llama-embed-nemotron-8b`
- Default family inference resolves to `llama-embed-nemotron`; explicit
  override via `--model-family llama-embed-nemotron` is available.
- Smoke-test path: `--smoke-test` restricts to the first 3 source classes,
  same protocol as for the other models.
