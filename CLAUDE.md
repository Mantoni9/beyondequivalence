# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OLALA is a Python-based **ontology matching system** for the OAEI (Ontology Alignment Evaluation Initiative) benchmark. It finds correspondences between entities (classes, properties) across two RDF knowledge graphs, outputting an `Alignment` — a set of `Correspondence` tuples with confidence scores.

## Running the System

There is no formal build system. Entry points are `if __name__ == "__main__"` blocks in matcher files, plus a dedicated experiment runner:

```bash
python MatcherSimple.py        # Runs lexical matching on anatomy + conference OAEI tracks
python MatcherTopN.py          # Tests top-N filtering configurations

# Full experiment runner (Anatomy track, baseline + LLM pipeline):
python run_experiment.py --model ~/models/Llama-3.1-8B-Instruct
python run_experiment.py --model ~/models/Llama-3.1-8B-Instruct --baseline-only
python run_experiment.py --model ~/models/Llama-3.1-8B-Instruct --wandb --threshold 0.6
```

OAEI datasets are downloaded automatically to `~/oaei_track_cache/` on first run. Results are written to `results/{timestamp}_results/`.

## Architecture

The system follows a **Strategy + Composition** pattern for matcher pipelines:

```
RDF Files → RDFGraphWrapper → Matcher Pipeline → Alignment → Evaluation
```

### Core Data Flow

1. `RDFGraphWrapper` loads RDF ontologies with SPARQL support and NetworkX integration
2. `Alignment` (a collection of `Correspondence` triples) is passed as input/output between matchers
3. Matchers implement `MatcherBase.match(kg_source, kg_target, input_alignment, parameters) -> Alignment`
4. `Evaluation.py` runs OAEI tracks, comparing output against reference alignments for P/R/F1

### Matcher Hierarchy

- `MatcherBase` — abstract interface all matchers implement
- `MatcherSimple` — lexical matching via normalized label comparison (tokenization, lowercasing, stopword removal)
- `MatcherCandidateGen` — semantic matching using sentence transformers + cosine similarity on entity embeddings
- `MatcherTopN` — post-processing filter, limits matches per entity to top-N by confidence
- `MatcherSequential` — pipes multiple matchers in sequence (composition)
- `MatcherFileLoader` — loads pre-computed alignments from files

A typical pipeline: `MatcherSimple → MatcherCandidateGen → MatcherTopN → MatcherLLMReranker`, composed via `MatcherSequential`.

### LLM Integration

`LLMBase` defines the abstract interface. Backend selection is automatic in `run_experiment.py` based on the `VLLM_BASE_URL` env var:

- `VLLM_BASE_URL` set → **`LLMOpenAI`** against a local vLLM server (cluster default — exported by the SLURM job script).
- `VLLM_BASE_URL` unset → **`LLMHuggingFace`** loading the model in-process (local development fallback).

**`LLMOpenAI`** — OpenAI-compatible HTTP backend:
- Batch processing via file uploads (when `batch_poll_interval` is set), otherwise synchronous per-request calls.
- `get_confidence_first_token` requests `max_tokens=1, logprobs=True, top_logprobs=20`, then computes `P(yes) / (P(yes) + P(no))` from the **max** logprob across positive/negative tokens in the top-20.
- `get_confidence_with_tools` performs batched multi-turn tool exploration before a final yes/no logprob call.
- Custom `base_url` enables local LLM servers (vLLM, Ollama).

**`LLMHuggingFace`** — local HuggingFace Transformers backend (fallback path):
- Loads model and tokenizer via `AutoModelForCausalLM` / `AutoTokenizer` with `trust_remote_code=True`.
- Uses `dtype` (not `torch_dtype`) in `from_pretrained` to avoid deprecation warnings; defaults to `torch.bfloat16`.
- `get_text_completion` runs greedy generation with `model.generate()`.
- `get_confidence_first_token` does a **single forward pass** (no generation), reads the logits at the last input position, applies softmax, and computes `P(yes) / (P(yes) + P(no))` by aggregating probabilities over all vocabulary tokens matched by regex patterns for positive ("yes"/"true") and negative ("no"/"false") tokens. Token sets are built once at init via `_initialize_positive_negative_tokens()`.
- Chat formatting is applied via `tokenizer.apply_chat_template(add_generation_prompt=True)` before every forward pass.
- FA2 is used only on the CUDA + full-precision path; NF4/8-bit paths use SDPA (transformers 5.5.x has an FA2 bug with bitsandbytes).

`prompt.py` provides a fluent `Prompt` builder for multi-turn conversations and pre-defined templates for embedding, reranking, and SPARQL agent interactions. Reranking prompts (`RERANKING_PROMPTS`) use placeholders `{source_url}`, `{target_url}`, `{source_kg}`, `{target_kg}` and are looked up by key (e.g. `"d"`).

### `MatcherLLMReranker`

Post-processing matcher that scores and filters candidates from `input_alignment` using an LLM:

1. For each `Correspondence` in `input_alignment`, the KG sub-graphs of both entities are extracted via a configurable `RDFGraphWrapper` description method (default: `description_one_gen`) and serialized to Turtle.
2. A reranking prompt (default: prompt `"d"`) is filled with the entity URIs and serialized sub-graphs.
3. `LLMHuggingFace.get_confidence_first_token()` scores all prompts in configurable **batches** (`batch_size`, default 8) to bound GPU memory usage.
4. Correspondences with score ≥ `threshold` (default 0.5) are kept; their confidence is replaced by the LLM score.

Key constructor parameters: `llm`, `prompt_id="d"`, `description="description_one_gen"`, `kg_format="turtle"`, `threshold=0.5`, `batch_size=8`.

### Key Parameter Keys

`ParameterConfigKeys.py` defines OAEI-standard parameter names used across matchers (language, matching targets, ontology format, etc.).

## Multi-Cluster Setup

Three execution environments are supported. Each has a corresponding `.env.<cluster>.template` file in the project root — copy it to `.env.<cluster>` (never committed) before running.

| Environment | User | Host / Path | Model | Device |
|---|---|---|---|---|
| **Local** (MacBook M4 Max) | — | `~/` | `~/models/Llama-3.1-8B-Instruct` | `mps` |
| **bwUniCluster** | `ma_amarkic` | `uc3.scc.kit.edu` · `/pfs/data6/home/ma/ma_ma/ma_amarkic/melt-project/src` | `$WORK/models/Llama-3.3-70B-Instruct` | `cuda` |
| **DWS** (Uni Mannheim) | `amarkic` | `dws-login-01.informatik.uni-mannheim.de` · `/work/amarkic/beyondequivalence` | `/work/amarkic/models/Llama-3.3-70B-Instruct` (NF4 4-bit + Flash Attention 2) | `cuda` |

> **DWS login nodes:** `dws-login-01` is tried first; on SSH timeout (5 s) the script automatically falls back to `dws-login-02.informatik.uni-mannheim.de`.

### Environment templates

```
.env.local.template   → copy to .env.local   (DEVICE=mps,  CLUSTER=local)
.env.bwuni.template   → copy to .env.bwuni   (DEVICE=cuda, CLUSTER=bwuni)
.env.dws.template     → copy to .env.dws     (DEVICE=cuda, CLUSTER=dws, LOAD_IN_4BIT=true)
```

`run_experiment.py` calls `load_dotenv()` at startup and reads `MODEL_PATH`, `DEVICE`, `CLUSTER` from whichever `.env` file is present.

**vLLM env vars** (cluster `.env` files) — consumed by the `*_vllm.sh` job scripts:
- `VLLM_TENSOR_PARALLEL` (e.g. `2`) — passed as `--tensor-parallel-size`.
- `VLLM_QUANTIZATION` (e.g. `awq` on DWS, unset on bwUni A100s) — passed as `--quantization`.
- `VLLM_DTYPE` (e.g. `float16` for AWQ, defaults to `float16`) — passed as `--dtype`.
- `VLLM_MAX_MODEL_LEN` (e.g. `8192`) — passed as `--max-model-len`.
- `VLLM_BASE_URL` is **not** set in `.env` — the job script computes it as `http://localhost:$((8000 + JOB_ID % 1000))/v1` and exports it before launching `run_experiment.py`.

**Quantization choice on DWS:** AWQ (INT4, GEMM kernels, group_size=128). Model: `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` at `/work/amarkic/models/Llama-3.3-70B-Instruct-AWQ-INT4`. Fits ~38 GB/GPU on 2× A40 with TP=2. **Do NOT use `bitsandbytes`** — produces NaN logits with Llama-3.3-70B + bf16 compute (verified on DWS 2026-04-26: greedy outputs `!!!!!`, JSON-encode fails on `nan` logprobs). AWQ produces correct outputs (`text='Yes.'`, `confidence=0.9999`) and loads in ~18 s vs. ~127 s for bnb.

**HuggingFace fallback path:** `LOAD_IN_4BIT=true` activates NF4 4-bit quantization (`bnb_4bit_quant_type="nf4"`, `compute_dtype=bfloat16`) via `bitsandbytes`. The 70B model requires ~35 GB VRAM in NF4 — fits on a single A6000 (48 GB). The MPS/CPU fallback to float32 only applies when neither `LOAD_IN_4BIT` nor `LOAD_IN_8BIT` is set and CUDA is absent.

### DWS-specific vLLM workarounds

`jobs/job_dws_vllm.sh` sets several environment workarounds that are mandatory on this cluster:

- `LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"` — DWS system libstdc++ is too old (missing `CXXABI_1.3.15`); the conda env's libstdc++ must come first.
- `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` — A6000 nodes have no NVLink P2P, NCCL hangs without this (only relevant at TP>1, but harmless to set always).
- `--enforce-eager` — disables CUDA-graph capture; avoids the V1 `shm_broadcast` deadlock and is currently required on this cluster. (vLLM 0.19+ ignores `VLLM_USE_V1`; V1 is the only engine.)
- `--gpu-memory-utilization 0.92` — leaves slack on A40 (49 GB) so the activation buffers + KV cache don't OOM under load. With AWQ-INT4 the model itself uses ~18.6 GB/GPU and KV cache gets ~24 GB, leaving headroom.

### Python package installs

Always install via the conda env to avoid mixing with system Python:

```bash
conda run -n melt-olala python -m pip install <package>
```

### sync_clusters.sh

```bash
# First-time setup (install conda env, create directories)
./sync_clusters.sh bwuni --setup
./sync_clusters.sh dws   --setup
./sync_clusters.sh all   --setup

# Sync code (git pull on the cluster)
./sync_clusters.sh bwuni --sync
./sync_clusters.sh dws   --sync
./sync_clusters.sh all   --sync

# Submit a SLURM job
./sync_clusters.sh bwuni --job jobs/job_bwuni_70b.sh
./sync_clusters.sh dws   --job jobs/job_dws_70b.sh
```

### SLURM job scripts

vLLM scripts are the primary path; the `_70b.sh` scripts run the legacy in-process `LLMHuggingFace` backend.

| Script | Cluster | Backend | Partition / GPUs | Mem | Time |
|---|---|---|---|---|---|
| `jobs/job_bwuni_vllm.sh` | bwUniCluster | vLLM (no quantization) | `gpu_a100_il` · 2× A100 | 200 G | 48 h |
| `jobs/job_dws_vllm.sh`   | DWS          | vLLM (AWQ INT4)        | `gpu-vram-48gb` · 2× A40/A6000 | 100 G | 6 h |
| `jobs/job_bwuni_70b.sh`  | bwUniCluster | `LLMHuggingFace` (legacy) | `gpu_a100_il` · 2× A100 | 300 G | 48 h |
| `jobs/job_dws_70b.sh`    | DWS          | `LLMHuggingFace` NF4 (legacy) | `gpu-vram-48gb` · 2× A6000 | 100 G | 24 h |

vLLM is not in `environment.yml`; the job scripts install it on first run via `python -m pip install vllm`.

## DWS verified baseline (2026-04-26)

First successful end-to-end run on DWS (2× A40, TP=2, AWQ-INT4, vLLM 0.19.1):

- **Track**: anatomy_track / mouse-human-suite
- **Pipeline**: `MatcherCandidateGen(all-MiniLM-L6-v2) → MatcherTopN(5) → MatcherLLMReranker(prompt=d, threshold=0.6, batch_size=2)`
- **F1**: 0.765 (P=0.838, R=0.703, TP=1066, FP=206, FN=450)
- **Runtime**: 22 min (~1.65 s per LLM call, sequential)
- **W&B**: `antonio-markic-university-of-mannheim/olala-ontology-matching`
- **Throughput bottleneck**: `MatcherLLMReranker` calls vLLM sequentially despite vLLM supporting concurrency. Parallelising with `asyncio.gather` would cut runtime by ~5×. Marlin kernels (`--quantization awq_marlin`) would add another 2–3× decode speedup. Neither is required for correctness.
