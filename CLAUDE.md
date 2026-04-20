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

`LLMBase` defines the abstract interface. Two concrete backends exist:

**`LLMOpenAI`** — OpenAI-compatible HTTP backend supporting:
- Batch processing via file uploads
- Confidence estimation from token logprobs (scanning vocabulary for positive/negative tokens like "yes"/"no")
- Custom base URLs for local LLM servers (e.g., vLLM, Ollama)

**`LLMHuggingFace`** — local HuggingFace Transformers backend for models such as LLaMA 3:
- Loads model and tokenizer via `AutoModelForCausalLM` / `AutoTokenizer` with `trust_remote_code=True`
- Uses `dtype` (not `torch_dtype`) in `from_pretrained` to avoid deprecation warnings; defaults to `torch.bfloat16`
- `get_text_completion` runs greedy generation with `model.generate()`
- `get_confidence_first_token` does a **single forward pass** (no generation), reads the logits at the last input position, applies softmax, and computes `P(yes) / (P(yes) + P(no))` by aggregating probabilities over all vocabulary tokens matched by regex patterns for positive ("yes"/"true") and negative ("no"/"false") tokens. Token sets are built once at init via `_initialize_positive_negative_tokens()`.
- Chat formatting is applied via `tokenizer.apply_chat_template(add_generation_prompt=True)` before every forward pass.

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
| **DWS** (Uni Mannheim) | `amarkic` | `dws-login-01.informatik.uni-mannheim.de` · `/work/amarkic/olala` | `/work/amarkic/models/Llama-3.3-70B-Instruct` (8-bit, bitsandbytes) | `cuda` |

> **DWS login nodes:** `dws-login-01` is tried first; on SSH timeout (5 s) the script automatically falls back to `dws-login-02.informatik.uni-mannheim.de`.

### Environment templates

```
.env.local.template   → copy to .env.local   (DEVICE=mps,  CLUSTER=local)
.env.bwuni.template   → copy to .env.bwuni   (DEVICE=cuda, CLUSTER=bwuni)
.env.dws.template     → copy to .env.dws     (DEVICE=cuda, CLUSTER=dws, LOAD_IN_8BIT=true)
```

`run_experiment.py` calls `load_dotenv()` at startup and reads `MODEL_PATH`, `DEVICE`, `CLUSTER` from whichever `.env` file is present.

`LOAD_IN_8BIT=true` in `.env.dws` activates 8-bit quantization via `bitsandbytes` in `LLMHuggingFace`, allowing the 70B model to run on 2× A6000 (96 GB VRAM total). The MPS/CPU fallback to float32 only applies in the non-8-bit path.

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

| Script | Cluster | Partition | GPUs | Mem | Time |
|---|---|---|---|---|---|
| `jobs/job_bwuni_70b.sh` | bwUniCluster | `gpu_a100_il` | 2× A100 | 300 G | 48 h |
| `jobs/job_dws_70b.sh` | DWS | `gpu-vram-48gb` | 2× A6000 | 100 G | 24 h |

### Pending manual steps

- **GitHub:** Create a remote repo and run `git remote add origin <url> && git push -u origin main`.
- **DWS model:** Upload `Llama-3.3-70B-Instruct` to `/work/amarkic/models/` on DWS before submitting jobs.
- **bwUniCluster model:** Verify `$WORK/models/Llama-3.3-70B-Instruct` is present before submitting the 70B job.
