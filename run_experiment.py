"""
run_experiment.py — Anatomy-Track experiment runner

Pipelines
---------
1. MatcherSimple (Baseline)
2. MatcherCandidateGen → MatcherTopN(5) → MatcherLLMReranker (full pipeline)

Usage
-----
    python run_experiment.py --model ~/models/Llama-3.1-8B-Instruct [OPTIONS]

Options
-------
    --model PATH          Path to a HuggingFace-compatible model directory (required)
    --prompt-id ID        Reranking prompt key (default: d)
    --description NAME    RDFGraphWrapper description method (default: description_one_gen)
    --threshold FLOAT     LLM confidence threshold for filtering (default: 0.5)
    --batch-size INT      Prompts per LLM forward pass (default: 8)
    --top-n INT           Candidates per entity kept by MatcherTopN (default: 5)
    --embedding-model STR Sentence-transformer model for MatcherCandidateGen
                          (default: all-MiniLM-L6-v2)
    --embedding-prompt-id ID  Query prompt key from EMBEDDING_PROMPTS passed to
                          MatcherCandidateGen (one/two/three/four/five or empty;
                          default: "" = no prompt)
    --timestamp STR       Fixed timestamp string for result folder (default: current time)
    --baseline-only       Run only the baseline, skip the LLM pipeline
    --pipeline-only       Run only the full pipeline, skip the baseline
    --wandb               Enable Weights & Biases logging (default: disabled)
    --wandb-project STR   W&B project name (default: olala-ontology-matching)
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from Evaluation import evaluate, run_oaei_tracks
from LLMHuggingFace import LLMHuggingFace
from MatcherCandidateGen import MatcherCandidateGen
from MatcherLLMReranker import MatcherLLMReranker
from MatcherSequential import MatcherSequential
from MatcherSimple import MatcherSimple
from MatcherTopN import MatcherTopN

ANATOMY_TRACK = ("anatomy_track", "anatomy_track-default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Anatomy-Track ontology matching experiment."
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to HuggingFace model directory for LLMHuggingFace.",
    )
    parser.add_argument(
        "--prompt-id",
        default="d",
        metavar="ID",
        help="Reranking prompt key from RERANKING_PROMPTS (default: d).",
    )
    parser.add_argument(
        "--description",
        default="description_one_gen",
        metavar="NAME",
        help="RDFGraphWrapper method used to build KG context (default: description_one_gen).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="LLM confidence threshold to keep a correspondence (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="INT",
        help="Prompts per LLM forward pass (default: 8).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="INT",
        help="Candidates per entity passed to MatcherLLMReranker (default: 5).",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        metavar="STR",
        help="Sentence-transformer model for MatcherCandidateGen (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--embedding-prompt-id",
        default="",
        metavar="ID",
        help=(
            "Query prompt key from EMBEDDING_PROMPTS for MatcherCandidateGen "
            "(one/two/three/four/five or empty string for no prompt; default: \"\")."
        ),
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        metavar="STR",
        help="Fixed timestamp string for result folder (default: current time).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the MatcherSimple baseline.",
    )
    mode.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Run only the full LLM pipeline (skip baseline).",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="olala-ontology-matching",
        metavar="STR",
        help="W&B project name (default: olala-ontology-matching).",
    )

    return parser.parse_args()


def build_systems(args: argparse.Namespace) -> list:
    systems = []

    # ── 1. Baseline ────────────────────────────────────────────────────────
    if not args.pipeline_only:
        systems.append(MatcherSimple())
        logging.info("Registered system: MatcherSimple (baseline)")

    # ── 2. Full pipeline ───────────────────────────────────────────────────
    if not args.baseline_only:
        model_path = str(Path(args.model).expanduser().resolve())
        logging.info(f"Loading LLM from {model_path} …")
        llm = LLMHuggingFace(model_path)

        candidate_gen = MatcherCandidateGen(
            model=args.embedding_model,
            description="description_one_gen",
            query_prompt_id=args.embedding_prompt_id,
        )
        top_n = MatcherTopN(n=args.top_n)
        reranker = MatcherLLMReranker(
            llm=llm,
            prompt_id=args.prompt_id,
            description=args.description,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )

        pipeline = MatcherSequential([candidate_gen, top_n, reranker])
        systems.append(pipeline)
        logging.info(f"Registered system: {pipeline}")

    return systems


def _init_wandb(args: argparse.Namespace):
    """Initialise a W&B run and return the run object (or None on failure)."""
    try:
        import wandb
    except ImportError:
        logging.error(
            "wandb is not installed. Install it with: pip install wandb"
        )
        sys.exit(1)

    model_path = str(Path(args.model).expanduser().resolve())
    run = wandb.init(
        project=args.wandb_project,
        config={
            "model_path": model_path,
            "prompt_id": args.prompt_id,
            "description": args.description,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "top_n": args.top_n,
            "embedding_model": args.embedding_model,
            "embedding_prompt_id": args.embedding_prompt_id,
            "track": ANATOMY_TRACK[0],
            "baseline_only": args.baseline_only,
            "pipeline_only": args.pipeline_only,
        },
    )
    logging.info(f"W&B run initialised: {run.url}")
    return run


def _log_results_to_wandb(results, args: argparse.Namespace) -> None:
    """Evaluate each TestCase and log metrics to the active W&B run."""
    import wandb

    model_path = str(Path(args.model).expanduser().resolve())

    for tc in results:
        precision, recall, f1, tp, fp, fn = evaluate(tc.reference, tc.alignment)

        wandb.log({
            "system_name": tc.system_name,
            "testcase_name": tc.testcase_name,
            "track_name": tc.track_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            # pipeline hyper-parameters — useful when comparing runs
            "model_path": model_path,
            "threshold": args.threshold,
            "top_n": args.top_n,
            "prompt_id": args.prompt_id,
        })
        logging.info(
            f"W&B logged — system={tc.system_name} testcase={tc.testcase_name} "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    wandb_run = None
    if args.wandb:
        wandb_run = _init_wandb(args)

    systems = build_systems(args)
    if not systems:
        logging.error("No systems to run. Check --baseline-only / --pipeline-only flags.")
        sys.exit(1)

    logging.info(
        f"Starting experiment on {ANATOMY_TRACK[0]} with {len(systems)} system(s)."
    )

    results = run_oaei_tracks(
        systems=systems,
        tracks=[ANATOMY_TRACK],
        timestamp_replacement=args.timestamp,
    )

    if wandb_run is not None:
        _log_results_to_wandb(results, args)
        wandb_run.finish()

    logging.info("Experiment complete. Results written to results/.")


if __name__ == "__main__":
    main()
