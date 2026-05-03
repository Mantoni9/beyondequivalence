"""
cleanup_wandb_runs.py — delete W&B runs that match a specific
(group, git_sha) combination.

Use case: a re-launched sweep produced a handful of runs under the wrong
SHA before being scancelled, and those runs need to disappear so the W&B
group stays clean.

Defaults to dry-run. Pass --delete to actually remove them.

    conda run -n melt-olala python cleanup_wandb_runs.py \\
        --entity antonio-markic-university-of-mannheim \\
        --project beyondequivalence-retrieval-stage1 \\
        --group subB_descablation_2026-05-03_12-18-11_2d92b24 \\
        --git-sha ecd16dc

    # When the dry-run output looks right:
    conda run -n melt-olala python cleanup_wandb_runs.py ... --delete
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Delete W&B runs by (group, git_sha).")
    p.add_argument("--entity",  required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group",   required=True,
                   help="W&B group; only runs in this group are eligible.")
    p.add_argument("--git-sha", required=True,
                   help="Match runs whose config.git_sha equals this. Required "
                        "to keep cleanup narrow — never delete runs without an "
                        "exact SHA filter.")
    p.add_argument("--delete", action="store_true",
                   help="Actually delete. Without this flag, runs only "
                        "the dry-run preview.")
    args = p.parse_args()

    try:
        import wandb  # noqa: F401
        from wandb.apis.public import Api
    except ImportError:
        sys.exit("wandb not installed — install via the conda env first.")

    api = Api()
    runs = api.runs(
        path=f"{args.entity}/{args.project}",
        filters={"group": args.group, "config.git_sha": args.git_sha},
        per_page=200,
    )

    matched = list(runs)
    if not matched:
        print(f"No matches for group={args.group!r} git_sha={args.git_sha!r}.")
        return

    print(f"Matched {len(matched)} run(s) in {args.entity}/{args.project}:")
    for r in matched:
        ds = r.config.get("dataset", "?")
        model = r.config.get("model_arg") or r.config.get("model", "?")
        variant = r.config.get("instruction_variant", "?")
        desc = r.config.get("description", "?")
        tid = r.config.get("template_id", "—")
        state = r.state
        print(f"  [{state:>10s}] {r.name}  ({model}/{variant}/{desc}/{tid}/{ds})  id={r.id}")

    if not args.delete:
        print()
        print(f"DRY-RUN. Re-run with --delete to remove these {len(matched)} run(s).")
        return

    print()
    print(f"Deleting {len(matched)} run(s) ...")
    n_ok = n_fail = 0
    for r in matched:
        try:
            r.delete(delete_artifacts=True)
            n_ok += 1
        except Exception as e:
            print(f"  FAIL: {r.name}: {e}")
            n_fail += 1
    print(f"Done. deleted={n_ok}  failed={n_fail}")


if __name__ == "__main__":
    main()
