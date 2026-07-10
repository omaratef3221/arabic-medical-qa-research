"""
Revision-R1 orchestrator: iterates the full revision run matrix, skips every
configuration that already exists (finished W&B run with the same name OR a
local eval/results.json), and launches the rest sequentially. Kill it and
restart it at any time - completed runs are never re-run.

Phases:
  stats       Task 1: inference-only prediction dumps for existing checkpoints
              (scripts/reinfer_all.py), then scripts/stats.py and
              scripts/breakdowns.py. No training.
  seeds       Task 2: seeds 1337/2024 for the four Stage-2-only configs.
  sweep       Task 3: LoRA rank sweep, LR/dropout one-factor ablations at the
              best rank, attention-only ablation, QLoRA. Best configurations
              are re-resolved from W&B between stages.
  newmodels   Task 4: Fanar-1-9B / SILMA-9B / Qwen2.5-7B x
              {zero-shot, Stage-2 LoRA (best sweep config), Stage-2 Full}.
  stage1diag  Task 5: Stage-1-only checkpoint evals + AraMed perplexity
              report + the Jais Stage-1-full 3-epoch ablation.
  all         Everything above, in that order.

Every launched run logs to W&B ({model}_s1-{s1}_s2-{s2}[_suffix] naming)
tagged `revision-r1`, and writes predictions/{run_name}.parquet.

Usage:
  python scripts/run_revision.py --phase all --dry-run   # plan only
  python scripts/run_revision.py --phase seeds           # execute
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (  # noqa: E402
    PROJECT_ROOT, RunSpec, fetch_wandb_runs, run_completed,
    seeds_matrix, sweep_rank_matrix, sweep_lr_dropout_matrix,
    sweep_attnonly_matrix, qlora_matrix, newmodels_matrix, stage1diag_matrix,
)

PHASES = ["stats", "seeds", "sweep", "newmodels", "stage1diag", "all"]


# ---------------------------------------------------------------------------
# GPU memory sanity check (the trainable-parameter-percentage assertion runs
# inside train/finetuning.py right before training; this is the pre-launch
# check that the GPUs are actually free)
# ---------------------------------------------------------------------------

def gpu_memory_check() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            print("  [gpu-check] nvidia-smi unavailable; skipping check")
            return True
        ok = True
        for line in out.stdout.strip().splitlines():
            idx, used, total = [int(x.strip()) for x in line.split(",")]
            free = total - used
            flag = "" if free > 4000 else "  <-- LOW FREE MEMORY"
            print(f"  [gpu-check] GPU {idx}: {used}/{total} MiB used{flag}")
            if free <= 4000:
                ok = False
        if not ok:
            print("  [gpu-check] WARNING: a GPU has <4 GiB free; "
                  "a previous process may still be running.")
        return ok
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  [gpu-check] nvidia-smi not found (non-GPU host?); skipping check")
        return True


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def plan_specs(specs: list, results: list) -> list:
    """Print skip/launch status for each spec; return the ones to launch."""
    to_launch = []
    for spec in specs:
        done, reason = run_completed(spec.run_name)
        if done:
            print(f"SKIP (exists): {spec.run_name}  [{reason}]")
            results.append(("SKIP", spec))
        else:
            kind = "train+eval" if spec.train else "eval-only"
            note = f"  -- {spec.note}" if spec.note else ""
            print(f"LAUNCH ({kind}): {spec.run_name}{note}")
            results.append(("LAUNCH", spec))
            to_launch.append(spec)
    return to_launch


def execute_specs(specs: list, eval_batch_size: int, failures: list):
    for i, spec in enumerate(specs, 1):
        # Re-check right before launching: an earlier restart of this
        # orchestrator may have completed the run in the meantime.
        done, reason = run_completed(spec.run_name)
        if done:
            print(f"SKIP (exists): {spec.run_name}  [{reason}]")
            continue
        print(f"\n{'='*70}\n[{i}/{len(specs)}] {spec.run_name}\n{'='*70}")
        gpu_memory_check()
        cmd = ([sys.executable, os.path.join(PROJECT_ROOT, "main.py")]
               + spec.main_py_args()
               + ["--eval_batch_size", str(eval_batch_size)])
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"FAILED: {spec.run_name} (exit {result.returncode}) -- continuing")
            failures.append(spec.run_name)


def run_script(script: str, extra: list, dry_run: bool):
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script)] + extra
    if dry_run and script in ("reinfer_all.py", "eval_stage1_checkpoints.py"):
        cmd.append("--dry-run")
    elif dry_run:
        print(f"WOULD RUN: {' '.join(cmd)}")
        return 0
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_stats(args, results, failures):
    print(f"\n{'#'*70}\n# Phase: stats (Task 1 - no training)\n{'#'*70}")
    run_script("reinfer_all.py",
               ["--eval_batch_size", str(args.eval_batch_size)], args.dry_run)
    run_script("stats.py", [], args.dry_run)
    run_script("breakdowns.py", [], args.dry_run)


def phase_seeds(args, results, failures):
    print(f"\n{'#'*70}\n# Phase: seeds (Task 2 - seeds 1337/2024)\n{'#'*70}")
    to_launch = plan_specs(seeds_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)


def phase_sweep(args, results, failures):
    print(f"\n{'#'*70}\n# Phase: sweep (Task 3 - LoRA sweep + QLoRA)\n{'#'*70}")

    print("\n--- 3.1 rank sweep: r in {8, 32, 64}, alpha = 2r ---")
    to_launch = plan_specs(sweep_rank_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)
        fetch_wandb_runs(force=True)  # rank results feed the next stages

    print("\n--- 3.2 LR / dropout one-factor ablations at best rank ---")
    to_launch = plan_specs(sweep_lr_dropout_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)
        fetch_wandb_runs(force=True)

    print("\n--- 3.3 attention-only target-module ablation ---")
    to_launch = plan_specs(sweep_attnonly_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)
        fetch_wandb_runs(force=True)

    print("\n--- 3.4 QLoRA (NF4 + double quant + paged AdamW) ---")
    to_launch = plan_specs(qlora_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)


def phase_newmodels(args, results, failures):
    print(f"\n{'#'*70}\n# Phase: newmodels (Task 4 - Fanar / SILMA / Qwen)\n{'#'*70}")
    to_launch = plan_specs(newmodels_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)


def phase_stage1diag(args, results, failures):
    print(f"\n{'#'*70}\n# Phase: stage1diag (Task 5 - Stage 1 diagnostics)\n{'#'*70}")
    run_script("eval_stage1_checkpoints.py",
               ["--perplexity", "--eval_batch_size", str(args.eval_batch_size)],
               args.dry_run)
    to_launch = plan_specs(stage1diag_matrix(), results)
    if not args.dry_run:
        execute_specs(to_launch, args.eval_batch_size, failures)


PHASE_FUNCS = {
    "stats": phase_stats,
    "seeds": phase_seeds,
    "sweep": phase_sweep,
    "newmodels": phase_newmodels,
    "stage1diag": phase_stage1diag,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the launch/skip plan without running anything.")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"W&B project: {common.WANDB_ENTITY}/{common.WANDB_PROJECT}  "
          f"(revision tag: {common.REVISION_TAG})")
    print("Querying W&B for finished runs...")
    runs = fetch_wandb_runs()
    finished = sum(1 for r in runs.values() if r["state"] == "finished")
    print(f"  {len(runs)} runs found, {finished} finished.")

    results: list = []   # (status, spec) for the summary
    failures: list = []

    phases = ["stats", "seeds", "sweep", "newmodels", "stage1diag"] \
        if args.phase == "all" else [args.phase]
    for phase in phases:
        PHASE_FUNCS[phase](args, results, failures)

    n_skip = sum(1 for s, _ in results if s == "SKIP")
    n_launch = sum(1 for s, _ in results if s == "LAUNCH")
    print(f"\n{'='*70}")
    if args.dry_run:
        print(f"DRY RUN SUMMARY: {n_launch} runs would be launched, "
              f"{n_skip} skipped (already exist).")
    else:
        print(f"SUMMARY: {n_launch} runs launched, {n_skip} skipped, "
              f"{len(failures)} failed.")
        if failures:
            print("Failed runs (re-run this orchestrator to retry):")
            for f in failures:
                print(f"  - {f}")
    print("After all phases: python scripts/make_paper_tables.py regenerates "
          "all LaTeX tables and figures from the prediction files.")


if __name__ == "__main__":
    main()
