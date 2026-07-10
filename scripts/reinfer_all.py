"""
Inference-only re-evaluation of every existing checkpoint.

Loads each of the 14 pre-revision conditions (12 trained + 2 zero-shot
baselines for Llama-3.1-8B and Jais-2-8B-Chat) and runs evaluation ONLY, to
produce the per-sample prediction files predictions/{run_name}.parquet with
question_id, gold_label, pred_label and the raw A-E logits. Nothing is
retrained and no W&B runs are created or modified (evals run with --no_wandb).

Additionally (--scan, on by default) any outputs/<run_name> directory whose
name matches the run-naming convention and whose parquet is missing gets
re-inferred too, so revision runs that crashed between eval and parquet
writing are repaired automatically.

Resumable: conditions whose parquet already exists are skipped with
"SKIP (exists): <run_name>".

Usage:
  python scripts/reinfer_all.py [--dry-run] [--no-scan] [--eval_batch_size 16]
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    PROJECT_ROOT, OUTPUTS_DIR,
    existing_conditions_matrix, has_predictions, predictions_path,
    resolve_checkpoint_dir, RunSpec,
)
from utils.model_registry import REGISTRY  # noqa: E402

RUN_NAME_RE = re.compile(
    r"^(?P<short>.+)_s1-(?P<s1>none|lora|full|full3ep)"
    r"_s2-(?P<s2>none|lora|full|qlora)(?P<suffix>_.*)?$"
)


def parse_run_name(run_name: str) -> RunSpec | None:
    """Reconstruct a RunSpec (for eval only) from a run-name string."""
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return None
    short = m.group("short")
    model = next((hf_id for hf_id, spec in REGISTRY.items()
                  if spec.short_name == short), None)
    if model is None:
        return None
    s1 = "full" if m.group("s1") == "full3ep" else m.group("s1")
    s2 = m.group("s2")
    extra = []
    if s2 == "qlora":
        s2 = "lora"
        extra.append("--qlora")
    return RunSpec(run_name=run_name, model=model, stage1_method=s1,
                   stage2_method=s2, phase="reinfer", train=False,
                   extra_args=extra, note="scanned from outputs/")


def eval_args(spec: RunSpec, batch_size: int) -> list:
    """main.py invocation for an inference-only pass (no W&B, no training)."""
    return [
        sys.executable, os.path.join(PROJECT_ROOT, "main.py"),
        "--model", spec.model,
        "--stage1_method", spec.stage1_method,
        "--stage2_method", spec.stage2_method,
        "--output_dir", spec.output_dir,
        "--run_name", spec.run_name,
        "--eval_batch_size", str(batch_size),
        "--do_eval",
        "--no_wandb",
    ] + [a for a in spec.extra_args if a == "--qlora"]


def collect_specs(scan: bool) -> list:
    specs = {s.run_name: s for s in existing_conditions_matrix()}
    if scan and os.path.isdir(OUTPUTS_DIR):
        for entry in sorted(os.listdir(OUTPUTS_DIR)):
            parsed = parse_run_name(entry)
            if parsed and parsed.run_name not in specs:
                specs[parsed.run_name] = parsed
    return list(specs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without loading any model.")
    parser.add_argument("--no-scan", action="store_true",
                        help="Only the 14 pre-revision conditions; skip the outputs/ scan.")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()

    specs = collect_specs(scan=not args.no_scan)
    to_run, failures = [], []

    print(f"\n{'='*70}\nRe-inference plan ({len(specs)} conditions)\n{'='*70}")
    for spec in specs:
        if has_predictions(spec.run_name):
            print(f"SKIP (exists): {spec.run_name}")
            continue
        needs_ckpt = spec.stage2_method != "none" or spec.stage1_method != "none"
        if needs_ckpt:
            stage = "stage2" if spec.stage2_method != "none" else "stage1"
            ckpt = resolve_checkpoint_dir(spec.run_name, stage)
            if ckpt is None:
                print(f"MISSING CHECKPOINT: {spec.run_name} "
                      f"(no {stage}/ under {spec.output_dir}) -- skipped")
                failures.append(spec.run_name)
                continue
        print(f"REINFER: {spec.run_name} -> {predictions_path(spec.run_name)}")
        to_run.append(spec)

    if args.dry_run:
        print(f"\nDry run: {len(to_run)} conditions would be re-inferred, "
              f"{len(failures)} unresolvable.")
        return

    succeeded = 0
    for i, spec in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] Re-inferring {spec.run_name} ...")
        cmd = eval_args(spec, args.eval_batch_size)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"[reinfer_all] FAILED: {spec.run_name} "
                  f"(exit {result.returncode}) -- continuing")
            failures.append(spec.run_name)
        else:
            succeeded += 1

    print(f"\nDone. {succeeded} succeeded, {len(failures)} failed/missing.")
    if failures:
        print("Unresolved conditions:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
