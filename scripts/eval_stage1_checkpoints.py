"""
Stage 1 diagnostics (Task 5): catastrophic forgetting + perplexity.

1. Evaluates every existing Stage-1-only checkpoint (outputs/*/stage1,
   before any Stage 2) zero-shot on the MedAraBench test set, saving
   per-sample predictions as predictions/{model}_s1-{method}_s2-none.parquet.
   This measures whether domain adaptation alone helps or hurts MCQ accuracy
   (catastrophic forgetting test). Inference only - nothing is trained.

2. With --perplexity: computes AraMed validation perplexity (same 2% split,
   seed 42, answer-token masking as in training) for the base model (before)
   and each Stage 1 checkpoint (after), logs the table to a dedicated W&B run
   tagged revision-r1, and writes results/stage1_perplexity.csv.

Checkpoints are discovered from outputs/*/stage1/training_args.json and
deduplicated by (model, method) - exp01/exp03 style duplicates trained with
identical config and seed collapse to one entry.

Usage:
  python scripts/eval_stage1_checkpoints.py [--dry-run] [--perplexity]
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    PROJECT_ROOT, OUTPUTS_DIR, RESULTS_DIR, REVISION_TAG, WANDB_PROJECT,
    has_predictions, predictions_path,
)
from utils.model_registry import short_name_for  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_stage1_checkpoints() -> dict[tuple, str]:
    """
    Return {(model_name, method): stage1_dir} for every unique Stage 1
    checkpoint under outputs/. Prefers training_args.json metadata; falls
    back to adapter_config.json presence + directory-name heuristics.
    """
    found: dict[tuple, str] = {}
    if not os.path.isdir(OUTPUTS_DIR):
        return found

    for entry in sorted(os.listdir(OUTPUTS_DIR)):
        stage1_dir = os.path.join(OUTPUTS_DIR, entry, "stage1")
        if not os.path.isdir(stage1_dir):
            continue
        meta_path = os.path.join(stage1_dir, "training_args.json")
        model_name, method = None, None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                model_name = meta.get("model_name")
                method = meta.get("method")
            except (json.JSONDecodeError, OSError):
                pass
        if model_name is None or method is None:
            # Heuristic fallback for checkpoints saved without metadata
            method = ("lora" if os.path.isfile(
                os.path.join(stage1_dir, "adapter_config.json")) else "full")
            lowered = entry.lower()
            if "llama" in lowered:
                model_name = "meta-llama/Llama-3.1-8B"
            elif "jais" in lowered:
                model_name = "inceptionai/Jais-2-8B-Chat"
            else:
                print(f"  WARNING: cannot identify model for {stage1_dir}; skipped")
                continue
        key = (model_name, method)
        if key not in found:
            found[key] = stage1_dir
    return found


# ---------------------------------------------------------------------------
# Zero-shot MedAraBench evaluation of Stage 1 checkpoints (via main.py)
# ---------------------------------------------------------------------------

def eval_checkpoint_cmd(model_name: str, method: str, ckpt: str,
                        run_name: str, batch_size: int) -> list:
    output_dir = os.path.join(OUTPUTS_DIR, run_name)
    return [
        sys.executable, os.path.join(PROJECT_ROOT, "main.py"),
        "--model", model_name,
        "--stage1_method", method,
        "--stage2_method", "none",
        "--stage1_checkpoint", ckpt,
        "--output_dir", output_dir,
        "--run_name", run_name,
        "--eval_batch_size", str(batch_size),
        "--do_eval",
        "--no_wandb",
    ]


# ---------------------------------------------------------------------------
# AraMed validation perplexity
# ---------------------------------------------------------------------------

def compute_aramed_perplexity(model, tokenizer, max_samples: int | None,
                              max_seq_length: int = 512) -> float:
    """
    Perplexity over the answer tokens of the AraMed validation split
    (2% of train, seed 42 - identical to the Stage 1 training split).
    """
    import torch
    from data.read_data import load_aramed
    from utils.prompt_template import _tokenize_aramed

    full = load_aramed(split="train", data_dir=os.path.join(PROJECT_ROOT, "Files/datasets/"))
    val = full.train_test_split(test_size=0.02, seed=42)["test"]
    if max_samples is not None:
        val = val.select(range(min(max_samples, len(val))))

    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(len(val)):
            enc = _tokenize_aramed(val[i], tokenizer, max_seq_length)
            input_ids = torch.tensor([enc["input_ids"]], device=model.device)
            labels = torch.tensor([enc["labels"]], device=model.device)
            n_target = int((labels != -100).sum())
            if n_target == 0:
                continue
            out = model(input_ids=input_ids, labels=labels)
            total_nll += float(out.loss) * n_target
            total_tokens += n_target
            if (i + 1) % 200 == 0:
                print(f"    perplexity: {i + 1}/{len(val)} samples...")
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def run_perplexity_report(checkpoints: dict, max_samples: int | None,
                          no_wandb: bool):
    import torch
    from utils.get_model import load_model_and_tokenizer, load_from_checkpoint

    rows = []
    base_ppl_cache: dict[str, float] = {}

    for (model_name, method), ckpt in sorted(checkpoints.items()):
        short = short_name_for(model_name)

        if model_name not in base_ppl_cache:
            print(f"\n[ppl] Base model (before Stage 1): {model_name}")
            model, tokenizer = load_model_and_tokenizer(model_name, method="full")
            for p in model.parameters():
                p.requires_grad = False
            base_ppl_cache[model_name] = compute_aramed_perplexity(
                model, tokenizer, max_samples)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[ppl]   before = {base_ppl_cache[model_name]:.3f}")

        print(f"\n[ppl] Stage 1 checkpoint: {short} s1-{method} ({ckpt})")
        model, tokenizer = load_from_checkpoint(
            checkpoint_path=ckpt, base_model_name=model_name, method=method)
        after = compute_aramed_perplexity(model, tokenizer, max_samples)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[ppl]   after = {after:.3f}")

        rows.append({
            "model": short, "stage1_method": method,
            "ppl_before": base_ppl_cache[model_name], "ppl_after": after,
            "checkpoint": ckpt,
        })

    import pandas as pd
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "stage1_perplexity.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nPerplexity CSV written to: {csv_path}")

    if not no_wandb:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            name="stage1_perplexity_report",
            tags=[REVISION_TAG, "stage1-diagnostics"],
            resume="allow",
            notes="AraMed validation perplexity before/after Stage 1 domain adaptation",
        )
        table = wandb.Table(
            columns=["Model", "Stage 1 method", "PPL before", "PPL after"],
            data=[[r["model"], r["stage1_method"],
                   round(r["ppl_before"], 3), round(r["ppl_after"], 3)]
                  for r in rows],
        )
        wandb.log({"stage1/perplexity": table})
        for r in rows:
            wandb.summary[f"ppl_before/{r['model']}"] = r["ppl_before"]
            wandb.summary[f"ppl_after/{r['model']}_s1-{r['stage1_method']}"] = r["ppl_after"]
        run.finish()
        print("Perplexity table logged to W&B: stage1_perplexity_report")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--perplexity", action="store_true",
                        help="Also compute AraMed validation perplexity before/after Stage 1.")
    parser.add_argument("--max_ppl_samples", type=int, default=None,
                        help="Cap AraMed validation samples for perplexity (default: all).")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    checkpoints = discover_stage1_checkpoints()
    print(f"Discovered {len(checkpoints)} unique Stage 1 checkpoints:")
    for (model_name, method), ckpt in sorted(checkpoints.items()):
        print(f"  {short_name_for(model_name)} s1-{method}: {ckpt}")

    plan = []
    for (model_name, method), ckpt in sorted(checkpoints.items()):
        run_name = f"{short_name_for(model_name)}_s1-{method}_s2-none"
        if has_predictions(run_name):
            print(f"SKIP (exists): {run_name}")
            continue
        print(f"EVAL: {run_name} -> {predictions_path(run_name)}")
        plan.append((model_name, method, ckpt, run_name))

    if args.dry_run:
        print(f"\nDry run: {len(plan)} Stage-1 checkpoint evals would run"
              + (", plus a perplexity report" if args.perplexity else "") + ".")
        return

    for i, (model_name, method, ckpt, run_name) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] Evaluating {run_name} ...")
        result = subprocess.run(
            eval_checkpoint_cmd(model_name, method, ckpt, run_name,
                                args.eval_batch_size),
            cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"FAILED: {run_name} (exit {result.returncode}) -- continuing")

    if args.perplexity:
        run_perplexity_report(checkpoints, args.max_ppl_samples, args.no_wandb)


if __name__ == "__main__":
    main()
