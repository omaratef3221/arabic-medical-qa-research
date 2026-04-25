"""
Weights & Biases logging utilities.

Centralises all W&B interactions so training scripts stay clean.
Logs:
  - Experiment config and hyperparameters (as W&B config)
  - Dataset statistics (sizes, cleaning stats) as a W&B Table + summary
  - Training loss curve (via Trainer's built-in report_to="wandb")
  - Validation loss after each epoch (logged manually)
  - Final evaluation metrics (accuracy, macro-F1, per-specialty breakdown)
  - HF Hub repo links (as W&B summary fields)
"""

import os
from typing import Any


def _wandb():
    """Lazy import so the module loads even if wandb isn't installed."""
    try:
        import wandb
        return wandb
    except ImportError:
        raise ImportError("wandb is not installed. Run: pip install wandb")


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def init_run(
    project: str,
    run_name: str,
    config: dict,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> Any:
    """
    Initialise (or resume) a W&B run.

    Call once at the start of main.py, before any training.

    Args:
        project:  W&B project name
        run_name: human-readable run name, e.g. "exp01-llama-lora-lora"
        config:   flat dict of all hyperparameters and experiment settings
        tags:     list of string tags
        notes:    free-text notes attached to the run

    Returns:
        The wandb.Run object (or a no-op stub if wandb is disabled).
    """
    wandb = _wandb()
    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags or [],
        notes=notes or "",
        resume="allow",
    )
    return run


def finish_run():
    """Mark the W&B run as finished."""
    wandb = _wandb()
    if wandb.run is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def log_dataset_stats(
    aramed_train_size: int,
    aramed_test_size: int,
    medarabench_raw_size: int,
    medarabench_clean_size: int,
    medarabench_test_size: int,
    cleaning_breakdown: dict | None = None,
):
    """
    Log dataset sizes and cleaning statistics to W&B.

    Creates:
      - A W&B Table with one row per dataset split
      - Individual summary scalars for quick access
      - Optional cleaning breakdown table

    Args:
        aramed_train_size:       number of AraMed training samples
        aramed_test_size:        number of AraMed test samples
        medarabench_raw_size:    MedAraBench train before cleaning
        medarabench_clean_size:  MedAraBench train after cleaning
        medarabench_test_size:   MedAraBench test set size
        cleaning_breakdown:      dict with keys like 'invalid_answers',
                                 'empty_questions', 'missing_option_e', 'duplicates'
    """
    wandb = _wandb()
    if wandb.run is None:
        return

    # Summary table
    table = wandb.Table(
        columns=["Dataset", "Split", "Samples", "Notes"],
        data=[
            ["AraMed", "train", aramed_train_size, "Open-ended Arabic medical QA (Stage 1)"],
            ["AraMed", "test", aramed_test_size, "Held-out AraMed samples"],
            ["MedAraBench", "train (raw)", medarabench_raw_size, "Before cleaning"],
            ["MedAraBench", "train (clean)", medarabench_clean_size, "After dedup + answer validation"],
            ["MedAraBench", "test", medarabench_test_size, "Evaluation benchmark"],
        ],
    )
    wandb.log({"dataset/overview": table})

    # Scalars for quick dashboard access
    wandb.summary.update({
        "data/aramed_train": aramed_train_size,
        "data/aramed_test": aramed_test_size,
        "data/medarabench_raw": medarabench_raw_size,
        "data/medarabench_clean": medarabench_clean_size,
        "data/medarabench_test": medarabench_test_size,
        "data/samples_removed": medarabench_raw_size - medarabench_clean_size,
        "data/removal_pct": round(
            (medarabench_raw_size - medarabench_clean_size) / medarabench_raw_size * 100, 2
        ),
    })

    if cleaning_breakdown:
        clean_table = wandb.Table(
            columns=["Reason", "Removed"],
            data=[[k.replace("_", " ").title(), v] for k, v in cleaning_breakdown.items()],
        )
        wandb.log({"dataset/cleaning_breakdown": clean_table})


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

def log_validation_loss(val_loss: float, epoch: int, stage: str):
    """
    Log validation loss for a given epoch and stage.

    Args:
        val_loss: float loss value
        epoch:    1-indexed epoch number
        stage:    "stage1" or "stage2"
    """
    wandb = _wandb()
    if wandb.run is None:
        return
    wandb.log({f"{stage}/val_loss": val_loss, f"{stage}/epoch": epoch})


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def log_eval_metrics(
    accuracy: float,
    macro_f1: float,
    per_specialty: dict,
    stage: str = "eval",
):
    """
    Log final evaluation metrics to W&B.

    Args:
        accuracy:      overall accuracy (0–1)
        macro_f1:      overall macro F1 (0–1)
        per_specialty: dict mapping specialty → {accuracy, macro_f1, count}
        stage:         prefix for metric keys, e.g. "eval" or "stage2/eval"
    """
    wandb = _wandb()
    if wandb.run is None:
        return

    # Top-level metrics
    wandb.log({
        f"{stage}/accuracy": accuracy,
        f"{stage}/macro_f1": macro_f1,
    })
    wandb.summary.update({
        f"{stage}/accuracy": accuracy,
        f"{stage}/macro_f1": macro_f1,
    })

    # Per-specialty breakdown as a W&B Table
    rows = []
    for spec, vals in sorted(per_specialty.items()):
        rows.append([spec, vals["count"], round(vals["accuracy"], 4), round(vals["macro_f1"], 4)])

    if rows:
        spec_table = wandb.Table(
            columns=["Specialty", "Count", "Accuracy", "Macro F1"],
            data=rows,
        )
        wandb.log({f"{stage}/per_specialty": spec_table})

        # Also log individual specialty accuracy as a bar chart-friendly dict
        specialty_acc = {f"{stage}/specialty/{s[0]}": s[2] for s in rows}
        wandb.log(specialty_acc)


# ---------------------------------------------------------------------------
# HF Hub link
# ---------------------------------------------------------------------------

def log_hf_repo(repo_id: str, stage: str):
    """
    Record the HuggingFace Hub repo link in the W&B run summary.

    Args:
        repo_id: full HF repo ID, e.g. "your-org/llama-3.1-8b-lora-lora-stage2"
        stage:   "stage1" or "stage2"
    """
    wandb = _wandb()
    if wandb.run is None:
        return
    url = f"https://huggingface.co/{repo_id}"
    wandb.summary.update({f"hf_hub/{stage}_repo": repo_id, f"hf_hub/{stage}_url": url})
    print(f"W&B: logged HF Hub link for {stage}: {url}")
