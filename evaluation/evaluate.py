"""
Evaluation using log-probability method (no generation).

For each test sample the model does a single forward pass and the predicted
answer is the option letter (A/B/C/D/E) whose first-token logit is highest
at the last prompt position.

Final metrics are logged to W&B via wandb_logger.
"""

import os
import json
import csv
import torch
import pandas as pd
from datasets import Dataset

from data.read_data import load_medarabench
from data.clean_data import clean_medarabench
from utils.prompt_template import format_medarabench_sample
from utils.metrics import compute_all_metrics, compute_per_specialty_metrics
from utils.model_registry import get_spec, resolve_answer_token_ids
from utils import wandb_logger


# ---------------------------------------------------------------------------
# Token-ID helpers
# ---------------------------------------------------------------------------

def _get_answer_token_ids(tokenizer, model_spec=None) -> dict[str, int]:
    """
    Resolve and verify the single-token ID for each answer letter A-E.
    Delegates to the model registry, which raises if any letter does not
    encode to exactly one token (see utils/model_registry.py).
    """
    return resolve_answer_token_ids(tokenizer, spec=model_spec)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    test_dataset: Dataset,
    output_dir: str,
    batch_size: int = 16,
    data_dir: str = "Files/datasets/",
    run_name: str | None = None,
    predictions_dir: str | None = None,
    model_spec=None,
):
    """
    Evaluate a model on the MedAraBench test set using log-probability selection.

    Args:
        model:           loaded (and optionally merged) causal LM
        tokenizer:       corresponding tokenizer
        test_dataset:    HuggingFace Dataset with MedAraBench test samples
        output_dir:      directory to write results.json and predictions.csv
        batch_size:      number of samples per inference batch
        data_dir:        dataset root (unused here, kept for API consistency)
        run_name:        if set, per-sample predictions (with raw A-E logits)
                         are written to {predictions_dir}/{run_name}.parquet
        predictions_dir: root for parquet prediction files
                         (default: <project>/predictions)
        model_spec:      ModelSpec for answer-token resolution (optional)

    Returns:
        dict with 'accuracy', 'macro_f1', and per-specialty scores
    """
    model.eval()

    # Left-padding for batched inference (align right edge of sequences)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    token_ids = _get_answer_token_ids(tokenizer, model_spec=model_spec)

    # Determine which answer letters appear in this test set
    all_answers = set(test_dataset["answer"])
    active_letters = sorted(
        [l for l in ["A", "B", "C", "D", "E"] if l in all_answers or l in token_ids]
    )
    active_token_ids = [token_ids[l] for l in active_letters]

    predictions: list[str] = []
    references: list[str] = []
    specialties: list[str] = []
    per_sample_records: list[dict] = []

    samples = [test_dataset[i] for i in range(len(test_dataset))]

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        prompts = [format_medarabench_sample(s, include_answer=False) for s in batch]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]

        answer_logits = last_token_logits[:, active_token_ids]  # [batch, num_letters]
        predicted_indices = answer_logits.argmax(dim=-1).cpu().tolist()
        raw_logits = answer_logits.float().cpu().tolist()  # [batch][num_letters]

        for i, sample in enumerate(batch):
            pred_letter = active_letters[predicted_indices[i]]
            ref_letter = str(sample.get("answer", "")).strip().upper()

            predictions.append(pred_letter)
            references.append(ref_letter)
            specialties.append(
                str(sample.get("specialty", sample.get("umbrella_specialty", "Unknown")))
            )

            record = {
                "question_id": sample.get("question_id", batch_start + i),
                "question": sample.get("question", ""),
                "reference": ref_letter,
                "prediction": pred_letter,
                "correct": pred_letter == ref_letter,
                "specialty": specialties[-1],
                "umbrella_specialty": str(sample.get("umbrella_specialty", "Unknown")),
                "level": str(sample.get("level", "Unknown")),
            }
            for j, letter in enumerate(active_letters):
                record[f"logit_{letter}"] = raw_logits[i][j]
            per_sample_records.append(record)

        if (batch_start // batch_size) % 20 == 0:
            done = min(batch_start + batch_size, len(samples))
            print(f"  Evaluated {done}/{len(samples)} samples...")

    tokenizer.padding_side = original_padding_side

    # ------------------------------------------------------------------ #
    # Compute metrics
    # ------------------------------------------------------------------ #
    overall = compute_all_metrics(predictions, references)
    per_spec = compute_per_specialty_metrics(predictions, references, specialties)

    results = {
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "total_samples": len(predictions),
        "per_specialty_scores": per_spec,
        "answer_token_ids": {l: token_ids[l] for l in active_letters},
    }

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Macro F1 : {results['macro_f1']:.4f}")
    print(f"  Samples  : {results['total_samples']}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------ #
    # Log to W&B
    # ------------------------------------------------------------------ #
    wandb_logger.log_eval_metrics(
        accuracy=overall["accuracy"],
        macro_f1=overall["macro_f1"],
        per_specialty=per_spec,
        stage="eval",
    )

    # ------------------------------------------------------------------ #
    # Save to disk
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_path}")

    preds_path = os.path.join(output_dir, "predictions.csv")
    fieldnames = ["question", "reference", "prediction", "correct", "specialty"]
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(per_sample_records)
    print(f"Predictions saved to: {preds_path}")

    # ------------------------------------------------------------------ #
    # Per-sample parquet with raw answer logits (for statistical testing)
    # ------------------------------------------------------------------ #
    if run_name is not None:
        parquet_path = save_predictions_parquet(
            per_sample_records, run_name, active_letters, predictions_dir
        )
        results["predictions_parquet"] = parquet_path

    return results


def save_predictions_parquet(
    per_sample_records: list[dict],
    run_name: str,
    letters: list[str],
    predictions_dir: str | None = None,
) -> str:
    """
    Write predictions/{run_name}.parquet with one row per test sample:
    question_id, gold_label, pred_label, logit_A..logit_E (+ metadata columns
    specialty, umbrella_specialty, level used by scripts/breakdowns.py).
    """
    if predictions_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictions_dir = os.path.join(project_root, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    df = pd.DataFrame(per_sample_records)
    df = df.rename(columns={"reference": "gold_label", "prediction": "pred_label"})
    cols = (["question_id", "gold_label", "pred_label"]
            + [f"logit_{l}" for l in letters]
            + ["specialty", "umbrella_specialty", "level"])
    df = df[[c for c in cols if c in df.columns]]

    parquet_path = os.path.join(predictions_dir, f"{run_name}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Per-sample prediction parquet saved to: {parquet_path}")
    return parquet_path


def run_evaluation(
    model,
    tokenizer,
    output_dir: str,
    batch_size: int = 16,
    data_dir: str = "Files/datasets/",
    max_samples: int | None = None,
    run_name: str | None = None,
    predictions_dir: str | None = None,
    model_spec=None,
):
    """
    Convenience wrapper: loads the MedAraBench test set and evaluates.

    Args:
        model:           loaded causal LM (merged if LoRA, ready for inference)
        tokenizer:       corresponding tokenizer
        output_dir:      directory to write eval results
        batch_size:      inference batch size
        data_dir:        dataset root
        max_samples:     cap test set size (for dry-run / smoke tests)
        run_name:        if set, save predictions/{run_name}.parquet with raw logits
        predictions_dir: root for parquet prediction files
        model_spec:      ModelSpec for answer-token resolution

    Returns:
        dict with evaluation metrics
    """
    print("Loading and cleaning MedAraBench test set...")
    raw_test = load_medarabench(split="test", data_dir=data_dir)
    test_dataset = clean_medarabench(raw_test)
    if max_samples is not None:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    print(f"  Test samples: {len(raw_test):,} raw → {len(test_dataset):,} clean")

    return evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        data_dir=data_dir,
        run_name=run_name,
        predictions_dir=predictions_dir,
        model_spec=model_spec,
    )
