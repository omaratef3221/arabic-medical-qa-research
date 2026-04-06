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
from datasets import Dataset

from data.read_data import load_medarabench
from utils.prompt_template import format_medarabench_sample
from utils.metrics import compute_all_metrics, compute_per_specialty_metrics
from utils import wandb_logger


# ---------------------------------------------------------------------------
# Token-ID helpers
# ---------------------------------------------------------------------------

def _get_answer_token_ids(tokenizer) -> dict[str, int]:
    """
    Return the first token ID for each answer letter.

    Tries both plain ("A") and space-prefixed (" A") variants and picks
    whichever is a single-token encoding.  Falls back to the plain variant.
    """
    candidates = ["A", "B", "C", "D", "E"]
    token_ids: dict[str, int] = {}

    for letter in candidates:
        plain_ids = tokenizer.encode(letter, add_special_tokens=False)
        space_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)

        if len(plain_ids) == 1:
            token_ids[letter] = plain_ids[0]
        elif len(space_ids) == 1:
            token_ids[letter] = space_ids[0]
        else:
            token_ids[letter] = plain_ids[0]

    print("Answer token IDs:")
    for letter, tid in token_ids.items():
        decoded = tokenizer.decode([tid])
        print(f"  {letter} → id={tid}  decoded={decoded!r}")

    return token_ids


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
):
    """
    Evaluate a model on the MedAraBench test set using log-probability selection.

    Args:
        model:        loaded (and optionally merged) causal LM
        tokenizer:    corresponding tokenizer
        test_dataset: HuggingFace Dataset with MedAraBench test samples
        output_dir:   directory to write results.json and predictions.csv
        batch_size:   number of samples per inference batch
        data_dir:     dataset root (unused here, kept for API consistency)

    Returns:
        dict with 'accuracy', 'macro_f1', and per-specialty scores
    """
    model.eval()

    # Left-padding for batched inference (align right edge of sequences)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    token_ids = _get_answer_token_ids(tokenizer)

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

        for i, sample in enumerate(batch):
            pred_letter = active_letters[predicted_indices[i]]
            ref_letter = str(sample.get("answer", "")).strip().upper()

            predictions.append(pred_letter)
            references.append(ref_letter)
            specialties.append(
                str(sample.get("specialty", sample.get("umbrella_specialty", "Unknown")))
            )

            per_sample_records.append({
                "question": sample.get("question", ""),
                "reference": ref_letter,
                "prediction": pred_letter,
                "correct": pred_letter == ref_letter,
                "specialty": specialties[-1],
            })

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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_records)
    print(f"Predictions saved to: {preds_path}")

    return results


def run_evaluation(
    model,
    tokenizer,
    output_dir: str,
    batch_size: int = 16,
    data_dir: str = "Files/datasets/",
):
    """
    Convenience wrapper: loads the MedAraBench test set and evaluates.

    Args:
        model:      loaded causal LM (merged if LoRA, ready for inference)
        tokenizer:  corresponding tokenizer
        output_dir: directory to write eval results
        batch_size: inference batch size
        data_dir:   dataset root

    Returns:
        dict with evaluation metrics
    """
    print("Loading MedAraBench test set...")
    test_dataset = load_medarabench(split="test", data_dir=data_dir)
    print(f"  Loaded {len(test_dataset):,} test samples")

    return evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        data_dir=data_dir,
    )
