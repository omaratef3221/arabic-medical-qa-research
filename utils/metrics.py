from sklearn.metrics import f1_score, accuracy_score


def compute_accuracy(predictions: list, references: list) -> float:
    """
    Compute simple accuracy: correct predictions / total samples.

    Args:
        predictions: list of predicted answer letters, e.g. ["A", "B", "C"]
        references:  list of ground-truth answer letters

    Returns:
        Accuracy as a float in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references."
        )
    return float(accuracy_score(references, predictions))


def compute_macro_f1(predictions: list, references: list) -> float:
    """
    Compute macro-averaged F1 score across answer classes (A, B, C, D, E).

    Args:
        predictions: list of predicted answer letters
        references:  list of ground-truth answer letters

    Returns:
        Macro F1 as a float in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references."
        )
    return float(
        f1_score(references, predictions, average="macro", zero_division=0)
    )


def compute_all_metrics(predictions: list, references: list) -> dict:
    """
    Compute both accuracy and macro F1 in a single call.

    Returns:
        dict with keys 'accuracy' and 'macro_f1'
    """
    acc = compute_accuracy(predictions, references)
    f1 = compute_macro_f1(predictions, references)
    return {"accuracy": acc, "macro_f1": f1}


def compute_per_specialty_metrics(
    predictions: list,
    references: list,
    specialties: list,
) -> dict:
    """
    Compute accuracy and macro-F1 grouped by medical specialty.

    Args:
        predictions: list of predicted answer letters
        references:  list of ground-truth answer letters
        specialties: list of specialty labels (same length)

    Returns:
        dict mapping specialty → {accuracy, macro_f1, count}
    """
    from collections import defaultdict

    groups: dict = defaultdict(lambda: {"preds": [], "refs": []})
    for pred, ref, spec in zip(predictions, references, specialties):
        groups[spec]["preds"].append(pred)
        groups[spec]["refs"].append(ref)

    results = {}
    for spec, data in sorted(groups.items()):
        preds = data["preds"]
        refs = data["refs"]
        results[spec] = {
            "accuracy": compute_accuracy(preds, refs),
            "macro_f1": compute_macro_f1(preds, refs),
            "count": len(preds),
        }
    return results
