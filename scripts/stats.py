"""
Statistical analysis over the per-sample prediction files.

From predictions/{run_name}.parquet (question_id, gold_label, pred_label,
logit_A..logit_E) this script computes:

  1. 95% bootstrap confidence intervals (10,000 resamples, seed 42) for
     Accuracy and Macro F1 of every configuration.
  2. Paired McNemar tests on accuracy for each Full-vs-LoRA contrast holding
     model and Stage 1 method fixed (exact binomial when discordant pairs
     < 25, otherwise continuity-corrected chi-square).
  3. Paired bootstrap tests for the Macro F1 delta of the same contrasts
     (delta, 95% CI of delta, p-value; same resample indices for both systems).
  4. Mean +/- std aggregation across seeds {42, 1337, 2024} for the
     multi-seed configurations (Task 2).

Outputs a single results/statistics.csv plus a booktabs LaTeX table
(results/statistics_table.tex). LaTeX output is ASCII-only, uses --- for
dashes and \\% for percent signs.

Usage:
  python scripts/stats.py [--n_boot 10000] [--seed 42]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PREDICTIONS_DIR, RESULTS_DIR  # noqa: E402
from utils.model_registry import REGISTRY  # noqa: E402

LETTERS = ["A", "B", "C", "D", "E"]
LETTER_TO_INT = {l: i for i, l in enumerate(LETTERS)}
K = len(LETTERS)

SEED_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_predictions(predictions_dir: str) -> dict[str, pd.DataFrame]:
    """Load every parquet prediction file, sorted by question_id."""
    files = sorted(glob.glob(os.path.join(predictions_dir, "*.parquet")))
    out = {}
    for path in files:
        run_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_parquet(path).sort_values("question_id").reset_index(drop=True)
        out[run_name] = df
    return out


def to_int_labels(series: pd.Series) -> np.ndarray:
    codes = series.map(LETTER_TO_INT)
    if codes.isna().any():
        bad = series[codes.isna()].unique()
        raise ValueError(f"Labels outside A-E in predictions: {bad}")
    return codes.to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Vectorised metric core (confusion-matrix based, fast enough for 10k boots)
# ---------------------------------------------------------------------------

def _confusion_from_codes(codes: np.ndarray) -> np.ndarray:
    """codes = gold * K + pred; returns K x K confusion matrix."""
    return np.bincount(codes, minlength=K * K).reshape(K, K)


def _accuracy_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(cm.sum(), 1))


def _macro_f1_cm(cm: np.ndarray) -> float:
    """Macro F1 over the fixed label set A-E (zero_division=0), matching
    sklearn.f1_score(average='macro') on the full test set where all five
    gold labels occur."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def bootstrap_ci(gold: np.ndarray, pred: np.ndarray,
                 n_boot: int, seed: int) -> dict:
    """Percentile bootstrap 95% CI for accuracy and macro F1."""
    n = len(gold)
    codes = gold * K + pred
    cm_full = _confusion_from_codes(codes)
    rng = np.random.default_rng(seed)

    accs = np.empty(n_boot)
    f1s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = _confusion_from_codes(codes[idx])
        accs[i] = _accuracy_cm(cm)
        f1s[i] = _macro_f1_cm(cm)

    return {
        "accuracy": _accuracy_cm(cm_full),
        "acc_ci_lo": float(np.percentile(accs, 2.5)),
        "acc_ci_hi": float(np.percentile(accs, 97.5)),
        "macro_f1": _macro_f1_cm(cm_full),
        "f1_ci_lo": float(np.percentile(f1s, 2.5)),
        "f1_ci_hi": float(np.percentile(f1s, 97.5)),
        "n": n,
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    Paired McNemar test on accuracy. b = A right & B wrong, c = A wrong &
    B right. Exact binomial for b + c < 25, else chi-square with continuity
    correction (Edwards).
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"mcnemar_b": b, "mcnemar_c": c, "mcnemar_stat": 0.0, "mcnemar_p": 1.0,
                "mcnemar_kind": "exact"}
    if b + c < 25:
        p = scipy_stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        return {"mcnemar_b": b, "mcnemar_c": c, "mcnemar_stat": float(min(b, c)),
                "mcnemar_p": float(p), "mcnemar_kind": "exact"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = scipy_stats.chi2.sf(stat, df=1)
    return {"mcnemar_b": b, "mcnemar_c": c, "mcnemar_stat": float(stat),
            "mcnemar_p": float(p), "mcnemar_kind": "chi2_cc"}


def paired_bootstrap_delta(gold: np.ndarray, pred_a: np.ndarray,
                           pred_b: np.ndarray, n_boot: int, seed: int) -> dict:
    """
    Paired bootstrap for the Macro F1 and accuracy deltas (A minus B): the
    SAME resample indices are applied to both systems. p-value is the
    two-sided bootstrap sign probability with add-one smoothing.
    """
    n = len(gold)
    codes_a = gold * K + pred_a
    codes_b = gold * K + pred_b
    rng = np.random.default_rng(seed)

    d_f1 = np.empty(n_boot)
    d_acc = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cm_a = _confusion_from_codes(codes_a[idx])
        cm_b = _confusion_from_codes(codes_b[idx])
        d_f1[i] = _macro_f1_cm(cm_a) - _macro_f1_cm(cm_b)
        d_acc[i] = _accuracy_cm(cm_a) - _accuracy_cm(cm_b)

    cm_a_full = _confusion_from_codes(codes_a)
    cm_b_full = _confusion_from_codes(codes_b)

    def _p(deltas):
        lo = (np.sum(deltas <= 0) + 1) / (n_boot + 1)
        hi = (np.sum(deltas >= 0) + 1) / (n_boot + 1)
        return float(min(1.0, 2 * min(lo, hi)))

    return {
        "f1_delta": _macro_f1_cm(cm_a_full) - _macro_f1_cm(cm_b_full),
        "f1_delta_ci_lo": float(np.percentile(d_f1, 2.5)),
        "f1_delta_ci_hi": float(np.percentile(d_f1, 97.5)),
        "f1_delta_p": _p(d_f1),
        "acc_delta": _accuracy_cm(cm_a_full) - _accuracy_cm(cm_b_full),
        "acc_delta_ci_lo": float(np.percentile(d_acc, 2.5)),
        "acc_delta_ci_hi": float(np.percentile(d_acc, 97.5)),
    }


# ---------------------------------------------------------------------------
# Contrast discovery
# ---------------------------------------------------------------------------

def full_vs_lora_contrasts(preds: dict) -> list[tuple[str, str, str, str]]:
    """
    Return (model_short, stage1_method, full_run, lora_run) for every
    Full-vs-LoRA contrast with model and Stage 1 method held fixed.
    """
    shorts = sorted({spec.short_name for spec in REGISTRY.values()},
                    key=len, reverse=True)
    contrasts = []
    for short in shorts:
        for s1 in ("none", "lora", "full", "full3ep"):
            full_run = f"{short}_s1-{s1}_s2-full"
            lora_run = f"{short}_s1-{s1}_s2-lora"
            if full_run in preds and lora_run in preds:
                contrasts.append((short, s1, full_run, lora_run))
    return contrasts


def align_pair(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Align two prediction frames on question_id (inner join, sorted)."""
    merged = df_a.merge(df_b, on="question_id", suffixes=("_a", "_b"))
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        print(f"  WARNING: question_id mismatch "
              f"({len(df_a)} vs {len(df_b)} -> {len(merged)} aligned)")
    gold = to_int_labels(merged["gold_label_a"])
    gold_b = to_int_labels(merged["gold_label_b"])
    if not np.array_equal(gold, gold_b):
        raise ValueError("Gold labels disagree between paired prediction files")
    return gold, to_int_labels(merged["pred_label_a"]), to_int_labels(merged["pred_label_b"])


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def seed_aggregates(config_rows: list[dict]) -> list[dict]:
    """Mean +/- std across seeds for runs following the _seed{N} convention
    (the suffix-less run is seed 42)."""
    by_base: dict[str, dict[int, dict]] = {}
    names = {r["run_name"] for r in config_rows}
    for row in config_rows:
        m = SEED_RE.match(row["run_name"])
        if m:
            by_base.setdefault(m.group("base"), {})[int(m.group("seed"))] = row
        elif any(SEED_RE.match(n) and SEED_RE.match(n).group("base") == row["run_name"]
                 for n in names):
            by_base.setdefault(row["run_name"], {})[42] = row

    out = []
    for base, seed_rows in sorted(by_base.items()):
        if len(seed_rows) < 2:
            continue
        accs = [r["accuracy"] for r in seed_rows.values()]
        f1s = [r["macro_f1"] for r in seed_rows.values()]
        out.append({
            "kind": "seed_agg",
            "run_name": base,
            "seeds": ",".join(str(s) for s in sorted(seed_rows)),
            "n_seeds": len(seed_rows),
            "accuracy": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "macro_f1": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# LaTeX output (ASCII only; --- for dashes; \% for percent)
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def latex_statistics_table(config_rows, contrast_rows, seed_rows) -> str:
    lines = [
        "% Auto-generated by scripts/stats.py --- do not edit by hand.",
        "% Requires \\usepackage{booktabs}.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Accuracy and Macro F1 on the MedAraBench test set with 95\\%",
        "bootstrap confidence intervals (10{,}000 resamples).}",
        "\\label{tab:statistics}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Configuration & Accuracy (\\%) & Macro F1 (\\%) \\\\",
        "\\midrule",
    ]
    for row in config_rows:
        name = row["run_name"].replace("_", "\\_")
        acc = f"{_pct(row['accuracy'])} [{_pct(row['acc_ci_lo'])}, {_pct(row['acc_ci_hi'])}]"
        f1 = f"{_pct(row['macro_f1'])} [{_pct(row['f1_ci_lo'])}, {_pct(row['f1_ci_hi'])}]"
        lines.append(f"{name} & {acc} & {f1} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    if contrast_rows:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Full vs.\\ LoRA contrasts (model and Stage 1 method held",
            "fixed): McNemar test on accuracy and paired-bootstrap Macro F1 delta.}",
            "\\label{tab:contrasts}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "Model & Stage 1 & $\\Delta$Acc (\\%) & McNemar $p$ & "
            "$\\Delta$F1 (\\%) [95\\% CI] & $p$ \\\\",
            "\\midrule",
        ]
        for row in contrast_rows:
            model = row["model"].replace("_", "\\_")
            dacc = _pct(row["acc_delta"])
            df1 = (f"{_pct(row['f1_delta'])} "
                   f"[{_pct(row['f1_delta_ci_lo'])}, {_pct(row['f1_delta_ci_hi'])}]")
            lines.append(
                f"{model} & {row['stage1']} & {dacc} & "
                f"{row['mcnemar_p']:.4f} & {df1} & {row['f1_delta_p']:.4f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    if seed_rows:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Multi-seed stability (seeds 42, 1337, 2024): "
            "mean $\\pm$ std.}",
            "\\label{tab:seeds}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Configuration & Accuracy (\\%) & Macro F1 (\\%) \\\\",
            "\\midrule",
        ]
        for row in seed_rows:
            name = row["run_name"].replace("_", "\\_")
            lines.append(
                f"{name} & {_pct(row['accuracy'])} $\\pm$ {_pct(row['acc_std'])} & "
                f"{_pct(row['macro_f1'])} $\\pm$ {_pct(row['f1_std'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    text = "\n".join(lines)
    if not text.isascii():
        bad = {c for c in text if not c.isascii()}
        raise ValueError(f"Non-ASCII characters in LaTeX output: {bad}")
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", default=PREDICTIONS_DIR)
    parser.add_argument("--out_dir", default=RESULTS_DIR)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preds = load_all_predictions(args.predictions_dir)
    if not preds:
        print(f"No prediction files in {args.predictions_dir}. "
              f"Run scripts/reinfer_all.py first.")
        sys.exit(1)
    print(f"Loaded {len(preds)} prediction files from {args.predictions_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Per-configuration bootstrap CIs
    config_rows = []
    for run_name, df in sorted(preds.items()):
        gold = to_int_labels(df["gold_label"])
        pred = to_int_labels(df["pred_label"])
        ci = bootstrap_ci(gold, pred, args.n_boot, args.seed)
        config_rows.append({"kind": "config", "run_name": run_name, **ci})
        print(f"  {run_name}: acc={ci['accuracy']:.4f} "
              f"[{ci['acc_ci_lo']:.4f}, {ci['acc_ci_hi']:.4f}]  "
              f"f1={ci['macro_f1']:.4f} [{ci['f1_ci_lo']:.4f}, {ci['f1_ci_hi']:.4f}]")

    # 2+3) Full-vs-LoRA contrasts: McNemar + paired bootstrap F1 delta
    contrast_rows = []
    for model, s1, full_run, lora_run in full_vs_lora_contrasts(preds):
        print(f"Contrast: {full_run} vs {lora_run}")
        gold, pred_full, pred_lora = align_pair(preds[full_run], preds[lora_run])
        row = {
            "kind": "contrast", "model": model, "stage1": s1,
            "run_name": f"{full_run} vs {lora_run}",
            "run_full": full_run, "run_lora": lora_run,
            "n": len(gold),
        }
        row.update(mcnemar_test(pred_full == gold, pred_lora == gold))
        row.update(paired_bootstrap_delta(gold, pred_full, pred_lora,
                                          args.n_boot, args.seed))
        contrast_rows.append(row)
        print(f"  dAcc={row['acc_delta']:+.4f}  McNemar p={row['mcnemar_p']:.4g}  "
              f"dF1={row['f1_delta']:+.4f} "
              f"[{row['f1_delta_ci_lo']:+.4f}, {row['f1_delta_ci_hi']:+.4f}] "
              f"p={row['f1_delta_p']:.4g}")

    # 4) Multi-seed aggregation
    seed_rows = seed_aggregates(config_rows)
    for row in seed_rows:
        print(f"Seeds {row['seeds']} {row['run_name']}: "
              f"acc={row['accuracy']:.4f}+/-{row['acc_std']:.4f}  "
              f"f1={row['macro_f1']:.4f}+/-{row['f1_std']:.4f}")

    # Single CSV
    all_rows = config_rows + contrast_rows + seed_rows
    csv_path = os.path.join(args.out_dir, "statistics.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nStatistics CSV written to: {csv_path}")

    # LaTeX table
    tex_path = os.path.join(args.out_dir, "statistics_table.tex")
    with open(tex_path, "w", encoding="ascii") as f:
        f.write(latex_statistics_table(config_rows, contrast_rows, seed_rows))
    print(f"LaTeX table written to: {tex_path}")


if __name__ == "__main__":
    main()
