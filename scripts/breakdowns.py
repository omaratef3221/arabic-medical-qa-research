"""
Per-configuration error breakdowns from the prediction parquet files.

For every predictions/{run_name}.parquet this produces:
  - class-wise precision / recall / F1 per answer letter (CSV)
  - a 5x5 confusion matrix (CSV + annotated heatmap figure)
and across configurations:
  - accuracy broken down by medical specialty (umbrella), difficulty /
    academic year (Y1-Y5) and gold answer label (CSVs + comparison heatmaps)

Figure style (paper figures): serif font, YlGnBu sequential colormap,
annotated cell values with luminance-aware ink, and a red outline marking the
best configuration's row in every comparison heatmap.

Usage:
  python scripts/breakdowns.py [--predictions_dir ...] [--out_dir ...]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PREDICTIONS_DIR, RESULTS_DIR  # noqa: E402

LETTERS = ["A", "B", "C", "D", "E"]
LEVELS = ["Y1", "Y2", "Y3", "Y4", "Y5"]
HIGHLIGHT_RED = "#d62728"  # reserved highlight; never used as a data color

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotate(ax, matrix, fmt="{:.2f}", threshold=None):
    """Write each cell value with ink that stays readable on the YlGnBu ramp."""
    if threshold is None:
        vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
        threshold = 0.6 * vmax
    for (i, j), val in np.ndenumerate(matrix):
        if np.isnan(val):
            continue
        color = "white" if val > threshold else "#1a1a2e"
        ax.text(j, i, fmt.format(val), ha="center", va="center",
                color=color, fontsize=8)


def _outline_row(ax, row_idx, n_cols):
    ax.add_patch(Rectangle(
        (-0.5, row_idx - 0.5), n_cols, 1,
        fill=False, edgecolor=HIGHLIGHT_RED, linewidth=2.0, zorder=5,
    ))


def load_predictions(predictions_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.parquet"))):
        run_name = os.path.splitext(os.path.basename(path))[0]
        out[run_name] = pd.read_parquet(path)
    return out


def accuracy_by(df: pd.DataFrame, column: str) -> pd.Series:
    correct = (df["gold_label"] == df["pred_label"])
    return correct.groupby(df[column]).mean()


# ---------------------------------------------------------------------------
# Per-configuration outputs
# ---------------------------------------------------------------------------

def classwise_report(df: pd.DataFrame) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        df["gold_label"], df["pred_label"], labels=LETTERS, zero_division=0)
    return pd.DataFrame({
        "letter": LETTERS, "precision": p, "recall": r, "f1": f1,
        "support": support,
    })


def confusion_df(df: pd.DataFrame) -> pd.DataFrame:
    cm = confusion_matrix(df["gold_label"], df["pred_label"], labels=LETTERS)
    return pd.DataFrame(cm, index=LETTERS, columns=LETTERS)


def plot_confusion(cm: pd.DataFrame, run_name: str, out_path: str):
    # Row-normalised for color, raw counts as annotations
    counts = cm.to_numpy().astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, row_sums, out=np.zeros_like(counts),
                     where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(LETTERS)), LETTERS)
    ax.set_yticks(range(len(LETTERS)), LETTERS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title(f"Confusion matrix: {run_name}")
    _annotate_with_norm(ax, counts, norm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Row-normalised share")
    fig.savefig(out_path)
    plt.close(fig)


def _annotate_with_norm(ax, counts, norm):
    for (i, j), val in np.ndenumerate(counts):
        color = "white" if norm[i, j] > 0.6 else "#1a1a2e"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                color=color, fontsize=8)


# ---------------------------------------------------------------------------
# Cross-configuration comparison heatmaps
# ---------------------------------------------------------------------------

def comparison_heatmap(table: pd.DataFrame, best_run: str, title: str,
                       xlabel: str, out_path: str):
    """
    Heatmap of accuracy with configurations as rows and category values as
    columns; the best configuration's row gets a red outline.
    """
    data = table.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    fig_h = max(2.0, 0.34 * n_rows + 1.2)
    fig_w = max(4.0, 0.85 * n_cols + 2.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0,
                   vmax=np.nanmax(data) if np.isfinite(data).any() else 1.0,
                   aspect="auto")
    ax.set_xticks(range(n_cols), [str(c) for c in table.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(n_rows), [str(r) for r in table.index])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _annotate(ax, data)
    if best_run in table.index:
        _outline_row(ax, list(table.index).index(best_run), n_cols)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Accuracy")
    fig.savefig(out_path)
    plt.close(fig)


def build_breakdown_table(preds: dict, column: str,
                          column_order: list | None = None) -> pd.DataFrame:
    rows = {}
    for run_name, df in preds.items():
        if column not in df.columns:
            continue
        rows[run_name] = accuracy_by(df, column)
    table = pd.DataFrame(rows).T  # configs x categories
    if column_order:
        ordered = [c for c in column_order if c in table.columns]
        rest = [c for c in table.columns if c not in ordered]
        table = table[ordered + rest]
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", default=PREDICTIONS_DIR)
    parser.add_argument("--out_dir", default=os.path.join(RESULTS_DIR, "breakdowns"))
    args = parser.parse_args()

    preds = load_predictions(args.predictions_dir)
    if not preds:
        print(f"No prediction files in {args.predictions_dir}. "
              f"Run scripts/reinfer_all.py first.")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loaded {len(preds)} prediction files")

    # Best configuration = highest overall accuracy (marked red in figures)
    overall_acc = {name: float((df["gold_label"] == df["pred_label"]).mean())
                   for name, df in preds.items()}
    best_run = max(overall_acc, key=overall_acc.get)
    print(f"Best configuration: {best_run} (acc={overall_acc[best_run]:.4f})")

    # Per-configuration: class-wise report + confusion matrix
    for run_name, df in sorted(preds.items()):
        report = classwise_report(df)
        report.to_csv(os.path.join(args.out_dir, f"classwise_{run_name}.csv"),
                      index=False)
        cm = confusion_df(df)
        cm.to_csv(os.path.join(args.out_dir, f"confusion_{run_name}.csv"))
        plot_confusion(cm, run_name,
                       os.path.join(args.out_dir, f"confusion_{run_name}.png"))
    print(f"Per-configuration class-wise reports and confusion matrices saved")

    # Cross-configuration breakdowns
    breakdowns = [
        ("umbrella_specialty", None, "Accuracy by medical specialty", "Specialty",
         "acc_by_specialty"),
        ("level", LEVELS, "Accuracy by academic year (difficulty)", "Academic year",
         "acc_by_level"),
        ("gold_label", LETTERS, "Accuracy by gold answer label", "Gold label",
         "acc_by_gold_label"),
    ]
    for column, order, title, xlabel, stem in breakdowns:
        table = build_breakdown_table(preds, column, order)
        if table.empty:
            print(f"  WARNING: no data for breakdown column {column!r}")
            continue
        table.to_csv(os.path.join(args.out_dir, f"{stem}.csv"))
        comparison_heatmap(table, best_run, title, xlabel,
                           os.path.join(args.out_dir, f"{stem}.png"))
        print(f"  {stem}: {table.shape[0]} configs x {table.shape[1]} categories")

    # Cross-configuration class-wise F1 heatmap
    f1_rows = {name: classwise_report(df).set_index("letter")["f1"]
               for name, df in preds.items()}
    f1_table = pd.DataFrame(f1_rows).T[LETTERS]
    f1_table.to_csv(os.path.join(args.out_dir, "f1_by_letter.csv"))
    comparison_heatmap(f1_table, best_run, "Class-wise F1 by answer letter",
                       "Answer letter",
                       os.path.join(args.out_dir, "f1_by_letter.png"))

    print(f"\nAll breakdowns written to: {args.out_dir}")


if __name__ == "__main__":
    main()
