import csv
import os
import pandas as pd
from datasets import Dataset


def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read a CSV file tolerantly, handling:
      - Unterminated quoted strings (common in Arabic medical text with stray quotes)
      - Embedded newlines inside cells
      - Encoding issues

    Strategy: try the fast C engine first; on parse error fall back to the
    Python engine with quoting=QUOTE_NONE so stray quotes are treated as
    ordinary characters and bad lines are skipped rather than crashing.
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        print(f"  [read_csv] C engine failed on {path}, retrying with Python engine (quoting=QUOTE_NONE)...")
        return pd.read_csv(
            path,
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
            encoding="utf-8",
            encoding_errors="replace",
        )


def load_aramed(split="train", data_dir="Files/datasets/"):
    """
    Load AraMed dataset (open-ended QA, Stage 1 domain adaptation).

    Columns used:
      - 'Question description' (primary question text; falls back to 'Question title')
      - 'Answer details 1' (correct doctor answer)

    Returns a HuggingFace Dataset with columns: question, answer
    """
    filename = "Train.csv" if split == "train" else "Test.csv"
    path = os.path.join(data_dir, "AraMed", filename)
    df = _read_csv_robust(path)

    # Build question: prefer description, fall back to title
    df["question"] = df["Question description"].fillna(df["Question title"]).fillna("").str.strip()
    df["answer"] = df["Answer details 1"].fillna("").str.strip()

    # Drop rows where either field is empty
    df = df[df["question"].str.len() > 0]
    df = df[df["answer"].str.len() > 0]

    result = df[["question", "answer"]].reset_index(drop=True)
    return Dataset.from_pandas(result)


def load_medarabench(split="train", data_dir="Files/datasets/"):
    """
    Load MedAraBench dataset (MCQ, Stage 2 task fine-tuning and evaluation).

    Columns used:
      - 'Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Option E'
      - 'Correct Answer'
      - 'Medical Specialty', 'umbrella_specialty', 'Level'

    Returns a HuggingFace Dataset with standardized columns.
    """
    filename = "Train.csv" if split == "train" else "Test.csv"
    path = os.path.join(data_dir, "MedAraBench", filename)
    df = _read_csv_robust(path)

    # Stable per-sample identifier for prediction files / statistical tests.
    # 'Question Number' is unique in Test.csv; fall back to row position.
    if "Question Number" in df.columns and df["Question Number"].is_unique:
        df["question_id"] = df["Question Number"]
    else:
        df["question_id"] = range(len(df))

    # Keep only relevant columns
    keep_cols = [
        "question_id",
        "Question", "Option A", "Option B", "Option C", "Option D", "Option E",
        "Correct Answer", "Medical Specialty", "umbrella_specialty", "Level", "Group",
    ]
    # Some columns may be absent (e.g. 'Group' in test)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Rename for consistency
    rename_map = {
        "Question": "question",
        "Option A": "option_a",
        "Option B": "option_b",
        "Option C": "option_c",
        "Option D": "option_d",
        "Option E": "option_e",
        "Correct Answer": "answer",
        "Medical Specialty": "specialty",
        "umbrella_specialty": "umbrella_specialty",
        "Level": "level",
        "Group": "group",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Fill NaN option_e with empty string
    if "option_e" in df.columns:
        df["option_e"] = df["option_e"].fillna("")
    else:
        df["option_e"] = ""

    # Ensure string types
    for col in ["question", "option_a", "option_b", "option_c", "option_d", "option_e", "answer"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.reset_index(drop=True)
    return Dataset.from_pandas(df)
