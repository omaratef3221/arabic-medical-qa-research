import re
from datasets import Dataset


# Valid single-letter answers (uppercase only after normalization)
VALID_ANSWERS = {"A", "B", "C", "D", "E"}

# Regex: exactly one letter optionally surrounded by whitespace/punctuation
_SINGLE_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\s*\.?\s*$")

# Patterns that indicate multi-label answers
_MULTI_LABEL_RE = re.compile(
    r"[A-Ea-e]\s*[\+\,\&]\s*[A-Ea-e]"   # A+B, A,B, A&B
    r"|and\s+[A-Ea-e]"                    # A and B
    r"|[A-Ea-e]\s+or\s+[A-Ea-e]",        # AorB
    re.IGNORECASE,
)


def _normalize_answer(raw: str) -> str | None:
    """
    Return the uppercase single letter (A-E) if valid, else None.
    Accepts: 'a', 'B', ' C ', 'D.' etc.
    Rejects: multi-label, text answers, NaN strings.
    """
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan":
        return None
    if _MULTI_LABEL_RE.search(raw):
        return None
    m = _SINGLE_LETTER_RE.match(raw)
    if m:
        letter = m.group(1).upper()
        return letter if letter in VALID_ANSWERS else None
    return None


def clean_medarabench(dataset: Dataset) -> Dataset:
    """
    Clean MedAraBench training data.

    Cleaning steps:
      1. Normalize answer to a single uppercase letter in {A, B, C, D, E}.
      2. Remove samples with multi-label, text, or missing answers.
      3. Remove samples where the question text is empty.
      4. For samples with answer E, verify that option_e is non-empty.
      5. Remove duplicate questions (keep first occurrence).

    Prints cleaning statistics.
    """
    df = dataset.to_pandas()
    original_count = len(df)

    # Step 1 & 2: Normalize and filter answer
    df["answer_clean"] = df["answer"].apply(_normalize_answer)
    invalid_answer_mask = df["answer_clean"].isna()
    removed_invalid = int(invalid_answer_mask.sum())
    df = df[~invalid_answer_mask].copy()
    df["answer"] = df["answer_clean"]
    df = df.drop(columns=["answer_clean"])

    # Step 3: Remove empty questions
    empty_q_mask = df["question"].str.strip().str.len() == 0
    removed_empty_q = int(empty_q_mask.sum())
    df = df[~empty_q_mask].copy()

    # Step 4: If answer is E but option_e is blank, drop
    if "option_e" in df.columns:
        missing_e_mask = (df["answer"] == "E") & (df["option_e"].str.strip().str.len() == 0)
        removed_missing_e = int(missing_e_mask.sum())
        df = df[~missing_e_mask].copy()
    else:
        removed_missing_e = 0

    # Step 5: Deduplicate by question text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["question"], keep="first")
    removed_duplicates = before_dedup - len(df)

    final_count = len(df)
    removed_total = original_count - final_count

    print("=" * 50)
    print("MedAraBench Cleaning Statistics")
    print("=" * 50)
    print(f"  Original samples  : {original_count:>7,}")
    print(f"  Invalid answers   : {removed_invalid:>7,}")
    print(f"  Empty questions   : {removed_empty_q:>7,}")
    print(f"  Missing option E  : {removed_missing_e:>7,}")
    print(f"  Duplicates removed: {removed_duplicates:>7,}")
    print(f"  Total removed     : {removed_total:>7,}")
    print(f"  Final samples     : {final_count:>7,}")
    print("=" * 50)

    df = df.reset_index(drop=True)
    return Dataset.from_pandas(df)
