# Datasets & Preprocessing — Complete Reference

This document fully describes the datasets used and every preprocessing step applied to them. All numbers and code references have been validated against the live codebase and verified by executing the preprocessing pipeline on the actual data files.

---

## Table of Contents

1. [Datasets](#1-datasets)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Where Cleaning Is Applied](#3-where-cleaning-is-applied-validated)
4. [Validation Splits](#4-validation-splits)
5. [Why AraMed Has No Cleaning Step](#5-aramed-has-no-cleaning-step--why)
6. [Final Pipeline Numbers](#6-final-pipeline-numbers-end-to-end)

---

## 1. Datasets

The pipeline uses **two distinct Arabic medical datasets**, each serving a different role.

### 1.1 AraMed — Stage 1 (Domain Adaptation)

**Purpose:** Continual pre-training on open-ended Arabic medical QA to inject domain knowledge into the base LLM before task fine-tuning.

| Property | Value |
|---|---|
| **File location** | `Files/datasets/AraMed/Train.csv`, `Files/datasets/AraMed/Test.csv` |
| **Format** | Open-ended (question + free-text doctor answer) — **NOT** multiple choice |
| **Train size** | **109,834 samples** |
| **Test size** | **27,459 samples** (unused; we use a 2% holdout from train for validation) |
| **Language** | Arabic |
| **Domain** | General medical Q&A (consumer health, clinical scenarios) |

**Source columns (CSV → loader):**

- `Question description` → primary question text (falls back to `Question title` if NaN)
- `Answer details 1` → correct doctor answer
- Other columns in the CSV are discarded

**Output columns after `load_aramed()`:** `['question', 'answer']`

### 1.2 MedAraBench — Stage 2 (Task Fine-tuning) + Evaluation

**Purpose:** Supervised fine-tuning on Arabic medical multiple-choice questions, then evaluation on a held-out test split.

| Property | Value |
|---|---|
| **File location** | `Files/datasets/MedAraBench/Train.csv`, `Files/datasets/MedAraBench/Test.csv` |
| **Format** | 5-choice MCQ (A/B/C/D/E) with specialty labels |
| **Train size (raw)** | **19,891 samples** |
| **Train size (cleaned)** | **17,638 samples** |
| **Test size (raw)** | **4,959 samples** |
| **Test size (cleaned)** | **4,761 samples** |
| **Language** | Arabic |
| **Domain** | Medical board-exam style MCQ across many specialties |

**Source columns (CSV → loader, after rename):**

| CSV column | Internal name |
|---|---|
| `Question` | `question` |
| `Option A` | `option_a` |
| `Option B` | `option_b` |
| `Option C` | `option_c` |
| `Option D` | `option_d` |
| `Option E` | `option_e` |
| `Correct Answer` | `answer` |
| `Medical Specialty` | `specialty` |
| `umbrella_specialty` | `umbrella_specialty` |
| `Level` | `level` |
| `Group` | `group` (test-only; absent → skipped) |

---

## 2. Preprocessing Pipeline

### 2.1 Robust CSV Loading — `_read_csv_robust()`

**Source:** [data/read_data.py:7-29](data/read_data.py#L7-L29)

The AraMed `Train.csv` contains malformed quoted strings (stray quote characters in Arabic medical text — e.g., row ~108,971 triggers `pd.errors.ParserError: EOF inside string`). The loader uses a **two-pass fallback strategy**:

```python
def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)                    # Fast C engine
    except pd.errors.ParserError:
        return pd.read_csv(
            path,
            engine="python",                                          # Slower but tolerant
            quoting=csv.QUOTE_NONE,        # Treat stray quotes as literal characters
            on_bad_lines="skip",            # Skip malformed rows instead of crashing
            encoding="utf-8",
            encoding_errors="replace",      # Replace invalid bytes instead of crashing
        )
```

**Why both engines:** The C engine is ~10x faster but strict about quoting. The Python engine + `QUOTE_NONE` ignores quote semantics entirely, treating every character literally.

### 2.2 AraMed Loading — `load_aramed()`

**Source:** [data/read_data.py:32-55](data/read_data.py#L32-L55)

```python
# Build question: prefer description, fall back to title
df["question"] = df["Question description"].fillna(df["Question title"]).fillna("").str.strip()
df["answer"]   = df["Answer details 1"].fillna("").str.strip()

# Drop rows where either field is empty
df = df[df["question"].str.len() > 0]
df = df[df["answer"].str.len() > 0]
```

Preprocessing steps:

1. Read CSV with robust reader
2. **Build question column:** Primary = `Question description`; fallback to `Question title` if description is NaN; default to empty string
3. **Build answer column:** Use `Answer details 1` (the verified doctor answer)
4. Strip whitespace from both
5. **Drop empty rows:** Any sample with empty question OR empty answer is dropped
6. Reset index, return as HuggingFace `Dataset`

**No cleaning beyond emptiness filter** — AraMed answers are free-text, so there's no letter-validation step.

### 2.3 MedAraBench Loading — `load_medarabench()`

**Source:** [data/read_data.py:58-110](data/read_data.py#L58-L110)

```python
# Keep only relevant columns (skip 'Group' if absent)
keep_cols = ["Question", "Option A", ..., "Group"]
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()

# Rename to consistent lowercase
df = df.rename(columns={...})

# Fill NaN option_e with empty string (some MCQs have only 4 options)
df["option_e"] = df["option_e"].fillna("")

# Coerce all text fields to stripped strings
for col in ["question", "option_a", ..., "answer"]:
    df[col] = df[col].fillna("").astype(str).str.strip()
```

Preprocessing steps:

1. Read CSV with robust reader
2. **Keep only relevant columns** (handles missing `Group` column in test split)
3. **Rename to consistent lowercase snake_case**
4. **Fill missing option_e:** Some MCQs have only 4 options (A–D); the missing E becomes an empty string (not NaN)
5. **Type coercion:** All text fields → string, NaN → empty string, strip whitespace
6. Reset index, return as `Dataset`

**Note:** `load_medarabench()` does NOT clean the data — that's a separate explicit step.

### 2.4 MedAraBench Cleaning — `clean_medarabench()`

**Source:** [data/clean_data.py:38-96](data/clean_data.py#L38-L96)

The raw MedAraBench `Correct Answer` column contains garbage: Arabic medical text, multi-letter answers (`A+B`, `C, A`), question marks, lowercase letters, out-of-range letters (`F`, `V`, `S`), free-text rejections (`None of the above`), etc. The cleaning function is a **5-step filter** with regex-based answer normalization.

#### Step 1 & 2: Answer Normalization & Multi-Label Rejection

```python
VALID_ANSWERS = {"A", "B", "C", "D", "E"}

_SINGLE_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\s*\.?\s*$")
_MULTI_LABEL_RE = re.compile(
    r"[A-Ea-e]\s*[\+\,\&]\s*[A-Ea-e]"   # A+B, A,B, A&B
    r"|and\s+[A-Ea-e]"                   # A and B
    r"|[A-Ea-e]\s+or\s+[A-Ea-e]",        # A or B
    re.IGNORECASE,
)

def _normalize_answer(raw: str) -> str | None:
    raw = str(raw).strip()
    if not raw or raw.lower() == "nan": return None     # Empty / NaN
    if _MULTI_LABEL_RE.search(raw):     return None     # A+B, A or B, etc.
    m = _SINGLE_LETTER_RE.match(raw)
    if m:
        letter = m.group(1).upper()
        return letter if letter in VALID_ANSWERS else None  # Valid A-E
    return None                                              # Free-text answer
```

**What this accepts:** `"A"`, `"b"`, `" C "`, `"D."`, `"e "` → normalized to uppercase A–E

**What this rejects:** `"A+B"`, `"C, A"`, `"A and B"`, `"None of the above"`, `"الجسم اللوزي"`, `"?"`, `"F"`, `"V"`, NaN, empty

#### Step 3: Empty Question Filter

```python
empty_q_mask = df["question"].str.strip().str.len() == 0
df = df[~empty_q_mask]
```

Drops samples where the question text is empty (after stripping whitespace).

#### Step 4: Missing Option-E Filter

```python
missing_e_mask = (df["answer"] == "E") & (df["option_e"].str.strip().str.len() == 0)
df = df[~missing_e_mask]
```

If the correct answer is `E` but `option_e` is empty/missing, the sample is contradictory and is dropped.

#### Step 5: Question-Text Deduplication

```python
df = df.drop_duplicates(subset=["question"], keep="first")
```

Removes duplicate questions (keeps the first occurrence). Note: this deduplicates by question text only, ignoring option ordering.

#### Cleaning Statistics (Verified by running on actual data)

| Step | Train (19,891 → 17,638) | Test (4,959 → 4,761) |
|---|---|---|
| Invalid answers (Step 1+2) | **79 removed** | **21 removed** |
| Empty questions (Step 3) | **1 removed** | **1 removed** |
| Missing option E (Step 4) | **56 removed** | **10 removed** |
| Duplicates (Step 5) | **2,117 removed** | **166 removed** |
| **Total removed** | **2,253** | **198** |
| **Final samples** | **17,638** | **4,761** |

---

## 3. Where Cleaning Is Applied (Validated)

Cleaning is invoked at **three call sites** in the codebase:

| Location | Purpose | Code |
|---|---|---|
| [main.py:182](main.py#L182) | Compute dataset statistics for W&B logging at run start | `med_clean = clean_medarabench(med_raw)` |
| [train/finetuning.py:236](train/finetuning.py#L236) | Clean training data before Stage 2 SFTTrainer | `clean_dataset = clean_medarabench(raw_dataset)` |
| [evaluation/evaluate.py:227](evaluation/evaluate.py#L227) | Clean test data before evaluation | `test_dataset = clean_medarabench(raw_test)` |

**Critical history:** Until April 2026, `evaluate.py` was NOT cleaning the test set — metrics were computed against 4,959 polluted reference labels including garbage answers. This was fixed by adding the cleaning call at line 227.

---

## 4. Validation Splits

After cleaning, the train splits are further divided into train/val for per-epoch validation loss tracking. Both stages use HuggingFace's seeded `train_test_split()` for reproducibility.

### Stage 1 (AraMed)

**Source:** [train/adaptation.py:121](train/adaptation.py#L121)

```python
split = full_dataset.train_test_split(test_size=val_split, seed=train_cfg.get("seed", 42))
```

| Parameter | Value |
|---|---|
| `val_split` default | **0.02** (2%) |
| `seed` | 42 |
| Effective Train | **107,637 samples** |
| Effective Val | **2,196 samples** |

### Stage 2 (MedAraBench cleaned)

**Source:** [train/finetuning.py:238](train/finetuning.py#L238)

```python
split = clean_dataset.train_test_split(test_size=val_split, seed=train_cfg.get("seed", 42))
```

| Parameter | Value |
|---|---|
| `val_split` default | **0.05** (5%) |
| `seed` | 42 |
| Effective Train | **16,756 samples** |
| Effective Val | **881 samples** |

---

## 5. AraMed Has No Cleaning Step — Why?

AraMed is used for **open-ended language modeling** (Stage 1 domain adaptation), so:

- Answers are free-text doctor responses, not letters
- There's no concept of a "valid" answer to enforce
- The only quality filter is: drop samples with empty question OR empty answer (handled at load time)

This contrasts with MedAraBench where answers MUST be in `{A, B, C, D, E}` for log-probability evaluation to work.

---

## 6. Final Pipeline Numbers (End-to-End)

| Stage | Dataset | Source | After Loading | After Cleaning | After Val Split |
|---|---|---|---|---|---|
| **Stage 1 (S1)** | AraMed Train | 109,834 | 109,834 | — (no cleaning) | 107,637 train / 2,196 val |
| **Stage 2 (S2)** | MedAraBench Train | 19,891 | 19,891 | 17,638 | 16,756 train / 881 val |
| **Evaluation** | MedAraBench Test | 4,959 | 4,959 | 4,761 | — (used in full) |

These are the exact numbers reported in W&B `dataset/*` summary fields and printed at the start of every run.

---

## Appendix: Example Garbage Answers Filtered from MedAraBench

A representative sample of the 100+ unique invalid answers that the cleaning step filters out:

**Multi-label answers:**
- `'A+B'`, `'A+E'`, `'A+B+C+D+E'`, `'A,B,C,D'`, `'C+D'`, `'B+D'`, `'D+E'`
- `'C or D'`, `'AorB'`, `'C, A'`

**Free-text/Arabic answers (treated as full text instead of letter):**
- `'الجسم اللوزي'` (the amygdala)
- `'المحفظة الباطنة'` (internal capsule)
- `'التلم المركزي'` (central sulcus)
- `'العقدة الأذنية'` (otic ganglion)
- `'كل ما سبق صح'` (all of the above)
- `'ولا بنية مما علا'` (none of the above)
- `'مراكز الحس'` (sensory centers)

**Out-of-range letters:**
- `'F'`, `'S'`, `'V'`, `'A\``

**Invalid markers:**
- `'?'`, `'None of the above'`, NaN, empty strings
