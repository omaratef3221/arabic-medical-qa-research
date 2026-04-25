from datasets import Dataset


# ---------------------------------------------------------------------------
# Stage 1: AraMed — open-ended QA domain adaptation
# ---------------------------------------------------------------------------

def format_aramed_sample(sample: dict) -> str:
    """
    Format a single AraMed sample as a question-answer pair for causal LM training.

    Expected keys: 'question', 'answer'
    """
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    return (
        f"### Question:\n{question}\n\n"
        f"### Answer:\n{answer}"
    )


# ---------------------------------------------------------------------------
# Stage 2 / Evaluation: MedAraBench — MCQ
# ---------------------------------------------------------------------------

def format_medarabench_sample(sample: dict, include_answer: bool = True) -> str:
    """
    Format a single MedAraBench MCQ sample.

    When include_answer=True  → used for training (loss computed on answer token).
    When include_answer=False → used for evaluation (model predicts next token).

    Expected keys: 'question', 'option_a', 'option_b', 'option_c', 'option_d',
                   optionally 'option_e', 'answer'
    """
    question = str(sample.get("question", "")).strip()
    opt_a = str(sample.get("option_a", "")).strip()
    opt_b = str(sample.get("option_b", "")).strip()
    opt_c = str(sample.get("option_c", "")).strip()
    opt_d = str(sample.get("option_d", "")).strip()
    opt_e = str(sample.get("option_e", "")).strip()

    options_block = (
        f"A) {opt_a}\n"
        f"B) {opt_b}\n"
        f"C) {opt_c}\n"
        f"D) {opt_d}"
    )
    if opt_e:
        options_block += f"\nE) {opt_e}"

    prompt = (
        f"### Question:\n{question}\n\n"
        f"### Options:\n{options_block}\n\n"
        f"### Answer:\n"
    )

    if include_answer:
        answer = str(sample.get("answer", "")).strip()
        return prompt + answer
    return prompt


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize_aramed(sample: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Tokenize a single AraMed sample with label masking:
    only the answer portion contributes to the loss.
    """
    full_text = format_aramed_sample(sample)

    # Split at the answer boundary to find prompt length
    answer_marker = "### Answer:\n"
    split_idx = full_text.rfind(answer_marker)
    prompt_text = full_text[: split_idx + len(answer_marker)]
    answer_text = full_text[split_idx + len(answer_marker) :]

    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=True, truncation=False
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]

    # Add EOS token at the end if not already present
    if full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]
        full_ids = full_ids[:max_seq_length]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    labels = labels[: len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _tokenize_medarabench(sample: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Tokenize a single MedAraBench sample with label masking:
    only the answer letter contributes to the loss.
    """
    full_text = format_medarabench_sample(sample, include_answer=True)

    answer_marker = "### Answer:\n"
    split_idx = full_text.rfind(answer_marker)
    prompt_text = full_text[: split_idx + len(answer_marker)]

    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=True, truncation=False
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]

    if full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]
        full_ids = full_ids[:max_seq_length]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    labels = labels[: len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def tokenize_and_format(
    dataset: Dataset,
    tokenizer,
    prompt_fn: str,  # "aramed" or "medarabench"
    max_seq_length: int = 2048,
) -> Dataset:
    """
    Apply formatting + tokenization to a full HuggingFace Dataset.

    Args:
        dataset:        HuggingFace Dataset
        tokenizer:      HuggingFace tokenizer (already configured)
        prompt_fn:      "aramed"  → uses AraMed formatting
                        "medarabench" → uses MedAraBench MCQ formatting
        max_seq_length: maximum token length (longer samples are truncated)

    Returns:
        Tokenized HuggingFace Dataset with columns:
        input_ids, attention_mask, labels
    """
    if prompt_fn == "aramed":
        fn = lambda s: _tokenize_aramed(s, tokenizer, max_seq_length)
    elif prompt_fn == "medarabench":
        fn = lambda s: _tokenize_medarabench(s, tokenizer, max_seq_length)
    else:
        raise ValueError(f"Unknown prompt_fn: {prompt_fn!r}. Choose 'aramed' or 'medarabench'.")

    tokenized = dataset.map(
        fn,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing ({prompt_fn})",
    )
    tokenized.set_format("torch")
    return tokenized
