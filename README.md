# LoRA vs. Full Fine-Tuning for Arabic Medical Question Answering

**NeurIPS Paper:** *LoRA vs. Full Fine-Tuning for Arabic Medical Question Answering: A Systematic Comparison Across General-Purpose and Arabic-Centric Large Language Models*

A complete training and evaluation pipeline comparing LoRA and full fine-tuning across Llama-3.1 and Jais-2 base models on Arabic medical multiple-choice question answering (MCQ), using a two-stage fine-tuning strategy.

---

## Overview

The pipeline implements a two-stage fine-tuning approach:

| Stage | Dataset | Task |
|-------|---------|------|
| **Stage 1** — Domain Adaptation | AraMed (~110K QA pairs) | Open-ended Arabic medical QA |
| **Stage 2** — Task Fine-tuning | MedAraBench (~17.6K MCQ samples) | Multiple-choice QA (A/B/C/D/E) |

Models are evaluated on the **MedAraBench test set** (4,959 samples) using a log-probability method (no text generation) — the predicted answer is the option letter with the highest logit at the final prompt token.

---

## Directory Structure

```
script/
├── configs/
│   ├── lora.yaml             # LoRA hyperparameters (r=16, alpha=32, dropout=0.05)
│   └── full_ft.yaml          # Full fine-tuning hyperparameters
├── data/
│   ├── read_data.py          # Dataset loaders → HuggingFace Dataset
│   └── clean_data.py         # MedAraBench cleaning (dedup, answer validation)
├── evaluation/
│   └── evaluate.py           # Log-prob evaluation on MedAraBench test set
├── Files/
│   └── datasets/
│       ├── AraMed/
│       │   ├── Train.csv     # ~110K Arabic medical QA pairs (Stage 1)
│       │   └── Test.csv
│       └── MedAraBench/
│           ├── Train.csv     # ~19.9K MCQ samples (→ 17.6K after cleaning)
│           └── Test.csv      # 4,959 MCQ samples (evaluation benchmark)
├── train/
│   ├── adaptation.py         # Stage 1 training (SFTTrainer on AraMed)
│   └── finetuning.py         # Stage 2 training (SFTTrainer on MedAraBench)
├── utils/
│   ├── get_model.py          # Model/tokenizer loading, LoRA setup, merging
│   ├── metrics.py            # Accuracy + Macro-F1 (overall + per-specialty)
│   └── prompt_template.py    # Prompt formatting + tokenization with label masking
├── main.py                   # CLI orchestrator
├── run.sh                    # Experiment runner (all 18 experiments)
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU (A100 80GB recommended for 8B models; multi-GPU or 4-bit quantization required for 70B models).

---

## Usage

### Run a single experiment

```bash
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --stage1_method lora \
    --stage2_method lora \
    --output_dir outputs/exp01_llama_lora_lora \
    --do_train --do_eval
```

### CLI arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--model` | HuggingFace model ID | e.g. `meta-llama/Llama-3.1-8B`, `inceptionai/Jais-2-8B-Chat` |
| `--stage1_method` | `lora`, `full`, `none` | Stage 1 method; `none` skips Stage 1 |
| `--stage2_method` | `lora`, `full`, `none` | Stage 2 method; `none` = zero-shot evaluation only |
| `--output_dir` | path | Root directory for checkpoints and results |
| `--do_train` | flag | Run training |
| `--do_eval` | flag | Run evaluation on MedAraBench test set |
| `--stage1_checkpoint` | path | Skip Stage 1 training and use this existing checkpoint |
| `--data_dir` | path | Dataset root (default: `Files/datasets/`) |
| `--eval_batch_size` | int | Inference batch size (default: 16) |

### Run all 18 experiments

```bash
bash run.sh
```

### Run specific experiments

```bash
bash run.sh 1 5 13    # run experiments 1, 5, and 13 only
```

---

## Experiments

### Primary (1–8): Two-stage pipeline

| Exp | Model | Stage 1 | Stage 2 |
|-----|-------|---------|---------|
| 1 | Llama-3.1-8B | LoRA | LoRA |
| 2 | Llama-3.1-8B | Full | Full |
| 3 | Llama-3.1-8B | LoRA | Full |
| 4 | Llama-3.1-8B | Full | LoRA |
| 5 | Jais-2-8B-Chat | LoRA | LoRA |
| 6 | Jais-2-8B-Chat | Full | Full |
| 7 | Jais-2-8B-Chat | LoRA | Full |
| 8 | Jais-2-8B-Chat | Full | LoRA |

### Baselines (9–12): No domain adaptation

| Exp | Model | Stage 1 | Stage 2 |
|-----|-------|---------|---------|
| 9 | Llama-3.1-8B | — | LoRA |
| 10 | Llama-3.1-8B | — | Full |
| 11 | Jais-2-8B-Chat | — | LoRA |
| 12 | Jais-2-8B-Chat | — | Full |

### Zero-shot (13–14): No fine-tuning

| Exp | Model |
|-----|-------|
| 13 | Llama-3.1-8B |
| 14 | Jais-2-8B-Chat |

### Scale ablation (15–18): 70B models (QLoRA only)

| Exp | Model | Stage 1 | Stage 2 |
|-----|-------|---------|---------|
| 15 | Llama-3.1-70B | LoRA | LoRA |
| 16 | Llama-3.1-70B | — | LoRA |
| 17 | Jais-2-70B-Chat | LoRA | LoRA |
| 18 | Jais-2-70B-Chat | — | LoRA |

70B models automatically use 4-bit QLoRA quantization (BitsAndBytes NF4).

---

## Output Structure

Each experiment writes to its `--output_dir`:

```
outputs/exp01_llama_lora_lora/
├── stage1/
│   ├── adapter_config.json       # LoRA adapter config
│   ├── adapter_model.safetensors # LoRA weights
│   ├── tokenizer files
│   └── training_args.json        # Logged hyperparameters
├── stage2/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.json
└── eval/
    ├── results.json              # {accuracy, macro_f1, per_specialty_scores}
    └── predictions.csv           # Per-sample predictions vs. ground truth
```

---

## Key Design Decisions

### Label masking
During training, loss is computed **only on the answer portion** of each sample. Prompt tokens are masked with `-100` so the model learns to predict answers, not reproduce questions.

### Mixed-method experiments
- **LoRA → Full** (Exp 3, 7): The Stage 1 LoRA adapter is merged into the base model weights (`merge_and_unload()`) before Stage 2 full fine-tuning.
- **Full → LoRA** (Exp 4, 8): The Stage 1 full checkpoint is loaded, then a fresh LoRA adapter is applied on top for Stage 2.

### Log-probability evaluation
No text generation is used. For each test sample:
1. The prompt (without the answer) is tokenized and passed through the model.
2. Logits at the final token position are extracted.
3. The predicted answer is `argmax` over the token IDs for `A`, `B`, `C`, `D`, `E`.

This is deterministic, fast, and avoids generation artifacts.

### MedAraBench data cleaning
Raw training data (19,891 samples) is cleaned to ~17,638 by removing:
- Multi-label answers (`A+B`, `A,B`, etc.)
- Text answers (not a single letter)
- Lowercase answers (normalized to uppercase)
- Samples where answer is `E` but `Option E` is blank
- Duplicate questions (keep first occurrence)

---

## Hyperparameters

| | LoRA | Full FT |
|-|------|---------|
| Learning rate | 2e-4 | 2e-5 |
| Batch size (per device) | 4 | 2 |
| Gradient accumulation | 4 | 8 |
| Effective batch size | 16 | 16 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |
| Weight decay | — | 0.01 |
| Stage 1 epochs | 3 | 3 |
| Stage 2 epochs | 5 | 5 |
| Max seq length | 2048 | 2048 |
| Precision | bf16 | bf16 |
| LoRA r | 16 | — |
| LoRA alpha | 32 | — |
| LoRA dropout | 0.05 | — |
| LoRA target modules | q/k/v/o/gate/up/down\_proj | — |

---

## Metrics

- **Accuracy**: correct predictions / total samples
- **Macro F1**: sklearn `f1_score(average='macro')` across answer classes
- Both reported overall and broken down by `Medical Specialty`
