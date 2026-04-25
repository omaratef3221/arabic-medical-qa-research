# LoRA vs. Full Fine-Tuning for Arabic Medical Question Answering

**Experimental codebase for a NeurIPS 2026 submission.**

A complete two-stage fine-tuning pipeline comparing LoRA and full fine-tuning across Llama-3.1-8B and Jais-2-8B-Chat on Arabic medical multiple-choice QA, using log-probability evaluation on the MedAraBench benchmark.

---

## Table of Contents

1. [Research Overview](#research-overview)
2. [Project Structure](#project-structure)
3. [Datasets](#datasets)
4. [Data Preprocessing](#data-preprocessing)
5. [Models](#models)
6. [Two-Stage Training Pipeline](#two-stage-training-pipeline)
7. [Evaluation Protocol](#evaluation-protocol)
8. [Experiment Grid](#experiment-grid)
9. [Hyperparameters](#hyperparameters)
10. [Prompt Templates](#prompt-templates)
11. [Logging and Tracking](#logging-and-tracking)
12. [Infrastructure](#infrastructure)
13. [Usage](#usage)
14. [Critical Bugs Fixed](#critical-bugs-fixed)
15. [Reproducibility](#reproducibility)

---

## Research Overview

### Research Question

*Does two-stage fine-tuning (domain adaptation followed by task-specific training) improve Arabic medical QA performance over direct task fine-tuning, and how does LoRA compare to full fine-tuning at each stage?*

### Methodology

The pipeline implements a two-stage training approach:

- **Stage 1 (Domain Adaptation):** Continual pre-training on ~110K Arabic medical QA pairs from AraMed to inject domain knowledge into the base LLM.
- **Stage 2 (Task-Specific Fine-tuning):** Supervised fine-tuning on ~17.6K cleaned multiple-choice questions from MedAraBench to align the model with the downstream MCQ format.

Each stage independently uses **LoRA** (parameter-efficient) or **full fine-tuning** (all parameters), yielding a 2x2 experimental matrix per model. Additional baselines skip Stage 1 entirely or evaluate zero-shot.

### Key Contributions

1. First systematic comparison of LoRA vs. full fine-tuning for Arabic medical LLMs across a two-stage pipeline.
2. Evaluation across two model families: an English-dominant model (Llama-3.1) and an Arabic-native model (Jais-2).
3. Log-probability evaluation protocol that eliminates generation-related confounds (decoding strategy, stopping criteria).

---

## Project Structure

```
script/
├── main.py                     # Pipeline orchestrator (CLI entry point)
├── run.sh                      # Shell runner for sequential experiments
├── requirements.txt            # Python dependencies
├── .env                        # Credentials (WANDB_API_KEY, HF_TOKEN) — not committed
│
├── configs/
│   ├── lora.yaml               # LoRA hyperparameters + training config
│   └── full_ft.yaml            # Full fine-tuning training config
│
├── data/
│   ├── read_data.py            # Dataset loading (AraMed, MedAraBench)
│   └── clean_data.py           # MedAraBench cleaning pipeline
│
├── train/
│   ├── adaptation.py           # Stage 1: domain adaptation on AraMed
│   └── finetuning.py           # Stage 2: task-specific fine-tuning on MedAraBench
│
├── evaluation/
│   └── evaluate.py             # Log-probability evaluation on MedAraBench test set
│
├── utils/
│   ├── get_model.py            # Model loading, LoRA application, checkpoint management
│   ├── prompt_template.py      # Prompt formatting for both datasets
│   ├── metrics.py              # Accuracy, Macro F1, per-specialty metrics
│   ├── wandb_logger.py         # Weights & Biases logging utilities
│   ├── hf_hub.py               # HuggingFace Hub upload utilities
│   └── env_loader.py           # .env file parser for credentials
│
├── slurm/
│   ├── jobscript.sh            # Single-experiment SLURM submission
│   ├── jobscript_4_gpu.sh      # Multi-experiment 4-GPU SLURM submission
│   ├── job_array.sh            # Array job for parallel experiments
│   └── logs/                   # SLURM stdout/stderr logs
│
└── Files/
    └── datasets/
        ├── AraMed Train.csv    # AraMed training set (~110K samples)
        ├── AraMed Test.csv     # AraMed test set
        ├── MedAraBench Train.csv  # MedAraBench training set (~19.9K raw)
        └── MedAraBench Test.csv   # MedAraBench test set (~4,959 raw)
```

---

## Datasets

### AraMed (Stage 1 — Domain Adaptation)

| Property | Value |
|----------|-------|
| **Purpose** | Domain adaptation — inject Arabic medical knowledge |
| **Format** | Open-ended QA pairs (question + free-text answer) |
| **Train size** | ~110,000 samples |
| **Language** | Arabic |
| **Source** | Arabic medical QA corpus |
| **Columns** | `question`, `answer` |

### MedAraBench (Stage 2 — Task Fine-tuning & Evaluation)

| Property | Value |
|----------|-------|
| **Purpose** | Task-specific fine-tuning and evaluation |
| **Format** | 5-choice MCQ (A/B/C/D/E) with medical specialty labels |
| **Train size (raw)** | 19,891 samples |
| **Train size (cleaned)** | 17,638 samples |
| **Test size (raw)** | 4,959 samples |
| **Test size (cleaned)** | 4,761 samples |
| **Language** | Arabic |
| **Columns** | `question`, `option_a`–`option_e`, `answer`, `specialty` |

---

## Data Preprocessing

### AraMed CSV Robustness (`data/read_data.py`)

The AraMed training CSV contains malformed quoted strings in Arabic medical text (stray quote characters causing `ParserError` at row 108,971). The loader uses a two-pass strategy:

```python
def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", quoting=csv.QUOTE_NONE,
                           on_bad_lines="skip", encoding="utf-8",
                           encoding_errors="replace")
```

1. **First pass:** Standard C engine CSV reader.
2. **Fallback:** Python engine with `QUOTE_NONE` and `on_bad_lines="skip"` to handle malformed rows gracefully.

### MedAraBench Cleaning (`data/clean_data.py`)

The MedAraBench dataset undergoes a 5-step cleaning pipeline applied to **both train and test splits** (after the cleaning bug fix described in [Critical Bugs Fixed](#critical-bugs-fixed)):

| Step | Description | Train Removed | Test Removed |
|------|-------------|--------------|--------------|
| 1 | Normalize answer to single uppercase letter (A–E) | — | — |
| 2 | Remove multi-label answers (e.g., "A+B", "A and B") | 79 | 21 |
| 3 | Remove samples with empty question text | 1 | 1 |
| 4 | Remove samples where answer=E but `option_e` is blank | 56 | 10 |
| 5 | Deduplicate by question text (keep first) | 2,117 | 166 |
| **Total** | | **2,253** | **198** |

**Final cleaned counts:**
- **Train:** 17,638 samples (from 19,891 raw)
- **Test:** 4,761 samples (from 4,959 raw)

Both splits guarantee answer values are exclusively `['A', 'B', 'C', 'D', 'E']`.

Garbage filtered out includes: Arabic medical text in answer field (e.g., `'الجسم اللوزي'`), multi-label answers (`'A+B'`, `'C, A'`, `'D+E'`), question marks (`'?'`), out-of-range letters (`'F'`, `'V'`, `'S'`), and free-text rejections (`'None of the above'`).

### Validation Splits

| Dataset | Train/Val Split | Strategy |
|---------|-----------------|----------|
| AraMed (Stage 1) | 98% / 2% | Random split, seed=42 |
| MedAraBench (Stage 2) | 95% / 5% | Random split, seed=42 |

---

## Models

### Base Models

| Model | Parameters | Architecture | Language Focus | HuggingFace ID |
|-------|-----------|-------------|----------------|----------------|
| **Llama-3.1-8B** | 8B | LlamaForCausalLM | English-dominant, multilingual | `meta-llama/Llama-3.1-8B` |
| **Jais-2-8B-Chat** | 8B | GPT-style | Arabic-native, bilingual | `inceptionai/Jais-2-8B-Chat` |

Both models load in **bfloat16** with `device_map="auto"`.

### Model Loading (`utils/get_model.py`)

- **LoRA:** Loads base model in bf16, applies `PeftModel` with LoRA adapter via `get_peft_model()`. Calls `enable_input_require_grads()` on the base before wrapping (required for gradient checkpointing + LoRA).
- **Full:** Loads base model in bf16, sets `requires_grad=True` for all parameters.
- **QLoRA (70B models):** Uses `BitsAndBytesConfig` with NF4 quantization and double quantization.

### LoRA Configuration

```yaml
r: 16                    # rank
lora_alpha: 32            # scaling factor (alpha/r = 2)
lora_dropout: 0.05
target_modules:           # all linear projection layers
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
task_type: CAUSAL_LM
bias: none
```

**Trainable parameters:** ~0.5% of total model parameters (LoRA, ~42M of 8B) vs. 100% (full fine-tuning).

### Checkpoint Management — All 6 Stage 1 → Stage 2 Transitions

| Stage 1 | Stage 2 | Transition Logic |
|---------|---------|------------------|
| **None** | LoRA | Fresh base model + fresh LoRA adapter |
| **None** | Full | Fresh base model, all params trainable |
| **LoRA** | LoRA | Load S1 adapter, **merge_and_unload()**, apply fresh trainable LoRA on top |
| **LoRA** | Full | Load S1 adapter, **merge_and_unload()**, set all params trainable |
| **Full** | LoRA | Load full S1 checkpoint, apply fresh trainable LoRA |
| **Full** | Full | Load full S1 checkpoint, set all params trainable |

**Why merge_and_unload() is critical for LoRA→LoRA:** PEFT's `PeftModel.from_pretrained()` loads the adapter in inference mode with `requires_grad=False`. Without merging, no fresh adapter is added → grad_norm stays at 0 → no learning. See [Critical Bugs Fixed](#critical-bugs-fixed).

---

## Two-Stage Training Pipeline

### Stage 1: Domain Adaptation (`train/adaptation.py`)

**Objective:** Continual pre-training on AraMed to inject Arabic medical domain knowledge.

**Framework:** `SFTTrainer` from TRL with `SFTConfig`.

| Parameter | LoRA | Full FT |
|-----------|------|---------|
| Optimizer | AdamW 8-bit | AdamW 8-bit |
| Learning rate | 2e-4 | 2e-5 |
| LR scheduler | Cosine decay | Cosine decay |
| Warmup ratio | 3% | 3% |
| Effective batch size | 16 (8 × 2 accum.) | 16 (4 × 4 accum.) |
| Max sequence length | 512 tokens | 512 tokens |
| Precision | bfloat16 | bfloat16 |
| Gradient checkpointing | Enabled | Enabled |
| Epochs | 1 | 1 |

**Why 1 epoch for Stage 1:** With ~110K samples, a single epoch provides sufficient domain exposure. Multiple epochs risk overfitting to AraMed's QA format. Follows continual pre-training practice in LLaMA-2 and BioMedLM.

**Sequence length justification:** AraMed has p99=~450 tokens. The 512-token limit truncates only ~1% of samples.

### Stage 2: Task-Specific Fine-tuning (`train/finetuning.py`)

**Objective:** Supervised fine-tuning on MedAraBench MCQ to align the model with the 5-choice answer format.

| Parameter | LoRA | Full FT |
|-----------|------|---------|
| Optimizer | AdamW 8-bit | AdamW 8-bit |
| Learning rate | 2e-4 | 2e-5 |
| LR scheduler | Cosine decay | Cosine decay |
| Warmup ratio | 3% | 3% |
| Effective batch size | 16 | 16 |
| Max sequence length | 512 tokens | 512 tokens |
| Precision | bfloat16 | bfloat16 |
| Gradient checkpointing | Enabled | Enabled |
| Epochs | 3 | 3 |
| Validation eval | Per epoch | Per epoch |
| Per-device eval batch size | 1 | 1 |
| Eval accumulation steps | 8 | 8 |

**Sequence length justification:** MedAraBench MCQ prompts are short (p99=~85 tokens). The 512-token limit is more than sufficient.

### Stage 2 Sanity Check (post-bug-fix)

Before training begins, the pipeline verifies trainable parameters fall in expected ranges:

```python
# In train/finetuning.py
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
pct = 100 * trainable / total

if method == "lora":
    assert 0.01 < pct < 10.0, "LoRA Stage 2 should have 0.01-10% trainable params"
elif method == "full":
    assert pct > 99.0, "Full Stage 2 should have ~100% trainable params"

# Verify optimizer is tracking these parameters
trainer.create_optimizer()
optimizer_params = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
assert optimizer_params > 0, "Optimizer has no parameters to update!"
```

The job aborts immediately if either check fails — saving hours of zero-gradient training.

### Optimizer: AdamW 8-bit

Both stages use `adamw_8bit` from bitsandbytes, storing optimizer momentum in 8-bit precision. Reduces optimizer memory by ~50% (from ~32GB to ~8GB for 8B models) with negligible convergence impact.

---

## Evaluation Protocol

### Log-Probability Method (`evaluation/evaluate.py`)

The evaluation uses **log-probability selection** (no text generation):

1. Format each test MCQ as a prompt (question + options A–E, without the answer).
2. Run a single forward pass through the model.
3. Extract logits at the **last token position** of the prompt.
4. Select logits corresponding to the token IDs of letters A, B, C, D, E.
5. The predicted answer is `argmax` over these 5 logits.

```
Predicted answer = argmax over l in {A,B,C,D,E} of logit(token_id(l))
```

**Advantages over generation-based evaluation:**
- Deterministic (no sampling/temperature/decoding strategy confounds)
- Fast (single forward pass, no autoregressive generation)
- Reproducible across runs

### Token ID Resolution

The evaluator resolves answer token IDs for each tokenizer by trying both plain (`"A"`) and space-prefixed (`" A"`) encodings, selecting whichever produces a single token. Handles differences between Llama and Jais tokenizers.

### Test Set Cleaning (post-bug-fix)

The MedAraBench test set is now **cleaned before evaluation** (this was previously skipped — see [Critical Bugs Fixed](#critical-bugs-fixed)):

```python
# In evaluation/evaluate.py:run_evaluation()
raw_test = load_medarabench(split="test", data_dir=data_dir)
test_dataset = clean_medarabench(raw_test)
print(f"Test samples: {len(raw_test):,} raw → {len(test_dataset):,} clean")
```

This ensures:
- All reference labels are valid `A`–`E` letters
- No samples with empty questions, missing options, or duplicate questions
- Metrics computed on 4,761 clean samples instead of 4,959 polluted ones

### Inference Details

- **Padding side:** Left (for batched inference)
- **Batch size:** 16 (configurable)
- **Truncation:** 2,048 tokens max
- **Precision:** bfloat16

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correctly predicted answers |
| **Macro F1** | Unweighted mean of per-class F1 scores across all 5 answer labels |
| **Per-specialty accuracy** | Accuracy broken down by medical specialty |
| **Per-specialty Macro F1** | Macro F1 broken down by medical specialty |

Metrics computed using scikit-learn (`accuracy_score`, `f1_score(average='macro')`).

### Predictions CSV

Per-sample predictions are saved to `outputs/{exp}/eval/predictions.csv` with `quoting=csv.QUOTE_ALL` to prevent Arabic question text containing commas from corrupting the CSV columns.

---

## Experiment Grid

### 8B Experiments (14 total)

| Exp | Model | Stage 1 | Stage 2 | Category |
|-----|-------|---------|---------|----------|
| 1 | Llama-3.1-8B | LoRA | LoRA | Two-stage |
| 2 | Llama-3.1-8B | Full | Full | Two-stage |
| 3 | Llama-3.1-8B | LoRA | Full | Two-stage (mixed) |
| 4 | Llama-3.1-8B | Full | LoRA | Two-stage (mixed) |
| 5 | Jais-2-8B-Chat | LoRA | LoRA | Two-stage |
| 6 | Jais-2-8B-Chat | Full | Full | Two-stage |
| 7 | Jais-2-8B-Chat | LoRA | Full | Two-stage (mixed) |
| 8 | Jais-2-8B-Chat | Full | LoRA | Two-stage (mixed) |
| 9 | Llama-3.1-8B | None | LoRA | Baseline (no S1) |
| 10 | Llama-3.1-8B | None | Full | Baseline (no S1) |
| 11 | Jais-2-8B-Chat | None | LoRA | Baseline (no S1) |
| 12 | Jais-2-8B-Chat | None | Full | Baseline (no S1) |
| 13 | Llama-3.1-8B | None | None | Zero-shot |
| 14 | Jais-2-8B-Chat | None | None | Zero-shot |

### 70B Scale Ablation (4 total)

| Exp | Model | Stage 1 | Stage 2 | Notes |
|-----|-------|---------|---------|-------|
| 15 | Llama-3.1-70B | LoRA | LoRA | QLoRA (4-bit) |
| 16 | Llama-3.1-70B | None | LoRA | QLoRA baseline |
| 17 | Jais-2-70B-Chat | LoRA | LoRA | QLoRA (4-bit) |
| 18 | Jais-2-70B-Chat | None | LoRA | QLoRA baseline |

---

## Hyperparameters

### LoRA Configuration (`configs/lora.yaml`)

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  task_type: CAUSAL_LM
  bias: none

training:
  optimizer: adamw_8bit
  learning_rate: 0.0002       # 2e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2   # effective batch size = 16
  max_seq_length: 512
  bf16: true
  gradient_checkpointing: true
  seed: 42

stages:
  domain_adaptation:
    num_train_epochs: 1
  task_specific:
    num_train_epochs: 3
```

### Full Fine-tuning Configuration (`configs/full_ft.yaml`)

```yaml
training:
  optimizer: adamw_8bit
  learning_rate: 0.00002      # 2e-5 (10x lower than LoRA)
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.01
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch size = 16
  max_seq_length: 512
  bf16: true
  gradient_checkpointing: true
  seed: 42

stages:
  domain_adaptation:
    num_train_epochs: 1
  task_specific:
    num_train_epochs: 3
```

**Key design choices:**
- **Learning rate:** LoRA uses 10x higher LR (2e-4 vs 2e-5) — higher LR compensates for the restricted parameter space.
- **Weight decay:** Applied only to full FT (0.01) as a regularizer; LoRA's low-rank constraint provides implicit regularization.
- **Batch size:** Full FT uses smaller per-device batch (4 vs 8) due to higher memory from optimizer states; gradient accumulation maintains the effective batch size of 16.

---

## Prompt Templates

### AraMed (Stage 1 — Open-ended QA)

```
### Question:
{question}

### Answer:
{answer}
```

### MedAraBench (Stage 2 & Evaluation — MCQ)

**Training (with answer):**
```
### Question:
{question}

### Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

### Answer:
{answer_letter}
```

**Evaluation (without answer):**
```
### Question:
{question}

### Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
E) {option_e}

### Answer:

```

The model's prediction is determined by which answer letter token (A/B/C/D/E) receives the highest logit at the position immediately after `### Answer:\n`.

---

## Logging and Tracking

### Weights & Biases (`utils/wandb_logger.py`)

All experiments log to a shared W&B project with:

| Logged Item | When | Details |
|-------------|------|---------|
| **Training loss** | Every 50 steps | Automatic via SFTTrainer `report_to="wandb"` |
| **Validation loss** | End of each epoch | Custom `_ValLossCallback` |
| **Dataset statistics** | Run initialization | Train/val/test sizes, cleaning breakdown |
| **Evaluation accuracy** | After eval | Overall accuracy |
| **Evaluation Macro F1** | After eval | Overall macro F1 |
| **Per-specialty results** | After eval | W&B Table with per-specialty accuracy and F1 |
| **HF repo link** | After upload | Link to uploaded model on HuggingFace Hub |

**Run naming convention:** `{model-short}_s1-{method}_s2-{method}`
Example: `llama-3.1-8b_s1-lora_s2-lora`

The run name is set both via `wandb.init(name=...)` AND `os.environ["WANDB_NAME"]` AND `SFTConfig.run_name=...` to prevent the trainer from overriding with a random name (e.g., "crashed-deep-yogurt"). All three layers must agree.

**Tags:** Model name, stage 1 method, stage 2 method, model size (8b/70b).

### HuggingFace Hub (`utils/hf_hub.py`)

Checkpoints are automatically uploaded to HuggingFace Hub after each training stage:

- **Repo naming for Stage 1:** `{owner}/{model-short}-s1-{method}-aramed`
- **Repo naming for Stage 2:** `{owner}/{model-short}-s1-{s1-method}-s2-{s2-method}-medarabench`
- **Owner:** Resolved dynamically from `api.whoami()` using the `HF_TOKEN`
- **Metadata:** Training args, dataset stats, and stage info saved as model card

Stage 1 checkpoints can be reused as `--stage1_checkpoint omaratef3221/llama-3.1-8b-s1-lora-aramed` to skip retraining.

### Credentials (`.env`)

```env
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=arabic-medical-llm
HF_TOKEN=your_hf_token_here
```

Loaded automatically by `utils/env_loader.py` at pipeline startup. Environment variables already set take precedence (env > `.env` file). No manual `wandb login` or `huggingface-cli login` required.

---

## Infrastructure

### Hardware

| Resource | 8B Experiments | 70B Experiments |
|----------|---------------|-----------------|
| **GPU** | 1× or 4× NVIDIA A10G (24 GB) | 4× NVIDIA A10G or 8× A100 |
| **Instance** | AWS g5.12xlarge | AWS g5.12xlarge / p4d.24xlarge |
| **CPU** | 8–24 cores per job | 24+ cores per job |
| **Memory** | 60–180 GB RAM | 180+ GB RAM |

### SLURM Partitions (AWS ParallelCluster)

| Partition | GPUs | Status |
|-----------|------|--------|
| `dcv-1gpu-g5-ond` | 1× A10G | On-demand |
| `dcv-4gpu-g5-ond` | 4× A10G | On-demand |
| `gpu-g5-spt` | 4× A10G | Spot |
| `dcv-p4d-ond` | 8× A100 | On-demand |
| `dcv-p4d-spt` | 8× A100 | Spot |

### SLURM Job Submission

**Single experiment:**
```bash
sbatch slurm/jobscript.sh
```

**Multiple experiments on 4-GPU node:**
```bash
sbatch slurm/jobscript_4_gpu.sh
```

The `jobscript_4_gpu.sh` runs:
- **Exp 4 & 8:** Full Stage 1 + LoRA Stage 2 (full train, both stages)
- **Exps 1, 2, 3, 5, 6, 7:** Stage 2 retrain only, reusing Stage 1 from HuggingFace Hub
- **Exps 9–12:** Baseline experiments (no Stage 1)
- **Exps 13 & 14:** Zero-shot evaluation only

**Switch partition on the fly:**
```bash
sbatch --partition=gpu-g5-spt slurm/jobscript_4_gpu.sh
```

### Memory Optimizations

| Optimization | Memory Saved | Description |
|---|---|---|
| `adamw_8bit` optimizer | ~50% optimizer states | 8-bit Adam momentum via bitsandbytes |
| `gradient_checkpointing` | ~40% activation memory | Recompute activations during backward pass |
| `enable_input_require_grads()` | — | Required when combining gradient checkpointing + LoRA on a frozen base; without it, gradients don't flow |
| `max_seq_length=512` | ~4× less attention memory | Justified by dataset statistics (p99 < 500 tokens) |
| `per_device_eval_batch_size=1` | Prevents eval OOM | Evaluation pass uses less memory than training |
| `eval_accumulation_steps=8` | Prevents eval OOM | Accumulates predictions before moving to CPU |
| bfloat16 training | 2× vs fp32 | Standard mixed-precision training |

---

## Usage

### Prerequisites

```bash
# Create conda environment
conda create -n medical_llm python=3.10 -y
conda activate medical_llm

# Install dependencies
pip install -r requirements.txt

# Note: pyarrow may need to be installed via conda due to glibc version requirements:
# conda install -c conda-forge pyarrow -y
```

### Local Dry-Run (Mac/CPU — no GPU required)

```bash
# Zero-shot evaluation on 50 samples
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --stage1_method none --stage2_method none \
    --output_dir outputs/test_dryrun \
    --do_eval --dry_run
```

The `--dry_run` flag automatically:
- Caps training to 100 samples, evaluation to 50 samples
- Forces 1 epoch
- Disables bfloat16 (uses fp32 for CPU compatibility)
- Disables W&B logging and HF Hub upload

### Full Experiments (GPU)

```bash
# Experiment 1: Llama + LoRA → LoRA (two-stage)
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --stage1_method lora --stage2_method lora \
    --output_dir outputs/exp01_llama_lora_lora \
    --do_train --do_eval

# Stage 2 retrain only — reuse Stage 1 from HF Hub
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --stage1_method lora --stage2_method lora \
    --stage1_checkpoint omaratef3221/llama-3.1-8b-s1-lora-aramed \
    --output_dir outputs/exp01_llama_lora_lora \
    --do_train --do_eval

# Zero-shot evaluation
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --stage1_method none --stage2_method none \
    --output_dir outputs/exp13_llama_zeroshot \
    --do_eval
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | HuggingFace model identifier |
| `--stage1_method` | `lora` | Stage 1 method: `lora`, `full`, or `none` |
| `--stage2_method` | `lora` | Stage 2 method: `lora`, `full`, or `none` |
| `--output_dir` | (required) | Output directory for checkpoints and results |
| `--do_train` | `False` | Run training |
| `--do_eval` | `False` | Run evaluation |
| `--stage1_checkpoint` | `None` | Pre-existing Stage 1 checkpoint path or HF repo ID |
| `--data_dir` | `Files/datasets/` | Dataset root directory |
| `--eval_batch_size` | `16` | Evaluation inference batch size |
| `--wandb_project` | `arabic-medical-llm` | W&B project name |
| `--no_wandb` | `False` | Disable W&B logging |
| `--hf_private` | `False` | Make uploaded HF repos private |
| `--max_train_samples` | `None` | Cap training set size |
| `--max_eval_samples` | `None` | Cap validation/test set size |
| `--dry_run` | `False` | Smoke test mode (100 train, 50 eval, 1 epoch, fp32) |

---

## Critical Bugs Fixed

This codebase went through several rounds of debugging during the experimental campaign. The fixes below were essential for correct results.

### Bug 1: Test Set Was Never Cleaned (evaluation/evaluate.py)

**Symptom:** Evaluation accuracy and Macro F1 were artificially deflated. The MedAraBench test set contained 198 samples with garbage answers (Arabic medical text, multi-label like `'A+B'`, `'?'`, `'V'`, etc.) that were being evaluated as if they had valid letter answers.

**Root cause:** `run_evaluation()` loaded the raw test set without calling `clean_medarabench()`, even though the cleaning function existed and was being applied to training data.

**Fix:** Added cleaning in `evaluation/evaluate.py:225-227`:
```python
raw_test = load_medarabench(split="test", data_dir=data_dir)
test_dataset = clean_medarabench(raw_test)
```

**Impact:** Test set went from 4,959 polluted → 4,761 clean. All previously-reported metrics were slightly pessimistic.

### Bug 2: Stage 2 Training Did Nothing for LoRA → LoRA (train/finetuning.py)

**Symptom:** During Stage 2 of `lora→lora` experiments, `grad_norm` was 0 for every step, training loss was stuck at ~2.7, and token accuracy was flat at 0.50. After 10+ hours of "training", model performance was identical to the Stage 1 checkpoint.

**Root cause:** When loading a Stage 1 LoRA adapter via `PeftModel.from_pretrained()`, the adapter is loaded in **inference mode with `requires_grad=False`**. The original code took this loaded model and passed it directly to the trainer without applying a fresh, trainable adapter — resulting in zero trainable parameters and zero gradient flow.

**Fix:** Rewrote the Stage 2 loading logic in `train/finetuning.py` for all 6 scenarios. For `lora→lora`:

```python
# Load S1 adapter (inference mode)
s1_model, tokenizer = load_from_checkpoint(...)

# Merge S1 adapter into base weights
merged_base = s1_model.merge_and_unload()

# Required when combining gradient checkpointing + LoRA on a frozen base
if hasattr(merged_base, "enable_input_require_grads"):
    merged_base.enable_input_require_grads()

# Apply a fresh trainable LoRA adapter on top
peft_cfg = _build_lora_config(lora_cfg)
model = get_peft_model(merged_base, peft_cfg)
```

**Mandatory sanity check** added before training to catch any future regressions:
```python
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if method == "lora":
    assert 0.01 < pct < 10.0, "LoRA Stage 2 should have 0.01-10% trainable"
elif method == "full":
    assert pct > 99.0, "Full Stage 2 should have ~100% trainable"

trainer.create_optimizer()
optimizer_params = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
assert optimizer_params > 0, "Optimizer has no parameters to update!"
```

**Verified working:** After the fix, `grad_norm` values are 0.15–0.55 (non-zero), loss declines from 2.1 → 1.5+, and `mean_token_accuracy` rises from 0.58 → 0.65+ within the first epoch.

### Bug 3: W&B Run Names Were Random (`crashed-deep-yogurt`)

**Symptom:** W&B runs appeared with random names like `crashed-deep-yogurt` instead of the intended `llama-3.1-8b_s1-lora_s2-lora`.

**Root cause:** `SFTConfig(run_name=None)` in `train/adaptation.py` and `train/finetuning.py` caused HuggingFace Trainer to override the name set by `wandb.init()`.

**Fix:** Set the run name in three places (any one of which can override the others):
1. `os.environ["WANDB_NAME"]` in `main.py` before init
2. `wandb.init(name=run_name)` in `wandb_logger.init_run()`
3. `SFTConfig(run_name=...)` in both training scripts

```python
run_name=os.environ.get("WANDB_NAME") or f"{model_name.split('/')[-1].lower()}_s1-{method}_s2-..."
```

### Bug 4: PyYAML Scientific Notation Parsed as String

**Symptom:** `TypeError: '<=' not supported between float and str` for `learning_rate`.

**Root cause:** PyYAML parses `2e-4` as the **string** `"2e-4"` (it requires a decimal point or explicit exponent sign for float parsing).

**Fix:**
1. Use decimal notation in YAMLs (`learning_rate: 0.0002`, `learning_rate: 0.00002`)
2. Defensive cast in `_load_config()`:
   ```python
   for key in ("learning_rate", "warmup_ratio", "weight_decay"):
       if key in train:
           train[key] = float(train[key])
   ```

### Bug 5: AraMed CSV ParserError at Row 108,971

**Symptom:** `pd.errors.ParserError: EOF inside string` when loading AraMed Train.csv.

**Root cause:** Stray quote characters in Arabic medical text confuse the C engine CSV parser.

**Fix:** Added a fallback to Python engine with `quoting=csv.QUOTE_NONE` and `on_bad_lines="skip"` in `_read_csv_robust()`.

### Bug 6: TRL 1.0.0 Breaking API Changes

Multiple keyword renames between TRL versions broke training:

| Old (broken) | New (fixed) |
|---|---|
| `evaluation_strategy="epoch"` | `eval_strategy="epoch"` |
| `tokenizer=tokenizer` (in SFTTrainer) | `processing_class=tokenizer` |
| `max_seq_length=...` (in SFTTrainer) | `max_length=...` (in SFTConfig) |
| `formatting_func` returns `[str]` | Returns `str` (TRL 1.0 calls per-sample) |

### Bug 7: Predictions CSV Had Arabic Text in Wrong Columns

**Symptom:** `predictions.csv` showed Arabic question text appearing in the prediction column.

**Root cause:** Default CSV writer doesn't quote fields containing commas; Arabic medical questions often contain commas.

**Fix:** Use `csv.QUOTE_ALL`:
```python
writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
```

### Bug 8: End-of-Epoch Validation OOM

**Symptom:** Training completed normally, then OOM during the per-epoch validation pass on 24GB A10G GPUs.

**Fix:** Reduced eval memory footprint:
```python
sft_config = SFTConfig(
    per_device_eval_batch_size=1,
    eval_accumulation_steps=8,
    ...
)
```

---

## Reproducibility

### Random Seeds

All experiments use `seed=42` for:
- Model weight initialization (`transformers.set_seed()`)
- Dataset splitting (train/validation)
- Training data shuffling

### Software Versions

| Package | Version |
|---------|---------|
| PyTorch | >= 2.1.0 |
| Transformers | >= 4.40.0 |
| PEFT | >= 0.10.0 |
| TRL | 1.0.0 |
| bitsandbytes | >= 0.43.0 |
| datasets | >= 2.18.0 |
| accelerate | >= 0.28.0 |
| scikit-learn | >= 1.4.0 |
| wandb | 0.18.7 (pinned) |
| huggingface_hub | >= 0.23.0 |
| Python | 3.10 |

**Notes:**
- `wandb` pinned to 0.18.7 because 0.19+ requires a Go compiler for `wandb-core`
- `flash-attn` is **NOT** used — the AWS cluster's glibc 2.31 is too old (flash-attn requires GLIBC 2.32+)
- `pyarrow` should be installed via conda, not pip (avoids CMake 3.25 requirement)

### Output Structure

Each experiment produces:

```
outputs/exp01_llama_lora_lora/
├── stage1/
│   ├── adapter_model.safetensors   # LoRA weights (or full model)
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── training_args.json          # Metadata
├── stage2/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── training_args.json
└── eval/
    ├── results.json                # Accuracy, Macro F1, per-specialty scores
    └── predictions.csv             # Per-sample predictions (with QUOTE_ALL)
```

### Verifying a Successful Run

A correctly trained Stage 2 LoRA experiment should show:

| Indicator | Expected Value |
|---|---|
| `Trainable params` | ~42M (~0.5% of 8B) |
| `Optimizer tracking` | ~42M parameters |
| `Sanity check PASSED` | Printed before training starts |
| `grad_norm` (early steps) | 0.1–0.6 (non-zero) |
| `loss` trajectory | Declining over training |
| `mean_token_accuracy` | Rising over training |
| `train answers` (after cleaning) | `['A', 'B', 'C', 'D', 'E']` |
| `test answers` (after cleaning) | `['A', 'B', 'C', 'D', 'E']` |
