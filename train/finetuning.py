"""
Stage 2: Task-Specific Fine-tuning on MedAraBench (MCQ) dataset.

Handles all combinations:
  - LoRA → LoRA   (continue with same adapter)
  - Full → Full   (continue with full model)
  - LoRA → Full   (merge adapter, then full fine-tune)
  - Full → LoRA   (load full checkpoint, apply fresh LoRA adapter)
  - None → LoRA   (fresh base model + LoRA)
  - None → Full   (fresh base model, full fine-tune)

Logs training loss and per-epoch validation loss to W&B.
Uploads the final checkpoint to HuggingFace Hub.
"""

import os
import json
import yaml
import transformers
from transformers import TrainerCallback

from trl import SFTTrainer, SFTConfig
from peft import get_peft_model

from data.read_data import load_medarabench
from data.clean_data import clean_medarabench
from utils.get_model import (
    load_model_and_tokenizer,
    load_from_checkpoint,
    merge_lora_and_save,
    _build_lora_config,
)
from utils.prompt_template import format_medarabench_sample
from utils import wandb_logger


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # PyYAML parses scientific notation (e.g. 2e-4) as strings — cast floats explicitly
    train = cfg.get("training", {})
    for key in ("learning_rate", "warmup_ratio", "weight_decay"):
        if key in train:
            train[key] = float(train[key])
    return cfg


class _ValLossCallback(TrainerCallback):
    """Log validation loss to W&B at the end of each evaluation."""

    def __init__(self, stage: str):
        self.stage = stage

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        val_loss = metrics.get("eval_loss")
        if val_loss is not None:
            epoch = int(state.epoch) if state.epoch else state.global_step
            wandb_logger.log_validation_loss(val_loss, epoch=epoch, stage=self.stage)
        return control


def run_task_finetuning(
    model_name: str,
    method: str,
    config_path: str,
    output_dir: str,
    stage1_checkpoint: str | None = None,
    stage1_method: str | None = None,
    data_dir: str = "Files/datasets/",
    load_in_4bit: bool = False,
    hf_token: str | None = None,
    hf_private: bool = False,
    val_split: float = 0.05,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    dry_run: bool = False,
):
    """
    Run Stage 2 task-specific fine-tuning on MedAraBench.

    Args:
        model_name:        HuggingFace model identifier
        method:            Stage 2 fine-tuning method: "lora" or "full"
        config_path:       path to lora.yaml or full_ft.yaml for Stage 2
        output_dir:        directory to save Stage 2 checkpoint
        stage1_checkpoint: path to Stage 1 checkpoint (None → use fresh base model)
        stage1_method:     method used in Stage 1 ("lora", "full", or None)
        data_dir:          root of dataset directory
        load_in_4bit:      enable QLoRA (for 70B models)
        hf_token:          HF API token (falls back to HF_TOKEN env var).
                           If None and HF_TOKEN env var is unset, upload is skipped.
        hf_private:        whether the HF repo should be private
        val_split:         fraction of cleaned training data held out for val loss
        max_train_samples: cap training set size (for dry-run / smoke tests)
        max_eval_samples:  cap validation set size (for dry-run / smoke tests)
        dry_run:           if True: force 1 epoch, fp32, skip HF upload
    """
    cfg = _load_config(config_path)
    train_cfg = cfg["training"]
    num_epochs = 1 if dry_run else cfg["stages"]["task_specific"]["num_train_epochs"]
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    lora_cfg = cfg.get("lora", {})
    use_bf16 = train_cfg.get("bf16", True) and not dry_run

    transformers.set_seed(train_cfg.get("seed", 42))

    print(f"\n{'='*60}")
    print(f"Stage 2: Task-Specific Fine-tuning")
    print(f"  Model       : {model_name}")
    print(f"  S1 method   : {stage1_method or 'none'}")
    print(f"  S2 method   : {method}")
    print(f"  S1 ckpt     : {stage1_checkpoint or 'N/A (fresh base)'}")
    print(f"  Epochs      : {num_epochs}")
    print(f"  Output      : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Load model depending on Stage 1 / Stage 2 method combination
    # ------------------------------------------------------------------ #
    if stage1_checkpoint is None:
        # No Stage 1 — load fresh base model
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            method=method,
            lora_config=lora_cfg,
            load_in_4bit=load_in_4bit,
        )

    elif stage1_method == "lora" and method == "full":
        # LoRA Stage 1 → Full Stage 2: merge adapter first
        print("Merging Stage 1 LoRA adapter into base model...")
        merged_dir = os.path.join(output_dir, "_merged_stage1")
        os.makedirs(merged_dir, exist_ok=True)

        s1_model, s1_tokenizer = load_from_checkpoint(
            checkpoint_path=stage1_checkpoint,
            base_model_name=model_name,
            method="lora",
            load_in_4bit=load_in_4bit,
        )
        model = merge_lora_and_save(s1_model, merged_dir, tokenizer=s1_tokenizer)
        tokenizer = s1_tokenizer
        for param in model.parameters():
            param.requires_grad = True

    elif stage1_method == "full" and method == "lora":
        # Full Stage 1 → LoRA Stage 2: load full checkpoint, add fresh LoRA
        print("Loading Stage 1 full checkpoint and applying fresh LoRA adapter...")
        model, tokenizer = load_from_checkpoint(
            checkpoint_path=stage1_checkpoint,
            base_model_name=model_name,
            method="full",
            load_in_4bit=load_in_4bit,
        )
        peft_cfg = _build_lora_config(lora_cfg)
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    else:
        # Same method (lora→lora or full→full): load from checkpoint directly
        model, tokenizer = load_from_checkpoint(
            checkpoint_path=stage1_checkpoint,
            base_model_name=model_name,
            method=stage1_method,
            load_in_4bit=load_in_4bit,
        )

    # ------------------------------------------------------------------ #
    # Load, clean, and split MedAraBench training data
    # ------------------------------------------------------------------ #
    print("Loading and cleaning MedAraBench training data...")
    raw_dataset = load_medarabench(split="train", data_dir=data_dir)
    clean_dataset = clean_medarabench(raw_dataset)

    split = clean_dataset.train_test_split(test_size=val_split, seed=train_cfg.get("seed", 42))
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    print(f"  Train: {len(train_dataset):,}  |  Val: {len(eval_dataset):,}")

    def formatting_func(sample):
        return format_medarabench_sample(sample, include_answer=True)

    # ------------------------------------------------------------------ #
    # SFTConfig
    # ------------------------------------------------------------------ #
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        eval_accumulation_steps=8,
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        bf16=use_bf16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 50),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        eval_strategy="epoch",
        seed=train_cfg.get("seed", 42),
        optim=train_cfg.get("optimizer", "adamw_torch"),
        max_length=max_seq_length,
        packing=False,
        report_to="wandb",
        run_name=os.environ.get("WANDB_NAME") or f"{model_name.split('/')[-1].lower()}_s1-{stage1_method or 'none'}_s2-{method}",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        callbacks=[_ValLossCallback(stage="stage2")],
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print("Starting Stage 2 training...")
    trainer.train()

    print(f"\nSaving Stage 2 checkpoint to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ------------------------------------------------------------------ #
    # Save metadata
    # ------------------------------------------------------------------ #
    meta = {
        "model_name": model_name,
        "stage1_method": stage1_method,
        "stage2_method": method,
        "stage": "task_specific",
        "num_epochs": num_epochs,
        "config_path": config_path,
        "stage1_checkpoint": stage1_checkpoint,
        "train_samples": len(train_dataset),
        "val_samples": len(eval_dataset),
    }
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Upload to HF Hub
    # ------------------------------------------------------------------ #
    hf_repo_id = None
    resolved_token = hf_token or os.environ.get("HF_TOKEN")
    if resolved_token and not dry_run:
        from utils.hf_hub import upload_checkpoint_to_hub
        hf_repo_id = upload_checkpoint_to_hub(
            checkpoint_dir=output_dir,
            model_name=model_name,
            stage="stage2",
            s1_method=stage1_method or "none",
            s2_method=method,
            hf_token=resolved_token,
            private=hf_private,
            extra_metadata=meta,
        )
        wandb_logger.log_hf_repo(hf_repo_id, stage="stage2")

    print("Stage 2 complete.")
    return output_dir, hf_repo_id
