"""
Stage 1: Domain Adaptation Training on AraMed dataset.

Trains a causal LM (Llama-3.1-8B or Jais-2-8B-Chat) on ~110K Arabic
open-ended medical QA pairs using either LoRA or full fine-tuning.

Logs training loss and validation loss to W&B.
Uploads the final checkpoint to HuggingFace Hub.
"""

import os
import json
import yaml
import transformers
from transformers import TrainerCallback

from trl import SFTTrainer, SFTConfig

from data.read_data import load_aramed
from utils.get_model import load_model_and_tokenizer
from utils.prompt_template import format_aramed_sample
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
    """Compute and log validation loss to W&B at the end of each epoch."""

    def __init__(self, eval_dataset, stage: str):
        self.eval_dataset = eval_dataset
        self.stage = stage

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("model")  # not available here — use on_evaluate instead
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        val_loss = metrics.get("eval_loss")
        if val_loss is not None:
            epoch = int(state.epoch) if state.epoch else state.global_step
            wandb_logger.log_validation_loss(val_loss, epoch=epoch, stage=self.stage)
        return control


def run_domain_adaptation(
    model_name: str,
    method: str,
    config_path: str,
    output_dir: str,
    data_dir: str = "Files/datasets/",
    load_in_4bit: bool = False,
    hf_token: str | None = None,
    hf_private: bool = False,
    val_split: float = 0.02,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    dry_run: bool = False,
):
    """
    Run Stage 1 domain adaptation on AraMed.

    Args:
        model_name:        HuggingFace model identifier
        method:            "lora" or "full"
        config_path:       path to lora.yaml or full_ft.yaml
        output_dir:        directory to save Stage 1 checkpoint
        data_dir:          root of dataset directory
        load_in_4bit:      enable QLoRA quantization (for 70B models)
        hf_token:          HF API token (falls back to HF_TOKEN env var).
                           If None and HF_TOKEN env var is unset, upload is skipped.
        hf_private:        whether the HF repo should be private
        val_split:         fraction of training data to use for validation loss tracking
        max_train_samples: cap training set size (for dry-run / smoke tests)
        max_eval_samples:  cap validation set size (for dry-run / smoke tests)
        dry_run:           if True: force 1 epoch, fp32, skip HF upload
    """
    cfg = _load_config(config_path)
    train_cfg = cfg["training"]
    num_epochs = 1 if dry_run else cfg["stages"]["domain_adaptation"]["num_train_epochs"]
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    lora_cfg = cfg.get("lora", {})
    use_bf16 = train_cfg.get("bf16", True) and not dry_run

    transformers.set_seed(train_cfg.get("seed", 42))

    print(f"\n{'='*60}")
    print(f"Stage 1: Domain Adaptation")
    print(f"  Model  : {model_name}")
    print(f"  Method : {method}")
    print(f"  Epochs : {num_epochs}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        method=method,
        lora_config=lora_cfg,
        load_in_4bit=load_in_4bit,
    )

    # ------------------------------------------------------------------ #
    # Load and split AraMed data
    # ------------------------------------------------------------------ #
    print("Loading AraMed training data...")
    full_dataset = load_aramed(split="train", data_dir=data_dir)
    split = full_dataset.train_test_split(test_size=val_split, seed=train_cfg.get("seed", 42))
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    print(f"  Train: {len(train_dataset):,}  |  Val: {len(eval_dataset):,}")

    def formatting_func(sample):
        return format_aramed_sample(sample)

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
        report_to="wandb",   # training loss streamed to W&B automatically
        run_name=None,       # run name already set by init_run() in main.py
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        callbacks=[_ValLossCallback(eval_dataset, stage="stage1")],
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print("Starting Stage 1 training...")
    trainer.train()

    print(f"\nSaving Stage 1 checkpoint to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ------------------------------------------------------------------ #
    # Save metadata
    # ------------------------------------------------------------------ #
    meta = {
        "model_name": model_name,
        "method": method,
        "stage": "domain_adaptation",
        "num_epochs": num_epochs,
        "config_path": config_path,
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
            stage="stage1",
            s1_method=method,
            s2_method=None,
            hf_token=resolved_token,
            private=hf_private,
            extra_metadata=meta,
        )
        wandb_logger.log_hf_repo(hf_repo_id, stage="stage1")

    print("Stage 1 complete.")
    return output_dir, hf_repo_id
