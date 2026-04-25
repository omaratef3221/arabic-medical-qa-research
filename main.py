"""
Pipeline Orchestrator for Arabic Medical LLM Fine-tuning.

Runs any combination of Stage 1 (domain adaptation) and Stage 2
(task-specific fine-tuning) followed by evaluation on MedAraBench.

Credentials (WANDB_API_KEY, HF_TOKEN) are loaded automatically from a .env
file in the project root, then from environment variables.

Example usage:

  # Exp 1: Llama + LoRA → LoRA
  python main.py \\
      --model meta-llama/Llama-3.1-8B \\
      --stage1_method lora \\
      --stage2_method lora \\
      --output_dir outputs/exp01_llama_lora_lora \\
      --do_train --do_eval

  # Zero-shot (no training)
  python main.py \\
      --model meta-llama/Llama-3.1-8B \\
      --stage1_method none \\
      --stage2_method none \\
      --output_dir outputs/exp13_llama_zeroshot \\
      --do_eval
"""

import argparse
import gc
import os
import sys
import torch

# Make sure project root is on the path when called from a different cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env early so all subsequent imports see the env vars
from utils.env_loader import load_env
load_env()


def _config_path_for(method: str, script_dir: str) -> str:
    if method == "lora":
        return os.path.join(script_dir, "configs", "lora.yaml")
    return os.path.join(script_dir, "configs", "full_ft.yaml")


def _is_70b(model_name: str) -> bool:
    return "70b" in model_name.lower()


def _make_run_name(model_name: str, stage1_method: str, stage2_method: str) -> str:
    short = model_name.split("/")[-1].lower()
    return f"{short}_s1-{stage1_method}_s2-{stage2_method}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arabic Medical LLM Fine-tuning Pipeline"
    )
    parser.add_argument("--model", required=True,
        help="HuggingFace model name (e.g. meta-llama/Llama-3.1-8B)")
    parser.add_argument("--stage1_method", default="lora",
        choices=["lora", "full", "none"],
        help="Stage 1 training method. 'none' skips Stage 1.")
    parser.add_argument("--stage2_method", default="lora",
        choices=["lora", "full", "none"],
        help="Stage 2 training method. 'none' = zero-shot evaluation only.")
    parser.add_argument("--output_dir", required=True,
        help="Root output directory for this experiment.")
    parser.add_argument("--do_train", action="store_true",
        help="Run training (Stage 1 and/or Stage 2 as applicable).")
    parser.add_argument("--do_eval", action="store_true",
        help="Run evaluation on MedAraBench test set.")
    parser.add_argument("--stage1_checkpoint", default=None,
        help="Path to an existing Stage 1 checkpoint (skips Stage 1 training).")
    parser.add_argument("--data_dir", default="Files/datasets/",
        help="Root directory of datasets.")
    parser.add_argument("--eval_batch_size", type=int, default=16,
        help="Batch size for evaluation inference.")

    # W&B
    parser.add_argument("--wandb_project", default=None,
        help="W&B project name. Overrides WANDB_PROJECT env var. "
             "Pass --no_wandb to disable W&B entirely.")
    parser.add_argument("--no_wandb", action="store_true",
        help="Disable W&B logging.")

    # HuggingFace Hub
    parser.add_argument("--hf_private", action="store_true",
        help="Make HF Hub repos private. "
             "Upload is automatic when HF_TOKEN is set in .env or environment.")

    # Dry-run / local testing
    parser.add_argument("--max_train_samples", type=int, default=None,
        help="Truncate training data to this many samples. "
             "Use for quick smoke-tests on CPU/MPS (e.g. --max_train_samples 100).")
    parser.add_argument("--max_eval_samples", type=int, default=None,
        help="Truncate validation/test data to this many samples.")
    parser.add_argument("--dry_run", action="store_true",
        help="Shorthand for a 1-epoch, fp32, 100-sample smoke test on CPU/MPS. "
             "Automatically sets --max_train_samples 100 --max_eval_samples 50 "
             "--no_wandb and disables bf16/HF upload.")

    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    # --dry_run applies safe defaults for local CPU/MPS smoke-testing
    if args.dry_run:
        args.max_train_samples = args.max_train_samples or 100
        args.max_eval_samples = args.max_eval_samples or 50
        args.no_wandb = True
        print("Dry-run mode: max_train_samples=100, max_eval_samples=50, W&B disabled, bf16 disabled, 1 epoch")

    load_in_4bit = _is_70b(args.model) and not args.dry_run

    # ------------------------------------------------------------------
    # Resolve credentials from env (already loaded from .env)
    # ------------------------------------------------------------------
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "arabic-medical-llm")
    hf_token = os.environ.get("HF_TOKEN")  # resolved from .env; repo owner derived automatically

    # ------------------------------------------------------------------
    # W&B run initialisation
    # ------------------------------------------------------------------
    from utils import wandb_logger

    if not args.no_wandb:
        import yaml

        # Build a flat config dict from all relevant settings
        run_config = {
            "model": args.model,
            "stage1_method": args.stage1_method,
            "stage2_method": args.stage2_method,
            "output_dir": args.output_dir,
            "load_in_4bit": load_in_4bit,
        }
        # Merge hyperparameters from both configs
        for method in {args.stage1_method, args.stage2_method} - {"none"}:
            cfg_path = _config_path_for(method, script_dir)
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            prefix = f"s1_" if method == args.stage1_method else f"s2_"
            for k, v in cfg.get("training", {}).items():
                run_config[f"{prefix}{k}"] = v
            if "lora" in cfg:
                for k, v in cfg["lora"].items():
                    run_config[f"lora_{k}"] = v

        run_name = _make_run_name(args.model, args.stage1_method, args.stage2_method)
        os.environ["WANDB_NAME"] = run_name
        tags = [
            args.model.split("/")[-1],
            f"s1_{args.stage1_method}",
            f"s2_{args.stage2_method}",
            "70b" if load_in_4bit else "8b",
        ]

        wandb_logger.init_run(
            project=wandb_project,
            run_name=run_name,
            config=run_config,
            tags=tags,
            notes=f"Experiment: {args.model} | Stage1={args.stage1_method} | Stage2={args.stage2_method}",
        )

        # Log dataset statistics once per run
        from data.read_data import load_aramed, load_medarabench
        from data.clean_data import clean_medarabench

        print("Computing dataset statistics for W&B...")
        aramed_train = load_aramed(split="train", data_dir=args.data_dir)
        aramed_test = load_aramed(split="test", data_dir=args.data_dir)
        med_raw = load_medarabench(split="train", data_dir=args.data_dir)
        med_clean = clean_medarabench(med_raw)
        med_test = load_medarabench(split="test", data_dir=args.data_dir)

        wandb_logger.log_dataset_stats(
            aramed_train_size=len(aramed_train),
            aramed_test_size=len(aramed_test),
            medarabench_raw_size=len(med_raw),
            medarabench_clean_size=len(med_clean),
            medarabench_test_size=len(med_test),
            cleaning_breakdown={
                "invalid_answers": len(med_raw) - len(med_clean),
            },
        )
        del aramed_train, aramed_test, med_raw, med_clean, med_test
        gc.collect()

    # ------------------------------------------------------------------
    # Stage dirs
    # ------------------------------------------------------------------
    stage1_dir = os.path.join(args.output_dir, "stage1")
    stage2_dir = os.path.join(args.output_dir, "stage2")
    eval_dir = os.path.join(args.output_dir, "eval")

    stage1_checkpoint = args.stage1_checkpoint
    stage2_checkpoint = None

    # ------------------------------------------------------------------
    # Stage 1: Domain Adaptation
    # ------------------------------------------------------------------
    if args.stage1_method != "none" and args.do_train and stage1_checkpoint is None:
        from train.adaptation import run_domain_adaptation

        s1_config = _config_path_for(args.stage1_method, script_dir)
        os.makedirs(stage1_dir, exist_ok=True)

        stage1_checkpoint, _ = run_domain_adaptation(
            model_name=args.model,
            method=args.stage1_method,
            config_path=s1_config,
            output_dir=stage1_dir,
            data_dir=args.data_dir,
            load_in_4bit=load_in_4bit,
            hf_token=hf_token,
            hf_private=args.hf_private,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            dry_run=args.dry_run,
        )

        gc.collect()
        torch.cuda.empty_cache()

    elif args.stage1_method != "none" and stage1_checkpoint is not None:
        print(f"Using existing Stage 1 checkpoint: {stage1_checkpoint}")

    # ------------------------------------------------------------------
    # Stage 2: Task-Specific Fine-tuning
    # ------------------------------------------------------------------
    if args.stage2_method != "none" and args.do_train:
        from train.finetuning import run_task_finetuning

        s2_config = _config_path_for(args.stage2_method, script_dir)
        os.makedirs(stage2_dir, exist_ok=True)

        stage2_checkpoint, _ = run_task_finetuning(
            model_name=args.model,
            method=args.stage2_method,
            config_path=s2_config,
            output_dir=stage2_dir,
            stage1_checkpoint=stage1_checkpoint,
            stage1_method=args.stage1_method if args.stage1_method != "none" else None,
            data_dir=args.data_dir,
            load_in_4bit=load_in_4bit,
            hf_token=hf_token,
            hf_private=args.hf_private,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            dry_run=args.dry_run,
        )

        gc.collect()
        torch.cuda.empty_cache()

    elif args.stage2_method != "none" and not args.do_train:
        if os.path.isdir(stage2_dir):
            stage2_checkpoint = stage2_dir
        elif stage1_checkpoint:
            stage2_checkpoint = stage1_checkpoint

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if args.do_eval:
        from evaluation.evaluate import run_evaluation
        from utils.get_model import load_model_and_tokenizer, load_from_checkpoint, merge_lora_and_save

        os.makedirs(eval_dir, exist_ok=True)

        if args.stage2_method == "none" and args.stage1_method == "none":
            print("Zero-shot evaluation — loading base model...")
            model, tokenizer = load_model_and_tokenizer(
                model_name=args.model,
                method="full",
                load_in_4bit=load_in_4bit,
            )
            for p in model.parameters():
                p.requires_grad = False

        elif args.stage2_method == "none" and args.stage1_method != "none":
            ckpt = stage1_checkpoint or stage1_dir
            print(f"Evaluating Stage 1 checkpoint: {ckpt}")
            model, tokenizer = load_from_checkpoint(
                checkpoint_path=ckpt,
                base_model_name=args.model,
                method=args.stage1_method,
                load_in_4bit=load_in_4bit,
            )
            if args.stage1_method == "lora":
                merged_dir = os.path.join(eval_dir, "_merged")
                model = merge_lora_and_save(model, merged_dir, tokenizer)

        else:
            ckpt = stage2_checkpoint or stage2_dir
            print(f"Evaluating Stage 2 checkpoint: {ckpt}")
            eval_method = args.stage2_method
            model, tokenizer = load_from_checkpoint(
                checkpoint_path=ckpt,
                base_model_name=args.model,
                method=eval_method,
                load_in_4bit=load_in_4bit,
            )
            if eval_method == "lora":
                merged_dir = os.path.join(eval_dir, "_merged")
                model = merge_lora_and_save(model, merged_dir, tokenizer)

        results = run_evaluation(
            model=model,
            tokenizer=tokenizer,
            output_dir=eval_dir,
            batch_size=args.eval_batch_size,
            data_dir=args.data_dir,
            max_samples=args.max_eval_samples,
        )

        print(f"\nExperiment complete.")
        print(f"  Accuracy : {results['accuracy']:.4f}")
        print(f"  Macro F1 : {results['macro_f1']:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Finish W&B run
    # ------------------------------------------------------------------
    if not args.no_wandb:
        wandb_logger.finish_run()

    print("\nAll done.")


if __name__ == "__main__":
    main()
