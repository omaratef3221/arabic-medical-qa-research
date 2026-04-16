#!/bin/bash
# =============================================================================
# SLURM array job — runs each experiment as an independent parallel job
#
# Usage:
#   # 8B experiments (1-14) — 1 GPU each on uos-hpc-queue-1:
#   sbatch --array=1-14 slurm/job_array.sh
#
#   # 70B experiments (15-18) — 4 GPUs each on gpu-g5-spt:
#   sbatch --array=15-18 --partition=gpu-g5-spt --gres=gpu:a10g:4 --mem=180G slurm/job_array.sh
#   # Alternative 4-GPU partitions: dcv-4gpu-g5-ond
#
#   # Specific experiments:
#   sbatch --array=1,5,9,13 slurm/job_array.sh
#
#   # Limit concurrency (e.g. max 4 running at once):
#   sbatch --array=1-14%4 slurm/job_array.sh
#
# Each array task maps SLURM_ARRAY_TASK_ID → experiment number.
# =============================================================================

#SBATCH --job-name=arabic-med-%a
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=1-14                         # default: 8B experiments only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1                    # 1× A10G per 8B task (override to 4 for 70B)
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --partition=dcv-1gpu-g5-ond          # default partition for 8B jobs

# ---- (optional) email notifications ----
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.com

# =============================================================================
# Environment setup
# =============================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medical_llm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

EXP_ID=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "Array job   : $SLURM_ARRAY_JOB_ID"
echo "Task (Exp)  : $EXP_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo "============================================"

# =============================================================================
# Experiment definitions — maps task ID → main.py arguments
# =============================================================================

case $EXP_ID in
  # --- Primary: two-stage pipeline ---
  1)  ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method lora  --stage2_method lora  --output_dir outputs/exp01_llama_lora_lora  --do_train --do_eval" ;;
  2)  ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method full  --stage2_method full  --output_dir outputs/exp02_llama_full_full  --do_train --do_eval" ;;
  3)  ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method lora  --stage2_method full  --output_dir outputs/exp03_llama_lora_full  --do_train --do_eval" ;;
  4)  ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method full  --stage2_method lora  --output_dir outputs/exp04_llama_full_lora  --do_train --do_eval" ;;
  5)  ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method lora  --stage2_method lora  --output_dir outputs/exp05_jais_lora_lora  --do_train --do_eval" ;;
  6)  ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method full  --stage2_method full  --output_dir outputs/exp06_jais_full_full  --do_train --do_eval" ;;
  7)  ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method lora  --stage2_method full  --output_dir outputs/exp07_jais_lora_full  --do_train --do_eval" ;;
  8)  ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method full  --stage2_method lora  --output_dir outputs/exp08_jais_full_lora  --do_train --do_eval" ;;

  # --- Baselines: no Stage 1 ---
  9)  ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method none  --stage2_method lora  --output_dir outputs/exp09_llama_none_lora  --do_train --do_eval" ;;
  10) ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method none  --stage2_method full  --output_dir outputs/exp10_llama_none_full  --do_train --do_eval" ;;
  11) ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method none  --stage2_method lora  --output_dir outputs/exp11_jais_none_lora  --do_train --do_eval" ;;
  12) ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method none  --stage2_method full  --output_dir outputs/exp12_jais_none_full  --do_train --do_eval" ;;

  # --- Zero-shot ---
  13) ARGS="--model meta-llama/Llama-3.1-8B  --stage1_method none  --stage2_method none  --output_dir outputs/exp13_llama_zeroshot  --do_eval" ;;
  14) ARGS="--model inceptionai/Jais-2-8B-Chat  --stage1_method none  --stage2_method none  --output_dir outputs/exp14_jais_zeroshot  --do_eval" ;;

  # --- Scale ablation: 70B QLoRA ---
  15) ARGS="--model meta-llama/Llama-3.1-70B  --stage1_method lora  --stage2_method lora  --output_dir outputs/exp15_llama70b_lora_lora  --do_train --do_eval" ;;
  16) ARGS="--model meta-llama/Llama-3.1-70B  --stage1_method none  --stage2_method lora  --output_dir outputs/exp16_llama70b_none_lora  --do_train --do_eval" ;;
  17) ARGS="--model inceptionai/Jais-2-70B-Chat  --stage1_method lora  --stage2_method lora  --output_dir outputs/exp17_jais70b_lora_lora  --do_train --do_eval" ;;
  18) ARGS="--model inceptionai/Jais-2-70B-Chat  --stage1_method none  --stage2_method lora  --output_dir outputs/exp18_jais70b_none_lora  --do_train --do_eval" ;;

  *) echo "Unknown experiment ID: $EXP_ID"; exit 1 ;;
esac

echo "Running: python main.py $ARGS"
python main.py $ARGS

echo "============================================"
echo "Finished : $(date)"
echo "============================================"
