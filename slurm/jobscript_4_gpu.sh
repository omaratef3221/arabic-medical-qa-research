#!/bin/bash
# =============================================================================
# SLURM job script — 4-GPU partition
#
# Runs:
#   - Exp 4 & 8: Full Stage 1 + LoRA Stage 2 (full train, both stages)
#   - Exps 1, 2, 3, 5, 6, 7, 9–14: Retrain Stage 2 only, reusing existing
#     Stage 1 checkpoints from HuggingFace Hub
#
# All experiments use the cleaned MedAraBench data (cleaning is applied in
# both train/finetuning.py and evaluation/evaluate.py).
# =============================================================================

#SBATCH --job-name=arabic_med_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --partition=dcv-4gpu-g5-ond
##SBATCH --qos=qos_dcv_4gpu_g5_ond
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# =============================================================================
# Environment setup
# =============================================================================

module load anaconda3
eval "$(conda shell.bash hook)"
source activate medical_llm

cd /home/oelgendy/arabic-medical-llm/script

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo "============================================"

nvidia-smi

# =============================================================================
# HuggingFace owner (resolved dynamically via HF_TOKEN)
# Stage 1 checkpoints uploaded by previous runs live under this namespace.
# =============================================================================
HF_OWNER="omaratef3221"

# =============================================================================
# Helper: run a full experiment (Stage 1 + Stage 2 + Eval)
# =============================================================================
run_full () {
  local NAME=$1
  local MODEL=$2
  local S1=$3
  local S2=$4
  local OUT=$5

  echo ""
  echo "============================================"
  echo "FULL TRAIN: $NAME ($MODEL, s1=$S1, s2=$S2)"
  echo "Started: $(date)"
  echo "============================================"

  python main.py \
    --model "$MODEL" \
    --stage1_method "$S1" \
    --stage2_method "$S2" \
    --output_dir "$OUT" \
    --do_train --do_eval

  echo "$NAME finished at: $(date)"
}

# =============================================================================
# Helper: retrain Stage 2 only, using an existing Stage 1 checkpoint from HF
# =============================================================================
run_stage2 () {
  local NAME=$1
  local MODEL=$2
  local S1=$3
  local S2=$4
  local OUT=$5
  local S1_CKPT=$6   # HF repo name or local path; empty for "no Stage 1"

  echo ""
  echo "============================================"
  echo "STAGE-2 RETRAIN: $NAME ($MODEL, s1=$S1, s2=$S2)"
  echo "Started: $(date)"
  echo "S1 checkpoint: ${S1_CKPT:-<none>}"
  echo "============================================"

  if [ -z "$S1_CKPT" ]; then
    # No Stage 1 (baseline experiments)
    python main.py \
      --model "$MODEL" \
      --stage1_method "$S1" \
      --stage2_method "$S2" \
      --output_dir "$OUT" \
      --do_train --do_eval
  else
    python main.py \
      --model "$MODEL" \
      --stage1_method "$S1" \
      --stage2_method "$S2" \
      --stage1_checkpoint "$S1_CKPT" \
      --output_dir "$OUT" \
      --do_train --do_eval
  fi

  echo "$NAME finished at: $(date)"
}

# =============================================================================
# Helper: re-evaluate zero-shot (no training, just eval with cleaned test set)
# =============================================================================
run_zeroshot_eval () {
  local NAME=$1
  local MODEL=$2
  local OUT=$3

  echo ""
  echo "============================================"
  echo "ZERO-SHOT EVAL: $NAME ($MODEL)"
  echo "Started: $(date)"
  echo "============================================"

  python main.py \
    --model "$MODEL" \
    --stage1_method none \
    --stage2_method none \
    --output_dir "$OUT" \
    --do_eval

  echo "$NAME finished at: $(date)"
}

# =============================================================================
# Exp 4 & 8 — FULL training (both Stage 1 and Stage 2)
# =============================================================================
run_full "Exp04" "meta-llama/Llama-3.1-8B"     "full" "lora" "outputs/exp04_llama_full_lora"
run_full "Exp08" "inceptionai/Jais-2-8B-Chat"  "full" "lora" "outputs/exp08_jais_full_lora"

# =============================================================================
# Exps 1, 2, 3, 5, 6, 7 — Stage 2 retrain using existing Stage 1 on HF
# Repo naming convention from utils/hf_hub.py:
#   {owner}/{model-short}-s1-{s1_method}-medarabench  (after Stage 1 upload)
# =============================================================================
run_stage2 "Exp01" "meta-llama/Llama-3.1-8B"    "lora" "lora" "outputs/exp01_llama_lora_lora" \
           "${HF_OWNER}/llama-3.1-8b-s1-lora-aramed"

run_stage2 "Exp02" "meta-llama/Llama-3.1-8B"    "full" "full" "outputs/exp02_llama_full_full" \
           "${HF_OWNER}/llama-3.1-8b-s1-full-aramed"

run_stage2 "Exp03" "meta-llama/Llama-3.1-8B"    "lora" "full" "outputs/exp03_llama_lora_full" \
           "${HF_OWNER}/llama-3.1-8b-s1-lora-aramed"

run_stage2 "Exp05" "inceptionai/Jais-2-8B-Chat" "lora" "lora" "outputs/exp05_jais_lora_lora" \
           "${HF_OWNER}/jais-2-8b-chat-s1-lora-aramed"

run_stage2 "Exp06" "inceptionai/Jais-2-8B-Chat" "full" "full" "outputs/exp06_jais_full_full" \
           "${HF_OWNER}/jais-2-8b-chat-s1-full-aramed"

run_stage2 "Exp07" "inceptionai/Jais-2-8B-Chat" "lora" "full" "outputs/exp07_jais_lora_full" \
           "${HF_OWNER}/jais-2-8b-chat-s1-lora-aramed"

# =============================================================================
# Exps 9–12 — No Stage 1, Stage 2 only (baseline experiments)
# =============================================================================
run_stage2 "Exp09" "meta-llama/Llama-3.1-8B"    "none" "lora" "outputs/exp09_llama_none_lora" ""
run_stage2 "Exp10" "meta-llama/Llama-3.1-8B"    "none" "full" "outputs/exp10_llama_none_full" ""
run_stage2 "Exp11" "inceptionai/Jais-2-8B-Chat" "none" "lora" "outputs/exp11_jais_none_lora" ""
run_stage2 "Exp12" "inceptionai/Jais-2-8B-Chat" "none" "full" "outputs/exp12_jais_none_full" ""

# =============================================================================
# Exps 13 & 14 — Zero-shot re-evaluation only (no training needed)
# =============================================================================
run_zeroshot_eval "Exp13" "meta-llama/Llama-3.1-8B"    "outputs/exp13_llama_zeroshot"
run_zeroshot_eval "Exp14" "inceptionai/Jais-2-8B-Chat" "outputs/exp14_jais_zeroshot"

echo ""
echo "============================================"
echo "All experiments finished at: $(date)"
echo "============================================"
