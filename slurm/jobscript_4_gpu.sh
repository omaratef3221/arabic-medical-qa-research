#!/bin/bash
# =============================================================================
# SLURM job script — 4-GPU partition
#
# Runs ONLY two experiments:
#   - Exp 04: Llama-3.1-8B   Full Stage 1 → LoRA Stage 2  (full train both stages)
#   - Exp 08: Jais-2-8B-Chat Full Stage 1 → LoRA Stage 2  (Stage 1 reused from HF)
#
# Exp 08's Stage 1 (Jais full) was successfully uploaded to HF in a previous
# run (Omaratef3221/jais-2-8b-chat-s1-full-aramed) so we skip Stage 1 training
# and only retrain Stage 2 — saves ~7 hours.
#
# All experiments use the cleaned MedAraBench data (cleaning is applied in
# both train/finetuning.py and evaluation/evaluate.py).
#
# IMPORTANT (2026-04-23): The Stage 2 loading logic in train/finetuning.py was
# fixed — previously the LoRA S1 adapter was loaded frozen and no fresh S2
# adapter was applied, causing grad_norm=0 and no learning. The new logic
# merges the S1 adapter into the base weights, then applies a fresh trainable
# S2 adapter. A sanity check verifies trainable params are non-zero before
# training starts.
#
# Also fixed (2026-04-24): HF Hub upload now overwrites PEFT's auto-generated
# README (which had a local path in `base_model:`) and is wrapped in try/except
# so an upload failure no longer blocks evaluation.
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

HF_OWNER="omaratef3221"

# =============================================================================
# Exp 04: Llama-3.1-8B  Full Stage 1 → LoRA Stage 2  (full train, both stages)
# =============================================================================
echo ""
echo "============================================"
echo "FULL TRAIN: Exp04 (Llama-3.1-8B, s1=full, s2=lora)"
echo "Started: $(date)"
echo "============================================"

python main.py \
    --model meta-llama/Llama-3.1-8B \
    --stage1_method full \
    --stage2_method lora \
    --output_dir outputs/exp04_llama_full_lora \
    --do_train --do_eval

echo "Exp04 finished at: $(date)"

# =============================================================================
# Exp 08: Jais-2-8B-Chat  Full Stage 1 → LoRA Stage 2
# Stage 1 reused from HF (uploaded successfully in previous run).
# =============================================================================
echo ""
echo "============================================"
echo "STAGE-2 RETRAIN: Exp08 (Jais-2-8B-Chat, s1=full, s2=lora)"
echo "Started: $(date)"
echo "S1 checkpoint: ${HF_OWNER}/jais-2-8b-chat-s1-full-aramed"
echo "============================================"

python main.py \
    --model inceptionai/Jais-2-8B-Chat \
    --stage1_method full \
    --stage2_method lora \
    --stage1_checkpoint "${HF_OWNER}/jais-2-8b-chat-s1-full-aramed" \
    --output_dir outputs/exp08_jais_full_lora \
    --do_train --do_eval

echo "Exp08 finished at: $(date)"

echo ""
echo "============================================"
echo "All experiments finished at: $(date)"
echo "============================================"
