#!/bin/bash
# =============================================================================
# SLURM job script — Revision R1 orchestrator (4x A10G, g5.12xlarge)
#
# Runs one phase of scripts/run_revision.py on a 4-GPU node. The orchestrator
# skips every configuration that already has a finished W&B run or a local
# eval/results.json, so this job can be resubmitted any number of times
# (walltime kill, node failure, ...) and it continues where it left off.
#
# Usage (from the head node, inside the repo's script/ directory):
#   sbatch slurm/revision_r1.sh stats        # Task 1: predictions + stats (no training)
#   sbatch slurm/revision_r1.sh seeds        # Task 2
#   sbatch slurm/revision_r1.sh sweep        # Task 3
#   sbatch slurm/revision_r1.sh newmodels    # Task 4
#   sbatch slurm/revision_r1.sh stage1diag   # Task 5
#   sbatch slurm/revision_r1.sh all          # everything, in order
# =============================================================================

#SBATCH --job-name=revision_r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --partition=dcv-4gpu-g5-ond
##SBATCH --qos=qos_dcv_4gpu_g5_ond
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# =============================================================================
# Environment setup
# =============================================================================

module load anaconda3
eval "$(conda shell.bash hook)"
source activate medical_llm

cd /home/oelgendy/arabic-medical-llm/script
mkdir -p slurm/logs

PHASE="${1:-all}"

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Phase       : $PHASE"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo "============================================"

nvidia-smi

# Show the plan in the log before executing (also proves W&B connectivity)
python scripts/run_revision.py --phase "$PHASE" --dry-run

python scripts/run_revision.py --phase "$PHASE"

echo ""
echo "============================================"
echo "Phase '$PHASE' finished at: $(date)"
echo "Resubmit this script to retry any failed runs; completed runs are skipped."
echo "============================================"
