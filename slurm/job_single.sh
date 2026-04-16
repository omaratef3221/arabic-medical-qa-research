#!/bin/bash
# =============================================================================
# SLURM job script — single experiment
#
# Usage:
#   sbatch slurm/job_single.sh 1          # run experiment 1
#   sbatch slurm/job_single.sh 1 2 3      # run experiments 1, 2, 3 sequentially
#   sbatch slurm/job_single.sh            # run all 18 experiments sequentially
#
# Adjust the #SBATCH directives below to match your cluster's partition names,
# GPU type, and time limits before submitting.
# =============================================================================

#SBATCH --job-name=arabic-medical-llm
#SBATCH --output=slurm/logs/%x_%j.out        # stdout  → slurm/logs/arabic-medical-llm_<jobid>.out
#SBATCH --error=slurm/logs/%x_%j.err         # stderr  → slurm/logs/arabic-medical-llm_<jobid>.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                    # data loading workers
#SBATCH --gres=gpu:a10g:4                    # 4× A10G — gpu-g5-spt node
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-g5-spt

# ---- (optional) email notifications ----
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.com

# =============================================================================
# Environment setup
# =============================================================================

# Load required modules (uncomment / adjust for your cluster's module system)
# module purge
# module load cuda/12.1
# module load python/3.10

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medical_llm          # change to your env name if different

# Move to project script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo "Experiments : ${@:-all}"
echo "============================================"

# =============================================================================
# Run experiments
# =============================================================================

bash run.sh "$@"

echo "============================================"
echo "Finished : $(date)"
echo "============================================"
