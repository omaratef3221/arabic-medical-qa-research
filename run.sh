#!/bin/bash
# =============================================================================
# Experiment Runner — Arabic Medical LLM Fine-tuning
# LoRA vs. Full Fine-Tuning for Arabic Medical Question Answering
# =============================================================================
#
# Usage:
#   bash run.sh              # run all 18 experiments sequentially
#   bash run.sh 1            # run only experiment 1
#   bash run.sh 1 5 13       # run experiments 1, 5, and 13
#
# Prerequisites:
#   pip install torch transformers peft trl bitsandbytes datasets \
#               accelerate scikit-learn pandas pyyaml
#
# Run from the script/ directory:
#   cd path/to/script && bash run.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which experiments to run (default: all)
EXPS_TO_RUN=("$@")

should_run() {
    local exp_num="$1"
    if [ "${#EXPS_TO_RUN[@]}" -eq 0 ]; then
        return 0  # run all
    fi
    for e in "${EXPS_TO_RUN[@]}"; do
        if [ "$e" = "$exp_num" ]; then
            return 0
        fi
    done
    return 1
}

log() {
    echo ""
    echo "============================================"
    echo " $1"
    echo "============================================"
}

# =============================================================================
# PRIMARY EXPERIMENTS (1-8): Two-stage pipeline
# =============================================================================

if should_run 1; then
    log "Exp 1: Llama-3.1-8B  |  LoRA -> LoRA"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method lora \
        --stage2_method lora \
        --output_dir outputs/exp01_llama_lora_lora \
        --do_train --do_eval
fi

if should_run 2; then
    log "Exp 2: Llama-3.1-8B  |  Full -> Full"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method full \
        --stage2_method full \
        --output_dir outputs/exp02_llama_full_full \
        --do_train --do_eval
fi

if should_run 3; then
    log "Exp 3: Llama-3.1-8B  |  LoRA -> Full"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method lora \
        --stage2_method full \
        --output_dir outputs/exp03_llama_lora_full \
        --do_train --do_eval
fi

if should_run 4; then
    log "Exp 4: Llama-3.1-8B  |  Full -> LoRA"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method full \
        --stage2_method lora \
        --output_dir outputs/exp04_llama_full_lora \
        --do_train --do_eval
fi

if should_run 5; then
    log "Exp 5: Jais-2-8B-Chat  |  LoRA -> LoRA"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method lora \
        --stage2_method lora \
        --output_dir outputs/exp05_jais_lora_lora \
        --do_train --do_eval
fi

if should_run 6; then
    log "Exp 6: Jais-2-8B-Chat  |  Full -> Full"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method full \
        --stage2_method full \
        --output_dir outputs/exp06_jais_full_full \
        --do_train --do_eval
fi

if should_run 7; then
    log "Exp 7: Jais-2-8B-Chat  |  LoRA -> Full"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method lora \
        --stage2_method full \
        --output_dir outputs/exp07_jais_lora_full \
        --do_train --do_eval
fi

if should_run 8; then
    log "Exp 8: Jais-2-8B-Chat  |  Full -> LoRA"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method full \
        --stage2_method lora \
        --output_dir outputs/exp08_jais_full_lora \
        --do_train --do_eval
fi

# =============================================================================
# BASELINE EXPERIMENTS (9-12): No Stage 1 domain adaptation
# =============================================================================

if should_run 9; then
    log "Exp 9: Llama-3.1-8B  |  None -> LoRA"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method none \
        --stage2_method lora \
        --output_dir outputs/exp09_llama_none_lora \
        --do_train --do_eval
fi

if should_run 10; then
    log "Exp 10: Llama-3.1-8B  |  None -> Full"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method none \
        --stage2_method full \
        --output_dir outputs/exp10_llama_none_full \
        --do_train --do_eval
fi

if should_run 11; then
    log "Exp 11: Jais-2-8B-Chat  |  None -> LoRA"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method none \
        --stage2_method lora \
        --output_dir outputs/exp11_jais_none_lora \
        --do_train --do_eval
fi

if should_run 12; then
    log "Exp 12: Jais-2-8B-Chat  |  None -> Full"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method none \
        --stage2_method full \
        --output_dir outputs/exp12_jais_none_full \
        --do_train --do_eval
fi

# =============================================================================
# ZERO-SHOT EXPERIMENTS (13-14): No fine-tuning at all
# =============================================================================

if should_run 13; then
    log "Exp 13: Llama-3.1-8B  |  Zero-shot"
    python main.py \
        --model meta-llama/Llama-3.1-8B \
        --stage1_method none \
        --stage2_method none \
        --output_dir outputs/exp13_llama_zeroshot \
        --do_eval
fi

if should_run 14; then
    log "Exp 14: Jais-2-8B-Chat  |  Zero-shot"
    python main.py \
        --model inceptionai/Jais-2-8B-Chat \
        --stage1_method none \
        --stage2_method none \
        --output_dir outputs/exp14_jais_zeroshot \
        --do_eval
fi

# =============================================================================
# SCALE ABLATION EXPERIMENTS (15-18): 70B models, LoRA only (QLoRA)
# =============================================================================

if should_run 15; then
    log "Exp 15: Llama-3.1-70B  |  LoRA -> LoRA  (QLoRA)"
    python main.py \
        --model meta-llama/Llama-3.1-70B \
        --stage1_method lora \
        --stage2_method lora \
        --output_dir outputs/exp15_llama70b_lora_lora \
        --do_train --do_eval
fi

if should_run 16; then
    log "Exp 16: Llama-3.1-70B  |  None -> LoRA  (QLoRA)"
    python main.py \
        --model meta-llama/Llama-3.1-70B \
        --stage1_method none \
        --stage2_method lora \
        --output_dir outputs/exp16_llama70b_none_lora \
        --do_train --do_eval
fi

if should_run 17; then
    log "Exp 17: Jais-2-70B-Chat  |  LoRA -> LoRA  (QLoRA)"
    python main.py \
        --model inceptionai/Jais-2-70B-Chat \
        --stage1_method lora \
        --stage2_method lora \
        --output_dir outputs/exp17_jais70b_lora_lora \
        --do_train --do_eval
fi

if should_run 18; then
    log "Exp 18: Jais-2-70B-Chat  |  None -> LoRA  (QLoRA)"
    python main.py \
        --model inceptionai/Jais-2-70B-Chat \
        --stage1_method none \
        --stage2_method lora \
        --output_dir outputs/exp18_jais70b_none_lora \
        --do_train --do_eval
fi

echo ""
echo "All selected experiments completed."
