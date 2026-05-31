#!/bin/bash
# ==============================================================================
#                 Enhanced_RRSIS_UOT v3 — GPU Training Script
# ==============================================================================
# Architecture: SAM3 + Dynamic LoRA + Multi-Scale OT + Differentiable GPG
#
# Core Enhancements:
#   1. Text-Guided Dynamic LoRA (text-conditioned vision adapters)
#   2. Contrastive Language-Image Loss (InfoNCE V-L alignment)
#   3. Multi-Scale OT Feature Alignment (differentiable Sinkhorn)
#   4. OHEM + Focal Dice + Boundary Loss (hard pixel mining)
#   5. Differentiable GPG (spatial soft-argmax, end-to-end gradient flow)
#
# v3 Fix: Removed gradient blockers (@torch.no_grad, topk, floor_div)
#         from GPG and Sinkhorn modules. Segmentation loss now optimizes
#         prompt placement end-to-end.
#
# LR Schedule: Linear Warmup (3 epochs) → Cosine Decay with Floor (eta_min=3e-6)
# Early Stopping: patience=8 (auto-stop if no improvement for 8 epochs)
#
# Target Performance: oIoU ≥ 82%, mIoU ≥ 72% on RRSIS-D validation
#
# Usage:
#   bash fine.sh [dataset_name] [data_root] [sam3_ckpt]
#
# Examples:
#   bash fine.sh rrsis_d /path/to/data ./pre-trained-weights/sam3.pt
#   bash fine.sh rrsis_hr /path/to/data ./pre-trained-weights/sam3.pt
# ==============================================================================

# Default parameters
DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
SAM3_CKPT=${3:-./pre-trained-weights/sam3.pt}
OUTPUT_DIR="./output/${DATASET}_enhanced_v3"

# Highlight setup information
echo "=============================================================================="
echo "🚀 Enhanced_RRSIS_UOT v3 — Differentiable GPG Training"
echo "📊 Dataset:     ${DATASET}"
echo "📂 Data Root:   ${DATA_ROOT}"
echo "💾 SAM3 Ckpt:   ${SAM3_CKPT}"
echo "📁 Output Dir:  ${OUTPUT_DIR}"
echo "🔧 Enhancements: Dynamic LoRA | Contrastive | Multi-Scale OT | OHEM | Diff-GPG"
echo "📈 Schedule:    3-epoch warmup → Cosine decay (floor=3e-6, never zero)"
echo "🛑 Early Stop:  patience=8"
echo "🆕 v3 Fix:      End-to-end differentiable prompt generation"
echo "=============================================================================="

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Ensure pre-trained weight exists or check parent dir fallback
if [ ! -f "${SAM3_CKPT}" ]; then
    echo "⚠️  [WARNING] Pre-trained weight not found at '${SAM3_CKPT}'"
    FALLBACK="../sam3_pre_trained_weights/sam3.pt"
    if [ -f "${FALLBACK}" ]; then
        echo "🔍 [INFO] Found fallback checkpoint: '${FALLBACK}'"
        SAM3_CKPT=${FALLBACK}
    else
        echo "❌ [ERROR] Could not find pre-trained weights."
    fi
fi

# Run training with optimal hyperparameters for v3
python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ${SAM3_CKPT} \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 50 \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --weight_decay 0.01 \
    --weight_decay_decoder 0.005 \
    --warmup_epochs 3 \
    --eta_min 3e-6 \
    --patience 8 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    --contrastive_weight 0.1 \
    --ohem_hard_ratio 0.3 \
    --ot_reg 0.1 \
    --ot_num_iter 10 \
    --num_ot_scales 3 \
    --focal_gamma 2.0

echo "🎉 Training complete! Output saved to ${OUTPUT_DIR}"
