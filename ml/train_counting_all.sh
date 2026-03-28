#!/bin/bash
# =============================================================================
# Train All Models for Counting Task
# =============================================================================
# This script trains all model architectures on the counting task
# across all difficulty datasets.
#
# Models:
#   - JEPA (Multi-modal with visual teacher)
#   - JEPA Acoustic-Only
#   - LeWM (Acoustic-only)
#   - JEPA+SigReg (Multi-modal with Gaussian regularization)
#
# Usage:
#   ./train_counting_all.sh [epochs] [patience]
#
# Arguments:
#   epochs   - Number of training epochs (default: 100)
#   patience - Early stopping patience (default: 15)
# =============================================================================

set -e  # Exit on error

# Configuration
EPOCHS=${1:-100}
PATIENCE=${2:-15}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================${NC}\n"
}

print_section() {
    echo -e "\n${YELLOW}--- $1 ---${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python is available
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3.9 or higher."
    exit 1
fi

cd "${SCRIPT_DIR}"

print_header "🔢 Training All Models for COUNTING Task"
echo "Configuration:"
echo "  Task:     counting"
echo "  Epochs:   ${EPOCHS}"
echo "  Patience: ${PATIENCE}"
echo "  Weights:  ${WEIGHTS_DIR}"
echo ""

# Create weights directory structure
print_section "Setting Up Directory Structure"
for model in jepa jepa_acoustic lewm jepa_sigreg; do
    for dataset in easy medium hard extreme; do
        mkdir -p "${WEIGHTS_DIR}/${model}_${dataset}"
    done
done
print_success "Created all weight directories"

# Track training results (use simple arrays for macOS bash 3.x compatibility)
TRAINING_RESULTS=()

# Uppercase function for bash 3.x compatibility
to_upper() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

# Training function
train_model() {
    local model_type=$1
    local dataset=$2
    local weights_dir="${WEIGHTS_DIR}/${model_type}_${dataset}"
    
    print_section "Training $(to_upper ${model_type}) on $(to_upper ${dataset}) Dataset (Counting Task)"
    echo "Weights will be saved to: ${weights_dir}"
    
    case $model_type in
        "jepa")
            python3 train.py jepa --dataset ${dataset} --epochs ${EPOCHS} \
                --patience ${PATIENCE} --weights-dir "${weights_dir}" \
                --task counting --with-aug
            ;;
        "jepa_acoustic")
            # Train standard JEPA, will evaluate acoustic-only later
            python3 train.py jepa --dataset ${dataset} --epochs ${EPOCHS} \
                --patience ${PATIENCE} --weights-dir "${weights_dir}" \
                --task counting --with-aug
            ;;
        "lewm")
            python3 train.py lewm --dataset ${dataset} --epochs ${EPOCHS} \
                --patience ${PATIENCE} --weights-dir "${weights_dir}" \
                --task counting
            ;;
        "jepa_sigreg")
            python3 train_jepa_sigreg.py --dataset ${dataset} --epochs ${EPOCHS} \
                --patience ${PATIENCE} --weights-dir "${weights_dir}" \
                --task counting --with-aug
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        print_success "${model_type^^} ${dataset^^} training complete!"
        TRAINING_RESULTS+=("${model_type}_${dataset}:✓ Success")
    else
        print_error "${model_type^^} ${dataset^^} training failed!"
        TRAINING_RESULTS+=("${model_type}_${dataset}:✗ Failed")
    fi
}

# Train all models
print_header "📚 Starting Training Pipeline"

# JEPA Models (Multi-modal with visual teacher)
print_section "JEPA Models (Multi-modal)"
for dataset in easy medium hard extreme; do
    train_model "jepa" "${dataset}"
done

# JEPA Acoustic-Only (same weights, different evaluation)
print_section "JEPA Acoustic-Only (uses same weights as JEPA)"
for dataset in easy medium hard extreme; do
    # No training needed - uses same weights as JEPA
    TRAINING_RESULTS["jepa_acoustic_${dataset}"]="✓ (shares JEPA weights)"
done

# LeWM Models (Acoustic-only)
print_section "LeWM Models (Acoustic-only)"
for dataset in easy medium hard extreme; do
    train_model "lewm" "${dataset}"
done

# JEPA+SigReg Models (Multi-modal with SigReg)
print_section "JEPA+SigReg Models (Multi-modal + Gaussian Reg.)"
for dataset in easy medium hard extreme; do
    train_model "jepa_sigreg" "${dataset}"
done

# Print summary
print_header "📊 Training Summary"
echo ""
printf "%-25s | %-15s\n" "Model" "Status"
printf "%-25s-+-%-15s\n" "-------------------------" "---------------"

for result in "${TRAINING_RESULTS[@]}"; do
    model="${result%%:*}"
    status="${result##*:}"
    printf "%-25s | %-15s\n" "${model}" "${status}"
done

echo ""
print_header "✅ Training Pipeline Complete!"
echo ""
echo "Next steps:"
echo "  1. Evaluate models in simulation OR"
echo "  2. Run evaluation script: ./evaluate_counting_all.sh"
echo "  3. Generate results table: python3 generate_table.py --task counting"
echo ""
echo "Weights saved in: ${WEIGHTS_DIR}"
echo ""
