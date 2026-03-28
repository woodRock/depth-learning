#!/bin/bash
# =============================================================================
# Train All Models Script
# =============================================================================
# This script trains JEPA and LeWM models on all difficulty datasets
# and saves weights in the correct directories for the Bevy simulation.
#
# Usage:
#   ./train_all_models.sh [epochs] [patience]
#
# Arguments:
#   epochs   - Number of training epochs (default: 100)
#   patience - Early stopping patience (default: 15)
#
# Example:
#   ./train_all_models.sh 100 15
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

# Helper functions
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

# Navigate to script directory
cd "${SCRIPT_DIR}"

print_header "🚀 Training All Models for Depth Learning Benchmark"
echo "Configuration:"
echo "  Epochs:   ${EPOCHS}"
echo "  Patience: ${PATIENCE}"
echo "  Weights:  ${WEIGHTS_DIR}"
echo ""

# Create weights directory structure
print_section "Setting Up Directory Structure"
for dataset in easy medium hard extreme; do
    mkdir -p "${WEIGHTS_DIR}/jepa_${dataset}"
    mkdir -p "${WEIGHTS_DIR}/lewm_${dataset}"
    print_success "Created directories for ${dataset}"
done

# Track training results
declare -A TRAINING_RESULTS

# Training function
train_model() {
    local model_type=$1
    local dataset=$2
    local weights_dir="${WEIGHTS_DIR}/${model_type}_${dataset}"
    
    print_section "Training ${model_type^^} on ${dataset^^} Dataset"
    echo "Weights will be saved to: ${weights_dir}"
    
    if python3 train.py ${model_type} \
        --dataset ${dataset} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --weights-dir "${weights_dir}"; then
        print_success "${model_type^^} ${dataset^^} training complete!"
        TRAINING_RESULTS["${model_type}_${dataset}"]="✓ Success"
    else
        print_error "${model_type^^} ${dataset^^} training failed!"
        TRAINING_RESULTS["${model_type}_${dataset}"]="✗ Failed"
    fi
}

# Train all models
print_header "📚 Starting Training Pipeline"

# JEPA Models (Multi-modal with visual teacher)
print_section "JEPA Models (Multi-modal)"
for dataset in easy medium hard extreme; do
    train_model "jepa" "${dataset}"
done

# LeWM Models (Acoustic-only)
print_section "LeWM Models (Acoustic-only)"
for dataset in easy medium hard extreme; do
    train_model "lewm" "${dataset}"
done

# Print summary
print_header "📊 Training Summary"
echo ""
printf "%-20s | %-15s\n" "Model" "Status"
printf "%-20s-+-%-15s\n" "--------------------" "---------------"

for key in "${!TRAINING_RESULTS[@]}"; do
    printf "%-20s | %-15s\n" "${key}" "${TRAINING_RESULTS[$key]}"
done

echo ""
print_header "✅ Training Pipeline Complete!"
echo ""
echo "Next steps:"
echo "  1. Start the Python server:  python3 serve.py"
echo "  2. Run the Bevy simulation:  cargo run --release"
echo "  3. Evaluate models in simulation using the Test Evaluation UI"
echo ""
echo "Weights are saved in: ${WEIGHTS_DIR}"
echo "  - jepa_easy/, jepa_medium/, jepa_hard/, jepa_extreme/"
echo "  - lewm_easy/, lewm_medium/, lewm_hard/, lewm_extreme/"
echo ""
