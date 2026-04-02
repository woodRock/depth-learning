#!/bin/bash

# Experiment runner for depth learning models
# Runs all models on all datasets 10 times with fixed seeds for reproducibility

SEEDS=(42 123 777 999 2024 1337 555 101 888 1234)
DATASETS=("easy" "medium" "hard" "extreme")
MODELS=("jepa" "lewm" "lewm_plus" "decoder" "fusion" "translator" "mae")

# Default settings
EPOCHS=30 # Reduced for experiment sweep, adjust as needed
BATCH_SIZE=32

echo "Starting experiment sweep: ${#MODELS[@]} models x ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "--------------------------------------------------------------------------------"
            echo "Running Model: $model | Dataset: $dataset | Seed: $seed"
            echo "--------------------------------------------------------------------------------"
            
            # Construct command based on model type
            case $model in
                jepa)
                    depth train jepa --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task presence --with-aug
                    ;;
                lewm)
                    depth train lewm --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task presence --with-aug
                    ;;
                lewm_plus)
                    depth train lewm_plus --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task presence --with-aug
                    ;;
                decoder)
                    depth train decoder --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --with-aug
                    ;;
                fusion)
                    depth train fusion --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task presence --with-aug
                    ;;
                translator)
                    depth train translator --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task presence --with-aug
                    ;;
                mae)
                    depth train mae --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --with-aug
                    ;;
            esac
            
            if [ $? -ne 0 ]; then
                echo "Error occurred during $model on $dataset with seed $seed. Continuing with next experiment..."
            fi
        done
    done
done

echo "Experiment sweep completed!"
