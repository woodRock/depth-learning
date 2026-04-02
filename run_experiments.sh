#!/bin/bash

# Experiment runner for depth learning models
# Queues all models on all datasets 10 times with fixed seeds using task-spooler (ts)

SEEDS=(42 123 777 999 2024 1337 555 101 888 1234)
DATASETS=("easy" "medium" "hard" "extreme")
MODELS=("jepa" "lewm" "lewm_plus" "fusion" "translator")

# Default settings
EPOCHS=30 # Reduced for experiment sweep, adjust as needed
BATCH_SIZE=32
TASK=${1:-counting} # Default to counting, allow override via first argument

echo "Queuing experiment sweep (Task: $TASK): ${#MODELS[@]} models x ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds"
echo "Usage: ./run_experiments.sh [task_name] (default: counting)"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Construct command based on model type
            case $model in
                jepa)
                    ts -G 1 depth train jepa --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                lewm)
                    ts -G 1 depth train lewm --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                lewm_plus)
                    ts -G 1 depth train lewm_plus --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                fusion)
                    ts -G 1 depth train fusion --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                translator)
                    ts -G 1 depth train translator --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
            esac
        done
    done
done

echo "Success: All tasks have been added to the task-spooler queue."
echo "Use 'ts' to view the queue status."
