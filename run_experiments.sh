#!/bin/bash

# Experiment runner for depth learning models
# Queues all models on all datasets 10 times with fixed seeds using task (task spooler)

SEEDS=(42 123 777 999 2024 1337 555 101 888 1234)
DATASETS=("easy" "medium" "hard" "extreme")
MODELS=("jepa" "lewm" "lewm_plus" "fusion" "translator" "vit_jepa")

# Default settings
EPOCHS=30 # Reduced for experiment sweep, adjust as needed
BATCH_SIZE=32
TASK=${1:-counting} # Default to counting, allow override via first argument
directory="/vol/ecrg-solar/woodj4/depth-learning"

echo "Queuing experiment sweep (Task: $TASK): ${#MODELS[@]} models x ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds"
echo "Usage: ./run_experiments.sh [task_name] (default: counting)"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Construct command based on model type
            case $model in
                jepa)
                    task -G 1 -d "$directory" depth train jepa --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                lewm)
                    task -G 1 -d "$directory" depth train lewm --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                lewm_plus)
                    task -G 1 -d "$directory" depth train lewm_plus --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                fusion)
                    task -G 1 -d "$directory" depth train fusion --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                translator)
                    task -G 1 -d "$directory" depth train translator --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug
                    ;;
                vit_jepa)
                    # Pretrain ViT for 50 epochs then train JEPA for $EPOCHS epochs
                    task -G 1 -d "$directory" depth train vit_jepa --model transformer --dataset "$dataset" --seed "$seed" --epochs "$EPOCHS" --task "$TASK" --with-aug --vit-pretrain-epochs 50
                    ;;
            esac
        done
    done
done

echo "Success: All tasks have been added to the task spooler queue."
echo "Use 'task' to view the queue status."
