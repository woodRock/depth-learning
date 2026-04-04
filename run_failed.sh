#!/bin/bash

# Re-runs failed experiments from failed.csv
# Each row gets a seed from the SEEDS array (cycling if needed)

SEEDS=(42 123 777 999 2024 1337 555 101 888 1234)
directory="/vol/ecrg-solar/woodj4/depth-learning"

echo "Queuing failed experiments from failed.csv"

seed_index=0

# Read header to find column indices dynamically
header=$(head -n 1 failed.csv)
IFS=',' read -ra header_cols <<< "$header"

col_task=-1
col_dataset=-1
col_architecture=-1
col_epochs=-1

for i in "${!header_cols[@]}"; do
    col="${header_cols[$i]//\"/}"
    case "$col" in
        task)         col_task=$i ;;
        dataset)      col_dataset=$i ;;
        architecture) col_architecture=$i ;;
        epochs)       col_epochs=$i ;;
    esac
done

if [[ $col_task -eq -1 || $col_dataset -eq -1 || $col_architecture -eq -1 || $col_epochs -eq -1 ]]; then
    echo "Error: failed.csv is missing one or more required columns (task, dataset, architecture, epochs)"
    exit 1
fi

# Skip header row with tail -n +2
while IFS=',' read -r -u 3 -a row || [[ ${#row[@]} -gt 0 ]]; do
    exp_task="${row[$col_task]//\"/}"
    dataset="${row[$col_dataset]//\"/}"
    architecture="${row[$col_architecture]//\"/}"
    epochs="${row[$col_epochs]//\"/}"

    # Skip empty rows
    [[ -z "$exp_task" && -z "$dataset" && -z "$architecture" ]] && continue

    seed="${SEEDS[$seed_index % ${#SEEDS[@]}]}"
    ((seed_index++))

    case $architecture in
        jepa)
            task -G 1 -d "$directory" depth train jepa --model transformer --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$exp_task" --with-aug
            ;;
        lewm)
            task -G 1 -d "$directory" depth train lewm --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$exp_task" --with-aug
            ;;
        lewm_plus)
            task -G 1 -d "$directory" depth train lewm_plus --model transformer --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$exp_task" --with-aug
            ;;
        fusion)
            task -G 1 -d "$directory" depth train fusion --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$exp_task" --with-aug
            ;;
        translator)
            task -G 1 -d "$directory" depth train translator --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$exp_task" --with-aug
            ;;
        *)
            echo "Unknown architecture: $architecture (skipping $name)"
            ;;
    esac
done 3< <(tail -n +2 failed.csv)

echo "Success: All failed experiments have been re-queued."
echo "Use 'task' to view the queue status."
