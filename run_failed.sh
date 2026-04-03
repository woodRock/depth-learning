#!/bin/bash

# Re-runs failed experiments from failed.csv
# Each row gets a seed from the SEEDS array (cycling if needed)

SEEDS=(42 123 777 999 2024 1337 555 101 888 1234)
directory="/vol/ecrg-solar/woodj4/depth-learning"

echo "Queuing failed experiments from failed.csv"

seed_index=0

# Skip header row with tail -n +2
while IFS=',' read -r name state created runtime task _ _ _ _ dataset architecture epochs _; do
    # Strip surrounding quotes
    task="${task//\"/}"
    dataset="${dataset//\"/}"
    architecture="${architecture//\"/}"
    epochs="${epochs//\"/}"

    seed="${SEEDS[$seed_index % ${#SEEDS[@]}]}"
    ((seed_index++))

    case $architecture in
        jepa)
            task -G 1 -d "$directory" depth train jepa --model transformer --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$task" --with-aug
            ;;
        lewm)
            task -G 1 -d "$directory" depth train lewm --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$task" --with-aug
            ;;
        lewm_plus)
            task -G 1 -d "$directory" depth train lewm_plus --model transformer --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$task" --with-aug
            ;;
        fusion)
            task -G 1 -d "$directory" depth train fusion --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$task" --with-aug
            ;;
        translator)
            task -G 1 -d "$directory" depth train translator --dataset "$dataset" --seed "$seed" --epochs "$epochs" --task "$task" --with-aug
            ;;
        *)
            echo "Unknown architecture: $architecture (skipping $name)"
            ;;
    esac
done < <(tail -n +2 failed.csv)

echo "Success: All failed experiments have been re-queued."
echo "Use 'task' to view the queue status."
