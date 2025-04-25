#!/bin/bash

set -e

NUM_THREADS=4  # How many GPUs do you have?
PY_SCRIPT=algorithms/reconstruct_faces.py

declare -a TASKS=(
  "lfw retinaface arcface"
  "lfw mtcnn facenet"
  "agedb_30 retinaface arcface"
  "agedb_30 mtcnn facenet"
  "cfp_fp retinaface arcface"
  "cfp_fp mtcnn facenet"
)

for TASK in "${TASKS[@]}"; do
  IFS=' ' read -r dataset aligner model <<< "$TASK"
  echo "🧪 Starting reconstruction: $dataset / $aligner / $model"

  JSON_PATH="runs/preprocessed_data/${dataset}_${aligner}/samples.json"
  OUTNAME="${dataset}_${aligner}_${model}"

  for i in $(seq 0 $((NUM_THREADS-1))); do
    echo "  → Launching shard $i on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python $PY_SCRIPT \
      --json "$JSON_PATH" \
      --model "$model" \
      --gpu "0" \
      --threads $NUM_THREADS \
      --index $i \
      --outname "$OUTNAME" &
  done

  wait
  echo "Finished: $dataset / $aligner / $model"
done

echo "All reconstructions completed."
