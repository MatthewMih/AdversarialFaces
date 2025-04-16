#!/bin/bash

set -e

echo "Starting evaluation..."

# LFW
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/lfw_retinaface/samples.json \
  --model arcface \
  --exp lfw_retina_arcface \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/lfw_mtcnn/samples.json \
  --model facenet \
  --exp lfw_mtcnn_facenet \
  --flip

# AgeDB-30
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/agedb_30_retinaface/samples.json \
  --model arcface \
  --exp agedb_retina_arcface \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/agedb_30_mtcnn/samples.json \
  --model facenet \
  --exp agedb_mtcnn_facenet \
  --flip

# CFP-FP
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/cfp_fp_retinaface/samples.json \
  --model arcface \
  --exp cfp_retina_arcface \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/cfp_fp_mtcnn/samples.json \
  --model facenet \
  --exp cfp_mtcnn_facenet \
  --flip


echo "All evaluations completed."
