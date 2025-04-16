#!/bin/bash

set -e  # если что-то падает — скрипт останавливается

echo "Starting full preprocessing pipeline..."

# # LFW
# source activate py3_10
# python3 algorithms/preprocess_faces.py \
#   --json runs/extracted_data/lfw/samples.json \
#   --dataset lfw \
#   --aligner mtcnn

# source activate ins_env
# python3 algorithms/preprocess_faces.py \
#   --json runs/extracted_data/lfw/samples.json \
#   --dataset lfw \
#   --aligner retinaface

# # AgeDB-30
# source activate py3_10
# python3 algorithms/preprocess_faces.py \
#   --json runs/extracted_data/agedb_30/samples.json \
#   --dataset agedb_30 \
#   --aligner mtcnn

# source activate ins_env
# python3 algorithms/preprocess_faces.py \
#   --json runs/extracted_data/agedb_30/samples.json \
#   --dataset agedb_30 \
#   --aligner retinaface

# CFP-FP
source activate py3_10
python3 algorithms/preprocess_faces.py \
  --json runs/extracted_data/cfp_fp/cfp-dataset/folds_fp.json \
  --dataset cfp_fp \
  --aligner mtcnn

source activate ins_env
python3 algorithms/preprocess_faces.py \
  --json runs/extracted_data/cfp_fp/cfp-dataset/folds_fp.json \
  --dataset cfp_fp \
  --aligner retinaface

source activate py3_10

