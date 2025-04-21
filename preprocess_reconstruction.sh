#!/bin/bash

set -e  # если что-то падает — скрипт останавливается

echo "Starting full preprocessing pipeline for reconstructed images..."

# # LFW
# source activate py3_10
# python3 algorithms/preprocess_faces_2.py \
#   --json runs/reconstructions/reconstruction_lfw_retinaface_arcface/samples.json \
#   --dataset lfw \
#   --aligner mtcnn \
#   --out_dir runs/preprocessed_for_another_model_reconstructions

source activate ins_env # rerun
python3 algorithms/preprocess_faces_2.py \
  --json runs/reconstructions/reconstruction_lfw_mtcnn_facenet/samples.json \
  --dataset lfw \
  --aligner retinaface \
  --out_dir runs/preprocessed_for_another_model_reconstructions

# # AgeDB-30
# source activate py3_10
# python3 algorithms/preprocess_faces_2.py \
#   --json runs/reconstructions/reconstruction_agedb_30_retinaface_arcface/samples.json \
#   --dataset agedb_30 \
#   --aligner mtcnn \
#   --out_dir runs/preprocessed_for_another_model_reconstructions

# source activate ins_env
# python3 algorithms/preprocess_faces_2.py \
#   --json runs/reconstructions/reconstruction_agedb_30_mtcnn_facenet/samples.json \
#   --dataset agedb_30 \
#   --aligner retinaface \
#   --out_dir runs/preprocessed_for_another_model_reconstructions

# # CFP-FP
# source activate py3_10
# python3 algorithms/preprocess_faces_2.py \
#   --json runs/reconstructions/reconstruction_cfp_fp_retinaface_arcface/samples.json \
#   --dataset cfp_fp \
#   --aligner mtcnn \
#   --out_dir runs/preprocessed_for_another_model_reconstructions

# source activate ins_env
# python3 algorithms/preprocess_faces_2.py \
#   --json runs/reconstructions/reconstruction_cfp_fp_mtcnn_facenet/samples.json \
#   --dataset cfp_fp \
#   --aligner retinaface \
#   --out_dir runs/preprocessed_for_another_model_reconstructions

source activate py3_10

