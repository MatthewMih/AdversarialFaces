#!/bin/bash

set -e

echo "Starting evaluation..."


# LFW
python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_lfw_retinaface_arcface/samples.json \
  --model arcface \
  --exp lfw_retina_arcface_raw_reconstruction \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_lfw_mtcnn_facenet/samples.json \
  --model facenet \
  --exp lfw_mtcnn_facenet_raw_reconstruction \
  --flip

# AgeDB-30
python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_agedb_30_retinaface_arcface/samples.json \
  --model arcface \
  --exp agedb_retina_arcface_raw_reconstruction \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_agedb_30_mtcnn_facenet/samples.json \
  --model facenet \
  --exp agedb_mtcnn_facenet_raw_reconstruction \
  --flip

# CFP-FP
python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_cfp_fp_retinaface_arcface/samples.json \
  --model arcface \
  --exp cfp_retina_arcface_raw_reconstruction \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_cfp_fp_mtcnn_facenet/samples.json \
  --model facenet \
  --exp cfp_mtcnn_facenet_raw_reconstruction \
  --flip

echo "Starting full preprocessing pipeline for reconstructed images..."

# LFW
source activate py3_10
python3 algorithms/preprocess_faces.py \
  --json runs/reconstructions/reconstruction_lfw_retinaface_arcface/samples.json \
  --dataset lfw \
  --aligner mtcnn \
  --out_dir runs/preprocessed_for_another_model_reconstructions

source activate ins_env # rerun
python3 algorithms/preprocess_faces.py \
  --json runs/reconstructions/reconstruction_lfw_mtcnn_facenet/samples.json \
  --dataset lfw \
  --aligner retinaface \
  --out_dir runs/preprocessed_for_another_model_reconstructions

# AgeDB-30
python3 algorithms/preprocess_faces.py \
  --json runs/reconstructions/reconstruction_agedb_30_retinaface_arcface/samples.json \
  --dataset agedb_30 \
  --aligner mtcnn \
  --out_dir runs/preprocessed_for_another_model_reconstructions

python3 algorithms/preprocess_faces.py \
  --json runs/reconstructions/reconstruction_agedb_30_mtcnn_facenet/samples.json \
  --dataset agedb_30 \
  --aligner retinaface \
  --out_dir runs/preprocessed_for_another_model_reconstructions

# CFP-FP
python3 algorithms/preprocess_faces.py \
  --json runs/reconstructions/reconstruction_cfp_fp_retinaface_arcface/samples.json \
  --dataset cfp_fp \
  --aligner mtcnn \
  --out_dir runs/preprocessed_for_another_model_reconstructions

python3 algorithms/preprocess_faces.py \
  --json runs/reconstructions/reconstruction_cfp_fp_mtcnn_facenet/samples.json \
  --dataset cfp_fp \
  --aligner retinaface \
  --out_dir runs/preprocessed_for_another_model_reconstructions

echo "Starting evaluation..."


# LFW
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/lfw_retinaface/samples.json \
  --model arcface \
  --exp lfw_reconstructed_from_facenet_evaluated_by_arcface \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/lfw_mtcnn/samples.json \
  --model facenet \
  --exp lfw_reconstructed_from_arcface_evaluated_by_facenet \
  --flip

# AgeDB-30
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/agedb_30_retinaface/samples.json \
  --model arcface \
  --exp agedb_reconstructed_from_facenet_evaluated_by_arcface \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/agedb_30_mtcnn/samples.json \
  --model facenet \
  --exp lagedb_reconstructed_from_arcface_evaluated_by_facenet \
  --flip

# CFP-FP
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/cfp_fp_retinaface/samples.json \
  --model arcface \
  --exp cfp_reconstructed_from_facenet_evaluated_by_arcface \
  --flip

python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/cfp_fp_mtcnn/samples.json \
  --model facenet \
  --exp cfp_reconstructed_from_arcface_evaluated_by_facenet \
  --flip


echo "All evaluations completed."

