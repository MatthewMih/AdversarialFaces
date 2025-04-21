#!/bin/bash

set -e

echo "Starting evaluation..."


# LFW
python3 algorithms/eval_accuracy.py \
  --json runs/preprocessed_for_another_model_reconstructions/lfw_retinaface/samples.json \
  --model arcface \
  --exp lfw_reconstructed_from_facenet_evaluated_by_arcface \
  --flip

# python3 algorithms/eval_accuracy.py \
#   --json runs/preprocessed_for_another_model_reconstructions/lfw_mtcnn/samples.json \
#   --model facenet \
#   --exp lfw_reconstructed_from_arcface_evaluated_by_facenet \
#   --flip

# # AgeDB-30
# python3 algorithms/eval_accuracy.py \
#   --json runs/preprocessed_for_another_model_reconstructions/agedb_30_retinaface/samples.json \
#   --model arcface \
#   --exp agedb_reconstructed_from_facenet_evaluated_by_arcface \
#   --flip

# python3 algorithms/eval_accuracy.py \
#   --json runs/preprocessed_for_another_model_reconstructions/agedb_30_mtcnn/samples.json \
#   --model facenet \
#   --exp lagedb_reconstructed_from_arcface_evaluated_by_facenet \
#   --flip

# # CFP-FP
# python3 algorithms/eval_accuracy.py \
#   --json runs/preprocessed_for_another_model_reconstructions/cfp_fp_retinaface/samples.json \
#   --model arcface \
#   --exp cfp_reconstructed_from_facenet_evaluated_by_arcface \
#   --flip

# python3 algorithms/eval_accuracy.py \
#   --json runs/preprocessed_for_another_model_reconstructions/cfp_fp_mtcnn/samples.json \
#   --model facenet \
#   --exp cfp_reconstructed_from_arcface_evaluated_by_facenet \
#   --flip


echo "All evaluations completed."
