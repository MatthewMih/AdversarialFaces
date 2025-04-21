#!/bin/bash

set -e

echo "Starting evaluation..."


# # LFW
# python3 algorithms/eval_accuracy.py \
#   --json runs/reconstructions/reconstruction_lfw_retinaface_arcface/samples.json \
#   --model arcface \
#   --exp lfw_retina_arcface_raw_reconstruction \
#   --flip

# python3 algorithms/eval_accuracy.py \
#   --json runs/reconstructions/reconstruction_lfw_mtcnn_facenet/samples.json \
#   --model facenet \
#   --exp lfw_mtcnn_facenet_raw_reconstruction \
#   --flip

# # AgeDB-30
# python3 algorithms/eval_accuracy.py \
#   --json runs/reconstructions/reconstruction_agedb_30_retinaface_arcface/samples.json \
#   --model arcface \
#   --exp agedb_retina_arcface_raw_reconstruction \
#   --flip

python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_agedb_30_mtcnn_facenet/samples.json \
  --model facenet \
  --exp agedb_mtcnn_facenet_raw_reconstruction \
  --flip

# # CFP-FP
# python3 algorithms/eval_accuracy.py \
#   --json runs/reconstructions/reconstruction_cfp_fp_retinaface_arcface/samples.json \
#   --model arcface \
#   --exp cfp_retina_arcface_raw_reconstruction \
#   --flip

python3 algorithms/eval_accuracy.py \
  --json runs/reconstructions/reconstruction_cfp_fp_mtcnn_facenet/samples.json \
  --model facenet \
  --exp cfp_mtcnn_facenet_raw_reconstruction \
  --flip


echo "All evaluations completed."
