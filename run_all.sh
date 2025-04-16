python3 algorithms/extract_data.py --dataset lfw
python3 algorithms/extract_data.py --dataset agedb_30
python3 algorithms/extract_data.py --dataset cfp_fp

echo "Starting full preprocessing pipeline..."

# LFW
source activate py3_10
python3 algorithms/preprocess_faces.py \
  --json runs/extracted_data/lfw/samples.json \
  --dataset lfw \
  --aligner mtcnn

source activate ins_env # rerun
python3 algorithms/preprocess_faces.py \
  --json runs/extracted_data/lfw/samples.json \
  --dataset lfw \
  --aligner retinaface

# AgeDB-30
source activate py3_10
python3 algorithms/preprocess_faces.py \
  --json runs/extracted_data/agedb_30/samples.json \
  --dataset agedb_30 \
  --aligner mtcnn

source activate ins_env
python3 algorithms/preprocess_faces.py \
  --json runs/extracted_data/agedb_30/samples.json \
  --dataset agedb_30 \
  --aligner retinaface

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



echo "Starting evaluation..."

# LFW
python algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/lfw_retinaface/samples.json \
  --model arcface \
  --exp lfw_retina_arcface \
  --flip

python algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/lfw_mtcnn/samples.json \
  --model facenet \
  --exp lfw_mtcnn_facenet \
  --flip

# AgeDB-30
python algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/agedb_30_retinaface/samples.json \
  --model arcface \
  --exp agedb_retina_arcface \
  --flip

python algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/agedb_30_mtcnn/samples.json \
  --model facenet \
  --exp agedb_mtcnn_facenet \
  --flip

# CFP-FP
python algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/cfp_fp_retinaface/samples.json \
  --model arcface \
  --exp cfp_retina_arcface\
  --flip

python algorithms/eval_accuracy.py \
  --json runs/preprocessed_data/cfp_fp_mtcnn/samples.json \
  --model facenet \
  --exp cfp_mtcnn_facenet \
  --flip


echo "All evaluations completed."
