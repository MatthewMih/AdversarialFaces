import os
import argparse
import json
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def to_abs(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(ROOT, path))

def load_pairs_and_paths(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_paths = set()
    if 'folds' in data and data['folds']:
        for fold in data['folds']:
            for p1, p2, _ in fold:
                all_paths.add(to_abs(p1))
                all_paths.add(to_abs(p2))
    else:
        for p1, p2, _ in data['pairs']:
            all_paths.add(to_abs(p1))
            all_paths.add(to_abs(p2))

    return data, list(all_paths)

def preprocess_faces(json_path, out_base, dataset_name, aligner_type):
    if aligner_type == "mtcnn":
        from mtcnn_aligner import get_aligned_face
    elif aligner_type == "retinaface":
        from retina_aligner import get_aligned_face
    else:
        raise ValueError("Aligner must be 'mtcnn' or 'retinaface'.")

    out_images_dir = os.path.join(out_base, f"{dataset_name}_{aligner_type}", "images")
    os.makedirs(out_images_dir, exist_ok=True)

    data, all_paths = load_pairs_and_paths(json_path)
    common_root = os.path.commonpath(all_paths)

    old_to_new = {}
    dataset_roots = {
        "cfp_fp": os.path.join(ROOT, "runs", "extracted_data", "cfp_fp", "cfp-dataset", "Images"),
        "lfw": os.path.join(ROOT, "runs", "extracted_data", "lfw", "images"),
        "agedb_30": os.path.join(ROOT, "runs", "extracted_data", "agedb_30", "images"),
    }
    images_root_dir = dataset_roots.get(dataset_name, os.path.join(ROOT, "runs"))

    for path in tqdm(all_paths, desc=f"Preprocessing with {aligner_type}"):
        rel_path = os.path.relpath(path, start=common_root)
        save_path = os.path.join(out_images_dir, rel_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"\n\n\n\n {path}\n\n{save_path}\n\n\n\n")
        aligned = get_aligned_face(path)
        aligned.save(save_path)
        old_to_new[path] = save_path

    if 'folds' in data and data['folds']:
        new_folds = []
        for fold in data['folds']:
            new_fold = []
            for p1, p2, label in fold:
                new_fold.append((old_to_new[to_abs(p1)], old_to_new[to_abs(p2)], label))
            new_folds.append(new_fold)
        result = {'pairs': None, 'folds': new_folds}
    else:
        new_pairs = []
        for p1, p2, label in data['pairs']:
            new_pairs.append((old_to_new[to_abs(p1)], old_to_new[to_abs(p2)], label))
        result = {'pairs': new_pairs, 'folds': None}

    json_out_path = os.path.join(out_base, f"{dataset_name}_{aligner_type}", "samples.json")
    with open(json_out_path, 'w') as f:
        json.dump(result, f)

    print(f"Saved updated JSON to {json_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to input JSON with pairs/folds")
    parser.add_argument("--aligner", choices=["mtcnn", "retinaface"], required=True, help="Preprocessing method")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., lfw, agedb_30, cfp_fp)")
    parser.add_argument("--out_dir", default="runs/preprocessed_data", help="Base output directory")

    args = parser.parse_args()
    preprocess_faces(args.json, args.out_dir, args.dataset, args.aligner)
