import os, json, tqdm, tarfile, zipfile
import argparse
from datasets import load_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def parse_txt_lfw(file_path):
    list_positive, list_negative = [], []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("10"):
                continue
            parts = line.split()
            if len(parts) == 3:
                name, num1, num2 = parts
                list_positive.append((name, int(num1), int(num2)))
            elif len(parts) == 4:
                name1, num1, name2, num2 = parts
                list_negative.append((name1, int(num1), name2, int(num2)))
            else:
                raise ValueError(f"Неверная строка: {parts}")
    return list_positive, list_negative

def extract_lfw():
    print("Extracting LFW...")
    images_dir = os.path.join(ROOT, "runs", "extracted_data", "lfw", "images")
    os.makedirs(images_dir, exist_ok=True)
    archive_path = os.path.join(ROOT, "data", "lfw.tgz")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("lfw/"):
                member.name = member.name[len("lfw/"):]  # remove lfw/ subdir prefix
            tar.extract(member, path=images_dir)
    print(f"Extracted images to {images_dir}")

    txt_path = os.path.join(ROOT, "data", "pairs_lfw.txt")
    pos, neg = parse_txt_lfw(txt_path)

    pairs = []
    for name, idx1, idx2 in pos:
        pairs.append((f'{images_dir}/{name}/{name}_{idx1:04d}.jpg',
                      f'{images_dir}/{name}/{name}_{idx2:04d}.jpg', 1))
    for name1, idx1, name2, idx2 in neg:
        pairs.append((f'{images_dir}/{name1}/{name1}_{idx1:04d}.jpg',
                      f'{images_dir}/{name2}/{name2}_{idx2:04d}.jpg', 0))

    json_out = os.path.join(ROOT, "runs", "extracted_data", "lfw", "samples.json")
    with open(json_out, 'w') as f:
        json.dump({'pairs': pairs, 'folds': None}, f)
    print(f"Saved pairs to {json_out}")

def load_extract_agedb_30():
    print("Loading and extracting AgeDB-30...")
    out_dir = os.path.join(ROOT, "runs", "extracted_data", "agedb_30", "images")
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("cat-claws/face-verification", split='agedb_30')

    pairs = []
    for idx, sample in enumerate(tqdm.tqdm(ds, desc='Extracting AgeDB_30')):
        pth1 = os.path.join(out_dir, f"{idx}_a.png")
        pth2 = os.path.join(out_dir, f"{idx}_b.png")
        sample['image1'].save(pth1)
        sample['image2'].save(pth2)
        pairs.append((pth1, pth2, int(sample['target'])))

    json_out = os.path.join(ROOT, "runs", "extracted_data", "agedb_30", "samples.json")
    with open(json_out, 'w') as f:
        json.dump({'pairs': pairs, 'folds': None}, f)

    print(f"Saved AgeDB-30 to {json_out}")

def extract_cfp_fp():
    print("Extracting CFP-FP...")
    archive_path = os.path.join(ROOT, "data", "cfp-dataset.zip")
    out_dir = os.path.join(ROOT, "runs", "extracted_data", "cfp_fp")
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    print(f"Extracted CFP-FP to {out_dir}")
    
    print("Generating CFP-FP fold pairs...")
    base_dir = os.path.join(ROOT, "runs", "extracted_data", "cfp_fp", "cfp-dataset")

    f_path = os.path.join(base_dir, "Protocol", "Pair_list_F.txt")
    p_path = os.path.join(base_dir, "Protocol", "Pair_list_P.txt")

    with open(f_path, "r") as f:
        frontal_lines = [x.strip() for x in f]
    with open(p_path, "r") as f:
        profile_lines = [x.strip() for x in f]

    num2path_f = {}
    num2path_p = {}

    for line in frontal_lines:
        num, relpath = line.split()
        num2path_f[int(num)] = os.path.join(base_dir, relpath[3:])
    for line in profile_lines:
        num, relpath = line.split()
        num2path_p[int(num)] = os.path.join(base_dir, relpath[3:])

    folds = []
    for fold_idx in range(1, 11):
        fold = []

        same_path = os.path.join(base_dir, "Protocol", "Split", "FP", f"{fold_idx:02d}", "same.txt")
        diff_path = os.path.join(base_dir, "Protocol", "Split", "FP", f"{fold_idx:02d}", "diff.txt")

        with open(same_path, "r") as f:
            for line in f:
                i, j = map(int, line.strip().split(','))
                fold.append((num2path_f[i], num2path_p[j], 1))

        with open(diff_path, "r") as f:
            for line in f:
                i, j = map(int, line.strip().split(','))
                fold.append((num2path_f[i], num2path_p[j], 0))

        folds.append(fold)

    out_json = os.path.join(base_dir, "folds_fp.json")
    with open(out_json, "w") as f:
        json.dump({'pairs': None, 'folds': folds}, f)

    print(f"Saved CFP-FP folds to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lfw", "agedb_30", "cfp_fp"],
                        help="Which dataset to extract")
    args = parser.parse_args()

    if args.dataset == "lfw":
        extract_lfw()
    elif args.dataset == "agedb_30":
        load_extract_agedb_30()
    elif args.dataset == "cfp_fp":
        extract_cfp_fp()
