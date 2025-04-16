import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import KFold
from facenet_pytorch import InceptionResnetV1
from iresnet import iresnet100

torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def vectorize(img, size):
    if img.size != (size, size):
        img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255
    return torch.tensor(arr).permute(2, 0, 1).reshape(3 * size * size)

def get_image(path):
    return Image.open(path).convert("RGB")

arcface_model = iresnet100().to(device).eval()
arcface_model.load_state_dict(torch.load(f"{ROOT}/models/ms1mv3_arcface_r100_fp16_backbone.pth"))

def arcface_embed(v, flip=False):
    x = 2 * v.reshape(-1, 3, 112, 112).to(device) - 1
    emb = arcface_model(x[:, [2,1,0], :, :])
    if flip:
        emb_f = arcface_model(torch.flip(x, [3]))
        emb = emb + emb_f
    return emb


facenet_model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
def facenet_embed(v, flip=False):
    x = 2 * v.reshape(-1, 3, 160, 160).to(device) - 1
    emb = facenet_model(x[:, [2,1,0], :, :])
    if flip:
        emb_f = facenet_model(torch.flip(x, [3]))
        emb = emb + emb_f
    return emb


def find_best_threshold(scores, labels):
    thresholds = np.linspace(min(scores), max(scores), 1000)
    best_acc = 0
    best_thresh = thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc

def cross_val_accuracy(folds, emb_cache, sim_fn):
    accs = []
    for i, fold in enumerate(folds):
        train = [s for j, f in enumerate(folds) if j != i for s in f]
        test = folds[i]

        train_scores = [sim_fn(emb_cache[a], emb_cache[b]).item() for a, b, _ in train]
        train_labels = [l for _, _, l in train]
        best_thresh, _ = find_best_threshold(np.array(train_scores), np.array(train_labels))

        test_scores = [sim_fn(emb_cache[a], emb_cache[b]).item() for a, b, _ in test]
        test_labels = [l for _, _, l in test]
        preds = (np.array(test_scores) >= best_thresh).astype(int)
        acc = np.mean(preds == test_labels)
        accs.append(acc)
        print(f"  Fold {i+1}: acc = {acc:.4f} (thresh = {best_thresh:.4f})")

    return np.mean(accs), np.std(accs)

def build_embedding_cache(paths, embed_fn, size, flip):
    cache = {}
    for path in tqdm(paths, desc="Embedding images"):
        img = get_image(path)
        vec = vectorize(img, size)
        emb = embed_fn(vec, flip).detach().cpu()
        cache[path] = emb
    return cache

def main(json_path, model_type, exp_name, flip=False):
    with open(json_path) as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    folds = data.get("folds", [])

    if not folds and pairs:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        folds = [[] for _ in range(10)]
        for fold_idx, (_, test_idx) in enumerate(kf.split(pairs)):
            for i in test_idx:
                folds[fold_idx].append(pairs[i])
        print("No folds found — created 10-fold split from flat pairs.")

    all_paths = set()
    for fold in folds:
        for a, b, _ in fold:
            all_paths.add(a)
            all_paths.add(b)

    if model_type == "arcface":
        embed_fn = arcface_embed
        input_size = 112
    elif model_type == "facenet":
        embed_fn = facenet_embed
        input_size = 160
    else:
        raise ValueError("Unknown model type")

    emb_cache = build_embedding_cache(sorted(all_paths), embed_fn, input_size, flip)
    acc, std = cross_val_accuracy(folds, emb_cache, torch.nn.functional.cosine_similarity)

    log_path = os.path.join(ROOT, "runs", "results.txt")
    with open(log_path, "a") as f:
        f.write(f"{exp_name}: acc = {acc:.4f}, std = {std:.4f}, model = {model_type}, flip = {flip}\n")

    print(f"Accuracy: {acc:.4f} ± {std:.4f} (logged to {log_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to samples.json")
    parser.add_argument("--model", required=True, choices=["arcface", "facenet"])
    parser.add_argument("--exp", required=True, help="Experiment name (used in log)")
    parser.add_argument("--flip", action="store_true", help="Use horizontal flip during embedding")

    args = parser.parse_args()
    print(f'Starting eval: {args.json}, {args.model} -- {args.exp}')
    main(args.json, args.model, args.exp, flip=args.flip)
