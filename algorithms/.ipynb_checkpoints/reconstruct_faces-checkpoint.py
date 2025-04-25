import os
import sys
import json
import torch
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1, MTCNN
from iresnet import iresnet100

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--json", required=True)
parser.add_argument("--model", choices=["arcface", "facenet"], required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--outname", default=None, help="Name suffix for output directory")
args = parser.parse_args()

thread = args.index
num_threads = args.threads
device = torch.device(f"cuda:{args.gpu}")




torch.set_grad_enabled(False)

num_iters = 15000
test_iters = 500
bs = 32

# === Пути ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_NAME = f"reconstruction_{args.outname or args.model}"
EXP_DIR = os.path.join(ROOT, "runs", "reconstructions", EXP_NAME)
OUT_IMG_DIR = os.path.join(EXP_DIR, "images")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

arcface_model = iresnet100().eval().to(device)
arcface_model.load_state_dict(torch.load(os.path.join(ROOT, "models", "ms1mv3_arcface_r100_fp16_backbone.pth")))

facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def embed_arcface(v):
    x = v.reshape(-1, 3, 112, 112).to(device)
    x = 2 * x - 1
    return arcface_model(x)#arcface_model(x[:, [2,1,0], :, :])


def embed_facenet(v):
    x = v.reshape(-1, 3, 112, 112).to(device)
    x = torch.nn.functional.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
    x = 2 * x - 1
    return facenet_model(x) #facenet_model(x[:, [2,1,0], :, :])

embed_primary = embed_arcface if args.model == "arcface" else embed_facenet
sim_fn = lambda x, emb2: torch.cosine_similarity(embed_primary(x), emb2)


with open(args.json) as f:
    data = json.load(f)

if isinstance(data, list):
    images_to_process = sorted(data)
    mode = 'list'
elif 'folds' in data and data['folds']:
    images_to_process = sorted(set(p[0] for fold in data['folds'] for p in fold if p[2] == 1))
    mode = 'folds'
elif 'pairs' in data and data['pairs']:
    images_to_process = sorted(set(p[0] for p in data['pairs'] if p[2] == 1))
    mode = 'pairs'
else:
    raise ValueError("Unsupported JSON format")

images_to_process = images_to_process[thread::num_threads]
print(f"Found {len(images_to_process)} images to process for thread {thread}")

eigenfaces = torch.load(os.path.join(ROOT, "models", "eigenvectors_denormalized_Square_ffhq.pt")).to(device)
mean_face = torch.load(os.path.join(ROOT, "models", "mtcnn_crops_ffhq_average_vector_square.pt")).to(device)

def vectorize(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((112, 112))
    x = np.array(img).astype(np.float32) / 255
    return torch.tensor(x).permute(2, 0, 1).reshape(3*112*112)

def imaginize(x):
    img = x.reshape(3, 112, 112).permute(1, 2, 0).cpu().detach().numpy() * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.data = [vectorize(Image.open(p)) for p in tqdm.tqdm(paths, desc="Loading images")]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.paths[idx]

dataloader = DataLoader(ImageDataset(images_to_process), batch_size=bs, shuffle=False)

def train_iteration(coordinates, emb_target, sigma, lr):
    noise = torch.normal(0, sigma, size=(coordinates.shape[0], eigenfaces.shape[0])).to(device)
    dv = torch.einsum('mc,bm->bc', eigenfaces, noise)
    v = mean_face + torch.einsum('mc,bm->bc', eigenfaces, coordinates)
    v_plus = v + dv
    v_minus = v - dv
    grad_coef = -eigenfaces.shape[1] * (sim_fn(v_plus, emb_target) - sim_fn(v_minus, emb_target))[..., None] / (2 * sigma)
    v_new = v - lr * grad_coef * dv
    coordinates = coordinates - lr * grad_coef * noise
    querries = 2
    return v_new, querries, coordinates

results = {}
updated_json = {'pairs': [], 'folds': []}
for batch_idx, (v0, paths) in enumerate(tqdm.tqdm(dataloader, desc=f"Thread {thread}")):
    for pth in paths:
        results[pth] = {}
    v0 = v0.to(device)  ## flatten-faces
    emb_target = embed_primary(v0)

    LR = 1e-5
    pretrain_coords = []
    pretrain_scores = []
    total_querries = 0
    for attempt_idx in tqdm.tqdm(range(10)):
        coordinates = torch.zeros((v0.shape[0], len(eigenfaces))).to(device)
        for iter_ in range(500):
            v, querries, coordinates = train_iteration(coordinates, emb_target, sigma=0.3, lr=LR)
            total_querries += querries
        sims = sim_fn(v, emb_target)
        pretrain_coords.append(coordinates)
        pretrain_scores.append(sims)
    pretrain_coords = torch.stack(pretrain_coords) # [10, B, 1024]
    pretrain_scores = torch.stack(pretrain_scores) # [10, B]

    coordinates = []
    for b in range(len(v0)):
        best_idx = 0
        for attempt_idx in range(len(pretrain_scores)):
            if pretrain_scores[attempt_idx, b] >= pretrain_scores[best_idx, b]:
                best_idx = attempt_idx
        coordinates.append(pretrain_coords[best_idx, b])
    coords = torch.stack(coordinates)
    print('Created init coords')

    for iter_ in tqdm.trange(num_iters, desc=f"{EXP_NAME}, thread {thread}, batch {batch_idx+1} of {len(dataloader)}"):
        v_rec, querries, coords = train_iteration(coords, emb_target, sigma=0.3, lr=LR)
        total_querries += querries

        if iter_ % test_iters == 0 or iter_ == num_iters - 1:
            arcface_sims = []
            facenet_sims = []
            for i in range(len(v0)):
                arcf = torch.cosine_similarity(embed_arcface(v_rec[i:i+1]), embed_arcface(v0[i:i+1])).item()
                fcnt = torch.cosine_similarity(embed_facenet(v_rec[i:i+1]), embed_facenet(v0[i:i+1])).item()
                arcface_sims.append(arcf)
                facenet_sims.append(fcnt)
                results[paths[i]] = results.get(paths[i], {})
                results[paths[i]][iter_] = {
                    'querries': total_querries,
                    'arcface_sim': float(arcf),
                    'facenet_sim': float(fcnt),
                    'coords': [float(x) for x in coords[i]]
                }
            print(f"[Iter {iter_}] ArcFace avg: {np.mean(arcface_sims):.4f}, FaceNet avg: {np.mean(facenet_sims):.4f}")
            with open(os.path.join(EXP_DIR, f"results_thread{thread}.json"), 'w') as f:
                json.dump(results, f)

    for img, path in zip(v_rec, paths):
        rel_path = os.path.relpath(path, start=os.path.join(ROOT, "runs", "preprocessed_data"))
        new_path = os.path.join(OUT_IMG_DIR, rel_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        imaginize(img).save(new_path)

        if mode == "pairs":
            for p in data['pairs']:
                if p[0] == path:
                    updated_json['pairs'].append([new_path, p[1], p[2]])
        elif mode == "folds":
            for fold in data['folds']:
                updated_json['folds'].append([
                    [new_path if p[0] == path else p[0], p[1], p[2]] for p in fold
                ])
        elif mode == "list":
            updated_json.setdefault('list', []).append(new_path)


if thread == 0:
    reconstructed_json = {'pairs': [], 'folds': []}
    root_runs = os.path.join(ROOT, "runs")

    if mode == "pairs":
        for p in data['pairs']:
            if p[2] == 1:
                rel_path = os.path.relpath(p[0], start=os.path.join(ROOT, "runs", "preprocessed_data"))
                new_path = os.path.join("runs", "reconstructions", EXP_NAME, "images", rel_path)
                reconstructed_json['pairs'].append([new_path, p[1], p[2]])
            else:
                reconstructed_json['pairs'].append(p)

    elif mode == "folds":
        for fold in data['folds']:
            new_fold = []
            for p in fold:
                if p[2] == 1:
                    rel_path = os.path.relpath(p[0], start=os.path.join(ROOT, "runs", "preprocessed_data"))
                    new_path = os.path.join("runs", "reconstructions", EXP_NAME, "images", rel_path)
                    new_fold.append([new_path, p[1], p[2]])
                else:
                    new_fold.append(p)
            reconstructed_json['folds'].append(new_fold)

    elif mode == "list":
        reconstructed_json = []
        for p in data:
            rel_path = os.path.relpath(p, start=os.path.join(ROOT, "runs", "preprocessed_data"))
            new_path = os.path.join("runs", "reconstructions", EXP_NAME, "images", rel_path)
            reconstructed_json.append(new_path)

    out_json_path = os.path.join(EXP_DIR, "samples.json")
    with open(out_json_path, 'w') as f:
        json.dump(reconstructed_json, f, indent=2)

    print("Global samples.json written with paths under runs/")
