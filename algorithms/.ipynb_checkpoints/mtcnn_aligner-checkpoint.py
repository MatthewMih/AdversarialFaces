import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=160, margin=0, selection_method='center_weighted_size', device='cuda' if torch.cuda.is_available() else 'cpu')

def get_aligned_face(path):
    """
    Открывает изображение, выравнивает через MTCNN, возвращает PIL.Image.
    В случае ошибки — паддит, в крайнем случае — возвращает resize.
    """
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)  # (H, W, 3)

    aligned = mtcnn(img)
    if aligned is None:
        print(f"[!] Face not found initially: {path}")
        
        h, w, _ = img_np.shape
        pad_h = h // 2
        pad_w = w // 2

        padded_np = np.zeros((2*h, 2*w, 3), dtype=img_np.dtype)
        padded_np[pad_h:pad_h + h, pad_w:pad_w + w] = img_np
        padded_img = Image.fromarray(padded_np)

        aligned = mtcnn(padded_img)
        if aligned is None:
            print(f"[!!] Even with full padding, face not found: {path}")
            return img.resize((160, 160))  # fallback

    aligned = (aligned + 1) / 2  # [-1, 1] → [0, 1]
    img_arr = (aligned.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)
