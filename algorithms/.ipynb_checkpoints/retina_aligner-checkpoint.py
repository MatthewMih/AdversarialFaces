import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


crop_size = 112  # for ArcFace
face_analysis = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_analysis.prepare(ctx_id=0, det_size=(640, 640))


def align_face(img, landmarks, output_size=112):
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    dst = np.array(landmarks, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, M, (output_size, output_size), borderValue=0.0)

def center_crop_resize(img, size):
    h, w = img.shape[:2]
    if h > w:
        top = (h - w) // 2
        img = img[top:top + w, :]
    elif w > h:
        left = (w - h) // 2
        img = img[:, left:left + h]
    return cv2.resize(img, (size, size))

def get_aligned_face(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    h, w, c = img.shape
    if h != w:
        size = max(h, w)
        padded_img = np.zeros((size, size, c), dtype=np.uint8)
        top = (size - h) // 2
        left = (size - w) // 2
        padded_img[top:top + h, left:left + w] = img
        img = padded_img

    img = np.array(Image.fromarray(img).resize((250, 250)))

    faces = face_analysis.get(img, max_num=1)
    if not faces:
        img = cv2.copyMakeBorder(img, img.shape[0] // 2, img.shape[0] // 2,
                                 img.shape[1] // 2, img.shape[1] // 2,
                                 cv2.BORDER_CONSTANT, value=0)
        faces = face_analysis.get(img, max_num=1)

    if faces:
        kps = faces[0].kps
        if kps is not None and kps.shape == (5, 2) and np.isfinite(kps).all():
            aligned = align_face(img, kps, output_size=crop_size)
        else:
            aligned = center_crop_resize(img, crop_size)
    else:
        aligned = center_crop_resize(img, crop_size)

    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(aligned_rgb)
