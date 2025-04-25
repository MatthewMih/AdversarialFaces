import glob, tqdm, torch, math
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
mtcnn = MTCNN()
torch.set_grad_enabled(False) 

device = 'cuda'
images = glob.glob('YOUR_PATH_TO_FFHQ/thumbnails128x128/*.*')


HW = 112
def vectorize(img):
    if img.size != (HW, HW):
        x = img.resize((HW, HW))
    else:
        x = img
    x = np.array(x).astype(np.float32) / 255
    x = torch.tensor(x).permute(2, 0, 1).reshape(3*HW*HW)
    return x

def imaginize(x):
    img = x.reshape(3, HW, HW).permute(1, 2, 0).cpu().detach().numpy() * 255
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img

HW = 112
def get_square_face(image, mtcnn, resize=HW):
    detection = mtcnn.detect(image)
    if detection[0] is None:
        return None
    x1, y1, x2, y2 = detection[0][0]
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    hw = max(x2 - x1, y2 - y1)

    x1 = max(int(x_c - hw / 2), 0)
    x2 = min(int(x_c + hw / 2), image.size[0])

    y1 = max(int(y_c - hw / 2), 0)
    y2 = min(int(y_c + hw / 2), image.size[1])

    crop = Image.fromarray(np.array(image)[y1:y2, x1:x2])
    if resize is not None:
        crop = crop.resize((resize, resize)) 
    
    return crop

vectors = []
for img in tqdm.tqdm(images):
    crop = get_square_face(Image.open(img), mtcnn, HW)
    if crop is not None:
        vectors.append(vectorize(crop))
vectors = torch.stack(vectors)

torch.save(vectors, 'mtcnn_crops_ffhq_square.pt')

avg_vect = vectors.mean(dim=0)
torch.save(avg_vect, 'mtcnn_crops_ffhq_average_vector_square.pt')

U, S ,V = torch.pca_lowrank((vectors - avg_vect[None, ...]).to(device), q=1024, center=False)

torch.save(
    {
        'U': U.cpu().detach(),
        'S': S.cpu().detach(),
        'V': V.cpu().detach()
    },
    'pca_1024_USV_square_ffhq.pt'
)

basis_norms = S / math.sqrt(U.shape[0] - 1)
eigenvectors_denormalized = V.T * basis_norms[..., None]
torch.save(eigenvectors_denormalized.cpu().detach(), 'eigenvectors_denormalized_Square_ffhq.pt')