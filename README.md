### 🔧 How to Use

#### 📁 Datasets

- **LFW**  
  Download [`lfw.tgz`](https://www.kaggle.com/datasets/ashfaqsyed/labelled-faces-in-the-wild) and place it into the `data/` directory.

- **AgeDB-30**  
  No manual setup required — the script will automatically download and extract it.

- **CFP-FP**  
  Download [`cfp-dataset.zip`](http://www.cfpw.io/) and place it into the `data/` directory.

---

#### 🧠 Models

- **ArcFace Backbone**  
  Download `ms1mv3_arcface_r100_fp16_backbone.pth` and place it into the `models/` directory.

- **Eigenfaces & Average Face (for reconstruction)**  
  Download the following files and place them into the `models/` directory:
  
  - [`eigenvectors_denormalized_Square_ffhq.pt`](https://drive.google.com/file/d/1h7F0_iZl7R9Uh5MnKDqvfE3oF8Loh5Rg/view?usp=sharing)
  - [`mtcnn_crops_ffhq_average_vector_square.pt`](https://drive.google.com/file/d/1uv3ZxsVWeCOObjYh86to_CqOVbiwjlef/view?usp=sharing)
 

#### 🚀 Running the Full Pipeline

1. **Step 1: Prepare datasets and evaluate on original images**
   ```bash
   bash scripts/download_preprocess_eval_on_original_sets.sh
   ```
   - Unpacks datasets
   - Aligns all faces (MTCNN / RetinaFace)
   - Runs baselines with ArcFace and FaceNet on original (real) images

2. **Step 2: Reconstruct positive 1st faces in positive pairs with our method**
   ```bash
   bash scripts/reconstruct_faces.sh
   ```
   - It takes ~30 hours on 4×A100 GPUs to run
   - Runs our zero-order optimization algorithm
   - Restores the first face in all positive verification pairs
   - Supports ArcFace-based and FaceNet-based reconstructions

3. **Step 3: Evaluate reconstructed faces**
   ```bash
   bash scripts/run_eval_on_reconstructed.sh
   ```
   - Aligns reconstructed images (MTCNN / RetinaFace)
   - Runs ArcFace and FaceNet verification on restored faces

---

📝 **Results** are saved into:
```
runs/results.txt
```


### ⚙️ Optional: Generate Your Own PCA Basis

If you prefer to generate your own PCA eigenfaces instead of using the precomputed ones, run the following script:

```bash
python algorithms/generate_pca_basis.py
```

You need to specify the path to a folder with face images. We used the [FFHQ thumbnails dataset](https://github.com/NVlabs/ffhq-dataset) (`thumbnails128x128` version).

The script will save:
- A mean face vector `mtcnn_crops_ffhq_average_vector_square.pt`
- A PCA projection matrix `eigenvectors_denormalized_Square_ffhq.pt` (eigenfaces) 

Make sure to copy the generated files to the `models/` folder so that reconstruction scripts can find them.