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
