# 🐱 Cats vs Dogs Image Classifier 🐶

A deep learning project that classifies images as **cats** or **dogs** using a **Convolutional Neural Network (CNN)** built with **Keras**. The model is trained in a Jupyter notebook and deployed as a web app using **Streamlit** on **Hugging Face Spaces**.

---

## 🚀 Live Demo

👉 [**Click here to try the app**](https://huggingface.co/spaces/dsharma08k/cats-vs-dogs-classifier)  
Upload, paste, or link an image and get instant predictions!

---

## 📁 Files in This Repository

| File | Description |
|------|-------------|
| `Cats_vs_Dogs_Classification.ipynb` | Jupyter notebook for training the CNN model |
| `app.py` | Streamlit app script used for deployment |
| `requirements.txt` | Python dependencies for the app |
| `README.md` | Project overview and usage instructions |

> 🔒 Note: The trained model file (`best_model.keras`) is **not uploaded here** to save space. It is hosted within the Hugging Face Space.

---

## 🧠 Model Overview

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input Shape:** 256 x 256 x 3
- **Layers:**
  - Multiple `Conv2D + MaxPooling2D` layers
  - Flatten + Dense layers
  - Sigmoid output for binary classification
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

## 📊 Model Performance

| Metric        | Score        |
|---------------|--------------|
| **Validation Accuracy** | ~85% |
| **Precision (macro)**   | 85%  |
| **Recall (macro)**      | 84%  |
| **F1 Score (macro)**    | 84%  |
| **Confusion Matrix**    | Included in the notebook |

---

## 🧪 Dataset

- **Name:** Dogs vs Cats
- **Source:** [Kaggle: Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Classes:** Binary — 0 = Cat, 1 = Dog
- **Used Images:** Subsampled ~25,000 images

---

## 🎯 Features of the Web App

- ✅ Upload image from your device (drag & drop / browse / paste)
- ✅ Paste an image URL
- ✅ Real-time prediction (Cat 🐱 or Dog 🐶)
- ✅ Confidence score shown via progress bar
- ✅ Clean and responsive Streamlit UI

---

## 🛠 Built With

- Python
- TensorFlow / Keras
- NumPy, Pillow
- Streamlit
- Hugging Face Spaces

---

## 🧾 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/cats-vs-dogs-classifier
   cd cats-vs-dogs-classifier
2. Install dependencies:
    pip install -r requirements.txt
   
3. Run the app:
    streamlit run app.py
   
⚠️ Make sure to place best_model.keras in the same directory if you run it locally.

