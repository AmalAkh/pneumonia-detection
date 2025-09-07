# Pneumonia Detection with CNN on Chest X-Ray Images

This is a small deep learning project that demonstrates how a **Convolutional Neural Network (CNN)** can be trained to detect **pneumonia** from chest X-ray images. The model is built using **PyTorch** and trained on the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## 📂 Project Structure

* train.ipynb - model, code, and so on in a notebook
* best-model - best model
* readme.md - readme
---

## 🧠 Model & Training

- Framework: **PyTorch**
- Architecture: Small **CNN** designed for image classification
- Dataset: Chest X-Ray Pneumonia (train/val/test split provided by Kaggle)
- Optimizer & Loss: Standard training setup with cross-entropy loss

---

## 📊 Results

- **Test Accuracy:** 88.94%  
- **Test Loss:** 0.2912  

These results show the model can reasonably distinguish between normal and pneumonia cases from X-rays.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/pneumonia-cnn.git
   cd pneumonia-cnn
2. Open the notebook and train model
    ```
    jupyter notebook train.ipynb
    ```