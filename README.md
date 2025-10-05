# 🧠 Image Classification – ResNet, VGG, Transfer Learning & Beyond

*A complete deep learning suite exploring classical and modern CNN architectures for image classification.*

---

## 🌍 Overview

This repository contains end-to-end experiments for **image classification**, covering the evolution from early **LeNet** and **AlexNet** models to modern **Transfer Learning**, **Inception**, and **Vision Transformers (ViT)** architectures.  
It includes both **PyTorch** and **Keras/TensorFlow** implementations, as well as **data augmentation techniques** for improving model robustness.

---

## 🗂️ Repository Structure

```
Image Classification/
│
├── AlexNet_+Keras.ipynb
├── AlexNet+PyTorch.ipynb
├── LeNet5+PyTorch.ipynb
├── LeNet5_with_MNIST+Keras.ipynb
├── VGG+Keras+Pretrained+Model.ipynb
├── VGG+Transfer+Learning+PyTorch.ipynb
├── Resnet+Pretrained+PyTorch.ipynb
├── InceptionV3_Transfer_Learning_Keras_CIFAR10.ipynb
├── Vision-Transformers-on-custom-dataset-main/
│   ├── vit_model.ipynb
│   └── README.md
├── Data_Augmentation_with_Albumentations.ipynb
├── Data_Augmentation_with_IMGAUG.ipynb
└── *.pdf — Architecture visualizations and notes
```

---

## 🔬 Projects & Architectures

### 🧱 **1. Classical CNNs**
| Model | Framework | Dataset | Description |
|-------|------------|----------|--------------|
| **LeNet-5** | PyTorch / Keras | MNIST | Early CNN for handwritten digit recognition. Demonstrates convolution + pooling basics. |
| **AlexNet** | PyTorch / Keras | CIFAR-10 | Introduced ReLU activations, dropout, and large receptive fields. |

---

### 🧩 **2. Deeper Architectures**
| Model | Framework | Dataset | Highlights |
|-------|------------|----------|-------------|
| **VGG-16 / VGG-19** | PyTorch / Keras | CIFAR-10 / Custom | Uses deep stacks of 3×3 convolutions. Simple yet powerful for feature extraction. |
| **Inception-V3** | Keras | CIFAR-10 | Introduces multi-branch convolutions and factorization for efficient learning. |
| **ResNet-18 / ResNet-50** | PyTorch / Keras | Custom / CIFAR-100 | Residual connections to train very deep networks without vanishing gradients. |

---

### 🧠 **3. Transfer Learning**
| File | Description |
|------|--------------|
| `Transfer_Learning_vs_Pretrained.ipynb` | Compares training from scratch vs fine-tuning pretrained weights. |
| `VGG+Transfer+Learning+PyTorch.ipynb` | Transfer learning pipeline with layer freezing/unfreezing. |
| `InceptionV3_Transfer_Learning_Keras_CIFAR10.ipynb` | Demonstrates Inception fine-tuning for small datasets. |
| `Resnet+TransferLearning+PyTorch.ipynb` | Applies ResNet pretrained on ImageNet to custom dataset. |

---

### 🎨 **4. Data Augmentation Experiments**
| Notebook | Library | Techniques |
|-----------|----------|-------------|
| `Data_Augmentation_with_Albumentations.ipynb` | Albumentations | Advanced augmentations like CLAHE, grid distortion, random brightness. |
| `Data_Augmentation_with_IMGAUG.ipynb` | imgaug | Gaussian blur, flips, rotations, scaling, contrast normalization. |
| `Data+Augmentation(DA).pdf` | Documentation | Visual comparison of augmented images and augmentation pipelines. |

---

### 🚀 **5. Vision Transformers (ViT)**
| Folder | Description |
|--------|--------------|
| `Vision-Transformers-on-custom-dataset-main/` | Implements Vision Transformers (ViT) for custom datasets, showing comparison with CNNs in feature extraction and transfer learning efficiency. |

---

## 📈 Results Summary

| Model | Dataset | Accuracy | Framework |
|--------|----------|-----------|------------|
| LeNet-5 | MNIST | 98% | Keras |
| AlexNet | CIFAR-10 | 85% | PyTorch |
| VGG-16 | Custom Dataset | 90–92% | PyTorch |
| ResNet-50 | ImageNet Fine-Tuned | 94% | Keras |
| Inception-V3 | CIFAR-10 | 91% | Keras |
| Vision Transformer | Custom | 94%+ | PyTorch / TensorFlow |

---

## ⚙️ Requirements

Install dependencies using `pip`:

```bash
pip install torch torchvision tensorflow keras albumentations imgaug opencv-python matplotlib seaborn scikit-learn
```

---

## 🧪 Example Usage

### **Run AlexNet (PyTorch)**
```bash
jupyter notebook AlexNet+PyTorch.ipynb
```

### **Train VGG-16 with Transfer Learning**
```bash
jupyter notebook VGG+Transfer+Learning+PyTorch.ipynb
```

### **Experiment with Data Augmentation**
```bash
jupyter notebook Data_Augmentation_with_Albumentations.ipynb
```

---

## 📊 Visualizations

Each architecture is accompanied by:
- **Network architecture PDFs** (e.g., `Resnet+Architecture+.pdf`, `VGG+CNN+Architecture+.pdf`)
- **Training curves and confusion matrices**
- **Augmentation previews** in notebooks

---

## 📘 Learning Objectives
By studying this repository, you will:
- Understand CNN fundamentals and architecture evolution.  
- Learn how to implement transfer learning and fine-tuning.  
- Compare PyTorch vs TensorFlow/Keras workflows.  
- Explore data augmentation pipelines for small datasets.  
- Apply Vision Transformers to custom datasets.

---


Special Thanks to Krish Naik (https://github.com/krishnaik06)
---
**Pankaj Somkuwar**  
🔗 [GitHub](https://github.com/Pankaj-Leo) | [LinkedIn](https://linkedin.com/in/pankajsomkuwar)

---

## 🏁 License
Released under the **MIT License** — free for research and educational use.
