# Image Classification with CNN on CIFAR-10 Dataset

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Results](#results)

---

## Project Overview
The goal of this project is to classify images from the CIFAR-10 dataset into one of 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

We use a **Convolutional Neural Network (CNN)** implemented in TensorFlow and Keras to achieve this task. The model is trained on 50,000 images and evaluated on 10,000 test images.

---

## Dataset
The CIFAR-10 dataset is loaded using TensorFlow's `datasets.cifar10.load_data()` function. The dataset is split into:
- **Training data**: 50,000 images
- **Test data**: 10,000 images

Each image is a 32x32 RGB image, and the pixel values are normalized to the range `[0, 1]`.

---

## Model Architecture
The CNN model consists of the following layers:
1. **Convolutional Layers**:
   - Conv2D (32 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)
   - Conv2D (64 filters, 3x3 kernel, ReLU activation)
   - MaxPooling2D (2x2 pool size)
   - Conv2D (64 filters, 3x3 kernel, ReLU activation)

2. **Fully Connected Layers**:
   - Flatten
   - Dense (64 units, ReLU activation)
   - Dense (10 units, softmax activation for classification)

The model is compiled using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss function.

---

## Results
The model achieves the following performance:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~75%
- **Test Accuracy**: ~73%
