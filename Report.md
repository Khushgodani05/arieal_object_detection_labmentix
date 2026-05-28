# Aerial Object Classification using Deep Learning

## 
Github Repository: [Aerial Object Classification](https://github.com/Khushgodani05/arieal_object_detection_labmentix)

Github Link: https://github.com/Khushgodani05/arieal_object_detection_labmentix

## Project Overview

This project focuses on developing a Deep Learning based image classification system capable of distinguishing between **Birds** and **Drones** using aerial imagery. The primary objective of the system is to build a robust classification pipeline that can generalize effectively on unseen test data while maintaining strong precision and recall.

The project implements:

* Data preprocessing
* Data augmentation
* Custom CNN architecture
* Model training and validation
* Early stopping
* Model checkpointing
* Performance evaluation using multiple metrics
* Final testing on unseen data

---

# Dataset Workflow

## 1. Dataset Understanding

The dataset consisted of aerial images belonging to two classes:

* Bird
* Drone

The dataset was divided into:

* Training Dataset
* Validation Dataset
* Testing Dataset

The testing dataset remained completely unseen during training to ensure unbiased evaluation.

---

# Data Preprocessing

The following preprocessing steps were applied:

* Image resizing to `224 × 224`
* Pixel normalization
* Tensor conversion
* Batch loading using PyTorch DataLoader

Normalization values used:

```python
Mean = [0.55566122, 0.5696501, 0.53595315]
Std  = [0.29024144, 0.28019984, 0.31316953]
```

---

# Data Augmentation

To improve generalization performance and reduce overfitting, multiple augmentation techniques were applied:

* Random Horizontal Flip
* Random Vertical Flip
* Random Rotation
* Random Grayscale Transformation

These augmentations increased dataset diversity and improved model robustness against real-world variations.

---

# Model Architecture

A custom Convolutional Neural Network (CNN) architecture was implemented using PyTorch.

## CNN Components

The model architecture included:

* Convolutional Layers
* Batch Normalization
* Max Pooling
* Dropout Regularization
* Fully Connected Dense Layers
* Sigmoid Output Activation

The output layer performs binary classification:

* `0 → Bird`
* `1 → Drone`

---

# Training Configuration

## Optimizer

Adam Optimizer was used with:

```python
Learning Rate = 0.001
```

## Loss Function

Binary Cross Entropy Loss (BCELoss) was used for binary classification.

## Batch Size

```python
Batch Size = 32
```

## Epochs

```python
Maximum Epochs = 15
```

---

# Early Stopping and Model Checkpointing

To avoid overfitting and preserve the best-performing model:

* Early stopping was implemented
* Model checkpointing was enabled

The model was automatically saved whenever validation accuracy improved.

Training stopped automatically when no significant improvement was observed for multiple epochs.

---

# Training Progress Analysis

## Epoch-wise Performance Summary

| Epoch | Train Accuracy | Validation Accuracy | Validation F1-Score |
| ----- | -------------- | ------------------- | ------------------- |
| 1     | 66.04%         | 73.80%              | 0.6877              |
| 2     | 71.64%         | 77.16%              | 0.7293              |
| 3     | 75.69%         | 80.53%              | 0.8067              |
| 4     | 78.63%         | 83.17%              | 0.8309              |
| 5     | 79.08%         | 80.05%              | 0.8118              |
| 6     | 81.10%         | 81.49%              | 0.8197              |
| 7     | 82.08%         | 83.89%              | 0.8453              |
| 8     | 83.21%         | 83.17%              | 0.8158              |
| 9     | 84.11%         | 72.36%              | 0.7667              |
| 10    | 84.15%         | 83.41%              | 0.8305              |
| 11    | 85.65%         | 83.89%              | 0.8431              |
| 12    | 85.39%         | 83.41%              | 0.8217              |

---

# Best Model Selection

The best-performing model was selected based on validation accuracy and validation stability.

The highest stable validation performance was achieved around:

* Validation Accuracy: **83.89%**
* Validation F1-Score: **0.8453**

The model from the best validation checkpoint was saved as:

```python
best_model.pth
```

---

# Observations During Training

## Performance Improvement

The model demonstrated continuous learning across epochs:

* Training accuracy improved from **66.04% → 85.39%**
* Validation accuracy improved from **73.80% → 83.89%**

This indicates successful feature learning and improved classification capability.

---

## Generalization Capability

The validation performance remained relatively stable across epochs, indicating:

* Good generalization
* Reduced overfitting
* Effective augmentation strategy

---

## Overfitting Observation

At Epoch 9:

* Validation accuracy dropped significantly to **72.36%**
* Validation loss increased sharply

This indicated temporary overfitting or unstable learning behavior.

However, subsequent epochs recovered performance successfully.

---

# Final Testing on Unseen Data

After training completion, the best saved model was evaluated on completely unseen test data.

This evaluation represents the real-world performance of the system.

---

# Test Dataset Results

## Final Test Accuracy

```python
87.50%
```

## Precision

```python
0.7582
```

## Recall

```python
0.9718
```

## F1-Score

```python
0.8519
```

---

# Test Confusion Matrix

```python
[[99 22]
 [ 2 69]]
```

---

# Interpretation of Confusion Matrix

| Actual Class | Predicted Correctly | Misclassified |
| ------------ | ------------------- | ------------- |
| Bird         | 99                  | 22            |
| Drone        | 69                  | 2             |

The model demonstrated:

* Very high drone detection capability
* Extremely low false negatives for drones
* Strong recall performance

---

# Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| Bird  | 0.98      | 0.82   | 0.89     |
| Drone | 0.76      | 0.97   | 0.85     |

---

# Result Analysis

## Strengths of the Model

### 1. Excellent Recall

The recall score of:

```python
0.9718
```

indicates that the model successfully identifies almost all drone instances.

This is particularly important in surveillance and aerial monitoring systems where missing a drone detection could be critical.

---

## 2. Strong Generalization

The model achieved:

```python
87.50% Test Accuracy
```

on completely unseen data, showing good real-world generalization.

---

## 3. Stable F1-Score

The F1-score of:

```python
0.8519
```

demonstrates a strong balance between precision and recall.

---


# Conclusion

The developed aerial object classification system successfully demonstrates the effectiveness of Deep Learning for distinguishing between birds and drones.

Key achievements include:

* Successful custom CNN implementation
* Strong unseen test accuracy
* High drone recall performance
* Effective early stopping and checkpointing
* Robust evaluation pipeline

The model achieved:

```python
87.50% Test Accuracy
```

with strong recall and F1-score, making it a promising solution for aerial surveillance and object monitoring applications.
