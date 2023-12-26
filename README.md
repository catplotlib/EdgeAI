# Edge AI with TensorFlow on CIFAR-10 Dataset

## Overview

This project demonstrates the use of TensorFlow and TensorFlow Model Optimization Toolkit to train, optimize, and evaluate a deep learning model on the CIFAR-10 dataset. The primary focus is on implementing and comparing different model optimization techniques, including quantization and pruning, for efficient deployment in resource-constrained environments like edge devices.

## Setup and Prerequisites

To run this project, you need access to Google Colab and a Google Drive account for model storage. The primary dependencies include:

- TensorFlow
- TensorFlow Datasets
- TensorFlow Model Optimization Toolkit
- NumPy

Mounting Google Drive in the Colab notebook is required for saving and loading the models.

## Data Loading and Preprocessing

The CIFAR-10 dataset, a collection of 60,000 32x32 color images in 10 classes, is loaded and preprocessed. The preprocessing steps involve resizing the images to 160x160 and normalizing pixel values.

## Model Architecture and Training

The MobileNetV2 architecture is used as the base model. Its layers are frozen to leverage transfer learning, and a global average pooling layer followed by a dense layer with softmax activation is added on top. The model is compiled and trained on the preprocessed CIFAR-10 dataset.

## Model Optimization Techniques

### Quantization

The trained model is converted to the TensorFlow Lite format with quantization. Quantization reduces the model size and is suitable for deployment on devices with limited resources.

### Pruning

The model is also optimized using pruning, a technique that systematically removes weights from the model. A pruning schedule is defined to balance the model's size and performance. The pruning process is applied to the model, and the pruned model is retrained.

## Model Evaluation

### TensorFlow Lite Model Evaluation

A TensorFlow Lite interpreter is set up to evaluate the quantized model's performance. The test dataset is processed and used to measure the model's accuracy.

### Final Evaluation and Comparison

After optimizing and evaluating the models, the following results were obtained:

- Original Model Accuracy: 81.47%
- Quantized Model Accuracy: 79.44%
- Pruned Model Accuracy: 53.81%

These results illustrate the trade-offs between model complexity and performance:

- The **original model** shows the highest accuracy, which is expected as it retains all its parameters and complexity.
- The **quantized model** demonstrates a slight decrease in accuracy. This minor reduction is a favorable outcome, considering the significant benefits in terms of reduced model size and faster inference, making it suitable for edge devices.
- The **pruned model** shows a substantial decrease in accuracy. This suggests that the pruning might have been too aggressive, leading to a loss of important features necessary for making accurate predictions. It indicates a need to fine-tune the pruning process, balancing model size reduction with performance retention.


## Execution Instructions

To execute the project, follow these steps:

1. Mount Google Drive in Google Colab.
2. Run the cells in sequence, starting from data loading and preprocessing, followed by model training, optimization, and evaluation.
3. Observe the output at each stage, especially the accuracy metrics for each model.

## Conclusion and Future Work

This project demonstrates the effectiveness of model optimization techniques in preparing models for edge deployment. Future work could explore combining pruning and quantization, experimenting with different architectures, and deploying the optimized models on actual edge devices.