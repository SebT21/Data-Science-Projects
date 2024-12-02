# Emotion Recognition with CNNs

This repository contains the code and report for a group project completed as part of the **ECE1508 Deep Learning** course during my Master's in Engineering (MEng) at the **University of Toronto**. The project focused on recognizing human emotions from grayscale images using various convolutional neural network (CNN) architectures.

## Project Objective

The project aimed to classify human emotions into seven categories: **fear, disgust, neutral, happy, angry, surprise, and sad**, leveraging a dataset of approximately 36,000 grayscale images. We implemented and compared the performance of four CNN models:

1. **AlexNet** (my main contribution)
2. **ResNet-18**
3. **VGG-16**
4. **Custom CNN**

## Features

- **Preprocessing and Augmentation**: Customized data handling to adapt to grayscale images and address class imbalance.
- **Model Evaluation**: Analyzed models using metrics like accuracy, F1 score, and feature map visualizations.
- **AlexNet Focus**: Modified the AlexNet architecture to fit the dataset's specifications, such as resizing input images and adjusting for seven-class output.

## Contributions

- **AlexNet Development**: Implemented and optimized AlexNet for emotion recognition, including architectural adjustments and hyperparameter tuning.
- **Report Writing**: Authored and edited the majority of the final report, ensuring clear communication of objectives, methods, and findings.

## Results

- **ResNet-18** achieved the highest accuracy (68.1%) among all models, but class imbalance remained a significant challenge.
- F1 scores revealed a need for better handling of underrepresented emotion categories to improve overall performance.
- Visualizations of feature maps highlighted how each model processed emotional cues differently.

## Repository Contents

- **`AlexNet.ipynb`**: Notebook for all steps used for the modified AlexNet architecture, including data processing and results.
- **`Project Report`**: The full project report documenting our methods, analyses, and conclusions.

## Acknowledgments

This project was a collaborative effort with my teammates:
- **João Atz Dick** (ResNet-18)
- **Chenhao Hong** (VGG-16)
- **Pengyu Pan** (Custom CNN)
