# Enhancing Text Classification: A Comparative Study of CNN, SVM, and RNN Models

This repository contains materials from a group project completed for the **ECE1513 Introduction to Machine Learning** course during my Master's in Engineering (MEng) at the **University of Toronto**. The project focused on exploring and comparing the effectiveness of different machine learning models for topical text classification.

## Project Objective

The aim of this project was to evaluate the performance of three machine learning models—**Support Vector Machines (SVMs)**, **Convolutional Neural Networks (CNNs)**, and **Recurrent Neural Networks (RNNs)**—on textual datasets with varying characteristics. The analysis focused on aspects such as:

- Accuracy
- Computational efficiency
- Adaptability to high-dimensional data

## Features

- **SVM Implementation**: Leveraged kernel functions and hyperparameter tuning (C, gamma) to optimize text classification performance.
- **TextCNN Implementation**: Adapted convolutional neural networks to capture local patterns within text for sentence-level classification.
- **BiLSTM Implementation**: Utilized bidirectional LSTMs to capture both forward and backward context in text sequences.

## Contributions

- **SVM Development**: Designed and implemented the SVM framework, including preprocessing with TF-IDF vectorization, hyperparameter tuning, and model evaluation.
- **Report Writing**: Authored significant portions of the report, ensuring the presentation of methodologies, results, and conclusions was clear and well-structured.
- **Analysis**: Conducted detailed performance evaluations and comparisons across datasets, preprocessing techniques, and models.

## Results

- **SVMs** achieved the highest accuracy across all datasets, with their strength in handling high-dimensional feature spaces.
- **TextCNNs** performed well for capturing local textual patterns but were less effective than SVMs in general scenarios.
- **BiLSTMs** excelled in processing sequential data but required significant computational resources and were prone to overfitting without proper tuning.

### Datasets Used

- **Banking77**: Online banking queries labeled with 77 domains.
- **10 Newsgroups**: A collection of news articles spanning 10 topics.
- **AG News**: A large dataset of news articles categorized into 4 labels.

### Preprocessing Techniques

1. Tokenization, case lowering, and removing non-alphabet characters.
2. TF-IDF vectorization for SVMs and word indexing for neural networks.

## Repository Contents

- **`code/`**: Python scripts implementing SVM, TextCNN, and BiLSTM models, along with preprocessing and evaluation.
- **`report/`**: The comprehensive project report documenting the methodologies, results, and conclusions.

## Acknowledgments

This project was a collaborative effort with my teammates:
- **Chen Wang**: TextCNN implementation
- **Pas Panthasen**: BiLSTM implementation
- **Yan Pan Chung**: Analysis of preprocessing and datasets
