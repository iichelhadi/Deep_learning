# Deep_learning

This repository contains a collection of deep learning projects that explore various machine learning models and techniques. The current projects focus on sentiment analysis using Hugging Face's Transformers and image classification tasks.

## Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [Sentiment Analysis with Hugging Face](#sentiment-analysis-with-hugging-face)
  - [Image Classifier](#image-classifier)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Overview

This repository is structured into several independent deep learning projects. Each project is self-contained with its own set of scripts, models, and datasets. The current projects are:

1. **Sentiment Analysis with Hugging Face**: A sentiment analysis model fine-tuned on text data using a pre-trained BERT model.
2. **Image Classifier**: Projects focusing on classifying images from different datasets, including CIFAR-10 and sc-islet images.

## Projects

### Sentiment Analysis with Hugging Face

- **Folder**: `Sentiment_analysis_huggingface`
- **Description**: This project uses Hugging Face's Transformers to fine-tune a BERT model for sentiment analysis, classifying text data into positive, negative, or neutral sentiments.
- **Key Files**: 
  - `huggingface_sentiment_analysis.ipynb`: The main notebook containing all the code for the sentiment analysis task.

### Image Classifier

This section contains two sub-projects:

1. **CIFAR-10 Image Classifier**
   - **Folder**: `Image_classifier/Cifar10`
   - **Description**: A convolutional neural network (CNN) model for classifying images in the CIFAR-10 dataset.
   - **Key Files**: 
     - `Torch_image_classification.ipynb`: The main notebook containing the code for CIFAR-10 image classification.

2. **sc-islet Functionality Classifier**
   - **Folder**: `Image_classifier/sc-islet_classifier`
   - **Description**: A model that classifies iPSC-derived sc-islets into functional or non-functional categories based on brightfield images using a Transformer-based model.
   - **Key Files**: 
     - `transformer_classifier.ipynb`: The main notebook containing the code for training the model.
     - `streamlit_app.py`: A Streamlit app for predicting sc-islet functionality from uploaded images.

## Installation

### General Requirements

- Python 3.x
- Additional dependencies as listed in the respective project folders.

## Usage
Each project has its own usage instructions, detailed in the respective README.md files located in each project's folder. Generally, you'll need to:
    Clone the repository:
```bash
git clone https://github.com/iichelhadi/Deep_learning.git
cd Deep_learning
```
Navigate to the specific project folder you are interested in and follow the instructions in the README.md of that folder.

You can install the general requirements using `pip`:

```bash
pip install torch torchvision transformers pandas numpy matplotlib pillow streamlit notebook
```

### Contact
For any questions or suggestions, feel free to reach out:

Name: Elhadi Iich
Email: e.iich@hotmail.nl
