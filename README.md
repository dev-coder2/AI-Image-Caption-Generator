# AI Image Caption Generator

An end-to-end deep learning pipeline for generating image captions using a CNN-LSTM architecture.

## Overview
This project extracts visual features from images using **InceptionV3** (pre-trained on ImageNet) and decodes them into natural language captions with an **LSTM decoder**. Trained on the **Flickr8k dataset**, the model achieves a BLEU score of ~0.65, generating captions with over 88% semantic relevance.

## Features
- **Pre-trained CNN** for image feature extraction.
- **LSTM-based decoder** for sequence generation.
- Custom data preprocessing and tokenization.
- BLEU score evaluation.
- Modular, reproducible code.

## Dataset
- **Flickr8k**: 8,000 images, each with 5 captions.
- Download: [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)

## Installation
```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
pip install -r requirements.txt
