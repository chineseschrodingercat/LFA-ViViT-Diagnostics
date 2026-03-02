# AI-Enhanced Kinetic Lateral Flow Assay (LFA) Platform

## Overview
This repository hosts the deep learning framework for a next-generation **Digital Point-of-Care Testing (POCT)** system. Unlike traditional LFA readers that rely on static end-point imaging, this system utilizes a **Video Vision Transformer (ViViT)** to analyze the full kinetic reaction profile of the assay.

By capturing temporal features of the fluid dynamics, our model achieves superior sensitivity and quantification precision compared to standard colorimetric methods, specifically optimized for low-cost hardware (Raspberry Pi integration).

## Key Features
* **Temporal Analysis**: Processes 32-frame video sequences to capture reaction kinetics using ViViT (B-16x2).
* **Robust Data Loading**: Custom `VideoClassificationDataset` handling temporal sampling, prefix alignment, and corrupted frame fallback.
* **High Precision**: Achieved **100% Accuracy** (on validation set) for distinguishing critical concentration cutoffs.
* **Hardware Optimized**: Designed to work with low-resolution inputs from embedded camera modules (Raspberry Pi HQ Camera).

## Project Structure
* `dataset.py`: Custom PyTorch Dataset class with OpenCV-based video processing and temporal sampling logic.
* `train.py`: Training pipeline using Hugging Face Transformers and PyTorch Lightning-style loops.
* `inference.py`: Deployment script for batch inference on new samples.

## Contributors
* **Minhao Liu** (Project Lead): Experimental design, hardware integration (Raspberry Pi/Microfluidics), T/C ratio algorithm, and system validation.
* **Pu Sun** (Model Architect): Implementation of the ViViT architecture, hyperparameter optimization, and training pipeline deployment.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
To train the model on a new dataset:
```bash
python train.py
```

To run inference using trained weights:
```bash
python inference.py --weights checkpoints/best_model.pth --test_dir ./data/test
```
