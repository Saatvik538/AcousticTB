# AcousticTB: TB Screening from Cough Analysis

An end-to-end machine learning pipeline for tuberculosis screening using cough audio analysis, designed for resource-limited clinical settings.

## Overview

This project implements a machine learning pipeline for TB screening using:
- **CNN**: Mel-spectrogram feature extraction with focal loss
- **XGBoost**: Ensemble learning on embeddings + clinical data
- **Data Augmentation**: ESC-50 environmental noise for robustness
- **Imbalanced Learning**: Handles 30% TB+, 70% TB- distribution

  Mel-Spectrograms → CNN Encoder → Embeddings (256D) 
                                      ↓
Clinical Features ────────────→ XGBoost Classifier
                                      ↓
                              Logistic Regression
                                   (Stacker)
                                      ↓
                              Final TB Prediction

## Performance

- **ROC-AUC:** 0.821 (Excellent discrimination)  
- **Specificity:** 78.8% (Surpasses WHO standards for TB screening tool)  
- **NPV:** 85.4% (Negative predictive value)  
- **Accuracy:** 75.2% (Overall performance)  

## Quick Start

1. **Clone repository**:
  \\\ash
  git clone https://github.com/Saatvik538/AcousticTB.git
  cd AcousticTB
  \\\

2. **Install dependencies**:
  \\\ash
  pip install tensorflow scikit-learn xgboost librosa matplotlib seaborn tqdm pandas numpy
  \\\

3. **Run data augmentation**:
  \\\python
  %run src/data_augmentation.py
  \\\

4. **Execute pipeline**: Open and run \AcousticTB.ipynb\

## Project Structure

\\\
AcousticTB/
├── AcousticTB.ipynb              # Main pipeline notebook
├── src/
│   ├── data_augmentation.py      # ESC-50 + Gaussian noise augmentation
│   └── utils.py                  # Helper functions
├── data/                         # TB cough dataset and processed features
│   ├── raw/solicited_coughs/
│   └── processed/
└── ESC-50/                       # Environmental sound dataset
   ├── audio/
   └── meta/
\\\

## Technical Approach

### Deep Learning Architecture 
CNN Encoder (256 parameters: 1.2M)
- Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.2)
- Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.3)  
- Conv2D(256) → BatchNorm → Conv2D(256) → MaxPool → Dropout(0.4)
- GlobalAvgPool → Dense(512) → Dense(256) → Dense(1, sigmoid)

### Focal Loss
- focal_loss = -α * (1-p_t)^γ * log(p_t)
- γ = 2.0: Strong focus on hard examples
- α = 0.25: Moderate class balancing
- Optimized for AUROC in a medical context

### Data Augmentation Strategy
- **Clean cough sounds**: Original recordings
- **Environmental noise**: ESC-50 real-world sounds (rain, traffic, etc.)
- **Gaussian noise**: Statistical backup augmentation
- **SNR=20dB**: Realistic noise levels

### Ensemble Architecture
1. **CNN**: Extracts 256-dim embeddings from mel-spectrograms
2. **XGBoost**: Combines embeddings + clinical features
3. **Logistic Regression**: Final stacking layer

### Parameters
XGBoost:
learning_rate: 0.094
n_estimators: 673
max_depth: 6
subsample: 0.842
colsample_bytree: 0.789
reg_alpha: 0.067
reg_lambda: 0.186
scale_pos_weight: 5.23

### LogReg: 
solver: lbfgs
max_iterations: 2000
regularization: L2 (C=1.0)
class_weight: balanced

## Medical Application

Designed for resource-limited clinical settings where:
- TB screening is critical, but diagnostic tools are limited
- High sensitivity required (avoid missing TB+ cases)
- Acceptable specificity to minimize false alarms
- Robust to environmental noise conditions

## Results Analysis

The model includes a comprehensive evaluation:
- **ROC/PR curves**: Performance visualization
- **Confusion matrices**: Clinical decision analysis  
- **Feature importance**: Interpretability for medical use
- **Noise resilience**: Performance across environments

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- XGBoost 1.6+
- Librosa 0.9+
- Scikit-learn 1.0+

## Dataset

- **TB Cough Data**: CODA TB Challenge dataset (restricted access)
- **Environmental Sounds**: ESC-50 (publicly available)
- **Processing**: Mel-spectrograms (64x64) at 22kHz

## Key Innovation

Advanced ensemble architecture combining deep learning and gradient boosting with **focal loss optimization** specifically tuned for medical screening scenarios where missing positive TB cases have significantly higher clinical cost than false positives.



