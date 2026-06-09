# Palm Oil Ripeness Classification (Mobile Deployment)

This repository contains the Machine Learning pipeline for an automated Palm Oil Ripeness Classification system. The goal of this project is to build a lightweight Convolutional Neural Network (CNN) capable of running entirely offline on edge devices (smartphones) to assist farmers in determining harvest readiness.

## Overview
The model classifies palm oil fresh fruit bunches into four distinct ripeness stages:
1. **Unripe** (Green)
2. **Underripe** (Yellow/Orange)
3. **Ripe** (Reddish-Orange)
4. **Overripe** (Dark Red/Decaying)

## Architecture
The core architecture uses **MobileNetV2** configured for Transfer Learning. MobileNetV2 was specifically chosen because of its highly optimized parameter count, ensuring fast inference times and small file sizes suitable for mobile applications without requiring an internet connection.

### Training Strategy
*   **Stage 1 (Feature Extraction):** MobileNetV2 base is frozen. Only the custom 4-class dense classification head is trained.
*   **Stage 2 (Fine-Tuning):** The final 30 layers of MobileNetV2 are unfrozen and trained with a very low learning rate (`1e-5`) to adapt the pretrained ImageNet features to palm oil specific textures.

## Project Structure
This repository contains the main notebook `palm_oil_ripeness.ipynb` which trains the production model exported for the mobile app.
*   **Accuracy:** ~64.5% (refer to the `results/` folder for full metrics)
*   **Key Feature:** Uses **Label Smoothing (`0.15`)**. Because the visual boundary between "Ripe" and "Underripe" is highly subjective, Label Smoothing intentionally makes the model cautious on borderline cases. 
*   **Result:** This model is exceptionally stable and virtually immune to real-world lighting variations (tested under +/- 30% artificial brightness).

## Results
The `results/` directory contains detailed evaluation metrics for the finetuned model, including:
*   `classification_report_finetuned.txt`
*   `confusion_matrix_finetuned.png`
*   `metrics_finetuned.csv`

## Mobile Export
The final trained production models are exported via TensorFlow Lite. While the code generates these in the `exports/` folder, they are ignored via `.gitignore` to prevent exceeding GitHub file size limits. The files generated include:
*   `mobilenetv2_palm_ripeness_fp32.tflite` (Standard 32-bit float model)
*   `mobilenetv2_palm_ripeness_fp16.tflite` (Quantized 16-bit float model for smaller file size)
*   `labels.txt` (Class label mapping for Android/iOS integration)

*Note: The `dataset/`, `exports/`, and raw `models/` checkpoints are ignored via `.gitignore`.*