# Bird Species Classification Using CNN

This repository contains the source code and report files for **Project 1: CNN-Based Image Classification** in Introduction to AI Programming.

## Project Goal
The goal is to classify images into 200 bird species using CNN-based image classification models. The project compares a custom baseline CNN, ResNet-based transfer learning, and EfficientNetV2-S.

## Dataset
- Dataset: CUB-200-2011 style bird species dataset
- Classes: 200 bird species
- Train: 5,994 images
- Validation: 2,897 images
- Test: 2,897 images
- The code uses bounding-box cropping before resizing images.

## Main Models
1. Custom baseline CNN
2. ResNet-18 / ResNet-50 transfer-learning experiments
3. EfficientNetV2-S final model

## Final Model Setting
- Architecture: EfficientNetV2-S
- Input size: 384 x 384
- Optimizer: Adam
- Learning rate: 0.00001
- Batch size: 32
- Epochs: 50
- Dropout: 0.5 in the uploaded final code; final report discusses the selected dropout 0.3 setting from tuning
- Weight decay: 0.0001
- Label smoothing: 0.1
- Scheduler: CosineAnnealingLR with eta_min = 0.00001
- Random seed: 42

## Reproducibility
The code fixes the random seed for Python random, NumPy, PyTorch, CUDA, and cuDNN. The validation/test split also uses `random_state=42`.

## How to Run Training and Evaluation
```bash
python final_train_eval_with_cases.py --batch_size 32 --lr 0.00001 --epochs 50 --optimizer adam
```

## Output Files
The code saves:
- `best_cub_model.pth`
- `bird2-bird_aug_EffNetV2.pth`
- `bird2-cub_aug_EffNetV2_learning_curve.png`
- `bird2-cub_aug_EffNetV2_confusion_matrix.png`
- `bird2-cub_aug_EffNetV2_report.txt`
- `correct_cases.png`
- `failure_cases.png`
- `top5_best_recall.csv`
- `top5_worst_recall.csv`
- `top10_confusion_pairs.csv`

## Inference Demo
The project presentation shows a Streamlit demo. The demo should load the trained checkpoint, preprocess an uploaded bird image, and display the predicted class, confidence score, and top-3 candidate classes.

## Final Submission Package
Submit the following four items:
1. Source code
2. Final report PDF
3. Presentation slides
4. GitHub repository link with this README and code

## GitHub Link
Replace this placeholder with the final repository URL:

`https://github.com/chaemin040508/bird-species-classification-cnn`
