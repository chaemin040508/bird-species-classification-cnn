import argparse
import os
import kagglehub
import time 
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, classification_report

# 0. 랜덤 시드 고정

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. CUB-200-2011 

class CUBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        images_file = os.path.join(root_dir, 'images.txt')
        split_file = os.path.join(root_dir, 'train_test_split.txt')
        bbox_file = os.path.join(root_dir, 'bounding_boxes.txt')
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        
        self.data = []
        
        id_to_path = {}
        with open(images_file, 'r') as f:
            for line in f:
                img_id, path = line.strip().split()
                id_to_path[img_id] = path
                
        id_to_split = {}
        with open(split_file, 'r') as f:
            for line in f:
                img_id, is_train = line.strip().split()
                id_to_split[img_id] = int(is_train)
                
        id_to_bbox = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id = parts[0]
                x, y, w, h = map(float, parts[1:])
                id_to_bbox[img_id] = (x, y, w, h)
                
        id_to_label = {}
        with open(labels_file, 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                id_to_label[img_id] = int(label) - 1 
                
        target_split = 1 if split == 'train' else 0
        for img_id in id_to_path:
            if id_to_split[img_id] == target_split:
                self.data.append({
                    'path': os.path.join(root_dir, 'images', id_to_path[img_id]),
                    'bbox': id_to_bbox[img_id],
                    'label': id_to_label[img_id]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')
        
        x, y, w, h = item['bbox']
        image = image.crop((x, y, x + w, y + h))
        
        if self.transform:
            image = self.transform(image)
            
        return image, item['label']

# 2.  EfficientNet V2 

class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes: int = 200, dropout_rate: float = 0.5):
        super().__init__()

        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),            
            nn.Linear(in_features, num_classes)    
        )

    def forward(self, x):
        return self.backbone(x)

# 3. Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset Download
    path = kagglehub.dataset_download("wenewone/cub2002011")
    data_root = os.path.join(path, "CUB_200_2011")

    # Transform
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandAugment(),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Dataset Load & Split
    train_dataset = CUBDataset(root_dir=data_root, split='train', transform=train_transform)
    full_test_dataset = CUBDataset(root_dir=data_root, split='test', transform=eval_transform)

    test_labels = [item['label'] for item in full_test_dataset.data]
    test_indices = list(range(len(full_test_dataset)))

    val_idx, test_idx = train_test_split(
        test_indices, 
        test_size=0.5, 
        stratify=test_labels, 
        random_state=42
    )

    val_dataset = Subset(full_test_dataset, val_idx)
    final_test_dataset = Subset(full_test_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(final_test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset Size - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(final_test_dataset)}")

    model = CustomEfficientNetV2(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)

    # Training Loop

    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

    print("Training Start")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # ---------------- Train ----------------
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # ---------------- Validation ----------------
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # ---------------- Scheduler Step ----------------
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_end_time = time.time()
        time_elapsed = epoch_end_time - epoch_start_time
        mins = int(time_elapsed // 60)
        secs = int(time_elapsed % 60)

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        print(
            f"Epoch [{epoch}/{args.epochs}] Time: {mins}m {secs}s | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), "./best_cub_model.pth")


    # Learning Curve

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_train_loss, label="Train Loss")
    plt.plot(history_val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_train_acc, label="Train Acc")
    plt.plot(history_val_acc, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("bird2-cub_aug_EffNetV2_learning_curve.png")
    plt.close()

    # Final Test 

    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []

    correct_cases = []
    failure_cases = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(images.size(0)):
                if predicted[i] == labels[i]:
                    if len(correct_cases) < 5:
                        correct_cases.append((images[i].cpu(), labels[i].item(), predicted[i].item()))
                else:
                    if len(failure_cases) < 5:
                        failure_cases.append((images[i].cpu(), labels[i].item(), predicted[i].item()))

    test_acc = test_correct / test_total

    print("=" * 50)
    print("Final Test Accuracy:", round(test_acc, 4))
    print("=" * 50)
    torch.save(model.state_dict(), "./bird2-bird_aug_EffNetV2.pth")
    print("bird2-Model saved as bird_aug_EffNetV2.pth")

    def unnormalize_and_plot(cases, filename, title_prefix):
        fig, axes = plt.subplots(1, len(cases), figsize=(15, 3.5))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for idx, (img_tensor, true_label, pred_label) in enumerate(cases):

            img = img_tensor.numpy().transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
            axes[idx].axis('off')
            
        plt.suptitle(f"{title_prefix} Cases", fontsize=16)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    if correct_cases:
        unnormalize_and_plot(correct_cases, "bird2-cub_aug_EffNetV2_correct_cases.png", "Correct")
    if failure_cases:
        unnormalize_and_plot(failure_cases, "bird2-cub_aug_EffNetV2_failure_cases.png", "Failure")

    # Confusion Matrix

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("bird2-cub_aug_EffNetV2_confusion_matrix.png")
    plt.close()

    # Classification Report

    report = classification_report(all_labels, all_preds, digits=4)

    with open("bird2-cub_aug_EffNetV2_report.txt", "w") as f:
        f.write(report)

    print("All files saved successfully. (Including Correct/Failure images)")

if __name__ == "__main__":
    main()
