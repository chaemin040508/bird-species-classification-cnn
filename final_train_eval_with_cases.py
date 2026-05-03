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

# ==============================================================================
# 0. 랜덤 시드 고정
# ==============================================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. CUB-200-2011 전용 커스텀 데이터셋 (BBox 및 Split 적용)
# ==============================================================================
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


# ==============================================================================
# 2. 🌟 EfficientNet V2 모델 구조 
# ==============================================================================
class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes: int = 200, dropout_rate: float = 0.5):
        super().__init__()
        # EfficientNet V2 Small 모델 불러오기
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        
        # EfficientNet은 마지막 분류기가 'classifier'라는 이름의 Sequential로 되어 있음
        # 보통 [0]이 Dropout, [1]이 Linear 층이므로 [1]의 in_features를 가져옴
        in_features = self.backbone.classifier[1].in_features

        # 새로운 분류기 층 덮어쓰기
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),            
            nn.Linear(in_features, num_classes)    
        )

    def forward(self, x):
        return self.backbone(x)


# ==============================================================================
# 3. Main
# ==============================================================================
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

    # Model & Optimizer & Scheduler 적용
    model = CustomEfficientNetV2(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

    print("Training Start")

    for epoch in range(1, args.epochs + 1):
        # 🌟 에포크 시작 시간 기록
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

        # 🌟 에포크 종료 시간 기록 및 걸린 시간 계산
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

    # ==========================================================================
    # Learning Curve
    # ==========================================================================
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

    # ==========================================================================
    # Final Test
    # ==========================================================================
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total

    print("=" * 50)
    print("Final Test Accuracy:", round(test_acc, 4))
    print("=" * 50)
    torch.save(model.state_dict(), "./bird2-bird_aug_EffNetV2.pth")
    print("bird2-Model saved as bird_aug_EffNetV2.pth")

    # ==========================================================================
    # Confusion Matrix
    # ==========================================================================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("bird2-cub_aug_EffNetV2_confusion_matrix.png")
    plt.close()

    # ==========================================================================
    # Classification Report
    # ==========================================================================
    report = classification_report(all_labels, all_preds, digits=4)

    with open("bird2-cub_aug_EffNetV2_report.txt", "w") as f:
        f.write(report)



    # ==========================================================================
    # Top-5 Correct / Failure Case Visualization and Class-wise Analysis
    # ==========================================================================
    # This section was added for the final report. It saves qualitative examples
    # and class-wise confusion summaries so the report can discuss not only the
    # final accuracy number, but also what kinds of images the model succeeds or
    # fails on.

    # Re-run final test once more while keeping the image tensors for visualization.
    case_images, case_true, case_pred = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu, labels_gpu = images.to(device), labels.to(device)
            outputs = model(images_gpu)
            _, predicted = torch.max(outputs, 1)
            for i in range(images.size(0)):
                case_images.append(images[i].cpu())
                case_true.append(int(labels[i].item()))
                case_pred.append(int(predicted[i].cpu().item()))

    def unnormalize_image(tensor_img):
        """Convert normalized tensor image back to displayable numpy image."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor_img * std + mean
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()

    def save_case_grid(indices, title, filename):
        plt.figure(figsize=(15, 3.5))
        for plot_idx, data_idx in enumerate(indices[:5]):
            plt.subplot(1, 5, plot_idx + 1)
            plt.imshow(unnormalize_image(case_images[data_idx]))
            plt.title(f"True: {case_true[data_idx]}\nPred: {case_pred[data_idx]}", fontsize=10)
            plt.axis("off")
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    correct_indices = [i for i, (t, p) in enumerate(zip(case_true, case_pred)) if t == p]
    failure_indices = [i for i, (t, p) in enumerate(zip(case_true, case_pred)) if t != p]
    save_case_grid(correct_indices, "Correct Cases", "correct_cases.png")
    save_case_grid(failure_indices, "Failure Cases", "failure_cases.png")

    # Class-wise recall table: this helps identify the best and weakest classes.
    class_correct = np.diag(cm)
    class_total = cm.sum(axis=1)
    class_recall = np.divide(
        class_correct,
        class_total,
        out=np.zeros_like(class_correct, dtype=float),
        where=class_total != 0
    )

    sorted_recall_idx = np.argsort(class_recall)
    with open("top5_best_recall.csv", "w") as f:
        f.write("class_id,correct,total,recall\n")
        for class_id in sorted_recall_idx[-5:][::-1]:
            f.write(f"{class_id},{class_correct[class_id]},{class_total[class_id]},{class_recall[class_id]:.4f}\n")

    with open("top5_worst_recall.csv", "w") as f:
        f.write("class_id,correct,total,recall\n")
        for class_id in sorted_recall_idx[:5]:
            f.write(f"{class_id},{class_correct[class_id]},{class_total[class_id]},{class_recall[class_id]:.4f}\n")

    # Top confusion pairs: off-diagonal cells with the largest counts.
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    flat_indices = np.argsort(cm_offdiag.ravel())[::-1]
    with open("top10_confusion_pairs.csv", "w") as f:
        f.write("true_class,pred_class,count\n")
        written = 0
        for flat_idx in flat_indices:
            true_class, pred_class = np.unravel_index(flat_idx, cm_offdiag.shape)
            count = cm_offdiag[true_class, pred_class]
            if count <= 0:
                break
            f.write(f"{true_class},{pred_class},{count}\n")
            written += 1
            if written == 10:
                break

    print("Saved correct_cases.png, failure_cases.png, top5_best_recall.csv, top5_worst_recall.csv, and top10_confusion_pairs.csv")

    print("All files saved successfully.")

if __name__ == "__main__":
    main()