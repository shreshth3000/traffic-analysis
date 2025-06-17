import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.normpath(os.path.join(script_dir, '../models'))
valid_dir = os.path.normpath(os.path.join(script_dir, '../images_car/valid'))

# Load models
with open(os.path.join(model_dir, 'svm_model.pkl'), 'rb') as f:
    svm = pickle.load(f)
with open(os.path.join(model_dir, 'xgb_model.pkl'), 'rb') as f:
    xgb = pickle.load(f)

# Feature extraction transform (should match training)
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Extract features from valid set
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Load EfficientNet-B2 (no head)
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

features = []
labels = []
img_paths = []
with torch.no_grad():
    for imgs, lbls in valid_loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        features.append(feats.cpu().numpy())
        labels.append(lbls.cpu().numpy())
    # Get image paths in order
    for path, _ in valid_dataset.samples:
        img_paths.append(os.path.basename(path))

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# SVM predictions
svm_pred = svm.predict(features)
# XGBoost predictions
xgb_pred = xgb.predict(features)

# Metrics for SVM
svm_accuracy = accuracy_score(labels, svm_pred)
svm_precision = precision_score(labels, svm_pred, average='binary', pos_label=1)
svm_recall = recall_score(labels, svm_pred, average='binary', pos_label=1)
svm_f1 = f1_score(labels, svm_pred, average='binary', pos_label=1)

# Metrics for XGBoost
xgb_accuracy = accuracy_score(labels, xgb_pred)
xgb_precision = precision_score(labels, xgb_pred, average='binary', pos_label=1)
xgb_recall = recall_score(labels, xgb_pred, average='binary', pos_label=1)
xgb_f1 = f1_score(labels, xgb_pred, average='binary', pos_label=1)

# Class mapping
idx_to_class = {v: k for k, v in valid_dataset.class_to_idx.items()}

# Misclassified images for SVM
svm_misclassified = [(img_paths[i], idx_to_class[labels[i]], idx_to_class[svm_pred[i]])
                     for i in range(len(labels)) if labels[i] != svm_pred[i]]
# Misclassified images for XGBoost
xgb_misclassified = [(img_paths[i], idx_to_class[labels[i]], idx_to_class[xgb_pred[i]])
                     for i in range(len(labels)) if labels[i] != xgb_pred[i]]

print("SVM Metrics:")
print(f"  Accuracy: {svm_accuracy*100:.2f}%")
print(f"  Precision: {svm_precision:.4f}")
print(f"  Recall: {svm_recall:.4f}")
print(f"  F1 Score: {svm_f1:.4f}")
if svm_misclassified:
    print("\nSVM Misclassified Images:")
    for img_name, actual, predicted in svm_misclassified:
        print(f" - {img_name}: actual = {actual}, predicted = {predicted}")
else:
    print("\nSVM: No misclassifications.")

print("\nXGBoost Metrics:")
print(f"  Accuracy: {xgb_accuracy*100:.2f}%")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall: {xgb_recall:.4f}")
print(f"  F1 Score: {xgb_f1:.4f}")
if xgb_misclassified:
    print("\nXGBoost Misclassified Images:")
    for img_name, actual, predicted in xgb_misclassified:
        print(f" - {img_name}: actual = {actual}, predicted = {predicted}")
else:
    print("\nXGBoost: No misclassifications.") 