import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle  # Use pickle for saving models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '../images_car/train'))

# Transforms for feature extraction
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load pretrained EfficientNet-B2 (without classifier head)
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
model.classifier = torch.nn.Identity()  # Remove classification head
model = model.to(device)
model.eval()

# Extract features
features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        feats = model(imgs)  # Output shape: [batch_size, 1408]
        features.append(feats.cpu().numpy())
        labels.append(lbls.cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels)

# ----- SVM -----
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_val)

# Save SVM model with pickle
with open(os.path.join(script_dir, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm, f)

# ----- XGBoost -----
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_val)

# Save XGBoost model with pickle
with open(os.path.join(script_dir, 'xgb_model.pkl'), 'wb') as f:
    pickle.dump(xgb, f)

# Metrics for SVM
svm_accuracy = accuracy_score(y_val, y_pred_svm)
svm_precision = precision_score(y_val, y_pred_svm, average='binary', pos_label=1)
svm_recall = recall_score(y_val, y_pred_svm, average='binary', pos_label=1)
svm_f1 = f1_score(y_val, y_pred_svm, average='binary', pos_label=1)

# Metrics for XGBoost
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)
xgb_precision = precision_score(y_val, y_pred_xgb, average='binary', pos_label=1)
xgb_recall = recall_score(y_val, y_pred_xgb, average='binary', pos_label=1)
xgb_f1 = f1_score(y_val, y_pred_xgb, average='binary', pos_label=1)

# DataFrames for results
svm_df = pd.DataFrame({'True': y_val, 'Pred': y_pred_svm})
xgb_df = pd.DataFrame({'True': y_val, 'Pred': y_pred_xgb})

print(f"SVM Metrics:")
print(f"  Accuracy: {svm_accuracy*100:.2f}%")
print(f"  Precision: {svm_precision:.4f}")
print(f"  Recall: {svm_recall:.4f}")
print(f"  F1 Score: {svm_f1:.4f}")

print(f"\nXGBoost Metrics:")
print(f"  Accuracy: {xgb_accuracy*100:.2f}%")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall: {xgb_recall:.4f}")
print(f"  F1 Score: {xgb_f1:.4f}")
