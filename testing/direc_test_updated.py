import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedDirectionCNN(nn.Module):
    def __init__(self):
        super(ImprovedDirectionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ImprovedDirectionCNN().to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '../models/direction_classifier_validation_V3.pth')
model_path = os.path.normpath(model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class_names = ['backward', 'forward']
class_to_idx = {'backward': 0, 'forward': 1}

test_root = os.path.join(script_dir, '../images_car/valid')
test_root = os.path.normpath(test_root)

y_true = []
y_pred = []
misclassified = []

for label in class_names:
    folder = os.path.join(test_root, label)
    if not os.path.exists(folder):
        print(f"Warning: folder '{folder}' does not exist.")
        continue

    for img_file in sorted(os.listdir(folder)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_file)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                predicted_label = class_names[predicted.item()]

            y_true.append(label)
            y_pred.append(predicted_label)
            if predicted_label != label:
                misclassified.append((img_file, label, predicted_label))

y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)

data = pd.DataFrame({
    'True': y_true_np,
    'Pred': y_pred_np
})

accuracy = accuracy_score(y_true_np, y_pred_np)
precision = precision_score(y_true_np, y_pred_np, pos_label='forward', average='binary')
recall = recall_score(y_true_np, y_pred_np, pos_label='forward', average='binary')
f1 = f1_score(y_true_np, y_pred_np, pos_label='forward', average='binary')

print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

if misclassified:
    print("\nMisclassified Images:")
    for img_name, actual, predicted in misclassified:
        print(f" - {img_name}: actual = {actual}, predicted = {predicted}")
else:
    print("\nNo misclassifications.")
