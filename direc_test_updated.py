import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture (must match training)
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

# Load model
model = ImprovedDirectionCNN().to(device)
model.load_state_dict(torch.load("./models/direction_classifier_validation_V1.pth", map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Class mapping
class_names = ['backward', 'forward']
class_to_idx = {'backward': 0, 'forward': 1}

# Folder structure: ./images_car_unclassified/backward/, ./images_car_unclassified/forward/
test_root = "./images_car/valid"

total = 0
correct = 0
misclassified = []

# Iterate through class folders
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

            total += 1
            if predicted_label == label:
                correct += 1
            else:
                misclassified.append((img_file, label, predicted_label))

# Results
accuracy = 100 * correct / total if total > 0 else 0
print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")

if misclassified:
    print("\nMisclassified Images:")
    for img_name, actual, predicted in misclassified:
        print(f" - {img_name}: actual = {actual}, predicted = {predicted}")
else:
    print("\nNo misclassifications.")
