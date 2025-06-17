import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomPerspective(0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '../images_car/train'))
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
val_size = int(0.2 * len(full_dataset))
train_dataset, val_dataset = random_split(full_dataset, [len(full_dataset)-val_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup: EfficientNet-B2
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

# Freeze the backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # 2 classes
model = model.to(device)

# Only head parameters will be trained
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop (head only)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = train_correct = train_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss/train_total:.4f}, Acc: {100*train_correct/train_total:.2f}% | "
          f"Val Loss: {val_loss/val_total:.4f}, Acc: {100*val_correct/val_total:.2f}%")

# Save
torch.save(model.state_dict(), "efficientnet_b2_direction_head_only.pth")
