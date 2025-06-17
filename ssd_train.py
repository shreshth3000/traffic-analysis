import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
from torch.optim import AdamW

data_root = "data/train/images"
anno_path = "data/train/labels/_annotations.coco.json"
batch_size = 4
num_epochs = 20
img_size = 300

def collate_fn(batch):
    return tuple(zip(*batch))

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_dataset = CocoDetection(root=data_root, annFile=anno_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = ssd300_vgg16(pretrained=True)
model.head.classification_head.num_classes = 2
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train()
model.to(device)
optimizer = AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]

        processed_targets = []
        for t in targets:
            boxes = []
            labels = []
            for obj in t:
                x, y, w, h = obj['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(obj['category_id'])
            boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.int64).to(device)
            processed_targets.append({'boxes': boxes, 'labels': labels})

        loss_dict = model(images, processed_targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"ssd_models/ssd_weights{epoch+1}.pt")