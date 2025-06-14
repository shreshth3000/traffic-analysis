import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Path to the image and models
image_path = './data/valid/images/test_mp4-13_jpg.rf.98cd77f75c4492f8f103aaf4ce2ca8f8.jpg' # Example image
vehicle_model_path = './models/yolo8m.pt'
lane_model_path = './models/lane_seg_weights.pt'
direction_model_path = './models/direction_classifier_validation_V2.pth'

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

vehicle_model = YOLO(vehicle_model_path)
lane_model = YOLO(lane_model_path)
direction_model = ImprovedDirectionCNN().to(device)
direction_model.load_state_dict(torch.load(direction_model_path, map_location=device))
direction_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])
class_names = ['backward', 'forward']

image = cv2.imread(image_path)
frame = cv2.resize(image, (1220, 700))
resize_frame = cv2.resize(frame, (640, 640))
rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)

# Vehicle Detection
vehicle_results = vehicle_model(rgb_frame, device="cuda" if torch.cuda.is_available() else "cpu")
scale_x = frame.shape[1] / 640
scale_y = frame.shape[0] / 640

desired_obj = [0, 3]

for result in vehicle_results:
    for box in result.boxes:
        if hasattr(box, 'cls') and (box.cls in desired_obj or not desired_obj):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            crop = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
            if crop.size != 0:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = direction_model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    label = class_names[predicted.item()]

                color = (0, 255, 0) if label == 'forward' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Lane Segmentation
lane_results = lane_model(rgb_frame, device="cuda" if torch.cuda.is_available() else "cpu")
lane_masks = []
total_lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
for result in lane_results:
    masks = result.masks
    if masks is not None:
        for mask in masks.data:
            mask_np = mask.cpu().numpy().astype('uint8')
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            color = (0, 255, 0)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask_resized == 1] = color
            frame = np.where(colored_mask > 0,
                             cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0),
                             frame)
            lane_masks.append(mask_resized)
            total_lane_mask |= mask_resized

# Collect vehicle boxes (from detection loop above)
vehicle_boxes = []
for result in vehicle_results:
    for box in result.boxes:
        if hasattr(box, 'cls') and (box.cls in desired_obj or not desired_obj):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            vehicle_boxes.append((x1, y1, x2, y2))

num_lanes = len(lane_masks)
assumed_lane_area = total_lane_mask.sum() / num_lanes if num_lanes > 0 else 1
lane_vehicle_areas = [0] * num_lanes

for x1, y1, x2, y2 in vehicle_boxes:
    box_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 1
    overlaps = [np.logical_and(box_mask, lane_mask).sum() for lane_mask in lane_masks]
    if overlaps:
        lane_idx = int(np.argmax(overlaps))
        box_area = (x2 - x1) * (y2 - y1)
        lane_vehicle_areas[lane_idx] += box_area

for i in range(num_lanes):
    density_ratio = lane_vehicle_areas[i] / assumed_lane_area
    if density_ratio > 0.4:
        status = "High"
        color = (0, 0, 255)      # Red
    elif density_ratio > 0.1:
        status = "Moderate"
        color = (0, 255, 255)    # Yellow
    else:
        status = "Low"
        color = (0, 255, 0)      # Green

    lane_mask_uint8 = (lane_masks[i] * 255).astype('uint8')
    colored_mask = np.zeros_like(frame, dtype=np.uint8)
    colored_mask[lane_masks[i] == 1] = color
    frame = np.where(colored_mask > 0,
                     cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0),
                     frame)

cv2.imshow("Detected Vehicles, Lanes, and Directions", frame)

# Legend
legend_x = frame.shape[1] - 220
legend_y = 30
cv2.putText(frame, 'Forward', (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(frame, 'Backward', (legend_x + 110, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Detected Vehicles, Lanes, and Directions", frame)
cv2.waitKey(0)
cv2.destroyAllWindows() 