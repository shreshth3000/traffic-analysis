import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# Path to the image and models
image_path = './data/valid/images/test2_mp4-16_jpg.rf.585413166ce3fb74dc3e975edc009020.jpg' # Example image

vehicle_model_path = './models/yolo8nbest.pt'
lane_model_path = './models/lane_seg_weights.pt'
direction_model_path = './models/efficientnet_b2_direction_classifier_V2_best.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vehicle_model = YOLO(vehicle_model_path).to(device)
lane_model = YOLO(lane_model_path).to(device)
# Load EfficientNet-B2 for direction classification
direction_model = models.efficientnet_b2()
num_features = direction_model.classifier[1].in_features
direction_model.classifier[1] = nn.Linear(num_features, 2)
direction_model.load_state_dict(torch.load(direction_model_path, map_location=device))
direction_model = direction_model.to(device)
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

lane_results = lane_model(rgb_frame, device="cuda" if torch.cuda.is_available() else "cpu")
lane_masks = []
total_lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
for result in lane_results:
    masks = result.masks
    if masks is not None:
        for mask in masks.data:
            mask_np = mask.cpu().numpy().astype('uint8')
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            lane_masks.append(mask_resized)
            total_lane_mask |= mask_resized

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
num_vehicles=len(vehicle_boxes)
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
        color = (0, 0, 255)      
    elif density_ratio > 0.1:
        status = "Moderate"
        color = (0, 255, 255)    
    else:
        status = "Low"
        color = (0, 255, 0)      

    lane_mask_uint8 = (lane_masks[i] * 255).astype('uint8')
    colored_mask = np.zeros_like(frame, dtype=np.uint8)
    colored_mask[lane_masks[i] == 1] = color
    frame = np.where(colored_mask > 0,
                     cv2.addWeighted(frame, 0.5, colored_mask, 0.6, 0),
                     frame)
    
    contours, _ = cv2.findContours(lane_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx = x + w // 2
        cy = y + h // 2
        cv2.putText(frame, f"Lane {i+1}: {status}", (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {num_vehicles}", (500,100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),2)

cv2.imshow("Detected Vehicles, Lanes, and Directions", frame)

legend_x = frame.shape[1] - 220
legend_y = 30
cv2.putText(frame, 'Forward', (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 70), 1)
cv2.putText(frame, 'Backward', (legend_x + 100, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255, 70), 1)


# # Lane density color legend (no background, more transparent, larger text)
# lane_legend_x = 30
# lane_legend_y = 30
# cv2.rectangle(frame, (lane_legend_x, lane_legend_y), (lane_legend_x + 30, lane_legend_y + 20), (0, 255, 0), cv2.FILLED)
# cv2.putText(frame, 'Low', (lane_legend_x + 40, lane_legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0, 70), 1)
# lane_legend_y += 30
# cv2.rectangle(frame, (lane_legend_x, lane_legend_y), (lane_legend_x + 30, lane_legend_y + 20), (0, 255, 255), cv2.FILLED)
# cv2.putText(frame, 'Moderate', (lane_legend_x + 40, lane_legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255, 70), 1)
# lane_legend_y += 30
# cv2.rectangle(frame, (lane_legend_x, lane_legend_y), (lane_legend_x + 30, lane_legend_y + 20), (0, 0, 255), cv2.FILLED)
# cv2.putText(frame, 'High', (lane_legend_x + 40, lane_legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255, 70), 1)


cv2.imshow("Detected Vehicles, Lanes, and Directions", frame)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# Save the processed image
cv2.imwrite('output.jpg', frame) 