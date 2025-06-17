import os
import numpy as np
import torch
import cv2 as cv
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

car_model = YOLO('models/yolo8m.pt')
lane_model = YOLO('models/lane_seg_weights.pt')

direction_model_path = 'models/efficientnet_b2_direction_classifier_V2_best.pth'

dev = "cuda" if torch.cuda.is_available() else "cpu"
# Load EfficientNet-B2 for direction classification
# (same as in main.py)
direction_model = models.efficientnet_b2()
num_features = direction_model.classifier[1].in_features
direction_model.classifier[1] = nn.Linear(num_features, 2)
direction_model.load_state_dict(torch.load(direction_model_path, map_location=dev))
direction_model = direction_model.to(dev)
direction_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])
class_names = ['backward', 'forward']



vid = cv.VideoCapture("demo/vid.mp4")
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
vid.set(cv.CAP_PROP_FPS, 60)

if not vid.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_w, frame_h = 1220, 700
desired_obj = [0, 3]

# VideoWriter to save output
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('demo/output.mp4', fourcc, 30, (frame_w, frame_h))

# Read the first frame for lane detection
istrue, first_frame = vid.read()
if not istrue:
    print("Error: Could not read the first frame.")
    exit()
first_frame = cv.resize(first_frame, (frame_w, frame_h))
resize_first_frame = cv.resize(first_frame, (640, 640))
rgb_first_frame = cv.cvtColor(resize_first_frame, cv.COLOR_BGR2RGB)
lane_results = lane_model(rgb_first_frame, device=dev)

# Reset video to the beginning
vid.set(cv.CAP_PROP_POS_FRAMES, 0)

while True:
    istrue, frame = vid.read()
    if not istrue:
        break

    frame = cv.resize(frame, (frame_w, frame_h))
    resize_frame = cv.resize(frame, (640, 640))
    rgb_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2RGB)

    car_results = car_model(rgb_frame, device=dev)

    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 640

    vehicle_boxes = []

    if car_results:
        for result in car_results:
            for box in result.boxes:
                if int(box.cls) in desired_obj:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    vehicle_boxes.append((x1, y1, x2, y2))
                    # --- Direction classification integration ---
                    crop = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
                    if crop.size != 0:
                        crop_pil = Image.fromarray(cv.cvtColor(crop, cv.COLOR_BGR2RGB))
                        input_tensor = transform(crop_pil).unsqueeze(0).to(dev)
                        with torch.no_grad():
                            output = direction_model(input_tensor)
                            _, predicted = torch.max(output, 1)
                            label = class_names[predicted.item()]
                        color = (0, 255, 0) if label == 'forward' else (0, 0, 255)
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                    else:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    lane_masks = []
    total_lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for result in lane_results:
        if result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy().astype('uint8')
                mask_resized = cv.resize(mask_np, (frame.shape[1], frame.shape[0]))
                lane_masks.append(mask_resized)
                total_lane_mask |= mask_resized  

                # Visualize mask on frame
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask_resized == 1] = (0, 255, 0)
                frame = np.where(colored_mask > 0,
                                cv.addWeighted(frame, 0.5, colored_mask, 0.5, 0),
                                frame)

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
        # Lane coloring based on density (like main.py)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[lane_masks[i] == 1] = color
        frame = np.where(colored_mask > 0,
                         cv.addWeighted(frame, 0.5, colored_mask, 0.6, 0),
                         frame)

        contours, _ = cv.findContours(lane_mask_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            cx = x + w // 2
            cy = y + h // 2
            cv.putText(frame, f"Lane {i+1}: {status}", (cx - 50, cy),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame,f"Vehicles: {len(vehicle_boxes)}",(100,100),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
    legend_x = frame.shape[1] - 220
    legend_y = 30
    cv.putText(frame, 'Forward', (legend_x, legend_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 70), 2)
    cv.putText(frame, 'Backward', (legend_x + 100, legend_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255, 70), 2)

    
    cv.imshow("vid", frame)
    out.write(frame)

    # Break if 'd' is pressed or window is closed
    if cv.waitKey(10) & 0xFF == ord("d"):
        break
    # Check if window was closed
    if cv.getWindowProperty("vid", cv.WND_PROP_VISIBLE) < 1:
        break

vid.release()
out.release()
cv.destroyAllWindows()
