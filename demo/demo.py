import os
import numpy as np
import torch
import cv2 as cv
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict, deque
import math

car_model = YOLO('models/yolo8nbest.pt')
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

# --- Car Tracker for Idle Detection ---
class CarTracker:
    def __init__(self, max_lost=30, idle_frames=20, dist_thresh=50):
        self.next_id = 0
        self.tracks = {}  # id: {'bbox': (x1, y1, x2, y2), 'centroid': (x, y), 'lost': 0, 'history': deque}
        self.max_lost = max_lost
        self.idle_frames = idle_frames
        self.dist_thresh = dist_thresh

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, detections):
        updated_tracks = {}
        used_detections = set()
        # Match detections to existing tracks
        for track_id, track in self.tracks.items():
            min_dist = float('inf')
            min_idx = -1
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                c1 = self._centroid(track['bbox'])
                c2 = self._centroid(det)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_dist < self.dist_thresh:
                # Update track
                bbox = detections[min_idx]
                centroid = self._centroid(bbox)
                history = track['history']
                history.append(centroid)
                if len(history) > self.idle_frames:
                    history.popleft()
                updated_tracks[track_id] = {
                    'bbox': bbox,
                    'centroid': centroid,
                    'lost': 0,
                    'history': history
                }
                used_detections.add(min_idx)
            else:
                # Track lost
                track['lost'] += 1
                if track['lost'] < self.max_lost:
                    updated_tracks[track_id] = track
        # Add new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                centroid = self._centroid(det)
                updated_tracks[self.next_id] = {
                    'bbox': det,
                    'centroid': centroid,
                    'lost': 0,
                    'history': deque([centroid], maxlen=self.idle_frames)
                }
                self.next_id += 1
        self.tracks = updated_tracks
        return self.tracks

    def get_idle(self):
        idle_ids = []
        for track_id, track in self.tracks.items():
            if len(track['history']) == self.idle_frames:
                # If all centroids in history are close, mark as idle
                c0 = track['history'][0]
                if all(math.hypot(c0[0]-c[0], c0[1]-c[1]) < 5 for c in track['history']):
                    idle_ids.append(track_id)
        return idle_ids

car_tracker = CarTracker(idle_frames=20)

vid = cv.VideoCapture("demo/traffic_last5.mp4")
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
vid.set(cv.CAP_PROP_FPS, 60)

if not vid.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_w, frame_h = 1220, 700
desired_obj = [0, 3]

# VideoWriter to save output
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('demo/output.mp4', fourcc, 30, (frame_w, frame_h))

# # Read the first frame for lane detection
# istrue, first_frame = vid.read()
# if not istrue:
#     print("Error: Could not read the first frame.")
#     exit()
# first_frame = cv.resize(first_frame, (frame_w, frame_h))
# resize_first_frame = cv.resize(first_frame, (640, 640))
# rgb_first_frame = cv.cvtColor(resize_first_frame, cv.COLOR_BGR2RGB)
# lane_results = lane_model(rgb_first_frame, device=dev)

# Reset video to the beginning
vid.set(cv.CAP_PROP_POS_FRAMES, 0)

class LaneTracker:
    def __init__(self, confidence_threshold=0.5, iou_threshold=0.7):
        self.previous_masks = None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
        
    def update(self, new_masks, confidence_scores):
        if self.previous_masks is None:
            self.previous_masks = new_masks
            return new_masks
            
        updated_masks = []
        for i, (new_mask, conf_score) in enumerate(zip(new_masks, confidence_scores)):
            if i < len(self.previous_masks):
                prev_mask = self.previous_masks[i]
                iou = self.calculate_iou(new_mask, prev_mask)
                
                # Update mask if confidence is high enough or IoU is good
                if conf_score > self.confidence_threshold or iou > self.iou_threshold:
                    updated_masks.append(new_mask)
                else:
                    updated_masks.append(prev_mask)
            else:
                # New lane detected
                if conf_score > self.confidence_threshold:
                    updated_masks.append(new_mask)
                else:
                    # Skip low confidence new lanes
                    continue
                    
        self.previous_masks = updated_masks
        return updated_masks

# Initialize lane tracker
lane_tracker = LaneTracker(confidence_threshold=0.5, iou_threshold=0.7)

while True:
    istrue, frame = vid.read()
    if not istrue:
        break

    frame = cv.resize(frame, (frame_w, frame_h))
    resize_frame = cv.resize(frame, (640, 640))
    rgb_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2RGB)

    car_results = car_model(rgb_frame, device=dev)
    lane_results = lane_model(rgb_frame, device=dev)

    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 640

    vehicle_boxes = []
    detected_boxes_with_labels = []

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
                    # --- Direction classification integration (restored) ---
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
                        detected_boxes_with_labels.append(((x1, y1, x2, y2), label))
                    else:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        detected_boxes_with_labels.append(((x1, y1, x2, y2), None))

    # --- Car tracking for idle detection ---
    tracks = car_tracker.update(vehicle_boxes)
    idle_ids = set(car_tracker.get_idle())

    lane_masks = []
    confidence_scores = []
    total_lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for result in lane_results:
        if result.masks is not None:
            for mask, conf in zip(result.masks.data, result.boxes.conf):
                mask_np = mask.cpu().numpy().astype('uint8')
                mask_resized = cv.resize(mask_np, (frame.shape[1], frame.shape[0]))
                lane_masks.append(mask_resized)
                confidence_scores.append(conf.item())

    # Update lane masks with confidence thresholding
    lane_masks = lane_tracker.update(lane_masks, confidence_scores)
    
    # Update total lane mask
    total_lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for mask in lane_masks:
        total_lane_mask |= mask

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

    # Draw 'idle' label for tracked cars
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = track['bbox']
        if track_id in idle_ids:
            cv.putText(frame, 'idle', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv.imshow("vid", frame)
    # out.write(frame)

    # Break if 'd' is pressed or window is closed
    if cv.waitKey(10) & 0xFF == ord("d"):
        break
    # Check if window was closed
    if cv.getWindowProperty("vid", cv.WND_PROP_VISIBLE) < 1:
        break

vid.release()
# out.release()
cv.destroyAllWindows()
