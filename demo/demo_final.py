import os
import numpy as np
import torch
import cv2 as cv
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import deque
import math
from concurrent.futures import ThreadPoolExecutor

dev = "cuda" if torch.cuda.is_available() else "cpu"

car_model = YOLO('models/yolo8nbest.pt')
car_model.fuse()  
car_model.to(dev).half()
lane_model = YOLO('models/lane_seg_weights.pt')

direction_model_path = 'models/efficientnet_b2_direction_classifier_V2_best.pth'
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

class CarTracker:
    def __init__(self, max_lost=30, idle_frames=30, dist_thresh=50):
        self.next_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.idle_frames = idle_frames
        self.dist_thresh = dist_thresh
        self.idle_threshold = 5

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, detections):
        updated_tracks = {}
        used_detections = set()
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
                track['lost'] += 1
                if track['lost'] < self.max_lost:
                    updated_tracks[track_id] = track
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
                c0 = track['history'][0]
                if all(math.hypot(c0[0]-c[0], c0[1]-c[1]) < 5 for c in track['history']):
                    idle_ids.append(track_id)
        return idle_ids

car_tracker = CarTracker(idle_frames=20)

vid = cv.VideoCapture("demo/traffic.mp4")
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
frame_w, frame_h = 1220, 700
desired_obj = [0, 3]

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
                if conf_score > self.confidence_threshold or iou > self.iou_threshold:
                    updated_masks.append(new_mask)
                else:
                    updated_masks.append(prev_mask)
            else:
                if conf_score > self.confidence_threshold:
                    updated_masks.append(new_mask)
        self.previous_masks = updated_masks
        return updated_masks

lane_tracker = LaneTracker(confidence_threshold=0.5, iou_threshold=0.7)

@torch.no_grad()
def run_inference(frame):
    with ThreadPoolExecutor(max_workers=2) as executor:
        car_future = executor.submit(car_model.predict, frame, 
                                   imgsz=640, conf=0.3, device=dev,
                                   half=True, max_det=20, iou=0.5, agnostic_nms=True)
        lane_future = executor.submit(lane_model, frame, device=dev)
        return car_future.result(), lane_future.result()

while True:
    istrue, frame = vid.read()
    if not istrue:
        break

    frame = cv.resize(frame, (frame_w, frame_h))
    small_frame = cv.resize(frame, (640, 640))
    rgb_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    car_results, lane_results = run_inference(rgb_frame)
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
                    crop = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
                    if crop.size != 0:
                        crop_pil = Image.fromarray(cv.cvtColor(crop, cv.COLOR_BGR2RGB))
                        input_tensor = transform(crop_pil).unsqueeze(0).to(dev)
                        with torch.no_grad():
                            output = direction_model(input_tensor)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            confidence_threshold = 1      
                            if probabilities[0, 0] >= confidence_threshold:
                                label = 'backward'
                            elif probabilities[0, 1] >= confidence_threshold:
                                label = 'forward'
                            else:
                                label = 'uncertain'  # Or you can default to one direction
                        color = (0, 255, 0) if label == 'forward' else (0, 0, 255)
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                        detected_boxes_with_labels.append(((x1, y1, x2, y2), label))
                    else:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        detected_boxes_with_labels.append(((x1, y1, x2, y2), None))

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

    lane_masks = lane_tracker.update(lane_masks, confidence_scores)
    total_lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for mask in lane_masks:
        total_lane_mask |= mask

    num_lanes = len(lane_masks)
    assumed_lane_area = total_lane_mask.sum() / num_lanes if num_lanes > 0 else 1
    lane_vehicle_areas = [0] * num_lanes
    lane_vehicle_counts = [0] * num_lanes

    for x1, y1, x2, y2 in vehicle_boxes:
        box_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1
        overlaps = [np.logical_and(box_mask, lane_mask).sum() for lane_mask in lane_masks]
        if overlaps:
            lane_idx = np.argmax(overlaps)
            lane_vehicle_counts[lane_idx] += 1

    for x1, y1, x2, y2 in vehicle_boxes:
        box_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1
        overlaps = [np.logical_and(box_mask, lane_mask).sum() for lane_mask in lane_masks]
        if overlaps:
            lane_idx = int(np.argmax(overlaps))
            box_area = (x2 - x1) * (y2 - y1)
            lane_vehicle_areas[lane_idx] += box_area

    idle_vehicles = car_tracker.get_idle()
    idle_per_lane = [0] * num_lanes
        
    for track_id in idle_vehicles:
        x1, y1, x2, y2 = tracks[track_id]['bbox']
        box_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1
        overlaps = [np.logical_and(box_mask, lane_mask).sum() for lane_mask in lane_masks]
        if overlaps:
            lane_idx = np.argmax(overlaps)
            idle_per_lane[lane_idx] += 1

    for i in range(num_lanes):
        area_ratio = lane_vehicle_areas[i] / assumed_lane_area
        vehicle_count = lane_vehicle_counts[i]
        total_iou = 0
        valid_boxes = 0
        for x1, y1, x2, y2 in vehicle_boxes:
            box_mask = np.zeros_like(lane_masks[i])
            box_mask[y1:y2, x1:x2] = 1
            intersection = np.logical_and(box_mask, lane_masks[i]).sum()
            union = np.logical_or(box_mask, lane_masks[i]).sum()
            if union > 0:
                total_iou += intersection / union
                valid_boxes += 1
        avg_iou = total_iou / valid_boxes if valid_boxes > 0 else 0
        idle_factor = min(1.0, idle_per_lane[i] / 3)
        density_score = 0.5*area_ratio + 0.5*(vehicle_count/9) + 0.3*avg_iou + 0.2*idle_factor
                
        if density_score > 0.45:
            status = "High"
            color = (0, 0, 255)
        elif density_score > 0.15:
            status = "Moderate" 
            color = (0, 255, 255)
        else:
            status = "Low"
            color = (0, 255, 0)

        lane_mask_uint8 = (lane_masks[i] * 255).astype('uint8')
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
            cv.putText(frame, f"Total Vehicles: {len(vehicle_boxes)}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv.putText(frame, f"Vehicles: {lane_vehicle_counts[i]}", 
               (cx - 50, cy + 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv.putText(frame, f"Idle: {idle_per_lane[i]}", (cx-50, cy+60),
                 cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    legend_x = frame.shape[1] - 220
    legend_y = 30
    cv.putText(frame, 'Forward', (legend_x, legend_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 70), 2)
    cv.putText(frame, 'Backward', (legend_x + 100, legend_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255, 70), 2)

    for track_id, track in tracks.items():
        x1, y1, x2, y2 = track['bbox']
        if track_id in idle_ids:
            cv.putText(frame, 'idle', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv.imshow("vid", frame)
    out.write(frame)

    if cv.waitKey(10) & 0xFF == ord("d"):
        break
    if cv.getWindowProperty("vid", cv.WND_PROP_VISIBLE) < 1:
        break

vid.release()
out.release()
cv.destroyAllWindows()