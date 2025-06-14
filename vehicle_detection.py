import cv2
import numpy as np
from ultralytics import YOLO

# Path to the image and models
image_path = './data/valid/images/5_mp4-27_jpg.rf.30a8975089fb66bc245019de9d868801.jpg'
  # Example image
vehicle_model_path = './models/yolo8m.pt'
lane_model_path = './models/lane_seg_weights.pt'

# Load the YOLO models
vehicle_model = YOLO(vehicle_model_path)
lane_model = YOLO(lane_model_path)

# Read and preprocess the image
image = cv2.imread(image_path)
frame = cv2.resize(image, (1220, 700))
resize_frame = cv2.resize(frame, (640, 640))
rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)

# --- Vehicle Detection ---
vehicle_results = vehicle_model(rgb_frame, device="cpu")
scale_x = frame.shape[1] / 640
scale_y = frame.shape[0] / 640

desired_obj = [0, 3]  # Use the same as car_detect.py, if relevant

for result in vehicle_results:
    for box in result.boxes:
        if hasattr(box, 'cls') and (box.cls in desired_obj or not desired_obj):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- Lane Segmentation ---
lane_results = lane_model(rgb_frame, device="cpu")
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
                             cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0),
                             frame)

cv2.imshow("Detected Vehicles and Lanes", frame)
cv2.waitKey(0)
cv2.destroyAllWindows() 