import cv2
from ultralytics import YOLO

# Path to the image and model
image_path = './data/valid/images/10_mp4-13_jpg.rf.aff71e875ee297d3086b715b7d6aaf26.jpg'  # Example image
model_path = './models/yolo8m.pt'

# Load the YOLO model
model = YOLO(model_path)

# Read and preprocess the image
image = cv2.imread(image_path)
frame = cv2.resize(image, (1220, 700))
resize_frame = cv2.resize(frame, (640, 640))
rgb_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)

# Run inference
results = model(rgb_frame, device="cpu")
scale_x = frame.shape[1] / 640
scale_y = frame.shape[0] / 640

desired_obj = [0, 3]  # Use the same as car_detect.py, if relevant

for result in results:
    for box in result.boxes:
        if hasattr(box, 'cls') and (box.cls in desired_obj or not desired_obj):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Vehicles", frame)
cv2.waitKey(0)
cv2.destroyAllWindows() 