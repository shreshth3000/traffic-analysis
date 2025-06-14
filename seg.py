import numpy as np
from ultralytics import YOLO
import cv2 as cv


model = YOLO('./models/lane_seg_weights.pt')  


vid = cv.VideoCapture("vid.mp4")
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
vid.set(cv.CAP_PROP_FPS, 60)

while True:
    istrue, frame = vid.read()
    if not istrue:
        break
    frame = cv.resize(frame, (1220, 700))

    resize_frame = cv.resize(frame, (640, 640))
    rgb_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2RGB)

    results = model(rgb_frame, device="cpu")

    if results is None:
        continue

    for result in results:
        masks = result.masks
        if masks is not None:
            for mask in masks.data:
                mask_np = mask.cpu().numpy().astype('uint8')
                mask_resized = cv.resize(mask_np, (frame.shape[1], frame.shape[0]))
            
                color = (0, 255, 0) 
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask_resized == 1] = color
          
                frame = np.where(colored_mask > 0,
                                 cv.addWeighted(frame, 0.5, colored_mask, 0.5, 0),
                                 frame)

    cv.imshow("Lane Segmentation", frame)

    if cv.waitKey(10) & 0xFF == ord('d'):
        break

cv.destroyAllWindows()
vid.release()
