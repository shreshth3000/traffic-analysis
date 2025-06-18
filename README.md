# Traffic-analysis

## DEMO:
Run file [demo_improved.py](https://github.com/shreshth3000/traffic-analysis/blob/main/demo/demo_improved.py) to view the video demonstration of the model.

## Output format

**Sample Output:**

![Sample Output](sample_output.jpg)
1. Cars with green bounding boxes are moving forwards (away the camera) and cars with red boundinf boxes are moving backwards(towards from the camera)
2. The lane colours signify the traffic in that particular lane.

## Below are the different approaches with instructions to run and results for object detection.

### Model:
#### 1. YOLO 8 nano pretrained 
- [Training File](https://github.com/shreshth3000/traffic-analysis/blob/main/yolotrained.ipynb)
- [best.pt](https://github.com/shreshth3000/traffic-analysis/blob/main/models/yolo8nbest.pt)
##### Dependencies:
```
pip install ultralytics
```
##### Results:
- mAP@0.50: 0.9755
- mAP@0.50-0.95: 0.7320
- Precision: 0.9004
- Recall: 0.9391

#### 2. YOLOv8 trained from scratch

##### Dependencies:
```
pip install ultralytics
```
##### Results:
- mAP@0.50: 0.96
- mAP@0.50-0.95: 0.73
- Precision: 0.92
- Recall: 0.90

#### 3. DeTR:

| Metric | IoU/ Area/ MaxDets | ResNet-50 | ResNet-101 |
|--------|------------------------------------------|-------|-------|
| **AP** | @[IoU=0.50:0.95 / area=all / maxDets=100] | 0.613 | 0.627 |
| **AP** | @[IoU=0.50      / area=all / maxDets=100] | 0.915 | 0.945 |
| **AP** | @[IoU=0.75      / area=all / maxDets=100] | 0.727 | 0.758 |
| **AP** | @[IoU=0.50:0.95 / area=small / maxDets=100] | 0.379 | 0.382 |
| **AP** | @[IoU=0.50:0.95 / area=medium / maxDets=100] | 0.640 | 0.658 |
| **AP** | @[IoU=0.50:0.95 / area=large / maxDets=100] | 0.806 | 0.821 |
| **AR** | @[IoU=0.50:0.95 / area=all / maxDets=1] | 0.472 | 0.481 |
| **AR** | @[IoU=0.50:0.95 / area=all / maxDets=10] | 0.712 | 0.729 |
| **AR** | @[IoU=0.50:0.95 / area=small / maxDets=100] | 0.567 | 0.582 |
| **AR** | @[IoU=0.50:0.95 / area=medium / maxDets=100] | 0.735 | 0.754 |
| **AR** | @[IoU=0.50:0.95 / area=large / maxDets=100] | 0.858 | 0.860 |

