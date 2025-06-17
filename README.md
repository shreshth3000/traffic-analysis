# Traffic-analysis

## DEMO:
Run file [demo.py](https://github.com/shreshth3000/traffic-analysis/blob/main/demo/demo.py) to view video demonstration of model.

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

## Below are the different approaches with instructions to run and results for lane detection.





## Below are the different approaches with instructions to run and results for direction detection.
