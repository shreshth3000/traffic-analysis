import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import uuid
import cv2
import numpy as np

val_root = "data/valid/images"
val_anno_path = "data/valid/labels/_annotations.coco.json"
batch_size = 1
img_size = 300
model_weights = "models/best_ssd_weights.pt"

resize_size = img_size
orig_size = 640
scale_x = orig_size / resize_size
scale_y = orig_size / resize_size

def collate_fn(batch):
    return tuple(zip(*batch))

def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return image

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

val_dataset = CocoDetection(root=val_root, annFile=val_anno_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = ssd300_vgg16(weights=None, weights_backbone=None)
model.head.classification_head.num_classes = 2
model.load_state_dict(torch.load(model_weights))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

device = next(model.parameters()).device
coco_gt = COCO(val_anno_path)
coco_results = []
os.makedirs("ssd_val_vis", exist_ok=True)

with torch.no_grad():
    for i, (image, targets) in enumerate(val_loader):
        image_tensor = image[0].to(device)
        output = model([image_tensor])[0]

        img_info = val_dataset.coco.loadImgs(val_dataset.ids[i])[0]
        image_path = os.path.join(val_root, img_info['file_name'])
        original_image = cv2.imread(image_path)

        pred_boxes = output['boxes'].cpu().numpy()
        pred_scores = output['scores'].cpu().numpy()
        pred_labels = output['labels'].cpu().numpy()

        mask = pred_scores > 0.5
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]

        scaled_pred_boxes = []
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            scaled_pred_boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ])
        scaled_pred_boxes = np.array(scaled_pred_boxes)

        original_image = draw_boxes(original_image, scaled_pred_boxes, (0, 0, 255), "pred")

        gt_boxes = []
        for obj in targets[0]:
            x, y, w, h = obj['bbox']
            gt_boxes.append([x, y, x + w, y + h])
        original_image = draw_boxes(original_image, gt_boxes, (0, 255, 0), "gt")

        cv2.imwrite(f"ssd_val_vis/img_{i:03}.jpg", original_image)

        image_id = targets[0][0]['image_id']
        for box, score, label in zip(scaled_pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box
            coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            result = {
                "image_id": image_id,
                "category_id": 1,
                "bbox": coco_box,
                "score": float(score),
                "id": uuid.uuid4().int >> 64
            }
            coco_results.append(result)

if coco_results:
    result_path = "tmp_coco_results_val.json"
    with open(result_path, 'w') as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(result_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
