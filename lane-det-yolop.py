import torch
from torchvison import transforms
from PIL import Image


model = torch.hub.load('hustvl/yolop', 'yolop', pretrained = True)


img_path = "..data/valid/5_mp4-29_jpg.rf.de506ae3402dafeffb1e6427af894871.jpg"
img = Image.open(img_path)
img.show()



















# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # img_det: original image as numpy array (H, W, 3), RGB
# # da_seg_mask: drivable area mask (H, W), 0/1/2
# # ll_seg_mask: lane line mask (H, W), 0/1/2

# # Overlay drivable area (e.g., blue) and lane lines (e.g., yellow)
# def show_seg_result(img_det, seg_masks, is_demo=True):
#     da_seg_mask, ll_seg_mask = seg_masks
#     img = img_det.copy()
#     # Drivable area: blue overlay
#     img[da_seg_mask == 1] = img[da_seg_mask == 1] * 0.5 + np.array([0, 0, 255]) * 0.5
#     # Lane lines: yellow overlay
#     img[ll_seg_mask == 1] = img[ll_seg_mask == 1] * 0.5 + np.array([255, 255, 0]) * 0.5
#     return img.astype(np.uint8)

# # Example usage:
# result_img = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), is_demo=True)

# plt.figure(figsize=(12, 8))
# plt.imshow(result_img)
# plt.axis('off')
# plt.show()
