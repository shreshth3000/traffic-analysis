import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        
        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] ==1] = [255, 0, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    return img

# Add CUDA support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model to CUDA
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True).to(device)
model.eval()

# Normalization as in original code
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

img_path = "./data/valid/images/3_mp4-8_jpg.rf.4a784fedfdd385032614371ea5cb78fa.jpg"
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

with torch.no_grad():
    det_out, da_seg_out, ll_seg_out = model(img_tensor)

# Post-process da_seg_out
# da_seg_out: [1, 2, 640, 640]
da_seg_mask = da_seg_out.argmax(1).squeeze().cpu().numpy().astype(np.uint8)  # [640, 640]

# Convert to PIL image for visualization
da_seg_mask_img = Image.fromarray(da_seg_mask * 127)  # 0, 127, 254 for 3 classes
da_seg_mask_img.show()

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
