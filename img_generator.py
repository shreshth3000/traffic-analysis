import cv2
import os
import pandas as pd

# Load the CSV file
df = pd.read_csv('./data/train/labels/_annotations.csv')
df = df.iloc[1000::51]

# Directory containing the images
image_dir = './data/train/images'
# Output directory for cropped images
output_dir = 'images_car_unclassified'
os.makedirs(output_dir, exist_ok=True)

# Group by filename so we don't reload the same image multiple times
grouped = df.groupby('filename')

for filename, group in grouped:
    image_path = os.path.join(image_dir, filename)
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"[Warning] Image not found: {image_path}")
        continue
    
    # Load image
    image = cv2.imread(image_path)
    
    for idx, row in group.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Crop the region
        cropped = image[ymin:ymax, xmin:xmax]
        
        # Output filename: originalname_boxID.jpg
        crop_filename = f"{os.path.splitext(filename)[0]}_{idx}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        
        # Save cropped image
        cv2.imwrite(crop_path, cropped)

        print(f"[Saved] {crop_path}")

print("All bounding boxes cropped and saved.")
