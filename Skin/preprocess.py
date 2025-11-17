import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
BASE_PATH = r"C:\Skin"    
PART1 = os.path.join(BASE_PATH, "HAM10000_images_part_1")
PART2 = os.path.join(BASE_PATH, "HAM10000_images_part_2")
METADATA_PATH = os.path.join(BASE_PATH, "HAM10000_metadata.csv")

OUTPUT_DIR = "processed"
IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(IMG_OUT_DIR, exist_ok=True)

# =========================
# 1️⃣ Load Metadata
# =========================
print("📄 Loading metadata...")
meta = pd.read_csv(METADATA_PATH)
print(f"✅ Loaded {len(meta)} entries.")

# =========================
# 2️⃣ Find Image Paths
# =========================
def get_image_path(image_id):
    for folder in [PART1, PART2]:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

meta['image_path'] = meta['image_id'].apply(get_image_path)
meta.dropna(subset=['image_path'], inplace=True)
print(f"✅ Found {len(meta)} matching images in folders.")

# =========================
# 3️⃣ Preprocess Images (Resize + Normalize)
# =========================
print("🖼️ Preprocessing images (resizing to 224x224)...")

missing_images = 0
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    img_path = row['image_path']
    if not os.path.exists(img_path):
        missing_images += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        missing_images += 1
        continue

    img = cv2.resize(img, (224, 224))
    img = img / 255.0  
    save_path = os.path.join(IMG_OUT_DIR, f"{row['image_id']}.jpg")
    cv2.imwrite(save_path, (img * 255).astype(np.uint8))

print(f"✅ All images processed and saved to: {IMG_OUT_DIR}")
print(f"⚠️ Missing or unreadable images skipped: {missing_images}")


meta.drop(columns=['image_path'], inplace=True)

final_csv = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
meta.to_csv(final_csv, index=False)

print("✅ Final preprocessed CSV saved at:", final_csv)
print("🎉 Done! All images preprocessed and CSV ready for use.")
