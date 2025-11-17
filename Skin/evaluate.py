import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# =========================
# CONFIGURATION
# =========================
BASE_PATH = r"C:\Skin"
CSV_PATH = os.path.join(BASE_PATH, "processed", "preprocessed_data.csv")
IMG_DIR = os.path.join(BASE_PATH, "processed", "images")
MODEL_PATH = os.path.join(BASE_PATH, "skin_model.keras")

IMG_SIZE = 128

# =========================
# 1️⃣ Load Model
# =========================
print("📦 Loading fine-tuned model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# =========================
# 2️⃣ Load Data
# =========================
print("📄 Loading preprocessed CSV...")
df = pd.read_csv(CSV_PATH)
df["image_id"] = df["image_id"].astype(str) + ".jpg"

# Disease label mapping (same as training)
label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
inverse_label_map = {v: k for k, v in label_map.items()}

# =========================
# 3️⃣ Predict Each Image (Safely & In Order)
# =========================
print("🖼️ Processing all images and mapping with CSV entries...")
predicted_labels = []
actual_labels = []
image_ids = []

missing_images = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_id = row["image_id"]
    actual_class = row["dx"]
    img_path = os.path.join(IMG_DIR, image_id)

    image_ids.append(image_id)
    actual_labels.append(actual_class)

    if not os.path.exists(img_path):
        predicted_labels.append("missing")
        missing_images += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        predicted_labels.append("missing")
        missing_images += 1
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    pred_class = inverse_label_map[np.argmax(pred[0])]
    predicted_labels.append(pred_class)

print(f"✅ Completed predictions for {len(df)} images (missing: {missing_images})")

# =========================
# 4️⃣ Evaluation (Ignore missing)
# =========================
eval_df = pd.DataFrame({
    "image_id": image_ids,
    "actual": actual_labels,
    "predicted": predicted_labels
})

# Filter out missing entries for metrics
eval_clean = eval_df[eval_df["predicted"] != "missing"]

y_true = [label_map[a] for a in eval_clean["actual"]]
y_pred = [label_map[p] for p in eval_clean["predicted"]]

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Accuracy (on available images): {acc*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(label_map.keys()), digits=3))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# =========================
# 5️⃣ Save Full Results
# =========================
output_csv = os.path.join(BASE_PATH, "model_predictions_mapped.csv")
eval_df.to_csv(output_csv, index=False)
print(f"\n💾 All predictions mapped and saved to: {output_csv}")
