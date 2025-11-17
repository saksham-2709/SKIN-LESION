import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
BASE_PATH = r"C:\Skin"
CSV_PATH = os.path.join(BASE_PATH, "processed", "preprocessed_data.csv")
IMG_DIR = os.path.join(BASE_PATH, "processed", "images")
MODEL_PATH = os.path.join(BASE_PATH, "skin_model.keras")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10   # 👈 Total 10 epochs only

# =========================
# 1️⃣ Load Preprocessed CSV
# =========================
print("📄 Loading preprocessed CSV...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} records.")

df["image_id"] = df["image_id"].astype(str) + ".jpg"

# =========================
# 2️⃣ Balance Dataset
# =========================
print("⚖️ Balancing dataset before generator setup...")

balanced_df = pd.DataFrame()
for label in df['dx'].unique():
    subset = df[df['dx'] == label]
    if len(subset) > 500:
        subset = resample(subset, replace=False, n_samples=500, random_state=42)
    else:
        subset = resample(subset, replace=True, n_samples=500, random_state=42)
    balanced_df = pd.concat([balanced_df, subset])

print(f"✅ Balanced dataset: {balanced_df['dx'].value_counts().to_dict()}")

# =========================
# 3️⃣ Data Generators
# =========================
print("🧩 Setting up ImageDataGenerator...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    validation_split=0.2
)

classes_list = sorted(df["dx"].unique().tolist())

train_gen = train_datagen.flow_from_dataframe(
    dataframe=balanced_df,
    directory=IMG_DIR,
    x_col="image_id",
    y_col="dx",
    classes=classes_list,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    subset="training",
    batch_size=BATCH_SIZE,
    seed=42
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=balanced_df,
    directory=IMG_DIR,
    x_col="image_id",
    y_col="dx",
    classes=classes_list,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    subset="validation",
    batch_size=BATCH_SIZE,
    seed=42
)

print(f"✅ Found {train_gen.n} training and {val_gen.n} validation images.")
num_classes = len(train_gen.class_indices)
print(f"🧠 Classes: {train_gen.class_indices}")

# =========================
# 4️⃣ Compute Class Weights
# =========================
print("⚖️ Computing class weights for imbalance...")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(classes_list),
    y=df['dx']
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("✅ Computed class weights:", class_weight_dict)

# =========================
# 5️⃣ Build Model
# =========================
print("⚙️ Building EfficientNetB0 model...")

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True   # 👈 Unfreeze full model for single-phase fine-tuning

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# 6️⃣ Training (10 Epochs Total)
# =========================
print("🚀 Training EfficientNetB0 for 10 total epochs...")

checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# =========================
# 7️⃣ Save Model
# =========================
model.save(MODEL_PATH)
print(f"\n💾 Final model saved successfully at: {MODEL_PATH}")
print("🎉 Training complete!")


