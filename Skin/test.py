import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import cv2

# =========================
# CONFIGURATION
# =========================
BASE_PATH = r"C:\Skin"
CSV_PATH = os.path.join(BASE_PATH, "processed", "preprocessed_data.csv")
IMG_DIR = os.path.join(BASE_PATH, "processed", "images")
MODEL_PATH = os.path.join(BASE_PATH, "skin_model_final.keras")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 5
SEED = 42

# =========================
# 1️⃣ LOAD DATA
# =========================
print("📄 Loading CSV...")
df = pd.read_csv(CSV_PATH)
df["image_id"] = df["image_id"].astype(str) + ".jpg"
print(f"✅ Loaded {len(df)} images across {df['dx'].nunique()} classes.")

# =========================
# 2️⃣ DATA GENERATORS
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="image_id",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    subset="training",
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_gen = datagen.flow_from_dataframe(
    df,
    directory=IMG_DIR,
    x_col="image_id",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    subset="validation",
    batch_size=BATCH_SIZE,
    seed=SEED
)

num_classes = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())
print("🧠 Classes:", train_gen.class_indices)

# =========================
# 3️⃣ CLASS WEIGHTS (Bias Mitigation)
# =========================
# Helps handle imbalance (nv has ~6000 samples)
labels = df['dx']
cw = compute_class_weight('balanced', classes=np.array(class_names), y=labels)
class_weight_dict = {train_gen.class_indices[name]: float(weight) for name, weight in zip(class_names, cw)}
print("⚖ Class Weights:", class_weight_dict)

# =========================
# 4️⃣ BUILD MODEL
# =========================
print("⚙ Building EfficientNetB0 model...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Callbacks
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7, verbose=1)

# =========================
# 5️⃣ PHASE 1 – Train Classifier Head
# =========================
print("\n🚀 Phase 1: Training classifier head (frozen base)...")
base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# =========================
# 6️⃣ PHASE 2 – Fine-tune Full Model
# =========================
print("\n🔧 Phase 2: Fine-tuning EfficientNet layers...")
base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False  # keep early layers frozen

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# =========================
# 7️⃣ SAVE FINAL MODEL
# =========================
model.save(MODEL_PATH)
print(f"\n💾 Model saved successfully at: {MODEL_PATH}")
print("🎉 Training complete!")

# =========================
# 8️⃣ PLOT TRAINING CURVES
# =========================
def plot_history(history1, history2):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

plot_history(history1, history2)

# =========================
# 9️⃣ GRAD-CAM GENERATION
# =========================
def generate_gradcam(model, img_path, class_idx=None):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("efficientnetb0").output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM Visualization ({class_names[class_idx]})")
    plt.axis('off')
    plt.show()

# Example usage:
# test_img_path = os.path.join(IMG_DIR, df.iloc[0]["image_id"])
# generate_gradcam(model, test_img_path)

print("\n✅ Grad-CAM function ready! Use it for explainability visualization.")
print("⚖️ Class rebalancing, data augmentation, and Grad-CAM ensure fairness & interpretability.")
