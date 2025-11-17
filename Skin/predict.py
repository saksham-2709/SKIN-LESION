import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


BASE_PATH = r"C:\Skin"
MODEL_PATH = os.path.join(BASE_PATH, "skin_model2.keras")
IMG_SIZE = 128

# Load the trained model
print("📦 Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def predict_skin_disease(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Could not read image file.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    predicted_class = np.argmax(preds[0])
    confidence = preds[0][predicted_class] * 100

    print("\n🧠 Prediction Results:")
    for i, label in enumerate(class_labels):
        print(f"   {label:>6s}: {preds[0][i]*100:.2f}%")

    print(f"\n✅ Final Prediction: {class_labels[predicted_class].upper()} ({confidence:.2f}% confidence)\n")


test_image = r"C:\Skin\processed\images\ISIC_0032342.jpg"
predict_skin_disease(test_image)
