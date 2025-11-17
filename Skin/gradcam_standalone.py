"""
Standalone Grad-CAM visualization script for testing and professional image generation.
Usage:
    python gradcam_standalone.py --model skin_model_final.keras --image path/to/image.jpg --output results/output
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

# Configuration
IMG_SIZE = 128

def generate_gradcam_overlay(model, img_array, last_conv_layer_name=None, save_path=None, original_img=None):
    """
    Generate professional Grad-CAM visualization with vivid color overlay.
    
    Args:
        model: Keras model
        img_array: Preprocessed image array (normalized, batch dimension included)
        last_conv_layer_name: Name of the last convolutional layer (auto-detected if None)
        save_path: Path to save images (without extension)
        original_img: Original image array (RGB, 0-255 range) for overlay
    
    Returns:
        tuple: (pure_heatmap, overlay_image) as numpy arrays
    """
    try:
        if model is None:
            return None, None
        
        # Get the last convolutional layer
        if last_conv_layer_name is None:
            # Try to find EfficientNet's last conv layer or any conv layer
            for layer in reversed(model.layers):
                try:
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                        if output_shape and len(output_shape) == 4:
                            last_conv_layer_name = layer.name
                            break
                except:
                    continue
        
        if last_conv_layer_name is None:
            print("Warning: Could not find convolutional layer for Grad-CAM")
            return None, None
        
        print(f"Using layer: {last_conv_layer_name}")
        
        # Create a model that outputs the last conv layer and the predictions
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )
        except Exception as e:
            print(f"Error creating grad model: {e}")
            return None, None
        
        # Compute gradient
        img_array_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array_tensor, training=False)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Get gradients with respect to conv_outputs
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None, None
        
        # Global Average Pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to focus on positive contributions
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize heatmap to 0-1 range
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        else:
            heatmap = tf.zeros_like(heatmap)
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Get original image dimensions
        if original_img is not None:
            orig_h, orig_w = original_img.shape[:2]
        else:
            # Use model input size
            orig_h, orig_w = IMG_SIZE, IMG_SIZE
            # Create a placeholder original image if not provided
            original_img = (img_array[0] * 255).astype(np.uint8)
            # Convert from RGB (if model expects RGB) to proper format
            if len(original_img.shape) == 3:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Resize heatmap to original image size for proper overlay
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-255 range for colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply JET colormap for vivid colors (red/yellow = high activation, blue = low)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Ensure original image is in BGR format and proper size
        if original_img.shape[:2] != (orig_h, orig_w):
            original_img = cv2.resize(original_img, (orig_w, orig_h))
        
        # Convert original image to RGB if needed (for overlay, we'll convert back to BGR)
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        elif original_img.shape[2] == 4:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
        
        # Create overlay: blend heatmap with original image
        # Alpha = 0.5 for semi-transparent overlay (professional look)
        overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)
        
        # Save images if save_path is provided
        if save_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            # Save pure heatmap
            cv2.imwrite(f"{save_path}_heatmap.jpg", heatmap_colored)
            # Save overlay
            cv2.imwrite(f"{save_path}_overlay.jpg", overlay)
            print(f"Grad-CAM images saved:")
            print(f"  - Pure heatmap: {save_path}_heatmap.jpg")
            print(f"  - Overlay: {save_path}_overlay.jpg")
        
        return heatmap_colored, overlay
    
    except Exception as e:
        print(f"Error generating Grad-CAM overlay: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_image(image_path):
    """Load and preprocess image for model prediction."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original for overlay
    original_img = img.copy()
    
    # Resize to model input size
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_normalized, axis=0)
    
    return img_array, original_img

def main():
    parser = argparse.ArgumentParser(description='Generate professional Grad-CAM visualizations')
    parser.add_argument('--model', type=str, required=True, help='Path to Keras model file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='results/gradcam_output', help='Output path (without extension)')
    parser.add_argument('--layer', type=str, default=None, help='Name of convolutional layer (auto-detect if not provided)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print("Model loaded successfully!")
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    img_array, original_img = preprocess_image(args.image)
    
    # Convert original to BGR for OpenCV
    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    # Generate Grad-CAM
    print("Generating Grad-CAM visualization...")
    heatmap, overlay = generate_gradcam_overlay(
        model=model,
        img_array=img_array,
        last_conv_layer_name=args.layer,
        save_path=args.output,
        original_img=original_img_bgr
    )
    
    if heatmap is not None and overlay is not None:
        print("Grad-CAM generation completed successfully!")
        
        # Display class predictions
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        print(f"Predicted class index: {class_idx}, Confidence: {confidence:.2f}%")
    else:
        print("Failed to generate Grad-CAM visualization")

if __name__ == '__main__':
    main()





