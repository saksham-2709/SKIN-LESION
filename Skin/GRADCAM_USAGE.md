# Enhanced Grad-CAM Visualization Usage

## Overview

The enhanced Grad-CAM function produces professional, publication-quality visualizations with vivid color overlays similar to those used in research papers.

## Features

- **Vivid Color Mapping**: Uses JET colormap (red/yellow for high activation, blue/purple for low)
- **Professional Overlay**: Semi-transparent overlay (alpha=0.5) on original image
- **Dual Output**: Returns both pure heatmap and overlay
- **Auto-detection**: Automatically finds the last convolutional layer
- **High Quality**: Proper normalization and interpolation for clear visualizations

## Usage in Flask API

The enhanced function is automatically used in the `/predict` endpoint:

```python
# In app.py - automatically used
gradcam_heatmap, gradcam_overlay = generate_gradcam_overlay(
    model=model,
    img_array=img_array,
    last_conv_layer_name=None,  # Auto-detect
    save_path=None,  # Don't save in API
    original_img=original_img_bgr
)
```

## Standalone Script Usage

Use the standalone script for testing and batch processing:

```bash
# Basic usage
python gradcam_standalone.py \
    --model skin_model_final.keras \
    --image path/to/image.jpg \
    --output results/gradcam_output

# With specific layer
python gradcam_standalone.py \
    --model skin_model_final.keras \
    --image path/to/image.jpg \
    --output results/gradcam_output \
    --layer "block7a_expand_conv"
```

## Function Signature

```python
def generate_gradcam_overlay(
    model,                    # Keras model
    img_array,                # Preprocessed image (normalized, with batch dimension)
    last_conv_layer_name=None, # Layer name (auto-detected if None)
    save_path=None,           # Path to save images (without extension)
    original_img=None         # Original image (BGR, 0-255 range)
) -> tuple:                   # Returns (heatmap, overlay)
```

## Output Files

When `save_path` is provided, two files are created:
- `{save_path}_heatmap.jpg`: Pure color heatmap
- `{save_path}_overlay.jpg`: Heatmap overlay on original image

## Color Scheme

- **Red/Yellow**: High activation areas (important for prediction)
- **Blue/Purple**: Low activation areas
- **Transparency**: 50% overlay for natural appearance

## Example

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('skin_model_final.keras')

# Load and preprocess image
img = cv2.imread('skin_lesion.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (128, 128))
img_normalized = img_resized / 255.0
img_array = np.expand_dims(img_normalized, axis=0)

# Generate Grad-CAM
heatmap, overlay = generate_gradcam_overlay(
    model=model,
    img_array=img_array,
    save_path='results/gradcam_skin_lesion',
    original_img=img
)

# Results saved to:
# - results/gradcam_skin_lesion_heatmap.jpg
# - results/gradcam_skin_lesion_overlay.jpg
```

## Tips

1. **Layer Selection**: If auto-detection fails, inspect model layers:
   ```python
   for layer in model.layers:
       if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
           print(layer.name)
   ```

2. **Image Quality**: Use original resolution images for better overlay quality

3. **Customization**: Adjust alpha blending in `cv2.addWeighted()` for different transparency levels

4. **Colormap**: Change `cv2.COLORMAP_JET` to other colormaps:
   - `cv2.COLORMAP_VIRIDIS`: Green-blue scale
   - `cv2.COLORMAP_HOT`: Red-yellow scale
   - `cv2.COLORMAP_COOL`: Blue-cyan scale





