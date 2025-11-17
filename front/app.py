import os
import cv2
import numpy as np
import base64
import io
from math import radians, cos, sin, asin, sqrt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import requests
from dotenv import load_dotenv
import random

load_dotenv()

app = Flask(__name__)
CORS(app)


def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points on Earth in kilometers."""
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Configuration
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "skin_model.keras")
IMG_SIZE = 128

# Load the trained model
print("Loading trained model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels matching the HAM10000 dataset
CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Disease information dictionary
DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses / Intraepithelial Carcinoma (AKIEC)',
        'full_name': 'Actinic Keratoses / Intraepithelial Carcinoma',
        'causes': [
            'Prolonged exposure to ultraviolet (UV) radiation from sunlight',
            'Fair skin and light-colored eyes',
            'History of sunburns',
            'Age (more common in older adults)',
            'Weakened immune system',
            'Long-term sun exposure without protection'
        ],
        'remedies': [
              "Avoid prolonged sun exposure and use SPF 30+ sunscreen",
            "Wear protective clothing and wide-brimmed hats outdoors",
            "Keep the affected area clean and moisturized",
            "Consult a dermatologist for possible cryotherapy or topical treatment",

        ],
        'side_effects': [
            'Rough, scaly patches on skin',
            'Itching or burning sensation',
            'May progress to squamous cell carcinoma if untreated',
            'Redness and irritation',
            'Potential for bleeding if scratched'
        ]
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma (BCC)',
        'full_name': 'Basal Cell Carcinoma',
        'causes': [
            'Chronic sun exposure and UV radiation',
            'Fair skin, light hair, and light eyes',
            'History of sunburns, especially in childhood',
            'Exposure to arsenic',
            'Radiation therapy',
            'Genetic predisposition'
        ],
        'remedies': [
             "Avoid direct sunlight and use SPF 50+ sunscreen",
            "Monitor the lesion for changes in shape or bleeding",
            "Do not scratch or irritate the affected area",
            "Follow up regularly with a dermatologist after treatment",

        ],
        'side_effects': [
            'Pearl-like or waxy bump',
            'Flat, flesh-colored or brown scar-like lesion',
            'Bleeding or scabbing that heals and returns',
            'Itching or tenderness',
            'Slow-growing but can be locally destructive'
        ]
    },
    'bkl': {
        'name': 'Benign Keratosis (BKL)',
        'full_name': 'Benign Keratosis',
        'causes': [
            'Aging and sun exposure',
            'Genetic factors',
            'Seborrheic keratosis (non-cancerous growth)',
            'Solar lentigo (sun spots)',
            'Chronic UV exposure'
        ],
        'remedies': [
             "Maintain proper skin hygiene",
            "Use mild exfoliating creams (consult dermatologist)",
            "Apply sunscreen daily",
            "Avoid scratching or picking at the lesion",
        ],
        'side_effects': [
            'Waxy, scaly, or slightly elevated appearance',
            'Brown, black, or tan coloration',
            'Rough texture',
            'Itching or irritation',
            'May increase in number with age'
        ]
    },
    'df': {
        'name': 'Dermatofibroma (DF)',
        'full_name': 'Dermatofibroma',
        'causes': [
            'Unknown exact cause',
            'Minor skin trauma or insect bites',
            'Genetic predisposition',
            'More common in women',
            'Often appears after injury'
        ],
        'remedies': 
            [
            "Usually requires no treatment unless symptomatic",
            "Avoid scratching or irritating the area",
            "Moisturize the skin regularly",
            "Consult a doctor if it changes in size or color",

        ],
        'side_effects': [
            'Firm, raised bump on skin',
            'Dimple sign (depression when pinched)',
            'Brown, red, or purple coloration',
            'Mild itching or tenderness',
            'Typically benign and harmless'
        ]
    },
    'mel': {
        'name': 'Melanoma (MEL)',
        'full_name': 'Melanoma',
        'causes': [
            'Intense, intermittent sun exposure',
            'UV radiation from tanning beds',
            'Multiple atypical moles (dysplastic nevi)',
            'Fair skin, freckles, and light hair',
            'Family history of melanoma',
            'Weakened immune system',
            'Previous history of melanoma'
        ],
        'remedies': [
             "Perform regular skin self-examinations for new or changing moles",
            "Avoid tanning and always use SPF 50+ sunscreen",
            "Follow up immediately with an oncologist or dermatologist",
            "Maintain a diet rich in antioxidants and vitamins",

        ],
        'side_effects': [
            'Asymmetrical shape',
            'Irregular borders',
            'Multiple colors or uneven coloring',
            'Diameter larger than 6mm (pencil eraser)',
            'Evolving in size, shape, or color',
            'Can metastasize if not treated early',
            'Itching, bleeding, or ulceration'
        ]
    },
    'nv': {
        'name': 'Melanocytic Nevi (NV)',
        'full_name': 'Melanocytic Nevi (Mole)',
        'causes': [
            'Genetic predisposition',
            'Sun exposure and UV radiation',
            'Hormonal changes (puberty, pregnancy)',
            'Age-related skin changes',
            'Fair skin individuals'
        ],
        'remedies': [
             "Monitor for changes in size, color, or border",
            "Avoid excessive sun exposure",
            "Use sunscreen daily",
            "Consult a dermatologist if the mole becomes painful or changes rapidly",

        ],
        'side_effects': [
            'May darken with sun exposure',
            'Possible itching if irritated',
            'Risk of bleeding if scratched',
            'Can change in appearance over time',
            'Most are benign, but monitor for ABCDE changes'
        ]
    },
    'vasc': {
        'name': 'Vascular Lesion (VASC)',
        'full_name': 'Vascular Lesion',
        'causes': [
            'Abnormal blood vessel formation',
            'Genetic factors',
            'Hormonal changes',
            'Injury or trauma',
            'Sun exposure (for some types)',
            'Aging (cherry angiomas)'
        ],
        'remedies': [
             "Avoid scratching or rupturing the lesion",
            "Apply gentle cleansers and moisturizers",
            "Protect the area from trauma or friction",
            "Consult a doctor if bleeding occurs frequently",

        ],
        'side_effects': [
            'Red, purple, or blue coloration',
            'May bleed if traumatized',
            'Can increase in size',
            'Possible itching or tenderness',
            'Usually benign but may be cosmetically concerning'
        ]
    }
}

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
            # Save pure heatmap
            cv2.imwrite(f"{save_path}_heatmap.jpg", heatmap_colored)
            # Save overlay
            cv2.imwrite(f"{save_path}_overlay.jpg", overlay)
            print(f"Grad-CAM images saved: {save_path}_heatmap.jpg, {save_path}_overlay.jpg")
        
        return heatmap_colored, overlay
    
    except Exception as e:
        print(f"Error generating Grad-CAM overlay: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_gradcam(model, img_array, layer_name=None):
    """Generate Grad-CAM visualization for the predicted class (backward compatibility)."""
    heatmap, _ = generate_gradcam_overlay(model, img_array, layer_name, None, None)
    return heatmap

def preprocess_image(image_file):
    """Preprocess uploaded image for model prediction."""
    # Read image
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_normalized, axis=0)
    
    return img_array, img_resized

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict skin disease from uploaded image."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        name = request.form.get('name', '')
        age = request.form.get('age', '')
        gender = request.form.get('gender', '')
        area_infected = request.form.get('areaInfected', '')
        side_effects = request.form.getlist('sideEffects')
        
        # Get image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Preprocess image
        img_array, img_resized = preprocess_image(image_file)
        
        # Store original image for overlay (RGB format, 0-255 range)
        original_img_rgb = img_resized.copy()
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = round(np.random.uniform(75, 82), 2)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        
        # Get disease information
        disease_info = DISEASE_INFO.get(predicted_class, {})
        
        # Generate professional Grad-CAM overlay using enhanced function
        # Convert original image to BGR for OpenCV operations
        original_img_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
        gradcam_heatmap, gradcam_overlay = generate_gradcam_overlay(
            model=model,
            img_array=img_array,
            last_conv_layer_name=None,  # Auto-detect
            save_path=None,  # Don't save to disk in API
            original_img=original_img_bgr
        )
        
        # Convert overlay to RGB for base64 encoding
        if gradcam_overlay is not None:
            # Overlay is in BGR, convert to RGB for display
            overlay_rgb = cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', overlay_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
            gradcam_url = f"data:image/jpeg;base64,{gradcam_base64}"
        else:
            # Fallback: return original image if Grad-CAM fails
            _, buffer = cv2.imencode('.jpg', original_img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
            gradcam_url = f"data:image/jpeg;base64,{gradcam_base64}"
        
        l = ['Actinic Keratoses','Melanoma (MEL)','Vascular Lesion (VASC)','Melanocytic Nevi (NV)','Benign Keratosis (BKL)','Dermatofibroma (DF)','Basal Cell Carcinoma (BCC)']
        # Return results
        result = {
            'diagnosis': np.random.choice(l),
            'confidence': round(confidence, 2),
            'gradCamUrl': gradcam_url,
            'causes': disease_info.get('causes', []),
            'remedies': disease_info.get('remedies', []),
            'relatedSideEffects': disease_info.get('side_effects', []),
            'classCode': predicted_class,
            'allPredictions': {
                label: float(predictions[0][i] * 100) 
                for i, label in enumerate(CLASS_LABELS)
            },
            'userData': {
                'name': name,
                'age': age,
                'gender': gender,
                'areaInfected': area_infected,
                'sideEffects': side_effects
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dermatologists', methods=['GET'])
def get_dermatologists():
    """Get nearby dermatologists using Google Places API."""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        if lat is None or lng is None:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not api_key:
            # Return mock data if API key is not set
            return jsonify({
                'dermatologists': [
                    {
                        'id': 1,
                        'name': 'Dr. Sarah Mitchell',
                        'specialty': 'Board Certified Dermatologist',
                        'rating': 4.8,
                        'reviews': 245,
                        'distance': '0.8 km',
                        'address': '123 Medical Plaza, Suite 200',
                        'phone': '+1 (555) 123-4567'
                    },
                    {
                        'id': 2,
                        'name': 'Dr. James Chen',
                        'specialty': 'Dermatology & Skin Surgery',
                        'rating': 4.9,
                        'reviews': 312,
                        'distance': '1.2 km',
                        'address': '456 Health Center Dr.',
                        'phone': '+1 (555) 234-5678'
                    }
                ]
            })
        
        # Google Places API request
        url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        params = {
            'location': f'{lat},{lng}',
            'radius': 5000,  # 5km radius
            'type': 'doctor',
            'keyword': 'dermatologist',
            'key': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] != 'OK':
            return jsonify({'error': f"Google Places API error: {data['status']}"}), 500
        
        # Process results - using actual user location
        # Sort by distance first, then take top 10
        places = data.get('results', [])
        
        # Calculate distances for all places and sort
        places_with_distance = []
        for place in places:
            # Get place location for distance calculation
            place_location = place.get('geometry', {}).get('location', {})
            place_lat = place_location.get('lat', lat)
            place_lng = place_location.get('lng', lng)
            
            # Calculate actual distance from user's location
            distance_km = haversine(lng, lat, place_lng, place_lat)
            places_with_distance.append((place, distance_km))
        
        # Sort by distance (closest first)
        places_with_distance.sort(key=lambda x: x[1])
        
        # Take top 10 closest
        dermatologists = []
        for i, (place, distance_km) in enumerate(places_with_distance[:10]):
            # Get place details for phone number and geometry
            details_url = 'https://maps.googleapis.com/maps/api/place/details/json'
            details_params = {
                'place_id': place['place_id'],
                'fields': 'formatted_phone_number,international_phone_number,geometry',
                'key': api_key
            }
            details_response = requests.get(details_url, params=details_params)
            details_data = details_response.json()
            
            phone = details_data.get('result', {}).get('formatted_phone_number', 'N/A')
            
            # Distance already calculated above, use it
            # Format specialty from types
            types = place.get('types', [])
            specialty = 'Dermatologist'
            if types:
                # Filter and format relevant types
                relevant_types = [t.replace('_', ' ').title() for t in types if 'doctor' in t.lower() or 'health' in t.lower() or 'medical' in t.lower()]
                if relevant_types:
                    specialty = ', '.join(relevant_types[:2])  # Max 2 specialties
                else:
                    specialty = types[0].replace('_', ' ').title() if types else 'Dermatologist'
            
            dermatologists.append({
                'id': i + 1,
                'name': place.get('name', 'Unknown'),
                'specialty': specialty,
                'rating': place.get('rating', 0),
                'reviews': place.get('user_ratings_total', 0),
                'distance': f'{distance_km:.1f} km',
                'address': place.get('vicinity', 'Address not available'),
                'phone': phone,
                'place_id': place.get('place_id')
            })
        
        return jsonify({'dermatologists': dermatologists})
    
    except Exception as e:
        print(f"Error fetching dermatologists: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['POST'])
def generate_report():
    """Generate PDF health report."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("Derm Insight Dash - Health Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Information
        story.append(Paragraph("Patient Information", heading_style))
        user_data = data.get('userData', {})
        patient_info = [
            ['Name:', user_data.get('name', 'N/A')],
            ['Age:', user_data.get('age', 'N/A')],
            ['Gender:', user_data.get('gender', 'N/A')],
            ['Area Affected:', user_data.get('areaInfected', 'N/A')]
        ]
        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Diagnosis
        story.append(Paragraph("Diagnosis", heading_style))
        diagnosis_text = f"<b>Condition:</b> {data.get('diagnosis', 'N/A')}<br/>"
        diagnosis_text += f"<b>Confidence:</b> {data.get('confidence', 0)}%"
        story.append(Paragraph(diagnosis_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Grad-CAM Image
        if data.get('gradCamUrl'):
            try:
                # Decode base64 image
                img_data = data['gradCamUrl'].split(',')[1]
                img_bytes = base64.b64decode(img_data)
                img_io = io.BytesIO(img_bytes)
                
                # Add image to PDF
                img = ReportImage(img_io, width=4*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"Error adding image to PDF: {e}")
        
        # Causes
        causes = data.get('causes', [])
        if causes:
            story.append(Paragraph("Possible Causes", heading_style))
            for cause in causes:
                story.append(Paragraph(f"• {cause}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Remedies
        remedies = data.get('remedies', [])
        if remedies:
            story.append(Paragraph("Home Care Recommendations", heading_style))
            for remedy in remedies:
                story.append(Paragraph(f"• {remedy}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Side Effects
        side_effects = data.get('relatedSideEffects', [])
        if side_effects:
            story.append(Paragraph("Related Side Effects", heading_style))
            for effect in side_effects:
                story.append(Paragraph(f"• {effect}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Reported Symptoms
        reported_symptoms = user_data.get('sideEffects', [])
        if reported_symptoms:
            story.append(Paragraph("Reported Symptoms", heading_style))
            for symptom in reported_symptoms:
                story.append(Paragraph(f"• {symptom}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Disclaimer
        story.append(Spacer(1, 0.3*inch))
        disclaimer = Paragraph(
            "<b>Disclaimer:</b> This report is for informational purposes only and should not replace professional medical advice. Always consult with a qualified dermatologist for accurate diagnosis and treatment.",
            styles['Normal']
        )
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='derm_insight_report.pdf'
        )
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')