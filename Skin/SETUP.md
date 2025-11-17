# Setup Guide for Derm Insight Dash

## Quick Start

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make sure skin_model_final.keras is in the root directory
# If you have a different model, update MODEL_PATH in app.py

# (Optional) Set up Google Places API key
# Create a .env file with:
# GOOGLE_PLACES_API_KEY=your_key_here

# Start Flask server
python app.py
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd derm-insight-dash

# Install dependencies
npm install

# (Optional) Create .env file for API URL
# VITE_API_URL=http://localhost:5000

# Start development server
npm run dev
```

### 3. Access the Application

- Frontend: http://localhost:8080
- Backend API: http://localhost:5000

## Model Requirements

The application expects a Keras model file (`skin_model_final.keras`) with:
- Input shape: (128, 128, 3)
- Output: 7 classes (akiec, bcc, bkl, df, mel, nv, vasc)
- Model architecture: Compatible with Grad-CAM (should have convolutional layers)

## Troubleshooting

### Model Not Loading
- Check if `skin_model_final.keras` exists in the root directory
- Verify the model file is not corrupted
- Check console for error messages

### Grad-CAM Not Working
- The model needs to have convolutional layers
- Check if the model architecture is compatible
- The function will gracefully fall back to original image if Grad-CAM fails

### Google Places API
- If API key is not set, the app will use mock data
- Get API key from: https://console.cloud.google.com/google/maps-apis
- Enable "Places API" in Google Cloud Console

### CORS Issues
- Make sure Flask-CORS is installed
- Check that backend is running on port 5000
- Verify frontend proxy configuration in vite.config.ts

## Testing

### Test Image Upload
1. Navigate to http://localhost:8080
2. Fill in personal details
3. Upload a skin lesion image
4. Select symptoms
5. Click "Analyze"
6. View results and download PDF report

### Test API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Test prediction (requires image file)
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.jpg" \
  -F "name=Test User" \
  -F "age=30" \
  -F "gender=male" \
  -F "areaInfected=arm"
```

## Production Deployment

### Backend
- Use a production WSGI server (e.g., Gunicorn)
- Set environment variables properly
- Enable HTTPS
- Configure CORS for production domain

### Frontend
- Build production bundle: `npm run build`
- Serve with a web server (e.g., Nginx)
- Update API URL in environment variables
- Enable HTTPS

## Support

For issues or questions, please check:
1. Console logs for error messages
2. Network tab for API errors
3. Backend logs for server errors


