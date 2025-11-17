# Derm Insight Dash - Smart Skin Analyzer

An AI-powered skin anomaly detection system that uses deep learning to analyze skin lesions and provide diagnostic insights. Built with Flask (Python) backend and React + Tailwind CSS frontend.

## 🚀 Features

- **AI-Powered Analysis**: Uses a trained Keras model based on HAM10000 dataset
- **Grad-CAM Visualization**: Visual explanation of model predictions
- **Disease Information**: Comprehensive details about causes, remedies, and side effects
- **Dermatologist Finder**: Locate nearby dermatologists using Google Maps Places API
- **PDF Report Generation**: Download detailed health reports
- **Modern UI**: Responsive design with glassmorphism effects

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+ and npm/yarn
- Trained Keras model (`skin_model.keras`) in the root directory
- (Optional) Google Places API Key for dermatologist finder

## 🛠️ Installation

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google Places API key (optional)
   ```

3. **Verify model file:**
   - Ensure `skin_model.keras` exists in the root directory
   - If you have a different model file, update `MODEL_PATH` in `app.py`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ../dataverse
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Create environment file (optional):**
   ```bash
   # Create .env file in dataverse directory
   VITE_API_URL=http://localhost:5000
   ```

## 🚀 Running the Application

### Start Backend Server

```bash
# From root directory
python app.py
```

The Flask server will start on `http://localhost:5000`

### Start Frontend Development Server

```bash
# From dataverse directory
npm run dev
# or
yarn dev
```

The React app will start on `http://localhost:8080` (or the port configured in vite.config.ts)

## 📖 API Endpoints

### `POST /predict`
Analyzes a skin lesion image and returns prediction results.

**Request:**
- `multipart/form-data`
- Fields: `image` (file), `name`, `age`, `gender`, `areaInfected`, `sideEffects[]`

**Response:**
```json
{
  "diagnosis": "Melanocytic Nevi (Mole)",
  "confidence": 87.5,
  "gradCamUrl": "data:image/jpeg;base64,...",
  "causes": [...],
  "remedies": [...],
  "relatedSideEffects": [...],
  "classCode": "nv",
  "allPredictions": {...},
  "userData": {...}
}
```

### `GET /dermatologists`
Fetches nearby dermatologists based on coordinates.

**Parameters:**
- `lat` (float): Latitude
- `lng` (float): Longitude

**Response:**
```json
{
  "dermatologists": [
    {
      "id": 1,
      "name": "Dr. John Doe",
      "specialty": "...",
      "rating": 4.8,
      "reviews": 245,
      "distance": "0.8 km",
      "address": "...",
      "phone": "..."
    }
  ]
}
```

### `POST /report`
Generates a PDF health report.

**Request:**
- `application/json`
- Body: Complete diagnostic data (same structure as `/predict` response)

**Response:**
- PDF file download

## 🏗️ Project Structure

```
.
├── app.py                  # Flask backend application
├── requirements.txt        # Python dependencies
├── skin_model.keras        # Trained Keras model
├── .env.example           # Environment variables template
├── ../dataverse/          # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── ...
│   └── package.json
└── README.md
```

## 🔧 Configuration

### Model Configuration
Edit `MODEL_PATH` and `IMG_SIZE` in `app.py` if using a different model.

### Disease Classes
The model predicts 7 classes:
- `akiec`: Actinic Keratoses / Intraepithelial Carcinoma
- `bcc`: Basal Cell Carcinoma
- `bkl`: Benign Keratosis
- `df`: Dermatofibroma
- `mel`: Melanoma
- `nv`: Melanocytic Nevi
- `vasc`: Vascular Lesion

## 🧪 Testing

### Test Backend
```bash
# Health check
curl http://localhost:5000/health
```

### Test Frontend
Navigate to `http://localhost:8080` and upload a test image.

## 📝 Notes

- The model is trained on the HAM10000 dataset
- Grad-CAM visualization provides interpretability
- Google Places API is optional - the app will use mock data if not configured
- All medical information is for educational purposes only

## ⚠️ Disclaimer

This tool is for educational purposes only and should not replace professional medical advice. Always consult with a qualified dermatologist for accurate diagnosis and treatment.

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contributing guidelines here]


