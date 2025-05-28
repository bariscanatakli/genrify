# Genrify - Separated Architecture Documentation

## Overview
Successfully separated the music genre classification system into independent frontend and backend services.

## Architecture

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Next.js       │◄──────────────►│   FastAPI       │
│   Frontend      │                │   Backend       │
│   (Port 3001)   │                │   (Port 8888)   │
└─────────────────┘                └─────────────────┘
       │                                     │
       │                                     │
    TypeScript                          Python + TensorFlow
    React UI                            AI Model Processing
```

## Components

### FastAPI Backend (`/server/`)
- **Location**: `/home/baris/genrify/server/main.py`
- **Port**: 8888
- **Features**:
  - Health check endpoint (`/health`)
  - Prediction endpoints (`/predict`, `/predict-with-viz`)
  - GPU acceleration support
  - CORS enabled for frontend communication
  - TensorFlow optimization
  - ~13 second prediction time (68% faster than original)

### Next.js Frontend (`/app/`)
- **Location**: `/home/baris/genrify/app/page.tsx`
- **Port**: 3001
- **Features**:
  - Modern React UI with Tailwind CSS
  - Real-time backend health monitoring
  - File upload with progress tracking
  - Results visualization
  - GPU toggle option

## Running the Application

### 1. Start Backend Server
```bash
cd /home/baris/genrify/server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 2. Start Frontend
```bash
cd /home/baris/genrify
npm install
npm run dev
```

### 3. Access Application
- **Frontend UI**: http://localhost:3001
- **Backend API**: http://localhost:8888/docs
- **Health Check**: http://localhost:8888/health

## API Endpoints

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true
}
```

### Prediction
```
POST /predict
Content-Type: multipart/form-data
Body: 
  - file: audio file (MP3, WAV, M4A)
  - use_gpu: boolean (optional)

Response: {
  "predicted_genre": "rock",
  "confidence": 0.85,
  "genre_probabilities": {...},
  "processing_time": 13.2
}
```

## Performance Metrics

| Method | Processing Time | Improvement |
|--------|----------------|-------------|
| Original Pipeline | 42.0s | Baseline |
| FastAPI Backend | ~13.3s | 68% faster |
| Old API Route | 41.8s | 0% faster |

## Benefits Achieved

✅ **Separated Concerns**: Python AI backend, TypeScript frontend  
✅ **Scalable Backend**: Can handle multiple clients independently  
✅ **API-First Design**: Reusable backend for different frontends  
✅ **Independent Deployment**: Backend and frontend can be deployed separately  
✅ **GPU Optimization Maintained**: All performance optimizations preserved  
✅ **Developer Experience**: Clear separation, better debugging, easier maintenance  

## Testing

Run end-to-end tests:
```bash
python test_separated_architecture.py
```

Expected results:
- ✅ Backend (FastAPI): PASS
- ✅ Frontend (Next.js): PASS  
- ✅ Integration: PASS

## Environment Variables

### Frontend (`.env.local`)
```
NEXT_PUBLIC_API_URL=http://localhost:8888
```

### Backend (`server/.env`)
```
TF_CPP_MIN_LOG_LEVEL=2
TF_ENABLE_ONEDNN_OPTS=1
```

## File Structure

```
/home/baris/genrify/
├── server/                    # FastAPI Backend
│   ├── main.py               # Main application
│   ├── requirements.txt      # Python dependencies
│   ├── start_server.sh       # Startup script
│   └── README.md            # Backend documentation
├── app/                      # Next.js Frontend
│   ├── page.tsx             # Main React component
│   ├── layout.tsx           # App layout
│   └── globals.css          # Styles
├── models/                   # AI Models
│   └── optimized_cnn_model.keras
├── package.json             # Node.js dependencies
├── .env.local               # Environment variables
└── test_separated_architecture.py  # E2E tests
```

## Next Steps

1. **Production Deployment**: Configure for production with proper domains
2. **Authentication**: Add user authentication if needed
3. **Rate Limiting**: Implement API rate limiting for production
4. **Monitoring**: Add logging and monitoring for both services
5. **Docker**: Containerize both services for easier deployment
