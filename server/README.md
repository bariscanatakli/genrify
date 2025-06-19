# Music Genre Classification Server

FastAPI backend server for AI-powered music genre classification.

## Architecture

```
genrify/
├── server/           # FastAPI Backend
│   ├── main.py      # Main server application
│   ├── requirements.txt
│   └── start_server.sh
├── app/             # Next.js Frontend
│   └── page_new.tsx # New client-only frontend
└── models/          # AI Models
    └── optimized_cnn_model.keras
```

## Quick Start

### 1. Install Server Dependencies
```bash
cd server
source ../venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Backend Server
```bash
# Option 1: Using script (standard)
cd server
./start_server.sh

# Option 1b: Using GPU-optimized script
cd server
./start-gpu.sh

# Option 2: Using npm script (from root directory)
npm run server

# Option 2b: Using npm script for development (port 8000)
npm run server:dev

# Option 3: Direct uvicorn
cd server
source ../venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

### 3. Start Frontend (separate terminal)
```bash
npm run dev
```

### 4. Start Both Together
```bash
npm run dev:full
```

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server status, GPU availability, and model loading status

### Basic Prediction
- **POST** `/predict`
- Upload MP3/WAV/M4A file
- Returns genre prediction with probabilities

### Prediction with Visualization
- **POST** `/predict-with-viz`
- Upload audio file
- Returns prediction + visualization data

### Music Recommendations
- **POST** `/recommend`
- Upload audio file to get similar song recommendations
- Returns list of recommended tracks

### Batch Processing
- **POST** `/predict-batch` or `/batch-predict`
- Upload multiple audio files for batch processing
- Returns predictions for all files

### Ensemble Prediction
- **POST** `/predict-ensemble`
- Advanced prediction using multiple models
- Returns enhanced prediction accuracy

### Model Statistics
- **GET** `/model-stats`
- Returns detailed model performance statistics

### Audio Processing
- **POST** `/audio/process`
- Raw audio processing endpoint

### API Documentation
- **GET** `/docs` - Interactive Swagger UI
- **GET** `/redoc` - ReDoc documentation

## Performance

- **Processing Time**: ~13 seconds per song
- **GPU Acceleration**: Automatic detection and usage
- **Supported Formats**: MP3, WAV, M4A
- **Max File Size**: No explicit limit (reasonable files recommended)
- **Default Port**: 8888 (production scripts), 8000 (development npm script)

## Port Configuration

The server runs on different ports depending on how it's started:
- **Production/Scripts**: Port 8888 (`./start_server.sh`, `./start-gpu.sh`, `python main.py`)
- **Development**: Port 8000 (`npm run server:dev`)
- **Frontend expects**: Port 8888 by default (configurable via `NEXT_PUBLIC_API_URL`)

## Environment Variables

The server automatically sets optimal TensorFlow configurations:
- `TF_FORCE_GPU_ALLOW_GROWTH=true`
- `TF_GPU_THREAD_MODE=gpu_private`
- `TF_XLA_FLAGS=--tf_xla_enable_xla_devices`
- `CUDA_CACHE_DISABLE=0`
- `TF_ENABLE_ONEDNN_OPTS=1`

## Frontend Configuration

Update `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8888
```

## Testing

Test the API directly:
```bash
curl -X POST "http://localhost:8888/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-song.mp3" \
  -F "use_gpu=true"
```

## Benefits of Separation

1. **Independent Scaling**: Scale backend and frontend separately
2. **Technology Choice**: Use Python for AI, TypeScript for UI
3. **Better Performance**: Dedicated server for heavy processing
4. **API Reusability**: Backend can serve multiple clients
5. **Development**: Work on frontend/backend independently
6. **Deployment**: Deploy to different servers/containers

## Production Deployment

### Backend (FastAPI)
- Deploy to cloud server with GPU support
- Use gunicorn with uvicorn workers
- Set up load balancing for multiple instances

### Frontend (Next.js)
- Deploy to Vercel, Netlify, or static hosting
- Update `NEXT_PUBLIC_API_URL` to production backend URL

### Docker Option
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim
COPY server/ /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
```
