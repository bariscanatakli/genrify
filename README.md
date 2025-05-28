# 🎵 Genrify - Music Genre Classifier

An AI-powered music genre classification system with **separated architecture** - FastAPI backend for AI processing and Next.js frontend for user interface. This application uses deep learning models to analyze audio files and predict their genres with high accuracy.

## 🚀 Features

- **⚡ Separated Architecture**: Independent FastAPI backend and Next.js frontend
- **🎯 Real-time Audio Analysis**: Upload audio files and get instant genre predictions (~13 seconds)
- **🧠 Deep Learning Models**: Optimized CNN architecture with GPU acceleration
- **📊 Interactive Visualizations**: Real-time progress tracking and confidence scores
- **🎵 Comprehensive Genre Support**: 8 major genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
- **🔧 Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS
- **📈 Performance Optimized**: 68% faster than original pipeline
- **🌐 API-First Design**: Reusable backend for multiple frontends

## 🏗️ Separated Architecture

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

### Backend (FastAPI)
- **AI Processing**: TensorFlow-based genre classification
- **GPU Acceleration**: CUDA support for faster inference
- **RESTful API**: Clean endpoints for health checks and predictions
- **Performance**: ~13 second processing time (68% improvement)

### Backend (Python/TensorFlow)
- **CNN Classifier**: Optimized convolutional neural network for genre classification
- **Autoencoder**: Dense embedding generation for music similarity
- **Triplet Network**: Discriminative embeddings for enhanced recommendations
- **Audio Processing**: Librosa-based feature extraction (spectrograms, MFCC, tempo, etc.)
### Frontend (Next.js)
- **React Components**: Modern UI with real-time feedback
- **API Integration**: Communicates with FastAPI backend
- **Real-time Updates**: Dynamic progress tracking and status indicators
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development  
- **Tailwind CSS** - Utility-first styling
- **React Hooks** - State management

### Backend
- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - Deep learning models
- **CUDA/cuDNN** - GPU acceleration
- **Librosa** - Audio processing and feature extraction
- **NumPy** - Data manipulation
- **Uvicorn** - ASGI server

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone <repository>
cd genrify
```

### 2. Start Backend Server
```bash
cd server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 3. Start Frontend
```bash
# In a new terminal
cd genrify
npm install
npm run dev
```

### 4. Access Application
- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8888/docs
- **Health Check**: http://localhost:8888/health

## 📦 Detailed Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- CUDA Toolkit 11.2+ and cuDNN (optional, for GPU acceleration)
- At least 4GB RAM (for ML models)

### Frontend Setup
```bash
# Clone the repository
git clone [your-repo-url]
cd music-recommender

# Install dependencies
npm install

# Start development server
npm run dev
```

### Python Dependencies Setup

#### Using Virtual Environment (Recommended)
```bash
# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac
source venv/bin/activate
# On Windows
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# For GPU support (if CUDA is installed)
pip install tensorflow-gpu faiss-gpu
```

### CUDA Setup for GPU Acceleration

#### 1. Check GPU Compatibility
First, verify your GPU is compatible with CUDA:
```bash
# For NVIDIA GPUs
nvidia-smi
```

#### 2. Install CUDA Toolkit and cuDNN
For TensorFlow 2.x:
- Install [CUDA Toolkit 11.2+](https://developer.nvidia.com/cuda-toolkit-archive)
- Install [cuDNN 8.1+](https://developer.nvidia.com/cudnn)

#### 3. Verify CUDA Installation
You can verify your CUDA installation:
```bash
# Check CUDA version
nvcc --version

# Check if TensorFlow can access the GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 4. Install TensorFlow with GPU Support
```bash
# Inside your activated virtual environment
pip install tensorflow
```

#### 5. Enable GPU Acceleration
The application will automatically detect and use your GPU if available.

#### 6. Troubleshooting
If you encounter issues with GPU acceleration:
- Ensure CUDA and cuDNN versions are compatible with your TensorFlow version
- Check TensorFlow logs for any specific device errors
- Try setting memory growth: `TF_FORCE_GPU_ALLOW_GROWTH=true`

#### Quick setup with npm scripts
```bash
# Create venv and install dependencies
npm run venv:setup

# Run the app with venv activated
npm run dev:venv
```

#### Manual Installation (Not Recommended)
```bash
# Install Python dependencies globally
pip install tensorflow librosa numpy pandas matplotlib scikit-learn faiss-cpu
```

### Model Setup
Ensure the trained models are in the `models/` directory:
- `optimized_cnn_model.keras`
- `metadata.npy`
- Additional models in `models/backup_models/`

## 🚀 Usage

1. **Start the Application**
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000)

2. **Upload Audio File**
   - Drag and drop or click to select an audio file
   - Supported formats: MP3, WAV, FLAC, M4A
   - Maximum file size: 50MB

3. **Analysis Process**
   - Audio preprocessing and feature extraction
   - Model inference with confidence scoring
   - Visualization generation (spectrograms, features)
   - Genre prediction with probability distribution

4. **View Results**
   - Primary genre prediction with confidence score
   - Detailed probability breakdown for all genres
   - Audio feature visualizations
   - Similar track recommendations (when available)

## 📊 Model Performance

### Genre Classification Accuracy
- **Overall Accuracy**: ~67% on validation set
- **Best Performing**: Hip-Hop (78% recall), International (77% precision)
- **Training Data**: 16,000+ balanced samples across 8 genres
- **Architecture**: CNN with SpecAugment, GroupNorm, and focal loss

### Recommendation Quality
- **MAP@10**: 0.60+ for triplet embedding model
- **Similarity Metrics**: Cosine similarity with L2 normalization
- **Search Speed**: <100ms for 10k+ track database with FAISS

## 🔧 Configuration

### Environment Variables
Create a `.env.local` file:
```env
# Optional: Configure model paths
MODEL_PATH=/path/to/models
AUDIO_UPLOAD_LIMIT=52428800  # 50MB

# GPU/CUDA Configuration
TF_FORCE_GPU_ALLOW_GROWTH=true  # Prevent TensorFlow from allocating all GPU memory
TF_GPU_THREAD_MODE=gpu_private  # Improve GPU thread management
```

### Model Configuration
Edit `app/api/process.py` to adjust:
- Segment length for audio processing
- Confidence thresholds
- Genre names and mappings
- Feature extraction parameters

## 📁 Project Structure

```
music-recommender/
├── app/
│   ├── api/                 # API routes and ML processing
│   │   ├── process.py       # Main audio processing pipeline
│   │   ├── predict/         # Prediction endpoint
│   │   └── check-models/    # Model status checking
│   ├── components/          # React components
│   │   ├── ClassificationVisualization.tsx
│   │   ├── DependencyStatus.tsx
│   │   └── GenreRadar.tsx
│   ├── utils/               # Utility functions
│   └── page.tsx             # Main application page
├── models/                  # Trained ML models
│   ├── optimized_cnn_model.keras
│   ├── metadata.npy
│   └── backup_models/
├── public/                  # Static assets
└── music_recommender_pipeline.ipynb  # Training notebook
```

## 🧠 Model Training

The project includes a comprehensive Jupyter notebook (`music_recommender_pipeline.ipynb`) for:

1. **Data Preparation**: FMA dataset processing and balancing
2. **Feature Extraction**: Mel spectrograms with librosa
3. **Data Augmentation**: SpecAugment for robustness
4. **Model Training**: CNN, autoencoder, and triplet networks
5. **Evaluation**: Confusion matrices, classification reports, MAP@k
6. **Embedding Analysis**: t-SNE visualization and similarity heatmaps

### Training New Models
```bash
# Open the training notebook
jupyter notebook music_recommender_pipeline.ipynb

# Follow the step-by-step process:
# 1. Data preprocessing
# 2. Model architecture definition
# 3. Training with callbacks
# 4. Evaluation and visualization
# 5. Model export
```

## 🐛 Troubleshooting

### Common Issues

**Models not loading**
- Check `models/` directory contains required files
- Verify Python dependencies are installed
- Check file permissions

**Audio processing errors**
- Ensure librosa is properly installed
- Verify audio file format is supported
- Check file isn't corrupted

**Memory issues**
- Reduce batch size in model configuration
- Use smaller audio segments
- Consider using CPU-only inference

**Dependency issues**
- Install exact package versions from requirements
- Use virtual environment to avoid conflicts

**GPU/CUDA issues**
- Verify CUDA and cuDNN are correctly installed and compatible with TensorFlow
- Check GPU memory usage with `nvidia-smi`
- Try disabling GPU with `CUDA_VISIBLE_DEVICES=-1`
- Update GPU drivers to the latest version
- Set `TF_FORCE_GPU_ALLOW_GROWTH=true` to prevent memory allocation issues

### Performance Optimization

**For better inference speed:**
- Use TensorFlow GPU if available
- Optimize model with TensorFlow Lite
- Implement model quantization
- Use FAISS GPU for large-scale search
- Set proper GPU memory allocation with `tf.config.experimental.set_memory_growth`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript/Python best practices
- Add tests for new features
- Update documentation
- Ensure models are compatible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FMA Dataset**: Free Music Archive for training data
- **Librosa**: Audio processing capabilities
- **TensorFlow**: Deep learning framework
- **FAISS**: Efficient similarity search
- **Next.js**: React framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the training notebook for ML details

---

Built with ❤️ for music lovers and AI enthusiasts.
