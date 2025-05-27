# 🎵 Music Genre Classifier & Recommender

An AI-powered music genre classification and recommendation system built with Next.js and TensorFlow. This application uses deep learning models to analyze audio files, predict their genres, and provide intelligent music recommendations.

## 🚀 Features

- **Real-time Audio Analysis**: Upload audio files and get instant genre predictions
- **Deep Learning Models**: Multiple CNN architectures including optimized classifiers, autoencoders, and triplet networks
- **Interactive Visualizations**: Comprehensive audio feature visualization including spectrograms, MFCC, and prediction confidence
- **Music Recommendations**: FAISS-powered similarity search for finding similar tracks
- **Comprehensive Genre Support**: Supports 8 major genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
- **Fallback System**: Graceful degradation with mock predictions when ML dependencies are unavailable
- **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS

## 🏗️ Architecture

### Frontend (Next.js)
- **React Components**: Interactive UI with real-time feedback
- **Audio Processing**: Client-side audio file handling and validation
- **Visualization**: D3.js-powered charts and spectrograms
- **Real-time Updates**: Dynamic progress tracking and status indicators

### Backend (Python/TensorFlow)
- **CNN Classifier**: Optimized convolutional neural network for genre classification
- **Autoencoder**: Dense embedding generation for music similarity
- **Triplet Network**: Discriminative embeddings for enhanced recommendations
- **Audio Processing**: Librosa-based feature extraction (spectrograms, MFCC, tempo, etc.)
- **FAISS Integration**: Efficient similarity search for recommendations

### Models
- **Primary Classifier**: `optimized_cnn_model.keras` - Main genre classification model
- **Triplet Encoder**: Enhanced embedding model for similarity matching
- **Autoencoder**: Dimensionality reduction and feature learning
- **Backup Models**: Fallback systems ensuring reliability

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **D3.js** - Data visualization
- **React Hooks** - State management

### Backend & ML
- **Python 3.11+** - Core ML processing
- **TensorFlow/Keras** - Deep learning models
- **Librosa** - Audio processing and feature extraction
- **NumPy/Pandas** - Data manipulation
- **Matplotlib** - Visualization generation
- **FAISS** - Efficient similarity search
- **Scikit-learn** - ML utilities

### Development Tools
- **ESLint** - Code linting
- **PostCSS** - CSS processing
- **Sharp** - Image optimization

## 📦 Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
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
```bash
# Install Python dependencies
pip install tensorflow librosa numpy pandas matplotlib scikit-learn faiss-cpu

# For GPU support (optional)
pip install tensorflow-gpu faiss-gpu
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

### Performance Optimization

**For better inference speed:**
- Use TensorFlow GPU if available
- Optimize model with TensorFlow Lite
- Implement model quantization
- Use FAISS GPU for large-scale search

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
