import sys
import os
import json
import random
from pathlib import Path
import keras
keras.config.enable_unsafe_deserialization()

# Suppress TensorFlow C++ logs (INFO, WARNING). Must be set before TensorFlow import.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow' # Explicitly set backend

# Helper to print to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Try importing libraries, but provide fallbacks if they're not available
DEPENDENCIES = {
    'librosa': False,
    'tensorflow': False,
    'faiss': False,
    'numpy': False,
    'matplotlib': False
}

# Set default value for DEPENDENCIES_AVAILABLE
DEPENDENCIES_AVAILABLE = False
ML_READY = False
SEARCH_READY = False
AUDIO_READY = False
EMBEDDINGS_AVAILABLE = False
MODELS_AVAILABLE = False

try:
    import numpy as np
    DEPENDENCIES['numpy'] = True
except ImportError:
    eprint("[WARNING] numpy not available")
    
try:
    import tensorflow as tf
    DEPENDENCIES['tensorflow'] = True
    # Set lower logging level for TensorFlow
    tf.get_logger().setLevel('ERROR')
except ImportError:
    eprint("[WARNING] TensorFlow not available")
    
try:
    import librosa
    import librosa.display
    DEPENDENCIES['librosa'] = True
except ImportError:
    eprint("[WARNING] librosa not available")
    
try:
    import matplotlib
    import matplotlib.pyplot as plt
    DEPENDENCIES['matplotlib'] = True
except ImportError:
    eprint("[WARNING] matplotlib not available")
    
try:
    import faiss
    DEPENDENCIES['faiss'] = True
except ImportError:
    eprint("[WARNING] FAISS not available")

# Set status flags based on dependencies
DEPENDENCIES_AVAILABLE = all([DEPENDENCIES['numpy'], DEPENDENCIES['tensorflow'], DEPENDENCIES['librosa']])
ML_READY = all([DEPENDENCIES['numpy'], DEPENDENCIES['tensorflow']])
SEARCH_READY = all([DEPENDENCIES['numpy'], DEPENDENCIES['faiss']])
AUDIO_READY = all([DEPENDENCIES['numpy'], DEPENDENCIES['librosa']])

eprint(f"[STATUS] Library availability: librosa={DEPENDENCIES['librosa']}, tensorflow={DEPENDENCIES['tensorflow']}, faiss={DEPENDENCIES['faiss']}, numpy={DEPENDENCIES['numpy']}")
eprint(f"[STATUS] Functionality: ML={ML_READY}, Search={SEARCH_READY}, Audio={AUDIO_READY}, Dependencies={DEPENDENCIES_AVAILABLE}")

def ensure_metadata_exists():
    """Ensure metadata file exists, creating a basic one if needed"""
    global metadata, embeddings
    
    if not DEPENDENCIES['numpy']:
        eprint("[WARNING] Cannot create metadata: NumPy missing")
        return False
    
    metadata_path = MODEL_DIR / "metadata.npy"
    
    # If metadata already exists and is loaded, nothing to do
    if metadata is not None:
        return True
        
    # Try to load existing metadata
    if metadata_path.exists():
        try:
            metadata = np.load(metadata_path, allow_pickle=True).item()
            eprint("[OK] Loaded existing metadata")
            return True
        except Exception as e:
            eprint(f"Error loading metadata: {e}")
    
    # Create basic metadata from embeddings
    if embeddings is not None:
        try:
            num_tracks = embeddings.shape[0]
            metadata = {
                "track_ids": np.arange(num_tracks),
                "genres": np.random.randint(0, len(GENRE_NAMES), num_tracks),
                "titles": {str(i): f"Track {i}" for i in range(num_tracks)},
                "genre_names": GENRE_NAMES
            }
            np.save(metadata_path, metadata)
            eprint(f"[OK] Created basic metadata for {num_tracks} tracks")
            return True
        except Exception as e:
            eprint(f"Error creating metadata: {e}")
    
    eprint("[WARNING] Cannot create metadata: embeddings missing")
    return False

# --- Custom TensorFlow layer definitions for model loading --- #
if DEPENDENCIES['tensorflow']:
    # SpecAugment for data augmentation
    @tf.keras.utils.register_keras_serializable(package="Custom", name="SpecAugment")
    class SpecAugment(tf.keras.layers.Layer):
        def __init__(self, max_f=25, max_t=40, **kwargs):
            super().__init__(**kwargs)
            self.max_f, self.max_t = max_f, max_t
            self.h, self.w = 128, 128
            self.freq_axis = tf.reshape(tf.range(self.h), (self.h, 1, 1))
            self.time_axis = tf.reshape(tf.range(self.w), (1, self.w, 1))

        def call(self, x, training=None):
            if not training:
                return x
            f  = tf.random.uniform([], 0, self.max_f + 1, tf.int32)
            f0 = tf.random.uniform([], 0, self.h - f + 1, tf.int32)
            t  = tf.random.uniform([], 0, self.max_t + 1, tf.int32)
            t0 = tf.random.uniform([], 0, self.w - t + 1, tf.int32)
            mask_f = tf.cast(~tf.logical_and(self.freq_axis >= f0,
                                            self.freq_axis < f0 + f), x.dtype)
            mask_t = tf.cast(~tf.logical_and(self.time_axis >= t0,
                                            self.time_axis < t0 + t), x.dtype)
            return x * mask_f * mask_t
            
        def get_config(self):
            config = super().get_config()
            config.update({
                "max_f": self.max_f,
                "max_t": self.max_t
            })
            return config

    # GroupNorm layer
    @tf.keras.utils.register_keras_serializable(package="Custom", name="GroupNorm")
    class GroupNorm(tf.keras.layers.Layer):
        def __init__(self, groups=8, epsilon=1e-5, **kwargs):
            super().__init__(**kwargs)
            self.groups, self.eps = groups, epsilon

        def build(self, inp_shape):
            self.gamma = self.add_weight(name="gamma", shape=(inp_shape[-1],),
                                        initializer="ones", trainable=True)
            self.beta  = self.add_weight(name="beta", shape=(inp_shape[-1],),
                                        initializer="zeros", trainable=True)

        def call(self, x):
            N, H, W, C = tf.shape(x)[0], *x.shape[1:]
            G = self.groups
            x = tf.reshape(x, [N, H, W, G, C // G])
            mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
            x = (x - mean) / tf.sqrt(var + self.eps)
            x = tf.reshape(x, [N, H, W, C])
            return x * self.gamma + self.beta
            
        def get_config(self):
            config = super().get_config()
            config.update({
                "groups": self.groups,
                "epsilon": self.eps
            })
            return config

    # Lambda wrapper for L2 normalization
    @tf.keras.utils.register_keras_serializable()
    class L2NormalizationLayer(tf.keras.layers.Layer):
        """Custom layer to replace the Lambda layer with L2 normalization"""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
        def call(self, inputs):
            # Import tensorflow explicitly inside the call method
            import tensorflow as tf  # Local import for serialization safety
            return tf.math.l2_normalize(inputs, axis=1)
            
        def get_config(self):
            return super().get_config()
    
    # Define a standalone function for Lambda layers that need l2_normalize
    def l2_normalize_fn(x):
        """Standalone function for Lambda layer to normalize vectors"""
        import tensorflow as tf  # Explicit import inside function
        return tf.math.l2_normalize(x, axis=1)

# Constants
SAMPLE_RATE = 22050
N_MELS = 128
MODEL_DIR = Path("models")
EMBEDDINGS_PATH = MODEL_DIR / "triplet_embeddings.npy"
GENRE_NAMES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

# Load models with custom handling for serialization issues
def load_models():
    global tf
    
    if not DEPENDENCIES_AVAILABLE:
        return None, None, None, None, None
        
    # Ensure model directory exists
    MODEL_DIR.mkdir(exist_ok=True)
    
    try:
        eprint("Loading models from:", MODEL_DIR)
        
        # Custom objects dictionary for loading models
        custom_objects = {}
        if DEPENDENCIES['tensorflow']:
            custom_objects = {
                "SpecAugment": SpecAugment,
                "GroupNorm": GroupNorm,
                "L2NormalizationLayer": L2NormalizationLayer,
                "l2_normalize_fn": l2_normalize_fn
            }
          # CNN Classifier model - load with special handling
        classifier_model = None
        classifier_path = MODEL_DIR / "optimized_cnn_model.keras"
        if classifier_path.exists() and DEPENDENCIES['tensorflow']:
            try:
                # Load without compilation to avoid loss function issues
                classifier_model = tf.keras.models.load_model(classifier_path, compile=False, custom_objects=custom_objects)
                eprint("[OK] Classifier model loaded without compilation")
            except Exception as e:
                eprint(f"Classifier loading failed: {e}")
                # Fallback: try with minimal custom objects
                try:
                    minimal_objects = {"SpecAugment": SpecAugment, "GroupNorm": GroupNorm}
                    classifier_model = tf.keras.models.load_model(classifier_path, compile=False, custom_objects=minimal_objects)
                    eprint("[OK] Classifier model loaded with minimal custom objects")
                except Exception as e2:
                    eprint(f"Minimal loading also failed: {e2}")
        else:
            eprint("[WARNING] Classifier model not found or tensorflow not available")
        
        # Autoencoder (encoder part) for embeddings
        encoder_model = None
        encoder_path = MODEL_DIR / "auto_encoder.keras"
        if encoder_path.exists() and DEPENDENCIES['tensorflow']:
            try:
                encoder_model = tf.keras.models.load_model(encoder_path, compile=False, custom_objects=custom_objects)
                eprint("[OK] Autoencoder model loaded")
            except Exception as e:
                eprint(f"Error loading autoencoder: {e}")
        else:
            eprint("[INFO] Autoencoder path does not exist or tensorflow not available")
        
        # Load triplet encoder
        triplet_encoder = None
        triplet_path = MODEL_DIR / "base_encoder.keras"
        if triplet_path.exists() and DEPENDENCIES['tensorflow']:
            try:
                triplet_encoder = tf.keras.models.load_model(triplet_path, compile=False, custom_objects=custom_objects)
                eprint("[OK] Triplet encoder loaded")
            except Exception as e:
                eprint(f"Error loading triplet encoder: {e}")
                # Try again with a different approach
                try:
                    eprint("Attempting to load triplet encoder with basic layers only...")
                    triplet_encoder = tf.keras.models.load_model(
                        triplet_path, 
                        compile=False,
                        # Less strict loading
                        custom_objects=custom_objects
                    )
                    eprint("[OK] Triplet encoder loaded with alternative method")
                except Exception as e2:
                    eprint(f"All attempts to load triplet encoder failed: {e2}")
        else:
            eprint("[INFO] Triplet encoder path does not exist or tensorflow not available")
        
        # Load embeddings
        embeddings = None
        if EMBEDDINGS_PATH.exists():
            try:
                embeddings = np.load(EMBEDDINGS_PATH)
                eprint(f"[OK] Embeddings loaded: {embeddings.shape}")
            except Exception as e:
                eprint(f"Error loading embeddings: {e}")
        else:
            eprint("[INFO] Embeddings file does not exist")
            
        # Load or create metadata
        metadata = None
        metadata_path = MODEL_DIR / "metadata.npy"
        if metadata_path.exists():
            try:
                metadata = np.load(metadata_path, allow_pickle=True).item()
                eprint("[OK] Metadata loaded")
            except Exception as e:
                eprint(f"Error loading metadata: {e}")
        elif embeddings is not None and DEPENDENCIES['numpy']:            
            # Create basic metadata if not found
            try:
                num_tracks = embeddings.shape[0]
                metadata = {
                    "track_ids": np.arange(num_tracks),
                    "genres": np.random.randint(0, len(GENRE_NAMES), num_tracks),
                    "titles": {str(i): f"Track {i}" for i in range(num_tracks)},
                    "genre_names": GENRE_NAMES
                }
                np.save(metadata_path, metadata)
                eprint("[OK] Metadata created and saved")
            except Exception as e:
                eprint(f"Error creating metadata: {e}")
        else:
            eprint("[WARNING] Metadata not available and can't be created")
        
        return classifier_model, encoder_model, triplet_encoder, None, metadata  # Return None for index, we'll create it separately
        
    except Exception as e:
        eprint(f"Error during model loading process: {e}")
        return None, None, None, None, None

# --- Embedding normalization and FAISS index creation --- #
def load_embeddings_and_index():
    if not EMBEDDINGS_PATH.exists():
        eprint(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        return None, None
    embeddings = np.load(EMBEDDINGS_PATH)
    eprint(f"Loaded embeddings shape: {embeddings.shape}")
    # Print norms before normalization
    norms_before = np.linalg.norm(embeddings, axis=1)
    eprint(f"Norms before normalization: min={norms_before.min():.6f}, max={norms_before.max():.6f}, mean={norms_before.mean():.6f}")
    # L2-normalize embeddings
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    # Print norms after normalization
    norms_after = np.linalg.norm(embeddings, axis=1)
    eprint(f"Norms after normalization: min={norms_after.min():.6f}, max={norms_after.max():.6f}, mean={norms_after.mean():.6f}")
    # Build FAISS index (IndexFlatIP for cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    eprint(f"FAISS index created with {index.ntotal} vectors (dim={dim})")
    return embeddings, index

# Initialize models and other resources
classifier_model = None
encoder_model = None
triplet_encoder = None
metadata = None
embeddings = None
index = None

def ensure_models_initialized():
    global classifier_model, encoder_model, triplet_encoder, metadata, embeddings, index, SEARCH_READY, EMBEDDINGS_AVAILABLE, MODELS_AVAILABLE
    
    # If the models aren't loaded yet, try to load them
    if classifier_model is None or triplet_encoder is None:
        classifier_model, encoder_model, triplet_encoder, _, metadata = load_models()
    
    # If embeddings and index aren't loaded yet, try to load them
    if embeddings is None or index is None:
        embeddings, index = load_embeddings_and_index()
    
    # Create metadata if it doesn't exist
    if metadata is None and embeddings is not None:
        ensure_metadata_exists()
    
    # Set status flags
    EMBEDDINGS_AVAILABLE = embeddings is not None
    MODELS_AVAILABLE = classifier_model is not None or triplet_encoder is not None
    
    # Set SEARCH_READY flag based on successful loading of index and embeddings
    if index is not None and embeddings is not None and metadata is not None and DEPENDENCIES['faiss'] and DEPENDENCIES['numpy']:
        SEARCH_READY = True
        
    return (classifier_model is not None, encoder_model is not None, 
            triplet_encoder is not None, index is not None, metadata is not None)

if DEPENDENCIES_AVAILABLE:
    ensure_models_initialized()

def recommend_similar_tracks(query_embedding, top_k=5):
    # Ensure query embedding is L2-normalized
    query = np.array(query_embedding, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query)
    # Search index
    D, I = index.search(query, top_k)
    # Print top similarity scores for debug
    eprint(f"Top similarity scores: {D[0]}")
    return I[0], D[0]

def audio_to_melspec(audio_path, fixed_length=128):
    """Convert audio file to mel spectrogram with fixed length"""
    if not DEPENDENCIES['librosa'] or not DEPENDENCIES['numpy']:
        eprint("[WARNING] Cannot process audio: librosa or numpy missing")
        return None
        
    try:
        # Use Path to handle Windows paths properly
        path_obj = Path(audio_path)
        y, sr = librosa.load(path_obj, sr=SAMPLE_RATE, mono=True)
        
        # Mel spectrogram extraction
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS)
        logmel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [0,1]
        logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-6)
        
        # Fix length
        logmel = librosa.util.fix_length(logmel, size=fixed_length, axis=1)
        
        return logmel[..., np.newaxis]  # Add channel dimension
    except Exception as e:
        eprint(f"Error processing audio: {e}")
        return None

def extract_segments(audio_path, segment_seconds=30, overlap=0.5):
    """Extract overlapping segments from an audio file"""
    if not DEPENDENCIES['librosa']:
        eprint("[WARNING] Cannot extract segments: librosa missing")
        return None
    
    try:
        # Load full audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Calculate segment length and hop size in samples
        segment_len = int(sr * segment_seconds)
        hop = int(segment_len * (1 - overlap))
        
        # Extract segments
        segments = []
        for start in range(0, len(y), hop):
            seg = y[start:start+segment_len]
            
            # Pad if segment is shorter than required length
            if len(seg) < segment_len:
                seg = np.pad(seg, (0, segment_len - len(seg)), mode="constant")
                
            # Convert segment to mel spectrogram
            mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS)
            logmel = librosa.power_to_db(mel, ref=np.max)
            logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-6)
            logmel = librosa.util.fix_length(logmel, size=128, axis=1)
            
            segments.append(logmel[..., np.newaxis])
            
        return segments
    except Exception as e:
        eprint(f"Error extracting segments: {e}")
        return None

def predict_genre(audio_path, segment_seconds=30, overlap=0.5, min_segments=1):
    """Predict genre from audio file using segment-based approach"""
    global tf
    
    if not DEPENDENCIES_AVAILABLE or not ML_READY or classifier_model is None:
        eprint("[WARNING] Using mock predictions - one or more components unavailable")
        return mock_predict(audio_path)
    
    try:
        # Extract segments from the audio file
        segments = extract_segments(audio_path, segment_seconds, overlap)
        
        if segments is None or len(segments) < min_segments:
            eprint(f"[WARNING] Not enough valid segments (got {0 if segments is None else len(segments)}), using full audio")
            mel = audio_to_melspec(audio_path)
            if mel is None:
                eprint("[WARNING] Could not process audio, using mock predictions")
                return mock_predict(audio_path)
            segments = [mel]
        
        # Make predictions for each segment
        probs_list = []
        for segment in segments:
            segment_batch = np.expand_dims(segment, axis=0)  # Add batch dimension
            try:
                segment_probs = classifier_model.predict(segment_batch, verbose=0)[0]
                probs_list.append(segment_probs)
            except Exception as e:
                eprint(f"Error predicting segment: {e}")
                continue
        
        if not probs_list:
            eprint("[WARNING] No valid predictions from segments, using mock predictions")
            return mock_predict(audio_path)
        
        # Average probabilities across all segments
        avg_probs = np.mean(probs_list, axis=0)
        
        # Get predicted genre and confidence
        pred_idx = np.argmax(avg_probs)
        confidence = float(avg_probs[pred_idx])
        
        # Return results
        return {
            "genre_probabilities": {genre: float(prob) for genre, prob in zip(GENRE_NAMES, avg_probs)},
            "predicted_genre": GENRE_NAMES[pred_idx],
            "confidence": confidence
        }
    except Exception as e:
        eprint(f"Error during genre prediction: {e}")
        return mock_predict(audio_path)

def predict_genre_with_visualization(audio_path, segment_seconds=30, overlap=0.5, min_segments=1):
    """Predict genre from audio file with comprehensive visualization data"""
    global tf
    
    if not DEPENDENCIES_AVAILABLE or not ML_READY or classifier_model is None:
        eprint("[WARNING] Using mock predictions with mock visualization - one or more components unavailable")
        result = mock_predict(audio_path)
        result['visualization_data'] = generate_mock_visualization()
        return result
    
    try:
        # Load audio for analysis
        y, sr = librosa.load(audio_path, sr=None)
        eprint(f"[INFO] Loaded audio: {len(y)} samples at {sr} Hz")
        
        # Generate visualization data
        visualization_data = {}
        
        # 1. Generate spectrograms
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Raw spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Linear-frequency power spectrogram')
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            spectrogram_b64 = base64.b64encode(buffer.getvalue()).decode()
            visualization_data['spectrogram'] = spectrogram_b64
            plt.close()
            
            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-frequency spectrogram')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            mel_spectrogram_b64 = base64.b64encode(buffer.getvalue()).decode()
            visualization_data['mel_spectrogram'] = mel_spectrogram_b64
            plt.close()
            
        except Exception as e:
            eprint(f"[WARNING] Could not generate spectrograms: {e}")
        
        # 2. Extract detailed audio features
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            visualization_data['mfcc_features'] = mfcc.tolist()
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            visualization_data['chroma_features'] = chroma.tolist()
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            visualization_data['spectral_features'] = {
                'spectral_centroid': spectral_centroids.tolist(),
                'spectral_rolloff': spectral_rolloff.tolist(),
                'zero_crossing_rate': zero_crossing_rate.tolist()
            }
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            visualization_data['tempo'] = float(tempo)
            
        except Exception as e:
            eprint(f"[WARNING] Could not extract audio features: {e}")
        
        # 3. Perform genre prediction (reuse existing logic)
        segments = extract_segments(audio_path, segment_seconds, overlap)
        
        if segments is None or len(segments) < min_segments:
            eprint(f"[WARNING] Not enough valid segments (got {0 if segments is None else len(segments)}), using full audio")
            mel = audio_to_melspec(audio_path)
            if mel is None:
                eprint("[WARNING] Could not process audio, using mock predictions")
                result = mock_predict(audio_path)
                result['visualization_data'] = visualization_data
                return result
            segments = [mel]
        
        # Make predictions for each segment
        probs_list = []
        raw_predictions = []
        
        for segment in segments:
            segment_batch = np.expand_dims(segment, axis=0)  # Add batch dimension
            try:
                segment_probs = classifier_model.predict(segment_batch, verbose=0)[0]
                probs_list.append(segment_probs)
                raw_predictions.append(segment_probs.tolist())
            except Exception as e:
                eprint(f"Error predicting segment: {e}")
                continue
        
        if not probs_list:
            eprint("[WARNING] No valid predictions from segments, using mock predictions")
            result = mock_predict(audio_path)
            result['visualization_data'] = visualization_data
            return result
        
        # Average probabilities across all segments
        avg_probs = np.mean(probs_list, axis=0)
        
        # Add raw model predictions to visualization
        visualization_data['model_predictions'] = avg_probs.tolist()
        
        # Get predicted genre and confidence
        pred_idx = np.argmax(avg_probs)
        confidence = float(avg_probs[pred_idx])
        
        # Return results with visualization data
        result = {
            "genre_probabilities": {genre: float(prob) for genre, prob in zip(GENRE_NAMES, avg_probs)},
            "predicted_genre": GENRE_NAMES[pred_idx],
            "confidence": confidence,
            "visualization_data": visualization_data
        }
        
        return result
        
    except Exception as e:
        eprint(f"Error during genre prediction with visualization: {e}")
        result = mock_predict(audio_path)
        result['visualization_data'] = generate_mock_visualization()
        return result

def generate_mock_visualization():
    """Generate mock visualization data for development/fallback"""
    import random
    
    # Generate mock MFCC features
    mfcc_features = []
    for i in range(13):
        mfcc_features.append([random.uniform(-1, 1) for _ in range(100)])
    
    # Generate mock spectral features
    spectral_features = {
        'spectral_centroid': [random.uniform(1000, 3000) for _ in range(100)],
        'spectral_rolloff': [random.uniform(2000, 5000) for _ in range(100)],
        'zero_crossing_rate': [random.uniform(0.01, 0.1) for _ in range(100)]
    }
    
    # Generate mock model predictions
    model_predictions = [random.uniform(0, 1) for _ in range(8)]
    # Normalize to sum to 1
    total = sum(model_predictions)
    model_predictions = [p / total for p in model_predictions]
    
    return {
        'mfcc_features': mfcc_features,
        'spectral_features': spectral_features,
        'tempo': random.uniform(60, 180),
        'model_predictions': model_predictions
    }

def get_recommendations(audio_path, top_k=5):
    """Get similar song recommendations based purely on audio embedding similarity"""
    # Make sure models are initialized
    status = ensure_models_initialized()
    classifier_ok, encoder_ok, triplet_ok, index_ok, metadata_ok = status
    
    # This condition should now correctly reflect if triplet_encoder loaded
    if not DEPENDENCIES_AVAILABLE or not SEARCH_READY or not triplet_ok or not index_ok or not metadata_ok:
        eprint("[WARNING] Using mock recommendations - one or more components unavailable.")
        eprint(f"  Dependencies available: {DEPENDENCIES_AVAILABLE}")
        eprint(f"  Search ready (Faiss & NumPy): {SEARCH_READY}")
        eprint(f"  Triplet encoder loaded: {triplet_ok}")
        eprint(f"  FAISS index loaded: {index_ok}")
        eprint(f"  Metadata loaded: {metadata_ok}")
        return mock_recommendations(audio_path, top_k)
        
    mel = audio_to_melspec(audio_path)
    if mel is None:
        eprint("[WARNING] Could not process audio, using mock recommendations")
        return mock_recommendations(audio_path, top_k)
        
    try:
        # Get embedding from triplet encoder
        mel = mel.reshape(1, *mel.shape)
        
        # Get embedding for query audio - use only triplet encoder, no need for genre classification
        try:
            # First attempt with the original model
            query_embedding = triplet_encoder.predict(mel, verbose=0)
        except Exception as e:
            eprint(f"Error during triplet encoder prediction: {e}")
            eprint("Trying workaround for Lambda layer issue...")
            
            # Use a completely rebuilt model approach first
            try:
                eprint("Rebuilding model with safe Lambda layer...")
                # Create a new model with the same architecture but replace Lambda with L2NormalizationLayer
                inputs = triplet_encoder.input
                x = inputs
                for layer in triplet_encoder.layers[1:-1]:  # Skip input and Lambda layer
                    x = layer(x)
                
                # Add L2 normalization explicitly
                outputs = L2NormalizationLayer()(x)
                safe_model = tf.keras.models.Model(inputs, outputs)
                
                # Get embedding from safe model
                query_embedding = safe_model.predict(mel, verbose=0)
                eprint("[OK] Rebuilt model worked successfully")
            except Exception as e2:
                eprint(f"Model rebuild failed: {e2}")
                
                # Try the layer extraction workaround as a last resort
                try:
                    # Find last non-Lambda layer
                    last_regular_layer = None
                    for layer in triplet_encoder.layers:
                        if not isinstance(layer, tf.keras.layers.Lambda):
                            last_regular_layer = layer
                    
                    if last_regular_layer:
                        # Create intermediate model up to the last regular layer
                        temp_model = tf.keras.models.Model(
                            inputs=triplet_encoder.input,
                            outputs=last_regular_layer.output
                        )
                        # Get embedding and normalize manually
                        query_embedding = temp_model.predict(mel, verbose=0)
                        
                        # Manual L2 normalization that doesn't rely on TensorFlow's high-level API
                        norm = np.sqrt(np.sum(np.square(query_embedding), axis=1, keepdims=True))
                        query_embedding = query_embedding / (norm + 1e-10)
                        eprint("[OK] Layer extraction workaround successful")
                    else:
                        eprint("Could not find suitable layer for extraction")
                        return mock_recommendations(audio_path, top_k)
                except Exception as e3:
                    eprint(f"All workarounds failed: {e3}")
                    return mock_recommendations(audio_path, top_k)
        
        # Normalize for cosine similarity
        query_embedding = query_embedding.astype('float32')
        
        # Manual normalization to ensure unit vectors with same scale as notebook
        norm = np.sqrt(np.sum(np.square(query_embedding), axis=1, keepdims=True))
        if norm.min() > 0:  # Avoid division by zero
            query_embedding = query_embedding / norm
        else:
            eprint("[WARNING] Zero norm vector detected")
            
        # Verify embedding quality by checking norm - should be very close to 1.0
        embedding_norm = np.linalg.norm(query_embedding)
        eprint(f"[DEBUG] Query embedding norm: {embedding_norm:.6f} (should be ~1.0)")
        
        # Explicitly normalize with FAISS to match notebook approach
        faiss.normalize_L2(query_embedding)
        
        # Use the dedicated recommend_similar_tracks function
        indices, similarities = recommend_similar_tracks(query_embedding, top_k+15)
        
        # Format results without genre-based sorting
        recommendations = []
        idx_used = set()  # To avoid duplicate recommendations
        
        # Simply collect all unique recommendations with their similarity scores
        for i, (idx, score) in enumerate(zip(indices, similarities)):
            if idx in idx_used:
                continue
                
            if metadata and "track_ids" in metadata and idx < len(metadata["track_ids"]):
                track_id = metadata["track_ids"][idx]
                genre_idx = metadata.get("genres", [])[idx] if idx < len(metadata.get("genres", [])) else 0
                genre = GENRE_NAMES[genre_idx] if genre_idx < len(GENRE_NAMES) else "Unknown"
                title = metadata.get("titles", {}).get(str(track_id), f"Track {track_id}")
                
                # Use similarity score from FAISS
                recommendations.append({
                    "id": str(track_id),
                    "title": title,
                    "genre": genre,  # Keep genre info for display only
                    "similarity": float(score)  # Ensure it's a Python float
                })
                idx_used.add(idx)
                
                # Stop when we have enough recommendations
                if len(recommendations) >= top_k:
                    break
        
        # Print stats about recommendations
        eprint(f"[INFO] Found {len(recommendations)} recommendations based on pure audio similarity")
        if len(recommendations) > 0:
            eprint(f"[INFO] Similarity range: {recommendations[0]['similarity']:.4f} - {recommendations[-1]['similarity']:.4f}")
            genres_found = set(r['genre'] for r in recommendations)
            eprint(f"[INFO] Genres in results: {', '.join(genres_found)}")
        
        return recommendations
    except Exception as e:
        eprint(f"Error during recommendation: {e}")
        return mock_recommendations(audio_path, top_k)

# Mock functions that don't depend on unavailable libraries
def mock_predict(audio_path):
    """Generate mock genre predictions for when real prediction is unavailable"""
    # Extract filename to get some variability
    filename = str(Path(audio_path).name)
    # Simple hash of filename for deterministic but varied mock results
    file_hash = sum(ord(c) for c in filename)
    
    # Seed the RNG with the filename hash for deterministic results
    random.seed(file_hash)
    
    # Generate random probabilities
    raw_probs = [random.random() for _ in range(len(GENRE_NAMES))]
    # Normalize to sum to 1
    total = sum(raw_probs)
    probs = [p / total for p in raw_probs]
    
    # Pick the highest probability as the predicted genre
    max_idx = probs.index(max(probs))
    max_prob = probs[max_idx]
    
    return {
        "genre_probabilities": {genre: float(prob) for genre, prob in zip(GENRE_NAMES, probs)},
        "predicted_genre": GENRE_NAMES[max_idx],
        "confidence": float(max_prob),
        "using_mock": True
    }

def mock_recommendations(audio_path, top_k=5):
    """Generate mock recommendations when real recommendations are unavailable"""
    # Get mock genre prediction to make mock recommendations consistent with predicted genre
    pred = mock_predict(audio_path)
    primary_genre = pred["predicted_genre"]
    
    # Deterministic seed based on filename
    filename = str(Path(audio_path).name)
    file_hash = sum(ord(c) for c in filename)
    random.seed(file_hash)
    
    # Generate mock recommendations
    recommendations = []
    
    for i in range(top_k):
        # 70% chance of matching the predicted primary genre
        if random.random() < 0.7:
            genre = primary_genre
        else:
            # Otherwise pick a random genre
            genre = random.choice(GENRE_NAMES)
        
        # Generate decreasing similarity scores
        similarity = 0.95 - (i * 0.05) 
        
        recommendations.append({
            "id": str(10000 + i),
            "title": f"Sample Track {i+1}",
            "genre": genre, 
            "similarity": similarity,
            "using_mock": True
        })
    
    return recommendations

# Enable UTF-8 output in all environments
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def output_json(obj):
    """Output JSON with a clear marker for parsing and add dependency status."""
    # Add dependency status to the object if it's a dict or the first item of a list
    status_payload = {
        "libraries": DEPENDENCIES,
        "functionality": {
            "ml_ready": ML_READY,
            "search_ready": SEARCH_READY,
            "audio_ready": AUDIO_READY,
            "embeddings_available": EMBEDDINGS_AVAILABLE,
            "models_available": MODELS_AVAILABLE
        }
    }
    if isinstance(obj, dict):
        obj['dependency_status'] = status_payload
    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        obj[0]['dependency_status'] = status_payload
    elif isinstance(obj, list) and len(obj) == 0: # Handle empty list case
        obj = {"data": [], "dependency_status": status_payload}

    # This is the only place printing to STDOUT for API calls
    sys.stdout.write("\n===JSON_START===\n") # Use sys.stdout.write for more control
    sys.stdout.write(json.dumps(obj))
    sys.stdout.write("\n===JSON_END===\n")
    sys.stdout.flush()

if __name__ == "__main__":
    # Test code to run when executing this script directly
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Processing {audio_path}...")
        
        if len(sys.argv) > 2 and sys.argv[2] == "recommend":
            result = get_recommendations(audio_path)
            print(json.dumps(result, indent=2))
        else:
            result = predict_genre(audio_path)
            print(json.dumps(result, indent=2))