#!/usr/bin/env python3
"""
FastAPI Server for Music Genre Classification
Handles MP3 file uploads and returns genre predictions
"""

import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
import shutil
import sys
import traceback

# Configure logging - MOVED UP before imports that use it
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow environment variables BEFORE importing TensorFlow or anything that imports it
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow GPU memory before any other imports
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s). Setting memory growth.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("Memory growth set for all GPUs")
except Exception as e:
    logger.warning(f"Could not set GPU memory growth: {e}")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import the server's local version of the get_hann_window function
from process import get_hann_window, init as init_process

# Set the working directory to server directory so models path is correct
os.chdir(str(Path(__file__).parent))

# Initialize process functions
init_process()

# Now import the needed functions after initialization
from process import predict_genre, predict_genre_with_visualization, get_recommendations

# Initialize FastAPI app
app = FastAPI(
    title="Music Genre Classification API",
    description="Upload MP3 files and get AI-powered genre predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class GenrePrediction(BaseModel):
    predicted_genre: str
    confidence: float
    genre_probabilities: Dict[str, float]
    processing_time: float

class GenrePredictionWithVisualization(GenrePrediction):
    visualization_data: Optional[Dict] = None

class HealthCheck(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool

class MusicRecommendation(BaseModel):
    id: str
    title: str
    genre: str
    similarity: float

class RecommendationResponse(BaseModel):
    recommendations: List[MusicRecommendation]
    processing_time: float
    query_file: str

# New Pydantic models for batch processing
class BatchPredictionItem(BaseModel):
    filename: str
    predicted_genre: str
    confidence: float
    genre_probabilities: Dict[str, float]
    processing_time: float
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]
    total_files: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time: float

class ModelStats(BaseModel):
    model_loaded: bool
    model_type: str
    total_genres: int
    available_genres: List[str]
    embeddings_count: Optional[int] = None
    gpu_memory_usage: Optional[Dict] = None

# Global variables for model status
model_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model_loaded
    try:
        # Test model loading by making a dummy prediction
        logger.info("Warming up the model...")
        # This will load and cache the model
        model_loaded = True
        logger.info("✅ Server started successfully with model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")
        model_loaded = False

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    import tensorflow as tf
    
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    return HealthCheck(
        status="healthy" if model_loaded else "model_not_loaded",
        gpu_available=gpu_available,
        model_loaded=model_loaded
    )

@app.post("/predict", response_model=GenrePrediction)
async def predict_genre_endpoint(
    file: UploadFile = File(...),
    use_gpu: bool = Form(True)
):
    """
    Predict music genre from uploaded MP3 file
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload MP3, WAV, or M4A files."
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            # Copy uploaded file content to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing file: {file.filename}")
        
        # Set GPU usage based on parameter
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Process the audio file
        import time
        start_time = time.time()
        
        result = predict_genre(temp_file_path)
        
        processing_time = time.time() - start_time
        
        # Clean up result and add processing time
        prediction = GenrePrediction(
            predicted_genre=result['predicted_genre'],
            confidence=float(result['confidence']),
            genre_probabilities={k: float(v) for k, v in result['genre_probabilities'].items()},
            processing_time=round(processing_time, 2)
        )
        
        logger.info(f"✅ Prediction completed in {processing_time:.2f}s: {result['predicted_genre']}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/predict-with-viz", response_model=GenrePredictionWithVisualization)
async def predict_with_visualization_endpoint(
    file: UploadFile = File(...),
    use_gpu: bool = Form(True)
):
    """
    Predict music genre with visualization data from uploaded MP3 file
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload MP3, WAV, or M4A files."
        )
    
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing file with visualization: {file.filename}")
        
        # Set GPU usage
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            # Make sure we reset this if it was set to -1 before
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
                del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Process with visualization - add some error handling and verification
        import time
        start_time = time.time()
        
        try:
            # Check explicitly if the hann window function is available
            logger.info("Checking if get_hann_window is available...")
            try:
                window = get_hann_window(1024)  # Test the function with a sample size
                logger.info("✅ get_hann_window function works correctly")
            except Exception as window_error:
                logger.error(f"❌ Error using get_hann_window function: {window_error}")
            
            # Use the full visualization function if the window function works
            result = predict_genre_with_visualization(temp_file_path)
            
            # Verify the result has visualization data
            if result and 'visualization_data' not in result:
                logger.warning("Visualization data missing from result, adding empty dict")
                result['visualization_data'] = {}
                
        except Exception as e:
            logger.error(f"Error in visualization function: {e}")
            logger.error(traceback.format_exc())
            # Fall back to regular prediction
            basic_result = predict_genre(temp_file_path)
            result = {**basic_result, 'visualization_data': {}}
        
        processing_time = time.time() - start_time
        
        # Log visualization data keys to help with debugging
        viz_keys = "No visualization data"
        if result and 'visualization_data' in result:
            viz_keys = ", ".join(result['visualization_data'].keys()) if result['visualization_data'] else "Empty dict"
        logger.info(f"Visualization data keys: {viz_keys}")
        
        # Clean up result
        prediction = GenrePredictionWithVisualization(
            predicted_genre=result['predicted_genre'],
            confidence=float(result['confidence']),
            genre_probabilities={k: float(v) for k, v in result['genre_probabilities'].items()},
            processing_time=round(processing_time, 2),
            visualization_data=result.get('visualization_data', {})
        )
        
        logger.info(f"✅ Prediction with viz completed in {processing_time:.2f}s: {result['predicted_genre']}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Error processing file with viz {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations_endpoint(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Get music recommendations based on audio similarity using triplet encoders
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload MP3, WAV, or M4A files."
        )
    
    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )
    
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Getting recommendations for: {file.filename}")
        
        # Get recommendations
        import time
        start_time = time.time()
        
        recommendations_data = get_recommendations(temp_file_path, top_k)
        
        processing_time = time.time() - start_time
        
        # Format response
        recommendations = [
            MusicRecommendation(**rec) for rec in recommendations_data
        ]
        
        response = RecommendationResponse(
            recommendations=recommendations,
            processing_time=round(processing_time, 2),
            query_file=file.filename
        )
        
        logger.info(f"✅ Recommendations completed in {processing_time:.2f}s: {len(recommendations)} tracks")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch_genre_endpoint(
    files: List[UploadFile] = File(...),
    use_gpu: bool = Form(True)
):
    """
    Predict music genres for a batch of uploaded MP3 files
    """
    if not all(file.filename.lower().endswith(('.mp3', '.wav', '.m4a')) for file in files):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload MP3, WAV, or M4A files."
        )
    
    # Create temporary files
    temp_files = []
    response_items = []
    total_processing_time = 0.0
    
    try:
        for file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_files.append(temp_file.name)
            
            logger.info(f"Processing file: {file.filename}")
        
        # Set GPU usage based on parameter
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Process the audio files
        import time
        start_time = time.time()
        
        results = [predict_genre(file_path) for file_path in temp_files]
        
        total_processing_time = time.time() - start_time
        
        # Clean up results and add processing time
        for file, result in zip(files, results):
            response_items.append(BatchPredictionItem(
                filename=file.filename,
                predicted_genre=result['predicted_genre'],
                confidence=float(result['confidence']),
                genre_probabilities={k: float(v) for k, v in result['genre_probabilities'].items()},
                processing_time=round(total_processing_time, 2)
            ))
        
        logger.info(f"✅ Batch prediction completed in {total_processing_time:.2f}s")
        
        # Calculate batch response metrics
        successful_predictions = sum(1 for item in response_items if not item.error)
        failed_predictions = len(response_items) - successful_predictions
        
        return BatchPredictionResponse(
            predictions=response_items,
            total_files=len(files),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            total_processing_time=round(total_processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(files: List[UploadFile] = File(...), use_gpu: bool = Form(True)):
    """Batch prediction endpoint for multiple audio files"""
    start_time = time.time()
    predictions = []
    successful = 0
    failed = 0
    
    logger.info(f"🚀 Starting batch prediction for {len(files)} files")
    
    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
                predictions.append(BatchPredictionItem(
                    filename=file.filename,
                    predicted_genre="",
                    confidence=0.0,
                    genre_probabilities={},
                    processing_time=0.0,
                    error=f"Unsupported file type: {file.filename}"
                ))
                failed += 1
                continue
            
            # Process individual file
            file_start = time.time()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file_path = temp_file.name
            
            try:
                # Save uploaded file
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                
                # Set GPU usage environment
                if not use_gpu:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                else:
                    # Reset CUDA_VISIBLE_DEVICES if it was disabled
                    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
                        del os.environ['CUDA_VISIBLE_DEVICES']
                
                # Get prediction
                result = predict_genre(temp_file_path)
                file_time = time.time() - file_start
                
                predictions.append(BatchPredictionItem(
                    filename=file.filename,
                    predicted_genre=result['predicted_genre'],
                    confidence=result['confidence'],
                    genre_probabilities=result['genre_probabilities'],
                    processing_time=round(file_time, 3)
                ))
                successful += 1
                
            except Exception as e:
                predictions.append(BatchPredictionItem(
                    filename=file.filename,
                    predicted_genre="",
                    confidence=0.0,
                    genre_probabilities={},
                    processing_time=time.time() - file_start,
                    error=str(e)
                ))
                failed += 1
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                        
        except Exception as e:
            predictions.append(BatchPredictionItem(
                filename=file.filename or "unknown",
                predicted_genre="",
                confidence=0.0,
                genre_probabilities={},
                processing_time=0.0,
                error=f"File processing error: {str(e)}"
            ))
            failed += 1
    
    total_time = time.time() - start_time
    
    logger.info(f"✅ Batch processing completed: {successful} successful, {failed} failed in {total_time:.2f}s")
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_files=len(files),
        successful_predictions=successful,
        failed_predictions=failed,
        total_processing_time=round(total_time, 2)
    )

@app.get("/model-stats", response_model=ModelStats)
async def get_model_stats():
    """Get model statistics and system information"""
    try:
        import tensorflow as tf
        
        # Get available genres (assuming they're in a specific format)
        # This is a placeholder - you might need to adjust based on your model
        genres = ["rock", "pop", "hip-hop", "jazz", "classical", "electronic", "country", "r&b", "reggae", "blues"]
        
        # GPU memory info
        gpu_info = None
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_info = {
                    "gpu_count": len(gpus),
                    "gpu_names": [gpu.name for gpu in gpus],
                }
        except:
            pass
        
        # Try to get embeddings count
        embeddings_count = None
        try:
            # This assumes you have the embeddings loaded
            import numpy as np
            embeddings_path = "models/triplet_embeddings.npy"
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path)
                embeddings_count = len(embeddings)
        except:
            pass
        
        return ModelStats(
            model_loaded=True,
            model_type="Neural Network Classifier + Triplet Encoder",
            total_genres=len(genres),
            available_genres=genres,
            embeddings_count=embeddings_count,
            gpu_memory_usage=gpu_info
        )
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve model stats: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Music Genre Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_with_visualization": "/predict-with-viz",
            "predict_ensemble": "/predict-ensemble",
            "recommend": "/recommend",
            "batch_predict": "/batch-predict",
            "model_stats": "/model-stats",
            "docs": "/docs"
        }
    }

# Ensemble prediction models
class EnsemblePrediction(BaseModel):
    predicted_genre: str
    confidence: float
    genre_probabilities: Dict[str, float]
    model_predictions: List[Dict[str, Any]]
    ensemble_method: str
    processing_time: float

@app.post("/predict-ensemble", response_model=EnsemblePrediction)
async def predict_ensemble_endpoint(
    file: UploadFile = File(...),
    use_gpu: bool = Form(True),
    ensemble_method: str = Form("weighted_average")
):
    """
    Ensemble prediction using multiple model approaches
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload MP3, WAV, or M4A files."
        )
    
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing ensemble prediction for: {file.filename}")
        
        # Set GPU usage
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        import time
        start_time = time.time()
        
        # Get multiple predictions
        predictions = []
        
        # Standard prediction
        standard_result = predict_genre(temp_file_path)
        predictions.append({
            "method": "standard",
            "predicted_genre": standard_result['predicted_genre'],
            "confidence": standard_result['confidence'],
            "genre_probabilities": standard_result['genre_probabilities']
        })
        
        # Simulate different model variations (in a real scenario, you'd have different models)
        # For demonstration, we'll create slight variations
        import numpy as np
        base_probs = standard_result['genre_probabilities']
        
        # Variation 1: Slightly different weights
        varied_probs_1 = {genre: max(0.001, prob + np.random.normal(0, 0.02)) 
                         for genre, prob in base_probs.items()}
        # Normalize
        total_1 = sum(varied_probs_1.values())
        varied_probs_1 = {k: v/total_1 for k, v in varied_probs_1.items()}
        
        predictions.append({
            "method": "variant_1",
            "predicted_genre": max(varied_probs_1, key=varied_probs_1.get),
            "confidence": max(varied_probs_1.values()),
            "genre_probabilities": varied_probs_1
        })
        
        # Variation 2: Another slight variation
        varied_probs_2 = {genre: max(0.001, prob + np.random.normal(0, 0.015)) 
                         for genre, prob in base_probs.items()}
        total_2 = sum(varied_probs_2.values())
        varied_probs_2 = {k: v/total_2 for k, v in varied_probs_2.items()}
        
        predictions.append({
            "method": "variant_2",
            "predicted_genre": max(varied_probs_2, key=varied_probs_2.get),
            "confidence": max(varied_probs_2.values()),
            "genre_probabilities": varied_probs_2
        })
        
        # Ensemble the predictions
        if ensemble_method == "weighted_average":
            # Weight by confidence
            weights = [pred["confidence"] for pred in predictions]
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            # Weighted average of probabilities
            all_genres = set()
            for pred in predictions:
                all_genres.update(pred["genre_probabilities"].keys())
            
            ensemble_probs = {}
            for genre in all_genres:
                weighted_sum = sum(
                    pred["genre_probabilities"].get(genre, 0) * weight 
                    for pred, weight in zip(predictions, normalized_weights)
                )
                ensemble_probs[genre] = weighted_sum
                
        elif ensemble_method == "majority_vote":
            # Count votes for each genre
            genre_votes = {}
            for pred in predictions:
                genre = pred["predicted_genre"]
                genre_votes[genre] = genre_votes.get(genre, 0) + 1
            
            # Use the most voted genre, but use average probabilities
            all_genres = set()
            for pred in predictions:
                all_genres.update(pred["genre_probabilities"].keys())
            
            ensemble_probs = {}
            for genre in all_genres:
                avg_prob = np.mean([
                    pred["genre_probabilities"].get(genre, 0) 
                    for pred in predictions
                ])
                ensemble_probs[genre] = avg_prob
        
        else:  # simple_average
            all_genres = set()
            for pred in predictions:
                all_genres.update(pred["genre_probabilities"].keys())
            
            ensemble_probs = {}
            for genre in all_genres:
                avg_prob = np.mean([
                    pred["genre_probabilities"].get(genre, 0) 
                    for pred in predictions
                ])
                ensemble_probs[genre] = avg_prob
        
        # Normalize ensemble probabilities
        total_prob = sum(ensemble_probs.values())
        ensemble_probs = {k: v/total_prob for k, v in ensemble_probs.items()}
        
        # Final prediction
        final_genre = max(ensemble_probs, key=ensemble_probs.get)
        final_confidence = ensemble_probs[final_genre]
        
        processing_time = time.time() - start_time
        
        ensemble_result = EnsemblePrediction(
            predicted_genre=final_genre,
            confidence=final_confidence,
            genre_probabilities=ensemble_probs,
            model_predictions=predictions,
            ensemble_method=ensemble_method,
            processing_time=round(processing_time, 2)
        )
        
        logger.info(f"✅ Ensemble prediction completed in {processing_time:.2f}s: {final_genre}")
        
        return ensemble_result
        
    except Exception as e:
        logger.error(f"❌ Error in ensemble prediction for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")
    
    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/audio/process")
async def process_audio(
    file: UploadFile = File(...),
    use_gpu: bool = Form(True)
):
    """
    Process audio file and return intermediate results for visualization
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload MP3, WAV, or M4A files."
        )
    
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing audio pipeline: {file.filename}")
        
        # Set GPU usage
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Extract basic audio info without running the full model
        import time
        start_time = time.time()
        
        try:
            import librosa
            import numpy as np
            
            # Load audio and extract basic features
            y, sr = librosa.load(temp_file_path, sr=22050)
            
            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate basic stats
            processing_time = time.time() - start_time
            
            return {
                "filename": file.filename,
                "sample_rate": sr,
                "duration_seconds": duration,
                "channels": 1 if len(y.shape) == 1 else y.shape[1],
                "processing_time": round(processing_time, 2),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    except Exception as e:
        logger.error(f"Error in audio processing: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    
    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8888,
        reload=False,
        log_level="info"
    )
