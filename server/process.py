"""
Server-side audio processing functions for music genre classification
Implemented locally to avoid import errors from app directory
"""
import numpy as np
import sys

# Make sure this function is defined first and properly exported
def get_hann_window(window_length):
    """
    Compatibility wrapper for hann window that works across different SciPy versions.
    Some versions have it in scipy.signal, others in scipy.signal.windows
    """
    try:
        # Try newer SciPy structure first (scipy.signal.windows)
        from scipy.signal.windows import hann
        return hann(window_length)
    except ImportError:
        try:
            # Try older SciPy structure (scipy.signal)
            from scipy.signal import hann
            return hann(window_length)
        except ImportError:
            # Last resort - implement a simple hann window ourselves
            import numpy as np
            print("[INFO] Using NumPy implementation of hann window function")
            return 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_length) / (window_length - 1)))

# Import model loaders and prediction functions from app's process.py
# Use relative imports to avoid the GPU memory conflicts
def import_app_functions():
    """Import functions from app's process.py with safe error handling"""
    try:
        # Add app directories to path
        import os
        from pathlib import Path
        app_path = Path(__file__).parent.parent / "app"
        api_path = app_path / "api"
        
        sys.path.append(str(app_path))
        sys.path.append(str(api_path))
        
        # Import only specific functions after patching
        from api.process import predict_genre, predict_genre_with_visualization, get_recommendations
        
        return predict_genre, predict_genre_with_visualization, get_recommendations
    except ImportError as e:
        print(f"Error importing app functions: {e}")
        return None, None, None

# Import functions only when needed to avoid early GPU initialization
predict_genre, predict_genre_with_visualization, get_recommendations = None, None, None

def init():
    """Initialize functions to avoid circular imports and preserve GPU memory settings"""
    global predict_genre, predict_genre_with_visualization, get_recommendations
    predict_genre, predict_genre_with_visualization, get_recommendations = import_app_functions()
