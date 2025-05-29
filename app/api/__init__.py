# Make the important functions from process.py directly importable from the api package
from .process import (
    get_hann_window,
    predict_genre,
    predict_genre_with_visualization,
    get_recommendations,
    mock_predict,
    generate_mock_visualization
)

__all__ = [
    'get_hann_window',
    'predict_genre',
    'predict_genre_with_visualization',
    'get_recommendations',
    'mock_predict',
    'generate_mock_visualization'
]
