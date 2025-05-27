#!/bin/bash
# Activate the Python virtual environment
source venv/bin/activate

# Print Python and package versions for verification
echo "Using Python from venv:"
python --version
echo ""
echo "TensorFlow version:"
python -c "import tensorflow as tf; print(tf.__version__)"
echo ""

# Show CUDA and GPU information
echo "CUDA and GPU Information:"
python -c "import tensorflow as tf; print(f'CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}'); print(f'GPU available: {tf.test.is_gpu_available()}'); print(f'Physical GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
echo ""

echo "Librosa version:"
python -c "import librosa; print(librosa.__version__)"
echo ""
echo "FAISS version:"
python -c "import faiss; print(faiss.__version__)"
echo ""
echo "Virtual environment is active. You can now run your Next.js app with 'npm run dev'"
