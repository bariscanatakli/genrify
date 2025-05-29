#!/bin/bash
# Helper script to start server with GPU acceleration

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Music Genre Classification Server with GPU Acceleration${NC}"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️ NVIDIA GPU drivers not detected - will still attempt to start with TensorFlow GPU${NC}"
else
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi
fi

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo -e "${RED}❌ Virtual environment not found at ../venv. Please run from the main genrify directory:${NC}"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r server/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}📦 Activating virtual environment...${NC}"
source ../venv/bin/activate

# Set environment variables for optimal GPU memory usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
export CUDA_CACHE_DISABLE=0
export TF_ENABLE_ONEDNN_OPTS=1
export TF_CPP_MIN_LOG_LEVEL=2

echo -e "${GREEN}🖥️ Starting FastAPI server with GPU acceleration${NC}"
echo -e "${BLUE}📖 API Documentation available at: http://localhost:8888/docs${NC}"
echo -e "${BLUE}🏥 Health check available at: http://localhost:8888/health${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
