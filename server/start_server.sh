#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Music Genre Classification Server${NC}"

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run from the main genrify directory:${NC}"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}📦 Activating virtual environment...${NC}"
source ../venv/bin/activate

# Install server requirements
echo -e "${BLUE}📥 Installing server dependencies...${NC}"
pip install -r requirements.txt

# Check if models exist
if [ ! -f "../models/optimized_cnn_model.keras" ]; then
    echo -e "${RED}❌ Model file not found. Please ensure the model is available at:${NC}"
    echo "../models/optimized_cnn_model.keras"
    exit 1
fi

echo -e "${GREEN}✅ Starting FastAPI server on http://localhost:8888${NC}"
echo -e "${BLUE}📖 API Documentation available at: http://localhost:8888/docs${NC}"
echo -e "${BLUE}🏥 Health check available at: http://localhost:8888/health${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
