#!/usr/bin/env node
/**
 * This script helps run Next.js with the correct Python virtual environment.
 * It ensures that the Python dependencies are available before starting the Next.js development server.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Path to the virtual environment
const venvPath = path.join(__dirname, 'venv');
const venvBin = path.join(venvPath, 'bin');
const venvPython = path.join(venvBin, 'python');

// Check if venv exists
if (!fs.existsSync(venvPath)) {
  console.log('🔧 Python virtual environment not found. Creating one...');
  try {
    execSync('python -m venv venv', { stdio: 'inherit' });
    console.log('✅ Virtual environment created!');
  } catch (error) {
    console.error('❌ Failed to create virtual environment:', error.message);
    process.exit(1);
  }
}

// Check if requirements are installed
console.log('🔄 Checking Python dependencies...');
try {
  // Activate venv and check for dependencies
  execSync(`${venvPython} -c "import tensorflow, librosa, faiss, numpy, matplotlib; print('All dependencies available')"`, 
    { stdio: 'inherit' });
} catch (error) {
  console.log('📦 Installing required Python dependencies...');
  try {
    execSync(`${venvPython} -m pip install -r requirements.txt`, { stdio: 'inherit' });
    console.log('✅ Dependencies installed successfully!');
  } catch (installError) {
    console.error('❌ Failed to install dependencies:', installError.message);
    console.log('Please run: npm run venv:setup');
    process.exit(1);
  }
}

// Check for CUDA/GPU support
console.log('🔍 Checking for GPU/CUDA support...');
try {
  const cudaInfo = execSync(
    `${venvPython} -c "import tensorflow as tf; print(f'CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}'); print(f'GPU available: {tf.test.is_gpu_available()}'); print(f'Physical GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"`,
    { encoding: 'utf8' }
  );
  
  console.log(cudaInfo);
  
  if (cudaInfo.includes('Physical GPUs: 0') || !cudaInfo.includes('GPU available: True')) {
    console.log('⚠️ No GPU detected. Running in CPU mode.');
  } else {
    console.log('🚀 GPU detected! TensorFlow will use GPU acceleration.');
  }
} catch (error) {
  console.log('⚠️ Could not check GPU support:', error.message);
}

// Run Next.js dev server
console.log('🚀 Starting Next.js development server with Python environment...');
process.env.PATH = `${venvBin}:${process.env.PATH}`;

// Start the Next.js development server
const nextDev = spawn('npx', ['next', 'dev'], { 
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHON_PATH: venvPython
  }
});

// Handle process exit
nextDev.on('exit', (code) => {
  process.exit(code);
});

// Handle process signals
process.on('SIGINT', () => {
  nextDev.kill('SIGINT');
});

process.on('SIGTERM', () => {
  nextDev.kill('SIGTERM');
});
