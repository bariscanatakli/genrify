import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';

const execAsync = promisify(exec);

// More rigorous package checking script that tests actual functionality
const testPackageScript = (pkg: string) => {
  switch (pkg) {
    case 'numpy':
      return `import numpy as np; a = np.array([1,2,3]); print('ok')`;
    case 'tensorflow':
      // Use the dedicated module for TensorFlow testing
      return `import sys; sys.path.append('${process.cwd()}/app/api'); from check_tf import check_tensorflow; check_tensorflow()`;
    case 'tensorflow-cuda':
      // Test CUDA support specifically
      return `import sys; sys.path.append('${process.cwd()}/app/api'); from check_tf import check_cuda; check_cuda()`;
    case 'faiss-cpu':
      return `import faiss; print('ok')`;  // Simplified test without index creation
    case 'librosa':
      return `import librosa; print('ok')`;
    case 'matplotlib':
      return `import matplotlib; print('ok')`; // Simplified test without plt
    default:
      return `import ${pkg.replace('-', '_')}; print('ok')`;
  }
};

export async function GET() {
  try {
    // First check if Python is available (use venv Python)
    try {
      await execAsync('./venv/bin/python --version');
    } catch (error) {
      return NextResponse.json({
        available: false,
        pythonAvailable: false,
        missingPackages: ['Python is not available in the venv path'],
        missingCritical: ['Python'],
        modelsLoaded: false,
        functionality: {
          ml_ready: false,
          search_ready: false,
          audio_ready: false,
          embeddings_available: false,
          models_available: false
        }
      });
    }
    
    // Check model availability - only need CNN classifier
    const modelDir = path.join(process.cwd(), 'models');
    let modelsLoaded = false;
    
    if (fs.existsSync(modelDir)) {
      const requiredModelFiles = ['optimized_cnn_model.keras'];
      modelsLoaded = requiredModelFiles.every(file => fs.existsSync(path.join(modelDir, file)));
    }
    
    // Check each package separately with a 5 second timeout
    // For classification, we only need these core packages
    const requiredPackages = [
      { name: 'librosa', optional: false, description: 'Audio processing' },
      { name: 'tensorflow', optional: false, description: 'Machine learning classification' },
      { name: 'numpy', optional: false, description: 'Numerical processing' },
      { name: 'matplotlib', optional: true, description: 'Visualization and spectrogram generation' }
    ];
    
    // Add CUDA check as separate test
    const additionalTests = [
      { name: 'tensorflow-cuda', optional: true, description: 'GPU acceleration' }
    ];
    
    const missingPackages: string[] = [];
    const missingCritical: string[] = [];
    const missingLibraries: Record<string, boolean> = {};
    
    // Try each package with the actual functionality test
    for (const pkg of [...requiredPackages, ...additionalTests]) {
      try {
        const testScript = testPackageScript(pkg.name);
        const { stdout, stderr } = await execAsync(`./venv/bin/python -c "${testScript}"`, { 
          timeout: 15000, // 15 seconds (increased from 5 seconds)
          env: { 
            ...process.env, 
            PYTHONIOENCODING: 'utf-8',
            TF_CPP_MIN_LOG_LEVEL: '3' // Suppress TensorFlow warnings
          }
        });
        
        // Check for successful output
        const success = stdout.includes('ok');
        if (success) {
          missingLibraries[pkg.name] = false; // not missing
          
          // Special handling for CUDA support information
          if (pkg.name === 'tensorflow-cuda') {
            // Extract GPU information from stdout
            const gpuAvailable = stdout.includes('GPU available:') && !stdout.includes('GPU available: 0');
            const gpuCount = parseInt(stdout.match(/GPU available: (\d+)/)?.[1] || '0');
            const cudaBuilt = stdout.includes('CUDA built: True');
            
            // Add these details to the response
            missingLibraries['cuda_available'] = !gpuAvailable;
            missingLibraries['cuda_built'] = !cudaBuilt;
            missingLibraries['gpu_count'] = gpuCount;
          }
        } else {
          throw new Error(`Test failed for ${pkg.name}`);
        }
      } catch (error) {
        console.log(`Package test failed for ${pkg.name}:`, error);
        missingPackages.push(pkg.name);
        missingLibraries[pkg.name] = true; // missing
        if (!pkg.optional) {
          missingCritical.push(pkg.name);
        }
        
        // Set CUDA flags if test failed
        if (pkg.name === 'tensorflow-cuda') {
          missingLibraries['cuda_available'] = true;
          missingLibraries['cuda_built'] = true;
          missingLibraries['gpu_count'] = 0;
        }
      }
    }
    
    // For classification, we need ML and audio processing capabilities
    const functionality = {
      ml_ready: !missingLibraries['tensorflow'] && !missingLibraries['numpy'] && modelsLoaded,
      audio_ready: !missingLibraries['librosa'] && !missingLibraries['numpy'],
      models_available: modelsLoaded,
      classification_ready: !missingLibraries['tensorflow'] && !missingLibraries['librosa'] && !missingLibraries['numpy'] && modelsLoaded,
      visualization_ready: !missingLibraries['matplotlib'] && !missingLibraries['librosa'] && !missingLibraries['numpy'],
      cuda_available: !missingLibraries['cuda_available'],
      gpu_acceleration: !missingLibraries['tensorflow-cuda'] && !missingLibraries['cuda_available']
    };
    
    const usingMock = missingPackages.length > 0 || !modelsLoaded;
    const available = missingCritical.length === 0 && modelsLoaded;

    return NextResponse.json({
      available,
      pythonAvailable: true,
      missingPackages,
      missingCritical,
      missingLibraries,
      functionality,
      modelsLoaded,
      usingMock,
      gpu_support: {
        available: !missingLibraries['cuda_available'],
        gpu_count: missingLibraries['gpu_count'] || 0,
        cuda_built: !missingLibraries['cuda_built']
      }
    });
  } catch (error) {
    console.error('Error checking dependencies:', error);
    return NextResponse.json({
      available: false,
      pythonAvailable: false,
      missingPackages: ['Error checking dependencies'],
      missingCritical: ['Error checking Python environment'],
      error: error instanceof Error ? error.message : String(error),
      modelsLoaded: false,
      usingMock: true,
      functionality: {
        ml_ready: false,
        audio_ready: false,
        models_available: false,
        classification_ready: false,
        cuda_available: false,
        gpu_acceleration: false
      },
      gpu_support: {
        available: false,
        gpu_count: 0,
        cuda_built: false
      }
    });
  }
}
