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
      return `import tensorflow as tf; print(tf.__version__); t = tf.constant([1,2,3]); print('ok')`;
    case 'faiss-cpu':
      return `import faiss; index = faiss.IndexFlatL2(5); print('ok')`;
    case 'librosa':
      return `import librosa; print('ok')`;
    case 'matplotlib':
      return `import matplotlib; import matplotlib.pyplot as plt; plt.figure(); print('ok')`;
    default:
      return `import ${pkg.replace('-', '_')}; print('ok')`;
  }
};

export async function GET() {
  try {
    // First check if Python is available
    try {
      await execAsync('python --version');
    } catch (error) {
      return NextResponse.json({
        available: false,
        pythonAvailable: false,
        missingPackages: ['Python is not available in the system path'],
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
    }    // Check each package separately with a 5 second timeout
    // For classification, we only need these core packages
    const requiredPackages = [
      { name: 'librosa', optional: false, description: 'Audio processing' },
      { name: 'tensorflow', optional: false, description: 'Machine learning classification' },
      { name: 'numpy', optional: false, description: 'Numerical processing' },
      { name: 'matplotlib', optional: true, description: 'Visualization and spectrogram generation' }
    ];
    
    const missingPackages: string[] = [];
    const missingCritical: string[] = [];
    const missingLibraries: Record<string, boolean> = {};
    
    // Try each package with the actual functionality test
    for (const pkg of requiredPackages) {
      try {
        const testScript = testPackageScript(pkg.name);
        const { stdout } = await execAsync(`python -c "${testScript}"`, { 
          timeout: 5000, // 5 seconds
          env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
        });
        
        // Check for successful output
        const success = stdout.includes('ok');
        if (success) {
          missingLibraries[pkg.name] = false; // not missing
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
      }
    }    // For classification, we need ML and audio processing capabilities
    const functionality = {
      ml_ready: !missingLibraries['tensorflow'] && !missingLibraries['numpy'] && modelsLoaded,
      audio_ready: !missingLibraries['librosa'] && !missingLibraries['numpy'],
      models_available: modelsLoaded,
      classification_ready: !missingLibraries['tensorflow'] && !missingLibraries['librosa'] && !missingLibraries['numpy'] && modelsLoaded,
      visualization_ready: !missingLibraries['matplotlib'] && !missingLibraries['librosa'] && !missingLibraries['numpy']
    };const usingMock = missingPackages.length > 0 || !modelsLoaded;
    const available = missingCritical.length === 0 && modelsLoaded;

    return NextResponse.json({
      available,
      pythonAvailable: true,
      missingPackages,
      missingCritical,
      missingLibraries,
      functionality,
      modelsLoaded,
      usingMock
    });
  } catch (error) {
    console.error('Error checking dependencies:', error);
    return NextResponse.json({
      available: false,
      pythonAvailable: false,
      missingPackages: ['Error checking dependencies'],
      missingCritical: ['Error checking Python environment'],
      error: error instanceof Error ? error.message : String(error),
      modelsLoaded: false,      usingMock: true,
      functionality: {
        ml_ready: false,
        audio_ready: false,
        models_available: false,
        classification_ready: false
      }
    });
  }
}
