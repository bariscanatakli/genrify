import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function GET() {
  try {
    // Check for Python
    try {
      await execAsync('./venv/bin/python --version');
    } catch (error) {
      return NextResponse.json({
        success: false,
        error: 'Python virtual environment not available',
        cuda_available: false,
        gpu_count: 0,
        details: {}
      });
    }

    // Run the CUDA check script
    const { stdout, stderr } = await execAsync(
      `./venv/bin/python -c "import sys; sys.path.append('${process.cwd()}/app/api'); from check_tf import check_cuda; check_cuda()"`,
      {
        timeout: 15000,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
          TF_CPP_MIN_LOG_LEVEL: '1'  // Show some important warnings
        }
      }
    );

    // Extract information from output
    const cudaBuilt = stdout.includes('CUDA built: True');
    const gpuAvailable = stdout.includes('GPU available for TF: True');
    const gpuCountMatch = stdout.match(/GPU devices available: (\d+)/);
    const gpuCount = gpuCountMatch ? parseInt(gpuCountMatch[1]) : 0;
    
    // Extract CUDA and cuDNN versions if available
    const cudaVersionMatch = stdout.match(/CUDA version: ([\d\.]+)/);
    const cudnnVersionMatch = stdout.match(/cuDNN version: ([\d\.]+)/);
    
    // Extract GPU details
    const gpuDetails: string[] = [];
    const gpuRegex = /GPU \d+: (.+)/g;
    let gpuMatch;
    while ((gpuMatch = gpuRegex.exec(stdout)) !== null) {
      gpuDetails.push(gpuMatch[1]);
    }
    
    // Check if GPU computation was successful
    const gpuComputationSuccessful = stdout.includes('GPU computation successful');

    // Handle errors in output
    const errorLines = stderr.split('\n').filter(line => line.trim().length > 0);
    const hasErrors = errorLines.length > 0;

    return NextResponse.json({
      success: true,
      cuda_available: cudaBuilt && gpuAvailable && gpuCount > 0,
      gpu_available: gpuAvailable,
      gpu_count: gpuCount,
      gpu_details: gpuDetails,
      cuda_built: cudaBuilt,
      gpu_computation_successful: gpuComputationSuccessful,
      details: {
        cuda_version: cudaVersionMatch ? cudaVersionMatch[1] : 'unknown',
        cudnn_version: cudnnVersionMatch ? cudnnVersionMatch[1] : 'unknown',
        errors: hasErrors ? errorLines : []
      },
      raw_output: stdout  // Include raw output for debugging
    });
  } catch (error) {
    console.error('Error checking CUDA:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      cuda_available: false,
      gpu_count: 0,
      details: {}
    });
  }
}
