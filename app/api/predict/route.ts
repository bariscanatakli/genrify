import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { writeFile } from 'fs/promises';
import path from 'path';
import os from 'os';
import { extractJsonFromOutput } from '../../utils/json-helpers'; // Import the utility

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    // Handle file upload
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Create a temporary file
    const tempDir = os.tmpdir();
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const filePath = path.join(tempDir, `upload-${uniqueSuffix}.mp3`);
    
    // Write the file to disk
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filePath, buffer);
    
    // Fix path format - replace backslashes with forward slashes
    const normalizedScriptPath = path.join(process.cwd(), 'app', 'api').replace(/\\/g, '/');
    const normalizedFilePath = filePath.replace(/\\/g, '/');
      // Check if visualization data is requested
    const includeVisualization = formData.get('include_visualization') === 'true';
    
    try {
      // Python command now calls the function and then output_json
      const command = includeVisualization 
        ? `python -c "import sys; sys.path.append('${normalizedScriptPath}'); from process import predict_genre_with_visualization, output_json; result = predict_genre_with_visualization('${normalizedFilePath}'); output_json(result)"`
        : `python -c "import sys; sys.path.append('${normalizedScriptPath}'); from process import predict_genre, output_json; result = predict_genre('${normalizedFilePath}'); output_json(result)"`;
      // console.log("Executing command for prediction:", command); // For debugging
        const { stdout, stderr } = await execAsync(command, {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, // TF_CPP_MIN_LOG_LEVEL is set in process.py
        maxBuffer: 1024 * 1024 * 50  // 50MB buffer for large outputs with visualization data
      });
      
      if (stderr) { // Log Python's stderr for debugging
        console.error("Python stderr (predict):", stderr);
      }
      
      try {
        const result = extractJsonFromOutput(stdout); // Use the utility
        // console.log("Successfully extracted JSON prediction result"); // For debugging
        return NextResponse.json(result);
      } catch (jsonError) {
        console.error("JSON extraction error (predict):", jsonError);
        console.error("Raw stdout from Python (predict - first 500 chars):", stdout.substring(0, 500));
        
        // Fall back to hardcoded mock data if JSON parsing fails
        return NextResponse.json({
          genre_probabilities: {
            "Electronic": 0.7,
            "Rock": 0.2,
            "Pop": 0.05,
            "Hip-Hop": 0.02,
            "Folk": 0.01,
            "Experimental": 0.01,
            "Instrumental": 0.005,
            "International": 0.005
          },
          predicted_genre: "Electronic",
          confidence: 0.7,
          using_mock: true,
          error: "JSON parsing failed after Python execution"
        });
      }
    } catch (pythonError) {
      console.error('Python execution failed (predict):', pythonError);
      
      // Fall back to client-side mock data
      return NextResponse.json({
        genre_probabilities: {
          "Electronic": 0.7,
          "Rock": 0.2,
          "Pop": 0.05,
          "Hip-Hop": 0.02,
          "Folk": 0.01,
          "Experimental": 0.01,
          "Instrumental": 0.005,
          "International": 0.005
        },
        predicted_genre: "Electronic",
        confidence: 0.7,
        using_mock: true,
        error: pythonError instanceof Error ? pythonError.message : String(pythonError)
      });
    }
  } catch (requestError) {
    console.error('Error processing request (predict):', requestError);
    return NextResponse.json({ 
      error: 'Failed to process audio file',
      details: requestError instanceof Error ? requestError.message : String(requestError) 
    }, { status: 500 });
  }
}
