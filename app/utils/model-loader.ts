import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// Check if all required models are available
export function verifyModels(): { available: boolean; missing: string[] } {
  const requiredModels = [
    'optimized_cnn_model.keras'
  ];
  
  const modelDir = path.join(process.cwd(), 'models');
  const missing: string[] = [];
  
  for (const model of requiredModels) {
    const modelPath = path.join(modelDir, model);
    if (!fs.existsSync(modelPath)) {
      missing.push(model);
    }
  }
    return {
    available: missing.length === 0,
    missing
  };
}
