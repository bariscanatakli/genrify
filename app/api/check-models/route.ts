import { NextResponse } from 'next/server';
import { verifyModels } from '../../utils/model-loader';

export async function GET() {
  try {
    // Check if models are available
    const modelStatus = verifyModels();
    
    return NextResponse.json(modelStatus);
  } catch (error) {
    console.error('Error checking models:', error);
    return NextResponse.json({ 
      available: false, 
      missing: ['Error checking model availability'],
      error: error instanceof Error ? error.message : String(error) 
    });
  }
}
