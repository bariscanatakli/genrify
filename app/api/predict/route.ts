import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.API_URL || 'http://localhost:8888';

export async function POST(request: NextRequest) {
  try {
    // Extract the form data from the request
    const formData = await request.formData();
    
    // Forward the request to FastAPI backend
    const endpoint = formData.get('include_visualization') === 'true' 
      ? `${API_BASE_URL}/predict-with-viz` 
      : `${API_BASE_URL}/predict`;
      
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || 'Prediction failed' },
        { status: response.status }
      );
    }

    // Return the prediction results directly from the API
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('Error in prediction API route:', error);
    return NextResponse.json(
      { error: 'Internal server error during prediction' },
      { status: 500 }
    );
  }
}
