import { useState, useCallback } from 'react';

interface GenrePrediction {
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  processing_time: number;
  visualization_data?: any;
}

interface AudioAnalysisOptions {
  apiUrl?: string;
  useGpu?: boolean;
  includeVisualization?: boolean;
}

export function useAudioAnalysis({ 
  apiUrl = 'http://localhost:8888',
  useGpu = true,
  includeVisualization = false
}: AudioAnalysisOptions = {}) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<GenrePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);

  const analyzeAudio = useCallback(async (audioFile: File = file!) => {
    if (!audioFile) {
      setError("No audio file provided");
      return null;
    }

    setLoading(true);
    setError(null);
    setProgress(0);

    // Reset existing results
    setResult(null);

    const formData = new FormData();
    formData.append('file', audioFile);
    formData.append('use_gpu', useGpu.toString());
    
    if (includeVisualization) {
      formData.append('include_visualization', 'true');
    }

    try {
      // Start progress simulation
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + (Math.random() * 5);
          return newProgress > 95 ? 95 : newProgress;
        });
      }, 300);

      // Determine endpoint based on visualization need
      const endpoint = includeVisualization 
        ? `${apiUrl}/predict-with-viz`
        : `${apiUrl}/predict`;

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      clearInterval(progressInterval);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze audio');
      }

      const prediction = await response.json();
      setResult(prediction);
      setProgress(100);
      
      return prediction;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiUrl, file, includeVisualization, useGpu]);

  const clearResults = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  return {
    file,
    setFile,
    loading,
    result,
    error,
    progress,
    analyzeAudio,
    clearResults
  };
}
