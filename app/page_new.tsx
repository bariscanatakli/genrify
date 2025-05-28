'use client';

import { useState } from 'react';
import ClassificationVisualization from './components/ClassificationVisualization';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888';

interface GenrePrediction {
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  processing_time: number;
  visualization_data?: any;
}

interface ServerHealth {
  status: string;
  gpu_available: boolean;
  model_loaded: boolean;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenrePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [includeVisualization, setIncludeVisualization] = useState(false);
  const [useGpu, setUseGpu] = useState(true);
  const [serverHealth, setServerHealth] = useState<ServerHealth | null>(null);

  // Check server health on component mount
  useState(() => {
    checkServerHealth();
  });

  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const health = await response.json();
      setServerHealth(health);
    } catch (err) {
      console.error('Server health check failed:', err);
      setServerHealth({ status: 'offline', gpu_available: false, model_loaded: false });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError('Please select an audio file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('use_gpu', useGpu.toString());

      const endpoint = includeVisualization ? '/predict-with-viz' : '/predict';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const prediction = await response.json();
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
            🎵 Genrify
          </h1>
          <p className="text-xl text-gray-300 mb-4">
            AI-Powered Music Genre Classification
          </p>
          
          {/* Server Status */}
          <div className="flex justify-center mb-6">
            <div className={`px-4 py-2 rounded-full text-sm font-medium ${
              serverHealth?.status === 'healthy' 
                ? 'bg-green-600 text-white' 
                : 'bg-red-600 text-white'
            }`}>
              Server: {serverHealth?.status === 'healthy' ? '🟢 Online' : '🔴 Offline'}
              {serverHealth?.gpu_available && ' | GPU: ✅'}
              {serverHealth?.model_loaded && ' | Model: ✅'}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto">
          {/* File Upload */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
            <h2 className="text-2xl font-bold text-white mb-4">Upload Audio File</h2>
            
            <div className="space-y-4">
              <div>
                <input
                  type="file"
                  accept=".mp3,.wav,.m4a"
                  onChange={handleFileChange}
                  className="w-full p-3 bg-white/20 rounded-lg text-white placeholder-gray-300"
                />
                <p className="text-sm text-gray-300 mt-2">
                  Supported formats: MP3, WAV, M4A
                </p>
              </div>

              {/* Options */}
              <div className="flex gap-4">
                <label className="flex items-center text-white">
                  <input
                    type="checkbox"
                    checked={includeVisualization}
                    onChange={(e) => setIncludeVisualization(e.target.checked)}
                    className="mr-2"
                  />
                  Include Visualization
                </label>
                
                <label className="flex items-center text-white">
                  <input
                    type="checkbox"
                    checked={useGpu}
                    onChange={(e) => setUseGpu(e.target.checked)}
                    className="mr-2"
                  />
                  Use GPU Acceleration
                </label>
              </div>

              <button
                onClick={handlePredict}
                disabled={!file || loading || serverHealth?.status !== 'healthy'}
                className="w-full py-3 px-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {loading ? '🤖 Analyzing...' : '🎯 Predict Genre'}
              </button>
            </div>
          </div>

          {/* Loading */}
          {loading && (
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 text-center">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p className="text-white">Processing your audio file...</p>
              <p className="text-gray-300 text-sm">This may take 10-15 seconds</p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-red-600/20 border border-red-500 rounded-xl p-6 mb-6">
              <h3 className="text-red-400 font-bold mb-2">❌ Error</h3>
              <p className="text-white">{error}</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-6">
              {/* Main Result */}
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                <h3 className="text-2xl font-bold text-white mb-4">🎯 Prediction Results</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-400">
                      {result.predicted_genre}
                    </div>
                    <div className="text-gray-300">Predicted Genre</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-400">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-gray-300">Confidence</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-400">
                      {result.processing_time}s
                    </div>
                    <div className="text-gray-300">Processing Time</div>
                  </div>
                </div>

                {/* Genre Probabilities */}
                <div>
                  <h4 className="text-lg font-bold text-white mb-3">Genre Probabilities</h4>
                  <div className="space-y-2">
                    {Object.entries(result.genre_probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([genre, probability]) => (
                        <div key={genre} className="flex items-center">
                          <div className="w-20 text-sm text-gray-300">{genre}</div>
                          <div className="flex-1 mx-3">
                            <div className="bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full"
                                style={{ width: `${probability * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          <div className="w-12 text-sm text-gray-300 text-right">
                            {(probability * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {/* Visualization */}
              {result.visualization_data && (
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                  <h3 className="text-2xl font-bold text-white mb-4">📊 Audio Analysis</h3>
                  <ClassificationVisualization visualizationData={result.visualization_data} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
