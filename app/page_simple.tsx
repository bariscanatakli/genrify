'use client';

import { useState, useEffect } from 'react';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888';

interface GenrePrediction {
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  processing_time: number;
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
  const [useGpu, setUseGpu] = useState(true);
  const [serverHealth, setServerHealth] = useState<ServerHealth | null>(null);

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

  // Check server health on component mount
  useEffect(() => {
    checkServerHealth();
  }, []);

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

      const response = await fetch(`${API_BASE_URL}/predict`, {
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
            AI-Powered Music Genre Classification - Separated Architecture
          </p>
          
          {/* Server Status */}
          <div className="flex justify-center mb-6">
            <div className={`px-4 py-2 rounded-full text-sm font-medium ${
              serverHealth?.status === 'healthy' 
                ? 'bg-green-600 text-white' 
                : 'bg-red-600 text-white'
            }`}>
              Backend: {serverHealth?.status === 'healthy' ? '🟢 Online' : '🔴 Offline'}
              {serverHealth?.gpu_available && ' | GPU: ✅'}
              {serverHealth?.model_loaded && ' | Model: ✅'}
            </div>
          </div>
          
          {/* Architecture Info */}
          <div className="text-center mb-6">
            <div className="inline-flex items-center space-x-4 bg-white/10 backdrop-blur-md rounded-lg px-6 py-3">
              <div className="text-sm text-gray-300">
                <span className="font-bold text-blue-400">FastAPI Backend</span> ⚡ <span className="font-bold text-green-400">Next.js Frontend</span>
              </div>
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
                  className="w-full p-3 bg-white/20 rounded-lg text-white placeholder-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                <p className="text-sm text-gray-300 mt-2">
                  Supported formats: MP3, WAV, M4A
                </p>
              </div>

              {/* GPU Option */}
              <div className="flex justify-center">
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
              <p className="text-gray-300 text-sm">FastAPI backend is analyzing the audio (~13 seconds)</p>
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
                          <div className="w-24 text-sm text-gray-300 font-medium">{genre}</div>
                          <div className="flex-1 mx-3">
                            <div className="bg-gray-700 rounded-full h-3">
                              <div
                                className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-500"
                                style={{ width: `${probability * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          <div className="w-12 text-sm text-gray-300 text-right font-medium">
                            {(probability * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {/* Architecture Info */}
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-3">🏗️ Separated Architecture Benefits</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div className="space-y-2">
                    <div className="text-green-400">✅ Independent Backend & Frontend</div>
                    <div className="text-green-400">✅ Scalable FastAPI Server</div>
                    <div className="text-green-400">✅ GPU Optimization Maintained</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-blue-400">⚡ API-First Design</div>
                    <div className="text-blue-400">⚡ Reusable Backend</div>
                    <div className="text-blue-400">⚡ Easy Deployment</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
