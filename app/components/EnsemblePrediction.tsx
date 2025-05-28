'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

interface ModelPrediction {
  method: string;
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
}

interface EnsemblePredictionResult {
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  model_predictions: ModelPrediction[];
  ensemble_method: string;
  processing_time: number;
}

interface EnsemblePredictionProps {
  audioFile: File | null;
  apiUrl?: string;
}

const EnsemblePrediction: React.FC<EnsemblePredictionProps> = ({ 
  audioFile, 
  apiUrl = 'http://localhost:8888' 
}) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EnsemblePredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ensembleMethod, setEnsembleMethod] = useState<string>('weighted_average');
  const [useGpu, setUseGpu] = useState(true);

  const ensembleMethods = [
    { value: 'weighted_average', label: 'Weighted Average (by confidence)', description: 'Weights predictions by their confidence scores' },
    { value: 'majority_vote', label: 'Majority Vote', description: 'Uses the most frequently predicted genre' },
    { value: 'simple_average', label: 'Simple Average', description: 'Simple average of all probability distributions' }
  ];

  const handleEnsemblePredict = async () => {
    if (!audioFile) {
      setError('Please select an audio file first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      formData.append('ensemble_method', ensembleMethod);
      formData.append('use_gpu', useGpu.toString());
      
      const response = await fetch(`${apiUrl}/predict/ensemble`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error('Error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getAgreementScore = () => {
    if (!result) return 0;
    
    const predictions = result.model_predictions.map(p => p.predicted_genre);
    const uniquePredictions = new Set(predictions);
    return ((predictions.length - uniquePredictions.size + 1) / predictions.length) * 100;
  };

  const getConfidenceVariance = () => {
    if (!result) return 0;
    const confidences = result.model_predictions.map(p => p.confidence);
    const mean = confidences.reduce((a, b) => a + b) / confidences.length;
    const variance = confidences.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / confidences.length;
    return Math.sqrt(variance);
  };

  return (
    <div className="space-y-8">
      {/* Enhanced Header */}
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="spectral-bars">
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
            <div className="spectral-bar"></div>
          </div>
        </div>
        <h2 className="text-3xl font-bold gradient-text-advanced mb-4 neon-glow">
          🎭 Ensemble Prediction
        </h2>
        <p className="text-gray-400">Combining multiple models for more accurate genre prediction</p>
      </div>

      <Card className="prism-glass dynamic-card neural-network rounded-3xl p-8 border-2 border-pink-400/20">
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <span className="text-4xl">🎯</span>
            <span className="gradient-text-advanced">Ensemble Prediction</span>
          </CardTitle>
          <CardDescription className="text-gray-300">
            Combine predictions from multiple models for improved accuracy
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">Ensemble Method</label>
              <select
                value={ensembleMethod}
                onChange={(e) => setEnsembleMethod(e.target.value)}
                className="w-full glass-card p-3 rounded-lg bg-gray-800/50 border border-gray-700 text-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                {ensembleMethods.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-sm text-gray-400">
                {ensembleMethods.find(m => m.value === ensembleMethod)?.description}
              </p>
            </div>
            
            <div>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useGpu}
                  onChange={(e) => setUseGpu(e.target.checked)}
                  className="rounded text-purple-500 focus:ring-purple-500 h-4 w-4"
                />
                <span className="text-gray-300 text-sm font-medium">Use GPU acceleration</span>
              </label>
            </div>

            <div className="text-center pt-4">
              <button
                onClick={handleEnsemblePredict}
                disabled={!audioFile || loading}
                className="morph-button px-8 py-3 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden ripple"
              >
                {loading ? (
                  <div className="flex items-center justify-center space-x-3">
                    <div className="spinner-premium"></div>
                    <span>🤖 Running Models...</span>
                  </div>
                ) : (
                  '🎭 Run Ensemble Prediction'
                )}
              </button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Loading */}
      {loading && (
        <Card className="bg-white/10 backdrop-blur-md border-blue-300/20">
          <CardContent className="text-center p-6">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-white">Running ensemble prediction...</p>
            <p className="text-gray-300 text-sm">
              Processing multiple model variants for improved accuracy
            </p>
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {error && (
        <Card className="bg-red-600/20 border border-red-500">
          <CardContent className="p-6">
            <h3 className="text-red-400 font-bold mb-2">❌ Error</h3>
            <p className="text-white">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Main Result */}
          <Card className="bg-white/10 backdrop-blur-md border-green-300/20">
            <CardHeader>
              <CardTitle className="text-white">🏆 Ensemble Result</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">
                    {result.predicted_genre}
                  </div>
                  <div className="text-gray-300 text-sm">Final Prediction</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-gray-300 text-sm">Ensemble Confidence</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400">
                    {getAgreementScore().toFixed(1)}%
                  </div>
                  <div className="text-gray-300 text-sm">Model Agreement</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-400">
                    {result.processing_time}s
                  </div>
                  <div className="text-gray-300 text-sm">Processing Time</div>
                </div>
              </div>

              <div className="bg-white/5 rounded-lg p-4">
                <h4 className="text-white font-medium mb-2">Ensemble Method: {result.ensemble_method}</h4>
                <p className="text-gray-300 text-sm">
                  Confidence Variance: ±{(getConfidenceVariance() * 100).toFixed(2)}%
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Individual Model Results */}
          <Card className="bg-white/10 backdrop-blur-md border-purple-300/20">
            <CardHeader>
              <CardTitle className="text-white">🧩 Individual Model Results</CardTitle>
              <CardDescription className="text-gray-300">
                Predictions from each model in the ensemble
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {result.model_predictions.map((model, index) => (
                  <div key={index} className="glass-card p-4 rounded-xl">
                    <div className="flex flex-col md:flex-row md:items-center justify-between mb-4">
                      <div className="mb-2 md:mb-0">
                        <h5 className="font-semibold text-purple-300">{model.method}</h5>
                        <p className="text-sm text-gray-400">
                          Model {index + 1} of {result.model_predictions.length}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xl font-bold text-green-400">{model.predicted_genre}</span>
                        <span className="text-sm px-2 py-1 bg-green-900/30 rounded-full text-green-400">
                          {(model.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {Object.entries(model.genre_probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 4)
                        .map(([genre, prob]) => (
                          <div key={genre} className="bg-gray-800/50 rounded-lg p-2 text-center">
                            <div className="text-xs text-gray-400">{genre}</div>
                            <div className="text-sm font-bold text-white">{(prob * 100).toFixed(1)}%</div>
                          </div>
                        ))
                      }
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Genre Distribution */}
          <Card className="bg-white/10 backdrop-blur-md border-blue-300/20">
            <CardHeader>
              <CardTitle className="text-white">📈 Genre Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(result.genre_probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 8)
                  .map(([genre, prob], index) => (
                    <div key={genre} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-300">{genre}</span>
                        <span className="text-gray-300">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            index === 0
                              ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                              : index === 1
                              ? 'bg-gradient-to-r from-blue-500 to-cyan-500'
                              : 'bg-gradient-to-r from-green-500 to-emerald-500'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default EnsemblePrediction;
