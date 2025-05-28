'use client';

import { useState, useEffect } from 'react';
import ClassificationVisualization from './components/ClassificationVisualization';
import GenreRadar from './components/GenreRadar';
import MusicRecommendations from './components/MusicRecommendations';
import AdvancedPrediction from './components/AdvancedPrediction';
import AudioProcessingPipeline from './components/AudioProcessingPipeline';
import BatchProcessing from './components/BatchProcessing';
import ModelDashboard from './components/ModelDashboard';
import EnsemblePrediction from './components/EnsemblePrediction';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888';

interface GenrePrediction {
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  processing_time: number;
  visualization_data?: {
    spectrogram?: string;
    mel_spectrogram?: string;
    mfcc_features?: number[][];
    chroma_features?: number[][];
    spectral_features?: {
      spectral_centroid: number[];
      spectral_rolloff: number[];
      zero_crossing_rate: number[];
    };
    tempo?: number;
    model_predictions?: number[];
  };
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
  const [showVisualization, setShowVisualization] = useState(false);
  const [useAdvancedPrediction, setUseAdvancedPrediction] = useState(false);
  const [activeTab, setActiveTab] = useState<'prediction' | 'recommendations' | 'advanced' | 'pipeline' | 'batch' | 'dashboard' | 'ensemble'>('prediction');

  useEffect(() => {
    checkServerHealth();
  }, []);

  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        const health = await response.json();
        setServerHealth(health);
      } else {
        setServerHealth({ status: 'unhealthy', gpu_available: false, model_loaded: false });
      }
    } catch {
      setServerHealth({ status: 'offline', gpu_available: false, model_loaded: false });
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
      setShowVisualization(false);
    }
  };

  const classifyMusic = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_gpu', useGpu.toString());
    
    if (useAdvancedPrediction) {
      formData.append('include_visualization', 'true');
    }

    try {
      const endpoint = useAdvancedPrediction 
        ? `${API_BASE_URL}/predict/advanced`
        : `${API_BASE_URL}/predict`;
        
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const prediction = await response.json();
      setResult(prediction);
      
      // Auto-show visualization if we have visualization data
      if (prediction.visualization_data) {
        setShowVisualization(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen morphing-bg">
      {/* Enhanced Floating Particles System */}
      <div className="floating-particles">
        {Array.from({ length: 50 }, (_, i) => (
          <div
            key={i}
            className="particle"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 15}s`,
              animationDuration: `${15 + Math.random() * 10}s`,
            }}
          />
        ))}
      </div>
      
      {/* Enhanced Animated background particles */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-purple-500/10 to-transparent rounded-full blur-3xl animate-spin" style={{animationDuration: '20s'}}></div>
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-blue-500/10 to-transparent rounded-full blur-3xl animate-spin" style={{animationDuration: '25s', animationDirection: 'reverse'}}></div>
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-pink-500/5 to-yellow-500/5 rounded-full blur-2xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-l from-green-500/5 to-blue-500/5 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header with enhanced animations */}
        <div className="text-center mb-12">
          <div className="float-animation">
            <h1 className="text-5xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-500 to-blue-500 mb-6 tracking-tight neon-glow">
              🎵 Genrify
            </h1>
          </div>
          <div className="flex justify-center mb-4">
            <div className="sound-wave">
              <div className="sound-bar"></div>
              <div className="sound-bar"></div>
              <div className="sound-bar"></div>
              <div className="sound-bar"></div>
              <div className="sound-bar"></div>
              <div className="sound-bar"></div>
            </div>
          </div>
          <p className="text-2xl text-gray-200 mb-6 font-light tracking-wide stagger-animation">
            AI-Powered Music Genre Classification
          </p>
          <p className="text-lg text-purple-200 mb-8 font-medium stagger-animation">
            ⚡ FastAPI Backend • 🎨 Next.js Frontend • 🚀 Separated Architecture
          </p>
          
          {/* Server Status with enhanced styling */}
          <div className="flex justify-center mb-8">
            <div className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
              serverHealth?.status === 'healthy' 
                ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg shadow-green-500/30 pulse-glow' 
                : 'bg-gradient-to-r from-red-500 to-pink-600 text-white shadow-lg shadow-red-500/30'
            }`}>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${serverHealth?.status === 'healthy' ? 'bg-green-300' : 'bg-red-300'} animate-pulse`}></div>
                <span>Backend: {serverHealth?.status === 'healthy' ? '🟢 Online' : '🔴 Offline'}</span>
                {serverHealth?.gpu_available && <span>• GPU: ✅</span>}
                {serverHealth?.model_loaded && <span>• Model: ✅</span>}
              </div>
            </div>
          </div>
          
          {/* Enhanced Architecture Info */}
          <div className="flex justify-center mb-8">
            <div className="glass-card-premium rounded-2xl px-8 py-4 magic-hover">
              <div className="flex items-center space-x-6 text-lg">
                <div className="text-blue-300 font-bold">FastAPI Backend</div>
                <div className="w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full animate-pulse"></div>
                <div className="text-green-300 font-bold">Next.js Frontend</div>
                <div className="w-2 h-2 bg-gradient-to-r from-blue-400 to-cyan-500 rounded-full animate-pulse"></div>
                <div className="text-purple-300 font-bold">Separated Design</div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          {/* Enhanced File Upload */}
          <div className="glass-card-premium rounded-2xl p-8 mb-8 magic-hover">
            <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6">
              <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-6 text-center">
                🎵 Upload Your Music
              </h2>
              
              <div className="space-y-6">
                <div className="relative">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="block w-full p-8 border-2 border-dashed border-purple-400/50 rounded-xl bg-gradient-to-br from-purple-500/5 to-pink-500/5 hover:from-purple-500/10 hover:to-pink-500/10 transition-all duration-300 cursor-pointer text-center magic-hover ripple"
                  >
                    <div className="space-y-4">
                      <div className="music-visualizer mx-auto w-fit">
                        <div className="vis-bar"></div>
                        <div className="vis-bar"></div>
                        <div className="vis-bar"></div>
                        <div className="vis-bar"></div>
                        <div className="vis-bar"></div>
                      </div>
                      <div className="text-2xl font-bold text-purple-300">
                        {file ? `🎵 ${file.name}` : '📁 Choose Audio File'}
                      </div>
                      <div className="text-gray-400">
                        Supported formats: MP3, WAV, FLAC, M4A
                      </div>
                    </div>
                  </label>
                </div>

                {/* Advanced Options */}
                <div className="flex flex-wrap gap-4 justify-center">
                  <label className="flex items-center space-x-2 glass-card rounded-lg px-4 py-2 cursor-pointer magic-hover">
                    <input
                      type="checkbox"
                      checked={useGpu}
                      onChange={(e) => setUseGpu(e.target.checked)}
                      className="rounded text-purple-500"
                    />
                    <span className="text-sm font-medium">🚀 Use GPU Acceleration</span>
                  </label>
                  
                  <label className="flex items-center space-x-2 glass-card rounded-lg px-4 py-2 cursor-pointer magic-hover">
                    <input
                      type="checkbox"
                      checked={useAdvancedPrediction}
                      onChange={(e) => setUseAdvancedPrediction(e.target.checked)}
                      className="rounded text-purple-500"
                    />
                    <span className="text-sm font-medium">🔬 Advanced Analysis</span>
                  </label>
                </div>

                {/* Enhanced Predict Button */}
                <div className="text-center">
                  <button
                    onClick={classifyMusic}
                    disabled={!file || loading}
                    className="btn-modern px-8 py-4 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden ripple"
                  >
                    {loading ? (
                      <div className="flex items-center justify-center space-x-3">
                        <div className="spinner-premium"></div>
                        <span>🎵 Analyzing Music...</span>
                      </div>
                    ) : (
                      '🚀 Classify Genre'
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Tab Navigation */}
          <div className="tab-container mb-8">
            <div className="relative flex flex-wrap gap-2">
              <div 
                className="tab-indicator"
                style={{
                  width: `${100 / 7}%`,
                  left: `${(['prediction', 'recommendations', 'advanced', 'pipeline', 'batch', 'dashboard', 'ensemble'].indexOf(activeTab)) * (100 / 7)}%`
                }}
              ></div>
              {[
                { id: 'prediction', label: '🎵 Prediction', icon: '🎯' },
                { id: 'recommendations', label: '💿 Recommendations', icon: '🎵' },
                { id: 'advanced', label: '🔬 Advanced', icon: '⚗️' },
                { id: 'pipeline', label: '🔄 Pipeline', icon: '⚙️' },
                { id: 'batch', label: '📦 Batch', icon: '🚀' },
                { id: 'dashboard', label: '📊 Dashboard', icon: '📈' },
                { id: 'ensemble', label: '🎭 Ensemble', icon: '🤖' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`tab-button flex-1 px-4 py-3 text-sm font-medium rounded-lg transition-all duration-300 relative z-10 ${
                    activeTab === tab.id 
                      ? 'text-white' 
                      : 'text-gray-300 hover:text-white'
                  }`}
                >
                  <span className="hidden sm:inline">{tab.label}</span>
                  <span className="sm:hidden text-xl">{tab.icon}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Enhanced Error Display */}
          {error && (
            <div className="glass-card-premium rounded-2xl p-6 mb-8 bg-gradient-to-r from-red-500/10 to-pink-500/10 border border-red-500/30">
              <div className="flex items-center space-x-3">
                <div className="text-red-400 text-2xl">⚠️</div>
                <div>
                  <h3 className="text-lg font-bold text-red-400 mb-2">Error</h3>
                  <p className="text-red-300">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Tab Content */}
          <div className="min-h-[400px]">
            {activeTab === 'prediction' && (
              <div className="space-y-8">
                {result && (
                  <div className="glass-card-premium rounded-2xl p-8 stagger-animation">
                    {/* Enhanced Quick Results */}
                    <div className="text-center mb-8">
                      <div className="text-6xl mb-4 animate-bounce">🎵</div>
                      <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-400 mb-4">
                        Genre Identified!
                      </h2>
                      <div className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-4 neon-glow">
                        {result.predicted_genre.toUpperCase()}
                      </div>
                      <div className="flex justify-center items-center space-x-4 text-lg">
                        <span className="text-gray-300">Confidence:</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-32 h-3 bg-gray-700 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-green-400 to-blue-400 rounded-full transition-all duration-1000"
                              style={{ width: `${result.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-green-400 font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      <p className="text-gray-400 mt-2">
                        Processing time: {result.processing_time.toFixed(2)}s
                      </p>
                    </div>

                    {/* Enhanced Genre Probabilities */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-4">
                          📊 Genre Probabilities
                        </h3>
                        <div className="space-y-3">
                          {Object.entries(result.genre_probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([genre, probability], index) => (
                              <div key={genre} className="stagger-animation" style={{animationDelay: `${index * 0.1}s`}}>
                                <div className="flex justify-between items-center mb-2">
                                  <span className="text-sm font-medium text-gray-300 capitalize">{genre}</span>
                                  <span className="text-sm font-bold text-purple-400">{(probability * 100).toFixed(1)}%</span>
                                </div>
                                <div className="bg-gray-700/50 rounded-full h-4 overflow-hidden">
                                  <div 
                                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-1000"
                                    style={{ 
                                      width: `${probability * 100}%`,
                                      animationDelay: `${index * 0.2}s`
                                    }}
                                  ></div>
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                      
                      {/* Enhanced Radar Chart */}
                      <div className="flex justify-center items-center">
                        <div className="glass-card rounded-2xl p-6">
                          <GenreRadar genreProbabilities={result.genre_probabilities} />
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Architecture Benefits */}
                    <div className="glass-card rounded-2xl p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 mt-8">
                      <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-6 text-center">
                        🏗️ Architecture Benefits
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="text-center space-y-3 stagger-animation">
                          <div className="text-4xl">⚡</div>
                          <h4 className="text-lg font-bold text-green-400">Performance</h4>
                          <div className="space-y-1 text-sm text-gray-300">
                            <div>✅ GPU Optimization</div>
                            <div>✅ FastAPI Speed</div>
                            <div>✅ Async Processing</div>
                          </div>
                        </div>
                        <div className="text-center space-y-3 stagger-animation">
                          <div className="text-4xl">🔧</div>
                          <h4 className="text-lg font-bold text-blue-400">Scalability</h4>
                          <div className="space-y-1 text-sm text-gray-300">
                            <div>✅ API-First Design</div>
                            <div>✅ Independent Services</div>
                            <div>✅ Easy Deployment</div>
                          </div>
                        </div>
                        <div className="text-center space-y-3 stagger-animation">
                          <div className="text-4xl">🎨</div>
                          <h4 className="text-lg font-bold text-purple-400">Experience</h4>
                          <div className="space-y-1 text-sm text-gray-300">
                            <div>✅ Modern UI</div>
                            <div>✅ Real-time Updates</div>
                            <div>✅ Responsive Design</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'advanced' && result && (
              <div className="glass-card-premium rounded-2xl p-6">
                <AdvancedPrediction 
                  genreProbabilities={result.genre_probabilities}
                  confidence={result.confidence}
                />
              </div>
            )}

            {activeTab === 'pipeline' && (
              <div className="glass-card-premium rounded-2xl p-6">
                <AudioProcessingPipeline 
                  isProcessing={loading}
                  audioFile={file}
                  onProcessingComplete={(result) => console.log('Processing complete:', result)}
                />
              </div>
            )}

            {activeTab === 'recommendations' && file && (
              <div className="glass-card-premium rounded-2xl p-6">
                <MusicRecommendations 
                  audioFile={file} 
                  isVisible={true}
                />
              </div>
            )}

            {activeTab === 'batch' && (
              <div className="glass-card-premium rounded-2xl p-6">
                <BatchProcessing apiUrl={API_BASE_URL} />
              </div>
            )}

            {activeTab === 'dashboard' && (
              <div className="glass-card-premium rounded-2xl p-6">
                <ModelDashboard apiUrl={API_BASE_URL} />
              </div>
            )}

            {activeTab === 'ensemble' && (
              <div className="glass-card-premium rounded-2xl p-6">
                <EnsemblePrediction audioFile={file} apiUrl={API_BASE_URL} />
              </div>
            )}
          </div>

          {/* Enhanced Visualization Component */}
          {showVisualization && result?.visualization_data && (
            <div className="glass-card-premium rounded-2xl p-6 mt-8">
              <ClassificationVisualization 
                visualizationData={result.visualization_data}
                isVisible={showVisualization}
              />
            </div>
          )}
        </div>

        {/* Enhanced Footer */}
        <div className="glass-card-premium rounded-2xl p-6 max-w-2xl mx-auto mt-12">
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <div className="music-visualizer">
                <div className="vis-bar"></div>
                <div className="vis-bar"></div>
                <div className="vis-bar"></div>
                <div className="vis-bar"></div>
                <div className="vis-bar"></div>
              </div>
            </div>
            <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              🎵 Genrify - Advanced Music Classification
            </h3>
            <p className="text-sm text-gray-400">
              Built with ❤️ using FastAPI, Next.js, TensorFlow, and Modern Web Technologies
            </p>
            <div className="flex justify-center space-x-4 text-sm text-gray-500">
              <span>🚀 High Performance</span>
              <span>•</span>
              <span>🎯 Accurate Predictions</span>
              <span>•</span>
              <span>🎨 Beautiful UI</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
