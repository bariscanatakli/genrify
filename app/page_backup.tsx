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

      // Use visualization endpoint if advanced prediction is enabled
      const endpoint = useAdvancedPrediction ? '/predict-with-viz' : '/predict';
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
            <div className="glass-card glass-card-hover rounded-2xl px-8 py-4">
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
          <div className="glass-card glass-card-hover rounded-3xl p-8 mb-8 border-2 border-purple-500/20">
            <h2 className="text-3xl font-bold text-white mb-6 text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
              🎶 Upload Your Music
            </h2>
            
            <div className="space-y-6">
              <div className="relative">
                <input
                  type="file"
                  accept=".mp3,.wav,.m4a"
                  onChange={handleFileChange}
                  className="w-full p-4 bg-white/10 backdrop-blur-md rounded-2xl text-white placeholder-gray-300 border-2 border-purple-300/20 hover:border-purple-300/40 transition-all duration-300 file:mr-4 file:py-3 file:px-6 file:rounded-full file:border-0 file:text-sm file:font-bold file:bg-gradient-to-r file:from-purple-500 file:to-pink-500 file:text-white hover:file:from-purple-600 hover:file:to-pink-600 file:transition-all file:duration-300"
                />
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/10 to-pink-500/10 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                <p className="text-sm text-purple-200 mt-3 text-center">
                  ✨ Supported formats: MP3, WAV, M4A • Maximum size: 100MB
                </p>
              </div>

              {/* Enhanced Options */}
              <div className="flex justify-center space-x-8">
                <label className="flex items-center text-white cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={useGpu}
                    onChange={(e) => setUseGpu(e.target.checked)}
                    className="mr-3 w-5 h-5 rounded bg-white/20 border-2 border-purple-400 text-purple-600 focus:ring-purple-500 focus:ring-2"
                  />
                  <span className="font-semibold group-hover:text-purple-300 transition-colors">
                    🚀 GPU Acceleration
                  </span>
                </label>
                
                <label className="flex items-center text-white cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={useAdvancedPrediction}
                    onChange={(e) => setUseAdvancedPrediction(e.target.checked)}
                    className="mr-3 w-5 h-5 rounded bg-white/20 border-2 border-blue-400 text-blue-600 focus:ring-blue-500 focus:ring-2"
                  />
                  <span className="font-semibold group-hover:text-blue-300 transition-colors">
                    🎨 Enable Visualizations
                  </span>
                </label>
              </div>

              <button
                onClick={handlePredict}
                disabled={!file || loading || serverHealth?.status !== 'healthy'}
                className="w-full py-4 px-8 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 text-white font-bold rounded-2xl hover:from-purple-700 hover:via-pink-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/30 btn-modern text-lg"
              >
                {loading ? (
                  <div className="flex items-center justify-center space-x-3">
                    <div className="animate-spin w-6 h-6 border-3 border-white border-t-transparent rounded-full"></div>
                    <span>🤖 AI is analyzing your music...</span>
                  </div>
                ) : (
                  <span>{useAdvancedPrediction ? '🎯 Predict Genre + Visualize' : '🎯 Predict Genre'}</span>
                )}
              </button>
            </div>
          </div>
              </button>
            </div>
          </div>

          {/* Enhanced Loading */}
          {loading && (
            <div className="glass-card rounded-3xl p-8 mb-8 text-center border-2 border-blue-400/30">
              <div className="flex flex-col items-center space-y-4">
                <div className="relative">
                  <div className="animate-spin w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                  <div className="absolute inset-0 animate-ping w-16 h-16 border-4 border-blue-300 border-t-transparent rounded-full opacity-20"></div>
                </div>
                <div className="space-y-2">
                  <h3 className="text-2xl font-bold text-white">🤖 AI is analyzing your music</h3>
                  <p className="text-gray-200 text-lg">
                    FastAPI backend is processing (~{useAdvancedPrediction ? '20-25' : '8-12'} seconds)
                  </p>
                  {useAdvancedPrediction && (
                    <p className="text-purple-300 font-semibold">
                      🎨 Generating spectrograms and audio features...
                    </p>
                  )}
                </div>
                {/* Progress animation */}
                <div className="w-full max-w-md">
                  <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                    <div className="progress-bar h-full animate-pulse"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced Error */}
          {error && (
            <div className="glass-card rounded-3xl p-8 mb-8 border-2 border-red-400/30 bg-red-500/10">
              <div className="text-center">
                <div className="text-6xl mb-4">❌</div>
                <h3 className="text-2xl font-bold text-red-400 mb-4">Oops! Something went wrong</h3>
                <p className="text-white text-lg bg-red-500/20 rounded-xl p-4">{error}</p>
              </div>
            </div>
          )}

          {/* Enhanced Results */}
          {result && (
            <div className="space-y-8">
              {/* Enhanced Tab Navigation */}
              <div className="glass-card rounded-3xl p-8 border-2 border-purple-400/20">
                {/* Enhanced tab buttons */}
                <div className="flex flex-wrap gap-3 mb-8 justify-center">
                  {[
                    { id: 'prediction', label: '🎵 Results', color: 'from-purple-500 to-pink-500' },
                    { id: 'advanced', label: '📊 Advanced', color: 'from-blue-500 to-cyan-500' },
                    { id: 'pipeline', label: '⚙️ Pipeline', color: 'from-green-500 to-teal-500' },
                    { id: 'recommendations', label: '🎯 Discover', color: 'from-orange-500 to-red-500' },
                    { id: 'batch', label: '📂 Batch', color: 'from-violet-500 to-purple-500' },
                    { id: 'dashboard', label: '📊 Dashboard', color: 'from-indigo-500 to-blue-500' },
                    { id: 'ensemble', label: '🎯 Ensemble', color: 'from-pink-500 to-rose-500' }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`px-6 py-3 rounded-2xl font-bold transition-all duration-300 transform hover:scale-105 tab-button ${
                        activeTab === tab.id
                          ? `bg-gradient-to-r ${tab.color} text-white shadow-lg active`
                          : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 hover:text-white'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Enhanced Quick Results */}
                <div className="bg-gradient-to-r from-purple-500/20 via-pink-500/20 to-blue-500/20 rounded-3xl p-8 mb-8 border border-purple-300/30">
                  <h3 className="text-2xl font-bold text-center text-white mb-6">🎉 Prediction Results</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center glass-card rounded-2xl p-6">
                      <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-500 mb-2">
                        {result.predicted_genre}
                      </div>
                      <div className="text-gray-300 font-semibold">🎭 Predicted Genre</div>
                    </div>
                    
                    <div className="text-center glass-card rounded-2xl p-6">
                      <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-500 mb-2">
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-gray-300 font-semibold">🎯 Confidence</div>
                    </div>
                    
                    <div className="text-center glass-card rounded-2xl p-6">
                      <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500 mb-2">
                        {result.processing_time}s
                      </div>
                      <div className="text-gray-300 font-semibold">⚡ Speed</div>
                    </div>
                  </div>
                </div>

                {/* Enhanced Tab Content */}
                {activeTab === 'prediction' && (
                  <div className="space-y-8">
                    {/* Enhanced Genre Probabilities */}
                    <div className="glass-card rounded-2xl p-6">
                      <div className="flex justify-between items-center mb-6">
                        <h4 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                          🎭 Genre Analysis
                        </h4>
                        {result.visualization_data && (
                          <button
                            onClick={() => setShowVisualization(!showVisualization)}
                            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 font-semibold"
                          >
                            {showVisualization ? '📊 Hide Visualizations' : '🎨 Show Visualizations'}
                          </button>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Enhanced Probability Bars */}
                        <div className="space-y-4">
                          {result && Object.entries(result.genre_probabilities)
                            .sort(([,a], [,b]) => b - a)
                            .map(([genre, probability]) => (
                              <div key={genre} className="group">
                                <div className="flex items-center justify-between mb-2">
                                  <div className="text-lg text-white font-bold">{genre}</div>
                                  <div className="text-lg text-purple-300 font-bold">
                                    {(probability * 100).toFixed(1)}%
                                  </div>
                                </div>
                                <div className="bg-gray-700/50 rounded-full h-4 overflow-hidden">
                                  <div
                                    className="progress-bar h-full transition-all duration-1000 ease-out"
                                    style={{ width: `${probability * 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            ))}
                        </div>
                        
                        {/* Enhanced Radar Chart */}
                        <div className="flex justify-center items-center">
                          <div className="glass-card rounded-2xl p-6">
                            {result && <GenreRadar genreProbabilities={result.genre_probabilities} />}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Architecture Benefits */}
                    <div className="glass-card rounded-2xl p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10">
                      <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-6 text-center">
                        🏗️ Architecture Benefits
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="text-center space-y-3">
                          <div className="text-4xl">⚡</div>
                          <h4 className="text-lg font-bold text-green-400">Performance</h4>
                          <div className="space-y-1 text-sm text-gray-300">
                            <div>✅ GPU Optimization</div>
                            <div>✅ FastAPI Speed</div>
                            <div>✅ Async Processing</div>
                          </div>
                        </div>
                        <div className="text-center space-y-3">
                          <div className="text-4xl">🔧</div>
                          <h4 className="text-lg font-bold text-blue-400">Scalability</h4>
                          <div className="space-y-1 text-sm text-gray-300">
                            <div>✅ API-First Design</div>
                            <div>✅ Independent Services</div>
                            <div>✅ Easy Deployment</div>
                          </div>
                        </div>
                        <div className="text-center space-y-3">
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

                {activeTab === 'advanced' && result && (
                  <div className="glass-card rounded-2xl p-6">
                    <AdvancedPrediction 
                      genreProbabilities={result.genre_probabilities}
                      confidence={result.confidence}
                    />
                  </div>
                )}

                {activeTab === 'pipeline' && (
                  <div className="glass-card rounded-2xl p-6">
                    <AudioProcessingPipeline 
                      isProcessing={loading}
                      audioFile={file}
                      onProcessingComplete={(result) => console.log('Processing complete:', result)}
                    />
                  </div>
                )}

                {activeTab === 'recommendations' && file && (
                  <div className="glass-card rounded-2xl p-6">
                    <MusicRecommendations 
                      audioFile={file} 
                      isVisible={true}
                    />
                  </div>
                )}

                {activeTab === 'batch' && (
                  <div className="glass-card rounded-2xl p-6">
                    <BatchProcessing apiUrl={API_BASE_URL} />
                  </div>
                )}

                {activeTab === 'dashboard' && (
                  <div className="glass-card rounded-2xl p-6">
                    <ModelDashboard apiUrl={API_BASE_URL} />
                  </div>
                )}

                {activeTab === 'ensemble' && (
                  <div className="glass-card rounded-2xl p-6">
                    <EnsemblePrediction audioFile={file} apiUrl={API_BASE_URL} />
                  </div>
                )}
              </div>

              {/* Enhanced Visualization Component */}
              {showVisualization && result?.visualization_data && activeTab === 'prediction' && (
                <div className="glass-card rounded-3xl p-8 border-2 border-purple-400/30">
                  <h3 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-6 text-center">
                    🎨 Audio Visualizations
                  </h3>
                  <ClassificationVisualization
                    isProcessing={loading}
                    visualizationData={result.visualization_data}
                    steps={[]}
                    genrePredictions={result.genre_probabilities}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Enhanced Footer */}
        <div className="text-center mt-16 pb-8">
          <div className="glass-card rounded-2xl p-6 max-w-2xl mx-auto">
            <p className="text-gray-300 mb-4">
              🚀 Powered by <span className="font-bold text-purple-400">FastAPI</span> + 
              <span className="font-bold text-blue-400"> Next.js</span> + 
              <span className="font-bold text-green-400"> PyTorch</span>
            </p>
            <p className="text-sm text-gray-400">
              Modern separated architecture for scalable AI music analysis
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}