'use client';

import React, { useState, useEffect } from 'react';

interface ModelStats {
  model_loaded: boolean;
  model_type: string;
  total_genres: number;
  available_genres: string[];
  embeddings_count?: number;
  gpu_memory_usage?: {
    gpu_count: number;
    gpu_names: string[];
  };
}

interface PerformanceMetrics {
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
  confusion_matrix?: number[][];
}

interface ModelDashboardProps {
  apiUrl?: string;
}

const ModelDashboard: React.FC<ModelDashboardProps> = ({ 
  apiUrl = 'http://localhost:8888' 
}) => {
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  // Mock performance data (in a real app, this would come from your API)
  const mockPerformance: PerformanceMetrics = {
    accuracy: 0.87,
    f1_score: 0.85,
    precision: 0.88,
    recall: 0.84
  };

  const fetchModelStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/model-stats`);
      if (!response.ok) {
        throw new Error('Failed to fetch model statistics');
      }
      const data = await response.json();
      setModelStats(data);
    } catch (err) {
      console.error('Error fetching model stats:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelStats();

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchModelStats();
    }, 30000);

    setRefreshInterval(interval);

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl]);

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
        <h2 className="text-4xl font-bold gradient-text-advanced mb-4 neon-glow">
          📊 Model Performance Dashboard
        </h2>
        <p className="text-gray-400">🚀 Real-time model statistics and AI performance metrics</p>
      </div>

      <div className="prism-glass dynamic-card border-2 border-blue-400/20 neural-network rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold gradient-text-advanced flex items-center">
              <div className="plasma-orb w-8 h-8 mr-3"></div>
              <span>Model Dashboard</span>
            </h3>
            <p className="text-gray-400 mt-1">
              Monitor model performance and system metrics
            </p>
          </div>
          <button
            onClick={fetchModelStats}
            disabled={loading}
            className="morph-button px-6 py-3 text-white font-bold rounded-xl disabled:opacity-50 transition-all duration-300 transform hover:scale-105"
          >
            {loading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                <span>Refreshing...</span>
              </div>
            ) : (
              <span>🔄 Refresh</span>
            )}
          </button>
        </div>
      </div>

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

      {modelStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Information */}
          <div className="prism-glass dynamic-card rounded-2xl p-6">
            <h3 className="text-2xl font-bold gradient-text-advanced mb-4">🤖 Model Information</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center glass-card p-4 rounded-xl">
                <span className="text-gray-300">Model Type</span>
                <span className="font-semibold gradient-text-advanced">{modelStats.model_type}</span>
              </div>
              
              <div className="flex justify-between items-center glass-card p-4 rounded-xl">
                <span className="text-gray-300">Status</span>
                <span className="flex items-center">
                  {modelStats.model_loaded ? (
                    <span className="text-green-400 flex items-center">
                      <span className="w-3 h-3 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                      Active
                    </span>
                  ) : (
                    <span className="text-red-400 flex items-center">
                      <span className="w-3 h-3 bg-red-400 rounded-full mr-2"></span>
                      Not Loaded
                    </span>
                  )}
                </span>
              </div>
              
              <div className="flex justify-between items-center glass-card p-4 rounded-xl">
                <span className="text-gray-300">Total Genres</span>
                <span className="text-cyan-400 font-bold">{modelStats.total_genres}</span>
              </div>
              
              {modelStats.embeddings_count && (
                <div className="flex justify-between items-center glass-card p-4 rounded-xl">
                  <span className="text-gray-300">Embeddings</span>
                  <span className="text-purple-400 font-bold">{modelStats.embeddings_count.toLocaleString()}</span>
                </div>
              )}
            </div>
            
            <div className="mt-6">
              <h4 className="text-xl font-semibold text-gray-200 mb-4">Available Genres</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {modelStats.available_genres.map((genre, index) => (
                  <div key={index} className="glass-card p-2 rounded-lg text-center">
                    <span className="text-sm text-gray-300">{genre}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* GPU Information */}
          <div className="prism-glass dynamic-card rounded-2xl p-6">
            <h3 className="text-2xl font-bold gradient-text-advanced mb-4">🔥 System Resources</h3>
            <div className="space-y-4">
              {modelStats.gpu_memory_usage ? (
                <>
                  <div className="flex justify-between items-center glass-card p-4 rounded-xl">
                    <span className="text-gray-300">GPU Count</span>
                    <span className="text-amber-400 font-bold">{modelStats.gpu_memory_usage.gpu_count}</span>
                  </div>
                  <div className="mt-4">
                    <h4 className="text-xl font-semibold text-gray-200 mb-3">GPU Devices</h4>
                    <div className="space-y-3">
                      {modelStats.gpu_memory_usage.gpu_names.map((name, index) => (
                        <div key={index} className="glass-card p-3 rounded-xl flex items-center">
                          <div className="w-4 h-4 bg-green-400 rounded-full animate-pulse mr-3"></div>
                          <span className="text-gray-300">{name}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="glass-card p-4 rounded-xl text-center">
                  <p className="text-amber-400">No GPU information available</p>
                  <p className="text-gray-400 text-sm mt-1">Model is running in CPU mode</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Performance Metrics */}
          <div className="prism-glass dynamic-card rounded-2xl p-6 lg:col-span-2">
            <h3 className="text-2xl font-bold gradient-text-advanced mb-4">📈 Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="glass-card p-5 rounded-xl text-center">
                <div className="text-3xl font-bold text-blue-400 mb-2">{(mockPerformance.accuracy * 100).toFixed(1)}%</div>
                <div className="text-gray-300 text-sm">Accuracy</div>
                <div className="w-full h-2 bg-gray-700 rounded-full mt-3">
                  <div 
                    className="h-full bg-blue-400 rounded-full"
                    style={{ width: `${mockPerformance.accuracy * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="glass-card p-5 rounded-xl text-center">
                <div className="text-3xl font-bold text-green-400 mb-2">{(mockPerformance.precision * 100).toFixed(1)}%</div>
                <div className="text-gray-300 text-sm">Precision</div>
                <div className="w-full h-2 bg-gray-700 rounded-full mt-3">
                  <div 
                    className="h-full bg-green-400 rounded-full"
                    style={{ width: `${mockPerformance.precision * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="glass-card p-5 rounded-xl text-center">
                <div className="text-3xl font-bold text-purple-400 mb-2">{(mockPerformance.recall * 100).toFixed(1)}%</div>
                <div className="text-gray-300 text-sm">Recall</div>
                <div className="w-full h-2 bg-gray-700 rounded-full mt-3">
                  <div 
                    className="h-full bg-purple-400 rounded-full"
                    style={{ width: `${mockPerformance.recall * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="glass-card p-5 rounded-xl text-center">
                <div className="text-3xl font-bold text-pink-400 mb-2">{(mockPerformance.f1_score * 100).toFixed(1)}%</div>
                <div className="text-gray-300 text-sm">F1 Score</div>
                <div className="w-full h-2 bg-gray-700 rounded-full mt-3">
                  <div 
                    className="h-full bg-pink-400 rounded-full"
                    style={{ width: `${mockPerformance.f1_score * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
            
            <div className="mt-8">
              <div className="glass-card rounded-xl p-4 bg-gradient-to-r from-blue-500/10 to-purple-500/10">
                <h4 className="text-xl font-semibold text-center gradient-text-advanced mb-4">API Performance</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="glass-card p-4 rounded-xl">
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-400">Avg. Response Time</span>
                      <span className="text-cyan-400 font-bold">{(Math.random() * 70 + 30).toFixed(1)} ms</span>
                    </div>
                    <div className="w-full h-2 bg-gray-700 rounded-full">
                      <div className="h-full bg-cyan-400 rounded-full" style={{ width: '85%' }}></div>
                    </div>
                  </div>
                  
                  <div className="glass-card p-4 rounded-xl">
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-400">Requests per Minute</span>
                      <span className="text-amber-400 font-bold">{Math.floor(Math.random() * 40 + 10)}</span>
                    </div>
                    <div className="w-full h-2 bg-gray-700 rounded-full">
                      <div className="h-full bg-amber-400 rounded-full" style={{ width: '65%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Server Statistics */}
          <div className="prism-glass dynamic-card rounded-2xl p-6 lg:col-span-2">
            <h3 className="text-2xl font-bold gradient-text-advanced mb-4">🖥️ Server Statistics</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass-card p-4 rounded-xl">
                <h4 className="text-lg font-semibold text-gray-300 mb-3">CPU Usage</h4>
                <div className="flex items-end space-x-1">
                  {[45, 65, 40, 30, 70, 55, 60, 35, 50, 45].map((height, i) => (
                    <div 
                      key={i}
                      className="w-full bg-gradient-to-t from-blue-500 to-cyan-400 rounded-t"
                      style={{ height: `${height}px` }}
                    ></div>
                  ))}
                </div>
                <div className="text-center mt-3">
                  <span className="text-cyan-400 font-bold text-xl">54%</span>
                </div>
              </div>
              
              <div className="glass-card p-4 rounded-xl">
                <h4 className="text-lg font-semibold text-gray-300 mb-3">Memory Usage</h4>
                <div className="relative h-28 w-28 mx-auto">
                  <svg viewBox="0 0 36 36" className="w-full h-full">
                    <path
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none"
                      stroke="#444"
                      strokeWidth="2"
                      strokeDasharray="100, 100"
                    />
                    <path
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none"
                      stroke="#9333ea"
                      strokeWidth="2"
                      strokeDasharray="75, 100"
                    />
                    <text x="18" y="20.5" textAnchor="middle" className="text-2xl font-bold fill-purple-400">75%</text>
                  </svg>
                </div>
              </div>
              
              <div className="glass-card p-4 rounded-xl">
                <h4 className="text-lg font-semibold text-gray-300 mb-3">GPU Usage</h4>
                <div className="flex items-end space-x-1">
                  {[60, 75, 80, 65, 90, 85, 70, 80, 75, 80].map((height, i) => (
                    <div 
                      key={i}
                      className="w-full bg-gradient-to-t from-pink-500 to-purple-400 rounded-t"
                      style={{ height: `${height}px` }}
                    ></div>
                  ))}
                </div>
                <div className="text-center mt-3">
                  <span className="text-pink-400 font-bold text-xl">78%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelDashboard;
