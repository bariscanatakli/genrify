'use client';

import { useState, useEffect, useRef } from 'react';
import ClassificationVisualization from './components/ClassificationVisualization';
import GenreRadar from './components/GenreRadar';
import AdvancedPrediction from './components/AdvancedPrediction';
import AudioProcessingPipeline from './components/AudioProcessingPipeline';
import BatchProcessing from './components/BatchProcessing';

// Redux imports
import { useAppDispatch, useAppSelector } from './redux/hooks';
import { setFile, setPredictionResult, clearPrediction } from './redux/slices/predictionSlice';
import { 
  setActiveTab,
  setNavHovered,
  setLoading,
  setError,
  setShowVisualization,
  setUseGpu,
  setUseAdvancedPrediction,
  setServerHealth
} from './redux/slices/uiSlice';
import { setPipelineData } from './redux/slices/pipelineSlice';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888';

// Tab navigation configuration
const navigationTabs = [
  { id: 'prediction', label: 'Prediction', icon: '🎯' },
  { id: 'advanced', label: 'Advanced', icon: '⚗️' },
  { id: 'pipeline', label: 'Pipeline', icon: '⚙️' },
  { id: 'spectrogram', label: 'Spectrogram', icon: '📊' }, // Fixed spelling
  { id: 'batch', label: 'Batch', icon: '📦' }
];

// Architecture highlights data
const architectureHighlights = [
  { 
    icon: '⚡', 
    title: 'Performance', 
    points: ['GPU Optimization', 'FastAPI Speed', 'Async Processing'], 
    color: 'text-green-400' 
  },
  { 
    icon: '🔧', 
    title: 'Scalability', 
    points: ['API-First Design', 'Modular Services', 'Cloud Ready'], 
    color: 'text-blue-400' 
  },
  { 
    icon: '🎨', 
    title: 'Experience', 
    points: ['Modern UI/UX', 'Real-time Updates', 'Responsive Design'], 
    color: 'text-purple-400' 
  }
];

export default function Home() {
  // Redux state
    const dispatch = useAppDispatch();
    const { activeTab, navHovered, loading, error, useGpu, useAdvancedPrediction, serverHealth, showVisualization } = useAppSelector(state => state.ui);
    const { file, result } = useAppSelector((state: { prediction: any }) => state.prediction);
    const { pipelineData } = useAppSelector((state: any) => state.pipeline);
    
    // Refs
  const navigationRef = useRef<HTMLDivElement>(null);

  // Effects
  useEffect(() => {
    checkServerHealth();
  }, []);

  // API functions
  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        const health = await response.json();
        dispatch(setServerHealth(health));
      } else {
        dispatch(setServerHealth({ status: 'unhealthy', gpu_available: false, model_loaded: false }));
      }
    } catch {
      dispatch(setServerHealth({ status: 'offline', gpu_available: false, model_loaded: false }));
    }
  };

  const classifyMusic = async () => {
    if (!file) return;

    dispatch(setLoading(true));
    dispatch(setError(null));
    
    // Show pipeline during classification
    dispatch(setActiveTab('pipeline'));

    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_gpu', useGpu.toString());

    // IMPORTANT: Add this parameter explicitly to request visualization data
    if (useAdvancedPrediction) {
      formData.append('include_visualization', 'true');
    }

    try {
      // Pipeline data request
      const pipelineResponse = await fetch(`${API_BASE_URL}/audio/process`, {
        method: 'POST',
        body: formData,
      });
      
      if (pipelineResponse.ok) {
        const pipelineResult = await pipelineResponse.json();
        dispatch(setPipelineData(pipelineResult));
      }

      // Use the correct endpoint based on whether we need visualization data
      const endpoint = useAdvancedPrediction
        ? `${API_BASE_URL}/predict-with-viz`  // Make sure this endpoint exists in your FastAPI server
        : `${API_BASE_URL}/predict`;

      console.log(`Using endpoint: ${endpoint} with advanced mode: ${useAdvancedPrediction}`);

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const prediction = await response.json();
      console.log('Received prediction:', prediction);

      // Log whether visualization data was received
      if (prediction.visualization_data) {
        console.log('Visualization data received');
      } else {
        console.warn('No visualization data in response');
      }

      dispatch(setPredictionResult(prediction));

      // Auto-show visualization if we have visualization data
      if (prediction.visualization_data) {
        dispatch(setShowVisualization(true));
        // Automatically switch to spectrogram tab to show visualizations
        dispatch(setActiveTab('spectrogram')); // Fixed spelling from 'spectogram' to 'spectrogram'
      }
    } catch (err) {
      dispatch(setError(err instanceof Error ? err.message : 'An error occurred'));
    } finally {
      dispatch(setLoading(false));
    }
  };

  // Event handlers
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      dispatch(setFile(selectedFile));
      dispatch(clearPrediction());
      dispatch(setError(null));
      dispatch(setShowVisualization(false));
    }
  };

  // Helper functions
  const handleTabChange = (tabId: string) => {
    dispatch(setActiveTab(tabId as any));
    
    // Enhanced scrolling behavior
    setTimeout(() => {
      const contentArea = document.getElementById('tab-content');
      if (contentArea) {
        const headerOffset = 100;
        const elementPosition = contentArea.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
        
        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    }, 100);
  };

  // UI Components
  const renderBackground = () => (
    <>
      {/* Background particles */}
      <div className="floating-particles fixed inset-0 w-full h-full pointer-events-none z-0" style={{ top: 0, left: 0 }}>
        {Array.from({ length: 75 }, (_, i) => (
          <div
            key={i}
            className="particle"
            style={{
              position: 'absolute',
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 15}s`,
              animationDuration: `${15 + Math.random() * 10}s`,
              pointerEvents: 'none',
            }}
          />
        ))}
      </div>

      {/* Enhanced Animated background particles */}
      <div className="absolute inset-0 overflow-hidden" style={{ pointerEvents: 'none' }}>
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-purple-500/10 to-transparent rounded-full blur-3xl animate-spin" style={{ animationDuration: '20s', pointerEvents: 'none' }}></div>
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-blue-500/10 to-transparent rounded-full blur-3xl animate-spin" style={{ animationDuration: '25s', animationDirection: 'reverse', pointerEvents: 'none' }}></div>
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-pink-500/5 to-yellow-500/5 rounded-full blur-2xl animate-pulse" style={{ pointerEvents: 'none' }}></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-l from-green-500/5 to-blue-500/5 rounded-full blur-2xl animate-pulse" style={{ animationDelay: '2s', pointerEvents: 'none' }}></div>
      </div>
    </>
  );

  const renderHeader = () => (
    <div className="text-center mb-12">
      <div>
        <h1
          className="text-3xl md:text-5xl font-extrabold gradient-text-advanced mb-4 tracking-tight leading-tight md:leading-[1.1] pb-1"
          style={{
            lineHeight: '1.1',
            paddingBottom: '0.25em',
            textShadow: '0 2px 8px rgba(80,0,120,0.15)',
            color: '#fff',
          }}
        >
          🎵 Genrify
        </h1>
      </div>
      
      <div className="flex justify-center mb-4">
        <div className="spectral-bars">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="spectral-bar"></div>
          ))}
        </div>
      </div>
      
      <p className="text-2xl text-gray-200 mb-6 font-light tracking-wide">
        AI-Powered Music Genre Classification
      </p>

      {/* Server Status */}
      <div className="flex justify-center mb-8">
        <div className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
          serverHealth?.status === 'healthy'
            ? 'bg-green-500/20 text-green-300 border border-green-500/30'
            : 'bg-red-500/20 text-red-300 border border-red-500/30'
        }`}>
          <div className="flex items-center space-x-2">
            <div className={`w-2.5 h-2.5 rounded-full ${serverHealth?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
            <span>Backend: {serverHealth?.status === 'healthy' ? 'Online' : 'Offline'}</span>
            {serverHealth?.status === 'healthy' && serverHealth?.gpu_available && <span className="text-xs opacity-80">(GPU Ready)</span>}
            {serverHealth?.status === 'healthy' && serverHealth?.model_loaded && <span className="text-xs opacity-80">(Model Loaded)</span>}
          </div>
        </div>
      </div>

      {/* Architecture Info */}
      <div className="flex justify-center mb-8">
        <div className="prism-glass dynamic-card enhanced-hover cyberpunk-grid rounded-2xl px-8 py-4 magic-hover">
          <div className="flex flex-wrap justify-center items-center gap-x-6 gap-y-2 text-lg">
            {[
              { label: 'FastAPI Backend', color: 'text-blue-300' },
              { label: 'Next.js Frontend', color: 'text-green-300' },
              { label: 'TensorFlow Model', color: 'text-purple-300' }
            ].map((tech, index, arr) => (
              <>
                <div key={tech.label} className={`${tech.color} font-bold flex items-center space-x-2`}>
                  <div className="plasma-orb w-4 h-4"></div>
                  <span>{tech.label}</span>
                </div>
                {index < arr.length - 1 && (
                  <div className="w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full animate-pulse hidden md:block"></div>
                )}
              </>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderNavigation = () => (
    <>
      {/* Desktop Navigation */}
      <div 
        ref={navigationRef}
        className="fixed left-0 top-1/2 transform -translate-y-1/2 z-50 hidden md:block"
      >
        <div 
          className="prism-glass dynamic-card rounded-r-xl p-2 ml-0 shadow-xl transition-all duration-300 ease-in-out overflow-hidden hover:shadow-purple-500/20 hover:shadow-lg"
          style={{ width: navHovered ? '180px' : '74px' }}
          onMouseEnter={() => dispatch(setNavHovered(true))}
          onMouseLeave={() => dispatch(setNavHovered(false))}
        >
          {navigationTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`flex items-center w-full p-2 mb-2 text-sm font-semibold rounded-lg transition-all duration-200 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-opacity-75 clickable ${
                activeTab === tab.id
                  ? 'bg-purple-600 text-white shadow-lg' 
                  : 'text-gray-300 hover:bg-purple-500/40 hover:text-white hover:shadow-md'
              }`}
              role="tab"
              aria-selected={activeTab === tab.id}
              tabIndex={0}
              title={tab.label}
              aria-label={tab.label}
            >
              {/* Icon */}
              <span className="text-2xl flex-shrink-0">
                {tab.icon}
              </span>
              
              {/* Text - visibility controlled by React state */}
              {navHovered && (
                <span className="whitespace-nowrap text-sm ml-3 transition-all duration-300">
                  {tab.label}
                </span>
              )}
            </button>
          ))}
        </div>
        
        {/* Back to top button */}
        <div className="absolute -bottom-16 left-0 w-full flex justify-center">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="prism-glass p-2 rounded-full shadow-lg hover:bg-purple-500/20 transition-all duration-200 clickable"
            title="Go to top"
            aria-label="Back to top"
          >
            <span className="text-xl">⬆️</span>
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden sticky top-0 z-50 mb-4">
        <div className="prism-glass dynamic-card p-2 rounded-xl shadow-xl">
          <div className="flex overflow-x-auto snap-x scrollbar-hide space-x-2">
            {navigationTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id)}
                className={`flex-shrink-0 snap-start flex flex-col items-center justify-center p-2 min-w-[64px] text-sm font-semibold rounded-lg transition-all duration-200 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-opacity-75 clickable ${
                  activeTab === tab.id
                    ? 'bg-purple-600 text-white shadow-lg' 
                    : 'text-gray-300 hover:bg-purple-500/40 hover:text-white hover:shadow-md'
                }`}
                role="tab"
                aria-selected={activeTab === tab.id}
                tabIndex={0}
                aria-label={tab.label}
              >
                <span className="text-xl mb-1">{tab.icon}</span>
                <span className="text-xs line-clamp-1">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </>
  );

  const renderFileUpload = () => (
    <div className="prism-glass dynamic-card enhanced-hover bio-luminescent temporal-distortion rounded-2xl p-8 mb-8 magic-hover">
      <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6">
        <h2 className="text-2xl font-bold gradient-text-advanced mb-6 text-center tracking-wide">
          🎵 Upload Your Music
        </h2>

        <div className="space-y-6">
          {/* File upload area */}
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
              className="clickable group block w-full p-8 border-2 border-dashed border-purple-400/50 rounded-xl bg-gradient-to-br from-purple-500/5 to-pink-500/5 hover:from-purple-500/15 hover:to-pink-500/15 hover:border-purple-400/70 transition-all duration-300 ease-in-out cursor-pointer text-center magic-hover ripple transform hover:scale-[1.02]"
              style={{ position: 'relative', zIndex: 20, pointerEvents: 'auto' }}
            >
              <div className="space-y-4 transition-transform duration-300 ease-in-out group-hover:scale-105">
                <div className="music-visualizer mx-auto w-fit">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="vis-bar"></div>
                  ))}
                </div>
                <div className="text-2xl font-bold text-purple-300 group-hover:text-purple-200 transition-colors duration-300">
                  {file ? `🎵 ${file.name}` : '📁 Choose Audio File'}
                </div>
                <div className="text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
                  Supported formats: MP3, WAV, FLAC, M4A
                </div>
              </div>
            </label>
          </div>

          {/* Options */}
          <div className="flex flex-wrap gap-4 justify-center">
            <label className="clickable flex items-center space-x-3 glass-card rounded-lg px-4 py-3 cursor-pointer magic-hover hover:bg-white/10 transition-colors duration-200" style={{ position: 'relative', zIndex: 20, pointerEvents: 'auto' }}>
              <input
                type="checkbox"
                checked={useGpu}
                onChange={(e) => dispatch(setUseGpu(e.target.checked))}
                className="h-5 w-5 rounded accent-purple-500 bg-transparent border-purple-400/50 focus:ring-2 focus:ring-purple-500 focus:ring-offset-0 focus:ring-offset-transparent"
              />
              <span className="text-sm font-medium">🚀 Use GPU Acceleration</span>
            </label>

            <label className="clickable flex items-center space-x-3 glass-card rounded-lg px-4 py-3 cursor-pointer magic-hover hover:bg-white/10 transition-colors duration-200" style={{ position: 'relative', zIndex: 20, pointerEvents: 'auto' }}>
              <input
                type="checkbox"
                checked={useAdvancedPrediction}
                onChange={(e) => dispatch(setUseAdvancedPrediction(e.target.checked))}
                className="h-5 w-5 rounded accent-purple-500 bg-transparent border-purple-400/50 focus:ring-2 focus:ring-purple-500 focus:ring-offset-0 focus:ring-offset-transparent"
              />
              <span className="text-sm font-medium">🔬 Advanced Analysis</span>
            </label>
          </div>

          {/* Predict Button */}
          <div className="text-center pt-2">
            <button
              onClick={classifyMusic}
              disabled={!file || loading}
              className="clickable morph-button px-10 py-4 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden ripple hover:brightness-110 transition-all duration-200 transform active:scale-95"
              style={{ position: 'relative', zIndex: 20, pointerEvents: 'auto' }}
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
  );

  const renderErrorMessage = () => {
    if (!error) return null;
    
    return (
      <div className="glass-card-premium rounded-2xl p-6 mb-8 bg-gradient-to-r from-red-500/20 to-pink-500/10 border border-red-500/40 shadow-xl">
        <div className="flex items-center space-x-4">
          <div className="text-red-400 text-3xl animate-pulse">⚠️</div>
          <div>
            <h3 className="text-xl font-bold text-red-300 mb-1">Error Occurred</h3>
            <p className="text-red-300/90">{error}</p>
          </div>
        </div>
      </div>
    );
  };

  const renderPredictionResults = () => {
    if (!result) return null;
    
    return (
      <div className="prism-glass enhanced-hover particle-physics rounded-2xl p-6 sm:p-8 stagger-animation shadow-2xl">
        {/* Quick Results */}
        <div className="text-center mb-10">
          <div className="text-7xl mb-4 inline-block animate-bounce" style={{ animationDuration: '1.5s' }}>🎵</div>
          <h2 className="text-4xl font-bold gradient-text-advanced mb-3">
            Genre Identified!
          </h2>
          <div className="text-6xl font-black gradient-text-advanced mb-5 neon-glow tracking-wider">
            {result.predicted_genre.toUpperCase()}
          </div>
          <div className="flex justify-center items-center space-x-3 text-lg">
            <span className="text-gray-300">Confidence:</span>
            <div className="flex items-center space-x-2">
              <div className="w-36 h-4 bg-gray-700/80 rounded-full overflow-hidden shadow-inner">
                <div
                  className="h-full bg-gradient-to-r from-green-400 via-cyan-400 to-blue-500 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${result.confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-green-300 font-bold text-xl">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
          <p className="text-gray-400 mt-3 text-sm">
            Processing time: {result.processing_time.toFixed(2)}s
          </p>
        </div>

        {/* Genre Probabilities and Radar */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          <div className="space-y-4">
            <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-5">
              📊 Genre Probabilities
            </h3>
            <div className="space-y-3.5">
              {Object.entries(result.genre_probabilities)
                .sort(([, a], [, b]) => (b as number) - (a as number))
                .slice(0, 7) // Show top 7 for cleaner UI
                .map(([genre, probability], index) => (
                  <div key={genre} className="stagger-animation" style={{ animationDelay: `${index * 0.08}s` }}>
                    <div className="flex justify-between items-center mb-1.5">
                      <span className="text-base font-medium text-gray-200 capitalize">{genre}</span>
                      <span className="text-base font-bold text-purple-300">{((probability as number) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bg-gray-700/60 rounded-full h-3.5 overflow-hidden shadow-inner">
                      <div
                        className="h-full bg-gradient-to-r from-purple-600 via-pink-500 to-red-500 rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${(probability as number) * 100}%`,
                          animationDelay: `${index * 0.15}s` // Stagger bar animation
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Radar Chart */}
          <div className="flex justify-center items-center pt-4 lg:pt-0">
            <div className="glass-card rounded-2xl p-4 sm:p-6 w-full max-w-md shadow-lg">
              <GenreRadar genreProbabilities={result.genre_probabilities} />
            </div>
          </div>
        </div>

        {/* Architecture Benefits */}
        <div className="glass-card rounded-2xl p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 mt-10 shadow-lg">
          <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 mb-6 text-center">
            🏗️ Architecture Highlights
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {architectureHighlights.map((benefit, idx) => (
              <div key={idx} className="text-center space-y-3 stagger-animation p-3 rounded-lg hover:bg-white/5 transition-colors" style={{ animationDelay: `${idx * 0.1}s` }}>
                <div className={`text-5xl ${benefit.color}`}>{benefit.icon}</div>
                <h4 className={`text-xl font-bold ${benefit.color}`}>{benefit.title}</h4>
                <div className="space-y-1 text-sm text-gray-300">
                  {benefit.points.map(point => <div key={point}>✅ {point}</div>)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderTabContent = () => (
    <div 
      id="tab-content" 
      className="min-h-[400px] scroll-mt-24"
    >
      {activeTab === 'prediction' && (
        <div className="space-y-8">
          {!result ? (
            <div className="text-center p-8">
              <h2 className="text-2xl font-bold mb-4 text-white">🎯 Genre Prediction</h2>
              <p className="text-gray-300">Upload an audio file and click "Classify Genre" to get started.</p>
            </div>
          ) : renderPredictionResults()}
        </div>
      )}

      {activeTab === 'advanced' && result && (
        <div className="prism-glass enhanced-hover quantum-entangled rounded-2xl p-6 shadow-2xl">
          <h2 className="text-2xl font-bold mb-4 text-white">🔬 Advanced AI Analysis</h2>
          <AdvancedPrediction
            genreProbabilities={result.genre_probabilities}
            confidence={result.confidence}
          />
        </div>
      )}

      {activeTab === 'pipeline' && (
        <div className="prism-glass enhanced-hover bio-luminescent rounded-2xl p-6 shadow-2xl">
          <h2 className="text-2xl font-bold mb-4 text-white">⚙️ Audio Processing Pipeline</h2>
          
          {/* Processing modes display */}
          <div className="mb-4 flex flex-wrap gap-2 justify-center">
            <div className={`px-3 py-1 rounded text-sm ${useGpu ? 'bg-green-500/20 text-green-300' : 'bg-gray-500/20 text-gray-300'}`}>
              {useGpu ? '🚀 GPU Accelerated' : '🖥️ CPU Mode'}
            </div>
            <div className={`px-3 py-1 rounded text-sm ${useAdvancedPrediction ? 'bg-purple-500/20 text-purple-300' : 'bg-gray-500/20 text-gray-300'}`}>
              {useAdvancedPrediction ? '🔬 Advanced Analysis' : '📊 Basic Analysis'}
            </div>
          </div>
          
          {!file ? (
            <div className="bg-gray-800/40 rounded-xl p-8 text-center">
              <p className="text-gray-300">Upload an audio file to see the processing pipeline in action.</p>
            </div>
          ) : (
            <AudioProcessingPipeline
              isProcessing={loading}
              audioFile={file}
              useGpu={useGpu}
              advancedAnalysis={useAdvancedPrediction}
              onProcessingComplete={(result) => {
                console.log('Processing complete:', result);
                if (result && result.success === false) {
                  // Show error message if pipeline failed
                  dispatch(setError(result.error || "Pipeline processing failed"));
                }
                dispatch(setPipelineData(result));
              }}
            />
          )}
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="prism-glass enhanced-hover cyberpunk-grid rounded-2xl p-6 shadow-2xl">
          <h2 className="text-2xl font-bold mb-4 text-white">📦 Batch Processing</h2>
          <BatchProcessing apiUrl={API_BASE_URL} />
        </div>
      )}

      {activeTab === 'spectrogram' && (
        <div className="prism-glass enhanced-hover quantum-entangled rounded-2xl p-6 shadow-2xl">
          <h2 className="text-2xl font-bold mb-4 text-white">📊 Spectrogram Analysis</h2>
          
          {!result?.visualization_data ? (
            <div className="bg-gray-800/40 rounded-xl p-8 text-center">
              <p className="text-gray-300">Run a prediction with Advanced Analysis enabled to see spectrograms.</p>
            </div>
          ) : (
            <div className="space-y-8">
              {/* Educational info section */}
              <div className="glass-card p-4 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg mb-6">
                <h4 className="text-lg font-medium text-blue-300 mb-2">What are spectrograms?</h4>
                <p className="text-sm text-gray-300">
                  Spectrograms are visual representations of the spectrum of frequencies in an audio signal as they vary with time.
                  They help identify patterns and characteristics in music that define its genre.
                </p>
              </div>
              
              {/* Linear Spectrogram with enhanced description */}
              <div className="glass-card p-6 rounded-lg">
                <h3 className="text-xl font-semibold mb-2 text-white">Linear Spectrogram</h3>
                <p className="text-sm text-gray-400 mb-4">
                  Shows frequency distribution over time with linear frequency scaling. 
                  Useful for analyzing higher frequency content.
                </p>
                {result.visualization_data.spectrogram ? (
                  <div className="relative">
                    <img 
                      src={result.visualization_data.spectrogram} 
                      alt="Spectrogram" 
                      className="w-full rounded-lg shadow-lg hover:scale-[1.02] transition-transform cursor-zoom-in"
                      onClick={() => window.open(result.visualization_data.spectrogram, '_blank')}
                    />
                    <div className="absolute bottom-2 right-2 bg-black/60 text-xs text-white px-2 py-1 rounded">
                      Click to enlarge
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-300">Spectrogram data not available</p>
                )}
              </div>
              
              {/* Mel Spectrogram with enhanced description */}
              <div className="glass-card p-6 rounded-lg">
                <h3 className="text-xl font-semibold mb-2 text-white">Mel Spectrogram</h3>
                <p className="text-sm text-gray-400 mb-4">
                  Uses mel scale which better approximates human auditory perception.
                  This representation is particularly useful for genre classification models.
                </p>
                {result.visualization_data.mel_spectrogram ? (
                  <div className="relative">
                    <img 
                      src={result.visualization_data.mel_spectrogram} 
                      alt="Mel Spectrogram" 
                      className="w-full rounded-lg shadow-lg hover:scale-[1.02] transition-transform cursor-zoom-in"
                      onClick={() => window.open(result.visualization_data.mel_spectrogram, '_blank')}
                    />
                    <div className="absolute bottom-2 right-2 bg-black/60 text-xs text-white px-2 py-1 rounded">
                      Click to enlarge
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-300">Mel spectrogram data not available</p>
                )}
              </div>
              
              {/* Genre characteristics section based on spectrograms */}
              <div className="glass-card p-6 rounded-lg bg-gradient-to-r from-purple-500/10 to-pink-500/10">
                <h3 className="text-xl font-semibold mb-3 text-white">Spectrogram Insight</h3>
                <p className="text-sm text-gray-300 mb-4">
                  Based on the spectrogram analysis, we can identify key characteristics of this {result.predicted_genre} track:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.predicted_genre === 'Electronic' && (
                    <>
                      <div className="p-3 bg-black/30 rounded-lg">
                        <span className="block text-blue-300 font-medium">Strong bass presence</span>
                        <span className="text-xs text-gray-400">Visible as concentrated energy in lower frequencies</span>
                      </div>
                      <div className="p-3 bg-black/30 rounded-lg">
                        <span className="block text-blue-300 font-medium">Repetitive patterns</span>
                        <span className="text-xs text-gray-400">Shown as recurring structures in the spectrogram</span>
                      </div>
                    </>
                  )}
                  {result.predicted_genre === 'Rock' && (
                    <>
                      <div className="p-3 bg-black/30 rounded-lg">
                        <span className="block text-blue-300 font-medium">Wide frequency range</span>
                        <span className="text-xs text-gray-400">Energy spread across low, mid, and high frequencies</span>
                      </div>
                      <div className="p-3 bg-black/30 rounded-lg">
                        <span className="block text-blue-300 font-medium">Strong mid-range presence</span>
                        <span className="text-xs text-gray-400">Characteristic of guitar and vocal dominance</span>
                      </div>
                    </>
                  )}
                  {/* Add more genre-specific characteristics */}
                  <div className="p-3 bg-black/30 rounded-lg">
                    <span className="block text-blue-300 font-medium">Frequency distribution</span>
                    <span className="text-xs text-gray-400">Typical of {result.predicted_genre} genre signature</span>
                  </div>
                  <div className="p-3 bg-black/30 rounded-lg">
                    <span className="block text-blue-300 font-medium">Temporal structure</span>
                    <span className="text-xs text-gray-400">Reflects rhythm patterns common in this genre</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Info message when needed content is missing */}
      {((activeTab === 'advanced' && !result)) && (
        <div className="prism-glass rounded-2xl p-8 text-center">
          <h3 className="text-xl font-bold mb-4 text-white">
            🔬 Advanced Analysis
          </h3>
          <p className="text-gray-300">
            Run a genre prediction first to see advanced analysis.
          </p>
        </div>
      )}
    </div>
  );

  const renderVisualization = () => {
    if (!showVisualization || !result?.visualization_data) return null;
    
    return (
      <div className="glass-card-premium rounded-2xl p-4 sm:p-6 mt-8 shadow-xl">
        <ClassificationVisualization
          visualizationData={result.visualization_data}
          isVisible={showVisualization}
        />
      </div>
    );
  };

  const renderFloatingButtons = () => (
    <>
      <button
        className="clickable fab-enhanced"
        style={{ bottom: '20px', right: '20px', position: 'fixed', zIndex: 50, pointerEvents: 'auto' }}
        onClick={() => document.getElementById('file-upload')?.click()}
        title="Quick Upload"
      >
        📁
      </button>

      <button
        className="clickable fab-enhanced"
        style={{ bottom: '90px', right: '20px', position: 'fixed', zIndex: 50, pointerEvents: 'auto' }}
        onClick={checkServerHealth}
        title="Refresh Status"
      >
        🔄
      </button>
    </>
  );

  // Main Render
  return (
    <div className="min-h-screen morphing-bg aurora-bg neural-network cosmic-dust hyperspace-tunnel">
      {renderBackground()}

      <div className="relative z-10 container mx-auto px-4 py-8">
        {renderHeader()}

        {/* Mobile and Desktop Navigation */}
        {renderNavigation()}

        {/* Content container */}
        <div className="flex justify-center">
          {/* Left spacing for desktop */}
          <div className="w-20 flex-shrink-0 hidden md:block"></div>
          
          {/* Main content */}
          <div className="max-w-3xl flex-grow">
            {renderFileUpload()}
            {renderErrorMessage()}
            <div id="main-content-area" className="scroll-mt-8">
              {renderTabContent()}
              {renderVisualization()}
            </div>
          </div>
          
          {/* Right spacing for desktop */}
          <div className="w-20 flex-shrink-0 hidden md:block"></div>
        </div>

        {renderFloatingButtons()}
      </div>
    </div>
  );
}