"use client";

import { useState, useRef, useEffect } from "react";
import { formatFileSize, isAudioFile } from "./utils/audio";
import DependencyStatus from "./components/DependencyStatus";
import ClassificationVisualization from "./components/ClassificationVisualization";

// Types
type Genre = string;

type GenrePrediction = {
  predicted_genre: Genre;
  confidence: number;
  genre_probabilities: Record<Genre, number>;
  using_mock?: boolean;
  dependency_status?: any;
};

export default function Home() {
  // State management
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<GenrePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelsStatus, setModelsStatus] = useState<{loading: boolean, available: boolean, missing: string[]}>({
    loading: true,
    available: false,
    missing: []
  });
  const [dependencyStatus, setDependencyStatus] = useState<{
    checked: boolean, 
    missing: string[], 
    critical: string[],
    modelsLoaded: boolean,
    usingMock: boolean,
    missingLibraries?: Record<string, boolean>,
    functionality?: {
      ml_ready: boolean,
      search_ready: boolean,
      audio_ready: boolean,
      embeddings_available: boolean,
      models_available: boolean,
    }
  }>({
    checked: false,
    missing: [],
    critical: [],
    modelsLoaded: false,
    usingMock: true
  });
  
  // Visualization state
  const [visualizationData, setVisualizationData] = useState<any>(null);
  const [processingSteps, setProcessingSteps] = useState<any[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Available genres from the classifier model
  const genres = [
    "Rock",
    "Electronic", 
    "Experimental",
    "Hip-Hop",
    "Folk",
    "Instrumental",
    "Pop",
    "International"
  ];

  // Check if models are available
  useEffect(() => {
    fetch('/api/check-models')
      .then(res => res.json())
      .then((data) => {
        setModelsStatus({
          loading: false,
          available: data.available,
          missing: data.missing || []
        });
      })
      .catch(err => {
        console.error('Error checking models:', err);
        setModelsStatus({
          loading: false,
          available: false,
          missing: ['Error checking models availability']
        });
      });
  }, []);

  // Check Python dependencies
  useEffect(() => {
    fetch('/api/check-dependencies')
      .then(res => res.json())
      .then(data => {
        setDependencyStatus({
          checked: true,
          missing: data.missingPackages || [],
          critical: data.missingCritical || [],
          modelsLoaded: data.modelsLoaded || false,
          usingMock: data.usingMock || false,
          missingLibraries: data.missingLibraries || {},
          functionality: data.functionality || undefined
        });
      })
      .catch(err => {
        console.error('Error checking dependencies:', err);
        setDependencyStatus({
          checked: true,
          missing: ['Error checking dependencies'],
          critical: ['Error checking Python environment'],
          modelsLoaded: false,
          usingMock: true
        });
      });
  }, []);

  const classifyAudio = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setVisualizationData(null);
    
    // Initialize processing steps
    const steps = [
      { step: 1, title: "Audio Loading", description: "Loading and preprocessing audio file", status: 'processing' as 'pending' | 'processing' | 'completed' | 'error' },
      { step: 2, title: "Feature Extraction", description: "Extracting MFCC, spectral, and temporal features", status: 'pending' as 'pending' | 'processing' | 'completed' | 'error' },
      { step: 3, title: "Spectrogram Generation", description: "Creating spectrograms for visualization", status: 'pending' as 'pending' | 'processing' | 'completed' | 'error' },
      { step: 4, title: "Model Prediction", description: "Running genre classification model", status: 'pending' as 'pending' | 'processing' | 'completed' | 'error' },
      { step: 5, title: "Results Processing", description: "Processing prediction results", status: 'pending' as 'pending' | 'processing' | 'completed' | 'error' }
    ];
    setProcessingSteps([...steps]);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('include_visualization', 'true');
      
      // Update step 1 to completed
      steps[0].status = 'completed';
      steps[1].status = 'processing';
      setProcessingSteps([...steps]);
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Classification failed: ${response.statusText}`);
      }
      
      const data: GenrePrediction & { visualization_data?: any } = await response.json();
      
      // Update all steps to completed
      steps.forEach(step => step.status = 'completed');
      setProcessingSteps([...steps]);
      
      setPrediction(data);
      
      // Set visualization data if available
      if (data.visualization_data) {
        setVisualizationData(data.visualization_data);
      }
      
    } catch (err) {
      console.error('Error during classification:', err);
      setError(err instanceof Error ? err.message : 'Failed to classify audio');
      
      // Mark current step as error
      const currentStep = steps.findIndex(s => s.status === 'processing');
      if (currentStep !== -1) {
        steps[currentStep].status = 'error';
        setProcessingSteps([...steps]);
      }
      
      // Set mock data during development/errors
      if (process.env.NODE_ENV === 'development') {
        setPrediction({
          predicted_genre: "Electronic",
          confidence: 0.85,
          genre_probabilities: {
            "Electronic": 0.85,
            "Experimental": 0.06,
            "Hip-Hop": 0.04,
            "Rock": 0.02,
            "Pop": 0.01,
            "Folk": 0.01,
            "Instrumental": 0.005,
            "International": 0.005
          },
          using_mock: true
        });
        
        // Set mock visualization data
        setVisualizationData({
          mfcc_features: Array.from({ length: 13 }, (_, i) => 
            Array.from({ length: 100 }, () => Math.random() * 2 - 1)
          ),
          spectral_features: {
            spectral_centroid: [1500.5],
            spectral_rolloff: [3000.2],
            zero_crossing_rate: [0.045]
          },
          tempo: 128.5,
          model_predictions: Array.from({ length: 8 }, () => Math.random())
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (isAudioFile(selectedFile)) {
        setFile(selectedFile);
        setPrediction(null);
      } else {
        setError("Please select an audio file (MP3, WAV, etc.)");
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (file) {
      await classifyAudio(file);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Sort genre probabilities by value for visualization
  const sortedProbabilities = prediction
    ? Object.entries(prediction.genre_probabilities)
        .sort(([,a], [,b]) => b - a)
    : [];

  // Update dependency status when prediction is received
  useEffect(() => {
    if (prediction && prediction.dependency_status) {
      setDependencyStatus(prev => ({
        ...prev,
        missingLibraries: prediction.dependency_status.libraries || {},
        functionality: prediction.dependency_status.functionality || undefined
      }));
    }
  }, [prediction]);

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-900">
      <main className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-8 h-8 text-white">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
            </svg>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Music Genre Classifier
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto leading-relaxed">
            Upload an audio file and discover its musical genre using advanced AI classification. 
            Get detailed probability scores across {genres.length} different genres.
          </p>
          
          {/* Status indicators */}
          {modelsStatus.loading ? (
            <div className="mt-4 flex items-center justify-center gap-2 text-blue-500 text-sm">
              <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
              Checking models availability...
            </div>
          ) : !dependencyStatus.checked ? (
            null
          ) : (
            !modelsStatus.available && (
              <div className="mt-4 text-amber-600 dark:text-amber-400 text-sm bg-amber-50 dark:bg-amber-900/20 px-4 py-2 rounded-lg inline-block">
                ⚠️ Some models not available: {modelsStatus.missing.join(', ')}
              </div>
            )
          )}
          
          <DependencyStatus status={dependencyStatus} />
          
          {prediction?.using_mock && (
            <div className="mt-3 text-amber-600 dark:text-amber-400 text-sm bg-amber-50 dark:bg-amber-900/20 px-4 py-2 rounded-lg inline-block">
              ⚠️ Using mock data for some functionality
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200 px-6 py-4 rounded-xl mb-8">
            <div className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
              </svg>
              <p className="font-medium">{error}</p>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="bg-white/80 dark:bg-slate-800/80 p-8 rounded-2xl backdrop-blur-sm shadow-xl border border-white/20 mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-8 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors">
              {!file ? (
                <div className="space-y-4">
                  <div className="flex justify-center">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/50 dark:to-purple-900/50 rounded-full flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 text-blue-600 dark:text-blue-400">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <p className="text-lg font-medium text-slate-700 dark:text-slate-200 mb-2">
                      Drop your audio file here
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                      Supports MP3, WAV, FLAC, and other audio formats
                    </p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    id="music-file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <label
                    htmlFor="music-file"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl cursor-pointer hover:from-blue-600 hover:to-purple-700 transition-all duration-200 font-medium shadow-lg"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
                    </svg>
                    Choose Audio File
                  </label>
                </div>
              ) : (
                <div className="flex items-center justify-between bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-green-100 to-blue-100 dark:from-green-900/50 dark:to-blue-900/50 rounded-lg flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-green-600 dark:text-green-400">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                      </svg>
                    </div>
                    <div className="text-left">
                      <p className="font-semibold text-slate-800 dark:text-slate-200 truncate max-w-xs">{file.name}</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                  </div>
                  <button 
                    type="button"
                    onClick={clearFile}
                    className="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 p-2 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
            
            <div className="flex justify-center">
              <button
                type="submit"
                disabled={!file || isAnalyzing}
                className={`px-8 py-3 rounded-xl font-semibold transition-all duration-200 ${
                  !file || isAnalyzing
                    ? "bg-slate-200 dark:bg-slate-700 cursor-not-allowed text-slate-400 dark:text-slate-500"
                    : "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105"
                }`}
              >
                {isAnalyzing ? (
                  <span className="flex items-center gap-3">
                    <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                    Analyzing Audio...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                    </svg>
                    Classify Genre
                  </span>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        {prediction && (
          <div className="bg-white/80 dark:bg-slate-800/80 p-8 rounded-2xl backdrop-blur-sm shadow-xl border border-white/20 mb-8">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-10 h-10 text-white">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold mb-2 text-slate-800 dark:text-slate-100">Classification Complete</h2>
              <div className="space-y-2">
                <p className="text-xl text-slate-600 dark:text-slate-300">
                  Detected Genre: <span className="font-bold text-blue-600 dark:text-blue-400">{prediction.predicted_genre}</span>
                </p>
                <div className="inline-flex items-center gap-2 bg-blue-50 dark:bg-blue-900/30 px-4 py-2 rounded-full">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                    {Math.round(prediction.confidence * 100)}% confidence
                  </span>
                </div>
              </div>
            </div>

            {/* Genre Probabilities Visualization */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                </svg>
                Genre Probability Distribution
              </h3>
              <div className="space-y-3">
                {sortedProbabilities.map(([genre, probability], index) => (
                  <div key={genre} className="group">
                    <div className="flex items-center justify-between mb-1">
                      <span className={`text-sm font-medium ${
                        index === 0 ? 'text-blue-600 dark:text-blue-400' : 'text-slate-600 dark:text-slate-300'
                      }`}>
                        {genre}
                      </span>
                      <span className={`text-sm font-bold ${
                        index === 0 ? 'text-blue-600 dark:text-blue-400' : 'text-slate-500 dark:text-slate-400'
                      }`}>
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3 overflow-hidden">
                      <div 
                        className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                          index === 0 
                            ? 'bg-gradient-to-r from-blue-500 to-blue-600' 
                            : 'bg-gradient-to-r from-slate-400 to-slate-500'
                        }`}
                        style={{ 
                          width: `${Math.max(2, probability * 100)}%`,
                          transitionDelay: `${index * 100}ms`
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Visualization Section */}
        {(prediction || isAnalyzing || processingSteps.length > 0) && (
          <div className="bg-white/80 dark:bg-slate-800/80 p-8 rounded-2xl backdrop-blur-sm shadow-xl border border-white/20 mb-8">
            <h2 className="text-2xl font-bold mb-6 text-slate-800 dark:text-slate-100 flex items-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-7 h-7 text-purple-500">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l-1-3m1 3l-1-3m-16.5-3h16.5" />
              </svg>
              Classification Analysis
            </h2>
            <ClassificationVisualization
              isProcessing={isAnalyzing}
              visualizationData={visualizationData}
              steps={processingSteps}
              genrePredictions={prediction?.genre_probabilities}
            />
          </div>
        )}

        {/* Info Section */}
        <div className="bg-white/80 dark:bg-slate-800/80 p-8 rounded-2xl backdrop-blur-sm shadow-xl border border-white/20">
          <h2 className="text-2xl font-bold mb-6 text-slate-800 dark:text-slate-100 flex items-center gap-3">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-7 h-7 text-blue-500">
              <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
            </svg>
            How It Works
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-bold">1</span>
                </div>
                <div>
                  <h3 className="font-semibold text-slate-800 dark:text-slate-200 mb-1">Audio Processing</h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Upload your audio file in any common format (MP3, WAV, FLAC)</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-bold">2</span>
                </div>
                <div>
                  <h3 className="font-semibold text-slate-800 dark:text-slate-200 mb-1">Feature Extraction</h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Convert audio to spectrogram and extract key musical features</p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-bold">3</span>
                </div>
                <div>
                  <h3 className="font-semibold text-slate-800 dark:text-slate-200 mb-1">AI Classification</h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Deep learning model analyzes patterns to predict genre</p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-bold">4</span>
                </div>
                <div>
                  <h3 className="font-semibold text-slate-800 dark:text-slate-200 mb-1">Results</h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Get detailed probabilities across {genres.length} different genres</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-8 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-xl">
            <h4 className="font-semibold text-slate-800 dark:text-slate-200 mb-2">Supported Genres</h4>
            <div className="flex flex-wrap gap-2">
              {genres.map((genre) => (
                <span 
                  key={genre}
                  className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-sm font-medium"
                >
                  {genre}
                </span>
              ))}
            </div>
          </div>
        </div>
      </main>

      <footer className="mt-16 text-center text-slate-500 dark:text-slate-400">
        <div className="max-w-2xl mx-auto pb-8">
          <p className="text-sm">
            Powered by advanced deep learning and convolutional neural networks for accurate music genre classification
          </p>
        </div>
      </footer>
    </div>
  );
}
