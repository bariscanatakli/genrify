"use client";

import React, { useState, useEffect } from 'react';

interface ClassificationStep {
  step: number;
  title: string;
  description: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  duration?: number;
  data?: any;
}

interface VisualizationData {
  spectrogram?: string; // Base64 encoded image
  mel_spectrogram?: string;
  mfcc_features?: number[][];
  chroma_features?: number[][];
  spectral_features?: {
    spectral_centroid: number[];
    spectral_rolloff: number[];
    zero_crossing_rate: number[];
  };
  tempo?: number;
  raw_features?: number[];
  model_predictions?: number[];
}

interface ClassificationVisualizationProps {
  isProcessing: boolean;
  visualizationData: VisualizationData | null;
  steps: ClassificationStep[];
  genrePredictions?: Record<string, number>;
}

const ClassificationVisualization: React.FC<ClassificationVisualizationProps> = ({
  isProcessing,
  visualizationData,
  steps,
  genrePredictions
}) => {
  const [activeTab, setActiveTab] = useState<'pipeline' | 'spectrogram' | 'features' | 'predictions'>('pipeline');

  const tabs = [
    { id: 'pipeline', label: 'Processing Pipeline', icon: '🔄' },
    { id: 'spectrogram', label: 'Spectrograms', icon: '📊' },
    { id: 'features', label: 'Audio Features', icon: '🎵' },
    { id: 'predictions', label: 'Model Output', icon: '🎯' }
  ];

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✅';
      case 'processing': return '⏳';
      case 'error': return '❌';
      default: return '⚪';
    }
  };

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 border-green-400/50';
      case 'processing': return 'text-blue-400 border-blue-400/50';
      case 'error': return 'text-red-400 border-red-400/50';
      default: return 'text-gray-400 border-gray-600/50';
    }
  };
  const ProcessingPipeline = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white mb-4">Classification Process</h3>
      
      {/* Explanation Section */}
      <div className="bg-orange-900/20 backdrop-blur-sm rounded-lg p-4 border-l-4 border-orange-400">
        <h4 className="text-md font-medium text-orange-300 mb-2">🔄 Understanding the Pipeline</h4>
        <div className="text-sm text-gray-300 space-y-2">
          <p><strong>Audio Loading:</strong> The system reads your audio file and converts it to a standardized format for analysis.</p>
          <p><strong>Feature Extraction:</strong> Key audio characteristics are extracted including spectrograms, MFCC coefficients, tempo, and spectral features.</p>
          <p><strong>Preprocessing:</strong> Features are normalized and prepared for the neural network model input.</p>
          <p><strong>Model Inference:</strong> The trained CNN model processes the features and generates genre predictions.</p>
          <p><strong>Post-processing:</strong> Raw model outputs are converted to human-readable probabilities and visualizations are generated.</p>
        </div>
      </div>
      
      {steps.map((step) => (
        <div 
          key={step.step}
          className={`p-4 rounded-lg border ${getStepColor(step.status)} bg-black/20 backdrop-blur-sm`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-xl">{getStepIcon(step.status)}</span>
              <div>
                <h4 className="font-medium">{step.title}</h4>
                <p className="text-sm opacity-80">{step.description}</p>
              </div>
            </div>
            {step.duration && (
              <span className="text-sm opacity-60">{step.duration}ms</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
  const SpectrogramView = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">Spectrograms</h3>
      
      {/* Explanation Section */}
      <div className="bg-blue-900/20 backdrop-blur-sm rounded-lg p-4 border-l-4 border-blue-400">
        <h4 className="text-md font-medium text-blue-300 mb-2">📚 Understanding Spectrograms</h4>
        <div className="text-sm text-gray-300 space-y-2">
          <p><strong>Spectrogram:</strong> A visual representation showing how the frequency content of audio changes over time. The X-axis represents time, Y-axis represents frequency, and color intensity shows the magnitude of each frequency component.</p>
          <p><strong>Raw Spectrogram:</strong> Shows the complete frequency spectrum using Short-Time Fourier Transform (STFT). Useful for seeing all frequency details but can be computationally intensive.</p>
          <p><strong>Mel Spectrogram:</strong> Uses the mel scale, which better matches human auditory perception. Lower frequencies are given more resolution, making it ideal for music analysis and machine learning models.</p>
        </div>
      </div>
      
      {visualizationData?.spectrogram && (
        <div className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
          <h4 className="text-md font-medium text-white mb-3">Raw Spectrogram</h4>
          <div className="text-xs text-gray-400 mb-3">
            Linear frequency scale • Full spectrum analysis • High computational detail
          </div>
          <div className="bg-black/40 rounded-lg p-4">
            <img 
              src={`data:image/png;base64,${visualizationData.spectrogram}`}
              alt="Audio Spectrogram"
              className="w-full h-auto rounded-lg"
            />
          </div>
        </div>
      )}

      {visualizationData?.mel_spectrogram && (
        <div className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
          <h4 className="text-md font-medium text-white mb-3">Mel Spectrogram</h4>
          <div className="text-xs text-gray-400 mb-3">
            Perceptually-motivated scale • Optimized for human hearing • ML model input
          </div>
          <div className="bg-black/40 rounded-lg p-4">
            <img 
              src={`data:image/png;base64,${visualizationData.mel_spectrogram}`}
              alt="Mel Spectrogram"
              className="w-full h-auto rounded-lg"
            />
          </div>
        </div>
      )}

      {!visualizationData?.spectrogram && !visualizationData?.mel_spectrogram && (
        <div className="text-center py-8 text-gray-400">
          <p>No spectrogram data available</p>
          <p className="text-sm">Upload and classify an audio file to see spectrograms</p>
        </div>
      )}
    </div>
  );
  const FeaturesView = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">Audio Features</h3>
      
      {/* Explanation Section */}
      <div className="bg-purple-900/20 backdrop-blur-sm rounded-lg p-4 border-l-4 border-purple-400">
        <h4 className="text-md font-medium text-purple-300 mb-2">🎵 Understanding Audio Features</h4>
        <div className="text-sm text-gray-300 space-y-2">
          <p><strong>MFCC (Mel-Frequency Cepstral Coefficients):</strong> Represents the shape of the spectral envelope. Critical for genre classification as different genres have distinct spectral patterns. The first few coefficients capture the most important timbral information.</p>
          <p><strong>Spectral Centroid:</strong> The "center of mass" of the spectrum. Higher values indicate brighter sounds (e.g., cymbals), lower values indicate darker sounds (e.g., bass).</p>
          <p><strong>Spectral Rolloff:</strong> The frequency below which 85% of the spectrum's energy is contained. Helps distinguish between harmonic and percussive content.</p>
          <p><strong>Zero Crossing Rate:</strong> How often the audio signal crosses zero. High values suggest noisy or percussive content, low values suggest tonal content.</p>
          <p><strong>Tempo (BPM):</strong> Beats per minute, fundamental for genre classification as different genres have characteristic tempo ranges.</p>
        </div>
      </div>
      
      {visualizationData?.mfcc_features && (
        <div className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
          <h4 className="text-md font-medium text-white mb-3">MFCC Features</h4>
          <div className="text-xs text-gray-400 mb-3">
            Mel-Frequency Cepstral Coefficients • Timbral characteristics • Genre signature
          </div>
          <div className="bg-black/40 rounded-lg p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {visualizationData.mfcc_features.slice(0, 4).map((coeff, index) => (
                <div key={index} className="space-y-2">
                  <p className="text-sm text-gray-300">MFCC {index + 1} {index === 0 ? '(Energy)' : index === 1 ? '(Pitch)' : index === 2 ? '(Timbre)' : '(Detail)'}</p>
                  <div className="h-16 bg-gray-800 rounded relative overflow-hidden">
                    <div className="absolute inset-0 flex items-end space-x-1 p-1">
                      {coeff.slice(0, 50).map((value, i) => (
                        <div
                          key={i}
                          className="bg-gradient-to-t from-purple-600 to-pink-400 flex-1 rounded-sm"
                          style={{ height: `${Math.abs(value) * 100}%` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {visualizationData?.spectral_features && (
        <div className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
          <h4 className="text-md font-medium text-white mb-3">Spectral Features</h4>
          <div className="text-xs text-gray-400 mb-3">
            Frequency domain characteristics • Brightness and texture indicators
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-black/40 rounded-lg p-3">
              <p className="text-sm text-gray-300 mb-1">Spectral Centroid</p>
              <p className="text-xs text-gray-500 mb-2">Brightness measure</p>
              <p className="text-lg text-blue-400">
                {visualizationData.spectral_features.spectral_centroid?.[0]?.toFixed(2) || 'N/A'} Hz
              </p>
            </div>
            <div className="bg-black/40 rounded-lg p-3">
              <p className="text-sm text-gray-300 mb-1">Spectral Rolloff</p>
              <p className="text-xs text-gray-500 mb-2">Energy distribution</p>
              <p className="text-lg text-green-400">
                {visualizationData.spectral_features.spectral_rolloff?.[0]?.toFixed(2) || 'N/A'} Hz
              </p>
            </div>
            <div className="bg-black/40 rounded-lg p-3">
              <p className="text-sm text-gray-300 mb-1">Zero Crossing Rate</p>
              <p className="text-xs text-gray-500 mb-2">Noisiness indicator</p>
              <p className="text-lg text-purple-400">
                {visualizationData.spectral_features.zero_crossing_rate?.[0]?.toFixed(4) || 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}

      {visualizationData?.tempo && (
        <div className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
          <h4 className="text-md font-medium text-white mb-3">Tempo Analysis</h4>
          <div className="text-xs text-gray-400 mb-3">
            Rhythmic foundation • Genre characteristic • Dance-ability factor
          </div>
          <div className="bg-black/40 rounded-lg p-4 text-center">
            <p className="text-3xl text-yellow-400 font-bold">{visualizationData.tempo.toFixed(1)}</p>
            <p className="text-sm text-gray-300">BPM (Beats Per Minute)</p>
            <p className="text-xs text-gray-500 mt-1">
              {visualizationData.tempo < 80 ? 'Slow/Ballad' : 
               visualizationData.tempo < 120 ? 'Moderate' : 
               visualizationData.tempo < 140 ? 'Upbeat' : 'Fast/Dance'}
            </p>
          </div>
        </div>
      )}

      {!visualizationData?.mfcc_features && !visualizationData?.spectral_features && (
        <div className="text-center py-8 text-gray-400">
          <p>No feature data available</p>
          <p className="text-sm">Audio features will be displayed after classification</p>
        </div>
      )}
    </div>
  );
  const PredictionsView = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-white mb-4">Model Predictions</h3>
      
      {/* Explanation Section */}
      <div className="bg-green-900/20 backdrop-blur-sm rounded-lg p-4 border-l-4 border-green-400">
        <h4 className="text-md font-medium text-green-300 mb-2">🎯 Understanding Model Output</h4>
        <div className="text-sm text-gray-300 space-y-2">
          <p><strong>Genre Probabilities:</strong> The model outputs confidence scores for each genre class. Higher percentages indicate stronger confidence. The sum of all probabilities equals 100%.</p>
          <p><strong>Prediction Ranking:</strong> Genres are sorted by confidence level. The top prediction is the model's best guess, but consider the confidence distribution - close scores suggest uncertainty.</p>
          <p><strong>Raw Model Output:</strong> The unprocessed neural network outputs before applying softmax normalization. These values show the model's internal decision process.</p>
          <p><strong>Confidence Interpretation:</strong> &gt;70% = High confidence, 50-70% = Moderate confidence, &lt;50% = Low confidence or ambiguous genre.</p>
        </div>
      </div>
      
      {genrePredictions && (
        <div className="space-y-4">
          <div className="text-sm text-gray-400 mb-3">
            Genre confidence ranking • Softmax probabilities • Model decision
          </div>
          {Object.entries(genrePredictions)
            .sort(([,a], [,b]) => b - a)
            .map(([genre, probability], index) => (
              <div key={genre} className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-white">{genre}</span>
                    {index === 0 && (
                      <span className="px-2 py-1 text-xs bg-green-600/30 text-green-300 rounded-full">
                        Top Prediction
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <span className="text-sm text-gray-300">{(probability * 100).toFixed(1)}%</span>
                    <div className="text-xs text-gray-500">
                      {probability > 0.7 ? 'High' : probability > 0.5 ? 'Moderate' : 'Low'} confidence
                    </div>
                  </div>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      index === 0 
                        ? 'bg-gradient-to-r from-green-400 to-emerald-500' 
                        : 'bg-gradient-to-r from-blue-400 to-purple-500'
                    }`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
              </div>
            ))}
        </div>
      )}

      {visualizationData?.model_predictions && (
        <div className="bg-black/20 backdrop-blur-sm rounded-lg p-4">
          <h4 className="text-md font-medium text-white mb-3">Raw Model Output</h4>
          <div className="text-xs text-gray-400 mb-3">
            Pre-softmax logits • Neural network activations • Internal decision values
          </div>
          <div className="bg-black/40 rounded-lg p-4">
            <div className="grid grid-cols-4 gap-2">
              {visualizationData.model_predictions.map((value, index) => (
                <div key={index} className="text-center">
                  <div className="text-sm text-gray-400">Class #{index}</div>
                  <div className="text-lg text-blue-400">{value.toFixed(4)}</div>
                  <div className="text-xs text-gray-500">
                    {value > 0 ? 'Positive' : 'Negative'} activation
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!genrePredictions && !visualizationData?.model_predictions && (
        <div className="text-center py-8 text-gray-400">
          <p>No prediction data available</p>
          <p className="text-sm">Model predictions will be displayed after classification</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="w-full max-w-6xl mx-auto">
      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 mb-6 border-b border-gray-600/50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-t-lg transition-all ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-purple-600/50 to-pink-600/50 border-b-2 border-purple-400 text-white'
                : 'bg-black/20 text-gray-400 hover:text-white hover:bg-black/30'
            }`}
          >
            <span className="text-lg">{tab.icon}</span>
            <span className="font-medium">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'pipeline' && <ProcessingPipeline />}
        {activeTab === 'spectrogram' && <SpectrogramView />}
        {activeTab === 'features' && <FeaturesView />}
        {activeTab === 'predictions' && <PredictionsView />}
      </div>

      {/* Processing Indicator */}
      {isProcessing && (
        <div className="fixed top-4 right-4 bg-blue-600/90 backdrop-blur-sm text-white px-4 py-2 rounded-lg shadow-lg">
          <div className="flex items-center space-x-2">
            <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
            <span>Processing audio...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClassificationVisualization;
