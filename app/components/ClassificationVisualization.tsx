"use client";

import React, { useState } from 'react';

interface VisualizationData {
  spectrogram?: string | null; // Allow null values
  mel_spectrogram?: string | null;
  mfcc_features?: number[][];
  chroma_features?: number[][];
  spectral_features?: {
    spectral_centroid: number[];
    spectral_rolloff: number[];
    zero_crossing_rate: number[];
  };
  tempo?: number;
  raw_features?: number[];
}

interface ClassificationVisualizationProps {
  visualizationData: VisualizationData | null;
  isVisible?: boolean;
}

const ClassificationVisualization: React.FC<ClassificationVisualizationProps> = ({
  visualizationData,
  isVisible = true
}) => {
  const [activeTab, setActiveTab] = useState<'spectrogram' | 'features'>('spectrogram');

  if (!isVisible || !visualizationData) {
    return (
      <div className="text-center p-6">
        <p className="text-gray-400">No visualization data available</p>
      </div>
    );
  }

  // Check if we have any visualization data
  const hasSpectrograms = visualizationData.spectrogram || visualizationData.mel_spectrogram;
  const hasFeatures = visualizationData.spectral_features || visualizationData.mfcc_features || visualizationData.tempo;

  if (!hasSpectrograms && !hasFeatures) {
    return (
      <div className="text-center p-6">
        <div className="bg-yellow-500/20 border border-yellow-500/40 rounded-lg p-4">
          <p className="text-yellow-200">Visualization generation failed. This may be due to a compatibility issue with the audio file.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-xl font-bold mb-4 text-center text-white">Audio Analysis Visualization</h3>
      
      {/* Tab Navigation */}
      <div className="flex justify-center mb-6">
        <div className="prism-glass rounded-lg p-1 flex">
          <button
            onClick={() => setActiveTab('spectrogram')}
            className={`px-4 py-2 rounded-md ${
              activeTab === 'spectrogram' 
                ? 'bg-purple-500/30 text-white' 
                : 'text-gray-400 hover:text-gray-200'
            }`}
            disabled={!hasSpectrograms}
          >
            Spectrograms
          </button>
          <button
            onClick={() => setActiveTab('features')}
            className={`px-4 py-2 rounded-md ${
              activeTab === 'features' 
                ? 'bg-purple-500/30 text-white' 
                : 'text-gray-400 hover:text-gray-200'
            }`}
            disabled={!hasFeatures}
          >
            Audio Features
          </button>
        </div>
      </div>
      
      {/* Spectrograms Tab */}
      {activeTab === 'spectrogram' && (
        <div className="space-y-6">
          {visualizationData.spectrogram ? (
            <div className="space-y-2">
              <h4 className="text-lg font-medium text-gray-200">Linear Spectrogram</h4>
              <div className="bg-black/30 rounded-lg p-2">
                <img 
                  src={`data:image/png;base64,${visualizationData.spectrogram}`} 
                  alt="Spectrogram" 
                  className="w-full rounded" 
                />
              </div>
            </div>
          ) : (
            <div className="text-center my-4 p-4 bg-gray-800/50 rounded-lg">
              <p className="text-gray-400">Linear spectrogram not available</p>
            </div>
          )}
          
          {visualizationData.mel_spectrogram ? (
            <div className="space-y-2">
              <h4 className="text-lg font-medium text-gray-200">Mel Spectrogram</h4>
              <div className="bg-black/30 rounded-lg p-2">
                <img 
                  src={`data:image/png;base64,${visualizationData.mel_spectrogram}`} 
                  alt="Mel Spectrogram" 
                  className="w-full rounded" 
                />
              </div>
            </div>
          ) : (
            <div className="text-center my-4 p-4 bg-gray-800/50 rounded-lg">
              <p className="text-gray-400">Mel spectrogram not available</p>
            </div>
          )}
        </div>
      )}
      
      {/* Features Tab */}
      {activeTab === 'features' && (
        <div className="space-y-6">
          {/* Tempo */}
          {visualizationData.tempo && (
            <div className="prism-glass rounded-lg p-4">
              <h4 className="text-lg font-medium text-gray-200 mb-2">Tempo</h4>
              <div className="text-center">
                <span className="text-4xl font-bold text-purple-300">{visualizationData.tempo.toFixed(1)}</span>
                <span className="text-lg text-gray-400 ml-2">BPM</span>
              </div>
            </div>
          )}
          
          {/* Spectral Features */}
          {visualizationData.spectral_features && (
            <div className="prism-glass rounded-lg p-4">
              <h4 className="text-lg font-medium text-gray-200 mb-4">Spectral Features</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-black/30 p-3 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Spectral Centroid</div>
                  <div className="text-xl font-semibold text-blue-300">
                    {visualizationData.spectral_features.spectral_centroid[0]?.toFixed(2) || 'N/A'}
                  </div>
                </div>
                <div className="bg-black/30 p-3 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Spectral Rolloff</div>
                  <div className="text-xl font-semibold text-green-300">
                    {visualizationData.spectral_features.spectral_rolloff[0]?.toFixed(2) || 'N/A'}
                  </div>
                </div>
                <div className="bg-black/30 p-3 rounded-lg">
                  <div className="text-sm text-gray-400 mb-1">Zero Crossing Rate</div>
                  <div className="text-xl font-semibold text-yellow-300">
                    {visualizationData.spectral_features.zero_crossing_rate[0]?.toFixed(4) || 'N/A'}
                  </div>
                </div>
              </div>
              
              {/* Feature Visualization with Overflow Prevention */}
              {visualizationData.spectral_features.spectral_centroid && (
                <div className="mt-4 bg-black/30 p-3 rounded-lg overflow-hidden">
                  <div className="text-sm text-gray-400 mb-2">Feature Waveform</div>
                  <div className="h-24 relative">
                    <div className="absolute inset-0 flex items-center">
                      {visualizationData.spectral_features.spectral_centroid.slice(0, 100).map((value, i) => {
                        // Normalize value to fit within visualization height
                        const normalizedHeight = Math.min(Math.abs((value / 5000) * 100), 90);
                        return (
                          <div 
                            key={i} 
                            className="flex-1"
                            style={{ position: 'relative', height: '100%' }}
                          >
                            <div 
                              className="absolute bottom-1/2 w-full bg-purple-500 rounded-sm"
                              style={{ 
                                height: `${normalizedHeight}%`,
                                transform: 'translateY(50%)',
                                maxHeight: '95%' 
                              }}
                            ></div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* MFCC Features */}
          {visualizationData.mfcc_features && visualizationData.mfcc_features.length > 0 && (
            <div className="prism-glass rounded-lg p-4">
              <h4 className="text-lg font-medium text-gray-200 mb-2">MFCC Features</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {visualizationData.mfcc_features.slice(0, 4).map((mfcc, i) => (
                  <div key={i} className="bg-black/30 p-2 rounded-lg">
                    <div className="text-sm text-gray-400 mb-1">MFCC #{i+1}</div>
                    <div className="h-20 relative overflow-hidden">
                      {/* Added overflow-hidden to prevent bars from overflowing */}
                      <div className="absolute inset-0 flex items-end w-full">
                        {mfcc.slice(0, 50).map((value, j) => (
                          <div 
                            key={j} 
                            className="flex-1 mx-0.5 rounded-t-sm bg-gradient-to-t from-purple-500 to-pink-400"
                            style={{ 
                              height: `${Math.min(Math.abs(value) * 100, 100)}%`, // Limit maximum height
                              maxHeight: '100%' // Ensure height never exceeds container
                            }}
                          ></div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ClassificationVisualization;
