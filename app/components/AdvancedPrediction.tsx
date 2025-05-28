'use client';

import React, { useState } from 'react';

interface PredictionQuality {
  confidence_level: 'high' | 'medium' | 'low';
  entropy: number;
  top_2_difference: number;
  prediction_certainty: string;
  recommendation: string;
}

interface AdvancedPredictionProps {
  genreProbabilities: Record<string, number>;
  confidence: number;
}

export default function AdvancedPrediction({ genreProbabilities, confidence }: AdvancedPredictionProps) {
  const [showDetails, setShowDetails] = useState(false);

  const analyzePredictionQuality = (): PredictionQuality => {
    const probs = Object.values(genreProbabilities).sort((a, b) => b - a);
    
    // Calculate entropy (uncertainty measure)
    const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
    
    // Difference between top 2 predictions
    const top2Diff = probs[0] - probs[1];
    
    // Determine confidence level
    let confidenceLevel: 'high' | 'medium' | 'low';
    let certainty: string;
    let recommendation: string;
    
    if (confidence > 0.7 && top2Diff > 0.3) {
      confidenceLevel = 'high';
      certainty = 'Very confident prediction with clear genre distinction';
      recommendation = 'Trust this result - the model is highly confident';
    } else if (confidence > 0.5 && top2Diff > 0.15) {
      confidenceLevel = 'medium';
      certainty = 'Moderate confidence with reasonable genre separation';
      recommendation = 'Good prediction, but consider alternative genres';
    } else {
      confidenceLevel = 'low';
      certainty = 'Low confidence - track may blend multiple genres';
      recommendation = 'Uncertain prediction - track might be experimental/fusion';
    }
    
    return {
      confidence_level: confidenceLevel,
      entropy,
      top_2_difference: top2Diff,
      prediction_certainty: certainty,
      recommendation
    };
  };

  const quality = analyzePredictionQuality();
  
  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-green-400 bg-green-900/20 border-green-500/30';
      case 'medium': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30';
      case 'low': return 'text-red-400 bg-red-900/20 border-red-500/30';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500/30';
    }
  };

  const sortedGenres = Object.entries(genreProbabilities)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 3); // Top 3 genres

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
          🔬 Advanced AI Analysis
        </h2>
        <p className="text-gray-400">Deep learning insights and prediction quality assessment</p>
      </div>

      {/* Enhanced Main Analysis Card */}
      <div className="prism-glass dynamic-card neural-network rounded-3xl p-8 border-2 border-cyan-400/20">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold gradient-text-advanced flex items-center">
            <div className="plasma-orb w-8 h-8 mr-3"></div>
            Prediction Quality Analysis
          </h3>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="morph-button px-4 py-2 text-white rounded-xl transition-all duration-300 transform hover:scale-105"
          >
            {showDetails ? '📊 Hide Details' : '🔍 Show Details'}
          </button>
        </div>
        
        {/* Enhanced Confidence Assessment */}
        <div className={`rounded-2xl border-2 p-6 mb-6 ${getConfidenceColor(quality.confidence_level)}`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <span className="text-3xl">
                {quality.confidence_level === 'high' ? '🎯' : 
                 quality.confidence_level === 'medium' ? '⚠️' : '❓'}
              </span>
              <div>
                <div className="text-xl font-bold">
                  Confidence: {quality.confidence_level.toUpperCase()}
                </div>
                <div className="text-lg opacity-80">{(confidence * 100).toFixed(1)}% Accuracy</div>
              </div>
            </div>
          </div>
          <div className="glass-card rounded-xl p-4 mb-4">
            <p className="text-lg font-medium mb-2">{quality.prediction_certainty}</p>
            <p className="text-sm opacity-90">💡 {quality.recommendation}</p>
          </div>
        </div>

        {/* Enhanced Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="glass-card rounded-2xl p-6 text-center">
            <div className="text-4xl mb-3">📊</div>
            <div className="text-sm text-gray-300 mb-2">Prediction Entropy</div>
            <div className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-2">
              {quality.entropy.toFixed(3)}
            </div>
            <div className="text-sm text-gray-400">
              {quality.entropy < 1.5 ? '🟢 Low uncertainty' : 
               quality.entropy < 2.5 ? '🟡 Moderate uncertainty' : '🔴 High uncertainty'}
            </div>
          </div>
          
          <div className="glass-card rounded-2xl p-6 text-center">
            <div className="text-4xl mb-3">⚔️</div>
            <div className="text-sm text-gray-300 mb-2">Top-2 Separation</div>
            <div className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-400 mb-2">
              {(quality.top_2_difference * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">
              {quality.top_2_difference > 0.3 ? '🎯 Clear winner' : 
               quality.top_2_difference > 0.15 ? '⚖️ Close competition' : '🤏 Very close call'}
            </div>
          </div>
        </div>

        {/* Enhanced Genre Competition */}
        <div className="glass-card rounded-2xl p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-2 border-blue-400/20">
          <h4 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-6 flex items-center">
            <span className="text-3xl mr-3">🏆</span>
            Genre Competition Analysis
          </h4>
          <div className="space-y-4">
            {sortedGenres.map(([genre, prob], index) => (
              <div key={genre} className="glass-card rounded-xl p-4 hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <span className="text-3xl">
                      {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                    </span>
                    <div>
                      <div className="text-xl font-bold text-white">{genre}</div>
                      <div className="text-sm text-gray-400">
                        {index === 0 ? 'Winner' : index === 1 ? 'Runner-up' : 'Third place'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-400">
                      {(prob * 100).toFixed(1)}%
                    </div>
                    <div className="w-32 bg-gray-700 rounded-full h-3 mt-2">
                      <div
                        className="progress-bar h-full transition-all duration-1000 ease-out"
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Enhanced Additional Details */}
        {showDetails && (
          <div className="mt-8 space-y-6">
            <div className="glass-card rounded-2xl p-6 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border-2 border-purple-400/20">
              <h4 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-4 flex items-center">
                <span className="text-3xl mr-3">🧠</span>
                Deep Learning Insights
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h5 className="text-lg font-semibold text-white">Model Behavior</h5>
                  <ul className="space-y-2 text-gray-300">
                    <li className="flex items-center space-x-2">
                      <span>🎯</span>
                      <span>Confidence threshold: {confidence > 0.8 ? 'Exceeded' : 'Within range'}</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <span>📊</span>
                      <span>Distribution: {quality.entropy < 2 ? 'Focused' : 'Scattered'}</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <span>⚡</span>
                      <span>Decision clarity: {quality.top_2_difference > 0.2 ? 'High' : 'Low'}</span>
                    </li>
                  </ul>
                </div>
                <div className="space-y-3">
                  <h5 className="text-lg font-semibold text-white">Recommendation</h5>
                  <div className="glass-card rounded-xl p-4">
                    <p className="text-gray-300">{quality.recommendation}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
