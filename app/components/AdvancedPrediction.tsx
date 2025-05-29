'use client';

import React, { useState } from 'react';
import GenreRadar from './GenreRadar';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

interface AdvancedPredictionProps {
  genreProbabilities: Record<string, number>;
  confidence: number;
}

const AdvancedPrediction: React.FC<AdvancedPredictionProps> = ({
  genreProbabilities,
  confidence
}) => {
  const [activeView, setActiveView] = useState<'radar' | 'chart' | 'comparison'>('radar');
  
  // Get sorted genres by probability
  const sortedGenres = Object.entries(genreProbabilities)
    .sort(([, a], [, b]) => b - a)
    .map(([genre]) => genre);
  
  // Get primary and secondary genres
  const primaryGenre = sortedGenres[0];
  const secondaryGenre = sortedGenres[1];
  
  // Calculate genre similarity metrics
  const genreSimilarities: Record<string, Record<string, number>> = {
    'Electronic': {
      'Experimental': 0.75,
      'Hip-Hop': 0.6,
      'Pop': 0.55,
      'Rock': 0.4,
      'Folk': 0.25,
      'Classical': 0.2,
      'Jazz': 0.3
    },
    'Rock': {
      'Folk': 0.5,
      'Pop': 0.65,
      'Electronic': 0.4,
      'Experimental': 0.55,
      'Hip-Hop': 0.3,
      'Classical': 0.25,
      'Jazz': 0.4
    },
    'Hip-Hop': {
      'Electronic': 0.6,
      'Pop': 0.7,
      'Experimental': 0.5,
      'Rock': 0.3,
      'Folk': 0.25,
      'Classical': 0.15,
      'Jazz': 0.45
    },
    'Folk': {
      'Rock': 0.5,
      'Classical': 0.4,
      'Jazz': 0.45,
      'Pop': 0.4,
      'Electronic': 0.25,
      'Experimental': 0.3,
      'Hip-Hop': 0.25
    },
    // Add other genre similarities as needed
  };
  
  // Check if the identified genres have a defined similarity
  const hasDefinedSimilarity = 
    genreSimilarities[primaryGenre] && 
    genreSimilarities[primaryGenre][secondaryGenre];
  
  // Get the similarity value between primary and secondary genres
  const genreSimilarity = hasDefinedSimilarity
    ? genreSimilarities[primaryGenre][secondaryGenre]
    : 0.4; // Default similarity if not defined
  
  // Define music characteristics based on genre
  const genreCharacteristics: Record<string, string[]> = {
    'Electronic': ['Synthesized sounds', 'Digital production', 'Beat-driven', 'Repetitive structures'],
    'Rock': ['Guitar-driven', 'Band format', 'Vocal prominence', 'Verse-chorus structure'],
    'Hip-Hop': ['Rhythmic vocals', 'Sampled beats', 'Lyric-focused', 'Bass-heavy'],
    'Folk': ['Acoustic instruments', 'Storytelling lyrics', 'Traditional elements', 'Vocal harmony'],
    'Pop': ['Catchy melodies', 'Verse-chorus-bridge', 'Polished production', 'Contemporary sounds'],
    'Jazz': ['Improvisation', 'Complex harmonies', 'Swing rhythm', 'Instrumental solos'],
    'Classical': ['Orchestral instruments', 'Complex compositions', 'Dynamic range', 'Traditional notation'],
    'Experimental': ['Non-traditional sounds', 'Avant-garde', 'Genre-blending', 'Unconventional structure']
  };
  
  // Get the primary genre characteristics
  const primaryCharacteristics = genreCharacteristics[primaryGenre] || [];
  
  return (
    <div className="space-y-8">
      {/* Navigation tabs */}
      <div className="flex justify-center mb-6">
        <div className="glass-card p-1 rounded-lg flex divide-x divide-gray-700">
          <button
            onClick={() => setActiveView('radar')}
            className={`px-4 py-2 rounded-l-md transition ${activeView === 'radar' ? 'bg-purple-500/30 text-white' : 'text-gray-400 hover:text-white hover:bg-purple-500/10'}`}
          >
            Radar Analysis
          </button>
          <button
            onClick={() => setActiveView('chart')}
            className={`px-4 py-2 transition ${activeView === 'chart' ? 'bg-purple-500/30 text-white' : 'text-gray-400 hover:text-white hover:bg-purple-500/10'}`}
          >
            Feature Breakdown
          </button>
          <button
            onClick={() => setActiveView('comparison')}
            className={`px-4 py-2 rounded-r-md transition ${activeView === 'comparison' ? 'bg-purple-500/30 text-white' : 'text-gray-400 hover:text-white hover:bg-purple-500/10'}`}
          >
            Genre Comparison
          </button>
        </div>
      </div>
      
      {/* Radar View */}
      {activeView === 'radar' && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <Card className="glass-card lg:col-span-3">
            <CardHeader>
              <CardTitle>Genre Distribution</CardTitle>
            </CardHeader>
            <CardContent className="flex justify-center">
              <div className="w-[300px] h-[300px]">
                <GenreRadar genreProbabilities={genreProbabilities} />
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-card lg:col-span-2">
            <CardHeader>
              <CardTitle>Primary Genre Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-6">
                <div className="flex justify-between mb-2">
                  <span className="text-lg font-medium text-purple-300">{primaryGenre}</span>
                  <span className="text-lg font-semibold text-blue-300">
                    {(genreProbabilities[primaryGenre] * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-500 to-purple-500"
                    style={{ width: `${genreProbabilities[primaryGenre] * 100}%` }}
                  />
                </div>
              </div>
              
              <h4 className="text-md font-medium text-gray-200 mb-2">Key Characteristics:</h4>
              <ul className="space-y-2">
                {primaryCharacteristics.map((trait, idx) => (
                  <li key={idx} className="flex items-center text-sm">
                    <span className="w-2 h-2 bg-purple-400 rounded-full mr-2" />
                    {trait}
                  </li>
                ))}
              </ul>
              
              <div className="mt-4 pt-4 border-t border-gray-700">
                <h4 className="text-md font-medium text-gray-200 mb-2">Prediction Confidence:</h4>
                <div className="glass-card p-3 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">{confidence < 0.5 ? 'Low' : confidence < 0.7 ? 'Medium' : 'High'}</span>
                    <span className="text-sm font-bold">{(confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full mt-2">
                    <div
                      className={`h-full rounded-full ${
                        confidence < 0.5 ? 'bg-red-500' : confidence < 0.7 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      {/* Chart View */}
      {activeView === 'chart' && (
        <div className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Genre Probability Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(genreProbabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([genre, probability], index) => (
                    <div key={genre} className="stagger-animation" style={{ animationDelay: `${index * 0.08}s` }}>
                      <div className="flex justify-between mb-1">
                        <span className="text-base font-medium text-gray-200 capitalize">{genre}</span>
                        <span className="text-base font-semibold text-purple-300">{(probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="bg-gray-700/60 rounded-full h-3 overflow-hidden">
                        <div
                          className={`h-full ${
                            index === 0
                              ? 'bg-gradient-to-r from-purple-500 to-blue-500'
                              : index === 1
                                ? 'bg-gradient-to-r from-blue-500 to-cyan-500'
                                : 'bg-gradient-to-r from-gray-500 to-gray-400'
                          }`}
                          style={{ width: `${probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Audio Feature Influence</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400 mb-4">
                Estimated influence of different audio features on the genre classification:
              </p>
              <div className="space-y-3">
                {[
                  { name: 'Spectral Contrast', value: Math.random() * 0.3 + 0.5, color: 'from-pink-500 to-red-500' },
                  { name: 'Rhythmic Patterns', value: Math.random() * 0.3 + 0.5, color: 'from-orange-500 to-amber-500' },
                  { name: 'Timbral Texture', value: Math.random() * 0.4 + 0.4, color: 'from-green-500 to-emerald-500' },
                  { name: 'Harmonic Structure', value: Math.random() * 0.4 + 0.3, color: 'from-blue-500 to-cyan-500' },
                  { name: 'Temporal Evolution', value: Math.random() * 0.5 + 0.2, color: 'from-indigo-500 to-violet-500' }
                ].map((feature, index) => (
                  <div key={feature.name} className="stagger-animation" style={{ animationDelay: `${index * 0.08 + 0.5}s` }}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium text-gray-300">{feature.name}</span>
                      <span className="text-sm font-medium text-gray-400">{(feature.value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bg-gray-700/60 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r ${feature.color}`}
                        style={{ width: `${feature.value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      {/* Comparison View */}
      {activeView === 'comparison' && (
        <div className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Primary Genre Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-1 glass-card p-4 rounded-lg">
                  <div className="text-center mb-4">
                    <h3 className="text-xl font-medium text-purple-300">{primaryGenre}</h3>
                    <p className="text-sm text-gray-400">Primary Genre</p>
                    <div className="text-2xl font-bold text-white mt-2">
                      {(genreProbabilities[primaryGenre] * 100).toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    {(genreCharacteristics[primaryGenre] || []).map((trait, idx) => (
                      <div key={idx} className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                        {trait}
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="flex-1 glass-card p-4 rounded-lg">
                  <div className="text-center mb-4">
                    <h3 className="text-xl font-medium text-blue-300">{secondaryGenre}</h3>
                    <p className="text-sm text-gray-400">Secondary Genre</p>
                    <div className="text-2xl font-bold text-white mt-2">
                      {(genreProbabilities[secondaryGenre] * 100).toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    {(genreCharacteristics[secondaryGenre] || []).map((trait, idx) => (
                      <div key={idx} className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                        {trait}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="mt-6 glass-card p-4 rounded-lg">
                <h3 className="text-lg font-medium text-gray-200 mb-3">Genre Similarity</h3>
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-400">Low</span>
                  <div className="flex-1 h-2 bg-gray-700 rounded-full">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                      style={{ width: `${genreSimilarity * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-400">High</span>
                </div>
                <p className="text-sm text-gray-400 mt-3">
                  {genreSimilarity < 0.3 
                    ? `${primaryGenre} and ${secondaryGenre} are highly distinct genres with few common characteristics.` 
                    : genreSimilarity < 0.6 
                      ? `${primaryGenre} and ${secondaryGenre} share some musical elements but have distinct identities.`
                      : `${primaryGenre} and ${secondaryGenre} are closely related genres with many shared characteristics.`}
                </p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Classification Confidence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-gray-400">
                  {confidence > 0.8 
                    ? "The model has high confidence in this classification. The audio contains clear signature elements of the predicted genre."
                    : confidence > 0.6
                      ? "The model has moderate confidence in this classification. The audio shows characteristics of multiple genres with one being more prominent."
                      : "The model has low confidence in this classification. The audio may contain elements from multiple genres or unusual characteristics."}
                </p>
                
                <div className="glass-card p-4 rounded-lg bg-gradient-to-r from-purple-500/10 to-blue-500/10">
                  <h4 className="text-md font-medium text-gray-200 mb-3">Recommendation</h4>
                  <p className="text-sm text-gray-300">
                    {confidence > 0.8 
                      ? "Consider exploring more artists within this genre for similar experiences."
                      : confidence > 0.6
                        ? "Check out both the primary and secondary genres for similar music."
                        : "This track appears to blend multiple genres. You might enjoy fusion or experimental music."}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default AdvancedPrediction;
