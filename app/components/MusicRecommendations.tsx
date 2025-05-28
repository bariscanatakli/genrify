'use client';

import React, { useState } from 'react';
import { Card, CardContent } from '../components/ui/card';

interface MusicRecommendation {
  id: string;
  title: string;
  genre: string;
  similarity: number;
}

interface RecommendationResponse {
  recommendations: MusicRecommendation[];
  processing_time: number;
  query_file: string;
}

interface MusicRecommendationsProps {
  audioFile: File | null;
  isVisible: boolean;
}

export default function MusicRecommendations({ audioFile, isVisible }: MusicRecommendationsProps) {
  const [recommendations, setRecommendations] = useState<MusicRecommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const getRecommendations = async () => {
    if (!audioFile) {
      setError('No audio file selected');
      return;
    }

    setIsLoading(true);
    setError(null);
    setRecommendations([]);

    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      formData.append('top_k', '8'); // Get 8 recommendations

      const response = await fetch('http://localhost:8888/recommend', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: RecommendationResponse = await response.json();
      setRecommendations(data.recommendations);
      setProcessingTime(data.processing_time);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get recommendations');
      console.error('Recommendation error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isVisible) return null;

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-400';
    if (similarity >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getSimilarityBadge = (similarity: number) => {
    if (similarity >= 0.8) return 'High';
    if (similarity >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">🎵 Music Recommendations</h2>
        <button
          onClick={getRecommendations}
          disabled={!audioFile || isLoading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Analyzing...</span>
            </div>
          ) : (
            'Get Similar Music'
          )}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {processingTime && (
        <div className="text-sm text-gray-400">
          Analysis completed in {processingTime}s using AI-powered audio similarity
        </div>
      )}

      {recommendations.length > 0 && (
        <div className="space-y-4">
          <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
            <h4 className="text-md font-medium text-blue-300 mb-2">🔍 How Recommendations Work</h4>
            <div className="text-sm text-gray-300 space-y-1">
              <p><strong>Audio Embeddings:</strong> Your music is converted to a 128-dimensional feature vector using triplet neural networks</p>
              <p><strong>Similarity Search:</strong> FAISS (Facebook AI Similarity Search) finds the closest matches in embedding space</p>
              <p><strong>Cosine Similarity:</strong> Scores range from 0-1, where 1.0 means identical audio characteristics</p>
            </div>
          </div>

          <div className="grid gap-4">
            {recommendations.map((rec, index) => (
              <Card key={rec.id} className="bg-black/20 backdrop-blur-sm border-gray-700/50">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <div className="text-2xl">
                          {index === 0 ? '🏆' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🎵'}
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{rec.title}</h3>
                          <p className="text-sm text-gray-400">Genre: {rec.genre}</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getSimilarityColor(rec.similarity)}`}>
                        {(rec.similarity * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {getSimilarityBadge(rec.similarity)} Match
                      </div>
                    </div>
                  </div>

                  {/* Similarity bar */}
                  <div className="mt-3">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>Audio Similarity</span>
                      <span>{rec.similarity.toFixed(3)}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          rec.similarity >= 0.8 ? 'bg-green-500' :
                          rec.similarity >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${rec.similarity * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="bg-gray-900/50 rounded-lg p-4 text-sm text-gray-400">
            <p><strong>Note:</strong> Recommendations are based on audio content similarity, not metadata matching. 
            High similarity scores indicate songs with similar acoustic properties, rhythm patterns, and timbral characteristics.</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Extracting audio embeddings and searching similarity space...</p>
          <p className="text-sm text-gray-500 mt-2">This may take 10-15 seconds</p>
        </div>
      )}

      {!isLoading && !error && recommendations.length === 0 && audioFile && (
        <div className="text-center py-8 text-gray-400">
          <div className="text-4xl mb-4">🎼</div>
          <p>Upload an audio file and click "Get Similar Music" to discover recommendations</p>
          <p className="text-sm text-gray-500 mt-2">
            Uses triplet neural networks trained on 16,000+ music tracks
          </p>
        </div>
      )}
    </div>
  );
}
