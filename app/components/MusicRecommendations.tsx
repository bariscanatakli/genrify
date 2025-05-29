'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Music, Loader2, ThumbsUp, RefreshCcw, AlertCircle, HeadphonesIcon } from 'lucide-react';

interface MusicRecommendation {
  id: string;
  title: string;
  genre: string;
  similarity: number;
}

interface MusicRecommendationsProps {
  audioFile: File | null;
  apiUrl?: string;
  isVisible?: boolean;
  maxRecommendations?: number;
}

const MusicRecommendations: React.FC<MusicRecommendationsProps> = ({
  audioFile,
  apiUrl = 'http://localhost:8888',
  isVisible = true,
  maxRecommendations = 5
}) => {
  const [recommendations, setRecommendations] = useState<MusicRecommendation[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);

  useEffect(() => {
    if (audioFile && isVisible) {
      getRecommendations();
    }
  }, [audioFile, isVisible]);

  const getRecommendations = async () => {
    if (!audioFile) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', audioFile);
      formData.append('top_k', maxRecommendations.toString());
      
      const response = await fetch(`${apiUrl}/recommend`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get recommendations');
      }
      
      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setProcessingTime(data.processing_time || 0);
      
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (!isVisible) return null;

  return (
    <div className="space-y-8">
      <div className="text-center mb-2">
        <p className="text-gray-400">
          Based on your uploaded track, you might like these similar songs:
        </p>
      </div>
      
      {loading ? (
        <Card className="glass-card p-8 text-center">
          <div className="flex flex-col items-center space-y-4">
            <Loader2 size={48} className="text-purple-400 animate-spin" />
            <div>
              <h3 className="text-xl font-medium mb-2">Finding Similar Music...</h3>
              <p className="text-gray-400">Analyzing audio patterns and comparing with our music database</p>
            </div>
          </div>
        </Card>
      ) : error ? (
        <Card className="glass-card p-6 border-red-500/30">
          <div className="flex items-center space-x-4">
            <AlertCircle size={32} className="text-red-400" />
            <div>
              <h3 className="text-lg font-medium mb-1">Error Finding Recommendations</h3>
              <p className="text-red-300">{error}</p>
            </div>
          </div>
          <div className="flex justify-center mt-4">
            <button
              onClick={getRecommendations}
              className="morph-button px-4 py-2 flex items-center space-x-2"
            >
              <RefreshCcw size={16} />
              <span>Try Again</span>
            </button>
          </div>
        </Card>
      ) : recommendations.length > 0 ? (
        <div className="space-y-4">
          <div className="glass-card p-4 rounded-lg mb-2">
            <p className="text-center text-sm text-gray-400">
              Recommendations processed in {processingTime.toFixed(2)}s
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recommendations.map((recommendation, index) => (
              <Card 
                key={recommendation.id}
                className={`dynamic-card stagger-animation hover:bg-purple-900/10 transition-all duration-300`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <CardContent className="p-4">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500/30 to-blue-500/30 flex items-center justify-center">
                      <HeadphonesIcon size={24} className="text-purple-300" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-medium line-clamp-1">{recommendation.title}</h3>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-purple-300">{recommendation.genre}</span>
                        <div className="flex items-center">
                          <ThumbsUp size={14} className="mr-1 text-blue-400" />
                          <span className="text-blue-300">
                            {(recommendation.similarity * 100).toFixed(1)}% match
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      ) : !audioFile ? (
        <Card className="glass-card p-8 text-center">
          <div className="flex flex-col items-center space-y-4">
            <Music size={48} className="text-gray-500" />
            <div>
              <h3 className="text-xl font-medium mb-2">Upload Music First</h3>
              <p className="text-gray-400">Upload an audio file to get music recommendations</p>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="glass-card p-8 text-center">
          <div className="flex flex-col items-center space-y-4">
            <Music size={48} className="text-gray-500" />
            <div>
              <h3 className="text-xl font-medium mb-2">No Recommendations Found</h3>
              <p className="text-gray-400">We couldn't find similar music for this track</p>
            </div>
          </div>
          <div className="flex justify-center mt-4">
            <button
              onClick={getRecommendations}
              className="morph-button px-4 py-2 flex items-center space-x-2"
            >
              <RefreshCcw size={16} />
              <span>Try Again</span>
            </button>
          </div>
        </Card>
      )}
    </div>
  );
};

export default MusicRecommendations;
