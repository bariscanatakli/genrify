"use client";

import React from 'react';

type DependencyStatus = {
  checked: boolean;
  missing: string[];
  critical: string[];
  modelsLoaded: boolean;
  usingMock: boolean;
  missingLibraries?: Record<string, boolean>;
  functionality?: {
    ml_ready: boolean;
    search_ready: boolean;
    audio_ready: boolean;
    embeddings_available: boolean;
    models_available: boolean;
  };
};

export default function DependencyStatus({ status }: { status: DependencyStatus }) {
  if (!status.checked) {
    return (
      <div className="mt-2 text-blue-500 text-sm animate-pulse">
        Checking system dependencies...
      </div>
    );
  }

  // All good - everything is available
  if (status.missing.length === 0 && status.modelsLoaded) {
    return (
      <div className="mt-2 text-green-500 text-sm">
        ✓ All dependencies installed and models loaded
      </div>
    );
  }

  // Models loaded but some libraries might show as missing even though they work
  if (status.modelsLoaded) {
    // Check what functionality is available
    const functionality = status.functionality || {
      ml_ready: false,
      search_ready: false,
      audio_ready: false,
      embeddings_available: true,
      models_available: true
    };

    // If all functionality is available, libraries are actually working
    if (functionality.ml_ready && functionality.search_ready && functionality.audio_ready) {
      return (
        <div className="mt-2 text-green-500 text-sm">
          ✓ All models and functionality available
        </div>
      );
    }

    // Some functionality limited
    return (
      <div className="mt-2">
        <div className="text-green-500 text-sm flex items-start gap-1.5">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          <div className="flex flex-col">
            <span>Models loaded successfully</span>
            <div className="text-xs mt-1 text-gray-600 dark:text-gray-400 flex flex-col gap-0.5">
              {functionality.embeddings_available && (
                <span className="text-green-500">✓ Embeddings available</span>
              )}
              
              {functionality.ml_ready ? (
                <span className="text-green-500">✓ ML processing available</span>
              ) : (
                <span className="text-yellow-500">◐ Using pre-computed ML results</span>
              )}
              
              {functionality.audio_ready ? (
                <span className="text-green-500">✓ Audio processing available</span>
              ) : (
                <span className="text-yellow-500">◐ Limited audio processing</span>
              )}
              
              {status.missing.length > 0 && (
                <span className="text-gray-600 dark:text-gray-400 mt-0.5">
                  Install missing: <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-xs">pip install {status.missing.join(' ')}</code>
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Critical dependencies missing
  return (
    <div className="mt-2">
      <div className="text-amber-500 text-sm">
        ⚠️ Missing critical dependencies. Using mock data.
      </div>
      <div className="block text-xs text-gray-600 dark:text-gray-400 mt-1">
        Install with: <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-xs">pip install {status.missing.join(' ')}</code>
      </div>
    </div>
  );
}
