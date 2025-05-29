'use client';

import React, { useCallback } from 'react';
import { UploadCloud, FileText, CheckCircle, XCircle, Loader2, Music, Trash2 } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

// Redux imports
import { useAppDispatch, useAppSelector } from '../redux/hooks';
import { 
  setBatchFiles, 
  addBatchFiles, 
  removeBatchFile,
  setBatchResults,
  setBatchSummary,
  setBatchProgress,
  clearBatchState,
  BatchPredictionItem
} from '../redux/slices/batchSlice';
import { setError } from '../redux/slices/uiSlice';

interface BatchProcessingProps {
  apiUrl?: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888';

export default function BatchProcessing({ apiUrl = API_URL }: BatchProcessingProps) {
  // Redux state
  const dispatch = useAppDispatch();
  const { files, results, summary, progress } = useAppSelector((state: any) => state.batch);
  const { error, loading } = useAppSelector(state => state.ui);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    dispatch(addBatchFiles(acceptedFiles.filter(file => file.type.startsWith('audio/'))));
    dispatch(setError(null));
    dispatch(clearBatchState());
  }, [dispatch]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: { 'audio/*': [] },
    multiple: true 
  });

  const removeFile = (fileName: string) => {
    dispatch(removeBatchFile(fileName));
  };

  const handleBatchPredict = async () => {
    if (files.length === 0) {
      dispatch(setError('Please select audio files first.'));
      return;
    }

    dispatch(setBatchProgress(0));
    dispatch(setError(null));
    dispatch(setBatchResults([]));
    dispatch(setBatchSummary(null));

    const formData = new FormData();
    files.forEach((file: File) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${apiUrl}/predict/batch`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Batch prediction failed');
      }

      const data = await response.json();
      dispatch(setBatchResults(data.predictions || []));
      dispatch(setBatchSummary(data));
      
      // Simulate progress for demo purposes if not provided by API
      let currentProgress = 0;
      const interval = setInterval(() => {
        currentProgress += 10;
        if (currentProgress <= 100) {
          dispatch(setBatchProgress(currentProgress));
        } else {
          clearInterval(interval);
        }
      }, 200);
      
      dispatch(setBatchProgress(100));

    } catch (err) {
      dispatch(setError(err instanceof Error ? err.message : 'An unknown error occurred'));
      dispatch(setBatchResults([]));
      dispatch(setBatchSummary(null));
    }
  };

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
          📦 Batch Processing
        </h2>
        <p className="text-gray-400">Process multiple audio files simultaneously for efficient genre classification</p>
      </div>

      {/* Enhanced File Upload Area */}
      <div className="prism-glass dynamic-card neural-network rounded-3xl p-8 border-2 border-purple-400/20">
        <div 
          {...getRootProps()} 
          className={`p-10 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 
            ${isDragActive ? 'border-purple-400 bg-purple-500/10' : 'border-gray-600 hover:border-purple-500'}
            text-center magic-hover ripple`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center space-y-4">
            <UploadCloud className="w-16 h-16 text-purple-400 animate-pulse" />
            {isDragActive ? (
              <p className="text-xl font-semibold text-purple-300">Drop files here...</p>
            ) : (
              <p className="text-xl font-semibold text-gray-300">Drag & drop audio files, or click to select</p>
            )}
            <p className="text-sm text-gray-500">Supports MP3, WAV, FLAC, M4A</p>
          </div>
        </div>

        {files.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-200 mb-3">Selected Files ({files.length}):</h3>
            <div className="max-h-60 overflow-y-auto space-y-2 pr-2 scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-gray-700/50">
              {files.map((file: File) => (
                <div key={file.name} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg hover:bg-gray-600/50 transition-all duration-200">
                  <div className="flex items-center space-x-3">
                    <Music className="w-5 h-5 text-purple-400" />
                    <span className="text-sm text-gray-300 truncate max-w-xs">{file.name}</span>
                  </div>
                  <button 
                    onClick={() => removeFile(file.name)} 
                    className="text-red-400 hover:text-red-300 transition-colors duration-200 p-1 rounded-full hover:bg-red-500/20"
                    aria-label="Remove file"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mt-8 text-center">
          <button
            onClick={handleBatchPredict}
            disabled={loading || files.length === 0}
            className="morph-button px-8 py-4 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden ripple flex items-center justify-center mx-auto"
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 mr-3 animate-spin" />
                Processing Batch...
              </>
            ) : (
              '🚀 Process Batch'
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-900/30 border border-red-500/50 rounded-lg text-red-300">
          <div className="flex items-center">
            <XCircle className="w-5 h-5 mr-2" />
            <span className="font-semibold">Error:</span>
          </div>
          <p className="ml-7">{error}</p>
        </div>
      )}

      {progress > 0 && progress < 100 && (
        <div className="mt-6">
          <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-500 ease-linear rounded-full"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="text-center text-sm text-gray-400 mt-2">Processing... {progress}%</p>
        </div>
      )}

      {summary && (
        <div className="prism-glass dynamic-card rounded-3xl p-8 mt-8 border-2 border-green-400/20">
          <h3 className="text-2xl font-bold gradient-text-advanced mb-6 text-center">Batch Processing Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-center">
            <div className="glass-card p-6 rounded-xl">
              <FileText className="w-10 h-10 mx-auto mb-3 text-blue-400" />
              <p className="text-3xl font-bold text-blue-300">{summary.total_files}</p>
              <p className="text-sm text-gray-400">Total Files</p>
            </div>
            <div className="glass-card p-6 rounded-xl">
              <CheckCircle className="w-10 h-10 mx-auto mb-3 text-green-400" />
              <p className="text-3xl font-bold text-green-300">{summary.successful_predictions}</p>
              <p className="text-sm text-gray-400">Successful</p>
            </div>
            <div className="glass-card p-6 rounded-xl">
              <XCircle className="w-10 h-10 mx-auto mb-3 text-red-400" />
              <p className="text-3xl font-bold text-red-300">{summary.failed_predictions}</p>
              <p className="text-sm text-gray-400">Failed</p>
            </div>
            <div className="glass-card p-6 rounded-xl">
              <Loader2 className="w-10 h-10 mx-auto mb-3 text-purple-400 animate-spin" />
              <p className="text-3xl font-bold text-purple-300">{summary.total_processing_time.toFixed(2)}s</p>
              <p className="text-sm text-gray-400">Total Time</p>
            </div>
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="mt-8 prism-glass dynamic-card rounded-3xl p-6 border-2 border-cyan-400/20">
          <h3 className="text-2xl font-bold gradient-text-advanced mb-6 text-center">Prediction Results</h3>
          <div className="overflow-x-auto scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-gray-700/50">
            <table className="w-full min-w-[600px] text-left">
              <thead className="border-b border-gray-700">
                <tr>
                  <th className="p-4 text-sm font-semibold text-gray-300">File Name</th>
                  <th className="p-4 text-sm font-semibold text-gray-300">Predicted Genre</th>
                  <th className="p-4 text-sm font-semibold text-gray-300">Confidence</th>
                  <th className="p-4 text-sm font-semibold text-gray-300">Time (s)</th>
                  <th className="p-4 text-sm font-semibold text-gray-300">Status</th>
                </tr>
              </thead>
              <tbody>
                {results.map((item: BatchPredictionItem, index: number) => (
                  <tr key={index} className="border-b border-gray-800 hover:bg-gray-700/30 transition-colors duration-200">
                    <td className="p-4 text-sm text-gray-400 truncate max-w-xs">{item.filename}</td>
                    <td className="p-4 text-sm font-semibold text-purple-300">{item.predicted_genre || 'N/A'}</td>
                    <td className="p-4 text-sm">
                      <div className="flex items-center space-x-2">
                        <span className={`${item.confidence > 0.7 ? 'text-green-400' : item.confidence > 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                        <div className="w-16 h-2 bg-gray-600 rounded-full">
                          <div 
                            className={`h-full rounded-full ${item.confidence > 0.7 ? 'bg-green-500' : item.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`}
                            style={{ width: `${item.confidence * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="p-4 text-sm text-gray-400">{item.processing_time.toFixed(2)}s</td>
                    <td className="p-4">
                      {item.error ? (
                        <span className="text-xs font-medium bg-red-500/20 text-red-300 rounded-full px-2 py-1 inline-flex items-center">
                          <XCircle className="w-3 h-3 mr-1" /> Error
                        </span>
                      ) : (
                        <span className="text-xs font-medium bg-green-500/20 text-green-300 rounded-full px-2 py-1 inline-flex items-center">
                          <CheckCircle className="w-3 h-3 mr-1" /> Success
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
