'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Music, Upload, FilePlus2, XCircle } from 'lucide-react';

interface AudioDropzoneProps {
  onFileChange: (file: File) => void;
  selectedFile: File | null;
  accept?: Record<string, string[]>;
  maxSize?: number;
  className?: string;
}

const AudioDropzone: React.FC<AudioDropzoneProps> = ({
  onFileChange,
  selectedFile,
  accept = {
    'audio/*': ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']
  },
  maxSize = 50 * 1024 * 1024, // 50MB
  className = '',
}) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileChange(acceptedFiles[0]);
      }
    },
    [onFileChange]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
  });

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileChange(null as unknown as File);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };

  return (
    <div
      {...getRootProps()}
      className={`clickable group block w-full p-8 border-2 border-dashed rounded-xl bg-gradient-to-br from-purple-500/5 to-pink-500/5 hover:from-purple-500/15 hover:to-pink-500/15 hover:border-purple-400/70 transition-all duration-300 ease-in-out cursor-pointer text-center magic-hover ripple transform hover:scale-[1.02] ${
        isDragActive ? 'border-purple-400 bg-purple-500/10' : 'border-purple-400/50'
      } ${isDragReject ? 'border-red-500 bg-red-500/10' : ''} ${className}`}
    >
      <input {...getInputProps()} />
      <div className="space-y-4 transition-transform duration-300 ease-in-out group-hover:scale-105">
        {selectedFile ? (
          <div className="space-y-4">
            <div className="music-visualizer mx-auto w-fit">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="vis-bar"></div>
              ))}
            </div>
            <div className="flex items-center justify-center">
              <Music size={24} className="text-purple-400 mr-2" />
              <span className="text-xl font-medium text-purple-300">{selectedFile.name}</span>
              <button
                onClick={clearFile}
                className="ml-3 text-gray-400 hover:text-red-400 transition-colors"
                aria-label="Remove file"
              >
                <XCircle size={20} />
              </button>
            </div>
            <div className="text-sm text-gray-400">
              {formatFileSize(selectedFile.size)} • {selectedFile.type || 'audio file'}
            </div>
          </div>
        ) : isDragActive ? (
          <div>
            <Upload size={48} className="mx-auto text-purple-400 animate-bounce" />
            <p className="text-2xl font-bold text-purple-300 mt-4">Drop your audio file here</p>
          </div>
        ) : (
          <div>
            <FilePlus2 size={48} className="mx-auto text-gray-400 group-hover:text-purple-400 transition-colors" />
            <p className="text-2xl font-bold text-gray-300 group-hover:text-purple-200 transition-colors mt-4">
              Choose Audio File
            </p>
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors mt-2">
              Drag & drop or click to browse
            </p>
            <p className="text-sm text-gray-500 mt-4">
              Supported formats: MP3, WAV, FLAC, M4A • Max size: {formatFileSize(maxSize)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioDropzone;
