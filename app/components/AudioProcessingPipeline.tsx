'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent } from './ui/card';

interface StepData {
  name: string;
  icon: string;
  status: 'waiting' | 'processing' | 'completed' | 'error';
  description: string;
  duration?: number;
  detailsHtml?: string;
}

interface AudioProcessingPipelineProps {
  isProcessing: boolean;
  audioFile: File | null;
  useGpu: boolean;
  advancedAnalysis: boolean;
  onProcessingComplete?: (result: any) => void;
}

const AudioProcessingPipeline: React.FC<AudioProcessingPipelineProps> = ({
  isProcessing,
  audioFile,
  useGpu,
  advancedAnalysis,
  onProcessingComplete
}) => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [steps, setSteps] = useState<StepData[]>([
    {
      name: 'Audio Loading',
      icon: '🎵',
      status: 'waiting',
      description: 'Loading and validating audio file'
    },
    {
      name: 'Feature Extraction',
      icon: '📊',
      status: 'waiting',
      description: 'Extracting mel spectrograms and audio features'
    },
    {
      name: 'Segmentation',
      icon: '✂️',
      status: 'waiting',
      description: 'Creating overlapping 30-second segments'
    },
    {
      name: 'Model Inference',
      icon: '🧠',
      status: 'waiting',
      description: 'Running neural network prediction'
    },
    {
      name: 'Post-Processing',
      icon: '📈',
      status: 'waiting',
      description: 'Aggregating predictions and generating results'
    }
  ]);

  // Reset steps when audio file changes
  useEffect(() => {
    if (audioFile) {
      resetSteps();
    }
  }, [audioFile]);

  // Processing simulation
  useEffect(() => {
    if (isProcessing && audioFile) {
      simulateProcessing();
    } else if (!isProcessing) {
      // If processing stopped and we were in the middle of it, 
      // update steps that were in progress to completed
      setSteps(prev => prev.map(step => 
        step.status === 'processing' ? {...step, status: 'completed'} : step
      ));
    }
  }, [isProcessing, audioFile]);

  const resetSteps = () => {
    setActiveStep(0);
    setSteps(steps.map(step => ({...step, status: 'waiting', duration: undefined, detailsHtml: undefined})));
  };

  const simulateProcessing = async () => {
    let stepDurations: number[] = [];
    
    // These are roughly representative of the actual durations in a real pipeline
    if (useGpu) {
      // GPU timings are faster
      stepDurations = [0.8, 3.2, 1.5, 2.1, 0.9];
    } else {
      // CPU timings are slower
      stepDurations = [1.2, 5.8, 2.2, 9.5, 1.3];
    }
    
    // Adjust for advanced analysis which takes longer
    if (advancedAnalysis) {
      stepDurations = stepDurations.map(d => d * 1.4);
    }
    
    // Process each step
    for (let i = 0; i < steps.length && isProcessing; i++) {
      // Update current step to processing
      setActiveStep(i);
      setSteps(prev => {
        const newSteps = [...prev];
        newSteps[i] = {...newSteps[i], status: 'processing'};
        return newSteps;
      });
      
      // Wait for the step duration
      await new Promise(resolve => setTimeout(resolve, stepDurations[i] * 1000));
      
      // Add details based on step
      let details = '';
      switch(i) {
        case 0:
          details = `<p>File: ${audioFile?.name}</p><p>Size: ${audioFile?.size ? (audioFile?.size / 1024 / 1024).toFixed(2) : '0.00'} MB</p>`;
          break;
        case 1:
          details = `<p>Generated Mel Spectrogram ${advancedAnalysis ? 'and detailed audio features' : ''}</p>`;
          break;
        case 2:
          details = `<p>Created ${Math.floor(Math.random() * 3) + 3} overlapping segments</p>`;
          break;
        case 3:
          details = useGpu 
            ? '<p>Using GPU acceleration for inference</p><p>Model: CNN with SpecAugment</p>' 
            : '<p>Using CPU for inference</p><p>Model: CNN with SpecAugment</p>';
          break;
        case 4:
          details = `<p>Aggregated predictions from all segments</p><p>Generated ${advancedAnalysis ? 'detailed visualizations' : 'basic output'}</p>`;
          break;
      }
      
      // Mark step as completed
      setSteps(prev => {
        const newSteps = [...prev];
        newSteps[i] = {
          ...newSteps[i], 
          status: 'completed', 
          duration: stepDurations[i],
          detailsHtml: details
        };
        return newSteps;
      });
    }
    
    // Processing complete
    if (onProcessingComplete) {
      onProcessingComplete({ 
        success: true, 
        steps: steps,
        total_time: stepDurations.reduce((a, b) => a + b, 0),
        gpu_used: useGpu
      });
    }
  };

  const getStepIcon = (step: StepData) => {
    switch (step.status) {
      case 'processing':
        return '⏳';
      case 'completed':
        return '✅';
      case 'error':
        return '❌';
      default:
        return step.icon;
    }
  };

  return (
    <div className="space-y-4">
      <div className="glass-card p-4 text-center">
        <h3 className="text-lg font-medium mb-2">
          {isProcessing 
            ? '🔄 Processing Pipeline Active' 
            : '⚙️ Audio Processing Pipeline'}
        </h3>
        <p className="text-sm text-gray-400">
          {isProcessing 
            ? `Using ${useGpu ? 'GPU' : 'CPU'} for ${advancedAnalysis ? 'advanced' : 'standard'} analysis` 
            : 'Step-by-step audio analysis process'}
        </p>
      </div>

      <div className="space-y-2">
        {steps.map((step, index) => (
          <Card 
            key={index}
            className={`
              transition-all duration-300 border-l-4
              ${activeStep === index && isProcessing ? 'bg-blue-900/20 border-l-blue-500' : ''}
              ${step.status === 'completed' ? 'bg-green-900/10 border-l-green-500' : ''}
              ${step.status === 'error' ? 'bg-red-900/10 border-l-red-500' : ''}
            `}
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`
                    w-10 h-10 rounded-full flex items-center justify-center text-xl
                    ${step.status === 'processing' ? 'bg-blue-500/20 animate-pulse' : ''}
                    ${step.status === 'completed' ? 'bg-green-500/20' : ''}
                    ${step.status === 'error' ? 'bg-red-500/20' : ''}
                    ${step.status === 'waiting' ? 'bg-gray-500/20' : ''}
                  `}>
                    {getStepIcon(step)}
                  </div>
                  <div>
                    <h4 className="font-medium">{step.name}</h4>
                    <p className="text-sm text-gray-400">{step.description}</p>
                    
                    {/* Step details after completion */}
                    {step.status === 'completed' && step.detailsHtml && (
                      <div 
                        className="text-xs text-gray-300 mt-2 pl-2 border-l-2 border-green-500/30"
                        dangerouslySetInnerHTML={{ __html: step.detailsHtml }} 
                      />
                    )}
                  </div>
                </div>
                
                {step.duration && (
                  <div className="text-right">
                    <span className="text-sm text-blue-300 font-mono">{step.duration.toFixed(1)}s</span>
                  </div>
                )}
                
                {step.status === 'processing' && (
                  <div className="w-16 flex justify-end">
                    <div className="spinner-premium"></div>
                  </div>
                )}
              </div>
              
              {step.status === 'processing' && (
                <div className="mt-3">
                  <div className="h-1 w-full bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 rounded-full animate-pulse"
                      style={{ width: `${Math.random() * 100}%` }}  
                    ></div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default AudioProcessingPipeline;
