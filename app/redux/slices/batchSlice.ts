import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface BatchPredictionItem {
  filename: string;
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  processing_time: number;
  error?: string;
}

export interface BatchProcessingResponse {
  predictions: BatchPredictionItem[];
  total_files: number;
  successful_predictions: number;
  failed_predictions: number;
  total_processing_time: number;
}

interface BatchState {
  files: File[];
  results: BatchPredictionItem[];
  summary: BatchProcessingResponse | null;
  progress: number;
}

const initialState: BatchState = {
  files: [],
  results: [],
  summary: null,
  progress: 0,
};

export const batchSlice = createSlice({
  name: 'batch',
  initialState,
  reducers: {
    setBatchFiles: (state, action: PayloadAction<File[]>) => {
      state.files = action.payload;
    },
    addBatchFiles: (state, action: PayloadAction<File[]>) => {
      state.files = [...state.files, ...action.payload];
    },
    removeBatchFile: (state, action: PayloadAction<string>) => {
      state.files = state.files.filter(file => file.name !== action.payload);
    },
    setBatchResults: (state, action: PayloadAction<BatchPredictionItem[]>) => {
      state.results = action.payload;
    },
    setBatchSummary: (state, action: PayloadAction<BatchProcessingResponse | null>) => {
      state.summary = action.payload;
    },
    setBatchProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    },
    clearBatchState: (state) => {
      state.results = [];
      state.summary = null;
      state.progress = 0;
    },
  },
});

export const { 
  setBatchFiles, 
  addBatchFiles, 
  removeBatchFile,
  setBatchResults,
  setBatchSummary,
  setBatchProgress,
  clearBatchState
} = batchSlice.actions;

export default batchSlice.reducer;
