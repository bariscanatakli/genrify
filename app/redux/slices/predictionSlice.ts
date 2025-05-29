import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface GenrePrediction {
  predicted_genre: string;
  confidence: number;
  genre_probabilities: Record<string, number>;
  processing_time: number;
  visualization_data?: {
    spectrogram?: string;
    mel_spectrogram?: string;
    mfcc_features?: number[][];
    chroma_features?: number[][];
    spectral_features?: {
      spectral_centroid: number[];
      spectral_rolloff: number[];
      zero_crossing_rate: number[];
    };
    tempo?: number;
    model_predictions?: number[];
  };
}

interface PredictionState {
  file: File | null;
  result: GenrePrediction | null;
}

const initialState: PredictionState = {
  file: null,
  result: null,
};

export const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    setFile: (state, action: PayloadAction<File | null>) => {
      state.file = action.payload;
    },
    setPredictionResult: (state, action: PayloadAction<GenrePrediction | null>) => {
      state.result = action.payload;
    },
    clearPrediction: (state) => {
      state.result = null;
    },
  },
});

export const { setFile, setPredictionResult, clearPrediction } = predictionSlice.actions;

export default predictionSlice.reducer;
