import { createSlice, PayloadAction } from '@reduxjs/toolkit';

// Make sure ActiveTab type includes 'spectrogram' with correct spelling
export type ActiveTab = 'prediction' | 'advanced' | 'pipeline' | 'spectrogram' | 'batch';

interface UiState {
  activeTab: ActiveTab;
  navHovered: boolean;
  loading: boolean;
  error: string | null;
  showVisualization: boolean;
  useGpu: boolean;
  useAdvancedPrediction: boolean;
  serverHealth: any;
}

const initialState: UiState = {
  activeTab: 'prediction',
  navHovered: false,
  loading: false,
  error: null,
  showVisualization: false,
  useGpu: true,
  useAdvancedPrediction: false,
  serverHealth: null,
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<ActiveTab>) => {
      state.activeTab = action.payload;
    },
    setNavHovered: (state, action: PayloadAction<boolean>) => {
      state.navHovered = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setShowVisualization: (state, action: PayloadAction<boolean>) => {
      state.showVisualization = action.payload;
    },
    setUseGpu: (state, action: PayloadAction<boolean>) => {
      state.useGpu = action.payload;
    },
    setUseAdvancedPrediction: (state, action: PayloadAction<boolean>) => {
      state.useAdvancedPrediction = action.payload;
    },
    setServerHealth: (state, action: PayloadAction<UiState['serverHealth']>) => {
      state.serverHealth = action.payload;
    },
  },
});

export const { 
  setActiveTab, 
  setNavHovered, 
  setLoading,
  setError,
  setShowVisualization,
  setUseGpu,
  setUseAdvancedPrediction,
  setServerHealth 
} = uiSlice.actions;

export default uiSlice.reducer;
