import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface PipelineState {
  pipelineData: any | null;
  currentStep: number;
}

const initialState: PipelineState = {
  pipelineData: null,
  currentStep: 0,
};

export const pipelineSlice = createSlice({
  name: 'pipeline',
  initialState,
  reducers: {
    setPipelineData: (state, action: PayloadAction<any>) => {
      state.pipelineData = action.payload;
    },
    setCurrentStep: (state, action: PayloadAction<number>) => {
      state.currentStep = action.payload;
    },
    clearPipelineData: (state) => {
      state.pipelineData = null;
      state.currentStep = 0;
    },
  },
});

export const { setPipelineData, setCurrentStep, clearPipelineData } = pipelineSlice.actions;

export default pipelineSlice.reducer;
