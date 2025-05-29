import { configureStore } from '@reduxjs/toolkit';
import predictionReducer from './slices/predictionSlice';
import uiReducer from './slices/uiSlice';
import batchReducer from './slices/batchSlice';
import pipelineReducer from './slices/pipelineSlice';

export const store = configureStore({
  reducer: {
    prediction: predictionReducer,
    ui: uiReducer,
    batch: batchReducer,
    pipeline: pipelineReducer,
  },
  // Disable serializable check for better developer experience with audio files
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false
    })
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
