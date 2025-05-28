#!/usr/bin/env python3
"""
Performance profiler to identify bottlenecks in audio processing
"""
import time
import sys
import os
from pathlib import Path

# Add the API directory to path
sys.path.append(str(Path(__file__).parent))

def profile_predict_genre(audio_path):
    """Profile the entire predict_genre pipeline to identify bottlenecks"""
    
    print(f"🎵 Profiling audio file: {audio_path}")
    total_start = time.time()
    
    # Import timing
    import_start = time.time()
    from process import predict_genre, extract_segments, audio_to_melspec, classifier_model
    import_time = time.time() - import_start
    print(f"📦 Import time: {import_time:.3f}s")
    
    # Check if model is loaded
    model_check_start = time.time()
    model_ready = classifier_model is not None
    model_check_time = time.time() - model_check_start
    print(f"🤖 Model check: {model_check_time:.3f}s (Ready: {model_ready})")
    
    if not model_ready:
        print("❌ Model not loaded, cannot profile")
        return
    
    # Segment extraction timing
    segment_start = time.time()
    segments = extract_segments(audio_path, segment_seconds=30, overlap=0.5)
    segment_time = time.time() - segment_start
    segment_count = len(segments) if segments else 0
    print(f"✂️ Segment extraction: {segment_time:.3f}s ({segment_count} segments)")
    
    if not segments:
        print("❌ No segments extracted, trying full audio")
        full_audio_start = time.time()
        mel = audio_to_melspec(audio_path)
        full_audio_time = time.time() - full_audio_start
        print(f"🎶 Full audio processing: {full_audio_time:.3f}s")
        if mel is not None:
            segments = [mel]
            segment_count = 1
        else:
            print("❌ Could not process audio")
            return
    
    # Model prediction timing
    prediction_start = time.time()
    
    # Test batch prediction
    try:
        import numpy as np
        batch_start = time.time()
        segment_batch = np.stack(segments, axis=0)
        batch_prep_time = time.time() - batch_start
        print(f"📊 Batch preparation: {batch_prep_time:.3f}s")
        
        model_start = time.time()
        probs_list = classifier_model.predict(segment_batch, verbose=0)
        model_time = time.time() - model_start
        print(f"🧠 Model inference (batch): {model_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        print("🔄 Falling back to sequential prediction...")
        
        sequential_start = time.time()
        probs_list = []
        for i, segment in enumerate(segments):
            seg_start = time.time()
            segment_batch = np.expand_dims(segment, axis=0)
            segment_probs = classifier_model.predict(segment_batch, verbose=0)[0]
            seg_time = time.time() - seg_start
            print(f"  Segment {i+1}: {seg_time:.3f}s")
            probs_list.append(segment_probs)
        sequential_time = time.time() - sequential_start
        print(f"🧠 Model inference (sequential): {sequential_time:.3f}s")
    
    prediction_time = time.time() - prediction_start
    print(f"🎯 Total prediction: {prediction_time:.3f}s")
    
    # Post-processing timing
    postprocess_start = time.time()
    if len(probs_list) > 0:
        import numpy as np
        avg_probs = np.mean(probs_list, axis=0)
        pred_idx = np.argmax(avg_probs)
        confidence = float(avg_probs[pred_idx])
    postprocess_time = time.time() - postprocess_start
    print(f"📈 Post-processing: {postprocess_time:.3f}s")
    
    total_time = time.time() - total_start
    print(f"\n⏱️ TOTAL TIME: {total_time:.3f}s")
    
    # Breakdown
    print(f"\n📊 TIME BREAKDOWN:")
    print(f"  Import: {import_time:.3f}s ({import_time/total_time*100:.1f}%)")
    print(f"  Segment extraction: {segment_time:.3f}s ({segment_time/total_time*100:.1f}%)")
    print(f"  Model prediction: {prediction_time:.3f}s ({prediction_time/total_time*100:.1f}%)")
    print(f"  Post-processing: {postprocess_time:.3f}s ({postprocess_time/total_time*100:.1f}%)")
    
    return {
        'total_time': total_time,
        'import_time': import_time,
        'segment_time': segment_time,
        'prediction_time': prediction_time,
        'postprocess_time': postprocess_time,
        'segment_count': segment_count
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python performance_profiler.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)
    
    profile_predict_genre(audio_file)
