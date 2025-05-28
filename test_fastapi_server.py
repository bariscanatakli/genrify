#!/usr/bin/env python3
"""
Test the FastAPI server with file upload
"""
import requests
import time
import os

def test_fastapi_server():
    # Server configuration
    API_URL = "http://localhost:8888"
    
    # Test file
    test_file = "/home/baris/genrify/Nirvana - Smells Like Teen Spirit (Official Music Video)-rock.mp3"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    print("🚀 Testing FastAPI Server")
    print("=" * 50)
    
    # 1. Health Check
    print("1. Health Check...")
    try:
        response = requests.get(f"{API_URL}/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   GPU Available: {health['gpu_available']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        
        if health['status'] != 'healthy':
            print("❌ Server not healthy")
            return
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # 2. File Upload and Prediction
    print("\n2. Genre Prediction...")
    try:
        with open(test_file, 'rb') as f:
            files = {
                'file': (os.path.basename(test_file), f, 'audio/mpeg')
            }
            data = {
                'use_gpu': 'true'
            }
            
            print(f"   Uploading: {os.path.basename(test_file)}")
            start_time = time.time()
            
            response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=120)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Prediction successful!")
                print(f"   🎵 Genre: {result['predicted_genre']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   ⏱️  Processing Time: {result['processing_time']:.2f}s")
                print(f"   🌐 Total Request Time: {duration:.2f}s")
                
                # Show top 3 probabilities
                print(f"   📈 Top Genres:")
                sorted_probs = sorted(result['genre_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for i, (genre, prob) in enumerate(sorted_probs[:3]):
                    print(f"      {i+1}. {genre}: {prob:.3f}")
                
                # Performance evaluation
                if duration < 20:
                    print(f"   🚀 EXCELLENT: {duration:.1f}s is much faster than original!")
                elif duration < 30:
                    print(f"   👍 GOOD: {duration:.1f}s is better than original")
                else:
                    print(f"   ⚠️  SLOW: {duration:.1f}s needs optimization")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 FastAPI Server Test Complete!")

if __name__ == "__main__":
    test_fastapi_server()
