#!/usr/bin/env python3
"""
Test script to verify API route performance after optimization
"""
import requests
import time
import os

def test_api_performance():
    # Test file path
    test_file = "/home/baris/genrify/Nirvana - Smells Like Teen Spirit (Official Music Video)-rock.mp3"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Available test files:")
        test_dir = "/home/baris/genrify/app/api/test_files"
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                if f.endswith('.mp3'):
                    print(f"  - {f}")
        return
    
    print(f"Testing API performance with: {os.path.basename(test_file)}")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:3000/api/predict"
    
    # Prepare the file
    with open(test_file, 'rb') as f:
        files = {
            'file': (os.path.basename(test_file), f, 'audio/mpeg')
        }
        data = {
            'use_gpu': 'true',  # Enable GPU optimization
            'include_visualization': 'false'  # Skip visualization for speed test
        }
        
        print("Starting API request...")
        start_time = time.time()
        
        try:
            response = requests.post(url, files=files, data=data, timeout=120)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"API Response Time: {duration:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                if 'predicted_genre' in result:
                    print(f"Predicted Genre: {result['predicted_genre']}")
                    print(f"Confidence: {result.get('confidence', 'N/A'):.3f}")
                    
                    if result.get('using_mock'):
                        print("⚠️  WARNING: Using mock data (optimization failed)")
                    else:
                        print("✅ Real prediction successful!")
                        
                        # Compare with our target (direct Python was ~12s)
                        if duration < 20:
                            print(f"🚀 EXCELLENT: {duration:.1f}s is much faster than original 42s!")
                        elif duration < 30:
                            print(f"👍 GOOD: {duration:.1f}s is better than original 42s")
                        else:
                            print(f"⚠️  SLOW: {duration:.1f}s is still slow compared to direct Python (~12s)")
                else:
                    print("❌ Invalid response format")
                    print(f"Response: {result}")
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out (>120s)")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_performance()
