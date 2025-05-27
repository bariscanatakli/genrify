# Simple test for TensorFlow functionality
import tensorflow as tf
import sys

def check_tensorflow():
    # Print version to validate import worked
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU/CUDA availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_details = []
        for gpu in gpus:
            try:
                gpu_details.append(tf.config.experimental.get_device_details(gpu))
            except:
                # Fallback for older TF versions
                gpu_details.append({"name": gpu.name})
        
        print(f"GPU available: {len(gpus)}")
        for i, details in enumerate(gpu_details):
            name = details.get('name', f'GPU:{i}')
            print(f"  GPU {i}: {name}")
        
        # Set memory growth to avoid taking all GPU memory
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth set for {gpu.name}")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
    else:
        print("No GPU available, using CPU")
    
    # Simple test that doesn't create models or use GPU
    tf_constant = tf.constant([1, 2, 3])
    print("TensorFlow test passed")
    print("ok")

def check_cuda():
    """Detailed check for CUDA availability and configuration"""
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check CUDA availability
    cuda_available = tf.test.is_built_with_cuda()
    print(f"CUDA built: {cuda_available}")
    
    # GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU devices available: {len(gpus)}")
    
    # Is GPU available for TensorFlow
    gpu_available = tf.test.is_gpu_available()
    print(f"GPU available for TF: {gpu_available}")
    
    # CUDA version - using direct attribute access if available
    if hasattr(tf.sysconfig, 'get_build_info'):
        build_info = tf.sysconfig.get_build_info()
        cuda_version = build_info.get('cuda_version', 'unknown')
        cudnn_version = build_info.get('cudnn_version', 'unknown')
        print(f"CUDA version: {cuda_version}")
        print(f"cuDNN version: {cudnn_version}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            
        # Test simple GPU operation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
                print(f"GPU computation successful: {c.numpy().sum()}")
        except Exception as e:
            print(f"GPU test failed: {e}")
    
    print("ok")

if __name__ == "__main__":
    # Check if an argument was provided
    if len(sys.argv) > 1 and sys.argv[1] == "cuda":
        check_cuda()
    else:
        check_tensorflow()
