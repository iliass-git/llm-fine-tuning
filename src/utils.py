import subprocess
import sys
import torch
import gc

def check_gpu():
    if not torch.cuda.is_available():
        print("\n⚠️ WARNING: No GPU detected!")
        print("Please ensure GPU is enabled and properly configured")
        sys.exit(1)
    print(f"\n📊 System Info:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def install_dependencies():
    print("\n📦 Installing required packages...")
    packages = [
        "transformers", "peft", "datasets", "bitsandbytes", 
        "accelerate", "trl", "sentencepiece", "protobuf"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + packages, check=True)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n🧹 GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB allocated, {torch.cuda.memory_reserved(0) / 1e9:.2f}GB reserved")
