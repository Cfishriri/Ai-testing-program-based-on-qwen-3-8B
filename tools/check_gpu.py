import sys
print("Python路径:", sys.executable)

import torch
import psutil

print("=" * 60)
print("环境检查")
print("=" * 60)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  显存: {props.total_memory / 1024**3:.2f} GB")

mem = psutil.virtual_memory()
print(f"\nCPU内存总量: {mem.total / 1024**3:.2f} GB")
print(f"CPU可用内存: {mem.available / 1024**3:.2f} GB")
