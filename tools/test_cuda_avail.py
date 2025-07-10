import torch
print(f"PyTorch ver: {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
print(f"CUDA ver: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device num: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name()}")
    print(f"curr GPU proper: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA unavailable")