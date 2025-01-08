import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Is CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")