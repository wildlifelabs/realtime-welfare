import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"TorchVision {torchvision.__version__}")
print(f"NMS supported: {torchvision.ops.nms}")
print("-----------")
print(torch.cuda.get_device_name(0))
print("-----------")
