import torch

def test_gpu():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    tensor = torch.randn(3,3).cuda()
    print(f"Tensor on GPU:\n{tensor}")

if __name__ == "__main__":
    test_gpu()
