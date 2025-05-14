import torch
print(torch.cuda.is_available())  # Should print True if GPU is usable
print(torch.cuda.get_device_name(0))  # Should print your GPU name if available


import torch
print(torch.version.cuda)
