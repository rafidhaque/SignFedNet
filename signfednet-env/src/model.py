# model.py
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from torch import Tensor
import torch

class TrafficSignModel(nn.Module):
    def __init__(self, num_classes: int = 43, freeze_base: bool = False):
        """
        Transfer learning model for traffic sign recognition.
        
        Args:
            num_classes (int): Number of output classes (43 for GTSRB)
            freeze_base (bool): Whether to freeze base ResNet50 layers
        """
        super().__init__()
        
        # 1. Load pre-trained ResNet50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 2. Freeze base layers if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # 3. Replace final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # 4. Initialize new layer
        self._init_weights(self.base_model.fc)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with input tensor"""
        return self.base_model(x)
    
    def _init_weights(self, layer: nn.Module):
        """Initialize weights for new layers"""
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.constant_(layer.bias, 0)
            
    @staticmethod
    def get_transform():
        """Get standard transforms for input preprocessing"""
        return {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

# Test the model
if __name__ == "__main__":
    # Test CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = TrafficSignModel().to(device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)  # Batch of 2 images
    output = model(dummy_input)
    
    print("\nModel test successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output classes: {output.shape[1]}")
