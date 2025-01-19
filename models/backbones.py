import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class SharedBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        
        if backbone_name == 'resnet18':
            # Use the latest way to load pretrained models
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            
            # Keep the 3-channel input as medical images are RGB
            # Just need to ensure the first conv layer is properly initialized
            if pretrained:
                # Save the pretrained weights
                conv1_weight = backbone.conv1.weight.data
                
                # Initialize new conv1 with pretrained weights
                backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                backbone.conv1.weight.data = conv1_weight
                
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 512
            
            # Add Batch Normalization for better feature normalization
            self.bn = nn.BatchNorm1d(self.feature_dim)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
            
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x