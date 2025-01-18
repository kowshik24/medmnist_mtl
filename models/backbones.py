import torch
import torch.nn as nn
import torchvision.models as models

class SharedBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            # Modify first conv layer for single-channel input
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = 512
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
            
    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)