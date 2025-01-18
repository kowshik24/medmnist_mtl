import torch.nn as nn
from .backbones import SharedBackbone
from .task_heads import TaskHead

class MTLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Initialize shared backbone
        self.backbone = SharedBackbone(
            config['model']['backbone'],
            pretrained=config['model']['pretrained']
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        task_classes = {
            'pathmnist': 9,
            'organmnist': 11,
            'bloodmnist': 8
        }
        
        for task in config['data']['tasks']:
            self.task_heads[task] = TaskHead(
                input_dim=config['model']['feature_dim'],
                num_classes=task_classes[task],
                hidden_layers=config['model']['task_specific_layers']
            )
            
    def forward(self, x, task=None):
        features = self.backbone(x)
        
        if task is not None:
            return self.task_heads[task](features)
        
        # Return predictions for all tasks
        return {task: head(features) for task, head in self.task_heads.items()}