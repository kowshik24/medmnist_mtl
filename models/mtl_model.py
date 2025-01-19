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
        
        # Task-specific heads with correct number of classes
        self.task_heads = nn.ModuleDict()
        self.task_classes = {
            'pathmnist': 9,
            'organmnist': 11,
            'bloodmnist': 8
        }
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        for task in config['data']['tasks']:
            self.task_heads[task] = TaskHead(
                input_dim=config['model']['feature_dim'],
                num_classes=self.task_classes[task.lower()],
                hidden_layers=config['model']['task_specific_layers']
            )
        
        self._initialize_weights()
            
    def forward(self, x, task=None):
        features = self.backbone(x)
        features = self.dropout(features)
        
        if task is not None:
            return self.task_heads[task](features)
        
        # Return predictions for all tasks
        return {task: head(features) for task, head in self.task_heads.items()}
    
    def _initialize_weights(self):
        """Initialize the weights using He initialization"""
        for task_head in self.task_heads.values():
            for m in task_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)