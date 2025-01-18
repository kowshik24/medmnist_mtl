import torch.nn as nn

class TaskHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Add hidden layers
        for _ in range(hidden_layers - 1):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            current_dim = current_dim // 2
            
        # Add final classification layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)