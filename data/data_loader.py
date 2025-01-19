import medmnist
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class ConvertToRGB:
    """Convert single channel image to RGB by repeating the channel 3 times"""
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.size(0) == 1 else x

class MTLDataManager:
    def __init__(self, config):
        self.config = config
        self.tasks = config['data']['tasks']
        self.datasets = {}
        self.dataloaders = {}
        
        # Updated transform pipeline with RGB conversion
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ConvertToRGB(),  # Convert single-channel to RGB
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def setup(self):
        task_info = {
            'pathmnist': {'class': medmnist.PathMNIST, 'classes': 9},
            'organmnist': {'class': medmnist.OrganAMNIST, 'classes': 11},
            'bloodmnist': {'class': medmnist.BloodMNIST, 'classes': 8}
        }
        
        for task in self.tasks:
            task_lower = task.lower()
            if task_lower not in task_info:
                raise ValueError(f"Unsupported task name: {task}")
                
            data_class = task_info[task_lower]['class']
            
            # Load datasets with info flag to get labels
            train_dataset = data_class(split='train', transform=self.transform, download=True)
            test_dataset = data_class(split='test', transform=self.transform, download=True)
            
            print(f"\nTask: {task}")
            print(f"Number of classes: {task_info[task_lower]['classes']}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            
            # Split training into train/val
            train_size = int((1 - self.config['data']['val_split']) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            self.datasets[task] = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
            
            # Create dataloaders with pin_memory for faster data transfer to GPU
            self.dataloaders[task] = {
                'train': DataLoader(
                    train_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=True,
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=True,
                    persistent_workers=True  # Keep workers alive between iterations
                ),
                'val': DataLoader(
                    val_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=True,
                    persistent_workers=True
                ),
                'test': DataLoader(
                    test_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=True,
                    persistent_workers=True
                )
            }
            
            print(f"DataLoaders created for {task}")
    
    def get_dataloaders(self):
        return self.dataloaders