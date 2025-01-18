import medmnist
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class MTLDataManager:
    def __init__(self, config):
        self.config = config
        self.tasks = config['data']['tasks']
        self.datasets = {}
        self.dataloaders = {}
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
    def setup(self):
        for task in self.tasks:
            # Load dataset
            data_class = getattr(medmnist, task.capitalize())
            train_dataset = data_class(split='train', transform=self.transform, download=True)
            test_dataset = data_class(split='test', transform=self.transform, download=True)
            
            # Split training into train/val
            train_size = int((1 - self.config['data']['val_split']) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            
            self.datasets[task] = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
            
            # Create dataloaders
            self.dataloaders[task] = {
                'train': DataLoader(
                    train_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=True,
                    num_workers=self.config['data']['num_workers']
                ),
                'val': DataLoader(
                    val_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['data']['num_workers']
                ),
                'test': DataLoader(
                    test_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['data']['num_workers']
                )
            }
    
    def get_dataloaders(self):
        return self.dataloaders