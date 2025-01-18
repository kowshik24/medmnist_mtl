import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm

class MTLTrainer:
    def __init__(self, model, dataloaders, config):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler']['eta_min']
        )
        
        # Setup logging
        self.writer = SummaryWriter(config['paths']['logs'])
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        task_losses = {task: 0 for task in self.dataloaders.keys()}
        
        # Create progress bar
        pbar = tqdm(total=sum(len(dl['train']) for dl in self.dataloaders.values()),
                   desc=f'Epoch {epoch}')
        
        # Training loop
        for task in self.dataloaders:
            for batch_idx, (data, target) in enumerate(self.dataloaders[task]['train']):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data, task)
                loss = self.criterion(output, target) * self.config['training']['task_weights'][task]
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                task_losses[task] += loss.item()
                
                pbar.update(1)
                
        pbar.close()
        
        # Log metrics
        self.writer.add_scalar('Loss/train', total_loss, epoch)
        for task in task_losses:
            self.writer.add_scalar(f'Loss/{task}/train', task_losses[task], epoch)
            
        return total_loss
        
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        task_metrics = {task: {'correct': 0, 'total': 0} for task in self.dataloaders.keys()}
        
        with torch.no_grad():
            for task in self.dataloaders:
                for data, target in self.dataloaders[task]['val']:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data, task)
                    
                    val_loss += self.criterion(output, target).item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    task_metrics[task]['correct'] += pred.eq(target.view_as(pred)).sum().item()
                    task_metrics[task]['total'] += target.size(0)
        
        # Calculate and log metrics
        for task in task_metrics:
            acc = 100. * task_metrics[task]['correct'] / task_metrics[task]['total']
            self.writer.add_scalar(f'Accuracy/{task}/val', acc, epoch)
            
        return val_loss
        
    def train(self):
        best_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint('best.pt')
                
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
                
    def save_checkpoint(self, filename):
        save_path = Path(self.config['paths']['checkpoints']) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, save_path)