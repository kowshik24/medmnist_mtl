import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import os

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
                
                # Ensure target is 1D
                target = target.view(-1)
                
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
                    
                    # Ensure target is 1D
                    target = target.view(-1)
                    
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
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': {task: [] for task in self.dataloaders.keys()}
        }
        
        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Store metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            
            # Store validation accuracies
            for task in self.dataloaders.keys():
                acc = self.calculate_accuracy(task, 'val')
                training_history['val_accuracy'][task].append(acc)
            
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint('best.pt')
                
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
        
        # After training, evaluate and save results
        self.save_training_history(training_history)
        test_results = self.evaluate()
        self.save_results(test_results)
        
    def calculate_accuracy(self, task, split='test'):
        """Calculate accuracy for a specific task and data split"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.dataloaders[task][split]:
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1)
                output = self.model(data, task)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return 100. * correct / total

    def evaluate(self):
        """Evaluate the model on test set"""
        self.model.eval()
        results = {}
        
        print("\nEvaluating on test set:")
        for task in self.dataloaders:
            accuracy = self.calculate_accuracy(task, 'test')
            results[task] = {
                'accuracy': accuracy,
                'predictions': []
            }
            print(f"{task}: {accuracy:.2f}%")
            
        return results
    
    def save_training_history(self, history):
        """Save training history plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path(self.config['paths']['results']) / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(results_dir / 'loss_curves.png')
        plt.close()
        
        # Plot validation accuracy for each task
        plt.figure(figsize=(10, 6))
        for task in history['val_accuracy']:
            plt.plot(history['val_accuracy'][task], label=f'{task}')
        plt.title('Validation Accuracy by Task')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(results_dir / 'accuracy_curves.png')
        plt.close()
        
    def save_results(self, results):
        """Save final evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path(self.config['paths']['results']) / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numerical results
        results_df = pd.DataFrame({
            task: {'accuracy': results[task]['accuracy']} 
            for task in results
        }).transpose()
        
        results_df.to_csv(results_dir / 'test_results.csv')
        
        # Plot final accuracies
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(results.keys()), 
                   y=[results[task]['accuracy'] for task in results])
        plt.title('Final Test Accuracy by Task')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Task')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / 'final_accuracies.png')
        plt.close()
        
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        save_path = Path(self.config['paths']['checkpoints']) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, save_path)