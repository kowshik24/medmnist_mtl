import yaml
from pathlib import Path
from data.data_loader import MTLDataManager
from models.mtl_model import MTLModel
from trainers.mtl_trainer import MTLTrainer

def main():
    # Load configuration
    with open('configs/mtl_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    for path in config['paths'].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Setup data
    data_manager = MTLDataManager(config)
    data_manager.setup()
    dataloaders = data_manager.get_dataloaders()
    
    # Initialize model
    model = MTLModel(config)
    
    # Initialize trainer
    trainer = MTLTrainer(model, dataloaders, config)
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main()