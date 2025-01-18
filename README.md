
# Multi-Task Learning for Medical Image Analysis

This repository implements a Multi-Task Learning (MTL) framework for medical image analysis using the MedMNIST dataset.

## Features
- Shared backbone architecture with task-specific heads
- Support for multiple MedMNIST tasks
- Configurable training parameters
- TensorBoard logging
- Checkpoint saving and loading

## Installation
```bash
git clone https://github.com/yourusername/medmnist-mtl
cd medmnist-mtl
pip install -r requirements.txt
```

## Usage
1. Configure the training parameters in `configs/mtl_config.yaml`
2. Run the training:
```bash
python main.py
```

## Project Structure
- `data/`: Data loading and preprocessing
- `models/`: Neural network architectures
- `trainers/`: Training and evaluation code
- `utils/`: Utility functions and metrics
- `configs/`: Configuration files
- `notebooks/`: Analysis notebooks

## Monitoring
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir logs
```

## Results
Training results and model checkpoints are saved in:
- `checkpoints/`: Model checkpoints
- `logs/`: TensorBoard logs
- `results/`: Evaluation results
```

This implementation:
1. Uses a shared backbone (ResNet18) with task-specific heads
2. Handles multiple MedMNIST tasks simultaneously
3. Implements proper logging and checkpointing
4. Uses configuration files for easy experimentation
5. Follows best practices for code organization

To use this code:
1. Clone the repository
2. Install requirements
3. Modify the config file as needed
4. Run `main.py`

The code will automatically:
- Download and prepare the MedMNIST datasets
- Train the multi-task model
- Log metrics to TensorBoard
- Save model checkpoints
- Evaluate performance on validation set

Additional features that could be added:
1. Gradient balancing strategies
2. More sophisticated loss weighting
3. Additional backbone architectures
4. Uncertainty estimation
5. Model interpretability tools

