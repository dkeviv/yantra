"""
Training Configuration for GraphSAGE Model

All hyperparameters and training settings
"""

import os
from pathlib import Path
from typing import Dict, Any


class TrainingConfig:
    """Configuration for GraphSAGE training"""
    
    # Model architecture
    INPUT_DIM = 978
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 512
    EMBEDDING_DIM = 256
    NUM_IMPORTS = 500
    NUM_BUG_TYPES = 50
    DROPOUT = 0.1
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    
    # Learning rate scheduler
    LR_SCHEDULER = "ReduceLROnPlateau"
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    LR_MIN = 1e-6
    
    # Early stopping
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 0.001
    
    # Loss weights (for multi-task learning)
    LOSS_WEIGHT_CODE = 1.0
    LOSS_WEIGHT_CONFIDENCE = 1.0
    LOSS_WEIGHT_IMPORTS = 0.5
    LOSS_WEIGHT_BUGS = 0.5
    
    # Gradient clipping
    GRADIENT_CLIP_VALUE = 1.0
    
    # Device configuration
    DEVICE = "mps" if os.environ.get("FORCE_MPS") else "cpu"
    # Will auto-detect MPS at runtime
    
    # Data paths
    DATASET_DIR = Path.home() / ".yantra" / "datasets" / "codecontests"
    TRAIN_FILE = DATASET_DIR / "train.jsonl"
    VAL_FILE = DATASET_DIR / "validation.jsonl"
    
    # Checkpoint paths
    CHECKPOINT_DIR = Path.home() / ".yantra" / "checkpoints" / "graphsage"
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
    LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pt"
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
    
    # Data loading
    NUM_WORKERS = 0  # DataLoader workers (0 = main process)
    PIN_MEMORY = True
    
    # Training limits (for testing)
    MAX_TRAIN_EXAMPLES = None  # None = use all
    MAX_VAL_EXAMPLES = None
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=" * 60)
        print("Training Configuration")
        print("=" * 60)
        
        print("\n📐 Model Architecture:")
        print(f"  Input dimension: {cls.INPUT_DIM}")
        print(f"  Hidden layers: {cls.HIDDEN_DIM_1} → {cls.HIDDEN_DIM_2} → {cls.EMBEDDING_DIM}")
        print(f"  Dropout: {cls.DROPOUT}")
        
        print("\n⚙️  Training Hyperparameters:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        
        print("\n📊 Loss Weights:")
        print(f"  Code embedding: {cls.LOSS_WEIGHT_CODE}")
        print(f"  Confidence: {cls.LOSS_WEIGHT_CONFIDENCE}")
        print(f"  Imports: {cls.LOSS_WEIGHT_IMPORTS}")
        print(f"  Bugs: {cls.LOSS_WEIGHT_BUGS}")
        
        print("\n📁 Paths:")
        print(f"  Dataset: {cls.DATASET_DIR}")
        print(f"  Checkpoints: {cls.CHECKPOINT_DIR}")
        
        print("\n🖥️  Device:")
        print(f"  Target device: {cls.DEVICE}")
        
        print("=" * 60)


# Create checkpoint directory if it doesn't exist
TrainingConfig.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    TrainingConfig.print_config()
