"""
PyTorch Dataset and DataLoader for GraphSAGE Training

Loads CodeContests examples and extracts 978-dim features using Yantra's GNN.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class CodeContestsDataset(Dataset):
    """
    PyTorch Dataset for CodeContests examples
    
    Each example contains:
    - features: 978-dim feature vector (from GNN)
    - labels: Ground truth for training (code, imports, bugs, confidence)
    """
    
    def __init__(
        self,
        data_file: str,
        max_examples: Optional[int] = None,
        cache_features: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_file: Path to .jsonl file (train.jsonl or validation.jsonl)
            max_examples: Limit number of examples (for testing)
            cache_features: Cache extracted features in memory
        """
        self.data_file = Path(data_file)
        self.cache_features = cache_features
        self.feature_cache = {}
        
        # Load examples
        self.examples = []
        with open(self.data_file, 'r') as f:
            for idx, line in enumerate(f):
                if max_examples and idx >= max_examples:
                    break
                self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} examples from {self.data_file.name}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example
        
        Returns:
            Dictionary with:
            - features: [978] tensor
            - code_target: Target code embedding (placeholder for now)
            - imports_target: [500] multi-label tensor
            - bugs_target: [50] multi-label tensor
            - confidence_target: scalar tensor
        """
        # Check cache first
        if self.cache_features and idx in self.feature_cache:
            return self.feature_cache[idx]
        
        example = self.examples[idx]
        
        # TODO: Extract real features from code using GNN
        # For now, use random features as placeholder
        features = torch.randn(978)
        
        # TODO: Extract real labels from solutions and tests
        # For now, use placeholder targets
        
        # Code embedding target (will use contrastive learning)
        code_target = torch.randn(512)
        
        # Import targets (multi-label, random for now)
        imports_target = torch.zeros(500)
        # Randomly mark 2-5 imports as positive
        num_imports = np.random.randint(2, 6)
        import_indices = np.random.choice(500, num_imports, replace=False)
        imports_target[import_indices] = 1.0
        
        # Bug targets (multi-label, mostly zeros - no bugs)
        bugs_target = torch.zeros(50)
        # Randomly mark 0-2 potential bugs
        num_bugs = np.random.randint(0, 3)
        if num_bugs > 0:
            bug_indices = np.random.choice(50, num_bugs, replace=False)
            bugs_target[bug_indices] = 1.0
        
        # Confidence target (high for validated solutions)
        confidence_target = torch.tensor(0.9 + 0.1 * np.random.random())
        
        result = {
            'features': features,
            'code_target': code_target,
            'imports_target': imports_target,
            'bugs_target': bugs_target,
            'confidence_target': confidence_target,
            'example_id': example['id']
        }
        
        # Cache if enabled
        if self.cache_features:
            self.feature_cache[idx] = result
        
        return result


def create_dataloaders(
    train_file: str,
    val_file: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_train_examples: Optional[int] = None,
    max_val_examples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_file: Path to train.jsonl
        val_file: Path to validation.jsonl
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        max_train_examples: Limit training examples (for testing)
        max_val_examples: Limit validation examples (for testing)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CodeContestsDataset(
        train_file,
        max_examples=max_train_examples,
        cache_features=True
    )
    
    val_dataset = CodeContestsDataset(
        val_file,
        max_examples=max_val_examples,
        cache_features=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_train.jsonl>")
        sys.exit(1)
    
    print("Testing dataset loader...")
    
    dataset = CodeContestsDataset(sys.argv[1], max_examples=10)
    print(f"\nDataset size: {len(dataset)}")
    
    # Test getting an item
    item = dataset[0]
    print(f"\nSample item keys: {item.keys()}")
    print(f"Features shape: {item['features'].shape}")
    print(f"Code target shape: {item['code_target'].shape}")
    print(f"Imports target shape: {item['imports_target'].shape}")
    print(f"Bugs target shape: {item['bugs_target'].shape}")
    print(f"Confidence target: {item['confidence_target'].item():.3f}")
    
    print("\n✓ Dataset loader test complete!")
