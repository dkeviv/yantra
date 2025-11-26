#!/usr/bin/env python3
"""
Training Script for GraphSAGE Model

Implements:
- Multi-task learning (code, confidence, imports, bugs)
- Checkpointing and model saving
- Validation and early stopping
- Learning rate scheduling
- Training metrics tracking

Usage:
    python src-python/training/train.py [--epochs 100] [--batch-size 32] [--device mps]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.graphsage import (
    GraphSAGEModel,
    create_model,
    save_checkpoint,
    load_checkpoint
)
from training.dataset import create_dataloaders
from training.config import TrainingConfig


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    
    Losses:
    1. Code embedding: Cosine embedding loss (similarity)
    2. Confidence: MSE loss
    3. Imports: Binary cross-entropy (multi-label)
    4. Bugs: Binary cross-entropy (multi-label)
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Code embedding loss (cosine similarity)
        # For now, use MSE as placeholder (will use contrastive learning later)
        code_loss = self.mse_loss(
            predictions['code_embedding'],
            targets['code_target']
        )
        
        # Confidence loss (MSE)
        confidence_loss = self.mse_loss(
            predictions['confidence'],
            targets['confidence_target']
        )
        
        # Imports loss (BCE for multi-label)
        imports_loss = self.bce_loss(
            predictions['imports'],
            targets['imports_target']
        )
        
        # Bugs loss (BCE for multi-label)
        bugs_loss = self.bce_loss(
            predictions['bugs'],
            targets['bugs_target']
        )
        
        # Weighted combination
        total_loss = (
            self.config.LOSS_WEIGHT_CODE * code_loss +
            self.config.LOSS_WEIGHT_CONFIDENCE * confidence_loss +
            self.config.LOSS_WEIGHT_IMPORTS * imports_loss +
            self.config.LOSS_WEIGHT_BUGS * bugs_loss
        )
        
        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'code': code_loss.item(),
            'confidence': confidence_loss.item(),
            'imports': imports_loss.item(),
            'bugs': bugs_loss.item()
        }
        
        return total_loss, loss_dict


def train_epoch(
    model: GraphSAGEModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    config: TrainingConfig
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'code': 0.0,
        'confidence': 0.0,
        'imports': 0.0,
        'bugs': 0.0
    }
    
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        features = batch['features'].to(device)
        targets = {
            'code_target': batch['code_target'].to(device),
            'confidence_target': batch['confidence_target'].to(device),
            'imports_target': batch['imports_target'].to(device),
            'bugs_target': batch['bugs_target'].to(device)
        }
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        
        # Compute loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRADIENT_CLIP_VALUE
        )
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        
        # Log progress
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_losses['total'] / (batch_idx + 1)
            print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(Avg: {avg_loss:.4f})")
    
    # Compute epoch averages
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def validate(
    model: GraphSAGEModel,
    val_loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'code': 0.0,
        'confidence': 0.0,
        'imports': 0.0,
        'bugs': 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            features = batch['features'].to(device)
            targets = {
                'code_target': batch['code_target'].to(device),
                'confidence_target': batch['confidence_target'].to(device),
                'imports_target': batch['imports_target'].to(device),
                'bugs_target': batch['bugs_target'].to(device)
            }
            
            # Forward pass
            predictions = model(features)
            
            # Compute loss
            loss, loss_dict = criterion(predictions, targets)
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
    
    # Compute averages
    num_batches = len(val_loader)
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def train(
    config: TrainingConfig,
    resume_from: str = None,
    device: str = None
) -> None:
    """
    Main training loop
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        device: Device override (cpu/mps/cuda)
    """
    # Setup device
    if device is None:
        device = config.DEVICE
    
    # Auto-detect MPS on Apple Silicon
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        device = "cpu"
    
    device = torch.device(device)
    print(f"🖥️  Using device: {device}")
    
    # Create model
    print("\n📐 Creating model...")
    model = create_model(str(device))
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.LR_PATIENCE,
        factor=config.LR_FACTOR,
        min_lr=config.LR_MIN
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and Path(resume_from).exists():
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
        model, optimizer, metadata = load_checkpoint(
            resume_from,
            model,
            optimizer,
            str(device)
        )
        start_epoch = metadata['epoch'] + 1
        best_val_loss = metadata.get('val_loss', float('inf'))
    
    # Create dataloaders
    print("\n📊 Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        str(config.TRAIN_FILE),
        str(config.VAL_FILE),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        max_train_examples=config.MAX_TRAIN_EXAMPLES,
        max_val_examples=config.MAX_VAL_EXAMPLES
    )
    
    # Create loss criterion
    criterion = MultiTaskLoss(config)
    
    # Training loop
    print(f"\n🚀 Starting training from epoch {start_epoch + 1}...")
    print("=" * 60)
    
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config
        )
        
        # Validate
        print("\n  Validating...")
        val_losses = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_losses['total'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Train Loss: {train_losses['total']:.4f}")
        print(f"    Val Loss: {val_losses['total']:.4f}")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, epoch,
                train_losses['total'],
                val_losses['total'],
                str(checkpoint_path),
                metadata={'losses': train_losses}
            )
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, epoch,
                train_losses['total'],
                val_losses['total'],
                str(config.BEST_MODEL_PATH),
                metadata={'best': True}
            )
            print(f"    ✓ New best model saved! Val Loss: {best_val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Save last model
        save_checkpoint(
            model, optimizer, epoch,
            train_losses['total'],
            val_losses['total'],
            str(config.LAST_MODEL_PATH)
        )
        
        # Early stopping
        if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train GraphSAGE model')
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'mps', 'cuda'],
        default=None,
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit training examples (for testing)'
    )
    
    args = parser.parse_args()
    
    # Override config with command-line args
    config = TrainingConfig
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.limit:
        config.MAX_TRAIN_EXAMPLES = args.limit
        config.MAX_VAL_EXAMPLES = args.limit // 5
    
    # Print configuration
    config.print_config()
    
    # Check dataset exists
    if not config.TRAIN_FILE.exists():
        print(f"\n❌ Error: Training data not found at {config.TRAIN_FILE}")
        print(f"   Please run: python scripts/download_codecontests.py")
        sys.exit(1)
    
    # Start training
    train(config, resume_from=args.resume, device=args.device)


if __name__ == "__main__":
    main()
