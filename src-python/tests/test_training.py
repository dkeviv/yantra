"""
Unit Tests for Training Pipeline

Tests:
- Dataset loading
- DataLoader creation
- Training configuration
- Multi-task loss computation
- Training epoch execution
"""

import pytest
import torch
import tempfile
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import CodeContestsDataset, create_dataloaders
from training.config import TrainingConfig
from training.train import MultiTaskLoss


class TestCodeContestsDataset:
    """Test dataset loading"""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create temporary sample data file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write sample data
            for i in range(10):
                data = {
                    'id': f'problem_{i}',  # Changed from 'name' to 'id'
                    'name': f'problem_{i}',
                    'description': f'Test problem {i}',
                    'public_tests': {'input': ['1'], 'output': ['2']},
                    'private_tests': {'input': ['3'], 'output': ['4']},
                    'generated_tests': {'input': ['5'], 'output': ['6']},
                    'solutions': {
                        'solution': [f'def solve():\n    return {i}']
                    }
                }
                f.write(json.dumps(data) + '\n')
            
            return f.name
    
    def test_dataset_creation(self, sample_data_file):
        """Test dataset can be created"""
        dataset = CodeContestsDataset(sample_data_file)
        assert len(dataset) == 10
    
    def test_dataset_getitem(self, sample_data_file):
        """Test dataset __getitem__ returns correct format"""
        dataset = CodeContestsDataset(sample_data_file)
        batch = dataset[0]
        
        # Should return a dict
        assert isinstance(batch, dict)
        assert 'features' in batch
        assert 'code_target' in batch
        assert 'confidence_target' in batch
        assert 'imports_target' in batch
        assert 'bugs_target' in batch
        
        # Check shapes
        assert batch['features'].shape == (978,)
        assert batch['code_target'].shape == (512,)
        assert batch['confidence_target'].shape == ()
        assert batch['imports_target'].shape == (500,)
        assert batch['bugs_target'].shape == (50,)
    
    def test_dataset_with_limit(self, sample_data_file):
        """Test dataset respects max_examples"""
        dataset = CodeContestsDataset(sample_data_file, max_examples=5)
        assert len(dataset) == 5
    
    def test_create_dataloaders(self, sample_data_file):
        """Test dataloader creation"""
        train_loader, val_loader = create_dataloaders(
            sample_data_file, sample_data_file,
            batch_size=2, num_workers=0
        )
        
        assert train_loader is not None
        assert val_loader is not None
        
        # Test batch
        batch = next(iter(train_loader))
        assert isinstance(batch, dict)
        assert 'features' in batch
        assert batch['features'].shape[0] == 2  # batch size
        assert batch['features'].shape[1] == 978


class TestTrainingConfig:
    """Test training configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = TrainingConfig()
        
        assert config.BATCH_SIZE == 32
        assert config.LEARNING_RATE == 0.001
        assert config.NUM_EPOCHS == 100
        assert config.EARLY_STOP_PATIENCE == 10
    
    def test_config_to_dict(self):
        """Test config serialization"""
        config = TrainingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'BATCH_SIZE' in config_dict
        assert 'LEARNING_RATE' in config_dict
    
    def test_config_attributes(self):
        """Test config has all required attributes"""
        config = TrainingConfig()
        
        # Check all required attributes exist
        assert hasattr(config, 'BATCH_SIZE')
        assert hasattr(config, 'LEARNING_RATE')
        assert hasattr(config, 'NUM_EPOCHS')
        assert hasattr(config, 'DEVICE')
        assert hasattr(config, 'CHECKPOINT_DIR')


class TestMultiTaskLoss:
    """Test multi-task loss computation"""
    
    def test_loss_creation(self):
        """Test loss can be created"""
        config = TrainingConfig()
        criterion = MultiTaskLoss(config)
        assert criterion is not None
    
    def test_loss_forward(self):
        """Test loss computation"""
        config = TrainingConfig()
        criterion = MultiTaskLoss(config)
        
        # Create sample predictions and targets
        batch_size = 4
        predictions = {
            'code_embedding': torch.randn(batch_size, 512),
            'confidence': torch.randn(batch_size),
            'imports': torch.sigmoid(torch.randn(batch_size, 500)),
            'bugs': torch.sigmoid(torch.randn(batch_size, 50))
        }
        
        targets = {
            'code_target': torch.randn(batch_size, 512),  
            'confidence_target': torch.rand(batch_size),  
            'imports_target': (torch.rand(batch_size, 500) > 0.5).float(),  
            'bugs_target': (torch.rand(batch_size, 50) > 0.5).float()  
        }
        
        loss, losses = criterion(predictions, targets)
        
        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)
        assert loss > 0
        
        # Check component losses (keys are simple names, not *_loss)
        assert 'total' in losses
        assert 'code' in losses
        assert 'confidence' in losses
        assert 'imports' in losses
        assert 'bugs' in losses
    
    def test_loss_weights(self):
        """Test loss weights are applied"""
        config = TrainingConfig()
        criterion = MultiTaskLoss(config)
        
        batch_size = 4
        predictions = {
            'code_embedding': torch.randn(batch_size, 512),
            'confidence': torch.randn(batch_size),
            'imports': torch.sigmoid(torch.randn(batch_size, 500)),
            'bugs': torch.sigmoid(torch.randn(batch_size, 50))
        }
        
        targets = {
            'code_target': torch.randn(batch_size, 512),
            'confidence_target': torch.rand(batch_size),
            'imports_target': (torch.rand(batch_size, 500) > 0.5).float(),
            'bugs_target': (torch.rand(batch_size, 50) > 0.5).float()
        }
        
        loss, losses = criterion(predictions, targets)
        
        # Loss should be weighted sum (using default weights from config)
        expected_loss = (
            config.LOSS_WEIGHT_CODE * losses['code'] +
            config.LOSS_WEIGHT_CONFIDENCE * losses['confidence'] +
            config.LOSS_WEIGHT_IMPORTS * losses['imports'] +
            config.LOSS_WEIGHT_BUGS * losses['bugs']
        )
        
        assert abs(losses['total'] - expected_loss) < 1e-5


class TestTrainingIntegration:
    """Integration tests for training pipeline"""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create temporary sample data file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(20):
                data = {
                    'id': f'problem_{i}',
                    'name': f'problem_{i}',
                    'description': f'Test problem {i}',
                    'public_tests': {'input': ['1'], 'output': ['2']},
                    'private_tests': {'input': ['3'], 'output': ['4']},
                    'generated_tests': {'input': ['5'], 'output': ['6']},
                    'solutions': {
                        'solution': [f'def solve():\n    return {i}']
                    }
                }
                f.write(json.dumps(data) + '\n')
            
            return f.name
    
    def test_end_to_end_training_step(self, sample_data_file):
        """Test one training step completes"""
        from model.graphsage import create_model
        
        # Setup
        config = TrainingConfig()
        model = create_model('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = MultiTaskLoss(config)
        
        train_loader, _ = create_dataloaders(
            sample_data_file, sample_data_file,
            batch_size=4,
            num_workers=0
        )
        
        # Training step
        model.train()
        batch = next(iter(train_loader))
        features = batch['features']
        
        optimizer.zero_grad()
        predictions = model(features)
        loss, losses = criterion(predictions, batch)
        loss.backward()
        optimizer.step()
        
        # Check loss is valid
        assert not torch.isnan(loss)
        assert loss > 0


class TestErrorHandling:
    """Test error handling"""
    
    def test_nonexistent_file(self):
        """Test dataset handles missing file"""
        with pytest.raises(FileNotFoundError):
            CodeContestsDataset('/nonexistent/file.jsonl')
    
    def test_invalid_json(self):
        """Test dataset handles invalid JSON gracefully"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('invalid json\n')
            f.write('more invalid\n')
            fname = f.name
        
        # Should skip invalid lines (or raise error on first bad line - depending on implementation)
        # Let's check if it raises an error
        try:
            dataset = CodeContestsDataset(fname)
            # If no error, should have 0 examples
            assert len(dataset) == 0
        except json.decoder.JSONDecodeError:
            # If it raises error on invalid JSON, that's also acceptable
            pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
