"""
Unit Tests for GraphSAGE Model and Python Bridge

Tests:
- Model creation and architecture
- Save/load functionality
- Prediction output format
- Bridge predict() function
- Error handling
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.graphsage import (
    GraphSAGEModel,
    create_model,
    save_checkpoint,
    load_checkpoint,
    save_model_for_inference,
    load_model_for_inference
)
from yantra_bridge import predict, get_model_info


class TestGraphSAGEModel:
    """Test GraphSAGE model architecture"""
    
    def test_model_creation(self):
        """Test model can be created"""
        model = create_model('cpu')
        assert model is not None
        assert isinstance(model, GraphSAGEModel)
    
    def test_model_parameters(self):
        """Test model has correct number of parameters"""
        model = create_model('cpu')
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params == 2_452_647  # Expected parameter count
    
    def test_model_forward_pass(self):
        """Test model forward pass works"""
        model = create_model('cpu')
        features = torch.randn(4, 978)  # Batch of 4
        
        predictions = model(features)
        
        assert 'code_embedding' in predictions
        assert 'confidence' in predictions
        assert 'imports' in predictions
        assert 'bugs' in predictions
        
        # Check output shapes
        assert predictions['code_embedding'].shape == (4, 512)
        assert predictions['confidence'].shape == (4,)
        assert predictions['imports'].shape == (4, 500)
        assert predictions['bugs'].shape == (4, 50)
    
    def test_model_eval_mode(self):
        """Test model can be set to eval mode"""
        model = create_model('cpu')
        model.eval()
        
        features = torch.randn(1, 978)
        with torch.no_grad():
            predictions = model(features)
        
        assert predictions is not None
    
    def test_predict_code_properties(self):
        """Test prediction helper function"""
        model = create_model('cpu')
        features = torch.randn(1, 978)
        
        result = model.predict_code_properties(features)
        
        assert 'code_embedding' in result
        assert 'confidence' in result
        assert 'predicted_imports' in result
        assert 'potential_bugs' in result
        
        # Check types
        assert isinstance(result['confidence'], float)
        assert isinstance(result['predicted_imports'], list)
        assert isinstance(result['potential_bugs'], list)
        assert 0.0 <= result['confidence'] <= 1.0


class TestModelSaveLoad:
    """Test model serialization"""
    
    def test_save_and_load_checkpoint(self):
        """Test checkpoint save/load cycle"""
        model = create_model('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            
            # Save
            save_checkpoint(
                model, optimizer, epoch=10,
                train_loss=0.5, val_loss=0.6,
                checkpoint_path=checkpoint_path
            )
            
            assert os.path.exists(checkpoint_path)
            
            # Load
            loaded_model, loaded_optimizer, metadata = load_checkpoint(
                checkpoint_path, device='cpu'
            )
            
            assert loaded_model is not None
            assert metadata['epoch'] == 10
            assert metadata['train_loss'] == 0.5
            assert metadata['val_loss'] == 0.6
    
    def test_save_and_load_for_inference(self):
        """Test inference model save/load"""
        model = create_model('cpu')
        model.eval()  # Set to eval mode for batch norm
        
        # Get original predictions
        features = torch.randn(1, 978)
        with torch.no_grad():
            original_output = model(features)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pt')
            
            # Save
            save_model_for_inference(model, model_path)
            assert os.path.exists(model_path)
            
            # Load
            loaded_model = load_model_for_inference(model_path, device='cpu')
            
            # Test loaded model produces same output
            with torch.no_grad():
                loaded_output = loaded_model(features)
            
            # Check outputs match (approximately)
            assert torch.allclose(
                original_output['confidence'],
                loaded_output['confidence'],
                atol=1e-5
            )


class TestPythonBridge:
    """Test yantra_bridge.py interface"""
    
    def test_predict_with_valid_features(self):
        """Test predict() with valid 978-dim features"""
        features = [0.5] * 978
        result = predict(features)
        
        assert isinstance(result, dict)
        assert 'code_suggestion' in result
        assert 'confidence' in result
        assert 'next_function' in result
        assert 'predicted_imports' in result
        assert 'potential_bugs' in result
        
        # Check types
        assert isinstance(result['code_suggestion'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['predicted_imports'], list)
        assert isinstance(result['potential_bugs'], list)
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_predict_with_invalid_length(self):
        """Test predict() rejects wrong feature length"""
        features = [0.5] * 100  # Wrong size
        
        with pytest.raises(ValueError, match="Expected 978 features"):
            predict(features)
    
    def test_predict_with_list(self):
        """Test predict() accepts list input"""
        features = [float(i) / 978 for i in range(978)]
        result = predict(features)
        
        assert result is not None
        assert 'confidence' in result
    
    def test_predict_with_tuple(self):
        """Test predict() accepts tuple input"""
        features = tuple([0.5] * 978)
        result = predict(features)
        
        assert result is not None
    
    def test_get_model_info(self):
        """Test get_model_info() returns status"""
        info = get_model_info()
        
        assert isinstance(info, dict)
        assert 'status' in info
        assert 'version' in info
        
        # Should have loaded status if model available
        if info['status'] == 'loaded':
            assert 'model_size_mb' in info


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_extreme_feature_values(self):
        """Test model handles extreme values"""
        model = create_model('cpu')
        model.eval()  # Set to eval mode
        
        # Very large values
        features_large = torch.full((2, 978), 1000.0)  # Batch size > 1
        with torch.no_grad():
            result = model(features_large)
        assert not torch.isnan(result['confidence']).any()
        
        # Very small values
        features_small = torch.full((2, 978), 0.001)
        with torch.no_grad():
            result = model(features_small)
        assert not torch.isnan(result['confidence']).any()
        
        # Negative values
        features_neg = torch.full((2, 978), -10.0)
        with torch.no_grad():
            result = model(features_neg)
        assert not torch.isnan(result['confidence']).any()
    
    def test_zero_features(self):
        """Test model handles all-zero features"""
        model = create_model('cpu')
        model.eval()  # Set to eval mode
        features = torch.zeros(2, 978)  # Batch size > 1
        
        with torch.no_grad():
            result = model(features)
        assert result is not None
        assert not torch.isnan(result['confidence']).any()
    
    def test_batch_size_one(self):
        """Test model handles single sample in eval mode"""
        model = create_model('cpu')
        model.eval()  # Must be in eval mode for batch size 1
        features = torch.randn(1, 978)
        
        with torch.no_grad():
            result = model(features)
        assert result['confidence'].shape == (1,)
    
    def test_large_batch(self):
        """Test model handles large batch"""
        model = create_model('cpu')
        features = torch.randn(128, 978)
        
        result = model(features)
        assert result['confidence'].shape == (128,)


class TestPerformance:
    """Test performance characteristics"""
    
    def test_inference_speed(self):
        """Test inference is fast enough (<10ms)"""
        import time
        
        model = create_model('cpu')
        model.eval()
        features = torch.randn(1, 978)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(features)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model(features)
                times.append((time.time() - start) * 1000)
        
        median_time = sorted(times)[len(times)//2]
        assert median_time < 10.0, f"Inference too slow: {median_time:.2f}ms"
    
    def test_model_size(self):
        """Test model size is reasonable"""
        model = create_model('cpu')
        size_mb = model.get_model_size_mb()
        
        # Should be less than 50 MB for this small model
        assert size_mb < 50.0, f"Model too large: {size_mb:.2f} MB"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
