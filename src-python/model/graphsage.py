"""
GraphSAGE Model for Code Property Prediction
Architecture: 978 → 512 → 512 → 256 → Multiple Prediction Heads
Target Model Size: ~140 MB

Prediction Heads:
1. Code Suggestion (embedding + decoder)
2. Confidence Score (0.0-1.0)
3. Next Function Call
4. Predicted Imports
5. Potential Bugs

Training Strategy:
- Success-only learning (pass_rate >= 0.9)
- Knowledge distillation from LLMs
- Incremental updates from working code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class SAGEConv(nn.Module):
    """GraphSAGE Convolution Layer with mean aggregation"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(SAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Transform neighbor features
        self.neighbor_linear = nn.Linear(in_features, out_features, bias=False)
        # Transform self features
        self.self_linear = nn.Linear(in_features, out_features, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Node features [batch_size, in_features] or [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes] (optional, for graph learning)
        Returns:
            Updated node features [batch_size, out_features]
        """
        # Self transformation
        self_features = self.self_linear(x)
        
        # Neighbor aggregation (mean)
        if adj is not None:
            # Normalize adjacency matrix
            degree = adj.sum(dim=1, keepdim=True) + 1e-6
            adj_norm = adj / degree
            
            # Aggregate neighbor features
            neighbor_features = torch.matmul(adj_norm, x)
            neighbor_features = self.neighbor_linear(neighbor_features)
        else:
            # No graph structure, use self features only
            neighbor_features = self.neighbor_linear(x)
        
        # Combine self and neighbor features
        output = self_features + neighbor_features
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


class GraphSAGEEncoder(nn.Module):
    """3-layer GraphSAGE encoder: 978 → 512 → 512 → 256"""
    
    def __init__(self, input_dim: int = 978, dropout: float = 0.1):
        super(GraphSAGEEncoder, self).__init__()
        
        self.conv1 = SAGEConv(input_dim, 512, dropout)
        self.conv2 = SAGEConv(512, 512, dropout)
        self.conv3 = SAGEConv(512, 256, dropout)
        
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode node features through 3-layer GraphSAGE
        Args:
            x: Input features [batch_size, 978]
            adj: Adjacency matrix (optional)
        Returns:
            Encoded features [batch_size, 256]
        """
        # Layer 1: 978 → 512
        x = self.conv1(x, adj)
        x = self.batch_norm1(x)
        
        # Layer 2: 512 → 512
        x = self.conv2(x, adj)
        x = self.batch_norm2(x)
        
        # Layer 3: 512 → 256
        x = self.conv3(x, adj)
        x = self.batch_norm3(x)
        
        return x


class CodeSuggestionHead(nn.Module):
    """Prediction head for code suggestions (generates embeddings)"""
    
    def __init__(self, input_dim: int = 256, embedding_dim: int = 512):
        super(CodeSuggestionHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate code embedding"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return embedding for similarity matching


class ConfidenceHead(nn.Module):
    """Prediction head for confidence score (0.0-1.0)"""
    
    def __init__(self, input_dim: int = 256):
        super(ConfidenceHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict confidence score"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Bound to [0, 1]
        return x.squeeze(-1)


class ImportPredictionHead(nn.Module):
    """Prediction head for import suggestions (multi-label classification)"""
    
    def __init__(self, input_dim: int = 256, num_imports: int = 500):
        super(ImportPredictionHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_imports)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict import probabilities"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Multi-label probabilities
        return x


class BugPredictionHead(nn.Module):
    """Prediction head for potential bugs (multi-label classification)"""
    
    def __init__(self, input_dim: int = 256, num_bug_types: int = 50):
        super(BugPredictionHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_bug_types)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict bug probabilities"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Multi-label probabilities
        return x


class GraphSAGEModel(nn.Module):
    """
    Complete GraphSAGE model for code property prediction
    
    Architecture:
    - Encoder: 978 → 512 → 512 → 256 (GraphSAGE layers)
    - Code Suggestion Head: 256 → 512 (embedding)
    - Confidence Head: 256 → 1 (scalar)
    - Import Head: 256 → 500 (multi-label)
    - Bug Head: 256 → 50 (multi-label)
    
    Model Size: ~140 MB
    """
    
    def __init__(
        self,
        input_dim: int = 978,
        num_imports: int = 500,
        num_bug_types: int = 50,
        dropout: float = 0.1
    ):
        super(GraphSAGEModel, self).__init__()
        
        # Encoder
        self.encoder = GraphSAGEEncoder(input_dim, dropout)
        
        # Prediction heads
        self.code_head = CodeSuggestionHead(256, 512)
        self.confidence_head = ConfidenceHead(256)
        self.import_head = ImportPredictionHead(256, num_imports)
        self.bug_head = BugPredictionHead(256, num_bug_types)
        
    def forward(
        self,
        features: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model
        
        Args:
            features: Node features [batch_size, 978]
            adj: Adjacency matrix (optional)
            
        Returns:
            Dictionary with predictions:
            - code_embedding: [batch_size, 512]
            - confidence: [batch_size]
            - imports: [batch_size, num_imports]
            - bugs: [batch_size, num_bug_types]
        """
        # Encode features
        encoded = self.encoder(features, adj)
        
        # Generate predictions
        predictions = {
            'code_embedding': self.code_head(encoded),
            'confidence': self.confidence_head(encoded),
            'imports': self.import_head(encoded),
            'bugs': self.bug_head(encoded),
        }
        
        return predictions
    
    def predict_code_properties(
        self,
        features: torch.Tensor,
        import_vocab: Optional[List[str]] = None,
        bug_vocab: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Predict code properties with post-processing
        
        Args:
            features: Node features [batch_size, 978]
            import_vocab: List of import names (for decoding)
            bug_vocab: List of bug type names (for decoding)
            threshold: Threshold for multi-label predictions
            
        Returns:
            Dictionary with decoded predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(features)
            
            # Decode imports
            import_probs = predictions['imports'][0]  # First batch item
            import_indices = (import_probs > threshold).nonzero(as_tuple=True)[0]
            
            if import_vocab is not None:
                predicted_imports = [import_vocab[i] for i in import_indices.tolist()]
            else:
                predicted_imports = [f"import_{i}" for i in import_indices.tolist()]
            
            # Decode bugs
            bug_probs = predictions['bugs'][0]
            bug_indices = (bug_probs > threshold).nonzero(as_tuple=True)[0]
            
            if bug_vocab is not None:
                potential_bugs = [bug_vocab[i] for i in bug_indices.tolist()]
            else:
                potential_bugs = [f"bug_type_{i}" for i in bug_indices.tolist()]
            
            return {
                'code_embedding': predictions['code_embedding'][0].cpu().numpy(),
                'confidence': predictions['confidence'][0].item(),
                'predicted_imports': predicted_imports,
                'potential_bugs': potential_bugs,
            }
    
    def get_model_size_mb(self) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def create_model(device: str = 'cpu') -> GraphSAGEModel:
    """
    Factory function to create GraphSAGE model
    
    Args:
        device: 'cpu' or 'cuda'
        
    Returns:
        Initialized GraphSAGE model
    """
    model = GraphSAGEModel(
        input_dim=978,
        num_imports=500,
        num_bug_types=50,
        dropout=0.1
    )
    
    model = model.to(device)
    
    # Print model info
    size_mb = model.get_model_size_mb()
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"GraphSAGE Model Created:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model Size: {size_mb:.2f} MB")
    print(f"  Target: ~140 MB")
    print(f"  Device: {device}")
    
    return model


def save_checkpoint(
    model: GraphSAGEModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    checkpoint_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with optimizer state and training metadata
    
    Args:
        model: GraphSAGE model to save
        optimizer: Optimizer with current state
        epoch: Current epoch number
        train_loss: Training loss at checkpoint
        val_loss: Validation loss (if available)
        checkpoint_path: Path to save checkpoint file
        metadata: Additional metadata (learning rate, metrics, etc.)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': {
            'input_dim': 978,
            'num_imports': 500,
            'num_bug_types': 50,
        },
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print(f"  Epoch: {epoch}, Train Loss: {train_loss:.4f}", end="")
    if val_loss is not None:
        print(f", Val Loss: {val_loss:.4f}")
    else:
        print()


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[GraphSAGEModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Tuple[GraphSAGEModel, Optional[torch.optim.Optimizer], Dict]:
    """
    Load model checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Existing model to load weights into (creates new if None)
        optimizer: Optimizer to restore state (optional)
        device: Device to load model onto
        
    Returns:
        Tuple of (model, optimizer, metadata)
        metadata contains: epoch, train_loss, val_loss, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model if not provided
    if model is None:
        model = create_model(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0.0),
        'val_loss': checkpoint.get('val_loss', None),
        'model_config': checkpoint.get('model_config', {}),
        'metadata': checkpoint.get('metadata', {})
    }
    
    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {metadata['epoch']}, Train Loss: {metadata['train_loss']:.4f}")
    
    return model, optimizer, metadata


def save_model_for_inference(model: GraphSAGEModel, save_path: str) -> None:
    """
    Save model weights only (for inference, no optimizer state)
    
    Args:
        model: Trained GraphSAGE model
        save_path: Path to save model weights
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 978,
            'num_imports': 500,
            'num_bug_types': 50,
        }
    }, save_path)
    
    size_mb = model.get_model_size_mb()
    print(f"✓ Model saved for inference: {save_path}")
    print(f"  Model size: {size_mb:.2f} MB")


def load_model_for_inference(model_path: str, device: str = 'cpu') -> GraphSAGEModel:
    """
    Load trained model for inference only
    
    Args:
        model_path: Path to saved model
        device: Device to load model onto
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded for inference: {model_path}")
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing GraphSAGE Model...")
    
    # Create model
    model = create_model('cpu')
    
    # Test forward pass
    batch_size = 4
    features = torch.randn(batch_size, 978)
    
    print(f"\nTesting forward pass with batch_size={batch_size}")
    predictions = model(features)
    
    print(f"  Code embedding shape: {predictions['code_embedding'].shape}")
    print(f"  Confidence shape: {predictions['confidence'].shape}")
    print(f"  Imports shape: {predictions['imports'].shape}")
    print(f"  Bugs shape: {predictions['bugs'].shape}")
    
    # Test prediction function
    print("\nTesting prediction function...")
    result = model.predict_code_properties(features[:1])
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Predicted imports: {len(result['predicted_imports'])}")
    print(f"  Potential bugs: {len(result['potential_bugs'])}")
    
    print("\n✓ Model test complete!")
