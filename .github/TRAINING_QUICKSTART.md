# GraphSAGE Training - Quick Start Guide

## Prerequisites ✅
- [x] M4 MacBook (or any Mac with Apple Silicon)
- [x] PyTorch with MPS support installed
- [x] datasets and tqdm packages installed

```bash
# Verify MPS is available
.venv/bin/python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Step 1: Download Full Dataset (~10 minutes)

```bash
cd /Users/vivekdurairaj/Projects/yantra

# Download all 13,328 examples, filter for ~8,000 Python solutions
.venv/bin/python scripts/download_codecontests.py --output ~/.yantra/datasets/codecontests
```

**Output:**
- `~/.yantra/datasets/codecontests/train.jsonl` (~6,400 examples)
- `~/.yantra/datasets/codecontests/validation.jsonl` (~1,600 examples)
- `~/.yantra/datasets/codecontests/stats.json`

---

## Step 2: Start Training (3-8 hours overnight)

```bash
# Full training with Apple Silicon GPU
.venv/bin/python src-python/training/train.py --device mps

# Quick test (2 epochs, 100 examples)
.venv/bin/python src-python/training/train.py --epochs 2 --limit 100 --device mps

# CPU fallback (if MPS issues)
.venv/bin/python src-python/training/train.py --device cpu
```

**Training Parameters:**
- Epochs: 100 (will early stop if converged)
- Batch size: 32
- Learning rate: 0.001 (adaptive via ReduceLROnPlateau)
- Device: MPS (Apple Silicon GPU)

**Expected Output:**
```
Epoch 1/100
------------------------------------------------------------
  Batch [10/200] Loss: 1.2345 (Avg: 1.3456)
  ...
  Validating...
  
  Epoch 1 Summary:
    Train Loss: 1.2345
    Val Loss: 1.1234
    Time: 180.5s
    LR: 0.001000
    ✓ New best model saved! Val Loss: 1.1234
```

---

## Step 3: Monitor Training

### Checkpoints Location
```bash
ls -lh ~/.yantra/checkpoints/graphsage/
```

**Files:**
- `best_model.pt` - Best validation loss (use this for inference!)
- `last_model.pt` - Most recent checkpoint (resume if interrupted)
- `checkpoint_epoch_5.pt` - Periodic checkpoints every 5 epochs
- `checkpoint_epoch_10.pt`
- ...

### Resume Training (if interrupted)
```bash
.venv/bin/python src-python/training/train.py \
  --resume ~/.yantra/checkpoints/graphsage/last_model.pt \
  --device mps
```

---

## Step 4: Use Trained Model

### Option A: Load in Python Bridge (yantra_bridge.py)

Edit `src-python/yantra_bridge.py`:
```python
def _ensure_model(device: str = "cpu"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    # Load trained model instead of creating new
    checkpoint_path = Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
    if checkpoint_path.exists():
        from model.graphsage import load_model_for_inference
        _MODEL = load_model_for_inference(str(checkpoint_path), device)
    else:
        # Fallback to untrained model
        _MODEL = _graphsage_mod.create_model(device)
    
    return _MODEL
```

### Option B: Load Directly in Python

```python
from model.graphsage import load_model_for_inference

# Load trained model
model = load_model_for_inference(
    "~/.yantra/checkpoints/graphsage/best_model.pt",
    device="mps"
)

# Run inference
import torch
features = torch.randn(1, 978)  # Your 978-dim feature vector
predictions = model(features)

print(f"Confidence: {predictions['confidence'].item():.3f}")
print(f"Predicted imports: {predictions['imports'].shape}")
```

---

## Training Configuration (Customize)

Edit `src-python/training/config.py`:

```python
class TrainingConfig:
    # Model
    INPUT_DIM = 978
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 32        # Reduce if OOM, increase if underutilized
    NUM_EPOCHS = 100       # More epochs = better learning
    LEARNING_RATE = 0.001  # Lower = slower but more stable
    
    # Loss weights (tune for your priority)
    LOSS_WEIGHT_CODE = 1.0
    LOSS_WEIGHT_CONFIDENCE = 1.0
    LOSS_WEIGHT_IMPORTS = 0.5  # Increase if import prediction important
    LOSS_WEIGHT_BUGS = 0.5     # Increase if bug detection important
```

---

## Troubleshooting

### 1. MPS Not Available
```bash
# Check PyTorch version
.venv/bin/python -c "import torch; print(torch.__version__)"

# Should be 2.10+ with MPS support
# If not, reinstall:
.venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### 2. Out of Memory (OOM)
```bash
# Reduce batch size
.venv/bin/python src-python/training/train.py --batch-size 16 --device mps

# Or use CPU (slower but more memory)
.venv/bin/python src-python/training/train.py --device cpu
```

### 3. Training Too Slow
```bash
# Check device being used (should see "mps")
# Check Activity Monitor → GPU History
# Reduce batch size if GPU underutilized
# Increase num_workers in config.py (multi-process data loading)
```

### 4. Loss Not Decreasing
- Check learning rate (might be too high or low)
- Verify dataset quality (real features vs random)
- Increase epochs (might need more time)
- Check loss weights (one task might dominate)

### 5. Dataset Download Fails
```bash
# Install/upgrade datasets
.venv/bin/pip install --upgrade datasets

# Check HuggingFace access
.venv/bin/python -c "from datasets import load_dataset; load_dataset('deepmind/code_contests', split='train[:10]')"
```

---

## Performance Benchmarks (Expected)

### M4 MacBook Pro
- **Dataset:** 8,000 examples
- **Batch size:** 32
- **Per epoch:** 2-5 minutes (MPS)
- **100 epochs:** 3-8 hours
- **GPU memory:** ~2-3 GB
- **CPU usage:** 20-40%

### Early Stopping
- Training will auto-stop if no improvement for 10 epochs
- Typical convergence: 40-60 epochs
- Expected time: **2-4 hours** (overnight)

---

## Commands Cheat Sheet

```bash
# Full training
.venv/bin/python src-python/training/train.py --device mps

# Quick test (2 epochs)
.venv/bin/python src-python/training/train.py --epochs 2 --limit 100 --device mps

# Custom settings
.venv/bin/python src-python/training/train.py \
  --epochs 50 \
  --batch-size 16 \
  --device mps

# Resume training
.venv/bin/python src-python/training/train.py \
  --resume ~/.yantra/checkpoints/graphsage/last_model.pt

# CPU fallback
.venv/bin/python src-python/training/train.py --device cpu

# Download dataset
.venv/bin/python scripts/download_codecontests.py

# Test model inference
.venv/bin/python -c "
from model.graphsage import load_model_for_inference
import torch
model = load_model_for_inference('~/.yantra/checkpoints/graphsage/best_model.pt', 'cpu')
print('Model loaded successfully!')
"
```

---

## Next Steps After Training

1. **Evaluate on test set** (not implemented yet)
2. **Integrate with Yantra orchestrator** (load in yantra_bridge.py)
3. **Benchmark inference latency** (target: <10ms)
4. **Collect real training data** (from GNN feature extraction)
5. **Retrain with real features** (placeholder features currently)
6. **Knowledge distillation** (from LLM outputs)
7. **Incremental learning** (online updates from new code)

---

**Last Updated:** November 25, 2025  
**Status:** ✅ Ready for production training
