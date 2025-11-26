"""
Yantra Bridge - Python side
Provides the interface between Rust and the GraphSAGE model

This module will be called from Rust via PyO3 for:
- Model inference
- Feature processing
- Prediction generation
"""

__version__ = "0.1.0"

import threading
from typing import List, Optional
from pathlib import Path

# Defer importing the graphsage module until needed. This avoids hard failures
# at import time when native dependencies (like torch) are not available to the
# embedded Python interpreter used by Rust tests. _graphsage_mod will be set
# by _ensure_model when a real import is attempted.
_graphsage_mod = None
_graphsage_import_error = None


# Module-level singleton for the model
_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_TRAINED = False  # Track if we loaded trained weights


def _ensure_model(device: str = "cpu"):
    """Lazily create and return the GraphSAGE model instance."""
    global _MODEL, _MODEL_TRAINED
    if _MODEL is not None:
        return _MODEL

    # Try importing the graphsage implementation on demand. Capture and store
    # any import error so callers can return a graceful fallback.
    global _graphsage_mod, _graphsage_import_error
    if _graphsage_mod is None and _graphsage_import_error is None:
        try:
            from model import graphsage as _g
            _graphsage_mod = _g
        except Exception as e:
            _graphsage_import_error = e

    if _graphsage_mod is None:
        raise ImportError(f"Cannot import GraphSAGE model: {_graphsage_import_error}")

    with _MODEL_LOCK:
        if _MODEL is None:
            # Try to load trained model from checkpoint
            checkpoint_path = Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
            
            if checkpoint_path.exists():
                try:
                    print(f"[yantra_bridge] Loading trained model from: {checkpoint_path}")
                    _MODEL = _graphsage_mod.load_model_for_inference(str(checkpoint_path), device)
                    _MODEL_TRAINED = True
                    print(f"[yantra_bridge] ✓ Trained model loaded successfully")
                except Exception as e:
                    print(f"[yantra_bridge] ⚠️  Failed to load trained model: {e}")
                    print(f"[yantra_bridge] Falling back to untrained model")
                    _MODEL = _graphsage_mod.create_model(device)
                    _MODEL_TRAINED = False
            else:
                print(f"[yantra_bridge] No trained model found at {checkpoint_path}")
                print(f"[yantra_bridge] Creating untrained model (predictions will be random)")
                _MODEL = _graphsage_mod.create_model(device)
                _MODEL_TRAINED = False
    
    return _MODEL


def predict(features: List[float]) -> dict:
    """Run model inference on a 978-dim feature vector.

    Args:
        features: List of 978 floats

    Returns:
        dict matching the Rust ModelPrediction structure:
            - code_suggestion: str
            - confidence: float (0.0-1.0)
            - next_function: Optional[str]
            - predicted_imports: List[str]
            - potential_bugs: List[str]
    """
    if not isinstance(features, (list, tuple)):
        # Allow numpy arrays and other sequences that implement list() conversion
        try:
            features = list(features)
        except Exception:
            raise ValueError("features must be a sequence of floats")

    if len(features) != 978:
        raise ValueError(f"Expected 978 features, got {len(features)}")

    # Try to run the model. If model import or torch are not available in the
    # embedded interpreter (common in constrained test environments), return
    # a sensible placeholder instead of raising.
    try:
        model = _ensure_model(device="cpu")
        import torch
        # Convert to tensor of shape [1, 978]
        tensor = torch.tensor([features], dtype=torch.float32)

        # Use model-provided postprocessing helper
        result = model.predict_code_properties(tensor)

        # Build a friendly code suggestion string from predicted imports and confidence
        imports = result.get("predicted_imports", []) or []
        potential_bugs = result.get("potential_bugs", []) or []
        confidence = float(result.get("confidence", 0.0))

        import_lines = "\n".join([f"import {imp}" for imp in imports]) if imports else ""

        code_suggestion = (
            f"{import_lines}\n\n# Suggested by GraphSAGE (confidence={confidence:.3f})\n"
            "def suggested_function():\n    pass\n"
        )

        return {
            "code_suggestion": code_suggestion,
            "confidence": confidence,
            "next_function": None,
            "predicted_imports": imports,
            "potential_bugs": potential_bugs,
        }
    except ImportError as ie:
        # Log import error and return placeholder; tests expect bridge to work
        print(f"[yantra_bridge] ImportError while loading model: {ie}")
        return {
            "code_suggestion": "# GraphSAGE model not available in this environment\npass",
            "confidence": 0.0,
            "next_function": None,
            "predicted_imports": [],
            "potential_bugs": [],
        }
    except Exception as e:
        # For other runtime errors, also return a placeholder but log the error
        print(f"[yantra_bridge] Error during prediction: {e}")
        return {
            "code_suggestion": "# GraphSAGE prediction error\npass",
            "confidence": 0.0,
            "next_function": None,
            "predicted_imports": [],
            "potential_bugs": [],
        }


def get_model_info() -> dict:
    """Return information about the loaded model (or import error)."""
    if _graphsage_mod is None:
        return {
            "status": "import_error",
            "version": __version__,
            "message": f"GraphSAGE import failed: {_graphsage_import_error}",
        }

    try:
        model = _ensure_model(device="cpu")
        size_mb = model.get_model_size_mb()
        return {
            "status": "loaded",
            "version": __version__,
            "model_size_mb": float(size_mb),
            "trained": _MODEL_TRAINED,
            "checkpoint": str(Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt") if _MODEL_TRAINED else None,
        }
    except Exception as e:
        return {
            "status": "error",
            "version": __version__,
            "message": str(e),
        }
