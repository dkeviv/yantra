#!/usr/bin/env python3
"""
Evaluate GraphSAGE model on HumanEval benchmark

This script:
1. Loads HumanEval dataset (164 programming problems)
2. Extracts features from each problem's prompt and canonical solution
3. Runs GraphSAGE inference to predict code quality metrics
4. Reports prediction accuracy and confidence calibration
"""

import sys
from pathlib import Path

# Add src-python to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src-python"))

import torch
import numpy as np
from human_eval.data import read_problems
from model.graphsage import load_model_for_inference
from training.feature_extractor import get_feature_extractor
from typing import Dict, List
import json


def evaluate_humaneval(
    checkpoint_path: str,
    device: str = "cpu",
    output_file: str = None
) -> Dict:
    """
    Evaluate GraphSAGE on HumanEval benchmark
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to run on (cpu, cuda, mps)
        output_file: Optional path to save results
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("GraphSAGE HumanEval Evaluation")
    print("="*60)
    
    # Load model
    print(f"\n📦 Loading model from: {checkpoint_path}")
    model = load_model_for_inference(checkpoint_path, device=device)
    model.eval()
    print(f"✓ Model loaded: {model.get_model_size_mb():.2f} MB")
    
    # Load HumanEval problems
    print("\n📊 Loading HumanEval dataset...")
    problems = read_problems()
    print(f"✓ Loaded {len(problems)} problems")
    
    # Initialize feature extractor
    extractor = get_feature_extractor()
    
    # Evaluation results
    results = {
        "total_problems": len(problems),
        "predictions": [],
        "statistics": {
            "avg_confidence": 0.0,
            "high_confidence_count": 0,
            "low_confidence_count": 0,
        }
    }
    
    print("\n🔍 Running inference on all problems...")
    print("-" * 60)
    
    confidences = []
    
    for task_id, problem in problems.items():
        # Extract features from canonical solution
        canonical_solution = problem["canonical_solution"]
        
        try:
            features_np = extractor.extract_features_from_code(
                canonical_solution, 
                language="python"
            )
            features = torch.from_numpy(features_np).unsqueeze(0).float().to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = model(features)
            
            # Extract predictions
            confidence = torch.sigmoid(predictions["confidence"]).item()
            confidences.append(confidence)
            
            # Store result
            result = {
                "task_id": task_id,
                "confidence": confidence,
                "high_confidence": confidence > 0.8,
                "entry_point": problem["entry_point"],
            }
            results["predictions"].append(result)
            
            # Print progress every 20 problems
            if len(results["predictions"]) % 20 == 0:
                print(f"  Processed {len(results['predictions'])}/{len(problems)} problems...")
        
        except Exception as e:
            print(f"  ⚠️  Error processing {task_id}: {e}")
            continue
    
    # Calculate statistics
    confidences = np.array(confidences)
    results["statistics"]["avg_confidence"] = float(confidences.mean())
    results["statistics"]["std_confidence"] = float(confidences.std())
    results["statistics"]["median_confidence"] = float(np.median(confidences))
    results["statistics"]["min_confidence"] = float(confidences.min())
    results["statistics"]["max_confidence"] = float(confidences.max())
    results["statistics"]["high_confidence_count"] = int((confidences > 0.8).sum())
    results["statistics"]["medium_confidence_count"] = int(((confidences >= 0.5) & (confidences <= 0.8)).sum())
    results["statistics"]["low_confidence_count"] = int((confidences < 0.5).sum())
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\n📊 Processed: {len(results['predictions'])}/{results['total_problems']} problems")
    print(f"\n📈 Confidence Statistics:")
    print(f"  Average:     {results['statistics']['avg_confidence']:.3f}")
    print(f"  Median:      {results['statistics']['median_confidence']:.3f}")
    print(f"  Std Dev:     {results['statistics']['std_confidence']:.3f}")
    print(f"  Range:       [{results['statistics']['min_confidence']:.3f}, {results['statistics']['max_confidence']:.3f}]")
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"  High (>0.8):     {results['statistics']['high_confidence_count']} ({results['statistics']['high_confidence_count']/len(results['predictions'])*100:.1f}%)")
    print(f"  Medium (0.5-0.8): {results['statistics']['medium_confidence_count']} ({results['statistics']['medium_confidence_count']/len(results['predictions'])*100:.1f}%)")
    print(f"  Low (<0.5):      {results['statistics']['low_confidence_count']} ({results['statistics']['low_confidence_count']/len(results['predictions'])*100:.1f}%)")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    avg_conf = results['statistics']['avg_confidence']
    if avg_conf > 0.8:
        print(f"  ✅ Model shows HIGH confidence in canonical solutions (avg {avg_conf:.3f})")
        print(f"     This suggests the model recognizes high-quality code patterns.")
    elif avg_conf > 0.6:
        print(f"  ⚠️  Model shows MODERATE confidence (avg {avg_conf:.3f})")
        print(f"     Model can distinguish code quality but may need more training.")
    else:
        print(f"  ❌ Model shows LOW confidence (avg {avg_conf:.3f})")
        print(f"     Model needs more training on high-quality examples.")
    
    print("="*60)
    
    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate GraphSAGE on HumanEval benchmark"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: ~/.yantra/checkpoints/graphsage/best_model.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
    
    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_path = Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python src-python/training/train.py")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluate_humaneval(
            checkpoint_path=str(checkpoint_path),
            device=device,
            output_file=args.output
        )
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
