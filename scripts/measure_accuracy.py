#!/usr/bin/env python3
"""
Measure GraphSAGE Prediction Accuracy on HumanEval

This script:
1. Extracts features from HumanEval canonical solutions
2. Gets GraphSAGE confidence predictions
3. Executes test cases to get ground truth (pass/fail)
4. Compares predictions vs actual results
5. Reports accuracy, precision, recall, F1-score
"""

import sys
from pathlib import Path

# Add src-python to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src-python"))

import torch
import numpy as np
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from model.graphsage import load_model_for_inference
from training.feature_extractor import get_feature_extractor
from typing import Dict, List, Tuple
import json
import tempfile
import os


def run_tests_on_solution(problem: Dict) -> bool:
    """
    Execute test cases on canonical solution to get ground truth
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Create a test file with the solution and tests
    code = problem["prompt"] + problem["canonical_solution"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]
    
    # Execute in isolated namespace
    namespace = {}
    try:
        # Execute the solution
        exec(code, namespace)
        # Execute the tests
        exec(test_code, namespace)
        # Call check function
        namespace["check"](namespace[entry_point])
        return True
    except Exception as e:
        return False


def evaluate_accuracy(
    checkpoint_path: str,
    device: str = "cpu",
    confidence_threshold: float = 0.7
) -> Dict:
    """
    Evaluate GraphSAGE prediction accuracy on HumanEval
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to run on (cpu, cuda, mps)
        confidence_threshold: Threshold for binary classification (>threshold = predict pass)
    
    Returns:
        Dictionary with accuracy metrics
    """
    print("\n" + "="*70)
    print("GraphSAGE Accuracy Evaluation on HumanEval")
    print("="*70)
    
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
    
    print(f"\n🔍 Running predictions and executing tests...")
    print(f"   Confidence threshold: {confidence_threshold}")
    print("-" * 70)
    
    # Storage for results
    predictions = []  # Model confidence scores
    ground_truth = []  # Actual test results (True/False)
    binary_predictions = []  # Binary predictions based on threshold
    
    task_results = []
    
    for idx, (task_id, problem) in enumerate(problems.items(), 1):
        try:
            # 1. Get model prediction
            canonical_solution = problem["canonical_solution"]
            features_np = extractor.extract_features_from_code(
                canonical_solution, 
                language="python"
            )
            features = torch.from_numpy(features_np).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                model_output = model(features)
            
            confidence = torch.sigmoid(model_output["confidence"]).item()
            binary_pred = confidence > confidence_threshold
            
            # 2. Get ground truth by running tests
            actual_pass = run_tests_on_solution(problem)
            
            # 3. Store results
            predictions.append(confidence)
            ground_truth.append(actual_pass)
            binary_predictions.append(binary_pred)
            
            task_results.append({
                "task_id": task_id,
                "confidence": confidence,
                "predicted": binary_pred,
                "actual": actual_pass,
                "correct": binary_pred == actual_pass
            })
            
            # Progress indicator
            if idx % 20 == 0:
                print(f"   Processed {idx}/{len(problems)} problems...")
        
        except Exception as e:
            print(f"   ⚠️  Error processing {task_id}: {e}")
            continue
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    binary_predictions = np.array(binary_predictions)
    
    # Calculate metrics
    correct = binary_predictions == ground_truth
    accuracy = correct.sum() / len(correct)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = ((binary_predictions == True) & (ground_truth == True)).sum()
    fp = ((binary_predictions == True) & (ground_truth == False)).sum()
    tn = ((binary_predictions == False) & (ground_truth == False)).sum()
    fn = ((binary_predictions == False) & (ground_truth == True)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Results dictionary
    results = {
        "total_problems": len(predictions),
        "threshold": confidence_threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        },
        "ground_truth_distribution": {
            "passing": int(ground_truth.sum()),
            "failing": int((~ground_truth).sum())
        },
        "prediction_distribution": {
            "predict_pass": int(binary_predictions.sum()),
            "predict_fail": int((~binary_predictions).sum())
        },
        "confidence_stats": {
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
            "min": float(predictions.min()),
            "max": float(predictions.max())
        },
        "per_problem_results": task_results
    }
    
    # Print results
    print("\n" + "="*70)
    print("ACCURACY EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total problems: {results['total_problems']}")
    print(f"   Passing solutions: {results['ground_truth_distribution']['passing']} ({results['ground_truth_distribution']['passing']/results['total_problems']*100:.1f}%)")
    print(f"   Failing solutions: {results['ground_truth_distribution']['failing']} ({results['ground_truth_distribution']['failing']/results['total_problems']*100:.1f}%)")
    
    print(f"\n🎯 Classification Metrics (threshold={confidence_threshold}):")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1*100:.2f}%")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Positives:  {tp:3d}  (Correctly predicted PASS)")
    print(f"   True Negatives:  {tn:3d}  (Correctly predicted FAIL)")
    print(f"   False Positives: {fp:3d}  (Predicted PASS, actually FAIL)")
    print(f"   False Negatives: {fn:3d}  (Predicted FAIL, actually PASS)")
    
    print(f"\n📊 Model Confidence Statistics:")
    print(f"   Mean:   {results['confidence_stats']['mean']:.3f}")
    print(f"   Std:    {results['confidence_stats']['std']:.3f}")
    print(f"   Range:  [{results['confidence_stats']['min']:.3f}, {results['confidence_stats']['max']:.3f}]")
    
    print(f"\n💡 Interpretation:")
    if accuracy > 0.9:
        print(f"   ✅ EXCELLENT accuracy ({accuracy*100:.1f}%)")
    elif accuracy > 0.7:
        print(f"   ✓ GOOD accuracy ({accuracy*100:.1f}%)")
    elif accuracy > 0.5:
        print(f"   ⚠️  MODERATE accuracy ({accuracy*100:.1f}%)")
    else:
        print(f"   ❌ LOW accuracy ({accuracy*100:.1f}%) - needs more training")
    
    if precision > 0.8 and recall > 0.8:
        print(f"   ✅ Well-balanced predictions (precision={precision:.2f}, recall={recall:.2f})")
    elif precision > recall + 0.2:
        print(f"   ⚠️  Conservative predictions (high precision, low recall)")
    elif recall > precision + 0.2:
        print(f"   ⚠️  Aggressive predictions (high recall, low precision)")
    
    print("="*70)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Measure GraphSAGE accuracy on HumanEval"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for binary classification (default: 0.7)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Checkpoint path
    if args.checkpoint is None:
        checkpoint_path = Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Run accuracy evaluation
    try:
        results = evaluate_accuracy(
            checkpoint_path=str(checkpoint_path),
            device=device,
            confidence_threshold=args.threshold
        )
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")
    
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
