#!/usr/bin/env python3
"""
Benchmark GraphSAGE inference latency.

This script measures the inference performance of the trained GraphSAGE model
to validate it meets the <10ms per prediction target for production use.

Metrics reported:
- Average inference time
- P50 (median), P95, P99 percentiles
- Min/Max times
- Throughput (predictions/second)

Usage:
    python scripts/benchmark_inference.py [--iterations 1000] [--warmup 10]
"""

import sys
import time
from pathlib import Path
import argparse
import statistics

# Add src-python to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src-python"))

try:
    import torch
    from model.graphsage import GraphSAGEModel, load_model_for_inference
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("Make sure you've installed PyTorch and the model is available.")
    sys.exit(1)


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def benchmark_inference(
    model: GraphSAGEModel,
    device: str,
    num_iterations: int = 1000,
    num_warmup: int = 10,
    batch_size: int = 1,
) -> dict:
    """
    Benchmark model inference latency.

    Args:
        model: GraphSAGEModel to benchmark
        device: Device to run on ('cpu', 'cuda', or 'mps')
        num_iterations: Number of inference iterations to run
        num_warmup: Number of warmup iterations (not counted)
        batch_size: Batch size for inference

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"GraphSAGE Inference Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device.upper()}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    print(f"{'='*60}\n")

    model.eval()
    model.to(device)

    # Create random input features (978 dims as expected by model)
    input_dim = 978
    features = torch.randn(batch_size, input_dim, device=device)

    # Warmup phase
    print("Running warmup...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(features)

    # Synchronize before benchmarking (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    # Benchmark phase
    print(f"Running {num_iterations} iterations...")
    latencies = []

    with torch.no_grad():
        for i in range(num_iterations):
            start = time.perf_counter()
            _ = model(features)

            # Synchronize to ensure computation is complete
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            latency_ms = (end - start) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)

            # Progress indicator every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations...")

    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = percentile(latencies, 50)
    p95_latency = percentile(latencies, 95)
    p99_latency = percentile(latencies, 99)
    min_latency = min(latencies)
    max_latency = max(latencies)
    throughput = 1000 / avg_latency  # predictions per second

    results = {
        "device": device,
        "batch_size": batch_size,
        "iterations": num_iterations,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_per_sec": throughput,
    }

    return results


def print_results(results: dict, target_ms: float = 10.0):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Device:              {results['device'].upper()}")
    print(f"Batch size:          {results['batch_size']}")
    print(f"Iterations:          {results['iterations']}")
    print(f"{'-'*60}")
    print("Latency Metrics:")
    print(f"  Average:           {results['avg_latency_ms']:.3f} ms")
    print(f"  Median (P50):      {results['median_latency_ms']:.3f} ms")
    print(f"  P95:               {results['p95_latency_ms']:.3f} ms")
    print(f"  P99:               {results['p99_latency_ms']:.3f} ms")
    print(f"  Min:               {results['min_latency_ms']:.3f} ms")
    print(f"  Max:               {results['max_latency_ms']:.3f} ms")
    print(f"{'-'*60}")
    print(f"Throughput:          {results['throughput_per_sec']:.1f} predictions/sec")
    print(f"{'='*60}")

    # Check if target is met
    if results['avg_latency_ms'] <= target_ms:
        print(f"✅ SUCCESS: Average latency ({results['avg_latency_ms']:.3f} ms) meets target (<{target_ms} ms)")
    else:
        print(f"⚠️  WARNING: Average latency ({results['avg_latency_ms']:.3f} ms) exceeds target (<{target_ms} ms)")

    if results['p95_latency_ms'] <= target_ms:
        print(f"✅ SUCCESS: P95 latency ({results['p95_latency_ms']:.3f} ms) meets target (<{target_ms} ms)")
    else:
        print(f"⚠️  WARNING: P95 latency ({results['p95_latency_ms']:.3f} ms) exceeds target (<{target_ms} ms)")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GraphSAGE inference latency"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of inference iterations to run (default: 1000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: ~/.yantra/checkpoints/graphsage/best_model.pt)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=10.0,
        help="Target latency in milliseconds (default: 10.0)",
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
        checkpoint_path = (
            Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
        )
    else:
        checkpoint_path = Path(args.checkpoint)

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python src-python/training/train.py")
        sys.exit(1)

    print(f"Loading model from: {checkpoint_path}")

    try:
        # Load trained model
        model = load_model_for_inference(str(checkpoint_path), device=device)
        print(f"✅ Successfully loaded trained model")
        print(f"   Model size: {model.get_model_size_mb():.2f} MB")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run benchmark
    try:
        results = benchmark_inference(
            model=model,
            device=device,
            num_iterations=args.iterations,
            num_warmup=args.warmup,
            batch_size=args.batch_size,
        )

        # Print results
        print_results(results, target_ms=args.target)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
