#!/usr/bin/env python3
"""
GraphSAGE Model Inference Benchmark

Measures:
- Model loading time
- Single prediction latency
- Batch prediction throughput
- Memory usage

Usage:
    python src-python/training/benchmark.py
"""

import sys
import time
import statistics
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.graphsage import load_model_for_inference
from yantra_bridge import predict


def benchmark_model_loading():
    """Benchmark model loading time"""
    print("=" * 60)
    print("1. Model Loading Benchmark")
    print("=" * 60)
    
    checkpoint_path = Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ No trained model found at: {checkpoint_path}")
        print("   Run training first: python src-python/training/train.py")
        return None
    
    # Cold load
    start = time.time()
    model = load_model_for_inference(str(checkpoint_path), device='cpu')
    cold_load_time = time.time() - start
    
    print(f"\n✓ Model loaded successfully")
    print(f"  Cold load time: {cold_load_time*1000:.2f}ms")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")
    
    return model


def benchmark_single_prediction(model):
    """Benchmark single prediction latency"""
    print("\n" + "=" * 60)
    print("2. Single Prediction Latency")
    print("=" * 60)
    
    # Generate random features
    features = torch.randn(1, 978)
    
    # Warmup (compile/optimize)
    print("\nWarming up...")
    for _ in range(10):
        _ = model(features)
    
    # Benchmark
    print("Running 1000 predictions...")
    times = []
    for _ in range(1000):
        start = time.time()
        with torch.no_grad():
            predictions = model(features)
        times.append((time.time() - start) * 1000)
    
    print(f"\n✓ Single Prediction Latency:")
    print(f"  Mean:   {statistics.mean(times):.3f}ms")
    print(f"  Median: {statistics.median(times):.3f}ms")
    print(f"  P95:    {sorted(times)[int(len(times)*0.95)]:.3f}ms")
    print(f"  P99:    {sorted(times)[int(len(times)*0.99)]:.3f}ms")
    print(f"  Min:    {min(times):.3f}ms")
    print(f"  Max:    {max(times):.3f}ms")
    
    target_met = statistics.median(times) < 10.0
    print(f"\n  Target (<10ms): {'✅ PASS' if target_met else '❌ FAIL'}")


def benchmark_batch_prediction(model):
    """Benchmark batch prediction throughput"""
    print("\n" + "=" * 60)
    print("3. Batch Prediction Throughput")
    print("=" * 60)
    
    batch_sizes = [1, 8, 16, 32, 64]
    
    print("\nBatch Size | Throughput (predictions/sec) | Latency per prediction")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        features = torch.randn(batch_size, 978)
        
        # Warmup
        for _ in range(5):
            _ = model(features)
        
        # Benchmark
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(features)
        elapsed = time.time() - start
        
        throughput = (batch_size * num_iterations) / elapsed
        latency_per_pred = (elapsed / (batch_size * num_iterations)) * 1000
        
        print(f"{batch_size:10} | {throughput:28.1f} | {latency_per_pred:.3f}ms")


def benchmark_python_bridge():
    """Benchmark Python bridge (Rust ↔ Python)"""
    print("\n" + "=" * 60)
    print("4. Python Bridge (yantra_bridge.py)")
    print("=" * 60)
    
    features = [0.5] * 978
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        _ = predict(features)
    
    # Benchmark
    print("Running 1000 predictions through bridge...")
    times = []
    for _ in range(1000):
        start = time.time()
        result = predict(features)
        times.append((time.time() - start) * 1000)
    
    print(f"\n✓ Bridge Prediction Latency:")
    print(f"  Mean:   {statistics.mean(times):.3f}ms")
    print(f"  Median: {statistics.median(times):.3f}ms")
    print(f"  P95:    {sorted(times)[int(len(times)*0.95)]:.3f}ms")
    print(f"  P99:    {sorted(times)[int(len(times)*0.99)]:.3f}ms")
    
    # Bridge overhead (compared to direct model call)
    # First prediction includes model loading
    direct_latency = 0.21  # From previous benchmark
    bridge_overhead = statistics.median(times) - direct_latency
    
    print(f"\n  Bridge overhead: ~{bridge_overhead:.3f}ms")
    print(f"  (includes tensor conversion + dict creation)")


def benchmark_mps_vs_cpu():
    """Compare MPS (Apple Silicon GPU) vs CPU performance"""
    print("\n" + "=" * 60)
    print("5. MPS vs CPU Performance")
    print("=" * 60)
    
    checkpoint_path = Path.home() / ".yantra" / "checkpoints" / "graphsage" / "best_model.pt"
    
    if not torch.backends.mps.is_available():
        print("\n⚠️  MPS not available, skipping GPU benchmark")
        return
    
    print("\nLoading models...")
    model_cpu = load_model_for_inference(str(checkpoint_path), device='cpu')
    model_mps = load_model_for_inference(str(checkpoint_path), device='mps')
    
    batch_size = 32
    features_cpu = torch.randn(batch_size, 978)
    features_mps = features_cpu.to('mps')
    
    # Warmup
    for _ in range(10):
        _ = model_cpu(features_cpu)
        _ = model_mps(features_mps)
    
    # CPU benchmark
    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_cpu(features_cpu)
    cpu_time = time.time() - start
    
    # MPS benchmark
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_mps(features_mps)
    mps_time = time.time() - start
    
    cpu_throughput = (batch_size * num_iterations) / cpu_time
    mps_throughput = (batch_size * num_iterations) / mps_time
    speedup = cpu_time / mps_time
    
    print(f"\n✓ Performance Comparison (batch_size={batch_size}):")
    print(f"  CPU: {cpu_throughput:.1f} predictions/sec")
    print(f"  MPS: {mps_throughput:.1f} predictions/sec")
    print(f"  Speedup: {speedup:.2f}x faster on MPS")


def main():
    print("\n" + "=" * 60)
    print("GraphSAGE Model Inference Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()
    
    # Run benchmarks
    model = benchmark_model_loading()
    
    if model is not None:
        benchmark_single_prediction(model)
        benchmark_batch_prediction(model)
        benchmark_python_bridge()
        benchmark_mps_vs_cpu()
    
    print("\n" + "=" * 60)
    print("✓ Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
