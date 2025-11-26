/// Benchmark tests for PyO3 bridge overhead
/// Target: <2ms overhead for Rust → Python → Rust roundtrip

#[cfg(test)]
mod bench_tests {
    use super::super::pyo3_bridge::{FeatureVector, PythonBridge};
    use std::time::Instant;

    #[test]
    fn benchmark_bridge_overhead() {
        let bridge = PythonBridge::new();
        bridge.initialize().expect("Failed to initialize bridge");

        // Create a sample 978-dimensional feature vector
        let mut features = vec![0.5f32; 978];
        // Add some variation
        for i in 0..978 {
            features[i] = (i as f32) / 978.0;
        }
        let feature_vector = FeatureVector::new(features).expect("Failed to create feature vector");

        // Warm-up call
        let _ = bridge.predict(&feature_vector);

        // Benchmark: 100 calls to measure average overhead
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            bridge.predict(&feature_vector).expect("Prediction failed");
        }
        
        let duration = start.elapsed();
        let avg_ms = duration.as_millis() as f64 / iterations as f64;
        
        println!("\n=== PyO3 Bridge Benchmark ===");
        println!("Total time for {} calls: {:?}", iterations, duration);
        println!("Average overhead per call: {:.3} ms", avg_ms);
        println!("Target: <2ms");
        
        if avg_ms < 2.0 {
            println!("✓ PASSED: {:.1}x better than target!", 2.0 / avg_ms);
        } else {
            println!("✗ FAILED: {:.1}x slower than target", avg_ms / 2.0);
        }
        
        // Assert that we meet the <2ms target
        assert!(
            avg_ms < 2.0,
            "Bridge overhead {:.3}ms exceeds 2ms target",
            avg_ms
        );
    }

    #[test]
    fn benchmark_echo_call() {
        let bridge = PythonBridge::new();
        bridge.initialize().expect("Failed to initialize bridge");

        // Benchmark simple echo call (minimal Python interaction)
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            bridge.test_echo().expect("Echo failed");
        }
        
        let duration = start.elapsed();
        let avg_us = duration.as_micros() as f64 / iterations as f64;
        
        println!("\n=== Echo Call Benchmark ===");
        println!("Total time for {} calls: {:?}", iterations, duration);
        println!("Average time per call: {:.1} µs", avg_us);
        
        // Echo should be very fast (<100µs)
        assert!(
            avg_us < 100.0,
            "Echo call overhead {:.1}µs exceeds 100µs",
            avg_us
        );
    }

    #[test]
    fn benchmark_feature_conversion() {
        use pyo3::Python;
        
        // Benchmark feature vector conversion to Python
        let mut features = vec![0.5f32; 978];
        for i in 0..978 {
            features[i] = (i as f32) / 978.0;
        }
        let feature_vector = FeatureVector::new(features).expect("Failed to create feature vector");

        let iterations = 10000;
        let start = Instant::now();
        
        Python::with_gil(|py| {
            for _ in 0..iterations {
                let _ = feature_vector.to_python(py).expect("Conversion failed");
            }
        });
        
        let duration = start.elapsed();
        let avg_us = duration.as_micros() as f64 / iterations as f64;
        
        println!("\n=== Feature Vector Conversion Benchmark ===");
        println!("Total time for {} conversions: {:?}", iterations, duration);
        println!("Average time per conversion: {:.2} µs", avg_us);
        
        // Conversion should be reasonable (<50µs for 978 floats)
        assert!(
            avg_us < 50.0,
            "Feature conversion {:.2}µs exceeds 50µs target",
            avg_us
        );
    }
}
