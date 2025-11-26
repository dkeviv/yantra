// PyO3 Bridge Module
// Provides bidirectional Rust ↔ Python communication for GraphSAGE model integration
//
// Architecture:
// - Rust (GNN, features) → Python (GraphSAGE model) → Rust (predictions)
// - Target: <2ms overhead for bridge calls
// - Handles Python exceptions gracefully
// - Thread-safe Python GIL management

pub mod pyo3_bridge;

#[cfg(test)]
mod bench;

pub use pyo3_bridge::{PythonBridge, FeatureVector, ModelPrediction};
