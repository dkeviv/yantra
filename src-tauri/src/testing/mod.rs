// File: src-tauri/src/testing/mod.rs
// Purpose: Testing module for test generation and execution
// Last Updated: December 3, 2025

pub mod generator;
pub mod runner;
pub mod executor;
pub mod retry;
pub mod executor_js;
pub mod generator_js;

// Re-export key types from executor (for GraphSAGE learning loop)
pub use executor::{PytestExecutor, TestExecutionResult, TestFailureInfo};

// Re-export key types from retry (for autonomous retry logic)
pub use retry::{RetryExecutor, RetryResult, RetryStrategy};

// Re-export JavaScript/TypeScript testing types
pub use executor_js::{JestExecutor, JestExecutionResult, JestFailureInfo};
pub use generator_js::{JestGenerator, JestGeneratorConfig};

// Re-export key types from runner (legacy test runner)
#[allow(unused_imports)]
pub use runner::{TestRunner, TestResult, FailureType};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGenerationRequest {
    pub code: String,
    pub language: String,
    pub file_path: String,
    pub coverage_target: f32, // e.g., 0.9 for 90%
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGenerationResponse {
    pub tests: String,
    pub test_count: usize,
    pub estimated_coverage: f32,
    pub fixtures: Vec<String>,
}

// Legacy types (kept for backwards compatibility, not used in new executor)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyTestExecutionRequest {
    pub test_file_path: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyTestFailure {
    pub test_name: String,
    pub message: String,
    pub traceback: String,
}
