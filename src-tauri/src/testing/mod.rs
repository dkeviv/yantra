// File: src-tauri/src/testing/mod.rs
// Purpose: Testing module for test generation and execution
// Last Updated: November 22, 2025

pub mod generator;
pub mod runner;

// Re-export key types
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionRequest {
    pub test_file_path: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionResult {
    pub success: bool,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration_seconds: f64,
    pub failures: Vec<TestFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFailure {
    pub test_name: String,
    pub message: String,
    pub traceback: String,
}
