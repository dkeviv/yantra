// Integration tests module
// Links all integration test files

pub mod execution_tests;
pub mod packaging_tests;
pub mod deployment_tests;

// Re-export commonly used types for tests
pub use tempfile::TempDir;
pub use std::path::PathBuf;
pub use std::fs;
