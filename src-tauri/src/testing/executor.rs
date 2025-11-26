// File: src-tauri/src/testing/executor.rs
// Purpose: Simple pytest executor with JSON output parsing for GraphSAGE learning
// Last Updated: November 25, 2025
//
// This module implements streamlined test execution for the learning loop:
// - Execute pytest with JSON report plugin
// - Parse pytest-json-report output (cleaner than XML)
// - Extract pass/fail counts for success-only learning
// - Fast execution (<100ms overhead)
//
// Usage in learning loop:
// 1. Generate code with GraphSAGE/LLM
// 2. Generate tests with LLM
// 3. Execute tests → get TestExecutionResult
// 4. If success=true → train GraphSAGE on (code, context)
// 5. If success=false → don't learn (or learn as negative example)

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Result of test execution (simplified for learning loop)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionResult {
    /// Overall success (all tests passed)
    pub success: bool,
    /// Number of tests that passed
    pub passed: usize,
    /// Number of tests that failed
    pub failed: usize,
    /// Number of tests that were skipped
    pub skipped: usize,
    /// Number of tests with errors
    pub errors: usize,
    /// Total number of tests
    pub total: usize,
    /// Execution time in seconds
    pub duration_seconds: f64,
    /// Pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Test failures (for debugging)
    pub failures: Vec<TestFailureInfo>,
    /// Coverage percentage (if available)
    pub coverage_percent: Option<f64>,
}

/// Individual test failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFailureInfo {
    pub test_name: String,
    pub error_type: String,
    pub error_message: String,
}

impl TestExecutionResult {
    /// Check if tests passed well enough to learn from
    /// Threshold: >90% pass rate
    pub fn is_learnable(&self) -> bool {
        self.pass_rate >= 0.9
    }

    /// Get quality score for confidence calculation
    pub fn quality_score(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.pass_rate
        }
    }
}

/// Pytest executor with JSON report output
pub struct PytestExecutor {
    workspace_path: PathBuf,
    python_command: String,
}

impl PytestExecutor {
    /// Create new pytest executor
    pub fn new(workspace_path: PathBuf) -> Self {
        PytestExecutor {
            workspace_path,
            python_command: "python".to_string(),
        }
    }

    /// Set custom Python command (e.g., "python3", "conda run -n myenv python")
    pub fn with_python_command(mut self, command: String) -> Self {
        self.python_command = command;
        self
    }

    /// Execute pytest and return results
    /// 
    /// # Arguments
    /// * `test_file` - Path to test file (relative to workspace)
    /// * `_timeout_seconds` - Maximum execution time (default: 300s, currently not enforced)
    /// 
    /// # Returns
    /// TestExecutionResult with pass/fail counts
    pub fn execute_tests(
        &self,
        test_file: &Path,
        _timeout_seconds: Option<u64>,
    ) -> Result<TestExecutionResult, String> {
        let start = Instant::now();
        
        // Build pytest command with JSON report
        let json_output_path = self.workspace_path.join(".pytest_result.json");
        
        // Build pytest arguments
        let mut args = vec![
            "-m".to_string(),
            "pytest".to_string(),
            "-v".to_string(),
            "--tb=short".to_string(),
            "--json-report".to_string(),
            format!("--json-report-file={}", json_output_path.display()),
        ];
        
        // Add test file path
        let test_path = self.workspace_path.join(test_file);
        args.push(test_path.to_string_lossy().to_string());
        
        // Execute pytest
        let output = Command::new(&self.python_command)
            .args(&args)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute pytest: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        
        // Parse JSON output if available
        if json_output_path.exists() {
            let json_content = std::fs::read_to_string(&json_output_path)
                .map_err(|e| format!("Failed to read pytest JSON output: {}", e))?;
            
            // Clean up JSON file
            let _ = std::fs::remove_file(&json_output_path);
            
            return self.parse_json_report(&json_content, duration);
        }
        
        // Fallback: Parse stdout/stderr
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{}\n{}", stdout, stderr);
        
        self.parse_pytest_output(&combined_output, duration, output.status.success())
    }

    /// Execute pytest with coverage
    pub fn execute_tests_with_coverage(
        &self,
        test_file: &Path,
        _timeout_seconds: Option<u64>,
    ) -> Result<TestExecutionResult, String> {
        let start = Instant::now();
        
        // Build pytest command with JSON report and coverage
        let json_output_path = self.workspace_path.join(".pytest_result.json");
        let coverage_json_path = self.workspace_path.join(".coverage.json");
        
        // Build pytest arguments
        let mut args = vec![
            "-m".to_string(),
            "pytest".to_string(),
            "-v".to_string(),
            "--tb=short".to_string(),
            "--json-report".to_string(),
            format!("--json-report-file={}", json_output_path.display()),
            "--cov=.".to_string(),
            "--cov-report=json".to_string(),
        ];
        
        // Add test file path
        let test_path = self.workspace_path.join(test_file);
        args.push(test_path.to_string_lossy().to_string());
        
        // Execute pytest
        let output = Command::new(&self.python_command)
            .args(&args)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute pytest with coverage: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        
        // Parse JSON output
        let mut result = if json_output_path.exists() {
            let json_content = std::fs::read_to_string(&json_output_path)
                .map_err(|e| format!("Failed to read pytest JSON output: {}", e))?;
            
            // Clean up JSON file
            let _ = std::fs::remove_file(&json_output_path);
            
            self.parse_json_report(&json_content, duration)?
        } else {
            // Fallback: Parse stdout/stderr
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined_output = format!("{}\n{}", stdout, stderr);
            
            self.parse_pytest_output(&combined_output, duration, output.status.success())?
        };
        
        // Parse coverage if available
        if coverage_json_path.exists() {
            if let Ok(coverage_content) = std::fs::read_to_string(&coverage_json_path) {
                if let Ok(coverage) = self.parse_coverage_json(&coverage_content) {
                    result.coverage_percent = Some(coverage);
                }
            }
            // Clean up coverage file
            let _ = std::fs::remove_file(&coverage_json_path);
        }
        
        Ok(result)
    }

    /// Parse pytest-json-report output
    fn parse_json_report(&self, json_content: &str, duration: f64) -> Result<TestExecutionResult, String> {
        #[derive(Deserialize)]
        struct PytestJsonReport {
            summary: PytestSummary,
            tests: Vec<PytestTest>,
        }

        #[derive(Deserialize)]
        struct PytestSummary {
            total: usize,
            passed: Option<usize>,
            failed: Option<usize>,
            skipped: Option<usize>,
            error: Option<usize>,
        }

        #[derive(Deserialize)]
        struct PytestTest {
            nodeid: String,
            outcome: String,
            #[serde(default)]
            call: Option<PytestCall>,
        }

        #[derive(Deserialize)]
        struct PytestCall {
            longrepr: Option<String>,
        }

        let report: PytestJsonReport = serde_json::from_str(json_content)
            .map_err(|e| format!("Failed to parse pytest JSON: {}", e))?;

        let passed = report.summary.passed.unwrap_or(0);
        let failed = report.summary.failed.unwrap_or(0);
        let skipped = report.summary.skipped.unwrap_or(0);
        let errors = report.summary.error.unwrap_or(0);
        let total = report.summary.total;

        let success = failed == 0 && errors == 0;
        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };

        // Extract failure information
        let mut failures = Vec::new();
        for test in report.tests {
            if test.outcome == "failed" || test.outcome == "error" {
                let error_message = test
                    .call
                    .and_then(|c| c.longrepr)
                    .unwrap_or_else(|| "Unknown error".to_string());

                // Extract error type from message
                let error_type = if error_message.contains("AssertionError") {
                    "AssertionError"
                } else if error_message.contains("ImportError") {
                    "ImportError"
                } else if error_message.contains("TypeError") {
                    "TypeError"
                } else if error_message.contains("ValueError") {
                    "ValueError"
                } else {
                    "Unknown"
                }
                .to_string();

                failures.push(TestFailureInfo {
                    test_name: test.nodeid,
                    error_type,
                    error_message: error_message.lines().next().unwrap_or("").to_string(),
                });
            }
        }

        Ok(TestExecutionResult {
            success,
            passed,
            failed,
            skipped,
            errors,
            total,
            duration_seconds: duration,
            pass_rate,
            failures,
            coverage_percent: None,
        })
    }

    /// Parse pytest stdout/stderr output (fallback)
    fn parse_pytest_output(
        &self,
        output: &str,
        duration: f64,
        _exit_success: bool,
    ) -> Result<TestExecutionResult, String> {
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        let mut errors = 0;

        // Parse summary line (e.g., "5 passed, 2 failed in 3.45s")
        for line in output.lines() {
            if line.contains("passed") || line.contains("failed") || line.contains("error") {
                if let Some(count) = extract_count(line, "passed") {
                    passed = count;
                }
                if let Some(count) = extract_count(line, "failed") {
                    failed = count;
                }
                if let Some(count) = extract_count(line, "skipped") {
                    skipped = count;
                }
                if let Some(count) = extract_count(line, "error") {
                    errors = count;
                }
            }
        }

        let total = passed + failed + skipped + errors;
        let success = failed == 0 && errors == 0;
        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };

        Ok(TestExecutionResult {
            success,
            passed,
            failed,
            skipped,
            errors,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
        })
    }

    /// Parse coverage.json output
    fn parse_coverage_json(&self, json_content: &str) -> Result<f64, String> {
        #[derive(Deserialize)]
        struct CoverageReport {
            totals: CoverageTotals,
        }

        #[derive(Deserialize)]
        struct CoverageTotals {
            percent_covered: f64,
        }

        let report: CoverageReport = serde_json::from_str(json_content)
            .map_err(|e| format!("Failed to parse coverage JSON: {}", e))?;

        Ok(report.totals.percent_covered)
    }
}

/// Extract count from pytest summary line
fn extract_count(line: &str, keyword: &str) -> Option<usize> {
    let words: Vec<&str> = line.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if word.contains(keyword) && i > 0 {
            if let Ok(count) = words[i - 1].parse::<usize>() {
                return Some(count);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_count() {
        assert_eq!(extract_count("5 passed in 3.45s", "passed"), Some(5));
        assert_eq!(extract_count("2 failed, 3 passed in 1.23s", "failed"), Some(2));
        assert_eq!(extract_count("2 failed, 3 passed in 1.23s", "passed"), Some(3));
    }

    #[test]
    fn test_execution_result_quality() {
        let result = TestExecutionResult {
            success: true,
            passed: 10,
            failed: 0,
            skipped: 0,
            errors: 0,
            total: 10,
            duration_seconds: 1.5,
            pass_rate: 1.0,
            failures: Vec::new(),
            coverage_percent: Some(95.0),
        };

        assert!(result.is_learnable());
        assert_eq!(result.quality_score(), 1.0);
    }

    #[test]
    fn test_execution_result_not_learnable() {
        let result = TestExecutionResult {
            success: false,
            passed: 5,
            failed: 5,
            skipped: 0,
            errors: 0,
            total: 10,
            duration_seconds: 1.5,
            pass_rate: 0.5,
            failures: Vec::new(),
            coverage_percent: None,
        };

        assert!(!result.is_learnable());
        assert_eq!(result.quality_score(), 0.5);
    }

    #[test]
    fn test_parse_pytest_output_success() {
        let executor = PytestExecutor::new(PathBuf::from("/tmp"));
        let output = "test_file.py::test_one PASSED\ntest_file.py::test_two PASSED\n\n====== 2 passed in 0.12s ======";
        
        let result = executor.parse_pytest_output(output, 0.12, true).unwrap();
        
        assert!(result.success);
        assert_eq!(result.passed, 2);
        assert_eq!(result.failed, 0);
        assert_eq!(result.total, 2);
        assert_eq!(result.pass_rate, 1.0);
    }

    #[test]
    fn test_parse_pytest_output_failure() {
        let executor = PytestExecutor::new(PathBuf::from("/tmp"));
        let output = "test_file.py::test_one PASSED\ntest_file.py::test_two FAILED\n\n====== 1 passed, 1 failed in 0.15s ======";
        
        let result = executor.parse_pytest_output(output, 0.15, false).unwrap();
        
        assert!(!result.success);
        assert_eq!(result.passed, 1);
        assert_eq!(result.failed, 1);
        assert_eq!(result.total, 2);
        assert_eq!(result.pass_rate, 0.5);
    }
}
