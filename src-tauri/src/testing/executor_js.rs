// File: src-tauri/src/testing/executor_js.rs
// Purpose: Jest executor for JavaScript/TypeScript testing
// Dependencies: Node.js, Jest
// Last Updated: December 3, 2025
//
// Implements Jest test execution for JavaScript/TypeScript projects:
// - Execute Jest with JSON reporters
// - Parse Jest JSON output
// - Extract pass/fail counts
// - Coverage reporting
// - Fast execution with parallel tests
//
// Usage:
// 1. Generate JavaScript/TypeScript code
// 2. Generate Jest tests
// 3. Execute tests → get TestExecutionResult
// 4. Auto-retry with fixes if failures

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Result of Jest test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JestExecutionResult {
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
    pub failures: Vec<JestFailureInfo>,
    /// Coverage percentage (if available)
    pub coverage_percent: Option<f64>,
}

/// Individual Jest test failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JestFailureInfo {
    pub test_name: String,
    pub suite_name: String,
    pub error_type: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
}

impl JestExecutionResult {
    /// Check if tests passed well enough to learn from
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

    /// Convert to generic TestExecutionResult for compatibility
    pub fn to_generic(&self) -> super::executor::TestExecutionResult {
        super::executor::TestExecutionResult {
            success: self.success,
            passed: self.passed,
            failed: self.failed,
            skipped: self.skipped,
            errors: self.errors,
            total: self.total,
            duration_seconds: self.duration_seconds,
            pass_rate: self.pass_rate,
            failures: self.failures.iter().map(|f| {
                super::executor::TestFailureInfo {
                    test_name: f.test_name.clone(),
                    error_type: f.error_type.clone(),
                    error_message: f.error_message.clone(),
                }
            }).collect(),
            coverage_percent: self.coverage_percent,
        }
    }
}

/// Jest executor for JavaScript/TypeScript projects
pub struct JestExecutor {
    workspace_path: PathBuf,
    node_command: String,
    use_npm: bool, // true for npm, false for yarn
}

impl JestExecutor {
    /// Create new Jest executor
    pub fn new(workspace_path: PathBuf) -> Self {
        JestExecutor {
            workspace_path,
            node_command: "node".to_string(),
            use_npm: true,
        }
    }

    /// Set custom Node command
    pub fn with_node_command(mut self, command: String) -> Self {
        self.node_command = command;
        self
    }

    /// Use Yarn instead of npm
    pub fn with_yarn(mut self) -> Self {
        self.use_npm = false;
        self
    }

    /// Execute Jest and return results
    /// 
    /// # Arguments
    /// * `test_pattern` - Test file pattern (e.g., "*.test.js" or specific file)
    /// * `timeout_seconds` - Maximum execution time (default: 300s)
    /// 
    /// # Returns
    /// JestExecutionResult with test outcomes
    pub async fn execute(
        &self,
        test_pattern: Option<&str>,
        timeout_seconds: Option<u64>,
    ) -> Result<JestExecutionResult, String> {
        let start = Instant::now();
        let timeout = timeout_seconds.unwrap_or(300);

        // Build Jest command
        let mut cmd = if self.use_npm {
            let mut c = Command::new("npm");
            c.arg("test");
            c
        } else {
            let mut c = Command::new("yarn");
            c.arg("test");
            c
        };

        // Add Jest arguments
        cmd.arg("--")
            .arg("--json") // JSON output
            .arg("--no-coverage") // Skip coverage for speed (add --coverage to enable)
            .arg("--maxWorkers=4"); // Parallel execution

        if let Some(pattern) = test_pattern {
            cmd.arg(pattern);
        }

        cmd.current_dir(&self.workspace_path);

        // Execute Jest
        let output = tokio::time::timeout(
            tokio::time::Duration::from_secs(timeout),
            tokio::task::spawn_blocking(move || cmd.output()),
        )
        .await
        .map_err(|_| "Jest execution timeout".to_string())?
        .map_err(|e| format!("Failed to spawn Jest process: {}", e))?
        .map_err(|e| format!("Jest execution failed: {}", e))?;

        let duration = start.elapsed().as_secs_f64();

        // Parse Jest JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Jest writes JSON to stdout
        let result = self.parse_jest_output(&stdout)?;

        // If parsing failed, check stderr
        if !output.status.success() && result.total == 0 {
            return Err(format!(
                "Jest failed with no test results. stderr: {}",
                stderr
            ));
        }

        Ok(JestExecutionResult {
            duration_seconds: duration,
            ..result
        })
    }

    /// Execute Jest with coverage
    pub async fn execute_with_coverage(
        &self,
        test_pattern: Option<&str>,
    ) -> Result<JestExecutionResult, String> {
        let start = Instant::now();

        // Build Jest command with coverage
        let mut cmd = if self.use_npm {
            let mut c = Command::new("npm");
            c.arg("test");
            c
        } else {
            let mut c = Command::new("yarn");
            c.arg("test");
            c
        };

        cmd.arg("--")
            .arg("--json")
            .arg("--coverage")
            .arg("--coverageReporters=json-summary")
            .arg("--maxWorkers=4");

        if let Some(pattern) = test_pattern {
            cmd.arg(pattern);
        }

        cmd.current_dir(&self.workspace_path);

        // Execute Jest
        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute Jest: {}", e))?;

        let duration = start.elapsed().as_secs_f64();

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut result = self.parse_jest_output(&stdout)?;

        // Parse coverage from coverage-summary.json
        let coverage_file = self.workspace_path.join("coverage/coverage-summary.json");
        if coverage_file.exists() {
            if let Ok(coverage_data) = std::fs::read_to_string(&coverage_file) {
                result.coverage_percent = self.parse_coverage(&coverage_data);
            }
        }

        Ok(JestExecutionResult {
            duration_seconds: duration,
            ..result
        })
    }

    /// Parse Jest JSON output
    fn parse_jest_output(&self, json_output: &str) -> Result<JestExecutionResult, String> {
        // Jest JSON output structure:
        // {
        //   "numTotalTests": 10,
        //   "numPassedTests": 8,
        //   "numFailedTests": 2,
        //   "numPendingTests": 0,
        //   "testResults": [...]
        // }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct JestOutput {
            num_total_tests: usize,
            num_passed_tests: usize,
            num_failed_tests: usize,
            num_pending_tests: usize,
            test_results: Vec<JestTestResult>,
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct JestTestResult {
            assertion_results: Vec<JestAssertion>,
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct JestAssertion {
            full_name: String,
            title: String,
            status: String,
            failure_messages: Vec<String>,
        }

        let parsed: JestOutput = serde_json::from_str(json_output)
            .map_err(|e| format!("Failed to parse Jest output: {}", e))?;

        let mut failures = Vec::new();

        // Extract failures
        for test_result in &parsed.test_results {
            for assertion in &test_result.assertion_results {
                if assertion.status == "failed" {
                    let error_message = assertion.failure_messages.join("\n");
                    
                    // Extract error type from message (e.g., "TypeError:", "ReferenceError:")
                    let error_type = error_message
                        .lines()
                        .find(|line| line.contains("Error"))
                        .and_then(|line| {
                            if let Some(pos) = line.find("Error") {
                                Some(line[..pos + 5].trim().to_string())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| "Error".to_string());

                    failures.push(JestFailureInfo {
                        test_name: assertion.title.clone(),
                        suite_name: assertion.full_name.clone(),
                        error_type,
                        error_message: error_message.clone(),
                        stack_trace: Some(error_message),
                    });
                }
            }
        }

        let total = parsed.num_total_tests;
        let passed = parsed.num_passed_tests;
        let failed = parsed.num_failed_tests;
        let skipped = parsed.num_pending_tests;

        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };

        Ok(JestExecutionResult {
            success: failed == 0,
            passed,
            failed,
            skipped,
            errors: 0, // Jest doesn't separate errors from failures
            total,
            duration_seconds: 0.0, // Will be filled by caller
            pass_rate,
            failures,
            coverage_percent: None,
        })
    }

    /// Parse coverage from coverage-summary.json
    fn parse_coverage(&self, coverage_json: &str) -> Option<f64> {
        #[derive(Deserialize)]
        struct CoverageSummary {
            total: CoverageMetrics,
        }

        #[derive(Deserialize)]
        struct CoverageMetrics {
            lines: CoverageDetail,
        }

        #[derive(Deserialize)]
        struct CoverageDetail {
            pct: f64,
        }

        serde_json::from_str::<CoverageSummary>(coverage_json)
            .ok()
            .map(|summary| summary.total.lines.pct)
    }

    /// Check if Jest is installed in the project
    pub fn is_jest_available(&self) -> bool {
        let package_json = self.workspace_path.join("package.json");
        if !package_json.exists() {
            return false;
        }

        // Check if jest is in dependencies or devDependencies
        if let Ok(content) = std::fs::read_to_string(&package_json) {
            content.contains("\"jest\"")
        } else {
            false
        }
    }

    /// Install Jest if not present
    pub async fn ensure_jest_installed(&self) -> Result<(), String> {
        if self.is_jest_available() {
            return Ok(());
        }

        println!("📦 Installing Jest...");

        let mut cmd = if self.use_npm {
            let mut c = Command::new("npm");
            c.arg("install")
                .arg("--save-dev")
                .arg("jest")
                .arg("@types/jest");
            c
        } else {
            let mut c = Command::new("yarn");
            c.arg("add")
                .arg("--dev")
                .arg("jest")
                .arg("@types/jest");
            c
        };

        cmd.current_dir(&self.workspace_path);

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to install Jest: {}", e))?;

        if !output.status.success() {
            return Err(format!(
                "Jest installation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        println!("✅ Jest installed successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jest_executor_creation() {
        let executor = JestExecutor::new(PathBuf::from("/tmp/test"));
        assert_eq!(executor.workspace_path, PathBuf::from("/tmp/test"));
        assert!(executor.use_npm);
    }

    #[test]
    fn test_parse_jest_output() {
        let executor = JestExecutor::new(PathBuf::from("/tmp/test"));
        
        let json_output = r#"{
            "numTotalTests": 3,
            "numPassedTests": 2,
            "numFailedTests": 1,
            "numPendingTests": 0,
            "testResults": [
                {
                    "assertionResults": [
                        {
                            "fullName": "Calculator add",
                            "title": "should add two numbers",
                            "status": "passed",
                            "failureMessages": []
                        },
                        {
                            "fullName": "Calculator subtract",
                            "title": "should subtract two numbers",
                            "status": "failed",
                            "failureMessages": ["TypeError: Cannot read property 'x' of undefined"]
                        }
                    ]
                }
            ]
        }"#;

        let result = executor.parse_jest_output(json_output).unwrap();
        
        assert_eq!(result.total, 3);
        assert_eq!(result.passed, 2);
        assert_eq!(result.failed, 1);
        assert!(!result.success);
        assert_eq!(result.failures.len(), 1);
        assert_eq!(result.failures[0].error_type, "TypeError");
    }
}
