// File: src-tauri/src/testing/executor_unified.rs
// Purpose: Unified test executor that routes to language-specific executors
// Last Updated: December 4, 2025
//
// Supports all 13 implemented languages with their respective test runners:
// - Python (pytest)
// - JavaScript/TypeScript (Jest)
// - Rust (cargo test)
// - Go (go test)
// - Java (JUnit via Maven/Gradle)
// - C (Unity)
// - C++ (Google Test)
// - Ruby (RSpec)
// - PHP (PHPUnit)
// - Swift (swift test)
// - Kotlin (JUnit via Gradle)

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Generic test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTestResult {
    /// Overall success (all tests passed)
    pub success: bool,
    /// Number of tests that passed
    pub passed: usize,
    /// Number of tests that failed
    pub failed: usize,
    /// Number of tests that were skipped
    pub skipped: usize,
    /// Total number of tests
    pub total: usize,
    /// Execution time in seconds
    pub duration_seconds: f64,
    /// Pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Test failures (for debugging)
    pub failures: Vec<TestFailure>,
    /// Coverage percentage (if available)
    pub coverage_percent: Option<f64>,
    /// Language of the tests
    pub language: String,
    /// Test framework used
    pub framework: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFailure {
    pub test_name: String,
    pub error_type: String,
    pub error_message: String,
}

impl UnifiedTestResult {
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
}

/// Unified test executor
pub struct UnifiedTestExecutor {
    workspace_path: PathBuf,
}

impl UnifiedTestExecutor {
    pub fn new(workspace_path: PathBuf) -> Self {
        UnifiedTestExecutor { workspace_path }
    }

    /// Execute tests for any supported language
    pub async fn execute_tests(
        &self,
        test_file: &Path,
        language: &str,
        timeout_seconds: u64,
    ) -> Result<UnifiedTestResult, String> {
        let language_lower = language.to_lowercase();
        
        match language_lower.as_str() {
            "python" | "py" => self.execute_pytest(test_file, timeout_seconds).await,
            "javascript" | "js" | "jsx" | "typescript" | "ts" | "tsx" => {
                self.execute_jest(test_file, timeout_seconds).await
            }
            "rust" | "rs" => self.execute_cargo_test(test_file, timeout_seconds).await,
            "go" => self.execute_go_test(test_file, timeout_seconds).await,
            "java" => self.execute_junit_java(test_file, timeout_seconds).await,
            "kotlin" | "kt" => self.execute_junit_kotlin(test_file, timeout_seconds).await,
            "c" => self.execute_unity_test(test_file, timeout_seconds).await,
            "cpp" | "c++" | "cc" | "cxx" => self.execute_gtest(test_file, timeout_seconds).await,
            "ruby" | "rb" => self.execute_rspec(test_file, timeout_seconds).await,
            "php" => self.execute_phpunit(test_file, timeout_seconds).await,
            "swift" => self.execute_xctest(test_file, timeout_seconds).await,
            _ => Err(format!("Unsupported language for test execution: {}", language)),
        }
    }

    /// Execute pytest (Python)
    async fn execute_pytest(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("python")
            .args(&["-m", "pytest", "--json-report", "--json-report-file=.pytest-report.json"])
            .arg(test_file)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute pytest: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        
        // Parse JSON report
        let report_path = self.workspace_path.join(".pytest-report.json");
        let result = if report_path.exists() {
            self.parse_pytest_report(&report_path, duration)?
        } else {
            self.parse_pytest_stdout(&output.stdout, duration)?
        };
        
        Ok(result)
    }

    /// Execute Jest (JavaScript/TypeScript)
    async fn execute_jest(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("npm")
            .args(&["test", "--", "--json", "--coverage"])
            .arg(test_file)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute Jest: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_jest_output(&output.stdout, duration)
    }

    /// Execute cargo test (Rust)
    async fn execute_cargo_test(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("cargo")
            .args(&["test", "--", "--test-threads=1", "--nocapture"])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute cargo test: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_cargo_test_output(&output.stdout, duration)
    }

    /// Execute go test (Go)
    async fn execute_go_test(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("go")
            .args(&["test", "-v", "-coverprofile=coverage.out"])
            .arg(test_file)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute go test: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_go_test_output(&output.stdout, duration)
    }

    /// Execute JUnit (Java)
    async fn execute_junit_java(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        // Try Maven first, then Gradle
        let output = if self.workspace_path.join("pom.xml").exists() {
            Command::new("mvn")
                .args(&["test", "-Dtest=" ])
                .arg(test_file.file_stem().unwrap())
                .current_dir(&self.workspace_path)
                .output()
                .map_err(|e| format!("Failed to execute Maven test: {}", e))?
        } else {
            Command::new("gradle")
                .args(&["test", "--tests"])
                .arg(test_file.file_stem().unwrap())
                .current_dir(&self.workspace_path)
                .output()
                .map_err(|e| format!("Failed to execute Gradle test: {}", e))?
        };
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_junit_output(&output.stdout, duration)
    }

    /// Execute JUnit (Kotlin)
    async fn execute_junit_kotlin(&self, test_file: &Path, timeout: u64) -> Result<UnifiedTestResult, String> {
        // Kotlin uses same JUnit infrastructure as Java
        self.execute_junit_java(test_file, timeout).await
    }

    /// Execute Unity (C)
    async fn execute_unity_test(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        // Unity tests are typically compiled and run as executables
        let test_executable = test_file.with_extension("");
        
        let output = Command::new(&test_executable)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute Unity test: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_unity_output(&output.stdout, duration)
    }

    /// Execute Google Test (C++)
    async fn execute_gtest(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        // Google Test executables are typically compiled and run
        let test_executable = test_file.with_extension("");
        
        let output = Command::new(&test_executable)
            .args(&["--gtest_output=json:gtest-results.json"])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute Google Test: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        
        // Parse JSON output if available
        let report_path = self.workspace_path.join("gtest-results.json");
        if report_path.exists() {
            self.parse_gtest_json(&report_path, duration)
        } else {
            self.parse_gtest_output(&output.stdout, duration)
        }
    }

    /// Execute RSpec (Ruby)
    async fn execute_rspec(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("rspec")
            .args(&["--format", "json", "--out", "rspec-results.json"])
            .arg(test_file)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute RSpec: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        
        let report_path = self.workspace_path.join("rspec-results.json");
        if report_path.exists() {
            self.parse_rspec_json(&report_path, duration)
        } else {
            self.parse_rspec_output(&output.stdout, duration)
        }
    }

    /// Execute PHPUnit (PHP)
    async fn execute_phpunit(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("phpunit")
            .args(&["--log-junit", "phpunit-results.xml"])
            .arg(test_file)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute PHPUnit: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_phpunit_output(&output.stdout, duration)
    }

    /// Execute XCTest (Swift)
    async fn execute_xctest(&self, test_file: &Path, _timeout: u64) -> Result<UnifiedTestResult, String> {
        let start = Instant::now();
        
        let output = Command::new("swift")
            .args(&["test", "--parallel"])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute Swift test: {}", e))?;
        
        let duration = start.elapsed().as_secs_f64();
        self.parse_xctest_output(&output.stdout, duration)
    }

    // ============================================================================
    // Output Parsers
    // ============================================================================

    fn parse_pytest_report(&self, report_path: &Path, duration: f64) -> Result<UnifiedTestResult, String> {
        let content = std::fs::read_to_string(report_path)
            .map_err(|e| format!("Failed to read pytest report: {}", e))?;
        
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse pytest JSON: {}", e))?;
        
        let summary = &json["summary"];
        let passed = summary["passed"].as_u64().unwrap_or(0) as usize;
        let failed = summary["failed"].as_u64().unwrap_or(0) as usize;
        let skipped = summary["skipped"].as_u64().unwrap_or(0) as usize;
        let total = summary["total"].as_u64().unwrap_or(0) as usize;
        
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Python".to_string(),
            framework: "pytest".to_string(),
        })
    }

    fn parse_pytest_stdout(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        
        for line in output.lines() {
            if line.contains(" passed") {
                if let Some(num) = line.split_whitespace().next() {
                    passed = num.parse().unwrap_or(0);
                }
            }
            if line.contains(" failed") {
                if let Some(num) = line.split_whitespace().next() {
                    failed = num.parse().unwrap_or(0);
                }
            }
            if line.contains(" skipped") {
                if let Some(num) = line.split_whitespace().next() {
                    skipped = num.parse().unwrap_or(0);
                }
            }
        }
        
        let total = passed + failed + skipped;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Python".to_string(),
            framework: "pytest".to_string(),
        })
    }

    fn parse_jest_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        // Jest outputs JSON when --json flag is used
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&output) {
            let passed = json["numPassedTests"].as_u64().unwrap_or(0) as usize;
            let failed = json["numFailedTests"].as_u64().unwrap_or(0) as usize;
            let total = json["numTotalTests"].as_u64().unwrap_or(0) as usize;
            let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
            
            return Ok(UnifiedTestResult {
                success: failed == 0,
                passed,
                failed,
                skipped: 0,
                total,
                duration_seconds: duration,
                pass_rate,
                failures: Vec::new(),
                coverage_percent: None,
                language: "JavaScript".to_string(),
                framework: "Jest".to_string(),
            });
        }
        
        Err("Failed to parse Jest output".to_string())
    }

    fn parse_cargo_test_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for line in output.lines() {
            if line.contains("test result:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == "passed;" && i > 0 {
                        passed = parts[i - 1].parse().unwrap_or(0);
                    }
                    if *part == "failed;" && i > 0 {
                        failed = parts[i - 1].parse().unwrap_or(0);
                    }
                }
            }
        }
        
        let total = passed + failed;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Rust".to_string(),
            framework: "cargo test".to_string(),
        })
    }

    fn parse_go_test_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for line in output.lines() {
            if line.starts_with("PASS") {
                passed += 1;
            } else if line.starts_with("FAIL") {
                failed += 1;
            }
        }
        
        let total = passed + failed;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Go".to_string(),
            framework: "go test".to_string(),
        })
    }

    fn parse_junit_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        
        for line in output.lines() {
            if line.contains("Tests run:") {
                let parts: Vec<&str> = line.split(',').collect();
                for part in parts {
                    if part.contains("Failures:") {
                        failed = part.split(':').nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0);
                    }
                    if part.contains("Skipped:") {
                        skipped = part.split(':').nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0);
                    }
                }
            }
        }
        
        let total = passed + failed + skipped;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Java".to_string(),
            framework: "JUnit".to_string(),
        })
    }

    fn parse_unity_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for line in output.lines() {
            if line.contains("PASS") || line.contains("OK") {
                passed += 1;
            } else if line.contains("FAIL") {
                failed += 1;
            }
        }
        
        let total = passed + failed;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "C".to_string(),
            framework: "Unity".to_string(),
        })
    }

    fn parse_gtest_json(&self, report_path: &Path, duration: f64) -> Result<UnifiedTestResult, String> {
        let content = std::fs::read_to_string(report_path)
            .map_err(|e| format!("Failed to read gtest report: {}", e))?;
        
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse gtest JSON: {}", e))?;
        
        let tests = json["tests"].as_u64().unwrap_or(0) as usize;
        let failures = json["failures"].as_u64().unwrap_or(0) as usize;
        let passed = tests - failures;
        
        let pass_rate = if tests > 0 { passed as f64 / tests as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failures == 0,
            passed,
            failed: failures,
            skipped: 0,
            total: tests,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "C++".to_string(),
            framework: "Google Test".to_string(),
        })
    }

    fn parse_gtest_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for line in output.lines() {
            if line.contains("[  PASSED  ]") {
                passed += 1;
            } else if line.contains("[  FAILED  ]") {
                failed += 1;
            }
        }
        
        let total = passed + failed;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "C++".to_string(),
            framework: "Google Test".to_string(),
        })
    }

    fn parse_rspec_json(&self, report_path: &Path, duration: f64) -> Result<UnifiedTestResult, String> {
        let content = std::fs::read_to_string(report_path)
            .map_err(|e| format!("Failed to read rspec report: {}", e))?;
        
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse rspec JSON: {}", e))?;
        
        let summary = &json["summary"];
        let passed = summary["example_count"].as_u64().unwrap_or(0) as usize;
        let failed = summary["failure_count"].as_u64().unwrap_or(0) as usize;
        let total = passed + failed;
        
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Ruby".to_string(),
            framework: "RSpec".to_string(),
        })
    }

    fn parse_rspec_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for line in output.lines() {
            if line.contains(" examples, ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(num) = parts.first() {
                    let total: usize = num.parse().unwrap_or(0);
                    for (i, part) in parts.iter().enumerate() {
                        if *part == "failures" && i > 0 {
                            failed = parts[i - 1].parse().unwrap_or(0);
                        }
                    }
                    passed = total - failed;
                }
            }
        }
        
        let total = passed + failed;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Ruby".to_string(),
            framework: "RSpec".to_string(),
        })
    }

    fn parse_phpunit_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;
        
        for line in output.lines() {
            if line.contains("Tests:") {
                let parts: Vec<&str> = line.split(',').collect();
                for part in parts {
                    if part.contains("Failures:") {
                        failed = part.split(':').nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0);
                    }
                    if part.contains("Skipped:") {
                        skipped = part.split(':').nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0);
                    }
                }
            }
        }
        
        let total = passed + failed + skipped;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "PHP".to_string(),
            framework: "PHPUnit".to_string(),
        })
    }

    fn parse_xctest_output(&self, stdout: &[u8], duration: f64) -> Result<UnifiedTestResult, String> {
        let output = String::from_utf8_lossy(stdout);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for line in output.lines() {
            if line.contains("Test Case") && line.contains("passed") {
                passed += 1;
            } else if line.contains("Test Case") && line.contains("failed") {
                failed += 1;
            }
        }
        
        let total = passed + failed;
        let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
        
        Ok(UnifiedTestResult {
            success: failed == 0,
            passed,
            failed,
            skipped: 0,
            total,
            duration_seconds: duration,
            pass_rate,
            failures: Vec::new(),
            coverage_percent: None,
            language: "Swift".to_string(),
            framework: "XCTest".to_string(),
        })
    }
}
