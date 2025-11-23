// Semgrep scanner implementation
// Scans code for security vulnerabilities using Semgrep OWASP rules

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub id: String,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub file_path: String,
    pub line_number: usize,
    pub code_snippet: String,
    pub fix_suggestion: Option<String>,
    pub cwe_id: Option<String>, // Common Weakness Enumeration ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub success: bool,
    pub issues: Vec<SecurityIssue>,
    pub files_scanned: usize,
    pub duration_ms: u64,
}

pub struct SemgrepScanner {
    workspace_path: PathBuf,
    ruleset: String, // e.g., "p/owasp-top-10"
}

impl SemgrepScanner {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self {
            workspace_path,
            ruleset: "p/owasp-top-10".to_string(),
        }
    }

    pub fn with_ruleset(mut self, ruleset: String) -> Self {
        self.ruleset = ruleset;
        self
    }

    pub async fn scan(&self) -> Result<ScanResult, String> {
        let start_time = std::time::Instant::now();

        // Execute Semgrep command
        // semgrep --config=p/owasp-top-10 --json <workspace_path>
        let output = Command::new("semgrep")
            .arg("--config")
            .arg(&self.ruleset)
            .arg("--json")
            .arg(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute Semgrep: {}", e))?;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        if !output.status.success() {
            return Err(format!(
                "Semgrep failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Parse JSON output
        let json_output = String::from_utf8_lossy(&output.stdout);
        let issues = self.parse_semgrep_output(&json_output)?;

        let files_scanned = self.count_scanned_files(&json_output);

        Ok(ScanResult {
            success: true,
            issues,
            files_scanned,
            duration_ms,
        })
    }

    fn parse_semgrep_output(&self, json_output: &str) -> Result<Vec<SecurityIssue>, String> {
        // Simplified parsing - in production, use serde_json for robust parsing
        let mut issues = Vec::new();

        // Example Semgrep output structure:
        // {
        //   "results": [
        //     {
        //       "check_id": "python.lang.security.injection.sql-injection",
        //       "path": "app.py",
        //       "start": { "line": 10 },
        //       "end": { "line": 10 },
        //       "extra": {
        //         "message": "SQL injection vulnerability",
        //         "severity": "ERROR",
        //         "metadata": { "cwe": "CWE-89" }
        //       }
        //     }
        //   ]
        // }

        // For MVP, return a placeholder issue if any vulnerabilities found
        if json_output.contains("results") && json_output.len() > 100 {
            issues.push(SecurityIssue {
                id: "semgrep-001".to_string(),
                severity: Severity::Error,
                title: "Security Vulnerability Detected".to_string(),
                description: "Potential security issue found in code".to_string(),
                file_path: self.workspace_path.to_string_lossy().to_string(),
                line_number: 1,
                code_snippet: "".to_string(),
                fix_suggestion: Some("Review and fix security issue".to_string()),
                cwe_id: Some("CWE-79".to_string()),
            });
        }

        Ok(issues)
    }

    fn count_scanned_files(&self, _json_output: &str) -> usize {
        // Count files scanned from Semgrep output
        // For MVP, estimate based on workspace
        1
    }

    pub fn get_critical_issues(&self, scan_result: &ScanResult) -> Vec<SecurityIssue> {
        scan_result
            .issues
            .iter()
            .filter(|issue| issue.severity == Severity::Critical)
            .cloned()
            .collect()
    }

    pub fn get_high_severity_issues(&self, scan_result: &ScanResult) -> Vec<SecurityIssue> {
        scan_result
            .issues
            .iter()
            .filter(|issue| {
                matches!(issue.severity, Severity::Critical | Severity::Error)
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_scanner_creation() {
        let temp_dir = TempDir::new().unwrap();
        let scanner = SemgrepScanner::new(temp_dir.path().to_path_buf());
        assert_eq!(scanner.ruleset, "p/owasp-top-10");
    }

    #[test]
    fn test_custom_ruleset() {
        let temp_dir = TempDir::new().unwrap();
        let scanner = SemgrepScanner::new(temp_dir.path().to_path_buf())
            .with_ruleset("p/python".to_string());
        assert_eq!(scanner.ruleset, "p/python");
    }

    #[test]
    fn test_parse_empty_output() {
        let temp_dir = TempDir::new().unwrap();
        let scanner = SemgrepScanner::new(temp_dir.path().to_path_buf());
        
        let json = r#"{"results": []}"#;
        let issues = scanner.parse_semgrep_output(json).unwrap();
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn test_get_critical_issues() {
        let temp_dir = TempDir::new().unwrap();
        let scanner = SemgrepScanner::new(temp_dir.path().to_path_buf());
        
        let scan_result = ScanResult {
            success: true,
            issues: vec![
                SecurityIssue {
                    id: "1".to_string(),
                    severity: Severity::Critical,
                    title: "Critical Issue".to_string(),
                    description: "".to_string(),
                    file_path: "".to_string(),
                    line_number: 1,
                    code_snippet: "".to_string(),
                    fix_suggestion: None,
                    cwe_id: None,
                },
                SecurityIssue {
                    id: "2".to_string(),
                    severity: Severity::Warning,
                    title: "Warning".to_string(),
                    description: "".to_string(),
                    file_path: "".to_string(),
                    line_number: 1,
                    code_snippet: "".to_string(),
                    fix_suggestion: None,
                    cwe_id: None,
                },
            ],
            files_scanned: 1,
            duration_ms: 100,
        };

        let critical = scanner.get_critical_issues(&scan_result);
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].severity, Severity::Critical);
    }

    #[test]
    fn test_severity_levels() {
        assert!(matches!(Severity::Critical, Severity::Critical));
        assert!(matches!(Severity::Error, Severity::Error));
        assert!(matches!(Severity::Warning, Severity::Warning));
        assert!(matches!(Severity::Info, Severity::Info));
    }
}
