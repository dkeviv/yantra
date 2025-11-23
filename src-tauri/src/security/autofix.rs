// Auto-fixer for security vulnerabilities
// Generates fixes for common security issues

use crate::security::semgrep::SecurityIssue;
#[cfg(test)]
use crate::security::semgrep::Severity;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct FixSuggestion {
    pub issue_id: String,
    pub fix_type: FixType,
    pub original_code: String,
    pub fixed_code: String,
    pub explanation: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FixType {
    InputValidation,
    OutputEncoding,
    SqlParameterization,
    CsrfProtection,
    XssProtection,
    AuthenticationFix,
    AuthorizationFix,
    CryptographyFix,
    Other,
}

pub struct AutoFixer {
    #[allow(dead_code)]
    workspace_path: PathBuf,
}

impl AutoFixer {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }

    pub async fn generate_fixes(&self, issues: &[SecurityIssue]) -> Vec<FixSuggestion> {
        let mut fixes = Vec::new();

        for issue in issues {
            if let Some(fix) = self.generate_fix(issue).await {
                fixes.push(fix);
            }
        }

        fixes
    }

    async fn generate_fix(&self, issue: &SecurityIssue) -> Option<FixSuggestion> {
        // Detect fix type based on CWE ID or description
        let fix_type = self.detect_fix_type(issue);
        
        match fix_type {
            FixType::SqlParameterization => self.fix_sql_injection(issue),
            FixType::XssProtection => self.fix_xss(issue),
            FixType::InputValidation => self.fix_input_validation(issue),
            _ => None,
        }
    }

    fn detect_fix_type(&self, issue: &SecurityIssue) -> FixType {
        // Detect based on CWE ID
        if let Some(cwe) = &issue.cwe_id {
            return match cwe.as_str() {
                "CWE-89" => FixType::SqlParameterization,
                "CWE-79" => FixType::XssProtection,
                "CWE-20" => FixType::InputValidation,
                "CWE-352" => FixType::CsrfProtection,
                "CWE-327" => FixType::CryptographyFix,
                _ => FixType::Other,
            };
        }

        // Detect based on description
        let desc_lower = issue.description.to_lowercase();
        if desc_lower.contains("sql injection") {
            FixType::SqlParameterization
        } else if desc_lower.contains("xss") || desc_lower.contains("cross-site scripting") {
            FixType::XssProtection
        } else if desc_lower.contains("csrf") {
            FixType::CsrfProtection
        } else {
            FixType::Other
        }
    }

    fn fix_sql_injection(&self, issue: &SecurityIssue) -> Option<FixSuggestion> {
        // Example: Convert string concatenation to parameterized query
        // Original: f"SELECT * FROM users WHERE id = {user_id}"
        // Fixed: "SELECT * FROM users WHERE id = %s", (user_id,)
        
        Some(FixSuggestion {
            issue_id: issue.id.clone(),
            fix_type: FixType::SqlParameterization,
            original_code: issue.code_snippet.clone(),
            fixed_code: "Use parameterized queries with placeholders".to_string(),
            explanation: "Replace string concatenation with parameterized queries to prevent SQL injection".to_string(),
            confidence: 0.9,
        })
    }

    fn fix_xss(&self, issue: &SecurityIssue) -> Option<FixSuggestion> {
        // Example: Add output encoding
        // Original: return f"<div>{user_input}</div>"
        // Fixed: return f"<div>{html.escape(user_input)}</div>"
        
        Some(FixSuggestion {
            issue_id: issue.id.clone(),
            fix_type: FixType::XssProtection,
            original_code: issue.code_snippet.clone(),
            fixed_code: "Add HTML encoding/escaping to user input".to_string(),
            explanation: "Escape user input before rendering in HTML to prevent XSS attacks".to_string(),
            confidence: 0.85,
        })
    }

    fn fix_input_validation(&self, issue: &SecurityIssue) -> Option<FixSuggestion> {
        Some(FixSuggestion {
            issue_id: issue.id.clone(),
            fix_type: FixType::InputValidation,
            original_code: issue.code_snippet.clone(),
            fixed_code: "Add input validation and sanitization".to_string(),
            explanation: "Validate and sanitize all user input before processing".to_string(),
            confidence: 0.75,
        })
    }

    pub fn apply_fixes(&self, fixes: &[FixSuggestion]) -> Result<usize, String> {
        // In production, this would modify files
        // For MVP, return count of fixes that would be applied
        
        let high_confidence_fixes = fixes
            .iter()
            .filter(|f| f.confidence >= 0.8)
            .count();
        
        Ok(high_confidence_fixes)
    }

    pub fn get_high_confidence_fixes(&self, fixes: &[FixSuggestion]) -> Vec<FixSuggestion> {
        fixes
            .iter()
            .filter(|f| f.confidence >= 0.8)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_autofixer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let fixer = AutoFixer::new(temp_dir.path().to_path_buf());
        assert!(fixer.workspace_path.exists());
    }

    #[test]
    fn test_detect_fix_type_sql_injection() {
        let temp_dir = TempDir::new().unwrap();
        let fixer = AutoFixer::new(temp_dir.path().to_path_buf());
        
        let issue = SecurityIssue {
            id: "1".to_string(),
            severity: Severity::Critical,
            title: "SQL Injection".to_string(),
            description: "Possible SQL injection vulnerability".to_string(),
            file_path: "".to_string(),
            line_number: 1,
            code_snippet: "".to_string(),
            fix_suggestion: None,
            cwe_id: Some("CWE-89".to_string()),
        };

        let fix_type = fixer.detect_fix_type(&issue);
        assert_eq!(fix_type, FixType::SqlParameterization);
    }

    #[test]
    fn test_detect_fix_type_xss() {
        let temp_dir = TempDir::new().unwrap();
        let fixer = AutoFixer::new(temp_dir.path().to_path_buf());
        
        let issue = SecurityIssue {
            id: "2".to_string(),
            severity: Severity::Error,
            title: "XSS".to_string(),
            description: "Cross-site scripting vulnerability detected".to_string(),
            file_path: "".to_string(),
            line_number: 1,
            code_snippet: "".to_string(),
            fix_suggestion: None,
            cwe_id: Some("CWE-79".to_string()),
        };

        let fix_type = fixer.detect_fix_type(&issue);
        assert_eq!(fix_type, FixType::XssProtection);
    }

    #[test]
    fn test_fix_sql_injection() {
        let temp_dir = TempDir::new().unwrap();
        let fixer = AutoFixer::new(temp_dir.path().to_path_buf());
        
        let issue = SecurityIssue {
            id: "1".to_string(),
            severity: Severity::Critical,
            title: "SQL Injection".to_string(),
            description: "SQL injection".to_string(),
            file_path: "".to_string(),
            line_number: 1,
            code_snippet: "SELECT * FROM users WHERE id = {}".to_string(),
            fix_suggestion: None,
            cwe_id: Some("CWE-89".to_string()),
        };

        let fix = fixer.fix_sql_injection(&issue).unwrap();
        assert_eq!(fix.fix_type, FixType::SqlParameterization);
        assert!(fix.confidence >= 0.8);
    }

    #[test]
    fn test_high_confidence_fixes() {
        let temp_dir = TempDir::new().unwrap();
        let fixer = AutoFixer::new(temp_dir.path().to_path_buf());
        
        let fixes = vec![
            FixSuggestion {
                issue_id: "1".to_string(),
                fix_type: FixType::SqlParameterization,
                original_code: "".to_string(),
                fixed_code: "".to_string(),
                explanation: "".to_string(),
                confidence: 0.9,
            },
            FixSuggestion {
                issue_id: "2".to_string(),
                fix_type: FixType::XssProtection,
                original_code: "".to_string(),
                fixed_code: "".to_string(),
                explanation: "".to_string(),
                confidence: 0.6,
            },
        ];

        let high_confidence = fixer.get_high_confidence_fixes(&fixes);
        assert_eq!(high_confidence.len(), 1);
        assert!(high_confidence[0].confidence >= 0.8);
    }
}
