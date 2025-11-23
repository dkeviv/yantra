// Security module - Semgrep integration for vulnerability scanning
// Purpose: Scan code for security vulnerabilities and generate auto-fixes

pub mod semgrep;
pub mod autofix;

pub use semgrep::{SemgrepScanner, SecurityIssue, Severity};
pub use autofix::{AutoFixer, FixSuggestion};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_module_exports() {
        // Verify module exports are accessible
        let _scanner: Option<SemgrepScanner> = None;
        let _fixer: Option<AutoFixer> = None;
    }
}
