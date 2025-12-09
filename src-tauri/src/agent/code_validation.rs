// File: src-tauri/src/agent/code_validation.rs
// Purpose: Universal code validation state (SM-CG-014)
// Last Updated: December 9, 2025
//
// Validates code with language-appropriate tools before testing.
// Supports:
// - Compiled languages: cargo check, go build, javac, tsc --noEmit
// - Interpreted languages: mypy, pylint, eslint, py_compile, import checks
//
// Performance targets:
// - Validation: <5s per file
// - Batch validation: <30s for 10 files
// - Error extraction: <100ms

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::process::Command as TokioCommand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub success: bool,
    pub language: String,
    pub file_path: PathBuf,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub duration_ms: u64,
    pub validator_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub message: String,
    pub error_type: ErrorType,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub message: String,
    pub warning_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErrorType {
    Syntax,
    Type,
    Import,
    Undefined,
    Compilation,
    Lint,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Kotlin,
    C,
    Cpp,
    Ruby,
    PHP,
    Swift,
}

impl Language {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "js" => Some(Language::JavaScript),
            "ts" => Some(Language::TypeScript),
            "tsx" => Some(Language::TypeScript),
            "jsx" => Some(Language::JavaScript),
            "go" => Some(Language::Go),
            "java" => Some(Language::Java),
            "kt" => Some(Language::Kotlin),
            "c" => Some(Language::C),
            "cpp" | "cc" | "cxx" => Some(Language::Cpp),
            "h" | "hpp" => Some(Language::Cpp),
            "rb" => Some(Language::Ruby),
            "php" => Some(Language::PHP),
            "swift" => Some(Language::Swift),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Language::Rust => "rust".to_string(),
            Language::Python => "python".to_string(),
            Language::JavaScript => "javascript".to_string(),
            Language::TypeScript => "typescript".to_string(),
            Language::Go => "go".to_string(),
            Language::Java => "java".to_string(),
            Language::Kotlin => "kotlin".to_string(),
            Language::C => "c".to_string(),
            Language::Cpp => "cpp".to_string(),
            Language::Ruby => "ruby".to_string(),
            Language::PHP => "php".to_string(),
            Language::Swift => "swift".to_string(),
        }
    }
}

/// Universal code validator
pub struct CodeValidator {
    workspace_path: PathBuf,
}

impl CodeValidator {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }

    /// Validate code file with appropriate validator
    pub async fn validate_file(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let start = std::time::Instant::now();

        // Detect language from extension
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| "No file extension found".to_string())?;

        let language = Language::from_extension(extension)
            .ok_or_else(|| format!("Unsupported language: {}", extension))?;

        // Run appropriate validator
        let result = match language {
            Language::Rust => self.validate_rust(file_path).await?,
            Language::Python => self.validate_python(file_path).await?,
            Language::JavaScript | Language::TypeScript => self.validate_js_ts(file_path, language).await?,
            Language::Go => self.validate_go(file_path).await?,
            Language::Java => self.validate_java(file_path).await?,
            Language::Kotlin => self.validate_kotlin(file_path).await?,
            Language::C | Language::Cpp => self.validate_c_cpp(file_path, language).await?,
            Language::Ruby => self.validate_ruby(file_path).await?,
            Language::PHP => self.validate_php(file_path).await?,
            Language::Swift => self.validate_swift(file_path).await?,
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ValidationResult {
            success: result.errors.is_empty(),
            language: language.to_string(),
            file_path: file_path.to_path_buf(),
            errors: result.errors,
            warnings: result.warnings,
            duration_ms,
            validator_used: result.validator_used,
        })
    }

    /// Validate Rust code with cargo check
    async fn validate_rust(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("cargo")
            .arg("check")
            .arg("--message-format=json")
            .current_dir(&self.workspace_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run cargo check: {}", e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_cargo_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "rust".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "cargo check".to_string(),
        })
    }

    /// Validate Python code with multiple validators
    async fn validate_python(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // 1. Syntax check with py_compile
        let syntax_check = TokioCommand::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(file_path)
            .output()
            .await;

        if let Ok(output) = syntax_check {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                errors.extend(self.parse_python_syntax_errors(&stderr, file_path));
            }
        }

        // 2. Type checking with mypy (if available)
        let mypy_check = TokioCommand::new("mypy")
            .arg("--no-error-summary")
            .arg(file_path)
            .output()
            .await;

        if let Ok(output) = mypy_check {
            let stdout = String::from_utf8_lossy(&output.stdout);
            errors.extend(self.parse_mypy_errors(&stdout));
        }

        // 3. Linting with pylint (if available)
        let pylint_check = TokioCommand::new("pylint")
            .arg("--output-format=parseable")
            .arg(file_path)
            .output()
            .await;

        if let Ok(output) = pylint_check {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let (lint_errors, lint_warnings) = self.parse_pylint_output(&stdout);
            errors.extend(lint_errors);
            warnings.extend(lint_warnings);
        }

        Ok(ValidationResult {
            success: errors.is_empty(),
            language: "python".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings,
            duration_ms: 0,
            validator_used: "py_compile + mypy + pylint".to_string(),
        })
    }

    /// Validate JavaScript/TypeScript with eslint and tsc
    async fn validate_js_ts(&self, file_path: &Path, language: Language) -> Result<ValidationResult, String> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // For TypeScript, run tsc --noEmit
        if language == Language::TypeScript {
            let tsc_check = TokioCommand::new("tsc")
                .arg("--noEmit")
                .arg(file_path)
                .output()
                .await;

            if let Ok(output) = tsc_check {
                let stdout = String::from_utf8_lossy(&output.stdout);
                errors.extend(self.parse_tsc_errors(&stdout));
            }
        }

        // Run eslint
        let eslint_check = TokioCommand::new("eslint")
            .arg("--format=json")
            .arg(file_path)
            .output()
            .await;

        if let Ok(output) = eslint_check {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let (eslint_errors, eslint_warnings) = self.parse_eslint_output(&stdout);
            errors.extend(eslint_errors);
            warnings.extend(eslint_warnings);
        }

        Ok(ValidationResult {
            success: errors.is_empty(),
            language: language.to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings,
            duration_ms: 0,
            validator_used: if language == Language::TypeScript {
                "tsc + eslint".to_string()
            } else {
                "eslint".to_string()
            },
        })
    }

    /// Validate Go code with go build
    async fn validate_go(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("go")
            .arg("build")
            .arg("-o")
            .arg("/dev/null")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run go build: {}", e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_go_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "go".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "go build".to_string(),
        })
    }

    /// Validate Java code with javac
    async fn validate_java(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("javac")
            .arg("-Xlint")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run javac: {}", e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_javac_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "java".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "javac".to_string(),
        })
    }

    /// Validate Kotlin code with kotlinc
    async fn validate_kotlin(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("kotlinc")
            .arg("-no-stdlib")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run kotlinc: {}", e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_kotlin_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "kotlin".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "kotlinc".to_string(),
        })
    }

    /// Validate C/C++ code with clang
    async fn validate_c_cpp(&self, file_path: &Path, language: Language) -> Result<ValidationResult, String> {
        let compiler = if language == Language::C { "clang" } else { "clang++" };
        
        let output = TokioCommand::new(compiler)
            .arg("-fsyntax-only")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run {}: {}", compiler, e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_clang_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: language.to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: format!("{} -fsyntax-only", compiler),
        })
    }

    /// Validate Ruby code with ruby -c
    async fn validate_ruby(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("ruby")
            .arg("-c")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run ruby -c: {}", e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_ruby_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "ruby".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "ruby -c".to_string(),
        })
    }

    /// Validate PHP code with php -l
    async fn validate_php(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("php")
            .arg("-l")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run php -l: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let errors = self.parse_php_errors(&stdout);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "php".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "php -l".to_string(),
        })
    }

    /// Validate Swift code with swiftc
    async fn validate_swift(&self, file_path: &Path) -> Result<ValidationResult, String> {
        let output = TokioCommand::new("swiftc")
            .arg("-typecheck")
            .arg(file_path)
            .output()
            .await
            .map_err(|e| format!("Failed to run swiftc: {}", e))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = self.parse_swift_errors(&stderr);

        Ok(ValidationResult {
            success: output.status.success(),
            language: "swift".to_string(),
            file_path: file_path.to_path_buf(),
            errors,
            warnings: Vec::new(),
            duration_ms: 0,
            validator_used: "swiftc -typecheck".to_string(),
        })
    }

    // Error parsers for each language
    fn parse_cargo_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(message) = json.get("message") {
                    if let Some(msg_text) = message.get("message").and_then(|m| m.as_str()) {
                        errors.push(ValidationError {
                            line: None,
                            column: None,
                            message: msg_text.to_string(),
                            error_type: ErrorType::Compilation,
                            severity: Severity::Error,
                        });
                    }
                }
            }
        }

        errors
    }

    fn parse_python_syntax_errors(&self, output: &str, file_path: &Path) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("SyntaxError") || line.contains("IndentationError") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Syntax,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_mypy_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("error:") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Type,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_pylint_output(&self, output: &str) -> (Vec<ValidationError>, Vec<ValidationWarning>) {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for line in output.lines() {
            if line.contains(":E") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Lint,
                    severity: Severity::Error,
                });
            } else if line.contains(":W") {
                warnings.push(ValidationWarning {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    warning_type: "lint".to_string(),
                });
            }
        }

        (errors, warnings)
    }

    fn parse_tsc_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("error TS") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Type,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_eslint_output(&self, output: &str) -> (Vec<ValidationError>, Vec<ValidationWarning>) {
        (Vec::new(), Vec::new()) // Simplified for now
    }

    fn parse_go_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if !line.is_empty() {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Compilation,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_javac_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("error:") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Compilation,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_kotlin_errors(&self, output: &str) -> Vec<ValidationError> {
        self.parse_javac_errors(output) // Similar format
    }

    fn parse_clang_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("error:") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Compilation,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_ruby_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("SyntaxError") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Syntax,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_php_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("Parse error") || line.contains("Fatal error") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Syntax,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }

    fn parse_swift_errors(&self, output: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        for line in output.lines() {
            if line.contains("error:") {
                errors.push(ValidationError {
                    line: None,
                    column: None,
                    message: line.to_string(),
                    error_type: ErrorType::Compilation,
                    severity: Severity::Error,
                });
            }
        }

        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("js"), Some(Language::JavaScript));
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("go"), Some(Language::Go));
        assert_eq!(Language::from_extension("unknown"), None);
    }

    #[tokio::test]
    async fn test_validator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let validator = CodeValidator::new(temp_dir.path().to_path_buf());
        assert_eq!(validator.workspace_path, temp_dir.path());
    }
}
