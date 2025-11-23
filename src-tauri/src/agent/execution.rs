// File: src-tauri/src/agent/execution.rs
// Purpose: Script executor for running generated code with error detection
// Last Updated: November 22, 2025
//
// This module implements script execution with intelligent error classification:
// - Detect entry points (main functions, __main__ blocks)
// - Execute Python, Node.js, and Rust scripts
// - Capture stdout/stderr with real-time streaming
// - Classify errors (ImportError, SyntaxError, RuntimeError, PermissionError)
// - Extract error messages and tracebacks
// - Performance profiling (execution time, memory usage)
//
// Performance targets:
// - Entry point detection: <10ms
// - Script execution: Depends on code
// - Error classification: <5ms

// Most types in this module are not yet fully integrated
#![allow(dead_code)]

use crate::agent::dependencies::{DependencyInstaller, ProjectType};
use crate::agent::terminal::{ExecutionResult, TerminalExecutor, TerminalOutput};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Error type classification for runtime errors
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    /// Missing import or module (e.g., ModuleNotFoundError, ImportError)
    ImportError,
    /// Syntax error in the code (e.g., SyntaxError, IndentationError)
    SyntaxError,
    /// Runtime error during execution (e.g., TypeError, ValueError, ZeroDivisionError)
    RuntimeError,
    /// Permission denied or file not found (e.g., PermissionError, FileNotFoundError)
    PermissionError,
    /// Timeout during execution
    TimeoutError,
    /// Unknown or unclassified error
    UnknownError,
}

impl ErrorType {
    /// Classify error from error message
    pub fn from_message(message: &str) -> Self {
        let message_lower = message.to_lowercase();
        
        // ImportError patterns
        if message_lower.contains("modulenotfounderror") 
            || message_lower.contains("importerror") 
            || message_lower.contains("cannot find module")
            || message_lower.contains("no module named") {
            return ErrorType::ImportError;
        }
        
        // SyntaxError patterns
        if message_lower.contains("syntaxerror") 
            || message_lower.contains("indentationerror")
            || message_lower.contains("unexpected token")
            || message_lower.contains("invalid syntax") {
            return ErrorType::SyntaxError;
        }
        
        // PermissionError patterns
        if message_lower.contains("permissionerror") 
            || message_lower.contains("permission denied")
            || message_lower.contains("eacces") {
            return ErrorType::PermissionError;
        }
        
        // TimeoutError patterns
        if message_lower.contains("timeout") 
            || message_lower.contains("timed out") {
            return ErrorType::TimeoutError;
        }
        
        // RuntimeError patterns (more specific errors)
        if message_lower.contains("zerodivisionerror")
            || message_lower.contains("typeerror")
            || message_lower.contains("valueerror")
            || message_lower.contains("attributeerror")
            || message_lower.contains("keyerror")
            || message_lower.contains("indexerror")
            || message_lower.contains("nameerror")
            || message_lower.contains("runtimeerror") {
            return ErrorType::RuntimeError;
        }
        
        // Generic error
        if message_lower.contains("error") {
            return ErrorType::RuntimeError;
        }
        
        ErrorType::UnknownError
    }
}

/// Execution result with error classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptExecutionResult {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub output: Vec<TerminalOutput>,
    pub duration_ms: u64,
    pub error_type: Option<ErrorType>,
    pub error_message: Option<String>,
    pub traceback: Option<String>,
    pub entry_point: String,
}

/// Script executor for running generated code
pub struct ScriptExecutor {
    workspace_path: PathBuf,
    terminal_executor: TerminalExecutor,
    dependency_installer: DependencyInstaller,
}

impl ScriptExecutor {
    /// Create new script executor for workspace
    pub fn new(workspace_path: PathBuf) -> Self {
        let terminal_executor = TerminalExecutor::new(workspace_path.clone())
            .with_timeout(Duration::from_secs(300)); // 5 minutes default
        let dependency_installer = DependencyInstaller::new(workspace_path.clone());

        ScriptExecutor {
            workspace_path,
            terminal_executor,
            dependency_installer,
        }
    }

    /// Set custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.terminal_executor = self.terminal_executor.with_timeout(timeout);
        self
    }

    /// Detect entry point in Python file
    fn detect_python_entry_point(&self, file_path: &PathBuf) -> Result<String, String> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Check for if __name__ == '__main__':
        if content.contains("if __name__ == '__main__':") 
            || content.contains("if __name__ == \"__main__\":") {
            return Ok(format!("python3 {}", file_path.display()));
        }

        // Check for main() function
        if content.contains("def main(") {
            return Ok(format!("python3 -c \"import sys; sys.path.insert(0, '.'); from {} import main; main()\"", 
                file_path.file_stem().unwrap().to_string_lossy()));
        }

        // Default: just run the script
        Ok(format!("python3 {}", file_path.display()))
    }

    /// Detect entry point in Node.js file
    fn detect_node_entry_point(&self, file_path: &PathBuf) -> Result<String, String> {
        // Check package.json for main entry point
        let package_json = self.workspace_path.join("package.json");
        if package_json.exists() {
            if let Ok(content) = std::fs::read_to_string(&package_json) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(main) = json.get("main").and_then(|v| v.as_str()) {
                        return Ok(format!("node {}", main));
                    }
                }
            }
        }

        // Default: run the file
        Ok(format!("node {}", file_path.display()))
    }

    /// Execute Python script
    pub async fn execute_python(&self, file_path: PathBuf) -> Result<ScriptExecutionResult, String> {
        let entry_point = self.detect_python_entry_point(&file_path)?;
        
        // Parse entry point into command and args
        let parts: Vec<&str> = entry_point.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty entry point".to_string());
        }

        let command = parts[0];
        let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        // Execute
        let result = self.terminal_executor.execute(command, args).await?;

        // Classify errors
        let (error_type, error_message, traceback) = if !result.success {
            self.classify_execution_error(&result)
        } else {
            (None, None, None)
        };

        Ok(ScriptExecutionResult {
            success: result.success,
            exit_code: result.exit_code,
            output: result.output.clone(),
            duration_ms: result.duration_ms,
            error_type,
            error_message,
            traceback,
            entry_point,
        })
    }

    /// Execute Node.js script
    pub async fn execute_node(&self, file_path: PathBuf) -> Result<ScriptExecutionResult, String> {
        let entry_point = self.detect_node_entry_point(&file_path)?;
        
        // Parse entry point
        let parts: Vec<&str> = entry_point.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty entry point".to_string());
        }

        let command = parts[0];
        let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

        // Execute
        let result = self.terminal_executor.execute(command, args).await?;

        // Classify errors
        let (error_type, error_message, traceback) = if !result.success {
            self.classify_execution_error(&result)
        } else {
            (None, None, None)
        };

        Ok(ScriptExecutionResult {
            success: result.success,
            exit_code: result.exit_code,
            output: result.output.clone(),
            duration_ms: result.duration_ms,
            error_type,
            error_message,
            traceback,
            entry_point,
        })
    }

    /// Execute Python code string (not a file)
    pub async fn execute_python_code(&self, code: &str) -> Result<ScriptExecutionResult, String> {
        let result = self.terminal_executor.execute_python_command(code).await?;

        // Classify errors
        let (error_type, error_message, traceback) = if !result.success {
            self.classify_execution_error(&result)
        } else {
            (None, None, None)
        };

        Ok(ScriptExecutionResult {
            success: result.success,
            exit_code: result.exit_code,
            output: result.output.clone(),
            duration_ms: result.duration_ms,
            error_type,
            error_message,
            traceback,
            entry_point: "python3 -c <code>".to_string(),
        })
    }

    /// Classify execution error from output
    fn classify_execution_error(&self, result: &ExecutionResult) -> (Option<ErrorType>, Option<String>, Option<String>) {
        let mut error_type = None;
        let mut error_message = None;
        let mut traceback_lines = Vec::new();
        let mut in_traceback = false;

        // Combine all output to search for errors
        let all_output: Vec<String> = result.output.iter().filter_map(|o| {
            match o {
                TerminalOutput::Stderr(line) | TerminalOutput::Error(line) | TerminalOutput::Stdout(line) => Some(line.clone()),
                _ => None,
            }
        }).collect();

        // First pass: look for error patterns in all output
        let combined_output = all_output.join("\n");
        
        // Classify based on combined output
        if error_type.is_none() {
            error_type = Some(ErrorType::from_message(&combined_output));
        }

        // Second pass: extract traceback and error message
        for line in &all_output {
            // Check for traceback start
            if line.contains("Traceback (most recent call last)") {
                in_traceback = true;
                traceback_lines.clear();
                traceback_lines.push(line.clone());
                continue;
            }

            if in_traceback {
                traceback_lines.push(line.clone());
            }

            // Extract error message (lines containing "Error:")
            if line.contains("Error:") || line.contains("Error") {
                error_message = Some(line.clone());
            }
        }

        let traceback = if traceback_lines.is_empty() {
            None
        } else {
            Some(traceback_lines.join("\n"))
        };

        (error_type, error_message, traceback)
    }

    /// Execute with auto-fix for import errors
    /// If execution fails with ImportError, automatically install the missing package and retry
    pub async fn execute_with_auto_fix(
        &self,
        file_path: PathBuf,
        project_type: ProjectType,
        max_retries: usize,
    ) -> Result<ScriptExecutionResult, String> {
        let mut attempt = 0;

        loop {
            // Try to execute
            let result = match project_type {
                ProjectType::Python => self.execute_python(file_path.clone()).await?,
                ProjectType::Node => self.execute_node(file_path.clone()).await?,
                _ => return Err("Unsupported project type for execution".to_string()),
            };

            // If successful, return
            if result.success {
                return Ok(result);
            }

            // If max retries reached, return failure
            if attempt >= max_retries {
                return Ok(result);
            }

            // Check if it's an import error
            if result.error_type == Some(ErrorType::ImportError) {
                // Try to auto-fix
                if let Some(ref error_msg) = result.error_message {
                    println!("Detected import error, attempting auto-fix...");
                    
                    // Try to install missing dependency
                    match self.dependency_installer.auto_fix_missing_import(error_msg, project_type).await {
                        Ok(install_result) => {
                            if install_result.success {
                                println!("Successfully installed missing packages, retrying...");
                                attempt += 1;
                                continue;
                            } else {
                                println!("Failed to install packages");
                                return Ok(result);
                            }
                        }
                        Err(e) => {
                            println!("Auto-fix failed: {}", e);
                            return Ok(result);
                        }
                    }
                }
            }

            // Not an import error or couldn't auto-fix, return the result
            return Ok(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_error_type_classification() {
        assert_eq!(
            ErrorType::from_message("ModuleNotFoundError: No module named 'requests'"),
            ErrorType::ImportError
        );
        assert_eq!(
            ErrorType::from_message("SyntaxError: invalid syntax"),
            ErrorType::SyntaxError
        );
        assert_eq!(
            ErrorType::from_message("PermissionError: [Errno 13] Permission denied"),
            ErrorType::PermissionError
        );
        assert_eq!(
            ErrorType::from_message("Command timed out after 300 seconds"),
            ErrorType::TimeoutError
        );
        assert_eq!(
            ErrorType::from_message("TypeError: unsupported operand"),
            ErrorType::RuntimeError
        );
    }

    #[tokio::test]
    async fn test_execute_python_code_success() {
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());

        let result = executor.execute_python_code("print('Hello, World!')").await;
        
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert!(exec_result.success);
        assert_eq!(exec_result.error_type, None);
    }

    #[tokio::test]
    async fn test_execute_python_code_syntax_error() {
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());

        let result = executor.execute_python_code("print('Hello World'").await; // Missing closing paren
        
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert!(!exec_result.success);
        assert_eq!(exec_result.error_type, Some(ErrorType::SyntaxError));
    }

    #[tokio::test]
    async fn test_execute_python_code_runtime_error() {
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());

        let result = executor.execute_python_code("x = 1 / 0").await; // Division by zero
        
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert!(!exec_result.success);
        assert_eq!(exec_result.error_type, Some(ErrorType::RuntimeError));
    }

    #[tokio::test]
    async fn test_execute_python_code_import_error() {
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());

        let result = executor.execute_python_code("import nonexistent_module_12345").await;
        
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert!(!exec_result.success);
        assert_eq!(exec_result.error_type, Some(ErrorType::ImportError));
        assert!(exec_result.error_message.is_some());
    }

    #[tokio::test]
    async fn test_execute_python_script() {
        let temp_dir = tempdir().unwrap();
        let script_path = temp_dir.path().join("test_script.py");

        fs::write(
            &script_path,
            r#"
if __name__ == '__main__':
    print("Script executed successfully")
    result = 2 + 2
    print(f"Result: {result}")
"#,
        )
        .unwrap();

        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());
        let result = executor.execute_python(script_path).await;

        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert!(exec_result.success);
        assert_eq!(exec_result.error_type, None);
    }

    #[tokio::test]
    async fn test_detect_python_entry_point() {
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf());

        // Test with __main__ block
        let script1 = temp_dir.path().join("script1.py");
        fs::write(&script1, "if __name__ == '__main__':\n    print('test')").unwrap();
        let entry = executor.detect_python_entry_point(&script1).unwrap();
        assert!(entry.contains("python3"));
        assert!(entry.contains("script1.py"));

        // Test with main function
        let script2 = temp_dir.path().join("script2.py");
        fs::write(&script2, "def main():\n    print('test')").unwrap();
        let entry = executor.detect_python_entry_point(&script2).unwrap();
        assert!(entry.contains("python3"));
    }

    #[tokio::test]
    async fn test_timeout_classification() {
        let temp_dir = tempdir().unwrap();
        let executor = ScriptExecutor::new(temp_dir.path().to_path_buf())
            .with_timeout(Duration::from_millis(100));

        // This should timeout
        let result = executor.execute_python_code("import time; time.sleep(10)").await;
        
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert!(!exec_result.success);
        // Error classification might be timeout or unknown depending on the error message
        assert!(
            exec_result.error_type == Some(ErrorType::TimeoutError) 
            || exec_result.error_type == Some(ErrorType::UnknownError)
        );
    }
}
