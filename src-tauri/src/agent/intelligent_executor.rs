// Intelligent Executor: Command execution with classification and monitoring
// Purpose: Execute commands with appropriate strategy based on classification

use super::command_classifier::{CommandCategory, CommandClassifier, ExecutionStrategy};
use super::status_emitter::{StatusEmitter, TaskStatus};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    pub command: String,
    pub category: CommandCategory,
    pub timed_out: bool,
}

/// Background task handle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTask {
    pub task_id: String,
    pub command: String,
    pub started_at: String,
    pub status: TaskStatus,
}

/// Intelligent Executor
pub struct IntelligentExecutor {
    status_emitter: Option<Arc<tokio::sync::Mutex<StatusEmitter>>>,
}

impl IntelligentExecutor {
    /// Create new intelligent executor
    pub fn new() -> Self {
        Self {
            status_emitter: None,
        }
    }
    
    /// Create with status emitter for progress tracking
    pub fn with_status_emitter(mut self, emitter: Arc<tokio::sync::Mutex<StatusEmitter>>) -> Self {
        self.status_emitter = Some(emitter);
        self
    }
    
    /// Execute command with intelligent strategy selection
    pub async fn execute(&self, command: &str) -> Result<ExecutionResult, String> {
        // Classify command
        let classification = CommandClassifier::classify(command);
        
        // Choose execution strategy
        match classification.strategy {
            ExecutionStrategy::Synchronous => {
                self.execute_synchronous(command, &classification).await
            }
            ExecutionStrategy::Asynchronous => {
                self.execute_asynchronous(command, &classification).await
            }
            ExecutionStrategy::Background => {
                Err("Background commands should be started with execute_background".to_string())
            }
        }
    }
    
    /// Execute command synchronously (for quick commands)
    async fn execute_synchronous(
        &self,
        command: &str,
        classification: &super::command_classifier::CommandClassification,
    ) -> Result<ExecutionResult, String> {
        let start = Instant::now();
        
        // Emit start event
        if let Some(emitter) = &self.status_emitter {
            let task_id = format!("exec_{}", uuid::Uuid::new_v4());
            let emitter = emitter.lock().await;
            emitter.register_task(task_id.clone(), format!("Executing: {}", command)).await;
        }
        
        // Parse command and args
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty command".to_string());
        }
        
        let cmd = parts[0];
        let args = &parts[1..];
        
        // Execute with optional timeout
        let output = if let Some(timeout_secs) = classification.timeout_seconds {
            let result = timeout(
                Duration::from_secs(timeout_secs),
                async {
                    Command::new(cmd)
                        .args(args)
                        .output()
                        .await
                }
            ).await;
            
            match result {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => return Err(format!("Execution failed: {}", e)),
                Err(_) => {
                    return Ok(ExecutionResult {
                        success: false,
                        exit_code: None,
                        stdout: String::new(),
                        stderr: "Command timed out".to_string(),
                        duration_ms: start.elapsed().as_millis() as u64,
                        command: command.to_string(),
                        category: classification.category.clone(),
                        timed_out: true,
                    });
                }
            }
        } else {
            Command::new(cmd)
                .args(args)
                .output()
                .await
                .map_err(|e| format!("Execution failed: {}", e))?
        };
        
        let duration_ms = start.elapsed().as_millis() as u64;
        
        Ok(ExecutionResult {
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration_ms,
            command: command.to_string(),
            category: classification.category.clone(),
            timed_out: false,
        })
    }
    
    /// Execute command asynchronously with progress updates (for medium/long commands)
    async fn execute_asynchronous(
        &self,
        command: &str,
        classification: &super::command_classifier::CommandClassification,
    ) -> Result<ExecutionResult, String> {
        let start = Instant::now();
        let task_id = format!("exec_{}", uuid::Uuid::new_v4());
        
        // Register task
        if let Some(emitter) = &self.status_emitter {
            let emitter = emitter.lock().await;
            emitter.register_task(task_id.clone(), format!("Executing: {}", command)).await;
        }
        
        // Parse command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty command".to_string());
        }
        
        let cmd = parts[0];
        let args = &parts[1..];
        
        // Start async process
        let mut child = TokioCommand::new(cmd)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn command: {}", e))?;
        
        // Capture stdout
        let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
        let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;
        
        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();
        
        let mut stdout_lines = Vec::new();
        let mut stderr_lines = Vec::new();
        
        // Stream output with progress updates
        let emitter_clone = self.status_emitter.clone();
        let task_id_clone = task_id.clone();
        
        // Read stdout in background
        let stdout_handle = tokio::spawn(async move {
            let mut lines = Vec::new();
            while let Ok(Some(line)) = stdout_reader.next_line().await {
                lines.push(line.clone());
                
                // Update progress every 10 lines
                if lines.len() % 10 == 0 {
                    if let Some(emitter) = &emitter_clone {
                        let emitter = emitter.lock().await;
                        emitter.update_progress(
                            &task_id_clone,
                            50.0, // Arbitrary progress for async
                            Some(format!("Processing... ({} lines)", lines.len()))
                        ).await;
                    }
                }
            }
            lines
        });
        
        // Read stderr in background
        let stderr_handle = tokio::spawn(async move {
            let mut lines = Vec::new();
            while let Ok(Some(line)) = stderr_reader.next_line().await {
                lines.push(line);
            }
            lines
        });
        
        // Wait for process with optional timeout
        let wait_result = if let Some(timeout_secs) = classification.timeout_seconds {
            timeout(
                Duration::from_secs(timeout_secs),
                child.wait()
            ).await
        } else {
            Ok(child.wait().await)
        };
        
        // Collect output
        stdout_lines = stdout_handle.await.map_err(|e| format!("Failed to read stdout: {}", e))?;
        stderr_lines = stderr_handle.await.map_err(|e| format!("Failed to read stderr: {}", e))?;
        
        let duration_ms = start.elapsed().as_millis() as u64;
        
        // Check for timeout
        let (exit_status, timed_out) = match wait_result {
            Ok(Ok(status)) => (Some(status), false),
            Ok(Err(e)) => return Err(format!("Process wait failed: {}", e)),
            Err(_) => {
                // Kill the process on timeout
                let _ = child.kill().await;
                (None, true)
            }
        };
        
        let success = exit_status.as_ref().map(|s| s.success()).unwrap_or(false);
        let exit_code = exit_status.and_then(|s| s.code());
        
        // Complete task
        if let Some(emitter) = &self.status_emitter {
            let emitter = emitter.lock().await;
            if success {
                emitter.complete_task(&task_id).await;
            } else {
                let error = if timed_out {
                    "Command timed out".to_string()
                } else {
                    stderr_lines.join("\n")
                };
                emitter.fail_task(&task_id, error).await;
            }
        }
        
        Ok(ExecutionResult {
            success,
            exit_code,
            stdout: stdout_lines.join("\n"),
            stderr: stderr_lines.join("\n"),
            duration_ms,
            command: command.to_string(),
            category: classification.category.clone(),
            timed_out,
        })
    }
    
    /// Start background command (for dev servers, watch modes)
    pub async fn execute_background(&self, command: &str) -> Result<BackgroundTask, String> {
        let task_id = format!("bg_{}", uuid::Uuid::new_v4());
        
        // Register background task
        if let Some(emitter) = &self.status_emitter {
            let emitter = emitter.lock().await;
            emitter.register_task(task_id.clone(), format!("Background: {}", command)).await;
        }
        
        // Parse command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty command".to_string());
        }
        
        let cmd = parts[0];
        let args = &parts[1..];
        
        // Start background process
        let _child = TokioCommand::new(cmd)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to spawn background command: {}", e))?;
        
        // Note: In production, should track child process ID for termination
        
        Ok(BackgroundTask {
            task_id: task_id.clone(),
            command: command.to_string(),
            started_at: chrono::Utc::now().to_rfc3339(),
            status: TaskStatus::Running,
        })
    }
}

impl Default for IntelligentExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_execute_quick_command() {
        let executor = IntelligentExecutor::new();
        let result = executor.execute("echo hello").await.unwrap();
        
        assert!(result.success);
        assert_eq!(result.category, CommandCategory::Quick);
        assert!(result.stdout.contains("hello"));
        assert!(!result.timed_out);
    }
    
    #[tokio::test]
    async fn test_execute_with_timeout() {
        let executor = IntelligentExecutor::new();
        
        // Command that should timeout (sleep for 10s with 1s timeout via classification)
        let result = executor.execute("sleep 10").await.unwrap();
        
        // Should timeout
        assert!(result.timed_out || !result.success);
    }
    
    #[tokio::test]
    async fn test_command_classification() {
        let executor = IntelligentExecutor::new();
        
        // Quick command
        let result = executor.execute("ls").await.unwrap();
        assert_eq!(result.category, CommandCategory::Quick);
        
        // Would be medium (but don't actually run tests in test)
        // let result = executor.execute("cargo test").await;
        // assert_eq!(result.category, CommandCategory::Medium);
    }
}
