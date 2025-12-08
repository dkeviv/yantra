// File: src-tauri/src/agent/status_emitter.rs
// Purpose: Event system for real-time UI progress updates
// Dependencies: serde, tokio, tauri
// Last Updated: December 3, 2025

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub task_id: String,
    pub status: TaskStatus,
    pub progress_percent: Option<f64>,
    pub message: String,
    pub timestamp: u64,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProgress {
    pub task_id: String,
    pub task_name: String,
    pub status: TaskStatus,
    pub progress_percent: f64,
    pub current_step: Option<String>,
    pub total_steps: Option<usize>,
    pub completed_steps: Option<usize>,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub error_message: Option<String>,
}

pub struct StatusEmitter {
    tasks: Arc<RwLock<HashMap<String, TaskProgress>>>,
    events: Arc<RwLock<Vec<ProgressEvent>>>,
}

impl StatusEmitter {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new task
    pub async fn register_task(&self, task_id: String, task_name: String) {
        let task = TaskProgress {
            task_id: task_id.clone(),
            task_name,
            status: TaskStatus::Pending,
            progress_percent: 0.0,
            current_step: None,
            total_steps: None,
            completed_steps: None,
            start_time: Self::current_timestamp(),
            end_time: None,
            error_message: None,
        };

        let mut tasks = self.tasks.write().await;
        tasks.insert(task_id.clone(), task);

        self.emit_event(ProgressEvent {
            task_id,
            status: TaskStatus::Pending,
            progress_percent: Some(0.0),
            message: "Task registered".to_string(),
            timestamp: Self::current_timestamp(),
            metadata: None,
        })
        .await;
    }

    /// Update task status
    pub async fn update_status(&self, task_id: &str, status: TaskStatus, message: String) {
        let mut tasks = self.tasks.write().await;
        
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = status.clone();
            
            if status == TaskStatus::Running && task.start_time == 0 {
                task.start_time = Self::current_timestamp();
            }
            
            if matches!(status, TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled) {
                task.end_time = Some(Self::current_timestamp());
                task.progress_percent = if status == TaskStatus::Completed { 100.0 } else { task.progress_percent };
            }
        }

        self.emit_event(ProgressEvent {
            task_id: task_id.to_string(),
            status,
            progress_percent: None,
            message,
            timestamp: Self::current_timestamp(),
            metadata: None,
        })
        .await;
    }

    /// Update task progress
    pub async fn update_progress(
        &self,
        task_id: &str,
        progress_percent: f64,
        current_step: Option<String>,
    ) {
        let mut tasks = self.tasks.write().await;
        
        if let Some(task) = tasks.get_mut(task_id) {
            task.progress_percent = progress_percent.clamp(0.0, 100.0);
            task.current_step = current_step.clone();
            
            if let (Some(completed), Some(total)) = (task.completed_steps, task.total_steps) {
                task.progress_percent = (completed as f64 / total as f64) * 100.0;
            }
        }

        let message = current_step.clone().unwrap_or_else(|| format!("Progress: {}%", progress_percent as u32));

        self.emit_event(ProgressEvent {
            task_id: task_id.to_string(),
            status: TaskStatus::Running,
            progress_percent: Some(progress_percent),
            message,
            timestamp: Self::current_timestamp(),
            metadata: None,
        })
        .await;
    }

    /// Update step progress
    pub async fn update_steps(&self, task_id: &str, completed_steps: usize, total_steps: usize) {
        let mut tasks = self.tasks.write().await;
        
        if let Some(task) = tasks.get_mut(task_id) {
            task.completed_steps = Some(completed_steps);
            task.total_steps = Some(total_steps);
            task.progress_percent = (completed_steps as f64 / total_steps as f64) * 100.0;
        }

        let progress_percent = (completed_steps as f64 / total_steps as f64) * 100.0;

        self.emit_event(ProgressEvent {
            task_id: task_id.to_string(),
            status: TaskStatus::Running,
            progress_percent: Some(progress_percent),
            message: format!("Step {}/{}", completed_steps, total_steps),
            timestamp: Self::current_timestamp(),
            metadata: None,
        })
        .await;
    }

    /// Mark task as failed
    pub async fn fail_task(&self, task_id: &str, error_message: String) {
        let mut tasks = self.tasks.write().await;
        
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = TaskStatus::Failed;
            task.error_message = Some(error_message.clone());
            task.end_time = Some(Self::current_timestamp());
        }

        self.emit_event(ProgressEvent {
            task_id: task_id.to_string(),
            status: TaskStatus::Failed,
            progress_percent: None,
            message: error_message,
            timestamp: Self::current_timestamp(),
            metadata: None,
        })
        .await;
    }

    /// Mark task as completed
    pub async fn complete_task(&self, task_id: &str) {
        let mut tasks = self.tasks.write().await;
        
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = TaskStatus::Completed;
            task.progress_percent = 100.0;
            task.end_time = Some(Self::current_timestamp());
        }

        self.emit_event(ProgressEvent {
            task_id: task_id.to_string(),
            status: TaskStatus::Completed,
            progress_percent: Some(100.0),
            message: "Task completed successfully".to_string(),
            timestamp: Self::current_timestamp(),
            metadata: None,
        })
        .await;
    }

    /// Get task progress
    pub async fn get_task_progress(&self, task_id: &str) -> Option<TaskProgress> {
        let tasks = self.tasks.read().await;
        tasks.get(task_id).cloned()
    }

    /// Get all tasks
    pub async fn get_all_tasks(&self) -> Vec<TaskProgress> {
        let tasks = self.tasks.read().await;
        tasks.values().cloned().collect()
    }

    /// Get recent events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<ProgressEvent> {
        let events = self.events.read().await;
        events.iter().rev().take(limit).cloned().collect()
    }

    /// Clear completed tasks
    pub async fn clear_completed(&self) {
        let mut tasks = self.tasks.write().await;
        tasks.retain(|_, task| !matches!(task.status, TaskStatus::Completed | TaskStatus::Cancelled));
    }

    /// Internal: Emit event
    async fn emit_event(&self, event: ProgressEvent) {
        let mut events = self.events.write().await;
        events.push(event);
        
        // Keep only last 1000 events
        if events.len() > 1000 {
            let new_start = events.len() - 1000;
            events.drain(0..new_start);
        }
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}

impl Default for StatusEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_task() {
        let emitter = StatusEmitter::new();
        emitter.register_task("task1".to_string(), "Test Task".to_string()).await;
        
        let task = emitter.get_task_progress("task1").await;
        assert!(task.is_some());
        assert_eq!(task.unwrap().status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn test_update_progress() {
        let emitter = StatusEmitter::new();
        emitter.register_task("task1".to_string(), "Test Task".to_string()).await;
        emitter.update_progress("task1", 50.0, Some("Halfway done".to_string())).await;
        
        let task = emitter.get_task_progress("task1").await.unwrap();
        assert_eq!(task.progress_percent, 50.0);
        assert_eq!(task.current_step, Some("Halfway done".to_string()));
    }

    #[tokio::test]
    async fn test_complete_task() {
        let emitter = StatusEmitter::new();
        emitter.register_task("task1".to_string(), "Test Task".to_string()).await;
        emitter.complete_task("task1").await;
        
        let task = emitter.get_task_progress("task1").await.unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(task.progress_percent, 100.0);
    }

    #[tokio::test]
    async fn test_fail_task() {
        let emitter = StatusEmitter::new();
        emitter.register_task("task1".to_string(), "Test Task".to_string()).await;
        emitter.fail_task("task1", "Something went wrong".to_string()).await;
        
        let task = emitter.get_task_progress("task1").await.unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.error_message, Some("Something went wrong".to_string()));
    }

    #[tokio::test]
    async fn test_clear_completed() {
        let emitter = StatusEmitter::new();
        emitter.register_task("task1".to_string(), "Task 1".to_string()).await;
        emitter.register_task("task2".to_string(), "Task 2".to_string()).await;
        emitter.complete_task("task1").await;
        
        emitter.clear_completed().await;
        
        let tasks = emitter.get_all_tasks().await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "task2");
    }
}
