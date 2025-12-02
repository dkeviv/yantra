// File: src-tauri/src/agent/task_queue.rs
// Purpose: Task queue system for tracking agent tasks with persistence
// Created: November 29, 2025

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Task priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// A task in the queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub description: String,
    pub status: TaskStatus,
    pub priority: TaskPriority,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

impl Task {
    /// Create a new pending task
    pub fn new(id: String, description: String, priority: TaskPriority) -> Self {
        Self {
            id,
            description,
            status: TaskStatus::Pending,
            priority,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
            metadata: None,
        }
    }

    /// Mark task as in progress
    pub fn start(&mut self) {
        self.status = TaskStatus::InProgress;
        self.started_at = Some(Utc::now());
    }

    /// Mark task as completed
    pub fn complete(&mut self) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Mark task as failed
    pub fn fail(&mut self, error: String) {
        self.status = TaskStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.error = Some(error);
    }
}

/// Task queue manager
pub struct TaskQueue {
    tasks: Arc<Mutex<Vec<Task>>>,
    storage_path: PathBuf,
}

impl TaskQueue {
    /// Create a new task queue
    pub fn new(storage_path: PathBuf) -> Result<Self, String> {
        let mut queue = Self {
            tasks: Arc::new(Mutex::new(Vec::new())),
            storage_path,
        };

        // Load existing tasks from disk
        queue.load()?;

        Ok(queue)
    }

    /// Add a task to the queue
    pub fn add_task(&self, task: Task) -> Result<(), String> {
        self.tasks.lock().unwrap().push(task);
        self.save()?;
        Ok(())
    }

    /// Get all tasks
    pub fn get_all_tasks(&self) -> Vec<Task> {
        self.tasks.lock().unwrap().clone()
    }

    /// Get pending tasks
    pub fn get_pending_tasks(&self) -> Vec<Task> {
        self.tasks
            .lock()
            .unwrap()
            .iter()
            .filter(|t| t.status == TaskStatus::Pending)
            .cloned()
            .collect()
    }

    /// Get current task (first in-progress task)
    pub fn get_current_task(&self) -> Option<Task> {
        self.tasks
            .lock()
            .unwrap()
            .iter()
            .find(|t| t.status == TaskStatus::InProgress)
            .cloned()
    }

    /// Get task by ID
    pub fn get_task(&self, id: &str) -> Option<Task> {
        self.tasks
            .lock()
            .unwrap()
            .iter()
            .find(|t| t.id == id)
            .cloned()
    }

    /// Update task status
    pub fn update_task_status(&self, id: &str, status: TaskStatus) -> Result<(), String> {
        let mut tasks = self.tasks.lock().unwrap();
        let task = tasks
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or_else(|| format!("Task {} not found", id))?;

        match status {
            TaskStatus::InProgress => task.start(),
            TaskStatus::Completed => task.complete(),
            TaskStatus::Failed => task.fail("Task failed".to_string()),
            TaskStatus::Pending => {
                task.status = TaskStatus::Pending;
                task.started_at = None;
                task.completed_at = None;
                task.error = None;
            }
        }

        drop(tasks);
        self.save()?;
        Ok(())
    }

    /// Complete a task
    pub fn complete_task(&self, id: &str) -> Result<(), String> {
        self.update_task_status(id, TaskStatus::Completed)
    }

    /// Fail a task with error
    pub fn fail_task(&self, id: &str, error: String) -> Result<(), String> {
        let mut tasks = self.tasks.lock().unwrap();
        let task = tasks
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or_else(|| format!("Task {} not found", id))?;

        task.fail(error);
        drop(tasks);
        self.save()?;
        Ok(())
    }

    /// Remove completed tasks older than retention period
    pub fn cleanup_old_tasks(&self, retention_days: i64) -> Result<usize, String> {
        let cutoff = Utc::now() - chrono::Duration::days(retention_days);
        let mut tasks = self.tasks.lock().unwrap();
        
        let original_count = tasks.len();
        tasks.retain(|t| {
            if t.status == TaskStatus::Completed || t.status == TaskStatus::Failed {
                t.completed_at.map_or(true, |completed| completed > cutoff)
            } else {
                true
            }
        });
        
        let removed = original_count - tasks.len();
        drop(tasks);
        
        if removed > 0 {
            self.save()?;
        }
        
        Ok(removed)
    }

    /// Get task statistics
    pub fn get_stats(&self) -> TaskStats {
        let tasks = self.tasks.lock().unwrap();
        
        let total = tasks.len();
        let pending = tasks.iter().filter(|t| t.status == TaskStatus::Pending).count();
        let in_progress = tasks.iter().filter(|t| t.status == TaskStatus::InProgress).count();
        let completed = tasks.iter().filter(|t| t.status == TaskStatus::Completed).count();
        let failed = tasks.iter().filter(|t| t.status == TaskStatus::Failed).count();

        TaskStats {
            total,
            pending,
            in_progress,
            completed,
            failed,
        }
    }

    /// Save tasks to disk
    fn save(&self) -> Result<(), String> {
        let tasks = self.tasks.lock().unwrap();
        let json = serde_json::to_string_pretty(&*tasks)
            .map_err(|e| format!("Failed to serialize tasks: {}", e))?;

        // Ensure directory exists
        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create storage directory: {}", e))?;
        }

        fs::write(&self.storage_path, json)
            .map_err(|e| format!("Failed to write tasks to disk: {}", e))?;

        Ok(())
    }

    /// Load tasks from disk
    fn load(&mut self) -> Result<(), String> {
        if !self.storage_path.exists() {
            return Ok(());
        }

        let json = fs::read_to_string(&self.storage_path)
            .map_err(|e| format!("Failed to read tasks from disk: {}", e))?;

        let loaded_tasks: Vec<Task> = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize tasks: {}", e))?;

        *self.tasks.lock().unwrap() = loaded_tasks;

        Ok(())
    }
}

/// Task statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStats {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_add_task() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        let queue = TaskQueue::new(path).unwrap();

        let task = Task::new(
            "task-1".to_string(),
            "Test task".to_string(),
            TaskPriority::Medium,
        );

        queue.add_task(task).unwrap();
        assert_eq!(queue.get_all_tasks().len(), 1);
    }

    #[test]
    fn test_task_lifecycle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        let queue = TaskQueue::new(path).unwrap();

        let task = Task::new(
            "task-1".to_string(),
            "Test task".to_string(),
            TaskPriority::High,
        );

        queue.add_task(task).unwrap();
        
        // Start task
        queue.update_task_status("task-1", TaskStatus::InProgress).unwrap();
        let current = queue.get_current_task();
        assert!(current.is_some());
        assert_eq!(current.unwrap().status, TaskStatus::InProgress);

        // Complete task
        queue.complete_task("task-1").unwrap();
        let task = queue.get_task("task-1").unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.completed_at.is_some());
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tasks.json");

        // Create queue and add task
        {
            let queue = TaskQueue::new(path.clone()).unwrap();
            let task = Task::new(
                "task-1".to_string(),
                "Persistent task".to_string(),
                TaskPriority::Low,
            );
            queue.add_task(task).unwrap();
        }

        // Recreate queue and verify task persisted
        {
            let queue = TaskQueue::new(path).unwrap();
            assert_eq!(queue.get_all_tasks().len(), 1);
            let task = queue.get_task("task-1").unwrap();
            assert_eq!(task.description, "Persistent task");
        }
    }

    #[test]
    fn test_get_stats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        let queue = TaskQueue::new(path).unwrap();

        // Add various tasks
        queue.add_task(Task::new("1".into(), "Task 1".into(), TaskPriority::High)).unwrap();
        queue.add_task(Task::new("2".into(), "Task 2".into(), TaskPriority::Medium)).unwrap();
        queue.add_task(Task::new("3".into(), "Task 3".into(), TaskPriority::Low)).unwrap();

        queue.update_task_status("1", TaskStatus::InProgress).unwrap();
        queue.complete_task("2").unwrap();

        let stats = queue.get_stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.in_progress, 1);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 0);
    }
}
