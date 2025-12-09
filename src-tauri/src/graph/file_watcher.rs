// File: src-tauri/src/graph/file_watcher.rs
// Purpose: File system watcher for automatic graph updates (DEP-027)
// Last Updated: December 9, 2025
//
// Monitors workspace files for changes and triggers incremental graph updates.
// Uses notify crate for cross-platform file system events.
//
// Performance targets:
// - Event detection: <5ms
// - Debouncing: 100ms (batch multiple changes)
// - Graph update trigger: <50ms

use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time;

/// File change event for graph updates
#[derive(Debug, Clone)]
pub struct FileChangeEvent {
    pub path: PathBuf,
    pub event_type: FileEventType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileEventType {
    Created,
    Modified,
    Deleted,
    Renamed { from: PathBuf, to: PathBuf },
}

/// File watcher for automatic graph synchronization
pub struct FileWatcher {
    workspace_path: PathBuf,
    watcher: Option<RecommendedWatcher>,
    pending_changes: Arc<RwLock<HashSet<PathBuf>>>,
    change_tx: mpsc::UnboundedSender<FileChangeEvent>,
    enabled: Arc<RwLock<bool>>,
}

impl FileWatcher {
    /// Create new file watcher for workspace
    pub fn new(
        workspace_path: PathBuf,
        change_tx: mpsc::UnboundedSender<FileChangeEvent>,
    ) -> Result<Self, String> {
        Ok(Self {
            workspace_path,
            watcher: None,
            pending_changes: Arc::new(RwLock::new(HashSet::new())),
            change_tx,
            enabled: Arc::new(RwLock::new(false)),
        })
    }

    /// Start watching workspace for file changes
    pub fn start(&mut self) -> Result<(), String> {
        let change_tx = self.change_tx.clone();
        let workspace_path = self.workspace_path.clone();
        let pending_changes = self.pending_changes.clone();
        let enabled = self.enabled.clone();

        let mut watcher = RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                if let Ok(event) = result {
                    // Filter out non-source code files
                    if !Self::should_watch_event(&event, &workspace_path) {
                        return;
                    }

                    // Convert notify event to our event type
                    if let Some(file_event) = Self::convert_event(event) {
                        // Add to pending changes for debouncing
                        let pending = pending_changes.clone();
                        let tx = change_tx.clone();
                        let enabled_flag = enabled.clone();

                        tokio::spawn(async move {
                            if !*enabled_flag.read().await {
                                return;
                            }

                            pending.write().await.insert(file_event.path.clone());

                            // Send event immediately (debouncing happens downstream)
                            let _ = tx.send(file_event);
                        });
                    }
                }
            },
            Config::default(),
        )
        .map_err(|e| format!("Failed to create file watcher: {}", e))?;

        // Watch workspace recursively
        watcher
            .watch(&self.workspace_path, RecursiveMode::Recursive)
            .map_err(|e| format!("Failed to watch workspace: {}", e))?;

        self.watcher = Some(watcher);
        
        // Enable watching
        tokio::spawn(async move {
            *enabled.write().await = true;
        });

        Ok(())
    }

    /// Stop watching workspace
    pub async fn stop(&mut self) -> Result<(), String> {
        *self.enabled.write().await = false;
        self.watcher = None;
        Ok(())
    }

    /// Check if file should be watched (filter out non-source files)
    fn should_watch_event(event: &Event, workspace_path: &Path) -> bool {
        for path in &event.paths {
            // Skip hidden files and directories
            if path
                .components()
                .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
            {
                continue;
            }

            // Skip node_modules, target, build directories
            let path_str = path.to_string_lossy();
            if path_str.contains("node_modules")
                || path_str.contains("/target/")
                || path_str.contains("/build/")
                || path_str.contains("/dist/")
                || path_str.contains("/.yantra/")
                || path_str.contains("/__pycache__/")
            {
                continue;
            }

            // Only watch source code files
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy();
                if matches!(
                    ext_str.as_ref(),
                    "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "java" | "kt" 
                    | "c" | "cpp" | "h" | "hpp" | "rb" | "php" | "swift" | "json" 
                    | "toml" | "yaml" | "yml" | "md" | "ydoc"
                ) {
                    return true;
                }
            }
        }

        false
    }

    /// Convert notify event to our event type
    fn convert_event(event: Event) -> Option<FileChangeEvent> {
        match event.kind {
            EventKind::Create(_) => {
                if let Some(path) = event.paths.first() {
                    return Some(FileChangeEvent {
                        path: path.clone(),
                        event_type: FileEventType::Created,
                    });
                }
            }
            EventKind::Modify(_) => {
                if let Some(path) = event.paths.first() {
                    return Some(FileChangeEvent {
                        path: path.clone(),
                        event_type: FileEventType::Modified,
                    });
                }
            }
            EventKind::Remove(_) => {
                if let Some(path) = event.paths.first() {
                    return Some(FileChangeEvent {
                        path: path.clone(),
                        event_type: FileEventType::Deleted,
                    });
                }
            }
            _ => {}
        }

        None
    }

    /// Get pending changes (for batch processing)
    pub async fn get_pending_changes(&self) -> Vec<PathBuf> {
        let changes = self.pending_changes.read().await;
        changes.iter().cloned().collect()
    }

    /// Clear pending changes after processing
    pub async fn clear_pending_changes(&self) {
        self.pending_changes.write().await.clear();
    }
}

/// File change debouncer - batches rapid changes
pub struct FileChangeDebouncer {
    debounce_duration: Duration,
    pending_files: Arc<RwLock<HashSet<PathBuf>>>,
}

impl FileChangeDebouncer {
    /// Create new debouncer with specified duration
    pub fn new(debounce_duration: Duration) -> Self {
        Self {
            debounce_duration,
            pending_files: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Add file to pending batch
    pub async fn add(&self, path: PathBuf) {
        self.pending_files.write().await.insert(path);
    }

    /// Start debounce timer and return batched changes
    pub async fn debounce(&self) -> Vec<PathBuf> {
        time::sleep(self.debounce_duration).await;
        
        let mut files = self.pending_files.write().await;
        let result: Vec<PathBuf> = files.iter().cloned().collect();
        files.clear();
        
        result
    }

    /// Get pending count
    pub async fn pending_count(&self) -> usize {
        self.pending_files.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_file_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let watcher = FileWatcher::new(temp_dir.path().to_path_buf(), tx);
        assert!(watcher.is_ok());
    }

    #[tokio::test]
    async fn test_should_watch_source_files() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![test_file],
            attrs: Default::default(),
        };

        assert!(FileWatcher::should_watch_event(&event, temp_dir.path()));
    }

    #[tokio::test]
    async fn test_ignore_hidden_files() {
        let temp_dir = TempDir::new().unwrap();
        let hidden_file = temp_dir.path().join(".hidden");
        
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![hidden_file],
            attrs: Default::default(),
        };

        assert!(!FileWatcher::should_watch_event(&event, temp_dir.path()));
    }

    #[tokio::test]
    async fn test_ignore_node_modules() {
        let temp_dir = TempDir::new().unwrap();
        let node_file = temp_dir.path().join("node_modules/package/index.js");
        
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![node_file],
            attrs: Default::default(),
        };

        assert!(!FileWatcher::should_watch_event(&event, temp_dir.path()));
    }

    #[tokio::test]
    async fn test_debouncer() {
        let debouncer = FileChangeDebouncer::new(Duration::from_millis(50));
        
        debouncer.add(PathBuf::from("file1.rs")).await;
        debouncer.add(PathBuf::from("file2.rs")).await;
        
        assert_eq!(debouncer.pending_count().await, 2);
        
        let changes = debouncer.debounce().await;
        assert_eq!(changes.len(), 2);
        assert_eq!(debouncer.pending_count().await, 0);
    }
}
