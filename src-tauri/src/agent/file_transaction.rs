// File: src-tauri/src/agent/file_transaction.rs
// Purpose: File Operations Transaction System with atomic batch writes and rollback
// Last Updated: December 9, 2025

use rusqlite::{params, Connection, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use chrono::Utc;

/// Maximum file path length (Windows: 260, macOS/Linux: 4096)
const MAX_PATH_LENGTH: usize = 255;

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    InProgress,
    Committed,
    RolledBack,
    Failed,
}

/// File operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileOperation {
    Write { path: String, content: String },
    Delete { path: String },
    Move { from: String, to: String },
}

/// Transaction log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLogEntry {
    pub transaction_id: String,
    pub operation_index: usize,
    pub operation_type: String,
    pub target_path: String,
    pub backup_path: Option<String>,
    pub status: TransactionStatus,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub error_message: Option<String>,
}

/// File write request for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFileWriteRequest {
    pub files: Vec<FileWriteEntry>,
    pub atomic: bool,
    pub create_backups: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWriteEntry {
    pub path: String,
    pub content: String,
    pub dependency_order: Option<usize>,
}

/// Transaction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResult {
    pub transaction_id: String,
    pub success: bool,
    pub files_written: usize,
    pub files_backed_up: usize,
    pub message: String,
    pub errors: Vec<String>,
}

/// File transaction manager
pub struct FileTransactionManager {
    db_path: PathBuf,
    backup_dir: PathBuf,
}

impl FileTransactionManager {
    /// Create new transaction manager
    pub fn new(workspace_path: &Path) -> Result<Self, String> {
        let db_path = workspace_path.join(".yantra").join("file_transactions.db");
        let backup_dir = workspace_path.join(".yantra").join("backups");

        // Ensure directories exist
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create .yantra directory: {}", e))?;
        }
        fs::create_dir_all(&backup_dir)
            .map_err(|e| format!("Failed to create backup directory: {}", e))?;

        let manager = Self { db_path, backup_dir };
        manager.init_database()?;
        Ok(manager)
    }

    /// Initialize transaction log database
    fn init_database(&self) -> Result<(), String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open transaction database: {}", e))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS transaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                operation_index INTEGER NOT NULL,
                operation_type TEXT NOT NULL,
                target_path TEXT NOT NULL,
                backup_path TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                UNIQUE(transaction_id, operation_index)
            );
            
            CREATE INDEX IF NOT EXISTS idx_transaction_id ON transaction_log(transaction_id);
            CREATE INDEX IF NOT EXISTS idx_status ON transaction_log(status);
            CREATE INDEX IF NOT EXISTS idx_created_at ON transaction_log(created_at);
            
            CREATE TABLE IF NOT EXISTS transaction_metadata (
                transaction_id TEXT PRIMARY KEY,
                total_operations INTEGER NOT NULL,
                completed_operations INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                rollback_attempted BOOLEAN DEFAULT 0,
                rollback_successful BOOLEAN
            );"
        )
        .map_err(|e| format!("Failed to create transaction tables: {}", e))?;

        Ok(())
    }

    /// Sanitize file path (SVC-024)
    pub fn sanitize_path(&self, path: &str) -> Result<String, String> {
        // Remove invalid characters
        let invalid_chars = ['<', '>', ':', '"', '|', '?', '*'];
        let mut sanitized = path.to_string();
        
        for ch in invalid_chars {
            sanitized = sanitized.replace(ch, "_");
        }

        // Remove control characters
        sanitized = sanitized.chars()
            .filter(|c| !c.is_control())
            .collect();

        // Trim whitespace
        sanitized = sanitized.trim().to_string();

        // Check length (leave room for extensions and backup suffixes)
        if sanitized.len() > MAX_PATH_LENGTH {
            return Err(format!(
                "Path too long: {} characters (max: {})",
                sanitized.len(),
                MAX_PATH_LENGTH
            ));
        }

        // Ensure path doesn't start with / or contain ..
        if sanitized.starts_with('/') || sanitized.contains("..") {
            return Err("Path cannot start with / or contain ..".to_string());
        }

        Ok(sanitized)
    }

    /// Execute batch file write with transaction (SVC-025)
    pub fn batch_write(&self, request: BatchFileWriteRequest) -> Result<TransactionResult, String> {
        let transaction_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        // Sanitize all paths first
        let mut sanitized_files = Vec::new();
        for file in &request.files {
            let sanitized_path = self.sanitize_path(&file.path)?;
            sanitized_files.push(FileWriteEntry {
                path: sanitized_path,
                content: file.content.clone(),
                dependency_order: file.dependency_order,
            });
        }

        // Sort by dependency order if specified
        sanitized_files.sort_by_key(|f| f.dependency_order.unwrap_or(0));

        // Create transaction metadata
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        conn.execute(
            "INSERT INTO transaction_metadata (transaction_id, total_operations, status, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![&transaction_id, sanitized_files.len(), "InProgress", &now],
        ).map_err(|e| format!("Failed to create transaction metadata: {}", e))?;

        let mut files_written = 0;
        let mut files_backed_up = 0;
        let mut errors = Vec::new();
        let mut backups = HashMap::new();

        // Get workspace base path for file existence checks
        let base_path = self.backup_dir.parent()
            .ok_or("Invalid backup directory")?
            .parent()
            .ok_or("Invalid workspace path")?;

        // Execute writes
        for (idx, file) in sanitized_files.iter().enumerate() {
            // Log operation start
            let full_path = base_path.join(&file.path);
            let operation_type = if full_path.exists() {
                "update"
            } else {
                "create"
            };

            conn.execute(
                "INSERT INTO transaction_log 
                 (transaction_id, operation_index, operation_type, target_path, status, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![&transaction_id, idx, operation_type, &file.path, "Pending", &now],
            ).map_err(|e| format!("Failed to log operation: {}", e))?;

            // Create backup if requested and file exists
            let backup_path = if request.create_backups && full_path.exists() {
                match self.create_backup(&file.path) {
                    Ok(backup) => {
                        files_backed_up += 1;
                        backups.insert(file.path.clone(), backup.clone());
                        Some(backup)
                    }
                    Err(e) => {
                        errors.push(format!("Backup failed for {}: {}", file.path, e));
                        if request.atomic {
                            // Rollback on first error in atomic mode
                            self.rollback_transaction(&transaction_id, &backups)?;
                            return Err(format!("Transaction rolled back due to backup failure: {}", e));
                        }
                        None
                    }
                }
            } else {
                None
            };

            // Write file
            match self.write_file_atomic(&file.path, &file.content) {
                Ok(_) => {
                    files_written += 1;
                    
                    // Update log
                    let complete_time = Utc::now().to_rfc3339();
                    conn.execute(
                        "UPDATE transaction_log 
                         SET status = ?1, backup_path = ?2, completed_at = ?3
                         WHERE transaction_id = ?4 AND operation_index = ?5",
                        params!["Committed", backup_path, &complete_time, &transaction_id, idx],
                    ).map_err(|e| format!("Failed to update log: {}", e))?;
                }
                Err(e) => {
                    errors.push(format!("Write failed for {}: {}", file.path, e));
                    
                    // Update log with error
                    conn.execute(
                        "UPDATE transaction_log 
                         SET status = ?1, error_message = ?2
                         WHERE transaction_id = ?3 AND operation_index = ?4",
                        params!["Failed", &e, &transaction_id, idx],
                    ).map_err(|e| format!("Failed to update log: {}", e))?;

                    if request.atomic {
                        // Rollback all changes
                        self.rollback_transaction(&transaction_id, &backups)?;
                        return Err(format!("Transaction rolled back due to write failure: {}", e));
                    }
                }
            }
        }

        // Update transaction metadata
        let final_status = if errors.is_empty() {
            "Committed"
        } else if request.atomic {
            "RolledBack"
        } else {
            "Failed"
        };

        let complete_time = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE transaction_metadata 
             SET completed_operations = ?1, status = ?2, completed_at = ?3
             WHERE transaction_id = ?4",
            params![files_written, final_status, &complete_time, &transaction_id],
        ).map_err(|e| format!("Failed to update transaction metadata: {}", e))?;

        Ok(TransactionResult {
            transaction_id,
            success: errors.is_empty(),
            files_written,
            files_backed_up,
            message: if errors.is_empty() {
                format!("Successfully wrote {} files", files_written)
            } else {
                format!("Wrote {} files with {} errors", files_written, errors.len())
            },
            errors,
        })
    }

    /// Create backup of file
    fn create_backup(&self, file_path: &str) -> Result<String, String> {
        // Convert relative path to absolute within workspace
        let base_path = self.backup_dir.parent()
            .ok_or("Invalid backup directory")?
            .parent()
            .ok_or("Invalid workspace path")?;
        
        let full_path = base_path.join(file_path);
        
        if !full_path.exists() {
            return Err("File does not exist".to_string());
        }

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let file_name = full_path.file_name()
            .ok_or("Invalid file name")?
            .to_string_lossy();
        
        let backup_name = format!("{}_{}", timestamp, file_name);
        let backup_path = self.backup_dir.join(&backup_name);

        fs::copy(&full_path, &backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))?;

        Ok(backup_path.to_string_lossy().to_string())
    }

    /// Atomic file write using temp file + rename
    fn write_file_atomic(&self, file_path: &str, content: &str) -> Result<(), String> {
        // Convert relative path to absolute within workspace
        let base_path = self.backup_dir.parent()
            .ok_or("Invalid backup directory")?
            .parent()
            .ok_or("Invalid workspace path")?;
        
        let full_path = base_path.join(file_path);
        
        // Create parent directories
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directories: {}", e))?;
        }

        // Write to temp file
        let temp_path = format!("{}.tmp", full_path.display());
        fs::write(&temp_path, content)
            .map_err(|e| format!("Failed to write temp file: {}", e))?;

        // Atomic rename
        fs::rename(&temp_path, &full_path)
            .map_err(|e| {
                // Clean up temp file on failure
                let _ = fs::remove_file(&temp_path);
                format!("Failed to rename temp file: {}", e)
            })?;

        Ok(())
    }

    /// Rollback transaction (SVC-028)
    pub fn rollback_transaction(
        &self,
        transaction_id: &str,
        backups: &HashMap<String, String>,
    ) -> Result<(), String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        // Get workspace base path
        let base_path = self.backup_dir.parent()
            .ok_or("Invalid backup directory")?
            .parent()
            .ok_or("Invalid workspace path")?;

        // Mark rollback attempt
        conn.execute(
            "UPDATE transaction_metadata SET rollback_attempted = 1 WHERE transaction_id = ?1",
            params![transaction_id],
        ).map_err(|e| format!("Failed to mark rollback attempt: {}", e))?;

        // Get all operations for this transaction
        let mut stmt = conn
            .prepare(
                "SELECT operation_index, target_path, backup_path, status 
                 FROM transaction_log 
                 WHERE transaction_id = ?1 
                 ORDER BY operation_index DESC", // Reverse order for rollback
            )
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let operations: Vec<(usize, String, Option<String>, String)> = stmt
            .query_map(params![transaction_id], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                ))
            })
            .map_err(|e| format!("Failed to query operations: {}", e))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to collect operations: {}", e))?;

        let mut rollback_errors = Vec::new();

        for (idx, target_path, backup_path, status) in operations {
            // Only rollback committed operations
            if status != "Committed" {
                continue;
            }

            let full_target_path = base_path.join(&target_path);

            // Restore from backup or delete new file
            let result = if let Some(backup) = backup_path.or_else(|| backups.get(&target_path).cloned()) {
                // Restore from backup
                fs::copy(&backup, &full_target_path)
                    .map(|_| ())
                    .map_err(|e| format!("Failed to restore from backup: {}", e))
            } else {
                // Delete newly created file
                fs::remove_file(&full_target_path)
                    .map_err(|e| format!("Failed to delete file: {}", e))
            };

            match result {
                Ok(_) => {
                    conn.execute(
                        "UPDATE transaction_log SET status = ?1 WHERE transaction_id = ?2 AND operation_index = ?3",
                        params!["RolledBack", transaction_id, idx],
                    ).ok();
                }
                Err(e) => {
                    rollback_errors.push(format!("Failed to rollback {}: {}", target_path, e));
                }
            }
        }

        let rollback_successful = rollback_errors.is_empty();
        conn.execute(
            "UPDATE transaction_metadata 
             SET status = ?1, rollback_successful = ?2, completed_at = ?3
             WHERE transaction_id = ?4",
            params![
                "RolledBack",
                rollback_successful,
                Utc::now().to_rfc3339(),
                transaction_id
            ],
        ).map_err(|e| format!("Failed to update metadata: {}", e))?;

        if !rollback_successful {
            return Err(format!("Rollback completed with errors: {:?}", rollback_errors));
        }

        Ok(())
    }

    /// Get transaction log entries (for audit trail)
    pub fn get_transaction_log(&self, transaction_id: &str) -> Result<Vec<TransactionLogEntry>, String> {
        let conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let mut stmt = conn
            .prepare(
                "SELECT transaction_id, operation_index, operation_type, target_path, 
                        backup_path, status, created_at, completed_at, error_message
                 FROM transaction_log 
                 WHERE transaction_id = ?1 
                 ORDER BY operation_index",
            )
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let entries = stmt
            .query_map(params![transaction_id], |row| {
                Ok(TransactionLogEntry {
                    transaction_id: row.get(0)?,
                    operation_index: row.get(1)?,
                    operation_type: row.get(2)?,
                    target_path: row.get(3)?,
                    backup_path: row.get(4)?,
                    status: match row.get::<_, String>(5)?.as_str() {
                        "Pending" => TransactionStatus::Pending,
                        "InProgress" => TransactionStatus::InProgress,
                        "Committed" => TransactionStatus::Committed,
                        "RolledBack" => TransactionStatus::RolledBack,
                        _ => TransactionStatus::Failed,
                    },
                    created_at: row.get(6)?,
                    completed_at: row.get(7)?,
                    error_message: row.get(8)?,
                })
            })
            .map_err(|e| format!("Failed to query log: {}", e))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to collect entries: {}", e))?;

        Ok(entries)
    }

    /// Clean up old backups (older than 30 days)
    pub fn cleanup_old_backups(&self, days: u64) -> Result<usize, String> {
        let cutoff = Utc::now() - chrono::Duration::days(days as i64);
        let mut removed = 0;

        let entries = fs::read_dir(&self.backup_dir)
            .map_err(|e| format!("Failed to read backup directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let metadata = entry.metadata()
                .map_err(|e| format!("Failed to read metadata: {}", e))?;

            if let Ok(modified) = metadata.modified() {
                let modified_time: chrono::DateTime<Utc> = modified.into();
                if modified_time < cutoff {
                    fs::remove_file(entry.path())
                        .map_err(|e| format!("Failed to remove backup: {}", e))?;
                    removed += 1;
                }
            }
        }

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_path_sanitization() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileTransactionManager::new(temp_dir.path()).unwrap();

        // Valid path
        assert!(manager.sanitize_path("src/main.rs").is_ok());

        // Invalid characters
        let sanitized = manager.sanitize_path("src/main<>:*.rs").unwrap();
        assert!(!sanitized.contains('<'));
        assert!(!sanitized.contains('>'));

        // Path traversal attempt
        assert!(manager.sanitize_path("../secret.txt").is_err());

        // Too long path
        let long_path = "a".repeat(300);
        assert!(manager.sanitize_path(&long_path).is_err());
    }

    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileTransactionManager::new(temp_dir.path()).unwrap();
        
        let file_path = temp_dir.path().join("test.txt").to_str().unwrap().to_string();
        
        manager.write_file_atomic(&file_path, "test content").unwrap();
        
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_batch_write_atomic_success() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileTransactionManager::new(temp_dir.path()).unwrap();

        let request = BatchFileWriteRequest {
            files: vec![
                FileWriteEntry {
                    path: "file1.txt".to_string(),  // Relative path
                    content: "content 1".to_string(),
                    dependency_order: Some(1),
                },
                FileWriteEntry {
                    path: "file2.txt".to_string(),  // Relative path
                    content: "content 2".to_string(),
                    dependency_order: Some(2),
                },
            ],
            atomic: true,
            create_backups: false,
        };

        let result = manager.batch_write(request).unwrap();
        assert!(result.success);
        assert_eq!(result.files_written, 2);
        
        // Verify files exist
        assert!(temp_dir.path().join("file1.txt").exists());
        assert!(temp_dir.path().join("file2.txt").exists());
    }

    #[test]
    fn test_batch_write_with_backup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileTransactionManager::new(temp_dir.path()).unwrap();

        let file_path = "existing.txt";
        // Create file in workspace
        fs::write(temp_dir.path().join(file_path), "original content").unwrap();

        let request = BatchFileWriteRequest {
            files: vec![FileWriteEntry {
                path: file_path.to_string(),
                content: "new content".to_string(),
                dependency_order: None,
            }],
            atomic: true,
            create_backups: true,
        };

        let result = manager.batch_write(request).unwrap();
        assert!(result.success);
        assert_eq!(result.files_backed_up, 1);

        let content = fs::read_to_string(temp_dir.path().join(file_path)).unwrap();
        assert_eq!(content, "new content");
    }

    #[test]
    fn test_transaction_log() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileTransactionManager::new(temp_dir.path()).unwrap();

        let request = BatchFileWriteRequest {
            files: vec![FileWriteEntry {
                path: "test.txt".to_string(),
                content: "test".to_string(),
                dependency_order: None,
            }],
            atomic: true,
            create_backups: false,
        };

        let result = manager.batch_write(request).unwrap();
        
        let log = manager.get_transaction_log(&result.transaction_id).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].status, TransactionStatus::Committed);
    }

    #[test]
    fn test_rollback() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileTransactionManager::new(temp_dir.path()).unwrap();

        let file_path = "test.txt";
        let full_path = temp_dir.path().join(file_path);
        
        // Create original file
        fs::write(&full_path, "original").unwrap();

        // Create backup
        let backup = manager.create_backup(file_path).unwrap();
        
        // Modify file
        fs::write(&full_path, "modified").unwrap();

        // Verify modified
        let content = fs::read_to_string(&full_path).unwrap();
        assert_eq!(content, "modified");

        // Now create a transaction and rollback
        let mut backups = HashMap::new();
        backups.insert(file_path.to_string(), backup);

        // Create a fake transaction in the database
        let transaction_id = "test-transaction";
        let conn = Connection::open(&manager.db_path).unwrap();
        conn.execute(
            "INSERT INTO transaction_metadata (transaction_id, total_operations, status, created_at)
             VALUES (?1, 1, 'InProgress', ?2)",
            params![transaction_id, Utc::now().to_rfc3339()],
        ).unwrap();
        
        conn.execute(
            "INSERT INTO transaction_log 
             (transaction_id, operation_index, operation_type, target_path, backup_path, status, created_at)
             VALUES (?1, 0, 'update', ?2, ?3, 'Committed', ?4)",
            params![transaction_id, file_path, backups.get(file_path), Utc::now().to_rfc3339()],
        ).unwrap();

        // Perform rollback
        manager.rollback_transaction(transaction_id, &backups).unwrap();

        // File should be restored
        let content = fs::read_to_string(&full_path).unwrap();
        assert_eq!(content, "original");
    }
}
