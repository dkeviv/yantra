// Database Migration Manager: Schema versioning and migrations
// Purpose: Track and apply database schema changes with rollback support

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

/// Migration direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MigrationDirection {
    Up,   // Apply migration
    Down, // Rollback migration
}

/// Migration file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    pub version: u32,
    pub name: String,
    pub up_sql: String,
    pub down_sql: String,
    pub applied_at: Option<String>,
}

/// Migration history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationHistory {
    pub version: u32,
    pub name: String,
    pub applied_at: String,
}

/// Migration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    pub success: bool,
    pub version: u32,
    pub direction: MigrationDirection,
    pub message: String,
    pub error: Option<String>,
}

/// Migration Manager
pub struct MigrationManager {
    migrations_dir: PathBuf,
    migrations: HashMap<u32, Migration>,
}

impl MigrationManager {
    /// Create new migration manager
    pub fn new(migrations_dir: PathBuf) -> Result<Self, String> {
        if !migrations_dir.exists() {
            fs::create_dir_all(&migrations_dir)
                .map_err(|e| format!("Failed to create migrations directory: {}", e))?;
        }
        
        let mut manager = Self {
            migrations_dir,
            migrations: HashMap::new(),
        };
        
        manager.load_migrations()?;
        Ok(manager)
    }
    
    /// Load all migrations from directory
    fn load_migrations(&mut self) -> Result<(), String> {
        let entries = fs::read_dir(&self.migrations_dir)
            .map_err(|e| format!("Failed to read migrations directory: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("sql") {
                if let Some(migration) = self.parse_migration_file(&path)? {
                    self.migrations.insert(migration.version, migration);
                }
            }
        }
        
        Ok(())
    }
    
    /// Parse migration file
    fn parse_migration_file(&self, path: &Path) -> Result<Option<Migration>, String> {
        let filename = path.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| "Invalid filename".to_string())?;
        
        // Expected format: 001_create_users_table.sql
        let parts: Vec<&str> = filename.splitn(2, '_').collect();
        if parts.len() != 2 {
            return Ok(None); // Skip invalid files
        }
        
        let version: u32 = parts[0].parse()
            .map_err(|_| format!("Invalid version number: {}", parts[0]))?;
        let name = parts[1].to_string();
        
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read migration file: {}", e))?;
        
        // Split by -- DOWN marker
        let sections: Vec<&str> = content.split("-- DOWN").collect();
        
        let up_sql = sections[0]
            .trim_start_matches("-- UP")
            .trim()
            .to_string();
        
        let down_sql = if sections.len() > 1 {
            sections[1].trim().to_string()
        } else {
            String::new()
        };
        
        Ok(Some(Migration {
            version,
            name,
            up_sql,
            down_sql,
            applied_at: None,
        }))
    }
    
    /// Get all pending migrations
    pub fn get_pending_migrations(&self, current_version: u32) -> Vec<&Migration> {
        let mut pending: Vec<_> = self.migrations.values()
            .filter(|m| m.version > current_version)
            .collect();
        
        pending.sort_by_key(|m| m.version);
        pending
    }
    
    /// Get migration by version
    pub fn get_migration(&self, version: u32) -> Option<&Migration> {
        self.migrations.get(&version)
    }
    
    /// Get all migrations
    pub fn get_all_migrations(&self) -> Vec<&Migration> {
        let mut all: Vec<_> = self.migrations.values().collect();
        all.sort_by_key(|m| m.version);
        all
    }
    
    /// Create new migration file
    pub fn create_migration(&self, name: &str) -> Result<PathBuf, String> {
        // Find next version number
        let next_version = self.migrations.keys()
            .max()
            .map(|v| v + 1)
            .unwrap_or(1);
        
        let filename = format!("{:03}_{}.sql", next_version, name);
        let path = self.migrations_dir.join(&filename);
        
        let template = format!(
            "-- UP\n-- Add your UP migration SQL here\n\n\n-- DOWN\n-- Add your DOWN rollback SQL here\n"
        );
        
        fs::write(&path, template)
            .map_err(|e| format!("Failed to create migration file: {}", e))?;
        
        Ok(path)
    }
    
    /// Validate migration (check SQL syntax - simplified)
    pub fn validate_migration(&self, version: u32) -> Result<bool, String> {
        let migration = self.get_migration(version)
            .ok_or_else(|| format!("Migration {} not found", version))?;
        
        // Basic validation
        if migration.up_sql.is_empty() {
            return Err("UP SQL is empty".to_string());
        }
        
        // Check for dangerous operations in DOWN
        let dangerous_keywords = vec!["DROP TABLE", "DROP DATABASE", "TRUNCATE"];
        for keyword in dangerous_keywords {
            if migration.down_sql.to_uppercase().contains(keyword) {
                // Warning, but not error
                eprintln!("Warning: DOWN migration contains dangerous operation: {}", keyword);
            }
        }
        
        Ok(true)
    }
}

/// Migration executor (simplified - would integrate with database::connection_manager)
pub struct MigrationExecutor {
    manager: MigrationManager,
}

impl MigrationExecutor {
    /// Create new migration executor
    pub fn new(migrations_dir: PathBuf) -> Result<Self, String> {
        Ok(Self {
            manager: MigrationManager::new(migrations_dir)?,
        })
    }
    
    /// Apply migration
    pub fn migrate_up(&self, version: u32) -> Result<MigrationResult, String> {
        let migration = self.manager.get_migration(version)
            .ok_or_else(|| format!("Migration {} not found", version))?;
        
        // Validate first
        self.manager.validate_migration(version)?;
        
        // In production, execute migration.up_sql against database
        // For now, just return success
        Ok(MigrationResult {
            success: true,
            version,
            direction: MigrationDirection::Up,
            message: format!("Applied migration {}: {}", version, migration.name),
            error: None,
        })
    }
    
    /// Rollback migration
    pub fn migrate_down(&self, version: u32) -> Result<MigrationResult, String> {
        let migration = self.manager.get_migration(version)
            .ok_or_else(|| format!("Migration {} not found", version))?;
        
        if migration.down_sql.is_empty() {
            return Err("DOWN migration is empty - cannot rollback".to_string());
        }
        
        // In production, execute migration.down_sql against database
        Ok(MigrationResult {
            success: true,
            version,
            direction: MigrationDirection::Down,
            message: format!("Rolled back migration {}: {}", version, migration.name),
            error: None,
        })
    }
    
    /// Migrate to specific version
    pub fn migrate_to(&self, target_version: u32, current_version: u32) -> Result<Vec<MigrationResult>, String> {
        let mut results = Vec::new();
        
        if target_version > current_version {
            // Migrate up
            let pending = self.manager.get_pending_migrations(current_version);
            for migration in pending {
                if migration.version <= target_version {
                    results.push(self.migrate_up(migration.version)?);
                }
            }
        } else if target_version < current_version {
            // Migrate down
            let mut versions: Vec<_> = self.manager.get_all_migrations()
                .iter()
                .filter(|m| m.version > target_version && m.version <= current_version)
                .map(|m| m.version)
                .collect();
            
            versions.sort_by(|a, b| b.cmp(a)); // Descending order
            
            for version in versions {
                results.push(self.migrate_down(version)?);
            }
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;
    
    #[test]
    fn test_create_migration() {
        let temp_dir = tempdir().unwrap();
        let manager = MigrationManager::new(temp_dir.path().to_path_buf()).unwrap();
        
        let path = manager.create_migration("create_users").unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("001_create_users.sql"));
    }
    
    #[test]
    fn test_parse_migration() {
        let temp_dir = tempdir().unwrap();
        let migration_path = temp_dir.path().join("001_test.sql");
        
        let content = "-- UP\nCREATE TABLE users (id INT);\n\n-- DOWN\nDROP TABLE users;";
        std::fs::write(&migration_path, content).unwrap();
        
        let manager = MigrationManager::new(temp_dir.path().to_path_buf()).unwrap();
        let migration = manager.get_migration(1).unwrap();
        
        assert_eq!(migration.version, 1);
        assert_eq!(migration.name, "test");
        assert!(migration.up_sql.contains("CREATE TABLE"));
        assert!(migration.down_sql.contains("DROP TABLE"));
    }
    
    #[test]
    fn test_get_pending_migrations() {
        let temp_dir = tempdir().unwrap();
        
        // Create multiple migrations
        for i in 1..=3 {
            let path = temp_dir.path().join(format!("{:03}_migration.sql", i));
            let content = format!("-- UP\nSELECT {};\n\n-- DOWN\n-- rollback", i);
            std::fs::write(&path, content).unwrap();
        }
        
        let manager = MigrationManager::new(temp_dir.path().to_path_buf()).unwrap();
        let pending = manager.get_pending_migrations(1);
        
        assert_eq!(pending.len(), 2); // Versions 2 and 3
        assert_eq!(pending[0].version, 2);
        assert_eq!(pending[1].version, 3);
    }
    
    #[test]
    fn test_migration_executor() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("001_test.sql");
        
        let content = "-- UP\nCREATE TABLE test;\n\n-- DOWN\nDROP TABLE test;";
        std::fs::write(&path, content).unwrap();
        
        let executor = MigrationExecutor::new(temp_dir.path().to_path_buf()).unwrap();
        
        let result = executor.migrate_up(1).unwrap();
        assert!(result.success);
        assert_eq!(result.direction, MigrationDirection::Up);
        
        let result = executor.migrate_down(1).unwrap();
        assert!(result.success);
        assert_eq!(result.direction, MigrationDirection::Down);
    }
}
