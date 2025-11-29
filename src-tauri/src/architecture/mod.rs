// Architecture View System - Main Module
// Purpose: Coordinate architecture storage, types, and operations
// Created: November 27, 2025

pub mod types;
pub mod storage;
pub mod commands;
pub mod deviation_detector;

pub use types::*;
pub use storage::ArchitectureStorage;
pub use commands::*;
pub use deviation_detector::{DeviationDetector, DeviationCheck, AlignmentResult, Deviation, Severity};

use std::path::PathBuf;

/// Architecture manager - high-level API for architecture operations
pub struct ArchitectureManager {
    storage: ArchitectureStorage,
}

impl ArchitectureManager {
    /// Create new architecture manager with default storage location
    pub fn new() -> Result<Self, String> {
        let storage_path = Self::get_default_storage_path()?;
        let storage = ArchitectureStorage::new(&storage_path)
            .map_err(|e| format!("Failed to initialize storage: {}", e))?;
        
        Ok(Self { storage })
    }

    /// Create architecture manager with custom storage path
    pub fn with_path(path: PathBuf) -> Result<Self, String> {
        let storage = ArchitectureStorage::new(&path)
            .map_err(|e| format!("Failed to initialize storage: {}", e))?;
        
        Ok(Self { storage })
    }

    /// Get default storage path: ~/.yantra/architecture.db
    fn get_default_storage_path() -> Result<PathBuf, String> {
        let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
        let yantra_dir = home.join(".yantra");
        
        // Create .yantra directory if it doesn't exist
        if !yantra_dir.exists() {
            std::fs::create_dir_all(&yantra_dir)
                .map_err(|e| format!("Failed to create .yantra directory: {}", e))?;
        }
        
        Ok(yantra_dir.join("architecture.db"))
    }

    /// Create a new architecture
    pub fn create_architecture(&self, name: String, description: String) -> Result<Architecture, String> {
        let id = uuid::Uuid::new_v4().to_string();
        let architecture = Architecture::new(id, name, description);
        
        self.storage.create_architecture(&architecture)
            .map_err(|e| format!("Failed to create architecture: {}", e))?;
        
        Ok(architecture)
    }

    /// Get architecture by ID
    pub fn get_architecture(&self, id: &str) -> Result<Option<Architecture>, String> {
        self.storage.get_architecture(id)
            .map_err(|e| format!("Failed to get architecture: {}", e))
    }

    /// Create a component
    pub fn create_component(
        &self,
        architecture_id: &str,
        name: String,
        description: String,
        category: String,
        position: Position,
    ) -> Result<Component, String> {
        let id = uuid::Uuid::new_v4().to_string();
        let component = Component::new_planned(id, name, description, category, position);
        
        self.storage.create_component(architecture_id, &component)
            .map_err(|e| format!("Failed to create component: {}", e))?;
        
        Ok(component)
    }

    /// Update a component
    pub fn update_component(&self, component: &Component) -> Result<(), String> {
        self.storage.update_component(component)
            .map_err(|e| format!("Failed to update component: {}", e))
    }

    /// Delete a component
    pub fn delete_component(&self, component_id: &str) -> Result<(), String> {
        self.storage.delete_component(component_id)
            .map_err(|e| format!("Failed to delete component: {}", e))
    }

    /// Create a connection
    pub fn create_connection(
        &self,
        architecture_id: &str,
        source_id: String,
        target_id: String,
        connection_type: ConnectionType,
        description: String,
    ) -> Result<Connection, String> {
        let id = uuid::Uuid::new_v4().to_string();
        let connection = Connection::new(id, source_id, target_id, connection_type, description);
        
        self.storage.create_connection(architecture_id, &connection)
            .map_err(|e| format!("Failed to create connection: {}", e))?;
        
        Ok(connection)
    }

    /// Delete a connection
    pub fn delete_connection(&self, connection_id: &str) -> Result<(), String> {
        self.storage.delete_connection(connection_id)
            .map_err(|e| format!("Failed to delete connection: {}", e))
    }

    /// Save architecture version
    pub fn save_version(
        &self,
        architecture_id: &str,
        commit_message: String,
    ) -> Result<ArchitectureVersion, String> {
        // Get current architecture
        let architecture = self.get_architecture(architecture_id)?
            .ok_or_else(|| format!("Architecture {} not found", architecture_id))?;
        
        // Get current version number
        let versions = self.storage.list_versions(architecture_id)
            .map_err(|e| format!("Failed to list versions: {}", e))?;
        let next_version = versions.first().map(|v| v.version + 1).unwrap_or(1);
        
        // Create version
        let version = ArchitectureVersion {
            id: uuid::Uuid::new_v4().to_string(),
            architecture_id: architecture_id.to_string(),
            version: next_version,
            commit_message,
            snapshot: architecture,
            created_at: chrono::Utc::now().timestamp(),
        };
        
        self.storage.save_version(&version)
            .map_err(|e| format!("Failed to save version: {}", e))?;
        
        Ok(version)
    }

    /// List all versions for an architecture
    pub fn list_versions(&self, architecture_id: &str) -> Result<Vec<ArchitectureVersion>, String> {
        self.storage.list_versions(architecture_id)
            .map_err(|e| format!("Failed to list versions: {}", e))
    }

    /// Restore architecture to a specific version
    pub fn restore_version(&self, version_id: &str) -> Result<Architecture, String> {
        let version = self.storage.get_version(version_id)
            .map_err(|e| format!("Failed to get version: {}", e))?
            .ok_or_else(|| format!("Version {} not found", version_id))?;
        
        // Update architecture with snapshot data
        let architecture = version.snapshot;
        self.storage.create_architecture(&architecture)
            .map_err(|e| format!("Failed to restore architecture: {}", e))?;
        
        Ok(architecture)
    }

    /// Verify database integrity
    pub fn verify_integrity(&self) -> Result<bool, String> {
        self.storage.verify_integrity()
            .map_err(|e| format!("Failed to verify integrity: {}", e))
    }

    /// Create backup of architecture database
    pub fn create_backup(&self, backup_path: PathBuf) -> Result<(), String> {
        self.storage.create_backup(&backup_path)
            .map_err(|e| format!("Failed to create backup: {}", e))
    }
}

impl Default for ArchitectureManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default ArchitectureManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_architecture_manager_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let manager = ArchitectureManager::with_path(db_path).unwrap();
        
        assert!(manager.verify_integrity().unwrap());
    }

    #[test]
    fn test_full_workflow() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let manager = ArchitectureManager::with_path(db_path).unwrap();
        
        // Create architecture
        let arch = manager.create_architecture(
            "My App".to_string(),
            "Full stack application".to_string(),
        ).unwrap();
        
        // Create components
        let frontend = manager.create_component(
            &arch.id,
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        ).unwrap();
        
        let backend = manager.create_component(
            &arch.id,
            "Backend".to_string(),
            "API Server".to_string(),
            "backend".to_string(),
            Position { x: 300.0, y: 100.0 },
        ).unwrap();
        
        // Create connection
        let _connection = manager.create_connection(
            &arch.id,
            frontend.id.clone(),
            backend.id.clone(),
            ConnectionType::ApiCall,
            "REST API calls".to_string(),
        ).unwrap();
        
        // Get architecture
        let retrieved = manager.get_architecture(&arch.id).unwrap().unwrap();
        assert_eq!(retrieved.components.len(), 2);
        assert_eq!(retrieved.connections.len(), 1);
        
        // Save version
        let version = manager.save_version(&arch.id, "Initial architecture".to_string()).unwrap();
        assert_eq!(version.version, 1);
        
        // List versions
        let versions = manager.list_versions(&arch.id).unwrap();
        assert_eq!(versions.len(), 1);
    }
}
