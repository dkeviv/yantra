// Architecture View System - SQLite Storage
// Purpose: Persistent storage for architecture components, connections, and versions
// Created: November 27, 2025

use crate::architecture::types::{
    Architecture, Component, ComponentType, Connection, ConnectionType, 
    ArchitectureVersion, Position
};
use rusqlite::{Connection as SqliteConnection, Result as SqliteResult, params};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Architecture storage manager with SQLite backend
pub struct ArchitectureStorage {
    conn: Arc<Mutex<SqliteConnection>>,
}

impl ArchitectureStorage {
    /// Create new storage and initialize database schema
    pub fn new<P: AsRef<Path>>(db_path: P) -> SqliteResult<Self> {
        let conn = SqliteConnection::open(db_path)?;
        
        // Enable WAL mode for better concurrency using execute_batch
        conn.execute_batch("
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
        ")?;
        
        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        
        storage.initialize_schema()?;
        Ok(storage)
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        
        // Use batch_execute for all schema creation to avoid ExecuteReturnedResults error
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS architectures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                metadata TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS components (
                id TEXT PRIMARY KEY,
                architecture_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                component_type TEXT NOT NULL,
                category TEXT NOT NULL,
                position_x REAL NOT NULL,
                position_y REAL NOT NULL,
                metadata TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (architecture_id) REFERENCES architectures(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS connections (
                id TEXT PRIMARY KEY,
                architecture_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                connection_type TEXT NOT NULL,
                description TEXT,
                metadata TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (architecture_id) REFERENCES architectures(id) ON DELETE CASCADE,
                FOREIGN KEY (source_id) REFERENCES components(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES components(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS component_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                role TEXT,
                FOREIGN KEY (component_id) REFERENCES components(id) ON DELETE CASCADE,
                UNIQUE(component_id, file_path)
            );

            CREATE TABLE IF NOT EXISTS architecture_versions (
                id TEXT PRIMARY KEY,
                architecture_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                commit_message TEXT NOT NULL,
                snapshot TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (architecture_id) REFERENCES architectures(id) ON DELETE CASCADE,
                UNIQUE(architecture_id, version)
            );

            CREATE INDEX IF NOT EXISTS idx_components_architecture 
                ON components(architecture_id);
            
            CREATE INDEX IF NOT EXISTS idx_connections_architecture 
                ON connections(architecture_id);
            
            CREATE INDEX IF NOT EXISTS idx_component_files_component 
                ON component_files(component_id);
            
            CREATE INDEX IF NOT EXISTS idx_versions_architecture 
                ON architecture_versions(architecture_id, version DESC);
        ")?;

        Ok(())
    }

    /// Create a new architecture
    pub fn create_architecture(&self, architecture: &Architecture) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        let metadata_json = serde_json::to_string(&architecture.metadata).unwrap_or_default();
        
        conn.execute(
            "INSERT INTO architectures (id, name, description, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &architecture.id,
                &architecture.name,
                &architecture.description,
                &metadata_json,
                architecture.created_at,
                architecture.updated_at,
            ],
        )?;
        
        Ok(())
    }

    /// Get architecture by ID
    pub fn get_architecture(&self, id: &str) -> SqliteResult<Option<Architecture>> {
        let conn = self.conn.lock().unwrap();
        
        // Get base architecture
        let mut stmt = conn.prepare(
            "SELECT id, name, description, metadata, created_at, updated_at 
             FROM architectures WHERE id = ?1"
        )?;
        
        let arch = stmt.query_row(params![id], |row| {
            let metadata_json: String = row.get(3)?;
            let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();
            
            Ok(Architecture {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                components: Vec::new(), // Will be populated below
                connections: Vec::new(), // Will be populated below
                metadata,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
            })
        });
        
        if arch.is_err() {
            return Ok(None);
        }
        
        let mut architecture = arch?;
        
        // Get components
        architecture.components = self.get_components_for_architecture(&architecture.id)?;
        
        // Get connections
        architecture.connections = self.get_connections_for_architecture(&architecture.id)?;
        
        Ok(Some(architecture))
    }

    /// Create a component
    pub fn create_component(&self, architecture_id: &str, component: &Component) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        let component_type_json = serde_json::to_string(&component.component_type).unwrap_or_default();
        let metadata_json = serde_json::to_string(&component.metadata).unwrap_or_default();
        
        conn.execute(
            "INSERT INTO components 
             (id, architecture_id, name, description, component_type, category, 
              position_x, position_y, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &component.id,
                architecture_id,
                &component.name,
                &component.description,
                &component_type_json,
                &component.category,
                component.position.x,
                component.position.y,
                &metadata_json,
                component.created_at,
                component.updated_at,
            ],
        )?;
        
        // Insert component files
        for file in &component.files {
            self.add_component_file(&component.id, file, "primary")?;
        }
        
        Ok(())
    }

    /// Update a component
    pub fn update_component(&self, component: &Component) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        let component_type_json = serde_json::to_string(&component.component_type).unwrap_or_default();
        let metadata_json = serde_json::to_string(&component.metadata).unwrap_or_default();
        
        conn.execute(
            "UPDATE components 
             SET name = ?1, description = ?2, component_type = ?3, category = ?4,
                 position_x = ?5, position_y = ?6, metadata = ?7, updated_at = ?8
             WHERE id = ?9",
            params![
                &component.name,
                &component.description,
                &component_type_json,
                &component.category,
                component.position.x,
                component.position.y,
                &metadata_json,
                component.updated_at,
                &component.id,
            ],
        )?;
        
        // Update component files (simple approach: delete and re-insert)
        conn.execute("DELETE FROM component_files WHERE component_id = ?1", params![&component.id])?;
        for file in &component.files {
            self.add_component_file(&component.id, file, "primary")?;
        }
        
        Ok(())
    }

    /// Delete a component
    pub fn delete_component(&self, component_id: &str) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM components WHERE id = ?1", params![component_id])?;
        Ok(())
    }

    /// Get all components for an architecture
    fn get_components_for_architecture(&self, architecture_id: &str) -> SqliteResult<Vec<Component>> {
        let conn = self.conn.lock().unwrap();
        
        // First, get all components
        let mut stmt = conn.prepare(
            "SELECT id, name, description, component_type, category, 
                    position_x, position_y, metadata, created_at, updated_at
             FROM components WHERE architecture_id = ?1"
        )?;
        
        let component_rows: Vec<_> = stmt.query_map(params![architecture_id], |row| {
            Ok((
                row.get::<_, String>(0)?,  // id
                row.get::<_, String>(1)?,  // name
                row.get::<_, String>(2)?,  // description
                row.get::<_, String>(3)?,  // component_type
                row.get::<_, String>(4)?,  // category
                row.get::<_, f64>(5)?,     // position_x
                row.get::<_, f64>(6)?,     // position_y
                row.get::<_, String>(7)?,  // metadata
                row.get::<_, i64>(8)?,     // created_at
                row.get::<_, i64>(9)?,     // updated_at
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
        
        // Then get files for each component
        let mut components = Vec::new();
        for (id, name, description, component_type_json, category, pos_x, pos_y, metadata_json, created_at, updated_at) in component_rows {
            let component_type = serde_json::from_str(&component_type_json).unwrap_or(ComponentType::Planned);
            let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();
            
            // Get files for this component
            let mut file_stmt = conn.prepare("SELECT file_path FROM component_files WHERE component_id = ?1")?;
            let files: Vec<String> = file_stmt.query_map(params![&id], |row| row.get(0))?
                .collect::<Result<Vec<_>, _>>()?;
            
            components.push(Component {
                id,
                name,
                description,
                component_type,
                category,
                position: Position { x: pos_x, y: pos_y },
                files,
                metadata,
                created_at,
                updated_at,
            });
        }
        
        Ok(components)
    }

    /// Create a connection
    pub fn create_connection(&self, architecture_id: &str, connection: &Connection) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        let connection_type_json = serde_json::to_string(&connection.connection_type).unwrap_or_default();
        let metadata_json = serde_json::to_string(&connection.metadata).unwrap_or_default();
        
        conn.execute(
            "INSERT INTO connections 
             (id, architecture_id, source_id, target_id, connection_type, 
              description, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &connection.id,
                architecture_id,
                &connection.source_id,
                &connection.target_id,
                &connection_type_json,
                &connection.description,
                &metadata_json,
                connection.created_at,
                connection.updated_at,
            ],
        )?;
        
        Ok(())
    }

    /// Delete a connection
    pub fn delete_connection(&self, connection_id: &str) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM connections WHERE id = ?1", params![connection_id])?;
        Ok(())
    }

    /// Get all connections for an architecture
    fn get_connections_for_architecture(&self, architecture_id: &str) -> SqliteResult<Vec<Connection>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, connection_type, description, 
                    metadata, created_at, updated_at
             FROM connections WHERE architecture_id = ?1"
        )?;
        
        let connections = stmt.query_map(params![architecture_id], |row| {
            let connection_type_json: String = row.get(3)?;
            let connection_type = serde_json::from_str(&connection_type_json).unwrap_or(ConnectionType::DataFlow);
            
            let metadata_json: String = row.get(5)?;
            let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();
            
            Ok(Connection {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                connection_type,
                description: row.get(4)?,
                metadata,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        
        Ok(connections)
    }

    /// Add a file to a component
    fn add_component_file(&self, component_id: &str, file_path: &str, role: &str) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO component_files (component_id, file_path, role)
             VALUES (?1, ?2, ?3)",
            params![component_id, file_path, role],
        )?;
        Ok(())
    }

    /// Get all files for a component
    fn get_component_files(&self, component_id: &str) -> SqliteResult<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT file_path FROM component_files WHERE component_id = ?1"
        )?;
        
        let files = stmt.query_map(params![component_id], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(files)
    }

    /// Save architecture version
    pub fn save_version(&self, version: &ArchitectureVersion) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        let snapshot_json = serde_json::to_string(&version.snapshot).unwrap_or_default();
        
        conn.execute(
            "INSERT INTO architecture_versions 
             (id, architecture_id, version, commit_message, snapshot, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &version.id,
                &version.architecture_id,
                version.version,
                &version.commit_message,
                &snapshot_json,
                version.created_at,
            ],
        )?;
        
        Ok(())
    }

    /// Get all versions for an architecture
    pub fn list_versions(&self, architecture_id: &str) -> SqliteResult<Vec<ArchitectureVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, architecture_id, version, commit_message, snapshot, created_at
             FROM architecture_versions 
             WHERE architecture_id = ?1
             ORDER BY version DESC"
        )?;
        
        let versions = stmt.query_map(params![architecture_id], |row| {
            let snapshot_json: String = row.get(4)?;
            let snapshot = serde_json::from_str(&snapshot_json).unwrap_or_else(|_| {
                Architecture::new("".to_string(), "".to_string(), "".to_string())
            });
            
            Ok(ArchitectureVersion {
                id: row.get(0)?,
                architecture_id: row.get(1)?,
                version: row.get(2)?,
                commit_message: row.get(3)?,
                snapshot,
                created_at: row.get(5)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        
        Ok(versions)
    }

    /// Get specific version
    pub fn get_version(&self, version_id: &str) -> SqliteResult<Option<ArchitectureVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, architecture_id, version, commit_message, snapshot, created_at
             FROM architecture_versions WHERE id = ?1"
        )?;
        
        let version = stmt.query_row(params![version_id], |row| {
            let snapshot_json: String = row.get(4)?;
            let snapshot = serde_json::from_str(&snapshot_json).unwrap_or_else(|_| {
                Architecture::new("".to_string(), "".to_string(), "".to_string())
            });
            
            Ok(ArchitectureVersion {
                id: row.get(0)?,
                architecture_id: row.get(1)?,
                version: row.get(2)?,
                commit_message: row.get(3)?,
                snapshot,
                created_at: row.get(5)?,
            })
        });
        
        match version {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Verify database integrity
    pub fn verify_integrity(&self) -> SqliteResult<bool> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("PRAGMA integrity_check")?;
        let result: String = stmt.query_row([], |row| row.get(0))?;
        Ok(result == "ok")
    }

    /// Create backup
    pub fn create_backup<P: AsRef<Path>>(&self, backup_path: P) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        let mut backup_conn = SqliteConnection::open(backup_path)?;
        
        let backup = rusqlite::backup::Backup::new(&*conn, &mut backup_conn)?;
        backup.step(-1)?; // Copy entire database
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_storage_initialization() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = ArchitectureStorage::new(&db_path).unwrap();
        
        // Verify tables were created
        assert!(storage.verify_integrity().unwrap());
    }

    #[test]
    fn test_create_and_get_architecture() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = ArchitectureStorage::new(&db_path).unwrap();
        
        let arch = Architecture::new(
            "arch-1".to_string(),
            "Test App".to_string(),
            "Test description".to_string(),
        );
        
        storage.create_architecture(&arch).unwrap();
        let retrieved = storage.get_architecture("arch-1").unwrap();
        
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "Test App");
    }

    #[test]
    fn test_component_crud() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = ArchitectureStorage::new(&db_path).unwrap();
        
        // Create architecture
        let arch = Architecture::new(
            "arch-1".to_string(),
            "Test App".to_string(),
            "Test description".to_string(),
        );
        storage.create_architecture(&arch).unwrap();
        
        // Create component
        let component = Component::new_planned(
            "comp-1".to_string(),
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        );
        storage.create_component("arch-1", &component).unwrap();
        
        // Retrieve architecture with component
        let retrieved = storage.get_architecture("arch-1").unwrap().unwrap();
        assert_eq!(retrieved.components.len(), 1);
        assert_eq!(retrieved.components[0].name, "Frontend");
        
        // Delete component
        storage.delete_component("comp-1").unwrap();
        let retrieved = storage.get_architecture("arch-1").unwrap().unwrap();
        assert_eq!(retrieved.components.len(), 0);
    }

    #[test]
    fn test_versioning() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = ArchitectureStorage::new(&db_path).unwrap();
        
        // Create architecture
        let arch = Architecture::new(
            "arch-1".to_string(),
            "Test App".to_string(),
            "Test description".to_string(),
        );
        storage.create_architecture(&arch).unwrap();
        
        // Save version
        let version = ArchitectureVersion {
            id: "ver-1".to_string(),
            architecture_id: "arch-1".to_string(),
            version: 1,
            commit_message: "Initial version".to_string(),
            snapshot: arch.clone(),
            created_at: chrono::Utc::now().timestamp(),
        };
        storage.save_version(&version).unwrap();
        
        // List versions
        let versions = storage.list_versions("arch-1").unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].commit_message, "Initial version");
    }
}
