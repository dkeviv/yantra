// Architecture View System - Core Types
// Purpose: Define data structures for components, connections, and architecture versions
// Created: November 27, 2025

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a component in the architecture (e.g., Frontend, Backend, Database)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Component {
    pub id: String,
    pub name: String,
    pub description: String,
    pub component_type: ComponentType,
    pub category: String, // "frontend", "backend", "database", "external"
    pub position: Position,
    pub files: Vec<String>, // List of file paths belonging to this component
    pub metadata: HashMap<String, String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Component types with visual indicators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentType {
    /// 📋 Planned - No files yet (0/0)
    Planned,
    /// 🔄 InProgress - Some files implemented (e.g., 2/5)
    InProgress { completed: usize, total: usize },
    /// ✅ Implemented - All files complete (e.g., 5/5)
    Implemented { total: usize },
    /// ⚠️ Misaligned - Code doesn't match architecture
    Misaligned { reason: String },
}

/// Position for visual layout in React Flow
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Represents a connection between components
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Connection {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub connection_type: ConnectionType,
    pub description: String,
    pub metadata: HashMap<String, String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Connection types with different visual styling
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    /// → Data flow (solid arrow)
    DataFlow,
    /// ⇢ API call (dashed arrow)
    ApiCall,
    /// ⤳ Event (curved arrow)
    Event,
    /// ⋯> Dependency (dotted arrow)
    Dependency,
    /// ⇄ Bidirectional (double arrow)
    Bidirectional,
}

/// Complete architecture snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    pub id: String,
    pub name: String,
    pub description: String,
    pub components: Vec<Component>,
    pub connections: Vec<Connection>,
    pub metadata: HashMap<String, String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Architecture version for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureVersion {
    pub id: String,
    pub architecture_id: String,
    pub version: i32,
    pub commit_message: String,
    pub snapshot: Architecture,
    pub created_at: i64,
}

/// Mapping between components and their files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentFile {
    pub component_id: String,
    pub file_path: String,
    pub role: String, // "primary", "test", "config", etc.
}

/// Validation result for code-architecture alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_aligned: bool,
    pub misalignments: Vec<Misalignment>,
    pub warnings: Vec<String>,
}

/// Represents a misalignment between code and architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Misalignment {
    pub component_id: String,
    pub misalignment_type: MisalignmentType,
    pub description: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MisalignmentType {
    /// Component has fewer files than expected
    MissingFiles,
    /// Component has extra files not in architecture
    ExtraFiles,
    /// Connection missing in code (no imports)
    MissingConnection,
    /// Connection in code but not in architecture
    ExtraConnection,
    /// Component category doesn't match file location
    WrongCategory,
}

impl Component {
    /// Create a new planned component
    pub fn new_planned(id: String, name: String, description: String, category: String, position: Position) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            name,
            description,
            component_type: ComponentType::Planned,
            category,
            position,
            files: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Update component status based on file count
    pub fn update_status(&mut self, expected_files: usize) {
        let actual_files = self.files.len();
        self.component_type = if actual_files == 0 {
            ComponentType::Planned
        } else if actual_files < expected_files {
            ComponentType::InProgress {
                completed: actual_files,
                total: expected_files,
            }
        } else {
            ComponentType::Implemented {
                total: actual_files,
            }
        };
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Get visual status indicator
    pub fn status_indicator(&self) -> &str {
        match &self.component_type {
            ComponentType::Planned => "📋",
            ComponentType::InProgress { .. } => "🔄",
            ComponentType::Implemented { .. } => "✅",
            ComponentType::Misaligned { .. } => "⚠️",
        }
    }

    /// Get status text for UI
    pub fn status_text(&self) -> String {
        match &self.component_type {
            ComponentType::Planned => "0/0 files".to_string(),
            ComponentType::InProgress { completed, total } => format!("{}/{} files", completed, total),
            ComponentType::Implemented { total } => format!("{}/{} files", total, total),
            ComponentType::Misaligned { reason } => format!("Misaligned: {}", reason),
        }
    }
}

impl Connection {
    /// Create a new connection
    pub fn new(
        id: String,
        source_id: String,
        target_id: String,
        connection_type: ConnectionType,
        description: String,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            source_id,
            target_id,
            connection_type,
            description,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Get arrow type for visualization
    pub fn arrow_type(&self) -> &str {
        match self.connection_type {
            ConnectionType::DataFlow => "→",
            ConnectionType::ApiCall => "⇢",
            ConnectionType::Event => "⤳",
            ConnectionType::Dependency => "⋯>",
            ConnectionType::Bidirectional => "⇄",
        }
    }
}

impl Architecture {
    /// Create a new empty architecture
    pub fn new(id: String, name: String, description: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            name,
            description,
            components: Vec::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a component
    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Add a connection
    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.push(connection);
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Find component by ID
    pub fn get_component(&self, id: &str) -> Option<&Component> {
        self.components.iter().find(|c| c.id == id)
    }

    /// Find component by ID (mutable)
    pub fn get_component_mut(&mut self, id: &str) -> Option<&mut Component> {
        self.components.iter_mut().find(|c| c.id == id)
    }

    /// Remove component and its connections
    pub fn remove_component(&mut self, id: &str) -> bool {
        let initial_len = self.components.len();
        self.components.retain(|c| c.id != id);
        self.connections.retain(|c| c.source_id != id && c.target_id != id);
        self.updated_at = chrono::Utc::now().timestamp();
        self.components.len() < initial_len
    }

    /// Remove connection
    pub fn remove_connection(&mut self, id: &str) -> bool {
        let initial_len = self.connections.len();
        self.connections.retain(|c| c.id != id);
        self.updated_at = chrono::Utc::now().timestamp();
        self.connections.len() < initial_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_creation() {
        let component = Component::new_planned(
            "comp-1".to_string(),
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        );
        assert_eq!(component.name, "Frontend");
        assert_eq!(component.status_indicator(), "📋");
        assert_eq!(component.status_text(), "0/0 files");
    }

    #[test]
    fn test_component_status_updates() {
        let mut component = Component::new_planned(
            "comp-1".to_string(),
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        );

        // Add files
        component.files.push("App.tsx".to_string());
        component.files.push("index.tsx".to_string());
        component.update_status(5);

        assert_eq!(component.status_indicator(), "🔄");
        assert_eq!(component.status_text(), "2/5 files");

        // Complete all files
        component.files.push("store.ts".to_string());
        component.files.push("utils.ts".to_string());
        component.files.push("types.ts".to_string());
        component.update_status(5);

        assert_eq!(component.status_indicator(), "✅");
        assert_eq!(component.status_text(), "5/5 files");
    }

    #[test]
    fn test_connection_creation() {
        let connection = Connection::new(
            "conn-1".to_string(),
            "frontend".to_string(),
            "backend".to_string(),
            ConnectionType::ApiCall,
            "REST API calls".to_string(),
        );
        assert_eq!(connection.arrow_type(), "⇢");
    }

    #[test]
    fn test_architecture_operations() {
        let mut arch = Architecture::new(
            "arch-1".to_string(),
            "My App".to_string(),
            "Full stack app".to_string(),
        );

        // Add component
        let component = Component::new_planned(
            "comp-1".to_string(),
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        );
        arch.add_component(component);
        assert_eq!(arch.components.len(), 1);

        // Find component
        let found = arch.get_component("comp-1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Frontend");

        // Remove component
        let removed = arch.remove_component("comp-1");
        assert!(removed);
        assert_eq!(arch.components.len(), 0);
    }
}
