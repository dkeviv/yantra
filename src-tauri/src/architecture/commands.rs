// Architecture View System - Tauri Commands
// Purpose: Expose architecture operations to the frontend
// Created: November 28, 2025

use crate::architecture::{
    Architecture, ArchitectureManager, Component, Connection, ConnectionType, 
    Position, ArchitectureVersion
};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

/// Global architecture manager state
pub struct ArchitectureState {
    pub manager: Mutex<ArchitectureManager>,
}

impl ArchitectureState {
    pub fn new() -> Result<Self, String> {
        let manager = ArchitectureManager::new()?;
        Ok(Self {
            manager: Mutex::new(manager),
        })
    }
}

/// Request to create a new architecture
#[derive(Debug, Deserialize)]
pub struct CreateArchitectureRequest {
    pub name: String,
    pub description: String,
}

/// Request to create a component
#[derive(Debug, Deserialize)]
pub struct CreateComponentRequest {
    pub architecture_id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub position: Position,
}

/// Request to update a component
#[derive(Debug, Deserialize)]
pub struct UpdateComponentRequest {
    pub component: Component,
}

/// Request to create a connection
#[derive(Debug, Deserialize)]
pub struct CreateConnectionRequest {
    pub architecture_id: String,
    pub source_id: String,
    pub target_id: String,
    pub connection_type: ConnectionType,
    pub description: String,
}

/// Request to save a version
#[derive(Debug, Deserialize)]
pub struct SaveVersionRequest {
    pub architecture_id: String,
    pub commit_message: String,
}

/// Export format options
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Markdown,
    Mermaid,
    Json,
}

/// Request to export architecture
#[derive(Debug, Deserialize)]
pub struct ExportArchitectureRequest {
    pub architecture_id: String,
    pub format: ExportFormat,
}

/// Response wrapper for success/error handling
#[derive(Debug, Serialize)]
pub struct CommandResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> CommandResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

/// Create a new architecture
#[tauri::command]
pub fn create_architecture(
    state: State<ArchitectureState>,
    request: CreateArchitectureRequest,
) -> CommandResponse<Architecture> {
    let manager = state.manager.lock().unwrap();
    
    match manager.create_architecture(request.name, request.description) {
        Ok(architecture) => CommandResponse::success(architecture),
        Err(e) => CommandResponse::error(e),
    }
}

/// Get architecture by ID
#[tauri::command]
pub fn get_architecture(
    state: State<ArchitectureState>,
    architecture_id: String,
) -> CommandResponse<Architecture> {
    let manager = state.manager.lock().unwrap();
    
    match manager.get_architecture(&architecture_id) {
        Ok(Some(architecture)) => CommandResponse::success(architecture),
        Ok(None) => CommandResponse::error(format!("Architecture {} not found", architecture_id)),
        Err(e) => CommandResponse::error(e),
    }
}

/// Create a component
#[tauri::command]
pub fn create_component(
    state: State<ArchitectureState>,
    request: CreateComponentRequest,
) -> CommandResponse<Component> {
    let manager = state.manager.lock().unwrap();
    
    match manager.create_component(
        &request.architecture_id,
        request.name,
        request.description,
        request.category,
        request.position,
    ) {
        Ok(component) => CommandResponse::success(component),
        Err(e) => CommandResponse::error(e),
    }
}

/// Update a component
#[tauri::command]
pub fn update_component(
    state: State<ArchitectureState>,
    request: UpdateComponentRequest,
) -> CommandResponse<()> {
    let manager = state.manager.lock().unwrap();
    
    match manager.update_component(&request.component) {
        Ok(_) => CommandResponse::success(()),
        Err(e) => CommandResponse::error(e),
    }
}

/// Delete a component
#[tauri::command]
pub fn delete_component(
    state: State<ArchitectureState>,
    component_id: String,
) -> CommandResponse<()> {
    let manager = state.manager.lock().unwrap();
    
    match manager.delete_component(&component_id) {
        Ok(_) => CommandResponse::success(()),
        Err(e) => CommandResponse::error(e),
    }
}

/// Create a connection
#[tauri::command]
pub fn create_connection(
    state: State<ArchitectureState>,
    request: CreateConnectionRequest,
) -> CommandResponse<Connection> {
    let manager = state.manager.lock().unwrap();
    
    match manager.create_connection(
        &request.architecture_id,
        request.source_id,
        request.target_id,
        request.connection_type,
        request.description,
    ) {
        Ok(connection) => CommandResponse::success(connection),
        Err(e) => CommandResponse::error(e),
    }
}

/// Delete a connection
#[tauri::command]
pub fn delete_connection(
    state: State<ArchitectureState>,
    connection_id: String,
) -> CommandResponse<()> {
    let manager = state.manager.lock().unwrap();
    
    match manager.delete_connection(&connection_id) {
        Ok(_) => CommandResponse::success(()),
        Err(e) => CommandResponse::error(e),
    }
}

/// Save architecture version
#[tauri::command]
pub fn save_architecture_version(
    state: State<ArchitectureState>,
    request: SaveVersionRequest,
) -> CommandResponse<ArchitectureVersion> {
    let manager = state.manager.lock().unwrap();
    
    match manager.save_version(&request.architecture_id, request.commit_message) {
        Ok(version) => CommandResponse::success(version),
        Err(e) => CommandResponse::error(e),
    }
}

/// List all versions for an architecture
#[tauri::command]
pub fn list_architecture_versions(
    state: State<ArchitectureState>,
    architecture_id: String,
) -> CommandResponse<Vec<ArchitectureVersion>> {
    let manager = state.manager.lock().unwrap();
    
    match manager.list_versions(&architecture_id) {
        Ok(versions) => CommandResponse::success(versions),
        Err(e) => CommandResponse::error(e),
    }
}

/// Restore architecture to a specific version
#[tauri::command]
pub fn restore_architecture_version(
    state: State<ArchitectureState>,
    version_id: String,
) -> CommandResponse<Architecture> {
    let manager = state.manager.lock().unwrap();
    
    match manager.restore_version(&version_id) {
        Ok(architecture) => CommandResponse::success(architecture),
        Err(e) => CommandResponse::error(e),
    }
}

/// Export architecture to different formats
#[tauri::command]
pub fn export_architecture(
    state: State<ArchitectureState>,
    request: ExportArchitectureRequest,
) -> CommandResponse<String> {
    let manager = state.manager.lock().unwrap();
    
    let architecture = match manager.get_architecture(&request.architecture_id) {
        Ok(Some(arch)) => arch,
        Ok(None) => return CommandResponse::error(format!("Architecture {} not found", request.architecture_id)),
        Err(e) => return CommandResponse::error(e),
    };
    
    let exported = match request.format {
        ExportFormat::Markdown => export_to_markdown(&architecture),
        ExportFormat::Mermaid => export_to_mermaid(&architecture),
        ExportFormat::Json => export_to_json(&architecture),
    };
    
    match exported {
        Ok(content) => CommandResponse::success(content),
        Err(e) => CommandResponse::error(e),
    }
}

/// Export architecture to Markdown format
fn export_to_markdown(architecture: &Architecture) -> Result<String, String> {
    let mut md = String::new();
    
    md.push_str(&format!("# {}\n\n", architecture.name));
    md.push_str(&format!("{}\n\n", architecture.description));
    
    md.push_str("## Components\n\n");
    for component in &architecture.components {
        md.push_str(&format!("### {} {}\n", component.status_indicator(), component.name));
        md.push_str(&format!("- **Category:** {}\n", component.category));
        md.push_str(&format!("- **Status:** {}\n", component.status_text()));
        md.push_str(&format!("- **Description:** {}\n", component.description));
        if !component.files.is_empty() {
            md.push_str("- **Files:**\n");
            for file in &component.files {
                md.push_str(&format!("  - `{}`\n", file));
            }
        }
        md.push_str("\n");
    }
    
    md.push_str("## Connections\n\n");
    for connection in &architecture.connections {
        let source = architecture.components.iter()
            .find(|c| c.id == connection.source_id)
            .map(|c| c.name.as_str())
            .unwrap_or("Unknown");
        let target = architecture.components.iter()
            .find(|c| c.id == connection.target_id)
            .map(|c| c.name.as_str())
            .unwrap_or("Unknown");
        
        md.push_str(&format!("- {} {} {} {}: {}\n",
            source,
            connection.arrow_type(),
            target,
            match connection.connection_type {
                ConnectionType::DataFlow => "(Data Flow)",
                ConnectionType::ApiCall => "(API Call)",
                ConnectionType::Event => "(Event)",
                ConnectionType::Dependency => "(Dependency)",
                ConnectionType::Bidirectional => "(Bidirectional)",
            },
            connection.description
        ));
    }
    
    Ok(md)
}

/// Export architecture to Mermaid diagram format
fn export_to_mermaid(architecture: &Architecture) -> Result<String, String> {
    let mut mermaid = String::from("```mermaid\ngraph LR\n");
    
    // Add components as nodes
    for component in &architecture.components {
        let status = component.status_indicator();
        let label = format!("{} {}", status, component.name);
        let safe_id = component.id.replace("-", "_");
        mermaid.push_str(&format!("    {}[\"{}\"]\n", safe_id, label));
    }
    
    mermaid.push_str("\n");
    
    // Add connections as edges
    for connection in &architecture.connections {
        let safe_source = connection.source_id.replace("-", "_");
        let safe_target = connection.target_id.replace("-", "_");
        let arrow = match connection.connection_type {
            ConnectionType::DataFlow => "-->",
            ConnectionType::ApiCall => "-.->",
            ConnectionType::Event => "==>",
            ConnectionType::Dependency => "..->",
            ConnectionType::Bidirectional => "<-->",
        };
        mermaid.push_str(&format!("    {} {}|{}| {}\n",
            safe_source,
            arrow,
            connection.description,
            safe_target
        ));
    }
    
    mermaid.push_str("```\n");
    Ok(mermaid)
}

/// Export architecture to JSON format
fn export_to_json(architecture: &Architecture) -> Result<String, String> {
    serde_json::to_string_pretty(architecture)
        .map_err(|e| format!("Failed to serialize architecture: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::path::PathBuf;

    fn create_test_state() -> ArchitectureState {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let manager = ArchitectureManager::with_path(db_path).unwrap();
        ArchitectureState {
            manager: Mutex::new(manager),
        }
    }

    #[test]
    fn test_create_architecture_command() {
        let state = create_test_state();
        let request = CreateArchitectureRequest {
            name: "Test App".to_string(),
            description: "Test description".to_string(),
        };
        
        let response = create_architecture(State::from(&state), request);
        assert!(response.success);
        assert!(response.data.is_some());
    }

    #[test]
    fn test_create_component_command() {
        let state = create_test_state();
        
        // First create architecture
        let arch_request = CreateArchitectureRequest {
            name: "Test App".to_string(),
            description: "Test".to_string(),
        };
        let arch_response = create_architecture(State::from(&state), arch_request);
        let arch_id = arch_response.data.unwrap().id;
        
        // Then create component
        let comp_request = CreateComponentRequest {
            architecture_id: arch_id,
            name: "Frontend".to_string(),
            description: "React UI".to_string(),
            category: "frontend".to_string(),
            position: Position { x: 100.0, y: 100.0 },
        };
        
        let response = create_component(State::from(&state), comp_request);
        assert!(response.success);
        assert!(response.data.is_some());
    }

    #[test]
    fn test_export_to_markdown() {
        let mut arch = Architecture::new(
            "arch-1".to_string(),
            "My App".to_string(),
            "Full stack app".to_string(),
        );
        
        let component = Component::new_planned(
            "comp-1".to_string(),
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        );
        arch.add_component(component);
        
        let md = export_to_markdown(&arch).unwrap();
        assert!(md.contains("# My App"));
        assert!(md.contains("## Components"));
        assert!(md.contains("### 📋 Frontend"));
    }

    #[test]
    fn test_export_to_mermaid() {
        let mut arch = Architecture::new(
            "arch-1".to_string(),
            "My App".to_string(),
            "Full stack app".to_string(),
        );
        
        let frontend = Component::new_planned(
            "comp-1".to_string(),
            "Frontend".to_string(),
            "React UI".to_string(),
            "frontend".to_string(),
            Position { x: 100.0, y: 100.0 },
        );
        arch.add_component(frontend);
        
        let mermaid = export_to_mermaid(&arch).unwrap();
        assert!(mermaid.contains("```mermaid"));
        assert!(mermaid.contains("graph LR"));
        assert!(mermaid.contains("comp_1[\"📋 Frontend\"]"));
    }
}
