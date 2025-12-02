// Architecture Analyzer - Generate architecture from existing code using GNN
// Purpose: Analyze codebase dependencies and infer architecture components
// Created: November 28, 2025

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::gnn::GNNEngine;
use super::types::{
    Architecture, Component, ComponentType, Connection, ConnectionType, Position,
};

/// Analyzer that generates architecture from code using GNN
pub struct ArchitectureAnalyzer {
    gnn: Arc<tokio::sync::Mutex<GNNEngine>>,
}

/// Component grouping based on directory structure
#[derive(Debug, Clone)]
struct FileGroup {
    name: String,
    files: Vec<PathBuf>,
    directory: PathBuf,
    category: String,
}

impl ArchitectureAnalyzer {
    /// Create new architecture analyzer
    pub fn new(gnn: Arc<tokio::sync::Mutex<GNNEngine>>) -> Self {
        Self { gnn }
    }

    /// Generate architecture from existing codebase
    pub async fn generate_from_code(
        &self,
        project_root: &Path,
    ) -> Result<Architecture, String> {
        // 1. Group files into logical components
        let file_groups = self.group_files_by_structure(project_root).await?;

        // 2. Convert file groups to components
        let components = self.file_groups_to_components(&file_groups)?;

        // 3. Infer connections from GNN dependencies
        let connections = self.infer_connections(&components).await?;

        // 4. Create architecture
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().timestamp();

        Ok(Architecture {
            id,
            name: "Analyzed Architecture".to_string(),
            description: format!("Auto-generated from codebase at {}", project_root.display()),
            components,
            connections,
            metadata: {
                let mut map = HashMap::new();
                map.insert("generation_method".to_string(), "gnn_analysis".to_string());
                map.insert("project_root".to_string(), project_root.to_string_lossy().to_string());
                map
            },
            created_at: timestamp,
            updated_at: timestamp,
        })
    }

    /// Group files by directory structure into logical components
    async fn group_files_by_structure(&self, project_root: &Path) -> Result<Vec<FileGroup>, String> {
        let gnn = self.gnn.lock().await;
        
        // Get all files from GNN
        let nodes = gnn.get_graph().get_all_nodes();
        let all_files: Vec<PathBuf> = nodes.iter()
            .filter(|n| !n.file_path.is_empty())
            .map(|n| PathBuf::from(&n.file_path))
            .collect();

        let mut groups: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

        // Group files by their parent directory (one level deep)
        for file in all_files {
            if let Some(parent) = file.parent() {
                // Skip if file is directly in project root
                if parent == project_root {
                    continue;
                }

                // Get the first-level subdirectory
                let relative = file.strip_prefix(project_root)
                    .map_err(|e| format!("Failed to get relative path: {}", e))?;
                
                let first_dir = relative.components()
                    .next()
                    .and_then(|c| c.as_os_str().to_str())
                    .ok_or("Failed to get first directory")?;

                let group_dir = project_root.join(first_dir);
                groups.entry(group_dir.clone()).or_insert_with(Vec::new).push(file);
            }
        }

        // Convert to FileGroups with inferred metadata
        let mut file_groups = Vec::new();
        for (dir, files) in groups {
            let name = dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .to_string();

            let category = self.infer_category(&name, &files);

            file_groups.push(FileGroup {
                name: Self::format_component_name(&name),
                files,
                directory: dir,
                category,
            });
        }

        Ok(file_groups)
    }

    /// Infer category from directory name and files
    fn infer_category(&self, dir_name: &str, files: &[PathBuf]) -> String {
        let name_lower = dir_name.to_lowercase();

        // Check for common patterns
        if name_lower.contains("frontend") || name_lower.contains("ui") || name_lower.contains("client") {
            return "frontend".to_string();
        }

        if name_lower.contains("backend") || name_lower.contains("api") || name_lower.contains("server") {
            return "backend".to_string();
        }

        if name_lower.contains("database") || name_lower.contains("db") || name_lower.contains("storage") {
            return "database".to_string();
        }

        if name_lower.contains("external") || name_lower.contains("third-party") {
            return "external".to_string();
        }

        // Check file extensions
        let has_ui_files = files.iter().any(|f| {
            f.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e, "tsx" | "jsx" | "html" | "vue" | "svelte"))
                .unwrap_or(false)
        });

        if has_ui_files {
            return "frontend".to_string();
        }

        // Default to backend
        "backend".to_string()
    }

    /// Format component name (capitalize words, remove underscores)
    fn format_component_name(name: &str) -> String {
        name.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().collect::<String>() + chars.as_str()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Convert file groups to architecture components
    fn file_groups_to_components(&self, groups: &[FileGroup]) -> Result<Vec<Component>, String> {
        let mut components = Vec::new();
        let timestamp = chrono::Utc::now().timestamp();

        for (index, group) in groups.iter().enumerate() {
            let component_id = format!("comp_{}", index + 1);

            // Calculate position in grid layout
            let x = (index % 3) as f64 * 250.0 + 50.0;
            let y = (index / 3) as f64 * 150.0 + 50.0;

            // Determine status based on file count
            let component_type = if group.files.is_empty() {
                ComponentType::Planned
            } else {
                ComponentType::Implemented { total: group.files.len() }
            };

            let component = Component {
                id: component_id,
                name: group.name.clone(),
                description: format!(
                    "{} component with {} files",
                    group.name,
                    group.files.len()
                ),
                component_type,
                category: group.category.clone(),
                position: Position { x, y },
                files: group
                    .files
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect(),
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("directory".to_string(), group.directory.to_string_lossy().to_string());
                    map.insert("file_count".to_string(), group.files.len().to_string());
                    map
                },
                created_at: timestamp,
                updated_at: timestamp,
            };

            components.push(component);
        }

        Ok(components)
    }

    /// Infer connections between components based on GNN dependencies
    async fn infer_connections(&self, components: &[Component]) -> Result<Vec<Connection>, String> {
        let gnn = self.gnn.lock().await;
        let mut connections = Vec::new();
        let mut connection_set = HashSet::new();
        let timestamp = chrono::Utc::now().timestamp();

        // Get all nodes from GNN graph and iterate through their edges
        let graph = gnn.get_graph();
        let nodes = graph.get_all_nodes();

        for source_node in nodes {
            // Get all outgoing edges from this node (imports, calls, etc.)
            let calls = graph.get_outgoing_edges(&source_node.id, crate::gnn::EdgeType::Calls);
            let imports = graph.get_outgoing_edges(&source_node.id, crate::gnn::EdgeType::Imports);
            
            // Combine all edge types
            let all_edges = calls.into_iter().chain(imports.into_iter());

            for edge in all_edges {
                // Find which components own the source and target nodes
                let source_comp = components.iter().find(|c| {
                    c.files.iter().any(|f| f == &source_node.file_path)
                });
                
                // Find target node (need to store nodes to avoid temporary value issue)
                let all_nodes = graph.get_all_nodes();
                let target_node = all_nodes.iter().find(|n| n.id == edge.target_id);
                
                let target_comp = if let Some(target) = target_node {
                    components.iter().find(|c| {
                        c.files.iter().any(|f| f == &target.file_path)
                    })
                } else {
                    None
                };

                if let (Some(source), Some(target)) = (source_comp, target_comp) {
                    // Don't create self-connections
                    if source.id == target.id {
                        continue;
                    }

                    // Create unique connection key to avoid duplicates
                    let connection_key = format!("{}->{}", source.id, target.id);
                    if connection_set.contains(&connection_key) {
                        continue;
                    }
                    connection_set.insert(connection_key.clone());

                    // Infer connection type based on categories
                    let connection_type = Self::infer_connection_type(
                        &source.category,
                        &target.category,
                    );

                    connections.push(Connection {
                        id: format!("conn_{}", connections.len() + 1),
                        source_id: source.id.clone(),
                        target_id: target.id.clone(),
                        connection_type,
                        description: format!("Dependency from {} to {}", source.name, target.name),
                        metadata: HashMap::new(),
                        created_at: timestamp,
                        updated_at: timestamp,
                    });
                }
            }
        }

        Ok(connections)
    }

    /// Infer connection type based on component categories
    fn infer_connection_type(
        source_category: &str,
        target_category: &str,
    ) -> ConnectionType {
        // Frontend to Backend = API Call
        // Infer connection type based on category pairs
        match (source_category, target_category) {
            // Frontend to Backend = API Call
            ("frontend", "backend") => ConnectionType::ApiCall,
            
            // Backend to Database = Data Flow
            ("backend", "database") => ConnectionType::DataFlow,
            
            // Backend to External = API Call
            ("backend", "external") => ConnectionType::ApiCall,
            
            // Database to Backend = Data Flow (reverse)
            ("database", "backend") => ConnectionType::DataFlow,
            
            // Default to Dependency for other combinations
            _ => ConnectionType::Dependency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_component_name() {
        assert_eq!(
            ArchitectureAnalyzer::format_component_name("auth_service"),
            "Auth Service"
        );
        assert_eq!(
            ArchitectureAnalyzer::format_component_name("user_management"),
            "User Management"
        );
    }

    #[test]
    fn test_infer_connection_type() {
        // Frontend to Backend
        let conn_type = ArchitectureAnalyzer::infer_connection_type("frontend", "backend");
        assert!(matches!(conn_type, ConnectionType::ApiCall));

        // Backend to Database
        let conn_type = ArchitectureAnalyzer::infer_connection_type("backend", "database");
        assert!(matches!(conn_type, ConnectionType::DataFlow));
        
        // Backend to External
        let conn_type = ArchitectureAnalyzer::infer_connection_type("backend", "external");
        assert!(matches!(conn_type, ConnectionType::ApiCall));
        
        // Other combinations default to Dependency
        let conn_type = ArchitectureAnalyzer::infer_connection_type("frontend", "frontend");
        assert!(matches!(conn_type, ConnectionType::Dependency));
    }
}
