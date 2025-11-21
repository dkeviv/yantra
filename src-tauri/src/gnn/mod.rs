// File: src-tauri/src/gnn/mod.rs
// Purpose: Graph Neural Network module for code dependency tracking
// Dependencies: tree-sitter, petgraph, rusqlite
// Last Updated: November 20, 2025

pub mod parser;
pub mod graph;
pub mod persistence;

use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeNode {
    pub id: String,
    pub node_type: NodeType,
    pub name: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Function,
    Class,
    Variable,
    Import,
    Module,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEdge {
    pub edge_type: EdgeType,
    pub source_id: String,
    pub target_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    Calls,
    Uses,
    Imports,
    Inherits,
    Defines,
}

pub struct GNNEngine {
    graph: graph::CodeGraph,
    db: persistence::Database,
}

impl GNNEngine {
    /// Create a new GNN engine with database at specified path
    pub fn new(db_path: &Path) -> Result<Self, String> {
        let db = persistence::Database::new(db_path)
            .map_err(|e| format!("Failed to initialize database: {}", e))?;
        
        let graph = graph::CodeGraph::new();
        
        Ok(Self { graph, db })
    }
    
    /// Parse a Python file and add its nodes and edges to the graph
    pub fn parse_file(&mut self, file_path: &Path) -> Result<(), String> {
        let code = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        let (nodes, edges) = parser::parse_python_file(&code, file_path)?;
        
        for node in nodes {
            self.graph.add_node(node);
        }
        
        for edge in edges {
            self.graph.add_edge(edge)?;
        }
        
        Ok(())
    }
    
    /// Build graph for entire project directory (two-pass approach)
    pub fn build_graph(&mut self, project_path: &Path) -> Result<(), String> {
        // Collect all Python files first
        let mut python_files = Vec::new();
        self.collect_python_files(project_path, &mut python_files)?;
        
        // Pass 1: Parse all files and add nodes only
        let mut all_edges = Vec::new();
        for file_path in &python_files {
            let code = std::fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            match parser::parse_python_file(&code, file_path) {
                Ok((nodes, edges)) => {
                    // Add all nodes to graph
                    for node in nodes {
                        self.graph.add_node(node);
                    }
                    // Store edges for second pass
                    all_edges.extend(edges);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to parse {}: {}", file_path.display(), e);
                }
            }
        }
        
        // Pass 2: Add all edges now that all nodes exist
        for edge in all_edges {
            if let Err(e) = self.graph.add_edge(edge) {
                // Only warn, don't fail - some edges might be to external modules
                eprintln!("Warning: Failed to add edge: {}", e);
            }
        }
        
        self.persist()?;
        Ok(())
    }
    
    /// Collect all Python files in directory tree
    fn collect_python_files(&self, dir_path: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
        if !dir_path.is_dir() {
            return Ok(());
        }
        
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            
            if path.is_dir() {
                // Skip common ignore directories
                let dir_name = path.file_name().unwrap().to_str().unwrap();
                if dir_name == "node_modules" || dir_name == "__pycache__" 
                    || dir_name == ".git" || dir_name == "venv" || dir_name == ".yantra" {
                    continue;
                }
                self.collect_python_files(&path, files)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("py") {
                files.push(path);
            }
        }
        
        Ok(())
    }
    
    /// Recursively scan directory for Python files (deprecated - use build_graph instead)
    #[allow(dead_code)]
    fn scan_directory(&mut self, dir_path: &Path) -> Result<(), String> {
        if !dir_path.is_dir() {
            return Ok(());
        }
        
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            
            if path.is_dir() {
                // Skip common ignore directories
                let dir_name = path.file_name().unwrap().to_str().unwrap();
                if dir_name == "node_modules" || dir_name == "__pycache__" 
                    || dir_name == ".git" || dir_name == "venv" {
                    continue;
                }
                self.scan_directory(&path)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("py") {
                if let Err(e) = self.parse_file(&path) {
                    eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                }
            }
        }
        
        Ok(())
    }
    
    /// Persist graph to database
    fn persist(&self) -> Result<(), String> {
        self.db.save_graph(&self.graph)
            .map_err(|e| format!("Failed to save graph: {}", e))
    }
    
    /// Load graph from database
    pub fn load(&mut self) -> Result<(), String> {
        self.graph = self.db.load_graph()
            .map_err(|e| format!("Failed to load graph: {}", e))?;
        Ok(())
    }
    
    /// Get all dependencies of a node
    pub fn get_dependencies(&self, node_id: &str) -> Vec<CodeNode> {
        self.graph.get_dependencies(node_id)
    }
    
    /// Get all dependents of a node (what depends on this)
    pub fn get_dependents(&self, node_id: &str) -> Vec<CodeNode> {
        self.graph.get_dependents(node_id)
    }
    
    /// Find a node by name and file
    pub fn find_node(&self, name: &str, file_path: Option<&str>) -> Option<&CodeNode> {
        self.graph.find_node(name, file_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_gnn_engine_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let engine = GNNEngine::new(&db_path);
        assert!(engine.is_ok());
    }
}
