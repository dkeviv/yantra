// File: src-tauri/src/gnn/mod.rs
// Purpose: Graph Neural Network module for code dependency tracking
// Dependencies: tree-sitter, petgraph, rusqlite
// Last Updated: December 3, 2025

pub mod parser;
pub mod parser_js;
pub mod parser_rust;
pub mod parser_go;
pub mod parser_java;
pub mod parser_c;
pub mod parser_cpp;
pub mod parser_ruby;
pub mod parser_php;
pub mod parser_swift;
pub mod parser_kotlin;
pub mod graph;
pub mod persistence;
pub mod incremental;
pub mod features;
pub mod embeddings;
pub mod hnsw_index;
pub mod query;
pub mod version_tracker;
pub mod completion;
pub mod package_tracker;
pub mod auto_refresh;
pub mod file_watcher;

// Re-export main types
pub use graph::CodeGraph;
pub use auto_refresh::{AutoRefreshManager, FileTracker, RefreshResult, RefreshStats};
pub use file_watcher::FileWatcher;

// Re-export query types
pub use query::{
    QueryBuilder, QueryFilter, QueryResults, OrderDirection,
    Aggregator, PathFinder, TransactionManager,
};

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeNode {
    pub id: String,
    pub node_type: NodeType,
    pub name: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    
    // Semantic layer (optional)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub semantic_embedding: Option<Vec<f32>>,
    
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub code_snippet: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub docstring: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Function,
    Class,
    Variable,
    Import,
    Module,
    Package {
        name: String,
        version: String,
        language: PackageLanguage,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackageLanguage {
    Python,
    JavaScript,
    Rust,
    Go,
    Java,
    Ruby,
    PHP,
}

impl Default for NodeType {
    fn default() -> Self {
        NodeType::Module
    }
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
    /// Test file tests a source function/class
    Tests,
    /// Test file depends on source file (general relationship)
    TestDependency,
    /// File uses a package at specific version
    UsesPackage,
    /// Package depends on another package (transitive dependency)
    DependsOn,
    /// Package conflicts with another package version
    ConflictsWith,
}

pub struct GNNEngine {
    graph: graph::CodeGraph,
    db: persistence::Database,
    incremental_tracker: incremental::IncrementalTracker,
    package_tracker: package_tracker::PackageTracker,
}

// SAFETY: GNNEngine is always accessed through a Mutex, which provides exclusive access
// The SQLite connection is never accessed concurrently because the Mutex ensures
// only one thread can access the GNNEngine at a time
unsafe impl Send for GNNEngine {}
unsafe impl Sync for GNNEngine {}

impl GNNEngine {
    /// Create a new GNN engine with database at specified path
    pub fn new(db_path: &Path) -> Result<Self, String> {
        let db = persistence::Database::new(db_path)
            .map_err(|e| format!("Failed to initialize database: {}", e))?;
        
        let graph = graph::CodeGraph::new();
        let incremental_tracker = incremental::IncrementalTracker::new();
        let package_tracker = package_tracker::PackageTracker::new();
        
        Ok(Self { 
            graph, 
            db,
            incremental_tracker,
            package_tracker,
        })
    }
    
    /// Parse a file (Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin) and add its nodes and edges to the graph
    pub fn parse_file(&mut self, file_path: &Path) -> Result<(), String> {
        let code = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        let extension = file_path.extension().and_then(|s| s.to_str());
        
        let (nodes, edges) = match extension {
            Some("py") => parser::parse_python_file(&code, file_path)?,
            Some("js") | Some("jsx") => parser_js::parse_javascript_file(&code, file_path)?,
            Some("ts") => parser_js::parse_typescript_file(&code, file_path)?,
            Some("tsx") => parser_js::parse_tsx_file(&code, file_path)?,
            Some("rs") => parser_rust::parse_rust_file(&code, file_path)?,
            Some("go") => parser_go::parse_go_file(&code, file_path)?,
            Some("java") => parser_java::parse_java_file(&code, file_path)?,
            Some("c") | Some("h") => parser_c::parse_c_file(&code, file_path)?,
            Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hxx") => parser_cpp::parse_cpp_file(&code, file_path)?,
            Some("rb") => parser_ruby::parse_ruby_file(&code, file_path)?,
            Some("php") => parser_php::parse_php_file(&code, file_path)?,
            Some("swift") => parser_swift::parse_swift_file(&code, file_path)?,
            Some("kt") | Some("kts") => parser_kotlin::parse_kotlin_file(&code, file_path)?,
            _ => return Err(format!("Unsupported file extension: {:?}", extension)),
        };
        
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
        
        // Pass 1: Parse all files and collect nodes and edges
        let mut all_nodes = Vec::new();
        let mut all_edges = Vec::new();
        for file_path in &python_files {
            let code = std::fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            match parser::parse_python_file(&code, file_path) {
                Ok((nodes, edges)) => {
                    all_nodes.extend(nodes);
                    all_edges.extend(edges);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to parse {}: {}", file_path.display(), e);
                }
            }
        }
        
        // Pass 2: Generate embeddings for nodes with code snippets (optional, lazy)
        if all_nodes.iter().any(|n| n.code_snippet.is_some()) {
            if let Ok(mut embedder) = embeddings::EmbeddingGenerator::new(embeddings::EmbeddingModel::MiniLM) {
                println!("Generating semantic embeddings for {} nodes...", all_nodes.len());
                let start = std::time::Instant::now();
                
                for node in &mut all_nodes {
                    if node.code_snippet.is_some() {
                        match embedder.generate_embedding(node) {
                            Ok(embedding) => {
                                node.semantic_embedding = Some(embedding);
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to generate embedding for {}: {}", node.id, e);
                            }
                        }
                    }
                }
                
                let duration = start.elapsed();
                println!("Generated embeddings in {:?} ({:.2}ms per node)", 
                    duration, 
                    duration.as_secs_f64() * 1000.0 / all_nodes.len() as f64
                );
            } else {
                eprintln!("Warning: Failed to initialize embedding generator, skipping semantic embeddings");
            }
        }
        
        // Pass 3: Add all nodes to graph
        for node in all_nodes {
            self.graph.add_node(node);
        }
        
        // Pass 4: Add all edges now that all nodes exist
        for edge in all_edges {
            if let Err(e) = self.graph.add_edge(edge) {
                // Only warn, don't fail - some edges might be to external modules
                eprintln!("Warning: Failed to add edge: {}", e);
            }
        }
        
        self.persist()?;
        Ok(())
    }
    
    /// Collect all supported source files in directory tree (Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin)
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
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                // Support Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin
                if matches!(ext, "py" | "js" | "ts" | "jsx" | "tsx" | 
                           "rs" | "go" | "java" |
                           "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "hxx" |
                           "rb" | "php" | "swift" | "kt" | "kts") {
                    files.push(path);
                }
            }
        }
        
        Ok(())
    }
    
    /// Recursively scan directory for supported files (deprecated - use build_graph instead)
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
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                // Support all languages
                if matches!(ext, "py" | "js" | "ts" | "jsx" | "tsx" | 
                           "rs" | "go" | "java" |
                           "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "hxx" |
                           "rb" | "php" | "swift" | "kt" | "kts") {
                    if let Err(e) = self.parse_file(&path) {
                        eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                    }
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
    
    /// Get reference to the underlying graph
    pub fn get_graph(&self) -> &graph::CodeGraph {
        &self.graph
    }
    
    /// Get the total count of nodes in the graph
    pub fn get_node_count(&self) -> usize {
        self.graph.get_all_nodes().len()
    }
    
    /// Get all nodes in the graph
    pub fn get_all_nodes(&self) -> Vec<&CodeNode> {
        self.graph.get_all_nodes()
    }
    
    /// Identify if a file is a test file based on naming conventions
    pub fn is_test_file(file_path: &Path) -> bool {
        let file_name = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        let path_str = file_path.to_str().unwrap_or("");
        
        // Python test patterns: test_*.py, *_test.py, tests/ directory
        // JavaScript test patterns: *.test.js, *.test.ts, *.spec.js, *.spec.ts, __tests__/ directory
        file_name.starts_with("test_") 
            || file_name.ends_with("_test.py")
            || file_name.ends_with("_test.js")
            || file_name.ends_with("_test.ts")
            || file_name.contains(".test.")
            || file_name.contains(".spec.")
            || path_str.contains("/tests/")
            || path_str.contains("/test/")
            || path_str.contains("/__tests__/")
            || path_str.contains("/spec/")
    }
    
    /// Map test file to its corresponding source file
    pub fn find_source_file_for_test(&self, test_file_path: &Path) -> Option<String> {
        let file_name = test_file_path.file_name()?.to_str()?;
        
        // Remove test patterns to find source file name
        let source_name = file_name
            .replace("test_", "")
            .replace("_test", "")
            .replace(".test", "")
            .replace(".spec", "");
        
        // Try to find matching source file in graph (exclude test files)
        for node in self.graph.get_all_nodes() {
            let node_path = Path::new(&node.file_path);
            
            // Skip if this is itself a test file
            if Self::is_test_file(node_path) {
                continue;
            }
            
            // Check if this source file matches the expected name
            if node.file_path.ends_with(&source_name) {
                return Some(node.file_path.clone());
            }
        }
        
        None
    }
    
    /// Create test-to-source edges for all test files in the graph
    pub fn create_test_edges(&mut self) -> Result<usize, String> {
        let mut test_edges_created = 0;
        let all_nodes: Vec<CodeNode> = self.graph.get_all_nodes().into_iter().cloned().collect();
        
        for node in &all_nodes {
            let file_path = Path::new(&node.file_path);
            
            if Self::is_test_file(file_path) {
                // Create TestDependency edge from test file to source file
                if let Some(source_file) = self.find_source_file_for_test(file_path) {
                    // Find all nodes in source file
                    for source_node in &all_nodes {
                        if source_node.file_path == source_file {
                            // Create Tests edge if test function tests source function
                            // Pattern: test_function_name tests function_name
                            if node.name.starts_with("test_") {
                                let tested_name = node.name.strip_prefix("test_").unwrap_or("");
                                if source_node.name == tested_name || source_node.name.contains(tested_name) {
                                    let edge = CodeEdge {
                                        edge_type: EdgeType::Tests,
                                        source_id: node.id.clone(),
                                        target_id: source_node.id.clone(),
                                    };
                                    self.graph.add_edge(edge)?;
                                    test_edges_created += 1;
                                }
                            }
                            
                            // Also create general TestDependency edge
                            let dep_edge = CodeEdge {
                                edge_type: EdgeType::TestDependency,
                                source_id: node.id.clone(),
                                target_id: source_node.id.clone(),
                            };
                            if let Err(_) = self.graph.add_edge(dep_edge) {
                                // Edge might already exist, ignore
                            } else {
                                test_edges_created += 1;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(test_edges_created)
    }
    
    /// Incremental update: process only changed files (<50ms target per file)
    pub fn incremental_update_file(&mut self, file_path: &Path) -> Result<incremental::UpdateMetrics, String> {
        use std::time::Instant;
        let start = Instant::now();
        
        // Use incremental tracker to handle caching
        let (nodes, edges, mut metrics) = incremental::incremental_update_file(
            &mut self.incremental_tracker,
            file_path,
            |code, path| parser::parse_python_file(code, path),
        )?;
        
        // Remove old nodes from this file in the graph
        let existing_nodes: Vec<CodeNode> = self.graph
            .get_all_nodes()
            .iter()
            .filter(|n| n.file_path == file_path.to_str().unwrap())
            .map(|n| (*n).clone())
            .collect();
        
        // For now, we rebuild affected edges (future: incremental edge updates)
        // This is still fast because we only reparse changed files
        
        // Add new nodes
        for node in nodes {
            self.graph.add_node(node);
        }
        
        // Add edges (with fuzzy matching for external references)
        for edge in edges {
            let _ = self.graph.add_edge(edge); // Ignore errors for external modules
        }
        
        // Persist after update
        self.persist()?;
        
        // Update total duration
        metrics.duration_ms = start.elapsed().as_millis() as u64;
        
        Ok(metrics)
    }
    
    /// Incremental update for multiple files (batch processing)
    pub fn incremental_update_files(&mut self, file_paths: &[&Path]) -> Result<Vec<incremental::UpdateMetrics>, String> {
        let mut all_metrics = Vec::new();
        
        for file_path in file_paths {
            let metrics = self.incremental_update_file(file_path)?;
            all_metrics.push(metrics);
        }
        
        Ok(all_metrics)
    }
    
    /// Check if a file needs updating (is dirty)
    pub fn is_file_dirty(&self, file_path: &Path) -> Result<bool, String> {
        self.incremental_tracker.is_file_dirty(file_path)
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> incremental::CacheStats {
        self.incremental_tracker.stats()
    }
    
    /// Get graph node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
    
    /// Get graph edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
    
    /// Refresh graph if files are stale (DEP-028, DEP-029)
    /// Checks all tracked files and updates those that changed
    pub fn refresh_if_stale(&mut self) -> Result<usize, String> {
        let mut updated_count = 0;
        
        // Get list of all files from graph
        let all_files = self.list_all_files()?;
        
        // Check each file and update if dirty
        for file_path_str in all_files {
            let file_path = PathBuf::from(&file_path_str);
            
            // Check if file exists and is dirty
            if file_path.exists() {
                match self.is_file_dirty(&file_path) {
                    Ok(true) => {
                        // File is dirty, update it
                        match self.incremental_update_file(&file_path) {
                            Ok(_metrics) => {
                                updated_count += 1;
                            }
                            Err(e) => {
                                eprintln!("Failed to update {}: {}", file_path.display(), e);
                            }
                        }
                    }
                    Ok(false) => {
                        // File is clean, skip
                    }
                    Err(e) => {
                        eprintln!("Failed to check if {} is dirty: {}", file_path.display(), e);
                    }
                }
            }
        }
        
        Ok(updated_count)
    }
    
    /// Parse package dependencies and add to graph
    pub fn parse_packages(&mut self, project_path: &Path) -> Result<usize, String> {
        // Parse all package manifests
        let packages = self.package_tracker.parse_project(project_path)?;
        
        // Convert packages to nodes
        for package in &packages {
            let node = package_tracker::PackageTracker::package_to_node(package);
            self.graph.add_node(node);
        }
        
        // Create dependency edges between packages
        let edges = package_tracker::PackageTracker::create_package_edges(&packages);
        for edge in &edges {
            // Ignore errors for missing target packages (external dependencies)
            let _ = self.graph.add_edge(edge.clone());
        }
        
        println!("Parsed {} packages with {} dependency edges", packages.len(), edges.len());
        
        Ok(packages.len())
    }
    
    /// Get all packages in the graph
    pub fn get_packages(&self) -> Vec<&CodeNode> {
        self.graph.get_all_nodes()
            .into_iter()
            .filter(|n| matches!(n.node_type, NodeType::Package { .. }))
            .collect()
    }
    
    /// Find files using a specific package
    pub fn get_files_using_package(&self, package_name: &str, version: Option<&str>) -> Vec<String> {
        let mut files = Vec::new();
        
        // Find package node
        for node in self.graph.get_all_nodes() {
            if let NodeType::Package { name, version: pkg_version, .. } = &node.node_type {
                if name == package_name {
                    if let Some(req_version) = version {
                        if pkg_version != req_version {
                            continue;
                        }
                    }
                    
                    // Find files with UsesPackage edges to this package
                    let dependents = self.graph.get_dependents(&node.id);
                    for dep in dependents {
                        if !matches!(dep.node_type, NodeType::Package { .. }) {
                            files.push(dep.file_path.clone());
                        }
                    }
                }
            }
        }
        
        files.sort();
        files.dedup();
        files
    }
    
    /// Find packages used by a specific file
    pub fn get_packages_used_by_file(&self, file_path: &str) -> Vec<String> {
        let mut packages = Vec::new();
        
        // Find nodes in this file
        let file_nodes = self.graph.get_nodes_in_file(file_path);
        
        for node in file_nodes {
            // Find UsesPackage edges
            let dependencies = self.graph.get_dependencies(&node.id);
            for dep in dependencies {
                if let NodeType::Package { name, version, .. } = &dep.node_type {
                    packages.push(format!("{}=={}", name, version));
                }
            }
        }
        
        packages.sort();
        packages.dedup();
        packages
    }
    
    /// List all file paths in the graph
    pub fn list_all_files(&self) -> Result<Vec<String>, String> {
        let mut files: Vec<String> = self.graph.get_all_nodes()
            .iter()
            .map(|node| node.file_path.clone())
            .collect();
        
        files.sort();
        files.dedup();
        Ok(files)
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
