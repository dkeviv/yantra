// File: src-tauri/src/gnn/incremental.rs
// Purpose: Incremental GNN updates with <50ms per file change
// Dependencies: std::collections, std::time
// Last Updated: November 25, 2025

use super::{CodeNode, CodeEdge, NodeType};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Tracks dirty state and cached features for incremental updates
#[derive(Debug)]
pub struct IncrementalTracker {
    /// Map from file path to its last modified timestamp
    file_timestamps: HashMap<PathBuf, std::time::SystemTime>,
    
    /// Set of dirty file paths that need reprocessing
    dirty_files: HashSet<PathBuf>,
    
    /// Map from node ID to cached parsed data (nodes only, edges computed separately)
    node_cache: HashMap<String, CodeNode>,
    
    /// Map from file path to node IDs it contains
    file_to_nodes: HashMap<PathBuf, Vec<String>>,
    
    /// Map from node ID to node IDs that depend on it (for propagation)
    dependency_map: HashMap<String, HashSet<String>>,
}

impl IncrementalTracker {
    pub fn new() -> Self {
        Self {
            file_timestamps: HashMap::new(),
            dirty_files: HashSet::new(),
            node_cache: HashMap::new(),
            file_to_nodes: HashMap::new(),
            dependency_map: HashMap::new(),
        }
    }

    /// Check if a file has changed since last processing
    pub fn is_file_dirty(&self, file_path: &Path) -> Result<bool, String> {
        let metadata = std::fs::metadata(file_path)
            .map_err(|e| format!("Failed to get file metadata: {}", e))?;
        
        let modified = metadata.modified()
            .map_err(|e| format!("Failed to get modification time: {}", e))?;
        
        Ok(match self.file_timestamps.get(file_path) {
            Some(cached_time) => modified > *cached_time,
            None => true, // File not seen before
        })
    }

    /// Mark a file as dirty (needs reprocessing)
    pub fn mark_dirty(&mut self, file_path: &Path) {
        self.dirty_files.insert(file_path.to_path_buf());
        
        // Also mark all nodes that depend on nodes in this file as dirty
        if let Some(node_ids) = self.file_to_nodes.get(file_path) {
            for node_id in node_ids {
                if let Some(dependents) = self.dependency_map.get(node_id) {
                    for dependent_id in dependents {
                        // Find the file containing the dependent node
                        if let Some(dependent_node) = self.node_cache.get(dependent_id) {
                            self.dirty_files.insert(PathBuf::from(&dependent_node.file_path));
                        }
                    }
                }
            }
        }
    }

    /// Update file timestamp after processing
    pub fn update_timestamp(&mut self, file_path: &Path) -> Result<(), String> {
        let metadata = std::fs::metadata(file_path)
            .map_err(|e| format!("Failed to get file metadata: {}", e))?;
        
        let modified = metadata.modified()
            .map_err(|e| format!("Failed to get modification time: {}", e))?;
        
        self.file_timestamps.insert(file_path.to_path_buf(), modified);
        self.dirty_files.remove(file_path);
        
        Ok(())
    }

    /// Cache nodes from a file
    pub fn cache_nodes(&mut self, file_path: &Path, nodes: Vec<CodeNode>) {
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        
        // Store nodes in cache
        for node in nodes {
            self.node_cache.insert(node.id.clone(), node);
        }
        
        // Update file-to-nodes mapping
        self.file_to_nodes.insert(file_path.to_path_buf(), node_ids);
    }

    /// Get cached nodes for a file
    pub fn get_cached_nodes(&self, file_path: &Path) -> Option<Vec<CodeNode>> {
        let node_ids = self.file_to_nodes.get(file_path)?;
        let nodes: Vec<CodeNode> = node_ids
            .iter()
            .filter_map(|id| self.node_cache.get(id))
            .cloned()
            .collect();
        
        if nodes.len() == node_ids.len() {
            Some(nodes)
        } else {
            None // Some nodes missing from cache
        }
    }

    /// Update dependency map from edges
    pub fn update_dependencies(&mut self, edges: &[CodeEdge]) {
        for edge in edges {
            self.dependency_map
                .entry(edge.target_id.clone())
                .or_insert_with(HashSet::new)
                .insert(edge.source_id.clone());
        }
    }

    /// Remove nodes for a file (when file is deleted or being reprocessed)
    pub fn remove_file_nodes(&mut self, file_path: &Path) {
        if let Some(node_ids) = self.file_to_nodes.remove(file_path) {
            for node_id in node_ids {
                self.node_cache.remove(&node_id);
                
                // Remove from dependency map
                self.dependency_map.remove(&node_id);
                
                // Remove as dependent from other nodes
                for deps in self.dependency_map.values_mut() {
                    deps.remove(&node_id);
                }
            }
        }
        
        self.file_timestamps.remove(file_path);
        self.dirty_files.remove(file_path);
    }

    /// Get all dirty files
    pub fn get_dirty_files(&self) -> Vec<PathBuf> {
        self.dirty_files.iter().cloned().collect()
    }

    /// Clear dirty flags
    pub fn clear_dirty(&mut self) {
        self.dirty_files.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            cached_files: self.file_to_nodes.len(),
            cached_nodes: self.node_cache.len(),
            dirty_files: self.dirty_files.len(),
            dependencies: self.dependency_map.len(),
        }
    }
}

impl Default for IncrementalTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the incremental cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cached_files: usize,
    pub cached_nodes: usize,
    pub dirty_files: usize,
    pub dependencies: usize,
}

/// Measures update performance
#[derive(Debug, Clone)]
pub struct UpdateMetrics {
    pub duration_ms: u64,
    pub files_processed: usize,
    pub nodes_updated: usize,
    pub edges_updated: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Performs incremental update on a single file
pub fn incremental_update_file(
    tracker: &mut IncrementalTracker,
    file_path: &Path,
    parse_fn: impl Fn(&str, &Path) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String>,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>, UpdateMetrics), String> {
    let start = Instant::now();
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    // Check if file is dirty
    let is_dirty = tracker.is_file_dirty(file_path)?;
    
    let (nodes, edges) = if !is_dirty {
        // Try to use cached nodes
        if let Some(cached_nodes) = tracker.get_cached_nodes(file_path) {
            cache_hits += cached_nodes.len();
            
            // Still need to recompute edges (they depend on other files)
            let code = std::fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            let (_, edges) = parse_fn(&code, file_path)?;
            
            (cached_nodes, edges)
        } else {
            cache_misses += 1;
            
            // Parse fresh
            let code = std::fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            let (nodes, edges) = parse_fn(&code, file_path)?;
            
            // Cache for next time
            tracker.cache_nodes(file_path, nodes.clone());
            tracker.update_timestamp(file_path)?;
            
            (nodes, edges)
        }
    } else {
        cache_misses += 1;
        
        // Remove old nodes for this file
        tracker.remove_file_nodes(file_path);
        
        // Parse fresh
        let code = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        let (nodes, edges) = parse_fn(&code, file_path)?;
        
        // Cache the new nodes
        tracker.cache_nodes(file_path, nodes.clone());
        tracker.update_timestamp(file_path)?;
        
        (nodes, edges)
    };

    // Update dependency tracking
    tracker.update_dependencies(&edges);

    let duration_ms = start.elapsed().as_millis() as u64;

    let metrics = UpdateMetrics {
        duration_ms,
        files_processed: 1,
        nodes_updated: nodes.len(),
        edges_updated: edges.len(),
        cache_hits,
        cache_misses,
    };

    Ok((nodes, edges, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: &str, file_path: &str) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            node_type: NodeType::Function,
            name: id.to_string(),
            file_path: file_path.to_string(),
            line_start: 1,
            line_end: 10,
            ..Default::default()
        }
    }

    #[test]
    fn test_cache_and_retrieve() {
        let mut tracker = IncrementalTracker::new();
        let path = PathBuf::from("/test/file.py");
        
        let nodes = vec![
            create_test_node("func1", "/test/file.py"),
            create_test_node("func2", "/test/file.py"),
        ];
        
        tracker.cache_nodes(&path, nodes.clone());
        
        let cached = tracker.get_cached_nodes(&path).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].id, "func1");
        assert_eq!(cached[1].id, "func2");
    }

    #[test]
    fn test_dirty_propagation() {
        let mut tracker = IncrementalTracker::new();
        
        // Setup: func1 depends on func2
        let edges = vec![CodeEdge {
            edge_type: super::super::EdgeType::Calls,
            source_id: "func1".to_string(),
            target_id: "func2".to_string(),
        }];
        
        tracker.update_dependencies(&edges);
        
        // Cache nodes in different files
        tracker.cache_nodes(
            &PathBuf::from("/test/file1.py"),
            vec![create_test_node("func1", "/test/file1.py")],
        );
        tracker.cache_nodes(
            &PathBuf::from("/test/file2.py"),
            vec![create_test_node("func2", "/test/file2.py")],
        );
        
        // Mark file2 as dirty
        tracker.mark_dirty(&PathBuf::from("/test/file2.py"));
        
        // Verify propagation: file1 should also be dirty (func1 depends on func2)
        let dirty = tracker.get_dirty_files();
        assert!(dirty.contains(&PathBuf::from("/test/file2.py")));
        assert!(dirty.contains(&PathBuf::from("/test/file1.py")));
    }

    #[test]
    fn test_remove_file_nodes() {
        let mut tracker = IncrementalTracker::new();
        let path = PathBuf::from("/test/file.py");
        
        let nodes = vec![
            create_test_node("func1", "/test/file.py"),
            create_test_node("func2", "/test/file.py"),
        ];
        
        tracker.cache_nodes(&path, nodes);
        assert_eq!(tracker.stats().cached_nodes, 2);
        
        tracker.remove_file_nodes(&path);
        assert_eq!(tracker.stats().cached_nodes, 0);
        assert!(tracker.get_cached_nodes(&path).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let mut tracker = IncrementalTracker::new();
        
        tracker.cache_nodes(
            &PathBuf::from("/test/file1.py"),
            vec![create_test_node("func1", "/test/file1.py")],
        );
        tracker.cache_nodes(
            &PathBuf::from("/test/file2.py"),
            vec![create_test_node("func2", "/test/file2.py")],
        );
        
        tracker.mark_dirty(&PathBuf::from("/test/file1.py"));
        
        let stats = tracker.stats();
        assert_eq!(stats.cached_files, 2);
        assert_eq!(stats.cached_nodes, 2);
        assert_eq!(stats.dirty_files, 1);
    }
}
