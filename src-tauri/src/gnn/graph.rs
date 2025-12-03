// File: src-tauri/src/gnn/graph.rs
// Purpose: Graph data structures using petgraph with HNSW semantic indexing
// Dependencies: petgraph, hnsw_rs, serde
// Last Updated: December 3, 2025

use super::{CodeNode, CodeEdge, EdgeType};
use super::hnsw_index::HnswIndex;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CodeGraph {
    graph: DiGraph<CodeNode, EdgeType>,
    // Map from node ID to graph index for quick lookup
    node_map: HashMap<String, NodeIndex>,
    // HNSW index for fast semantic similarity search (O(log n) vs O(n))
    hnsw_index: Option<Arc<HnswIndex>>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            hnsw_index: None,
        }
    }
    
    /// Create new graph with HNSW indexing enabled
    /// 
    /// # Arguments
    /// * `max_nodes` - Expected maximum number of nodes (for index sizing)
    /// * `embedding_dim` - Embedding vector dimensions (384 for MiniLM)
    pub fn new_with_hnsw(max_nodes: usize, embedding_dim: usize) -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            hnsw_index: Some(Arc::new(HnswIndex::new(embedding_dim, max_nodes))),
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, node: CodeNode) {
        let node_id = node.id.clone();
        
        // Check if node already exists
        if self.node_map.contains_key(&node_id) {
            return;
        }
        
        // Add to HNSW index if enabled and node has embedding
        if let Some(ref hnsw) = self.hnsw_index {
            if node.semantic_embedding.is_some() {
                if let Err(e) = hnsw.add_node(&node) {
                    eprintln!("Warning: Failed to add node to HNSW index: {}", e);
                }
            }
        }
        
        let index = self.graph.add_node(node);
        self.node_map.insert(node_id, index);
    }
    
    /// Add an edge between two nodes
    pub fn add_edge(&mut self, edge: CodeEdge) -> Result<(), String> {
        let source_idx = self.node_map
            .get(&edge.source_id)
            .ok_or_else(|| format!("Source node not found: {}", edge.source_id))?;
        
        // Try exact match first
        let target_idx = if let Some(idx) = self.node_map.get(&edge.target_id) {
            Some(*idx)
        } else {
            // Try fuzzy match: extract just the function/class name from target_id
            // target_id format: /path/to/file.py::function_name
            if let Some(target_name) = edge.target_id.split("::").last() {
                // Skip built-ins and methods (containing dots)
                if !target_name.contains('.') && !target_name.starts_with("self.") {
                    // Look for any node with this name
                    self.node_map.iter()
                        .find(|(id, _)| {
                            id.split("::").last() == Some(target_name)
                        })
                        .map(|(_, idx)| *idx)
                } else {
                    None
                }
            } else {
                None
            }
        };
        
        if let Some(target_idx) = target_idx {
            self.graph.add_edge(*source_idx, target_idx, edge.edge_type);
            Ok(())
        } else {
            Err(format!("Target node not found: {}", edge.target_id))
        }
    }
    
    /// Get all dependencies of a node (nodes that this node depends on)
    pub fn get_dependencies(&self, node_id: &str) -> Vec<CodeNode> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            let neighbors = self.graph.neighbors(node_idx);
            neighbors
                .filter_map(|idx| self.graph.node_weight(idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get all dependents of a node (nodes that depend on this node)
    pub fn get_dependents(&self, node_id: &str) -> Vec<CodeNode> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            let neighbors = self.graph.neighbors_directed(
                node_idx,
                petgraph::Direction::Incoming
            );
            neighbors
                .filter_map(|idx| self.graph.node_weight(idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Find a node by name and optionally file path
    pub fn find_node(&self, name: &str, file_path: Option<&str>) -> Option<&CodeNode> {
        self.graph.node_weights().find(|node| {
            if node.name == name {
                if let Some(path) = file_path {
                    // Match if the node's file_path ends with the search path
                    // This allows searching with just filename instead of full path
                    node.file_path.ends_with(path) || node.file_path == path
                } else {
                    true
                }
            } else {
                false
            }
        })
    }
    
    /// Get all nodes in the graph
    pub fn get_all_nodes(&self) -> Vec<&CodeNode> {
        self.graph.node_weights().collect()
    }
    
    /// Get a node by ID
    #[allow(dead_code)]
    pub fn get_node(&self, node_id: &str) -> Option<&CodeNode> {
        let idx = self.node_map.get(node_id)?;
        self.graph.node_weight(*idx)
    }

    /// Get total number of nodes
    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get total number of edges
    #[allow(dead_code)]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_map.clear();
    }
    
    /// Export graph as nodes and edges for serialization
    pub fn export(&self) -> (Vec<CodeNode>, Vec<(String, String, EdgeType)>) {
        let nodes: Vec<CodeNode> = self.graph.node_weights().cloned().collect();
        
        let edges: Vec<(String, String, EdgeType)> = self.graph
            .edge_indices()
            .filter_map(|edge_idx| {
                let (source_idx, target_idx) = self.graph.edge_endpoints(edge_idx)?;
                let source = self.graph.node_weight(source_idx)?;
                let target = self.graph.node_weight(target_idx)?;
                let edge_type = self.graph.edge_weight(edge_idx)?;
                
                Some((source.id.clone(), target.id.clone(), edge_type.clone()))
            })
            .collect();
        
        (nodes, edges)
    }
    
    /// Import graph from nodes and edges
    pub fn import(&mut self, nodes: Vec<CodeNode>, edges: Vec<(String, String, EdgeType)>) {
        self.clear();
        
        // Add all nodes first
        for node in nodes {
            self.add_node(node);
        }
        
        // Then add edges
        for (source_id, target_id, edge_type) in edges {
            let edge = CodeEdge {
                edge_type,
                source_id,
                target_id,
            };
            let _ = self.add_edge(edge);
        }
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::NodeType;
    
    #[test]
    fn test_add_node() {
        let mut graph = CodeGraph::new();
        
        let node = CodeNode {
            id: "test::func1".to_string(),
            node_type: NodeType::Function,
            name: "func1".to_string(),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 5,
            ..Default::default()
        };
        
        graph.add_node(node);
        assert_eq!(graph.node_count(), 1);
    }
    
    #[test]
    fn test_add_edge() {
        let mut graph = CodeGraph::new();
        
        let node1 = CodeNode {
            id: "test::func1".to_string(),
            node_type: NodeType::Function,
            name: "func1".to_string(),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 5,
            ..Default::default()
        };
        
        let node2 = CodeNode {
            id: "test::func2".to_string(),
            node_type: NodeType::Function,
            name: "func2".to_string(),
            file_path: "test.py".to_string(),
            line_start: 7,
            line_end: 10,
            ..Default::default()
        };
        
        graph.add_node(node1);
        graph.add_node(node2);
        
        let edge = CodeEdge {
            edge_type: EdgeType::Calls,
            source_id: "test::func1".to_string(),
            target_id: "test::func2".to_string(),
        };
        
        assert!(graph.add_edge(edge).is_ok());
        assert_eq!(graph.edge_count(), 1);
    }
    
    #[test]
    fn test_get_dependencies() {
        let mut graph = CodeGraph::new();
        
        // func1 calls func2
        let node1 = CodeNode {
            id: "test::func1".to_string(),
            node_type: NodeType::Function,
            name: "func1".to_string(),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 5,
            ..Default::default()
        };
        
        let node2 = CodeNode {
            id: "test::func2".to_string(),
            node_type: NodeType::Function,
            name: "func2".to_string(),
            file_path: "test.py".to_string(),
            line_start: 7,
            line_end: 10,
            ..Default::default()
        };
        
        graph.add_node(node1);
        graph.add_node(node2);
        
        let edge = CodeEdge {
            edge_type: EdgeType::Calls,
            source_id: "test::func1".to_string(),
            target_id: "test::func2".to_string(),
        };
        
        graph.add_edge(edge).unwrap();
        
        let deps = graph.get_dependencies("test::func1");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].name, "func2");
    }
}

impl CodeGraph {
    /// Get incoming edges of a specific type for a node
    pub fn get_incoming_edges(&self, node_id: &str, edge_type: EdgeType) -> Vec<CodeEdge> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            self.graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .filter(|edge_ref| *edge_ref.weight() == edge_type)
                .filter_map(|edge_ref| {
                    let source_node = self.graph.node_weight(edge_ref.source())?;
                    let target_node = self.graph.node_weight(edge_ref.target())?;
                    Some(CodeEdge {
                        edge_type: edge_type.clone(),
                        source_id: source_node.id.clone(),
                        target_id: target_node.id.clone(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get outgoing edges of a specific type for a node
    pub fn get_outgoing_edges(&self, node_id: &str, edge_type: EdgeType) -> Vec<CodeEdge> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            self.graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .filter(|edge_ref| *edge_ref.weight() == edge_type)
                .filter_map(|edge_ref| {
                    let source_node = self.graph.node_weight(edge_ref.source())?;
                    let target_node = self.graph.node_weight(edge_ref.target())?;
                    Some(CodeEdge {
                        edge_type: edge_type.clone(),
                        source_id: source_node.id.clone(),
                        target_id: target_node.id.clone(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all nodes in a specific file
    pub fn get_nodes_in_file(&self, file_path: &str) -> Vec<CodeNode> {
        self.graph
            .node_weights()
            .filter(|node| node.file_path == file_path)
            .cloned()
            .collect()
    }

    /// Get all dependencies recursively (all reachable nodes)
    pub fn get_all_dependencies(&self, node_id: &str) -> Vec<String> {
        use petgraph::visit::Bfs;
        use std::collections::HashSet;

        if let Some(&start_idx) = self.node_map.get(node_id) {
            let mut visited = HashSet::new();
            let mut bfs = Bfs::new(&self.graph, start_idx);

            while let Some(idx) = bfs.next(&self.graph) {
                if let Some(node) = self.graph.node_weight(idx) {
                    visited.insert(node.id.clone());
                }
            }

            // Remove the starting node itself
            visited.remove(node_id);
            visited.into_iter().collect()
        } else {
            Vec::new()
        }
    }

    /// Find semantically similar nodes using embeddings
    /// 
    /// Uses HNSW index for O(log n) search if available, falls back to linear scan.
    /// Returns nodes sorted by similarity score (highest first).
    /// Only nodes with embeddings are considered.
    /// 
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `min_similarity` - Minimum cosine similarity threshold (0.0 to 1.0)
    /// * `max_results` - Maximum number of results to return
    pub fn find_similar_nodes(
        &self,
        query_embedding: &[f32],
        min_similarity: f32,
        max_results: usize,
    ) -> Vec<(CodeNode, f32)> {
        // Use HNSW index if available (O(log n) search - FERRARI MVP requirement)
        if let Some(ref hnsw) = self.hnsw_index {
            match hnsw.search(query_embedding, max_results * 2) {
                Ok(results) => {
                    // Filter by minimum similarity and convert to CodeNode
                    let filtered: Vec<(CodeNode, f32)> = results
                        .into_iter()
                        .filter(|(_, similarity)| *similarity >= min_similarity)
                        .filter_map(|(node_id, similarity)| {
                            self.node_map.get(&node_id).and_then(|&idx| {
                                self.graph.node_weight(idx).map(|node| (node.clone(), similarity))
                            })
                        })
                        .take(max_results)
                        .collect();
                    
                    return filtered;
                }
                Err(e) => {
                    eprintln!("Warning: HNSW search failed, falling back to linear scan: {}", e);
                    // Fall through to linear scan
                }
            }
        }
        
        // Fallback: Linear scan (O(n) - slow for large graphs, only used if HNSW unavailable)
        use super::embeddings::EmbeddingGenerator;
        
        let mut results: Vec<(CodeNode, f32)> = self
            .graph
            .node_weights()
            .filter_map(|node| {
                if let Some(embedding) = &node.semantic_embedding {
                    let similarity = EmbeddingGenerator::cosine_similarity(
                        query_embedding,
                        embedding,
                    );
                    
                    if similarity >= min_similarity {
                        Some((node.clone(), similarity))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        results.truncate(max_results);
        
        results
    }

    /// Find nodes semantically similar to a target node
    /// 
    /// Convenience method that uses the target node's embedding.
    pub fn find_similar_to_node(
        &self,
        target_node_id: &str,
        min_similarity: f32,
        max_results: usize,
    ) -> Result<Vec<(CodeNode, f32)>, String> {
        let target_idx = self
            .node_map
            .get(target_node_id)
            .ok_or_else(|| format!("Node not found: {}", target_node_id))?;
        
        let target_node = self
            .graph
            .node_weight(*target_idx)
            .ok_or_else(|| "Node weight not found".to_string())?;
        
        let embedding = target_node
            .semantic_embedding
            .as_ref()
            .ok_or_else(|| format!("Node {} has no embedding", target_node_id))?;
        
        Ok(self.find_similar_nodes(embedding, min_similarity, max_results))
    }

    /// Find semantically similar nodes within N hops of target
    /// 
    /// Combines structural (BFS) and semantic (embeddings) search.
    /// This is more powerful than pure semantic search as it respects
    /// code structure while finding semantically related code.
    pub fn find_similar_in_neighborhood(
        &self,
        target_node_id: &str,
        max_hops: usize,
        min_similarity: f32,
        max_results: usize,
    ) -> Result<Vec<(CodeNode, f32)>, String> {
        use petgraph::visit::Bfs;
        use std::collections::HashSet;
        use super::embeddings::EmbeddingGenerator;
        
        let target_idx = self
            .node_map
            .get(target_node_id)
            .ok_or_else(|| format!("Node not found: {}", target_node_id))?;
        
        let target_node = self
            .graph
            .node_weight(*target_idx)
            .ok_or_else(|| "Node weight not found".to_string())?;
        
        let target_embedding = target_node
            .semantic_embedding
            .as_ref()
            .ok_or_else(|| format!("Node {} has no embedding", target_node_id))?;
        
        // BFS to find nodes within max_hops
        let mut neighborhood = HashSet::new();
        let mut bfs = Bfs::new(&self.graph, *target_idx);
        let mut depth_map: HashMap<NodeIndex, usize> = HashMap::new();
        depth_map.insert(*target_idx, 0);
        
        while let Some(idx) = bfs.next(&self.graph) {
            let depth = *depth_map.get(&idx).unwrap_or(&0);
            
            if depth <= max_hops {
                neighborhood.insert(idx);
                
                // Track depth for neighbors
                for neighbor in self.graph.neighbors(idx) {
                    depth_map.entry(neighbor).or_insert(depth + 1);
                }
            }
        }
        
        // Filter by semantic similarity
        let mut results: Vec<(CodeNode, f32)> = neighborhood
            .iter()
            .filter_map(|&idx| {
                let node = self.graph.node_weight(idx)?;
                
                // Skip target node itself
                if node.id == target_node_id {
                    return None;
                }
                
                if let Some(embedding) = &node.semantic_embedding {
                    let similarity = EmbeddingGenerator::cosine_similarity(
                        target_embedding,
                        embedding,
                    );
                    
                    if similarity >= min_similarity {
                        Some((node.clone(), similarity))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);
        
        Ok(results)
    }
}

