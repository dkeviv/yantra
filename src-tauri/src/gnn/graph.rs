// File: src-tauri/src/gnn/graph.rs
// Purpose: Graph data structures using petgraph
// Dependencies: petgraph, serde
// Last Updated: November 20, 2025

use super::{CodeNode, CodeEdge, EdgeType};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CodeGraph {
    graph: DiGraph<CodeNode, EdgeType>,
    // Map from node ID to graph index for quick lookup
    node_map: HashMap<String, NodeIndex>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, node: CodeNode) {
        let node_id = node.id.clone();
        
        // Check if node already exists
        if self.node_map.contains_key(&node_id) {
            return;
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
        };
        
        let node2 = CodeNode {
            id: "test::func2".to_string(),
            node_type: NodeType::Function,
            name: "func2".to_string(),
            file_path: "test.py".to_string(),
            line_start: 7,
            line_end: 10,
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
        };
        
        let node2 = CodeNode {
            id: "test::func2".to_string(),
            node_type: NodeType::Function,
            name: "func2".to_string(),
            file_path: "test.py".to_string(),
            line_start: 7,
            line_end: 10,
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
