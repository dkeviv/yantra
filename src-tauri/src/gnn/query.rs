// File: src-tauri/src/gnn/query.rs
// Purpose: Agentic database query operations for GNN graph data
// Dependencies: SQLite, CodeGraph
// Last Updated: December 3, 2025
//
// Provides structured query interface for code graph exploration:
// - Complex query composition (AND/OR/NOT filters)
// - Transaction management for atomicity
// - Query optimization and indexing
// - Aggregation operations (count, group by)
// - Path queries (find all paths between nodes)
// - Neighborhood queries (N-hop neighbors)
// - Batch operations with rollback
//
// Usage:
// 1. Create QueryBuilder with filters
// 2. Execute query → get QueryResults
// 3. Use results for analysis/code generation
// 4. Transactions for multiple operations

use crate::gnn::{CodeGraph, CodeNode, NodeType};
use rusqlite::{Connection, Transaction, params};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Query builder for complex graph queries
pub struct QueryBuilder {
    filters: Vec<QueryFilter>,
    limit: Option<usize>,
    offset: Option<usize>,
    order_by: Option<(String, OrderDirection)>,
}

impl QueryBuilder {
    /// Create new query builder
    pub fn new() -> Self {
        QueryBuilder {
            filters: Vec::new(),
            limit: None,
            offset: None,
            order_by: None,
        }
    }

    /// Add filter condition
    pub fn filter(mut self, filter: QueryFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set result offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Set order by field
    pub fn order_by(mut self, field: String, direction: OrderDirection) -> Self {
        self.order_by = Some((field, direction));
        self
    }

    /// Execute query on graph
    pub fn execute(&self, graph: &CodeGraph) -> Result<QueryResults, String> {
        let nodes = graph.get_all_nodes();
        let mut filtered: Vec<&CodeNode> = nodes.iter()
            .copied()
            .filter(|node| self.matches_filters(node))
            .collect();

        // Apply ordering
        if let Some((field, direction)) = &self.order_by {
            filtered.sort_by(|a, b| {
                let cmp = match field.as_str() {
                    "name" => a.name.cmp(&b.name),
                    "file_path" => a.file_path.cmp(&b.file_path),
                    "line_start" => a.line_start.cmp(&b.line_start),
                    _ => std::cmp::Ordering::Equal,
                };
                match direction {
                    OrderDirection::Asc => cmp,
                    OrderDirection::Desc => cmp.reverse(),
                }
            });
        }

        // Apply pagination
        let start = self.offset.unwrap_or(0);
        let end = if let Some(limit) = self.limit {
            (start + limit).min(filtered.len())
        } else {
            filtered.len()
        };

        let results: Vec<CodeNode> = filtered[start..end]
            .iter()
            .map(|&node| node.clone())
            .collect();

        Ok(QueryResults {
            nodes: results,
            total_count: filtered.len(),
        })
    }

    /// Check if node matches all filters
    fn matches_filters(&self, node: &CodeNode) -> bool {
        self.filters.iter().all(|filter| filter.matches(node))
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Query filter conditions
#[derive(Debug, Clone)]
pub enum QueryFilter {
    /// Node type equals
    NodeType(NodeType),
    /// File path equals
    FilePath(String),
    /// File path contains
    FilePathContains(String),
    /// Name equals
    Name(String),
    /// Name contains
    NameContains(String),
    /// Line range (start, end)
    LineRange(usize, usize),
    /// Has embedding
    HasEmbedding(bool),
    /// AND combination
    And(Vec<QueryFilter>),
    /// OR combination
    Or(Vec<QueryFilter>),
    /// NOT negation
    Not(Box<QueryFilter>),
}

impl QueryFilter {
    /// Check if node matches this filter
    pub fn matches(&self, node: &CodeNode) -> bool {
        match self {
            QueryFilter::NodeType(node_type) => &node.node_type == node_type,
            QueryFilter::FilePath(path) => node.file_path == *path,
            QueryFilter::FilePathContains(substr) => node.file_path.contains(substr),
            QueryFilter::Name(name) => node.name == *name,
            QueryFilter::NameContains(substr) => node.name.contains(substr),
            QueryFilter::LineRange(start, end) => {
                node.line_start >= *start && node.line_end <= *end
            }
            QueryFilter::HasEmbedding(has) => node.semantic_embedding.is_some() == *has,
            QueryFilter::And(filters) => filters.iter().all(|f| f.matches(node)),
            QueryFilter::Or(filters) => filters.iter().any(|f| f.matches(node)),
            QueryFilter::Not(filter) => !filter.matches(node),
        }
    }
}

/// Query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResults {
    pub nodes: Vec<CodeNode>,
    pub total_count: usize,
}

impl QueryResults {
    /// Get first result
    pub fn first(&self) -> Option<&CodeNode> {
        self.nodes.first()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get count
    pub fn count(&self) -> usize {
        self.nodes.len()
    }
}

/// Order direction
#[derive(Debug, Clone, Copy)]
pub enum OrderDirection {
    Asc,
    Desc,
}

/// Aggregation operations
pub struct Aggregator;

impl Aggregator {
    /// Count nodes by type
    pub fn count_by_type(graph: &CodeGraph) -> HashMap<NodeType, usize> {
        let mut counts = HashMap::new();
        for node in graph.get_all_nodes() {
            *counts.entry(node.node_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Count nodes by file
    pub fn count_by_file(graph: &CodeGraph) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for node in graph.get_all_nodes() {
            *counts.entry(node.file_path.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get files with most nodes
    pub fn top_files(graph: &CodeGraph, limit: usize) -> Vec<(String, usize)> {
        let counts = Self::count_by_file(graph);
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(limit);
        sorted
    }

    /// Calculate average lines per function
    pub fn avg_function_lines(graph: &CodeGraph) -> f64 {
        let all_nodes = graph.get_all_nodes();
        let functions: Vec<_> = all_nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Function))
            .collect();

        if functions.is_empty() {
            return 0.0;
        }

        let total_lines: usize = functions.iter()
            .map(|n| n.line_end - n.line_start + 1)
            .sum();

        total_lines as f64 / functions.len() as f64
    }
}

/// Path finder for graph traversal
pub struct PathFinder<'a> {
    graph: &'a CodeGraph,
}

impl<'a> PathFinder<'a> {
    /// Create new path finder
    pub fn new(graph: &'a CodeGraph) -> Self {
        PathFinder { graph }
    }

    /// Find all paths between two nodes (up to max_depth)
    pub fn find_paths(
        &self,
        start_id: &str,
        end_id: &str,
        max_depth: usize,
    ) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut current_path = Vec::new();
        let mut visited = HashSet::new();

        self.dfs_paths(start_id, end_id, &mut current_path, &mut visited, max_depth, &mut paths);
        paths
    }

    /// Find shortest path between two nodes
    pub fn find_shortest_path(&self, start_id: &str, end_id: &str) -> Option<Vec<String>> {
        let mut queue = std::collections::VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        queue.push_back(start_id.to_string());
        visited.insert(start_id.to_string());

        while let Some(current) = queue.pop_front() {
            if current == end_id {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = end_id.to_string();
                
                while node != start_id {
                    path.push(node.clone());
                    node = parent.get(&node)?.clone();
                }
                path.push(start_id.to_string());
                path.reverse();
                
                return Some(path);
            }

            // Get neighbors (nodes this depends on)
            if let Some(node) = self.graph.get_all_nodes().iter().find(|n| n.id == current) {
                for dep in self.graph.get_dependencies(&node.id) {
                    if !visited.contains(&dep.id) {
                        visited.insert(dep.id.clone());
                        parent.insert(dep.id.clone(), current.clone());
                        queue.push_back(dep.id.clone());
                    }
                }
            }
        }

        None
    }

    /// Get N-hop neighborhood of a node
    pub fn get_neighborhood(&self, node_id: &str, hops: usize) -> HashSet<String> {
        let mut neighborhood = HashSet::new();
        let mut current_level = HashSet::new();
        current_level.insert(node_id.to_string());

        for _ in 0..hops {
            let mut next_level = HashSet::new();
            
            for id in &current_level {
                neighborhood.insert(id.clone());
                
                if let Some(node) = self.graph.get_all_nodes().iter().find(|n| n.id == *id) {
                    for dep in self.graph.get_dependencies(&node.id) {
                        if !neighborhood.contains(&dep.id) {
                            next_level.insert(dep.id.clone());
                        }
                    }
                }
            }
            
            current_level = next_level;
        }

        neighborhood
    }

    /// DFS helper for finding all paths
    fn dfs_paths(
        &self,
        current: &str,
        target: &str,
        path: &mut Vec<String>,
        visited: &mut HashSet<String>,
        max_depth: usize,
        paths: &mut Vec<Vec<String>>,
    ) {
        if path.len() > max_depth {
            return;
        }

        path.push(current.to_string());
        visited.insert(current.to_string());

        if current == target {
            paths.push(path.clone());
        } else {
            if let Some(node) = self.graph.get_all_nodes().iter().find(|n| n.id == current) {
                for dep in self.graph.get_dependencies(&node.id) {
                    if !visited.contains(&dep.id) {
                        self.dfs_paths(&dep.id, target, path, visited, max_depth, paths);
                    }
                }
            }
        }

        path.pop();
        visited.remove(current);
    }
}

/// Transaction manager for batch operations
pub struct TransactionManager {
    db_path: String,
}

impl TransactionManager {
    /// Create new transaction manager
    pub fn new(db_path: String) -> Self {
        TransactionManager { db_path }
    }

    /// Execute batch operations in transaction
    pub fn execute_batch<F>(&self, operations: F) -> Result<(), String>
    where
        F: FnOnce(&Transaction) -> Result<(), String>,
    {
        let mut conn = Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        let tx = conn.transaction()
            .map_err(|e| format!("Failed to start transaction: {}", e))?;

        match operations(&tx) {
            Ok(_) => {
                tx.commit()
                    .map_err(|e| format!("Failed to commit transaction: {}", e))?;
                Ok(())
            }
            Err(e) => {
                tx.rollback()
                    .map_err(|e2| format!("Failed to rollback transaction: {}. Original error: {}", e2, e))?;
                Err(e)
            }
        }
    }

    /// Batch insert nodes
    pub fn batch_insert_nodes(&self, nodes: Vec<CodeNode>) -> Result<(), String> {
        self.execute_batch(|tx| {
            for node in nodes {
                tx.execute(
                    "INSERT INTO nodes (id, name, node_type, file_path, line_start, line_end) 
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        node.id,
                        node.name,
                        format!("{:?}", node.node_type),
                        node.file_path,
                        node.line_start,
                        node.line_end,
                    ],
                ).map_err(|e| format!("Failed to insert node: {}", e))?;
            }
            Ok(())
        })
    }

    /// Batch delete nodes by IDs
    pub fn batch_delete_nodes(&self, node_ids: Vec<String>) -> Result<(), String> {
        self.execute_batch(|tx| {
            for id in node_ids {
                tx.execute("DELETE FROM nodes WHERE id = ?1", params![id])
                    .map_err(|e| format!("Failed to delete node: {}", e))?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: &str, name: &str, node_type: NodeType) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            name: name.to_string(),
            node_type,
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 10,
            semantic_embedding: None,
            code_snippet: None,
            docstring: None,
        }
    }

    #[test]
    fn test_query_builder() {
        let builder = QueryBuilder::new()
            .filter(QueryFilter::NodeType(NodeType::Function))
            .limit(10);

        assert_eq!(builder.filters.len(), 1);
        assert_eq!(builder.limit, Some(10));
    }

    #[test]
    fn test_filter_matches() {
        let node = create_test_node("1", "test_func", NodeType::Function);
        
        assert!(QueryFilter::NodeType(NodeType::Function).matches(&node));
        assert!(QueryFilter::Name("test_func".to_string()).matches(&node));
        assert!(!QueryFilter::NodeType(NodeType::Class).matches(&node));
    }

    #[test]
    fn test_filter_and() {
        let node = create_test_node("1", "test_func", NodeType::Function);
        
        let filter = QueryFilter::And(vec![
            QueryFilter::NodeType(NodeType::Function),
            QueryFilter::Name("test_func".to_string()),
        ]);
        
        assert!(filter.matches(&node));
    }

    #[test]
    fn test_filter_or() {
        let node = create_test_node("1", "test_func", NodeType::Function);
        
        let filter = QueryFilter::Or(vec![
            QueryFilter::NodeType(NodeType::Class),
            QueryFilter::Name("test_func".to_string()),
        ]);
        
        assert!(filter.matches(&node));
    }

    #[test]
    fn test_filter_not() {
        let node = create_test_node("1", "test_func", NodeType::Function);
        
        let filter = QueryFilter::Not(Box::new(QueryFilter::NodeType(NodeType::Class)));
        
        assert!(filter.matches(&node));
    }

    #[test]
    fn test_order_direction() {
        let asc = OrderDirection::Asc;
        let desc = OrderDirection::Desc;
        
        assert!(matches!(asc, OrderDirection::Asc));
        assert!(matches!(desc, OrderDirection::Desc));
    }
}
