// File: src-tauri/src/gnn/persistence.rs
// Purpose: SQLite persistence for graph data
// Dependencies: rusqlite
// Last Updated: November 20, 2025

use super::{CodeNode, NodeType, EdgeType};
use super::graph::CodeGraph;
use rusqlite::{Connection, params, Result as SqlResult};
use std::path::Path;

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Create or open database at specified path
    pub fn new(db_path: &Path) -> SqlResult<Self> {
        let conn = Connection::open(db_path)?;
        
        let db = Self { conn };
        db.create_tables()?;
        
        Ok(db)
    }
    
    /// Create tables if they don't exist
    fn create_tables(&self) -> SqlResult<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL
            )",
            [],
        )?;
        
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                edge_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )",
            [],
        )?;
        
        // Create indices for faster queries
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)",
            [],
        )?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path)",
            [],
        )?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
            [],
        )?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
            [],
        )?;
        
        Ok(())
    }
    
    /// Save entire graph to database
    pub fn save_graph(&self, graph: &CodeGraph) -> SqlResult<()> {
        // Begin transaction for better performance
        let tx = self.conn.unchecked_transaction()?;
        
        // Clear existing data
        tx.execute("DELETE FROM edges", [])?;
        tx.execute("DELETE FROM nodes", [])?;
        
        let (nodes, edges) = graph.export();
        
        // Insert nodes
        for node in nodes {
            tx.execute(
                "INSERT INTO nodes (id, node_type, name, file_path, line_start, line_end)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    node.id,
                    node_type_to_string(&node.node_type),
                    node.name,
                    node.file_path,
                    node.line_start,
                    node.line_end,
                ],
            )?;
        }
        
        // Insert edges
        for (source_id, target_id, edge_type) in edges {
            tx.execute(
                "INSERT INTO edges (edge_type, source_id, target_id)
                 VALUES (?1, ?2, ?3)",
                params![
                    edge_type_to_string(&edge_type),
                    source_id,
                    target_id,
                ],
            )?;
        }
        
        tx.commit()?;
        Ok(())
    }
    
    /// Load entire graph from database
    pub fn load_graph(&self) -> SqlResult<CodeGraph> {
        let mut graph = CodeGraph::new();
        
        // Load nodes
        let mut stmt = self.conn.prepare(
            "SELECT id, node_type, name, file_path, line_start, line_end FROM nodes"
        )?;
        
        let nodes: Vec<CodeNode> = stmt.query_map([], |row| {
            Ok(CodeNode {
                id: row.get(0)?,
                node_type: string_to_node_type(&row.get::<_, String>(1)?),
                name: row.get(2)?,
                file_path: row.get(3)?,
                line_start: row.get(4)?,
                line_end: row.get(5)?,
            })
        })?
        .collect::<SqlResult<Vec<_>>>()?;
        
        // Load edges
        let mut stmt = self.conn.prepare(
            "SELECT edge_type, source_id, target_id FROM edges"
        )?;
        
        let edges: Vec<(String, String, EdgeType)> = stmt.query_map([], |row| {
            Ok((
                row.get(1)?,
                row.get(2)?,
                string_to_edge_type(&row.get::<_, String>(0)?),
            ))
        })?
        .collect::<SqlResult<Vec<_>>>()?;
        
        graph.import(nodes, edges);
        
        Ok(graph)
    }
    
    /// Get statistics about the graph
    pub fn get_stats(&self) -> SqlResult<GraphStats> {
        let node_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM nodes",
            [],
            |row| row.get(0),
        )?;
        
        let edge_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges",
            [],
            |row| row.get(0),
        )?;
        
        let file_count: i64 = self.conn.query_row(
            "SELECT COUNT(DISTINCT file_path) FROM nodes",
            [],
            |row| row.get(0),
        )?;
        
        Ok(GraphStats {
            node_count: node_count as usize,
            edge_count: edge_count as usize,
            file_count: file_count as usize,
        })
    }
}

#[derive(Debug)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub file_count: usize,
}

fn node_type_to_string(node_type: &NodeType) -> &str {
    match node_type {
        NodeType::Function => "function",
        NodeType::Class => "class",
        NodeType::Variable => "variable",
        NodeType::Import => "import",
        NodeType::Module => "module",
    }
}

fn string_to_node_type(s: &str) -> NodeType {
    match s {
        "function" => NodeType::Function,
        "class" => NodeType::Class,
        "variable" => NodeType::Variable,
        "import" => NodeType::Import,
        "module" => NodeType::Module,
        _ => NodeType::Function, // Default fallback
    }
}

fn edge_type_to_string(edge_type: &EdgeType) -> &str {
    match edge_type {
        EdgeType::Calls => "calls",
        EdgeType::Uses => "uses",
        EdgeType::Imports => "imports",
        EdgeType::Inherits => "inherits",
        EdgeType::Defines => "defines",
    }
}

fn string_to_edge_type(s: &str) -> EdgeType {
    match s {
        "calls" => EdgeType::Calls,
        "uses" => EdgeType::Uses,
        "imports" => EdgeType::Imports,
        "inherits" => EdgeType::Inherits,
        "defines" => EdgeType::Defines,
        _ => EdgeType::Calls, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_database_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = Database::new(&db_path);
        assert!(db.is_ok());
    }
    
    #[test]
    fn test_save_and_load_graph() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = Database::new(&db_path).unwrap();
        
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
        
        // Save
        assert!(db.save_graph(&graph).is_ok());
        
        // Load
        let loaded_graph = db.load_graph().unwrap();
        assert_eq!(loaded_graph.node_count(), 1);
        
        let node = loaded_graph.get_node("test::func1").unwrap();
        assert_eq!(node.name, "func1");
    }
}
