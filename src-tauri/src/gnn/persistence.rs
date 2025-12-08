// File: src-tauri/src/gnn/persistence.rs
// Purpose: SQLite persistence for graph data with WAL mode and connection pooling
// Dependencies: rusqlite, r2d2, r2d2_sqlite
// Last Updated: December 8, 2025 - Added WAL mode and connection pooling

use super::{CodeNode, NodeType, EdgeType};
use super::graph::CodeGraph;
use rusqlite::{params, Result as SqlResult};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use std::path::Path;

pub struct Database {
    pool: Pool<SqliteConnectionManager>,
}

impl Database {
    /// Create or open database at specified path with WAL mode and connection pooling
    pub fn new(db_path: &Path) -> Result<Self, String> {
        let manager = SqliteConnectionManager::file(db_path)
            .with_init(|conn| {
                // Enable WAL mode for better concurrency and corruption protection
                conn.execute_batch("
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA busy_timeout=5000;
                ")?;
                Ok(())
            });
        
        let pool = Pool::builder()
            .max_size(10)  // Max 10 connections for concurrent access
            .min_idle(Some(2))  // Keep at least 2 idle connections ready
            .build(manager)
            .map_err(|e| format!("Failed to create connection pool: {}", e))?;
        
        let db = Self { pool };
        db.create_tables()
            .map_err(|e| format!("Failed to create tables: {}", e))?;
        
        Ok(db)
    }
    
    /// Get a connection from the pool
    fn get_conn(&self) -> Result<PooledConnection<SqliteConnectionManager>, String> {
        self.pool.get().map_err(|e| format!("Failed to get connection: {}", e))
    }
    
    /// Create tables if they don't exist
    fn create_tables(&self) -> SqlResult<()> {
        let conn = self.get_conn().map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::Other, e))))?;
        
        conn.execute(
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
        
        conn.execute(
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)",
            [],
        )?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path)",
            [],
        )?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
            [],
        )?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
            [],
        )?;
        
        Ok(())
    }
    
    /// Save entire graph to database
    pub fn save_graph(&self, graph: &CodeGraph) -> Result<(), String> {
        let conn = self.get_conn()?;
        
        // Begin transaction for better performance
        let tx = conn.unchecked_transaction()
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;
        
        // Clear existing data
        tx.execute("DELETE FROM edges", [])
            .map_err(|e| format!("Failed to delete edges: {}", e))?;
        tx.execute("DELETE FROM nodes", [])
            .map_err(|e| format!("Failed to delete nodes: {}", e))?;
        
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
            ).map_err(|e| format!("Failed to insert node: {}", e))?;
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
            ).map_err(|e| format!("Failed to insert edge: {}", e))?;
        }
        
        tx.commit().map_err(|e| format!("Failed to commit transaction: {}", e))?;
        Ok(())
    }
    
    /// Load entire graph from database
    pub fn load_graph(&self) -> Result<CodeGraph, String> {
        let conn = self.get_conn()?;
        let mut graph = CodeGraph::new();
        
        // Load nodes
        let mut stmt = conn.prepare(
            "SELECT id, node_type, name, file_path, line_start, line_end FROM nodes"
        ).map_err(|e| format!("Failed to prepare nodes query: {}", e))?;
        
        let nodes: Vec<CodeNode> = stmt.query_map([], |row| {
            Ok(CodeNode {
                id: row.get(0)?,
                node_type: string_to_node_type(&row.get::<_, String>(1)?),
                name: row.get(2)?,
                file_path: row.get(3)?,
                line_start: row.get(4)?,
                line_end: row.get(5)?,
                ..Default::default()
            })
        }).map_err(|e| format!("Failed to query nodes: {}", e))?
        .collect::<SqlResult<Vec<_>>>()
        .map_err(|e| format!("Failed to collect nodes: {}", e))?;
        
        // Load edges
        let mut stmt = conn.prepare(
            "SELECT edge_type, source_id, target_id FROM edges"
        ).map_err(|e| format!("Failed to prepare edges query: {}", e))?;
        
        let edges: Vec<(String, String, EdgeType)> = stmt.query_map([], |row| {
            Ok((
                row.get(1)?,
                row.get(2)?,
                string_to_edge_type(&row.get::<_, String>(0)?),
            ))
        }).map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<SqlResult<Vec<_>>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;
        
        graph.import(nodes, edges);
        
        Ok(graph)
    }
    
    /// Get statistics about the graph
    pub fn get_stats(&self) -> Result<GraphStats, String> {
        let conn = self.get_conn()?;
        
        let node_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM nodes",
            [],
            |row| row.get(0),
        ).map_err(|e| format!("Failed to count nodes: {}", e))?;
        
        let edge_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM edges",
            [],
            |row| row.get(0),
        ).map_err(|e| format!("Failed to count edges: {}", e))?;
        
        let file_count: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT file_path) FROM nodes",
            [],
            |row| row.get(0),
        ).map_err(|e| format!("Failed to count files: {}", e))?;
        
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
        NodeType::Package { .. } => "package",
    }
}

fn string_to_node_type(s: &str) -> NodeType {
    match s {
        "function" => NodeType::Function,
        "class" => NodeType::Class,
        "variable" => NodeType::Variable,
        "import" => NodeType::Import,
        "module" => NodeType::Module,
        "package" => NodeType::Module, // Packages will be reconstructed from JSON
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
        EdgeType::Tests => "tests",
        EdgeType::TestDependency => "test_dependency",
        EdgeType::UsesPackage => "uses_package",
        EdgeType::DependsOn => "depends_on",
        EdgeType::ConflictsWith => "conflicts_with",
    }
}

fn string_to_edge_type(s: &str) -> EdgeType {
    match s {
        "calls" => EdgeType::Calls,
        "uses" => EdgeType::Uses,
        "imports" => EdgeType::Imports,
        "inherits" => EdgeType::Inherits,
        "defines" => EdgeType::Defines,
        "tests" => EdgeType::Tests,
        "test_dependency" => EdgeType::TestDependency,
        "uses_package" => EdgeType::UsesPackage,
        "depends_on" => EdgeType::DependsOn,
        "conflicts_with" => EdgeType::ConflictsWith,
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
            ..Default::default()
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
