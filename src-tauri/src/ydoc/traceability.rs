// YDoc Traceability Query Layer
// Last updated: December 8, 2025
//
// This module provides graph traversal queries for traceability.
// Enables requirement → architecture → spec → code → test chains.

use rusqlite::{Connection, Result as SqliteResult, Error as SqliteError};
use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum TraceabilityError {
    DatabaseError(SqliteError),
    NotFound(String),
    InvalidQuery(String),
}

impl fmt::Display for TraceabilityError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TraceabilityError::DatabaseError(err) => write!(f, "Database error: {}", err),
            TraceabilityError::NotFound(msg) => write!(f, "Not found: {}", msg),
            TraceabilityError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
        }
    }
}

impl Error for TraceabilityError {}

impl From<SqliteError> for TraceabilityError {
    fn from(err: SqliteError) -> Self {
        TraceabilityError::DatabaseError(err)
    }
}

type Result<T> = std::result::Result<T, TraceabilityError>;

/// Traceability entity with metadata
#[derive(Debug, Clone)]
pub struct TraceabilityEntity {
    pub id: String,
    pub entity_type: String,
    pub doc_id: String,
    pub content: String,
    pub depth: usize,
}

/// Edge in the traceability graph
#[derive(Debug, Clone)]
pub struct TraceabilityEdgeInfo {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub metadata: Option<String>,
}

/// Traceability query interface
pub struct TraceabilityQuery {
    db_path: String,
}

impl TraceabilityQuery {
    pub fn new(db_path: String) -> Self {
        Self { db_path }
    }

    /// Get a database connection
    fn get_connection(&self) -> SqliteResult<Connection> {
        Connection::open(&self.db_path)
    }

    /// Find all code implementations for a requirement
    /// Traverses: REQ → traces_to → ARCH → traces_to → SPEC → implements → Code
    pub fn find_code_for_requirement(&self, requirement_id: &str) -> Result<Vec<TraceabilityEntity>> {
        let conn = self.get_connection()?;
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with the requirement
        queue.push_back((requirement_id.to_string(), 0));
        visited.insert(requirement_id.to_string());

        while let Some((current_id, depth)) = queue.pop_front() {
            // Get all outgoing edges
            let mut stmt = conn.prepare(
                "SELECT target_block_id, edge_type FROM graph_edges WHERE source_block_id = ?"
            )?;

            let edges: Vec<(String, String)> = stmt
                .query_map([&current_id], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?
                .collect::<SqliteResult<Vec<_>>>()?;

            for (target_id, edge_type) in edges {
                if visited.contains(&target_id) {
                    continue;
                }
                visited.insert(target_id.clone());

                // Get target block info
                let mut block_stmt = conn.prepare(
                    "SELECT b.block_id, b.yantra_type, b.doc_id, b.content 
                     FROM blocks b 
                     WHERE b.block_id = ?"
                )?;

                if let Ok(entity) = block_stmt.query_row([&target_id], |row| {
                    Ok(TraceabilityEntity {
                        id: row.get(0)?,
                        entity_type: row.get(1)?,
                        doc_id: row.get(2)?,
                        content: row.get(3)?,
                        depth: depth + 1,
                    })
                }) {
                    // Check if this is code (entity_type contains "code" or is an implementation)
                    if edge_type == "implements" || entity.entity_type.to_lowercase().contains("code") {
                        results.push(entity.clone());
                    }

                    // Continue traversal for traces_to and implements edges
                    if edge_type == "traces_to" || edge_type == "implements" {
                        queue.push_back((target_id, depth + 1));
                    }
                }
            }
        }

        if results.is_empty() {
            Err(TraceabilityError::NotFound(format!(
                "No code found for requirement '{}'",
                requirement_id
            )))
        } else {
            Ok(results)
        }
    }

    /// Find all documentation for a code block
    /// Traverses: Code → realized_in/implements (reverse) → SPEC → traces_to (reverse) → ARCH → traces_to (reverse) → REQ
    pub fn find_docs_for_code(&self, code_id: &str) -> Result<Vec<TraceabilityEntity>> {
        let conn = self.get_connection()?;
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with the code block
        queue.push_back((code_id.to_string(), 0));
        visited.insert(code_id.to_string());

        while let Some((current_id, depth)) = queue.pop_front() {
            // Get all incoming edges (reverse traversal)
            let mut stmt = conn.prepare(
                "SELECT source_block_id, edge_type FROM graph_edges WHERE target_block_id = ?"
            )?;

            let edges: Vec<(String, String)> = stmt
                .query_map([&current_id], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?
                .collect::<SqliteResult<Vec<_>>>()?;

            for (source_id, edge_type) in edges {
                if visited.contains(&source_id) {
                    continue;
                }
                visited.insert(source_id.clone());

                // Get source block info
                let mut block_stmt = conn.prepare(
                    "SELECT b.block_id, b.yantra_type, b.doc_id, b.content 
                     FROM blocks b 
                     WHERE b.block_id = ?"
                )?;

                if let Ok(entity) = block_stmt.query_row([&source_id], |row| {
                    Ok(TraceabilityEntity {
                        id: row.get(0)?,
                        entity_type: row.get(1)?,
                        doc_id: row.get(2)?,
                        content: row.get(3)?,
                        depth: depth + 1,
                    })
                }) {
                    // Documentation types: REQ, ARCH, SPEC, ADR, etc.
                    if is_documentation_type(&entity.entity_type) {
                        results.push(entity.clone());
                    }

                    // Continue traversal for traces_to, implements, realized_in edges
                    if edge_type == "traces_to" || edge_type == "implements" || edge_type == "realized_in" {
                        queue.push_back((source_id, depth + 1));
                    }
                }
            }
        }

        if results.is_empty() {
            Err(TraceabilityError::NotFound(format!(
                "No documentation found for code '{}'",
                code_id
            )))
        } else {
            Ok(results)
        }
    }

    /// Impact analysis: Find all entities affected by changes to a given entity
    /// Performs bidirectional BFS to find both upstream and downstream dependencies
    pub fn impact_analysis(&self, entity_id: &str) -> Result<HashMap<String, Vec<TraceabilityEntity>>> {
        let conn = self.get_connection()?;
        let mut downstream = Vec::new();  // Entities that depend on this one
        let mut upstream = Vec::new();     // Entities this one depends on
        let mut visited = HashSet::new();

        // Downstream: Follow outgoing edges
        let mut queue = VecDeque::new();
        queue.push_back((entity_id.to_string(), 0));
        visited.insert(entity_id.to_string());

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= 5 {
                continue; // Limit depth to avoid infinite loops
            }

            let mut stmt = conn.prepare(
                "SELECT target_block_id, edge_type FROM graph_edges WHERE source_block_id = ?"
            )?;

            let edges: Vec<(String, String)> = stmt
                .query_map([&current_id], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?
                .collect::<SqliteResult<Vec<_>>>()?;

            for (target_id, _edge_type) in edges {
                if visited.contains(&target_id) {
                    continue;
                }
                visited.insert(target_id.clone());

                // Get target block info
                let mut block_stmt = conn.prepare(
                    "SELECT b.block_id, b.yantra_type, b.doc_id, b.content 
                     FROM blocks b 
                     WHERE b.block_id = ?"
                )?;

                if let Ok(entity) = block_stmt.query_row([&target_id], |row| {
                    Ok(TraceabilityEntity {
                        id: row.get(0)?,
                        entity_type: row.get(1)?,
                        doc_id: row.get(2)?,
                        content: row.get(3)?,
                        depth: depth + 1,
                    })
                }) {
                    downstream.push(entity);
                    queue.push_back((target_id, depth + 1));
                }
            }
        }

        // Upstream: Follow incoming edges (reset visited for new traversal)
        visited.clear();
        queue.clear();
        queue.push_back((entity_id.to_string(), 0));
        visited.insert(entity_id.to_string());

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= 5 {
                continue;
            }

            let mut stmt = conn.prepare(
                "SELECT source_block_id, edge_type FROM graph_edges WHERE target_block_id = ?"
            )?;

            let edges: Vec<(String, String)> = stmt
                .query_map([&current_id], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?
                .collect::<SqliteResult<Vec<_>>>()?;

            for (source_id, _edge_type) in edges {
                if visited.contains(&source_id) {
                    continue;
                }
                visited.insert(source_id.clone());

                let mut block_stmt = conn.prepare(
                    "SELECT b.block_id, b.yantra_type, b.doc_id, b.content 
                     FROM blocks b 
                     WHERE b.block_id = ?"
                )?;

                if let Ok(entity) = block_stmt.query_row([&source_id], |row| {
                    Ok(TraceabilityEntity {
                        id: row.get(0)?,
                        entity_type: row.get(1)?,
                        doc_id: row.get(2)?,
                        content: row.get(3)?,
                        depth: depth + 1,
                    })
                }) {
                    upstream.push(entity);
                    queue.push_back((source_id, depth + 1));
                }
            }
        }

        let mut result = HashMap::new();
        result.insert("downstream".to_string(), downstream);
        result.insert("upstream".to_string(), upstream);
        Ok(result)
    }

    /// Find all test blocks for a code block
    /// Traverses: Code → tested_by → Test
    pub fn find_tests_for_code(&self, code_id: &str) -> Result<Vec<TraceabilityEntity>> {
        let conn = self.get_connection()?;
        let mut results = Vec::new();

        let mut stmt = conn.prepare(
            "SELECT b.block_id, b.yantra_type, b.doc_id, b.content
             FROM graph_edges e
             JOIN blocks b ON e.target_block_id = b.block_id
             WHERE e.source_block_id = ? AND e.edge_type = 'tested_by'"
        )?;

        let tests = stmt.query_map([code_id], |row| {
            Ok(TraceabilityEntity {
                id: row.get(0)?,
                entity_type: row.get(1)?,
                doc_id: row.get(2)?,
                content: row.get(3)?,
                depth: 1,
            })
        })?;

        for test in tests {
            if let Ok(entity) = test {
                results.push(entity);
            }
        }

        if results.is_empty() {
            Err(TraceabilityError::NotFound(format!(
                "No tests found for code '{}'",
                code_id
            )))
        } else {
            Ok(results)
        }
    }

    /// Find all requirements without test coverage
    /// Returns REQs that don't have a complete chain to tested code
    pub fn find_untested_requirements(&self) -> Result<Vec<TraceabilityEntity>> {
        let conn = self.get_connection()?;
        let mut results = Vec::new();

        // Get all requirement blocks
        let mut stmt = conn.prepare(
            "SELECT block_id, yantra_type, doc_id, content 
             FROM blocks 
             WHERE yantra_type = 'requirement'"
        )?;

        let requirements = stmt.query_map([], |row| {
            Ok(TraceabilityEntity {
                id: row.get(0)?,
                entity_type: row.get(1)?,
                doc_id: row.get(2)?,
                content: row.get(3)?,
                depth: 0,
            })
        })?;

        for req in requirements {
            if let Ok(requirement) = req {
                // Check if requirement has path to tested code
                let has_tests = self.requirement_has_test_coverage(&conn, &requirement.id)?;
                if !has_tests {
                    results.push(requirement);
                }
            }
        }

        Ok(results)
    }

    /// Helper: Check if a requirement has test coverage
    fn requirement_has_test_coverage(&self, conn: &Connection, req_id: &str) -> Result<bool> {
        // Try to find code for this requirement
        let code_blocks = match self.find_code_for_requirement(req_id) {
            Ok(blocks) => blocks,
            Err(_) => return Ok(false), // No code = no tests
        };

        // Check if any code block has tests
        for code_block in code_blocks {
            let mut stmt = conn.prepare(
                "SELECT COUNT(*) FROM graph_edges 
                 WHERE source_block_id = ? AND edge_type = 'tested_by'"
            )?;

            let count: i64 = stmt.query_row([&code_block.id], |row| row.get(0))?;
            if count > 0 {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get full traceability chain for an entity (both directions)
    pub fn get_traceability_chain(&self, entity_id: &str) -> Result<HashMap<String, Vec<TraceabilityEntity>>> {
        let mut result = HashMap::new();

        // Get the entity itself
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT block_id, yantra_type, doc_id, content FROM blocks WHERE block_id = ?"
        )?;

        let entity: TraceabilityEntity = stmt.query_row([entity_id], |row| {
            Ok(TraceabilityEntity {
                id: row.get(0)?,
                entity_type: row.get(1)?,
                doc_id: row.get(2)?,
                content: row.get(3)?,
                depth: 0,
            })
        })?;

        result.insert("entity".to_string(), vec![entity.clone()]);

        // Get upstream (dependencies)
        let impact = self.impact_analysis(entity_id)?;
        if let Some(upstream) = impact.get("upstream") {
            result.insert("dependencies".to_string(), upstream.clone());
        }
        if let Some(downstream) = impact.get("downstream") {
            result.insert("dependents".to_string(), downstream.clone());
        }

        // Get tests if this is code
        if entity.entity_type.to_lowercase().contains("code") {
            if let Ok(tests) = self.find_tests_for_code(entity_id) {
                result.insert("tests".to_string(), tests);
            }
        }

        Ok(result)
    }

    /// Get all edges for visualization
    pub fn get_all_edges(&self) -> Result<Vec<TraceabilityEdgeInfo>> {
        let conn = self.get_connection()?;
        let mut results = Vec::new();

        let mut stmt = conn.prepare(
            "SELECT source_block_id, target_block_id, edge_type, metadata 
             FROM graph_edges"
        )?;

        let edges = stmt.query_map([], |row| {
            Ok(TraceabilityEdgeInfo {
                source_id: row.get(0)?,
                target_id: row.get(1)?,
                edge_type: row.get(2)?,
                metadata: row.get(3)?,
            })
        })?;

        for edge in edges {
            if let Ok(e) = edge {
                results.push(e);
            }
        }

        Ok(results)
    }

    /// Get statistics about traceability coverage
    pub fn get_coverage_stats(&self) -> Result<HashMap<String, i64>> {
        let conn = self.get_connection()?;
        let mut stats = HashMap::new();

        // Total requirements
        let total_reqs: i64 = conn.query_row(
            "SELECT COUNT(*) FROM blocks WHERE yantra_type = 'requirement'",
            [],
            |row| row.get(0),
        )?;
        stats.insert("total_requirements".to_string(), total_reqs);

        // Requirements with code
        let reqs_with_code: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT source_block_id) FROM graph_edges e
             JOIN blocks b ON e.source_block_id = b.block_id
             WHERE b.yantra_type = 'requirement' AND e.edge_type IN ('traces_to', 'implements')",
            [],
            |row| row.get(0),
        )?;
        stats.insert("requirements_with_code".to_string(), reqs_with_code);

        // Total code blocks
        let total_code: i64 = conn.query_row(
            "SELECT COUNT(*) FROM blocks WHERE yantra_type LIKE '%code%'",
            [],
            |row| row.get(0),
        )?;
        stats.insert("total_code_blocks".to_string(), total_code);

        // Code blocks with tests
        let code_with_tests: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT source_block_id) FROM graph_edges 
             WHERE edge_type = 'tested_by'",
            [],
            |row| row.get(0),
        )?;
        stats.insert("code_with_tests".to_string(), code_with_tests);

        // Total edges
        let total_edges: i64 = conn.query_row(
            "SELECT COUNT(*) FROM graph_edges",
            [],
            |row| row.get(0),
        )?;
        stats.insert("total_edges".to_string(), total_edges);

        Ok(stats)
    }
}

/// Helper: Check if a type is a documentation type
fn is_documentation_type(yantra_type: &str) -> bool {
    matches!(
        yantra_type.to_lowercase().as_str(),
        "requirement" | "adr" | "architecture" | "specification" | "plan" | "tech_guide" | "api_guide"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_db() -> (TempDir, String) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db").to_str().unwrap().to_string();
        
        let conn = Connection::open(&db_path).unwrap();
        
        // Create tables
        conn.execute_batch(
            "CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL,
                title TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                yantra_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL
            );
            CREATE TABLE graph_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_block_id TEXT NOT NULL,
                target_block_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                metadata TEXT
            );"
        ).unwrap();

        // Insert test data
        conn.execute(
            "INSERT INTO blocks (block_id, doc_id, yantra_type, content, created_at, modified_at)
             VALUES ('req-1', 'doc-1', 'requirement', 'User must login', '2025-01-01', '2025-01-01'),
                    ('arch-1', 'doc-2', 'architecture', 'Auth system', '2025-01-01', '2025-01-01'),
                    ('spec-1', 'doc-3', 'specification', 'Login endpoint', '2025-01-01', '2025-01-01'),
                    ('code-1', 'doc-4', 'code', 'function login()', '2025-01-01', '2025-01-01'),
                    ('test-1', 'doc-5', 'test', 'test_login()', '2025-01-01', '2025-01-01')",
            [],
        ).unwrap();

        conn.execute(
            "INSERT INTO graph_edges (source_block_id, target_block_id, edge_type)
             VALUES ('req-1', 'arch-1', 'traces_to'),
                    ('arch-1', 'spec-1', 'traces_to'),
                    ('spec-1', 'code-1', 'implements'),
                    ('code-1', 'test-1', 'tested_by')",
            [],
        ).unwrap();

        (temp_dir, db_path)
    }

    #[test]
    fn test_find_code_for_requirement() {
        let (_temp, db_path) = setup_test_db();
        let query = TraceabilityQuery::new(db_path);

        let result = query.find_code_for_requirement("req-1").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "code-1");
    }

    #[test]
    fn test_find_docs_for_code() {
        let (_temp, db_path) = setup_test_db();
        let query = TraceabilityQuery::new(db_path);

        let result = query.find_docs_for_code("code-1").unwrap();
        assert!(result.len() >= 1);
        // Should find spec-1, arch-1, req-1
        let ids: Vec<String> = result.iter().map(|e| e.id.clone()).collect();
        assert!(ids.contains(&"spec-1".to_string()));
    }

    #[test]
    fn test_find_tests_for_code() {
        let (_temp, db_path) = setup_test_db();
        let query = TraceabilityQuery::new(db_path);

        let result = query.find_tests_for_code("code-1").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "test-1");
    }

    #[test]
    fn test_impact_analysis() {
        let (_temp, db_path) = setup_test_db();
        let query = TraceabilityQuery::new(db_path);

        let result = query.impact_analysis("spec-1").unwrap();
        assert!(result.contains_key("downstream"));
        assert!(result.contains_key("upstream"));
        
        let downstream = result.get("downstream").unwrap();
        assert!(downstream.iter().any(|e| e.id == "code-1"));
    }

    #[test]
    fn test_get_coverage_stats() {
        let (_temp, db_path) = setup_test_db();
        let query = TraceabilityQuery::new(db_path);

        let stats = query.get_coverage_stats().unwrap();
        assert_eq!(stats.get("total_requirements").unwrap(), &1);
        assert_eq!(stats.get("total_edges").unwrap(), &4);
    }
}
