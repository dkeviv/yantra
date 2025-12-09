// File: src-tauri/src/ydoc/database.rs
// Purpose: YDoc SQLite database schema and operations
// Last Updated: December 8, 2025

use rusqlite::{params, Connection, Result as SqlResult};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::{DocumentType, BlockType, BlockStatus};

/// Document metadata stored in SQLite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,              // UUID
    pub doc_type: DocumentType,
    pub title: String,
    pub file_path: String,       // Relative path to .ydoc file
    pub created_by: String,      // "user" or "agent"
    pub created_at: String,      // ISO-8601
    pub modified_at: String,     // ISO-8601
    pub version: String,         // Default "1.0.0"
    pub status: String,          // Default "draft"
}

/// Block metadata stored in SQLite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMetadata {
    pub id: String,              // Block UUID
    pub doc_id: String,          // Foreign key to documents
    pub cell_index: usize,       // Position in document
    pub cell_type: String,       // "markdown", "code", "raw"
    pub yantra_type: BlockType,
    pub content: String,         // Actual markdown/code content
    pub created_by: String,      // "user" or "agent"
    pub created_at: String,      // ISO-8601
    pub modified_by: String,     // Last modifier
    pub modified_at: String,     // ISO-8601
    pub modifier_id: String,     // "user-123" or "agent-task-456"
    pub status: BlockStatus,
}

/// Traceability edge connecting docs to code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceabilityEdge {
    pub id: Option<i64>,         // Auto-increment
    pub source_id: String,       // block_id, file_id, function_id
    pub source_type: String,     // "doc_block", "code_file", "function", "class"
    pub target_id: String,       // Target entity ID
    pub target_type: String,     // Target entity type
    pub edge_type: String,       // "traces_to", "implements", "realized_in", "tested_by", "documents", "has_issue"
    pub created_at: String,      // ISO-8601
    pub metadata: Option<String>, // JSON: {line_range, confidence, etc.}
}

/// YDoc database with connection pooling + WAL mode
pub struct YDocDatabase {
    pool: Pool<SqliteConnectionManager>,
    pub db_path: String,
}

impl YDocDatabase {
    /// Create new YDoc database with WAL mode and connection pooling
    pub fn new(db_path: &Path) -> Result<Self, String> {
        let db_path_str = db_path.to_str()
            .ok_or_else(|| "Invalid database path".to_string())?
            .to_string();
        
        let manager = SqliteConnectionManager::file(db_path)
            .with_init(|conn| {
                // Enable WAL mode for concurrent reads/writes
                conn.execute_batch("
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA busy_timeout=5000;
                    PRAGMA foreign_keys=ON;
                ")?;
                Ok(())
            });

        let pool = Pool::builder()
            .max_size(10)
            .min_idle(Some(2))
            .build(manager)
            .map_err(|e| format!("Failed to create connection pool: {}", e))?;

        let db = Self { 
            pool,
            db_path: db_path_str,
        };
        db.create_tables()?;
        Ok(db)
    }

    /// Get pooled connection
    fn get_conn(&self) -> Result<PooledConnection<SqliteConnectionManager>, String> {
        self.pool.get()
            .map_err(|e| format!("Failed to get database connection: {}", e))
    }

    /// Get a direct rusqlite connection (for manager operations)
    pub fn get_connection(&self) -> Result<Connection, rusqlite::Error> {
        Connection::open(&self.db_path)
    }

    /// Create all YDoc tables with FTS5 for full-text search
    fn create_tables(&self) -> Result<(), String> {
        let conn = self.get_conn()?;

        // Documents table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                version TEXT DEFAULT '1.0.0',
                status TEXT DEFAULT 'draft'
            )",
            [],
        ).map_err(|e| format!("Failed to create documents table: {}", e))?;

        // Blocks table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS blocks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                cell_index INTEGER NOT NULL,
                cell_type TEXT NOT NULL,
                yantra_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                modified_by TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                modifier_id TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )",
            [],
        ).map_err(|e| format!("Failed to create blocks table: {}", e))?;

        // Graph edges table for traceability
        conn.execute(
            "CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )",
            [],
        ).map_err(|e| format!("Failed to create graph_edges table: {}", e))?;

        // Create FTS5 virtual table for full-text search
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
                block_id UNINDEXED,
                content,
                yantra_type UNINDEXED,
                doc_id UNINDEXED
            )",
            [],
        ).map_err(|e| format!("Failed to create FTS5 table: {}", e))?;

        // Create indices for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blocks_doc_id ON blocks(doc_id)",
            [],
        ).map_err(|e| format!("Failed to create blocks doc_id index: {}", e))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blocks_yantra_type ON blocks(yantra_type)",
            [],
        ).map_err(|e| format!("Failed to create blocks yantra_type index: {}", e))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id, source_type)",
            [],
        ).map_err(|e| format!("Failed to create edges source index: {}", e))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id, target_type)",
            [],
        ).map_err(|e| format!("Failed to create edges target index: {}", e))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type)",
            [],
        ).map_err(|e| format!("Failed to create edges type index: {}", e))?;

        Ok(())
    }

    /// Insert a new document
    pub fn insert_document(&self, doc: &DocumentMetadata) -> Result<(), String> {
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT INTO documents (id, doc_type, title, file_path, created_by, created_at, modified_at, version, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                doc.id,
                doc.doc_type.code(),
                doc.title,
                doc.file_path,
                doc.created_by,
                doc.created_at,
                doc.modified_at,
                doc.version,
                doc.status,
            ],
        ).map_err(|e| format!("Failed to insert document: {}", e))?;
        Ok(())
    }

    /// Insert a new block and update FTS5
    pub fn insert_block(&self, block: &BlockMetadata) -> Result<(), String> {
        let conn = self.get_conn()?;
        
        // Insert into blocks table
        conn.execute(
            "INSERT INTO blocks (id, doc_id, cell_index, cell_type, yantra_type, content,
                                created_by, created_at, modified_by, modified_at, modifier_id, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                block.id,
                block.doc_id,
                block.cell_index,
                block.cell_type,
                format!("{:?}", block.yantra_type), // Convert enum to string
                block.content,
                block.created_by,
                block.created_at,
                block.modified_by,
                block.modified_at,
                block.modifier_id,
                format!("{:?}", block.status), // Convert enum to string
            ],
        ).map_err(|e| format!("Failed to insert block: {}", e))?;

        // Update FTS5 index
        conn.execute(
            "INSERT INTO blocks_fts (block_id, content, yantra_type, doc_id)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                block.id,
                block.content,
                format!("{:?}", block.yantra_type),
                block.doc_id,
            ],
        ).map_err(|e| format!("Failed to update FTS5 index: {}", e))?;

        Ok(())
    }

    /// Insert a traceability edge
    pub fn insert_edge(&self, edge: &TraceabilityEdge) -> Result<i64, String> {
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT INTO graph_edges (source_id, source_type, target_id, target_type, edge_type, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                edge.source_id,
                edge.source_type,
                edge.target_id,
                edge.target_type,
                edge.edge_type,
                edge.created_at,
                edge.metadata,
            ],
        ).map_err(|e| format!("Failed to insert edge: {}", e))?;

        Ok(conn.last_insert_rowid())
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: &str) -> Result<Option<DocumentMetadata>, String> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, doc_type, title, file_path, created_by, created_at, modified_at, version, status
             FROM documents WHERE id = ?1"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let doc = match stmt.query_row([doc_id], |row| {
            let doc_type_code: String = row.get(1)?;
            Ok(DocumentMetadata {
                id: row.get(0)?,
                doc_type: DocumentType::from_code(&doc_type_code)
                    .unwrap_or(DocumentType::TechGuide),
                title: row.get(2)?,
                file_path: row.get(3)?,
                created_by: row.get(4)?,
                created_at: row.get(5)?,
                modified_at: row.get(6)?,
                version: row.get(7)?,
                status: row.get(8)?,
            })
        }) {
            Ok(doc) => Some(doc),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(format!("Failed to get document: {}", e)),
        };

        Ok(doc)
    }

    /// Get all blocks for a document
    pub fn get_blocks_for_document(&self, doc_id: &str) -> Result<Vec<BlockMetadata>, String> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, doc_id, cell_index, cell_type, yantra_type, content,
                    created_by, created_at, modified_by, modified_at, modifier_id, status
             FROM blocks WHERE doc_id = ?1 ORDER BY cell_index"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let blocks = stmt.query_map([doc_id], |row| {
            Ok(BlockMetadata {
                id: row.get(0)?,
                doc_id: row.get(1)?,
                cell_index: row.get(2)?,
                cell_type: row.get(3)?,
                yantra_type: BlockType::Requirement, // TODO: Parse from string
                content: row.get(5)?,
                created_by: row.get(6)?,
                created_at: row.get(7)?,
                modified_by: row.get(8)?,
                modified_at: row.get(9)?,
                modifier_id: row.get(10)?,
                status: BlockStatus::Draft, // TODO: Parse from string
            })
        })
        .map_err(|e| format!("Failed to query blocks: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect blocks: {}", e))?;

        Ok(blocks)
    }

    /// Full-text search across all blocks
    pub fn search_blocks(&self, query: &str) -> Result<Vec<BlockMetadata>, String> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare(
            "SELECT b.id, b.doc_id, b.cell_index, b.cell_type, b.yantra_type, b.content,
                    b.created_by, b.created_at, b.modified_by, b.modified_at, b.modifier_id, b.status
             FROM blocks b
             JOIN blocks_fts fts ON b.id = fts.block_id
             WHERE blocks_fts MATCH ?1
             ORDER BY rank"
        ).map_err(|e| format!("Failed to prepare search statement: {}", e))?;

        let blocks = stmt.query_map([query], |row| {
            Ok(BlockMetadata {
                id: row.get(0)?,
                doc_id: row.get(1)?,
                cell_index: row.get(2)?,
                cell_type: row.get(3)?,
                yantra_type: BlockType::Requirement, // TODO: Parse from string
                content: row.get(5)?,
                created_by: row.get(6)?,
                created_at: row.get(7)?,
                modified_by: row.get(8)?,
                modified_at: row.get(9)?,
                modifier_id: row.get(10)?,
                status: BlockStatus::Draft, // TODO: Parse from string
            })
        })
        .map_err(|e| format!("Failed to execute search: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect search results: {}", e))?;

        Ok(blocks)
    }

    /// Get edges from a source
    pub fn get_edges_from_source(&self, source_id: &str, edge_type: Option<&str>) -> Result<Vec<TraceabilityEdge>, String> {
        let conn = self.get_conn()?;
        
        let (sql, params): (String, Vec<String>) = if let Some(et) = edge_type {
            ("SELECT id, source_id, source_type, target_id, target_type, edge_type, created_at, metadata
              FROM graph_edges WHERE source_id = ?1 AND edge_type = ?2".to_string(),
             vec![source_id.to_string(), et.to_string()])
        } else {
            ("SELECT id, source_id, source_type, target_id, target_type, edge_type, created_at, metadata
              FROM graph_edges WHERE source_id = ?1".to_string(),
             vec![source_id.to_string()])
        };

        let mut stmt = conn.prepare(&sql)
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let edges = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
            Ok(TraceabilityEdge {
                id: row.get(0)?,
                source_id: row.get(1)?,
                source_type: row.get(2)?,
                target_id: row.get(3)?,
                target_type: row.get(4)?,
                edge_type: row.get(5)?,
                created_at: row.get(6)?,
                metadata: row.get(7)?,
            })
        })
        .map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;

        Ok(edges)
    }

    /// Get edges to a target (reverse lookup)
    pub fn get_edges_to_target(&self, target_id: &str, edge_type: Option<&str>) -> Result<Vec<TraceabilityEdge>, String> {
        let conn = self.get_conn()?;
        
        let (sql, params): (String, Vec<String>) = if let Some(et) = edge_type {
            ("SELECT id, source_id, source_type, target_id, target_type, edge_type, created_at, metadata
              FROM graph_edges WHERE target_id = ?1 AND edge_type = ?2".to_string(),
             vec![target_id.to_string(), et.to_string()])
        } else {
            ("SELECT id, source_id, source_type, target_id, target_type, edge_type, created_at, metadata
              FROM graph_edges WHERE target_id = ?1".to_string(),
             vec![target_id.to_string()])
        };

        let mut stmt = conn.prepare(&sql)
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let edges = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
            Ok(TraceabilityEdge {
                id: row.get(0)?,
                source_id: row.get(1)?,
                source_type: row.get(2)?,
                target_id: row.get(3)?,
                target_type: row.get(4)?,
                edge_type: row.get(5)?,
                created_at: row.get(6)?,
                metadata: row.get(7)?,
            })
        })
        .map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;

        Ok(edges)
    }

    /// Delete a document and all its blocks (CASCADE)
    pub fn delete_document(&self, doc_id: &str) -> Result<(), String> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM documents WHERE id = ?1", [doc_id])
            .map_err(|e| format!("Failed to delete document: {}", e))?;
        Ok(())
    }

    /// Update document metadata
    pub fn update_document(&self, doc: &DocumentMetadata) -> Result<(), String> {
        let conn = self.get_conn()?;
        conn.execute(
            "UPDATE documents SET title = ?1, modified_at = ?2, version = ?3, status = ?4
             WHERE id = ?5",
            params![doc.title, doc.modified_at, doc.version, doc.status, doc.id],
        ).map_err(|e| format!("Failed to update document: {}", e))?;
        Ok(())
    }

    /// Archive old test results (>30 days) with summary statistics
    /// Returns number of archived documents
    pub fn archive_old_test_results(&self, days_threshold: i64) -> Result<usize, String> {
        let conn = self.get_conn()?;
        
        // Calculate cutoff date (30 days ago by default)
        let cutoff_date = chrono::Utc::now() - chrono::Duration::days(days_threshold);
        let cutoff_str = cutoff_date.to_rfc3339();
        
        // Find test result documents older than threshold
        let mut stmt = conn.prepare(
            "SELECT id, title, created_at, modified_at FROM documents 
             WHERE doc_type = 'RESULT' AND modified_at < ?1"
        ).map_err(|e| format!("Failed to prepare archive query: {}", e))?;
        
        let old_docs: Vec<(String, String, String, String)> = stmt.query_map([&cutoff_str], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .map_err(|e| format!("Failed to query old test results: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect old test results: {}", e))?;
        
        let count = old_docs.len();
        
        // For each old test document, create summary and archive
        for (doc_id, title, created_at, modified_at) in old_docs {
            // Get block count and test statistics
            let mut block_stmt = conn.prepare(
                "SELECT COUNT(*), 
                        SUM(CASE WHEN content LIKE '%PASS%' OR content LIKE '%✓%' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN content LIKE '%FAIL%' OR content LIKE '%✗%' THEN 1 ELSE 0 END)
                 FROM blocks WHERE doc_id = ?1"
            ).map_err(|e| format!("Failed to prepare stats query: {}", e))?;
            
            let (total_blocks, passed, failed): (i64, i64, i64) = block_stmt.query_row([&doc_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            }).map_err(|e| format!("Failed to get block stats: {}", e))?;
            
            // Create archive summary as a new block in a special archive document
            let summary = format!(
                "ARCHIVED: {}\nOriginal ID: {}\nCreated: {}\nModified: {}\nBlocks: {}\nPassed: {}\nFailed: {}",
                title, doc_id, created_at, modified_at, total_blocks, passed, failed
            );
            
            // Get or create archive document
            let archive_doc_id = "archive-test-results";
            let archive_exists = conn.query_row(
                "SELECT COUNT(*) FROM documents WHERE id = ?1",
                [archive_doc_id],
                |row| row.get::<_, i64>(0)
            ).map_err(|e| format!("Failed to check archive document: {}", e))?;
            
            if archive_exists == 0 {
                // Create archive document
                let now = chrono::Utc::now().to_rfc3339();
                conn.execute(
                    "INSERT INTO documents (id, doc_type, title, file_path, created_by, created_at, modified_at, version, status)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![
                        archive_doc_id,
                        "RESULT",
                        "Archived Test Results",
                        "testing/archive.ydoc",
                        "system",
                        now.clone(),
                        now,
                        "1.0.0",
                        "archived"
                    ],
                ).map_err(|e| format!("Failed to create archive document: {}", e))?;
            }
            
            // Add summary block to archive
            let archive_block_id = format!("archive-{}", doc_id);
            let now = chrono::Utc::now().to_rfc3339();
            conn.execute(
                "INSERT INTO blocks (id, doc_id, cell_index, cell_type, yantra_type, content, created_by, created_at, modified_by, modified_at, modifier_id, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    archive_block_id,
                    archive_doc_id,
                    0,
                    "markdown",
                    "testresult",
                    summary,
                    "system",
                    now.clone(),
                    "system",
                    now.clone(),
                    "archiver",
                    "archived"
                ],
            ).map_err(|e| format!("Failed to create archive block: {}", e))?;
            
            // Delete original document and its blocks
            conn.execute("DELETE FROM documents WHERE id = ?1", [&doc_id])
                .map_err(|e| format!("Failed to delete archived document: {}", e))?;
        }
        
        Ok(count)
    }

    /// Get archived test results summary
    pub fn get_archived_test_results(&self) -> Result<Vec<String>, String> {
        let conn = self.get_conn()?;
        
        let mut stmt = conn.prepare(
            "SELECT content FROM blocks WHERE doc_id = 'archive-test-results' ORDER BY created_at DESC"
        ).map_err(|e| format!("Failed to prepare archive query: {}", e))?;
        
        let summaries = stmt.query_map([], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|e| format!("Failed to query archived results: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect archived results: {}", e))?;
        
        Ok(summaries)
    }

    /// Clean up archive by removing entries older than specified days
    pub fn cleanup_archive(&self, days_to_keep: i64) -> Result<usize, String> {
        let conn = self.get_conn()?;
        
        let cutoff_date = chrono::Utc::now() - chrono::Duration::days(days_to_keep);
        let cutoff_str = cutoff_date.to_rfc3339();
        
        let deleted = conn.execute(
            "DELETE FROM blocks WHERE doc_id = 'archive-test-results' AND created_at < ?1",
            [&cutoff_str],
        ).map_err(|e| format!("Failed to cleanup archive: {}", e))?;
        
        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use chrono::Utc;

    #[test]
    fn test_create_database() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_ydoc.db");
        
        let db = YDocDatabase::new(&db_path);
        assert!(db.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_insert_and_get_document() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_ydoc_insert.db");
        
        let db = YDocDatabase::new(&db_path).unwrap();

        let doc = DocumentMetadata {
            id: "doc-123".to_string(),
            doc_type: DocumentType::Requirements,
            title: "Test Requirements".to_string(),
            file_path: "/ydocs/requirements/test.ydoc".to_string(),
            created_by: "user".to_string(),
            created_at: Utc::now().to_rfc3339(),
            modified_at: Utc::now().to_rfc3339(),
            version: "1.0.0".to_string(),
            status: "draft".to_string(),
        };

        assert!(db.insert_document(&doc).is_ok());

        let retrieved = db.get_document("doc-123").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().title, "Test Requirements");

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
