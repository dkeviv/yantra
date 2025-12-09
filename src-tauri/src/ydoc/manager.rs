// YDoc Manager Module
// Last updated: December 8, 2025
//
// This module coordinates all YDoc operations, managing the lifecycle
// of documents and ensuring consistency between database, files, and memory.

use std::path::{Path, PathBuf};
use std::error::Error;
use std::fmt;
use rusqlite::{Connection, Transaction};
use crate::ydoc::{
    YDocDatabase, YDocFileOps, TraceabilityQuery,
    DocumentType, BlockType,
    parse_ydoc_file, write_ydoc_file, serialize_ydoc,
    YDocFile, YDocCell, YantraMetadata, GraphEdge,
};

#[derive(Debug)]
pub enum ManagerError {
    DatabaseError(String),
    FileError(String),
    ParseError(String),
    NotFound(String),
    ValidationError(String),
    TransactionError(String),
}

impl fmt::Display for ManagerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ManagerError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            ManagerError::FileError(msg) => write!(f, "File error: {}", msg),
            ManagerError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ManagerError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ManagerError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            ManagerError::TransactionError(msg) => write!(f, "Transaction error: {}", msg),
        }
    }
}

impl Error for ManagerError {}

type Result<T> = std::result::Result<T, ManagerError>;

/// YDoc Manager - coordinates database, parser, and file operations
pub struct YDocManager {
    db: YDocDatabase,
    project_root: PathBuf,
    ydocs_root: PathBuf,
}

impl YDocManager {
    /// Create a new YDoc manager
    pub fn new(db_path: String, project_root: PathBuf) -> Result<Self> {
        let db = YDocDatabase::new(Path::new(&db_path))
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Initialize ydocs folder structure
        let ydocs_root = YDocFileOps::initialize_ydoc_folder(&project_root)
            .map_err(|e| ManagerError::FileError(e.to_string()))?;
        
        Ok(Self {
            db,
            project_root,
            ydocs_root,
        })
    }

    /// Create a new document (both in DB and on disk)
    pub fn create_document(
        &self,
        doc_id: String,
        doc_type: DocumentType,
        title: String,
        version: String,
        created_by: String,
    ) -> Result<YDocFile> {
        // Create YDocFile
        let mut doc = YDocFile::new(doc_id.clone(), doc_type.clone(), title.clone(), created_by.clone());
        
        // Validate
        doc.validate().map_err(|e| ManagerError::ValidationError(e.to_string()))?;
        
        // Insert into database
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Get file path before inserting
        let file_path = self.get_document_path(&doc_id, &doc_type);
        
        conn.execute(
            "INSERT INTO documents (id, doc_type, title, file_path, created_by, created_at, modified_at, version, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                &doc_id,
                &format!("{:?}", doc_type),
                &title,
                &file_path.to_str().unwrap_or(""),
                &created_by,
                &doc.metadata.created_at,
                &doc.metadata.modified_at,
                &version,
                "draft", // status
            ],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Write to disk
        write_ydoc_file(&doc, &file_path)
            .map_err(|e| ManagerError::FileError(e.to_string()))?;
        
        Ok(doc)
    }

    /// Create a new block and add to document
    pub fn create_block(
        &self,
        doc_id: String,
        block_id: String,
        yantra_type: BlockType,
        content: String,
        created_by: String,
    ) -> Result<()> {
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Verify document exists
        let doc_exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE id = ?)",
            [&doc_id],
            |row| row.get(0),
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        if !doc_exists {
            return Err(ManagerError::NotFound(format!("Document '{}' not found", doc_id)));
        }
        
        // Insert block
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO blocks (id, doc_id, cell_index, cell_type, yantra_type, content, created_by, created_at, modified_by, modified_at, modifier_id, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                &block_id,
                &doc_id,
                0, // cell_index - will be updated when syncing from disk
                "markdown", // cell_type - default to markdown
                &format!("{:?}", yantra_type),
                &content,
                "user", // created_by
                &now,
                "user", // modified_by
                &now,
                "user-1", // modifier_id
                "draft", // status
            ],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Update document on disk
        self.sync_document_to_disk(&doc_id)?;
        
        Ok(())
    }

    /// Create a graph edge (traceability link)
    pub fn create_edge(
        &self,
        source_block_id: String,
        target_block_id: String,
        edge_type: String,
        metadata: Option<String>,
    ) -> Result<()> {
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Verify both blocks exist
        for block_id in &[&source_block_id, &target_block_id] {
            let exists: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM blocks WHERE id = ?)",
                [block_id],
                |row| row.get(0),
            ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
            
            if !exists {
                return Err(ManagerError::NotFound(format!("Block '{}' not found", block_id)));
            }
        }
        
        // Insert edge
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO graph_edges (source_id, source_type, target_id, target_type, edge_type, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                &source_block_id,
                "doc_block", // source_type
                &target_block_id,
                "doc_block", // target_type
                &edge_type,
                &now,
                &metadata,
            ],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Update document files (both source and target documents)
        let source_doc_id: String = conn.query_row(
            "SELECT doc_id FROM blocks WHERE id = ?",
            [&source_block_id],
            |row| row.get(0),
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        let target_doc_id: String = conn.query_row(
            "SELECT doc_id FROM blocks WHERE id = ?",
            [&target_block_id],
            |row| row.get(0),
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        self.sync_document_to_disk(&source_doc_id)?;
        if source_doc_id != target_doc_id {
            self.sync_document_to_disk(&target_doc_id)?;
        }
        
        Ok(())
    }

    /// Load a document from disk and sync to database
    pub fn load_document(&self, doc_id: &str) -> Result<YDocFile> {
        // Try to find the document file
        let doc_path = self.find_document_file(doc_id)?;
        
        // Parse the file
        let doc = parse_ydoc_file(&doc_path)
            .map_err(|e| ManagerError::ParseError(e.to_string()))?;
        
        // Sync to database
        self.sync_document_to_db(&doc)?;
        
        Ok(doc)
    }

    /// Save a document (update both DB and disk)
    pub fn save_document(&self, doc: &YDocFile) -> Result<()> {
        // Validate
        doc.validate().map_err(|e| ManagerError::ValidationError(e.to_string()))?;
        
        // Use transaction for atomic updates
        let mut conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        let tx = conn.transaction()
            .map_err(|e| ManagerError::TransactionError(e.to_string()))?;
        
        // Compute file path for the document
        let doc_type = self.parse_document_type(&doc.metadata.yantra_doc_type)?;
        let file_path = self.get_document_path(&doc.metadata.yantra_doc_id, &doc_type);
        
        // Update document metadata
        tx.execute(
            "INSERT OR REPLACE INTO documents (id, doc_type, title, file_path, created_by, created_at, modified_at, version, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                &doc.metadata.yantra_doc_id,
                &doc.metadata.yantra_doc_type,
                &doc.metadata.yantra_title,
                &file_path.to_str().unwrap_or(""),
                &doc.metadata.created_by,
                &doc.metadata.created_at,
                &doc.metadata.modified_at,
                &doc.metadata.yantra_version,
                "draft", // status
            ],
        ).map_err(|e| ManagerError::TransactionError(e.to_string()))?;
        
        // Delete old blocks for this document
        tx.execute(
            "DELETE FROM blocks WHERE doc_id = ?",
            [&doc.metadata.yantra_doc_id],
        ).map_err(|e| ManagerError::TransactionError(e.to_string()))?;
        
        // Insert all blocks
        for (idx, cell) in doc.cells.iter().enumerate() {
            if let Some(yantra) = &cell.metadata.yantra {
                tx.execute(
                    "INSERT INTO blocks (id, doc_id, cell_index, cell_type, yantra_type, content, created_by, created_at, modified_by, modified_at, modifier_id, status)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    rusqlite::params![
                        &yantra.yantra_id,
                        &doc.metadata.yantra_doc_id,
                        idx, // cell_index
                        &cell.cell_type, // cell_type from YDocCell
                        &yantra.yantra_type,
                        &cell.get_content(),
                        &yantra.created_by,
                        &yantra.created_at,
                        &yantra.modified_by,
                        &yantra.modified_at,
                        &yantra.modifier_id,
                        &yantra.status,
                    ],
                ).map_err(|e| ManagerError::TransactionError(e.to_string()))?;
                
                // Insert edges for this block
                for edge in &yantra.graph_edges {
                    let now = chrono::Utc::now().to_rfc3339();
                    tx.execute(
                        "INSERT OR IGNORE INTO graph_edges (source_id, source_type, target_id, target_type, edge_type, created_at, metadata)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        rusqlite::params![
                            &yantra.yantra_id,
                            "doc_block", // source_type
                            &edge.target_id,
                            &edge.target_type,
                            &edge.edge_type,
                            &now,
                            &edge.metadata.as_ref().map(|v| v.to_string()),
                        ],
                    ).map_err(|e| ManagerError::TransactionError(e.to_string()))?;
                }
            }
        }
        
        tx.commit().map_err(|e| ManagerError::TransactionError(e.to_string()))?;
        
        // Write to disk
        let doc_type = self.parse_document_type(&doc.metadata.yantra_doc_type)?;
        let file_path = self.get_document_path(&doc.metadata.yantra_doc_id, &doc_type);
        write_ydoc_file(doc, &file_path)
            .map_err(|e| ManagerError::FileError(e.to_string()))?;
        
        Ok(())
    }

    /// Delete a document (from both DB and disk)
    pub fn delete_document(&self, doc_id: &str, delete_file: bool) -> Result<()> {
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Get document type for file path
        let doc_type_str: String = conn.query_row(
            "SELECT doc_type FROM documents WHERE id = ?",
            [doc_id],
            |row| row.get(0),
        ).map_err(|_| ManagerError::NotFound(format!("Document '{}' not found", doc_id)))?;
        
        let doc_type = self.parse_document_type(&doc_type_str)?;
        
        // Delete from database (cascade will handle blocks and edges due to foreign keys)
        conn.execute(
            "DELETE FROM graph_edges WHERE source_id IN (SELECT id FROM blocks WHERE doc_id = ?)",
            [doc_id],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        conn.execute(
            "DELETE FROM graph_edges WHERE target_id IN (SELECT id FROM blocks WHERE doc_id = ?)",
            [doc_id],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        conn.execute(
            "DELETE FROM blocks WHERE doc_id = ?",
            [doc_id],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        conn.execute(
            "DELETE FROM documents WHERE id = ?",
            [doc_id],
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Delete file if requested
        if delete_file {
            let file_path = self.get_document_path(doc_id, &doc_type);
            if file_path.exists() {
                std::fs::remove_file(&file_path)
                    .map_err(|e| ManagerError::FileError(e.to_string()))?;
            }
        }
        
        Ok(())
    }

    /// List all documents in the database
    pub fn list_documents(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        let mut stmt = conn.prepare(
            "SELECT id, doc_type, title FROM documents ORDER BY modified_at DESC"
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        let docs = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        }).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        let mut result = Vec::new();
        for doc in docs {
            result.push(doc.map_err(|e| ManagerError::DatabaseError(e.to_string()))?);
        }
        
        Ok(result)
    }

    /// Get document metadata by ID
    pub fn get_document_metadata(&self, doc_id: &str) -> Result<(String, String, String, String)> {
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        conn.query_row(
            "SELECT doc_type, title, version, modified_at FROM documents WHERE id = ?",
            [doc_id],
            |row| Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
            )),
        ).map_err(|_| ManagerError::NotFound(format!("Document '{}' not found", doc_id)))
    }

    /// Search blocks by content
    pub fn search_blocks(&self, query: &str) -> Result<Vec<(String, String, String, String)>> {
        let results = self.db.search_blocks(query)
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Convert BlockMetadata to tuple format (id, yantra_type, doc_id, content)
        Ok(results.iter().map(|block| {
            (
                block.id.clone(),
                format!("{:?}", block.yantra_type),
                block.doc_id.clone(),
                block.content.clone(),
            )
        }).collect())
    }

    /// Get traceability chain for a block
    pub fn get_traceability_chain(&self, block_id: &str) -> Result<std::collections::HashMap<String, Vec<crate::ydoc::TraceabilityEntity>>> {
        let query = TraceabilityQuery::new(self.db.db_path.clone());
        query.get_traceability_chain(block_id)
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))
    }

    /// Get traceability coverage statistics
    pub fn get_coverage_stats(&self) -> Result<std::collections::HashMap<String, i64>> {
        let query = TraceabilityQuery::new(self.db.db_path.clone());
        query.get_coverage_stats()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))
    }

    /// Export document to Markdown
    pub fn export_to_markdown(&self, doc_id: &str) -> Result<String> {
        let doc = self.load_document(doc_id)?;
        YDocFileOps::export_to_markdown(&doc)
            .map_err(|e| ManagerError::FileError(e.to_string()))
    }

    /// Export document to HTML
    pub fn export_to_html(&self, doc_id: &str) -> Result<String> {
        let doc = self.load_document(doc_id)?;
        YDocFileOps::export_to_html(&doc)
            .map_err(|e| ManagerError::FileError(e.to_string()))
    }

    /// Import document from Markdown file
    pub fn import_from_markdown(&self, path: &Path, doc_type: DocumentType, doc_id: String) -> Result<YDocFile> {
        let doc = YDocFileOps::import_from_markdown(path, doc_type, doc_id)
            .map_err(|e| ManagerError::ParseError(e.to_string()))?;
        
        // Save to database and disk
        self.save_document(&doc)?;
        
        Ok(doc)
    }

    /// Archive old test results (>30 days by default)
    pub fn archive_old_test_results(&self, days_threshold: i64) -> Result<usize> {
        self.db.archive_old_test_results(days_threshold)
            .map_err(|e| ManagerError::DatabaseError(e))
    }

    /// Get archived test results summaries
    pub fn get_archived_test_results(&self) -> Result<Vec<String>> {
        self.db.get_archived_test_results()
            .map_err(|e| ManagerError::DatabaseError(e))
    }

    /// Clean up old archive entries
    pub fn cleanup_archive(&self, days_to_keep: i64) -> Result<usize> {
        self.db.cleanup_archive(days_to_keep)
            .map_err(|e| ManagerError::DatabaseError(e))
    }

    // --- Private helper methods ---

    /// Sync document from database to disk
    fn sync_document_to_disk(&self, doc_id: &str) -> Result<()> {
        let conn = self.db.get_connection()
            .map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        // Get document metadata
        let (doc_type_str, title, version, modified_at): (String, String, String, String) = conn.query_row(
            "SELECT doc_type, title, version, modified_at FROM documents WHERE id = ?",
            [doc_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        ).map_err(|_| ManagerError::NotFound(format!("Document '{}' not found", doc_id)))?;
        
        let doc_type = self.parse_document_type(&doc_type_str)?;
        
        // Reconstruct YDocFile from database
        let mut doc = YDocFile::new(doc_id.to_string(), doc_type.clone(), title, "system".to_string());
        doc.metadata.yantra_version = version;
        doc.metadata.modified_at = modified_at;
        
        // Get blocks
        let mut stmt = conn.prepare(
            "SELECT id, yantra_type, content, created_at, modified_at FROM blocks WHERE doc_id = ? ORDER BY rowid"
        ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        let blocks = stmt.query_map([doc_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        }).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
        
        for block in blocks {
            let (block_id, yantra_type, content, created_at, modified_at) = 
                block.map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
            
            // Get edges for this block
            let mut edge_stmt = conn.prepare(
                "SELECT target_block_id, edge_type, metadata FROM graph_edges WHERE source_block_id = ?"
            ).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
            
            let edges = edge_stmt.query_map([&block_id], |row| {
                Ok(GraphEdge {
                    target_id: row.get(0)?,
                    target_type: "unknown".to_string(), // Will be filled by frontend
                    edge_type: row.get(1)?,
                    metadata: row.get::<_, Option<String>>(2)?.map(|s| {
                        serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                    }),
                })
            }).map_err(|e| ManagerError::DatabaseError(e.to_string()))?;
            
            let mut edge_vec = Vec::new();
            for edge in edges {
                edge_vec.push(edge.map_err(|e| ManagerError::DatabaseError(e.to_string()))?);
            }
            
            // Parse block type
            let block_type_enum = self.parse_block_type(&yantra_type)?;
            
            // Create cell
            let mut cell = YDocCell::new_markdown(
                content,
                block_id.clone(),
                block_type_enum,
                "system".to_string(),
            );
            
            if let Some(yantra) = &mut cell.metadata.yantra {
                yantra.created_at = created_at;
                yantra.modified_at = modified_at;
                yantra.graph_edges = edge_vec;
            }
            
            doc.add_cell(cell);
        }
        
        // Write to disk
        let file_path = self.get_document_path(doc_id, &doc_type);
        write_ydoc_file(&doc, &file_path)
            .map_err(|e| ManagerError::FileError(e.to_string()))?;
        
        Ok(())
    }

    /// Sync document from disk to database
    fn sync_document_to_db(&self, doc: &YDocFile) -> Result<()> {
        // Use save_document which handles the full sync
        self.save_document(doc)
    }

    /// Get the file path for a document
    fn get_document_path(&self, doc_id: &str, doc_type: &DocumentType) -> PathBuf {
        let folder = YDocFileOps::get_folder_for_type(&self.ydocs_root, doc_type);
        folder.join(format!("{}.ydoc", doc_id))
    }

    /// Find a document file by searching all folders
    fn find_document_file(&self, doc_id: &str) -> Result<PathBuf> {
        let filename = format!("{}.ydoc", doc_id);
        
        // Search in all subfolders
        let subfolders = ["requirements", "adrs", "architecture", "specifications", "plans", 
                          "technical", "api", "user", "testing", "results", "changes", "decisions"];
        
        for subfolder in &subfolders {
            let path = self.ydocs_root.join(subfolder).join(&filename);
            if path.exists() {
                return Ok(path);
            }
        }
        
        Err(ManagerError::NotFound(format!("Document file '{}' not found", filename)))
    }

    /// Parse DocumentType from string
    fn parse_document_type(&self, type_str: &str) -> Result<DocumentType> {
        match type_str {
            "Requirements" => Ok(DocumentType::Requirements),
            "ADR" => Ok(DocumentType::ADR),
            "Architecture" => Ok(DocumentType::Architecture),
            "TechSpec" => Ok(DocumentType::TechSpec),
            "ProjectPlan" => Ok(DocumentType::ProjectPlan),
            "TechGuide" => Ok(DocumentType::TechGuide),
            "APIGuide" => Ok(DocumentType::APIGuide),
            "UserGuide" => Ok(DocumentType::UserGuide),
            "TestingPlan" => Ok(DocumentType::TestingPlan),
            "TestResults" => Ok(DocumentType::TestResults),
            "ChangeLog" => Ok(DocumentType::ChangeLog),
            "DecisionsLog" => Ok(DocumentType::DecisionsLog),
            _ => Err(ManagerError::ValidationError(format!("Unknown document type: {}", type_str))),
        }
    }

    /// Parse BlockType from string
    fn parse_block_type(&self, type_str: &str) -> Result<BlockType> {
        match type_str {
            "Requirement" => Ok(BlockType::Requirement),
            "ADR" => Ok(BlockType::ADR),
            "Architecture" => Ok(BlockType::Architecture),
            "Specification" => Ok(BlockType::Specification),
            "Task" => Ok(BlockType::Task),
            "TechDoc" => Ok(BlockType::TechDoc),
            "APIDoc" => Ok(BlockType::APIDoc),
            "UserDoc" => Ok(BlockType::UserDoc),
            "TestPlan" => Ok(BlockType::TestPlan),
            "TestResult" => Ok(BlockType::TestResult),
            "Change" => Ok(BlockType::Change),
            "Decision" => Ok(BlockType::Decision),
            _ => Err(ManagerError::ValidationError(format!("Unknown block type: {}", type_str))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_test_manager() -> (TempDir, YDocManager) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db").to_str().unwrap().to_string();
        let project_root = temp_dir.path().to_path_buf();
        
        let manager = YDocManager::new(db_path, project_root).unwrap();
        (temp_dir, manager)
    }

    #[test]
    fn test_create_document() {
        let (_temp, manager) = setup_test_manager();
        
        let doc = manager.create_document(
            "test-doc-1".to_string(),
            DocumentType::Requirements,
            "Test Document".to_string(),
            "1.0.0".to_string(),
            "test-user".to_string(),
        ).unwrap();
        
        assert_eq!(doc.metadata.yantra_doc_id, "test-doc-1");
        assert_eq!(doc.metadata.yantra_title, "Test Document");
    }

    #[test]
    fn test_create_and_load_document() {
        let (_temp, manager) = setup_test_manager();
        
        manager.create_document(
            "test-doc-2".to_string(),
            DocumentType::ADR,
            "Architecture Decision".to_string(),
            "1.0.0".to_string(),
            "architect".to_string(),
        ).unwrap();
        
        let loaded = manager.load_document("test-doc-2").unwrap();
        assert_eq!(loaded.metadata.yantra_doc_id, "test-doc-2");
        assert_eq!(loaded.metadata.yantra_doc_type, "ADR");
    }

    #[test]
    fn test_list_documents() {
        let (_temp, manager) = setup_test_manager();
        
        manager.create_document(
            "doc-1".to_string(),
            DocumentType::Requirements,
            "Doc 1".to_string(),
            "1.0.0".to_string(),
            "user".to_string(),
        ).unwrap();
        
        manager.create_document(
            "doc-2".to_string(),
            DocumentType::ADR,
            "Doc 2".to_string(),
            "1.0.0".to_string(),
            "user".to_string(),
        ).unwrap();
        
        let docs = manager.list_documents().unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_delete_document() {
        let (_temp, manager) = setup_test_manager();
        
        manager.create_document(
            "doc-to-delete".to_string(),
            DocumentType::Requirements,
            "Delete Me".to_string(),
            "1.0.0".to_string(),
            "user".to_string(),
        ).unwrap();
        
        manager.delete_document("doc-to-delete", true).unwrap();
        
        let result = manager.load_document("doc-to-delete");
        assert!(result.is_err());
    }
}
