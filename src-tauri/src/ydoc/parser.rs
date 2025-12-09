// YDoc Parser Module
// Last updated: December 8, 2025
//
// This module handles parsing and serialization of .ydoc files (ipynb-compatible format).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::error::Error;
use std::fmt;
use crate::ydoc::{DocumentType, BlockType};

#[derive(Debug)]
pub enum YDocError {
    ParseError(String),
    IoError(std::io::Error),
    SerdeError(serde_json::Error),
    ValidationError(String),
}

impl fmt::Display for YDocError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            YDocError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            YDocError::IoError(err) => write!(f, "IO error: {}", err),
            YDocError::SerdeError(err) => write!(f, "JSON error: {}", err),
            YDocError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl Error for YDocError {}

impl From<std::io::Error> for YDocError {
    fn from(err: std::io::Error) -> Self {
        YDocError::IoError(err)
    }
}

impl From<serde_json::Error> for YDocError {
    fn from(err: serde_json::Error) -> Self {
        YDocError::SerdeError(err)
    }
}

type Result<T> = std::result::Result<T, YDocError>;

/// Yantra-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YantraMetadata {
    pub yantra_id: String,
    pub yantra_type: String,
    pub created_by: String,
    pub created_at: String,
    pub modified_by: String,
    pub modified_at: String,
    pub modifier_id: String,
    #[serde(default)]
    pub graph_edges: Vec<GraphEdge>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_status")]
    pub status: String,
}

fn default_status() -> String {
    "draft".to_string()
}

/// Graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub target_id: String,
    pub target_type: String,
    pub edge_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Cell metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yantra: Option<YantraMetadata>,
    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}

impl Default for CellMetadata {
    fn default() -> Self {
        Self {
            yantra: None,
            other: HashMap::new(),
        }
    }
}

/// Notebook cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YDocCell {
    pub cell_type: String,
    pub source: Vec<String>,
    #[serde(default)]
    pub metadata: CellMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_count: Option<u32>,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YDocDocumentMetadata {
    pub yantra_doc_id: String,
    pub yantra_doc_type: String,
    pub yantra_title: String,
    pub yantra_version: String,
    pub created_by: String,
    pub created_at: String,
    pub modified_at: String,
    #[serde(flatten)]
    pub other: HashMap<String, serde_json::Value>,
}

/// Complete YDoc file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YDocFile {
    pub nbformat: u32,
    pub nbformat_minor: u32,
    pub metadata: YDocDocumentMetadata,
    pub cells: Vec<YDocCell>,
}

impl YDocFile {
    pub fn new(doc_id: String, doc_type: DocumentType, title: String, created_by: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            nbformat: 4,
            nbformat_minor: 5,
            metadata: YDocDocumentMetadata {
                yantra_doc_id: doc_id,
                yantra_doc_type: doc_type.code().to_string(),
                yantra_title: title,
                yantra_version: "1.0.0".to_string(),
                created_by: created_by.clone(),
                created_at: now.clone(),
                modified_at: now,
                other: HashMap::new(),
            },
            cells: Vec::new(),
        }
    }

    pub fn add_cell(&mut self, cell: YDocCell) {
        self.cells.push(cell);
    }

    pub fn validate(&self) -> Result<()> {
        if DocumentType::from_code(&self.metadata.yantra_doc_type).is_none() {
            return Err(YDocError::ValidationError(
                format!("Invalid document type: {}", self.metadata.yantra_doc_type)
            ));
        }
        for (idx, cell) in self.cells.iter().enumerate() {
            if !["markdown", "code", "raw"].contains(&cell.cell_type.as_str()) {
                return Err(YDocError::ValidationError(
                    format!("Cell {}: Invalid cell_type '{}'", idx, cell.cell_type)
                ));
            }
        }
        Ok(())
    }
}

impl YDocCell {
    pub fn new_markdown(content: String, yantra_id: String, yantra_type: BlockType, created_by: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            cell_type: "markdown".to_string(),
            source: content.lines().map(|s| s.to_string()).collect(),
            metadata: CellMetadata {
                yantra: Some(YantraMetadata {
                    yantra_id,
                    yantra_type: yantra_type_to_string(&yantra_type),
                    created_by: created_by.clone(),
                    created_at: now.clone(),
                    modified_by: created_by.clone(),
                    modified_at: now.clone(),
                    modifier_id: created_by,
                    graph_edges: Vec::new(),
                    tags: Vec::new(),
                    status: "draft".to_string(),
                }),
                other: HashMap::new(),
            },
            outputs: None,
            execution_count: None,
        }
    }

    pub fn get_content(&self) -> String {
        self.source.join("\n")
    }

    pub fn add_edge(&mut self, edge: GraphEdge) {
        if let Some(yantra) = &mut self.metadata.yantra {
            yantra.graph_edges.push(edge);
        }
    }
}

fn yantra_type_to_string(block_type: &BlockType) -> String {
    match block_type {
        BlockType::Requirement => "requirement",
        BlockType::ADR => "adr",
        BlockType::Architecture => "architecture",
        BlockType::Specification => "specification",
        BlockType::Task => "task",
        BlockType::TechDoc => "techdoc",
        BlockType::APIDoc => "apidoc",
        BlockType::UserDoc => "userdoc",
        BlockType::TestPlan => "testplan",
        BlockType::TestResult => "testresult",
        BlockType::Change => "change",
        BlockType::Decision => "decision",
    }.to_string()
}

pub fn parse_ydoc_file(path: &Path) -> Result<YDocFile> {
    let content = fs::read_to_string(path)?;
    parse_ydoc_content(&content)
}

pub fn parse_ydoc_content(content: &str) -> Result<YDocFile> {
    let doc: YDocFile = serde_json::from_str(content)?;
    doc.validate()?;
    Ok(doc)
}

pub fn serialize_ydoc(doc: &YDocFile) -> Result<String> {
    doc.validate()?;
    let json = serde_json::to_string_pretty(doc)?;
    Ok(json)
}

pub fn write_ydoc_file(doc: &YDocFile, path: &Path) -> Result<()> {
    let json = serialize_ydoc(doc)?;
    fs::write(path, json)?;
    Ok(())
}

pub struct YDocParser;
impl YDocParser {
    pub fn parse_file(path: &Path) -> Result<YDocFile> {
        parse_ydoc_file(path)
    }
    pub fn parse_content(content: &str) -> Result<YDocFile> {
        parse_ydoc_content(content)
    }
    pub fn serialize(doc: &YDocFile) -> Result<String> {
        serialize_ydoc(doc)
    }
    pub fn write_file(doc: &YDocFile, path: &Path) -> Result<()> {
        write_ydoc_file(doc, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_minimal_ydoc() {
        let content = r#"{
            "cells": [],
            "metadata": {
                "yantra_doc_id": "test-doc-001",
                "yantra_doc_type": "SPEC",
                "yantra_title": "Test Document",
                "yantra_version": "1.0.0",
                "created_by": "tester",
                "created_at": "2025-12-08T10:00:00Z",
                "modified_at": "2025-12-08T10:00:00Z"
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }"#;
        
        let result = parse_ydoc_content(content);
        assert!(result.is_ok(), "Failed to parse minimal ydoc: {:?}", result.err());
        
        let doc = result.unwrap();
        assert_eq!(doc.cells.len(), 0);
        assert_eq!(doc.metadata.yantra_doc_id, "test-doc-001");
        assert_eq!(doc.metadata.yantra_doc_type, "SPEC");
    }

    #[test]
    fn test_create_new_ydoc() {
        let doc = YDocFile::new(
            "doc-001".to_string(),
            DocumentType::TechSpec,
            "Test Spec".to_string(),
            "test_user".to_string(),
        );
        
        assert_eq!(doc.metadata.yantra_doc_id, "doc-001");
        assert_eq!(doc.metadata.yantra_doc_type, "SPEC");
        assert_eq!(doc.metadata.yantra_title, "Test Spec");
        assert_eq!(doc.cells.len(), 0);
    }

    #[test]
    fn test_add_cell_to_doc() {
        let mut doc = YDocFile::new(
            "doc-002".to_string(),
            DocumentType::Requirements,
            "Requirements".to_string(),
            "tester".to_string(),
        );
        
        let cell = YDocCell {
            cell_type: "markdown".to_string(),
            source: vec!["# Test Requirement".to_string()],
            metadata: CellMetadata {
                yantra: Some(YantraMetadata {
                    yantra_id: "req-001".to_string(),
                    yantra_type: "requirement".to_string(),
                    created_at: "2025-12-08T10:00:00Z".to_string(),
                    modified_at: "2025-12-08T10:00:00Z".to_string(),
                    created_by: "tester".to_string(),
                    modified_by: "tester".to_string(),
                    modifier_id: "tester-id".to_string(),
                    status: "active".to_string(),
                    graph_edges: vec![],
                    tags: vec![],
                }),
                other: HashMap::new(),
            },
            outputs: None,
            execution_count: None,
        };
        
        doc.add_cell(cell);
        assert_eq!(doc.cells.len(), 1);
        assert_eq!(doc.cells[0].cell_type, "markdown");
    }

    #[test]
    fn test_serialize_and_deserialize() {
        let doc = YDocFile::new(
            "doc-003".to_string(),
            DocumentType::Architecture,
            "Architecture Doc".to_string(),
            "architect".to_string(),
        );
        
        let serialized = serialize_ydoc(&doc).unwrap();
        assert!(serialized.contains("doc-003"));
        assert!(serialized.contains("ARCH"));
        
        let deserialized = parse_ydoc_content(&serialized).unwrap();
        assert_eq!(deserialized.metadata.yantra_doc_id, "doc-003");
    }

    #[test]
    fn test_write_and_read_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.ydoc");
        
        let doc = YDocFile::new(
            "doc-004".to_string(),
            DocumentType::TestingPlan,
            "Test Plan".to_string(),
            "qa_engineer".to_string(),
        );
        
        write_ydoc_file(&doc, &file_path).unwrap();
        assert!(file_path.exists());
        
        let read_doc = parse_ydoc_file(&file_path).unwrap();
        assert_eq!(read_doc.metadata.yantra_doc_id, "doc-004");
        assert_eq!(read_doc.metadata.yantra_doc_type, "TEST");
    }

    #[test]
    fn test_invalid_json() {
        let invalid = "{ invalid json }";
        let result = parse_ydoc_content(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_cell_with_graph_edges() {
        let mut doc = YDocFile::new(
            "doc-005".to_string(),
            DocumentType::TechSpec,
            "Spec with edges".to_string(),
            "dev".to_string(),
        );
        
        let edge1 = GraphEdge {
            edge_type: "implements".to_string(),
            target_type: "requirement".to_string(),
            target_id: "req-001".to_string(),
            metadata: None,
        };
        
        let edge2 = GraphEdge {
            edge_type: "tests".to_string(),
            target_type: "code_file".to_string(),
            target_id: "code-001".to_string(),
            metadata: Some(serde_json::json!({"coverage": "95%"})),
        };
        
        let cell = YDocCell {
            cell_type: "code".to_string(),
            source: vec!["function test() {}".to_string()],
            metadata: CellMetadata {
                yantra: Some(YantraMetadata {
                    yantra_id: "block-001".to_string(),
                    yantra_type: "specification".to_string(),
                    created_at: "2025-12-08T10:00:00Z".to_string(),
                    modified_at: "2025-12-08T10:00:00Z".to_string(),
                    created_by: "dev".to_string(),
                    modified_by: "dev".to_string(),
                    modifier_id: "dev-id".to_string(),
                    status: "active".to_string(),
                    graph_edges: vec![edge1, edge2],
                    tags: vec!["api".to_string(), "backend".to_string()],
                }),
                other: HashMap::new(),
            },
            outputs: None,
            execution_count: None,
        };
        
        doc.add_cell(cell);
        
        let yantra = doc.cells[0].metadata.yantra.as_ref().unwrap();
        assert_eq!(yantra.graph_edges.len(), 2);
        assert_eq!(yantra.graph_edges[0].edge_type, "implements");
        assert_eq!(yantra.graph_edges[1].edge_type, "tests");
        assert!(yantra.graph_edges[1].metadata.is_some());
        assert_eq!(yantra.tags.len(), 2);
    }

    #[test]
    fn test_multiple_cells_with_content() {
        let mut doc = YDocFile::new(
            "doc-006".to_string(),
            DocumentType::TechGuide,
            "Multi-cell Doc".to_string(),
            "writer".to_string(),
        );
        
        // Add markdown cell
        let cell1 = YDocCell {
            cell_type: "markdown".to_string(),
            source: vec![
                "# Introduction".to_string(),
                "This is the intro.".to_string(),
            ],
            metadata: CellMetadata::default(),
            outputs: None,
            execution_count: None,
        };
        
        // Add code cell
        let cell2 = YDocCell {
            cell_type: "code".to_string(),
            source: vec!["console.log('test');".to_string()],
            metadata: CellMetadata::default(),
            outputs: Some(vec![]),
            execution_count: Some(1),
        };
        
        doc.add_cell(cell1);
        doc.add_cell(cell2);
        
        assert_eq!(doc.cells.len(), 2);
        assert_eq!(doc.cells[0].source.join("\n"), "# Introduction\nThis is the intro.");
        assert_eq!(doc.cells[1].execution_count, Some(1));
    }

    #[test]
    fn test_validation() {
        let doc = YDocFile::new(
            "doc-007".to_string(),
            DocumentType::ADR,
            "ADR Test".to_string(),
            "architect".to_string(),
        );
        
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn test_yantra_type_conversion() {
        assert_eq!(yantra_type_to_string(&BlockType::Requirement), "requirement");
        assert_eq!(yantra_type_to_string(&BlockType::ADR), "adr");
        assert_eq!(yantra_type_to_string(&BlockType::Architecture), "architecture");
        assert_eq!(yantra_type_to_string(&BlockType::TestPlan), "testplan");
        assert_eq!(yantra_type_to_string(&BlockType::Specification), "specification");
    }
}
