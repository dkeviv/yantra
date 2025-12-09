// YDoc Tauri Commands
// Last updated: December 8, 2025
//
// This module exposes YDoc functionality to the frontend via Tauri commands.

use std::path::PathBuf;
use std::sync::Mutex;
use tauri::State;
use serde::{Deserialize, Serialize};

use crate::ydoc::{
    YDocManager, DocumentType, BlockType,
    YDocFile, TraceabilityEntity,
};

/// Global YDoc manager state
pub struct YDocState {
    pub manager: Mutex<Option<YDocManager>>,
}

impl YDocState {
    pub fn new() -> Self {
        Self {
            manager: Mutex::new(None),
        }
    }
}

/// Request to create a new document
#[derive(Debug, Deserialize)]
pub struct CreateDocumentRequest {
    pub doc_id: String,
    pub doc_type: String,
    pub title: String,
    pub version: String,
    pub created_by: String,
}

/// Request to create a new block
#[derive(Debug, Deserialize)]
pub struct CreateBlockRequest {
    pub doc_id: String,
    pub block_id: String,
    pub yantra_type: String,
    pub content: String,
    pub created_by: String,
}

/// Request to create a traceability edge
#[derive(Debug, Deserialize)]
pub struct CreateEdgeRequest {
    pub source_block_id: String,
    pub target_block_id: String,
    pub edge_type: String,
    pub metadata: Option<String>,
}

/// Response containing document metadata
#[derive(Debug, Serialize)]
pub struct DocumentMetadataResponse {
    pub doc_type: String,
    pub title: String,
    pub version: String,
    pub modified_at: String,
}

/// Response containing traceability chain
#[derive(Debug, Serialize)]
pub struct TraceabilityChainResponse {
    pub entity: Vec<TraceabilityEntityData>,
    pub dependencies: Vec<TraceabilityEntityData>,
    pub dependents: Vec<TraceabilityEntityData>,
    pub tests: Vec<TraceabilityEntityData>,
}

/// Serializable traceability entity data
#[derive(Debug, Serialize)]
pub struct TraceabilityEntityData {
    pub id: String,
    pub entity_type: String,
    pub doc_id: String,
    pub content: String,
    pub depth: usize,
}

impl From<&TraceabilityEntity> for TraceabilityEntityData {
    fn from(entity: &TraceabilityEntity) -> Self {
        Self {
            id: entity.id.clone(),
            entity_type: entity.entity_type.clone(),
            doc_id: entity.doc_id.clone(),
            content: entity.content.clone(),
            depth: entity.depth,
        }
    }
}

/// Initialize YDoc system
#[tauri::command]
pub async fn ydoc_initialize(
    project_root: String,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let project_path = PathBuf::from(&project_root);
    let db_path = project_path.join(".yantra").join("ydoc.db");
    
    // Create .yantra directory if it doesn't exist
    let yantra_dir = project_path.join(".yantra");
    std::fs::create_dir_all(&yantra_dir)
        .map_err(|e| format!("Failed to create .yantra directory: {}", e))?;
    
    let manager = YDocManager::new(
        db_path.to_str().unwrap().to_string(),
        project_path,
    ).map_err(|e| format!("Failed to initialize YDoc manager: {}", e))?;
    
    let mut state_guard = state.manager.lock().unwrap();
    *state_guard = Some(manager);
    
    Ok("YDoc initialized successfully".to_string())
}

/// Create a new document
#[tauri::command]
pub async fn ydoc_create_document(
    request: CreateDocumentRequest,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let doc_type = parse_document_type(&request.doc_type)?;
    
    let doc = manager.create_document(
        request.doc_id.clone(),
        doc_type,
        request.title,
        request.version,
        request.created_by,
    ).map_err(|e| format!("Failed to create document: {}", e))?;
    
    Ok(format!("Document '{}' created successfully", doc.metadata.yantra_doc_id))
}

/// Create a new block
#[tauri::command]
pub async fn ydoc_create_block(
    request: CreateBlockRequest,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let block_type = parse_block_type(&request.yantra_type)?;
    
    manager.create_block(
        request.doc_id,
        request.block_id.clone(),
        block_type,
        request.content,
        request.created_by,
    ).map_err(|e| format!("Failed to create block: {}", e))?;
    
    Ok(format!("Block '{}' created successfully", request.block_id))
}

/// Create a traceability edge
#[tauri::command]
pub async fn ydoc_create_edge(
    request: CreateEdgeRequest,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.create_edge(
        request.source_block_id.clone(),
        request.target_block_id.clone(),
        request.edge_type,
        request.metadata,
    ).map_err(|e| format!("Failed to create edge: {}", e))?;
    
    Ok(format!("Edge from '{}' to '{}' created successfully", 
        request.source_block_id, request.target_block_id))
}

/// Load a document
#[tauri::command]
pub async fn ydoc_load_document(
    doc_id: String,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let doc = manager.load_document(&doc_id)
        .map_err(|e| format!("Failed to load document: {}", e))?;
    
    // Serialize the YDocFile to JSON
    serde_json::to_string_pretty(&doc)
        .map_err(|e| format!("Failed to serialize document: {}", e))
}

/// List all documents
#[tauri::command]
pub async fn ydoc_list_documents(
    state: State<'_, YDocState>,
) -> Result<Vec<(String, String, String)>, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.list_documents()
        .map_err(|e| format!("Failed to list documents: {}", e))
}

/// Get document metadata
#[tauri::command]
pub async fn ydoc_get_document_metadata(
    doc_id: String,
    state: State<'_, YDocState>,
) -> Result<DocumentMetadataResponse, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let (doc_type, title, version, modified_at) = manager.get_document_metadata(&doc_id)
        .map_err(|e| format!("Failed to get document metadata: {}", e))?;
    
    Ok(DocumentMetadataResponse {
        doc_type,
        title,
        version,
        modified_at,
    })
}

/// Search blocks by content
#[tauri::command]
pub async fn ydoc_search_blocks(
    query: String,
    state: State<'_, YDocState>,
) -> Result<Vec<(String, String, String, String)>, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.search_blocks(&query)
        .map_err(|e| format!("Failed to search blocks: {}", e))
}

/// Get traceability chain for a block
#[tauri::command]
pub async fn ydoc_get_traceability_chain(
    block_id: String,
    state: State<'_, YDocState>,
) -> Result<TraceabilityChainResponse, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let chain = manager.get_traceability_chain(&block_id)
        .map_err(|e| format!("Failed to get traceability chain: {}", e))?;
    
    let entity = chain.get("entity")
        .map(|v| v.iter().map(|e| e.into()).collect())
        .unwrap_or_default();
    
    let dependencies = chain.get("dependencies")
        .map(|v| v.iter().map(|e| e.into()).collect())
        .unwrap_or_default();
    
    let dependents = chain.get("dependents")
        .map(|v| v.iter().map(|e| e.into()).collect())
        .unwrap_or_default();
    
    let tests = chain.get("tests")
        .map(|v| v.iter().map(|e| e.into()).collect())
        .unwrap_or_default();
    
    Ok(TraceabilityChainResponse {
        entity,
        dependencies,
        dependents,
        tests,
    })
}

/// Get traceability coverage statistics
#[tauri::command]
pub async fn ydoc_get_coverage_stats(
    state: State<'_, YDocState>,
) -> Result<std::collections::HashMap<String, i64>, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.get_coverage_stats()
        .map_err(|e| format!("Failed to get coverage stats: {}", e))
}

/// Export document to Markdown
#[tauri::command]
pub async fn ydoc_export_to_markdown(
    doc_id: String,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.export_to_markdown(&doc_id)
        .map_err(|e| format!("Failed to export to Markdown: {}", e))
}

/// Export document to HTML
#[tauri::command]
pub async fn ydoc_export_to_html(
    doc_id: String,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.export_to_html(&doc_id)
        .map_err(|e| format!("Failed to export to HTML: {}", e))
}

/// Delete a document
#[tauri::command]
pub async fn ydoc_delete_document(
    doc_id: String,
    delete_file: bool,
    state: State<'_, YDocState>,
) -> Result<String, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.delete_document(&doc_id, delete_file)
        .map_err(|e| format!("Failed to delete document: {}", e))?;
    
    Ok(format!("Document '{}' deleted successfully", doc_id))
}

/// Archive old test results (>30 days by default)
#[tauri::command]
pub async fn ydoc_archive_old_test_results(
    days_threshold: Option<i64>,
    state: State<'_, YDocState>,
) -> Result<usize, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let threshold = days_threshold.unwrap_or(30);
    manager.archive_old_test_results(threshold)
        .map_err(|e| format!("Failed to archive test results: {}", e))
}

/// Get archived test results summaries
#[tauri::command]
pub async fn ydoc_get_archived_test_results(
    state: State<'_, YDocState>,
) -> Result<Vec<String>, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    manager.get_archived_test_results()
        .map_err(|e| format!("Failed to get archived test results: {}", e))
}

/// Clean up old archive entries
#[tauri::command]
pub async fn ydoc_cleanup_archive(
    days_to_keep: Option<i64>,
    state: State<'_, YDocState>,
) -> Result<usize, String> {
    let state_guard = state.manager.lock().unwrap();
    let manager = state_guard.as_ref()
        .ok_or_else(|| "YDoc not initialized".to_string())?;
    
    let days = days_to_keep.unwrap_or(365); // Keep 1 year by default
    manager.cleanup_archive(days)
        .map_err(|e| format!("Failed to cleanup archive: {}", e))
}

// Helper functions to parse types from strings

fn parse_document_type(type_str: &str) -> Result<DocumentType, String> {
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
        _ => Err(format!("Unknown document type: {}", type_str)),
    }
}

fn parse_block_type(type_str: &str) -> Result<BlockType, String> {
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
        _ => Err(format!("Unknown block type: {}", type_str)),
    }
}
