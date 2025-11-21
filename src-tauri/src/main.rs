// File: src-tauri/src/main.rs
// Purpose: Tauri application entry point with file system operations and GNN
// Dependencies: tauri, serde_json, std::fs
// Last Updated: November 20, 2025

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

mod gnn;
mod llm;
mod testing;

#[derive(Debug, Serialize, Deserialize)]
struct FileEntry {
    name: String,
    path: String,
    is_directory: bool,
    size: Option<u64>,
}

#[derive(Debug, Serialize)]
struct FileOperationResult {
    success: bool,
    message: String,
}

// File system commands

/// Read file contents as string
#[tauri::command]
fn read_file(path: String) -> Result<String, String> {
    fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file: {}", e))
}

/// Write content to file
#[tauri::command]
fn write_file(path: String, content: String) -> Result<FileOperationResult, String> {
    fs::write(&path, content)
        .map(|_| FileOperationResult {
            success: true,
            message: format!("File saved successfully: {}", path),
        })
        .map_err(|e| format!("Failed to write file: {}", e))
}

/// List directory contents
#[tauri::command]
fn read_dir(path: String) -> Result<Vec<FileEntry>, String> {
    let entries = fs::read_dir(&path)
        .map_err(|e| format!("Failed to read directory: {}", e))?;
    
    let mut file_entries = Vec::new();
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let metadata = entry.metadata()
            .map_err(|e| format!("Failed to read metadata: {}", e))?;
        
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path().to_string_lossy().to_string();
        let is_directory = metadata.is_dir();
        let size = if is_directory { None } else { Some(metadata.len()) };
        
        file_entries.push(FileEntry {
            name,
            path,
            is_directory,
            size,
        });
    }
    
    // Sort: directories first, then alphabetically
    file_entries.sort_by(|a, b| {
        match (a.is_directory, b.is_directory) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });
    
    Ok(file_entries)
}

/// Check if path exists
#[tauri::command]
fn path_exists(path: String) -> bool {
    PathBuf::from(path).exists()
}

/// Get file metadata
#[tauri::command]
fn get_file_info(path: String) -> Result<FileEntry, String> {
    let path_buf = PathBuf::from(&path);
    let metadata = fs::metadata(&path_buf)
        .map_err(|e| format!("Failed to get file info: {}", e))?;
    
    let name = path_buf
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    
    Ok(FileEntry {
        name,
        path: path.clone(),
        is_directory: metadata.is_dir(),
        size: if metadata.is_dir() { None } else { Some(metadata.len()) },
    })
}

// GNN commands

/// Analyze a project and build the dependency graph
#[tauri::command]
fn analyze_project(project_path: String) -> Result<String, String> {
    let db_path = Path::new(&project_path).join(".yantra").join("graph.db");
    
    // Create .yantra directory if it doesn't exist
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create .yantra directory: {}", e))?;
    }
    
    let mut engine = gnn::GNNEngine::new(&db_path)?;
    engine.build_graph(Path::new(&project_path))?;
    
    Ok(format!("Successfully analyzed project at {}", project_path))
}

/// Get dependencies of a node
#[tauri::command]
fn get_dependencies(project_path: String, node_id: String) -> Result<Vec<gnn::CodeNode>, String> {
    let db_path = Path::new(&project_path).join(".yantra").join("graph.db");
    
    let mut engine = gnn::GNNEngine::new(&db_path)?;
    engine.load()?;
    
    Ok(engine.get_dependencies(&node_id))
}

/// Get dependents of a node (reverse dependencies)
#[tauri::command]
fn get_dependents(project_path: String, node_id: String) -> Result<Vec<gnn::CodeNode>, String> {
    let db_path = Path::new(&project_path).join(".yantra").join("graph.db");
    
    let mut engine = gnn::GNNEngine::new(&db_path)?;
    engine.load()?;
    
    Ok(engine.get_dependents(&node_id))
}

/// Find a node by name and optional file path
#[tauri::command]
fn find_node(project_path: String, name: String, file_path: Option<String>) -> Result<Option<gnn::CodeNode>, String> {
    let db_path = Path::new(&project_path).join(".yantra").join("graph.db");
    
    let mut engine = gnn::GNNEngine::new(&db_path)?;
    engine.load()?;
    
    let file_path_str = file_path.as_deref();
    Ok(engine.find_node(&name, file_path_str).cloned())
}

// LLM Configuration commands

/// Get LLM configuration (sanitized, no API keys)
#[tauri::command]
fn get_llm_config(app_handle: tauri::AppHandle) -> Result<llm::config::SanitizedConfig, String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let manager = llm::config::LLMConfigManager::new(&config_dir)?;
    Ok(manager.get_sanitized_config())
}

/// Set primary LLM provider
#[tauri::command]
fn set_llm_provider(app_handle: tauri::AppHandle, provider: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    
    let provider_enum = match provider.to_lowercase().as_str() {
        "claude" => llm::LLMProvider::Claude,
        "openai" => llm::LLMProvider::OpenAI,
        _ => return Err(format!("Invalid provider: {}. Use 'claude' or 'openai'", provider)),
    };
    
    manager.set_primary_provider(provider_enum)
}

/// Set Claude API key
#[tauri::command]
fn set_claude_key(app_handle: tauri::AppHandle, api_key: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_claude_key(api_key)
}

/// Set OpenAI API key
#[tauri::command]
fn set_openai_key(app_handle: tauri::AppHandle, api_key: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_openai_key(api_key)
}

/// Clear API key for a provider
#[tauri::command]
fn clear_llm_key(app_handle: tauri::AppHandle, provider: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    
    let provider_enum = match provider.to_lowercase().as_str() {
        "claude" => llm::LLMProvider::Claude,
        "openai" => llm::LLMProvider::OpenAI,
        _ => return Err(format!("Invalid provider: {}", provider)),
    };
    
    manager.clear_api_key(provider_enum)
}

/// Update retry configuration
// LLM commands (Configuration already implemented above)

/// Generate code using LLM with GNN context
#[tauri::command]
async fn generate_code(
    app_handle: tauri::AppHandle,
    intent: String,
    file_path: Option<String>,
    target_node: Option<String>,
) -> Result<llm::CodeGenerationResponse, String> {
    use llm::context::assemble_context;
    use llm::CodeGenerationRequest;
    
    // Get configuration
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let manager = llm::config::LLMConfigManager::new(&config_dir)?;
    let config = manager.get_sanitized_config();
    
    // Verify at least one API key is configured
    if !config.has_claude_key && !config.has_openai_key {
        return Err("No LLM API keys configured. Please configure at least one provider in settings.".to_string());
    }
    
    // Initialize GNN engine for context
    let data_dir = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get data directory".to_string())?;
    
    let db_path = data_dir.join("gnn.db");
    let engine = gnn::GNNEngine::new(&db_path)?;
    
    // Assemble context from GNN
    let context = assemble_context(
        &engine,
        target_node.as_deref(),
        file_path.as_deref(),
    )?;
    
    // Create code generation request
    let request = CodeGenerationRequest {
        intent: intent.clone(),
        file_path: file_path.clone(),
        context,
        dependencies: Vec::new(), // TODO: Extract from GNN
    };
    
    // Get full config for orchestrator
    let llm_config = manager.get_config().clone();
    
    // Create orchestrator and generate code
    let orchestrator = llm::orchestrator::LLMOrchestrator::new(llm_config);
    orchestrator.generate_code(&request).await
        .map_err(|e| e.message)
}

/// Generate pytest tests for given code
#[tauri::command]
async fn generate_tests(
    app_handle: tauri::AppHandle,
    code: String,
    language: String,
    file_path: String,
    coverage_target: Option<f32>,
) -> Result<testing::TestGenerationResponse, String> {
    // Get configuration
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let manager = llm::config::LLMConfigManager::new(&config_dir)?;
    let config = manager.get_sanitized_config();
    
    // Verify at least one API key is configured
    if !config.has_claude_key && !config.has_openai_key {
        return Err("No LLM API keys configured. Please configure at least one provider in settings.".to_string());
    }
    
    // Create test generation request
    let request = testing::TestGenerationRequest {
        code,
        language,
        file_path,
        coverage_target: coverage_target.unwrap_or(0.9), // Default to 90%
    };
    
    // Get full config for test generator
    let llm_config = manager.get_config().clone();
    
    // Generate tests
    testing::generator::generate_tests(request, llm_config).await
}

#[tauri::command]
fn set_llm_retry_config(
    app_handle: tauri::AppHandle,
    max_retries: u32,
    timeout_seconds: u64,
) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_retry_config(max_retries, timeout_seconds)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            read_file,
            write_file,
            read_dir,
            path_exists,
            get_file_info,
            analyze_project,
            get_dependencies,
            get_dependents,
            find_node,
            get_llm_config,
            set_llm_provider,
            set_claude_key,
            set_openai_key,
            clear_llm_key,
            set_llm_retry_config,
            generate_code,
            generate_tests
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
