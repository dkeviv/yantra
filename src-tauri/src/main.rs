// File: src-tauri/src/main.rs
// Purpose: Tauri application entry point with file system operations and GNN
// Dependencies: tauri, serde_json, std::fs
// Last Updated: November 20, 2025

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use base64::Engine;
use serde::{Deserialize, Serialize};
use tauri::{CustomMenuItem, Menu, MenuItem, Submenu, State};
use tokio::sync::Mutex as TokioMutex;

mod gnn;
mod llm;
mod agent;
mod testing;
mod git;
mod documentation;
mod bridge;
mod terminal;
mod architecture;

use terminal::TerminalManager;
use architecture::commands as arch_commands;

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

/// Edit file with surgical precision
#[tauri::command]
fn edit_file(request: agent::file_editor::FileEditRequest) -> Result<agent::file_editor::FileEditResult, String> {
    let editor = agent::file_editor::FileEditor::new(false);
    editor.apply_edit(request)
}

/// Preview file edit without applying (dry run)
#[tauri::command]
fn preview_file_edit(request: agent::file_editor::FileEditRequest) -> Result<agent::file_editor::FileEditResult, String> {
    let editor = agent::file_editor::FileEditor::new(true);
    editor.apply_edit(request)
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
        "openrouter" => llm::LLMProvider::OpenRouter,
        "groq" => llm::LLMProvider::Groq,
        "gemini" => llm::LLMProvider::Gemini,
        _ => return Err(format!("Invalid provider: {}. Use 'claude', 'openai', 'openrouter', 'groq', or 'gemini'", provider)),
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

/// Set OpenRouter API key
#[tauri::command]
fn set_openrouter_key(app_handle: tauri::AppHandle, api_key: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_openrouter_key(api_key)
}

/// Set Groq API key
#[tauri::command]
fn set_groq_key(app_handle: tauri::AppHandle, api_key: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_groq_key(api_key)
}

/// Set Gemini API key
#[tauri::command]
fn set_gemini_key(app_handle: tauri::AppHandle, api_key: String) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_gemini_key(api_key)
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
        "openrouter" => llm::LLMProvider::OpenRouter,
        "groq" => llm::LLMProvider::Groq,
        "gemini" => llm::LLMProvider::Gemini,
        _ => return Err(format!("Invalid provider: {}. Use 'claude', 'openai', 'openrouter', 'groq', or 'gemini'", provider)),
    };
    
    manager.clear_api_key(provider_enum)
}

/// Get available models for a specific provider
#[tauri::command]
fn get_available_models(provider: String) -> Result<Vec<llm::models::ModelInfo>, String> {
    let provider_enum = match provider.to_lowercase().as_str() {
        "claude" => llm::LLMProvider::Claude,
        "openai" => llm::LLMProvider::OpenAI,
        "openrouter" => llm::LLMProvider::OpenRouter,
        "groq" => llm::LLMProvider::Groq,
        "gemini" => llm::LLMProvider::Gemini,
        "qwen" => llm::LLMProvider::Qwen,
        _ => return Err(format!("Invalid provider: {}", provider)),
    };
    
    Ok(llm::models::get_available_models(provider_enum))
}

/// Get default model for a specific provider
#[tauri::command]
fn get_default_model(provider: String) -> Result<String, String> {
    let provider_enum = match provider.to_lowercase().as_str() {
        "claude" => llm::LLMProvider::Claude,
        "openai" => llm::LLMProvider::OpenAI,
        "openrouter" => llm::LLMProvider::OpenRouter,
        "groq" => llm::LLMProvider::Groq,
        "gemini" => llm::LLMProvider::Gemini,
        "qwen" => llm::LLMProvider::Qwen,
        _ => return Err(format!("Invalid provider: {}", provider)),
    };
    
    Ok(llm::models::get_default_model(provider_enum))
}

/// Set selected models for the user
#[tauri::command]
fn set_selected_models(app_handle: tauri::AppHandle, model_ids: Vec<String>) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    manager.set_selected_models(model_ids)
}

/// Get selected models
#[tauri::command]
fn get_selected_models(app_handle: tauri::AppHandle) -> Result<Vec<String>, String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let manager = llm::config::LLMConfigManager::new(&config_dir)?;
    Ok(manager.get_selected_models())
}

/// Update retry configuration
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

// Documentation commands

/// Get all features
#[tauri::command]
fn get_features(workspace_path: String) -> Result<Vec<documentation::Feature>, String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    Ok(manager.get_features().to_vec())
}

/// Get all decisions
#[tauri::command]
fn get_decisions(workspace_path: String) -> Result<Vec<documentation::Decision>, String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    Ok(manager.get_decisions().to_vec())
}

/// Get all changes
#[tauri::command]
fn get_changes(workspace_path: String) -> Result<Vec<documentation::Change>, String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    Ok(manager.get_changes().to_vec())
}

/// Get all tasks from plan
#[tauri::command]
fn get_tasks(workspace_path: String) -> Result<Vec<documentation::Task>, String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    Ok(manager.get_tasks().to_vec())
}

/// Add a new feature from chat
#[tauri::command]
fn add_feature(
    workspace_path: String,
    title: String,
    description: String,
    extracted_from: String,
) -> Result<(), String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    manager.add_feature(title, description, extracted_from);
    Ok(())
}

// Database commands

/// Connect to a database
#[tauri::command]
async fn db_connect(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
    config: agent::database::ConnectionConfig,
) -> Result<(), String> {
    let manager = db_manager.lock().await;
    manager.connect(config).await
}

/// Execute a SELECT query
#[tauri::command]
async fn db_query(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
    db_name: String,
    query: String,
) -> Result<agent::database::QueryResult, String> {
    let manager = db_manager.lock().await;
    manager.query(&db_name, &query).await
}

/// Execute INSERT/UPDATE/DELETE
#[tauri::command]
async fn db_execute(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
    db_name: String,
    query: String,
) -> Result<agent::database::QueryResult, String> {
    let manager = db_manager.lock().await;
    manager.execute(&db_name, &query).await
}

/// Get database schema
#[tauri::command]
async fn db_schema(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
    db_name: String,
) -> Result<agent::database::SchemaInfo, String> {
    let manager = db_manager.lock().await;
    manager.get_schema(&db_name).await
}

/// List all database connections
#[tauri::command]
async fn db_list_connections(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
) -> Result<Vec<String>, String> {
    let manager = db_manager.lock().await;
    Ok(manager.list_connections().await)
}

/// Disconnect from a database
#[tauri::command]
async fn db_disconnect(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
    db_name: String,
) -> Result<(), String> {
    let manager = db_manager.lock().await;
    manager.disconnect(&db_name).await
}

/// Test database connection
#[tauri::command]
async fn db_test_connection(
    db_manager: tauri::State<'_, Arc<tokio::sync::Mutex<agent::database::DatabaseManager>>>,
    config: agent::database::ConnectionConfig,
) -> Result<bool, String> {
    let manager = db_manager.lock().await;
    manager.test_connection(&config).await
}

// Browser automation commands

/// Launch browser session
#[tauri::command]
async fn browser_launch(url: String, headless: bool) -> Result<String, String> {
    let mut session = browser::BrowserSession::new(url, headless);
    session.launch().await?;
    // Return session ID (for now, just a simple identifier)
    Ok("browser_session_1".to_string())
}

/// Navigate to URL
#[tauri::command]
async fn browser_navigate(url: String) -> Result<(), String> {
    // Note: In production, this would use a session manager
    // For now, creating a new session for each operation
    let mut session = browser::BrowserSession::new(url.clone(), true);
    session.launch().await?;
    session.navigate().await
}

/// Click an element
#[tauri::command]
async fn browser_click(url: String, selector: String) -> Result<(), String> {
    let mut session = browser::BrowserSession::new(url, true);
    session.launch().await?;
    session.navigate().await?;
    session.click(&selector).await
}

/// Type text into an element
#[tauri::command]
async fn browser_type(url: String, selector: String, text: String) -> Result<(), String> {
    let mut session = browser::BrowserSession::new(url, true);
    session.launch().await?;
    session.navigate().await?;
    session.type_text(&selector, &text).await
}

/// Capture screenshot
#[tauri::command]
async fn browser_screenshot(url: String, path: String) -> Result<(), String> {
    let mut session = browser::BrowserSession::new(url, true);
    session.launch().await?;
    session.navigate().await?;
    session.screenshot(&path).await
}

/// Execute JavaScript in browser context
#[tauri::command]
async fn browser_evaluate_js(url: String, script: String) -> Result<String, String> {
    let mut session = browser::BrowserSession::new(url, true);
    session.launch().await?;
    session.navigate().await?;
    session.execute_script(&script).await
}

/// Collect console messages
#[tauri::command]
async fn browser_console_logs(url: String, duration_seconds: u64) -> Result<Vec<browser::ConsoleMessage>, String> {
    let mut session = browser::BrowserSession::new(url, true);
    session.launch().await?;
    session.navigate().await?;
    session.collect_messages(duration_seconds).await
}

/// Close browser session
#[tauri::command]
async fn browser_close(url: String) -> Result<(), String> {
    let mut session = browser::BrowserSession::new(url, true);
    session.close().await
}

// File operations commands

/// Delete file with optional backup
#[tauri::command]
fn file_delete(request: agent::file_ops::FileDeleteRequest) -> Result<agent::file_ops::FileDeleteResult, String> {
    agent::file_ops::delete_file(request)
}

/// Move/rename file
#[tauri::command]
fn file_move(request: agent::file_ops::FileMoveRequest) -> Result<agent::file_ops::FileMoveResult, String> {
    agent::file_ops::move_file(request)
}

/// Get directory tree structure
#[tauri::command]
fn directory_tree(root_path: String, max_depth: Option<usize>) -> Result<agent::file_ops::DirectoryTreeNode, String> {
    agent::file_ops::get_directory_tree(root_path, max_depth)
}

/// Search for files by pattern
#[tauri::command]
fn file_search(request: agent::file_ops::FileSearchRequest) -> Result<Vec<agent::file_ops::FileSearchResult>, String> {
    agent::file_ops::search_files(request)
}

// Command classification

/// Classify a command for execution strategy
#[tauri::command]
fn classify_command(command: String) -> agent::command_classifier::CommandClassification {
    agent::command_classifier::CommandClassifier::classify(&command)
}

// Status tracking commands

/// Register a new task for progress tracking
#[tauri::command]
async fn status_register_task(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    task_id: String,
    task_name: String,
) -> Result<(), String> {
    let emitter = status_emitter.lock().await;
    emitter.register_task(task_id, task_name).await;
    Ok(())
}

/// Update task progress
#[tauri::command]
async fn status_update_progress(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    task_id: String,
    progress_percent: f64,
    current_step: Option<String>,
) -> Result<(), String> {
    let emitter = status_emitter.lock().await;
    emitter.update_progress(&task_id, progress_percent, current_step).await;
    Ok(())
}

/// Get task progress
#[tauri::command]
async fn status_get_progress(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    task_id: String,
) -> Result<Option<agent::status_emitter::TaskProgress>, String> {
    let emitter = status_emitter.lock().await;
    Ok(emitter.get_task_progress(&task_id).await)
}

/// Get all tasks
#[tauri::command]
async fn status_get_all_tasks(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
) -> Result<Vec<agent::status_emitter::TaskProgress>, String> {
    let emitter = status_emitter.lock().await;
    Ok(emitter.get_all_tasks().await)
}

/// Complete a task
#[tauri::command]
async fn status_complete_task(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    task_id: String,
) -> Result<(), String> {
    let emitter = status_emitter.lock().await;
    emitter.complete_task(&task_id).await;
    Ok(())
}

/// Fail a task
#[tauri::command]
async fn status_fail_task(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    task_id: String,
    error_message: String,
) -> Result<(), String> {
    let emitter = status_emitter.lock().await;
    emitter.fail_task(&task_id, error_message).await;
    Ok(())
}

// Add Decision command (continuing from documentation)

/// Add a new decision
#[tauri::command]
fn add_decision(
    workspace_path: String,
    title: String,
    context: String,
    decision: String,
    rationale: String,
) -> Result<(), String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    manager.add_decision(title, context, decision, rationale);
    Ok(())
}

/// Add a change log entry
#[tauri::command]
fn add_change(
    workspace_path: String,
    change_type: String,
    description: String,
    files: Vec<String>,
) -> Result<(), String> {
    let mut manager = documentation::DocumentationManager::new(PathBuf::from(&workspace_path));
    manager.load_from_files()?;
    
    let change_type_enum = match change_type.to_lowercase().as_str() {
        "file-added" => documentation::ChangeType::FileAdded,
        "file-modified" => documentation::ChangeType::FileModified,
        "file-deleted" => documentation::ChangeType::FileDeleted,
        "function-added" => documentation::ChangeType::FunctionAdded,
        "function-removed" => documentation::ChangeType::FunctionRemoved,
        _ => return Err(format!("Invalid change type: {}", change_type)),
    };
    
    manager.add_change(change_type_enum, description, files);
    Ok(())
}

/// Extract features from chat message using LLM
#[tauri::command]
async fn extract_features_from_chat(
    app_handle: tauri::AppHandle,
    chat_message: String,
    context: Option<String>,
) -> Result<documentation::extractor::FeatureExtractionResponse, String> {
    // Get LLM configuration
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let manager = llm::config::LLMConfigManager::new(&config_dir)?;
    let llm_config = manager.get_config();
    
    // Verify API key is configured
    if !llm_config.has_api_key() {
        return Err("No LLM API key configured. Please configure at least one provider.".to_string());
    }
    
    // Create extraction request
    let request = documentation::extractor::FeatureExtractionRequest {
        chat_message,
        context,
    };
    
    // Extract features using LLM
    documentation::extractor::extract_features_from_chat(request, llm_config).await
}

// Test Coverage commands

/// Get test coverage metrics from GNN
#[tauri::command]
fn get_test_coverage(workspace_path: String) -> Result<agent::orchestrator::TestCoverageMetrics, String> {
    use std::path::Path;
    
    // Create or load GNN engine
    let workspace = Path::new(&workspace_path);
    let db_path = workspace.join(".yantra").join("graph.db");
    
    if !db_path.exists() {
        return Err("GNN database not found. Please build the project graph first.".to_string());
    }
    
    let mut gnn = gnn::GNNEngine::new(&db_path)
        .map_err(|e| format!("Failed to load GNN: {}", e))?;
    
    // Build graph if empty
    let node_count = gnn.get_node_count();
    if node_count == 0 {
        gnn.build_graph(workspace)
            .map_err(|e| format!("Failed to build graph: {}", e))?;
    }
    
    // Create test edges if not already created
    let _ = gnn.create_test_edges();
    
    // Calculate coverage
    let metrics = agent::orchestrator::calculate_test_coverage(&gnn);
    
    Ok(metrics)
}

/// Get list of affected tests for changed files
#[tauri::command]
fn get_affected_tests(
    workspace_path: String,
    changed_files: Vec<String>,
) -> Result<Vec<String>, String> {
    use std::path::Path;
    
    let workspace = Path::new(&workspace_path);
    let db_path = workspace.join(".yantra").join("graph.db");
    
    if !db_path.exists() {
        return Err("GNN database not found. Please build the project graph first.".to_string());
    }
    
    let mut gnn = gnn::GNNEngine::new(&db_path)
        .map_err(|e| format!("Failed to load GNN: {}", e))?;
    
    // Build graph if empty
    let node_count = gnn.get_node_count();
    if node_count == 0 {
        gnn.build_graph(workspace)
            .map_err(|e| format!("Failed to build graph: {}", e))?;
    }
    
    // Create test edges if not already created
    let _ = gnn.create_test_edges();
    
    // Find affected tests
    let affected = agent::orchestrator::find_affected_tests(&gnn, &changed_files);
    
    Ok(affected)
}

// LLM commands

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

/// Execute pytest tests and return results
#[tauri::command]
async fn execute_tests(
    workspace_path: String,
    test_file: String,
    timeout_seconds: Option<u64>,
) -> Result<testing::TestExecutionResult, String> {
    let workspace = PathBuf::from(workspace_path);
    let test_path = Path::new(&test_file);
    
    let executor = testing::PytestExecutor::new(workspace);
    executor.execute_tests(test_path, timeout_seconds)
}

/// Execute pytest tests with coverage
#[tauri::command]
async fn execute_tests_with_coverage(
    workspace_path: String,
    test_file: String,
    timeout_seconds: Option<u64>,
) -> Result<testing::TestExecutionResult, String> {
    let workspace = PathBuf::from(workspace_path);
    let test_path = Path::new(&test_file);
    
    let executor = testing::PytestExecutor::new(workspace);
    executor.execute_tests_with_coverage(test_path, timeout_seconds)
}

// Git commands

/// Get git status
#[tauri::command]
fn git_status(workspace_path: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.status()
}

/// Add files to git staging
#[tauri::command]
fn git_add(workspace_path: String, files: Vec<String>) -> Result<(), String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.add_files(&files)
}

/// Commit staged changes
#[tauri::command]
fn git_commit(workspace_path: String, message: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.commit(&message)
}

/// Get git diff
#[tauri::command]
fn git_diff(workspace_path: String, file: Option<String>) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.diff(file.as_deref())
}

/// Get git log
#[tauri::command]
fn git_log(workspace_path: String, max_count: usize) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.log(max_count)
}

/// List git branches
#[tauri::command]
fn git_branch_list(workspace_path: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.branch_list()
}

/// Get current branch
#[tauri::command]
fn git_current_branch(workspace_path: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.current_branch()
}

/// Checkout branch
#[tauri::command]
fn git_checkout(workspace_path: String, branch: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.checkout(&branch)
}

/// Git pull
#[tauri::command]
fn git_pull(workspace_path: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.pull()
}

/// Git push
#[tauri::command]
fn git_push(workspace_path: String) -> Result<String, String> {
    let git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.push()
}

/// Get graph dependencies for visualization
#[tauri::command]
fn get_graph_dependencies(app_handle: tauri::AppHandle) -> Result<serde_json::Value, String> {
    // Get workspace path from app state or config
    // For now, return empty graph if no workspace is open
    let workspace_path = app_handle
        .path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?;
    
    let db_path = workspace_path.join("yantra.db");
    
    // Try to load existing GNN engine
    let gnn_result = gnn::GNNEngine::new(&db_path)
        .and_then(|mut engine| {
            engine.load()?;
            Ok(engine)
        });
    
    match gnn_result {
        Ok(engine) => {
            let graph = engine.get_graph();
            let all_nodes = graph.get_all_nodes();
            
            // Transform nodes to frontend format
            let nodes: Vec<serde_json::Value> = all_nodes
                .iter()
                .map(|node| {
                    serde_json::json!({
                        "id": node.id,
                        "label": node.name,
                        "type": match node.node_type {
                            gnn::NodeType::Function => "function",
                            gnn::NodeType::Class => "class",
                            gnn::NodeType::Import => "import",
                            gnn::NodeType::Variable => "variable",
                            gnn::NodeType::Module => "file",
                        },
                        "file_path": node.file_path,
                    })
                })
                .collect();
            
            // Get edges by checking dependencies for each node
            let mut edges: Vec<serde_json::Value> = Vec::new();
            for node in all_nodes.iter() {
                let deps = engine.get_dependencies(&node.id);
                for dep in deps {
                    edges.push(serde_json::json!({
                        "source": node.id,
                        "target": dep.id,
                        "type": "uses", // Simplified for MVP
                    }));
                }
            }
            
            Ok(serde_json::json!({
                "nodes": nodes,
                "edges": edges,
            }))
        }
        Err(_) => {
            // Return empty graph if no GNN data exists yet
            Ok(serde_json::json!({
                "nodes": [],
                "edges": [],
            }))
        }
    }
}

/// Execute terminal command with streaming output
#[tauri::command]
async fn execute_terminal_command(
    terminal_id: String,
    command: String,
    working_dir: Option<String>,
    window: tauri::Window,
) -> Result<i32, String> {
    use std::process::{Command, Stdio};
    use std::io::{BufRead, BufReader};
    use tokio::task;
    
    // Determine working directory
    let work_dir = working_dir.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    });
    
    // Execute command in separate thread to avoid blocking
    let terminal_id_clone = terminal_id.clone();
    let window_clone = window.clone();
    
    let exit_code = task::spawn_blocking(move || {
        // Use shell to execute command (supports pipes, redirects, etc.)
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(&work_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to execute command: {}", e))?;
        
        // Stream stdout
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let tid = terminal_id_clone.clone();
            let win = window_clone.clone();
            
            std::thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        let _ = win.emit("terminal-output", serde_json::json!({
                            "terminal_id": tid,
                            "output": line,
                            "stream": "stdout",
                        }));
                    }
                }
            });
        }
        
        // Stream stderr
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            let tid = terminal_id_clone.clone();
            let win = window_clone.clone();
            
            std::thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        let _ = win.emit("terminal-output", serde_json::json!({
                            "terminal_id": tid,
                            "output": line,
                            "stream": "stderr",
                        }));
                    }
                }
            });
        }
        
        // Wait for process to complete
        let status = child.wait()
            .map_err(|e| format!("Failed to wait for command: {}", e))?;
        
        // Send completion event
        let _ = window_clone.emit("terminal-complete", serde_json::json!({
            "terminal_id": terminal_id_clone,
            "exit_code": status.code().unwrap_or(-1),
        }));
        
        Ok::<i32, String>(status.code().unwrap_or(-1))
    }).await
    .map_err(|e| format!("Task join error: {}", e))??;
    
    Ok(exit_code)
}

// ========================================
// PTY-Based Terminal Commands (New Implementation)
// ========================================

/// Create a new PTY-based terminal session
#[tauri::command]
async fn create_pty_terminal(
    terminal_id: String,
    name: String,
    shell: Option<String>,
    terminal_manager: State<'_, Arc<TokioMutex<TerminalManager>>>,
    window: tauri::Window,
) -> Result<(), String> {
    let manager = terminal_manager.lock().await;
    manager.create_terminal(terminal_id, name, shell, window)?;
    Ok(())
}

/// Write input to a PTY terminal
#[tauri::command]
async fn write_pty_input(
    terminal_id: String,
    data: String,
    terminal_manager: State<'_, Arc<TokioMutex<TerminalManager>>>,
) -> Result<(), String> {
    let manager = terminal_manager.lock().await;
    // Decode base64 input
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&data)
        .map_err(|e| format!("Failed to decode input: {}", e))?;
    manager.write_input(&terminal_id, &decoded)?;
    Ok(())
}

/// Resize a PTY terminal
#[tauri::command]
async fn resize_pty_terminal(
    terminal_id: String,
    rows: u16,
    cols: u16,
    terminal_manager: State<'_, Arc<TokioMutex<TerminalManager>>>,
) -> Result<(), String> {
    let manager = terminal_manager.lock().await;
    manager.resize(&terminal_id, rows, cols)?;
    Ok(())
}

/// Close a PTY terminal
#[tauri::command]
async fn close_pty_terminal(
    terminal_id: String,
    terminal_manager: State<'_, Arc<TokioMutex<TerminalManager>>>,
) -> Result<(), String> {
    let manager = terminal_manager.lock().await;
    manager.close_terminal(&terminal_id)?;
    Ok(())
}

/// List all PTY terminals
#[tauri::command]
async fn list_pty_terminals(
    terminal_manager: State<'_, Arc<TokioMutex<TerminalManager>>>,
) -> Result<Vec<String>, String> {
    let manager = terminal_manager.lock().await;
    Ok(manager.list_terminals())
}

// Menu event handler
fn handle_menu_event(event: tauri::WindowMenuEvent) {
    match event.menu_item_id() {
        // File menu
        "new_file" => {
            let _ = event.window().emit("menu-new-file", ());
        }
        "new_folder" => {
            let _ = event.window().emit("menu-new-folder", ());
        }
        "open_folder" => {
            let _ = event.window().emit("menu-open-folder", ());
        }
        "save" => {
            let _ = event.window().emit("menu-save", ());
        }
        "save_all" => {
            let _ = event.window().emit("menu-save-all", ());
        }
        "close_folder" => {
            let _ = event.window().emit("menu-close-folder", ());
        }
        "close_window" => {
            let _ = event.window().close();
        }
        // Edit menu
        "undo" => {
            let _ = event.window().emit("menu-undo", ());
        }
        "redo" => {
            let _ = event.window().emit("menu-redo", ());
        }
        "cut" => {
            let _ = event.window().emit("menu-cut", ());
        }
        "copy" => {
            let _ = event.window().emit("menu-copy", ());
        }
        "paste" => {
            let _ = event.window().emit("menu-paste", ());
        }
        "select_all" => {
            let _ = event.window().emit("menu-select-all", ());
        }
        "find" => {
            let _ = event.window().emit("menu-find", ());
        }
        "replace" => {
            let _ = event.window().emit("menu-replace", ());
        }
        // Yantra menu
        "about" => {
            let _ = event.window().emit("menu-about", ());
        }
        "check_updates" => {
            let _ = event.window().emit("menu-check-updates", ());
        }
        "settings" => {
            let _ = event.window().emit("menu-settings", ());
        }
        // View menu
        "toggle_terminal" => {
            let _ = event.window().emit("menu-toggle-terminal", ());
        }
        "toggle_file_tree" => {
            let _ = event.window().emit("menu-toggle-file-tree", ());
        }
        "reset_layout" => {
            let _ = event.window().emit("menu-reset-layout", ());
        }
        _ => {}
    }
}

// Task Queue Commands

/// Get all tasks in the queue
#[tauri::command]
fn get_task_queue(app_handle: tauri::AppHandle) -> Result<Vec<agent::Task>, String> {
    let tasks_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("task_queue.json");
    
    let queue = agent::TaskQueue::new(tasks_path)?;
    Ok(queue.get_all_tasks())
}

/// Get the current in-progress task
#[tauri::command]
fn get_current_task(app_handle: tauri::AppHandle) -> Result<Option<agent::Task>, String> {
    let tasks_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("task_queue.json");
    
    let queue = agent::TaskQueue::new(tasks_path)?;
    Ok(queue.get_current_task())
}

/// Add a new task to the queue
#[tauri::command]
fn add_task(
    app_handle: tauri::AppHandle,
    id: String,
    description: String,
    priority: String,
) -> Result<(), String> {
    let tasks_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("task_queue.json");
    
    let priority_enum = match priority.to_lowercase().as_str() {
        "low" => agent::TaskPriority::Low,
        "medium" => agent::TaskPriority::Medium,
        "high" => agent::TaskPriority::High,
        "critical" => agent::TaskPriority::Critical,
        _ => return Err(format!("Invalid priority: {}", priority)),
    };
    
    let task = agent::Task::new(id, description, priority_enum);
    let queue = agent::TaskQueue::new(tasks_path)?;
    queue.add_task(task)?;
    Ok(())
}

/// Update task status
#[tauri::command]
fn update_task_status(
    app_handle: tauri::AppHandle,
    id: String,
    status: String,
) -> Result<(), String> {
    let tasks_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("task_queue.json");
    
    let status_enum = match status.to_lowercase().as_str() {
        "pending" => agent::TaskStatus::Pending,
        "inprogress" | "in-progress" => agent::TaskStatus::InProgress,
        "completed" => agent::TaskStatus::Completed,
        "failed" => agent::TaskStatus::Failed,
        _ => return Err(format!("Invalid status: {}", status)),
    };
    
    let queue = agent::TaskQueue::new(tasks_path)?;
    queue.update_task_status(&id, status_enum)?;
    Ok(())
}

/// Complete a task
#[tauri::command]
fn complete_task(app_handle: tauri::AppHandle, id: String) -> Result<(), String> {
    let tasks_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("task_queue.json");
    
    let queue = agent::TaskQueue::new(tasks_path)?;
    queue.complete_task(&id)?;
    Ok(())
}

/// Get task queue statistics
#[tauri::command]
fn get_task_stats(app_handle: tauri::AppHandle) -> Result<agent::TaskStats, String> {
    let tasks_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("task_queue.json");
    
    let queue = agent::TaskQueue::new(tasks_path)?;
    Ok(queue.get_stats())
}

fn main() {
    // Build minimal custom menu
    let file_menu = Submenu::new(
        "File",
        Menu::new()
            .add_item(CustomMenuItem::new("new_file", "New File").accelerator("Cmd+N"))
            .add_item(CustomMenuItem::new("new_folder", "New Folder").accelerator("Cmd+Shift+N"))
            .add_item(CustomMenuItem::new("open_folder", "Open Folder...").accelerator("Cmd+O"))
            .add_item(CustomMenuItem::new("separator1", "───────────────").disabled())
            .add_item(CustomMenuItem::new("save", "Save").accelerator("Cmd+S"))
            .add_item(CustomMenuItem::new("save_all", "Save All").accelerator("Cmd+Alt+S"))
            .add_item(CustomMenuItem::new("separator2", "───────────────").disabled())
            .add_item(CustomMenuItem::new("close_folder", "Close Folder"))
            .add_item(CustomMenuItem::new("close_window", "Close Window").accelerator("Cmd+W")),
    );

    let edit_menu = Submenu::new(
        "Actions",
        Menu::new()
            .add_item(CustomMenuItem::new("undo", "Undo").accelerator("Cmd+Z"))
            .add_item(CustomMenuItem::new("redo", "Redo").accelerator("Cmd+Shift+Z"))
            .add_item(CustomMenuItem::new("separator1", "───────────────").disabled())
            .add_item(CustomMenuItem::new("cut", "Cut").accelerator("Cmd+X"))
            .add_item(CustomMenuItem::new("copy", "Copy").accelerator("Cmd+C"))
            .add_item(CustomMenuItem::new("paste", "Paste").accelerator("Cmd+V"))
            .add_item(CustomMenuItem::new("select_all", "Select All").accelerator("Cmd+A"))
            .add_item(CustomMenuItem::new("separator2", "───────────────").disabled())
            .add_item(CustomMenuItem::new("find", "Find").accelerator("Cmd+F"))
            .add_item(CustomMenuItem::new("replace", "Replace").accelerator("Cmd+H")),
    );

    let yantra_menu = Submenu::new(
        "Yantra",
        Menu::new()
            .add_item(CustomMenuItem::new("about", "About Yantra"))
            .add_item(CustomMenuItem::new("check_updates", "Check for Updates..."))
            .add_item(CustomMenuItem::new("separator1", "───────────────").disabled())
            .add_item(CustomMenuItem::new("settings", "Settings...").accelerator("Cmd+,"))
            .add_item(CustomMenuItem::new("separator2", "───────────────").disabled())
            .add_native_item(MenuItem::Quit),
    );

    let view_menu = Submenu::new(
        "View",
        Menu::new()
            .add_item(CustomMenuItem::new("toggle_terminal", "Toggle Terminal").accelerator("Cmd+`"))
            .add_item(CustomMenuItem::new("toggle_file_tree", "Toggle File Tree").accelerator("Cmd+B"))
            .add_item(CustomMenuItem::new("separator1", "───────────────").disabled())
            .add_item(CustomMenuItem::new("reset_layout", "Reset Layout")),
    );

    let menu = Menu::new()
        .add_submenu(yantra_menu)
        .add_submenu(file_menu)
        .add_submenu(edit_menu)
        .add_submenu(view_menu);

    // Initialize terminal manager
    let terminal_manager = Arc::new(TokioMutex::new(TerminalManager::new()));
    
    // Initialize architecture state with GNN and LLM
    // Default database path in .yantra directory
    let home_dir = dirs::home_dir().expect("Cannot determine home directory");
    let yantra_dir = home_dir.join(".yantra");
    std::fs::create_dir_all(&yantra_dir).expect("Failed to create .yantra directory");
    
    let db_path = yantra_dir.join("graph.db");
    let gnn = Arc::new(tokio::sync::Mutex::new(
        gnn::GNNEngine::new(&db_path).expect("Failed to initialize GNN engine")
    ));
    
    // Initialize LLM with default config
    let llm_config = llm::LLMConfig {
        claude_api_key: None,
        openai_api_key: None,
        openrouter_api_key: None,
        groq_api_key: None,
        gemini_api_key: None,
        primary_provider: llm::LLMProvider::Claude,
        max_retries: 3,
        timeout_seconds: 30,
        selected_models: Vec::new(),
    };
    let llm = Arc::new(tokio::sync::Mutex::new(llm::orchestrator::LLMOrchestrator::new(llm_config)));
    
    let arch_state = arch_commands::ArchitectureState::new(gnn, llm)
        .expect("Failed to initialize architecture state");

    // Initialize database manager
    let db_manager = Arc::new(tokio::sync::Mutex::new(
        agent::database::DatabaseManager::new().expect("Failed to initialize database manager")
    ));
    
    // Initialize status emitter
    let status_emitter = Arc::new(tokio::sync::Mutex::new(agent::status_emitter::StatusEmitter::new()));

    tauri::Builder::default()
        .menu(menu)
        .manage(terminal_manager)
        .manage(arch_state)
        .manage(db_manager)
        .manage(status_emitter)
        .on_menu_event(|event| {
            handle_menu_event(event);
        })
        .invoke_handler(tauri::generate_handler![
            read_file,
            write_file,
            edit_file,
            preview_file_edit,
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
            set_openrouter_key,
            set_groq_key,
            set_gemini_key,
            clear_llm_key,
            get_available_models,
            get_default_model,
            set_selected_models,
            get_selected_models,
            set_llm_retry_config,
            generate_code,
            generate_tests,
            execute_tests,
            execute_tests_with_coverage,
            git_status,
            git_add,
            git_commit,
            git_diff,
            git_log,
            git_branch_list,
            git_current_branch,
            git_checkout,
            git_pull,
            git_push,
            get_graph_dependencies,
            execute_terminal_command,
            create_pty_terminal,
            write_pty_input,
            resize_pty_terminal,
            close_pty_terminal,
            list_pty_terminals,
            get_features,
            get_decisions,
            get_changes,
            get_tasks,
            add_feature,
            add_decision,
            add_change,
            extract_features_from_chat,
            // Database commands
            db_connect,
            db_query,
            db_execute,
            db_schema,
            db_list_connections,
            db_disconnect,
            db_test_connection,
            // Browser automation commands
            browser_launch,
            browser_navigate,
            browser_click,
            browser_type,
            browser_screenshot,
            browser_evaluate_js,
            browser_console_logs,
            browser_close,
            // File operations commands
            file_delete,
            file_move,
            directory_tree,
            file_search,
            // Command classification
            classify_command,
            // Status tracking commands
            status_register_task,
            status_update_progress,
            status_get_progress,
            status_get_all_tasks,
            status_complete_task,
            status_fail_task,
            // Test Coverage commands
            get_test_coverage,
            get_affected_tests,
            // Task Queue commands
            get_task_queue,
            get_current_task,
            add_task,
            update_task_status,
            complete_task,
            get_task_stats,
            // Architecture View commands
            arch_commands::generate_architecture_from_intent,
            arch_commands::generate_architecture_from_code,
            arch_commands::initialize_new_project,
            arch_commands::initialize_existing_project,
            arch_commands::review_existing_code,
            arch_commands::analyze_requirement_impact,
            arch_commands::is_project_initialized,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
