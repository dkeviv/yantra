// File: src-tauri/src/main.rs
// Purpose: Tauri application entry point with file system operations and GNN
// Dependencies: tauri, serde_json, std::fs
// Last Updated: November 20, 2025

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::collections::HashMap;
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
mod browser;
mod security;
mod ydoc;
mod ydoc_commands;
mod code_intelligence;

use terminal::TerminalManager;
use architecture::commands as arch_commands;
use ydoc_commands::YDocState;

// State for managing file watcher
struct FileWatcherState {
    watcher: TokioMutex<Option<gnn::file_watcher::FileWatcher>>,
}

impl FileWatcherState {
    fn new() -> Self {
        Self {
            watcher: TokioMutex::new(None),
        }
    }
}

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

/// Copy file or directory
#[tauri::command]
fn file_copy(source: String, destination: String) -> Result<FileOperationResult, String> {
    let source_path = PathBuf::from(&source);
    let dest_path = PathBuf::from(&destination);
    
    if !source_path.exists() {
        return Err(format!("Source path does not exist: {}", source));
    }
    
    if source_path.is_file() {
        // Copy single file
        fs::copy(&source_path, &dest_path)
            .map_err(|e| format!("Failed to copy file: {}", e))?;
        Ok(FileOperationResult {
            success: true,
            message: format!("File copied from {} to {}", source, destination),
        })
    } else {
        // Copy directory recursively
        copy_dir_recursive(&source_path, &dest_path)?;
        Ok(FileOperationResult {
            success: true,
            message: format!("Directory copied from {} to {}", source, destination),
        })
    }
}

/// Helper function to copy directory recursively
fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), String> {
    if !destination.exists() {
        fs::create_dir_all(destination)
            .map_err(|e| format!("Failed to create destination directory: {}", e))?;
    }
    
    for entry in fs::read_dir(source)
        .map_err(|e| format!("Failed to read source directory: {}", e))? {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let source_path = entry.path();
        let dest_path = destination.join(entry.file_name());
        
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &dest_path)?;
        } else {
            fs::copy(&source_path, &dest_path)
                .map_err(|e| format!("Failed to copy file: {}", e))?;
        }
    }
    
    Ok(())
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

// Code Intelligence commands

/// Parse AST from a file
#[tauri::command]
fn parse_ast(file_path: String) -> Result<code_intelligence::ASTNode, String> {
    let mut parser = code_intelligence::ast_parser::ASTParser::new();
    parser.parse_file(&file_path)
}

/// Parse AST from code snippet
#[tauri::command]
fn parse_ast_snippet(code: String, language: String) -> Result<code_intelligence::ASTNode, String> {
    let mut parser = code_intelligence::ast_parser::ASTParser::new();
    parser.parse_snippet(&code, &language)
}

/// Extract symbols from a file
#[tauri::command]
fn get_symbols(file_path: String) -> Result<Vec<code_intelligence::Symbol>, String> {
    let mut extractor = code_intelligence::symbol_extractor::SymbolExtractor::new();
    extractor.extract_symbols(&file_path)
}

/// Find all references to a symbol in a file
#[tauri::command]
fn get_references(file_path: String, symbol_name: String) -> Result<Vec<code_intelligence::Reference>, String> {
    let mut finder = code_intelligence::reference_finder::ReferenceFinder::new();
    finder.find_references_in_file(&file_path, &symbol_name)
}

/// Find all references to a symbol across project
#[tauri::command]
fn find_all_references(project_path: String, symbol_name: String) -> Result<Vec<code_intelligence::Reference>, String> {
    let mut finder = code_intelligence::reference_finder::ReferenceFinder::new();
    finder.find_references_in_project(&project_path, &symbol_name)
}

/// Find definition of a symbol in a file
#[tauri::command]
fn get_definition(file_path: String, symbol_name: String) -> Result<Option<code_intelligence::Reference>, String> {
    let mut finder = code_intelligence::reference_finder::ReferenceFinder::new();
    finder.find_definition_in_file(&file_path, &symbol_name)
}

/// Get scope information at a specific position
#[tauri::command]
fn get_scope(file_path: String, line: usize, column: usize) -> Result<code_intelligence::ScopeInfo, String> {
    let mut analyzer = code_intelligence::scope_analyzer::ScopeAnalyzer::new();
    analyzer.get_scope_at_position(&file_path, line, column)
}

/// Get all scopes in a file
#[tauri::command]
fn get_all_scopes(file_path: String) -> Result<Vec<code_intelligence::ScopeInfo>, String> {
    let mut analyzer = code_intelligence::scope_analyzer::ScopeAnalyzer::new();
    analyzer.get_all_scopes(&file_path)
}

/// Get code completions using Tree-sitter and GNN
#[tauri::command]
fn get_code_completions(request: gnn::completion::CompletionRequest) -> Result<Vec<gnn::completion::CompletionItem>, String> {
    let project_path = Path::new(&request.project_path);
    let provider = gnn::completion::CompletionProvider::new(Some(project_path))?;
    provider.get_completions(&request)
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

/// Set secondary LLM provider for automatic failover
#[tauri::command]
fn set_secondary_llm_provider(app_handle: tauri::AppHandle, provider: Option<String>) -> Result<(), String> {
    let config_dir = app_handle.path_resolver()
        .app_config_dir()
        .ok_or_else(|| "Failed to get config directory".to_string())?;
    
    let mut manager = llm::config::LLMConfigManager::new(&config_dir)?;
    
    let provider_enum = match provider {
        Some(p) => {
            match p.to_lowercase().as_str() {
                "claude" => Some(llm::LLMProvider::Claude),
                "openai" => Some(llm::LLMProvider::OpenAI),
                "openrouter" => Some(llm::LLMProvider::OpenRouter),
                "groq" => Some(llm::LLMProvider::Groq),
                "gemini" => Some(llm::LLMProvider::Gemini),
                "none" => None,
                _ => return Err(format!("Invalid provider: {}. Use 'claude', 'openai', 'openrouter', 'groq', 'gemini', or 'none'", p)),
            }
        }
        None => None,
    };
    
    manager.set_secondary_provider(provider_enum)
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

// API Management commands

/// Import OpenAPI specification from file
#[tauri::command]
fn api_import_spec(file_path: String) -> Result<agent::api_manager::ApiSpec, String> {
    agent::api_manager::ApiManager::import_spec(&file_path)
}

/// Validate API contract against used endpoints
#[tauri::command]
fn api_validate_contract(
    spec: agent::api_manager::ApiSpec,
    used_endpoints: Vec<(String, agent::api_manager::HttpMethod)>,
) -> Result<agent::api_manager::ValidationResult, String> {
    Ok(agent::api_manager::ApiManager::validate_contract(&spec, used_endpoints))
}

// Dependency & Environment Management commands

/// Validate dependencies before execution
#[tauri::command]
fn validate_dependencies(
    dependencies: Vec<agent::dependency_manager::Dependency>,
) -> Result<agent::dependency_manager::ValidationResult, String> {
    Ok(agent::dependency_manager::DependencyValidator::validate_dependencies(dependencies))
}

/// Get virtual environment info
#[tauri::command]
fn get_venv_info(project_path: String) -> Result<agent::dependency_manager::VenvInfo, String> {
    Ok(agent::dependency_manager::VenvManager::get_venv_info(&project_path))
}

/// Create virtual environment
#[tauri::command]
fn create_venv(project_path: String) -> Result<(), String> {
    agent::dependency_manager::VenvManager::create_venv(&project_path)
}

/// Enforce venv usage (returns activation command if not active)
#[tauri::command]
fn enforce_venv(project_path: String) -> Result<Option<String>, String> {
    agent::dependency_manager::VenvManager::enforce_venv(&project_path)
}

// Conflict Detection commands

/// Detect dependency and import conflicts
#[tauri::command]
fn detect_conflicts(
    dependencies: Vec<agent::conflict_detector::VersionRequirement>,
    imports: HashMap<String, Vec<String>>,
    definitions: HashMap<String, Vec<String>>,
) -> Result<agent::conflict_detector::ConflictDetectionResult, String> {
    Ok(agent::conflict_detector::ConflictDetector::detect_conflicts(
        dependencies,
        imports,
        definitions,
    ))
}

// Intelligent Executor commands

/// Execute command with intelligent strategy
#[tauri::command]
async fn intelligent_execute(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    command: String,
) -> Result<agent::intelligent_executor::ExecutionResult, String> {
    let executor = agent::intelligent_executor::IntelligentExecutor::new()
        .with_status_emitter(status_emitter.inner().clone());
    
    executor.execute(&command).await
}

/// Execute background command
#[tauri::command]
async fn execute_background(
    status_emitter: tauri::State<'_, Arc<tokio::sync::Mutex<agent::status_emitter::StatusEmitter>>>,
    command: String,
) -> Result<agent::intelligent_executor::BackgroundTask, String> {
    let executor = agent::intelligent_executor::IntelligentExecutor::new()
        .with_status_emitter(status_emitter.inner().clone());
    
    executor.execute_background(&command).await
}

// GNN Version Tracking commands

/// Track a new version of a GNN node
#[tauri::command]
fn gnn_track_version(
    node_id: String,
    node_type: String,
    file_path: String,
    line_start: u32,
    line_end: u32,
    content: String,
    author: Option<String>,
    commit_hash: Option<String>,
    change_reason: Option<String>,
) -> Result<(), String> {
    // In production, this would be managed by GNNEngine state
    // For now, create a temporary node for the example
    use crate::gnn::version_tracker::VersionTracker;
    use crate::gnn::{CodeNode, NodeType};
    
    let node = CodeNode {
        id: node_id.clone(),
        name: node_id.split("::").last().unwrap_or("unknown").to_string(),
        node_type: match node_type.as_str() {
            "function" => NodeType::Function,
            "class" => NodeType::Class,
            "module" => NodeType::Module,
            "import" => NodeType::Import,
            _ => NodeType::Variable,
        },
        file_path,
        line_start: line_start as usize,
        line_end: line_end as usize,
        semantic_embedding: None,
        code_snippet: Some(content.clone()),
        docstring: None,
    };
    
    // TODO: Integrate with global VersionTracker state
    let mut tracker = VersionTracker::new();
    tracker.track_version(&node, content, author, commit_hash, change_reason);
    
    Ok(())
}

/// Get version history for a node
#[tauri::command]
fn gnn_get_history(node_id: String) -> Result<Option<gnn::version_tracker::NodeHistory>, String> {
    // TODO: Access global VersionTracker state
    Ok(None)
}

/// Rollback node to specific version
#[tauri::command]
fn gnn_rollback(node_id: String, target_version: u32) -> Result<gnn::version_tracker::NodeVersion, String> {
    // TODO: Access global VersionTracker state and apply rollback
    Err("Not implemented yet - needs global state integration".to_string())
}

/// Generate diff between two versions
#[tauri::command]
fn gnn_diff(node_id: String, old_version: u32, new_version: u32) -> Result<gnn::version_tracker::VersionDiff, String> {
    // TODO: Access global VersionTracker state
    Err("Not implemented yet - needs global state integration".to_string())
}

// Database Migration commands

/// Create new migration file
#[tauri::command]
fn db_create_migration(migrations_dir: String, name: String) -> Result<String, String> {
    use agent::database::migration_manager::MigrationManager;
    use std::path::PathBuf;
    
    let manager = MigrationManager::new(PathBuf::from(migrations_dir))?;
    let path = manager.create_migration(&name)?;
    
    Ok(path.to_string_lossy().to_string())
}

/// Get all migrations
#[tauri::command]
fn db_get_migrations(migrations_dir: String) -> Result<Vec<agent::database::migration_manager::Migration>, String> {
    use agent::database::migration_manager::MigrationManager;
    use std::path::PathBuf;
    
    let manager = MigrationManager::new(PathBuf::from(migrations_dir))?;
    Ok(manager.get_all_migrations().into_iter().cloned().collect())
}

/// Apply migration
#[tauri::command]
fn db_migrate_up(migrations_dir: String, version: u32) -> Result<agent::database::migration_manager::MigrationResult, String> {
    use agent::database::migration_manager::MigrationExecutor;
    use std::path::PathBuf;
    
    let executor = MigrationExecutor::new(PathBuf::from(migrations_dir))?;
    executor.migrate_up(version)
}

/// Rollback migration
#[tauri::command]
fn db_migrate_down(migrations_dir: String, version: u32) -> Result<agent::database::migration_manager::MigrationResult, String> {
    use agent::database::migration_manager::MigrationExecutor;
    use std::path::PathBuf;
    
    let executor = MigrationExecutor::new(PathBuf::from(migrations_dir))?;
    executor.migrate_down(version)
}

// API Health & Rate Limit commands

/// Check API health
#[tauri::command]
async fn api_health_check(endpoint: String) -> Result<agent::api_health::HealthCheckResult, String> {
    let mut monitor = agent::api_health::HealthMonitor::new();
    Ok(monitor.check_health(&endpoint).await)
}

/// Get API uptime
#[tauri::command]
fn api_get_uptime(
    endpoint: String,
) -> Result<f32, String> {
    // TODO: Integrate with global HealthMonitor state
    Ok(0.0)
}

/// Record API request for rate limiting
#[tauri::command]
fn api_record_request(endpoint: String) -> Result<(), String> {
    // TODO: Integrate with global RateLimitTracker state
    Ok(())
}

/// Get rate limit info
#[tauri::command]
fn api_get_rate_limit(endpoint: String) -> Result<Option<agent::api_health::RateLimitInfo>, String> {
    // TODO: Integrate with global RateLimitTracker state
    Ok(None)
}

// Environment Snapshot & Validation commands

/// Create environment snapshot
#[tauri::command]
fn env_create_snapshot(project_path: String) -> Result<agent::environment::EnvironmentSnapshot, String> {
    use agent::environment::SnapshotManager;
    use std::path::Path;
    
    let path = Path::new(&project_path);
    let manager = SnapshotManager::new(path);
    manager.create_snapshot(path)
}

/// Rollback to snapshot
#[tauri::command]
fn env_rollback(project_path: String) -> Result<(), String> {
    use agent::environment::SnapshotManager;
    use std::path::Path;
    
    let path = Path::new(&project_path);
    let manager = SnapshotManager::new(path);
    let snapshot = manager.load_latest_snapshot()?;
    manager.rollback(&snapshot, path)
}

/// Validate environment
#[tauri::command]
fn env_validate(project_path: String) -> Result<agent::environment::ValidationResult, String> {
    use agent::environment::EnvironmentValidator;
    use std::path::Path;
    
    Ok(EnvironmentValidator::validate(Path::new(&project_path)))
}

/// Set secret
#[tauri::command]
fn secrets_set(key: String, value: String, description: Option<String>) -> Result<(), String> {
    use agent::secrets::SecretsManager;
    use std::path::PathBuf;
    
    // TODO: Get vault path and encryption key from config
    let vault_path = PathBuf::from(".yantra/secrets.json");
    let encryption_key: [u8; 32] = [0u8; 32]; // TODO: Generate/load proper key
    
    let manager = SecretsManager::new(vault_path, &encryption_key);
    manager.set_secret(&key, &value, description)
}

/// Get secret
#[tauri::command]
fn secrets_get(key: String) -> Result<String, String> {
    use agent::secrets::SecretsManager;
    use std::path::PathBuf;
    
    let vault_path = PathBuf::from(".yantra/secrets.json");
    let encryption_key: [u8; 32] = [0u8; 32]; // TODO: Generate/load proper key
    
    let manager = SecretsManager::new(vault_path, &encryption_key);
    manager.get_secret(&key)
}

/// Delete secret
#[tauri::command]
fn secrets_delete(key: String) -> Result<(), String> {
    use agent::secrets::SecretsManager;
    use std::path::PathBuf;
    
    let vault_path = PathBuf::from(".yantra/secrets.json");
    let encryption_key: [u8; 32] = [0u8; 32]; // TODO: Generate/load proper key
    
    let manager = SecretsManager::new(vault_path, &encryption_key);
    manager.delete_secret(&key)
}

/// List all secret keys
#[tauri::command]
fn secrets_list() -> Result<Vec<String>, String> {
    use agent::secrets::SecretsManager;
    use std::path::PathBuf;
    
    let vault_path = PathBuf::from(".yantra/secrets.json");
    let encryption_key: [u8; 32] = [0u8; 32]; // TODO: Generate/load proper key
    
    let manager = SecretsManager::new(vault_path, &encryption_key);
    manager.list_secrets()
}

/// Get secret metadata
#[tauri::command]
fn secrets_metadata(key: String) -> Result<agent::secrets::SecretMetadata, String> {
    use agent::secrets::SecretsManager;
    use std::path::PathBuf;
    
    let vault_path = PathBuf::from(".yantra/secrets.json");
    let encryption_key: [u8; 32] = [0u8; 32]; // TODO: Generate/load proper key
    
    let manager = SecretsManager::new(vault_path, &encryption_key);
    manager.get_metadata(&key)
}

/// Read DOCX file
#[tauri::command]
fn read_docx(file_path: String) -> Result<agent::document_readers::DocumentContent, String> {
    use agent::document_readers::DocxReader;
    use std::path::Path;
    
    DocxReader::read(Path::new(&file_path))
}

/// Read PDF file
#[tauri::command]
fn read_pdf(file_path: String) -> Result<agent::document_readers::DocumentContent, String> {
    use agent::document_readers::PdfReader;
    use std::path::Path;
    
    PdfReader::read(Path::new(&file_path))
}

/// Find affected tests
#[tauri::command]
fn find_affected_tests(
    changed_files: Vec<String>,
    strategy: String,
) -> Result<agent::affected_tests::TestImpactAnalysis, String> {
    use agent::affected_tests::{AffectedTestsRunner, FilterStrategy};
    use crate::gnn::GNNEngine;
    use std::path::PathBuf;
    
    // TODO: Get GNN from global state
    let gnn = GNNEngine::new(&PathBuf::from("project"))?;
    let runner = AffectedTestsRunner::new(gnn, String::new());
    
    let paths: Vec<PathBuf> = changed_files.iter().map(PathBuf::from).collect();
    let filter_strategy = match strategy.as_str() {
        "direct" => FilterStrategy::Direct,
        "transitive" => FilterStrategy::Transitive,
        _ => FilterStrategy::Full,
    };
    
    runner.find_affected_tests(&paths, filter_strategy)
}

/// Register project
#[tauri::command]
fn register_project(
    project_id: String,
    project_path: String,
    language: String,
) -> Result<agent::multi_project::ProjectConfig, String> {
    use agent::multi_project::MultiProjectManager;
    use std::path::PathBuf;
    
    let projects_dir = PathBuf::from(".yantra/projects");
    let manager = MultiProjectManager::new(projects_dir);
    
    manager.register_project(&project_id, PathBuf::from(project_path), &language)
}

/// List all projects
#[tauri::command]
fn list_projects() -> Result<Vec<agent::multi_project::ProjectConfig>, String> {
    use agent::multi_project::MultiProjectManager;
    use std::path::PathBuf;
    
    let projects_dir = PathBuf::from(".yantra/projects");
    let manager = MultiProjectManager::new(projects_dir);
    
    manager.list_projects()
}

/// Activate project
#[tauri::command]
fn activate_project(project_id: String) -> Result<agent::multi_project::ProjectEnvironment, String> {
    use agent::multi_project::MultiProjectManager;
    use std::path::PathBuf;
    
    let projects_dir = PathBuf::from(".yantra/projects");
    let manager = MultiProjectManager::new(projects_dir);
    
    manager.activate_project(&project_id)
}

/// Check project conflicts
#[tauri::command]
fn check_project_conflicts(project1: String, project2: String) -> Result<Vec<String>, String> {
    use agent::multi_project::MultiProjectManager;
    use std::path::PathBuf;
    
    let projects_dir = PathBuf::from(".yantra/projects");
    let manager = MultiProjectManager::new(projects_dir);
    
    manager.check_conflicts(&project1, &project2)
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
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    let status = git_mcp.status()?;
    serde_json::to_string(&status).map_err(|e| e.to_string())
}

/// Add files to git staging
#[tauri::command]
fn git_add(workspace_path: String, files: Vec<String>) -> Result<(), String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.add_files(&files)
}

/// Commit staged changes
#[tauri::command]
fn git_commit(workspace_path: String, message: String) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.commit(&message)
}

/// Get git diff
#[tauri::command]
fn git_diff(workspace_path: String, file: Option<String>) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.diff(file.as_deref())
}

/// Get git log
#[tauri::command]
fn git_log(workspace_path: String, max_count: usize) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    let commits = git_mcp.log(max_count)?;
    serde_json::to_string(&commits).map_err(|e| e.to_string())
}

/// List git branches
#[tauri::command]
fn git_branch_list(workspace_path: String) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    let branches = git_mcp.branch_list()?;
    serde_json::to_string(&branches).map_err(|e| e.to_string())
}

/// Get current branch
#[tauri::command]
fn git_current_branch(workspace_path: String) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.current_branch()
}

/// Checkout branch
#[tauri::command]
fn git_checkout(workspace_path: String, branch: String) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.checkout(&branch)
}

/// Git pull
#[tauri::command]
fn git_pull(workspace_path: String) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
    git_mcp.pull()
}

/// Git push
#[tauri::command]
fn git_push(workspace_path: String) -> Result<String, String> {
    let mut git_mcp = git::GitMcp::new(PathBuf::from(workspace_path));
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
                            gnn::NodeType::Package { .. } => "package",
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

// File Watcher Commands

/// Start file system watcher for automatic graph synchronization
#[tauri::command]
async fn start_file_watcher(
    workspace_path: String,
    gnn_state: State<'_, Arc<TokioMutex<gnn::GNNEngine>>>,
    watcher_state: State<'_, FileWatcherState>,
) -> Result<String, String> {
    // Check if watcher is already running
    {
        let current_watcher = watcher_state.watcher.lock().await;
        if current_watcher.is_some() {
            return Err("File watcher is already running. Stop it first.".to_string());
        }
    }
    
    let graph = Arc::clone(&gnn_state.inner());
    let workspace = PathBuf::from(&workspace_path);
    
    let mut watcher = gnn::file_watcher::FileWatcher::new(workspace.clone(), graph)
        .map_err(|e| format!("Failed to create file watcher: {}", e))?;
    
    watcher.start()
        .map_err(|e| format!("Failed to start file watcher: {}", e))?;
    
    // Store the watcher in state so it stays alive
    let mut state_watcher = watcher_state.watcher.lock().await;
    *state_watcher = Some(watcher);
    
    Ok(format!("File watcher started for: {}", workspace_path))
}

/// Stop file system watcher
#[tauri::command]
async fn stop_file_watcher(
    watcher_state: State<'_, FileWatcherState>,
) -> Result<String, String> {
    let mut state_watcher = watcher_state.watcher.lock().await;
    
    if let Some(mut watcher) = state_watcher.take() {
        watcher.stop();
        Ok("File watcher stopped successfully".to_string())
    } else {
        Err("File watcher is not running".to_string())
    }
}

/// Get file watcher status
#[tauri::command]
async fn get_file_watcher_status(
    watcher_state: State<'_, FileWatcherState>,
) -> Result<bool, String> {
    let state_watcher = watcher_state.watcher.lock().await;
    Ok(state_watcher.is_some())
}

// Code Validation Commands

/// Validate code file for syntax, type, and import errors
#[tauri::command]
async fn validate_code_file(
    file_path: String,
    workspace_path: String,
) -> Result<agent::CodeValidationResult, String> {
    let validator = agent::CodeValidator::new(PathBuf::from(workspace_path));
    validator.validate_file(&PathBuf::from(file_path))
        .await
        .map_err(|e| format!("Code validation failed: {}", e))
}

// Conversation Memory Commands

/// Create a new conversation
#[tauri::command]
async fn create_conversation(
    app_handle: tauri::AppHandle,
    initial_title: Option<String>,
) -> Result<agent::Conversation, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    memory.create_conversation(initial_title)
        .map_err(|e| format!("Failed to create conversation: {}", e))
}

/// Save a message to a conversation
#[tauri::command]
async fn save_message(
    app_handle: tauri::AppHandle,
    conversation_id: String,
    role: String,
    content: String,
    parent_message_id: Option<String>,
    tokens: Option<usize>,
    metadata: Option<String>,
) -> Result<agent::Message, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let message_role = match role.to_lowercase().as_str() {
        "user" => agent::MessageRole::User,
        "assistant" => agent::MessageRole::Assistant,
        "system" => agent::MessageRole::System,
        _ => return Err(format!("Invalid message role: {}", role)),
    };
    
    let metadata_json = metadata.and_then(|m| serde_json::from_str(&m).ok());
    
    memory.save_message(
        &conversation_id,
        message_role,
        content,
        tokens.unwrap_or(0),
        parent_message_id,
        metadata_json,
    ).map_err(|e| format!("Failed to save message: {}", e))
}

/// Load a conversation with its messages
#[tauri::command]
async fn load_conversation(
    app_handle: tauri::AppHandle,
    conversation_id: String,
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<(agent::Conversation, Vec<agent::Message>), String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    // load_conversation only takes conversation_id
    let conversation = memory.load_conversation(&conversation_id)
        .map_err(|e| format!("Failed to load conversation: {}", e))?;
    
    // Use load_recent_messages for paginated message retrieval
    let messages = memory.load_recent_messages(&conversation_id, limit.unwrap_or(50))
        .map_err(|e| format!("Failed to load messages: {}", e))?;
    
    Ok((conversation, messages))
}

/// Get the last active conversation
#[tauri::command]
async fn get_last_active_conversation(
    app_handle: tauri::AppHandle,
) -> Result<Option<agent::Conversation>, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    memory.get_last_active_conversation()
        .map_err(|e| format!("Failed to get last active conversation: {}", e))
}

/// Load recent messages from a conversation
#[tauri::command]
async fn load_recent_messages(
    app_handle: tauri::AppHandle,
    conversation_id: String,
    count: usize,
) -> Result<Vec<agent::Message>, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    memory.load_recent_messages(&conversation_id, count)
        .map_err(|e| format!("Failed to load recent messages: {}", e))
}

/// Search conversations by keyword, date range, tags, or session type
#[tauri::command]
async fn search_conversations(
    app_handle: tauri::AppHandle,
    keyword: Option<String>,
    start_date: Option<String>,
    end_date: Option<String>,
    tags: Option<Vec<String>>,
    session_type: Option<String>,
) -> Result<Vec<agent::Conversation>, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let session_type_enum = match session_type {
        Some(st) => Some(match st.to_lowercase().as_str() {
            "codegeneration" => agent::SessionType::CodeGeneration,
            "testing" => agent::SessionType::Testing,
            "deployment" => agent::SessionType::Deployment,
            "documentation" => agent::SessionType::Documentation,
            _ => return Err(format!("Invalid session type: {}", st)),
        }),
        None => None,
    };
    
    // Parse date strings to DateTime<Utc>
    let parsed_start_date = start_date
        .and_then(|d| chrono::DateTime::parse_from_rfc3339(&d).ok())
        .map(|d| d.with_timezone(&chrono::Utc));
    
    let parsed_end_date = end_date
        .and_then(|d| chrono::DateTime::parse_from_rfc3339(&d).ok())
        .map(|d| d.with_timezone(&chrono::Utc));
    
    let filter = agent::SearchFilter {
        keyword,
        start_date: parsed_start_date,
        end_date: parsed_end_date,
        tags: tags.unwrap_or_else(Vec::new),
        session_type: session_type_enum,
    };
    
    memory.search_conversations(&filter)
        .map_err(|e| format!("Failed to search conversations: {}", e))
}

/// Link a conversation to a session (code generation, testing, etc.)
#[tauri::command]
async fn link_to_session(
    app_handle: tauri::AppHandle,
    conversation_id: String,
    message_id: String,
    session_type: String,
    session_id: String,
    metadata: Option<String>,
) -> Result<agent::SessionLink, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let session_type_enum = match session_type.to_lowercase().as_str() {
        "codegeneration" => agent::SessionType::CodeGeneration,
        "testing" => agent::SessionType::Testing,
        "deployment" => agent::SessionType::Deployment,
        "documentation" => agent::SessionType::Documentation,
        _ => return Err(format!("Invalid session type: {}", session_type)),
    };
    
    let metadata_json = metadata.and_then(|m| serde_json::from_str(&m).ok());
    
    memory.link_to_session(
        &conversation_id,
        &message_id,
        session_type_enum,
        &session_id,
        metadata_json,
    ).map_err(|e| format!("Failed to link to session: {}", e))
}

/// Get session links for a conversation
#[tauri::command]
async fn get_session_links(
    app_handle: tauri::AppHandle,
    conversation_id: String,
) -> Result<Vec<agent::SessionLink>, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    memory.get_session_links(&conversation_id)
        .map_err(|e| format!("Failed to get session links: {}", e))
}

/// Export conversation to file format (markdown, json, plaintext)
#[tauri::command]
async fn export_conversation(
    app_handle: tauri::AppHandle,
    conversation_id: String,
    format: String,
    output_path: String,
) -> Result<String, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let export_format = match format.to_lowercase().as_str() {
        "markdown" | "md" => agent::ExportFormat::Markdown,
        "json" => agent::ExportFormat::Json,
        "plaintext" | "txt" => agent::ExportFormat::PlainText,
        _ => return Err(format!("Invalid export format: {}", format)),
    };
    
    let content = memory.export_conversation(&conversation_id, export_format)
        .map_err(|e| format!("Failed to export conversation: {}", e))?;
    
    // Write content to output path
    std::fs::write(&output_path, content)
        .map_err(|e| format!("Failed to write export file: {}", e))?;
    
    Ok(output_path)
}

/// Build semantic search index for conversations
#[tauri::command]
async fn build_semantic_search_index(
    app_handle: tauri::AppHandle,
) -> Result<usize, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let search = agent::ConversationSemanticSearch::new()
        .map_err(|e| format!("Failed to create semantic search: {}", e))?;
    
    search.build_index(&memory)
        .map_err(|e| format!("Failed to build index: {}", e))
}

/// Search conversations using semantic similarity
#[tauri::command]
async fn semantic_search_conversations(
    app_handle: tauri::AppHandle,
    query: String,
    top_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let search = agent::ConversationSemanticSearch::new()
        .map_err(|e| format!("Failed to create semantic search: {}", e))?;
    
    // Build index if not already built
    let _ = search.build_index(&memory);
    
    search.search(&query, top_k)
        .map_err(|e| format!("Semantic search failed: {}", e))
}

/// Hybrid search combining keyword and semantic matching
#[tauri::command]
async fn hybrid_search_conversations(
    app_handle: tauri::AppHandle,
    query: String,
    keyword_weight: f32,
    semantic_weight: f32,
    top_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db_path = app_handle.path_resolver()
        .app_data_dir()
        .ok_or_else(|| "Failed to get app data directory".to_string())?
        .join("conversations.db");
    
    let memory = agent::ConversationMemory::new(&db_path)
        .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
    
    let search = agent::ConversationSemanticSearch::new()
        .map_err(|e| format!("Failed to create semantic search: {}", e))?;
    
    // Build index if not already built
    let _ = search.build_index(&memory);
    
    search.hybrid_search(&memory, &query, keyword_weight, semantic_weight, top_k)
        .map_err(|e| format!("Hybrid search failed: {}", e))
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
        secondary_provider: None,
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

    // Initialize YDoc state
    let ydoc_state = YDocState::new();
    
    // Initialize file watcher state
    let file_watcher_state = FileWatcherState::new();

    tauri::Builder::default()
        .menu(menu)
        .manage(terminal_manager)
        .manage(arch_state)
        .manage(db_manager)
        .manage(status_emitter)
        .manage(ydoc_state)
        .manage(file_watcher_state)
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
            parse_ast,
            parse_ast_snippet,
            get_symbols,
            get_references,
            find_all_references,
            get_definition,
            get_scope,
            get_all_scopes,
            get_code_completions,
            get_llm_config,
            set_llm_provider,
            set_secondary_llm_provider,
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
            // Database migrations
            db_create_migration,
            db_get_migrations,
            db_migrate_up,
            db_migrate_down,
            // API health & rate limits
            api_health_check,
            api_get_uptime,
            api_record_request,
            api_get_rate_limit,
            // Environment snapshot & validation
            env_create_snapshot,
            env_rollback,
            env_validate,
            // Secrets management
            secrets_set,
            secrets_get,
            secrets_delete,
            secrets_list,
            secrets_metadata,
            // Document readers
            read_docx,
            read_pdf,
            // Affected tests
            find_affected_tests,
            // Multi-project isolation
            register_project,
            list_projects,
            activate_project,
            check_project_conflicts,
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
            // API management commands
            api_import_spec,
            api_validate_contract,
            // Dependency & environment commands
            validate_dependencies,
            get_venv_info,
            create_venv,
            enforce_venv,
            // Conflict detection
            detect_conflicts,
            // Intelligent execution
            intelligent_execute,
            execute_background,
            // GNN version tracking
            gnn_track_version,
            gnn_get_history,
            gnn_rollback,
            gnn_diff,
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
            // YDoc commands
            ydoc_commands::ydoc_initialize,
            ydoc_commands::ydoc_create_document,
            ydoc_commands::ydoc_create_block,
            ydoc_commands::ydoc_create_edge,
            ydoc_commands::ydoc_load_document,
            ydoc_commands::ydoc_list_documents,
            ydoc_commands::ydoc_get_document_metadata,
            ydoc_commands::ydoc_search_blocks,
            ydoc_commands::ydoc_get_traceability_chain,
            ydoc_commands::ydoc_get_coverage_stats,
            ydoc_commands::ydoc_export_to_markdown,
            ydoc_commands::ydoc_export_to_html,
            ydoc_commands::ydoc_delete_document,
            ydoc_commands::ydoc_archive_old_test_results,
            ydoc_commands::ydoc_get_archived_test_results,
            ydoc_commands::ydoc_cleanup_archive,
            // Architecture View commands
            arch_commands::generate_architecture_from_intent,
            arch_commands::generate_architecture_from_code,
            arch_commands::initialize_new_project,
            arch_commands::initialize_existing_project,
            arch_commands::review_existing_code,
            arch_commands::analyze_requirement_impact,
            arch_commands::is_project_initialized,
            // File watcher commands
            start_file_watcher,
            stop_file_watcher,
            get_file_watcher_status,
            // Code validation commands
            validate_code_file,
            // Conversation memory commands
            create_conversation,
            save_message,
            load_conversation,
            get_last_active_conversation,
            load_recent_messages,
            search_conversations,
            link_to_session,
            get_session_links,
            export_conversation,
            // Semantic search commands
            build_semantic_search_index,
            semantic_search_conversations,
            hybrid_search_conversations,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
