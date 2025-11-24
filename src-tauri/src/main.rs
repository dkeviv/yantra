// File: src-tauri/src/main.rs
// Purpose: Tauri application entry point with file system operations and GNN
// Dependencies: tauri, serde_json, std::fs
// Last Updated: November 20, 2025

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tauri::{CustomMenuItem, Menu, MenuItem, Submenu};

mod gnn;
mod llm;
mod agent;
mod testing;
mod git;
mod documentation;

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
        "find" => {
            let _ = event.window().emit("menu-find", ());
        }
        "replace" => {
            let _ = event.window().emit("menu-replace", ());
        }
        _ => {}
    }
}

fn main() {
    // Build minimal custom menu
    let file_menu = Submenu::new(
        "File",
        Menu::new()
            .add_item(CustomMenuItem::new("new_file", "New File").accelerator("Cmd+N"))
            .add_item(CustomMenuItem::new("new_folder", "New Folder").accelerator("Cmd+Shift+N"))
            .add_item(CustomMenuItem::new("open_folder", "Open Folder...").accelerator("Cmd+O"))
            .add_native_item(MenuItem::Separator)
            .add_item(CustomMenuItem::new("save", "Save").accelerator("Cmd+S"))
            .add_item(CustomMenuItem::new("save_all", "Save All").accelerator("Cmd+Alt+S"))
            .add_native_item(MenuItem::Separator)
            .add_item(CustomMenuItem::new("close_folder", "Close Folder"))
            .add_item(CustomMenuItem::new("close_window", "Close Window").accelerator("Cmd+W"))
            .add_native_item(MenuItem::Separator)
            .add_native_item(MenuItem::Quit),
    );

    let edit_menu = Submenu::new(
        "Edit",
        Menu::new()
            .add_native_item(MenuItem::Copy)
            .add_native_item(MenuItem::Paste)
            .add_native_item(MenuItem::Separator)
            .add_item(CustomMenuItem::new("find", "Find").accelerator("Cmd+F"))
            .add_item(CustomMenuItem::new("replace", "Replace").accelerator("Cmd+H")),
    );

    let menu = Menu::new()
        .add_submenu(file_menu)
        .add_submenu(edit_menu);

    tauri::Builder::default()
        .menu(menu)
        .on_menu_event(|event| {
            handle_menu_event(event);
        })
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
            generate_tests,
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
            get_features,
            get_decisions,
            get_changes,
            get_tasks,
            add_feature,
            add_decision,
            add_change,
            extract_features_from_chat
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
