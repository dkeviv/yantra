// File: src-tauri/src/agent/file_ops.rs
// Purpose: Additional file operations (delete, move, search, directory tree)
// Dependencies: std::fs, walkdir, glob
// Last Updated: December 3, 2025

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDeleteRequest {
    pub file_path: String,
    pub check_dependencies: bool,
    pub create_backup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDeleteResult {
    pub success: bool,
    pub message: String,
    pub backup_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMoveRequest {
    pub from_path: String,
    pub to_path: String,
    pub update_imports: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMoveResult {
    pub success: bool,
    pub message: String,
    pub imports_updated: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryTreeNode {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
    pub size: Option<u64>,
    pub children: Option<Vec<DirectoryTreeNode>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchRequest {
    pub root_path: String,
    pub pattern: String,
    pub is_regex: bool,
    pub search_content: bool,
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchResult {
    pub path: String,
    pub matches: Vec<SearchMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
}

/// Delete a file safely with optional backup
pub fn delete_file(request: FileDeleteRequest) -> Result<FileDeleteResult, String> {
    let path = Path::new(&request.file_path);
    
    if !path.exists() {
        return Err(format!("File not found: {}", request.file_path));
    }

    // Create backup if requested
    let backup_path = if request.create_backup {
        let backup = format!("{}.backup", request.file_path);
        fs::copy(&request.file_path, &backup)
            .map_err(|e| format!("Failed to create backup: {}", e))?;
        Some(backup)
    } else {
        None
    };

    // Delete the file
    fs::remove_file(&request.file_path)
        .map_err(|e| format!("Failed to delete file: {}", e))?;

    Ok(FileDeleteResult {
        success: true,
        message: format!("File deleted: {}", request.file_path),
        backup_path,
    })
}

/// Move/rename a file
pub fn move_file(request: FileMoveRequest) -> Result<FileMoveResult, String> {
    let from_path = Path::new(&request.from_path);
    let to_path = Path::new(&request.to_path);

    if !from_path.exists() {
        return Err(format!("Source file not found: {}", request.from_path));
    }

    // Create destination directory if it doesn't exist
    if let Some(parent) = to_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create destination directory: {}", e))?;
    }

    // Move the file
    fs::rename(&request.from_path, &request.to_path)
        .map_err(|e| format!("Failed to move file: {}", e))?;

    // TODO: Update imports in other files (requires GNN integration)
    let imports_updated = 0;

    Ok(FileMoveResult {
        success: true,
        message: format!("File moved: {} -> {}", request.from_path, request.to_path),
        imports_updated,
    })
}

/// Get directory tree structure
pub fn get_directory_tree(root_path: String, max_depth: Option<usize>) -> Result<DirectoryTreeNode, String> {
    let path = Path::new(&root_path);
    
    if !path.exists() {
        return Err(format!("Path not found: {}", root_path));
    }

    build_tree_node(path, 0, max_depth.unwrap_or(10))
}

fn build_tree_node(path: &Path, depth: usize, max_depth: usize) -> Result<DirectoryTreeNode, String> {
    let metadata = fs::metadata(path)
        .map_err(|e| format!("Failed to read metadata: {}", e))?;
    
    let is_directory = metadata.is_dir();
    let size = if is_directory { None } else { Some(metadata.len()) };
    
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    
    let path_str = path.to_string_lossy().to_string();

    let children = if is_directory && depth < max_depth {
        let mut child_nodes = Vec::new();
        
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                // Skip hidden files and common ignore patterns
                let entry_name = entry.file_name();
                let entry_name_str = entry_name.to_string_lossy();
                
                if entry_name_str.starts_with('.') 
                    || entry_name_str == "node_modules"
                    || entry_name_str == "target"
                    || entry_name_str == "__pycache__" {
                    continue;
                }
                
                if let Ok(child) = build_tree_node(&entry.path(), depth + 1, max_depth) {
                    child_nodes.push(child);
                }
            }
        }
        
        Some(child_nodes)
    } else {
        None
    };

    Ok(DirectoryTreeNode {
        name,
        path: path_str,
        is_directory,
        size,
        children,
    })
}

/// Search for files by pattern
pub fn search_files(request: FileSearchRequest) -> Result<Vec<FileSearchResult>, String> {
    let root_path = Path::new(&request.root_path);
    
    if !root_path.exists() {
        return Err(format!("Root path not found: {}", request.root_path));
    }

    let mut results = Vec::new();
    let max_results = request.max_results.unwrap_or(1000);

    if request.is_regex {
        search_with_regex(&request, &mut results, max_results)?;
    } else {
        search_with_glob(&request, &mut results, max_results)?;
    }

    Ok(results)
}

fn search_with_glob(request: &FileSearchRequest, results: &mut Vec<FileSearchResult>, max_results: usize) -> Result<(), String> {
    let pattern = if request.pattern.contains('/') {
        request.pattern.clone()
    } else {
        format!("**/{}", request.pattern)
    };

    let search_pattern = format!("{}/{}", request.root_path, pattern);
    
    for entry in glob::glob(&search_pattern)
        .map_err(|e| format!("Invalid glob pattern: {}", e))?
        .flatten()
        .take(max_results)
    {
        if results.len() >= max_results {
            break;
        }

        let path_str = entry.to_string_lossy().to_string();
        
        if request.search_content && entry.is_file() {
            if let Ok(content) = fs::read_to_string(&entry) {
                let matches = find_content_matches(&content, &request.pattern);
                if !matches.is_empty() {
                    results.push(FileSearchResult {
                        path: path_str,
                        matches,
                    });
                }
            }
        } else {
            results.push(FileSearchResult {
                path: path_str,
                matches: vec![],
            });
        }
    }

    Ok(())
}

fn search_with_regex(request: &FileSearchRequest, results: &mut Vec<FileSearchResult>, max_results: usize) -> Result<(), String> {
    use regex::Regex;
    
    let re = Regex::new(&request.pattern)
        .map_err(|e| format!("Invalid regex pattern: {}", e))?;

    for entry in WalkDir::new(&request.root_path)
        .max_depth(10)
        .into_iter()
        .flatten()
        .take(max_results)
    {
        if results.len() >= max_results {
            break;
        }

        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();

        if request.search_content && path.is_file() {
            if let Ok(content) = fs::read_to_string(path) {
                let matches = find_regex_matches(&content, &re);
                if !matches.is_empty() {
                    results.push(FileSearchResult {
                        path: path_str,
                        matches,
                    });
                }
            }
        } else if re.is_match(&path_str) {
            results.push(FileSearchResult {
                path: path_str,
                matches: vec![],
            });
        }
    }

    Ok(())
}

fn find_content_matches(content: &str, pattern: &str) -> Vec<SearchMatch> {
    let mut matches = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        if let Some(pos) = line.find(pattern) {
            matches.push(SearchMatch {
                line_number: line_num + 1,
                line_content: line.to_string(),
                match_start: pos,
                match_end: pos + pattern.len(),
            });
        }
    }
    
    matches
}

fn find_regex_matches(content: &str, re: &regex::Regex) -> Vec<SearchMatch> {
    let mut matches = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        for mat in re.find_iter(line) {
            matches.push(SearchMatch {
                line_number: line_num + 1,
                line_content: line.to_string(),
                match_start: mat.start(),
                match_end: mat.end(),
            });
        }
    }
    
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_delete_file_with_backup() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "test content").unwrap();

        let result = delete_file(FileDeleteRequest {
            file_path: file_path.to_string_lossy().to_string(),
            check_dependencies: false,
            create_backup: true,
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.backup_path.is_some());
        assert!(!file_path.exists());
    }

    #[test]
    fn test_move_file() {
        let temp_dir = TempDir::new().unwrap();
        let from_path = temp_dir.path().join("from.txt");
        let to_path = temp_dir.path().join("to.txt");
        fs::write(&from_path, "test content").unwrap();

        let result = move_file(FileMoveRequest {
            from_path: from_path.to_string_lossy().to_string(),
            to_path: to_path.to_string_lossy().to_string(),
            update_imports: false,
        });

        assert!(result.is_ok());
        assert!(!from_path.exists());
        assert!(to_path.exists());
    }

    #[test]
    fn test_directory_tree() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir_all(temp_dir.path().join("subdir")).unwrap();
        fs::write(temp_dir.path().join("file1.txt"), "content").unwrap();
        fs::write(temp_dir.path().join("subdir/file2.txt"), "content").unwrap();

        let result = get_directory_tree(temp_dir.path().to_string_lossy().to_string(), Some(2));
        assert!(result.is_ok());
        
        let tree = result.unwrap();
        assert!(tree.is_directory);
        assert!(tree.children.is_some());
    }
}
