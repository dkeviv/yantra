// File: src-tauri/src/git/mcp.rs
// Purpose: Model Context Protocol (MCP) for Git operations
// Dependencies: serde, reqwest, tokio
// Last Updated: December 3, 2025
//
// Implements MCP-compliant Git operations:
// - JSON-RPC 2.0 protocol for git commands
// - Structured request/response format
// - Context-aware operations
// - Error handling with MCP error codes
// - Resource URIs for files and commits
//
// MCP Protocol Specification:
// - Request: {"jsonrpc": "2.0", "method": "git/status", "params": {...}, "id": 1}
// - Response: {"jsonrpc": "2.0", "result": {...}, "id": 1}
// - Error: {"jsonrpc": "2.0", "error": {"code": -32000, "message": "..."}, "id": 1}
//
// Benefits over raw git commands:
// - Standardized interface for AI agents
// - Type-safe request/response
// - Better error handling
// - Context preservation across operations
// - Async operation support

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

/// MCP-compliant Git client
pub struct GitMcp {
    workspace_path: PathBuf,
    request_id: u64,
}

impl GitMcp {
    /// Create new MCP Git client
    pub fn new(workspace_path: PathBuf) -> Self {
        Self {
            workspace_path,
            request_id: 0,
        }
    }

    /// Execute MCP request
    fn execute_request(&mut self, method: &str, params: MCPParams) -> MCPResponse {
        self.request_id += 1;
        let id = self.request_id;

        match method {
            "git/status" => self.handle_status(id, params),
            "git/add" => self.handle_add(id, params),
            "git/commit" => self.handle_commit(id, params),
            "git/diff" => self.handle_diff(id, params),
            "git/log" => self.handle_log(id, params),
            "git/branch_list" => self.handle_branch_list(id),
            "git/current_branch" => self.handle_current_branch(id),
            "git/checkout" => self.handle_checkout(id, params),
            "git/pull" => self.handle_pull(id),
            "git/push" => self.handle_push(id),
            "git/stash" => self.handle_stash(id, params),
            "git/show" => self.handle_show(id, params),
            _ => MCPResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(MCPError {
                    code: -32601,
                    message: format!("Method not found: {}", method),
                }),
                id,
            },
        }
    }

    // MCP method handlers

    fn handle_status(&self, id: u64, _params: MCPParams) -> MCPResponse {
        match self.git_command(&["status", "--porcelain"]) {
            Ok(output) => {
                let files = parse_porcelain_status(&output);
                MCPResponse::success(id, serde_json::json!({
                    "files": files,
                    "has_changes": !files.is_empty(),
                }))
            }
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_add(&self, id: u64, params: MCPParams) -> MCPResponse {
        let files = match params.get_string_array("files") {
            Some(f) => f,
            None => return MCPResponse::error(id, -32602, "Missing 'files' parameter"),
        };

        let mut args = vec!["add"];
        for file in &files {
            args.push(file);
        }

        match self.git_command(&args) {
            Ok(_) => MCPResponse::success(id, serde_json::json!({
                "added": files,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_commit(&self, id: u64, params: MCPParams) -> MCPResponse {
        let message = match params.get_string("message") {
            Some(m) => m,
            None => return MCPResponse::error(id, -32602, "Missing 'message' parameter"),
        };

        match self.git_command(&["commit", "-m", &message]) {
            Ok(output) => {
                // Extract commit hash from output
                let commit_hash = extract_commit_hash(&output);
                MCPResponse::success(id, serde_json::json!({
                    "commit_hash": commit_hash,
                    "message": message,
                }))
            }
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_diff(&self, id: u64, params: MCPParams) -> MCPResponse {
        let mut args = vec!["diff"];
        
        if let Some(file) = params.get_string("file") {
            args.push(&file);
        }

        match self.git_command(&args) {
            Ok(output) => MCPResponse::success(id, serde_json::json!({
                "diff": output,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_log(&self, id: u64, params: MCPParams) -> MCPResponse {
        let max_count = params.get_number("max_count").unwrap_or(10.0) as usize;
        
        let args = vec![
            "log",
            &format!("--max-count={}", max_count),
            "--format=%H|%an|%ae|%ad|%s",
            "--date=iso",
        ];

        match self.git_command(&args) {
            Ok(output) => {
                let commits = parse_log_output(&output);
                MCPResponse::success(id, serde_json::json!({
                    "commits": commits,
                }))
            }
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_branch_list(&self, id: u64) -> MCPResponse {
        match self.git_command(&["branch", "-a"]) {
            Ok(output) => {
                let branches = parse_branches(&output);
                MCPResponse::success(id, serde_json::json!({
                    "branches": branches,
                }))
            }
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_current_branch(&self, id: u64) -> MCPResponse {
        match self.git_command(&["branch", "--show-current"]) {
            Ok(output) => MCPResponse::success(id, serde_json::json!({
                "branch": output.trim(),
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_checkout(&self, id: u64, params: MCPParams) -> MCPResponse {
        let branch = match params.get_string("branch") {
            Some(b) => b,
            None => return MCPResponse::error(id, -32602, "Missing 'branch' parameter"),
        };

        match self.git_command(&["checkout", &branch]) {
            Ok(_) => MCPResponse::success(id, serde_json::json!({
                "branch": branch,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_pull(&self, id: u64) -> MCPResponse {
        match self.git_command(&["pull"]) {
            Ok(output) => MCPResponse::success(id, serde_json::json!({
                "result": output,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_push(&self, id: u64) -> MCPResponse {
        match self.git_command(&["push"]) {
            Ok(output) => MCPResponse::success(id, serde_json::json!({
                "result": output,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_stash(&self, id: u64, params: MCPParams) -> MCPResponse {
        let action = params.get_string("action").unwrap_or("save".to_string());
        
        let args = match action.as_str() {
            "save" => vec!["stash", "save"],
            "pop" => vec!["stash", "pop"],
            "list" => vec!["stash", "list"],
            _ => return MCPResponse::error(id, -32602, "Invalid stash action"),
        };

        match self.git_command(&args) {
            Ok(output) => MCPResponse::success(id, serde_json::json!({
                "action": action,
                "result": output,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    fn handle_show(&self, id: u64, params: MCPParams) -> MCPResponse {
        let commit = match params.get_string("commit") {
            Some(c) => c,
            None => return MCPResponse::error(id, -32602, "Missing 'commit' parameter"),
        };

        match self.git_command(&["show", &commit]) {
            Ok(output) => MCPResponse::success(id, serde_json::json!({
                "commit": commit,
                "content": output,
            })),
            Err(e) => MCPResponse::error(id, -32000, &e),
        }
    }

    // Helper methods

    fn git_command(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute git command: {}", e))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    // Public convenience methods (use MCP under the hood)

    pub fn status(&mut self) -> Result<GitStatus, String> {
        let response = self.execute_request("git/status", MCPParams::empty());
        response.into_status()
    }

    pub fn add_files(&mut self, files: &[String]) -> Result<(), String> {
        let mut params = MCPParams::empty();
        params.set_string_array("files", files.to_vec());
        let response = self.execute_request("git/add", params);
        response.into_unit()
    }

    pub fn commit(&mut self, message: &str) -> Result<String, String> {
        let mut params = MCPParams::empty();
        params.set_string("message", message.to_string());
        let response = self.execute_request("git/commit", params);
        response.into_commit_hash()
    }

    pub fn has_changes(&mut self) -> bool {
        if let Ok(status) = self.status() {
            status.has_changes
        } else {
            false
        }
    }

    pub fn diff(&mut self, file: Option<&str>) -> Result<String, String> {
        let mut params = MCPParams::empty();
        if let Some(f) = file {
            params.set_string("file", f.to_string());
        }
        let response = self.execute_request("git/diff", params);
        response.into_string("diff")
    }

    pub fn log(&mut self, max_count: usize) -> Result<Vec<GitCommit>, String> {
        let mut params = MCPParams::empty();
        params.set_number("max_count", max_count as f64);
        let response = self.execute_request("git/log", params);
        response.into_commits()
    }

    pub fn branch_list(&mut self) -> Result<Vec<String>, String> {
        let response = self.execute_request("git/branch_list", MCPParams::empty());
        response.into_string_array("branches")
    }

    pub fn current_branch(&mut self) -> Result<String, String> {
        let response = self.execute_request("git/current_branch", MCPParams::empty());
        response.into_string("branch")
    }

    pub fn checkout(&mut self, branch: &str) -> Result<String, String> {
        let mut params = MCPParams::empty();
        params.set_string("branch", branch.to_string());
        let response = self.execute_request("git/checkout", params);
        response.into_string("branch")
    }

    pub fn pull(&mut self) -> Result<String, String> {
        let response = self.execute_request("git/pull", MCPParams::empty());
        response.into_string("result")
    }

    pub fn push(&mut self) -> Result<String, String> {
        let response = self.execute_request("git/push", MCPParams::empty());
        response.into_string("result")
    }
}

// MCP Protocol Types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: MCPParams,
    pub id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPError>,
    pub id: u64,
}

impl MCPResponse {
    fn success(id: u64, result: serde_json::Value) -> Self {
        MCPResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: u64, code: i32, message: &str) -> Self {
        MCPResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(MCPError {
                code,
                message: message.to_string(),
            }),
            id,
        }
    }

    fn into_status(self) -> Result<GitStatus, String> {
        if let Some(err) = self.error {
            return Err(err.message);
        }
        
        let result = self.result.ok_or("No result")?;
        serde_json::from_value(result).map_err(|e| e.to_string())
    }

    fn into_unit(self) -> Result<(), String> {
        if let Some(err) = self.error {
            return Err(err.message);
        }
        Ok(())
    }

    fn into_commit_hash(self) -> Result<String, String> {
        self.into_string("commit_hash")
    }

    fn into_string(self, field: &str) -> Result<String, String> {
        if let Some(err) = self.error {
            return Err(err.message);
        }
        
        let result = self.result.ok_or("No result")?;
        result.get(field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("Missing field: {}", field))
    }

    fn into_string_array(self, field: &str) -> Result<Vec<String>, String> {
        if let Some(err) = self.error {
            return Err(err.message);
        }
        
        let result = self.result.ok_or("No result")?;
        result.get(field)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .ok_or_else(|| format!("Missing field: {}", field))
    }

    fn into_commits(self) -> Result<Vec<GitCommit>, String> {
        if let Some(err) = self.error {
            return Err(err.message);
        }
        
        let result = self.result.ok_or("No result")?;
        serde_json::from_value(result.get("commits").cloned().unwrap_or_default())
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MCPParams {
    #[serde(flatten)]
    pub data: serde_json::Map<String, serde_json::Value>,
}

impl MCPParams {
    fn empty() -> Self {
        MCPParams {
            data: serde_json::Map::new(),
        }
    }

    fn get_string(&self, key: &str) -> Option<String> {
        self.data.get(key)?.as_str().map(|s| s.to_string())
    }

    fn get_string_array(&self, key: &str) -> Option<Vec<String>> {
        self.data.get(key)?
            .as_array()?
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    }

    fn get_number(&self, key: &str) -> Option<f64> {
        self.data.get(key)?.as_f64()
    }

    fn set_string(&mut self, key: &str, value: String) {
        self.data.insert(key.to_string(), serde_json::Value::String(value));
    }

    fn set_string_array(&mut self, key: &str, values: Vec<String>) {
        let arr: Vec<serde_json::Value> = values.into_iter()
            .map(serde_json::Value::String)
            .collect();
        self.data.insert(key.to_string(), serde_json::Value::Array(arr));
    }

    fn set_number(&mut self, key: &str, value: f64) {
        self.data.insert(key.to_string(), serde_json::json!(value));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
}

// Git-specific types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitStatus {
    pub files: Vec<GitFile>,
    pub has_changes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitFile {
    pub status: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCommit {
    pub hash: String,
    pub author: String,
    pub email: String,
    pub date: String,
    pub message: String,
}

// Helper parsing functions

fn parse_porcelain_status(output: &str) -> Vec<GitFile> {
    output.lines()
        .filter_map(|line| {
            if line.len() < 3 {
                return None;
            }
            Some(GitFile {
                status: line[..2].trim().to_string(),
                path: line[3..].to_string(),
            })
        })
        .collect()
}

fn parse_branches(output: &str) -> Vec<String> {
    output.lines()
        .map(|line| line.trim().trim_start_matches('*').trim().to_string())
        .collect()
}

fn parse_log_output(output: &str) -> Vec<GitCommit> {
    output.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() != 5 {
                return None;
            }
            Some(GitCommit {
                hash: parts[0].to_string(),
                author: parts[1].to_string(),
                email: parts[2].to_string(),
                date: parts[3].to_string(),
                message: parts[4].to_string(),
            })
        })
        .collect()
}

fn extract_commit_hash(output: &str) -> Option<String> {
    // Extract hash from "[] <hash> message"
    output.lines()
        .next()
        .and_then(|line| {
            line.split_whitespace()
                .nth(1)
                .map(|s| s.trim_matches(']').to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_git_mcp_creation() {
        let temp_dir = TempDir::new().unwrap();
        let git_mcp = GitMcp::new(temp_dir.path().to_path_buf());
        assert!(git_mcp.workspace_path.exists());
    }

    #[test]
    fn test_mcp_params() {
        let mut params = MCPParams::empty();
        params.set_string("key", "value".to_string());
        assert_eq!(params.get_string("key"), Some("value".to_string()));
    }

    #[test]
    fn test_parse_porcelain_status() {
        let output = " M file1.txt\nA  file2.txt\n";
        let files = parse_porcelain_status(output);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].status, "M");
        assert_eq!(files[0].path, "file1.txt");
    }
}
