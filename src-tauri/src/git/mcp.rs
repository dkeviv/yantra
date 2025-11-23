// Model Context Protocol for Git
// Provides Git operations via MCP

use std::path::PathBuf;
use std::process::Command;

pub struct GitMcp {
    workspace_path: PathBuf,
}

impl GitMcp {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }

    pub fn status(&self) -> Result<String, String> {
        let output = Command::new("git")
            .arg("status")
            .arg("--porcelain")
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to get git status: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn add_files(&self, files: &[String]) -> Result<(), String> {
        for file in files {
            Command::new("git")
                .arg("add")
                .arg(file)
                .current_dir(&self.workspace_path)
                .output()
                .map_err(|e| format!("Failed to add file {}: {}", file, e))?;
        }
        Ok(())
    }

    pub fn commit(&self, message: &str) -> Result<String, String> {
        let output = Command::new("git")
            .arg("commit")
            .arg("-m")
            .arg(message)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to commit: {}", e))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn has_changes(&self) -> bool {
        if let Ok(status) = self.status() {
            !status.trim().is_empty()
        } else {
            false
        }
    }

    pub fn diff(&self, file: Option<&str>) -> Result<String, String> {
        let mut cmd = Command::new("git");
        cmd.arg("diff");
        
        if let Some(f) = file {
            cmd.arg(f);
        }
        
        let output = cmd
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to get git diff: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn log(&self, max_count: usize) -> Result<String, String> {
        let output = Command::new("git")
            .arg("log")
            .arg(format!("--max-count={}", max_count))
            .arg("--oneline")
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to get git log: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn branch_list(&self) -> Result<String, String> {
        let output = Command::new("git")
            .arg("branch")
            .arg("-a")
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to list branches: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn current_branch(&self) -> Result<String, String> {
        let output = Command::new("git")
            .arg("branch")
            .arg("--show-current")
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to get current branch: {}", e))?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub fn checkout(&self, branch: &str) -> Result<String, String> {
        let output = Command::new("git")
            .arg("checkout")
            .arg(branch)
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to checkout branch: {}", e))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn pull(&self) -> Result<String, String> {
        let output = Command::new("git")
            .arg("pull")
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to pull: {}", e))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn push(&self) -> Result<String, String> {
        let output = Command::new("git")
            .arg("push")
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to push: {}", e))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).to_string());
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
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
}
