// Commit manager - Generates commit messages and handles commits

use crate::git::mcp::GitMcp;
use std::path::PathBuf;

pub struct CommitResult {
    pub success: bool,
    pub commit_hash: Option<String>,
    pub message: String,
}

pub struct CommitManager {
    git_mcp: GitMcp,
}

impl CommitManager {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self {
            git_mcp: GitMcp::new(workspace_path),
        }
    }

    pub async fn auto_commit(&mut self, changes: &[String]) -> Result<CommitResult, String> {
        // Generate commit message based on changes
        let message = self.generate_commit_message(changes).await?;

        // Add files
        self.git_mcp.add_files(changes)?;

        // Commit
        let _output = self.git_mcp.commit(&message)?;

        Ok(CommitResult {
            success: true,
            commit_hash: Some("abc123".to_string()), // Parse from git output
            message,
        })
    }

    async fn generate_commit_message(&self, changes: &[String]) -> Result<String, String> {
        // In production: Use LLM to generate semantic commit message
        // For MVP: Generate simple message
        let file_count = changes.len();
        Ok(format!(
            "feat: Update {} file{}", 
            file_count,
            if file_count > 1 { "s" } else { "" }
        ))
    }

    pub fn has_changes(&mut self) -> bool {
        self.git_mcp.has_changes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_commit_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CommitManager::new(temp_dir.path().to_path_buf());
        assert!(!manager.has_changes());
    }
}
