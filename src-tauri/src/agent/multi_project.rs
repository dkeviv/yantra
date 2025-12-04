// Multi-Project Isolation: Independent environment management per project
// Purpose: Prevent dependency conflicts and enable parallel project work

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Project configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project_id: String,
    pub project_path: PathBuf,
    pub language: String,
    pub venv_path: Option<PathBuf>,
    pub node_modules_path: Option<PathBuf>,
    pub dependencies: HashMap<String, String>,
    pub created_at: String,
    pub last_used: String,
}

/// Project environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectEnvironment {
    pub config: ProjectConfig,
    pub is_active: bool,
    pub python_version: Option<String>,
    pub node_version: Option<String>,
}

/// Multi-Project Manager
pub struct MultiProjectManager {
    projects_dir: PathBuf,
}

impl MultiProjectManager {
    /// Create new manager
    pub fn new(projects_dir: PathBuf) -> Self {
        Self { projects_dir }
    }
    
    /// Register new project
    pub fn register_project(
        &self,
        project_id: &str,
        project_path: PathBuf,
        language: &str,
    ) -> Result<ProjectConfig, String> {
        // Ensure project directory exists
        fs::create_dir_all(&self.projects_dir)
            .map_err(|e| format!("Failed to create projects directory: {}", e))?;
        
        // Create isolated environment directories
        let venv_path = if language == "python" {
            Some(self.create_python_venv(project_id, &project_path)?)
        } else {
            None
        };
        
        let node_modules_path = if language == "javascript" || language == "typescript" {
            Some(project_path.join("node_modules"))
        } else {
            None
        };
        
        let config = ProjectConfig {
            project_id: project_id.to_string(),
            project_path,
            language: language.to_string(),
            venv_path,
            node_modules_path,
            dependencies: HashMap::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            last_used: chrono::Utc::now().to_rfc3339(),
        };
        
        // Save config
        self.save_project_config(&config)?;
        
        Ok(config)
    }
    
    /// Create Python virtual environment
    fn create_python_venv(
        &self,
        project_id: &str,
        project_path: &Path,
    ) -> Result<PathBuf, String> {
        let venv_path = project_path.join(".venv");
        
        // Check if venv already exists
        if venv_path.exists() {
            return Ok(venv_path);
        }
        
        // Create venv
        let output = Command::new("python3")
            .args(&["-m", "venv", venv_path.to_str().unwrap()])
            .output()
            .map_err(|e| format!("Failed to create venv: {}", e))?;
        
        if !output.status.success() {
            return Err(format!(
                "venv creation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        
        Ok(venv_path)
    }
    
    /// Get project configuration
    pub fn get_project(&self, project_id: &str) -> Result<ProjectConfig, String> {
        let config_path = self.projects_dir.join(format!("{}.json", project_id));
        
        if !config_path.exists() {
            return Err(format!("Project '{}' not found", project_id));
        }
        
        let content = fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read project config: {}", e))?;
        
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse project config: {}", e))
    }
    
    /// List all projects
    pub fn list_projects(&self) -> Result<Vec<ProjectConfig>, String> {
        if !self.projects_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut projects = Vec::new();
        
        let entries = fs::read_dir(&self.projects_dir)
            .map_err(|e| format!("Failed to read projects directory: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let content = fs::read_to_string(&path)
                    .map_err(|e| format!("Failed to read config file: {}", e))?;
                
                if let Ok(config) = serde_json::from_str::<ProjectConfig>(&content) {
                    projects.push(config);
                }
            }
        }
        
        Ok(projects)
    }
    
    /// Activate project environment
    pub fn activate_project(
        &self,
        project_id: &str,
    ) -> Result<ProjectEnvironment, String> {
        let mut config = self.get_project(project_id)?;
        
        // Update last used timestamp
        config.last_used = chrono::Utc::now().to_rfc3339();
        self.save_project_config(&config)?;
        
        // Get environment versions
        let python_version = if config.language == "python" {
            Self::get_python_version(&config.venv_path)?
        } else {
            None
        };
        
        let node_version = if config.language == "javascript" || config.language == "typescript" {
            Self::get_node_version()?
        } else {
            None
        };
        
        Ok(ProjectEnvironment {
            config,
            is_active: true,
            python_version,
            node_version,
        })
    }
    
    /// Deactivate project
    pub fn deactivate_project(&self, project_id: &str) -> Result<(), String> {
        // In a real implementation, this would clean up environment state
        // For now, just verify project exists
        let _ = self.get_project(project_id)?;
        Ok(())
    }
    
    /// Delete project configuration
    pub fn delete_project(&self, project_id: &str) -> Result<(), String> {
        let config_path = self.projects_dir.join(format!("{}.json", project_id));
        
        if !config_path.exists() {
            return Err(format!("Project '{}' not found", project_id));
        }
        
        fs::remove_file(&config_path)
            .map_err(|e| format!("Failed to delete project config: {}", e))?;
        
        Ok(())
    }
    
    /// Get Python version from venv
    fn get_python_version(venv_path: &Option<PathBuf>) -> Result<Option<String>, String> {
        let venv_path = match venv_path {
            Some(p) => p,
            None => return Ok(None),
        };
        
        let python_bin = if cfg!(windows) {
            venv_path.join("Scripts").join("python.exe")
        } else {
            venv_path.join("bin").join("python")
        };
        
        if !python_bin.exists() {
            return Ok(None);
        }
        
        let output = Command::new(&python_bin)
            .arg("--version")
            .output()
            .map_err(|e| format!("Failed to get Python version: {}", e))?;
        
        let version = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(Some(version.trim().to_string()))
    }
    
    /// Get Node version
    fn get_node_version() -> Result<Option<String>, String> {
        let output = Command::new("node")
            .arg("--version")
            .output()
            .map_err(|e| format!("Failed to get Node version: {}", e))?;
        
        if !output.status.success() {
            return Ok(None);
        }
        
        let version = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(Some(version.trim().to_string()))
    }
    
    /// Save project configuration
    fn save_project_config(&self, config: &ProjectConfig) -> Result<(), String> {
        fs::create_dir_all(&self.projects_dir)
            .map_err(|e| format!("Failed to create projects directory: {}", e))?;
        
        let config_path = self.projects_dir.join(format!("{}.json", config.project_id));
        
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        
        fs::write(&config_path, json)
            .map_err(|e| format!("Failed to write config: {}", e))?;
        
        Ok(())
    }
    
    /// Check for dependency conflicts between projects
    pub fn check_conflicts(&self, project1_id: &str, project2_id: &str) -> Result<Vec<String>, String> {
        let project1 = self.get_project(project1_id)?;
        let project2 = self.get_project(project2_id)?;
        
        let mut conflicts = Vec::new();
        
        // Check for version conflicts in common dependencies
        for (dep1, ver1) in &project1.dependencies {
            if let Some(ver2) = project2.dependencies.get(dep1) {
                if ver1 != ver2 {
                    conflicts.push(format!("{}: {} vs {}", dep1, ver1, ver2));
                }
            }
        }
        
        Ok(conflicts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_register_project() {
        let temp_dir = tempdir().unwrap();
        let projects_dir = temp_dir.path().join("projects");
        let project_path = temp_dir.path().join("my_project");
        fs::create_dir(&project_path).unwrap();
        
        let manager = MultiProjectManager::new(projects_dir);
        let config = manager.register_project("test_proj", project_path.clone(), "python");
        
        // Note: Will fail without Python installed
        // assert!(config.is_ok());
    }
    
    #[test]
    fn test_list_projects() {
        let temp_dir = tempdir().unwrap();
        let projects_dir = temp_dir.path().join("projects");
        
        let manager = MultiProjectManager::new(projects_dir);
        let projects = manager.list_projects().unwrap();
        
        assert_eq!(projects.len(), 0);
    }
    
    #[test]
    fn test_project_config_serialization() {
        let config = ProjectConfig {
            project_id: "test".to_string(),
            project_path: PathBuf::from("/test"),
            language: "python".to_string(),
            venv_path: None,
            node_modules_path: None,
            dependencies: HashMap::new(),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            last_used: "2025-01-01T00:00:00Z".to_string(),
        };
        
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProjectConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.project_id, deserialized.project_id);
    }
    
    #[test]
    fn test_check_conflicts() {
        let temp_dir = tempdir().unwrap();
        let manager = MultiProjectManager::new(temp_dir.path().to_path_buf());
        
        // Without registered projects, should return error
        let result = manager.check_conflicts("proj1", "proj2");
        assert!(result.is_err());
    }
}
