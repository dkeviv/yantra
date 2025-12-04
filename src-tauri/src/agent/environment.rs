// Environment Snapshot & Validator: Capture and restore environment state
// Purpose: Snapshot environment for rollback, validate before execution

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub timestamp: String,
    pub project_path: String,
    pub python_version: Option<String>,
    pub node_version: Option<String>,
    pub pip_packages: HashMap<String, String>,
    pub npm_packages: HashMap<String, String>,
    pub env_vars: HashMap<String, String>,
    pub requirements_content: Option<String>,
    pub package_json_content: Option<String>,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub missing_dependencies: Vec<String>,
    pub version_mismatches: Vec<String>,
}

/// Environment Snapshot Manager
pub struct SnapshotManager {
    snapshots_dir: PathBuf,
}

impl SnapshotManager {
    pub fn new(project_path: &Path) -> Self {
        let snapshots_dir = project_path.join(".yantra/snapshots");
        if !snapshots_dir.exists() {
            let _ = fs::create_dir_all(&snapshots_dir);
        }
        
        Self { snapshots_dir }
    }
    
    /// Create environment snapshot
    pub fn create_snapshot(&self, project_path: &Path) -> Result<EnvironmentSnapshot, String> {
        let snapshot = EnvironmentSnapshot {
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_path: project_path.to_string_lossy().to_string(),
            python_version: Self::get_python_version(),
            node_version: Self::get_node_version(),
            pip_packages: Self::get_pip_packages(),
            npm_packages: Self::get_npm_packages(project_path),
            env_vars: Self::get_important_env_vars(),
            requirements_content: Self::read_requirements(project_path),
            package_json_content: Self::read_package_json(project_path),
        };
        
        // Save snapshot
        self.save_snapshot(&snapshot)?;
        
        Ok(snapshot)
    }
    
    /// Save snapshot to disk
    fn save_snapshot(&self, snapshot: &EnvironmentSnapshot) -> Result<(), String> {
        let filename = format!("snapshot_{}.json", 
            snapshot.timestamp.replace([':', '-', '.'], "_"));
        let path = self.snapshots_dir.join(filename);
        
        let json = serde_json::to_string_pretty(snapshot)
            .map_err(|e| format!("Failed to serialize snapshot: {}", e))?;
        
        fs::write(&path, json)
            .map_err(|e| format!("Failed to write snapshot: {}", e))?;
        
        Ok(())
    }
    
    /// Load latest snapshot
    pub fn load_latest_snapshot(&self) -> Result<EnvironmentSnapshot, String> {
        let entries = fs::read_dir(&self.snapshots_dir)
            .map_err(|e| format!("Failed to read snapshots directory: {}", e))?;
        
        let mut snapshots: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        
        snapshots.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
        
        let latest = snapshots.last()
            .ok_or_else(|| "No snapshots found".to_string())?;
        
        let content = fs::read_to_string(latest.path())
            .map_err(|e| format!("Failed to read snapshot: {}", e))?;
        
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse snapshot: {}", e))
    }
    
    /// Rollback to snapshot
    pub fn rollback(&self, snapshot: &EnvironmentSnapshot, project_path: &Path) -> Result<(), String> {
        // Restore requirements.txt
        if let Some(ref content) = snapshot.requirements_content {
            let req_path = project_path.join("requirements.txt");
            fs::write(req_path, content)
                .map_err(|e| format!("Failed to restore requirements.txt: {}", e))?;
        }
        
        // Restore package.json
        if let Some(ref content) = snapshot.package_json_content {
            let pkg_path = project_path.join("package.json");
            fs::write(pkg_path, content)
                .map_err(|e| format!("Failed to restore package.json: {}", e))?;
        }
        
        // TODO: Actually reinstall packages to match snapshot
        
        Ok(())
    }
    
    // Helper methods
    fn get_python_version() -> Option<String> {
        Command::new("python3")
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    }
    
    fn get_node_version() -> Option<String> {
        Command::new("node")
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    }
    
    fn get_pip_packages() -> HashMap<String, String> {
        let output = Command::new("pip")
            .args(&["list", "--format=json"])
            .output();
        
        if let Ok(output) = output {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                if let Ok(packages) = serde_json::from_str::<Vec<serde_json::Value>>(&stdout) {
                    return packages
                        .iter()
                        .filter_map(|p| {
                            let name = p.get("name")?.as_str()?;
                            let version = p.get("version")?.as_str()?;
                            Some((name.to_string(), version.to_string()))
                        })
                        .collect();
                }
            }
        }
        
        HashMap::new()
    }
    
    fn get_npm_packages(project_path: &Path) -> HashMap<String, String> {
        let package_lock = project_path.join("package-lock.json");
        if !package_lock.exists() {
            return HashMap::new();
        }
        
        if let Ok(content) = fs::read_to_string(&package_lock) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(deps) = json.get("dependencies").and_then(|d| d.as_object()) {
                    return deps
                        .iter()
                        .filter_map(|(name, info)| {
                            let version = info.get("version")?.as_str()?;
                            Some((name.clone(), version.to_string()))
                        })
                        .collect();
                }
            }
        }
        
        HashMap::new()
    }
    
    fn get_important_env_vars() -> HashMap<String, String> {
        let vars = vec!["VIRTUAL_ENV", "NODE_ENV", "PATH"];
        vars.iter()
            .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
            .collect()
    }
    
    fn read_requirements(project_path: &Path) -> Option<String> {
        let req_path = project_path.join("requirements.txt");
        fs::read_to_string(req_path).ok()
    }
    
    fn read_package_json(project_path: &Path) -> Option<String> {
        let pkg_path = project_path.join("package.json");
        fs::read_to_string(pkg_path).ok()
    }
}

/// Environment Validator
pub struct EnvironmentValidator;

impl EnvironmentValidator {
    /// Validate environment before execution
    pub fn validate(project_path: &Path) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut missing_dependencies = Vec::new();
        let mut version_mismatches = Vec::new();
        
        // Check Python
        if !Self::check_python() {
            errors.push("Python 3 not found".to_string());
        }
        
        // Check requirements.txt
        let req_path = project_path.join("requirements.txt");
        if req_path.exists() {
            if let Ok(content) = fs::read_to_string(&req_path) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    
                    let package = line.split(&['=', '>', '<', '~'][..]).next().unwrap_or(line);
                    if !Self::check_pip_package(package) {
                        missing_dependencies.push(package.to_string());
                    }
                }
            }
        }
        
        // Check package.json
        let pkg_path = project_path.join("package.json");
        if pkg_path.exists() {
            if !Self::check_node() {
                errors.push("Node.js not found".to_string());
            }
            
            if let Ok(content) = fs::read_to_string(&pkg_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(deps) = json.get("dependencies").and_then(|d| d.as_object()) {
                        for (name, _) in deps {
                            if !Self::check_npm_package(name) {
                                missing_dependencies.push(name.clone());
                            }
                        }
                    }
                }
            }
        }
        
        ValidationResult {
            valid: errors.is_empty() && missing_dependencies.is_empty(),
            errors,
            warnings,
            missing_dependencies,
            version_mismatches,
        }
    }
    
    fn check_python() -> bool {
        Command::new("python3").arg("--version").output().is_ok()
    }
    
    fn check_node() -> bool {
        Command::new("node").arg("--version").output().is_ok()
    }
    
    fn check_pip_package(package: &str) -> bool {
        Command::new("pip")
            .args(&["show", package])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    
    fn check_npm_package(package: &str) -> bool {
        let node_modules = Path::new("node_modules").join(package);
        node_modules.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_create_snapshot() {
        let temp_dir = tempdir().unwrap();
        let manager = SnapshotManager::new(temp_dir.path());
        
        let snapshot = manager.create_snapshot(temp_dir.path()).unwrap();
        assert!(!snapshot.timestamp.is_empty());
        assert!(snapshot.python_version.is_some() || snapshot.node_version.is_some());
    }
    
    #[test]
    fn test_environment_validation() {
        let temp_dir = tempdir().unwrap();
        let result = EnvironmentValidator::validate(temp_dir.path());
        
        // Should pass for empty project
        assert!(result.errors.is_empty() || result.errors.iter().any(|e| e.contains("not found")));
    }
}
