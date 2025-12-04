// Dependency & Environment Management: Validation, venv enforcement, environment snapshots
// Purpose: Ensure dependencies are satisfied before execution, enforce Python virtual environments

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Dependency type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DependencyType {
    Python,      // pip package
    Node,        // npm package
    System,      // system binary (e.g., git, docker)
    Library,     // system library (e.g., libssl)
}

/// Dependency requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub dep_type: DependencyType,
    pub version_constraint: Option<String>, // e.g., ">=1.0.0", "^2.0"
    pub required: bool,
    pub description: Option<String>,
}

/// Dependency check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyStatus {
    pub dependency: Dependency,
    pub installed: bool,
    pub installed_version: Option<String>,
    pub satisfies_constraint: bool,
    pub error: Option<String>,
}

/// Validation result for all dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub total_dependencies: usize,
    pub satisfied: usize,
    pub missing: Vec<String>,
    pub version_mismatches: Vec<String>,
    pub statuses: Vec<DependencyStatus>,
}

/// Python virtual environment info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenvInfo {
    pub exists: bool,
    pub path: PathBuf,
    pub python_version: Option<String>,
    pub active: bool,
    pub packages_count: usize,
}

/// Environment snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub timestamp: String,
    pub project_path: String,
    pub venv_path: Option<String>,
    pub python_version: Option<String>,
    pub pip_packages: HashMap<String, String>, // name -> version
    pub npm_packages: HashMap<String, String>,
    pub system_binaries: HashSet<String>,
    pub environment_variables: HashMap<String, String>,
}

/// Dependency Validator
pub struct DependencyValidator;

impl DependencyValidator {
    /// Validate all dependencies (dry-run before execution)
    pub fn validate_dependencies(dependencies: Vec<Dependency>) -> ValidationResult {
        let mut statuses = Vec::new();
        let mut missing = Vec::new();
        let mut version_mismatches = Vec::new();
        let mut satisfied = 0;
        
        for dep in dependencies {
            let status = Self::check_dependency(&dep);
            
            if !status.installed {
                missing.push(dep.name.clone());
            } else if !status.satisfies_constraint {
                version_mismatches.push(format!(
                    "{} (installed: {}, required: {})",
                    dep.name,
                    status.installed_version.as_deref().unwrap_or("unknown"),
                    dep.version_constraint.as_deref().unwrap_or("any")
                ));
            } else {
                satisfied += 1;
            }
            
            statuses.push(status);
        }
        
        ValidationResult {
            valid: missing.is_empty() && version_mismatches.is_empty(),
            total_dependencies: statuses.len(),
            satisfied,
            missing,
            version_mismatches,
            statuses,
        }
    }
    
    /// Check single dependency
    fn check_dependency(dep: &Dependency) -> DependencyStatus {
        let (installed, installed_version) = match dep.dep_type {
            DependencyType::Python => Self::check_python_package(&dep.name),
            DependencyType::Node => Self::check_npm_package(&dep.name),
            DependencyType::System => Self::check_system_binary(&dep.name),
            DependencyType::Library => Self::check_system_library(&dep.name),
        };
        
        let satisfies_constraint = if installed && dep.version_constraint.is_some() {
            Self::check_version_constraint(
                installed_version.as_deref(),
                dep.version_constraint.as_deref().unwrap(),
            )
        } else {
            installed // If no constraint, just check if installed
        };
        
        DependencyStatus {
            dependency: dep.clone(),
            installed,
            installed_version,
            satisfies_constraint,
            error: if !installed {
                Some(format!("{} is not installed", dep.name))
            } else if !satisfies_constraint {
                Some(format!(
                    "{} version mismatch (installed: {}, required: {})",
                    dep.name,
                    installed_version.as_deref().unwrap_or("unknown"),
                    dep.version_constraint.as_deref().unwrap_or("any")
                ))
            } else {
                None
            },
        }
    }
    
    /// Check if Python package is installed
    fn check_python_package(name: &str) -> (bool, Option<String>) {
        // Try `pip show <package>`
        let output = Command::new("pip")
            .args(&["show", name])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let version = stdout
                    .lines()
                    .find(|line| line.starts_with("Version:"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|v| v.trim().to_string());
                (true, version)
            }
            _ => (false, None),
        }
    }
    
    /// Check if npm package is installed
    fn check_npm_package(name: &str) -> (bool, Option<String>) {
        // Check local node_modules first
        let node_modules = Path::new("node_modules").join(name).join("package.json");
        
        if node_modules.exists() {
            if let Ok(content) = fs::read_to_string(&node_modules) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    let version = json.get("version")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    return (true, version);
                }
            }
        }
        
        // Try `npm list <package>` for global
        let output = Command::new("npm")
            .args(&["list", "-g", name, "--depth=0"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Parse version from output like "package@1.0.0"
                let version = stdout
                    .lines()
                    .find(|line| line.contains(&format!("{}@", name)))
                    .and_then(|line| line.split('@').nth(1))
                    .map(|v| v.trim().to_string());
                (true, version)
            }
            _ => (false, None),
        }
    }
    
    /// Check if system binary exists
    fn check_system_binary(name: &str) -> (bool, Option<String>) {
        let output = Command::new("which")
            .arg(name)
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                // Try to get version
                let version_output = Command::new(name)
                    .arg("--version")
                    .output();
                
                let version = version_output
                    .ok()
                    .and_then(|out| {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        stdout.lines().next().map(|s| s.to_string())
                    });
                
                (true, version)
            }
            _ => (false, None),
        }
    }
    
    /// Check if system library exists (simplified)
    fn check_system_library(name: &str) -> (bool, Option<String>) {
        // This is platform-specific and simplified
        // On macOS/Linux, check common library paths
        let lib_paths = vec![
            "/usr/lib",
            "/usr/local/lib",
            "/lib",
        ];
        
        for lib_path in lib_paths {
            let path = Path::new(lib_path).join(format!("{}.so", name))
                .with_extension(""); // Will check .so, .dylib, etc.
            
            if path.with_extension("so").exists() 
                || path.with_extension("dylib").exists()
                || path.with_extension("a").exists() {
                return (true, None); // Version detection for libraries is complex
            }
        }
        
        (false, None)
    }
    
    /// Check version constraint (simplified semver-ish)
    fn check_version_constraint(installed: Option<&str>, constraint: &str) -> bool {
        if let Some(installed_ver) = installed {
            // Handle simple constraints: ">=1.0.0", "^2.0", "~1.2.3", "1.0.0"
            if constraint.starts_with(">=") {
                return Self::version_gte(installed_ver, &constraint[2..]);
            } else if constraint.starts_with('^') {
                return Self::version_caret(installed_ver, &constraint[1..]);
            } else if constraint.starts_with('~') {
                return Self::version_tilde(installed_ver, &constraint[1..]);
            } else {
                // Exact match
                return installed_ver == constraint;
            }
        }
        false
    }
    
    /// Check if version is >= constraint
    fn version_gte(installed: &str, constraint: &str) -> bool {
        let installed_parts: Vec<u32> = installed.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let constraint_parts: Vec<u32> = constraint.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        for i in 0..constraint_parts.len() {
            let inst = installed_parts.get(i).copied().unwrap_or(0);
            let cons = constraint_parts[i];
            
            if inst > cons {
                return true;
            } else if inst < cons {
                return false;
            }
        }
        
        true // Equal or installed has more parts
    }
    
    /// Check caret version (^1.2.3 allows >=1.2.3 <2.0.0)
    fn version_caret(installed: &str, constraint: &str) -> bool {
        let installed_parts: Vec<u32> = installed.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let constraint_parts: Vec<u32> = constraint.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if installed_parts.is_empty() || constraint_parts.is_empty() {
            return false;
        }
        
        // Major version must match
        if installed_parts[0] != constraint_parts[0] {
            return false;
        }
        
        // Check if >= constraint
        Self::version_gte(installed, constraint)
    }
    
    /// Check tilde version (~1.2.3 allows >=1.2.3 <1.3.0)
    fn version_tilde(installed: &str, constraint: &str) -> bool {
        let installed_parts: Vec<u32> = installed.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let constraint_parts: Vec<u32> = constraint.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if installed_parts.len() < 2 || constraint_parts.len() < 2 {
            return false;
        }
        
        // Major and minor must match
        if installed_parts[0] != constraint_parts[0] || installed_parts[1] != constraint_parts[1] {
            return false;
        }
        
        // Check if >= constraint
        Self::version_gte(installed, constraint)
    }
}

/// Virtual Environment Manager
pub struct VenvManager;

impl VenvManager {
    /// Get virtual environment info
    pub fn get_venv_info(project_path: &str) -> VenvInfo {
        let project = Path::new(project_path);
        let venv_path = project.join(".venv");
        
        let exists = venv_path.exists() && venv_path.join("bin").join("python").exists();
        
        let python_version = if exists {
            Self::get_venv_python_version(&venv_path)
        } else {
            None
        };
        
        let active = std::env::var("VIRTUAL_ENV")
            .ok()
            .map(|v| Path::new(&v) == venv_path)
            .unwrap_or(false);
        
        let packages_count = if exists {
            Self::count_venv_packages(&venv_path)
        } else {
            0
        };
        
        VenvInfo {
            exists,
            path: venv_path,
            python_version,
            active,
            packages_count,
        }
    }
    
    /// Create virtual environment
    pub fn create_venv(project_path: &str) -> Result<(), String> {
        let project = Path::new(project_path);
        let venv_path = project.join(".venv");
        
        if venv_path.exists() {
            return Err("Virtual environment already exists".to_string());
        }
        
        let output = Command::new("python3")
            .args(&["-m", "venv", venv_path.to_str().unwrap()])
            .output()
            .map_err(|e| format!("Failed to create venv: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to create venv: {}", stderr));
        }
        
        Ok(())
    }
    
    /// Enforce venv usage (check if active, return activation command if not)
    pub fn enforce_venv(project_path: &str) -> Result<Option<String>, String> {
        let info = Self::get_venv_info(project_path);
        
        if !info.exists {
            return Err("Virtual environment does not exist. Create it first.".to_string());
        }
        
        if info.active {
            Ok(None) // Already active
        } else {
            // Return activation command
            let activate_script = if cfg!(target_os = "windows") {
                info.path.join("Scripts").join("activate.bat")
            } else {
                info.path.join("bin").join("activate")
            };
            
            Ok(Some(format!("source {}", activate_script.display())))
        }
    }
    
    /// Get Python version in venv
    fn get_venv_python_version(venv_path: &Path) -> Option<String> {
        let python_bin = venv_path.join("bin").join("python");
        
        let output = Command::new(&python_bin)
            .arg("--version")
            .output()
            .ok()?;
        
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.trim().strip_prefix("Python ").map(|s| s.to_string())
        } else {
            None
        }
    }
    
    /// Count packages in venv
    fn count_venv_packages(venv_path: &Path) -> usize {
        let pip_bin = venv_path.join("bin").join("pip");
        
        let output = Command::new(&pip_bin)
            .args(&["list", "--format=json"])
            .output();
        
        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(packages) = serde_json::from_str::<Vec<serde_json::Value>>(&stdout) {
                    return packages.len();
                }
            }
        }
        
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_gte() {
        assert!(DependencyValidator::version_gte("1.2.3", "1.0.0"));
        assert!(DependencyValidator::version_gte("2.0.0", "1.9.9"));
        assert!(DependencyValidator::version_gte("1.2.3", "1.2.3"));
        assert!(!DependencyValidator::version_gte("1.0.0", "1.2.3"));
    }
    
    #[test]
    fn test_version_caret() {
        assert!(DependencyValidator::version_caret("1.2.3", "1.0.0"));
        assert!(DependencyValidator::version_caret("1.9.9", "1.0.0"));
        assert!(!DependencyValidator::version_caret("2.0.0", "1.0.0"));
        assert!(!DependencyValidator::version_caret("0.9.0", "1.0.0"));
    }
    
    #[test]
    fn test_version_tilde() {
        assert!(DependencyValidator::version_tilde("1.2.3", "1.2.0"));
        assert!(DependencyValidator::version_tilde("1.2.9", "1.2.0"));
        assert!(!DependencyValidator::version_tilde("1.3.0", "1.2.0"));
        assert!(!DependencyValidator::version_tilde("2.2.0", "1.2.0"));
    }
    
    #[test]
    fn test_validate_dependencies_mock() {
        let deps = vec![
            Dependency {
                name: "python".to_string(),
                dep_type: DependencyType::System,
                version_constraint: None,
                required: true,
                description: Some("Python interpreter".to_string()),
            },
        ];
        
        let result = DependencyValidator::validate_dependencies(deps);
        
        // System-dependent, but should not crash
        assert!(result.total_dependencies == 1);
    }
    
    #[test]
    fn test_venv_info() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let project_path = temp_dir.path().to_str().unwrap();
        
        let info = VenvManager::get_venv_info(project_path);
        
        assert!(!info.exists); // No venv created
        assert!(!info.active);
        assert_eq!(info.packages_count, 0);
    }
}
