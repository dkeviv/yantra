// File: src-tauri/src/agent/packaging.rs
// Purpose: Package builder for generating distributable artifacts
// Dependencies: tokio, serde, std::fs, std::process
// Last Updated: November 22, 2025

// Packaging functionality not yet fully integrated
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tokio::fs;

/// Package type to build
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageType {
    /// Python wheel package (.whl)
    PythonWheel,
    /// Docker container image
    DockerImage,
    /// Node.js npm package
    NpmPackage,
    /// Static site (HTML/CSS/JS)
    StaticSite,
    /// Executable binary (Rust)
    Binary,
}

/// Package configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageConfig {
    /// Package name
    pub name: String,
    /// Package version (semver)
    pub version: String,
    /// Package description
    pub description: String,
    /// Author information
    pub author: String,
    /// License (e.g., "MIT", "Apache-2.0")
    pub license: String,
    /// Entry point file (e.g., "main.py", "index.js")
    pub entry_point: Option<String>,
    /// Dependencies
    pub dependencies: HashMap<String, String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Package build result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageBuildResult {
    /// Build success
    pub success: bool,
    /// Package type built
    pub package_type: PackageType,
    /// Output artifact path
    pub artifact_path: Option<PathBuf>,
    /// Build output
    pub output: String,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Build duration in milliseconds
    pub duration_ms: u64,
    /// Artifact size in bytes
    pub artifact_size: Option<u64>,
}

/// Package builder
pub struct PackageBuilder {
    /// Workspace path
    workspace_path: PathBuf,
}

impl PackageBuilder {
    /// Create new package builder
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }

    /// Build package
    pub async fn build(
        &self,
        package_type: PackageType,
        config: PackageConfig,
    ) -> Result<PackageBuildResult, String> {
        let start_time = std::time::Instant::now();

        let result = match package_type {
            PackageType::PythonWheel => self.build_python_wheel(&config).await,
            PackageType::DockerImage => self.build_docker_image(&config).await,
            PackageType::NpmPackage => self.build_npm_package(&config).await,
            PackageType::StaticSite => self.build_static_site(&config).await,
            PackageType::Binary => self.build_binary(&config).await,
        };

        let duration_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok((artifact_path, output)) => {
                let artifact_size = if let Some(ref path) = artifact_path {
                    fs::metadata(path).await.ok().map(|m| m.len())
                } else {
                    None
                };

                Ok(PackageBuildResult {
                    success: true,
                    package_type,
                    artifact_path,
                    output,
                    error_message: None,
                    duration_ms,
                    artifact_size,
                })
            }
            Err(error) => Ok(PackageBuildResult {
                success: false,
                package_type,
                artifact_path: None,
                output: String::new(),
                error_message: Some(error),
                duration_ms,
                artifact_size: None,
            }),
        }
    }

    /// Build Python wheel package
    async fn build_python_wheel(
        &self,
        config: &PackageConfig,
    ) -> Result<(Option<PathBuf>, String), String> {
        // Generate setup.py
        let setup_py = self.generate_setup_py(config)?;
        let setup_path = self.workspace_path.join("setup.py");
        
        fs::write(&setup_path, setup_py)
            .await
            .map_err(|e| format!("Failed to write setup.py: {}", e))?;

        // Generate pyproject.toml for modern Python packaging
        let pyproject_toml = self.generate_pyproject_toml(config)?;
        let pyproject_path = self.workspace_path.join("pyproject.toml");
        
        fs::write(&pyproject_path, pyproject_toml)
            .await
            .map_err(|e| format!("Failed to write pyproject.toml: {}", e))?;

        // Build wheel using python -m build
        let output = Command::new("python")
            .args(["-m", "build", "--wheel"])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute build command: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Wheel build failed: {}", stderr));
        }

        // Find the generated wheel
        let dist_dir = self.workspace_path.join("dist");
        let mut entries = fs::read_dir(&dist_dir)
            .await
            .map_err(|e| format!("Failed to read dist directory: {}", e))?;

        let mut wheel_path = None;
        while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("whl") {
                wheel_path = Some(path);
                break;
            }
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok((wheel_path, stdout.to_string()))
    }

    /// Build Docker image
    async fn build_docker_image(
        &self,
        config: &PackageConfig,
    ) -> Result<(Option<PathBuf>, String), String> {
        // Generate Dockerfile
        let dockerfile = self.generate_dockerfile(config)?;
        let dockerfile_path = self.workspace_path.join("Dockerfile");
        
        fs::write(&dockerfile_path, dockerfile)
            .await
            .map_err(|e| format!("Failed to write Dockerfile: {}", e))?;

        // Generate .dockerignore
        let dockerignore = self.generate_dockerignore();
        let dockerignore_path = self.workspace_path.join(".dockerignore");
        
        fs::write(&dockerignore_path, dockerignore)
            .await
            .map_err(|e| format!("Failed to write .dockerignore: {}", e))?;

        // Build Docker image
        let image_tag = format!("{}:{}", config.name, config.version);
        let output = Command::new("docker")
            .args(["build", "-t", &image_tag, "."])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute docker build: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Docker build failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok((Some(dockerfile_path), stdout.to_string()))
    }

    /// Build npm package
    async fn build_npm_package(
        &self,
        config: &PackageConfig,
    ) -> Result<(Option<PathBuf>, String), String> {
        // Generate package.json
        let package_json = self.generate_package_json(config)?;
        let package_path = self.workspace_path.join("package.json");
        
        fs::write(&package_path, package_json)
            .await
            .map_err(|e| format!("Failed to write package.json: {}", e))?;

        // Run npm pack
        let output = Command::new("npm")
            .args(["pack"])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute npm pack: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("npm pack failed: {}", stderr));
        }

        // Find the generated tarball
        let tarball_name = format!("{}-{}.tgz", config.name, config.version);
        let tarball_path = self.workspace_path.join(&tarball_name);

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok((Some(tarball_path), stdout.to_string()))
    }

    /// Build static site
    async fn build_static_site(
        &self,
        config: &PackageConfig,
    ) -> Result<(Option<PathBuf>, String), String> {
        // Create dist directory
        let dist_dir = self.workspace_path.join("dist");
        fs::create_dir_all(&dist_dir)
            .await
            .map_err(|e| format!("Failed to create dist directory: {}", e))?;

        // Copy static files (HTML, CSS, JS)
        let src_dir = self.workspace_path.join("src");
        if !src_dir.exists() {
            return Err("Source directory not found".to_string());
        }

        // Simplified: just copy files (in production, would use bundler like webpack)
        let output = format!(
            "Built static site '{}' v{}\nOutput: {}",
            config.name,
            config.version,
            dist_dir.display()
        );

        Ok((Some(dist_dir), output))
    }

    /// Build binary (Rust)
    async fn build_binary(
        &self,
        config: &PackageConfig,
    ) -> Result<(Option<PathBuf>, String), String> {
        // Run cargo build --release
        let output = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(&self.workspace_path)
            .output()
            .map_err(|e| format!("Failed to execute cargo build: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Cargo build failed: {}", stderr));
        }

        // Find the binary in target/release
        let binary_name = &config.name;
        let binary_path = self.workspace_path.join("target/release").join(binary_name);

        if !binary_path.exists() {
            return Err(format!("Binary not found: {}", binary_path.display()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok((Some(binary_path), stdout.to_string()))
    }

    /// Generate setup.py for Python package
    fn generate_setup_py(&self, config: &PackageConfig) -> Result<String, String> {
        let deps = config
            .dependencies
            .iter()
            .map(|(k, v)| format!("'{}=={}'", k, v))
            .collect::<Vec<_>>()
            .join(", ");

        Ok(format!(
            r#"from setuptools import setup, find_packages

setup(
    name='{}',
    version='{}',
    description='{}',
    author='{}',
    license='{}',
    packages=find_packages(),
    install_requires=[{}],
    python_requires='>=3.7',
)
"#,
            config.name, config.version, config.description, config.author, config.license, deps
        ))
    }

    /// Generate pyproject.toml for modern Python packaging
    fn generate_pyproject_toml(&self, config: &PackageConfig) -> Result<String, String> {
        Ok(format!(
            r#"[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{}"
version = "{}"
description = "{}"
authors = [{{name = "{}"}}]
license = {{text = "{}"}}
requires-python = ">=3.7"
"#,
            config.name, config.version, config.description, config.author, config.license
        ))
    }

    /// Generate Dockerfile
    fn generate_dockerfile(&self, config: &PackageConfig) -> Result<String, String> {
        let entry_point = config
            .entry_point.as_deref()
            .unwrap_or("main.py");

        Ok(format!(
            r#"# Multi-stage build for Python application
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (if web app)
EXPOSE 8000

# Run application
CMD ["python", "{}"]
"#,
            entry_point
        ))
    }

    /// Generate .dockerignore
    fn generate_dockerignore(&self) -> String {
        r#"# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
.venv
venv/
ENV/

# Node.js
node_modules/
npm-debug.log
yarn-error.log

# Testing
.pytest_cache
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Build artifacts
dist/
build/
*.egg-info/

# OS
.DS_Store
Thumbs.db
"#
        .to_string()
    }

    /// Generate package.json for Node.js
    fn generate_package_json(&self, config: &PackageConfig) -> Result<String, String> {
        let deps: HashMap<String, String> = config
            .dependencies
            .iter()
            .map(|(k, v)| (k.clone(), format!("^{}", v)))
            .collect();

        let package = serde_json::json!({
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "main": config.entry_point.as_ref().unwrap_or(&"index.js".to_string()),
            "author": config.author,
            "license": config.license,
            "dependencies": deps,
        });

        serde_json::to_string_pretty(&package).map_err(|e| format!("JSON error: {}", e))
    }

    /// Detect recommended package type for workspace
    pub async fn detect_package_type(&self) -> PackageType {
        // Check for Python project
        if self.workspace_path.join("setup.py").exists()
            || self.workspace_path.join("pyproject.toml").exists()
        {
            return PackageType::PythonWheel;
        }

        // Check for Node.js project
        if self.workspace_path.join("package.json").exists() {
            return PackageType::NpmPackage;
        }

        // Check for Rust project
        if self.workspace_path.join("Cargo.toml").exists() {
            return PackageType::Binary;
        }

        // Check for Dockerfile
        if self.workspace_path.join("Dockerfile").exists() {
            return PackageType::DockerImage;
        }

        // Default to static site for web projects
        if self.workspace_path.join("index.html").exists() {
            return PackageType::StaticSite;
        }

        // Default to Docker image (most flexible)
        PackageType::DockerImage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_package_builder_creation() {
        let temp_dir = tempdir().unwrap();
        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());
        assert_eq!(builder.workspace_path, temp_dir.path());
    }

    #[test]
    fn test_generate_setup_py() {
        let temp_dir = tempdir().unwrap();
        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());

        let config = PackageConfig {
            name: "test-package".to_string(),
            version: "1.0.0".to_string(),
            description: "Test package".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            entry_point: Some("main.py".to_string()),
            dependencies: HashMap::new(),
            metadata: HashMap::new(),
        };

        let setup_py = builder.generate_setup_py(&config).unwrap();
        assert!(setup_py.contains("name='test-package'"));
        assert!(setup_py.contains("version='1.0.0'"));
        assert!(setup_py.contains("author='Test Author'"));
    }

    #[test]
    fn test_generate_dockerfile() {
        let temp_dir = tempdir().unwrap();
        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());

        let config = PackageConfig {
            name: "test-app".to_string(),
            version: "1.0.0".to_string(),
            description: "Test app".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            entry_point: Some("app.py".to_string()),
            dependencies: HashMap::new(),
            metadata: HashMap::new(),
        };

        let dockerfile = builder.generate_dockerfile(&config).unwrap();
        assert!(dockerfile.contains("FROM python:3.11-slim"));
        assert!(dockerfile.contains("CMD [\"python\", \"app.py\"]"));
    }

    #[test]
    fn test_generate_package_json() {
        let temp_dir = tempdir().unwrap();
        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());

        let mut deps = HashMap::new();
        deps.insert("express".to_string(), "4.18.0".to_string());

        let config = PackageConfig {
            name: "test-package".to_string(),
            version: "1.0.0".to_string(),
            description: "Test package".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            entry_point: Some("index.js".to_string()),
            dependencies: deps,
            metadata: HashMap::new(),
        };

        let package_json = builder.generate_package_json(&config).unwrap();
        assert!(package_json.contains("\"name\": \"test-package\""));
        assert!(package_json.contains("\"version\": \"1.0.0\""));
        assert!(package_json.contains("express"));
    }

    #[tokio::test]
    async fn test_detect_package_type_python() {
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("setup.py"), "# setup.py")
            .await
            .unwrap();

        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());
        let package_type = builder.detect_package_type().await;
        assert_eq!(package_type, PackageType::PythonWheel);
    }

    #[tokio::test]
    async fn test_detect_package_type_nodejs() {
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("package.json"), "{}")
            .await
            .unwrap();

        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());
        let package_type = builder.detect_package_type().await;
        assert_eq!(package_type, PackageType::NpmPackage);
    }

    #[tokio::test]
    async fn test_detect_package_type_rust() {
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("Cargo.toml"), "[package]")
            .await
            .unwrap();

        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());
        let package_type = builder.detect_package_type().await;
        assert_eq!(package_type, PackageType::Binary);
    }

    #[test]
    fn test_generate_dockerignore() {
        let temp_dir = tempdir().unwrap();
        let builder = PackageBuilder::new(temp_dir.path().to_path_buf());
        let dockerignore = builder.generate_dockerignore();
        assert!(dockerignore.contains("__pycache__"));
        assert!(dockerignore.contains(".git"));
        assert!(dockerignore.contains("node_modules"));
    }
}
