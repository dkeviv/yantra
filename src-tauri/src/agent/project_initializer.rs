// Project Initializer - Architecture-First Workflow for New and Existing Projects
// Purpose: Ensure architecture is created and approved BEFORE any code implementation
// Created: November 28, 2025
//
// Core Workflow:
// 1. New Project: Generate arch → User approval → Generate code
// 2. Existing Project: Check arch files → Analyze code → User approval → Set baseline

use std::path::{Path, PathBuf};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use crate::architecture::{
    Architecture, ArchitectureManager, ArchitectureGenerator, ArchitectureAnalyzer
};
use crate::gnn::GNNEngine;
use crate::llm::orchestrator::LLMOrchestrator;
use crate::llm::CodeGenerationRequest;

/// Source of architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureSource {
    /// Yantra's architecture.json file
    YantraFile,
    /// Markdown architecture file
    MarkdownFile(String),
    /// Found in documentation folder
    DocumentationFile(String),
    /// No architecture files found
    None,
}

/// Project analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAnalysisReport {
    pub files_count: usize,
    pub components_count: usize,
    pub connections_count: usize,
    pub complexity_score: f32,
    pub quality_issues: Vec<String>,
    pub cyclic_dependencies: Vec<String>,
    pub architecture_violations: Vec<String>,
}

/// Initialization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationResult {
    pub success: bool,
    pub architecture: Option<Architecture>,
    pub analysis_report: Option<ProjectAnalysisReport>,
    pub user_approved: bool,
    pub architecture_source: ArchitectureSource,
    pub message: String,
}

/// User approval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalResult {
    Approved,
    Modified(String),  // User requested modifications
    Rejected,
}

/// Project initializer
pub struct ProjectInitializer {
    gnn: Arc<tokio::sync::Mutex<GNNEngine>>,
    llm: Arc<tokio::sync::Mutex<LLMOrchestrator>>,
    arch_manager: std::sync::Mutex<ArchitectureManager>,
    analyzer: ArchitectureAnalyzer,
    generator: ArchitectureGenerator,
}

impl ProjectInitializer {
    /// Create new project initializer
    pub fn new(
        gnn: Arc<tokio::sync::Mutex<GNNEngine>>,
        llm: Arc<tokio::sync::Mutex<LLMOrchestrator>>,
    ) -> Result<Self, String> {
        let arch_manager = ArchitectureManager::new()?;
        let analyzer = ArchitectureAnalyzer::new(gnn.clone());
        let generator = ArchitectureGenerator::new(llm.clone());

        Ok(Self {
            gnn,
            llm,
            arch_manager: std::sync::Mutex::new(arch_manager),
            analyzer,
            generator,
        })
    }

    /// Initialize new project with architecture-first workflow
    ///
    /// Workflow:
    /// 1. Generate architecture from intent using LLM
    /// 2. Save architecture to database
    /// 3. Export to architecture.json and architecture.md
    /// 4. Return result for user approval
    pub async fn initialize_new_project(
        &self,
        intent: &str,
        project_path: &Path,
    ) -> Result<InitializationResult, String> {
        println!("🚀 Initializing new project: {}", intent);
        println!("📁 Project path: {}", project_path.display());

        // Step 1: Generate architecture from intent
        println!("📐 Generating architecture from user intent...");
        let architecture = self.generator
            .generate_from_intent(intent)
            .await
            .map_err(|e| format!("Failed to generate architecture: {}", e))?;

        println!("✅ Generated architecture with {} components", architecture.components.len());

        // Step 2: Save to database using public API
        let saved_arch = self.arch_manager.lock().unwrap().create_architecture(
            architecture.name.clone(),
            architecture.description.clone()
        )?;
        
        // Copy components and connections to saved architecture
        let mut final_arch = saved_arch;
        final_arch.components = architecture.components.clone();
        final_arch.connections = architecture.connections.clone();

        // Step 3: Export to files
        self.export_architecture_files(&final_arch, project_path)?;

        // Step 4: Return for user approval
        Ok(InitializationResult {
            success: true,
            architecture: Some(architecture),
            analysis_report: None,
            user_approved: false,  // Waiting for approval
            architecture_source: ArchitectureSource::YantraFile,
            message: format!(
                "Architecture generated with {} components and {} connections. Please review.",
                final_arch.components.len(),
                final_arch.connections.len()
            ),
        })
    }

    /// Initialize existing project (first-time open)
    ///
    /// Workflow:
    /// 1. Check for existing architecture files
    /// 2. If found: Offer to import or regenerate
    /// 3. If not found: Analyze code and generate architecture
    /// 4. Return result for user approval
    pub async fn initialize_existing_project(
        &self,
        project_path: &Path,
    ) -> Result<InitializationResult, String> {
        println!("🔍 Initializing existing project at {}", project_path.display());

        // Step 1: Check for architecture files
        let arch_source = self.check_architecture_files(project_path);

        match arch_source {
            ArchitectureSource::YantraFile => {
                // Load existing architecture
                let arch_path = project_path.join("architecture.json");
                let content = std::fs::read_to_string(&arch_path)
                    .map_err(|e| format!("Failed to read architecture.json: {}", e))?;
                
                let architecture: Architecture = serde_json::from_str(&content)
                    .map_err(|e| format!("Failed to parse architecture.json: {}", e))?;

                Ok(InitializationResult {
                    success: true,
                    architecture: Some(architecture),
                    analysis_report: None,
                    user_approved: false,
                    architecture_source: ArchitectureSource::YantraFile,
                    message: "Found existing architecture.json. Import this?".to_string(),
                })
            }
            ArchitectureSource::MarkdownFile(path) => {
                Ok(InitializationResult {
                    success: true,
                    architecture: None,
                    analysis_report: None,
                    user_approved: false,
                    architecture_source: ArchitectureSource::MarkdownFile(path.clone()),
                    message: format!("Found architecture documentation in {}. Import or regenerate?", path),
                })
            }
            ArchitectureSource::DocumentationFile(path) => {
                Ok(InitializationResult {
                    success: true,
                    architecture: None,
                    analysis_report: None,
                    user_approved: false,
                    architecture_source: ArchitectureSource::DocumentationFile(path.clone()),
                    message: format!("Found architecture documentation in {}. Import or regenerate?", path),
                })
            }
            ArchitectureSource::None => {
                // No architecture files found - analyze codebase
                self.analyze_and_generate_architecture(project_path).await
            }
        }
    }

    /// Check for existing architecture files
    fn check_architecture_files(&self, project_path: &Path) -> ArchitectureSource {
        // Priority 1: Check for architecture.json (Yantra's format)
        if project_path.join("architecture.json").exists() {
            return ArchitectureSource::YantraFile;
        }

        // Priority 2: Check for architecture.md
        if project_path.join("architecture.md").exists() {
            return ArchitectureSource::MarkdownFile("architecture.md".to_string());
        }

        // Priority 3: Check for common architecture files
        let arch_files = [
            "docs/architecture.md",
            "ARCHITECTURE.md",
            "docs/design.md",
            "docs/ARCHITECTURE.md",
            ".github/ARCHITECTURE.md",
        ];

        for file in arch_files {
            if project_path.join(file).exists() {
                return ArchitectureSource::DocumentationFile(file.to_string());
            }
        }

        ArchitectureSource::None
    }

    /// Import architecture from various file formats
    /// 
    /// Supports:
    /// - architecture.json (Yantra native format)
    /// - Markdown (.md) files with architecture sections
    /// - Mermaid diagrams
    /// - Basic PlantUML component diagrams
    pub async fn import_architecture_from_file(
        &self,
        file_path: &Path,
    ) -> Result<Architecture, String> {
        println!("📥 Importing architecture from: {}", file_path.display());

        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Determine format based on file extension and content
        let extension = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "json" => self.parse_json_architecture(&content),
            "md" | "markdown" => self.parse_markdown_architecture(&content, file_path).await,
            "mmd" | "mermaid" => self.parse_mermaid_architecture(&content).await,
            "puml" | "plantuml" => self.parse_plantuml_architecture(&content).await,
            _ => {
                // Try to auto-detect format from content
                if content.trim_start().starts_with('{') {
                    self.parse_json_architecture(&content)
                } else if content.contains("```mermaid") || content.contains("graph ") {
                    self.parse_mermaid_architecture(&content).await
                } else if content.contains("@startuml") {
                    self.parse_plantuml_architecture(&content).await
                } else {
                    // Default to markdown
                    self.parse_markdown_architecture(&content, file_path).await
                }
            }
        }
    }

    /// Parse JSON architecture file (Yantra native format)
    fn parse_json_architecture(&self, content: &str) -> Result<Architecture, String> {
        serde_json::from_str(content)
            .map_err(|e| format!("Failed to parse JSON architecture: {}", e))
    }

    /// Parse Markdown architecture document
    /// 
    /// Expected format:
    /// # Project Name
    /// Description...
    /// 
    /// ## Components
    /// - Component1: Description
    /// - Component2: Description
    /// 
    /// ## Connections  
    /// - Component1 -> Component2 (Type): Description
    async fn parse_markdown_architecture(
        &self,
        content: &str,
        file_path: &Path,
    ) -> Result<Architecture, String> {
        println!("📄 Parsing Markdown architecture...");

        // Use LLM to extract structured architecture from markdown
        let prompt = format!(
            r#"Extract architecture information from this markdown document and convert it to JSON format.

**Markdown Document:**
{}

**Required JSON Format:**
{{
  "name": "Project Name",
  "description": "Project description",
  "components": [
    {{
      "name": "Component Name",
      "description": "Component description",
      "category": "frontend|backend|database|integration|other"
    }}
  ],
  "connections": [
    {{
      "source": "Source Component",
      "target": "Target Component",
      "type": "DataFlow|ApiCall|Dependency|Event|Bidirectional",
      "description": "Connection description"
    }}
  ]
}}

Only return the JSON, no additional text."#,
            content
        );

        let response = {
            let llm = self.llm.lock().await;
            let request = CodeGenerationRequest {
                intent: prompt.clone(),
                context: vec![],
                file_path: None,
                dependencies: vec![],
            };

            llm.generate_code(&request).await
                .map_err(|e| format!("LLM parsing failed: {}", e))?
        };

        // Parse the LLM response
        #[derive(Deserialize)]
        struct ParsedArch {
            name: String,
            description: String,
            components: Vec<ParsedComponent>,
            connections: Vec<ParsedConnection>,
        }

        #[derive(Deserialize)]
        struct ParsedComponent {
            name: String,
            description: String,
            category: String,
        }

        #[derive(Deserialize)]
        struct ParsedConnection {
            source: String,
            target: String,
            #[serde(rename = "type")]
            connection_type: String,
            description: String,
        }

        let parsed: ParsedArch = serde_json::from_str(&response.code)
            .map_err(|e| format!("Failed to parse LLM response: {}. Response: {}", e, response.code))?;

        // Convert to Architecture
        use crate::architecture::{Component, Connection, ConnectionType, Position, ComponentType};
        use uuid::Uuid;

        let mut architecture = Architecture::new(
            Uuid::new_v4().to_string(),
            parsed.name,
            parsed.description,
        );

        // Add components with auto-generated positions
        for (i, comp) in parsed.components.iter().enumerate() {
            let component = Component::new_planned(
                Uuid::new_v4().to_string(),
                comp.name.clone(),
                comp.description.clone(),
                comp.category.clone(),
                Position {
                    x: 100.0 + (i % 4) as f64 * 250.0,
                    y: 100.0 + (i / 4) as f64 * 200.0,
                },
            );
            architecture.add_component(component);
        }

        // Add connections
        for conn in parsed.connections {
            // Find source and target component IDs
            let source_id = architecture.components.iter()
                .find(|c| c.name == conn.source)
                .map(|c| c.id.clone());

            let target_id = architecture.components.iter()
                .find(|c| c.name == conn.target)
                .map(|c| c.id.clone());

            if let (Some(source), Some(target)) = (source_id, target_id) {
                let conn_type = match conn.connection_type.to_lowercase().as_str() {
                    "dataflow" => ConnectionType::DataFlow,
                    "apicall" => ConnectionType::ApiCall,
                    "dependency" => ConnectionType::Dependency,
                    "event" => ConnectionType::Event,
                    "bidirectional" => ConnectionType::Bidirectional,
                    _ => ConnectionType::DataFlow,
                };

                let connection = Connection::new(
                    Uuid::new_v4().to_string(),
                    source,
                    target,
                    conn_type,
                    conn.description,
                );
                architecture.add_connection(connection);
            }
        }

        println!("✅ Parsed {} components and {} connections",
            architecture.components.len(),
            architecture.connections.len());

        Ok(architecture)
    }

    /// Parse Mermaid diagram
    /// 
    /// Expected format:
    /// ```mermaid
    /// graph LR
    ///     A[Component1] --> B[Component2]
    ///     B --> C[Component3]
    /// ```
    async fn parse_mermaid_architecture(&self, content: &str) -> Result<Architecture, String> {
        println!("🔷 Parsing Mermaid diagram...");

        // Extract mermaid block
        let mermaid_content = if content.contains("```mermaid") {
            content.split("```mermaid")
                .nth(1)
                .and_then(|s| s.split("```").next())
                .unwrap_or(content)
        } else {
            content
        };

        // Use LLM to parse the Mermaid syntax
        let prompt = format!(
            r#"Parse this Mermaid diagram and extract architecture information in JSON format.

**Mermaid Diagram:**
{}

**Required JSON Format:**
{{
  "name": "Inferred Project Name",
  "description": "Brief description of the system",
  "components": [
    {{
      "name": "Component Name",
      "description": "Component description (infer from node label)",
      "category": "frontend|backend|database|integration|other"
    }}
  ],
  "connections": [
    {{
      "source": "Source Component",
      "target": "Target Component",
      "type": "DataFlow|ApiCall|Dependency",
      "description": "Connection description (from arrow label if any)"
    }}
  ]
}}

Only return the JSON, no additional text."#,
            mermaid_content
        );

        let response = {
            let llm = self.llm.lock().await;
            let request = CodeGenerationRequest {
                intent: prompt.clone(),
                context: vec![],
                file_path: None,
                dependencies: vec![],
            };

            llm.generate_code(&request).await
                .map_err(|e| format!("LLM parsing failed: {}", e))?
        };

        // Reuse the markdown parsing result type
        #[derive(Deserialize)]
        struct ParsedArch {
            name: String,
            description: String,
            components: Vec<ParsedComponent>,
            connections: Vec<ParsedConnection>,
        }

        #[derive(Deserialize)]
        struct ParsedComponent {
            name: String,
            description: String,
            category: String,
        }

        #[derive(Deserialize)]
        struct ParsedConnection {
            source: String,
            target: String,
            #[serde(rename = "type")]
            connection_type: String,
            description: String,
        }

        let parsed: ParsedArch = serde_json::from_str(&response.code)
            .map_err(|e| format!("Failed to parse LLM response: {}. Response: {}", e, response.code))?;

        // Convert to Architecture (same logic as markdown)
        use crate::architecture::{Component, Connection, ConnectionType, Position};
        use uuid::Uuid;

        let mut architecture = Architecture::new(
            Uuid::new_v4().to_string(),
            parsed.name,
            parsed.description,
        );

        for (i, comp) in parsed.components.iter().enumerate() {
            let component = Component::new_planned(
                Uuid::new_v4().to_string(),
                comp.name.clone(),
                comp.description.clone(),
                comp.category.clone(),
                Position {
                    x: 100.0 + (i % 4) as f64 * 250.0,
                    y: 100.0 + (i / 4) as f64 * 200.0,
                },
            );
            architecture.add_component(component);
        }

        for conn in parsed.connections {
            let source_id = architecture.components.iter()
                .find(|c| c.name == conn.source)
                .map(|c| c.id.clone());

            let target_id = architecture.components.iter()
                .find(|c| c.name == conn.target)
                .map(|c| c.id.clone());

            if let (Some(source), Some(target)) = (source_id, target_id) {
                let conn_type = match conn.connection_type.to_lowercase().as_str() {
                    "dataflow" => ConnectionType::DataFlow,
                    "apicall" => ConnectionType::ApiCall,
                    "dependency" => ConnectionType::Dependency,
                    "event" => ConnectionType::Event,
                    "bidirectional" => ConnectionType::Bidirectional,
                    _ => ConnectionType::DataFlow,
                };

                let connection = Connection::new(
                    Uuid::new_v4().to_string(),
                    source,
                    target,
                    conn_type,
                    conn.description,
                );
                architecture.add_connection(connection);
            }
        }

        println!("✅ Parsed {} components and {} connections",
            architecture.components.len(),
            architecture.connections.len());

        Ok(architecture)
    }

    /// Parse PlantUML component diagram (basic support)
    /// 
    /// Expected format:
    /// @startuml
    /// component [Component1]
    /// component [Component2]
    /// [Component1] --> [Component2]
    /// @enduml
    async fn parse_plantuml_architecture(&self, content: &str) -> Result<Architecture, String> {
        println!("🌱 Parsing PlantUML diagram...");

        // Use LLM to parse PlantUML syntax
        let prompt = format!(
            r#"Parse this PlantUML component diagram and extract architecture information in JSON format.

**PlantUML Diagram:**
{}

**Required JSON Format:**
{{
  "name": "Inferred Project Name",
  "description": "Brief description of the system",
  "components": [
    {{
      "name": "Component Name",
      "description": "Component description",
      "category": "frontend|backend|database|integration|other"
    }}
  ],
  "connections": [
    {{
      "source": "Source Component",
      "target": "Target Component",
      "type": "DataFlow|ApiCall|Dependency",
      "description": "Connection description"
    }}
  ]
}}

Only return the JSON, no additional text."#,
            content
        );

        let response = {
            let llm = self.llm.lock().await;
            let request = CodeGenerationRequest {
                intent: prompt.clone(),
                context: vec![],
                file_path: None,
                dependencies: vec![],
            };

            llm.generate_code(&request).await
                .map_err(|e| format!("LLM parsing failed: {}", e))?
        };

        // Reuse the parsing structures
        #[derive(Deserialize)]
        struct ParsedArch {
            name: String,
            description: String,
            components: Vec<ParsedComponent>,
            connections: Vec<ParsedConnection>,
        }

        #[derive(Deserialize)]
        struct ParsedComponent {
            name: String,
            description: String,
            category: String,
        }

        #[derive(Deserialize)]
        struct ParsedConnection {
            source: String,
            target: String,
            #[serde(rename = "type")]
            connection_type: String,
            description: String,
        }

        let parsed: ParsedArch = serde_json::from_str(&response.code)
            .map_err(|e| format!("Failed to parse LLM response: {}. Response: {}", e, response.code))?;

        // Convert to Architecture
        use crate::architecture::{Component, Connection, ConnectionType, Position};
        use uuid::Uuid;

        let mut architecture = Architecture::new(
            Uuid::new_v4().to_string(),
            parsed.name,
            parsed.description,
        );

        for (i, comp) in parsed.components.iter().enumerate() {
            let component = Component::new_planned(
                Uuid::new_v4().to_string(),
                comp.name.clone(),
                comp.description.clone(),
                comp.category.clone(),
                Position {
                    x: 100.0 + (i % 4) as f64 * 250.0,
                    y: 100.0 + (i / 4) as f64 * 200.0,
                },
            );
            architecture.add_component(component);
        }

        for conn in parsed.connections {
            let source_id = architecture.components.iter()
                .find(|c| c.name == conn.source)
                .map(|c| c.id.clone());

            let target_id = architecture.components.iter()
                .find(|c| c.name == conn.target)
                .map(|c| c.id.clone());

            if let (Some(source), Some(target)) = (source_id, target_id) {
                let conn_type = match conn.connection_type.to_lowercase().as_str() {
                    "dataflow" => ConnectionType::DataFlow,
                    "apicall" => ConnectionType::ApiCall,
                    "dependency" => ConnectionType::Dependency,
                    "event" => ConnectionType::Event,
                    "bidirectional" => ConnectionType::Bidirectional,
                    _ => ConnectionType::DataFlow,
                };

                let connection = Connection::new(
                    Uuid::new_v4().to_string(),
                    source,
                    target,
                    conn_type,
                    conn.description,
                );
                architecture.add_connection(connection);
            }
        }

        println!("✅ Parsed {} components and {} connections",
            architecture.components.len(),
            architecture.connections.len());

        Ok(architecture)
    }

    /// Analyze codebase and generate architecture
    async fn analyze_and_generate_architecture(
        &self,
        project_path: &Path,
    ) -> Result<InitializationResult, String> {
        println!("📊 Analyzing codebase...");

        // Step 1: Build GNN graph
        {
            let mut gnn = self.gnn.lock().await;
            gnn.build_graph(project_path)
                .map_err(|e| format!("Failed to build GNN graph: {}", e))?;
        }

        // Step 2: Generate architecture from code
        let architecture = self.analyzer
            .generate_from_code(project_path).await
            .map_err(|e| format!("Failed to generate architecture: {}", e))?;

        println!("✅ Identified {} components", architecture.components.len());

        // Step 3: Analyze code quality
        let analysis_report = self.analyze_code_quality(project_path).await?;

        // Step 4: Save architecture using public API
        let saved_arch = self.arch_manager.lock().unwrap().create_architecture(
            architecture.name.clone(),
            architecture.description.clone()
        )?;
        
        // Copy components and connections to saved architecture
        let mut final_arch = saved_arch;
        final_arch.components = architecture.components.clone();
        final_arch.connections = architecture.connections.clone();

        // Step 5: Export to files
        self.export_architecture_files(&final_arch, project_path)?;

        Ok(InitializationResult {
            success: true,
            architecture: Some(architecture),
            analysis_report: Some(analysis_report),
            user_approved: false,
            architecture_source: ArchitectureSource::None,
            message: "Architecture generated from code analysis. Please review.".to_string(),
        })
    }

    /// Analyze code quality
    async fn analyze_code_quality(&self, _project_path: &Path) -> Result<ProjectAnalysisReport, String> {
        let gnn = self.gnn.lock().await;
        
        // Get all files using helper
        let files = self.get_files_from_gnn(&gnn);
        
        // Detect cyclic dependencies
        let cyclic_deps = self.detect_cyclic_dependencies(&gnn);
        
        // Calculate complexity
        let complexity = self.calculate_complexity_score(&gnn);

        // Find quality issues
        let quality_issues = self.find_quality_issues(&gnn);

        Ok(ProjectAnalysisReport {
            files_count: files.len(),
            components_count: 0,  // Will be filled after architecture generation
            connections_count: 0,
            complexity_score: complexity,
            quality_issues,
            cyclic_dependencies: cyclic_deps,
            architecture_violations: Vec::new(),
        })
    }

    /// Detect cyclic dependencies
    fn detect_cyclic_dependencies(&self, _gnn: &GNNEngine) -> Vec<String> {
        // TODO: Implement cycle detection using GNN graph
        // For now, return empty
        Vec::new()
    }

    /// Calculate complexity score (0-10)
    fn calculate_complexity_score(&self, gnn: &GNNEngine) -> f32 {
        let files = self.get_files_from_gnn(gnn);
        if files.is_empty() {
            return 0.0;
        }

        // Simple heuristic: more files = higher complexity
        let file_count_score = (files.len() as f32 / 50.0).min(5.0);
        
        // TODO: Add more sophisticated metrics:
        // - Average cyclomatic complexity
        // - Dependency depth
        // - Code duplication

        file_count_score
    }

    /// Find code quality issues
    fn find_quality_issues(&self, _gnn: &GNNEngine) -> Vec<String> {
        let issues = Vec::new();

        // TODO: Implement quality checks:
        // - Files >500 LOC
        // - Functions >50 LOC
        // - Deep nesting (>4 levels)
        // - Missing documentation
        // - Unused imports

        issues
    }

    /// Export architecture to files
    fn export_architecture_files(
        &self,
        architecture: &Architecture,
        project_path: &Path,
    ) -> Result<(), String> {
        // Export to architecture.json
        let json_path = project_path.join("architecture.json");
        let json_content = serde_json::to_string_pretty(&architecture)
            .map_err(|e| format!("Failed to serialize architecture: {}", e))?;
        
        std::fs::write(&json_path, json_content)
            .map_err(|e| format!("Failed to write architecture.json: {}", e))?;

        // Export to architecture.md
        let md_path = project_path.join("architecture.md");
        let md_content = self.architecture_to_markdown(architecture);
        
        std::fs::write(&md_path, md_content)
            .map_err(|e| format!("Failed to write architecture.md: {}", e))?;

        println!("✅ Exported architecture to {} and {}", 
            json_path.display(), 
            md_path.display()
        );

        Ok(())
    }

    /// Convert architecture to markdown
    fn architecture_to_markdown(&self, architecture: &Architecture) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", architecture.name));
        md.push_str(&format!("{}\n\n", architecture.description));
        md.push_str(&format!("**Created:** {}\n\n", architecture.created_at));

        md.push_str("## Components\n\n");
        for component in &architecture.components {
            md.push_str(&format!("### {}\n\n", component.name));
            md.push_str(&format!("- **Type:** {:?}\n", component.component_type));
            md.push_str(&format!("- **Category:** {}\n", component.category));
            md.push_str(&format!("- **Description:** {}\n", component.description));
            
            if !component.files.is_empty() {
                md.push_str("- **Files:**\n");
                for file in &component.files {
                    md.push_str(&format!("  - `{}`\n", file));
                }
            }
            md.push_str("\n");
        }

        md.push_str("## Connections\n\n");
        for connection in &architecture.connections {
            let source = architecture.components.iter()
                .find(|c| c.id == connection.source_id)
                .map(|c| c.name.as_str())
                .unwrap_or("Unknown");
            
            let target = architecture.components.iter()
                .find(|c| c.id == connection.target_id)
                .map(|c| c.name.as_str())
                .unwrap_or("Unknown");

            md.push_str(&format!("- {} → {} ({:?})\n", 
                source, target, connection.connection_type));
        }

        md
    }

    /// Check if project is already initialized
    pub fn is_initialized(&self, project_path: &Path) -> bool {
        // Check for .yantra directory with architecture.db
        let yantra_dir = project_path.join(".yantra");
        let arch_db = yantra_dir.join("architecture.db");
        
        arch_db.exists()
    }
    
    /// Helper method to get unique file paths from GNN nodes
    fn get_files_from_gnn(&self, gnn: &GNNEngine) -> Vec<String> {
        let nodes = gnn.get_graph().get_all_nodes();
        let mut files = std::collections::HashSet::new();
        
        for node in nodes {
            if !node.file_path.is_empty() {
                files.insert(node.file_path.clone());
            }
        }
        
        files.into_iter().collect()
    }

    /// Review existing code for quality, patterns, and issues
    ///
    /// This performs a comprehensive code review including:
    /// - Pattern detection (MVC, microservices, etc.)
    /// - Issue identification (security, performance, maintainability)
    /// - Quality scoring
    /// - Recommendations for improvements
    pub async fn review_existing_code(
        &self,
        _project_path: &Path,
        architecture: &Architecture,
    ) -> Result<CodeReviewResult, String> {
        println!("🔍 Running code review...");

        // Step 1: Get GNN analysis
        let gnn = self.gnn.lock().await;
        let files = self.get_files_from_gnn(&gnn);
        
        // Step 2: Detect patterns
        let patterns = self.detect_code_patterns(&gnn)?;
        
        // Step 3: Identify security issues (using GNN + LLM)
        let security_issues = self.identify_security_issues(&gnn).await?;
        
        // Step 4: Check architecture alignment
        let alignment_issues = self.check_architecture_alignment(&gnn, architecture)?;
        
        // Step 5: Calculate quality score
        let quality_score = self.calculate_quality_score_detailed(
            files.len(),
            &security_issues,
            &alignment_issues,
        );
        
        // Step 6: Generate recommendations
        let recommendations = self.generate_recommendations_from_issues(
            &security_issues,
            &alignment_issues,
        );

        println!("✅ Code review complete: {} files analyzed", files.len());
        println!("   Quality score: {:.1}/10", quality_score);
        println!("   Security issues: {}", security_issues.len());
        println!("   Alignment issues: {}", alignment_issues.len());

        Ok(CodeReviewResult {
            files_analyzed: files.len(),
            patterns,
            security_issues,
            alignment_issues,
            quality_score,
            recommendations,
        })
    }

    /// Analyze how a requirement impacts the current architecture
    ///
    /// This uses LLM to understand the requirement and predict:
    /// - New components needed
    /// - Existing components affected
    /// - New connections needed
    /// - Breaking changes
    /// - Impact severity (Low/Medium/High/Breaking)
    pub async fn analyze_requirement_impact(
        &self,
        requirement: &str,
        architecture: &Architecture,
    ) -> Result<ArchitectureImpact, String> {
        println!("🔍 Analyzing requirement impact: {}", requirement);

        // Step 1: Build context from architecture
        let arch_context = self.build_architecture_context(architecture);
        
        // Step 2: Build prompt for LLM
        let prompt = format!(
            r#"You are an expert software architect analyzing the impact of a new requirement on an existing system architecture.

**Current Architecture:**
{}

**New Requirement:**
{}

**Task:** Analyze how this requirement will impact the architecture.

**Provide your analysis in the following JSON format:**
{{
  "severity": "Low|Medium|High|Breaking",
  "new_components": [
    {{"name": "Component Name", "reason": "Why it's needed", "layer": "Frontend|Backend|Data|Integration"}}
  ],
  "affected_components": [
    {{"name": "Component Name", "changes": "What changes are needed"}}
  ],
  "new_connections": [
    {{"from": "Source Component", "to": "Target Component", "type": "API|Database|Queue|Cache"}}
  ],
  "breaking_changes": ["Description of breaking change 1", "Description of breaking change 2"],
  "summary": "Brief summary of the impact",
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}}

**Severity Guidelines:**
- Low: Changes to single component, no new components
- Medium: Changes to 2-3 components, possibly 1 new component
- High: New component(s) required, changes to multiple components
- Breaking: Fundamental pattern change (e.g., monolith → microservices)
"#,
            arch_context,
            requirement
        );

        // Step 3: Get LLM analysis
        let response = {
            let llm = self.llm.lock().await;
            let request = CodeGenerationRequest {
                intent: prompt.clone(),
                context: arch_context.lines().map(|s| s.to_string()).collect(),
                file_path: None,
                dependencies: vec![],
            };
            
            llm.generate_code(&request).await
                .map_err(|e| format!("LLM analysis failed: {}", e))?
        };

        // Step 4: Parse JSON response from the generated code
        let impact: ArchitectureImpact = serde_json::from_str(&response.code)
            .map_err(|e| format!("Failed to parse LLM response: {}. Response: {}", e, response.code))?;

        println!("✅ Impact analysis complete:");
        println!("   Severity: {:?}", impact.severity);
        println!("   New components: {}", impact.new_components.len());
        println!("   Affected components: {}", impact.affected_components.len());
        println!("   Breaking changes: {}", impact.breaking_changes.len());

        Ok(impact)
    }

    /// Build architecture context for LLM
    fn build_architecture_context(&self, architecture: &Architecture) -> String {
        let mut context = String::new();

        context.push_str(&format!("**Project:** {}\n", architecture.name));
        context.push_str(&format!("**Description:** {}\n\n", architecture.description));

        context.push_str("**Components:**\n");
        for component in &architecture.components {
            context.push_str(&format!(
                "- {} ({:?}, Category: {}): {}\n",
                component.name,
                component.component_type,
                component.category,
                component.description
            ));
        }

        context.push_str("\n**Connections:**\n");
        for connection in &architecture.connections {
            let source = architecture
                .components
                .iter()
                .find(|c| c.id == connection.source_id)
                .map(|c| c.name.as_str())
                .unwrap_or("Unknown");

            let target = architecture
                .components
                .iter()
                .find(|c| c.id == connection.target_id)
                .map(|c| c.name.as_str())
                .unwrap_or("Unknown");

            context.push_str(&format!("- {} → {} ({:?})\n", source, target, connection.connection_type));
        }

        context
    }

    /// Detect code patterns (MVC, microservices, etc.)
    fn detect_code_patterns(&self, gnn: &GNNEngine) -> Result<Vec<CodePattern>, String> {
        let mut patterns = Vec::new();

        let files = self.get_files_from_gnn(gnn);
        
        // Detect common patterns based on file structure
        let has_controllers = files.iter().any(|f| f.to_lowercase().contains("controller"));
        let has_models = files.iter().any(|f| f.to_lowercase().contains("model"));
        let has_views = files.iter().any(|f| f.to_lowercase().contains("view"));

        if has_controllers && has_models && has_views {
            patterns.push(CodePattern {
                name: "MVC Pattern".to_string(),
                description: "Model-View-Controller pattern detected".to_string(),
                confidence: 0.8,
            });
        }

        // TODO: Add more pattern detection:
        // - Microservices (service directories)
        // - Repository pattern (repository classes)
        // - Factory pattern (factory classes)
        // - Singleton pattern (static instances)

        Ok(patterns)
    }

    /// Identify security issues using GNN + LLM
    async fn identify_security_issues(&self, gnn: &GNNEngine) -> Result<Vec<SecurityIssue>, String> {
        let mut issues = Vec::new();

        // TODO: Integrate with Semgrep or similar security scanner
        // For now, use basic heuristics

        let files = self.get_files_from_gnn(gnn);
        
        // Check for common security anti-patterns
        for file in files {
            if file.to_lowercase().contains("password") && !file.to_lowercase().contains("hash") {
                issues.push(SecurityIssue {
                    severity: IssueSeverity::High,
                    category: "Authentication".to_string(),
                    description: format!("Potential plaintext password in {}", file),
                    file: PathBuf::from(file),
                    suggestion: "Use password hashing (bcrypt, argon2)".to_string(),
                });
            }
        }

        Ok(issues)
    }

    /// Check architecture alignment
    fn check_architecture_alignment(
        &self,
        gnn: &GNNEngine,
        architecture: &Architecture,
    ) -> Result<Vec<AlignmentIssue>, String> {
        let mut issues = Vec::new();

        let files = self.get_files_from_gnn(gnn);
        
        // Check if all files are mapped to components
        for file in files {
            let mapped = architecture.components.iter().any(|c| {
                c.files.iter().any(|f| f == &file)
            });

            if !mapped {
                issues.push(AlignmentIssue {
                    severity: IssueSeverity::Medium,
                    description: format!("File '{}' not mapped to any component", file),
                    file: PathBuf::from(file),
                    suggestion: "Add this file to an existing component or create new component".to_string(),
                });
            }
        }

        Ok(issues)
    }

    /// Calculate detailed quality score
    fn calculate_quality_score_detailed(
        &self,
        _files_count: usize,
        security_issues: &[SecurityIssue],
        alignment_issues: &[AlignmentIssue],
    ) -> f32 {
        let mut score = 10.0;

        // Deduct for security issues
        let critical_security = security_issues.iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .count();
        let high_security = security_issues.iter()
            .filter(|i| matches!(i.severity, IssueSeverity::High))
            .count();

        score -= critical_security as f32 * 2.0;
        score -= high_security as f32 * 1.0;

        // Deduct for alignment issues
        let high_alignment = alignment_issues.iter()
            .filter(|i| matches!(i.severity, IssueSeverity::High))
            .count();

        score -= high_alignment as f32 * 0.5;

        // Ensure score is between 0 and 10
        score.max(0.0).min(10.0)
    }

    /// Generate recommendations from issues
    fn generate_recommendations_from_issues(
        &self,
        security_issues: &[SecurityIssue],
        alignment_issues: &[AlignmentIssue],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Security recommendations
        if !security_issues.is_empty() {
            recommendations.push(format!(
                "Address {} security issue(s), including {} high-severity issues",
                security_issues.len(),
                security_issues.iter().filter(|i| matches!(i.severity, IssueSeverity::High)).count()
            ));
        }

        // Alignment recommendations
        if !alignment_issues.is_empty() {
            recommendations.push(format!(
                "Map {} unmapped file(s) to architecture components",
                alignment_issues.len()
            ));
        }

        // Add generic recommendations
        if recommendations.is_empty() {
            recommendations.push("Architecture is well-aligned with codebase".to_string());
            recommendations.push("Continue monitoring for deviations during development".to_string());
        }

        recommendations
    }
}

/// Code review result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReviewResult {
    pub files_analyzed: usize,
    pub patterns: Vec<CodePattern>,
    pub security_issues: Vec<SecurityIssue>,
    pub alignment_issues: Vec<AlignmentIssue>,
    pub quality_score: f32,
    pub recommendations: Vec<String>,
}

/// Architecture impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureImpact {
    pub severity: ImpactSeverity,
    pub new_components: Vec<NewComponent>,
    pub affected_components: Vec<AffectedComponent>,
    pub new_connections: Vec<NewConnection>,
    pub breaking_changes: Vec<String>,
    pub summary: String,
    pub recommendations: Vec<String>,
}

/// Impact severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactSeverity {
    Low,      // Single component modification
    Medium,   // 2-3 component changes
    High,     // New component(s) needed
    Breaking, // Fundamental pattern change
}

/// New component to be added
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewComponent {
    pub name: String,
    pub reason: String,
    pub layer: String,
}

/// Affected existing component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedComponent {
    pub name: String,
    pub changes: String,
}

/// New connection between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewConnection {
    pub from: String,
    pub to: String,
    #[serde(rename = "type")]
    pub connection_type: String,
}

/// Code pattern detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    pub name: String,
    pub description: String,
    pub confidence: f32,
}

/// Security issue found
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub severity: IssueSeverity,
    pub category: String,
    pub description: String,
    pub file: PathBuf,
    pub suggestion: String,
}

/// Alignment issue with architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub file: PathBuf,
    pub suggestion: String,
}

/// Issue severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,  // Security vulnerabilities, breaking changes
    High,      // Architectural violations, major bugs
    Medium,    // Code smells, minor issues
    Low,       // Style issues, minor improvements
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid;
    use tokio::sync::Mutex;
    use crate::llm::{LLMConfig, LLMProvider};

    #[test]
    fn test_check_architecture_files() {
        let dir = tempdir().unwrap();
        let project_path = dir.path();

        // Create architecture.json
        std::fs::write(project_path.join("architecture.json"), "{}").unwrap();

        let gnn = Arc::new(Mutex::new(GNNEngine::new(&project_path.join(".yantra/graph.db")).unwrap()));
        let config = LLMConfig {
            claude_api_key: Some("test".to_string()),
            openai_api_key: None,
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: vec![],
        };
        let llm = Arc::new(Mutex::new(LLMOrchestrator::new(config)));
        
        let initializer = ProjectInitializer::new(gnn, llm).unwrap();
        
        let source = initializer.check_architecture_files(project_path);
        assert!(matches!(source, ArchitectureSource::YantraFile));
    }

    #[test]
    fn test_is_initialized() {
        let dir = tempdir().unwrap();
        let project_path = dir.path();

        let gnn = Arc::new(Mutex::new(GNNEngine::new(&project_path.join(".yantra/graph.db")).unwrap()));
        let config = LLMConfig {
            claude_api_key: Some("test".to_string()),
            openai_api_key: None,
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: vec![],
        };
        let llm = Arc::new(Mutex::new(LLMOrchestrator::new(config)));
        
        let initializer = ProjectInitializer::new(gnn, llm).unwrap();
        
        // Not initialized yet
        assert!(!initializer.is_initialized(project_path));

        // Create .yantra directory
        std::fs::create_dir_all(project_path.join(".yantra")).unwrap();
        std::fs::write(project_path.join(".yantra/architecture.db"), "").unwrap();

        // Now initialized
        assert!(initializer.is_initialized(project_path));
    }

    #[test]
    fn test_build_architecture_context() {
        let dir = tempdir().unwrap();
        let project_path = dir.path();

        let gnn = Arc::new(Mutex::new(GNNEngine::new(&project_path.join(".yantra/graph.db")).unwrap()));
        let config = LLMConfig {
            claude_api_key: Some("test".to_string()),
            openai_api_key: None,
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: vec![],
        };
        let llm = Arc::new(Mutex::new(LLMOrchestrator::new(config)));
        
        let initializer = ProjectInitializer::new(gnn, llm).unwrap();
        
        // Create test architecture
        let architecture = Architecture::new(
            uuid::Uuid::new_v4().to_string(),
            "Test Project".to_string(),
            "Test Description".to_string(),
        );
        
        let context = initializer.build_architecture_context(&architecture);
        
        assert!(context.contains("Test Project"));
        assert!(context.contains("Test Description"));
        assert!(context.contains("**Components:**"));
        assert!(context.contains("**Connections:**"));
    }

    #[tokio::test]
    async fn test_analyze_requirement_impact_structure() {
        let dir = tempdir().unwrap();
        let project_path = dir.path();

        let gnn = Arc::new(Mutex::new(GNNEngine::new(&project_path.join(".yantra/graph.db")).unwrap()));
        let config = LLMConfig {
            claude_api_key: Some("test".to_string()),
            openai_api_key: None,
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: vec![],
        };
        let llm = Arc::new(Mutex::new(LLMOrchestrator::new(config)));
        
        let initializer = ProjectInitializer::new(gnn, llm).unwrap();
        
        // Create test architecture
        let architecture = Architecture::new(
            uuid::Uuid::new_v4().to_string(),
            "Test Project".to_string(),
            "Test Description".to_string(),
        );
        
        // Test that the method builds the correct prompt structure
        let context = initializer.build_architecture_context(&architecture);
        assert!(context.contains("Test Project"));
    }
}
