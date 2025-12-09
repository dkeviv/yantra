// YDoc File Operations Module
// Last updated: December 8, 2025
//
// This module handles file I/O operations for YDoc documents.
// Supports reading, writing, and exporting in various formats.

use std::path::{Path, PathBuf};
use std::fs;
use std::error::Error;
use std::fmt;
use crate::ydoc::parser::{YDocFile, YDocCell, CellMetadata};
use crate::ydoc::{DocumentType};

#[derive(Debug)]
pub enum FileOpsError {
    IoError(std::io::Error),
    ParseError(String),
    NotImplemented(String),
}

impl fmt::Display for FileOpsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FileOpsError::IoError(err) => write!(f, "IO error: {}", err),
            FileOpsError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            FileOpsError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
        }
    }
}

impl Error for FileOpsError {}

impl From<std::io::Error> for FileOpsError {
    fn from(err: std::io::Error) -> Self {
        FileOpsError::IoError(err)
    }
}

type Result<T> = std::result::Result<T, FileOpsError>;

pub struct YDocFileOps;

impl YDocFileOps {
    /// Initialize YDoc folder structure for a project
    pub fn initialize_ydoc_folder(project_root: &Path) -> Result<PathBuf> {
        let ydocs_path = project_root.join("ydocs");
        
        // Create main ydocs folder
        fs::create_dir_all(&ydocs_path)?;
        
        // Create subfolders for each document type
        let subfolders = [
            "requirements",    // REQ
            "adrs",            // ADR  
            "architecture",    // ARCH
            "specifications",  // SPEC
            "plans",           // PLAN
            "technical",       // TECH
            "api",             // API
            "user",            // USER
            "testing",         // TEST
            "results",         // RESULT
            "changes",         // CHANGE
            "decisions",       // DECISION
        ];
        
        for subfolder in &subfolders {
            let folder_path = ydocs_path.join(subfolder);
            fs::create_dir_all(&folder_path)?;
            
            // Create a MASTER.ydoc index file in each folder
            let master_file = folder_path.join("MASTER.ydoc");
            if !master_file.exists() {
                // Build the source lines properly
                let source_lines = vec![
                    format!("# {} Master Index", subfolder.to_uppercase()),
                    String::new(),
                    format!("This is the master index file for all {} documents.", subfolder),
                    String::new(),
                    "## Documents in this folder:".to_string(),
                    String::new(),
                    "- _Add links to documents here_".to_string(),
                ];
                
                // Create YDocFile using parser
                let mut master_doc = YDocFile::new(
                    format!("master-{}", subfolder),
                    match folder_to_doc_type(subfolder) {
                        "REQ" => crate::ydoc::DocumentType::Requirements,
                        "ADR" => crate::ydoc::DocumentType::ADR,
                        "ARCH" => crate::ydoc::DocumentType::Architecture,
                        "SPEC" => crate::ydoc::DocumentType::TechSpec,
                        "PLAN" => crate::ydoc::DocumentType::ProjectPlan,
                        "TECH" => crate::ydoc::DocumentType::TechGuide,
                        "API" => crate::ydoc::DocumentType::APIGuide,
                        "USER" => crate::ydoc::DocumentType::UserGuide,
                        "TEST" => crate::ydoc::DocumentType::TestingPlan,
                        "RESULT" => crate::ydoc::DocumentType::TestResults,
                        "CHANGE" => crate::ydoc::DocumentType::ChangeLog,
                        "DECISION" => crate::ydoc::DocumentType::DecisionsLog,
                        _ => crate::ydoc::DocumentType::Requirements,
                    },
                    format!("Master Index for {}", subfolder),
                    "system".to_string(),
                );
                
                // Add markdown cell with the content
                let cell = YDocCell {
                    cell_type: "markdown".to_string(),
                    source: source_lines,
                    metadata: CellMetadata::default(),
                    outputs: None,
                    execution_count: None,
                };
                master_doc.add_cell(cell);
                
                // Serialize and write
                let master_content = crate::ydoc::parser::serialize_ydoc(&master_doc)
                    .map_err(|e| FileOpsError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other, 
                        format!("Failed to serialize master doc: {}", e)
                    )))?;
                fs::write(&master_file, master_content)?;
            }
        }
        
        // Create a README.md in the ydocs root
        let readme_path = ydocs_path.join("README.md");
        if !readme_path.exists() {
            let readme_content = r#"# YDocs - Yantra Documentation System

This folder contains all project documentation in .ydoc format (Jupyter notebook compatible).

## Structure

- `requirements/` - Product requirements (REQ)
- `adrs/` - Architecture Decision Records (ADR)
- `architecture/` - System architecture documents (ARCH)
- `specifications/` - Technical specifications (SPEC)
- `plans/` - Project plans and roadmaps (PLAN)
- `technical/` - Technical documentation (TECH)
- `api/` - API documentation (API)
- `user/` - User guides (USER)
- `testing/` - Test plans and strategies (TEST)
- `results/` - Test results and reports (RESULT)
- `changes/` - Change logs (CHANGE)
- `decisions/` - Decision logs (DECISION)

## Usage

.ydoc files are Jupyter-notebook compatible and can be opened with:
- VS Code (with Jupyter extension)
- JupyterLab / Jupyter Notebook
- GitHub (renders automatically)

Each document contains blocks with metadata for traceability.
"#;
            fs::write(&readme_path, readme_content)?;
        }
        
        Ok(ydocs_path)
    }
    
    /// Export YDoc document to Markdown
    pub fn export_to_markdown(doc: &YDocFile) -> Result<String> {
        let mut markdown = String::new();
        
        // Add document header
        markdown.push_str(&format!("# {}\n\n", doc.metadata.yantra_title));
        markdown.push_str(&format!("**Type:** {}\n", doc.metadata.yantra_doc_type));
        markdown.push_str(&format!("**Version:** {}\n", doc.metadata.yantra_version));
        markdown.push_str(&format!("**Created:** {}\n", doc.metadata.created_at));
        markdown.push_str(&format!("**Modified:** {}\n\n", doc.metadata.modified_at));
        markdown.push_str("---\n\n");
        
        // Add each cell
        for cell in &doc.cells {
            match cell.cell_type.as_str() {
                "markdown" => {
                    markdown.push_str(&cell.get_content());
                    markdown.push_str("\n\n");
                    
                    // Add metadata if present
                    if let Some(yantra) = &cell.metadata.yantra {
                        markdown.push_str(&format!("<!-- Yantra ID: {} -->\n", yantra.yantra_id));
                        markdown.push_str(&format!("<!-- Type: {} -->\n", yantra.yantra_type));
                        markdown.push_str(&format!("<!-- Status: {} -->\n\n", yantra.status));
                    }
                }
                "code" => {
                    markdown.push_str("```\n");
                    markdown.push_str(&cell.get_content());
                    markdown.push_str("\n```\n\n");
                }
                "raw" => {
                    markdown.push_str(&cell.get_content());
                    markdown.push_str("\n\n");
                }
                _ => {}
            }
        }
        
        Ok(markdown)
    }
    
    /// Export YDoc document to HTML
    pub fn export_to_html(doc: &YDocFile) -> Result<String> {
        let mut html = String::new();
        
        // HTML header
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("  <title>{}</title>\n", doc.metadata.yantra_title));
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }\n");
        html.push_str("    .container { max-width: 900px; margin: 0 auto; padding: 20px; }\n");
        html.push_str("    .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }\n");
        html.push_str("    .metadata { color: #666; font-size: 0.9em; }\n");
        html.push_str("    .cell { margin: 20px 0; }\n");
        html.push_str("    .code { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }\n");
        html.push_str("    .yantra-meta { color: #999; font-size: 0.8em; border-top: 1px solid #eee; padding-top: 5px; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str("  <div class=\"container\">\n");
        
        // Document header
        html.push_str("    <div class=\"header\">\n");
        html.push_str(&format!("      <h1>{}</h1>\n", html_escape(&doc.metadata.yantra_title)));
        html.push_str("      <div class=\"metadata\">\n");
        html.push_str(&format!("        <strong>Type:</strong> {}<br>\n", doc.metadata.yantra_doc_type));
        html.push_str(&format!("        <strong>Version:</strong> {}<br>\n", doc.metadata.yantra_version));
        html.push_str(&format!("        <strong>Created:</strong> {}<br>\n", doc.metadata.created_at));
        html.push_str(&format!("        <strong>Modified:</strong> {}\n", doc.metadata.modified_at));
        html.push_str("      </div>\n");
        html.push_str("    </div>\n\n");
        
        // Cells
        for cell in &doc.cells {
            html.push_str("    <div class=\"cell\">\n");
            
            match cell.cell_type.as_str() {
                "markdown" => {
                    // Simple markdown to HTML (basic implementation)
                    let content = cell.get_content();
                    html.push_str("      <div class=\"markdown\">\n");
                    html.push_str(&markdown_to_html_simple(&content));
                    html.push_str("      </div>\n");
                }
                "code" => {
                    html.push_str("      <pre class=\"code\"><code>");
                    html.push_str(&html_escape(&cell.get_content()));
                    html.push_str("</code></pre>\n");
                }
                "raw" => {
                    html.push_str("      <div class=\"raw\">");
                    html.push_str(&html_escape(&cell.get_content()));
                    html.push_str("</div>\n");
                }
                _ => {}
            }
            
            // Add Yantra metadata
            if let Some(yantra) = &cell.metadata.yantra {
                html.push_str("      <div class=\"yantra-meta\">\n");
                html.push_str(&format!("        ID: {} | Type: {} | Status: {}\n", 
                    yantra.yantra_id, yantra.yantra_type, yantra.status));
                html.push_str("      </div>\n");
            }
            
            html.push_str("    </div>\n");
        }
        
        html.push_str("  </div>\n");
        html.push_str("</body>\n</html>");
        
        Ok(html)
    }
    
    /// Import Markdown file to YDoc format
    pub fn import_from_markdown(path: &Path, doc_type: DocumentType, doc_id: String) -> Result<YDocFile> {
        let content = fs::read_to_string(path)?;
        
        // Extract title from first heading or filename
        let title = content.lines()
            .find(|line| line.starts_with("# "))
            .map(|line| line.trim_start_matches("# ").to_string())
            .unwrap_or_else(|| path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Untitled")
                .to_string()
            );
        
        let mut doc = YDocFile::new(doc_id, doc_type, title, "user".to_string());
        
        // Split content into sections (simple implementation)
        let mut current_section = String::new();
        
        for line in content.lines() {
            if line.starts_with("##") && !current_section.is_empty() {
                // New section - save previous
                let cell = YDocCell::new_markdown(
                    current_section.trim().to_string(),
                    uuid::Uuid::new_v4().to_string(),
                    crate::ydoc::BlockType::Requirement,
                    "user".to_string(),
                );
                doc.add_cell(cell);
                current_section = String::new();
            }
            current_section.push_str(line);
            current_section.push('\n');
        }
        
        // Add final section
        if !current_section.trim().is_empty() {
            let cell = YDocCell::new_markdown(
                current_section.trim().to_string(),
                uuid::Uuid::new_v4().to_string(),
                crate::ydoc::BlockType::Requirement,
                "user".to_string(),
            );
            doc.add_cell(cell);
        }
        
        Ok(doc)
    }
    
    /// Get the appropriate subfolder for a document type
    pub fn get_folder_for_type(ydocs_root: &Path, doc_type: &DocumentType) -> PathBuf {
        let subfolder = match doc_type {
            DocumentType::Requirements => "requirements",
            DocumentType::ADR => "adrs",
            DocumentType::Architecture => "architecture",
            DocumentType::TechSpec => "specifications",
            DocumentType::ProjectPlan => "plans",
            DocumentType::TechGuide => "technical",
            DocumentType::APIGuide => "api",
            DocumentType::UserGuide => "user",
            DocumentType::TestingPlan => "testing",
            DocumentType::TestResults => "results",
            DocumentType::ChangeLog => "changes",
            DocumentType::DecisionsLog => "decisions",
        };
        
        ydocs_root.join(subfolder)
    }
}

/// Helper: Map folder name to DocumentType code
fn folder_to_doc_type(folder: &str) -> &str {
    match folder {
        "requirements" => "REQ",
        "adrs" => "ADR",
        "architecture" => "ARCH",
        "specifications" => "SPEC",
        "plans" => "PLAN",
        "technical" => "TECH",
        "api" => "API",
        "user" => "USER",
        "testing" => "TEST",
        "results" => "RESULT",
        "changes" => "CHANGE",
        "decisions" => "DECISION",
        _ => "UNKNOWN",
    }
}

/// Helper: Simple HTML escaping
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Helper: Very simple markdown to HTML conversion
fn markdown_to_html_simple(markdown: &str) -> String {
    let mut html = String::new();
    
    for line in markdown.lines() {
        if line.starts_with("# ") {
            html.push_str(&format!("        <h1>{}</h1>\n", html_escape(&line[2..])));
        } else if line.starts_with("## ") {
            html.push_str(&format!("        <h2>{}</h2>\n", html_escape(&line[3..])));
        } else if line.starts_with("### ") {
            html.push_str(&format!("        <h3>{}</h3>\n", html_escape(&line[4..])));
        } else if line.starts_with("- ") {
            html.push_str(&format!("        <li>{}</li>\n", html_escape(&line[2..])));
        } else if !line.is_empty() {
            html.push_str(&format!("        <p>{}</p>\n", html_escape(line)));
        } else {
            html.push_str("        <br>\n");
        }
    }
    
    html
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::ydoc::DocumentType;

    #[test]
    fn test_initialize_ydoc_folder() {
        let temp_dir = TempDir::new().unwrap();
        let result = YDocFileOps::initialize_ydoc_folder(temp_dir.path());
        
        assert!(result.is_ok());
        let ydocs_path = result.unwrap();
        assert!(ydocs_path.exists());
        assert!(ydocs_path.join("requirements").exists());
        assert!(ydocs_path.join("adrs").exists());
        assert!(ydocs_path.join("README.md").exists());
        assert!(ydocs_path.join("requirements/MASTER.ydoc").exists());
    }

    #[test]
    fn test_export_markdown() {
        let mut doc = YDocFile::new(
            "test-123".to_string(),
            DocumentType::Requirements,
            "Test Requirements".to_string(),
            "user".to_string(),
        );
        
        let cell = YDocCell::new_markdown(
            "# Requirement 1\nUser must login".to_string(),
            "block-1".to_string(),
            crate::ydoc::BlockType::Requirement,
            "user".to_string(),
        );
        doc.add_cell(cell);
        
        let markdown = YDocFileOps::export_to_markdown(&doc).unwrap();
        assert!(markdown.contains("Test Requirements"));
        assert!(markdown.contains("Requirement 1"));
        assert!(markdown.contains("REQ"));
    }

    #[test]
    fn test_export_html() {
        let mut doc = YDocFile::new(
            "test-456".to_string(),
            DocumentType::ADR,
            "Architecture Decision".to_string(),
            "agent".to_string(),
        );
        
        let cell = YDocCell::new_markdown(
            "## Decision\nUse PostgreSQL".to_string(),
            "block-2".to_string(),
            crate::ydoc::BlockType::ADR,
            "agent".to_string(),
        );
        doc.add_cell(cell);
        
        let html = YDocFileOps::export_to_html(&doc).unwrap();
        assert!(html.contains("<html>"));
        assert!(html.contains("Architecture Decision"));
        assert!(html.contains("PostgreSQL"));
    }

    #[test]
    fn test_get_folder_for_type() {
        let root = Path::new("/ydocs");
        
        let req_folder = YDocFileOps::get_folder_for_type(&root, &DocumentType::Requirements);
        assert_eq!(req_folder, Path::new("/ydocs/requirements"));
        
        let adr_folder = YDocFileOps::get_folder_for_type(&root, &DocumentType::ADR);
        assert_eq!(adr_folder, Path::new("/ydocs/adrs"));
    }
}
