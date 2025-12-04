// Document Readers: Extract text from DOCX and PDF files
// Purpose: Enable architecture analysis from documentation files

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Document content extracted from file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentContent {
    pub file_path: String,
    pub content_type: String,
    pub text: String,
    pub metadata: DocumentMetadata,
    pub tables: Vec<Table>,
    pub images: Vec<ImageInfo>,
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub created_at: Option<String>,
    pub modified_at: Option<String>,
    pub page_count: Option<usize>,
}

/// Table extracted from document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub rows: Vec<Vec<String>>,
    pub headers: Option<Vec<String>>,
}

/// Image information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    pub index: usize,
    pub alt_text: Option<String>,
    pub dimensions: Option<(u32, u32)>,
}

/// DOCX Reader
pub struct DocxReader;

impl DocxReader {
    /// Read DOCX file
    pub fn read(file_path: &Path) -> Result<DocumentContent, String> {
        let file = fs::File::open(file_path)
            .map_err(|e| format!("Failed to open DOCX file: {}", e))?;
        
        let mut docx = docx_rs::read_docx(&file)
            .map_err(|e| format!("Failed to parse DOCX: {}", e))?;
        
        // Extract text content
        let text = Self::extract_text(&docx);
        
        // Extract tables
        let tables = Self::extract_tables(&docx);
        
        // Extract metadata
        let metadata = Self::extract_metadata(&docx);
        
        // Extract image info
        let images = Self::extract_images(&docx);
        
        Ok(DocumentContent {
            file_path: file_path.to_string_lossy().to_string(),
            content_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
            text,
            metadata,
            tables,
            images,
        })
    }
    
    /// Extract text from DOCX
    fn extract_text(docx: &docx_rs::Docx) -> String {
        let mut text = String::new();
        
        for child in &docx.document.children {
            match child {
                docx_rs::DocumentChild::Paragraph(para) => {
                    for run in &para.children {
                        if let docx_rs::ParagraphChild::Run(r) = run {
                            for child in &r.children {
                                if let docx_rs::RunChild::Text(t) = child {
                                    text.push_str(&t.text);
                                }
                            }
                        }
                    }
                    text.push('\n');
                }
                docx_rs::DocumentChild::Table(table) => {
                    // Tables handled separately
                    text.push_str("[TABLE]\n");
                }
                _ => {}
            }
        }
        
        text
    }
    
    /// Extract tables from DOCX
    fn extract_tables(docx: &docx_rs::Docx) -> Vec<Table> {
        let mut tables = Vec::new();
        
        for child in &docx.document.children {
            if let docx_rs::DocumentChild::Table(table) = child {
                let mut rows = Vec::new();
                
                for row in &table.rows {
                    let mut cells = Vec::new();
                    
                    for cell in &row.cells {
                        let mut cell_text = String::new();
                        
                        for child in &cell.children {
                            if let docx_rs::TableCellContent::Paragraph(para) = child {
                                for run in &para.children {
                                    if let docx_rs::ParagraphChild::Run(r) = run {
                                        for child in &r.children {
                                            if let docx_rs::RunChild::Text(t) = child {
                                                cell_text.push_str(&t.text);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        cells.push(cell_text);
                    }
                    
                    rows.push(cells);
                }
                
                // Assume first row is header if table has > 1 row
                let headers = if rows.len() > 1 {
                    Some(rows[0].clone())
                } else {
                    None
                };
                
                tables.push(Table { rows, headers });
            }
        }
        
        tables
    }
    
    /// Extract metadata from DOCX
    fn extract_metadata(docx: &docx_rs::Docx) -> DocumentMetadata {
        DocumentMetadata {
            title: docx.doc_props.core.as_ref().and_then(|c| c.title.clone()),
            author: docx.doc_props.core.as_ref().and_then(|c| c.creator.clone()),
            created_at: docx.doc_props.core.as_ref().and_then(|c| c.created.clone()),
            modified_at: docx.doc_props.core.as_ref().and_then(|c| c.modified.clone()),
            page_count: None, // Not easily available in docx-rs
        }
    }
    
    /// Extract image information
    fn extract_images(docx: &docx_rs::Docx) -> Vec<ImageInfo> {
        let mut images = Vec::new();
        let mut index = 0;
        
        for child in &docx.document.children {
            if let docx_rs::DocumentChild::Paragraph(para) = child {
                for run in &para.children {
                    if let docx_rs::ParagraphChild::Run(r) = run {
                        for child in &r.children {
                            if let docx_rs::RunChild::Drawing(_) = child {
                                images.push(ImageInfo {
                                    index,
                                    alt_text: None,
                                    dimensions: None,
                                });
                                index += 1;
                            }
                        }
                    }
                }
            }
        }
        
        images
    }
}

/// PDF Reader
pub struct PdfReader;

impl PdfReader {
    /// Read PDF file
    pub fn read(file_path: &Path) -> Result<DocumentContent, String> {
        let bytes = fs::read(file_path)
            .map_err(|e| format!("Failed to read PDF file: {}", e))?;
        
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| format!("Failed to extract PDF text: {}", e))?;
        
        // Extract metadata using lopdf
        let metadata = Self::extract_metadata(file_path)?;
        
        Ok(DocumentContent {
            file_path: file_path.to_string_lossy().to_string(),
            content_type: "application/pdf".to_string(),
            text,
            metadata,
            tables: Vec::new(), // Table extraction from PDF is complex
            images: Vec::new(), // Image extraction from PDF is complex
        })
    }
    
    /// Extract metadata from PDF
    fn extract_metadata(file_path: &Path) -> Result<DocumentMetadata, String> {
        let doc = lopdf::Document::load(file_path)
            .map_err(|e| format!("Failed to load PDF: {}", e))?;
        
        let mut metadata = DocumentMetadata {
            title: None,
            author: None,
            created_at: None,
            modified_at: None,
            page_count: Some(doc.get_pages().len()),
        };
        
        // Try to extract metadata
        if let Ok(info) = doc.trailer.get(b"Info") {
            if let Ok(info_dict) = info.as_dict() {
                if let Ok(title) = info_dict.get(b"Title") {
                    if let Ok(s) = title.as_string() {
                        metadata.title = Some(String::from_utf8_lossy(&s).to_string());
                    }
                }
                
                if let Ok(author) = info_dict.get(b"Author") {
                    if let Ok(s) = author.as_string() {
                        metadata.author = Some(String::from_utf8_lossy(&s).to_string());
                    }
                }
                
                if let Ok(created) = info_dict.get(b"CreationDate") {
                    if let Ok(s) = created.as_string() {
                        metadata.created_at = Some(String::from_utf8_lossy(&s).to_string());
                    }
                }
                
                if let Ok(modified) = info_dict.get(b"ModDate") {
                    if let Ok(s) = modified.as_string() {
                        metadata.modified_at = Some(String::from_utf8_lossy(&s).to_string());
                    }
                }
            }
        }
        
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_document_content_structure() {
        let content = DocumentContent {
            file_path: "test.docx".to_string(),
            content_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
            text: "Sample text".to_string(),
            metadata: DocumentMetadata {
                title: Some("Test Document".to_string()),
                author: Some("Test Author".to_string()),
                created_at: None,
                modified_at: None,
                page_count: Some(1),
            },
            tables: vec![],
            images: vec![],
        };
        
        assert_eq!(content.file_path, "test.docx");
        assert_eq!(content.text, "Sample text");
        assert_eq!(content.metadata.title, Some("Test Document".to_string()));
    }
    
    #[test]
    fn test_table_structure() {
        let table = Table {
            rows: vec![
                vec!["Header 1".to_string(), "Header 2".to_string()],
                vec!["Row 1 Col 1".to_string(), "Row 1 Col 2".to_string()],
            ],
            headers: Some(vec!["Header 1".to_string(), "Header 2".to_string()]),
        };
        
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.headers.as_ref().unwrap().len(), 2);
    }
}
