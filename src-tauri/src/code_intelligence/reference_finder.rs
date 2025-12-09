// File: src-tauri/src/code_intelligence/reference_finder.rs
// Purpose: Find references and definitions of symbols
// Last Updated: December 9, 2025

use super::{Reference, detect_language};
use std::fs;
use std::path::Path;
use tree_sitter::{Parser, Node};

pub struct ReferenceFinder {
    parser: Parser,
}
impl ReferenceFinder {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }
    
    /// Find all references to a symbol in a file
    pub fn find_references_in_file(
        &mut self,
        file_path: &str,
        symbol_name: &str,
    ) -> Result<Vec<Reference>, String> {
        let language_name = detect_language(file_path)
            .ok_or_else(|| format!("Unsupported file type: {}", file_path))?;
        
        let language = super::get_tree_sitter_language(language_name)?;
        self.parser.set_language(&language)
            .map_err(|e| format!("Failed to set language: {}", e))?;
        
        let source_code = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        let tree = self.parser.parse(&source_code, None)
            .ok_or_else(|| "Failed to parse file".to_string())?;
        
        let root_node = tree.root_node();
        let mut references = Vec::new();
        
        self.find_references_recursive(
            &root_node,
            &source_code,
            file_path,
            symbol_name,
            &mut references,
        );
        
        Ok(references)
    }
    
    /// Find all references to a symbol across project
    pub fn find_references_in_project(
        &mut self,
        project_path: &str,
        symbol_name: &str,
    ) -> Result<Vec<Reference>, String> {
        let mut all_references = Vec::new();
        
        self.walk_project(Path::new(project_path), symbol_name, &mut all_references)?;
        
        Ok(all_references)
    }
    
    /// Find definition of a symbol in a file
    pub fn find_definition_in_file(
        &mut self,
        file_path: &str,
        symbol_name: &str,
    ) -> Result<Option<Reference>, String> {
        let references = self.find_references_in_file(file_path, symbol_name)?;
        Ok(references.into_iter().find(|r| r.is_definition))
    }
    
    fn walk_project(
        &mut self,
        path: &Path,
        symbol_name: &str,
        references: &mut Vec<Reference>,
    ) -> Result<(), String> {
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ["rs", "py", "js", "ts", "jsx", "tsx"].contains(&ext.to_str().unwrap_or("")) {
                    if let Ok(file_refs) = self.find_references_in_file(
                        path.to_str().unwrap(),
                        symbol_name,
                    ) {
                        references.extend(file_refs);
                    }
                }
            }
        } else if path.is_dir() {
            // Skip common directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if ["node_modules", "target", ".git", "__pycache__", "venv", ".venv"].contains(&name) {
                    return Ok(());
                }
            }
            
            for entry in fs::read_dir(path).map_err(|e| format!("Failed to read dir: {}", e))? {
                let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
                self.walk_project(&entry.path(), symbol_name, references)?;
            }
        }
        
        Ok(())
    }
    
    fn find_references_recursive(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        symbol_name: &str,
        references: &mut Vec<Reference>,
    ) {
        // Check if this node is an identifier matching the symbol name
        if node.kind() == "identifier" {
            if let Ok(text) = node.utf8_text(source.as_bytes()) {
                if text == symbol_name {
                    let start = node.start_position();
                    
                    // Determine if this is a definition
                    let is_definition = self.is_definition_node(node);
                    
                    // Get context (line of code)
                    let line_start = source[..node.start_byte()]
                        .rfind('\n')
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let line_end = source[node.end_byte()..]
                        .find('\n')
                        .map(|i| node.end_byte() + i)
                        .unwrap_or(source.len());
                    let context = source[line_start..line_end].trim().to_string();
                    
                    references.push(Reference {
                        file_path: file_path.to_string(),
                        line: start.row + 1,
                        column: start.column,
                        context,
                        is_definition,
                    });
                }
            }
        }
        
        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.find_references_recursive(&child, source, file_path, symbol_name, references);
            }
        }
    }
    
    fn is_definition_node(&self, node: &Node) -> bool {
        // Check if this identifier is in a definition context
        let mut current = Some(*node);
        while let Some(n) = current {
            let kind = n.kind();
            match kind {
                "function_item" | "function_definition" | "function_declaration" |
                "struct_item" | "class_definition" | "class_declaration" |
                "let_declaration" | "const_item" | "variable_declarator" => {
                    // Check if our identifier is the name field
                    if let Some(name_node) = n.child_by_field_name("name") {
                        if name_node.id() == node.id() {
                            return true;
                        }
                    }
                    // Check if it's the pattern in let declaration
                    if kind == "let_declaration" {
                        if let Some(pattern_node) = n.child_by_field_name("pattern") {
                            if pattern_node.id() == node.id() || self.contains_node(&pattern_node, node) {
                                return true;
                            }
                        }
                    }
                    return false;
                },
                _ => current = n.parent(),
            }
        }
        false
    }
    
    fn contains_node(&self, parent: &Node, target: &Node) -> bool {
        if parent.id() == target.id() {
            return true;
        }
        for i in 0..parent.child_count() {
            if let Some(child) = parent.child(i) {
                if self.contains_node(&child, target) {
                    return true;
                }
            }
        }
        false
    }
}
