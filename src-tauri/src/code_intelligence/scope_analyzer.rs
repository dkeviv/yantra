// File: src-tauri/src/code_intelligence/scope_analyzer.rs
// Purpose: Analyze scope information for code positions
// Last Updated: December 9, 2025

use super::{ScopeInfo, ScopeType, detect_language};
use std::fs;
use tree_sitter::{Parser, Node};

pub struct ScopeAnalyzer {
    parser: Parser,
}

impl ScopeAnalyzer {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }
    
    /// Get scope at a specific line and column in a file
    pub fn get_scope_at_position(
        &mut self,
        file_path: &str,
        line: usize,
        column: usize,
    ) -> Result<ScopeInfo, String> {
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
        
        // Convert line/column to byte offset (1-indexed to 0-indexed)
        let byte_offset = self.position_to_byte(&source_code, line - 1, column);
        
        // Find the innermost scope containing this position
        let scope = self.find_scope_at_byte(&root_node, byte_offset, &source_code, language_name)
            .unwrap_or_else(|| ScopeInfo {
                scope_type: ScopeType::Global,
                name: None,
                start_line: 1,
                end_line: source_code.lines().count(),
                parent_scope: None,
            });
        
        Ok(scope)
    }
    
    /// Get all scopes in a file
    pub fn get_all_scopes(&mut self, file_path: &str) -> Result<Vec<ScopeInfo>, String> {
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
        let mut scopes = Vec::new();
        
        self.collect_scopes_recursive(&root_node, &source_code, language_name, None, &mut scopes);
        
        Ok(scopes)
    }
    
    fn position_to_byte(&self, source: &str, line: usize, column: usize) -> usize {
        let mut byte_offset = 0;
        let mut current_line = 0;
        
        for ch in source.chars() {
            if current_line == line {
                if column == 0 {
                    return byte_offset;
                }
                return byte_offset + column.min(source.len() - byte_offset);
            }
            if ch == '\n' {
                current_line += 1;
            }
            byte_offset += ch.len_utf8();
        }
        
        byte_offset
    }
    
    fn find_scope_at_byte(
        &self,
        node: &Node,
        byte_offset: usize,
        source: &str,
        language: &str,
    ) -> Option<ScopeInfo> {
        // Check if this node contains the byte offset
        if node.start_byte() <= byte_offset && byte_offset <= node.end_byte() {
            // Check if this node defines a scope
            if let Some(scope_info) = self.node_to_scope(node, source, language) {
                // Try to find a more specific (inner) scope
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        if let Some(inner_scope) = self.find_scope_at_byte(&child, byte_offset, source, language) {
                            return Some(inner_scope);
                        }
                    }
                }
                return Some(scope_info);
            }
            
            // This node doesn't define a scope, but check children
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if let Some(scope) = self.find_scope_at_byte(&child, byte_offset, source, language) {
                        return Some(scope);
                    }
                }
            }
        }
        
        None
    }
    
    fn collect_scopes_recursive(
        &self,
        node: &Node,
        source: &str,
        language: &str,
        parent_name: Option<String>,
        scopes: &mut Vec<ScopeInfo>,
    ) {
        if let Some(mut scope_info) = self.node_to_scope(node, source, language) {
            scope_info.parent_scope = parent_name.clone();
            let current_name = scope_info.name.clone();
            scopes.push(scope_info);
            
            // Recurse with this scope as parent
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    self.collect_scopes_recursive(&child, source, language, current_name.clone(), scopes);
                }
            }
        } else {
            // Not a scope node, recurse with same parent
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    self.collect_scopes_recursive(&child, source, language, parent_name.clone(), scopes);
                }
            }
        }
    }
    
    fn node_to_scope(&self, node: &Node, source: &str, language: &str) -> Option<ScopeInfo> {
        let kind = node.kind();
        let start = node.start_position();
        let end = node.end_position();
        
        match language {
            "rust" => self.rust_node_to_scope(node, source, kind, start.row + 1, end.row + 1),
            "python" => self.python_node_to_scope(node, source, kind, start.row + 1, end.row + 1),
            "javascript" | "typescript" => self.js_node_to_scope(node, source, kind, start.row + 1, end.row + 1),
            _ => None,
        }
    }
    
    fn rust_node_to_scope(
        &self,
        node: &Node,
        source: &str,
        kind: &str,
        start_line: usize,
        end_line: usize,
    ) -> Option<ScopeInfo> {
        match kind {
            "function_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Function,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "impl_item" => {
                let type_node = node.child_by_field_name("type");
                let name = type_node
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| format!("impl {}", s));
                Some(ScopeInfo {
                    scope_type: ScopeType::Class,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "struct_item" | "enum_item" | "trait_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Class,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "block" => {
                Some(ScopeInfo {
                    scope_type: ScopeType::Block,
                    name: None,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "mod_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Module,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            _ => None,
        }
    }
    
    fn python_node_to_scope(
        &self,
        node: &Node,
        source: &str,
        kind: &str,
        start_line: usize,
        end_line: usize,
    ) -> Option<ScopeInfo> {
        match kind {
            "function_definition" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                
                // Check if it's a method (inside a class)
                let mut parent = node.parent();
                let mut scope_type = ScopeType::Function;
                while let Some(p) = parent {
                    if p.kind() == "class_definition" {
                        scope_type = ScopeType::Method;
                        break;
                    }
                    parent = p.parent();
                }
                
                Some(ScopeInfo {
                    scope_type,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "class_definition" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Class,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "module" => {
                Some(ScopeInfo {
                    scope_type: ScopeType::Module,
                    name: None,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            _ => None,
        }
    }
    
    fn js_node_to_scope(
        &self,
        node: &Node,
        source: &str,
        kind: &str,
        start_line: usize,
        end_line: usize,
    ) -> Option<ScopeInfo> {
        match kind {
            "function_declaration" | "function" | "arrow_function" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Function,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "class_declaration" | "class" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Class,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "method_definition" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());
                Some(ScopeInfo {
                    scope_type: ScopeType::Method,
                    name,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            "statement_block" => {
                Some(ScopeInfo {
                    scope_type: ScopeType::Block,
                    name: None,
                    start_line,
                    end_line,
                    parent_scope: None,
                })
            },
            _ => None,
        }
    }
}
