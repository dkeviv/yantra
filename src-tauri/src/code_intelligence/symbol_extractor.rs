// File: src-tauri/src/code_intelligence/symbol_extractor.rs
// Purpose: Extract symbols (functions, classes, variables) from code
// Last Updated: December 9, 2025

use super::{Symbol, SymbolKind, detect_language};
use std::fs;
use tree_sitter::{Parser, Node};

pub struct SymbolExtractor {
    parser: Parser,
}

impl SymbolExtractor {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }
    
    /// Extract all symbols from a file
    pub fn extract_symbols(&mut self, file_path: &str) -> Result<Vec<Symbol>, String> {
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
        let mut symbols = Vec::new();
        
        self.extract_symbols_recursive(&root_node, &source_code, file_path, language_name, &mut symbols);
        
        Ok(symbols)
    }
    
    fn extract_symbols_recursive(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        language: &str,
        symbols: &mut Vec<Symbol>,
    ) {
        let kind = node.kind();
        
        // Extract symbol based on node type and language
        if let Some(symbol) = self.extract_symbol_from_node(node, source, file_path, language, kind) {
            symbols.push(symbol);
        }
        
        // Recurse into children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                self.extract_symbols_recursive(&child, source, file_path, language, symbols);
            }
        }
    }
    
    fn extract_symbol_from_node(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        language: &str,
        kind: &str,
    ) -> Option<Symbol> {
        let start = node.start_position();
        let end = node.end_position();
        
        match language {
            "rust" => self.extract_rust_symbol(node, source, file_path, kind, start.row + 1, end.row + 1, start.column, end.column),
            "python" => self.extract_python_symbol(node, source, file_path, kind, start.row + 1, end.row + 1, start.column, end.column),
            "javascript" | "typescript" => self.extract_js_symbol(node, source, file_path, kind, start.row + 1, end.row + 1, start.column, end.column),
            _ => None,
        }
    }
    
    fn extract_rust_symbol(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        kind: &str,
        start_line: usize,
        end_line: usize,
        start_column: usize,
        end_column: usize,
    ) -> Option<Symbol> {
        match kind {
            "function_item" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Function,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: node.utf8_text(source.as_bytes()).ok().map(|s| s.lines().next().unwrap_or("").to_string()),
                })
            },
            "struct_item" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Struct,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "impl_item" => {
                let type_node = node.child_by_field_name("type")?;
                let name = type_node.utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name: format!("impl {}", name),
                    kind: SymbolKind::Class,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "let_declaration" | "const_item" | "static_item" => {
                let pattern = node.child_by_field_name("pattern")?;
                let name = pattern.utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: if kind == "const_item" || kind == "static_item" { SymbolKind::Constant } else { SymbolKind::Variable },
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "use_declaration" => {
                let text = node.utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name: text.clone(),
                    kind: SymbolKind::Import,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: Some(text),
                })
            },
            _ => None,
        }
    }
    
    fn extract_python_symbol(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        kind: &str,
        start_line: usize,
        end_line: usize,
        start_column: usize,
        end_column: usize,
    ) -> Option<Symbol> {
        match kind {
            "function_definition" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                
                // Check if it's a method (inside a class)
                let mut parent = node.parent();
                let mut is_method = false;
                while let Some(p) = parent {
                    if p.kind() == "class_definition" {
                        is_method = true;
                        break;
                    }
                    parent = p.parent();
                }
                
                Some(Symbol {
                    name,
                    kind: if is_method { SymbolKind::Method } else { SymbolKind::Function },
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: node.child_by_field_name("parameters")
                        .and_then(|p| p.utf8_text(source.as_bytes()).ok())
                        .map(|s| s.to_string()),
                })
            },
            "class_definition" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Class,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "import_statement" | "import_from_statement" => {
                let text = node.utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name: text.clone(),
                    kind: SymbolKind::Import,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: Some(text),
                })
            },
            _ => None,
        }
    }
    
    fn extract_js_symbol(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        kind: &str,
        start_line: usize,
        end_line: usize,
        start_column: usize,
        end_column: usize,
    ) -> Option<Symbol> {
        match kind {
            "function_declaration" | "function" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Function,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "class_declaration" | "class" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Class,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "method_definition" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Method,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "variable_declarator" => {
                let name = node.child_by_field_name("name")?
                    .utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name,
                    kind: SymbolKind::Variable,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: None,
                })
            },
            "import_statement" => {
                let text = node.utf8_text(source.as_bytes()).ok()?.to_string();
                Some(Symbol {
                    name: text.clone(),
                    kind: SymbolKind::Import,
                    file_path: file_path.to_string(),
                    start_line,
                    end_line,
                    start_column,
                    end_column,
                    scope: None,
                    signature: Some(text),
                })
            },
            _ => None,
        }
    }
}
