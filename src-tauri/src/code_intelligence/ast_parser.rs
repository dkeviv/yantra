// File: src-tauri/src/code_intelligence/ast_parser.rs
// Purpose: AST parsing using tree-sitter
// Last Updated: December 9, 2025

use super::{ASTNode, detect_language};
use std::fs;
use tree_sitter::{Parser, Node};

pub struct ASTParser {
    parser: Parser,
}

impl ASTParser {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }
    
    /// Parse file and return AST
    pub fn parse_file(&mut self, file_path: &str) -> Result<ASTNode, String> {
        let language_name = super::detect_language(file_path)
            .ok_or_else(|| format!("Unsupported file extension: {}", file_path))?;
        
        let language = super::get_tree_sitter_language(language_name)?;
        self.parser.set_language(&language)
            .map_err(|e| format!("Failed to set language: {}", e))?;
        
        let source_code = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        let tree = self.parser.parse(&source_code, None)
            .ok_or_else(|| "Failed to parse file".to_string())?;
        
        Ok(self.convert_node_to_ast(&tree.root_node(), &source_code))
    }

    /// Parse code snippet
    pub fn parse_snippet(&mut self, code: &str, language: &str) -> Result<ASTNode, String> {
        let ts_language = super::get_tree_sitter_language(language)?;
        self.parser.set_language(&ts_language)
            .map_err(|e| format!("Failed to set language: {}", e))?;
        
        let tree = self.parser.parse(code, None)
            .ok_or_else(|| "Failed to parse code".to_string())?;
        
        let root_node = tree.root_node();
        Ok(self.convert_node_to_ast(&root_node, code))
    }
    
    fn convert_node_to_ast(&self, node: &Node, source: &str) -> ASTNode {
        let start = node.start_position();
        let end = node.end_position();
        
        let text = if node.child_count() == 0 {
            node.utf8_text(source.as_bytes()).ok().map(|s| s.to_string())
        } else {
            None
        };
        
        let children: Vec<ASTNode> = (0..node.child_count())
            .filter_map(|i| node.child(i))
            .map(|child| self.convert_node_to_ast(&child, source))
            .collect();
        
        ASTNode {
            node_type: node.kind().to_string(),
            start_line: start.row + 1,
            end_line: end.row + 1,
            start_column: start.column,
            end_column: end.column,
            text,
            children,
        }
    }
}
