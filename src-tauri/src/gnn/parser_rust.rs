// File: src-tauri/src/gnn/parser_rust.rs
// Purpose: tree-sitter Rust parser for GNN
// Dependencies: tree-sitter, tree-sitter-rust
// Last Updated: November 30, 2025

use super::{CodeNode, CodeEdge, EdgeType, NodeType};
use std::path::Path;
use tree_sitter::{Parser, Tree};

/// Parse a Rust file and extract nodes and edges
pub fn parse_rust_file(
    code: &str,
    file_path: &Path,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::language())
        .map_err(|e| format!("Failed to set Rust language: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse Rust code".to_string())?;

    extract_nodes_and_edges(code, file_path, &tree)
}

/// Extract nodes and edges from parsed tree
fn extract_nodes_and_edges(
    code: &str,
    file_path: &Path,
    tree: &Tree,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let file_path_str = file_path.to_str().unwrap_or("");

    let root = tree.root_node();
    let mut cursor = root.walk();

    fn visit_node(
        node: &tree_sitter::Node,
        code: &str,
        file_path: &str,
        nodes: &mut Vec<CodeNode>,
        edges: &mut Vec<CodeEdge>,
    ) {
        match node.kind() {
            "function_item" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Function,
                            name: name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            code_snippet: extract_code_snippet(node, code),
                            docstring: extract_docstring(node, code),
                            ..Default::default()
                        });
                    }
                }
            }
            "struct_item" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Class,
                            name: name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            code_snippet: extract_code_snippet(node, code),
                            docstring: extract_docstring(node, code),
                            ..Default::default()
                        });
                    }
                }
            }
            "impl_item" => {
                if let Some(name_node) = node.child_by_field_name("type") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::impl::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Class,
                            name: format!("impl {}", name),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            ..Default::default()
                        });
                    }
                }
            }
            "trait_item" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::trait::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Class,
                            name: name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            code_snippet: extract_code_snippet(node, code),
                            docstring: extract_docstring(node, code),
                            ..Default::default()
                        });
                    }
                }
            }
            "use_declaration" => {
                if let Ok(use_text) = node.utf8_text(code.as_bytes()) {
                    let use_path = use_text
                        .trim_start_matches("use ")
                        .trim_end_matches(';')
                        .trim();
                    let start = node.start_position();
                    let id = format!("{}::use::{}::{}", file_path, use_path, start.row);

                    nodes.push(CodeNode {
                        id: id.clone(),
                        node_type: NodeType::Import,
                        name: use_path.to_string(),
                        file_path: file_path.to_string(),
                        line_start: start.row + 1,
                        line_end: start.row + 1,
                            code_snippet: extract_code_snippet(node, code),
                            docstring: None,
                            ..Default::default()
                    });

                    edges.push(CodeEdge {
                        edge_type: EdgeType::Imports,
                        source_id: id,
                        target_id: use_path.to_string(),
                    });
                }
            }
            _ => {}
        }

        // Recursively visit children
        let mut child_cursor = node.walk();
        for child in node.children(&mut child_cursor) {
            visit_node(&child, code, file_path, nodes, edges);
        }
    }

    visit_node(&root, code, file_path_str, &mut nodes, &mut edges);

    Ok((nodes, edges))
}


/// Extract code snippet from a node (with reasonable size limit)
fn extract_code_snippet(node: &tree_sitter::Node, code: &str) -> Option<String> {
    const MAX_SNIPPET_LENGTH: usize = 1000; // 1KB max per node
    
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    
    if start_byte >= code.len() || end_byte > code.len() || start_byte >= end_byte {
        return None;
    }
    
    let snippet = &code[start_byte..end_byte];
    
    // Truncate if too long
    if snippet.len() > MAX_SNIPPET_LENGTH {
        Some(format!("{}... [truncated]", &snippet[..900]))
    } else if !snippet.is_empty() {
        Some(snippet.to_string())
    } else {
        None
    }
}

/// Extract docstring/comment from a node
fn extract_docstring(node: &tree_sitter::Node, code: &str) -> Option<String> {
    // Look for comment or documentation nodes
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind.contains("comment") || kind.contains("doc") || kind.contains("string") {
            let text = match child.utf8_text(code.as_bytes()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            
            // Clean up common doc patterns
            let cleaned = text
                .trim()
                .trim_start_matches("/**")
                .trim_start_matches("/*")
                .trim_start_matches("//")
                .trim_start_matches('#')
                .trim_start_matches("\"\"\"")
                .trim_end_matches("*/")
                .trim_end_matches("\"\"\"")
                .trim()
                .to_string();
            
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_rust_function() {
        let code = r#"
fn calculate(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let path = PathBuf::from("/test/calc.rs");
        let (nodes, _edges) = parse_rust_file(code, &path).unwrap();

        let func_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Function))
            .collect();

        assert_eq!(func_nodes.len(), 1);
        assert_eq!(func_nodes[0].name, "calculate");
    }

    #[test]
    fn test_parse_rust_struct() {
        let code = r#"
struct Point {
    x: i32,
    y: i32,
}
"#;
        let path = PathBuf::from("/test/point.rs");
        let (nodes, _edges) = parse_rust_file(code, &path).unwrap();

        let struct_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Class))
            .collect();

        assert_eq!(struct_nodes.len(), 1);
        assert_eq!(struct_nodes[0].name, "Point");
    }

    #[test]
    fn test_parse_rust_use() {
        let code = r#"
use std::collections::HashMap;
use super::module;
"#;
        let path = PathBuf::from("/test/imports.rs");
        let (nodes, edges) = parse_rust_file(code, &path).unwrap();

        let import_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Import))
            .collect();

        assert_eq!(import_nodes.len(), 2);
        assert_eq!(edges.len(), 2);
    }
}
