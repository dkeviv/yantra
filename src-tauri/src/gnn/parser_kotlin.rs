// File: src-tauri/src/gnn/parser_kotlin.rs
// Purpose: tree-sitter Kotlin parser for GNN
// Dependencies: tree-sitter, tree-sitter-kotlin
// Last Updated: November 30, 2025

use super::{CodeNode, CodeEdge, EdgeType, NodeType};
use std::path::Path;
use tree_sitter::{Parser, Tree};

/// Parse a Kotlin file and extract nodes and edges
pub fn parse_kotlin_file(
    code: &str,
    file_path: &Path,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_kotlin::language())
        .map_err(|e| format!("Failed to set Kotlin language: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse Kotlin code".to_string())?;

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
            "function_declaration" => {
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
                            ..Default::default()
                        });
                    }
                }
            }
            "class_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::class::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Class,
                            name: name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            ..Default::default()
                        });
                    }
                }
            }
            "object_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::object::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Class,
                            name: name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            ..Default::default()
                        });
                    }
                }
            }
            "interface_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let end = node.end_position();
                        let id = format!("{}::interface::{}::{}", file_path, name, start.row);

                        nodes.push(CodeNode {
                            id,
                            node_type: NodeType::Class,
                            name: name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: end.row + 1,
                            ..Default::default()
                        });
                    }
                }
            }
            "import_header" => {
                if let Some(import_node) = node.child_by_field_name("identifier") {
                    if let Ok(import_name) = import_node.utf8_text(code.as_bytes()) {
                        let start = node.start_position();
                        let id = format!("{}::import::{}::{}", file_path, import_name, start.row);

                        nodes.push(CodeNode {
                            id: id.clone(),
                            node_type: NodeType::Import,
                            name: import_name.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: start.row + 1,
                            ..Default::default()
                        });

                        edges.push(CodeEdge {
                            edge_type: EdgeType::Imports,
                            source_id: id,
                            target_id: import_name.to_string(),
                        });
                    }
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
    fn test_parse_kotlin_function() {
        let code = r#"
fun add(a: Int, b: Int): Int {
    return a + b
}
"#;
        let path = PathBuf::from("/test/calc.kt");
        let (nodes, _edges) = parse_kotlin_file(code, &path).unwrap();

        let func_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Function))
            .collect();

        assert_eq!(func_nodes.len(), 1);
        assert_eq!(func_nodes[0].name, "add");
    }

    #[test]
    fn test_parse_kotlin_class() {
        let code = r#"
class Point(val x: Int, val y: Int) {
    fun distance(): Double {
        return Math.sqrt((x * x + y * y).toDouble())
    }
}
"#;
        let path = PathBuf::from("/test/point.kt");
        let (nodes, _edges) = parse_kotlin_file(code, &path).unwrap();

        let class_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Class))
            .collect();

        assert_eq!(class_nodes.len(), 1);
        assert_eq!(class_nodes[0].name, "Point");
    }

    #[test]
    fn test_parse_kotlin_import() {
        let code = r#"
import kotlin.math.sqrt
import java.util.Date
"#;
        let path = PathBuf::from("/test/main.kt");
        let (nodes, edges) = parse_kotlin_file(code, &path).unwrap();

        let import_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Import))
            .collect();

        assert_eq!(import_nodes.len(), 2);
        assert_eq!(edges.len(), 2);
    }
}
