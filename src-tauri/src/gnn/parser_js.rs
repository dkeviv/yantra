// File: src-tauri/src/gnn/parser_js.rs
// Purpose: tree-sitter JavaScript/TypeScript parser for GNN
// Dependencies: tree-sitter, tree-sitter-javascript, tree-sitter-typescript
// Last Updated: November 25, 2025

use super::{CodeNode, CodeEdge, EdgeType, NodeType};
use std::path::Path;
use tree_sitter::{Parser, Tree};

/// Parse a JavaScript file and extract nodes and edges
pub fn parse_javascript_file(
    code: &str,
    file_path: &Path,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_javascript::language())
        .map_err(|e| format!("Failed to set JavaScript language: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse JavaScript code".to_string())?;

    extract_nodes_and_edges(code, file_path, &tree, false)
}

/// Parse a TypeScript file and extract nodes and edges
pub fn parse_typescript_file(
    code: &str,
    file_path: &Path,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_typescript::language_typescript())
        .map_err(|e| format!("Failed to set TypeScript language: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse TypeScript code".to_string())?;

    extract_nodes_and_edges(code, file_path, &tree, true)
}

/// Parse a TSX file (TypeScript + JSX)
pub fn parse_tsx_file(
    code: &str,
    file_path: &Path,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_typescript::language_tsx())
        .map_err(|e| format!("Failed to set TSX language: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse TSX code".to_string())?;

    extract_nodes_and_edges(code, file_path, &tree, true)
}

/// Extract nodes and edges from parsed tree
fn extract_nodes_and_edges(
    code: &str,
    file_path: &Path,
    tree: &Tree,
    _is_typescript: bool,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let file_path_str = file_path.to_str().unwrap_or("");

    // Walk the tree manually for now (simpler than complex queries)
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
                        });
                    }
                }
            }
            "class_declaration" => {
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
                        });
                    }
                }
            }
            "import_statement" => {
                if let Some(source_node) = node.child_by_field_name("source") {
                    if let Ok(source_text) = source_node.utf8_text(code.as_bytes()) {
                        let source_module = source_text.trim_matches(|c| c == '"' || c == '\'');
                        let start = node.start_position();
                        let id = format!("{}::import::{}::{}", file_path, source_module, start.row);

                        nodes.push(CodeNode {
                            id: id.clone(),
                            node_type: NodeType::Import,
                            name: source_module.to_string(),
                            file_path: file_path.to_string(),
                            line_start: start.row + 1,
                            line_end: start.row + 1,
                        });

                        edges.push(CodeEdge {
                            edge_type: EdgeType::Imports,
                            source_id: id,
                            target_id: source_module.to_string(),
                        });
                    }
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                // Extract variable declarations
                let mut child_cursor = node.walk();
                for child in node.children(&mut child_cursor) {
                    if child.kind() == "variable_declarator" {
                        if let Some(name_node) = child.child_by_field_name("name") {
                            if let Ok(name) = name_node.utf8_text(code.as_bytes()) {
                                // Skip very short names
                                if name.len() >= 2 {
                                    let start = child.start_position();
                                    let id = format!("{}::{}::{}", file_path, name, start.row);

                                    nodes.push(CodeNode {
                                        id,
                                        node_type: NodeType::Variable,
                                        name: name.to_string(),
                                        file_path: file_path.to_string(),
                                        line_start: start.row + 1,
                                        line_end: start.row + 1,
                                    });
                                }
                            }
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_javascript_function() {
        let code = r#"
function add(a, b) {
    return a + b;
}

const multiply = (x, y) => x * y;
"#;
        let path = PathBuf::from("/test/math.js");
        let (nodes, _edges) = parse_javascript_file(code, &path).unwrap();

        // Should find at least the function declarations
        assert!(nodes.len() >= 2);
        
        let func_names: Vec<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(func_names.contains(&"add"));
        assert!(func_names.contains(&"multiply"));
    }

    #[test]
    fn test_parse_javascript_class() {
        let code = r#"
class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
}
"#;
        let path = PathBuf::from("/test/calculator.js");
        let (nodes, _edges) = parse_javascript_file(code, &path).unwrap();

        // Should find the class
        let class_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Class))
            .collect();
        
        assert_eq!(class_nodes.len(), 1);
        assert_eq!(class_nodes[0].name, "Calculator");
    }

    #[test]
    fn test_parse_javascript_imports() {
        let code = r#"
import { add, subtract } from './math';
import React from 'react';
"#;
        let path = PathBuf::from("/test/app.js");
        let (nodes, edges) = parse_javascript_file(code, &path).unwrap();

        // Should find import nodes
        let import_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Import))
            .collect();
        
        assert!(import_nodes.len() >= 1);
        
        // Should have import edges
        assert!(edges.len() >= 1);
        
        let import_edges: Vec<&CodeEdge> = edges
            .iter()
            .filter(|e| matches!(e.edge_type, EdgeType::Imports))
            .collect();
        
        assert!(import_edges.len() >= 1);
    }

    #[test]
    fn test_parse_typescript_function() {
        let code = r#"
function greet(name: string): string {
    return `Hello, ${name}!`;
}

const add = (a: number, b: number): number => a + b;
"#;
        let path = PathBuf::from("/test/utils.ts");
        let (nodes, _edges) = parse_typescript_file(code, &path).unwrap();

        // Should find functions
        let func_names: Vec<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(func_names.contains(&"greet"));
        assert!(func_names.contains(&"add"));
    }

    #[test]
    fn test_parse_tsx_component() {
        let code = r#"
import React from 'react';

function Button({ label }: { label: string }) {
    return <button>{label}</button>;
}

export default Button;
"#;
        let path = PathBuf::from("/test/Button.tsx");
        let (nodes, _edges) = parse_tsx_file(code, &path).unwrap();

        // Should find the Button function
        let func_nodes: Vec<&CodeNode> = nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Function))
            .collect();
        
        assert!(func_nodes.iter().any(|n| n.name == "Button"));
    }
}
