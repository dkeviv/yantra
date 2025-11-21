// File: src-tauri/src/gnn/parser.rs
// Purpose: Python code parser using tree-sitter
// Dependencies: tree-sitter, tree-sitter-python
// Last Updated: November 20, 2025

use super::{CodeNode, CodeEdge, NodeType, EdgeType};
use std::path::Path;
use tree_sitter::{Parser, Node};

/// Parse a Python file and extract nodes and edges
pub fn parse_python_file(
    code: &str,
    file_path: &Path,
) -> Result<(Vec<CodeNode>, Vec<CodeEdge>), String> {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_python::language())
        .map_err(|e| format!("Failed to set language: {}", e))?;
    
    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse code".to_string())?;
    
    let root_node = tree.root_node();
    
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    
    // Extract functions, classes, and other symbols
    extract_definitions(&root_node, code, file_path, &mut nodes, &mut edges)?;
    
    Ok((nodes, edges))
}

fn extract_definitions(
    node: &Node,
    code: &str,
    file_path: &Path,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_definition" => {
                extract_function(&child, code, file_path, nodes, edges)?;
            }
            "class_definition" => {
                extract_class(&child, code, file_path, nodes, edges)?;
            }
            "import_statement" | "import_from_statement" => {
                extract_import(&child, code, file_path, nodes, edges)?;
            }
            _ => {
                // Recursively process children
                extract_definitions(&child, code, file_path, nodes, edges)?;
            }
        }
    }
    
    Ok(())
}

fn extract_function(
    node: &Node,
    code: &str,
    file_path: &Path,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) -> Result<(), String> {
    // Get function name
    let name_node = node
        .child_by_field_name("name")
        .ok_or_else(|| "Function missing name".to_string())?;
    
    let name = get_node_text(&name_node, code);
    let file_path_str = file_path.to_str().unwrap_or("").to_string();
    
    // Create function node
    let func_id = format!("{}::{}", file_path_str, name);
    let func_node = CodeNode {
        id: func_id.clone(),
        node_type: NodeType::Function,
        name: name.clone(),
        file_path: file_path_str.clone(),
        line_start: node.start_position().row + 1,
        line_end: node.end_position().row + 1,
    };
    
    nodes.push(func_node);
    
    // Extract function calls within this function
    extract_function_calls(node, code, &func_id, &file_path_str, edges)?;
    
    Ok(())
}

fn extract_class(
    node: &Node,
    code: &str,
    file_path: &Path,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) -> Result<(), String> {
    // Get class name
    let name_node = node
        .child_by_field_name("name")
        .ok_or_else(|| "Class missing name".to_string())?;
    
    let name = get_node_text(&name_node, code);
    let file_path_str = file_path.to_str().unwrap_or("").to_string();
    
    // Create class node
    let class_id = format!("{}::{}", file_path_str, name);
    let class_node = CodeNode {
        id: class_id.clone(),
        node_type: NodeType::Class,
        name: name.clone(),
        file_path: file_path_str.clone(),
        line_start: node.start_position().row + 1,
        line_end: node.end_position().row + 1,
    };
    
    nodes.push(class_node);
    
    // Extract base classes (inheritance)
    if let Some(bases_node) = node.child_by_field_name("superclasses") {
        extract_inheritance(&bases_node, code, &class_id, &file_path_str, edges)?;
    }
    
    // Process methods within class - look in the body field
    if let Some(body_node) = node.child_by_field_name("body") {
        let mut cursor = body_node.walk();
        for child in body_node.children(&mut cursor) {
            if child.kind() == "function_definition" {
                extract_function(&child, code, file_path, nodes, edges)?;
            }
        }
    }
    
    Ok(())
}

fn extract_import(
    node: &Node,
    code: &str,
    file_path: &Path,
    nodes: &mut Vec<CodeNode>,
    _edges: &mut Vec<CodeEdge>,
) -> Result<(), String> {
    let file_path_str = file_path.to_str().unwrap_or("").to_string();
    
    // Extract module name
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "dotted_name" || child.kind() == "identifier" {
            let module_name = get_node_text(&child, code);
            
            // Create import node
            let import_id = format!("{}::import::{}", file_path_str, module_name);
            let import_node = CodeNode {
                id: import_id.clone(),
                node_type: NodeType::Import,
                name: module_name.clone(),
                file_path: file_path_str.clone(),
                line_start: node.start_position().row + 1,
                line_end: node.end_position().row + 1,
            };
            
            nodes.push(import_node);
        }
    }
    
    Ok(())
}

fn extract_function_calls(
    node: &Node,
    code: &str,
    caller_id: &str,
    file_path: &str,
    edges: &mut Vec<CodeEdge>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    
    for child in node.children(&mut cursor) {
        if child.kind() == "call" {
            // Get the function being called
            if let Some(func_node) = child.child_by_field_name("function") {
                let func_name = get_node_text(&func_node, code);
                let callee_id = format!("{}::{}", file_path, func_name);
                
                // Create call edge
                let edge = CodeEdge {
                    edge_type: EdgeType::Calls,
                    source_id: caller_id.to_string(),
                    target_id: callee_id,
                };
                
                edges.push(edge);
            }
        }
        
        // Recursively check children
        extract_function_calls(&child, code, caller_id, file_path, edges)?;
    }
    
    Ok(())
}

fn extract_inheritance(
    node: &Node,
    code: &str,
    class_id: &str,
    file_path: &str,
    edges: &mut Vec<CodeEdge>,
) -> Result<(), String> {
    let mut cursor = node.walk();
    
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" || child.kind() == "attribute" {
            let base_name = get_node_text(&child, code);
            let base_id = format!("{}::{}", file_path, base_name);
            
            // Create inheritance edge
            let edge = CodeEdge {
                edge_type: EdgeType::Inherits,
                source_id: class_id.to_string(),
                target_id: base_id,
            };
            
            edges.push(edge);
        }
    }
    
    Ok(())
}

fn get_node_text(node: &Node, code: &str) -> String {
    code[node.byte_range()].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_parse_simple_function() {
        let code = r#"
def hello_world():
    print("Hello, world!")
"#;
        
        let path = PathBuf::from("test.py");
        let result = parse_python_file(code, &path);
        
        assert!(result.is_ok());
        let (nodes, _edges) = result.unwrap();
        assert!(!nodes.is_empty());
        assert_eq!(nodes[0].name, "hello_world");
        assert_eq!(nodes[0].node_type, NodeType::Function);
    }
    
    #[test]
    fn test_parse_class() {
        let code = r#"
class MyClass:
    def __init__(self):
        pass
    
    def method(self):
        pass
"#;
        
        let path = PathBuf::from("test.py");
        let result = parse_python_file(code, &path);
        
        assert!(result.is_ok());
        let (nodes, _edges) = result.unwrap();
        
        // Debug: print what we extracted
        println!("Extracted {} nodes:", nodes.len());
        for node in &nodes {
            println!("  - {:?}: {} ({}:{})", node.node_type, node.name, node.line_start, node.line_end);
        }
        
        // Should have 1 class + 2 methods
        assert!(nodes.len() >= 3);
        
        let class_nodes: Vec<_> = nodes.iter()
            .filter(|n| n.node_type == NodeType::Class)
            .collect();
        assert_eq!(class_nodes.len(), 1);
        assert_eq!(class_nodes[0].name, "MyClass");
    }
}
