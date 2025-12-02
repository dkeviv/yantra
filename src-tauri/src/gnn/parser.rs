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
    // tree-sitter-python 0.23 uses LANGUAGE constant (LanguageFn)
    let lang_fn: extern "C" fn() -> *const std::os::raw::c_void = unsafe { std::mem::transmute(tree_sitter_python::LANGUAGE) };
    let language = unsafe { tree_sitter::Language::from_raw(lang_fn() as *const tree_sitter::ffi::TSLanguage) };
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to set Python language: {}", e))?;
    
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
        code_snippet: extract_code_snippet(node, code),
        docstring: extract_docstring(node, code),
        ..Default::default()
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
        code_snippet: extract_code_snippet(node, code),
        docstring: extract_docstring(node, code),
        ..Default::default()
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
                ..Default::default()
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

/// Extract code snippet from a node (with reasonable size limit)
fn extract_code_snippet(node: &Node, code: &str) -> Option<String> {
    const MAX_SNIPPET_LENGTH: usize = 1000; // 1KB max per node
    
    let snippet = get_node_text(node, code);
    
    // Truncate if too long (keep first 900 chars + indicator)
    if snippet.len() > MAX_SNIPPET_LENGTH {
        Some(format!("{}... [truncated]", &snippet[..900]))
    } else if !snippet.is_empty() {
        Some(snippet)
    } else {
        None
    }
}

/// Extract docstring from a function or class node
fn extract_docstring(node: &Node, code: &str) -> Option<String> {
    // In Python, docstring is usually the first statement in a function/class body
    // Look for a block node and then an expression_statement with a string
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                // Check if it's a string (docstring)
                if let Some(string_child) = child.child(0) {
                    if string_child.kind() == "string" {
                        let docstring = get_node_text(&string_child, code);
                        // Remove quotes and clean up
                        let cleaned = docstring
                            .trim()
                            .trim_start_matches("\"\"\"")
                            .trim_start_matches("'''")
                            .trim_start_matches('"')
                            .trim_start_matches('\'')
                            .trim_end_matches("\"\"\"")
                            .trim_end_matches("'''")
                            .trim_end_matches('"')
                            .trim_end_matches('\'')
                            .trim();
                        return Some(cleaned.to_string());
                    }
                }
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
