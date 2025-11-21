// File: src-tauri/src/llm/context.rs
// Purpose: Context assembly from GNN for LLM prompts
// Last Updated: November 21, 2025

use crate::gnn::{CodeNode, GNNEngine, NodeType};
use std::collections::{HashSet, VecDeque};

const MAX_CONTEXT_ITEMS: usize = 50;
const MAX_DEPTH: usize = 3;

/// Context item with priority for sorting
#[derive(Debug, Clone)]
struct ContextItem {
    content: String,
    priority: u32,
    depth: usize,
}

/// Assemble context for code generation from GNN
/// 
/// This function gathers relevant code context from the GNN for LLM prompts.
/// It performs a breadth-first traversal of dependencies starting from the target node.
/// 
/// # Arguments
/// * `engine` - The GNN engine containing the code graph
/// * `target_node` - Optional node name to start context gathering from
/// * `file_path` - Optional file path to limit context to or for target node lookup
/// 
/// # Returns
/// A vector of context strings, prioritized by relevance
pub fn assemble_context(
    engine: &GNNEngine,
    target_node: Option<&str>,
    file_path: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut context_items: Vec<ContextItem> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    
    // If target node specified, do BFS from that node
    if let Some(node_name) = target_node {
        if let Some(node) = engine.find_node(node_name, file_path) {
            gather_context_from_node(engine, &node.id, &mut context_items, &mut visited, 0)?;
        } else {
            return Err(format!("Node '{}' not found", node_name));
        }
    } 
    // If file path specified, gather all nodes from that file
    else if let Some(path) = file_path {
        gather_context_from_file(engine, path, &mut context_items)?;
    }
    // Otherwise, gather high-level context (imports and top-level definitions)
    else {
        gather_global_context(engine, &mut context_items)?;
    }
    
    // Sort by priority (higher first) and depth (shallower first)
    context_items.sort_by(|a, b| {
        b.priority.cmp(&a.priority)
            .then(a.depth.cmp(&b.depth))
    });
    
    // Limit to MAX_CONTEXT_ITEMS
    context_items.truncate(MAX_CONTEXT_ITEMS);
    
    // Extract just the content strings
    Ok(context_items.into_iter().map(|item| item.content).collect())
}

/// Gather context starting from a specific node using BFS
fn gather_context_from_node(
    engine: &GNNEngine,
    node_id: &str,
    context_items: &mut Vec<ContextItem>,
    visited: &mut HashSet<String>,
    start_depth: usize,
) -> Result<(), String> {
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    queue.push_back((node_id.to_string(), start_depth));
    
    let graph = engine.get_graph();
    
    while let Some((current_id, depth)) = queue.pop_front() {
        // Skip if already visited or too deep
        if visited.contains(&current_id) || depth > MAX_DEPTH {
            continue;
        }
        visited.insert(current_id.clone());
        
        // Find the node in the graph
        let node_opt = graph.get_all_nodes().iter()
            .find(|n| n.id == current_id)
            .copied();
        
        if let Some(node) = node_opt {
            // Add node to context
            let priority = calculate_priority(&node.node_type, depth);
            context_items.push(ContextItem {
                content: format_node_context(node),
                priority,
                depth,
            });
            
            // Get dependencies and add to queue
            let deps = engine.get_dependencies(&current_id);
            for dep in deps {
                if !visited.contains(&dep.id) {
                    queue.push_back((dep.id.clone(), depth + 1));
                }
            }
        }
    }
    
    Ok(())
}

/// Gather context from all nodes in a specific file
fn gather_context_from_file(
    engine: &GNNEngine,
    file_path: &str,
    context_items: &mut Vec<ContextItem>,
) -> Result<(), String> {
    // Get all nodes in the graph
    let graph = engine.get_graph();
    
    for node in graph.get_all_nodes() {
        if node.file_path == file_path {
            let priority = calculate_priority(&node.node_type, 0);
            context_items.push(ContextItem {
                content: format_node_context(node),
                priority,
                depth: 0,
            });
        }
    }
    
    Ok(())
}

/// Gather high-level global context (imports and top-level definitions)
fn gather_global_context(
    engine: &GNNEngine,
    context_items: &mut Vec<ContextItem>,
) -> Result<(), String> {
    let graph = engine.get_graph();
    
    for node in graph.get_all_nodes() {
        // Only include imports and top-level functions/classes
        match node.node_type {
            NodeType::Import => {
                context_items.push(ContextItem {
                    content: format_node_context(node),
                    priority: 5,
                    depth: 0,
                });
            }
            NodeType::Function | NodeType::Class => {
                // Only top-level (no parent in same file)
                context_items.push(ContextItem {
                    content: format_node_context(node),
                    priority: 3,
                    depth: 0,
                });
            }
            _ => {}
        }
    }
    
    Ok(())
}

/// Calculate priority based on node type and depth
fn calculate_priority(node_type: &NodeType, depth: usize) -> u32 {
    let base_priority: u32 = match node_type {
        NodeType::Import => 10,      // Imports are critical
        NodeType::Function => 8,     // Functions are very important
        NodeType::Class => 7,        // Classes are important
        NodeType::Variable => 5,     // Variables provide context
        NodeType::Module => 4,       // Module definitions
    };
    
    // Reduce priority based on depth (closer nodes are more relevant)
    base_priority.saturating_sub(depth as u32)
}

/// Format a node as context string for LLM
fn format_node_context(node: &CodeNode) -> String {
    match node.node_type {
        NodeType::Import => {
            format!("# Import from {}\nimport {}", node.file_path, node.name)
        }
        NodeType::Function => {
            format!(
                "# Function: {} ({}:{})\ndef {}(...): pass  # See implementation in {}",
                node.name, node.file_path, node.line_start, node.name, node.file_path
            )
        }
        NodeType::Class => {
            format!(
                "# Class: {} ({}:{})\nclass {}: pass  # See implementation in {}",
                node.name, node.file_path, node.line_start, node.name, node.file_path
            )
        }
        NodeType::Variable => {
            format!("# Variable: {} in {}", node.name, node.file_path)
        }
        NodeType::Module => {
            format!("# Module: {} ({})", node.name, node.file_path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::GNNEngine;
    use tempfile::tempdir;
    use std::fs;

    #[test]
    fn test_calculate_priority() {
        assert_eq!(calculate_priority(&NodeType::Import, 0), 10);
        assert_eq!(calculate_priority(&NodeType::Function, 0), 8);
        assert_eq!(calculate_priority(&NodeType::Class, 0), 7);
        assert_eq!(calculate_priority(&NodeType::Function, 2), 6); // 8 - 2
        assert_eq!(calculate_priority(&NodeType::Import, 11), 0); // saturating_sub
    }
    
    #[test]
    fn test_format_node_context() {
        let node = CodeNode {
            id: "test_id".to_string(),
            node_type: NodeType::Function,
            name: "test_func".to_string(),
            file_path: "test.py".to_string(),
            line_start: 10,
            line_end: 15,
        };
        
        let formatted = format_node_context(&node);
        assert!(formatted.contains("test_func"));
        assert!(formatted.contains("test.py"));
        assert!(formatted.contains("10"));
    }
    
    #[test]
    fn test_assemble_context_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let engine = GNNEngine::new(&db_path).unwrap();
        
        // Empty engine should return empty context
        let result = assemble_context(&engine, None, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
    
    #[test]
    fn test_assemble_context_from_file() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let test_file = dir.path().join("test.py");
        
        // Create a simple Python file
        fs::write(&test_file, "def foo():\n    pass\n").unwrap();
        
        let mut engine = GNNEngine::new(&db_path).unwrap();
        engine.parse_file(&test_file).unwrap();
        
        let result = assemble_context(&engine, None, Some(test_file.to_str().unwrap()));
        assert!(result.is_ok());
        let context = result.unwrap();
        assert!(!context.is_empty());
    }
}
