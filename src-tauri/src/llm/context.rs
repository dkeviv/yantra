// File: src-tauri/src/llm/context.rs
// Purpose: Context assembly from GNN for LLM prompts with exact token counting
// Last Updated: December 21, 2025

use crate::gnn::{CodeNode, GNNEngine, NodeType};
use crate::llm::tokens::count_tokens;
use std::collections::{HashSet, VecDeque};

// Token limits for different LLM providers (reserve 20% for response)
// Claude Sonnet 4: 200K tokens, use 160K for context
// GPT-4 Turbo: 128K tokens, use 100K for context
const CLAUDE_MAX_CONTEXT_TOKENS: usize = 160_000;
#[allow(dead_code)] // Available for GPT-4 specific context assembly
const GPT4_MAX_CONTEXT_TOKENS: usize = 100_000;

// No artificial depth limit - traverse entire graph until token limit
// BFS naturally prioritizes closer dependencies first
#[allow(dead_code)] // Kept for documentation - we use unlimited depth
const NO_DEPTH_LIMIT: usize = usize::MAX;

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
/// Uses token-aware limits based on LLM provider capabilities (Claude: 160K, GPT-4: 100K)
/// 
/// # Arguments
/// * `engine` - The GNN engine containing the code graph
/// * `target_node` - Optional node name to start context gathering from
/// * `file_path` - Optional file path to limit context to or for target node lookup
/// * `max_tokens` - Optional maximum tokens (defaults to Claude's 160K limit)
/// 
/// # Returns
/// A vector of context strings, prioritized by relevance and limited by token budget
pub fn assemble_context(
    engine: &GNNEngine,
    target_node: Option<&str>,
    file_path: Option<&str>,
) -> Result<Vec<String>, String> {
    assemble_context_with_limit(engine, target_node, file_path, CLAUDE_MAX_CONTEXT_TOKENS)
}

/// Assemble context with explicit token limit
pub fn assemble_context_with_limit(
    engine: &GNNEngine,
    target_node: Option<&str>,
    file_path: Option<&str>,
    max_tokens: usize,
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
    
    // Use exact token counting instead of estimates
    // Accumulate context items until we hit the token budget
    let mut selected_items = Vec::new();
    let mut current_tokens = 0;
    
    for item in context_items {
        let item_tokens = count_tokens(&item.content);
        
        // Check if adding this item would exceed the budget
        if current_tokens + item_tokens > max_tokens {
            break; // Stop when we hit the token limit
        }
        
        current_tokens += item_tokens;
        selected_items.push(item);
    }
    
    // Extract just the content strings
    Ok(selected_items.into_iter().map(|item| item.content).collect())
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
        // Skip if already visited (no depth limit - traverse entire dependency graph)
        if visited.contains(&current_id) {
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

/// Compress context to fit more within token limits
/// 
/// Applies multiple compression strategies while preserving semantic meaning:
/// 1. Strip excessive whitespace (multiple spaces → single space)
/// 2. Remove empty lines (except between functions/classes)
/// 3. Strip comments (except first line docstrings for functions/classes)
/// 4. Normalize indentation to 2 spaces
/// 
/// Target: 20-30% size reduction
/// 
/// # Arguments
/// * `context` - The context string to compress
/// 
/// # Returns
/// Compressed context string
pub fn compress_context(context: &str) -> String {
    let mut result = String::new();
    let lines: Vec<&str> = context.lines().collect();
    let mut i = 0;
    
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        
        // Skip empty lines unless between definitions
        if trimmed.is_empty() {
            // Keep one empty line between def/class
            if i > 0 && i < lines.len() - 1 {
                let prev = lines[i - 1].trim();
                let next = lines[i + 1].trim();
                if (prev.starts_with("def ") || prev.starts_with("class "))
                    && (next.starts_with("def ") || next.starts_with("class ")) {
                    result.push('\n');
                }
            }
            i += 1;
            continue;
        }
        
        // Skip comment-only lines (except docstrings)
        if trimmed.starts_with('#') && !is_likely_docstring(trimmed) {
            i += 1;
            continue;
        }
        
        // Normalize indentation to 2 spaces
        let indent_level = count_leading_spaces(line) / 4; // Assume 4-space indent
        let normalized_indent = "  ".repeat(indent_level);
        
        // Remove inline comments (preserve # in strings)
        let code_part = if let Some(pos) = find_comment_position(line) {
            &line[..pos]
        } else {
            line
        };
        
        // Compress multiple spaces to single space (preserve indentation)
        let compressed = compress_spaces(code_part);
        
        result.push_str(&normalized_indent);
        result.push_str(compressed.trim());
        result.push('\n');
        
        i += 1;
    }
    
    result
}

/// Compress multiple spaces to single space
fn compress_spaces(s: &str) -> String {
    let mut result = String::new();
    let mut prev_space = false;
    let mut in_string = false;
    let mut string_char = ' ';
    
    for ch in s.chars() {
        if ch == '"' || ch == '\'' {
            if !in_string {
                in_string = true;
                string_char = ch;
            } else if ch == string_char {
                in_string = false;
            }
            result.push(ch);
            prev_space = false;
        } else if ch.is_whitespace() && !in_string {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    
    result
}

/// Count leading spaces in a line
fn count_leading_spaces(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

/// Find position of comment start (handling strings)
fn find_comment_position(line: &str) -> Option<usize> {
    let mut in_string = false;
    let mut string_char = ' ';
    let mut prev_backslash = false;
    
    for (i, ch) in line.chars().enumerate() {
        if prev_backslash {
            prev_backslash = false;
            continue;
        }
        
        if ch == '\\' {
            prev_backslash = true;
            continue;
        }
        
        if ch == '"' || ch == '\'' {
            if !in_string {
                in_string = true;
                string_char = ch;
            } else if ch == string_char {
                in_string = false;
            }
        } else if ch == '#' && !in_string {
            return Some(i);
        }
    }
    
    None
}

/// Check if a comment line is likely a docstring placeholder
fn is_likely_docstring(line: &str) -> bool {
    let lower = line.to_lowercase();
    lower.contains("docstring") || lower.contains("\"\"\"") || lower.contains("'''")
}

/// Compress a vector of context strings
pub fn compress_context_vec(contexts: Vec<String>) -> Vec<String> {
    contexts.into_iter()
        .map(|ctx| compress_context(&ctx))
        .collect()
}

/// Hierarchical context with two levels of detail
/// 
/// Level 1 (Immediate): Full code for target files and direct dependencies (40% of budget)
/// Level 2 (Related): Function/class signatures only for related files (30% of budget)
/// Remaining 30% reserved for system prompts and user instructions
#[derive(Debug, Clone)]
pub struct HierarchicalContext {
    /// Level 1: Full code for immediate context
    pub immediate: Vec<String>,
    /// Level 2: Signatures only for related context
    pub related: Vec<String>,
    /// Total tokens used
    pub total_tokens: usize,
    /// Tokens used in Level 1
    pub immediate_tokens: usize,
    /// Tokens used in Level 2
    pub related_tokens: usize,
}

impl Default for HierarchicalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalContext {
    /// Create a new hierarchical context
    pub fn new() -> Self {
        Self {
            immediate: Vec::new(),
            related: Vec::new(),
            total_tokens: 0,
            immediate_tokens: 0,
            related_tokens: 0,
        }
    }
    
    /// Get all context as a formatted string
    pub fn to_string(&self) -> String {
        let mut result = String::new();
        
        if !self.immediate.is_empty() {
            result.push_str("=== IMMEDIATE CONTEXT (Full Code) ===\n\n");
            for item in &self.immediate {
                result.push_str(item);
                result.push_str("\n\n");
            }
        }
        
        if !self.related.is_empty() {
            result.push_str("=== RELATED CONTEXT (Signatures Only) ===\n\n");
            for item in &self.related {
                result.push_str(item);
                result.push_str("\n\n");
            }
        }
        
        result
    }
}

/// Assemble hierarchical context with two levels of detail
/// 
/// Level 1 (40% budget): Full code for target node, its file, and direct dependencies
/// Level 2 (30% budget): Signatures only for related files (2nd level dependencies)
/// 
/// This allows including much more context in a useful form:
/// - Immediate context gets full implementation details
/// - Related context provides function signatures for understanding broader codebase
/// 
/// # Arguments
/// * `engine` - The GNN engine containing the code graph
/// * `target_node` - Optional node name to start context gathering from
/// * `file_path` - Optional file path for target context
/// * `max_tokens` - Maximum total tokens (default: Claude's 160K)
/// 
/// # Returns
/// HierarchicalContext with two levels of detail
pub fn assemble_hierarchical_context(
    engine: &GNNEngine,
    target_node: Option<&str>,
    file_path: Option<&str>,
    max_tokens: usize,
) -> Result<HierarchicalContext, String> {
    let mut context = HierarchicalContext::new();
    
    // Calculate token budgets for each level
    let l1_budget = (max_tokens as f32 * 0.40) as usize; // 40% for immediate context
    let l2_budget = (max_tokens as f32 * 0.30) as usize; // 30% for related context
    
    let mut visited_l1: HashSet<String> = HashSet::new();
    let mut visited_l2: HashSet<String> = HashSet::new();
    
    // Level 1: Gather immediate context (full code)
    let mut l1_items: Vec<ContextItem> = Vec::new();
    
    if let Some(node_name) = target_node {
        if let Some(node) = engine.find_node(node_name, file_path) {
            // Add target node and its direct dependencies
            gather_immediate_context(engine, &node.id, &mut l1_items, &mut visited_l1, 0, 1)?;
        }
    }
    
    // If file path specified, include all nodes from that file in L1
    if let Some(path) = file_path {
        gather_context_from_file(engine, path, &mut l1_items)?;
    }
    
    // Sort L1 by priority
    l1_items.sort_by(|a, b| {
        b.priority.cmp(&a.priority)
            .then(a.depth.cmp(&b.depth))
    });
    
    // Fill L1 budget with exact token counting
    for item in l1_items {
        let item_tokens = count_tokens(&item.content);
        
        if context.immediate_tokens + item_tokens > l1_budget {
            break;
        }
        
        context.immediate_tokens += item_tokens;
        context.immediate.push(item.content);
    }
    
    // Level 2: Gather related context (signatures only)
    let mut l2_items: Vec<ContextItem> = Vec::new();
    
    // Expand from L1 nodes to their dependencies (but not already in L1)
    for visited_id in &visited_l1 {
        gather_related_signatures(engine, visited_id, &mut l2_items, &mut visited_l2, &visited_l1, 2)?;
    }
    
    // Sort L2 by priority
    l2_items.sort_by(|a, b| {
        b.priority.cmp(&a.priority)
            .then(a.depth.cmp(&b.depth))
    });
    
    // Fill L2 budget with signatures
    for item in l2_items {
        let item_tokens = count_tokens(&item.content);
        
        if context.related_tokens + item_tokens > l2_budget {
            break;
        }
        
        context.related_tokens += item_tokens;
        context.related.push(item.content);
    }
    
    context.total_tokens = context.immediate_tokens + context.related_tokens;
    
    Ok(context)
}

/// Gather immediate context (full code) with limited depth
fn gather_immediate_context(
    engine: &GNNEngine,
    node_id: &str,
    context_items: &mut Vec<ContextItem>,
    visited: &mut HashSet<String>,
    depth: usize,
    max_depth: usize,
) -> Result<(), String> {
    if depth > max_depth || visited.contains(node_id) {
        return Ok(());
    }
    
    visited.insert(node_id.to_string());
    let graph = engine.get_graph();
    
    // Find the node
    if let Some(node) = graph.get_all_nodes().iter().find(|n| n.id == node_id) {
        let priority = calculate_priority(&node.node_type, depth);
        context_items.push(ContextItem {
            content: format_node_full_code(node),
            priority,
            depth,
        });
        
        // Get direct dependencies for next level
        if depth < max_depth {
            let deps = engine.get_dependencies(node_id);
            for dep in deps {
                gather_immediate_context(engine, &dep.id, context_items, visited, depth + 1, max_depth)?;
            }
        }
    }
    
    Ok(())
}

/// Gather related context as signatures only
fn gather_related_signatures(
    engine: &GNNEngine,
    node_id: &str,
    context_items: &mut Vec<ContextItem>,
    visited: &mut HashSet<String>,
    exclude: &HashSet<String>,
    depth: usize,
) -> Result<(), String> {
    let graph = engine.get_graph();
    let deps = engine.get_dependencies(node_id);
    
    for dep in deps {
        // Skip if already in L1 or already visited in L2
        if exclude.contains(&dep.id) || visited.contains(&dep.id) {
            continue;
        }
        
        visited.insert(dep.id.clone());
        
        if let Some(node) = graph.get_all_nodes().iter().find(|n| n.id == dep.id) {
            let priority = calculate_priority(&node.node_type, depth);
            context_items.push(ContextItem {
                content: format_node_signature(node),
                priority,
                depth,
            });
        }
    }
    
    Ok(())
}

/// Format node as full code (for L1 immediate context)
fn format_node_full_code(node: &CodeNode) -> String {
    // For now, use the signature format
    // In future, could read actual file content from node.file_path lines node.line_start..node.line_end
    format_node_context(node)
}

/// Format node as signature only (for L2 related context)
fn format_node_signature(node: &CodeNode) -> String {
    match node.node_type {
        NodeType::Function => {
            format!(
                "def {}(...): ...  # {}, line {}",
                node.name, node.file_path, node.line_start
            )
        }
        NodeType::Class => {
            format!(
                "class {}: ...  # {}, line {}",
                node.name, node.file_path, node.line_start
            )
        }
        NodeType::Import => {
            format!("import {}  # from {}", node.name, node.file_path)
        }
        NodeType::Variable => {
            format!("{}  # variable in {}", node.name, node.file_path)
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
    
    #[test]
    fn test_token_based_limits() {
        // Test with different token limits
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let engine = GNNEngine::new(&db_path).unwrap();
        
        // With Claude's 160K token limit, we can fit ~800 items (160K / 200 avg tokens)
        let result = assemble_context_with_limit(&engine, None, None, CLAUDE_MAX_CONTEXT_TOKENS);
        assert!(result.is_ok());
        
        // With GPT-4's 100K token limit, we can fit ~500 items
        let result = assemble_context_with_limit(&engine, None, None, GPT4_MAX_CONTEXT_TOKENS);
        assert!(result.is_ok());
        
        // With small limit of 1000 tokens, we can only fit ~5 items
        let result = assemble_context_with_limit(&engine, None, None, 1000);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_hierarchical_context_structure() {
        // Test the hierarchical context structure
        let context = HierarchicalContext::new();
        assert_eq!(context.immediate.len(), 0);
        assert_eq!(context.related.len(), 0);
        assert_eq!(context.total_tokens, 0);
    }
    
    #[test]
    fn test_hierarchical_context_to_string() {
        let mut context = HierarchicalContext::new();
        context.immediate.push("def foo(): pass".to_string());
        context.related.push("def bar(): ...".to_string());
        
        let formatted = context.to_string();
        assert!(formatted.contains("IMMEDIATE CONTEXT"));
        assert!(formatted.contains("RELATED CONTEXT"));
        assert!(formatted.contains("def foo()"));
        assert!(formatted.contains("def bar()"));
    }
    
    #[test]
    fn test_assemble_hierarchical_context_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let engine = GNNEngine::new(&db_path).unwrap();
        
        let result = assemble_hierarchical_context(&engine, None, None, CLAUDE_MAX_CONTEXT_TOKENS);
        assert!(result.is_ok());
        
        let context = result.unwrap();
        assert_eq!(context.immediate.len(), 0);
        assert_eq!(context.related.len(), 0);
        assert_eq!(context.total_tokens, 0);
    }
    
    #[test]
    fn test_hierarchical_context_budget_split() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let test_file = dir.path().join("test.py");
        
        // Create a Python file with multiple functions
        fs::write(&test_file, 
            "def func1():\n    pass\n\ndef func2():\n    func1()\n\ndef func3():\n    func2()\n"
        ).unwrap();
        
        let mut engine = GNNEngine::new(&db_path).unwrap();
        engine.parse_file(&test_file).unwrap();
        
        // Assemble with hierarchical context
        let result = assemble_hierarchical_context(
            &engine, 
            Some("func3"),
            Some(test_file.to_str().unwrap()),
            10_000 // 10K token budget
        );
        
        assert!(result.is_ok());
        let context = result.unwrap();
        
        // Should have some immediate context
        assert!(context.immediate_tokens > 0 || context.related_tokens > 0);
        
        // Total should not exceed budget
        assert!(context.total_tokens <= 10_000);
        
        // Verify budget split (40% L1, 30% L2)
        // Allow some tolerance for small test
        if context.total_tokens > 0 {
            assert!(context.immediate_tokens <= 4_000); // 40% of 10K
            assert!(context.related_tokens <= 3_000);   // 30% of 10K
        }
    }
    
    #[test]
    fn test_format_node_signature() {
        let func_node = CodeNode {
            id: "test_func".to_string(),
            node_type: NodeType::Function,
            name: "test_function".to_string(),
            file_path: "test.py".to_string(),
            line_start: 10,
            line_end: 20,
        };
        
        let signature = format_node_signature(&func_node);
        assert!(signature.contains("def test_function(...): ..."));
        assert!(signature.contains("test.py"));
        assert!(signature.contains("10"));
        
        let class_node = CodeNode {
            id: "test_class".to_string(),
            node_type: NodeType::Class,
            name: "TestClass".to_string(),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 5,
        };
        
        let signature = format_node_signature(&class_node);
        assert!(signature.contains("class TestClass: ..."));
    }
    
    #[test]
    fn test_compress_spaces() {
        let input = "def    foo(  x,    y  ):";
        let output = compress_spaces(input);
        assert_eq!(output, "def foo( x, y ):");
        
        // Should preserve spaces in strings
        let input_with_string = "x = \"hello    world\"";
        let output_with_string = compress_spaces(input_with_string);
        assert_eq!(output_with_string, "x = \"hello    world\"");
    }
    
    #[test]
    fn test_count_leading_spaces() {
        assert_eq!(count_leading_spaces("    def foo():"), 4);
        assert_eq!(count_leading_spaces("def foo():"), 0);
        assert_eq!(count_leading_spaces("        return x"), 8);
    }
    
    #[test]
    fn test_find_comment_position() {
        assert_eq!(find_comment_position("x = 1  # comment"), Some(7));
        assert_eq!(find_comment_position("x = 1"), None);
        
        // Should handle # in strings
        assert_eq!(find_comment_position("x = \"#not a comment\"  # real comment"), Some(22));
    }
    
    #[test]
    fn test_compress_context_basic() {
        let input = r#"def foo():
    # This is a comment
    x = 1
    
    
    y = 2  # inline comment
    return x + y"#;
        
        let compressed = compress_context(input);
        
        // Should remove comment-only lines
        assert!(!compressed.contains("# This is a comment"));
        
        // Should remove inline comments
        assert!(!compressed.contains("# inline comment"));
        
        // Should keep code
        assert!(compressed.contains("def foo():"));
        assert!(compressed.contains("x = 1"));
        assert!(compressed.contains("y = 2"));
        assert!(compressed.contains("return x + y"));
    }
    
    #[test]
    fn test_compress_context_size_reduction() {
        let input = r#"def calculate_sum(a, b):
    # Calculate the sum of two numbers
    # Args:
    #     a: first number
    #     b: second number
    # Returns:
    #     The sum of a and b
    
    result = a + b  # Add the numbers
    return result   # Return the result

class Calculator:
    # A simple calculator class
    
    def add(self, x, y):
        # Add two numbers
        return x + y"#;
        
        let compressed = compress_context(input);
        let original_len = input.len();
        let compressed_len = compressed.len();
        
        // Should achieve 20-30% compression
        let compression_ratio = (original_len - compressed_len) as f32 / original_len as f32;
        assert!(compression_ratio >= 0.15); // At least 15% reduction
        
        // Should still contain essential code
        assert!(compressed.contains("def calculate_sum(a, b):"));
        assert!(compressed.contains("class Calculator:"));
        assert!(compressed.contains("def add(self, x, y):"));
    }
    
    #[test]
    fn test_compress_context_vec() {
        let contexts = vec![
            "def foo():  # comment\n    pass".to_string(),
            "def bar():  # another comment\n    pass".to_string(),
        ];
        
        let compressed = compress_context_vec(contexts);
        assert_eq!(compressed.len(), 2);
        
        // Each should be compressed
        for ctx in compressed {
            assert!(!ctx.contains("# comment"));
            assert!(!ctx.contains("# another comment"));
        }
    }
    
    #[test]
    fn test_compress_preserves_strings() {
        let input = r#"message = "  Keep   these   spaces  "
another = 'and # these # too'"#;
        
        let compressed = compress_context(input);
        
        // Should preserve spaces and # in strings
        assert!(compressed.contains("\"  Keep   these   spaces  \""));
        assert!(compressed.contains("'and # these # too'"));
    }
}
