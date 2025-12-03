// File: src-tauri/src/llm/context_depth.rs
// Purpose: 4-Level Context Depth system for optimal code generation
// Dependencies: GNN engine, embeddings
// Last Updated: December 3, 2025
//
// Implements graduated context assembly with 4 distinct levels:
// - L1 (Direct): Target node + direct dependencies (40% budget)
// - L2 (Adjacent): 1-hop transitive dependencies (30% budget)
// - L3 (Extended): 2-hop dependencies (20% budget)
// - L4 (Ecosystem): 3-hop + semantic neighbors (10% budget)
//
// Benefits:
// - Better context quality: Prioritizes most relevant code
// - Faster generation: Reduced noise, focused context
// - Higher accuracy: Critical dependencies always included
// - Semantic awareness: Uses embeddings for similarity
//
// Usage:
// 1. Call assemble_4_level_context() with target node
// 2. System builds hierarchical context layers
// 3. Each layer has token budget allocation
// 4. Returns optimized context for LLM

use crate::gnn::{CodeNode, GNNEngine, NodeType};
use crate::llm::tokens::count_tokens;
use std::collections::{HashMap, HashSet, VecDeque};

/// Token limits (reserve 20% for response)
const CLAUDE_MAX_CONTEXT_TOKENS: usize = 160_000;
const GPT4_MAX_CONTEXT_TOKENS: usize = 100_000;

/// 4-Level Context Structure
#[derive(Debug, Clone)]
pub struct FourLevelContext {
    /// L1: Direct dependencies (target + immediate deps)
    pub l1_direct: Vec<ContextItem>,
    /// L2: Adjacent dependencies (1-hop transitive)
    pub l2_adjacent: Vec<ContextItem>,
    /// L3: Extended dependencies (2-hop transitive)
    pub l3_extended: Vec<ContextItem>,
    /// L4: Ecosystem (3-hop + semantic neighbors)
    pub l4_ecosystem: Vec<ContextItem>,
    /// Total tokens used
    pub total_tokens: usize,
    /// Token budget per level
    pub level_budgets: [usize; 4],
}

impl FourLevelContext {
    /// Get all context items in priority order
    pub fn get_all_items(&self) -> Vec<ContextItem> {
        let mut items = Vec::new();
        items.extend(self.l1_direct.clone());
        items.extend(self.l2_adjacent.clone());
        items.extend(self.l3_extended.clone());
        items.extend(self.l4_ecosystem.clone());
        items
    }

    /// Get context as formatted strings
    pub fn to_strings(&self) -> Vec<String> {
        self.get_all_items()
            .into_iter()
            .map(|item| item.content)
            .collect()
    }

    /// Get summary statistics
    pub fn get_stats(&self) -> ContextStats {
        ContextStats {
            l1_count: self.l1_direct.len(),
            l2_count: self.l2_adjacent.len(),
            l3_count: self.l3_extended.len(),
            l4_count: self.l4_ecosystem.len(),
            total_tokens: self.total_tokens,
            l1_tokens: self.l1_direct.iter().map(|i| i.token_count).sum(),
            l2_tokens: self.l2_adjacent.iter().map(|i| i.token_count).sum(),
            l3_tokens: self.l3_extended.iter().map(|i| i.token_count).sum(),
            l4_tokens: self.l4_ecosystem.iter().map(|i| i.token_count).sum(),
        }
    }
}

/// Context statistics for monitoring
#[derive(Debug, Clone)]
pub struct ContextStats {
    pub l1_count: usize,
    pub l2_count: usize,
    pub l3_count: usize,
    pub l4_count: usize,
    pub total_tokens: usize,
    pub l1_tokens: usize,
    pub l2_tokens: usize,
    pub l3_tokens: usize,
    pub l4_tokens: usize,
}

/// Context item with depth and priority
#[derive(Debug, Clone)]
pub struct ContextItem {
    pub content: String,
    pub node_id: String,
    pub depth: usize,
    pub priority: u32,
    pub token_count: usize,
    pub similarity: Option<f64>, // For semantic L4 items
}

/// Configuration for 4-level context assembly
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum total tokens (default: Claude 160K)
    pub max_tokens: usize,
    /// Token budget allocation per level (must sum to 1.0)
    /// Default: [0.4, 0.3, 0.2, 0.1] = 40%/30%/20%/10%
    pub level_budgets: [f64; 4],
    /// Similarity threshold for L4 semantic neighbors (0.0-1.0)
    pub similarity_threshold: f64,
    /// Maximum L4 semantic results
    pub max_semantic_results: usize,
    /// Include implementation details (vs just signatures)
    pub include_implementations: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        ContextConfig {
            max_tokens: CLAUDE_MAX_CONTEXT_TOKENS,
            level_budgets: [0.4, 0.3, 0.2, 0.1], // L1: 40%, L2: 30%, L3: 20%, L4: 10%
            similarity_threshold: 0.70,
            max_semantic_results: 50,
            include_implementations: false, // Signatures only for better compression
        }
    }
}

impl ContextConfig {
    /// Create config for GPT-4 (100K token limit)
    pub fn for_gpt4() -> Self {
        ContextConfig {
            max_tokens: GPT4_MAX_CONTEXT_TOKENS,
            ..Default::default()
        }
    }

    /// Create config for Claude (160K token limit)
    pub fn for_claude() -> Self {
        ContextConfig {
            max_tokens: CLAUDE_MAX_CONTEXT_TOKENS,
            ..Default::default()
        }
    }
}

/// Assemble 4-level graduated context
/// 
/// # Arguments
/// * `engine` - GNN engine with code graph
/// * `target_node` - Starting node name
/// * `file_path` - Optional file path context
/// * `intent` - Optional natural language intent for semantic L4
/// * `config` - Context configuration
/// 
/// # Returns
/// FourLevelContext with hierarchical context layers
pub fn assemble_4_level_context(
    engine: &GNNEngine,
    target_node: &str,
    file_path: Option<&str>,
    intent: Option<&str>,
    config: Option<ContextConfig>,
) -> Result<FourLevelContext, String> {
    let config = config.unwrap_or_default();
    
    // Calculate token budgets per level
    let level_budgets: [usize; 4] = [
        (config.max_tokens as f64 * config.level_budgets[0]) as usize,
        (config.max_tokens as f64 * config.level_budgets[1]) as usize,
        (config.max_tokens as f64 * config.level_budgets[2]) as usize,
        (config.max_tokens as f64 * config.level_budgets[3]) as usize,
    ];

    // Find target node
    let target = engine.find_node(target_node, file_path)
        .ok_or_else(|| format!("Target node '{}' not found", target_node))?;

    let graph = engine.get_graph();
    
    // L1: Direct dependencies (target + immediate deps)
    let mut l1_items = Vec::new();
    let mut l1_tokens = 0;
    let mut visited_l1 = HashSet::new();
    
    // Add target node first
    let target_content = format_node_with_impl(&target, config.include_implementations);
    let target_tokens = count_tokens(&target_content);
    
    if target_tokens <= level_budgets[0] {
        l1_items.push(ContextItem {
            content: target_content,
            node_id: target.id.clone(),
            depth: 0,
            priority: 100, // Highest priority
            token_count: target_tokens,
            similarity: None,
        });
        l1_tokens += target_tokens;
        visited_l1.insert(target.id.clone());
    }
    
    // Add direct dependencies
    let direct_deps = engine.get_dependencies(&target.id);
    for dep in direct_deps {
        if visited_l1.contains(&dep.id) {
            continue;
        }
        
        let content = format_node_with_impl(&dep, config.include_implementations);
        let tokens = count_tokens(&content);
        
        if l1_tokens + tokens > level_budgets[0] {
            break; // L1 budget exhausted
        }
        
        let priority = calculate_priority(&dep.node_type, 1);
        l1_items.push(ContextItem {
            content,
            node_id: dep.id.clone(),
            depth: 1,
            priority,
            token_count: tokens,
            similarity: None,
        });
        
        l1_tokens += tokens;
        visited_l1.insert(dep.id.clone());
    }

    // L2: Adjacent (1-hop transitive dependencies)
    let mut l2_items = Vec::new();
    let mut l2_tokens = 0;
    let mut visited_l2 = visited_l1.clone();
    
    for l1_item in &l1_items {
        let deps = engine.get_dependencies(&l1_item.node_id);
        for dep in deps {
            if visited_l2.contains(&dep.id) {
                continue;
            }
            
            let content = format_node_with_impl(&dep, config.include_implementations);
            let tokens = count_tokens(&content);
            
            if l2_tokens + tokens > level_budgets[1] {
                break;
            }
            
            let priority = calculate_priority(&dep.node_type, 2);
            l2_items.push(ContextItem {
                content,
                node_id: dep.id.clone(),
                depth: 2,
                priority,
                token_count: tokens,
                similarity: None,
            });
            
            l2_tokens += tokens;
            visited_l2.insert(dep.id.clone());
        }
        
        if l2_tokens >= level_budgets[1] {
            break;
        }
    }

    // L3: Extended (2-hop transitive)
    let mut l3_items = Vec::new();
    let mut l3_tokens = 0;
    let mut visited_l3 = visited_l2.clone();
    
    for l2_item in &l2_items {
        let deps = engine.get_dependencies(&l2_item.node_id);
        for dep in deps {
            if visited_l3.contains(&dep.id) {
                continue;
            }
            
            let content = format_node_with_impl(&dep, config.include_implementations);
            let tokens = count_tokens(&content);
            
            if l3_tokens + tokens > level_budgets[2] {
                break;
            }
            
            let priority = calculate_priority(&dep.node_type, 3);
            l3_items.push(ContextItem {
                content,
                node_id: dep.id.clone(),
                depth: 3,
                priority,
                token_count: tokens,
                similarity: None,
            });
            
            l3_tokens += tokens;
            visited_l3.insert(dep.id.clone());
        }
        
        if l3_tokens >= level_budgets[2] {
            break;
        }
    }

    // L4: Ecosystem (3-hop + semantic neighbors)
    let mut l4_items = Vec::new();
    let mut l4_tokens = 0;
    let mut visited_l4 = visited_l3.clone();
    
    // First, add 3-hop dependencies
    for l3_item in &l3_items {
        if l4_tokens >= level_budgets[3] / 2 {
            break; // Reserve 50% of L4 for semantic
        }
        
        let deps = engine.get_dependencies(&l3_item.node_id);
        for dep in deps {
            if visited_l4.contains(&dep.id) {
                continue;
            }
            
            let content = format_node_with_impl(&dep, config.include_implementations);
            let tokens = count_tokens(&content);
            
            if l4_tokens + tokens > level_budgets[3] / 2 {
                break;
            }
            
            let priority = calculate_priority(&dep.node_type, 4);
            l4_items.push(ContextItem {
                content,
                node_id: dep.id.clone(),
                depth: 4,
                priority,
                token_count: tokens,
                similarity: None,
            });
            
            l4_tokens += tokens;
            visited_l4.insert(dep.id.clone());
        }
    }
    
    // Second, add semantic neighbors if intent provided
    if let Some(intent_text) = intent {
        let mut embedder = crate::gnn::embeddings::EmbeddingGenerator::default();
        if let Ok(intent_embedding) = embedder.generate_text_embedding(intent_text) {
            // Find similar nodes within 3-hop neighborhood
            if let Ok(similar_nodes) = graph.find_similar_in_neighborhood(
                &target.id,
                3, // max 3 hops
                config.similarity_threshold,
                config.max_semantic_results,
            ) {
                for (similar_node, similarity) in similar_nodes {
                    if visited_l4.contains(&similar_node.id) {
                        continue;
                    }
                    
                    let content = format_node_with_impl(&similar_node, config.include_implementations);
                    let tokens = count_tokens(&content);
                    
                    if l4_tokens + tokens > level_budgets[3] {
                        break;
                    }
                    
                    // Higher priority for more similar nodes
                    let priority = (50.0 + similarity * 50.0) as u32;
                    l4_items.push(ContextItem {
                        content,
                        node_id: similar_node.id.clone(),
                        depth: 4,
                        priority,
                        token_count: tokens,
                        similarity: Some(similarity),
                    });
                    
                    l4_tokens += tokens;
                    visited_l4.insert(similar_node.id.clone());
                }
            }
        }
    }

    let total_tokens = l1_tokens + l2_tokens + l3_tokens + l4_tokens;

    Ok(FourLevelContext {
        l1_direct: l1_items,
        l2_adjacent: l2_items,
        l3_extended: l3_items,
        l4_ecosystem: l4_items,
        total_tokens,
        level_budgets,
    })
}

/// Format node with or without implementation
fn format_node_with_impl(node: &CodeNode, include_impl: bool) -> String {
    if include_impl {
        format_node_full(node)
    } else {
        format_node_signature(node)
    }
}

/// Format node as signature only (compact)
fn format_node_signature(node: &CodeNode) -> String {
    match node.node_type {
        NodeType::Import => {
            format!("# Import: {} ({})", node.name, node.file_path)
        }
        NodeType::Function => {
            format!(
                "# Function: {} ({}:{})\ndef {}(...): ...",
                node.name, node.file_path, node.line_start, node.name
            )
        }
        NodeType::Class => {
            format!(
                "# Class: {} ({}:{})\nclass {}: ...",
                node.name, node.file_path, node.line_start, node.name
            )
        }
        NodeType::Variable => {
            format!("# Variable: {} = ... # {}", node.name, node.file_path)
        }
        NodeType::Module => {
            format!("# Module: {} ({})", node.name, node.file_path)
        }
    }
}

/// Format node with full implementation (if available)
fn format_node_full(node: &CodeNode) -> String {
    // In production, read actual code from file between line_start and line_end
    // For now, use signature format
    format_node_signature(node)
}

/// Calculate priority based on node type and depth
fn calculate_priority(node_type: &NodeType, depth: usize) -> u32 {
    let base_priority: u32 = match node_type {
        NodeType::Import => 90,
        NodeType::Function => 80,
        NodeType::Class => 70,
        NodeType::Variable => 60,
        NodeType::Module => 50,
    };
    
    // Reduce by 10 per depth level
    base_priority.saturating_sub((depth * 10) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();
        assert_eq!(config.max_tokens, CLAUDE_MAX_CONTEXT_TOKENS);
        assert_eq!(config.level_budgets[0], 0.4);
        assert_eq!(config.similarity_threshold, 0.70);
    }

    #[test]
    fn test_context_config_gpt4() {
        let config = ContextConfig::for_gpt4();
        assert_eq!(config.max_tokens, GPT4_MAX_CONTEXT_TOKENS);
    }

    #[test]
    fn test_calculate_priority() {
        assert_eq!(calculate_priority(&NodeType::Import, 0), 90);
        assert_eq!(calculate_priority(&NodeType::Import, 1), 80);
        assert_eq!(calculate_priority(&NodeType::Function, 0), 80);
        assert_eq!(calculate_priority(&NodeType::Function, 3), 50);
    }

    #[test]
    fn test_context_stats() {
        let context = FourLevelContext {
            l1_direct: vec![
                ContextItem {
                    content: "test".to_string(),
                    node_id: "1".to_string(),
                    depth: 0,
                    priority: 100,
                    token_count: 10,
                    similarity: None,
                }
            ],
            l2_adjacent: vec![],
            l3_extended: vec![],
            l4_ecosystem: vec![],
            total_tokens: 10,
            level_budgets: [64000, 48000, 32000, 16000],
        };

        let stats = context.get_stats();
        assert_eq!(stats.l1_count, 1);
        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.l1_tokens, 10);
    }
}
