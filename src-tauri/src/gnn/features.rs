// File: src-tauri/src/gnn/features.rs
// Purpose: Extract 986-dimensional feature vectors from GNN nodes for GraphSAGE
// Dependencies: tree-sitter, std
// Last Updated: November 30, 2025
//
// Feature Vector Structure (986 dimensions):
// - 974 base features (code properties, complexity, structure)
// - 12 language encoding (one-hot: Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin)
//
// Target: <1ms extraction time per node

use super::{CodeNode, NodeType, EdgeType, graph::CodeGraph};
use crate::bridge::pyo3_bridge::FeatureVector;
use std::collections::HashMap;

/// Feature extractor for converting GNN nodes to 986-dimensional vectors
pub struct FeatureExtractor {
    /// Cache for file-level statistics (avoid re-parsing)
    file_stats_cache: HashMap<String, FileStats>,
}

/// File-level statistics for feature extraction
#[derive(Debug, Clone)]
struct FileStats {
    total_lines: usize,
    total_nodes: usize,
    max_depth: usize,
    avg_complexity: f32,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            file_stats_cache: HashMap::new(),
        }
    }

    /// Extract 986-dimensional feature vector from a code node
    /// Returns FeatureVector ready for GraphSAGE model
    pub fn extract_features(
        &mut self,
        node: &CodeNode,
        graph: &CodeGraph,
    ) -> Result<FeatureVector, String> {
        let mut features = Vec::with_capacity(986);

        // --- SECTION 1: Node Identity Features (50 dims) ---
        self.extract_node_identity(&node, &mut features);

        // --- SECTION 2: Structural Features (100 dims) ---
        self.extract_structural_features(&node, graph, &mut features);

        // --- SECTION 3: Complexity Features (100 dims) ---
        self.extract_complexity_features(&node, &mut features);

        // --- SECTION 4: Dependency Features (150 dims) ---
        self.extract_dependency_features(&node, graph, &mut features);

        // --- SECTION 5: Context Features (100 dims) ---
        self.extract_context_features(&node, graph, &mut features);

        // --- SECTION 6: Semantic Features (200 dims) ---
        self.extract_semantic_features(&node, &mut features);

        // --- SECTION 7: Quality Features (100 dims) ---
        self.extract_quality_features(&node, &mut features);

        // --- SECTION 8: Temporal Features (50 dims) ---
        self.extract_temporal_features(&node, &mut features);

        // --- SECTION 9: Statistical Features (124 dims) ---
        self.extract_statistical_features(&node, graph, &mut features);

        // --- SECTION 10: Language Encoding (12 dims) ---
        self.extract_language_encoding(&node, &mut features);

        // Validate total dimensions
        assert_eq!(
            features.len(),
            986,
            "Feature vector must have exactly 986 dimensions, got {}",
            features.len()
        );

        FeatureVector::new(features)
    }

    /// Section 1: Node Identity Features (50 dims)
    /// One-hot encoding for node type + basic properties
    fn extract_node_identity(&self, node: &CodeNode, features: &mut Vec<f32>) {
        // Node type one-hot (6 dims - added Package)
        let mut node_type_vec = vec![0.0; 6];
        match node.node_type {
            NodeType::Function => node_type_vec[0] = 1.0,
            NodeType::Class => node_type_vec[1] = 1.0,
            NodeType::Variable => node_type_vec[2] = 1.0,
            NodeType::Import => node_type_vec[3] = 1.0,
            NodeType::Module => node_type_vec[4] = 1.0,
            NodeType::Package { .. } => node_type_vec[5] = 1.0,
        }
        features.extend(node_type_vec);

        // Name length (normalized, 1 dim)
        features.push(normalize(node.name.len() as f32, 0.0, 100.0));

        // Line span (normalized, 1 dim)
        let line_span = (node.line_end - node.line_start) as f32;
        features.push(normalize(line_span, 0.0, 1000.0));

        // Position in file (normalized, 1 dim)
        features.push(normalize(node.line_start as f32, 0.0, 10000.0));

        // Padding to 50 dims
        features.extend(vec![0.0; 42]);
    }

    /// Section 2: Structural Features (100 dims)
    /// Graph structure, relationships, nesting depth
    fn extract_structural_features(
        &mut self,
        node: &CodeNode,
        graph: &CodeGraph,
        features: &mut Vec<f32>,
    ) {
        // Incoming/outgoing edge counts by type
        let in_calls = graph.get_incoming_edges(&node.id, EdgeType::Calls).len() as f32;
        let out_calls = graph.get_outgoing_edges(&node.id, EdgeType::Calls).len() as f32;
        let in_uses = graph.get_incoming_edges(&node.id, EdgeType::Uses).len() as f32;
        let out_uses = graph.get_outgoing_edges(&node.id, EdgeType::Uses).len() as f32;
        let in_imports = graph.get_incoming_edges(&node.id, EdgeType::Imports).len() as f32;
        let out_imports = graph.get_outgoing_edges(&node.id, EdgeType::Imports).len() as f32;

        features.push(normalize(in_calls, 0.0, 50.0));
        features.push(normalize(out_calls, 0.0, 50.0));
        features.push(normalize(in_uses, 0.0, 50.0));
        features.push(normalize(out_uses, 0.0, 50.0));
        features.push(normalize(in_imports, 0.0, 20.0));
        features.push(normalize(out_imports, 0.0, 20.0));

        // Total degree (6 dims)
        let total_in = in_calls + in_uses + in_imports;
        let total_out = out_calls + out_uses + out_imports;
        features.push(normalize(total_in, 0.0, 100.0));
        features.push(normalize(total_out, 0.0, 100.0));

        // Padding to 100 dims
        features.extend(vec![0.0; 92]);
    }

    /// Section 3: Complexity Features (100 dims)
    /// Cyclomatic complexity, nesting depth, branching
    fn extract_complexity_features(&self, node: &CodeNode, features: &mut Vec<f32>) {
        // Simplified complexity estimation based on line count
        let line_count = (node.line_end - node.line_start) as f32;
        
        // Estimated cyclomatic complexity (higher for functions)
        let complexity = match node.node_type {
            NodeType::Function => line_count / 5.0, // ~1 complexity per 5 lines
            NodeType::Class => line_count / 10.0,
            _ => 1.0,
        };
        features.push(normalize(complexity, 0.0, 50.0));

        // Line count (normalized)
        features.push(normalize(line_count, 0.0, 500.0));

        // Padding to 100 dims
        features.extend(vec![0.0; 98]);
    }

    /// Section 4: Dependency Features (150 dims)
    /// Import patterns, external dependencies, coupling
    fn extract_dependency_features(
        &self,
        node: &CodeNode,
        graph: &CodeGraph,
        features: &mut Vec<f32>,
    ) {
        // Count unique dependencies
        let deps = graph.get_all_dependencies(&node.id);
        features.push(normalize(deps.len() as f32, 0.0, 100.0));

        // Count transitive dependencies (depth-2)
        let mut transitive_count = 0;
        for dep_id in &deps {
            transitive_count += graph.get_all_dependencies(dep_id).len();
        }
        features.push(normalize(transitive_count as f32, 0.0, 500.0));

        // Padding to 150 dims
        features.extend(vec![0.0; 148]);
    }

    /// Section 5: Context Features (100 dims)
    /// File context, module structure, neighboring nodes
    fn extract_context_features(
        &self,
        node: &CodeNode,
        graph: &CodeGraph,
        features: &mut Vec<f32>,
    ) {
        // File-level features
        let file_nodes = graph.get_nodes_in_file(&node.file_path);
        features.push(normalize(file_nodes.len() as f32, 0.0, 200.0));

        // Position in file (relative to other nodes)
        let earlier_nodes = file_nodes
            .iter()
            .filter(|n| n.line_start < node.line_start)
            .count();
        let relative_position = if !file_nodes.is_empty() {
            earlier_nodes as f32 / file_nodes.len() as f32
        } else {
            0.0
        };
        features.push(relative_position);

        // Padding to 100 dims
        features.extend(vec![0.0; 98]);
    }

    /// Section 6: Semantic Features (200 dims)
    /// Name embeddings, identifier patterns, naming conventions
    fn extract_semantic_features(&self, node: &CodeNode, features: &mut Vec<f32>) {
        // Name characteristics
        let name_has_underscore = if node.name.contains('_') { 1.0 } else { 0.0 };
        let name_is_camel_case = is_camel_case(&node.name);
        let name_is_snake_case = is_snake_case(&node.name);
        let name_is_uppercase = if node.name.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()) {
            1.0
        } else {
            0.0
        };

        features.push(name_has_underscore);
        features.push(name_is_camel_case);
        features.push(name_is_snake_case);
        features.push(name_is_uppercase);

        // Padding to 200 dims
        features.extend(vec![0.0; 196]);
    }

    /// Section 7: Quality Features (100 dims)
    /// Code quality indicators, potential issues
    fn extract_quality_features(&self, node: &CodeNode, features: &mut Vec<f32>) {
        // Function length (quality indicator)
        let is_too_long = if (node.line_end - node.line_start) > 50 {
            1.0
        } else {
            0.0
        };
        features.push(is_too_long);

        // Name quality (too short or too long)
        let name_too_short = if node.name.len() < 3 { 1.0 } else { 0.0 };
        let name_too_long = if node.name.len() > 50 { 1.0 } else { 0.0 };
        features.push(name_too_short);
        features.push(name_too_long);

        // Padding to 100 dims
        features.extend(vec![0.0; 97]);
    }

    /// Section 8: Temporal Features (50 dims)
    /// (Placeholder for future: last modified, change frequency)
    fn extract_temporal_features(&self, _node: &CodeNode, features: &mut Vec<f32>) {
        // Padding (temporal features require git history)
        features.extend(vec![0.0; 50]);
    }

    /// Section 9: Statistical Features (124 dims)
    /// Aggregate statistics, z-scores, outlier detection
    fn extract_statistical_features(
        &self,
        node: &CodeNode,
        graph: &CodeGraph,
        features: &mut Vec<f32>,
    ) {
        // File-level aggregates
        let file_nodes = graph.get_nodes_in_file(&node.file_path);
        
        // Average line count in file
        let avg_lines = if !file_nodes.is_empty() {
            file_nodes
                .iter()
                .map(|n| (n.line_end - n.line_start) as f32)
                .sum::<f32>()
                / file_nodes.len() as f32
        } else {
            0.0
        };
        features.push(normalize(avg_lines, 0.0, 100.0));

        // Node's deviation from average
        let node_lines = (node.line_end - node.line_start) as f32;
        let deviation = if avg_lines > 0.0 {
            (node_lines - avg_lines) / avg_lines
        } else {
            0.0
        };
        features.push(normalize(deviation, -5.0, 5.0));

        // Padding to 124 dims
        features.extend(vec![0.0; 122]);
    }

    /// Section 10: Language Encoding (12 dims)
    /// One-hot encoding: [Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin]
    fn extract_language_encoding(&self, node: &CodeNode, features: &mut Vec<f32>) {
        let path = std::path::Path::new(&node.file_path);
        let extension = path.extension().and_then(|s| s.to_str());

        let mut lang_vec = vec![0.0; 12];
        match extension {
            Some("py") => lang_vec[0] = 1.0,
            Some("js") | Some("jsx") => lang_vec[1] = 1.0,
            Some("ts") | Some("tsx") => lang_vec[2] = 1.0,
            Some("rs") => lang_vec[3] = 1.0,
            Some("go") => lang_vec[4] = 1.0,
            Some("java") => lang_vec[5] = 1.0,
            Some("c") | Some("h") => lang_vec[6] = 1.0,
            Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hxx") => lang_vec[7] = 1.0,
            Some("rb") => lang_vec[8] = 1.0,
            Some("php") => lang_vec[9] = 1.0,
            Some("swift") => lang_vec[10] = 1.0,
            Some("kt") | Some("kts") => lang_vec[11] = 1.0,
            _ => lang_vec[11] = 1.0, // Default to Kotlin slot for unknown
        }
        features.extend(lang_vec);
    }
}

/// Normalize value to [0, 1] range given min and max
fn normalize(value: f32, min: f32, max: f32) -> f32 {
    if max == min {
        return 0.5;
    }
    ((value - min) / (max - min)).max(0.0).min(1.0)
}

/// Check if name follows camelCase convention
fn is_camel_case(name: &str) -> f32 {
    if name.is_empty() {
        return 0.0;
    }
    let has_lowercase = name.chars().any(|c| c.is_lowercase());
    let has_uppercase = name.chars().any(|c| c.is_uppercase());
    let no_underscores = !name.contains('_');
    let starts_lowercase = name.chars().next().map_or(false, |c| c.is_lowercase());

    if has_lowercase && has_uppercase && no_underscores && starts_lowercase {
        1.0
    } else {
        0.0
    }
}

/// Check if name follows snake_case convention
fn is_snake_case(name: &str) -> f32 {
    if name.is_empty() {
        return 0.0;
    }
    let all_lower = name.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_lowercase());
    let has_underscores = name.contains('_');
    let no_uppercase = !name.chars().any(|c| c.is_uppercase());

    if all_lower && no_uppercase && (has_underscores || name.len() < 20) {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::{CodeNode, NodeType, CodeEdge, EdgeType};
    use crate::gnn::graph::CodeGraph;

    #[test]
    fn test_feature_extraction() {
        let mut extractor = FeatureExtractor::new();
        let graph = CodeGraph::new();

        let node = CodeNode {
            id: "test_func".to_string(),
            node_type: NodeType::Function,
            name: "calculate_sum".to_string(),
            file_path: "test.py".to_string(),
            line_start: 10,
            line_end: 25,
            ..Default::default()
        };

        let features = extractor.extract_features(&node, &graph).unwrap();
        
        // Verify 986 dimensions
        assert_eq!(features.features.len(), 986);
        
        // Verify language encoding (Python should be [1,0,0,0,0,0,0,0,0,0,0,0])
        assert_eq!(features.features[974], 1.0); // Python
        assert_eq!(features.features[975], 0.0); // JavaScript
        assert_eq!(features.features[976], 0.0); // TypeScript
        assert_eq!(features.features[977], 0.0); // Rust
        assert_eq!(features.features[978], 0.0); // Go
        assert_eq!(features.features[979], 0.0); // Java
        assert_eq!(features.features[980], 0.0); // C
        assert_eq!(features.features[981], 0.0); // C++
        assert_eq!(features.features[982], 0.0); // Ruby
        assert_eq!(features.features[983], 0.0); // PHP
        assert_eq!(features.features[984], 0.0); // Swift
        assert_eq!(features.features[985], 0.0); // Kotlin
    }

    #[test]
    fn test_language_encoding_javascript() {
        let mut extractor = FeatureExtractor::new();
        let graph = CodeGraph::new();

        let node = CodeNode {
            id: "test_func".to_string(),
            node_type: NodeType::Function,
            name: "calculateSum".to_string(),
            file_path: "test.js".to_string(),
            line_start: 5,
            line_end: 15,
            ..Default::default()
        };

        let features = extractor.extract_features(&node, &graph).unwrap();
        
        // Verify language encoding (JavaScript should be [0,1,0,0,0,0,0,0,0,0,0,0])
        assert_eq!(features.features[974], 0.0); // Python
        assert_eq!(features.features[975], 1.0); // JavaScript
        assert_eq!(features.features[976], 0.0); // TypeScript
        assert_eq!(features.features[977], 0.0); // Rust
    }

    #[test]
    fn test_naming_conventions() {
        assert_eq!(is_camel_case("camelCase"), 1.0);
        assert_eq!(is_camel_case("snake_case"), 0.0);
        assert_eq!(is_snake_case("snake_case"), 1.0);
        assert_eq!(is_snake_case("camelCase"), 0.0);
    }

    #[test]
    fn test_normalization() {
        assert_eq!(normalize(50.0, 0.0, 100.0), 0.5);
        assert_eq!(normalize(0.0, 0.0, 100.0), 0.0);
        assert_eq!(normalize(100.0, 0.0, 100.0), 1.0);
        assert_eq!(normalize(150.0, 0.0, 100.0), 1.0); // Clamped
        assert_eq!(normalize(-50.0, 0.0, 100.0), 0.0); // Clamped
    }

    #[test]
    fn test_extraction_performance() {
        use std::time::Instant;

        let mut extractor = FeatureExtractor::new();
        let mut graph = CodeGraph::new();

        // Create a more realistic graph with multiple nodes
        for i in 0..10 {
            let node = CodeNode {
                id: format!("func_{}", i),
                node_type: NodeType::Function,
                name: format!("function_{}", i),
                file_path: "test.py".to_string(),
                line_start: i * 10,
                line_end: i * 10 + 8,
                ..Default::default()
            };
            graph.add_node(node.clone());
        }

        // Add some edges
        for i in 0..9 {
            let edge = CodeEdge {
                edge_type: EdgeType::Calls,
                source_id: format!("func_{}", i),
                target_id: format!("func_{}", i + 1),
            };
            graph.add_edge(edge).ok();
        }

        // Warm-up
        let test_node = CodeNode {
            id: "test_func".to_string(),
            node_type: NodeType::Function,
            name: "test_function".to_string(),
            file_path: "test.py".to_string(),
            line_start: 100,
            line_end: 125,
            ..Default::default()
        };
        graph.add_node(test_node.clone());
        let _ = extractor.extract_features(&test_node, &graph);

        // Benchmark: 100 extractions
        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            extractor.extract_features(&test_node, &graph).unwrap();
        }

        let duration = start.elapsed();
        let avg_ms = duration.as_millis() as f64 / iterations as f64;

        println!("\n=== Feature Extraction Benchmark ===");
        println!("Total time for {} extractions: {:?}", iterations, duration);
        println!("Average time per extraction: {:.3} ms", avg_ms);
        println!("Target: <1ms");

        if avg_ms < 1.0 {
            println!("✓ PASSED: {:.1}x better than target!", 1.0 / avg_ms);
        } else {
            println!("✗ FAILED: {:.1}x slower than target", avg_ms / 1.0);
        }

        // Assert we meet the <1ms target
        assert!(
            avg_ms < 1.0,
            "Feature extraction {:.3}ms exceeds 1ms target",
            avg_ms
        );
    }
}
