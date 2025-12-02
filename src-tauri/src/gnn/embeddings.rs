// File: src-tauri/src/gnn/embeddings.rs
// Purpose: Lightweight semantic embeddings for code nodes (Rust-native)
// Dependencies: fastembed-rs (pure Rust, no Python)
// Last Updated: December 1, 2025
//
// Provides semantic understanding for GNN nodes without external vector DB.
// Uses quantized ONNX models for fast inference on CPU/GPU.
//
// Models:
// - all-MiniLM-L6-v2: 384 dims, 22MB, <10ms inference
// - BAAI/bge-small-en-v1.5: 384 dims, 33MB, <15ms inference
//
// Performance targets:
// - Embedding generation: <10ms per node
// - Batch processing: <100ms for 100 nodes
// - Memory: <50MB model cache

use super::CodeNode;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel as FastEmbedModel};

/// Embedding model type
#[derive(Debug, Clone, Copy)]
pub enum EmbeddingModel {
    /// all-MiniLM-L6-v2 (384 dims, fast, good for code)
    MiniLM,
    /// BAAI/bge-small-en-v1.5 (384 dims, better quality)
    BGESmall,
}

impl EmbeddingModel {
    pub fn dimensions(&self) -> usize {
        match self {
            EmbeddingModel::MiniLM => 384,
            EmbeddingModel::BGESmall => 384,
        }
    }
    
    fn to_fastembed_model(&self) -> FastEmbedModel {
        match self {
            EmbeddingModel::MiniLM => FastEmbedModel::AllMiniLML6V2,
            EmbeddingModel::BGESmall => FastEmbedModel::BGESmallENV15,
        }
    }
}

/// Semantic embedding generator for code nodes
/// 
/// Uses quantized ONNX models via fastembed-rs for pure Rust inference.
/// No Python dependencies, runs on CPU/GPU with auto device selection.
pub struct EmbeddingGenerator {
    model_type: EmbeddingModel,
    model: TextEmbedding,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl EmbeddingGenerator {
    /// Create new embedding generator with specified model
    pub fn new(model: EmbeddingModel) -> Result<Self, String> {
        // Initialize fastembed model with progress bar
        let fastembed_model = TextEmbedding::try_new(
            InitOptions::new(model.to_fastembed_model())
                .with_show_download_progress(true)
        ).map_err(|e| format!("Failed to initialize embedding model: {}", e))?;
        
        Ok(Self {
            model_type: model,
            model: fastembed_model,
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Generate embedding for code node
    /// 
    /// Combines function/class name, docstring, and code snippet
    /// into single semantic representation.
    pub fn generate_embedding(&mut self, node: &CodeNode) -> Result<Vec<f32>, String> {
        // Build text representation
        let text = self.build_text_representation(node);
        
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(embedding) = cache.get(&text) {
                return Ok(embedding.clone());
            }
        }
        
        // Generate embedding using fastembed
        let embeddings = self.model
            .embed(vec![text.clone()], None)
            .map_err(|e| format!("Embedding generation failed: {}", e))?;
        
        let embedding = embeddings
            .first()
            .ok_or("No embedding returned")?
            .to_vec();
        
        // Cache result
        {
            let mut cache = self.cache.write();
            cache.insert(text, embedding.clone());
        }
        
        Ok(embedding)
    }
    
    /// Generate embedding for arbitrary text (useful for intent/query matching)
    pub fn generate_text_embedding(&mut self, text: &str) -> Result<Vec<f32>, String> {
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(embedding) = cache.get(text) {
                return Ok(embedding.clone());
            }
        }
        
        // Generate embedding
        let embeddings = self.model
            .embed(vec![text.to_string()], None)
            .map_err(|e| format!("Embedding generation failed: {}", e))?;
        
        let embedding = embeddings
            .first()
            .ok_or("No embedding returned")?
            .to_vec();
        
        // Cache result
        {
            let mut cache = self.cache.write();
            cache.insert(text.to_string(), embedding.clone());
        }
        
        Ok(embedding)
    }
    
    /// Generate embeddings for multiple nodes (batch processing)
    pub fn generate_embeddings_batch(
        &mut self, 
        nodes: &[&CodeNode]
    ) -> Result<Vec<Vec<f32>>, String> {
        // Build text representations
        let texts: Vec<String> = nodes
            .iter()
            .map(|node| self.build_text_representation(node))
            .collect();
        
        // Generate embeddings in batch (much faster!)
        let embeddings = self.model
            .embed(texts, None)
            .map_err(|e| format!("Batch embedding generation failed: {}", e))?;
        
        Ok(embeddings)
    }
    
    /// Build text representation for embedding
    /// 
    /// Format: "{node_type}: {name}\n{docstring}\n{code_snippet}"
    fn build_text_representation(&self, node: &CodeNode) -> String {
        let mut parts = Vec::new();
        
        // Node type and name
        parts.push(format!("{:?}: {}", node.node_type, node.name));
        
        // Docstring if available
        if let Some(docstring) = &node.docstring {
            parts.push(docstring.clone());
        }
        
        // Code snippet (truncate to 500 chars for efficiency)
        if let Some(code) = &node.code_snippet {
            let truncated = if code.len() > 500 {
                format!("{}...", &code[..500])
            } else {
                code.clone()
            };
            parts.push(truncated);
        }
        
        parts.join("\n")
    }
    
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (magnitude_a * magnitude_b)
    }
    
    /// Clear embedding cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }
    
    /// Get cache size (number of cached embeddings)
    pub fn cache_size(&self) -> usize {
        let cache = self.cache.read();
        cache.len()
    }
}

impl Default for EmbeddingGenerator {
    fn default() -> Self {
        Self::new(EmbeddingModel::MiniLM)
            .expect("Failed to create default embedding generator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::NodeType;
    
    #[test]
    fn test_embedding_generator_creation() {
        let generator = EmbeddingGenerator::new(EmbeddingModel::MiniLM).unwrap();
        // Model successfully created - dimensions are known to be 384 for MiniLM
        assert!(generator.cache.read().is_empty());
    }
    
    #[test]
    fn test_text_representation() {
        let generator = EmbeddingGenerator::default();
        
        let node = CodeNode {
            id: "test::func".to_string(),
            node_type: NodeType::Function,
            name: "test_function".to_string(),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 10,
            semantic_embedding: None,
            code_snippet: Some("def test_function():\n    return 42".to_string()),
            docstring: Some("Test function docstring".to_string()),
            ..Default::default()
        };
        
        let text = generator.build_text_representation(&node);
        assert!(text.contains("Function: test_function"));
        assert!(text.contains("Test function docstring"));
        assert!(text.contains("def test_function"));
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((EmbeddingGenerator::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((EmbeddingGenerator::cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_embedding_caching() {
        let mut generator = EmbeddingGenerator::default();
        
        let node = CodeNode {
            id: "test::func".to_string(),
            node_type: NodeType::Function,
            name: "test_function".to_string(),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 10,
            semantic_embedding: None,
            code_snippet: Some("def test_function(): pass".to_string()),
            docstring: None,
            ..Default::default()
        };
        
        // Generate twice
        let _emb1 = generator.generate_embedding(&node).unwrap();
        let _emb2 = generator.generate_embedding(&node).unwrap();
        
        // Should be cached
        assert_eq!(generator.cache_size(), 1);
    }
}
