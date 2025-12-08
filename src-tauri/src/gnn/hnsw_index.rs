// File: src-tauri/src/gnn/hnsw_index.rs
// Purpose: HNSW (Hierarchical Navigable Small World) index for fast semantic similarity search
// Dependencies: hnsw_rs
// Last Updated: December 3, 2025
//
// Implements HNSW indexing for O(log n) similarity search instead of O(n) linear scan.
// This is a FERRARI MVP REQUIREMENT - spec explicitly requires HNSW from start.
//
// Performance targets:
// - Index build: <5s for 10k nodes
// - Search: <10ms for 10k+ nodes (vs 50ms linear scan)
// - Memory: ~1.5MB per 1k nodes with 384-dim vectors
//
// HNSW Parameters:
// - M (neighbors per layer): 16 (good balance for code search)
// - ef_construction: 200 (build quality)
// - ef_search: 50 (search quality vs speed)

use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;
use super::CodeNode;

/// HNSW index for semantic similarity search
/// 
/// Provides O(log n) similarity search using HNSW algorithm.
/// Automatically maintained as nodes are added/removed.
pub struct HnswIndex {
    /// HNSW index for fast similarity search
    index: Arc<RwLock<Hnsw<'static, f32, DistCosine>>>,
    /// Map from HNSW DataId to node_id (String)
    id_map: Arc<RwLock<Vec<String>>>,
    /// Map from node_id to HNSW DataId for updates
    reverse_map: Arc<RwLock<std::collections::HashMap<String, usize>>>,
    /// Dimensionality of embeddings
    dimensions: usize,
    /// Number of neighbors per layer (M parameter)
    neighbors: usize,
    /// Construction ef parameter
    ef_construction: usize,
    /// Search ef parameter
    ef_search: usize,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("dimensions", &self.dimensions)
            .field("neighbors", &self.neighbors)
            .field("ef_construction", &self.ef_construction)
            .field("ef_search", &self.ef_search)
            .field("nodes_count", &self.id_map.read().len())
            .finish()
    }
}

impl HnswIndex {
    /// Create new HNSW index
    /// 
    /// # Arguments
    /// * `dimensions` - Embedding vector dimensions (384 for MiniLM)
    /// * `max_elements` - Expected maximum number of nodes
    /// 
    /// # Returns
    /// New HNSW index ready for use
    pub fn new(dimensions: usize, max_elements: usize) -> Self {
        // HNSW parameters optimized for code similarity search
        let neighbors = 16; // M parameter - neighbors per layer
        let ef_construction = 200; // Construction quality
        let max_layers = 16; // Maximum graph layers
        
        // Create HNSW index with cosine distance
        let hnsw = Hnsw::<f32, DistCosine>::new(
            neighbors,
            max_elements,
            max_layers,
            ef_construction,
            DistCosine {},
        );
        
        Self {
            index: Arc::new(RwLock::new(hnsw)),
            id_map: Arc::new(RwLock::new(Vec::with_capacity(max_elements))),
            reverse_map: Arc::new(RwLock::new(std::collections::HashMap::with_capacity(max_elements))),
            dimensions,
            neighbors,
            ef_construction,
            ef_search: 50, // Default search quality
        }
    }
    
    /// Add node to HNSW index
    /// 
    /// # Arguments
    /// * `node` - Code node with embedding
    /// 
    /// # Returns
    /// Ok if added successfully, Err if node has no embedding or other error
    pub fn add_node(&self, node: &CodeNode) -> Result<(), String> {
        let embedding = node.semantic_embedding
            .as_ref()
            .ok_or_else(|| format!("Node {} has no embedding", node.id))?;
        
        if embedding.len() != self.dimensions {
            return Err(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            ));
        }
        
        // Get next DataId
        let data_id = {
            let id_map = self.id_map.read();
            id_map.len()
        };
        
        // Insert into HNSW index
        {
            let mut index = self.index.write();
            index.insert((embedding, data_id));
        }
        
        // Update mappings
        {
            let mut id_map = self.id_map.write();
            id_map.push(node.id.clone());
        }
        {
            let mut reverse_map = self.reverse_map.write();
            reverse_map.insert(node.id.clone(), data_id);
        }
        
        Ok(())
    }
    
    /// Remove node from HNSW index
    /// 
    /// Note: HNSW doesn't support efficient deletion, so we just mark it as deleted
    /// in our mappings. The actual HNSW graph is left intact for performance.
    /// A full rebuild is needed periodically if many deletions occur.
    pub fn remove_node(&self, node_id: &str) -> Result<(), String> {
        let mut reverse_map = self.reverse_map.write();
        reverse_map.remove(node_id)
            .ok_or_else(|| format!("Node {} not found in index", node_id))?;
        Ok(())
    }
    
    /// Update node in HNSW index
    /// 
    /// Since HNSW doesn't support efficient updates, we remove and re-add.
    /// This is acceptable since updates are rare compared to searches.
    pub fn update_node(&self, node: &CodeNode) -> Result<(), String> {
        // Check if node exists
        {
            let reverse_map = self.reverse_map.read();
            if !reverse_map.contains_key(&node.id) {
                return Err(format!("Node {} not found in index", node.id));
            }
        }
        
        // For HNSW, we need to rebuild with updated node
        // In practice, update the reverse_map to point to new DataId after re-add
        self.remove_node(&node.id)?;
        self.add_node(node)?;
        
        Ok(())
    }
    
    /// Search for similar nodes using HNSW index
    /// 
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `k` - Number of nearest neighbors to return
    /// 
    /// # Returns
    /// Vector of (node_id, similarity_score) tuples, sorted by similarity (descending)
    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        if query_embedding.len() != self.dimensions {
            return Err(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                query_embedding.len()
            ));
        }
        
        // Set search ef parameter (quality vs speed tradeoff)
        {
            let mut index = self.index.write();
            index.set_searching_mode(true);
        }
        
        // Perform HNSW search
        let neighbors = {
            let index = self.index.read();
            index.search(query_embedding, k, self.ef_search)
        };
        
        // Convert DataIds to node_ids with similarity scores
        let id_map = self.id_map.read();
        let reverse_map = self.reverse_map.read();
        
        let results: Vec<(String, f32)> = neighbors
            .iter()
            .filter_map(|neighbor| {
                let data_id = neighbor.d_id;
                if data_id < id_map.len() {
                    let node_id = &id_map[data_id];
                    // Only return if node still exists (not marked as deleted)
                    if reverse_map.contains_key(node_id) {
                        // Convert distance to similarity score
                        // Cosine distance: [0, 2], where 0 = identical
                        // Similarity: [0, 1], where 1 = identical
                        let similarity = 1.0 - (neighbor.distance / 2.0);
                        Some((node_id.clone(), similarity))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        Ok(results)
    }
    
    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let index = self.index.read();
        let id_map = self.id_map.read();
        let reverse_map = self.reverse_map.read();
        
        IndexStats {
            total_nodes: id_map.len(),
            active_nodes: reverse_map.len(),
            dimensions: self.dimensions,
            neighbors_per_layer: self.neighbors,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            index_layers: index.get_max_level(),
        }
    }
    
    /// Set search quality parameter (ef_search)
    /// 
    /// Higher values = better quality but slower search
    /// Typical range: 10-200
    /// Default: 50
    pub fn set_search_quality(&mut self, ef_search: usize) {
        self.ef_search = ef_search;
    }
    
    /// Check if index needs rebuild due to many deletions
    /// 
    /// Returns true if more than 20% of nodes have been deleted
    pub fn needs_rebuild(&self) -> bool {
        let id_map = self.id_map.read();
        let reverse_map = self.reverse_map.read();
        
        let total = id_map.len();
        let active = reverse_map.len();
        
        if total == 0 {
            return false;
        }
        
        let deletion_ratio = (total - active) as f32 / total as f32;
        deletion_ratio > 0.2
    }
    
    /// Rebuild index from scratch with only active nodes
    /// 
    /// Should be called periodically if many deletions occur.
    /// This creates a new HNSW index with only non-deleted nodes.
    pub fn rebuild(&mut self, nodes: &[CodeNode]) -> Result<(), String> {
        // Create new index
        let mut new_index = Hnsw::<f32, DistCosine>::new(
            self.neighbors,
            nodes.len(),
            16,
            self.ef_construction,
            DistCosine {},
        );
        
        let mut new_id_map = Vec::with_capacity(nodes.len());
        let mut new_reverse_map = std::collections::HashMap::with_capacity(nodes.len());
        
        // Insert all nodes
        for (data_id, node) in nodes.iter().enumerate() {
            if let Some(embedding) = &node.semantic_embedding {
                new_index.insert((embedding.as_slice(), data_id));
                new_id_map.push(node.id.clone());
                new_reverse_map.insert(node.id.clone(), data_id);
            }
        }
        
        // Replace old index
        *self.index.write() = new_index;
        *self.id_map.write() = new_id_map;
        *self.reverse_map.write() = new_reverse_map;
        
        Ok(())
    }
}

/// HNSW index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total nodes ever added
    pub total_nodes: usize,
    /// Active nodes (not deleted)
    pub active_nodes: usize,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Neighbors per layer (M parameter)
    pub neighbors_per_layer: usize,
    /// Construction ef parameter
    pub ef_construction: usize,
    /// Search ef parameter
    pub ef_search: usize,
    /// Number of layers in HNSW graph
    pub index_layers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_node(id: &str, embedding: Vec<f32>) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            node_type: super::super::NodeType::Function,
            name: format!("test_{}", id),
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 10,
            semantic_embedding: Some(embedding),
            ..Default::default()
        }
    }
    
    #[test]
    fn test_hnsw_basic_operations() {
        let index = HnswIndex::new(384, 1000);
        
        // Create test nodes with random embeddings
        let node1 = create_test_node("node1", vec![0.5; 384]);
        let node2 = create_test_node("node2", vec![0.6; 384]);
        
        // Add nodes
        assert!(index.add_node(&node1).is_ok());
        assert!(index.add_node(&node2).is_ok());
        
        // Search
        let results = index.search(&vec![0.5; 384], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "node1"); // Should be most similar
        
        // Stats
        let stats = index.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.active_nodes, 2);
    }
    
    #[test]
    fn test_hnsw_deletion() {
        let index = HnswIndex::new(384, 1000);
        
        let node1 = create_test_node("node1", vec![0.5; 384]);
        let node2 = create_test_node("node2", vec![0.6; 384]);
        
        index.add_node(&node1).unwrap();
        index.add_node(&node2).unwrap();
        
        // Remove node1
        assert!(index.remove_node("node1").is_ok());
        
        // Search should only return node2
        let results = index.search(&vec![0.5; 384], 2).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "node2");
        
        let stats = index.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.active_nodes, 1);
    }
}
