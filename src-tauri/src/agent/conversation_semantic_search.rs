// File: src-tauri/src/agent/conversation_semantic_search.rs
// Purpose: Semantic search for conversation memory using vector embeddings (CONV-005)
// Last Updated: December 9, 2025
//
// This module extends conversation memory with semantic search capabilities:
// - Generate embeddings for message content using fastembed
// - Build HNSW index for efficient similarity search
// - Search conversations by semantic similarity
// - Hybrid search combining keyword and semantic matching

use super::conversation_memory::{Conversation, Message, ConversationMemory};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Semantic search engine for conversations
pub struct ConversationSemanticSearch {
    embedding_model: Arc<Mutex<TextEmbedding>>,
    index: Arc<Mutex<Option<Hnsw<'static, f32, DistCosine>>>>,
    message_id_to_index: Arc<Mutex<HashMap<String, usize>>>,
    index_to_message_id: Arc<Mutex<HashMap<usize, String>>>,
    dimension: usize,
}

impl ConversationSemanticSearch {
    /// Create new semantic search engine
    /// Uses fastembed's default "sentence-transformers/all-MiniLM-L6-v2" model (384 dimensions)
    pub fn new() -> Result<Self, String> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        ).map_err(|e| format!("Failed to initialize embedding model: {}", e))?;

        Ok(Self {
            embedding_model: Arc::new(Mutex::new(model)),
            index: Arc::new(Mutex::new(None)),
            message_id_to_index: Arc::new(Mutex::new(HashMap::new())),
            index_to_message_id: Arc::new(Mutex::new(HashMap::new())),
            dimension: 384, // all-MiniLM-L6-v2 dimension
        })
    }

    /// Build HNSW index from all messages in conversation memory
    pub fn build_index(&self, memory: &ConversationMemory) -> Result<usize, String> {
        // Get all conversations
        let conversations = memory.search_conversations(&super::conversation_memory::SearchFilter {
            keyword: None,
            start_date: None,
            end_date: None,
            tags: vec![],
            session_type: None,
        }).map_err(|e| format!("Failed to load conversations: {}", e))?;

        let mut all_messages = Vec::new();
        for conversation in conversations {
            let messages = memory.load_recent_messages(&conversation.id, 1000)
                .map_err(|e| format!("Failed to load conversation messages: {}", e))?;
            all_messages.extend(messages);
        }

        if all_messages.is_empty() {
            return Ok(0);
        }

        // Generate embeddings for all messages
        let contents: Vec<String> = all_messages.iter()
            .map(|m| m.content.clone())
            .collect();

        let embeddings = self.generate_embeddings(&contents)?;

        // Build HNSW index
        let max_nb_connection = 16; // M parameter
        let ef_construction = 200; // efConstruction parameter
        let max_elements = all_messages.len();
        
        let hnsw: Hnsw<'static, f32, DistCosine> = Hnsw::new(
            max_nb_connection,
            max_elements,
            ef_construction,
            self.dimension,
            DistCosine,
        );

        // Insert embeddings into index
        let mut message_id_to_index = self.message_id_to_index.lock().unwrap();
        let mut index_to_message_id = self.index_to_message_id.lock().unwrap();

        for (idx, (message, embedding)) in all_messages.iter().zip(embeddings.iter()).enumerate() {
            let data = (embedding.as_slice(), idx);
            hnsw.insert(data);
            message_id_to_index.insert(message.id.clone(), idx);
            index_to_message_id.insert(idx, message.id.clone());
        }

        // Store the built index
        let mut index = self.index.lock().unwrap();
        *index = Some(hnsw);

        Ok(all_messages.len())
    }

    /// Generate embeddings for a list of texts
    fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut model = self.embedding_model.lock().unwrap();
        let embeddings = model.embed(texts.to_vec(), None)
            .map_err(|e| format!("Failed to generate embeddings: {}", e))?;
        Ok(embeddings)
    }

    /// Search for messages by semantic similarity
    /// Returns message IDs sorted by relevance
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<(String, f32)>, String> {
        let index = self.index.lock().unwrap();
        let hnsw = index.as_ref()
            .ok_or_else(|| "Index not built. Call build_index() first.".to_string())?;

        // Generate embedding for query
        let query_embeddings = self.generate_embeddings(&[query.to_string()])?;
        let query_embedding = query_embeddings.first()
            .ok_or_else(|| "Failed to generate query embedding".to_string())?;

        // Search HNSW index
        let ef_search = 50; // efSearch parameter
        let results = hnsw.search(query_embedding.as_slice(), top_k, ef_search);

        // Map indices back to message IDs
        let index_to_message_id = self.index_to_message_id.lock().unwrap();
        let mut message_results = Vec::new();

        for neighbour in results {
            if let Some(message_id) = index_to_message_id.get(&neighbour.d_id) {
                // Convert distance to similarity score (1 - cosine distance)
                let similarity = 1.0 - neighbour.distance;
                message_results.push((message_id.clone(), similarity));
            }
        }

        Ok(message_results)
    }

    /// Hybrid search combining keyword and semantic matching
    /// Returns conversation IDs with relevance scores
    pub fn hybrid_search(
        &self,
        memory: &ConversationMemory,
        query: &str,
        keyword_weight: f32,
        semantic_weight: f32,
        top_k: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        // Keyword search
        let keyword_results = memory.search_conversations(&super::conversation_memory::SearchFilter {
            keyword: Some(query.to_string()),
            start_date: None,
            end_date: None,
            tags: vec![],
            session_type: None,
        }).map_err(|e| format!("Keyword search failed: {}", e))?;

        // Semantic search
        let semantic_results = self.search(query, top_k * 2)?; // Get more results for merging

        // Combine scores
        let mut conversation_scores: HashMap<String, f32> = HashMap::new();

        // Add keyword scores
        for conversation in keyword_results {
            conversation_scores.insert(conversation.id.clone(), keyword_weight);
        }

        // Add semantic scores
        for (message_id, similarity) in semantic_results {
            // Get conversation ID for this message
            let score = semantic_weight * similarity;
            
            // For now, we'll use a simplified approach
            if let Ok(conversation) = self.get_conversation_for_message(memory, &message_id) {
                *conversation_scores.entry(conversation.id).or_insert(0.0) += score;
            }
        }

        // Sort by score and return top K
        let mut results: Vec<(String, f32)> = conversation_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// Get conversation for a message (helper method)
    fn get_conversation_for_message(
        &self,
        memory: &ConversationMemory,
        message_id: &str,
    ) -> Result<Conversation, String> {
        // This is a simplified implementation
        let conversations = memory.search_conversations(&super::conversation_memory::SearchFilter {
            keyword: None,
            start_date: None,
            end_date: None,
            tags: vec![],
            session_type: None,
        }).map_err(|e| format!("Failed to load conversations: {}", e))?;

        for conversation in conversations {
            let messages = memory.load_recent_messages(&conversation.id, 1000)
                .map_err(|e| format!("Failed to load conversation: {}", e))?;
            
            if messages.iter().any(|m| m.id == message_id) {
                return Ok(conversation);
            }
        }

        Err(format!("Message {} not found in any conversation", message_id))
    }

    /// Add a new message to the index (incremental update)
    pub fn add_message(&self, message: &Message) -> Result<(), String> {
        let mut index = self.index.lock().unwrap();
        
        if index.is_none() {
            return Err("Index not built. Call build_index() first.".to_string());
        }

        // Generate embedding for new message
        let embedding = self.generate_embeddings(&[message.content.clone()])?
            .into_iter()
            .next()
            .ok_or_else(|| "Failed to generate embedding".to_string())?;

        // Get next index
        let mut message_id_to_index = self.message_id_to_index.lock().unwrap();
        let mut index_to_message_id = self.index_to_message_id.lock().unwrap();
        let next_idx = message_id_to_index.len();

        // Insert into HNSW index
        if let Some(ref mut hnsw) = *index {
            let data: (&[f32], usize) = (embedding.as_slice(), next_idx);
            hnsw.insert(data);
        }

        // Update mappings
        message_id_to_index.insert(message.id.clone(), next_idx);
        index_to_message_id.insert(next_idx, message.id.clone());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::conversation_memory::{ConversationMemory, MessageRole};

    #[test]
    #[ignore] // Slow: downloads ML model
    fn test_semantic_search_creation() {
        let search = ConversationSemanticSearch::new();
        assert!(search.is_ok());
    }

    #[test]
    #[ignore] // Slow: downloads ML model
    fn test_embedding_generation() {
        let search = ConversationSemanticSearch::new().expect("Failed to create search engine");
        let texts = vec![
            "Create a Python function".to_string(),
            "Generate unit tests".to_string(),
        ];
        
        let embeddings = search.generate_embeddings(&texts);
        assert!(embeddings.is_ok());
        
        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384); // all-MiniLM-L6-v2 dimension
    }

    #[test]
    #[ignore] // Slow: downloads ML model and builds index
    fn test_build_and_search_index() {
        use std::path::Path;
        
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_semantic_search.db");
        let _ = std::fs::remove_file(&db_path);
        
        let memory = ConversationMemory::new(Path::new(&db_path))
            .expect("Failed to create memory");
        
        // Create test conversation with messages
        let conversation = memory.create_conversation(None)
            .expect("Failed to create conversation");
        
        memory.save_message(
            &conversation.id,
            MessageRole::User,
            "How do I create a Python function?".to_string(),
            10,
            None,
            None,
        ).expect("Failed to save message");
        
        memory.save_message(
            &conversation.id,
            MessageRole::Assistant,
            "Here's how to create a Python function: def example(): pass".to_string(),
            20,
            None,
            None,
        ).expect("Failed to save message");
        
        // Build search index
        let search = ConversationSemanticSearch::new()
            .expect("Failed to create search engine");
        let count = search.build_index(&memory)
            .expect("Failed to build index");
        
        assert_eq!(count, 2);
        
        // Search for similar messages
        let results = search.search("Python function creation", 5)
            .expect("Search failed");
        
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.5); // Should have high similarity
    }
}
