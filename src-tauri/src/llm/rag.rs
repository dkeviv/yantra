// File: src-tauri/src/llm/rag.rs
// Purpose: RAG (Retrieval-Augmented Generation) using ChromaDB
// Dependencies: chroma_client, embeddings
// Last Updated: December 3, 2025
//
// Implements learning from successful code generations:
// - Store successful code patterns with embeddings
// - Retrieve similar patterns for new tasks
// - Filter by language, context, success rate
// - Template library management
// - Pattern matching for common tasks
//
// Collections:
// - code_patterns: Successful code generations
// - test_patterns: Working test templates
// - fix_patterns: Successful bug fixes
//
// Usage:
// 1. After successful generation → store_pattern()
// 2. Before new generation → retrieve_patterns()
// 3. Use retrieved patterns as examples in prompt
// 4. Gradually build library of proven solutions

use super::chroma_client::{ChromaClient, Collection, Document, QueryResult};
use crate::gnn::embeddings::EmbeddingGenerator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::cell::RefCell;

/// RAG system for code pattern learning
pub struct RAGSystem {
    chroma: ChromaClient,
    embedder: RefCell<EmbeddingGenerator>,
    code_collection: Option<String>,
    test_collection: Option<String>,
    fix_collection: Option<String>,
}

impl RAGSystem {
    /// Create new RAG system
    pub fn new(chroma_url: Option<String>) -> Self {
        RAGSystem {
            chroma: ChromaClient::new(chroma_url),
            embedder: RefCell::new(EmbeddingGenerator::default()),
            code_collection: None,
            test_collection: None,
            fix_collection: None,
        }
    }

    /// Initialize RAG system (create collections)
    pub async fn initialize(&mut self) -> Result<(), String> {
        // Check if ChromaDB is running
        if !self.chroma.health_check().await {
            return Err("ChromaDB server is not running. Please start it with: chroma run --host localhost --port 8000".to_string());
        }

        // Create or get code patterns collection
        let code_collection = match self.chroma.get_collection("yantra_code_patterns").await {
            Ok(collection) => collection,
            Err(_) => {
                let mut metadata = HashMap::new();
                metadata.insert("description".to_string(), "Successful code generation patterns".to_string());
                self.chroma.create_collection("yantra_code_patterns", Some(metadata)).await?
            }
        };
        self.code_collection = Some(code_collection.id.clone());

        // Create or get test patterns collection
        let test_collection = match self.chroma.get_collection("yantra_test_patterns").await {
            Ok(collection) => collection,
            Err(_) => {
                let mut metadata = HashMap::new();
                metadata.insert("description".to_string(), "Working test templates".to_string());
                self.chroma.create_collection("yantra_test_patterns", Some(metadata)).await?
            }
        };
        self.test_collection = Some(test_collection.id.clone());

        // Create or get fix patterns collection
        let fix_collection = match self.chroma.get_collection("yantra_fix_patterns").await {
            Ok(collection) => collection,
            Err(_) => {
                let mut metadata = HashMap::new();
                metadata.insert("description".to_string(), "Successful bug fixes".to_string());
                self.chroma.create_collection("yantra_fix_patterns", Some(metadata)).await?
            }
        };
        self.fix_collection = Some(fix_collection.id.clone());

        println!("✅ RAG system initialized successfully");
        Ok(())
    }

    /// Store successful code pattern
    /// 
    /// # Arguments
    /// * `pattern` - Code pattern to store
    pub async fn store_code_pattern(&self, pattern: &CodePattern) -> Result<(), String> {
        let collection_id = self.code_collection.as_ref()
            .ok_or("RAG system not initialized")?;

        // Generate embedding for intent
        let embedding = self.embedder.borrow_mut().generate_text_embedding(&pattern.intent)?;

        // Prepare metadata
        let mut metadata = HashMap::new();
        metadata.insert("language".to_string(), pattern.language.clone());
        metadata.insert("intent".to_string(), pattern.intent.clone());
        metadata.insert("success_rate".to_string(), pattern.success_rate.to_string());
        metadata.insert("test_pass_rate".to_string(), pattern.test_pass_rate.to_string());
        metadata.insert("created_at".to_string(), pattern.created_at.clone());
        
        if let Some(framework) = &pattern.framework {
            metadata.insert("framework".to_string(), framework.clone());
        }

        // Create document
        let document = Document {
            id: pattern.id.clone(),
            document: format!(
                "Intent: {}\nLanguage: {}\nCode:\n{}",
                pattern.intent, pattern.language, pattern.code
            ),
            embedding,
            metadata,
        };

        self.chroma.add_documents(collection_id, vec![document]).await?;
        Ok(())
    }

    /// Store successful test pattern
    pub async fn store_test_pattern(&self, pattern: &TestPattern) -> Result<(), String> {
        let collection_id = self.test_collection.as_ref()
            .ok_or("RAG system not initialized")?;

        let embedding = self.embedder.borrow_mut().generate_text_embedding(&format!(
            "{} {} test",
            pattern.language, pattern.test_type
        ))?;

        let mut metadata = HashMap::new();
        metadata.insert("language".to_string(), pattern.language.clone());
        metadata.insert("test_type".to_string(), pattern.test_type.clone());
        metadata.insert("framework".to_string(), pattern.framework.clone());
        metadata.insert("success_rate".to_string(), pattern.success_rate.to_string());

        let document = Document {
            id: pattern.id.clone(),
            document: format!(
                "Language: {}\nFramework: {}\nTest:\n{}",
                pattern.language, pattern.framework, pattern.test_code
            ),
            embedding,
            metadata,
        };

        self.chroma.add_documents(collection_id, vec![document]).await?;
        Ok(())
    }

    /// Store successful fix pattern
    pub async fn store_fix_pattern(&self, pattern: &FixPattern) -> Result<(), String> {
        let collection_id = self.fix_collection.as_ref()
            .ok_or("RAG system not initialized")?;

        let embedding = self.embedder.borrow_mut().generate_text_embedding(&pattern.error_type)?;

        let mut metadata = HashMap::new();
        metadata.insert("error_type".to_string(), pattern.error_type.clone());
        metadata.insert("language".to_string(), pattern.language.clone());
        metadata.insert("success_rate".to_string(), pattern.success_rate.to_string());

        let document = Document {
            id: pattern.id.clone(),
            document: format!(
                "Error: {}\nFix:\n{}",
                pattern.error_message, pattern.fix_code
            ),
            embedding,
            metadata,
        };

        self.chroma.add_documents(collection_id, vec![document]).await?;
        Ok(())
    }

    /// Retrieve similar code patterns
    /// 
    /// # Arguments
    /// * `intent` - What the user wants to do
    /// * `language` - Programming language filter
    /// * `n_results` - Number of patterns to retrieve
    pub async fn retrieve_code_patterns(
        &self,
        intent: &str,
        language: Option<&str>,
        n_results: usize,
    ) -> Result<Vec<RetrievedPattern>, String> {
        let collection_id = self.code_collection.as_ref()
            .ok_or("RAG system not initialized")?;

        // Generate query embedding
        let query_embedding = self.embedder.borrow_mut().generate_text_embedding(intent)?;

        // Build filter
        let mut filter = HashMap::new();
        if let Some(lang) = language {
            filter.insert("language".to_string(), lang.to_string());
        }

        // Query ChromaDB
        let results = self.chroma.query(
            collection_id,
            query_embedding,
            n_results,
            if filter.is_empty() { None } else { Some(filter) },
        ).await?;

        // Convert to RetrievedPattern
        Ok(results.get_all().into_iter()
            .filter_map(|r| self.parse_code_pattern(&r).ok())
            .collect())
    }

    /// Retrieve similar test patterns
    pub async fn retrieve_test_patterns(
        &self,
        language: &str,
        framework: &str,
        n_results: usize,
    ) -> Result<Vec<RetrievedTestPattern>, String> {
        let collection_id = self.test_collection.as_ref()
            .ok_or("RAG system not initialized")?;

        let query_text = format!("{} {} test", language, framework);
        let query_embedding = self.embedder.borrow_mut().generate_text_embedding(&query_text)?;

        let mut filter = HashMap::new();
        filter.insert("language".to_string(), language.to_string());
        filter.insert("framework".to_string(), framework.to_string());

        let results = self.chroma.query(
            collection_id,
            query_embedding,
            n_results,
            Some(filter),
        ).await?;

        Ok(results.get_all().into_iter()
            .filter_map(|r| self.parse_test_pattern(&r).ok())
            .collect())
    }

    /// Retrieve similar fix patterns
    pub async fn retrieve_fix_patterns(
        &self,
        error_type: &str,
        language: &str,
        n_results: usize,
    ) -> Result<Vec<RetrievedFixPattern>, String> {
        let collection_id = self.fix_collection.as_ref()
            .ok_or("RAG system not initialized")?;

        let query_embedding = self.embedder.borrow_mut().generate_text_embedding(error_type)?;

        let mut filter = HashMap::new();
        filter.insert("language".to_string(), language.to_string());

        let results = self.chroma.query(
            collection_id,
            query_embedding,
            n_results,
            Some(filter),
        ).await?;

        Ok(results.get_all().into_iter()
            .filter_map(|r| self.parse_fix_pattern(&r).ok())
            .collect())
    }

    /// Get statistics
    pub async fn get_stats(&self) -> Result<RAGStats, String> {
        let code_count = if let Some(id) = &self.code_collection {
            self.chroma.count(id).await.unwrap_or(0)
        } else {
            0
        };

        let test_count = if let Some(id) = &self.test_collection {
            self.chroma.count(id).await.unwrap_or(0)
        } else {
            0
        };

        let fix_count = if let Some(id) = &self.fix_collection {
            self.chroma.count(id).await.unwrap_or(0)
        } else {
            0
        };

        Ok(RAGStats {
            code_patterns: code_count,
            test_patterns: test_count,
            fix_patterns: fix_count,
            total: code_count + test_count + fix_count,
        })
    }

    // Helper methods
    fn parse_code_pattern(&self, result: &QueryResult) -> Result<RetrievedPattern, String> {
        let metadata = &result.metadata;
        Ok(RetrievedPattern {
            id: result.id.clone(),
            code: result.document.clone(),
            similarity: 1.0 - result.distance, // Convert distance to similarity
            language: metadata.get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            intent: metadata.get("intent")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            success_rate: metadata.get("success_rate")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
        })
    }

    fn parse_test_pattern(&self, result: &QueryResult) -> Result<RetrievedTestPattern, String> {
        let metadata = &result.metadata;
        Ok(RetrievedTestPattern {
            id: result.id.clone(),
            test_code: result.document.clone(),
            similarity: 1.0 - result.distance,
            language: metadata.get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            framework: metadata.get("framework")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
        })
    }

    fn parse_fix_pattern(&self, result: &QueryResult) -> Result<RetrievedFixPattern, String> {
        let metadata = &result.metadata;
        Ok(RetrievedFixPattern {
            id: result.id.clone(),
            fix_code: result.document.clone(),
            similarity: 1.0 - result.distance,
            error_type: metadata.get("error_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
        })
    }
}

/// Code pattern to store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    pub id: String,
    pub intent: String,
    pub language: String,
    pub framework: Option<String>,
    pub code: String,
    pub success_rate: f64, // 0.0-1.0
    pub test_pass_rate: f64,
    pub created_at: String,
}

/// Test pattern to store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPattern {
    pub id: String,
    pub language: String,
    pub framework: String, // pytest, jest, etc.
    pub test_type: String, // unit, integration, etc.
    pub test_code: String,
    pub success_rate: f64,
}

/// Fix pattern to store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixPattern {
    pub id: String,
    pub error_type: String, // AttributeError, TypeError, etc.
    pub error_message: String,
    pub language: String,
    pub fix_code: String,
    pub success_rate: f64,
}

/// Retrieved code pattern
#[derive(Debug, Clone)]
pub struct RetrievedPattern {
    pub id: String,
    pub code: String,
    pub similarity: f32,
    pub language: String,
    pub intent: String,
    pub success_rate: f64,
}

/// Retrieved test pattern
#[derive(Debug, Clone)]
pub struct RetrievedTestPattern {
    pub id: String,
    pub test_code: String,
    pub similarity: f32,
    pub language: String,
    pub framework: String,
}

/// Retrieved fix pattern
#[derive(Debug, Clone)]
pub struct RetrievedFixPattern {
    pub id: String,
    pub fix_code: String,
    pub similarity: f32,
    pub error_type: String,
}

/// RAG statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGStats {
    pub code_patterns: usize,
    pub test_patterns: usize,
    pub fix_patterns: usize,
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_creation() {
        let rag = RAGSystem::new(None);
        assert!(rag.code_collection.is_none());
    }

    #[test]
    fn test_code_pattern() {
        let pattern = CodePattern {
            id: "test-1".to_string(),
            intent: "Calculate factorial".to_string(),
            language: "python".to_string(),
            framework: None,
            code: "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)".to_string(),
            success_rate: 0.95,
            test_pass_rate: 1.0,
            created_at: "2025-12-03".to_string(),
        };

        assert_eq!(pattern.language, "python");
        assert_eq!(pattern.success_rate, 0.95);
    }
}
