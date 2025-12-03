// File: src-tauri/src/llm/chroma_client.rs
// Purpose: ChromaDB client for vector database operations
// Dependencies: reqwest, serde_json
// Last Updated: December 3, 2025
//
// Implements ChromaDB HTTP API client for:
// - Collection management (create, get, delete)
// - Document insertion with embeddings
// - Similarity search (query)
// - Metadata filtering
// - Batch operations
//
// ChromaDB Setup:
// 1. Install: pip install chromadb
// 2. Run server: chroma run --host localhost --port 8000
// 3. API available at http://localhost:8000
//
// Usage:
// 1. Create client → create_collection()
// 2. Add documents with embeddings
// 3. Query for similar documents
// 4. Filter by metadata (language, success rate, etc.)

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// ChromaDB client for vector operations
pub struct ChromaClient {
    base_url: String,
    client: Client,
    timeout_seconds: u64,
}

impl ChromaClient {
    /// Create new ChromaDB client
    /// 
    /// # Arguments
    /// * `base_url` - ChromaDB server URL (default: http://localhost:8000)
    pub fn new(base_url: Option<String>) -> Self {
        ChromaClient {
            base_url: base_url.unwrap_or_else(|| "http://localhost:8000".to_string()),
            client: Client::new(),
            timeout_seconds: 30,
        }
    }

    /// Create collection
    /// 
    /// # Arguments
    /// * `name` - Collection name
    /// * `metadata` - Optional metadata
    pub async fn create_collection(
        &self,
        name: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Collection, String> {
        let url = format!("{}/api/v1/collections", self.base_url);
        
        let body = json!({
            "name": name,
            "metadata": metadata.unwrap_or_default(),
        });

        let response = self.client
            .post(&url)
            .json(&body)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("ChromaDB error ({}): {}", status, error_text));
        }

        let collection: Collection = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse collection response: {}", e))?;

        Ok(collection)
    }

    /// Get collection by name
    pub async fn get_collection(&self, name: &str) -> Result<Collection, String> {
        let url = format!("{}/api/v1/collections/{}", self.base_url, name);
        
        let response = self.client
            .get(&url)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| format!("Failed to get collection: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Collection '{}' not found", name));
        }

        let collection: Collection = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse collection: {}", e))?;

        Ok(collection)
    }

    /// Delete collection
    pub async fn delete_collection(&self, name: &str) -> Result<(), String> {
        let url = format!("{}/api/v1/collections/{}", self.base_url, name);
        
        let response = self.client
            .delete(&url)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| format!("Failed to delete collection: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Failed to delete collection ({}): {}", status, error_text));
        }

        Ok(())
    }

    /// Add documents to collection
    /// 
    /// # Arguments
    /// * `collection_id` - Collection ID
    /// * `documents` - Documents to add with IDs, embeddings, metadata
    pub async fn add_documents(
        &self,
        collection_id: &str,
        documents: Vec<Document>,
    ) -> Result<(), String> {
        let url = format!("{}/api/v1/collections/{}/add", self.base_url, collection_id);
        
        let ids: Vec<String> = documents.iter().map(|d| d.id.clone()).collect();
        let embeddings: Vec<Vec<f32>> = documents.iter().map(|d| d.embedding.clone()).collect();
        let metadatas: Vec<HashMap<String, String>> = documents.iter()
            .map(|d| d.metadata.clone())
            .collect();
        let documents_text: Vec<String> = documents.iter()
            .map(|d| d.document.clone())
            .collect();

        let body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "documents": documents_text,
        });

        let response = self.client
            .post(&url)
            .json(&body)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| format!("Failed to add documents: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Failed to add documents ({}): {}", status, error_text));
        }

        Ok(())
    }

    /// Query collection for similar documents
    /// 
    /// # Arguments
    /// * `collection_id` - Collection ID
    /// * `query_embedding` - Query embedding vector
    /// * `n_results` - Number of results to return
    /// * `where_filter` - Optional metadata filter
    pub async fn query(
        &self,
        collection_id: &str,
        query_embedding: Vec<f32>,
        n_results: usize,
        where_filter: Option<HashMap<String, String>>,
    ) -> Result<QueryResults, String> {
        let url = format!("{}/api/v1/collections/{}/query", self.base_url, collection_id);
        
        let mut body = json!({
            "query_embeddings": vec![query_embedding],
            "n_results": n_results,
        });

        if let Some(filter) = where_filter {
            body["where"] = json!(filter);
        }

        let response = self.client
            .post(&url)
            .json(&body)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| format!("Failed to query collection: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Query failed ({}): {}", status, error_text));
        }

        let results: QueryResults = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse query results: {}", e))?;

        Ok(results)
    }

    /// Count documents in collection
    pub async fn count(&self, collection_id: &str) -> Result<usize, String> {
        let url = format!("{}/api/v1/collections/{}/count", self.base_url, collection_id);
        
        let response = self.client
            .get(&url)
            .timeout(tokio::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await
            .map_err(|e| format!("Failed to count documents: {}", e))?;

        if !response.status().is_success() {
            return Err("Failed to count documents".to_string());
        }

        let count: usize = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse count: {}", e))?;

        Ok(count)
    }

    /// Check if ChromaDB server is running
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/api/v1/heartbeat", self.base_url);
        
        if let Ok(response) = self.client
            .get(&url)
            .timeout(tokio::time::Duration::from_secs(5))
            .send()
            .await
        {
            response.status().is_success()
        } else {
            false
        }
    }
}

/// ChromaDB collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub metadata: HashMap<String, String>,
}

/// Document to add to collection
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub document: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// Query results from ChromaDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResults {
    pub ids: Vec<Vec<String>>,
    pub distances: Vec<Vec<f32>>,
    pub documents: Vec<Vec<String>>,
    pub metadatas: Vec<Vec<HashMap<String, serde_json::Value>>>,
}

impl QueryResults {
    /// Get best result (first result from first query)
    pub fn get_best(&self) -> Option<QueryResult> {
        if self.ids.is_empty() || self.ids[0].is_empty() {
            return None;
        }

        Some(QueryResult {
            id: self.ids[0][0].clone(),
            distance: self.distances[0][0],
            document: self.documents[0][0].clone(),
            metadata: self.metadatas[0][0].clone(),
        })
    }

    /// Get all results as flat list
    pub fn get_all(&self) -> Vec<QueryResult> {
        let mut results = Vec::new();
        
        if self.ids.is_empty() {
            return results;
        }

        for i in 0..self.ids[0].len() {
            results.push(QueryResult {
                id: self.ids[0][i].clone(),
                distance: self.distances[0][i],
                document: self.documents[0][i].clone(),
                metadata: self.metadatas[0][i].clone(),
            });
        }

        results
    }
}

/// Single query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub id: String,
    pub distance: f32,
    pub document: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = ChromaClient::new(None);
        assert_eq!(client.base_url, "http://localhost:8000");
        assert_eq!(client.timeout_seconds, 30);
    }

    #[test]
    fn test_client_custom_url() {
        let client = ChromaClient::new(Some("http://example.com:9000".to_string()));
        assert_eq!(client.base_url, "http://example.com:9000");
    }

    #[tokio::test]
    async fn test_health_check() {
        let client = ChromaClient::new(None);
        // This will fail if ChromaDB is not running, but that's expected in tests
        let _ = client.health_check().await;
    }
}
