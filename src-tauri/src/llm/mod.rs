// File: src-tauri/src/llm/mod.rs
// Purpose: Multi-LLM orchestration module for code generation
// Dependencies: reqwest, tokio, serde
// Last Updated: December 3, 2025

pub mod claude;
pub mod openai;
pub mod openrouter;
pub mod groq;
pub mod gemini;
pub mod orchestrator;
pub mod prompts;
pub mod context;
pub mod context_depth;
pub mod chroma_client;
pub mod rag;
pub mod config;
pub mod tokens;
pub mod models;

// Re-export 4-level context types
pub use context_depth::{
    assemble_4_level_context, FourLevelContext, ContextItem as DepthContextItem,
    ContextConfig, ContextStats,
};

// Re-export RAG types
pub use rag::{
    RAGSystem, CodePattern, TestPattern, FixPattern,
    RetrievedPattern, RetrievedTestPattern, RetrievedFixPattern, RAGStats,
};
pub use chroma_client::{ChromaClient, Collection, Document, QueryResult};

use serde::{Deserialize, Serialize};

/// Configuration for LLM services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub claude_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub groq_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub primary_provider: LLMProvider,
    /// Secondary provider for automatic failover (optional)
    #[serde(default)]
    pub secondary_provider: Option<LLMProvider>,
    pub max_retries: u32,
    pub timeout_seconds: u64,
    /// User-selected models for each provider (model IDs)
    #[serde(default)]
    pub selected_models: Vec<String>,
}

/// Available LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLMProvider {
    Claude,
    OpenAI,
    OpenRouter,
    Groq,
    Gemini,
    Qwen,
}

/// Request for code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenerationRequest {
    pub intent: String,
    pub file_path: Option<String>,
    pub context: Vec<String>,
    pub dependencies: Vec<String>,
}

/// Response from code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenerationResponse {
    pub code: String,
    pub language: String,
    pub explanation: String,
    pub tests: Option<String>,
    pub provider: LLMProvider,
    pub tokens_used: u32,
}

/// Chat message for conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,  // "user" or "assistant"
    pub content: String,
}

/// Intent detected from user message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intent {
    #[serde(rename = "code_generation")]
    CodeGeneration,
    #[serde(rename = "code_modification")]
    CodeModification,
    #[serde(rename = "terminal_command")]
    TerminalCommand,
    #[serde(rename = "ui_control")]
    UIControl,
    #[serde(rename = "question")]
    Question,
    #[serde(rename = "general")]
    General,
}

/// Action to be taken based on intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAction {
    pub action_type: String,  // "generate_code", "run_command", "show_panel", etc.
    pub parameters: std::collections::HashMap<String, String>,
}

/// Response from chat API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub response: String,
    pub intent: Intent,
    pub action: Option<DetectedAction>,
}

/// Common error type for LLM operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMError {
    pub message: String,
    pub provider: Option<LLMProvider>,
    pub is_retryable: bool,
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LLM Error: {}", self.message)
    }
}

impl std::error::Error for LLMError {}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            claude_api_key: None,
            openai_api_key: None,
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            secondary_provider: None,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: Vec::new(),
        }
    }
}

impl LLMConfig {
    /// Check if at least one API key is configured
    pub fn has_api_key(&self) -> bool {
        self.claude_api_key.is_some() 
            || self.openai_api_key.is_some()
            || self.openrouter_api_key.is_some()
            || self.groq_api_key.is_some()
            || self.gemini_api_key.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LLMConfig::default();
        assert_eq!(config.primary_provider, LLMProvider::Claude);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_llm_config_has_api_key() {
        let mut config = LLMConfig::default();
        assert!(!config.has_api_key());
        
        config.claude_api_key = Some("test_key".to_string());
        assert!(config.has_api_key());
        
        config.claude_api_key = None;
        config.openai_api_key = Some("test_key".to_string());
        assert!(config.has_api_key());
    }

    #[test]
    fn test_code_generation_request_serialization() {
        let request = CodeGenerationRequest {
            intent: "Create a function to calculate fibonacci".to_string(),
            file_path: Some("math_utils.py".to_string()),
            context: vec!["import math".to_string()],
            dependencies: vec!["utils.py".to_string()],
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: CodeGenerationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.intent, request.intent);
    }
}