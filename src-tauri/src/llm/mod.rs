// File: src-tauri/src/llm/mod.rs
// Purpose: Multi-LLM orchestration module for code generation
// Dependencies: reqwest, tokio, serde
// Last Updated: November 21, 2025

pub mod claude;
pub mod openai;
pub mod orchestrator;
pub mod prompts;
pub mod context;
pub mod config;
pub mod tokens;

use serde::{Deserialize, Serialize};

/// Configuration for LLM services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub claude_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub primary_provider: LLMProvider,
    pub max_retries: u32,
    pub timeout_seconds: u64,
}

/// Available LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLMProvider {
    Claude,
    OpenAI,
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
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
        }
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
