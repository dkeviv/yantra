// File: src-tauri/src/llm/models.rs
// Purpose: Model definitions and availability for each LLM provider
// Last Updated: November 29, 2025

use serde::{Deserialize, Serialize};
use super::LLMProvider;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub supports_code: bool,
}

/// Get available models for a specific provider
pub fn get_available_models(provider: LLMProvider) -> Vec<ModelInfo> {
    match provider {
        LLMProvider::Claude => claude_models(),
        LLMProvider::OpenAI => openai_models(),
        LLMProvider::OpenRouter => openrouter_models(),
        LLMProvider::Groq => groq_models(),
        LLMProvider::Gemini => gemini_models(),
        LLMProvider::Qwen => qwen_models(),
    }
}

/// Claude (Anthropic) available models
fn claude_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "claude-3-5-sonnet-20241022".to_string(),
            name: "Claude 3.5 Sonnet (New)".to_string(),
            description: "Latest Claude 3.5 Sonnet with improved coding capabilities".to_string(),
            context_window: 200_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "claude-3-5-sonnet-20240620".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            description: "Fast and intelligent, best for complex reasoning".to_string(),
            context_window: 200_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "claude-3-opus-20240229".to_string(),
            name: "Claude 3 Opus".to_string(),
            description: "Most capable model, best for complex tasks".to_string(),
            context_window: 200_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "claude-3-sonnet-20240229".to_string(),
            name: "Claude 3 Sonnet".to_string(),
            description: "Balanced performance and speed".to_string(),
            context_window: 200_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "claude-3-haiku-20240307".to_string(),
            name: "Claude 3 Haiku".to_string(),
            description: "Fastest model, good for simple tasks".to_string(),
            context_window: 200_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
    ]
}

/// OpenAI available models
fn openai_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "gpt-4-turbo".to_string(),
            name: "GPT-4 Turbo".to_string(),
            description: "Latest GPT-4 with 128K context window".to_string(),
            context_window: 128_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "gpt-4-turbo-preview".to_string(),
            name: "GPT-4 Turbo Preview".to_string(),
            description: "Preview version of GPT-4 Turbo".to_string(),
            context_window: 128_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "gpt-4".to_string(),
            name: "GPT-4".to_string(),
            description: "Most capable GPT-4 model".to_string(),
            context_window: 8_192,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "gpt-4-32k".to_string(),
            name: "GPT-4 32K".to_string(),
            description: "GPT-4 with extended context window".to_string(),
            context_window: 32_768,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "gpt-3.5-turbo".to_string(),
            name: "GPT-3.5 Turbo".to_string(),
            description: "Fast and affordable, good for simple tasks".to_string(),
            context_window: 16_385,
            max_output_tokens: 4_096,
            supports_code: true,
        },
    ]
}

/// OpenRouter available models (comprehensive list with latest versions)
fn openrouter_models() -> Vec<ModelInfo> {
    vec![
        // ===== Claude Models (Anthropic) - Latest versions =====
        ModelInfo {
            id: "anthropic/claude-3.5-sonnet:beta".to_string(),
            name: "Claude 3.5 Sonnet (Latest)".to_string(),
            description: "Latest Claude 3.5 with improved coding and reasoning".to_string(),
            context_window: 200_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "anthropic/claude-3.5-sonnet".to_string(),
            name: "Claude 3.5 Sonnet".to_string(),
            description: "Best balance of intelligence and speed".to_string(),
            context_window: 200_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "anthropic/claude-3-opus".to_string(),
            name: "Claude 3 Opus".to_string(),
            description: "Most capable Claude model for complex tasks".to_string(),
            context_window: 200_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "anthropic/claude-3-sonnet".to_string(),
            name: "Claude 3 Sonnet".to_string(),
            description: "Balanced performance and cost".to_string(),
            context_window: 200_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "anthropic/claude-3-haiku".to_string(),
            name: "Claude 3 Haiku".to_string(),
            description: "Fastest Claude model".to_string(),
            context_window: 200_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        
        // ===== ChatGPT/OpenAI Models - Latest versions =====
        ModelInfo {
            id: "openai/chatgpt-4o-latest".to_string(),
            name: "ChatGPT-4o (Latest)".to_string(),
            description: "Latest GPT-4o with enhanced capabilities".to_string(),
            context_window: 128_000,
            max_output_tokens: 16_384,
            supports_code: true,
        },
        ModelInfo {
            id: "openai/gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            description: "Multimodal flagship model, fast and smart".to_string(),
            context_window: 128_000,
            max_output_tokens: 16_384,
            supports_code: true,
        },
        ModelInfo {
            id: "openai/gpt-4o-mini".to_string(),
            name: "GPT-4o Mini".to_string(),
            description: "Affordable and fast, great for most tasks".to_string(),
            context_window: 128_000,
            max_output_tokens: 16_384,
            supports_code: true,
        },
        ModelInfo {
            id: "openai/gpt-4-turbo".to_string(),
            name: "GPT-4 Turbo".to_string(),
            description: "Previous generation flagship".to_string(),
            context_window: 128_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "openai/gpt-4".to_string(),
            name: "GPT-4".to_string(),
            description: "Original GPT-4".to_string(),
            context_window: 8_192,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "openai/o1-preview".to_string(),
            name: "OpenAI o1 Preview".to_string(),
            description: "Advanced reasoning model for complex problems".to_string(),
            context_window: 128_000,
            max_output_tokens: 32_768,
            supports_code: true,
        },
        ModelInfo {
            id: "openai/o1-mini".to_string(),
            name: "OpenAI o1 Mini".to_string(),
            description: "Faster reasoning model".to_string(),
            context_window: 128_000,
            max_output_tokens: 65_536,
            supports_code: true,
        },
        
        // ===== Google Gemini Models - Latest versions =====
        ModelInfo {
            id: "google/gemini-2.0-flash-exp:free".to_string(),
            name: "Gemini 2.0 Flash (Free)".to_string(),
            description: "Latest Gemini 2.0 - experimental, free tier".to_string(),
            context_window: 1_000_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "google/gemini-pro-1.5".to_string(),
            name: "Gemini 1.5 Pro".to_string(),
            description: "Google's most capable model with huge context".to_string(),
            context_window: 2_000_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "google/gemini-flash-1.5".to_string(),
            name: "Gemini 1.5 Flash".to_string(),
            description: "Fast and efficient".to_string(),
            context_window: 1_000_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        
        // ===== Meta LLaMA Models - Latest versions =====
        ModelInfo {
            id: "meta-llama/llama-3.3-70b-instruct".to_string(),
            name: "LLaMA 3.3 70B Instruct".to_string(),
            description: "Latest LLaMA 3.3 - excellent for coding".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "meta-llama/llama-3.2-90b-vision-instruct".to_string(),
            name: "LLaMA 3.2 90B Vision".to_string(),
            description: "Multimodal LLaMA with vision capabilities".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "meta-llama/llama-3.1-405b-instruct".to_string(),
            name: "LLaMA 3.1 405B Instruct".to_string(),
            description: "Largest open-source model, GPT-4 class".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "meta-llama/llama-3.1-70b-instruct".to_string(),
            name: "LLaMA 3.1 70B Instruct".to_string(),
            description: "Balanced open-source model".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "meta-llama/llama-3.1-8b-instruct".to_string(),
            name: "LLaMA 3.1 8B Instruct".to_string(),
            description: "Fast and efficient open model".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        
        // ===== DeepSeek Models - Latest versions =====
        ModelInfo {
            id: "deepseek/deepseek-chat".to_string(),
            name: "DeepSeek Chat V3".to_string(),
            description: "Latest DeepSeek - excellent for coding, very affordable".to_string(),
            context_window: 64_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "deepseek/deepseek-coder".to_string(),
            name: "DeepSeek Coder".to_string(),
            description: "Specialized coding model, highly capable".to_string(),
            context_window: 64_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        
        // ===== Mistral Models - Latest versions =====
        ModelInfo {
            id: "mistralai/mistral-large".to_string(),
            name: "Mistral Large".to_string(),
            description: "Mistral's flagship model".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "mistralai/mistral-medium".to_string(),
            name: "Mistral Medium".to_string(),
            description: "Balanced Mistral model".to_string(),
            context_window: 32_768,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "mistralai/mixtral-8x22b-instruct".to_string(),
            name: "Mixtral 8x22B Instruct".to_string(),
            description: "Powerful mixture-of-experts model".to_string(),
            context_window: 64_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "mistralai/mixtral-8x7b-instruct".to_string(),
            name: "Mixtral 8x7B Instruct".to_string(),
            description: "Fast and efficient MoE model".to_string(),
            context_window: 32_768,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "mistralai/codestral-latest".to_string(),
            name: "Codestral".to_string(),
            description: "Mistral's specialized coding model".to_string(),
            context_window: 32_768,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        
        // ===== Qwen Models (Alibaba) - Latest versions =====
        ModelInfo {
            id: "qwen/qwen-2.5-72b-instruct".to_string(),
            name: "Qwen 2.5 72B Instruct".to_string(),
            description: "Latest Qwen, excellent multilingual model".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "qwen/qwen-2.5-coder-32b-instruct".to_string(),
            name: "Qwen 2.5 Coder 32B".to_string(),
            description: "Specialized coding model from Qwen".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        
        // ===== Other Notable Models =====
        ModelInfo {
            id: "x-ai/grok-beta".to_string(),
            name: "Grok Beta".to_string(),
            description: "xAI's Grok model with real-time knowledge".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "cohere/command-r-plus".to_string(),
            name: "Command R+".to_string(),
            description: "Cohere's flagship model, great for RAG".to_string(),
            context_window: 128_000,
            max_output_tokens: 4_096,
            supports_code: true,
        },
        ModelInfo {
            id: "perplexity/llama-3.1-sonar-large-128k-online".to_string(),
            name: "Perplexity Sonar Large".to_string(),
            description: "LLaMA with online search capabilities".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
    ]
}

/// Groq available models
fn groq_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "llama-3.1-70b-versatile".to_string(),
            name: "Llama 3.1 70B Versatile".to_string(),
            description: "Fast inference, high quality, versatile".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "llama-3.1-8b-instant".to_string(),
            name: "Llama 3.1 8B Instant".to_string(),
            description: "Ultra-fast, good for simple tasks".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "mixtral-8x7b-32768".to_string(),
            name: "Mixtral 8x7B".to_string(),
            description: "Fast mixture-of-experts, good quality".to_string(),
            context_window: 32_768,
            max_output_tokens: 32_768,
            supports_code: true,
        },
        ModelInfo {
            id: "gemma-7b-it".to_string(),
            name: "Gemma 7B".to_string(),
            description: "Google's open model, fast and efficient".to_string(),
            context_window: 8_192,
            max_output_tokens: 8_192,
            supports_code: true,
        },
    ]
}

/// Google Gemini available models
fn gemini_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "gemini-1.5-pro".to_string(),
            name: "Gemini 1.5 Pro".to_string(),
            description: "Latest model with best reasoning and 2M context".to_string(),
            context_window: 2_000_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "gemini-1.5-flash".to_string(),
            name: "Gemini 1.5 Flash".to_string(),
            description: "Fast and efficient, good for most tasks".to_string(),
            context_window: 1_000_000,
            max_output_tokens: 8_192,
            supports_code: true,
        },
        ModelInfo {
            id: "gemini-1.0-pro".to_string(),
            name: "Gemini 1.0 Pro".to_string(),
            description: "Earlier version, still capable".to_string(),
            context_window: 32_768,
            max_output_tokens: 2_048,
            supports_code: true,
        },
    ]
}

/// Qwen (Alibaba) available models - via OpenAI-compatible API
fn qwen_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "qwen-turbo".to_string(),
            name: "Qwen Turbo".to_string(),
            description: "Fast and efficient Qwen model".to_string(),
            context_window: 8_192,
            max_output_tokens: 2_048,
            supports_code: true,
        },
        ModelInfo {
            id: "qwen-plus".to_string(),
            name: "Qwen Plus".to_string(),
            description: "Enhanced Qwen model with better capabilities".to_string(),
            context_window: 32_768,
            max_output_tokens: 4_096,
            supports_code: true,
        },
    ]
}

/// Get default model for a provider
pub fn get_default_model(provider: LLMProvider) -> String {
    match provider {
        LLMProvider::Claude => "claude-3-5-sonnet-20241022".to_string(),
        LLMProvider::OpenAI => "gpt-4-turbo".to_string(),
        LLMProvider::OpenRouter => "anthropic/claude-3.5-sonnet".to_string(),
        LLMProvider::Groq => "llama-3.1-70b-versatile".to_string(),
        LLMProvider::Gemini => "gemini-1.5-pro".to_string(),
        LLMProvider::Qwen => "qwen-plus".to_string(),
    }
}
