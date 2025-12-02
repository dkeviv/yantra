// File: src-tauri/src/llm/config.rs
// Purpose: LLM configuration management and persistence
// Last Updated: November 20, 2025

use super::{LLMConfig, LLMProvider};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const CONFIG_FILE: &str = "llm_config.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfigManager {
    config: LLMConfig,
    config_path: PathBuf,
}

impl LLMConfigManager {
    /// Create new config manager with default config directory
    pub fn new(config_dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;

        let config_path = config_dir.join(CONFIG_FILE);
        
        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            LLMConfig::default()
        };

        Ok(Self {
            config,
            config_path,
        })
    }

    /// Get current configuration
    pub fn get_config(&self) -> &LLMConfig {
        &self.config
    }

    /// Update primary provider
    pub fn set_primary_provider(&mut self, provider: LLMProvider) -> Result<(), String> {
        self.config.primary_provider = provider;
        self.save()
    }

    /// Set Claude API key
    pub fn set_claude_key(&mut self, api_key: String) -> Result<(), String> {
        self.config.claude_api_key = Some(api_key);
        self.save()
    }

    /// Set OpenAI API key
    pub fn set_openai_key(&mut self, api_key: String) -> Result<(), String> {
        self.config.openai_api_key = Some(api_key);
        self.save()
    }

    /// Set OpenRouter API key
    pub fn set_openrouter_key(&mut self, api_key: String) -> Result<(), String> {
        self.config.openrouter_api_key = Some(api_key);
        self.save()
    }

    /// Set Groq API key
    pub fn set_groq_key(&mut self, api_key: String) -> Result<(), String> {
        self.config.groq_api_key = Some(api_key);
        self.save()
    }

    /// Set Gemini API key
    pub fn set_gemini_key(&mut self, api_key: String) -> Result<(), String> {
        self.config.gemini_api_key = Some(api_key);
        self.save()
    }

    /// Update retry settings
    pub fn set_retry_config(&mut self, max_retries: u32, timeout_seconds: u64) -> Result<(), String> {
        self.config.max_retries = max_retries;
        self.config.timeout_seconds = timeout_seconds;
        self.save()
    }

    /// Clear API key for a provider (for security)
    pub fn clear_api_key(&mut self, provider: LLMProvider) -> Result<(), String> {
        match provider {
            LLMProvider::Claude => self.config.claude_api_key = None,
            LLMProvider::OpenAI => self.config.openai_api_key = None,
            LLMProvider::OpenRouter => self.config.openrouter_api_key = None,
            LLMProvider::Groq => self.config.groq_api_key = None,
            LLMProvider::Gemini => self.config.gemini_api_key = None,
            LLMProvider::Qwen => self.config.openai_api_key = None, // Qwen uses OpenAI-compatible API
        }
        self.save()
    }

    /// Save configuration to disk
    fn save(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(&self.config_path, json)
            .map_err(|e| format!("Failed to write config file: {}", e))
    }

    /// Load configuration from file
    fn load_from_file(path: &Path) -> Result<LLMConfig, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config file: {}", e))
    }

    /// Get sanitized config (without API keys) for display
    pub fn get_sanitized_config(&self) -> SanitizedConfig {
        SanitizedConfig {
            has_claude_key: self.config.claude_api_key.is_some(),
            has_openai_key: self.config.openai_api_key.is_some(),
            has_openrouter_key: self.config.openrouter_api_key.is_some(),
            has_groq_key: self.config.groq_api_key.is_some(),
            has_gemini_key: self.config.gemini_api_key.is_some(),
            primary_provider: self.config.primary_provider,
            max_retries: self.config.max_retries,
            timeout_seconds: self.config.timeout_seconds,
            selected_models: self.config.selected_models.clone(),
        }
    }

    /// Set selected models for the user
    pub fn set_selected_models(&mut self, model_ids: Vec<String>) -> Result<(), String> {
        self.config.selected_models = model_ids;
        self.save()
    }

    /// Add a model to selected models
    pub fn add_selected_model(&mut self, model_id: String) -> Result<(), String> {
        if !self.config.selected_models.contains(&model_id) {
            self.config.selected_models.push(model_id);
            self.save()?;
        }
        Ok(())
    }

    /// Remove a model from selected models
    pub fn remove_selected_model(&mut self, model_id: &str) -> Result<(), String> {
        self.config.selected_models.retain(|m| m != model_id);
        self.save()
    }

    /// Get selected models
    pub fn get_selected_models(&self) -> Vec<String> {
        self.config.selected_models.clone()
    }
}

/// Sanitized configuration for frontend (no API keys)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedConfig {
    pub has_claude_key: bool,
    pub has_openai_key: bool,
    pub has_openrouter_key: bool,
    pub has_groq_key: bool,
    pub has_gemini_key: bool,
    pub primary_provider: LLMProvider,
    pub max_retries: u32,
    pub timeout_seconds: u64,
    pub selected_models: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_manager_creation() {
        let dir = tempdir().unwrap();
        let manager = LLMConfigManager::new(dir.path()).unwrap();
        
        let config = manager.get_config();
        assert_eq!(config.primary_provider, LLMProvider::Claude);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_set_primary_provider() {
        let dir = tempdir().unwrap();
        let mut manager = LLMConfigManager::new(dir.path()).unwrap();
        
        manager.set_primary_provider(LLMProvider::OpenAI).unwrap();
        assert_eq!(manager.get_config().primary_provider, LLMProvider::OpenAI);
        
        // Should persist
        let manager2 = LLMConfigManager::new(dir.path()).unwrap();
        assert_eq!(manager2.get_config().primary_provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_api_key_management() {
        let dir = tempdir().unwrap();
        let mut manager = LLMConfigManager::new(dir.path()).unwrap();
        
        manager.set_claude_key("test_claude_key".to_string()).unwrap();
        assert!(manager.get_config().claude_api_key.is_some());
        
        manager.clear_api_key(LLMProvider::Claude).unwrap();
        assert!(manager.get_config().claude_api_key.is_none());
    }

    #[test]
    fn test_sanitized_config() {
        let dir = tempdir().unwrap();
        let mut manager = LLMConfigManager::new(dir.path()).unwrap();
        
        manager.set_claude_key("secret_key".to_string()).unwrap();
        
        let sanitized = manager.get_sanitized_config();
        assert!(sanitized.has_claude_key);
        assert!(!sanitized.has_openai_key);
        assert_eq!(sanitized.primary_provider, LLMProvider::Claude);
    }
}
