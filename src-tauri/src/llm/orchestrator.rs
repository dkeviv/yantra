// File: src-tauri/src/llm/orchestrator.rs
// Purpose: Multi-LLM orchestrator with failover and circuit breaker
// Dependencies: claude, openai, openrouter modules
// Last Updated: November 29, 2025

use super::claude::ClaudeClient;
use super::openai::OpenAIClient;
use super::openrouter::OpenRouterClient;
use super::groq::GroqClient;
use super::gemini::GeminiClient;
use super::{CodeGenerationRequest, CodeGenerationResponse, LLMConfig, LLMError, LLMProvider};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,  // Normal operation
    Open,    // Failing, don't call
    HalfOpen, // Testing if recovered
}

/// Circuit breaker for a single provider
#[derive(Debug, Clone)]
struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    failure_threshold: u32,
    timeout_duration: Duration,
}

impl CircuitBreaker {
    fn new(failure_threshold: u32, timeout_seconds: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure_time: None,
            failure_threshold,
            timeout_duration: Duration::from_secs(timeout_seconds),
        }
    }

    fn record_success(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.last_failure_time = None;
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }

    fn can_attempt(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.timeout_duration {
                        self.state = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
}

/// Multi-LLM orchestrator with failover
pub struct LLMOrchestrator {
    config: LLMConfig,
    claude_client: Option<ClaudeClient>,
    openai_client: Option<OpenAIClient>,
    openrouter_client: Option<OpenRouterClient>,
    groq_client: Option<GroqClient>,
    gemini_client: Option<GeminiClient>,
    claude_circuit: Arc<RwLock<CircuitBreaker>>,
    openai_circuit: Arc<RwLock<CircuitBreaker>>,
    openrouter_circuit: Arc<RwLock<CircuitBreaker>>,
    groq_circuit: Arc<RwLock<CircuitBreaker>>,
    gemini_circuit: Arc<RwLock<CircuitBreaker>>,
}

impl LLMOrchestrator {
    pub fn new(config: LLMConfig) -> Self {
        let timeout = Duration::from_secs(config.timeout_seconds);

        let claude_client = config.claude_api_key.as_ref().map(|key| {
            ClaudeClient::new(key.clone(), config.timeout_seconds)
        });

        let openai_client = config.openai_api_key.as_ref().map(|key| {
            OpenAIClient::new(key.clone(), config.timeout_seconds)
        });

        let openrouter_client = config.openrouter_api_key.as_ref().map(|key| {
            OpenRouterClient::new(key.clone(), config.timeout_seconds)
        });

        let groq_client = config.groq_api_key.as_ref().map(|key| {
            GroqClient::new(key.clone(), timeout)
        });

        let gemini_client = config.gemini_api_key.as_ref().map(|key| {
            GeminiClient::new(key.clone(), timeout)
        });

        Self {
            config: config.clone(),
            claude_client,
            openai_client,
            openrouter_client,
            groq_client,
            gemini_client,
            claude_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
            openai_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
            openrouter_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
            groq_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
            gemini_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
        }
    }

    /// Get a reference to the LLM configuration
    pub fn config(&self) -> &LLMConfig {
        &self.config
    }

    /// Generate code using primary provider with automatic failover
    pub async fn generate_code(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        // Try primary provider
        let primary_result = match self.config.primary_provider {
            LLMProvider::Claude => self.try_claude(request).await,
            LLMProvider::OpenAI => self.try_openai(request).await,
            LLMProvider::OpenRouter => self.try_openrouter(request).await,
            LLMProvider::Groq => self.try_groq(request).await,
            LLMProvider::Gemini => self.try_gemini(request).await,
            LLMProvider::Qwen => self.try_openai(request).await, // Qwen uses OpenAI-compatible API
        };

        if let Ok(response) = primary_result {
            return Ok(response);
        }

        // Failover to secondary provider
        let secondary_result = match self.config.primary_provider {
            LLMProvider::Claude => self.try_openai(request).await,
            LLMProvider::OpenAI => self.try_claude(request).await,
            LLMProvider::OpenRouter => self.try_claude(request).await,
            LLMProvider::Groq => self.try_claude(request).await,
            LLMProvider::Gemini => self.try_claude(request).await,
            LLMProvider::Qwen => self.try_claude(request).await,
        };

        if let Ok(response) = secondary_result {
            eprintln!(
                "Warning: Failed over from {:?} to {:?}",
                self.config.primary_provider,
                response.provider
            );
            return Ok(response);
        }

        // Both providers failed
        Err(LLMError {
            message: "All LLM providers failed".to_string(),
            provider: None,
            is_retryable: true,
        })
    }

    async fn try_claude(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let client = self.claude_client.as_ref().ok_or_else(|| LLMError {
            message: "Claude API key not configured".to_string(),
            provider: Some(LLMProvider::Claude),
            is_retryable: false,
        })?;

        // Check circuit breaker
        let mut circuit = self.claude_circuit.write().await;
        if !circuit.can_attempt() {
            return Err(LLMError {
                message: "Claude circuit breaker is open".to_string(),
                provider: Some(LLMProvider::Claude),
                is_retryable: true,
            });
        }
        drop(circuit); // Release lock

        // Try with retries
        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }

            match client.generate_code(request).await {
                Ok(response) => {
                    let mut circuit = self.claude_circuit.write().await;
                    circuit.record_success();
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if !e.is_retryable {
                        break;
                    }
                }
            }
        }

        // All retries failed
        let mut circuit = self.claude_circuit.write().await;
        circuit.record_failure();

        Err(last_error.unwrap_or_else(|| LLMError {
            message: "Claude request failed after retries".to_string(),
            provider: Some(LLMProvider::Claude),
            is_retryable: false,
        }))
    }

    async fn try_openai(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let client = self.openai_client.as_ref().ok_or_else(|| LLMError {
            message: "OpenAI API key not configured".to_string(),
            provider: Some(LLMProvider::OpenAI),
            is_retryable: false,
        })?;

        let mut circuit = self.openai_circuit.write().await;
        if !circuit.can_attempt() {
            return Err(LLMError {
                message: "OpenAI circuit breaker is open".to_string(),
                provider: Some(LLMProvider::OpenAI),
                is_retryable: true,
            });
        }
        drop(circuit);

        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }

            match client.generate_code(request).await {
                Ok(response) => {
                    let mut circuit = self.openai_circuit.write().await;
                    circuit.record_success();
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if !e.is_retryable {
                        break;
                    }
                }
            }
        }

        let mut circuit = self.openai_circuit.write().await;
        circuit.record_failure();

        Err(last_error.unwrap_or_else(|| LLMError {
            message: "OpenAI request failed after retries".to_string(),
            provider: Some(LLMProvider::OpenAI),
            is_retryable: false,
        }))
    }

    async fn try_openrouter(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let client = self.openrouter_client.as_ref().ok_or_else(|| LLMError {
            message: "OpenRouter API key not configured".to_string(),
            provider: Some(LLMProvider::OpenRouter),
            is_retryable: false,
        })?;

        // Check circuit breaker
        let mut circuit = self.openrouter_circuit.write().await;
        if !circuit.can_attempt() {
            return Err(LLMError {
                message: "OpenRouter circuit breaker is open".to_string(),
                provider: Some(LLMProvider::OpenRouter),
                is_retryable: true,
            });
        }
        drop(circuit);

        let mut last_error = None;

        for attempt in 0..self.config.max_retries {
            // Exponential backoff
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }

            match client.generate_code(request).await {
                Ok(response) => {
                    let mut circuit = self.openrouter_circuit.write().await;
                    circuit.record_success();
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if !e.is_retryable {
                        break;
                    }
                }
            }
        }

        let mut circuit = self.openrouter_circuit.write().await;
        circuit.record_failure();

        Err(last_error.unwrap_or_else(|| LLMError {
            message: "OpenRouter request failed after retries".to_string(),
            provider: Some(LLMProvider::OpenRouter),
            is_retryable: false,
        }))
    }

    /// Try Groq with retry and circuit breaker
    async fn try_groq(&self, request: &CodeGenerationRequest) -> Result<CodeGenerationResponse, LLMError> {
        let Some(ref client) = self.groq_client else {
            return Err(LLMError {
                message: "Groq client not configured (no API key)".to_string(),
                provider: Some(LLMProvider::Groq),
                is_retryable: false,
            });
        };

        let mut circuit = self.groq_circuit.write().await;
        if !circuit.can_attempt() {
            return Err(LLMError {
                message: "Groq circuit breaker is open".to_string(),
                provider: Some(LLMProvider::Groq),
                is_retryable: false,
            });
        }
        drop(circuit);

        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }

            match client.generate_code(request).await {
                Ok(response) => {
                    let mut circuit = self.groq_circuit.write().await;
                    circuit.record_success();
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if !e.is_retryable {
                        break;
                    }
                }
            }
        }

        let mut circuit = self.groq_circuit.write().await;
        circuit.record_failure();

        Err(last_error.unwrap_or_else(|| LLMError {
            message: "Groq request failed after retries".to_string(),
            provider: Some(LLMProvider::Groq),
            is_retryable: false,
        }))
    }

    /// Try Gemini with retry and circuit breaker
    async fn try_gemini(&self, request: &CodeGenerationRequest) -> Result<CodeGenerationResponse, LLMError> {
        let Some(ref client) = self.gemini_client else {
            return Err(LLMError {
                message: "Gemini client not configured (no API key)".to_string(),
                provider: Some(LLMProvider::Gemini),
                is_retryable: false,
            });
        };

        let mut circuit = self.gemini_circuit.write().await;
        if !circuit.can_attempt() {
            return Err(LLMError {
                message: "Gemini circuit breaker is open".to_string(),
                provider: Some(LLMProvider::Gemini),
                is_retryable: false,
            });
        }
        drop(circuit);

        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }

            match client.generate_code(request).await {
                Ok(response) => {
                    let mut circuit = self.gemini_circuit.write().await;
                    circuit.record_success();
                    return Ok(response);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if !e.is_retryable {
                        break;
                    }
                }
            }
        }

        let mut circuit = self.gemini_circuit.write().await;
        circuit.record_failure();

        Err(last_error.unwrap_or_else(|| LLMError {
            message: "Gemini request failed after retries".to_string(),
            provider: Some(LLMProvider::Gemini),
            is_retryable: false,
        }))
    }

    /// Natural language chat with intent detection
    pub async fn chat(
        &self,
        message: &str,
        _conversation_history: &[super::ChatMessage],
    ) -> Result<super::ChatResponse, LLMError> {
        use super::{Intent, DetectedAction, ChatResponse};
        use std::collections::HashMap;

        // Simple intent detection without calling LLM (for now)
        let lower_message = message.to_lowercase();

        // Detect intent from message patterns
        let intent = if lower_message.contains("create") || lower_message.contains("generate") 
            || lower_message.contains("write") || lower_message.contains("build")
            || lower_message.contains("make") {
            Intent::CodeGeneration
        } else if lower_message.starts_with("run ") || lower_message.starts_with("execute ")
            || lower_message.contains("npm ") || lower_message.contains("cargo ")
            || lower_message.contains("python ") {
            Intent::TerminalCommand
        } else if lower_message.contains("show ") || lower_message.contains("open ")
            || lower_message.contains("close ") || lower_message.contains("toggle ") {
            Intent::UIControl
        } else if lower_message.contains("fix ") || lower_message.contains("update ")
            || lower_message.contains("modify ") || lower_message.contains("refactor ") {
            Intent::CodeModification
        } else if lower_message.contains("?") || lower_message.starts_with("what ")
            || lower_message.starts_with("how ") || lower_message.starts_with("why ") {
            Intent::Question
        } else {
            Intent::General
        };

        // Generate appropriate response based on intent
        let response = match intent {
            Intent::CodeGeneration => {
                format!("I understand you want to {}. Let me generate that code for you...\n\n✨ This will create production-quality code with:\n• Auto-generated unit tests\n• Security scanning\n• Dependency analysis\n\n(Code generation coming soon - LLM integration in progress)", message)
            }
            Intent::TerminalCommand => {
                // Extract and execute command
                let command = if let Some(cmd) = lower_message.strip_prefix("run ") {
                    cmd.to_string()
                } else if let Some(cmd) = lower_message.strip_prefix("execute ") {
                    cmd.to_string()
                } else {
                    message.to_string()
                };
                format!("Executing command: `{}`", command)
            }
            Intent::UIControl => {
                "I'll adjust the UI for you...".to_string()
            }
            Intent::CodeModification => {
                "I'll help you modify the code...".to_string()
            }
            Intent::Question => {
                "That's a great question! Let me help you understand...".to_string()
            }
            Intent::General => {
                "I'm Yantra, your AI development assistant. I can help you:\n\n• **Generate code**: \"create a React component\"\n• **Run commands**: \"run npm test\"\n• **Control UI**: \"show dependencies\"\n• **Answer questions**: \"how does this work?\"\n\nWhat would you like to do?".to_string()
            }
        };

        // Extract action based on intent
        let action = match intent {
            Intent::TerminalCommand => {
                let command = if let Some(cmd) = lower_message.strip_prefix("run ") {
                    cmd.to_string()
                } else if let Some(cmd) = lower_message.strip_prefix("execute ") {
                    cmd.to_string()
                } else {
                    message.to_string()
                };

                let mut params = HashMap::new();
                params.insert("command".to_string(), command);
                Some(DetectedAction {
                    action_type: "run_command".to_string(),
                    parameters: params,
                })
            }
            Intent::UIControl => {
                let mut params = HashMap::new();
                if lower_message.contains("dependencies") {
                    params.insert("panel".to_string(), "dependencies".to_string());
                } else if lower_message.contains("terminal") {
                    params.insert("panel".to_string(), "terminal".to_string());
                } else if lower_message.contains("file") {
                    params.insert("panel".to_string(), "filetree".to_string());
                }

                Some(DetectedAction {
                    action_type: "toggle_panel".to_string(),
                    parameters: params,
                })
            }
            Intent::CodeGeneration => {
                let mut params = HashMap::new();
                params.insert("intent".to_string(), message.to_string());
                Some(DetectedAction {
                    action_type: "generate_code".to_string(),
                    parameters: params,
                })
            }
            _ => None,
        };

        Ok(ChatResponse {
            response,
            intent,
            action,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_state() {
        let mut cb = CircuitBreaker::new(3, 60);
        assert_eq!(cb.state, CircuitState::Closed);
        assert!(cb.can_attempt());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, 60);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state, CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state, CircuitState::Open);
        assert!(!cb.can_attempt());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let mut cb = CircuitBreaker::new(2, 0); // 0 second timeout for testing
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state, CircuitState::Open);
        
        // After timeout, should transition to half-open
        std::thread::sleep(Duration::from_millis(10));
        assert!(cb.can_attempt());
        assert_eq!(cb.state, CircuitState::HalfOpen);

        // Success should close it
        cb.record_success();
        assert_eq!(cb.state, CircuitState::Closed);
        assert_eq!(cb.failure_count, 0);
    }

    #[test]
    fn test_orchestrator_creation() {
        let config = LLMConfig {
            claude_api_key: Some("test_claude_key".to_string()),
            openai_api_key: Some("test_openai_key".to_string()),
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: vec![],
        };

        let orchestrator = LLMOrchestrator::new(config);
        assert!(orchestrator.claude_client.is_some());
        assert!(orchestrator.openai_client.is_some());
    }
}
