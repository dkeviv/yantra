// File: src-tauri/src/llm/orchestrator.rs
// Purpose: Multi-LLM orchestrator with failover and circuit breaker
// Dependencies: claude, openai modules
// Last Updated: November 20, 2025

use super::claude::ClaudeClient;
use super::openai::OpenAIClient;
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
    claude_circuit: Arc<RwLock<CircuitBreaker>>,
    openai_circuit: Arc<RwLock<CircuitBreaker>>,
}

impl LLMOrchestrator {
    pub fn new(config: LLMConfig) -> Self {
        let claude_client = config.claude_api_key.as_ref().map(|key| {
            ClaudeClient::new(key.clone(), config.timeout_seconds)
        });

        let openai_client = config.openai_api_key.as_ref().map(|key| {
            OpenAIClient::new(key.clone(), config.timeout_seconds)
        });

        Self {
            config: config.clone(),
            claude_client,
            openai_client,
            claude_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
            openai_circuit: Arc::new(RwLock::new(CircuitBreaker::new(3, 60))),
        }
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
            LLMProvider::Qwen => self.try_openai(request).await, // Qwen uses OpenAI-compatible API
        };

        if let Ok(response) = primary_result {
            return Ok(response);
        }

        // Failover to secondary provider
        let secondary_result = match self.config.primary_provider {
            LLMProvider::Claude => self.try_openai(request).await,
            LLMProvider::OpenAI => self.try_claude(request).await,
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
            primary_provider: LLMProvider::Claude,
            max_retries: 3,
            timeout_seconds: 30,
        };

        let orchestrator = LLMOrchestrator::new(config);
        assert!(orchestrator.claude_client.is_some());
        assert!(orchestrator.openai_client.is_some());
    }
}
