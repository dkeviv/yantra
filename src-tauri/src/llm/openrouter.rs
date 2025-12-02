// File: src-tauri/src/llm/openrouter.rs
// Purpose: OpenRouter API client - provides access to multiple LLM providers
// Dependencies: reqwest, serde
// Last Updated: November 29, 2025

use super::{CodeGenerationRequest, CodeGenerationResponse, LLMError, LLMProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL: &str = "anthropic/claude-3.5-sonnet"; // Default model

#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    client: Client,
    api_key: String,
    model: String,
    #[allow(dead_code)]
    timeout: Duration,
}

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    total_tokens: Option<u32>,
}

impl OpenRouterClient {
    pub fn new(api_key: String, timeout_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            model: DEFAULT_MODEL.to_string(),
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    /// Set custom model (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo")
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    pub async fn generate_code(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_prompt(request);

        let openrouter_request = OpenRouterRequest {
            model: self.model.clone(),
            messages: vec![
                OpenRouterMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                OpenRouterMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            max_tokens: Some(4096),
            temperature: Some(0.2), // Lower temperature for more deterministic code
        };

        let response = self
            .client
            .post(OPENROUTER_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://yantra.dev") // Required by OpenRouter
            .header("X-Title", "Yantra") // Optional: shown in OpenRouter dashboard
            .json(&openrouter_request)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to call OpenRouter API: {}", e),
                provider: Some(LLMProvider::OpenRouter),
                is_retryable: true,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError {
                message: format!("OpenRouter API error ({}): {}", status, error_text),
                provider: Some(LLMProvider::OpenRouter),
                is_retryable: status.is_server_error(),
            });
        }

        let openrouter_response: OpenRouterResponse =
            response.json().await.map_err(|e| LLMError {
                message: format!("Failed to parse OpenRouter response: {}", e),
                provider: Some(LLMProvider::OpenRouter),
                is_retryable: false,
            })?;

        self.parse_response(openrouter_response)
    }

    fn build_system_prompt(&self) -> String {
        r#"You are an expert Python developer working on the Yantra AI-first development platform.
Your task is to generate production-quality Python code that:
1. Never breaks existing code
2. Follows PEP 8 style guidelines
3. Includes comprehensive type hints
4. Has detailed docstrings
5. Implements proper error handling
6. Is fully tested

Generate ONLY the requested code in Python. Format your response with code blocks:
```python
# Your code here
```

If generating tests:
```python
# tests
# Your test code here
```"#
            .to_string()
    }

    fn build_prompt(&self, request: &CodeGenerationRequest) -> String {
        let mut prompt = String::new();

        prompt.push_str(&format!("Task: {}\n\n", request.intent));

        if !request.context.is_empty() {
            prompt.push_str("Context:\n");
            for ctx in &request.context {
                prompt.push_str(&format!("{}\n", ctx));
            }
            prompt.push('\n');
        }

        if !request.dependencies.is_empty() {
            prompt.push_str("Dependencies:\n");
            for dep in &request.dependencies {
                prompt.push_str(&format!("- {}\n", dep));
            }
            prompt.push('\n');
        }

        if let Some(ref file_path) = request.file_path {
            prompt.push_str(&format!("Target File: {}\n\n", file_path));
        }

        prompt
    }

    fn parse_response(
        &self,
        response: OpenRouterResponse,
    ) -> Result<CodeGenerationResponse, LLMError> {
        if response.choices.is_empty() {
            return Err(LLMError {
                message: "No choices in OpenRouter response".to_string(),
                provider: Some(LLMProvider::OpenRouter),
                is_retryable: false,
            });
        }

        let message = &response.choices[0].message;
        let content = &message.content;

        // Extract code from response
        let code = self.extract_code_block(content, "python");
        let tests = self.extract_code_block(content, "tests");

        if code.is_empty() {
            return Err(LLMError {
                message: "No Python code found in OpenRouter response".to_string(),
                provider: Some(LLMProvider::OpenRouter),
                is_retryable: false,
            });
        }

        Ok(CodeGenerationResponse {
            code,
            language: "python".to_string(),
            explanation: content.clone(),
            tests: if tests.is_empty() { None } else { Some(tests) },
            provider: LLMProvider::OpenRouter,
            tokens_used: response
                .usage
                .and_then(|u| u.total_tokens)
                .unwrap_or(0) as u32,
        })
    }

    fn extract_code_block(&self, content: &str, marker: &str) -> String {
        let start_marker = format!("```{}", marker);
        let end_marker = "```";

        let mut code = String::new();
        let mut in_block = false;

        for line in content.lines() {
            if line.trim().starts_with(&start_marker) {
                in_block = true;
                continue;
            }
            if in_block && line.trim().starts_with(end_marker) {
                break;
            }
            if in_block {
                code.push_str(line);
                code.push('\n');
            }
        }

        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_block() {
        let client = OpenRouterClient::new("test_key".to_string(), 30);
        let content = r#"
Here's the solution:

```python
def hello():
    print("Hello, world!")
```

```tests
def test_hello():
    assert True
```
        "#;

        let code = client.extract_code_block(content, "python");
        assert!(code.contains("def hello():"));

        let tests = client.extract_code_block(content, "tests");
        assert!(tests.contains("def test_hello():"));
    }
}
