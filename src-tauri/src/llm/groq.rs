// File: src-tauri/src/llm/groq.rs
// Purpose: Groq API client for fast inference with LLaMA models
// API: https://console.groq.com/docs/quickstart
// Last Updated: November 29, 2025

use super::{CodeGenerationRequest, CodeGenerationResponse, LLMError, LLMProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const GROQ_API_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
const DEFAULT_MODEL: &str = "llama-3.1-70b-versatile"; // Fast, high-quality model

/// Groq API client (OpenAI-compatible)
pub struct GroqClient {
    client: Client,
    api_key: String,
    model: String,
    timeout: Duration,
}

/// Groq request structure (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct GroqMessage {
    role: String,
    content: String,
}

/// Groq response structure (OpenAI-compatible)
#[derive(Debug, Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
    usage: Option<GroqUsage>,
}

#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqResponseMessage,
}

#[derive(Debug, Deserialize)]
struct GroqResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct GroqUsage {
    total_tokens: Option<u32>,
}

impl GroqClient {
    /// Create new Groq client
    pub fn new(api_key: String, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            api_key,
            model: DEFAULT_MODEL.to_string(),
            timeout,
        }
    }

    /// Set custom model (e.g., llama-3.1-70b-versatile, mixtral-8x7b-32768)
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Generate code using Groq API
    pub async fn generate_code(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_prompt(request);

        let groq_request = GroqRequest {
            model: self.model.clone(),
            messages: vec![
                GroqMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                GroqMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            temperature: 0.2, // Low temperature for more focused code generation
            max_tokens: 4000,
        };

        let response = self
            .client
            .post(GROQ_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&groq_request)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Groq API request failed: {}", e),
                provider: Some(LLMProvider::Groq),
                is_retryable: e.is_timeout() || e.is_connect(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError {
                message: format!("Groq API error {}: {}", status, error_text),
                provider: Some(LLMProvider::Groq),
                is_retryable: status.is_server_error() || status == 429,
            });
        }

        let groq_response: GroqResponse = response.json().await.map_err(|e| LLMError {
            message: format!("Failed to parse Groq response: {}", e),
            provider: Some(LLMProvider::Groq),
            is_retryable: false,
        })?;

        self.parse_response(groq_response)
    }

    fn build_system_prompt(&self) -> String {
        r#"You are an expert Python developer. Generate clean, production-quality Python code.

Requirements:
- Write idiomatic Python code following PEP 8
- Include comprehensive docstrings
- Add type hints for all functions
- Include proper error handling
- Generate unit tests in a separate code block

Format your response with:
1. Code in ```python block
2. Tests in ```tests block (if applicable)
3. Brief explanation of the implementation"#
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
        response: GroqResponse,
    ) -> Result<CodeGenerationResponse, LLMError> {
        if response.choices.is_empty() {
            return Err(LLMError {
                message: "No choices in Groq response".to_string(),
                provider: Some(LLMProvider::Groq),
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
                message: "No Python code found in Groq response".to_string(),
                provider: Some(LLMProvider::Groq),
                is_retryable: false,
            });
        }

        Ok(CodeGenerationResponse {
            code,
            language: "python".to_string(),
            explanation: content.clone(),
            tests: if tests.is_empty() { None } else { Some(tests) },
            provider: LLMProvider::Groq,
            tokens_used: response
                .usage
                .and_then(|u| u.total_tokens)
                .unwrap_or(0),
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

        code.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_block() {
        let client = GroqClient::new("test_key".to_string(), Duration::from_secs(30));
        
        let content = r#"Here's the code:

```python
def hello():
    print("Hello, World!")
```

And here are the tests:

```tests
def test_hello():
    assert True
```"#;

        let code = client.extract_code_block(content, "python");
        assert!(code.contains("def hello()"));
        
        let tests = client.extract_code_block(content, "tests");
        assert!(tests.contains("test_hello"));
    }
}
