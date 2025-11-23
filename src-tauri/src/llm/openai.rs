// File: src-tauri/src/llm/openai.rs
// Purpose: OpenAI GPT-4 Turbo API client (fallback)
// Dependencies: reqwest, serde
// Last Updated: November 20, 2025

use super::{CodeGenerationRequest, CodeGenerationResponse, LLMError, LLMProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const OPENAI_MODEL: &str = "gpt-4-turbo-2024-04-09";

#[derive(Debug, Clone)]
pub struct OpenAIClient {
    client: Client,
    api_key: String,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    timeout: Duration,
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    total_tokens: u32,
}

impl OpenAIClient {
    pub fn new(api_key: String, timeout_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            model: OPENAI_MODEL.to_string(),
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    pub async fn generate_code(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_prompt(request);

        let openai_request = OpenAIRequest {
            model: OPENAI_MODEL.to_string(),
            messages: vec![
                OpenAIMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                OpenAIMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            max_tokens: 4096,
            temperature: 0.2, // Lower temperature for more deterministic code
        };

        let response = self
            .client
            .post(OPENAI_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&openai_request)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to call OpenAI API: {}", e),
                provider: Some(LLMProvider::OpenAI),
                is_retryable: true,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError {
                message: format!("OpenAI API error ({}): {}", status, error_text),
                provider: Some(LLMProvider::OpenAI),
                is_retryable: status.is_server_error(),
            });
        }

        let openai_response: OpenAIResponse = response.json().await.map_err(|e| LLMError {
            message: format!("Failed to parse OpenAI response: {}", e),
            provider: Some(LLMProvider::OpenAI),
            is_retryable: false,
        })?;

        self.parse_response(openai_response)
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

        prompt.push_str("User Intent:\n");
        prompt.push_str(&request.intent);
        prompt.push_str("\n\n");

        if let Some(file_path) = &request.file_path {
            prompt.push_str(&format!("File Path: {}\n\n", file_path));
        }

        if !request.context.is_empty() {
            prompt.push_str("Context (existing code):\n");
            for ctx in &request.context {
                prompt.push_str(ctx);
                prompt.push('\n');
            }
            prompt.push('\n');
        }

        if !request.dependencies.is_empty() {
            prompt.push_str("Dependencies:\n");
            for dep in &request.dependencies {
                prompt.push_str(&format!("- {}\n", dep));
            }
        }

        prompt
    }

    fn parse_response(
        &self,
        response: OpenAIResponse,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let text = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let code = self.extract_code_block(&text, "python");
        let tests = self.extract_tests_block(&text);

        if code.is_empty() {
            return Err(LLMError {
                message: "No code found in OpenAI response".to_string(),
                provider: Some(LLMProvider::OpenAI),
                is_retryable: false,
            });
        }

        Ok(CodeGenerationResponse {
            code,
            language: "python".to_string(),
            explanation: text,
            tests,
            provider: LLMProvider::OpenAI,
            tokens_used: response.usage.total_tokens,
        })
    }

    fn extract_code_block(&self, text: &str, language: &str) -> String {
        let start_marker = format!("```{}", language);
        let end_marker = "```";

        if let Some(start_pos) = text.find(&start_marker) {
            let after_start = start_pos + start_marker.len();
            if let Some(end_pos) = text[after_start..].find(end_marker) {
                let code = &text[after_start..after_start + end_pos];
                return code.trim().to_string();
            }
        }

        String::new()
    }

    fn extract_tests_block(&self, text: &str) -> Option<String> {
        if let Some(tests_pos) = text.find("# tests") {
            let after_tests = &text[tests_pos..];
            if let Some(code) = self.extract_code_block(after_tests, "python").into() {
                if !code.is_empty() {
                    return Some(code);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let client = OpenAIClient::new("test_key".to_string(), 30);
        let request = CodeGenerationRequest {
            intent: "Create a calculator".to_string(),
            file_path: Some("calc.py".to_string()),
            context: vec![],
            dependencies: vec![],
        };

        let prompt = client.build_prompt(&request);
        assert!(prompt.contains("Create a calculator"));
        assert!(prompt.contains("calc.py"));
    }
}
