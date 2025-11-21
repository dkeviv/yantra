// File: src-tauri/src/llm/claude.rs
// Purpose: Claude Sonnet 4 API client
// Dependencies: reqwest, serde
// Last Updated: November 20, 2025

use super::{CodeGenerationRequest, CodeGenerationResponse, LLMError, LLMProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const CLAUDE_API_URL: &str = "https://api.anthropic.com/v1/messages";
const CLAUDE_MODEL: &str = "claude-sonnet-4-20250514"; // Latest model as of Nov 2025

#[derive(Debug, Clone)]
pub struct ClaudeClient {
    client: Client,
    api_key: String,
    timeout: Duration,
}

#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ClaudeMessage>,
    system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    id: String,
    content: Vec<ClaudeContent>,
    usage: ClaudeUsage,
}

#[derive(Debug, Deserialize)]
struct ClaudeContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl ClaudeClient {
    pub fn new(api_key: String, timeout_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    pub async fn generate_code(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let prompt = self.build_prompt(request);
        let system_prompt = self.build_system_prompt();

        let claude_request = ClaudeRequest {
            model: CLAUDE_MODEL.to_string(),
            max_tokens: 4096,
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            system: Some(system_prompt),
        };

        let response = self
            .client
            .post(CLAUDE_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&claude_request)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Failed to call Claude API: {}", e),
                provider: Some(LLMProvider::Claude),
                is_retryable: true,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError {
                message: format!("Claude API error ({}): {}", status, error_text),
                provider: Some(LLMProvider::Claude),
                is_retryable: status.is_server_error(),
            });
        }

        let claude_response: ClaudeResponse = response.json().await.map_err(|e| LLMError {
            message: format!("Failed to parse Claude response: {}", e),
            provider: Some(LLMProvider::Claude),
            is_retryable: false,
        })?;

        self.parse_response(claude_response)
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

You will be provided with:
- User intent (what they want to build)
- Context from the codebase (existing functions, classes)
- Dependency information (what the code depends on)

Generate ONLY the requested code. Do not include explanations unless specifically asked.
Format your response as:
```python
# Your code here
```

If generating tests, separate them with:
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
                prompt.push_str("\n");
            }
            prompt.push_str("\n");
        }

        if !request.dependencies.is_empty() {
            prompt.push_str("Dependencies:\n");
            for dep in &request.dependencies {
                prompt.push_str(&format!("- {}\n", dep));
            }
            prompt.push_str("\n");
        }

        prompt.push_str("Generate production-quality Python code with tests.");

        prompt
    }

    fn parse_response(
        &self,
        response: ClaudeResponse,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let text = response
            .content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        // Extract code blocks
        let code = self.extract_code_block(&text, "python");
        let tests = self.extract_tests_block(&text);

        if code.is_empty() {
            return Err(LLMError {
                message: "No code found in Claude response".to_string(),
                provider: Some(LLMProvider::Claude),
                is_retryable: false,
            });
        }

        Ok(CodeGenerationResponse {
            code,
            language: "python".to_string(),
            explanation: text.clone(),
            tests,
            provider: LLMProvider::Claude,
            tokens_used: response.usage.input_tokens + response.usage.output_tokens,
        })
    }

    fn extract_code_block(&self, text: &str, language: &str) -> String {
        // Find ```python ... ``` blocks
        let start_marker = format!("```{}", language);
        let end_marker = "```";

        if let Some(start_pos) = text.find(&start_marker) {
            let after_start = start_pos + start_marker.len();
            if let Some(end_pos) = text[after_start..].find(end_marker) {
                let code = &text[after_start..after_start + end_pos];
                return code.trim().to_string();
            }
        }

        // If no code block found, return empty
        String::new()
    }

    fn extract_tests_block(&self, text: &str) -> Option<String> {
        // Look for # tests comment followed by code block
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
    fn test_extract_code_block() {
        let client = ClaudeClient::new("test_key".to_string(), 30);
        let text = r#"Here is the code:
```python
def hello():
    print("Hello")
```
That's it!"#;

        let code = client.extract_code_block(text, "python");
        assert!(code.contains("def hello()"));
        assert!(code.contains("print(\"Hello\")"));
    }

    #[test]
    fn test_build_prompt() {
        let client = ClaudeClient::new("test_key".to_string(), 30);
        let request = CodeGenerationRequest {
            intent: "Create a function".to_string(),
            file_path: Some("test.py".to_string()),
            context: vec!["import os".to_string()],
            dependencies: vec!["utils.py".to_string()],
        };

        let prompt = client.build_prompt(&request);
        assert!(prompt.contains("Create a function"));
        assert!(prompt.contains("test.py"));
        assert!(prompt.contains("import os"));
        assert!(prompt.contains("utils.py"));
    }
}
