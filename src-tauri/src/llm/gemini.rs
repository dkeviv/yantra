// File: src-tauri/src/llm/gemini.rs
// Purpose: Google Gemini API client for advanced reasoning
// API: https://ai.google.dev/docs
// Last Updated: November 29, 2025

use super::{CodeGenerationRequest, CodeGenerationResponse, LLMError, LLMProvider};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const GEMINI_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL: &str = "gemini-1.5-pro"; // Latest model with best reasoning

/// Google Gemini API client
pub struct GeminiClient {
    client: Client,
    api_key: String,
    model: String,
    timeout: Duration,
}

/// Gemini request structure
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    temperature: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

/// Gemini response structure
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
}

impl GeminiClient {
    /// Create new Gemini client
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

    /// Set custom model (e.g., gemini-1.5-pro, gemini-1.5-flash)
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Generate code using Gemini API
    pub async fn generate_code(
        &self,
        request: &CodeGenerationRequest,
    ) -> Result<CodeGenerationResponse, LLMError> {
        let prompt = self.build_prompt(request);

        let gemini_request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart { text: prompt }],
            }],
            generation_config: GeminiGenerationConfig {
                temperature: 0.2, // Low temperature for focused code generation
                max_output_tokens: 8000, // Gemini has large context window
            },
        };

        let url = format!(
            "{}/{}:generateContent?key={}",
            GEMINI_API_URL, self.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_request)
            .send()
            .await
            .map_err(|e| LLMError {
                message: format!("Gemini API request failed: {}", e),
                provider: Some(LLMProvider::Gemini),
                is_retryable: e.is_timeout() || e.is_connect(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError {
                message: format!("Gemini API error {}: {}", status, error_text),
                provider: Some(LLMProvider::Gemini),
                is_retryable: status.is_server_error() || status == 429,
            });
        }

        let gemini_response: GeminiResponse = response.json().await.map_err(|e| LLMError {
            message: format!("Failed to parse Gemini response: {}", e),
            provider: Some(LLMProvider::Gemini),
            is_retryable: false,
        })?;

        self.parse_response(gemini_response)
    }

    fn build_prompt(&self, request: &CodeGenerationRequest) -> String {
        let mut prompt = String::new();

        prompt.push_str("You are an expert Python developer. Generate clean, production-quality Python code.\n\n");
        
        prompt.push_str("Requirements:\n");
        prompt.push_str("- Write idiomatic Python code following PEP 8\n");
        prompt.push_str("- Include comprehensive docstrings\n");
        prompt.push_str("- Add type hints for all functions\n");
        prompt.push_str("- Include proper error handling\n");
        prompt.push_str("- Generate unit tests in a separate code block\n\n");

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

        prompt.push_str("\nFormat your response with:\n");
        prompt.push_str("1. Code in ```python block\n");
        prompt.push_str("2. Tests in ```tests block (if applicable)\n");
        prompt.push_str("3. Brief explanation of the implementation\n");

        prompt
    }

    fn parse_response(
        &self,
        response: GeminiResponse,
    ) -> Result<CodeGenerationResponse, LLMError> {
        if response.candidates.is_empty() {
            return Err(LLMError {
                message: "No candidates in Gemini response".to_string(),
                provider: Some(LLMProvider::Gemini),
                is_retryable: false,
            });
        }

        let candidate = &response.candidates[0];
        if candidate.content.parts.is_empty() {
            return Err(LLMError {
                message: "No parts in Gemini response".to_string(),
                provider: Some(LLMProvider::Gemini),
                is_retryable: false,
            });
        }

        let content = &candidate.content.parts[0].text;

        // Extract code from response
        let code = self.extract_code_block(content, "python");
        let tests = self.extract_code_block(content, "tests");

        if code.is_empty() {
            return Err(LLMError {
                message: "No Python code found in Gemini response".to_string(),
                provider: Some(LLMProvider::Gemini),
                is_retryable: false,
            });
        }

        Ok(CodeGenerationResponse {
            code,
            language: "python".to_string(),
            explanation: content.clone(),
            tests: if tests.is_empty() { None } else { Some(tests) },
            provider: LLMProvider::Gemini,
            tokens_used: response
                .usage_metadata
                .and_then(|u| u.total_token_count)
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
        let client = GeminiClient::new("test_key".to_string(), Duration::from_secs(30));
        
        let content = r#"Here's the implementation:

```python
def calculate(x: int, y: int) -> int:
    """Calculate sum."""
    return x + y
```

And here are the tests:

```tests
def test_calculate():
    assert calculate(2, 3) == 5
```"#;

        let code = client.extract_code_block(content, "python");
        assert!(code.contains("def calculate"));
        
        let tests = client.extract_code_block(content, "tests");
        assert!(tests.contains("test_calculate"));
    }
}
