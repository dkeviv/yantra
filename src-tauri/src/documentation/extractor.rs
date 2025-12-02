// File: src-tauri/src/documentation/extractor.rs
// Purpose: LLM-based feature extraction from chat conversations
// Dependencies: serde, crate::llm, crate::documentation
// Last Updated: November 24, 2025

use serde::{Deserialize, Serialize};
use crate::llm::{LLMConfig, LLMProvider};
use super::{Feature, FeatureStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionRequest {
    pub chat_message: String,
    pub context: Option<String>, // Previous messages for context
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionResponse {
    pub features: Vec<ExtractedFeature>,
    pub confidence: f32, // 0.0 - 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFeature {
    pub title: String,
    pub description: String,
    pub status: FeatureStatus,
}

/// Extract features from chat messages using LLM
pub async fn extract_features_from_chat(
    request: FeatureExtractionRequest,
    llm_config: &LLMConfig,
) -> Result<FeatureExtractionResponse, String> {
    // Build the extraction prompt
    let prompt = build_extraction_prompt(&request);
    
    // Call LLM based on provider
    let response = match llm_config.primary_provider {
        LLMProvider::Claude => {
            if let Some(ref api_key) = llm_config.claude_api_key {
                call_claude_for_extraction(&prompt, api_key, llm_config.timeout_seconds).await?
            } else {
                return Err("Claude API key not configured".to_string());
            }
        }
        LLMProvider::OpenAI => {
            if let Some(ref api_key) = llm_config.openai_api_key {
                call_openai_for_extraction(&prompt, api_key, llm_config.timeout_seconds).await?
            } else {
                return Err("OpenAI API key not configured".to_string());
            }
        }
        LLMProvider::OpenRouter => {
            if let Some(ref api_key) = llm_config.openrouter_api_key {
                call_openai_for_extraction(&prompt, api_key, llm_config.timeout_seconds).await?
            } else {
                return Err("OpenRouter API key not configured".to_string());
            }
        }
        LLMProvider::Groq => {
            if let Some(ref api_key) = llm_config.groq_api_key {
                call_openai_for_extraction(&prompt, api_key, llm_config.timeout_seconds).await?
            } else {
                return Err("Groq API key not configured".to_string());
            }
        }
        LLMProvider::Gemini => {
            return Err("Gemini provider not yet supported for feature extraction".to_string());
        }
        LLMProvider::Qwen => {
            return Err("Qwen provider not yet supported for feature extraction".to_string());
        }
    };
    
    // Parse the LLM response
    parse_extraction_response(&response)
}

/// Build the feature extraction prompt
fn build_extraction_prompt(request: &FeatureExtractionRequest) -> String {
    let context_section = if let Some(ctx) = &request.context {
        format!("Previous conversation context:\n{}\n\n", ctx)
    } else {
        String::new()
    };
    
    format!(
        r#"{}User message: "{}"

Analyze the user's message and extract any features they are requesting or describing. 
A feature is a capability, functionality, or system component the user wants to build.

For each feature found, provide:
1. Title: A concise name (3-7 words)
2. Description: What it does and why it's needed (1-2 sentences)
3. Status: "planned" (user just mentioned it) or "in-progress" (if they're working on it)

Return ONLY a JSON object in this exact format:
{{
    "features": [
        {{
            "title": "Feature Name",
            "description": "What it does",
            "status": "planned"
        }}
    ],
    "confidence": 0.85
}}

If no features are detected, return:
{{
    "features": [],
    "confidence": 0.0
}}

Important:
- Only extract actual feature requests, not questions or general discussion
- Be conservative - it's better to miss a feature than create a false positive
- Set confidence based on how explicit the user's request was (0.0-1.0)
- Return ONLY valid JSON, no explanatory text"#,
        context_section, request.chat_message
    )
}

/// Call Claude API for feature extraction
async fn call_claude_for_extraction(
    prompt: &str,
    api_key: &str,
    timeout_seconds: u64,
) -> Result<String, String> {
    use crate::llm::claude::ClaudeClient;
    use crate::llm::CodeGenerationRequest;
    
    let client = ClaudeClient::new(api_key.to_string(), timeout_seconds);
    
    // Use generate_code method with feature extraction as the "intent"
    let request = CodeGenerationRequest {
        intent: prompt.to_string(),
        file_path: None,
        context: vec![],
        dependencies: vec![],
    };
    
    client.generate_code(&request).await
        .map(|response| response.code)
        .map_err(|e| format!("Claude API error: {:?}", e))
}

/// Call OpenAI API for feature extraction
async fn call_openai_for_extraction(
    prompt: &str,
    api_key: &str,
    timeout_seconds: u64,
) -> Result<String, String> {
    use crate::llm::openai::OpenAIClient;
    use crate::llm::CodeGenerationRequest;
    
    let client = OpenAIClient::new(api_key.to_string(), timeout_seconds);
    
    // Use generate_code method with feature extraction as the "intent"
    let request = CodeGenerationRequest {
        intent: prompt.to_string(),
        file_path: None,
        context: vec![],
        dependencies: vec![],
    };
    
    client.generate_code(&request).await
        .map(|response| response.code)
        .map_err(|e| format!("OpenAI API error: {:?}", e))
}

/// Parse the LLM response into structured format
fn parse_extraction_response(response: &str) -> Result<FeatureExtractionResponse, String> {
    // Try to extract JSON from response (LLM might include explanation text)
    let json_str = extract_json_from_response(response)?;
    
    // Parse JSON
    serde_json::from_str::<FeatureExtractionResponse>(&json_str)
        .map_err(|e| format!("Failed to parse extraction response: {}", e))
}

/// Extract JSON object from LLM response (handles cases where LLM adds text)
fn extract_json_from_response(response: &str) -> Result<String, String> {
    // Find the first { and last }
    let start = response.find('{')
        .ok_or_else(|| "No JSON object found in response".to_string())?;
    let end = response.rfind('}')
        .ok_or_else(|| "No JSON object found in response".to_string())?;
    
    if end < start {
        return Err("Invalid JSON structure in response".to_string());
    }
    
    Ok(response[start..=end].to_string())
}

/// Convert extracted features to Feature objects
pub fn convert_to_features(
    extracted: Vec<ExtractedFeature>,
    extracted_from: String,
) -> Vec<Feature> {
    extracted
        .into_iter()
        .enumerate()
        .map(|(i, ef)| Feature {
            id: format!("chat_{}", i + 1),
            title: ef.title,
            description: ef.description,
            status: ef.status,
            extracted_from: extracted_from.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_from_response() {
        let response = r#"Here's what I found: {"features": [], "confidence": 0.5} Let me explain..."#;
        let result = extract_json_from_response(response);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("features"));
    }

    #[test]
    fn test_extract_json_no_json() {
        let response = "No JSON here";
        let result = extract_json_from_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_extraction_response() {
        let response = r#"{"features": [{"title": "Test", "description": "A test", "status": "planned"}], "confidence": 0.9}"#;
        let result = parse_extraction_response(response);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.features.len(), 1);
        assert_eq!(parsed.confidence, 0.9);
    }

    #[test]
    fn test_convert_to_features() {
        let extracted = vec![
            ExtractedFeature {
                title: "Feature 1".to_string(),
                description: "First feature".to_string(),
                status: FeatureStatus::Planned,
            }
        ];
        let features = convert_to_features(extracted, "Chat: test".to_string());
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].title, "Feature 1");
    }

    #[test]
    fn test_build_extraction_prompt() {
        let request = FeatureExtractionRequest {
            chat_message: "Add payment processing".to_string(),
            context: Some("Previous chat".to_string()),
        };
        let prompt = build_extraction_prompt(&request);
        assert!(prompt.contains("Add payment processing"));
        assert!(prompt.contains("Previous chat"));
        assert!(prompt.contains("JSON"));
    }
}
