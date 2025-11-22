// Token counting module using tiktoken-rs
// Provides exact token counting for different LLM providers

use tiktoken_rs::{cl100k_base, CoreBPE};
use std::sync::OnceLock;

/// Global tokenizer instance (cl100k_base used by Claude and GPT-4)
static TOKENIZER: OnceLock<CoreBPE> = OnceLock::new();

/// Initialize the tokenizer (called once)
fn get_tokenizer() -> &'static CoreBPE {
    TOKENIZER.get_or_init(|| {
        cl100k_base().expect("Failed to load cl100k_base tokenizer")
    })
}

/// Count tokens in a text string
/// 
/// Uses cl100k_base tokenizer (compatible with Claude and GPT-4)
/// Performance: <10ms for typical code snippets
pub fn count_tokens(text: &str) -> usize {
    let tokenizer = get_tokenizer();
    tokenizer.encode_with_special_tokens(text).len()
}

/// Count tokens for multiple text strings
/// 
/// More efficient than calling count_tokens multiple times
pub fn count_tokens_batch(texts: &[&str]) -> Vec<usize> {
    let tokenizer = get_tokenizer();
    texts.iter()
        .map(|text| tokenizer.encode_with_special_tokens(text).len())
        .collect()
}

/// Estimate if adding text would exceed token limit
/// 
/// Returns true if current_tokens + new_text would exceed limit
pub fn would_exceed_limit(current_tokens: usize, new_text: &str, limit: usize) -> bool {
    let new_tokens = count_tokens(new_text);
    current_tokens + new_tokens > limit
}

/// Truncate text to fit within token limit
/// 
/// Truncates at approximately the right byte position to stay under limit
/// Note: This is approximate and may need refinement for exact truncation
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let tokenizer = get_tokenizer();
    let tokens = tokenizer.encode_with_special_tokens(text);
    
    if tokens.len() <= max_tokens {
        return text.to_string();
    }
    
    // Decode only the first max_tokens tokens
    let truncated_tokens: Vec<usize> = tokens.iter()
        .take(max_tokens)
        .copied()
        .collect();
    
    match tokenizer.decode(truncated_tokens) {
        Ok(decoded) => decoded,
        Err(_) => {
            // Fallback: simple character truncation
            let chars_per_token = text.len() / tokens.len().max(1);
            let approx_chars = max_tokens * chars_per_token;
            text.chars().take(approx_chars).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_simple() {
        let text = "Hello, world!";
        let count = count_tokens(text);
        // "Hello, world!" is typically 4 tokens with cl100k_base
        assert!(count >= 3 && count <= 5, "Expected ~4 tokens, got {}", count);
    }

    #[test]
    fn test_count_tokens_code() {
        let code = r#"
def hello_world():
    print("Hello, world!")
    return True
"#;
        let count = count_tokens(code);
        // This should be around 15-25 tokens
        assert!(count >= 10 && count <= 30, "Expected ~15-25 tokens for code, got {}", count);
    }

    #[test]
    fn test_count_tokens_batch() {
        let texts = vec!["Hello", "World", "Test"];
        let counts = count_tokens_batch(&texts);
        assert_eq!(counts.len(), 3);
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_would_exceed_limit() {
        let current = 100;
        let text = "Hello world this is a test";
        let limit = 110;
        
        // Should not exceed (100 + ~5-6 tokens < 110)
        assert!(!would_exceed_limit(current, text, limit));
        
        // Should exceed with very low limit
        assert!(would_exceed_limit(current, text, 103));
    }

    #[test]
    fn test_truncate_to_tokens() {
        let long_text = "This is a very long text that needs to be truncated. ".repeat(10);
        let max_tokens = 20;
        
        let truncated = truncate_to_tokens(&long_text, max_tokens);
        let truncated_count = count_tokens(&truncated);
        
        // Should be at or under the limit
        assert!(truncated_count <= max_tokens, 
            "Truncated text has {} tokens, expected <= {}", 
            truncated_count, max_tokens);
    }

    #[test]
    fn test_empty_string() {
        let count = count_tokens("");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_unicode() {
        let text = "Hello 世界 🌍";
        let count = count_tokens(text);
        // Should handle unicode correctly
        assert!(count > 0);
    }

    #[test]
    fn test_performance() {
        // Warm up the tokenizer first (first call loads the model)
        let _ = count_tokens("warmup");
        
        // Now test performance on actual code
        let code = r#"
def fibonacci(n: int) -> int:
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"#;
        
        let start = std::time::Instant::now();
        let _count = count_tokens(code);
        let duration = start.elapsed();
        
        // After warmup, should be very fast
        assert!(duration.as_millis() < 10, 
            "Token counting took {}ms, expected <10ms (after warmup)", 
            duration.as_millis());
    }
}
