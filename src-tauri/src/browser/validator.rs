// Browser validator - Validates web applications

use crate::browser::cdp::{BrowserSession, ConsoleMessage};

pub struct ValidationResult {
    pub success: bool,
    pub errors: Vec<ConsoleMessage>,
    pub warnings: Vec<ConsoleMessage>,
    pub duration_ms: u64,
}

pub struct BrowserValidator {
    timeout_seconds: u64,
}

impl Default for BrowserValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserValidator {
    pub fn new() -> Self {
        Self {
            timeout_seconds: 30,
        }
    }

    pub async fn validate(&self, url: String) -> Result<ValidationResult, String> {
        let mut session = BrowserSession::new(url);
        let start = std::time::Instant::now();

        session.navigate().await?;
        let _messages = session.collect_messages(self.timeout_seconds).await?;

        let errors = session.get_errors();
        let warnings: Vec<ConsoleMessage> = Vec::new();

        Ok(ValidationResult {
            success: errors.is_empty(),
            errors,
            warnings,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = BrowserValidator::new();
        assert_eq!(validator.timeout_seconds, 30);
    }
}
