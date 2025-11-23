// Chrome DevTools Protocol client
// Monitors console output and JavaScript errors in browser

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsoleLevel {
    Log,
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleMessage {
    pub level: ConsoleLevel,
    pub message: String,
    pub source: String,
    pub line_number: Option<usize>,
    pub timestamp: u64,
}

pub struct BrowserSession {
    #[allow(dead_code)]
    url: String,
    messages: Vec<ConsoleMessage>,
}

impl BrowserSession {
    pub fn new(url: String) -> Self {
        Self {
            url,
            messages: Vec::new(),
        }
    }

    pub async fn navigate(&mut self) -> Result<(), String> {
        // Connect to Chrome via CDP and navigate to URL
        // For MVP: Placeholder implementation
        Ok(())
    }

    pub async fn collect_messages(&mut self, _duration_seconds: u64) -> Result<Vec<ConsoleMessage>, String> {
        // Listen to Console.messageAdded events
        // For MVP: Return empty list
        Ok(self.messages.clone())
    }

    pub fn get_errors(&self) -> Vec<ConsoleMessage> {
        self.messages
            .iter()
            .filter(|m| m.level == ConsoleLevel::Error)
            .cloned()
            .collect()
    }

    pub fn has_errors(&self) -> bool {
        self.messages.iter().any(|m| m.level == ConsoleLevel::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_session_creation() {
        let session = BrowserSession::new("http://localhost:3000".to_string());
        assert_eq!(session.url, "http://localhost:3000");
        assert_eq!(session.messages.len(), 0);
    }

    #[test]
    fn test_console_levels() {
        assert!(matches!(ConsoleLevel::Error, ConsoleLevel::Error));
    }
}
