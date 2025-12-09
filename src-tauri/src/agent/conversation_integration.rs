// File: src-tauri/src/agent/conversation_integration.rs
// Purpose: Integration layer between conversation memory and state machines
// Last Updated: December 9, 2025

use super::conversation_memory::{ConversationMemory, MessageRole, SessionType};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared conversation memory instance
pub struct ConversationContext {
    memory: Arc<Mutex<ConversationMemory>>,
    current_conversation_id: Arc<Mutex<Option<String>>>,
}

impl ConversationContext {
    /// Create new conversation context
    pub fn new(db_path: String) -> Result<Self, String> {
        let memory = ConversationMemory::new(&std::path::PathBuf::from(db_path))
            .map_err(|e| format!("Failed to initialize conversation memory: {}", e))?;
        
        Ok(Self {
            memory: Arc::new(Mutex::new(memory)),
            current_conversation_id: Arc::new(Mutex::new(None)),
        })
    }
    
    /// Get or create current conversation
    pub async fn get_or_create_conversation(&self) -> Result<String, String> {
        let mut current_id = self.current_conversation_id.lock().await;
        
        if let Some(id) = current_id.as_ref() {
            return Ok(id.clone());
        }
        
        // No current conversation, check for last active
        let memory = self.memory.lock().await;
        if let Some(conversation) = memory.get_last_active_conversation()
            .map_err(|e| format!("Failed to get last active conversation: {}", e))? {
            let id = conversation.id.clone();
            *current_id = Some(id.clone());
            return Ok(id);
        }
        
        // Create new conversation
        let conversation = memory.create_conversation(None)
            .map_err(|e| format!("Failed to create conversation: {}", e))?;
        let id = conversation.id.clone();
        *current_id = Some(id.clone());
        Ok(id)
    }
    
    /// Save user message and return conversation context
    pub async fn save_user_message(
        &self,
        content: &str,
        tokens: Option<usize>,
    ) -> Result<String, String> {
        let conversation_id = self.get_or_create_conversation().await?;
        
        let memory = self.memory.lock().await;
        memory.save_message(
            &conversation_id,
            MessageRole::User,
            content.to_string(),
            tokens.unwrap_or(0),
            None,
            None,
        ).map_err(|e| format!("Failed to save user message: {}", e))?;
        
        Ok(conversation_id)
    }
    
    /// Save assistant message
    pub async fn save_assistant_message(
        &self,
        content: &str,
        tokens: Option<usize>,
        metadata: Option<&str>,
    ) -> Result<String, String> {
        let conversation_id = self.get_or_create_conversation().await?;
        
        let memory = self.memory.lock().await;
        let metadata_json = metadata.and_then(|m| serde_json::from_str(m).ok());
        let message = memory.save_message(
            &conversation_id,
            MessageRole::Assistant,
            content.to_string(),
            tokens.unwrap_or(0),
            None,
            metadata_json,
        ).map_err(|e| format!("Failed to save assistant message: {}", e))?;
        
        Ok(message.id)
    }
    
    /// Get recent conversation context (last N messages)
    pub async fn get_recent_context(&self, count: usize) -> Result<String, String> {
        let conversation_id = self.get_or_create_conversation().await?;
        
        let memory = self.memory.lock().await;
        let messages = memory.load_recent_messages(&conversation_id, count)
            .map_err(|e| format!("Failed to load recent messages: {}", e))?;
        
        // Format messages as context string
        let context = messages.iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::System => "System",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        
        Ok(context)
    }
    
    /// Link current conversation to a code generation session
    pub async fn link_code_generation(
        &self,
        session_id: &str,
        code: &str,
    ) -> Result<(), String> {
        let conversation_id = self.get_or_create_conversation().await?;
        
        // Save the generated code as an assistant message
        let memory = self.memory.lock().await;
        let metadata = serde_json::json!({"session_id": session_id});
        let message = memory.save_message(
            &conversation_id,
            MessageRole::Assistant,
            format!("Generated code for session {}:\n\n```\n{}\n```", session_id, code),
            0,
            None,
            Some(metadata),
        ).map_err(|e| format!("Failed to save code generation message: {}", e))?;
        
        // Create session link
        let metadata = serde_json::json!({"code_length": code.len()});
        memory.link_to_session(
            &conversation_id,
            &message.id,
            SessionType::CodeGeneration,
            session_id,
            Some(metadata),
        ).map_err(|e| format!("Failed to link code generation session: {}", e))?;
        
        Ok(())
    }
    
    /// Link current conversation to a testing session
    pub async fn link_testing(
        &self,
        session_id: &str,
        test_code: &str,
        test_count: usize,
    ) -> Result<(), String> {
        let conversation_id = self.get_or_create_conversation().await?;
        
        // Save the generated tests as an assistant message
        let memory = self.memory.lock().await;
        let msg_metadata = serde_json::json!({"session_id": session_id, "test_count": test_count});
        let message = memory.save_message(
            &conversation_id,
            MessageRole::Assistant,
            format!("Generated {} tests for session {}:\n\n```\n{}\n```", test_count, session_id, test_code),
            0,
            None,
            Some(msg_metadata),
        ).map_err(|e| format!("Failed to save testing message: {}", e))?;
        
        // Create session link
        let link_metadata = serde_json::json!({"test_count": test_count});
        memory.link_to_session(
            &conversation_id,
            &message.id,
            SessionType::Testing,
            session_id,
            Some(link_metadata),
        ).map_err(|e| format!("Failed to link testing session: {}", e))?;
        
        Ok(())
    }
    
    /// Start a new conversation (for new chat sessions)
    pub async fn start_new_conversation(&self) -> Result<String, String> {
        let memory = self.memory.lock().await;
        let conversation = memory.create_conversation(None)
            .map_err(|e| format!("Failed to create conversation: {}", e))?;
        
        let mut current_id = self.current_conversation_id.lock().await;
        *current_id = Some(conversation.id.clone());
        
        Ok(conversation.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_conversation_context_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_conversation_context.db");
        let _ = std::fs::remove_file(&db_path); // Clean up from previous runs
        
        let context = ConversationContext::new(db_path.to_string_lossy().to_string())
            .expect("Failed to create conversation context");
        
        let conversation_id = context.get_or_create_conversation().await
            .expect("Failed to get or create conversation");
        
        assert!(!conversation_id.is_empty());
    }
    
    #[tokio::test]
    async fn test_save_and_retrieve_messages() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_messages.db");
        let _ = std::fs::remove_file(&db_path);
        
        let context = ConversationContext::new(db_path.to_string_lossy().to_string())
            .expect("Failed to create conversation context");
        
        // Save user message
        context.save_user_message("Create a Python function", Some(10)).await
            .expect("Failed to save user message");
        
        // Save assistant message
        context.save_assistant_message("def example():\n    pass", Some(20), None).await
            .expect("Failed to save assistant message");
        
        // Get recent context
        let recent = context.get_recent_context(5).await
            .expect("Failed to get recent context");
        
        assert!(recent.contains("Create a Python function"));
        assert!(recent.contains("def example()"));
    }
    
    #[tokio::test]
    async fn test_session_linking() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_session_linking.db");
        let _ = std::fs::remove_file(&db_path);
        
        let context = ConversationContext::new(db_path.to_string_lossy().to_string())
            .expect("Failed to create conversation context");
        
        // Save initial message
        context.save_user_message("Generate a test", Some(10)).await
            .expect("Failed to save user message");
        
        // Link to code generation
        context.link_code_generation("session-123", "def test(): pass").await
            .expect("Failed to link code generation");
        
        // Link to testing
        context.link_testing("test-456", "def test_example(): assert True", 1).await
            .expect("Failed to link testing");
    }
}
