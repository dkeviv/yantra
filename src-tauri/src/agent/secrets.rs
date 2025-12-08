// Secrets Manager: Encrypted storage for API keys and credentials
// Purpose: Secure storage and retrieval of sensitive information

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Secret entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Secret {
    pub key: String,
    pub value: String, // Encrypted
    pub description: Option<String>,
    pub created_at: String,
    pub last_accessed: Option<String>,
}

/// Secrets vault
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Vault {
    secrets: HashMap<String, Secret>,
}

/// Secrets Manager
pub struct SecretsManager {
    vault_path: PathBuf,
    cipher: Aes256Gcm,
}

impl SecretsManager {
    /// Create new secrets manager
    pub fn new(vault_path: PathBuf, encryption_key: &[u8; 32]) -> Self {
        let cipher = Aes256Gcm::new(encryption_key.into());
        
        Self {
            vault_path,
            cipher,
        }
    }
    
    /// Initialize vault
    pub fn init(&self) -> Result<(), String> {
        if self.vault_path.exists() {
            return Err("Vault already exists".to_string());
        }
        
        let vault = Vault {
            secrets: HashMap::new(),
        };
        
        self.save_vault(&vault)?;
        Ok(())
    }
    
    /// Set secret
    pub fn set_secret(
        &self,
        key: &str,
        value: &str,
        description: Option<String>,
    ) -> Result<(), String> {
        let encrypted_value = self.encrypt(value)?;
        
        let secret = Secret {
            key: key.to_string(),
            value: encrypted_value,
            description,
            created_at: chrono::Utc::now().to_rfc3339(),
            last_accessed: None,
        };
        
        let mut vault = self.load_vault()?;
        vault.secrets.insert(key.to_string(), secret);
        self.save_vault(&vault)?;
        
        Ok(())
    }
    
    /// Get secret
    pub fn get_secret(&self, key: &str) -> Result<String, String> {
        let mut vault = self.load_vault()?;
        
        let secret = vault.secrets.get_mut(key)
            .ok_or_else(|| format!("Secret '{}' not found", key))?;
        
        // Update last accessed
        secret.last_accessed = Some(chrono::Utc::now().to_rfc3339());
        let encrypted_value = secret.value.clone();
        self.save_vault(&vault)?;
        
        self.decrypt(&encrypted_value)
    }
    
    /// Delete secret
    pub fn delete_secret(&self, key: &str) -> Result<(), String> {
        let mut vault = self.load_vault()?;
        
        if vault.secrets.remove(key).is_none() {
            return Err(format!("Secret '{}' not found", key));
        }
        
        self.save_vault(&vault)?;
        Ok(())
    }
    
    /// List all secret keys
    pub fn list_secrets(&self) -> Result<Vec<String>, String> {
        let vault = self.load_vault()?;
        Ok(vault.secrets.keys().cloned().collect())
    }
    
    /// Get secret metadata (without decrypting value)
    pub fn get_metadata(&self, key: &str) -> Result<SecretMetadata, String> {
        let vault = self.load_vault()?;
        
        let secret = vault.secrets.get(key)
            .ok_or_else(|| format!("Secret '{}' not found", key))?;
        
        Ok(SecretMetadata {
            key: secret.key.clone(),
            description: secret.description.clone(),
            created_at: secret.created_at.clone(),
            last_accessed: secret.last_accessed.clone(),
        })
    }
    
    /// Encrypt value
    fn encrypt(&self, plaintext: &str) -> Result<String, String> {
        // Generate random nonce (12 bytes for AES-GCM)
        let nonce_bytes: [u8; 12] = rand::random();
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let ciphertext = self.cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| format!("Encryption failed: {}", e))?;
        
        // Combine nonce and ciphertext
        let mut combined = nonce_bytes.to_vec();
        combined.extend_from_slice(&ciphertext);
        
        // Encode as base64
        Ok(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, combined))
    }
    
    /// Decrypt value
    fn decrypt(&self, encrypted: &str) -> Result<String, String> {
        // Decode from base64
        let combined = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, encrypted)
            .map_err(|e| format!("Base64 decode failed: {}", e))?;
        
        if combined.len() < 12 {
            return Err("Invalid encrypted data".to_string());
        }
        
        // Split nonce and ciphertext
        let (nonce_bytes, ciphertext) = combined.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        let plaintext = self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("Decryption failed: {}", e))?;
        
        String::from_utf8(plaintext)
            .map_err(|e| format!("UTF-8 decode failed: {}", e))
    }
    
    /// Load vault from disk
    fn load_vault(&self) -> Result<Vault, String> {
        if !self.vault_path.exists() {
            return Ok(Vault {
                secrets: HashMap::new(),
            });
        }
        
        let content = fs::read_to_string(&self.vault_path)
            .map_err(|e| format!("Failed to read vault: {}", e))?;
        
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse vault: {}", e))
    }
    
    /// Save vault to disk
    fn save_vault(&self, vault: &Vault) -> Result<(), String> {
        // Ensure directory exists
        if let Some(parent) = self.vault_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create vault directory: {}", e))?;
        }
        
        let json = serde_json::to_string_pretty(vault)
            .map_err(|e| format!("Failed to serialize vault: {}", e))?;
        
        fs::write(&self.vault_path, json)
            .map_err(|e| format!("Failed to write vault: {}", e))?;
        
        Ok(())
    }
}

/// Secret metadata (without sensitive value)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretMetadata {
    pub key: String,
    pub description: Option<String>,
    pub created_at: String,
    pub last_accessed: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    fn create_test_manager() -> (SecretsManager, PathBuf) {
        let temp_dir = tempdir().unwrap();
        let vault_path = temp_dir.path().join("vault.json");
        let key: [u8; 32] = [0u8; 32]; // Test key
        
        let manager = SecretsManager::new(vault_path.clone(), &key);
        (manager, vault_path)
    }
    
    #[test]
    fn test_set_and_get_secret() {
        let (manager, _) = create_test_manager();
        
        manager.set_secret("api_key", "secret123", Some("Test API key".to_string())).unwrap();
        let value = manager.get_secret("api_key").unwrap();
        
        assert_eq!(value, "secret123");
    }
    
    #[test]
    fn test_delete_secret() {
        let (manager, _) = create_test_manager();
        
        manager.set_secret("temp", "value", None).unwrap();
        manager.delete_secret("temp").unwrap();
        
        assert!(manager.get_secret("temp").is_err());
    }
    
    #[test]
    fn test_list_secrets() {
        let (manager, _) = create_test_manager();
        
        manager.set_secret("key1", "val1", None).unwrap();
        manager.set_secret("key2", "val2", None).unwrap();
        
        let keys = manager.list_secrets().unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
    }
    
    #[test]
    fn test_encryption_decryption() {
        let (manager, _) = create_test_manager();
        
        let original = "sensitive data 🔐";
        let encrypted = manager.encrypt(original).unwrap();
        let decrypted = manager.decrypt(&encrypted).unwrap();
        
        assert_ne!(encrypted, original);
        assert_eq!(decrypted, original);
    }
    
    #[test]
    fn test_metadata() {
        let (manager, _) = create_test_manager();
        
        manager.set_secret("test", "value", Some("Description".to_string())).unwrap();
        let metadata = manager.get_metadata("test").unwrap();
        
        assert_eq!(metadata.key, "test");
        assert_eq!(metadata.description, Some("Description".to_string()));
        assert!(!metadata.created_at.is_empty());
    }
}
