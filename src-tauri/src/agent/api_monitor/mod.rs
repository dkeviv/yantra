// API Contract Monitor - OpenAPI/Swagger spec validation and breaking change detection
// Tracks API contracts, validates requests/responses, detects breaking changes

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};

pub mod contract_validator;
pub mod spec_parser;

use contract_validator::ContractValidator;
use spec_parser::SpecParser;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiContract {
    pub name: String,
    pub base_url: String,
    pub version: String,
    pub spec_format: SpecFormat,
    pub spec_content: String,
    pub endpoints: HashMap<String, Endpoint>,
    pub rate_limit: Option<RateLimit>,
    pub last_validated: DateTime<Utc>,
    pub health_check_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecFormat {
    OpenApiV3,
    OpenApiV2,
    Swagger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub path: String,
    pub method: String,
    pub request_schema: Option<serde_json::Value>,
    pub response_schema: Option<serde_json::Value>,
    pub required_params: Vec<String>,
    pub optional_params: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub requests_per_hour: Option<u32>,
    pub requests_per_day: Option<u32>,
    pub current_usage: u32,
    pub reset_time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub expected: String,
    pub actual: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub response_time_ms: u64,
    pub status_code: u16,
    pub last_check: DateTime<Utc>,
    pub consecutive_failures: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingChange {
    pub change_type: BreakingChangeType,
    pub endpoint: String,
    pub description: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakingChangeType {
    EndpointRemoved,
    RequiredParamAdded,
    ParamTypeChanged,
    ResponseSchemaChanged,
    StatusCodeChanged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct ApiMonitor {
    contracts: Arc<Mutex<HashMap<String, ApiContract>>>,
    validator: ContractValidator,
    parser: SpecParser,
    http_client: reqwest::Client,
}

impl ApiMonitor {
    pub fn new() -> Self {
        Self {
            contracts: Arc::new(Mutex::new(HashMap::new())),
            validator: ContractValidator::new(),
            parser: SpecParser::new(),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Import and register an API contract from OpenAPI/Swagger spec
    pub async fn import_api_spec(&self, name: String, spec_url_or_path: String) -> Result<String, String> {
        // Determine if it's a URL or file path
        let spec_content = if spec_url_or_path.starts_with("http://") || spec_url_or_path.starts_with("https://") {
            // Fetch from URL
            self.http_client
                .get(&spec_url_or_path)
                .send()
                .await
                .map_err(|e| format!("Failed to fetch spec: {}", e))?
                .text()
                .await
                .map_err(|e| format!("Failed to read spec: {}", e))?
        } else {
            // Read from file
            tokio::fs::read_to_string(&spec_url_or_path)
                .await
                .map_err(|e| format!("Failed to read spec file: {}", e))?
        };

        // Parse the spec
        let contract = self.parser.parse(&name, &spec_content).await?;

        // Store the contract
        let contract_id = format!("{}_{}", name, contract.version);
        self.contracts.lock().unwrap().insert(contract_id.clone(), contract);

        Ok(contract_id)
    }

    /// Validate a request against the API contract
    pub async fn validate_request(
        &self,
        contract_id: &str,
        endpoint: &str,
        method: &str,
        params: Option<&serde_json::Value>,
        body: Option<&serde_json::Value>,
    ) -> Result<ValidationResult, String> {
        let contracts = self.contracts.lock().unwrap();
        let contract = contracts
            .get(contract_id)
            .ok_or_else(|| format!("Contract not found: {}", contract_id))?;

        self.validator.validate_request(contract, endpoint, method, params, body)
    }

    /// Validate a response against the API contract
    pub async fn validate_response(
        &self,
        contract_id: &str,
        endpoint: &str,
        method: &str,
        status_code: u16,
        body: &serde_json::Value,
    ) -> Result<ValidationResult, String> {
        let contracts = self.contracts.lock().unwrap();
        let contract = contracts
            .get(contract_id)
            .ok_or_else(|| format!("Contract not found: {}", contract_id))?;

        self.validator.validate_response(contract, endpoint, method, status_code, body)
    }

    /// Detect breaking changes between two API versions
    pub async fn detect_breaking_changes(
        &self,
        old_contract_id: &str,
        new_contract_id: &str,
    ) -> Result<Vec<BreakingChange>, String> {
        let contracts = self.contracts.lock().unwrap();
        
        let old_contract = contracts
            .get(old_contract_id)
            .ok_or_else(|| format!("Old contract not found: {}", old_contract_id))?;
        
        let new_contract = contracts
            .get(new_contract_id)
            .ok_or_else(|| format!("New contract not found: {}", new_contract_id))?;

        self.validator.detect_breaking_changes(old_contract, new_contract)
    }

    /// Perform health check on API endpoint
    pub async fn health_check(&self, contract_id: &str) -> Result<HealthStatus, String> {
        let contracts = self.contracts.lock().unwrap();
        let contract = contracts
            .get(contract_id)
            .ok_or_else(|| format!("Contract not found: {}", contract_id))?;

        let health_url = contract.health_check_url.as_ref()
            .unwrap_or(&format!("{}/health", contract.base_url));

        let start = std::time::Instant::now();
        let result = self.http_client.get(health_url).send().await;
        let duration = start.elapsed().as_millis() as u64;

        match result {
            Ok(response) => {
                let status_code = response.status().as_u16();
                let healthy = status_code >= 200 && status_code < 300;

                Ok(HealthStatus {
                    healthy,
                    response_time_ms: duration,
                    status_code,
                    last_check: Utc::now(),
                    consecutive_failures: if healthy { 0 } else { 1 },
                })
            }
            Err(e) => {
                Ok(HealthStatus {
                    healthy: false,
                    response_time_ms: duration,
                    status_code: 0,
                    last_check: Utc::now(),
                    consecutive_failures: 1,
                })
            }
        }
    }

    /// Check and update rate limit status
    pub async fn check_rate_limit(&self, contract_id: &str) -> Result<RateLimitStatus, String> {
        let mut contracts = self.contracts.lock().unwrap();
        let contract = contracts
            .get_mut(contract_id)
            .ok_or_else(|| format!("Contract not found: {}", contract_id))?;

        if let Some(rate_limit) = &mut contract.rate_limit {
            // Reset counter if time has passed
            if Utc::now() >= rate_limit.reset_time {
                rate_limit.current_usage = 0;
                rate_limit.reset_time = Utc::now() + chrono::Duration::minutes(1);
            }

            let remaining = rate_limit.requests_per_minute.saturating_sub(rate_limit.current_usage);
            let at_limit = remaining == 0;

            Ok(RateLimitStatus {
                limit: rate_limit.requests_per_minute,
                remaining,
                reset_time: rate_limit.reset_time,
                at_limit,
            })
        } else {
            Ok(RateLimitStatus {
                limit: u32::MAX,
                remaining: u32::MAX,
                reset_time: Utc::now() + chrono::Duration::hours(24),
                at_limit: false,
            })
        }
    }

    /// Increment rate limit counter
    pub async fn increment_rate_limit(&self, contract_id: &str) -> Result<(), String> {
        let mut contracts = self.contracts.lock().unwrap();
        let contract = contracts
            .get_mut(contract_id)
            .ok_or_else(|| format!("Contract not found: {}", contract_id))?;

        if let Some(rate_limit) = &mut contract.rate_limit {
            rate_limit.current_usage += 1;
        }

        Ok(())
    }

    /// List all registered API contracts
    pub fn list_contracts(&self) -> Vec<String> {
        self.contracts.lock().unwrap().keys().cloned().collect()
    }

    /// Get contract details
    pub fn get_contract(&self, contract_id: &str) -> Option<ApiContract> {
        self.contracts.lock().unwrap().get(contract_id).cloned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub limit: u32,
    pub remaining: u32,
    pub reset_time: DateTime<Utc>,
    pub at_limit: bool,
}

impl Default for ApiMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_monitor_creation() {
        let monitor = ApiMonitor::new();
        assert_eq!(monitor.list_contracts().len(), 0);
    }

    #[tokio::test]
    async fn test_rate_limit_tracking() {
        let monitor = ApiMonitor::new();
        
        // Create a test contract with rate limit
        let mut contract = ApiContract {
            name: "test_api".to_string(),
            base_url: "https://api.example.com".to_string(),
            version: "1.0.0".to_string(),
            spec_format: SpecFormat::OpenApiV3,
            spec_content: String::new(),
            endpoints: HashMap::new(),
            rate_limit: Some(RateLimit {
                requests_per_minute: 60,
                requests_per_hour: None,
                requests_per_day: None,
                current_usage: 0,
                reset_time: Utc::now() + chrono::Duration::minutes(1),
            }),
            last_validated: Utc::now(),
            health_check_url: None,
        };

        let contract_id = "test_api_1.0.0".to_string();
        monitor.contracts.lock().unwrap().insert(contract_id.clone(), contract);

        // Check initial rate limit
        let status = monitor.check_rate_limit(&contract_id).await.unwrap();
        assert_eq!(status.remaining, 60);
        assert!(!status.at_limit);

        // Increment usage
        for _ in 0..60 {
            monitor.increment_rate_limit(&contract_id).await.unwrap();
        }

        // Check at limit
        let status = monitor.check_rate_limit(&contract_id).await.unwrap();
        assert_eq!(status.remaining, 0);
        assert!(status.at_limit);
    }
}
