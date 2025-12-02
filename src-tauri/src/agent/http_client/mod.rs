// Intelligent HTTP Client
// Provides HTTP client with circuit breaker, retry logic, rate limiting, and mock support

use reqwest::{Client, Method, Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower::Service;
use governor::{Quota, RateLimiter};
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};

/// HTTP request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequestConfig {
    pub url: String,
    pub method: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub retry_attempts: Option<u32>,
    pub follow_redirects: Option<bool>,
}

/// HTTP response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
    pub duration_ms: u64,
    pub attempts: u32,
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed,          // Normal operation
    Open,            // Failures exceeded threshold, rejecting requests
    HalfOpen,        // Testing if service recovered
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
struct CircuitStats {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<std::time::Instant>,
    opened_at: Option<std::time::Instant>,
}

/// Mock response for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockResponse {
    pub url_pattern: String,
    pub status_code: u16,
    pub body: String,
    pub headers: HashMap<String, String>,
    pub delay_ms: Option<u64>,
}

/// Intelligent HTTP Client
pub struct IntelligentHttpClient {
    client: Client,
    circuit_breaker: Arc<RwLock<HashMap<String, CircuitStats>>>,
    rate_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
    mock_responses: Arc<RwLock<Vec<MockResponse>>>,
    request_logs: Arc<RwLock<Vec<HttpRequestLog>>>,
}

/// Request log for tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HttpRequestLog {
    timestamp: chrono::DateTime<chrono::Utc>,
    method: String,
    url: String,
    status_code: u16,
    duration_ms: u64,
    attempts: u32,
}

impl IntelligentHttpClient {
    /// Create new HTTP client
    pub fn new() -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .cookie_store(true)
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        // Rate limiter: 100 requests per second
        let quota = Quota::per_second(std::num::NonZeroU32::new(100).unwrap());
        let rate_limiter = RateLimiter::direct(quota);

        Ok(Self {
            client,
            circuit_breaker: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(rate_limiter),
            mock_responses: Arc::new(RwLock::new(Vec::new())),
            request_logs: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Execute HTTP request with intelligence
    pub async fn request(&self, config: HttpRequestConfig) -> Result<HttpResponse, String> {
        let start = std::time::Instant::now();
        let max_attempts = config.retry_attempts.unwrap_or(3);

        // Check for mock response first
        if let Some(mock) = self.get_mock_response(&config.url).await {
            return self.handle_mock_response(mock, &config).await;
        }

        // Rate limiting
        self.rate_limiter.until_ready().await;

        // Circuit breaker check
        let host = self.extract_host(&config.url)?;
        if self.is_circuit_open(&host).await {
            return Err(format!("Circuit breaker open for host: {}", host));
        }

        let mut last_error = String::new();
        for attempt in 1..=max_attempts {
            match self.execute_request(&config).await {
                Ok(response) => {
                    // Record success
                    self.record_success(&host).await;
                    
                    let duration_ms = start.elapsed().as_millis() as u64;
                    
                    // Log request
                    self.log_request(&config, response.status().as_u16(), duration_ms, attempt).await;
                    
                    return self.parse_response(response, attempt, duration_ms).await;
                }
                Err(e) => {
                    last_error = e.clone();
                    
                    // Record failure
                    self.record_failure(&host).await;
                    
                    // Exponential backoff before retry
                    if attempt < max_attempts {
                        let backoff_ms = 100 * 2_u64.pow(attempt - 1);
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }

        Err(format!("Request failed after {} attempts: {}", max_attempts, last_error))
    }

    /// Execute HTTP request
    async fn execute_request(&self, config: &HttpRequestConfig) -> Result<Response, String> {
        let method = match config.method.to_uppercase().as_str() {
            "GET" => Method::GET,
            "POST" => Method::POST,
            "PUT" => Method::PUT,
            "DELETE" => Method::DELETE,
            "PATCH" => Method::PATCH,
            "HEAD" => Method::HEAD,
            "OPTIONS" => Method::OPTIONS,
            _ => return Err(format!("Unsupported HTTP method: {}", config.method)),
        };

        let mut request = self.client.request(method, &config.url);

        // Add headers
        for (key, value) in &config.headers {
            request = request.header(key, value);
        }

        // Add body
        if let Some(body) = &config.body {
            request = request.body(body.clone());
        }

        // Set timeout
        if let Some(timeout) = config.timeout_seconds {
            request = request.timeout(Duration::from_secs(timeout));
        }

        request
            .send()
            .await
            .map_err(|e| format!("Request execution failed: {}", e))
    }

    /// Parse response
    async fn parse_response(
        &self,
        response: Response,
        attempts: u32,
        duration_ms: u64,
    ) -> Result<HttpResponse, String> {
        let status_code = response.status().as_u16();
        
        let mut headers = HashMap::new();
        for (key, value) in response.headers() {
            headers.insert(
                key.to_string(),
                value.to_str().unwrap_or("").to_string(),
            );
        }

        let body = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response body: {}", e))?;

        Ok(HttpResponse {
            status_code,
            headers,
            body,
            duration_ms,
            attempts,
        })
    }

    /// GET request
    pub async fn get(&self, url: &str) -> Result<HttpResponse, String> {
        self.request(HttpRequestConfig {
            url: url.to_string(),
            method: "GET".to_string(),
            headers: HashMap::new(),
            body: None,
            timeout_seconds: None,
            retry_attempts: None,
            follow_redirects: Some(true),
        })
        .await
    }

    /// POST request
    pub async fn post(&self, url: &str, body: String) -> Result<HttpResponse, String> {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        self.request(HttpRequestConfig {
            url: url.to_string(),
            method: "POST".to_string(),
            headers,
            body: Some(body),
            timeout_seconds: None,
            retry_attempts: None,
            follow_redirects: Some(true),
        })
        .await
    }

    /// PUT request
    pub async fn put(&self, url: &str, body: String) -> Result<HttpResponse, String> {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        self.request(HttpRequestConfig {
            url: url.to_string(),
            method: "PUT".to_string(),
            headers,
            body: Some(body),
            timeout_seconds: None,
            retry_attempts: None,
            follow_redirects: Some(true),
        })
        .await
    }

    /// DELETE request
    pub async fn delete(&self, url: &str) -> Result<HttpResponse, String> {
        self.request(HttpRequestConfig {
            url: url.to_string(),
            method: "DELETE".to_string(),
            headers: HashMap::new(),
            body: None,
            timeout_seconds: None,
            retry_attempts: None,
            follow_redirects: Some(true),
        })
        .await
    }

    /// Circuit breaker: check if open
    async fn is_circuit_open(&self, host: &str) -> bool {
        let breakers = self.circuit_breaker.read().await;
        if let Some(stats) = breakers.get(host) {
            if stats.state == CircuitState::Open {
                // Check if should try half-open
                if let Some(opened_at) = stats.opened_at {
                    if opened_at.elapsed() > Duration::from_secs(30) {
                        // Try half-open after 30 seconds
                        drop(breakers);
                        let mut breakers = self.circuit_breaker.write().await;
                        if let Some(stats) = breakers.get_mut(host) {
                            stats.state = CircuitState::HalfOpen;
                        }
                        return false;
                    }
                }
                return true;
            }
        }
        false
    }

    /// Record successful request
    async fn record_success(&self, host: &str) {
        let mut breakers = self.circuit_breaker.write().await;
        let stats = breakers.entry(host.to_string()).or_insert(CircuitStats {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            opened_at: None,
        });

        stats.success_count += 1;
        
        // Reset circuit if in half-open and request succeeded
        if stats.state == CircuitState::HalfOpen {
            stats.state = CircuitState::Closed;
            stats.failure_count = 0;
        }
    }

    /// Record failed request
    async fn record_failure(&self, host: &str) {
        let mut breakers = self.circuit_breaker.write().await;
        let stats = breakers.entry(host.to_string()).or_insert(CircuitStats {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            opened_at: None,
        });

        stats.failure_count += 1;
        stats.last_failure_time = Some(std::time::Instant::now());

        // Open circuit if threshold exceeded (5 failures)
        if stats.failure_count >= 5 {
            stats.state = CircuitState::Open;
            stats.opened_at = Some(std::time::Instant::now());
        }
    }

    /// Extract host from URL
    fn extract_host(&self, url: &str) -> Result<String, String> {
        let parsed = url::Url::parse(url)
            .map_err(|e| format!("Invalid URL: {}", e))?;
        parsed.host_str()
            .map(|h| h.to_string())
            .ok_or_else(|| "No host in URL".to_string())
    }

    /// Add mock response for testing
    pub async fn add_mock(&self, mock: MockResponse) {
        let mut mocks = self.mock_responses.write().await;
        mocks.push(mock);
    }

    /// Get mock response if exists
    async fn get_mock_response(&self, url: &str) -> Option<MockResponse> {
        let mocks = self.mock_responses.read().await;
        mocks.iter().find(|m| url.contains(&m.url_pattern)).cloned()
    }

    /// Handle mock response
    async fn handle_mock_response(
        &self,
        mock: MockResponse,
        config: &HttpRequestConfig,
    ) -> Result<HttpResponse, String> {
        let start = std::time::Instant::now();

        // Simulate delay if specified
        if let Some(delay_ms) = mock.delay_ms {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(HttpResponse {
            status_code: mock.status_code,
            headers: mock.headers,
            body: mock.body,
            duration_ms,
            attempts: 1,
        })
    }

    /// Log request
    async fn log_request(&self, config: &HttpRequestConfig, status_code: u16, duration_ms: u64, attempts: u32) {
        let log = HttpRequestLog {
            timestamp: chrono::Utc::now(),
            method: config.method.clone(),
            url: config.url.clone(),
            status_code,
            duration_ms,
            attempts,
        };

        let mut logs = self.request_logs.write().await;
        logs.push(log);
    }

    /// Get request logs
    pub async fn get_logs(&self) -> Vec<HttpRequestLog> {
        let logs = self.request_logs.read().await;
        logs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = IntelligentHttpClient::new();
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_mock_response() {
        let client = IntelligentHttpClient::new().unwrap();
        
        client.add_mock(MockResponse {
            url_pattern: "example.com".to_string(),
            status_code: 200,
            body: "Mock response".to_string(),
            headers: HashMap::new(),
            delay_ms: Some(10),
        }).await;

        let response = client.get("https://example.com/api/test").await.unwrap();
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body, "Mock response");
    }

    #[test]
    fn test_extract_host() {
        let client = IntelligentHttpClient::new().unwrap();
        let host = client.extract_host("https://api.example.com:8080/path");
        assert_eq!(host.unwrap(), "api.example.com");
    }
}
