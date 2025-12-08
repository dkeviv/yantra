// API Health Monitor & Rate Limit Tracker
// Purpose: Monitor API availability and track rate limits

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Down,
    Unknown,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub endpoint: String,
    pub status: HealthStatus,
    pub response_time_ms: u64,
    pub status_code: Option<u16>,
    pub timestamp: String,
    pub error: Option<String>,
}

/// Rate limit info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub endpoint: String,
    pub limit: u32,
    pub remaining: u32,
    pub reset_at: String,
    pub window_seconds: u32,
}

/// Rate limit tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitTracker {
    limits: HashMap<String, RateLimitInfo>,
    #[serde(skip)]
    request_counts: HashMap<String, Vec<Instant>>,
}

impl RateLimitTracker {
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
            request_counts: HashMap::new(),
        }
    }
    
    /// Record API request
    pub fn record_request(&mut self, endpoint: &str) {
        let now = Instant::now();
        self.request_counts
            .entry(endpoint.to_string())
            .or_insert_with(Vec::new)
            .push(now);
        
        // Cleanup old requests (older than 1 hour)
        if let Some(requests) = self.request_counts.get_mut(endpoint) {
            requests.retain(|t| now.duration_since(*t) < Duration::from_secs(3600));
        }
    }
    
    /// Update rate limit from response headers
    pub fn update_from_headers(
        &mut self,
        endpoint: &str,
        limit: u32,
        remaining: u32,
        reset_timestamp: i64,
        window_seconds: u32,
    ) {
        self.limits.insert(
            endpoint.to_string(),
            RateLimitInfo {
                endpoint: endpoint.to_string(),
                limit,
                remaining,
                reset_at: chrono::DateTime::from_timestamp(reset_timestamp, 0)
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_default(),
                window_seconds,
            },
        );
    }
    
    /// Get rate limit info
    pub fn get_limit(&self, endpoint: &str) -> Option<&RateLimitInfo> {
        self.limits.get(endpoint)
    }
    
    /// Check if approaching rate limit
    pub fn is_approaching_limit(&self, endpoint: &str, threshold: f32) -> bool {
        if let Some(limit) = self.limits.get(endpoint) {
            let usage_pct = 1.0 - (limit.remaining as f32 / limit.limit as f32);
            usage_pct >= threshold
        } else {
            false
        }
    }
    
    /// Get request count in time window
    pub fn get_request_count(&self, endpoint: &str, window_seconds: u64) -> usize {
        if let Some(requests) = self.request_counts.get(endpoint) {
            let now = Instant::now();
            requests
                .iter()
                .filter(|t| now.duration_since(**t) < Duration::from_secs(window_seconds))
                .count()
        } else {
            0
        }
    }
}

impl Default for RateLimitTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// API Health Monitor
pub struct HealthMonitor {
    history: HashMap<String, Vec<HealthCheckResult>>,
    max_history: usize,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            max_history: 100,
        }
    }
    
    /// Perform health check
    pub async fn check_health(&mut self, endpoint: &str) -> HealthCheckResult {
        let start = Instant::now();
        
        // Perform HTTP request
        let result = match reqwest::get(endpoint).await {
            Ok(response) => {
                let status_code = response.status().as_u16();
                let status = if status_code >= 200 && status_code < 300 {
                    HealthStatus::Healthy
                } else if status_code >= 500 {
                    HealthStatus::Down
                } else {
                    HealthStatus::Degraded
                };
                
                HealthCheckResult {
                    endpoint: endpoint.to_string(),
                    status,
                    response_time_ms: start.elapsed().as_millis() as u64,
                    status_code: Some(status_code),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    error: None,
                }
            }
            Err(e) => HealthCheckResult {
                endpoint: endpoint.to_string(),
                status: HealthStatus::Down,
                response_time_ms: start.elapsed().as_millis() as u64,
                status_code: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
                error: Some(e.to_string()),
            },
        };
        
        // Store in history
        self.history
            .entry(endpoint.to_string())
            .or_insert_with(Vec::new)
            .push(result.clone());
        
        // Trim history
        if let Some(history) = self.history.get_mut(endpoint) {
            if history.len() > self.max_history {
                history.drain(0..history.len() - self.max_history);
            }
        }
        
        result
    }
    
    /// Get health history
    pub fn get_history(&self, endpoint: &str) -> Option<&Vec<HealthCheckResult>> {
        self.history.get(endpoint)
    }
    
    /// Get uptime percentage
    pub fn get_uptime(&self, endpoint: &str) -> f32 {
        if let Some(history) = self.history.get(endpoint) {
            if history.is_empty() {
                return 0.0;
            }
            
            let healthy_count = history
                .iter()
                .filter(|r| r.status == HealthStatus::Healthy)
                .count();
            
            (healthy_count as f32 / history.len() as f32) * 100.0
        } else {
            0.0
        }
    }
    
    /// Get average response time
    pub fn get_avg_response_time(&self, endpoint: &str) -> Option<u64> {
        if let Some(history) = self.history.get(endpoint) {
            if history.is_empty() {
                return None;
            }
            
            let total: u64 = history.iter().map(|r| r.response_time_ms).sum();
            Some(total / history.len() as u64)
        } else {
            None
        }
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rate_limit_tracking() {
        let mut tracker = RateLimitTracker::new();
        
        tracker.update_from_headers("/api/v1/users", 100, 50, 1234567890, 3600);
        
        let limit = tracker.get_limit("/api/v1/users").unwrap();
        assert_eq!(limit.limit, 100);
        assert_eq!(limit.remaining, 50);
        assert!(tracker.is_approaching_limit("/api/v1/users", 0.4));
    }
    
    #[test]
    fn test_request_counting() {
        let mut tracker = RateLimitTracker::new();
        
        tracker.record_request("/api/test");
        tracker.record_request("/api/test");
        tracker.record_request("/api/test");
        
        let count = tracker.get_request_count("/api/test", 60);
        assert_eq!(count, 3);
    }
    
    #[test]
    fn test_health_history() {
        let mut monitor = HealthMonitor::new();
        
        // Manually add results
        monitor.history.insert(
            "test".to_string(),
            vec![
                HealthCheckResult {
                    endpoint: "test".to_string(),
                    status: HealthStatus::Healthy,
                    response_time_ms: 100,
                    status_code: Some(200),
                    timestamp: "2025-12-03T00:00:00Z".to_string(),
                    error: None,
                },
                HealthCheckResult {
                    endpoint: "test".to_string(),
                    status: HealthStatus::Down,
                    response_time_ms: 5000,
                    status_code: None,
                    timestamp: "2025-12-03T00:01:00Z".to_string(),
                    error: Some("Timeout".to_string()),
                },
            ],
        );
        
        let uptime = monitor.get_uptime("test");
        assert_eq!(uptime, 50.0);
        
        let avg_time = monitor.get_avg_response_time("test").unwrap();
        assert_eq!(avg_time, 2550);
    }
}
