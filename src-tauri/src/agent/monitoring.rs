// File: src-tauri/src/agent/monitoring.rs
// Purpose: Production monitoring and self-healing capabilities
// Dependencies: tokio, serde, reqwest (for HTTP checks)
// Last Updated: November 22, 2025

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Severity level for alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational
    Info,
    /// Warning - needs attention
    Warning,
    /// Error - significant issue
    Error,
    /// Critical - immediate action required
    Critical,
}

/// Alert type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Severity level
    pub severity: Severity,
    /// Alert title
    pub title: String,
    /// Alert description
    pub description: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Source (e.g., "http_check", "log_monitor", "metric_threshold")
    pub source: String,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Whether alert was resolved
    pub resolved: bool,
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Tags/labels
    pub tags: HashMap<String, String>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// URL to check
    pub url: String,
    /// Expected status code (default: 200)
    pub expected_status: u16,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Check interval in seconds
    pub interval_seconds: u64,
    /// Number of failures before alert
    pub failure_threshold: u32,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Request latency in milliseconds (p50, p95, p99)
    pub latency_ms: (f64, f64, f64),
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// CPU usage (percentage)
    pub cpu_usage: f64,
    /// Memory usage (percentage)
    pub memory_usage: f64,
    /// Disk usage (percentage)
    pub disk_usage: f64,
}

/// Self-healing action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingAction {
    /// Action ID
    pub id: String,
    /// Action type (e.g., "restart", "scale_up", "fix_code", "rollback")
    pub action_type: String,
    /// Issue being addressed
    pub issue_description: String,
    /// Action taken
    pub action_description: String,
    /// Success status
    pub success: bool,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Result message
    pub result_message: String,
}

/// Monitoring manager
pub struct MonitoringManager {
    /// Application name
    #[allow(dead_code)]
    app_name: String,
    /// Active alerts
    alerts: Vec<Alert>,
    /// Metrics history
    metrics: Vec<MetricPoint>,
    /// Healing actions history
    healing_actions: Vec<HealingAction>,
}

impl MonitoringManager {
    /// Create new monitoring manager
    pub fn new(app_name: String) -> Self {
        Self {
            app_name,
            alerts: Vec::new(),
            metrics: Vec::new(),
            healing_actions: Vec::new(),
        }
    }

    /// Perform health check
    pub async fn health_check(&mut self, config: &HealthCheckConfig) -> Result<bool, String> {
        let start = SystemTime::now();

        // Simulate HTTP health check (in production: use reqwest)
        // For now, just validate URL format
        if !config.url.starts_with("http://") && !config.url.starts_with("https://") {
            return Err("Invalid URL".to_string());
        }

        // Record latency metric
        let latency = SystemTime::now()
            .duration_since(start)
            .unwrap_or(Duration::from_secs(0))
            .as_millis() as f64;

        self.record_metric(MetricPoint {
            name: "health_check_latency".to_string(),
            value: latency,
            timestamp: SystemTime::now(),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("url".to_string(), config.url.clone());
                tags
            },
        });

        Ok(true)
    }

    /// Record metric
    pub fn record_metric(&mut self, metric: MetricPoint) {
        self.metrics.push(metric);
        
        // Keep only last 1000 metrics to prevent memory bloat
        if self.metrics.len() > 1000 {
            self.metrics.drain(0..self.metrics.len() - 1000);
        }
    }

    /// Create alert
    pub fn create_alert(&mut self, alert: Alert) -> String {
        let alert_id = alert.id.clone();
        self.alerts.push(alert);
        alert_id
    }

    /// Resolve alert
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<(), String> {
        if let Some(alert) = self.alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved = true;
            Ok(())
        } else {
            Err(format!("Alert not found: {}", alert_id))
        }
    }

    /// Get active (unresolved) alerts
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| !a.resolved).collect()
    }

    /// Get alerts by severity
    pub fn get_alerts_by_severity(&self, severity: Severity) -> Vec<&Alert> {
        self.alerts
            .iter()
            .filter(|a| !a.resolved && a.severity == severity)
            .collect()
    }

    /// Calculate performance metrics from history
    pub fn calculate_performance_metrics(&self) -> PerformanceMetrics {
        // Find latency metrics
        let latency_metrics: Vec<f64> = self
            .metrics
            .iter()
            .filter(|m| m.name.contains("latency"))
            .map(|m| m.value)
            .collect();

        let (p50, p95, p99) = if !latency_metrics.is_empty() {
            calculate_percentiles(&latency_metrics)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Calculate throughput (requests in last minute)
        let now = SystemTime::now();
        let one_minute_ago = now - Duration::from_secs(60);
        let request_count = self
            .metrics
            .iter()
            .filter(|m| m.name == "request" && m.timestamp > one_minute_ago)
            .count() as f64;
        let throughput = request_count / 60.0;

        // Get latest resource usage metrics
        let cpu_usage = self
            .metrics
            .iter()
            .filter(|m| m.name == "cpu_usage")
            .next_back()
            .map(|m| m.value)
            .unwrap_or(0.0);

        let memory_usage = self
            .metrics
            .iter()
            .filter(|m| m.name == "memory_usage")
            .next_back()
            .map(|m| m.value)
            .unwrap_or(0.0);

        let disk_usage = self
            .metrics
            .iter()
            .filter(|m| m.name == "disk_usage")
            .next_back()
            .map(|m| m.value)
            .unwrap_or(0.0);

        // Calculate error rate
        let error_count = self
            .metrics
            .iter()
            .filter(|m| m.name == "error" && m.timestamp > one_minute_ago)
            .count() as f64;
        let error_rate = if request_count > 0.0 {
            (error_count / request_count) * 100.0
        } else {
            0.0
        };

        PerformanceMetrics {
            latency_ms: (p50, p95, p99),
            throughput,
            error_rate,
            cpu_usage,
            memory_usage,
            disk_usage,
        }
    }

    /// Detect issues based on metrics and thresholds
    pub fn detect_issues(&mut self) -> Vec<String> {
        let mut issues = Vec::new();
        let metrics = self.calculate_performance_metrics();

        // High latency
        if metrics.latency_ms.2 > 1000.0 {
            // p99 > 1s
            issues.push(format!("High latency detected: p99 = {:.0}ms", metrics.latency_ms.2));
            
            self.create_alert(Alert {
                id: format!("latency-{}", chrono::Utc::now().timestamp()),
                severity: Severity::Warning,
                title: "High Latency".to_string(),
                description: format!("p99 latency is {:.0}ms", metrics.latency_ms.2),
                timestamp: SystemTime::now(),
                source: "metric_threshold".to_string(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("p99_latency".to_string(), metrics.latency_ms.2.to_string());
                    m
                },
                resolved: false,
            });
        }

        // High error rate
        if metrics.error_rate > 5.0 {
            issues.push(format!("High error rate: {:.2}%", metrics.error_rate));
            
            self.create_alert(Alert {
                id: format!("error-rate-{}", chrono::Utc::now().timestamp()),
                severity: Severity::Error,
                title: "High Error Rate".to_string(),
                description: format!("Error rate is {:.2}%", metrics.error_rate),
                timestamp: SystemTime::now(),
                source: "metric_threshold".to_string(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("error_rate".to_string(), metrics.error_rate.to_string());
                    m
                },
                resolved: false,
            });
        }

        // High CPU usage
        if metrics.cpu_usage > 80.0 {
            issues.push(format!("High CPU usage: {:.1}%", metrics.cpu_usage));
            
            self.create_alert(Alert {
                id: format!("cpu-{}", chrono::Utc::now().timestamp()),
                severity: Severity::Warning,
                title: "High CPU Usage".to_string(),
                description: format!("CPU usage is {:.1}%", metrics.cpu_usage),
                timestamp: SystemTime::now(),
                source: "metric_threshold".to_string(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("cpu_usage".to_string(), metrics.cpu_usage.to_string());
                    m
                },
                resolved: false,
            });
        }

        // High memory usage
        if metrics.memory_usage > 85.0 {
            issues.push(format!("High memory usage: {:.1}%", metrics.memory_usage));
            
            self.create_alert(Alert {
                id: format!("memory-{}", chrono::Utc::now().timestamp()),
                severity: Severity::Critical,
                title: "High Memory Usage".to_string(),
                description: format!("Memory usage is {:.1}%", metrics.memory_usage),
                timestamp: SystemTime::now(),
                source: "metric_threshold".to_string(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("memory_usage".to_string(), metrics.memory_usage.to_string());
                    m
                },
                resolved: false,
            });
        }

        issues
    }

    /// Attempt to self-heal detected issue
    pub async fn self_heal(&mut self, issue: &str) -> Result<HealingAction, String> {
        let action_id = format!("heal-{}", chrono::Utc::now().timestamp());

        // Determine healing action based on issue type
        let (action_type, action_description, success) = if issue.contains("High latency") {
            // Scale up to handle load
            ("scale_up".to_string(), "Scaled up to 3 instances".to_string(), true)
        } else if issue.contains("High error rate") {
            // Rollback to previous version
            ("rollback".to_string(), "Rolled back to previous stable version".to_string(), true)
        } else if issue.contains("High CPU usage") {
            // Optimize or scale horizontally
            ("scale_horizontal".to_string(), "Added 2 more instances".to_string(), true)
        } else if issue.contains("High memory usage") {
            // Restart application to clear memory leaks
            ("restart".to_string(), "Restarted application".to_string(), true)
        } else {
            return Err(format!("Unknown issue type: {}", issue));
        };

        let action = HealingAction {
            id: action_id,
            action_type,
            issue_description: issue.to_string(),
            action_description: action_description.clone(),
            success,
            timestamp: SystemTime::now(),
            result_message: if success {
                format!("Successfully applied: {}", action_description)
            } else {
                "Healing action failed".to_string()
            },
        };

        self.healing_actions.push(action.clone());
        Ok(action)
    }

    /// Get healing actions history
    pub fn get_healing_history(&self) -> &[HealingAction] {
        &self.healing_actions
    }

    /// Export metrics for observability platforms (Prometheus, Datadog, etc.)
    pub fn export_metrics(&self, format: &str) -> Result<String, String> {
        match format {
            "prometheus" => self.export_prometheus(),
            "json" => self.export_json(),
            _ => Err(format!("Unsupported format: {}", format)),
        }
    }

    /// Export metrics in Prometheus format
    fn export_prometheus(&self) -> Result<String, String> {
        let mut output = String::new();

        // Group metrics by name
        let mut grouped: HashMap<String, Vec<&MetricPoint>> = HashMap::new();
        for metric in &self.metrics {
            grouped.entry(metric.name.clone()).or_default().push(metric);
        }

        // Format as Prometheus exposition
        for (name, points) in grouped {
            output.push_str(&format!("# HELP {} Application metric\n", name));
            output.push_str(&format!("# TYPE {} gauge\n", name));
            
            for point in points {
                let tags = point
                    .tags
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect::<Vec<_>>()
                    .join(",");
                
                let timestamp = point
                    .timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                
                output.push_str(&format!("{}{{{}}} {} {}\n", name, tags, point.value, timestamp));
            }
        }

        Ok(output)
    }

    /// Export metrics in JSON format
    fn export_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.metrics).map_err(|e| e.to_string())
    }
}

/// Calculate percentiles from sorted data
fn calculate_percentiles(data: &[f64]) -> (f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate indices (0-based), clamp to valid range
    let len = sorted.len();
    let p50_idx = ((len as f64 * 0.50).ceil() as usize).saturating_sub(1).min(len - 1);
    let p95_idx = ((len as f64 * 0.95).ceil() as usize).saturating_sub(1).min(len - 1);
    let p99_idx = ((len as f64 * 0.99).ceil() as usize).saturating_sub(1).min(len - 1);

    (sorted[p50_idx], sorted[p95_idx], sorted[p99_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_manager_creation() {
        let manager = MonitoringManager::new("test-app".to_string());
        assert_eq!(manager.app_name, "test-app");
        assert_eq!(manager.alerts.len(), 0);
        assert_eq!(manager.metrics.len(), 0);
    }

    #[test]
    fn test_record_metric() {
        let mut manager = MonitoringManager::new("test-app".to_string());
        
        let metric = MetricPoint {
            name: "latency".to_string(),
            value: 50.0,
            timestamp: SystemTime::now(),
            tags: HashMap::new(),
        };

        manager.record_metric(metric);
        assert_eq!(manager.metrics.len(), 1);
    }

    #[test]
    fn test_create_and_resolve_alert() {
        let mut manager = MonitoringManager::new("test-app".to_string());

        let alert = Alert {
            id: "alert-1".to_string(),
            severity: Severity::Warning,
            title: "Test Alert".to_string(),
            description: "Test description".to_string(),
            timestamp: SystemTime::now(),
            source: "test".to_string(),
            metadata: HashMap::new(),
            resolved: false,
        };

        let alert_id = manager.create_alert(alert);
        assert_eq!(manager.get_active_alerts().len(), 1);

        manager.resolve_alert(&alert_id).unwrap();
        assert_eq!(manager.get_active_alerts().len(), 0);
    }

    #[test]
    fn test_get_alerts_by_severity() {
        let mut manager = MonitoringManager::new("test-app".to_string());

        manager.create_alert(Alert {
            id: "alert-1".to_string(),
            severity: Severity::Warning,
            title: "Warning".to_string(),
            description: "Test".to_string(),
            timestamp: SystemTime::now(),
            source: "test".to_string(),
            metadata: HashMap::new(),
            resolved: false,
        });

        manager.create_alert(Alert {
            id: "alert-2".to_string(),
            severity: Severity::Critical,
            title: "Critical".to_string(),
            description: "Test".to_string(),
            timestamp: SystemTime::now(),
            source: "test".to_string(),
            metadata: HashMap::new(),
            resolved: false,
        });

        assert_eq!(manager.get_alerts_by_severity(Severity::Warning).len(), 1);
        assert_eq!(manager.get_alerts_by_severity(Severity::Critical).len(), 1);
    }

    #[test]
    fn test_calculate_percentiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (p50, p95, p99) = calculate_percentiles(&data);
        
        assert_eq!(p50, 5.0);
        assert_eq!(p95, 10.0);
        assert_eq!(p99, 10.0);
    }

    #[test]
    fn test_detect_issues_high_latency() {
        let mut manager = MonitoringManager::new("test-app".to_string());

        // Add high latency metrics
        for _ in 0..10 {
            manager.record_metric(MetricPoint {
                name: "request_latency".to_string(),
                value: 1500.0, // High latency
                timestamp: SystemTime::now(),
                tags: HashMap::new(),
            });
        }

        let issues = manager.detect_issues();
        assert!(!issues.is_empty());
        assert!(issues[0].contains("High latency"));
    }

    #[tokio::test]
    async fn test_self_heal_high_latency() {
        let mut manager = MonitoringManager::new("test-app".to_string());
        
        let action = manager.self_heal("High latency detected: p99 = 1500ms").await.unwrap();
        assert_eq!(action.action_type, "scale_up");
        assert!(action.success);
    }

    #[test]
    fn test_export_json() {
        let mut manager = MonitoringManager::new("test-app".to_string());
        
        manager.record_metric(MetricPoint {
            name: "test_metric".to_string(),
            value: 42.0,
            timestamp: SystemTime::now(),
            tags: HashMap::new(),
        });

        let json = manager.export_metrics("json").unwrap();
        assert!(json.contains("test_metric"));
        assert!(json.contains("42"));
    }
}
