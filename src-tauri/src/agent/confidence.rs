// File: src-tauri/src/agent/confidence.rs

// Module types not yet fully integrated
#![allow(dead_code)]
// Purpose: Confidence scoring system for auto-retry decisions
// Last Updated: November 21, 2025

use serde::{Deserialize, Serialize};

/// Individual confidence factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceFactor {
    /// Factor name
    pub name: String,
    /// Score (0.0-1.0)
    pub score: f32,
    /// Weight in overall calculation
    pub weight: f32,
}

/// Overall confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// LLM's own confidence (from response metadata)
    pub llm_confidence: f32,
    /// Test pass rate (% of tests passing)
    pub test_pass_rate: f32,
    /// Known failure pattern match similarity (0.0-1.0)
    pub known_failure_match: f32,
    /// Code complexity (0.0=complex, 1.0=simple)
    pub code_complexity: f32,
    /// Dependency impact (0.0=many changes, 1.0=few changes)
    pub dependency_impact: f32,
}

impl ConfidenceScore {
    /// Create new confidence score with default values
    pub fn new() -> Self {
        Self {
            llm_confidence: 1.0,
            test_pass_rate: 1.0,
            known_failure_match: 0.0,
            code_complexity: 1.0,
            dependency_impact: 1.0,
        }
    }

    /// Create from individual factors
    pub fn from_factors(
        llm_confidence: f32,
        test_pass_rate: f32,
        known_failure_match: f32,
        code_complexity: f32,
        dependency_impact: f32,
    ) -> Self {
        Self {
            llm_confidence: llm_confidence.clamp(0.0, 1.0),
            test_pass_rate: test_pass_rate.clamp(0.0, 1.0),
            known_failure_match: known_failure_match.clamp(0.0, 1.0),
            code_complexity: code_complexity.clamp(0.0, 1.0),
            dependency_impact: dependency_impact.clamp(0.0, 1.0),
        }
    }

    /// Calculate overall confidence score (weighted average)
    /// 
    /// Weights:
    /// - LLM confidence: 30%
    /// - Test pass rate: 25%
    /// - Known failure match: 25%
    /// - Code complexity: 10%
    /// - Dependency impact: 10%
    pub fn overall(&self) -> f32 {
        let score = self.llm_confidence * 0.30
            + self.test_pass_rate * 0.25
            + self.known_failure_match * 0.25
            + (1.0 - self.code_complexity) * 0.10  // Inverse: simpler is better
            + (1.0 - self.dependency_impact) * 0.10; // Inverse: fewer changes is better

        score.clamp(0.0, 1.0)
    }

    /// Check if should auto-retry
    /// 
    /// Returns true if overall confidence >= 0.5
    pub fn should_auto_retry(&self) -> bool {
        self.overall() >= 0.5
    }

    /// Check if should escalate to human
    /// 
    /// Returns true if overall confidence < 0.5
    pub fn should_escalate(&self) -> bool {
        self.overall() < 0.5
    }

    /// Get confidence level as string
    pub fn level(&self) -> &'static str {
        let score = self.overall();
        if score >= 0.8 {
            "High"
        } else if score >= 0.5 {
            "Medium"
        } else {
            "Low"
        }
    }

    /// Get individual factors with their contributions
    pub fn factors(&self) -> Vec<ConfidenceFactor> {
        vec![
            ConfidenceFactor {
                name: "LLM Confidence".to_string(),
                score: self.llm_confidence,
                weight: 0.30,
            },
            ConfidenceFactor {
                name: "Test Pass Rate".to_string(),
                score: self.test_pass_rate,
                weight: 0.25,
            },
            ConfidenceFactor {
                name: "Known Failure Match".to_string(),
                score: self.known_failure_match,
                weight: 0.25,
            },
            ConfidenceFactor {
                name: "Code Complexity".to_string(),
                score: 1.0 - self.code_complexity, // Inverted for display
                weight: 0.10,
            },
            ConfidenceFactor {
                name: "Dependency Impact".to_string(),
                score: 1.0 - self.dependency_impact, // Inverted for display
                weight: 0.10,
            },
        ]
    }

    /// Update LLM confidence from response metadata
    pub fn set_llm_confidence(&mut self, confidence: f32) {
        self.llm_confidence = confidence.clamp(0.0, 1.0);
    }

    /// Update test pass rate
    pub fn set_test_pass_rate(&mut self, passed: usize, total: usize) {
        self.test_pass_rate = if total > 0 {
            (passed as f32 / total as f32).clamp(0.0, 1.0)
        } else {
            1.0 // No tests = perfect pass rate (for now)
        };
    }

    /// Update known failure match score
    pub fn set_known_failure_match(&mut self, similarity: f32) {
        self.known_failure_match = similarity.clamp(0.0, 1.0);
    }

    /// Update code complexity score
    /// 
    /// Lower cyclomatic complexity = higher score
    pub fn set_code_complexity(&mut self, cyclomatic_complexity: usize) {
        // Normalize complexity (1-10 scale, inverted)
        // 1 = simple (score 1.0), 10+ = very complex (score 0.0)
        let normalized = (10.0 - cyclomatic_complexity.min(10) as f32) / 10.0;
        self.code_complexity = normalized.clamp(0.0, 1.0);
    }

    /// Update dependency impact score
    /// 
    /// Fewer files affected = higher score
    pub fn set_dependency_impact(&mut self, files_affected: usize) {
        // Normalize impact (1-20 files scale, inverted)
        // 1 file = minimal impact (score 1.0), 20+ files = major impact (score 0.0)
        let normalized = (20.0 - files_affected.min(20) as f32) / 20.0;
        self.dependency_impact = normalized.clamp(0.0, 1.0);
    }
}

impl Default for ConfidenceScore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_score_creation() {
        let score = ConfidenceScore::new();
        // Default: LLM=1.0 (0.30), Tests=1.0 (0.25), KnownFailures=0.0 (0.00),
        //          Complexity=1.0 (inverted=0.00), Deps=1.0 (inverted=0.00)
        // Total: 0.30 + 0.25 + 0.00 + 0.00 + 0.00 = 0.55
        assert_eq!(score.overall(), 0.55);
        assert_eq!(score.level(), "Medium");
        assert!(score.should_auto_retry());
        assert!(!score.should_escalate());
    }

    #[test]
    fn test_confidence_from_factors() {
        let score = ConfidenceScore::from_factors(0.9, 0.8, 0.7, 0.5, 0.6);
        
        // Weighted: 0.9*0.3 + 0.8*0.25 + 0.7*0.25 + 0.5*0.1 + 0.4*0.1
        //         = 0.27 + 0.2 + 0.175 + 0.05 + 0.04 = 0.735
        let overall = score.overall();
        assert!((overall - 0.735).abs() < 0.01, "Expected ~0.735, got {}", overall);
    }

    #[test]
    fn test_high_confidence() {
        let score = ConfidenceScore::from_factors(0.9, 1.0, 0.9, 0.0, 0.0);
        assert_eq!(score.level(), "High");
        assert!(score.should_auto_retry());
        assert!(!score.should_escalate());
    }

    #[test]
    fn test_medium_confidence() {
        let score = ConfidenceScore::from_factors(0.7, 0.6, 0.5, 0.5, 0.5);
        assert_eq!(score.level(), "Medium");
        assert!(score.should_auto_retry());
        assert!(!score.should_escalate());
    }

    #[test]
    fn test_low_confidence() {
        let score = ConfidenceScore::from_factors(0.3, 0.2, 0.1, 0.8, 0.9);
        assert_eq!(score.level(), "Low");
        assert!(!score.should_auto_retry());
        assert!(score.should_escalate());
    }

    #[test]
    fn test_test_pass_rate_update() {
        let mut score = ConfidenceScore::new();
        
        score.set_test_pass_rate(8, 10); // 80% pass rate
        assert_eq!(score.test_pass_rate, 0.8);

        score.set_test_pass_rate(0, 10); // 0% pass rate
        assert_eq!(score.test_pass_rate, 0.0);

        score.set_test_pass_rate(0, 0); // No tests
        assert_eq!(score.test_pass_rate, 1.0);
    }

    #[test]
    fn test_code_complexity_update() {
        let mut score = ConfidenceScore::new();
        
        score.set_code_complexity(1); // Very simple
        assert!((score.code_complexity - 0.9).abs() < 0.01);

        score.set_code_complexity(5); // Moderate
        assert!((score.code_complexity - 0.5).abs() < 0.01);

        score.set_code_complexity(10); // Very complex
        assert!((score.code_complexity - 0.0).abs() < 0.01);

        score.set_code_complexity(20); // Beyond scale
        assert!((score.code_complexity - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_dependency_impact_update() {
        let mut score = ConfidenceScore::new();
        
        score.set_dependency_impact(1); // Minimal impact
        assert!((score.dependency_impact - 0.95).abs() < 0.01);

        score.set_dependency_impact(10); // Moderate impact
        assert!((score.dependency_impact - 0.5).abs() < 0.01);

        score.set_dependency_impact(20); // Major impact
        assert!((score.dependency_impact - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_factors_list() {
        let score = ConfidenceScore::from_factors(0.9, 0.8, 0.7, 0.5, 0.6);
        let factors = score.factors();
        
        assert_eq!(factors.len(), 5);
        assert_eq!(factors[0].name, "LLM Confidence");
        assert_eq!(factors[0].score, 0.9);
        assert_eq!(factors[0].weight, 0.30);
    }

    #[test]
    fn test_clamping() {
        let score = ConfidenceScore::from_factors(1.5, -0.5, 2.0, -1.0, 3.0);
        
        // All values should be clamped to [0.0, 1.0]
        assert_eq!(score.llm_confidence, 1.0);
        assert_eq!(score.test_pass_rate, 0.0);
        assert_eq!(score.known_failure_match, 1.0);
        assert_eq!(score.code_complexity, 0.0);
        assert_eq!(score.dependency_impact, 1.0);
    }

    #[test]
    fn test_retry_threshold() {
        // Exactly 0.5 should auto-retry
        let score = ConfidenceScore::from_factors(0.5, 0.5, 0.5, 0.5, 0.5);
        assert!(score.should_auto_retry());
        assert!(!score.should_escalate());

        // Just below 0.5 should escalate
        let score = ConfidenceScore::from_factors(0.4, 0.4, 0.4, 0.6, 0.6);
        assert!(!score.should_auto_retry());
        assert!(score.should_escalate());
    }
}
