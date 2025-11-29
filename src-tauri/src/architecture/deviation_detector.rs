/**
 * Deviation Detector - Monitors code-architecture alignment
 * 
 * Purpose: Ensure generated/edited code aligns with planned architecture
 * 
 * Key Functions:
 * 1. monitor_code_generation() - Check BEFORE writing files (proactive)
 * 2. check_code_alignment() - Check AFTER file save (reactive)
 * 
 * Integrates with:
 * - GNN Engine (analyze dependencies)
 * - Architecture Manager (get expected dependencies)
 * - LLM Orchestrator (generate fix suggestions)
 * - Project Orchestrator (pause/resume code generation)
 */

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use crate::gnn::GNNEngine;
use crate::architecture::ArchitectureManager;

// ============================================================================
// TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationCheck {
    pub has_deviation: bool,
    pub violations: Vec<Violation>,
    pub severity: Severity,
    pub pause_generation: bool,
    pub user_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub is_aligned: bool,
    pub deviations: Vec<Deviation>,
    pub severity: Severity,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deviation {
    pub deviation_type: DeviationType,
    pub expected: String,
    pub actual: String,
    pub affected_file: PathBuf,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub import_path: String,
    pub reason: String,
    pub allowed_alternatives: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    None,       // No deviation
    Low,        // Minor deviation (e.g., extra utility import)
    Medium,     // Moderate (e.g., skip one layer but maintain pattern)
    High,       // Major violation (e.g., break layering completely)
    Critical,   // Catastrophic (e.g., circular dependencies)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviationType {
    UnexpectedDependency,   // Code imports something not in architecture
    MissingDependency,      // Architecture expects import, code doesn't have it
    WrongConnectionType,    // Using wrong communication pattern
    LayerViolation,         // Bypassing layers (e.g., Gateway → DB directly)
    CircularDependency,     // Creating cycle in directed graph
}

// ============================================================================
// DEVIATION DETECTOR
// ============================================================================

pub struct DeviationDetector {
    gnn_engine: Arc<Mutex<GNNEngine>>,
    architecture_manager: ArchitectureManager,
}

impl DeviationDetector {
    pub fn new(
        gnn_engine: Arc<Mutex<GNNEngine>>,
        architecture_manager: ArchitectureManager,
    ) -> Self {
        Self {
            gnn_engine,
            architecture_manager,
        }
    }

    /// Monitor code generation BEFORE writing files
    /// 
    /// Called by ProjectOrchestrator before saving generated code.
    /// If deviation detected, pauses generation and prompts user.
    pub async fn monitor_code_generation(
        &self,
        generated_code: &str,
        target_file: &Path,
        architecture_id: &str,
    ) -> Result<DeviationCheck, String> {
        // 1. Parse generated code to extract imports
        let imports = self.extract_imports_from_code(generated_code, target_file)?;
        
        // 2. Get expected dependencies from architecture
        let arch = self.architecture_manager.get_architecture(architecture_id)
            .map_err(|e| format!("Failed to load architecture: {}", e))?;
        
        // 3. Find which component this file belongs to
        let component = arch.components.iter()
            .find(|c| c.files.contains(&target_file.to_string_lossy().to_string()))
            .ok_or_else(|| format!("File {} not found in any component", target_file.display()))?;
        
        // 4. Get allowed dependencies for this component
        let allowed_deps = self.get_allowed_dependencies(&arch, &component.id)?;
        
        // 5. Check for violations
        let mut violations = Vec::new();
        
        for import in &imports {
            if !self.is_import_allowed(import, &allowed_deps) {
                violations.push(Violation {
                    import_path: import.clone(),
                    reason: format!("Component '{}' is not allowed to import '{}'", component.name, import),
                    allowed_alternatives: allowed_deps.clone(),
                });
            }
        }
        
        // 6. Calculate severity
        let severity = self.calculate_severity(&violations, &component.component_type);
        
        // 7. Generate user prompt if violations found
        let user_prompt = if !violations.is_empty() {
            Some(self.generate_deviation_prompt(&violations, component, &arch))
        } else {
            None
        };
        
        Ok(DeviationCheck {
            has_deviation: !violations.is_empty(),
            violations,
            severity,
            pause_generation: severity >= Severity::Medium,
            user_prompt,
        })
    }

    /// Check code alignment AFTER file save
    /// 
    /// Called by file watcher when user manually edits a file.
    /// Shows warning if code breaks architecture.
    pub async fn check_code_alignment(
        &self,
        file_path: &Path,
        architecture_id: &str,
    ) -> Result<AlignmentResult, String> {
        // 1. Get current architecture
        let arch = self.architecture_manager.get_architecture(architecture_id)
            .map_err(|e| format!("Failed to load architecture: {}", e))?;
        
        // 2. Find which component owns this file
        let component = arch.components.iter()
            .find(|c| c.files.contains(&file_path.to_string_lossy().to_string()))
            .ok_or_else(|| format!("File {} not found in any component", file_path.display()))?;
        
        // 3. Get GNN dependencies for this file
        let actual_deps = {
            let gnn = self.gnn_engine.lock().map_err(|e| format!("GNN lock error: {}", e))?;
            gnn.get_file_dependencies(file_path)
                .unwrap_or_else(|_| Vec::new())
        };
        
        // 4. Get expected dependencies from architecture
        let expected_deps = self.get_allowed_dependencies(&arch, &component.id)?;
        
        // 5. Compare actual vs expected
        let mut deviations = Vec::new();
        
        for dep in &actual_deps {
            if !self.is_import_allowed(&dep, &expected_deps) {
                deviations.push(Deviation {
                    deviation_type: DeviationType::UnexpectedDependency,
                    expected: format!("Only dependencies: {}", expected_deps.join(", ")),
                    actual: dep.clone(),
                    affected_file: file_path.to_path_buf(),
                    explanation: format!(
                        "File '{}' imports '{}' but architecture doesn't allow this dependency for component '{}'",
                        file_path.display(),
                        dep,
                        component.name
                    ),
                });
            }
        }
        
        // 6. Calculate severity
        let severity = self.calculate_severity_from_deviations(&deviations);
        
        // 7. Generate recommendations
        let recommendations = self.generate_recommendations(&deviations, component);
        
        Ok(AlignmentResult {
            is_aligned: deviations.is_empty(),
            deviations,
            severity,
            recommendations,
        })
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /// Extract imports from generated code (Python, JavaScript, TypeScript)
    fn extract_imports_from_code(&self, code: &str, file_path: &Path) -> Result<Vec<String>, String> {
        let extension = file_path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| "No file extension".to_string())?;
        
        match extension {
            "py" => self.extract_python_imports(code),
            "js" | "jsx" | "ts" | "tsx" => self.extract_js_imports(code),
            _ => Ok(Vec::new()),
        }
    }

    /// Extract Python imports (simple regex-based)
    fn extract_python_imports(&self, code: &str) -> Result<Vec<String>, String> {
        let mut imports = Vec::new();
        
        for line in code.lines() {
            let trimmed = line.trim();
            
            // import module
            if trimmed.starts_with("import ") {
                if let Some(module) = trimmed.strip_prefix("import ") {
                    let module = module.split_whitespace().next().unwrap_or("");
                    imports.push(module.to_string());
                }
            }
            
            // from module import ...
            if trimmed.starts_with("from ") {
                if let Some(rest) = trimmed.strip_prefix("from ") {
                    if let Some(module) = rest.split_whitespace().next() {
                        imports.push(module.to_string());
                    }
                }
            }
        }
        
        Ok(imports)
    }

    /// Extract JavaScript/TypeScript imports
    fn extract_js_imports(&self, code: &str) -> Result<Vec<String>, String> {
        let mut imports = Vec::new();
        
        for line in code.lines() {
            let trimmed = line.trim();
            
            // import ... from 'module'
            if trimmed.starts_with("import ") && trimmed.contains(" from ") {
                if let Some(from_part) = trimmed.split(" from ").nth(1) {
                    let module = from_part.trim_matches(&['\'', '"', ';', ' '][..]);
                    imports.push(module.to_string());
                }
            }
            
            // const ... = require('module')
            if trimmed.contains("require(") {
                if let Some(start) = trimmed.find("require(") {
                    if let Some(module_start) = trimmed[start..].find(&['\'', '"'][..]) {
                        if let Some(module_end) = trimmed[start + module_start + 1..].find(&['\'', '"'][..]) {
                            let module = &trimmed[start + module_start + 1..start + module_start + 1 + module_end];
                            imports.push(module.to_string());
                        }
                    }
                }
            }
        }
        
        Ok(imports)
    }

    /// Get allowed dependencies for a component from architecture
    fn get_allowed_dependencies(
        &self,
        arch: &crate::architecture::types::Architecture,
        component_id: &str,
    ) -> Result<Vec<String>, String> {
        let connections = arch.connections.iter()
            .filter(|c| c.source_id == component_id)
            .map(|c| {
                // Get target component name
                arch.components.iter()
                    .find(|comp| comp.id == c.target_id)
                    .map(|comp| comp.name.clone())
                    .unwrap_or_else(|| c.target_id.clone())
            })
            .collect();
        
        Ok(connections)
    }

    /// Check if an import is allowed based on allowed dependencies
    fn is_import_allowed(&self, import: &str, allowed_deps: &[String]) -> bool {
        // Allow standard library imports
        if self.is_standard_library(import) {
            return true;
        }
        
        // Check if import matches any allowed dependency
        allowed_deps.iter().any(|dep| {
            import.starts_with(dep) || 
            import.contains(dep) ||
            dep.to_lowercase() == import.to_lowercase()
        })
    }

    /// Check if import is from standard library
    fn is_standard_library(&self, import: &str) -> bool {
        // Python standard library
        let py_stdlib = ["os", "sys", "json", "time", "datetime", "re", "math", "random", "collections"];
        
        // JavaScript/Node standard library
        let js_stdlib = ["fs", "path", "http", "https", "crypto", "util", "events"];
        
        py_stdlib.contains(&import) || js_stdlib.contains(&import)
    }

    /// Calculate severity from violations
    fn calculate_severity(&self, violations: &[Violation], component_type: &str) -> Severity {
        if violations.is_empty() {
            return Severity::None;
        }
        
        // Critical: More than 3 violations or circular dependency detected
        if violations.len() > 3 {
            return Severity::Critical;
        }
        
        // High: Service layer bypass (e.g., Gateway → Database)
        if component_type.contains("Gateway") || component_type.contains("API") {
            for v in violations {
                if v.import_path.contains("database") || v.import_path.contains("db") {
                    return Severity::High;
                }
            }
        }
        
        // Medium: 2-3 violations
        if violations.len() >= 2 {
            return Severity::Medium;
        }
        
        // Low: Single minor violation
        Severity::Low
    }

    /// Calculate severity from deviations
    fn calculate_severity_from_deviations(&self, deviations: &[Deviation]) -> Severity {
        if deviations.is_empty() {
            return Severity::None;
        }
        
        // Check for critical deviations
        for dev in deviations {
            if matches!(dev.deviation_type, DeviationType::CircularDependency) {
                return Severity::Critical;
            }
            if matches!(dev.deviation_type, DeviationType::LayerViolation) {
                return Severity::High;
            }
        }
        
        // Medium: 2+ deviations
        if deviations.len() >= 2 {
            return Severity::Medium;
        }
        
        Severity::Low
    }

    /// Generate user-facing prompt for deviation
    fn generate_deviation_prompt(
        &self,
        violations: &[Violation],
        component: &crate::architecture::types::Component,
        _arch: &crate::architecture::types::Architecture,
    ) -> String {
        let mut prompt = format!(
            "⚠️ ARCHITECTURE DEVIATION DETECTED\n\n\
             I was about to generate code for component '{}', but it would violate \
             the architecture.\n\n",
            component.name
        );
        
        prompt.push_str("Violations:\n");
        for (i, v) in violations.iter().enumerate() {
            prompt.push_str(&format!(
                "{}. Attempted import: '{}'\n   Reason: {}\n\n",
                i + 1,
                v.import_path,
                v.reason
            ));
        }
        
        prompt.push_str("\nWhat would you like me to do?\n\n");
        prompt.push_str("1️⃣  Update Architecture - Allow these dependencies (changes architectural design)\n");
        prompt.push_str("2️⃣  Fix Code - Regenerate code that matches architecture (maintains design)\n");
        prompt.push_str("3️⃣  Cancel - Stop code generation for review\n\n");
        prompt.push_str("Recommended: Option 2 (maintain clean architecture)\n\n");
        prompt.push_str("Your choice (1/2/3): ");
        
        prompt
    }

    /// Generate recommendations for fixing deviations
    fn generate_recommendations(
        &self,
        deviations: &[Deviation],
        component: &crate::architecture::types::Component,
    ) -> Vec<String> {
        let mut recs = Vec::new();
        
        if deviations.is_empty() {
            recs.push("✅ Code aligns perfectly with architecture".to_string());
            return recs;
        }
        
        recs.push(format!(
            "Component '{}' has {} architectural deviation(s)",
            component.name,
            deviations.len()
        ));
        
        for dev in deviations {
            match dev.deviation_type {
                DeviationType::UnexpectedDependency => {
                    recs.push(format!(
                        "Remove import '{}' or update architecture to allow it",
                        dev.actual
                    ));
                },
                DeviationType::LayerViolation => {
                    recs.push("Refactor code to respect layer boundaries".to_string());
                },
                _ => {
                    recs.push(format!("Fix: {}", dev.explanation));
                },
            }
        }
        
        recs.push("\nRecommended action: Revert code changes and follow architecture".to_string());
        
        recs
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_python_imports() {
        let detector = create_test_detector();
        let code = r#"
import os
import sys
from database import query
from auth.service import verify_token
        "#;
        
        let imports = detector.extract_python_imports(code).unwrap();
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"database".to_string()));
        assert!(imports.contains(&"auth.service".to_string()));
    }

    #[test]
    fn test_severity_calculation() {
        let detector = create_test_detector();
        
        // No violations = None
        assert_eq!(detector.calculate_severity(&[], "Service"), Severity::None);
        
        // 1 violation = Low
        let violations = vec![Violation {
            import_path: "utils".to_string(),
            reason: "Minor".to_string(),
            allowed_alternatives: vec![],
        }];
        assert_eq!(detector.calculate_severity(&violations, "Service"), Severity::Low);
        
        // 4 violations = Critical
        let many_violations = vec![
            Violation { import_path: "a".to_string(), reason: "".to_string(), allowed_alternatives: vec![] },
            Violation { import_path: "b".to_string(), reason: "".to_string(), allowed_alternatives: vec![] },
            Violation { import_path: "c".to_string(), reason: "".to_string(), allowed_alternatives: vec![] },
            Violation { import_path: "d".to_string(), reason: "".to_string(), allowed_alternatives: vec![] },
        ];
        assert_eq!(detector.calculate_severity(&many_violations, "Service"), Severity::Critical);
    }

    fn create_test_detector() -> DeviationDetector {
        use std::path::PathBuf;
        let gnn = Arc::new(Mutex::new(GNNEngine::new(PathBuf::from("/tmp")).unwrap()));
        let arch_manager = ArchitectureManager::default();
        DeviationDetector::new(gnn, arch_manager)
    }
}
