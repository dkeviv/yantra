// File: src-tauri/src/architecture/refactoring.rs
// Purpose: Refactoring safety analysis to prevent breaking changes
// Dependencies: gnn/graph, architecture/analyzer
// Last Updated: December 3, 2025
//
// Implements safety checks for code refactoring to ensure "code that never breaks":
// 1. Breaking change detection (API changes, signature changes)
// 2. API compatibility checks (public vs private changes)
// 3. Dependency impact analysis (what breaks if we change this)
// 4. Safe refactoring suggestions
// 5. Rollback recommendations
//
// Performance targets:
// - Breaking change detection: <100ms
// - Dependency impact analysis: <500ms
// - Full safety check: <1s

use crate::gnn::{CodeGraph, CodeNode, NodeType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    /// Function/method signature changed
    SignatureChange,
    /// Function/method removed
    Removal,
    /// Function/method renamed
    Rename,
    /// Return type changed
    ReturnTypeChange,
    /// Access modifier changed (public to private)
    VisibilityChange,
    /// New function/method added
    Addition,
    /// Implementation changed (safe if signature same)
    ImplementationOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Safe,       // No breaking changes
    Low,        // Internal changes only
    Medium,     // Public API changes with backward compatibility
    High,       // Breaking changes to public API
    Critical,   // Breaks many dependents
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringChange {
    pub node_id: String,
    pub node_name: String,
    pub node_type: NodeType,
    pub change_type: ChangeType,
    pub old_signature: Option<String>,
    pub new_signature: Option<String>,
    pub is_public: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingChangeAnalysis {
    pub has_breaking_changes: bool,
    pub risk_level: RiskLevel,
    pub changes: Vec<RefactoringChange>,
    pub affected_files: Vec<String>,
    pub affected_count: usize,
    pub safe_to_proceed: bool,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyImpact {
    pub changed_node: String,
    pub direct_dependents: Vec<String>,
    pub indirect_dependents: Vec<String>,
    pub total_affected: usize,
    pub critical_dependents: Vec<String>,
}

/// Refactoring safety analyzer
pub struct RefactoringSafetyAnalyzer {
    graph: CodeGraph,
}

impl RefactoringSafetyAnalyzer {
    pub fn new(graph: CodeGraph) -> Self {
        Self { graph }
    }

    /// Analyze safety of proposed refactoring changes
    /// 
    /// # Arguments
    /// * `changes` - List of proposed changes
    /// 
    /// # Returns
    /// BreakingChangeAnalysis with risk assessment and recommendations
    pub fn analyze_changes(&self, changes: &[RefactoringChange]) -> BreakingChangeAnalysis {
        let mut has_breaking_changes = false;
        let mut affected_files = HashSet::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        let mut total_affected = 0;

        // Analyze each change
        for change in changes {
            // Check if this is a breaking change
            let is_breaking = self.is_breaking_change(change);
            
            if is_breaking {
                has_breaking_changes = true;
                warnings.push(format!(
                    "Breaking change detected: {} {} ({})",
                    change.node_type.to_string(),
                    change.node_name,
                    change.change_type.to_string()
                ));
            }

            // Analyze dependency impact
            let impact = self.analyze_dependency_impact(&change.node_id);
            total_affected += impact.total_affected;
            
            for file in impact.direct_dependents.iter().chain(impact.indirect_dependents.iter()) {
                affected_files.insert(file.clone());
            }

            // Generate suggestions
            if is_breaking && !impact.direct_dependents.is_empty() {
                suggestions.push(format!(
                    "Consider deprecating '{}' instead of removing/changing it",
                    change.node_name
                ));
                suggestions.push(format!(
                    "Add compatibility shim for '{}' to maintain backward compatibility",
                    change.node_name
                ));
            }

            if matches!(change.change_type, ChangeType::Rename) {
                suggestions.push(format!(
                    "Keep old name '{}' as alias pointing to new implementation",
                    change.old_signature.as_ref().unwrap_or(&change.node_name)
                ));
            }
        }

        // Calculate risk level
        let risk_level = self.calculate_risk_level(
            has_breaking_changes,
            total_affected,
            changes,
        );

        // Determine if safe to proceed
        let safe_to_proceed = matches!(risk_level, RiskLevel::Safe | RiskLevel::Low)
            || (matches!(risk_level, RiskLevel::Medium) && total_affected < 5);

        BreakingChangeAnalysis {
            has_breaking_changes,
            risk_level,
            changes: changes.to_vec(),
            affected_files: affected_files.into_iter().collect(),
            affected_count: total_affected,
            safe_to_proceed,
            warnings,
            suggestions,
        }
    }

    /// Check if a change is breaking
    fn is_breaking_change(&self, change: &RefactoringChange) -> bool {
        // Breaking changes:
        // 1. Remove public function/class
        // 2. Change signature of public function/method
        // 3. Change return type
        // 4. Make public API private
        // 5. Rename public function/class without alias

        if !change.is_public {
            // Changes to private APIs are generally safe
            return false;
        }

        matches!(
            change.change_type,
            ChangeType::Removal
                | ChangeType::SignatureChange
                | ChangeType::ReturnTypeChange
                | ChangeType::VisibilityChange
                | ChangeType::Rename
        )
    }

    /// Analyze dependency impact of a change
    pub fn analyze_dependency_impact(&self, node_id: &str) -> DependencyImpact {
        // Get direct dependents (who imports/calls this)
        let direct_dependents = self.graph.get_dependents(node_id);
        
        // Get indirect dependents (transitive dependencies)
        let mut indirect_dependents = HashSet::new();
        let mut visited = HashSet::new();
        
        for dependent in &direct_dependents {
            self.collect_transitive_dependents(
                &dependent.id,
                &mut indirect_dependents,
                &mut visited,
            );
        }

        // Remove direct dependents from indirect set
        for dependent in &direct_dependents {
            indirect_dependents.remove(&dependent.id);
        }

        // Identify critical dependents (entry points, main functions, exported APIs)
        let critical_dependents = self.identify_critical_dependents(&direct_dependents);

        let total_affected = direct_dependents.len() + indirect_dependents.len();

        DependencyImpact {
            changed_node: node_id.to_string(),
            direct_dependents: direct_dependents.iter().map(|n| n.id.clone()).collect(),
            indirect_dependents: indirect_dependents.into_iter().collect(),
            total_affected,
            critical_dependents,
        }
    }

    /// Collect transitive dependents recursively
    fn collect_transitive_dependents(
        &self,
        node_id: &str,
        indirect: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) {
        if visited.contains(node_id) {
            return; // Avoid cycles
        }
        visited.insert(node_id.to_string());

        let dependents = self.graph.get_dependents(node_id);
        for dependent in dependents {
            indirect.insert(dependent.id.clone());
            self.collect_transitive_dependents(&dependent.id, indirect, visited);
        }
    }

    /// Identify critical dependents (entry points, main functions, exported APIs)
    fn identify_critical_dependents(&self, dependents: &[CodeNode]) -> Vec<String> {
        dependents
            .iter()
            .filter(|node| {
                // Consider as critical if:
                // - Named "main" or "__main__"
                // - In __init__.py (exported API)
                // - Has "export" or "public" in name/path
                node.name == "main"
                    || node.name == "__main__"
                    || node.file_path.contains("__init__")
                    || node.file_path.contains("index")
                    || node.name.starts_with("export")
            })
            .map(|node| node.id.clone())
            .collect()
    }

    /// Calculate overall risk level
    fn calculate_risk_level(
        &self,
        has_breaking_changes: bool,
        affected_count: usize,
        changes: &[RefactoringChange],
    ) -> RiskLevel {
        if !has_breaking_changes {
            // Check if only implementation changes
            let only_impl = changes
                .iter()
                .all(|c| matches!(c.change_type, ChangeType::ImplementationOnly));
            
            if only_impl {
                return RiskLevel::Safe;
            } else {
                return RiskLevel::Low;
            }
        }

        // Has breaking changes - assess severity
        if affected_count == 0 {
            // Breaking changes but no dependents (unused code)
            return RiskLevel::Low;
        } else if affected_count < 5 {
            // Few dependents
            return RiskLevel::Medium;
        } else if affected_count < 20 {
            // Many dependents
            return RiskLevel::High;
        } else {
            // Critical number of dependents
            return RiskLevel::Critical;
        }
    }

    /// Detect changes between two versions of code
    pub fn detect_changes(
        old_code: &str,
        new_code: &str,
        file_path: &str,
    ) -> Result<Vec<RefactoringChange>, String> {
        // Parse both versions and compare
        // This is a simplified implementation - full version would use tree-sitter AST diff
        let mut changes = Vec::new();

        // Extract function signatures from old code
        let old_functions = Self::extract_function_signatures(old_code);
        let new_functions = Self::extract_function_signatures(new_code);

        // Detect removals
        for (name, signature) in &old_functions {
            if !new_functions.contains_key(name) {
                changes.push(RefactoringChange {
                    node_id: format!("{}::{}", file_path, name),
                    node_name: name.clone(),
                    node_type: NodeType::Function,
                    change_type: ChangeType::Removal,
                    old_signature: Some(signature.clone()),
                    new_signature: None,
                    is_public: !name.starts_with('_'),
                });
            } else if new_functions.get(name) != Some(signature) {
                // Signature changed
                changes.push(RefactoringChange {
                    node_id: format!("{}::{}", file_path, name),
                    node_name: name.clone(),
                    node_type: NodeType::Function,
                    change_type: ChangeType::SignatureChange,
                    old_signature: Some(signature.clone()),
                    new_signature: new_functions.get(name).cloned(),
                    is_public: !name.starts_with('_'),
                });
            }
        }

        // Detect additions
        for (name, signature) in &new_functions {
            if !old_functions.contains_key(name) {
                changes.push(RefactoringChange {
                    node_id: format!("{}::{}", file_path, name),
                    node_name: name.clone(),
                    node_type: NodeType::Function,
                    change_type: ChangeType::Addition,
                    old_signature: None,
                    new_signature: Some(signature.clone()),
                    is_public: !name.starts_with('_'),
                });
            }
        }

        Ok(changes)
    }

    /// Extract function signatures from code (simplified)
    fn extract_function_signatures(code: &str) -> HashMap<String, String> {
        let mut signatures = HashMap::new();

        // Simple regex-based extraction (Python functions)
        // Full implementation would use tree-sitter for accurate parsing
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("def ") && trimmed.contains('(') {
                if let Some(sig_end) = trimmed.find(':') {
                    let signature = trimmed[4..sig_end].trim();
                    if let Some(name_end) = signature.find('(') {
                        let name = signature[..name_end].trim().to_string();
                        signatures.insert(name, signature.to_string());
                    }
                }
            }
        }

        signatures
    }

    /// Generate safe refactoring recommendations
    pub fn recommend_safe_refactoring(
        &self,
        change: &RefactoringChange,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match change.change_type {
            ChangeType::Removal if change.is_public => {
                recommendations.push(format!(
                    "Instead of removing '{}', mark it as deprecated with @deprecated decorator",
                    change.node_name
                ));
                recommendations.push(
                    "Add deprecation warning to guide users to replacement".to_string()
                );
                recommendations.push(
                    "Keep function for 2-3 major versions before removal".to_string()
                );
            }
            ChangeType::SignatureChange if change.is_public => {
                recommendations.push(format!(
                    "Keep old signature of '{}' and add new overload",
                    change.node_name
                ));
                recommendations.push(
                    "Use *args, **kwargs to maintain backward compatibility".to_string()
                );
                recommendations.push(
                    "Create wrapper function with old signature calling new implementation".to_string()
                );
            }
            ChangeType::Rename if change.is_public => {
                recommendations.push(format!(
                    "Keep '{}' as alias to new name",
                    change.old_signature.as_ref().unwrap_or(&change.node_name)
                ));
                recommendations.push(
                    "Add deprecation notice to old name".to_string()
                );
            }
            _ => {
                recommendations.push("Change appears safe to proceed".to_string());
            }
        }

        recommendations
    }
}

impl ChangeType {
    fn to_string(&self) -> &str {
        match self {
            ChangeType::SignatureChange => "signature changed",
            ChangeType::Removal => "removed",
            ChangeType::Rename => "renamed",
            ChangeType::ReturnTypeChange => "return type changed",
            ChangeType::VisibilityChange => "visibility changed",
            ChangeType::Addition => "added",
            ChangeType::ImplementationOnly => "implementation changed",
        }
    }
}

impl NodeType {
    fn to_string(&self) -> &str {
        match self {
            NodeType::Function => "function",
            NodeType::Class => "class",
            NodeType::Variable => "variable",
            NodeType::Module => "module",
            NodeType::Import => "import",
            NodeType::Package { .. } => "package",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_function_removal() {
        let old_code = r#"
def hello():
    return "world"

def goodbye():
    return "farewell"
"#;

        let new_code = r#"
def hello():
    return "world"
"#;

        let changes = RefactoringSafetyAnalyzer::detect_changes(
            old_code,
            new_code,
            "test.py",
        ).unwrap();

        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].change_type, ChangeType::Removal);
        assert_eq!(changes[0].node_name, "goodbye");
    }

    #[test]
    fn test_detect_signature_change() {
        let old_code = r#"
def calculate(x, y):
    return x + y
"#;

        let new_code = r#"
def calculate(x, y, z):
    return x + y + z
"#;

        let changes = RefactoringSafetyAnalyzer::detect_changes(
            old_code,
            new_code,
            "test.py",
        ).unwrap();

        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].change_type, ChangeType::SignatureChange);
        assert_eq!(changes[0].node_name, "calculate");
    }

    #[test]
    fn test_is_breaking_change() {
        let graph = CodeGraph::new();
        let analyzer = RefactoringSafetyAnalyzer::new(graph);

        // Public removal is breaking
        let change = RefactoringChange {
            node_id: "test::func".to_string(),
            node_name: "func".to_string(),
            node_type: NodeType::Function,
            change_type: ChangeType::Removal,
            old_signature: Some("func()".to_string()),
            new_signature: None,
            is_public: true,
        };
        assert!(analyzer.is_breaking_change(&change));

        // Private removal is not breaking
        let change = RefactoringChange {
            node_id: "test::_func".to_string(),
            node_name: "_func".to_string(),
            node_type: NodeType::Function,
            change_type: ChangeType::Removal,
            old_signature: Some("_func()".to_string()),
            new_signature: None,
            is_public: false,
        };
        assert!(!analyzer.is_breaking_change(&change));
    }
}
