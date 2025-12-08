// Affected Tests Runner: Smart test selection based on code changes
// Purpose: Run only tests affected by code changes using GNN dependency tracking

use crate::gnn::GNNEngine;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Test impact analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestImpactAnalysis {
    pub changed_files: Vec<String>,
    pub affected_tests: Vec<String>,
    pub impact_graph: HashMap<String, Vec<String>>,
    pub analysis_time_ms: u64,
}

/// Test filtering strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterStrategy {
    /// Only direct dependencies
    Direct,
    /// Direct + transitive dependencies (1 level)
    Transitive,
    /// Full dependency tree
    Full,
}

/// Affected Tests Runner
pub struct AffectedTestsRunner {
    gnn: GNNEngine,
    test_pattern: String,
}

impl AffectedTestsRunner {
    /// Create new runner
    pub fn new(gnn: GNNEngine, test_pattern: String) -> Self {
        Self { gnn, test_pattern }
    }
    
    /// Find affected tests for changed files
    pub fn find_affected_tests(
        &self,
        changed_files: &[PathBuf],
        strategy: FilterStrategy,
    ) -> Result<TestImpactAnalysis, String> {
        let start = std::time::Instant::now();
        
        let changed_file_strings: Vec<String> = changed_files
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        
        // Find all test files
        let test_files = self.find_test_files()?;
        
        // Build dependency graph
        let mut affected_tests = HashSet::new();
        let mut impact_graph = HashMap::new();
        
        for changed_file in &changed_file_strings {
            let mut file_affected_tests = Vec::new();
            
            // Find tests that depend on this file
            for test_file in &test_files {
                if self.is_test_affected(changed_file, test_file, &strategy)? {
                    affected_tests.insert(test_file.clone());
                    file_affected_tests.push(test_file.clone());
                }
            }
            
            impact_graph.insert(changed_file.clone(), file_affected_tests);
        }
        
        let analysis_time_ms = start.elapsed().as_millis() as u64;
        
        Ok(TestImpactAnalysis {
            changed_files: changed_file_strings,
            affected_tests: affected_tests.into_iter().collect(),
            impact_graph,
            analysis_time_ms,
        })
    }
    
    /// Check if test is affected by changed file
    fn is_test_affected(
        &self,
        changed_file: &str,
        test_file: &str,
        strategy: &FilterStrategy,
    ) -> Result<bool, String> {
        // Get dependencies of test file
        let test_deps = self.gnn.get_dependencies(test_file);
        
        match strategy {
            FilterStrategy::Direct => {
                // Check if changed file is a direct dependency
                Ok(test_deps.iter().any(|node| node.file_path == changed_file))
            }
            FilterStrategy::Transitive => {
                // Check direct dependencies and their dependencies (1 level)
                if test_deps.iter().any(|node| node.file_path == changed_file) {
                    return Ok(true);
                }
                
                for dep in &test_deps {
                    let dep_deps = self.gnn.get_dependencies(&dep.id);
                    
                    if dep_deps.iter().any(|node| node.file_path == changed_file) {
                        return Ok(true);
                    }
                }
                
                Ok(false)
            }
            FilterStrategy::Full => {
                // Full transitive closure
                let mut visited = HashSet::new();
                let mut to_visit = vec![test_file.to_string()];
                
                while let Some(current) = to_visit.pop() {
                    if visited.contains(&current) {
                        continue;
                    }
                    visited.insert(current.clone());
                    
                    if current == changed_file {
                        return Ok(true);
                    }
                    
                    let deps = self.gnn.get_dependencies(&current);
                    to_visit.extend(deps.iter().map(|node| node.id.clone()));
                }
                
                Ok(false)
            }
        }
    }
    
    /// Find all test files in project
    fn find_test_files(&self) -> Result<Vec<String>, String> {
        // Get all files from GNN
        let all_files = self.gnn.list_all_files()
            .map_err(|e| format!("Failed to list files: {}", e))?;
        
        // Filter test files based on pattern
        let test_files: Vec<String> = all_files
            .into_iter()
            .filter(|f| self.is_test_file(f))
            .collect();
        
        Ok(test_files)
    }
    
    /// Check if file is a test file
    fn is_test_file(&self, file_path: &str) -> bool {
        let path = Path::new(file_path);
        
        // Check file name patterns
        if let Some(file_name) = path.file_name() {
            let name = file_name.to_string_lossy();
            
            // Python test patterns
            if name.starts_with("test_") && name.ends_with(".py") {
                return true;
            }
            
            // JavaScript test patterns
            if (name.ends_with(".test.js") || name.ends_with(".spec.js") ||
                name.ends_with(".test.ts") || name.ends_with(".spec.ts")) {
                return true;
            }
            
            // Custom pattern matching
            if !self.test_pattern.is_empty() {
                if let Ok(regex) = regex::Regex::new(&self.test_pattern) {
                    if regex.is_match(&name) {
                        return true;
                    }
                }
            }
        }
        
        // Check directory patterns
        if let Some(parent) = path.parent() {
            let parent_str = parent.to_string_lossy();
            if parent_str.contains("/tests/") || parent_str.contains("/test/") ||
               parent_str.contains("/__tests__/") {
                return true;
            }
        }
        
        false
    }
    
    /// Generate test command for affected tests
    pub fn generate_test_command(
        &self,
        affected_tests: &[String],
        language: &str,
    ) -> String {
        match language {
            "python" => {
                if affected_tests.is_empty() {
                    "pytest".to_string()
                } else {
                    format!("pytest {}", affected_tests.join(" "))
                }
            }
            "javascript" | "typescript" => {
                if affected_tests.is_empty() {
                    "npm test".to_string()
                } else {
                    let test_names: Vec<String> = affected_tests
                        .iter()
                        .map(|t| {
                            Path::new(t)
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string()
                        })
                        .collect();
                    format!("npm test -- {}", test_names.join(" "))
                }
            }
            _ => {
                // Generic command
                format!("run tests: {}", affected_tests.join(" "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_is_test_file_python() {
        let runner = AffectedTestsRunner {
            gnn: GNNEngine::new(&PathBuf::from("test")).expect("Failed to create GNN"),
            test_pattern: String::new(),
        };
        
        assert!(runner.is_test_file("tests/test_example.py"));
        assert!(runner.is_test_file("src/tests/test_foo.py"));
        assert!(!runner.is_test_file("src/example.py"));
    }
    
    #[test]
    fn test_is_test_file_javascript() {
        let runner = AffectedTestsRunner {
            gnn: GNNEngine::new(&PathBuf::from("test")).expect("Failed to create GNN"),
            test_pattern: String::new(),
        };
        
        assert!(runner.is_test_file("src/Component.test.js"));
        assert!(runner.is_test_file("src/utils.spec.ts"));
        assert!(runner.is_test_file("__tests__/app.js"));
        assert!(!runner.is_test_file("src/Component.js"));
    }
    
    #[test]
    fn test_generate_test_command_python() {
        let runner = AffectedTestsRunner {
            gnn: GNNEngine::new(&PathBuf::from("test")).expect("Failed to create GNN"),
            test_pattern: String::new(),
        };
        
        let tests = vec!["tests/test_a.py".to_string(), "tests/test_b.py".to_string()];
        let cmd = runner.generate_test_command(&tests, "python");
        
        assert!(cmd.contains("pytest"));
        assert!(cmd.contains("test_a.py"));
        assert!(cmd.contains("test_b.py"));
    }
    
    #[test]
    fn test_impact_analysis_structure() {
        let analysis = TestImpactAnalysis {
            changed_files: vec!["src/main.py".to_string()],
            affected_tests: vec!["tests/test_main.py".to_string()],
            impact_graph: HashMap::new(),
            analysis_time_ms: 10,
        };
        
        assert_eq!(analysis.changed_files.len(), 1);
        assert_eq!(analysis.affected_tests.len(), 1);
        assert_eq!(analysis.analysis_time_ms, 10);
    }
}
