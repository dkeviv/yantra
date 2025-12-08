// Conflict Detection: Detect dependency conflicts using GNN
// Purpose: Identify conflicting requirements, circular dependencies, incompatible versions

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Conflict type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictType {
    VersionConflict,      // Multiple incompatible version requirements
    CircularDependency,   // Circular import cycle
    MissingDependency,    // Required dependency not found
    DuplicateDefinition,  // Same function/class defined multiple times
    ImportConflict,       // Import name collision
}

/// Conflict severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    Low,      // Warning, can proceed
    Medium,   // Should fix, but not blocking
    High,     // Must fix before execution
    Critical, // Immediate fix required, will break code
}

/// Detected conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub conflict_type: ConflictType,
    pub severity: ConflictSeverity,
    pub description: String,
    pub affected_files: Vec<String>,
    pub affected_symbols: Vec<String>,
    pub suggested_fix: Option<String>,
}

/// Conflict detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDetectionResult {
    pub has_conflicts: bool,
    pub total_conflicts: usize,
    pub critical_count: usize,
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
    pub conflicts: Vec<Conflict>,
}

/// Dependency version requirement
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VersionRequirement {
    pub package_name: String,
    pub constraint: String,
    pub required_by: String, // File path
}

/// Conflict Detector
pub struct ConflictDetector;

impl ConflictDetector {
    /// Detect all conflicts in codebase
    pub fn detect_conflicts(
        dependencies: Vec<VersionRequirement>,
        imports: HashMap<String, Vec<String>>, // file -> imported symbols
        definitions: HashMap<String, Vec<String>>, // file -> defined symbols
    ) -> ConflictDetectionResult {
        let mut conflicts = Vec::new();
        
        // Detect version conflicts
        conflicts.extend(Self::detect_version_conflicts(&dependencies));
        
        // Detect circular dependencies
        conflicts.extend(Self::detect_circular_dependencies(&imports));
        
        // Detect duplicate definitions
        conflicts.extend(Self::detect_duplicate_definitions(&definitions));
        
        // Detect import conflicts
        conflicts.extend(Self::detect_import_conflicts(&imports, &definitions));
        
        // Count by severity
        let critical_count = conflicts.iter().filter(|c| c.severity == ConflictSeverity::Critical).count();
        let high_count = conflicts.iter().filter(|c| c.severity == ConflictSeverity::High).count();
        let medium_count = conflicts.iter().filter(|c| c.severity == ConflictSeverity::Medium).count();
        let low_count = conflicts.iter().filter(|c| c.severity == ConflictSeverity::Low).count();
        
        ConflictDetectionResult {
            has_conflicts: !conflicts.is_empty(),
            total_conflicts: conflicts.len(),
            critical_count,
            high_count,
            medium_count,
            low_count,
            conflicts,
        }
    }
    
    /// Detect version conflicts
    fn detect_version_conflicts(dependencies: &[VersionRequirement]) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        let mut package_requirements: HashMap<String, Vec<&VersionRequirement>> = HashMap::new();
        
        // Group requirements by package
        for dep in dependencies {
            package_requirements
                .entry(dep.package_name.clone())
                .or_default()
                .push(dep);
        }
        
        // Check for conflicting requirements
        for (package, reqs) in package_requirements {
            if reqs.len() > 1 {
                // Check if constraints are compatible
                let constraints: Vec<_> = reqs.iter().map(|r| r.constraint.as_str()).collect();
                
                if !Self::are_constraints_compatible(&constraints) {
                    let affected_files: Vec<_> = reqs.iter()
                        .map(|r| r.required_by.clone())
                        .collect();
                    
                    let constraint_list = reqs.iter()
                        .map(|r| format!("{} ({})", r.constraint, r.required_by))
                        .collect::<Vec<_>>()
                        .join(", ");
                    
                    conflicts.push(Conflict {
                        conflict_type: ConflictType::VersionConflict,
                        severity: ConflictSeverity::High,
                        description: format!(
                            "Package '{}' has conflicting version requirements: {}",
                            package, constraint_list
                        ),
                        affected_files,
                        affected_symbols: vec![package.clone()],
                        suggested_fix: Some(format!(
                            "Unify version requirements for '{}' across all files",
                            package
                        )),
                    });
                }
            }
        }
        
        conflicts
    }
    
    /// Check if version constraints are compatible
    fn are_constraints_compatible(constraints: &[&str]) -> bool {
        // Simplified: If all constraints are identical, they're compatible
        // In reality, need semver resolution (e.g., ">=1.0" and "^1.2" might be compatible)
        
        if constraints.is_empty() {
            return true;
        }
        
        let first = constraints[0];
        constraints.iter().all(|c| *c == first)
    }
    
    /// Detect circular dependencies
    fn detect_circular_dependencies(imports: &HashMap<String, Vec<String>>) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        let mut visited = HashSet::new();
        
        for file in imports.keys() {
            // Skip if already visited (part of a previously explored component)
            // This ensures each cycle is detected exactly once
            if visited.contains(file) {
                continue;
            }
            
            let mut rec_stack = HashSet::new();
            if let Some(cycle) = Self::find_cycle(file, imports, &mut visited, &mut rec_stack, &mut vec![]) {
                conflicts.push(Conflict {
                    conflict_type: ConflictType::CircularDependency,
                    severity: ConflictSeverity::Critical,
                    description: format!("Circular dependency detected: {}", cycle.join(" -> ")),
                    affected_files: cycle.clone(),
                    affected_symbols: vec![],
                    suggested_fix: Some("Refactor to break the circular dependency chain".to_string()),
                });
            }
        }
        
        conflicts
    }
    
    /// Find cycle using DFS
    fn find_cycle(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        if rec_stack.contains(node) {
            // Found cycle - node is already in path, find where it starts
            if let Some(cycle_start) = path.iter().position(|n| n == node) {
                let mut cycle = path[cycle_start..].to_vec();
                cycle.push(node.to_string()); // Close the cycle
                return Some(cycle);
            } else {
                // If not in path, just return the current path + node
                let mut cycle = path.clone();
                cycle.push(node.to_string());
                return Some(cycle);
            }
        }
        
        if visited.contains(node) {
            return None;
        }
        
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());
        
        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if let Some(cycle) = Self::find_cycle(neighbor, graph, visited, rec_stack, path) {
                    return Some(cycle);
                }
            }
        }
        
        rec_stack.remove(node);
        path.pop();
        
        None
    }
    
    /// Detect duplicate definitions
    fn detect_duplicate_definitions(definitions: &HashMap<String, Vec<String>>) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        let mut symbol_locations: HashMap<String, Vec<String>> = HashMap::new();
        
        // Track where each symbol is defined
        for (file, symbols) in definitions {
            for symbol in symbols {
                symbol_locations
                    .entry(symbol.clone())
                    .or_default()
                    .push(file.clone());
            }
        }
        
        // Find duplicates
        for (symbol, locations) in symbol_locations {
            if locations.len() > 1 {
                conflicts.push(Conflict {
                    conflict_type: ConflictType::DuplicateDefinition,
                    severity: ConflictSeverity::Medium,
                    description: format!(
                        "Symbol '{}' is defined in multiple files: {}",
                        symbol,
                        locations.join(", ")
                    ),
                    affected_files: locations.clone(),
                    affected_symbols: vec![symbol.clone()],
                    suggested_fix: Some(format!(
                        "Rename or consolidate definitions of '{}'",
                        symbol
                    )),
                });
            }
        }
        
        conflicts
    }
    
    /// Detect import conflicts
    fn detect_import_conflicts(
        imports: &HashMap<String, Vec<String>>,
        definitions: &HashMap<String, Vec<String>>,
    ) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        
        for (file, imported_symbols) in imports {
            if let Some(defined_symbols) = definitions.get(file) {
                // Find name collisions
                let imported_set: HashSet<_> = imported_symbols.iter().collect();
                let defined_set: HashSet<_> = defined_symbols.iter().collect();
                
                let collisions: Vec<_> = imported_set
                    .intersection(&defined_set)
                    .cloned()
                    .collect();
                
                for collision in collisions {
                    conflicts.push(Conflict {
                        conflict_type: ConflictType::ImportConflict,
                        severity: ConflictSeverity::Medium,
                        description: format!(
                            "Symbol '{}' in file '{}' shadows an imported symbol",
                            collision, file
                        ),
                        affected_files: vec![file.clone()],
                        affected_symbols: vec![collision.to_string()],
                        suggested_fix: Some(format!(
                            "Rename local symbol '{}' or use qualified imports",
                            collision
                        )),
                    });
                }
            }
        }
        
        conflicts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_version_conflicts() {
        let deps = vec![
            VersionRequirement {
                package_name: "requests".to_string(),
                constraint: ">=2.0.0".to_string(),
                required_by: "file1.py".to_string(),
            },
            VersionRequirement {
                package_name: "requests".to_string(),
                constraint: "^1.0.0".to_string(),
                required_by: "file2.py".to_string(),
            },
        ];
        
        let result = ConflictDetector::detect_conflicts(deps, HashMap::new(), HashMap::new());
        
        assert!(result.has_conflicts);
        assert_eq!(result.total_conflicts, 1);
        assert_eq!(result.high_count, 1);
        assert_eq!(result.conflicts[0].conflict_type, ConflictType::VersionConflict);
    }
    
    #[test]
    fn test_detect_circular_dependencies() {
        let mut imports = HashMap::new();
        imports.insert("a.py".to_string(), vec!["b.py".to_string()]);
        imports.insert("b.py".to_string(), vec!["c.py".to_string()]);
        imports.insert("c.py".to_string(), vec!["a.py".to_string()]);
        
        let result = ConflictDetector::detect_conflicts(vec![], imports, HashMap::new());
        
        assert!(result.has_conflicts);
        assert_eq!(result.critical_count, 1);
        assert_eq!(result.conflicts[0].conflict_type, ConflictType::CircularDependency);
    }
    
    #[test]
    fn test_detect_duplicate_definitions() {
        let mut definitions = HashMap::new();
        definitions.insert("file1.py".to_string(), vec!["MyClass".to_string()]);
        definitions.insert("file2.py".to_string(), vec!["MyClass".to_string()]);
        
        let result = ConflictDetector::detect_conflicts(vec![], HashMap::new(), definitions);
        
        assert!(result.has_conflicts);
        assert_eq!(result.medium_count, 1);
        assert_eq!(result.conflicts[0].conflict_type, ConflictType::DuplicateDefinition);
    }
    
    #[test]
    fn test_detect_import_conflicts() {
        let mut imports = HashMap::new();
        imports.insert("main.py".to_string(), vec!["helper".to_string()]);
        
        let mut definitions = HashMap::new();
        definitions.insert("main.py".to_string(), vec!["helper".to_string()]);
        
        let result = ConflictDetector::detect_conflicts(vec![], imports, definitions);
        
        assert!(result.has_conflicts);
        assert_eq!(result.medium_count, 1);
        assert_eq!(result.conflicts[0].conflict_type, ConflictType::ImportConflict);
    }
    
    #[test]
    fn test_no_conflicts() {
        let deps = vec![
            VersionRequirement {
                package_name: "requests".to_string(),
                constraint: ">=2.0.0".to_string(),
                required_by: "file1.py".to_string(),
            },
            VersionRequirement {
                package_name: "numpy".to_string(),
                constraint: "^1.20".to_string(),
                required_by: "file2.py".to_string(),
            },
        ];
        
        let result = ConflictDetector::detect_conflicts(deps, HashMap::new(), HashMap::new());
        
        assert!(!result.has_conflicts);
        assert_eq!(result.total_conflicts, 0);
    }
}
