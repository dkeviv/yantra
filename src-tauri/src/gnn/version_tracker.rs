// GNN Version-Level Tracker: Track code versions at node level for time-travel debugging
// Purpose: Version each GNN node to enable rollback, diff, and conflict detection

use super::{CodeNode, EdgeType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVersion {
    pub node_id: String,
    pub version: u32,
    pub timestamp: String,
    pub content: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub author: Option<String>,
    pub commit_hash: Option<String>,
    pub change_reason: Option<String>,
}

/// Version history for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHistory {
    pub node_id: String,
    pub current_version: u32,
    pub versions: Vec<NodeVersion>,
}

/// Diff between two versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    pub node_id: String,
    pub old_version: u32,
    pub new_version: u32,
    pub changes: Vec<DiffChange>,
}

/// Individual change in diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffChange {
    LineAdded { line_num: u32, content: String },
    LineRemoved { line_num: u32, content: String },
    LineModified { line_num: u32, old_content: String, new_content: String },
}

/// Version tracker for GNN nodes
pub struct VersionTracker {
    /// Node ID -> History
    histories: HashMap<String, NodeHistory>,
    /// Global version counter
    global_version: u64,
}

impl VersionTracker {
    /// Create new version tracker
    pub fn new() -> Self {
        Self {
            histories: HashMap::new(),
            global_version: 0,
        }
    }
    
    /// Track a new version of a node
    pub fn track_version(
        &mut self,
        node: &CodeNode,
        content: String,
        author: Option<String>,
        commit_hash: Option<String>,
        change_reason: Option<String>,
    ) {
        self.global_version += 1;
        
        let history = self.histories
            .entry(node.id.clone())
            .or_insert_with(|| NodeHistory {
                node_id: node.id.clone(),
                current_version: 0,
                versions: Vec::new(),
            });
        
        history.current_version += 1;
        
        let version = NodeVersion {
            node_id: node.id.clone(),
            version: history.current_version,
            timestamp: chrono::Utc::now().to_rfc3339(),
            content,
            file_path: node.file_path.clone(),
            line_start: node.line_start as u32,
            line_end: node.line_end as u32,
            author,
            commit_hash,
            change_reason,
        };
        
        history.versions.push(version);
    }
    
    /// Get current version of a node
    pub fn get_current_version(&self, node_id: &str) -> Option<&NodeVersion> {
        self.histories.get(node_id)
            .and_then(|h| h.versions.last())
    }
    
    /// Get specific version of a node
    pub fn get_version(&self, node_id: &str, version: u32) -> Option<&NodeVersion> {
        self.histories.get(node_id)
            .and_then(|h| {
                h.versions.iter()
                    .find(|v| v.version == version)
            })
    }
    
    /// Get all versions of a node
    pub fn get_history(&self, node_id: &str) -> Option<&NodeHistory> {
        self.histories.get(node_id)
    }
    
    /// Rollback node to specific version
    pub fn rollback(&mut self, node_id: &str, target_version: u32) -> Result<NodeVersion, String> {
        let history = self.histories.get_mut(node_id)
            .ok_or_else(|| format!("Node not found: {}", node_id))?;
        
        if target_version == 0 || target_version > history.current_version {
            return Err(format!(
                "Invalid version: {} (current: {})",
                target_version,
                history.current_version
            ));
        }
        
        // Find the target version
        let target = history.versions.iter()
            .find(|v| v.version == target_version)
            .ok_or_else(|| format!("Version {} not found", target_version))?
            .clone();
        
        // Add rollback as new version
        history.current_version += 1;
        
        let rollback_version = NodeVersion {
            node_id: node_id.to_string(),
            version: history.current_version,
            timestamp: chrono::Utc::now().to_rfc3339(),
            content: target.content.clone(),
            file_path: target.file_path.clone(),
            line_start: target.line_start,
            line_end: target.line_end,
            author: None,
            commit_hash: None,
            change_reason: Some(format!("Rollback to version {}", target_version)),
        };
        
        history.versions.push(rollback_version.clone());
        
        Ok(rollback_version)
    }
    
    /// Generate diff between two versions
    pub fn diff(&self, node_id: &str, old_version: u32, new_version: u32) -> Result<VersionDiff, String> {
        let old = self.get_version(node_id, old_version)
            .ok_or_else(|| format!("Version {} not found", old_version))?;
        
        let new = self.get_version(node_id, new_version)
            .ok_or_else(|| format!("Version {} not found", new_version))?;
        
        let changes = Self::compute_line_diff(&old.content, &new.content);
        
        Ok(VersionDiff {
            node_id: node_id.to_string(),
            old_version,
            new_version,
            changes,
        })
    }
    
    /// Compute line-by-line diff
    fn compute_line_diff(old_content: &str, new_content: &str) -> Vec<DiffChange> {
        let old_lines: Vec<&str> = old_content.lines().collect();
        let new_lines: Vec<&str> = new_content.lines().collect();
        
        let mut changes = Vec::new();
        
        // Simple line-by-line comparison (not optimal, but sufficient for MVP)
        let max_lines = old_lines.len().max(new_lines.len());
        
        for i in 0..max_lines {
            let old_line = old_lines.get(i).copied();
            let new_line = new_lines.get(i).copied();
            
            match (old_line, new_line) {
                (Some(old), Some(new)) if old != new => {
                    changes.push(DiffChange::LineModified {
                        line_num: (i + 1) as u32,
                        old_content: old.to_string(),
                        new_content: new.to_string(),
                    });
                }
                (Some(old), None) => {
                    changes.push(DiffChange::LineRemoved {
                        line_num: (i + 1) as u32,
                        content: old.to_string(),
                    });
                }
                (None, Some(new)) => {
                    changes.push(DiffChange::LineAdded {
                        line_num: (i + 1) as u32,
                        content: new.to_string(),
                    });
                }
                _ => {} // Lines are identical
            }
        }
        
        changes
    }
    
    /// Get nodes modified in time range
    pub fn get_modified_nodes(
        &self,
        start_time: &str,
        end_time: &str,
    ) -> Vec<String> {
        let mut modified = Vec::new();
        
        for (node_id, history) in &self.histories {
            if let Some(version) = history.versions.last() {
                if version.timestamp.as_str() >= start_time && version.timestamp.as_str() <= end_time {
                    modified.push(node_id.clone());
                }
            }
        }
        
        modified
    }
    
    /// Get global version number
    pub fn get_global_version(&self) -> u64 {
        self.global_version
    }
    
    /// Clear history for a node
    pub fn clear_history(&mut self, node_id: &str) {
        self.histories.remove(node_id);
    }
    
    /// Cleanup old versions (keep only last N versions)
    pub fn cleanup_old_versions(&mut self, keep_versions: usize) {
        for history in self.histories.values_mut() {
            if history.versions.len() > keep_versions {
                let start = history.versions.len() - keep_versions;
                history.versions.drain(..start);
            }
        }
    }
}

impl Default for VersionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_node(id: &str) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            name: "test_func".to_string(),
            node_type: super::super::NodeType::Function,
            file_path: "test.py".to_string(),
            line_start: 1,
            line_end: 10,
            code_snippet: Some("def test_func():\n    pass".to_string()),
            docstring: None,
            semantic_embedding: None,
        }
    }
    
    #[test]
    fn test_track_version() {
        let mut tracker = VersionTracker::new();
        let node = create_test_node("test::func1");
        
        tracker.track_version(
            &node,
            "def test_func():\n    pass".to_string(),
            Some("user@example.com".to_string()),
            Some("abc123".to_string()),
            Some("Initial version".to_string()),
        );
        
        let version = tracker.get_current_version("test::func1").unwrap();
        assert_eq!(version.version, 1);
        assert_eq!(version.author, Some("user@example.com".to_string()));
    }
    
    #[test]
    fn test_multiple_versions() {
        let mut tracker = VersionTracker::new();
        let node = create_test_node("test::func1");
        
        // Version 1
        tracker.track_version(&node, "v1".to_string(), None, None, None);
        
        // Version 2
        tracker.track_version(&node, "v2".to_string(), None, None, None);
        
        // Version 3
        tracker.track_version(&node, "v3".to_string(), None, None, None);
        
        let history = tracker.get_history("test::func1").unwrap();
        assert_eq!(history.current_version, 3);
        assert_eq!(history.versions.len(), 3);
    }
    
    #[test]
    fn test_rollback() {
        let mut tracker = VersionTracker::new();
        let node = create_test_node("test::func1");
        
        tracker.track_version(&node, "v1".to_string(), None, None, None);
        tracker.track_version(&node, "v2".to_string(), None, None, None);
        tracker.track_version(&node, "v3".to_string(), None, None, None);
        
        let rollback = tracker.rollback("test::func1", 1).unwrap();
        assert_eq!(rollback.content, "v1");
        assert_eq!(rollback.version, 4); // New version for rollback
        assert!(rollback.change_reason.unwrap().contains("Rollback"));
    }
    
    #[test]
    fn test_diff() {
        let mut tracker = VersionTracker::new();
        let node = create_test_node("test::func1");
        
        tracker.track_version(&node, "line1\nline2\nline3".to_string(), None, None, None);
        tracker.track_version(&node, "line1\nline2_modified\nline3\nline4".to_string(), None, None, None);
        
        let diff = tracker.diff("test::func1", 1, 2).unwrap();
        assert_eq!(diff.changes.len(), 2); // 1 modified, 1 added
    }
    
    #[test]
    fn test_cleanup() {
        let mut tracker = VersionTracker::new();
        let node = create_test_node("test::func1");
        
        for i in 1..=10 {
            tracker.track_version(&node, format!("v{}", i), None, None, None);
        }
        
        tracker.cleanup_old_versions(3);
        
        let history = tracker.get_history("test::func1").unwrap();
        assert_eq!(history.versions.len(), 3);
        assert_eq!(history.versions[0].version, 8);
        assert_eq!(history.versions[2].version, 10);
    }
}
