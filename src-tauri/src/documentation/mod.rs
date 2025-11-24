// File: src-tauri/src/documentation/mod.rs
// Purpose: Documentation extraction and management system
// Dependencies: serde, chrono, rusqlite
// Last Updated: November 24, 2025

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

pub mod extractor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: FeatureStatus,
    pub extracted_from: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FeatureStatus {
    Planned,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub title: String,
    pub context: String,
    pub decision: String,
    pub rationale: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub id: String,
    pub change_type: ChangeType,
    pub description: String,
    pub files: Vec<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChangeType {
    FileAdded,
    FileModified,
    FileDeleted,
    FunctionAdded,
    FunctionRemoved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub status: TaskStatus,
    pub milestone: String,
    pub dependencies: Vec<String>,
    pub requires_user_action: bool,
    pub user_action_instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskStatus {
    Completed,
    InProgress,
    Pending,
}

/// DocumentationManager handles extraction and storage of project documentation
pub struct DocumentationManager {
    workspace_path: PathBuf,
    features: Vec<Feature>,
    decisions: Vec<Decision>,
    changes: Vec<Change>,
    tasks: Vec<Task>,
}

impl DocumentationManager {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self {
            workspace_path,
            features: Vec::new(),
            decisions: Vec::new(),
            changes: Vec::new(),
            tasks: Vec::new(),
        }
    }

    /// Load documentation from existing markdown files in .github/
    pub fn load_from_files(&mut self) -> Result<(), String> {
        // Load from Project_Plan.md
        if let Ok(content) = fs::read_to_string(self.workspace_path.join("Project_Plan.md")) {
            self.extract_tasks_from_plan(&content);
        }

        // Load from Features.md
        if let Ok(content) = fs::read_to_string(self.workspace_path.join("Features.md")) {
            self.extract_features(&content);
        }

        // Load from Decision_Log.md
        if let Ok(content) = fs::read_to_string(self.workspace_path.join("Decision_Log.md")) {
            self.extract_decisions(&content);
        }

        Ok(())
    }

    /// Extract tasks from Project_Plan.md
    fn extract_tasks_from_plan(&mut self, content: &str) {
        let mut current_milestone = "MVP".to_string();
        let mut task_id = 0;

        for line in content.lines() {
            // Detect milestone headers
            if line.contains("Week") || line.contains("Phase") {
                current_milestone = line.trim().to_string();
            }

            // Extract tasks with checkboxes
            if line.trim().starts_with("- [") {
                task_id += 1;
                let is_completed = line.contains("[x]") || line.contains("[X]");
                let is_in_progress = line.contains("🔄") || line.contains("In Progress");
                
                let status = if is_completed {
                    TaskStatus::Completed
                } else if is_in_progress {
                    TaskStatus::InProgress
                } else {
                    TaskStatus::Pending
                };

                // Extract task title
                let title = line
                    .split(']')
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .trim_start_matches('*')
                    .trim()
                    .to_string();

                if !title.is_empty() {
                    self.tasks.push(Task {
                        id: task_id.to_string(),
                        title,
                        status,
                        milestone: current_milestone.clone(),
                        dependencies: Vec::new(),
                        requires_user_action: false,
                        user_action_instructions: None,
                    });
                }
            }
        }
    }

    /// Extract features from Features.md
    fn extract_features(&mut self, content: &str) {
        let mut feature_id = 0;
        let mut in_feature_section = false;

        for line in content.lines() {
            if line.starts_with("###") && line.contains("✅") {
                feature_id += 1;
                in_feature_section = true;
                
                let title = line
                    .trim_start_matches('#')
                    .trim()
                    .trim_start_matches("✅")
                    .trim()
                    .to_string();

                self.features.push(Feature {
                    id: feature_id.to_string(),
                    title,
                    description: String::new(),
                    status: FeatureStatus::Completed,
                    extracted_from: "Features.md".to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                });
            } else if in_feature_section && line.starts_with("####") && line.contains("Description") {
                in_feature_section = false;
            }
        }
    }

    /// Extract decisions from Decision_Log.md
    fn extract_decisions(&mut self, content: &str) {
        let mut decision_id = 0;

        for line in content.lines() {
            if line.starts_with("##") && !line.contains("Decision Log") {
                decision_id += 1;
                
                let title = line
                    .trim_start_matches('#')
                    .trim()
                    .to_string();

                self.decisions.push(Decision {
                    id: decision_id.to_string(),
                    title,
                    context: "Extracted from Decision_Log.md".to_string(),
                    decision: String::new(),
                    rationale: String::new(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                });
            }
        }
    }

    /// Add a new feature extracted from chat
    pub fn add_feature(&mut self, title: String, description: String, extracted_from: String) {
        let id = (self.features.len() + 1).to_string();
        self.features.push(Feature {
            id,
            title,
            description,
            status: FeatureStatus::Planned,
            extracted_from,
            timestamp: chrono::Utc::now().to_rfc3339(),
        });
    }

    /// Add a new decision
    pub fn add_decision(&mut self, title: String, context: String, decision: String, rationale: String) {
        let id = (self.decisions.len() + 1).to_string();
        self.decisions.push(Decision {
            id,
            title,
            context,
            decision,
            rationale,
            timestamp: chrono::Utc::now().to_rfc3339(),
        });
    }

    /// Add a change log entry
    pub fn add_change(&mut self, change_type: ChangeType, description: String, files: Vec<String>) {
        let id = (self.changes.len() + 1).to_string();
        self.changes.push(Change {
            id,
            change_type,
            description,
            files,
            timestamp: chrono::Utc::now().to_rfc3339(),
        });
    }

    /// Get all features
    pub fn get_features(&self) -> &[Feature] {
        &self.features
    }

    /// Get all decisions
    pub fn get_decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Get all changes
    pub fn get_changes(&self) -> &[Change] {
        &self.changes
    }

    /// Get all tasks
    pub fn get_tasks(&self) -> &[Task] {
        &self.tasks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_documentation_manager_creation() {
        let manager = DocumentationManager::new(PathBuf::from("/tmp/test"));
        assert_eq!(manager.features.len(), 0);
        assert_eq!(manager.decisions.len(), 0);
        assert_eq!(manager.changes.len(), 0);
        assert_eq!(manager.tasks.len(), 0);
    }

    #[test]
    fn test_add_feature() {
        let mut manager = DocumentationManager::new(PathBuf::from("/tmp/test"));
        manager.add_feature(
            "Test Feature".to_string(),
            "Test Description".to_string(),
            "Chat message".to_string(),
        );
        assert_eq!(manager.features.len(), 1);
        assert_eq!(manager.features[0].title, "Test Feature");
    }

    #[test]
    fn test_add_decision() {
        let mut manager = DocumentationManager::new(PathBuf::from("/tmp/test"));
        manager.add_decision(
            "Test Decision".to_string(),
            "Context".to_string(),
            "Decision made".to_string(),
            "Rationale".to_string(),
        );
        assert_eq!(manager.decisions.len(), 1);
    }

    #[test]
    fn test_add_change() {
        let mut manager = DocumentationManager::new(PathBuf::from("/tmp/test"));
        manager.add_change(
            ChangeType::FileModified,
            "Modified file".to_string(),
            vec!["test.rs".to_string()],
        );
        assert_eq!(manager.changes.len(), 1);
    }
}
