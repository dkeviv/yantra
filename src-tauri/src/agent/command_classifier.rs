// File: src-tauri/src/agent/command_classifier.rs
// Purpose: Intelligent command classification for execution strategy
// Dependencies: regex
// Last Updated: December 3, 2025

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommandCategory {
    Quick,      // <1s - Simple queries, file reads
    Medium,     // 5-20s - Builds, tests
    Long,       // 10-60s - Complex builds, deployments
    Infinite,   // Dev servers, watch modes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    Synchronous,   // Block and wait
    Asynchronous,  // Run in background with polling
    Background,    // Fire and forget
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandClassification {
    pub category: CommandCategory,
    pub strategy: ExecutionStrategy,
    pub estimated_duration_seconds: Option<u64>,
    pub requires_monitoring: bool,
    pub can_timeout: bool,
    pub timeout_seconds: Option<u64>,
}

pub struct CommandClassifier;

impl CommandClassifier {
    /// Classify a command to determine execution strategy
    pub fn classify(command: &str) -> CommandClassification {
        let cmd_lower = command.to_lowercase();

        // Infinite commands (dev servers, watch modes)
        if Self::is_infinite_command(&cmd_lower) {
            return CommandClassification {
                category: CommandCategory::Infinite,
                strategy: ExecutionStrategy::Background,
                estimated_duration_seconds: None,
                requires_monitoring: true,
                can_timeout: false,
                timeout_seconds: None,
            };
        }

        // Long commands (60s+)
        if Self::is_long_command(&cmd_lower) {
            return CommandClassification {
                category: CommandCategory::Long,
                strategy: ExecutionStrategy::Asynchronous,
                estimated_duration_seconds: Some(60),
                requires_monitoring: true,
                can_timeout: true,
                timeout_seconds: Some(300), // 5 minutes
            };
        }

        // Medium commands (5-20s)
        if Self::is_medium_command(&cmd_lower) {
            return CommandClassification {
                category: CommandCategory::Medium,
                strategy: ExecutionStrategy::Asynchronous,
                estimated_duration_seconds: Some(15),
                requires_monitoring: true,
                can_timeout: true,
                timeout_seconds: Some(120), // 2 minutes
            };
        }

        // Quick commands (<1s) - default
        CommandClassification {
            category: CommandCategory::Quick,
            strategy: ExecutionStrategy::Synchronous,
            estimated_duration_seconds: Some(1),
            requires_monitoring: false,
            can_timeout: true,
            timeout_seconds: Some(10),
        }
    }

    fn is_infinite_command(cmd: &str) -> bool {
        // Dev servers and watch modes
        cmd.contains("npm start")
            || cmd.contains("npm run dev")
            || cmd.contains("npm run serve")
            || cmd.contains("yarn start")
            || cmd.contains("yarn dev")
            || cmd.contains("python manage.py runserver")
            || cmd.contains("flask run")
            || cmd.contains("uvicorn")
            || cmd.contains("--watch")
            || cmd.contains("watch")
            || cmd.contains("serve")
            || cmd.contains("dev-server")
            || cmd.contains("nodemon")
            || cmd.contains("next dev")
            || cmd.contains("vite dev")
            || cmd.contains("cargo watch")
            || cmd.contains("cargo run") && !cmd.contains("--release")
    }

    fn is_long_command(cmd: &str) -> bool {
        // Complex builds, deployments, large test suites
        cmd.contains("cargo build --release")
            || cmd.contains("npm run build")
            || cmd.contains("yarn build")
            || cmd.contains("docker build")
            || cmd.contains("docker-compose up")
            || cmd.contains("deploy")
            || cmd.contains("terraform apply")
            || cmd.contains("kubectl apply")
            || cmd.contains("make install")
            || cmd.contains("webpack")
            || cmd.contains("parcel build")
    }

    fn is_medium_command(cmd: &str) -> bool {
        // Tests, linting, regular builds
        cmd.contains("npm test")
            || cmd.contains("yarn test")
            || cmd.contains("pytest")
            || cmd.contains("cargo test")
            || cmd.contains("cargo build") && !cmd.contains("--release")
            || cmd.contains("cargo check")
            || cmd.contains("cargo clippy")
            || cmd.contains("npm run lint")
            || cmd.contains("eslint")
            || cmd.contains("tsc")
            || cmd.contains("rustfmt")
            || cmd.contains("black")
            || cmd.contains("mypy")
            || cmd.contains("go test")
            || cmd.contains("go build")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_quick() {
        let result = CommandClassifier::classify("ls -la");
        assert_eq!(result.category, CommandCategory::Quick);
        assert!(matches!(result.strategy, ExecutionStrategy::Synchronous));
    }

    #[test]
    fn test_classify_medium_tests() {
        let result = CommandClassifier::classify("cargo test");
        assert_eq!(result.category, CommandCategory::Medium);
        assert!(matches!(result.strategy, ExecutionStrategy::Asynchronous));
        assert!(result.requires_monitoring);
    }

    #[test]
    fn test_classify_long_build() {
        let result = CommandClassifier::classify("cargo build --release");
        assert_eq!(result.category, CommandCategory::Long);
        assert!(matches!(result.strategy, ExecutionStrategy::Asynchronous));
    }

    #[test]
    fn test_classify_infinite_dev_server() {
        let result = CommandClassifier::classify("npm run dev");
        assert_eq!(result.category, CommandCategory::Infinite);
        assert!(matches!(result.strategy, ExecutionStrategy::Background));
        assert!(result.requires_monitoring);
        assert!(!result.can_timeout);
    }

    #[test]
    fn test_classify_watch_mode() {
        let result = CommandClassifier::classify("cargo watch -x test");
        assert_eq!(result.category, CommandCategory::Infinite);
    }

    #[test]
    fn test_classify_docker_build() {
        let result = CommandClassifier::classify("docker build -t myapp .");
        assert_eq!(result.category, CommandCategory::Long);
    }
}
