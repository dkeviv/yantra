// Git module - Model Context Protocol integration
// Purpose: Auto-commit successful code changes with AI-generated messages

pub mod mcp;
pub mod commit;

pub use mcp::GitMcp;
pub use commit::{CommitManager, CommitResult};
