// File: src-tauri/src/ydoc/mod.rs
// Purpose: YDoc documentation system - Full traceability from requirements to code
// Last Updated: December 8, 2025
//
// The YDoc system provides:
// - Block-based documentation with unique IDs
// - Graph-native traceability (requirement → architecture → spec → code → tests)
// - Git-friendly .ydoc files (ipynb-compatible JSON)
// - Full-text search across all documentation
// - Automatic impact analysis when code changes
//
// Architecture:
// - database.rs: SQLite schema (documents, blocks, graph_edges)
// - parser.rs: Parse .ydoc files (ipynb-compatible JSON)
// - manager.rs: CRUD operations, sync DB ↔ files
// - file_ops.rs: Disk I/O, folder structure management
// - traceability.rs: Query methods for requirement → code chains

pub mod database;
pub mod parser;
pub mod manager;
pub mod file_ops;
pub mod traceability;

pub use database::{YDocDatabase, DocumentMetadata, BlockMetadata, TraceabilityEdge};
pub use parser::{
    YDocFile, YDocCell, YDocError, YantraMetadata, GraphEdge, CellMetadata,
    parse_ydoc_file, parse_ydoc_content, serialize_ydoc, write_ydoc_file
};
pub use manager::{YDocManager, ManagerError};
pub use file_ops::{YDocFileOps, FileOpsError};
pub use traceability::{TraceabilityQuery, TraceabilityEntity, TraceabilityEdgeInfo, TraceabilityError};

/// Document types supported by YDoc
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DocumentType {
    Requirements,     // REQ - PRD, user intent, acceptance criteria
    ADR,              // ADR - Architecture Decision Records
    Architecture,     // ARCH - System design, component diagrams
    TechSpec,         // SPEC - Detailed behavior specifications
    ProjectPlan,      // PLAN - Tasks, milestones, timeline
    TechGuide,        // TECH - Internal technical documentation
    APIGuide,         // API - Endpoint/interface documentation
    UserGuide,        // USER - End-user documentation
    TestingPlan,      // TEST - Test strategy, coverage plan
    TestResults,      // RESULT - Historical test runs
    ChangeLog,        // CHANGE - What changed, when, by whom
    DecisionsLog,     // DECISION - Sign-offs, approvals, changes
}

impl DocumentType {
    pub fn code(&self) -> &str {
        match self {
            Self::Requirements => "REQ",
            Self::ADR => "ADR",
            Self::Architecture => "ARCH",
            Self::TechSpec => "SPEC",
            Self::ProjectPlan => "PLAN",
            Self::TechGuide => "TECH",
            Self::APIGuide => "API",
            Self::UserGuide => "USER",
            Self::TestingPlan => "TEST",
            Self::TestResults => "RESULT",
            Self::ChangeLog => "CHANGE",
            Self::DecisionsLog => "DECISION",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "REQ" => Some(Self::Requirements),
            "ADR" => Some(Self::ADR),
            "ARCH" => Some(Self::Architecture),
            "SPEC" => Some(Self::TechSpec),
            "PLAN" => Some(Self::ProjectPlan),
            "TECH" => Some(Self::TechGuide),
            "API" => Some(Self::APIGuide),
            "USER" => Some(Self::UserGuide),
            "TEST" => Some(Self::TestingPlan),
            "RESULT" => Some(Self::TestResults),
            "CHANGE" => Some(Self::ChangeLog),
            "DECISION" => Some(Self::DecisionsLog),
            _ => None,
        }
    }
}

/// Block types within YDoc documents
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BlockType {
    Requirement,
    ADR,
    Architecture,
    Specification,
    Task,
    TechDoc,
    APIDoc,
    UserDoc,
    TestPlan,
    TestResult,
    Change,
    Decision,
}

/// Block status lifecycle
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BlockStatus {
    Draft,
    Review,
    Approved,
    Deprecated,
}

impl Default for BlockStatus {
    fn default() -> Self {
        Self::Draft
    }
}
