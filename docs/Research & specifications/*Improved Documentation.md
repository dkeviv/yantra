# YDoc System - Technical Specification

**Version:** 1.0

**Status:** Draft

**Author:** Yantra AI Agent

**Created:** 2025-01-15

---

## 1. Overview

### 1.1 Purpose

YDoc is Yantra's unified documentation system that treats documentation as first-class nodes in the dependency graph. It eliminates the split experience between IDE and external documentation tools by making all project documentation editable, traceable, and graph-linked within Yantra.

### 1.2 Core Principles

* **Block DB as canonical source** - files are serialization, not storage
* **Graph-native** - every block links to requirements, code, and other docs
* **Agent-first editing** - LLM reads JSON, writes via tools
* **Git-friendly** - exportable, diffable, conflict-detectable
* **Full traceability** - requirement → architecture → spec → code → test → docs

### 1.3 File Format

YDoc uses notebook JSON schema (ipynb-compatible) with custom metadata:

* Extension: `.ydoc`
* Schema: ipynb-compatible (opens in VS Code, Jupyter, GitHub)
* Custom data: stored in cell `metadata` field
* Cell types: standard (`markdown`, `code`, `raw`)
* Yantra types: stored in `metadata.yantra_type`

---

## 2. Document Types

| Type          | Code         | Purpose                                   |
| ------------- | ------------ | ----------------------------------------- |
| Requirements  | `REQ`      | PRD, user intent, acceptance criteria     |
| ADR           | `ADR`      | Architecture Decision Records             |
| Architecture  | `ARCH`     | System design, component diagrams, flows  |
| Tech Spec     | `SPEC`     | Detailed behavior specifications          |
| Project Plan  | `PLAN`     | Tasks, milestones, timeline               |
| Tech Guide    | `TECH`     | Internal technical documentation          |
| API Guide     | `API`      | Endpoint/interface documentation          |
| User Guide    | `USER`     | End-user documentation                    |
| Testing Plan  | `TEST`     | Test strategy, coverage plan              |
| Test Results  | `RESULT`   | Historical test runs (smart archived)     |
| Change Log    | `CHANGE`   | What changed, when, by whom               |
| Decisions Log | `DECISION` | Sign-offs, approvals, requirement changes |

---

## 3. Block Schema

### 3.1 Cell Structure

```json
{
  "cell_type": "markdown",
  "source": ["Content as array of strings"],
  "metadata": {
    "yantra_id": "uuid-v4",
    "yantra_type": "requirement|adr|architecture|spec|...",
    "created_by": "user|agent",
    "created_at": "ISO-8601 timestamp",
    "modified_by": "user|agent",
    "modified_at": "ISO-8601 timestamp",
    "modifier_id": "user-123|agent-task-456",
    "graph_edges": ["REQ-001", "src/auth/oauth.rs:45-120"],
    "tags": ["auth", "oauth", "security"],
    "status": "draft|review|approved|deprecated"
  }
}
```

### 3.2 Document Structure

```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "yantra_doc_id": "uuid-v4",
    "yantra_doc_type": "requirement|adr|spec|...",
    "title": "Document Title",
    "created_by": "user|agent",
    "created_at": "ISO-8601 timestamp",
    "modified_at": "ISO-8601 timestamp",
    "version": "1.0.0",
    "status": "draft|review|approved|deprecated"
  },
  "cells": []
}
```

### 3.3 Metadata Fields

| Field           | Type              | Required | Description                          |
| --------------- | ----------------- | -------- | ------------------------------------ |
| `yantra_id`   | string (uuid)     | Yes      | Unique block identifier              |
| `yantra_type` | string            | Yes      | Block type (requirement, spec, etc.) |
| `created_by`  | string            | Yes      | `user`or `agent`                 |
| `created_at`  | string (ISO-8601) | Yes      | Creation timestamp                   |
| `modified_by` | string            | Yes      | Last modifier type                   |
| `modified_at` | string            | Yes      | Last modification timestamp          |
| `modifier_id` | string            | Yes      | User ID or agent task ID             |
| `graph_edges` | array[string]     | No       | Links to other blocks, code, docs    |
| `tags`        | array[string]     | No       | Searchable tags                      |
| `status`      | string            | No       | Block status                         |

---

## 4. Block Database

### 4.1 Technology

SQLite with FTS5 for full-text search.

### 4.2 Schema

```sql
-- Documents table
CREATE TABLE documents (
  id TEXT PRIMARY KEY,
  doc_type TEXT NOT NULL,
  title TEXT NOT NULL,
  file_path TEXT NOT NULL,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  modified_at TEXT NOT NULL,
  version TEXT DEFAULT '1.0.0',
  status TEXT DEFAULT 'draft'
);

-- Blocks table
CREATE TABLE blocks (
  id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  cell_index INTEGER NOT NULL,
  cell_type TEXT NOT NULL,
  yantra_type TEXT NOT NULL,
  content TEXT NOT NULL,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  modified_by TEXT NOT NULL,
  modified_at TEXT NOT NULL,
  modifier_id TEXT NOT NULL,
  status TEXT DEFAULT 'draft',
  FOREIGN KEY (doc_id) REFERENCES documents(id)
);

-- Graph edges table
CREATE TABLE graph_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_block_id TEXT NOT NULL,
  target_type TEXT NOT NULL,  -- 'block', 'code', 'file'
  target_ref TEXT NOT NULL,   -- block ID, file:line range, etc.
  edge_type TEXT NOT NULL,    -- 'implements', 'tests', 'documents', etc.
  created_at TEXT NOT NULL,
  FOREIGN KEY (source_block_id) REFERENCES blocks(id)
);

-- Tags table
CREATE TABLE block_tags (
  block_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  PRIMARY KEY (block_id, tag),
  FOREIGN KEY (block_id) REFERENCES blocks(id)
);

-- Full-text search
CREATE VIRTUAL TABLE blocks_fts USING fts5(
  content,
  content_rowid='rowid'
);
```

### 4.3 Indexing

* Primary: `blocks.id`, `documents.id`
* Foreign: `blocks.doc_id`, `graph_edges.source_block_id`
* Search: FTS5 on `blocks.content`
* Graph: Index on `graph_edges(source_block_id)`, `graph_edges(target_ref)`

---

## 5. Graph Integration

### 5.1 Edge Types

| Edge Type         | From         | To           | Description                   |
| ----------------- | ------------ | ------------ | ----------------------------- |
| `traces_to`     | Requirement  | Architecture | Requirement traced to design  |
| `implements`    | Architecture | Spec         | Design implemented in spec    |
| `realized_in`   | Spec         | Code         | Spec realized in code         |
| `tested_by`     | Requirement  | Test         | Requirement validated by test |
| `documented_in` | Code         | API Guide    | Code documented in guide      |
| `explained_in`  | Feature      | User Guide   | Feature explained for users   |
| `decided_by`    | Any          | ADR          | Decision recorded in ADR      |
| `approved_in`   | Any          | Decision Log | Sign-off recorded             |
| `changed_in`    | Any          | Change Log   | Change recorded               |

### 5.2 Graph Queries

```rust
// Find all code implementing a requirement
fn code_for_requirement(req_id: &str) -> Vec<CodeRef> {
    graph.traverse(req_id)
        .filter(|edge| edge.edge_type == "realized_in")
        .filter(|node| node.target_type == "code")
        .collect()
}

// Find all docs needing update when code changes
fn docs_affected_by_code(code_ref: &str) -> Vec<BlockRef> {
    graph.reverse_traverse(code_ref)
        .filter(|edge| edge.edge_type == "documented_in" 
                    || edge.edge_type == "explained_in")
        .collect()
}

// Full traceability chain for a requirement
fn trace_requirement(req_id: &str) -> TraceChain {
    TraceChain {
        requirement: req_id,
        architecture: graph.find_edges(req_id, "traces_to"),
        specs: graph.find_edges(req_id, "implements"),
        code: graph.find_edges(req_id, "realized_in"),
        tests: graph.find_edges(req_id, "tested_by"),
        docs: graph.find_edges(req_id, "documented_in"),
    }
}
```

### 5.3 Integration with Existing Graph

YDoc blocks become nodes in Yantra's existing petgraph dependency graph:

```rust
enum YantraNode {
    // Existing
    CodeFile(FileRef),
    Function(FunctionRef),
    Module(ModuleRef),
  
    // New - YDoc
    Document(DocRef),
    Block(BlockRef),
}

enum YantraEdge {
    // Existing
    Imports,
    Calls,
    Depends,
  
    // New - YDoc
    TracesTo,
    Implements,
    RealizedIn,
    TestedBy,
    DocumentedIn,
    ExplainedIn,
    DecidedBy,
    ApprovedIn,
    ChangedIn,
}
```

---

## 6. Agent Workflows

### 6.1 Agent Tools

```typescript
// Read operations
interface ReadTools {
  // Get full document as JSON
  read_ydoc(doc_id: string): YDoc;
  
  // Get specific block
  read_block(block_id: string): Block;
  
  // Search blocks by content
  search_blocks(query: string, filters?: BlockFilters): Block[];
  
  // Get graph edges for a block
  get_edges(block_id: string, edge_type?: string): Edge[];
  
  // Trace requirement to all related artifacts
  trace_requirement(req_id: string): TraceChain;
}

// Write operations
interface WriteTools {
  // Create new block
  create_block(
    doc_id: string,
    content: string,
    yantra_type: string,
    after_block_id?: string,
    graph_edges?: string[]
  ): Block;
  
  // Update block content
  update_block(
    block_id: string,
    content: string,
    graph_edges?: string[]
  ): Block;
  
  // Delete block
  delete_block(block_id: string): void;
  
  // Move block
  move_block(
    block_id: string,
    after_block_id: string
  ): void;
  
  // Add graph edge
  add_edge(
    source_block_id: string,
    target_ref: string,
    edge_type: string
  ): Edge;
  
  // Remove graph edge
  remove_edge(edge_id: string): void;
}

// Document operations
interface DocTools {
  // Create new document
  create_ydoc(
    doc_type: string,
    title: string,
    initial_content?: string
  ): YDoc;
  
  // Convert markdown to ydoc
  import_markdown(
    file_path: string,
    doc_type: string
  ): YDoc;
  
  // Export ydoc to format
  export_ydoc(
    doc_id: string,
    format: 'markdown' | 'html' | 'pdf' | 'docx'
  ): string;
}
```

### 6.2 Intake Workflow

User provides PRD (markdown or plain text):

```
1. Agent reads input file/text
2. Agent calls create_ydoc(type: 'requirement', title: 'PRD Title')
3. Agent parses content into logical blocks
4. For each block:
   a. Agent calls create_block(doc_id, content, 'requirement')
   b. Agent identifies entities (features, constraints, etc.)
   c. Agent adds tags via block metadata
5. Agent creates initial graph edges to existing code (if any)
6. Block DB updated, .ydoc file written
7. Git staged for commit
```

### 6.3 Code Generation Workflow

From Tech Spec to Code:

```
1. Agent reads spec blocks via read_ydoc()
2. Agent identifies implementation targets
3. For each target:
   a. Agent generates code
   b. Agent calls add_edge(spec_block_id, code_ref, 'realized_in')
4. Agent updates Change Log via create_block()
5. Agent updates affected API Guide blocks
```

### 6.4 Test Generation Workflow

Intent-based testing:

```
1. Agent reads requirement blocks
2. Agent reads spec blocks (behavior)
3. Agent reads code (implementation)
4. Agent generates tests that validate:
   a. Requirement intent (what user asked for)
   b. Spec behavior (how it should work)
   c. Code correctness (implementation details)
5. Agent links tests:
   a. add_edge(req_id, test_id, 'tested_by')
   b. add_edge(spec_id, test_id, 'tested_by')
   c. add_edge(code_ref, test_id, 'tested_by')
6. Agent updates Testing Plan
7. On test run: Agent creates Test Results entry
```

### 6.5 Documentation Update Workflow

When code changes:

```
1. Graph query: docs_affected_by_code(changed_code_ref)
2. For each affected block:
   a. Agent reads current content
   b. Agent reads new code
   c. Agent updates block via update_block()
   d. Agent creates Change Log entry
3. If significant: Agent creates Decision Log entry for review
```

---

## 7. Git Integration

### 7.1 File Structure

```
/ydocs
  /requirements
    MASTER.ydoc                 ← Vision, goals, epic index
    EPIC-auth.ydoc              ← Auth requirements (REQ-001 to REQ-015)
    EPIC-payments.ydoc
    EPIC-profile.ydoc
  
  /architecture
    MASTER.ydoc                 ← System overview, component index
    COMPONENT-backend.ydoc
    COMPONENT-frontend.ydoc
    COMPONENT-infrastructure.ydoc
  
  /specs
    MASTER.ydoc                 ← Spec index, cross-cutting concerns
    FEATURE-auth-flow.ydoc
    FEATURE-payment-flow.ydoc
  
  /adr
    ADR-001-database.ydoc
    ADR-002-auth-provider.ydoc
    ADR-003-hosting.ydoc
  
  /guides
    /tech
      MASTER.ydoc               ← Tech guide index
      SECTION-setup.ydoc
      SECTION-deployment.ydoc
      SECTION-debugging.ydoc
    /api
      MASTER.ydoc               ← API guide index
      MODULE-auth.ydoc
      MODULE-users.ydoc
      MODULE-payments.ydoc
    /user
      MASTER.ydoc               ← User guide index
      SECTION-getting-started.ydoc
      SECTION-features.ydoc
      SECTION-faq.ydoc
    
  /plans
    MASTER.ydoc                 ← Project plan overview
    SPRINT-001.ydoc
    SPRINT-002.ydoc
  
  /testing
    MASTER.ydoc                 ← Test strategy, coverage overview
    PLAN-auth.ydoc
    PLAN-payments.ydoc
    /results
      RESULT-2025-01-15.ydoc
      RESULT-2025-01-16.ydoc
    
  /logs
    CHANGE-LOG.ydoc
    DECISION-LOG.ydoc
```

### 7.2 Document Organization Rules

| Doc Type     | Folder               | Naming Convention            | Granularity                               |
| ------------ | -------------------- | ---------------------------- | ----------------------------------------- |
| Requirements | `/requirements`    | `EPIC-{name}.ydoc`         | One file per epic, blocks per requirement |
| Architecture | `/architecture`    | `COMPONENT-{name}.ydoc`    | One file per component                    |
| Tech Specs   | `/specs`           | `FEATURE-{name}.ydoc`      | One file per feature                      |
| ADR          | `/adr`             | `ADR-{number}-{name}.ydoc` | One file per decision                     |
| Tech Guide   | `/guides/tech`     | `SECTION-{name}.ydoc`      | One file per section                      |
| API Guide    | `/guides/api`      | `MODULE-{name}.ydoc`       | One file per module                       |
| User Guide   | `/guides/user`     | `SECTION-{name}.ydoc`      | One file per section                      |
| Project Plan | `/plans`           | `SPRINT-{number}.ydoc`     | One file per sprint                       |
| Testing Plan | `/testing`         | `PLAN-{name}.ydoc`         | One file per feature                      |
| Test Results | `/testing/results` | `RESULT-{date}.ydoc`       | One file per day                          |
| Change Log   | `/logs`            | `CHANGE-LOG.ydoc`          | Single file, append blocks                |
| Decision Log | `/logs`            | `DECISION-LOG.ydoc`        | Single file, append blocks                |

### 7.3 MASTER.ydoc Structure

Each folder's MASTER.ydoc serves as index and overview:

```json
{
  "metadata": {
    "yantra_doc_type": "master",
    "yantra_folder": "requirements",
    "title": "Product Requirements - Master"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["# Product Vision\n\nBuild an AI-powered..."],
      "metadata": { "yantra_type": "vision" }
    },
    {
      "cell_type": "markdown", 
      "source": ["# Goals\n\n1. Ship MVP by Q2..."],
      "metadata": { "yantra_type": "goals" }
    },
    {
      "cell_type": "markdown",
      "source": ["# Epic Index"],
      "metadata": { 
        "yantra_type": "index",
        "children": [
          {"doc": "EPIC-auth.ydoc", "title": "Authentication", "status": "in-progress"},
          {"doc": "EPIC-payments.ydoc", "title": "Payments", "status": "planned"},
          {"doc": "EPIC-profile.ydoc", "title": "User Profile", "status": "planned"}
        ]
      }
    }
  ]
}
```

### 7.4 Agent Project Initialization State Machine

When agent initializes a new project or onboards existing docs:

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT_INIT_START                       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  CHECK_EXISTING_YDOCS                       │
│         Does /ydocs folder exist with content?              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
                 ┌────────┴────────┐
                 ▼                 ▼
          ┌──────────┐      ┌──────────────┐
          │  EXISTS  │      │  NOT EXISTS  │
          └────┬─────┘      └──────┬───────┘
               ▼                   ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│   VALIDATE_STRUCTURE │  │       CREATE_FOLDER_STRUCTURE     │
│   Check for MASTER   │  │  Create all folders + MASTER.ydoc │
│   files, integrity   │  │  for each doc type                │
└──────────┬───────────┘  └───────────────┬───────────────────┘
           ▼                              ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│    REPAIR_IF_NEEDED  │  │       INIT_BLOCK_DB               │
│  Create missing      │  │  Create tables, indexes           │
│  MASTER files        │  │                                   │
└──────────┬───────────┘  └───────────────┬───────────────────┘
           ▼                              ▼
           └──────────────┬───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    INDEX_ALL_YDOCS                          │
│         Parse all .ydoc files into Block DB                 │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   BUILD_GRAPH_EDGES                         │
│      Extract graph_edges, build dependency graph            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  REGISTER_FILE_WATCHERS                     │
│       Watch /ydocs for external changes                     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT_INIT_COMPLETE                    │
└─────────────────────────────────────────────────────────────┘
```

### 7.5 Agent Document Creation State Machine

When agent creates new documentation:

```
┌─────────────────────────────────────────────────────────────┐
│                    DOC_CREATE_START                         │
│              Input: doc_type, title, content                │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  DETERMINE_LOCATION                         │
│    Map doc_type to folder + naming convention               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  CHECK_EXISTING_DOC                         │
│      Does doc already exist? Should append or create?       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
                 ┌────────┴────────┐
                 ▼                 ▼
          ┌──────────┐      ┌──────────────┐
          │  EXISTS  │      │  NOT EXISTS  │
          │  (append)│      │  (create)    │
          └────┬─────┘      └──────┬───────┘
               ▼                   ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│   LOAD_EXISTING_DOC  │  │       CREATE_NEW_YDOC             │
│                      │  │  Generate yantra_doc_id           │
│                      │  │  Set metadata                     │
└──────────┬───────────┘  └───────────────┬───────────────────┘
           ▼                              ▼
           └──────────────┬───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PARSE_CONTENT_TO_BLOCKS                  │
│         Split content into logical blocks                   │
│         Extract requirement IDs (REQ-XXX)                   │
│         Assign yantra_id to each block                      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ADD_BLOCKS_TO_DOC                        │
│         Append blocks to cells array                        │
│         Set created_by, timestamps                          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    UPDATE_BLOCK_DB                          │
│         Insert blocks into SQLite                           │
│         Update FTS index                                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    UPDATE_MASTER_INDEX                      │
│         Add doc reference to MASTER.ydoc                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CREATE_GRAPH_EDGES                       │
│         Link to existing requirements/code                  │
│         Update dependency graph                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    WRITE_YDOC_FILE                          │
│         Serialize to JSON, write to disk                    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOG_CHANGE                               │
│         Append entry to CHANGE-LOG.ydoc                     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE_FOR_GIT                            │
│         git add /ydocs/{path}                               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DOC_CREATE_COMPLETE                      │
│         Return: doc_id, block_ids, file_path                │
└─────────────────────────────────────────────────────────────┘
```

### 7.6 State Machine Implementation

```rust
enum ProjectInitState {
    Start,
    CheckExistingYdocs,
    CreateFolderStructure,
    ValidateStructure,
    RepairIfNeeded,
    InitBlockDb,
    IndexAllYdocs,
    BuildGraphEdges,
    RegisterFileWatchers,
    Complete,
    Failed { error: String, recoverable: bool },
}

enum DocCreateState {
    Start { doc_type: DocType, title: String, content: String },
    DetermineLocation { folder: String, filename: String },
    CheckExistingDoc { path: PathBuf },
    LoadExistingDoc { doc: YDoc },
    CreateNewYdoc { doc: YDoc },
    ParseContentToBlocks { blocks: Vec<Block> },
    AddBlocksToDoc,
    UpdateBlockDb,
    UpdateMasterIndex,
    CreateGraphEdges,
    WriteYdocFile,
    LogChange,
    StageForGit,
    Complete { doc_id: String, block_ids: Vec<String>, path: PathBuf },
    Failed { error: String, rollback: Option<RollbackAction> },
}

struct StateMachine<S> {
    state: S,
    context: StateContext,
    history: Vec<StateTransition>,
}

impl StateMachine<DocCreateState> {
    async fn transition(&mut self) -> Result<()> {
        let next_state = match &self.state {
            DocCreateState::Start { doc_type, title, content } => {
                let (folder, filename) = determine_location(doc_type, title);
                DocCreateState::DetermineLocation { folder, filename }
            }
            DocCreateState::DetermineLocation { folder, filename } => {
                let path = PathBuf::from(format!("/ydocs/{}/{}", folder, filename));
                DocCreateState::CheckExistingDoc { path }
            }
            // ... other transitions
            DocCreateState::Complete { .. } => return Ok(()),
            DocCreateState::Failed { error, .. } => return Err(error.into()),
        };
      
        self.history.push(StateTransition {
            from: self.state.clone(),
            to: next_state.clone(),
            timestamp: Utc::now(),
        });
      
        self.state = next_state;
        Ok(())
    }
}
```

### 7.7 Folder Creation on Init

```rust
const YDOC_FOLDERS: &[&str] = &[
    "requirements",
    "architecture", 
    "specs",
    "adr",
    "guides/tech",
    "guides/api",
    "guides/user",
    "plans",
    "testing",
    "testing/results",
    "logs",
];

async fn create_folder_structure(base: &Path) -> Result<()> {
    for folder in YDOC_FOLDERS {
        let path = base.join(folder);
        fs::create_dir_all(&path).await?;
      
        // Create MASTER.ydoc for folders that need it
        if should_have_master(folder) {
            let master = create_master_ydoc(folder);
            let master_path = path.join("MASTER.ydoc");
            write_ydoc(&master_path, &master).await?;
        }
    }
  
    // Create log files
    create_change_log(base).await?;
    create_decision_log(base).await?;
  
    Ok(())
}

fn should_have_master(folder: &str) -> bool {
    matches!(folder, 
        "requirements" | "architecture" | "specs" | 
        "guides/tech" | "guides/api" | "guides/user" |
        "plans" | "testing"
    )
}
```

### 7.2 Export on Save

When Block DB changes:

```
1. Serialize affected document to .ydoc JSON
2. Write to /ydocs/{type}/{filename}.ydoc
3. Optionally export markdown shadow to /docs/{filename}.md
4. Stage for git
```

### 7.3 Conflict Detection

On git pull or external file change:

```
1. Detect .ydoc file changed outside Yantra
2. Compare file metadata.modified_at with Block DB
3. If external change:
   a. Parse external .ydoc
   b. Diff against Block DB
   c. Show user: "Document changed externally"
   d. Options:
      - "Use external" → reimport to Block DB
      - "Keep Yantra" → overwrite file from Block DB
      - "Review" → show diff, manual merge
```

### 7.4 Diff Tooling

Integrate nbstripout or similar for clean diffs:

```bash
# .gitattributes
*.ydoc diff=ydoc

# .git/config
[diff "ydoc"]
  textconv = yantra ydoc-to-text
```

`yantra ydoc-to-text` extracts just the `source` content for readable diffs.

---

## 8. Smart Archiving

### 8.1 Test Results Retention

```rust
struct ArchivePolicy {
    // Keep all failures indefinitely
    keep_all_failures: true,
  
    // Keep last N passing runs per test
    keep_last_passes: 10,
  
    // Keep daily summary older than 30 days
    summarize_after_days: 30,
  
    // Delete raw results older than 90 days (keep summary)
    delete_raw_after_days: 90,
}
```

### 8.2 Archive Process

```
Daily job:
1. Query test results older than summarize_after_days
2. For each day:
   a. Generate summary (pass/fail counts, flaky tests)
   b. Keep failures, delete passing details
   c. Update RESULT archive document
3. Delete raw entries older than delete_raw_after_days
4. Vacuum Block DB
```

---

## 9. UI Integration

### 9.1 Editor Component

Use Nteract with Monaco swap:

```typescript
import { NotebookApp } from '@nteract/stateful-components';

// Custom cell renderer with Monaco
const YDocEditor: React.FC<{doc: YDoc}> = ({doc}) => {
  return (
    <NotebookApp
      notebook={doc}
      cellEditor={MonacoCellEditor}
      cellRenderer={YDocCellRenderer}
      onCellChange={handleCellChange}
    />
  );
};

// Cell change handler
const handleCellChange = async (cellId: string, content: string) => {
  await agent.update_block(cellId, content);
  // Block DB updated, file synced
};
```

### 9.2 Existing Views Integration

Feature/Plan/Changes/Decisions views query Block DB:

```typescript
// Feature view - query requirements
const features = await db.query(`
  SELECT * FROM blocks 
  WHERE yantra_type = 'requirement'
  AND status != 'deprecated'
`);

// Plan view - query project plan
const tasks = await db.query(`
  SELECT * FROM blocks 
  WHERE yantra_type = 'plan'
  ORDER BY cell_index
`);

// Changes view - query change log
const changes = await db.query(`
  SELECT * FROM blocks 
  WHERE doc_id = (SELECT id FROM documents WHERE doc_type = 'change')
  ORDER BY modified_at DESC
`);

// Decisions view - query decision log
const decisions = await db.query(`
  SELECT * FROM blocks 
  WHERE doc_id = (SELECT id FROM documents WHERE doc_type = 'decision')
  ORDER BY modified_at DESC
`);
```

### 9.3 Graph Visualization

Block → Code traceability in UI:

```typescript
// Show linked code when viewing requirement block
const LinkedCode: React.FC<{blockId: string}> = ({blockId}) => {
  const edges = useGraphEdges(blockId, 'realized_in');
  
  return (
    <Panel title="Implemented In">
      {edges.map(edge => (
        <CodeLink 
          key={edge.id}
          file={edge.target_ref}
          onClick={() => openInEditor(edge.target_ref)}
        />
      ))}
    </Panel>
  );
};
```

---

## 10. VS Code Compatibility

### 10.1 Settings

Add to VS Code settings for `.ydoc` rendering:

```json
{
  "workbench.editorAssociations": {
    "*.ydoc": "jupyter-notebook"
  },
  "files.associations": {
    "*.ydoc": "json"
  }
}
```

### 10.2 Future Extension (Post-MVP)

Simple VS Code extension for enhanced experience:

* Register `.ydoc` file type
* Custom cell type rendering
* Graph edge visualization
* Read-only preview (editing in Yantra)

---

## 11. Export Pipeline

### 11.1 Supported Formats

| Format   | Method            | Quality |
| -------- | ----------------- | ------- |
| Markdown | Direct extraction | High    |
| HTML     | Pandoc            | High    |
| PDF      | Pandoc + LaTeX    | High    |
| DOCX     | Pandoc + template | High    |
| IPYNB    | Rename extension  | Native  |

### 11.2 Export Command

```bash
yantra export <doc_id> --format <format> --output <path>

# Examples
yantra export REQ-001 --format markdown --output ./docs/requirements.md
yantra export SPEC-001 --format pdf --output ./specs/auth-spec.pdf
yantra export API-001 --format docx --output ./guides/api-guide.docx
```

### 11.3 Pandoc Integration

```rust
fn export_to_docx(doc: &YDoc, template: Option<&Path>) -> Result<Vec<u8>> {
    // 1. Convert ydoc to markdown
    let markdown = ydoc_to_markdown(doc);
  
    // 2. Write temp markdown file
    let temp_md = write_temp_file(&markdown)?;
  
    // 3. Run pandoc
    let mut cmd = Command::new("pandoc");
    cmd.args(&["-f", "markdown", "-t", "docx"]);
  
    if let Some(tpl) = template {
        cmd.args(&["--reference-doc", tpl.to_str().unwrap()]);
    }
  
    cmd.args(&["-o", "-", temp_md.to_str().unwrap()]);
  
    let output = cmd.output()?;
    Ok(output.stdout)
}
```

---

## 12. Implementation Plan

### 12.1 Phase 1: Core (MVP)

1. **Block DB schema** - SQLite tables, FTS5
2. **YDoc JSON serialization** - read/write .ydoc files
3. **Agent tools** - create, read, update, delete blocks
4. **Intake workflow** - markdown → ydoc conversion
5. **Git integration** - export on save, conflict detection
6. **Basic graph edges** - requirement → code linking

### 12.2 Phase 2: Editor

1. **Nteract integration** - notebook UI component
2. **Monaco swap** - consistent editor experience
3. **Views rewiring** - query Block DB instead of markdown
4. **Graph visualization** - show edges in UI

### 12.3 Phase 3: Full Traceability

1. **Complete edge types** - all doc-to-doc, doc-to-code edges
2. **Test integration** - intent-based test generation
3. **Change/Decision logs** - automated entries
4. **Smart archiving** - test results retention

### 12.4 Phase 4: Polish

1. **Export pipeline** - all formats via Pandoc
2. **VS Code extension** - enhanced .ydoc support
3. **Diff tooling** - clean git diffs
4. **Performance** - large doc handling

### 12.5 Phase 5: Confluence Integration

1. **MCP connection** - Atlassian MCP server setup
2. **Read-only import** - Confluence pages to ydoc
3. **Webhook integration** - change notifications
4. **Bidirectional sync** - push/pull with conflict resolution
5. **Agent-mediated conflicts** - user decision workflow

---

## 13. Success Metrics

| Metric                       | Target                      |
| ---------------------------- | --------------------------- |
| Doc-to-code traceability     | 100% of requirements linked |
| Agent edit success rate      | >95% (no corruption)        |
| Git conflict resolution      | <5 min average              |
| Test-to-requirement coverage | 100% of requirements tested |
| Doc freshness                | <24h behind code changes    |

---

## 14. Confluence Integration

### 14.1 Overview

Yantra supports bidirectional sync with Atlassian Confluence, allowing teams to continue using Confluence as their external documentation system while benefiting from Yantra's graph-based traceability and agent capabilities.

**Architecture:**

```
Confluence (External SSOT)
    ↕ Atlassian MCP Server (read/write)
    ↕ Confluence Webhooks (change notifications)
Yantra
    ↓
YDoc (Internal working copy)
    ↓
Block DB + Graph (tracking, agent access)
    ↓
Code, Tests, Docs (linked)
```

### 14.2 Integration Components

| Component            | Purpose                            | Technology                                  |
| -------------------- | ---------------------------------- | ------------------------------------------- |
| Atlassian MCP Server | Read/write Confluence pages        | Official `atlassian/atlassian-mcp-server` |
| Confluence Webhooks  | Push notifications on changes      | Confluence REST API                         |
| Sync Engine          | Bidirectional sync logic           | Yantra internal                             |
| Conflict Resolver    | Agent-mediated conflict resolution | Yantra agent                                |

### 14.3 Mapping Model

| Confluence               | YDoc                  | Notes                       |
| ------------------------ | --------------------- | --------------------------- |
| Page ID                  | `yantra_doc_id`     | 1:1 mapping                 |
| Page Version             | `version`           | For conflict detection      |
| Requirement ID (REQ-123) | `yantra_id`on block | Extracted from content      |
| Page Title               | `title`             | Synced                      |
| Page Sections (H1, H2)   | Blocks                | Parsed into separate blocks |
| Labels                   | `tags`              | Synced                      |
| Space hierarchy          | Graph edges           | Parent-child relationships  |
| Last Modified            | `modified_at`       | For sync decisions          |
| Modified By              | `modified_by`       | Track external vs internal  |

### 14.4 MCP Server Configuration

Using the official Atlassian Remote MCP Server:

```json
{
  "mcpServers": {
    "atlassian": {
      "type": "remote",
      "url": "https://mcp.atlassian.com",
      "auth": {
        "type": "oauth2",
        "scopes": [
          "read:confluence-content.all",
          "write:confluence-content",
          "read:confluence-space.summary",
          "offline_access"
        ]
      }
    }
  }
}
```

**Available MCP Tools:**

| Tool                       | Purpose                       |
| -------------------------- | ----------------------------- |
| `confluence_search`      | Search pages using CQL        |
| `confluence_get_page`    | Get page content and metadata |
| `confluence_create_page` | Create new page               |
| `confluence_update_page` | Update existing page          |
| `confluence_list_spaces` | List available spaces         |
| `confluence_get_space`   | Get space details             |

### 14.5 Webhook Configuration

Confluence Webhooks for change detection (separate from MCP):

```rust
struct ConfluenceWebhookConfig {
    base_url: String,           // https://your-domain.atlassian.net
    webhook_url: String,        // https://yantra.app/webhooks/confluence
    events: Vec<WebhookEvent>,
    auth: OAuth2Config,
}

enum WebhookEvent {
    PageCreated,
    PageUpdated,
    PageDeleted,
    PageRestored,
    CommentCreated,
    CommentUpdated,
    LabelAdded,
    LabelRemoved,
}
```

**Webhook Payload Processing:**

```rust
async fn handle_confluence_webhook(payload: WebhookPayload) -> Result<()> {
    match payload.event_type {
        "page_updated" => {
            let page_id = payload.page.id;
            let external_version = payload.page.version;
          
            // Check if we have this page synced
            if let Some(ydoc) = block_db.find_by_confluence_id(&page_id) {
                // Compare versions
                if external_version > ydoc.confluence_version {
                    // External change detected
                    trigger_sync_review(ydoc, payload).await?;
                }
            }
        }
        _ => { /* handle other events */ }
    }
    Ok(())
}
```

### 14.6 Sync Workflows

#### 14.6.1 Initial Import

```
1. User connects Confluence space to Yantra
2. Yantra calls confluence_list_pages(space_key)
3. For each page:
   a. Fetch full content via confluence_get_page(page_id)
   b. Parse HTML/Storage format into blocks
   c. Extract requirement IDs (REQ-123 pattern)
   d. Create ydoc with confluence_id reference
   e. Map requirement IDs to yantra_ids
   f. Create graph edges
4. Register webhooks for the space
5. Log import in Change Log
```

#### 14.6.2 Confluence → Yantra Sync (Webhook Triggered)

```
1. Webhook received: page_updated
2. Agent fetches updated page via MCP
3. Agent computes diff against Block DB:
   a. Parse external content into blocks
   b. Compare block-by-block with stored ydoc
   c. Classify changes (trivial/significant)
4. Agent presents changes to user:
   "Confluence page 'Auth Requirements' changed:
    - REQ-123: Added acceptance criteria
    - REQ-124: Modified description
    Accept / Reject / Review?"
5. User decides:
   - Accept: Update Block DB, create Change Log entry
   - Reject: Keep Yantra version, log Decision
   - Review: Show diff UI, manual merge
6. Create Decision Log entry with user's choice
7. If code/tests affected, notify user of downstream impact
```

#### 14.6.3 Yantra → Confluence Sync (User/Agent Triggered)

```
1. User/Agent modifies block in Yantra
2. Block marked as dirty (needs sync)
3. On sync trigger (manual or checkpoint):
   a. Agent: "Push changes to Confluence?"
   b. User confirms
4. Agent calls confluence_update_page via MCP:
   a. Fetch current Confluence version
   b. Check for conflicts (version mismatch)
   c. If conflict: trigger conflict resolution
   d. If clean: push update
5. Update ydoc.confluence_version
6. Create Change Log entry
7. Create Decision Log entry (user approved push)
```

### 14.7 Conflict Resolution

#### 14.7.1 Conflict Detection

```rust
struct SyncConflict {
    page_id: String,
    yantra_version: u32,
    confluence_version: u32,
    yantra_blocks: Vec<Block>,
    confluence_blocks: Vec<Block>,
    changed_in_yantra: Vec<BlockId>,
    changed_in_confluence: Vec<BlockId>,
    conflict_blocks: Vec<BlockId>,  // Changed in both
}

fn detect_conflict(ydoc: &YDoc, confluence_page: &Page) -> Option<SyncConflict> {
    // Compare versions
    if confluence_page.version <= ydoc.confluence_version {
        return None;  // No external changes
    }
  
    // Parse and diff
    let confluence_blocks = parse_to_blocks(&confluence_page.content);
    let yantra_blocks = &ydoc.cells;
  
    // Find blocks changed in each
    let changed_in_yantra = find_dirty_blocks(yantra_blocks);
    let changed_in_confluence = diff_blocks(yantra_blocks, confluence_blocks);
  
    // Conflict = changed in both
    let conflict_blocks: Vec<_> = changed_in_yantra
        .iter()
        .filter(|b| changed_in_confluence.contains(b))
        .collect();
  
    if conflict_blocks.is_empty() {
        None  // No conflicts, can auto-merge
    } else {
        Some(SyncConflict { /* ... */ })
    }
}
```

#### 14.7.2 Agent-Mediated Resolution

```
Agent prompt for conflict resolution:

"Conflict detected in 'Auth Requirements' (REQ-123):

YANTRA VERSION:
Users must authenticate via OAuth 2.0 with Google, GitHub, and Apple providers.

CONFLUENCE VERSION:
Users must authenticate via OAuth 2.0 with Google and GitHub providers.
Token refresh must happen automatically.

Changes:
- Yantra added: Apple provider
- Confluence added: Token refresh requirement

Options:
1. Keep Yantra version
2. Keep Confluence version
3. Merge both changes
4. Let me review manually

Which would you prefer?"
```

#### 14.7.3 Resolution Logging

```rust
struct ConflictResolution {
    conflict_id: String,
    page_id: String,
    block_ids: Vec<String>,
    resolution: ResolutionType,
    resolved_by: String,      // user-123 or agent
    resolved_at: DateTime,
    rationale: Option<String>,
}

enum ResolutionType {
    KeepYantra,
    KeepConfluence,
    Merged { merge_description: String },
    ManualEdit { diff: String },
}
```

All resolutions logged to Decision Log ydoc.

### 14.8 Requirement ID Handling

#### 14.8.1 ID Extraction

```rust
fn extract_requirement_ids(content: &str) -> Vec<RequirementId> {
    // Pattern: REQ-123, PROJ-456, etc.
    let re = Regex::new(r"\b([A-Z]+-\d+)\b").unwrap();
  
    re.captures_iter(content)
        .map(|cap| RequirementId {
            external_id: cap[1].to_string(),
            source: "confluence",
        })
        .collect()
}
```

#### 14.8.2 ID Mapping

```sql
CREATE TABLE requirement_id_map (
    yantra_id TEXT PRIMARY KEY,      -- Internal block ID
    external_id TEXT NOT NULL,        -- REQ-123
    external_source TEXT NOT NULL,    -- 'confluence', 'jira'
    page_id TEXT,                     -- Confluence page ID
    created_at TEXT NOT NULL,
    UNIQUE(external_id, external_source)
);
```

#### 14.8.3 Graph Edges with External IDs

```rust
// When creating edges, support both internal and external IDs
fn add_edge(source: &str, target: &str, edge_type: &str) -> Result<Edge> {
    // Resolve external IDs to internal
    let source_id = resolve_id(source)?;  // "REQ-123" → "block-uuid-001"
    let target_id = resolve_id(target)?;
  
    graph.add_edge(source_id, target_id, edge_type)
}

fn resolve_id(id: &str) -> Result<String> {
    if id.starts_with("block-") {
        Ok(id.to_string())  // Already internal
    } else {
        // Look up in mapping table
        db.query("SELECT yantra_id FROM requirement_id_map WHERE external_id = ?", id)
    }
}
```

### 14.9 Offline Handling

When Confluence is unreachable:

```rust
enum SyncStatus {
    Synced { last_sync: DateTime },
    Pending { changes: Vec<Change> },
    Offline { since: DateTime },
    Conflict { details: SyncConflict },
}

// Queue changes when offline
fn queue_sync(change: Change) -> Result<()> {
    if !confluence_reachable() {
        sync_queue.push(change);
        notify_user("Confluence offline. Changes queued for sync.");
        return Ok(());
    }
  
    // Online - sync immediately
    sync_to_confluence(change).await
}

// Process queue when back online
async fn process_sync_queue() {
    while let Some(change) = sync_queue.pop() {
        match sync_to_confluence(change).await {
            Ok(_) => log_change("synced"),
            Err(Conflict(c)) => trigger_conflict_resolution(c),
            Err(e) => requeue(change, e),
        }
    }
}
```

### 14.10 Permissions and Security

```rust
struct ConfluenceConnection {
    site_url: String,
    oauth_tokens: EncryptedTokens,
    scopes: Vec<String>,
    user_permissions: UserPermissions,
}

struct UserPermissions {
    can_read: Vec<SpaceKey>,
    can_write: Vec<SpaceKey>,
    is_admin: bool,
}

// Respect Confluence permissions
async fn check_write_permission(page_id: &str) -> Result<bool> {
    let page = confluence_mcp.get_page(page_id).await?;
    let user = get_current_user();
  
    // MCP respects permissions - if we can fetch, we can read
    // For write, check if user has edit permission
    Ok(page.permissions.can_edit(&user))
}
```

### 14.11 Implementation Plan

#### Phase 1: Read-Only Integration

1. MCP server connection
2. Import Confluence pages to ydoc
3. Requirement ID extraction and mapping
4. Graph edges to Confluence content

#### Phase 2: Webhook Integration

1. Webhook registration
2. Change detection
3. Agent notification of external changes
4. Manual sync review

#### Phase 3: Bidirectional Sync

1. Push changes to Confluence via MCP
2. Conflict detection
3. Agent-mediated resolution
4. Decision/Change logging

#### Phase 4: Advanced Features

1. Comment sync
2. Attachment handling
3. Multi-space support
4. Bulk operations

---

## Appendix A: Example YDoc

```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "yantra_doc_id": "550e8400-e29b-41d4-a716-446655440000",
    "yantra_doc_type": "requirement",
    "title": "OAuth Login Requirements",
    "created_by": "user",
    "created_at": "2025-01-15T10:00:00Z",
    "modified_at": "2025-01-15T14:30:00Z",
    "version": "1.0.0",
    "status": "approved"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# OAuth Login Requirements\n",
        "\n",
        "Users must be able to authenticate via OAuth 2.0 providers."
      ],
      "metadata": {
        "yantra_id": "block-001",
        "yantra_type": "requirement",
        "created_by": "user",
        "created_at": "2025-01-15T10:00:00Z",
        "modified_by": "agent",
        "modified_at": "2025-01-15T14:30:00Z",
        "modifier_id": "agent-task-123",
        "graph_edges": ["ARCH-001", "SPEC-001"],
        "tags": ["auth", "oauth", "security"],
        "status": "approved"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Acceptance Criteria\n",
        "\n",
        "1. Support Google OAuth\n",
        "2. Support GitHub OAuth\n",
        "3. Token refresh without user intervention\n",
        "4. Secure token storage"
      ],
      "metadata": {
        "yantra_id": "block-002",
        "yantra_type": "requirement",
        "created_by": "user",
        "created_at": "2025-01-15T10:05:00Z",
        "modified_by": "user",
        "modified_at": "2025-01-15T10:05:00Z",
        "modifier_id": "user-456",
        "graph_edges": ["TEST-001", "TEST-002", "TEST-003", "TEST-004"],
        "tags": ["acceptance-criteria"],
        "status": "approved"
      }
    }
  ]
}
```

---

## Appendix B: Agent Tool Examples

### Creating a Block

```
Agent receives: "Add a requirement for password reset"

Agent calls:
create_block(
  doc_id: "REQ-001-auth",
  content: "## Password Reset\n\nUsers must be able to reset their password via email verification.",
  yantra_type: "requirement",
  after_block_id: "block-002",
  graph_edges: ["SPEC-002-password-reset"]
)

Result: New block created with yantra_id "block-003"
```

### Updating a Block

```
Agent receives: "Update the OAuth requirement to include Apple Sign-In"

Agent calls:
update_block(
  block_id: "block-002",
  content: "## Acceptance Criteria\n\n1. Support Google OAuth\n2. Support GitHub OAuth\n3. Support Apple Sign-In\n4. Token refresh without user intervention\n5. Secure token storage"
)

Result: Block updated, modified_by set to "agent", Change Log entry created
```

### Tracing a Requirement

```
Agent receives: "What code implements the OAuth requirement?"

Agent calls:
trace_requirement(req_id: "block-001")

Result:
{
  requirement: "block-001",
  architecture: ["ARCH-001-block-003"],
  specs: ["SPEC-001-block-001", "SPEC-001-block-002"],
  code: ["src/auth/oauth.rs:45-120", "src/auth/providers/*.rs"],
  tests: ["tests/auth/oauth_test.rs"],
  docs: ["API-001-block-005"]
}
```
