# Yantra Platform - Requirements Table

Complete requirements organized in logical implementation buckets.

---

## 1. INFRASTRUCTURE LAYER

### 1.1 Language Support & Editor

| Req #   | Requirement Name             | Requirement Description                                                                                                     | Specification Reference | Planned Phase | Comment                                   |
| ------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------------- | ----------------------------------------- |
| INF-001 | Monaco Editor Integration    | Integrate Monaco editor for code editing with syntax highlighting                                                           | 3.1.1                   | Phase 1 - MVP | Core editor component                     |
| INF-002 | Tree-sitter Parser           | Implement Tree-sitter for AST parsing across 10 languages (Python, JS/TS, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin) | 3.1.1                   | Phase 1 - MVP | Foundation for code intelligence          |
| INF-003 | LSP Integration              | Integrate Language Server Protocol for autocomplete, hover, diagnostics                                                     | 3.1.1                   | Phase 1 - MVP | Type checking and validation              |
| INF-004 | LLM-Based Autocomplete       | Implement LLM-driven intelligent code completion (primary tier)                                                             | 3.1.1.1                 | Phase 1 - MVP | Context-aware multi-line suggestions      |
| INF-005 | GNN-Based Autocomplete       | Implement dependency graph-powered code suggestions (secondary tier)                                                        | 3.1.1.1                 | Phase 1 - MVP | Project-aware symbol suggestions          |
| INF-006 | Static Autocomplete Fallback | Implement keyword and snippet-based completion (fallback tier)                                                              | 3.1.1.1                 | Phase 1 - MVP | Instant fallback when LLM/GNN unavailable |

### 1.2 Dependency Graph (Core Differentiator)

| Req #   | Requirement Name          | Requirement Description                                                   | Specification Reference | Planned Phase       | Comment                                     |
| ------- | ------------------------- | ------------------------------------------------------------------------- | ----------------------- | ------------------- | ------------------------------------------- |
| DEP-001 | File-to-File Dependencies | Track import/include relationships between files                          | 3.1.2                   | Phase 1 - MVP       | Foundation of dependency tracking           |
| DEP-002 | Code Symbol Dependencies  | Track function calls, class usage, method invocations                     | 3.1.2                   | Phase 1 - MVP       | Fine-grained code relationships             |
| DEP-003 | Package Version Tracking  | Track exact package versions as separate nodes (numpy==1.24.0 vs 1.26.0)  | 3.1.2, 3.4.3.4          | Phase 1 - MVP       | Critical for version conflict detection     |
| DEP-004 | Tool Dependency Tracking  | Track build tool chains (webpack→babel→terser), test frameworks, linters  | 3.1.2                   | Phase 1 - MVP       | Complete dependency picture                 |
| DEP-005 | Package-to-File Mapping   | Track which files use which package versions                              | 3.1.2                   | Phase 1 - MVP       | Breaking change impact analysis             |
| DEP-006 | User-to-File Tracking     | Track which developer is working on which files                           | 3.1.2, 3.1.3            | Phase 2A - Post-MVP | Team collaboration, file locking foundation |
| DEP-007 | Git Checkout Tracking     | Track file modifications at Git checkout level                            | 3.1.2                   | Phase 2A - Post-MVP | Conflict prevention across branches         |
| DEP-008 | External API Tracking     | Track API endpoints as nodes with schema tracking                         | 3.1.2, 3.4.3.4          | Phase 3 - Post-MVP  | Cross-system dependency intelligence        |
| DEP-009 | Method Chain Tracking     | Track df.groupby().agg() level granularity                                | 3.1.2, 3.4.3.4          | Phase 3 - Post-MVP  | Deep API usage tracking                     |
| DEP-010 | HNSW Semantic Index       | Implement HNSW indexing with all-MiniLM-L6-v2 embeddings (384-dim)        | 3.1.2                   | Phase 1 - MVP       | Semantic similarity search                  |
| DEP-011 | Bidirectional Edges       | All graph edges must be bidirectional for reverse queries                 | 3.1.2                   | Phase 1 - MVP       | Query dependencies in both directions       |
| DEP-012 | Incremental Updates       | Fast incremental graph updates (<1ms per edge)                            | 3.1.2, 3.4.3.4          | Phase 1 - MVP       | Real-time graph maintenance                 |
| DEP-013 | Breaking Change Detection | Detect function signature changes, removed functions, return type changes | 3.1.2, 3.4.3.4          | Phase 1 - MVP       | Proactive conflict prevention               |

### 1.3 Extended Dependency Features

| Req #   | Requirement Name             | Requirement Description                                                      | Specification Reference | Planned Phase       | Comment                               |
| ------- | ---------------------------- | ---------------------------------------------------------------------------- | ----------------------- | ------------------- | ------------------------------------- |
| EXT-001 | File Locking System          | Implement dependency-aware file locking (NOTE: Phase 2A with Team of Agents) | 3.1.3                   | Phase 2A - Post-MVP | Prevents merge conflicts structurally |
| EXT-002 | Proactive Conflict Detection | Detect conflicts BEFORE code generation starts                               | 3.1.3                   | Phase 2A - Post-MVP | Warn about potential conflicts early  |
| EXT-003 | Work Visibility UI           | Show which developer is editing which files (MVP alternative to locking)     | 3.1.3                   | Phase 1 - MVP       | Manual coordination support           |
| EXT-004 | Developer Activity Tracking  | Track who edited what, when, and why                                         | 3.1.3                   | Phase 2A - Post-MVP | Audit trail and expertise mapping     |

### 1.4 YDoc Documentation System

| Req #    | Requirement Name           | Requirement Description                                                                                                                                             | Specification Reference     | Planned Phase      | Comment                              |
| -------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ------------------ | ------------------------------------ |
| YDOC-001 | Block Database             | Implement SQLite-based block storage with full-text search                                                                                                          | 3.1.4                       | Phase 1 - MVP      | Foundation of YDoc system            |
| YDOC-002 | 12 Document Types          | Support Requirements, Specifications, Architecture, API Docs, Test Plans, Change Logs, Decision Logs, User Guides, Developer Guides, Troubleshooting, FAQ, Glossary | 3.1.4                       | Phase 1 - MVP      | Comprehensive documentation coverage |
| YDOC-003 | Graph-Native Documentation | Create traceability edges between docs, code, tests, requirements                                                                                                   | 3.1.4                       | Phase 1 - MVP      | Full traceability chain              |
| YDOC-004 | MASTER.ydoc Files          | Implement folder-level index files with ordering and metadata                                                                                                       | 3.1.4                       | Phase 1 - MVP      | Hierarchical organization            |
| YDOC-005 | Smart Test Archiving       | Auto-archive test results >30 days, keep summary stats                                                                                                              | 3.1.4                       | Phase 1 - MVP      | Storage optimization                 |
| YDOC-006 | Nteract Monaco Integration | Render YDoc files with Monaco editor for consistency                                                                                                                | 3.1.4                       | Phase 1 - MVP      | Unified editing experience           |
| YDOC-007 | Confluence Integration     | Bidirectional sync with Confluence via MCP server                                                                                                                   | 3.1.4, SPEC-ydoc Section 14 | Phase 2 - Post-MVP | Enterprise documentation workflow    |
| YDOC-008 | Version Control            | Track document versions with timestamps and change metadata                                                                                                         | 3.1.4                       | Phase 1 - MVP      | Document history and rollback        |
| YDOC-009 | Yantra Metadata            | Custom metadata fields for traceability (yantra_id, linked_nodes, etc.)                                                                                             | 3.1.4                       | Phase 1 - MVP      | Graph integration                    |

### 1.5 Unlimited Context Solution

| Req #   | Requirement Name              | Requirement Description                                                                                                    | Specification Reference | Planned Phase | Comment                                |
| ------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------------- | -------------------------------------- |
| CTX-001 | Hierarchical Context Strategy | Implement hierarchical context assembly (current message, recent turns, direct deps, transitive deps, semantic similarity) | 3.1.5                   | Phase 1 - MVP | Enables any LLM including small models |
| CTX-002 | Dynamic Context Assembly      | Assemble context based on relevance and token budget                                                                       | 3.1.5                   | Phase 1 - MVP | Optimize context window usage          |
| CTX-003 | Compression Techniques        | Implement summarization, deduplication, truncation strategies                                                              | 3.1.5                   | Phase 1 - MVP | Fit more context in window             |
| CTX-004 | Token Budget Management       | Track and manage token usage across context levels                                                                         | 3.1.5                   | Phase 1 - MVP | Prevent context overflow               |
| CTX-005 | LLM-Agnostic Support          | Support Claude, GPT-4, Qwen Coder, any model with context limits                                                           | 3.1.5                   | Phase 1 - MVP | Flexibility in LLM choice              |

### 1.6 Yantra Codex (GNN Intelligence Layer)

| Req #     | Requirement Name       | Requirement Description                                                      | Specification Reference | Planned Phase      | Comment                           |
| --------- | ---------------------- | ---------------------------------------------------------------------------- | ----------------------- | ------------------ | --------------------------------- |
| CODEX-001 | GraphSAGE Architecture | Implement GraphSAGE neural network with 1024-dim embeddings, 150M parameters | 3.1.6                   | Phase 1 - MVP      | Actual GNN (not dependency graph) |
| CODEX-002 | Pattern Storage        | Store learned code patterns, bug patterns, test patterns                     | 3.1.6                   | Phase 1 - MVP      | Reusable knowledge base           |
| CODEX-003 | Continuous Learning    | Learn from bugs, tests, LLM mistakes, manual edits, code reviews             | 3.1.6                   | Phase 1 - MVP      | Improve over time                 |
| CODEX-004 | 15ms Inference         | Fast pattern matching and code generation                                    | 3.1.6                   | Phase 1 - MVP      | Real-time assistance              |
| CODEX-005 | Cost Optimization      | Reduce LLM usage by 96% after 12 months through learned patterns             | 3.1.6                   | Phase 1 - MVP      | Long-term cost savings            |
| CODEX-006 | Cloud Codex (Optional) | Opt-in shared learning across Yantra installations                           | 3.1.6                   | Phase 2 - Post-MVP | Community-driven improvements     |

### 1.7 Storage Architecture

| Req #    | Requirement Name          | Requirement Description                                                                                                                   | Specification Reference | Planned Phase                                                           | Comment                                                 |
| -------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------- |
| STOR-001 | Multi-Tier Storage System | Implement Tier 0 (Cloud), Tier 1 (petgraph + SQLite), Tier 2 (sled), Tier 3 (TOML), Tier 4 (HashMap/moka), Codex (separate SQLite + HNSW) | 3.1.7                   | Phase 1 MVP (Tiers 1,3,4 + Codex), Phase 2A (Tier 2), Phase 2B (Tier 0) | Optimized for different access patterns, Codex separate |
| STOR-002 | Tier 1 SQLite             | Local dependency graph, metadata, configuration (WAL mode, connection pooling)                                                            | 3.1.7                   | Phase 1 - MVP                                                           | Primary data store                                      |
| STOR-003 | Tier 2 sled               | Agent state, file locks, task queue (Phase 2A for multi-agent)                                                                            | 3.1.7                   | Phase 2A - Post-MVP                                                     | High-write coordination                                 |
| STOR-004 | Tier 3 Vector DB          | Semantic search, pattern matching, similarity queries                                                                                     | 3.1.7                   | Phase 1 - MVP                                                           | ML-powered search                                       |
| STOR-005 | Tier 4 File System        | Code files, assets, temporary data                                                                                                        | 3.1.7                   | Phase 1 - MVP                                                           | Standard file operations                                |
| STOR-006 | Tier 0 Cloud Graph DB     | Shared dependency graph for team coordination (Phase 2B)                                                                                  | 3.1.7                   | Phase 2B - Post-MVP                                                     | Cross-machine conflict prevention                       |
| STOR-007 | De-Duplication Index      | Prevent duplicate file/function/doc creation                                                                                              | 3.1.7                   | Phase 1 - MVP                                                           | Content-addressable storage                             |
| STOR-008 | WAL Mode                  | Enable Write-Ahead Logging for SQLite                                                                                                     | 3.1.7                   | Phase 1 - MVP                                                           | Concurrent reads/writes                                 |
| STOR-009 | Connection Pooling        | Reuse database connections, max 10 connections                                                                                            | 3.1.7                   | Phase 1 - MVP                                                           | Performance optimization                                |
| STOR-010 | Data Archiving            | Archive old data (test results >30 days, logs >90 days)                                                                                   | 3.1.7                   | Phase 1 - MVP                                                           | Storage management                                      |

### 1.8 Browser Integration

| Req #    | Requirement Name            | Requirement Description                                     | Specification Reference | Planned Phase      | Comment                      |
| -------- | --------------------------- | ----------------------------------------------------------- | ----------------------- | ------------------ | ---------------------------- |
| BROW-001 | CDP Protocol Support        | Integrate Chrome DevTools Protocol for browser automation   | 3.1.8                   | Phase 1 - MVP      | Foundation for UI validation |
| BROW-002 | System Browser Usage        | Use user's installed Chrome/Chromium/Edge (not bundled)     | 3.1.8                   | Phase 1 - MVP      | Zero-touch UX, no downloads  |
| BROW-003 | Browser Validation Workflow | Launch browser, navigate, execute scenarios, capture errors | 3.1.8                   | Phase 1 - MVP      | Automated UI testing         |
| BROW-004 | Console Error Capture       | Monitor and capture JavaScript console errors               | 3.1.8                   | Phase 1 - MVP      | Runtime error detection      |
| BROW-005 | Network Monitoring          | Track API calls, detect failures, monitor performance       | 3.1.8                   | Phase 1 - MVP      | API integration validation   |
| BROW-006 | Screenshot Capture          | Take screenshots at key steps for visual verification       | 3.1.8                   | Phase 1 - MVP      | Visual regression testing    |
| BROW-007 | Self-Healing Triggers       | Auto-fix browser errors detected in production              | 3.1.8                   | Phase 4 - Post-MVP | Production monitoring        |

### 1.9 Architecture View System

| Req #    | Requirement Name        | Requirement Description                                    | Specification Reference | Planned Phase      | Comment                          |
| -------- | ----------------------- | ---------------------------------------------------------- | ----------------------- | ------------------ | -------------------------------- |
| ARCH-001 | Agent-Driven Generation | Auto-generate architecture from code/intent/requirements   | 3.1.9                   | Phase 1 - MVP      | Architecture always current      |
| ARCH-002 | Deviation Detection     | Detect when code drifts from architecture                  | 3.1.9                   | Phase 1 - MVP      | Prevent architectural violations |
| ARCH-003 | Rule of 3 Versioning    | Keep current + 3 most recent versions (4 total)            | 3.1.9                   | Phase 2 - Post-MVP | Architecture history             |
| ARCH-004 | Human Approval Gate     | Require user approval before architecture changes          | 3.1.9                   | Phase 1 - MVP      | Prevent unwanted changes         |
| ARCH-005 | Continuous Alignment    | Monitor code changes and update architecture               | 3.1.9                   | Phase 1 - MVP      | Architecture never stale         |
| ARCH-006 | Visual Representation   | Display architecture as diagrams (components, connections) | 3.1.9                   | Phase 1 - MVP      | Easy understanding               |

### 1.10 Security Infrastructure

| Req #   | Requirement Name         | Requirement Description                                                     | Specification Reference | Planned Phase | Comment                  |
| ------- | ------------------------ | --------------------------------------------------------------------------- | ----------------------- | ------------- | ------------------------ |
| SEC-001 | Semgrep Integration      | Static analysis for OWASP Top 10 (SQL injection, XSS, CSRF, etc.)           | 3.1.12                  | Phase 1 - MVP | Code pattern security    |
| SEC-002 | Secrets Detection        | Detect hardcoded API keys, passwords, tokens (TruffleHog patterns)          | 3.1.12                  | Phase 1 - MVP | Prevent credential leaks |
| SEC-003 | Dependency Vuln Scanning | Check CVE database for vulnerable package versions                          | 3.1.12                  | Phase 1 - MVP | Supply chain security    |
| SEC-004 | License Compliance       | Verify license compatibility across dependencies                            | 3.1.12                  | Phase 1 - MVP | Legal compliance         |
| SEC-005 | Auto-Fix Patterns        | Automatically fix common issues (parameterized queries, escaping, env vars) | 3.1.12                  | Phase 1 - MVP | Reduce manual fixes      |
| SEC-006 | Parallel Scanning        | Run security scans with 4 workers concurrently                              | 3.1.12                  | Phase 1 - MVP | Fast feedback            |
| SEC-007 | Severity Blocking        | Block Critical/High issues, warn on Medium, log Low                         | 3.1.12                  | Phase 1 - MVP | Risk-based enforcement   |

---

## 2. AGENTIC LAYER

### 2.1 Agentic Framework (Four Pillars)

| Req #  | Requirement Name         | Requirement Description                                                                         | Specification Reference | Planned Phase | Comment           |
| ------ | ------------------------ | ----------------------------------------------------------------------------------------------- | ----------------------- | ------------- | ----------------- |
| AG-001 | PERCEIVE Primitives      | File operations, dependency analysis, code intelligence, test/validation, environment sensing   | 3.2, 3.3.1              | Phase 1 - MVP | Input layer       |
| AG-002 | REASON Primitives        | Pattern matching, risk assessment, architectural analysis, LLM consultation, confidence scoring | 3.2, 3.3.2              | Phase 1 - MVP | Decision layer    |
| AG-003 | ACT Primitives           | Code generation, file manipulation, test execution, deployment, browser automation, Git ops     | 3.2, 3.3.3              | Phase 1 - MVP | Output layer      |
| AG-004 | LEARN Primitives         | Pattern capture, feedback processing, Codex updates, analytics                                  | 3.2, 3.3.4              | Phase 1 - MVP | Improvement layer |
| AG-005 | Cross-Cutting Primitives | State management, context management, communication, error handling                             | 3.2, 3.3.5              | Phase 1 - MVP | Support functions |

### 2.2 Unified Tool Interface (UTI)

| Req #   | Requirement Name             | Requirement Description                                 | Specification Reference | Planned Phase      | Comment                     |
| ------- | ---------------------------- | ------------------------------------------------------- | ----------------------- | ------------------ | --------------------------- |
| UTI-001 | Protocol Abstraction         | Single API abstracting LSP, MCP, DAP, Builtin           | 3.3.0                   | Phase 1 - MVP      | Simplify tool integration   |
| UTI-002 | Protocol Selection Framework | Guidelines for when to use Builtin vs MCP vs LSP vs DAP | 3.3.0                   | Phase 1 - MVP      | Consistent tool decisions   |
| UTI-003 | LSP Support                  | Language server integration for code intelligence       | 3.3.0                   | Phase 1 - MVP      | Type checking, autocomplete |
| UTI-004 | MCP Support                  | Model Context Protocol for external tool integrations   | 3.3.0                   | Phase 1 - MVP      | Extensibility               |
| UTI-005 | DAP Support                  | Debug Adapter Protocol for debugging (Phase 2)          | 3.3.0                   | Phase 2 - Post-MVP | Interactive debugging       |
| UTI-006 | Builtin Tools                | Core tools implemented in Rust for performance          | 3.3.0                   | Phase 1 - MVP      | Fast, reliable operations   |

### 2.3 LLM Orchestration

| Req #   | Requirement Name       | Requirement Description                                  | Specification Reference | Planned Phase | Comment                         |
| ------- | ---------------------- | -------------------------------------------------------- | ----------------------- | ------------- | ------------------------------- |
| LLM-001 | Primary LLM Config     | Claude Sonnet 4 as default, user-configurable            | 3.4.1                   | Phase 1 - MVP | Flexibility in model choice     |
| LLM-002 | Secondary LLM Config   | GPT-4 Turbo as fallback, user-configurable               | 3.4.1                   | Phase 1 - MVP | Redundancy for reliability      |
| LLM-003 | Consultation Strategy  | Multi-LLM consultation after 2 failures                  | 3.4.1                   | Phase 1 - MVP | Improve success rate            |
| LLM-004 | Circuit Breaker        | Automatic fallback to secondary LLM on repeated failures | 3.4.1                   | Phase 1 - MVP | Fault tolerance                 |
| LLM-005 | Response Caching       | Cache LLM responses for identical inputs                 | 3.4.1                   | Phase 1 - MVP | Cost and latency optimization   |
| LLM-006 | Failover Sequence      | Primary → Retry → Secondary → Consultation → User        | 3.4.1                   | Phase 1 - MVP | Graceful degradation            |
| LLM-007 | Cost Tracking          | Track token usage and costs per operation                | 3.4.1                   | Phase 1 - MVP | Budget management               |
| LLM-008 | Model Allowlist        | Configurable list of allowed models                      | 3.4.1                   | Phase 1 - MVP | Control over AI usage           |
| LLM-009 | OpenRouter Integration | Use OpenRouter for LLM access with transparent pricing   | 3.4.1                   | Phase 1 - MVP | Unified API for multiple models |

### 2.4 Multi-LLM Team Orchestration (Phase 2)

| Req #    | Requirement Name        | Requirement Description                                                        | Specification Reference | Planned Phase       | Comment                  |
| -------- | ----------------------- | ------------------------------------------------------------------------------ | ----------------------- | ------------------- | ------------------------ |
| TEAM-001 | Lead Agent              | Coordinator agent for task decomposition and assignment                        | 3.4.1                   | Phase 2A - Post-MVP | Master role in team      |
| TEAM-002 | Specialist Agents       | Coding, Architecture, Testing, Documentation, UX agents with configurable LLMs | 3.4.1                   | Phase 2A - Post-MVP | Specialized roles        |
| TEAM-003 | Agent Coordination      | A2A protocol for inter-agent communication                                     | 3.4.1                   | Phase 2A - Post-MVP | Autonomous collaboration |
| TEAM-004 | Parallel Execution      | Multiple agents working on different files simultaneously                      | 3.4.1                   | Phase 2A - Post-MVP | 3-10x speedup            |
| TEAM-005 | Agent Instruction Files | Per-agent configuration and guidelines in .yantra/agents/                      | 3.4.1                   | Phase 2A - Post-MVP | Customizable behavior    |
| TEAM-006 | Performance Targets     | <30s Lead assignment overhead, <5ms file locks, <100ms messages                | 3.4.1                   | Phase 2A - Post-MVP | Efficient coordination   |

---

## 3. STATE MACHINES

### 3.1 Code Generation State Machine

| Req #     | Requirement Name             | Requirement Description                                                | Specification Reference   | Planned Phase                   | Comment                        |
| --------- | ---------------------------- | ---------------------------------------------------------------------- | ------------------------- | ------------------------------- | ------------------------------ |
| SM-CG-001 | ArchitectureGeneration State | Generate or import project architecture                                | 3.4.2.1 State 1           | Phase 1 - MVP                   | Foundation for all code        |
| SM-CG-002 | ArchitectureReview State     | Human approval gate for architecture changes                           | 3.4.2.1 State 2           | Phase 1 - MVP                   | Prevent unwanted changes       |
| SM-CG-003 | DependencyAssessment State   | Analyze package/tool requirements, CVE scan, web search integration    | 3.4.2.1 State 3           | Phase 1 - MVP                   | Prevent vulnerable deps        |
| SM-CG-004 | TaskDecomposition State      | Break feature into atomic tasks with file mappings                     | 3.4.2.1 State 4           | Phase 1 - MVP                   | Explicit planning              |
| SM-CG-005 | DependencySequencing State   | Topological sort for correct task execution order                      | 3.4.2.1 State 5           | Phase 1 - MVP                   | Prevent dependency violations  |
| SM-CG-006 | ConflictCheck State          | Query active work visibility (MVP) or file locks (Phase 2A)            | 3.4.2.1 State 6           | Phase 1 MVP / Phase 2A Enhanced | Coordination support           |
| SM-CG-007 | PlanGeneration State         | Create execution plan with estimates and priorities                    | 3.4.2.1 State 7           | Phase 1 - MVP                   | Scope clarity                  |
| SM-CG-008 | BlastRadiusAnalysis State    | Analyze impact before execution (P0 feature)                           | 3.4.2.1 State 8, 3.4.2.1A | Phase 1 - MVP                   | Informed decision-making       |
| SM-CG-009 | PlanReview State             | Optional approval for >5 tasks or multi-file changes                   | 3.4.2.1 State 9           | Phase 1 - MVP                   | Alignment on complex features  |
| SM-CG-010 | EnvironmentSetup State       | Create venv, install deps, validate setup                              | 3.4.2.1 State 10          | Phase 1 - MVP                   | Ready environment              |
| SM-CG-011 | FileLockAcquisition State    | Acquire file locks before editing (NOTE: Phase 2A with Team of Agents) | 3.4.2.1 State 11          | Phase 2A - Post-MVP             | Structural conflict prevention |
| SM-CG-012 | ContextAssembly State        | Load relevant context using hierarchical strategy                      | 3.4.2.1 State 12          | Phase 1 - MVP                   | Optimal LLM input              |
| SM-CG-013 | CodeGeneration State         | Generate code using Yantra Codex + Multi-LLM consultation              | 3.4.2.1 State 13          | Phase 1 - MVP                   | Core code generation           |
| SM-CG-014 | DependencyValidation State   | Validate against dependency graph for breaking changes                 | 3.4.2.1 State 14          | Phase 1 - MVP                   | Prevent breaks                 |
| SM-CG-015 | BrowserValidation State      | Validate UI in actual browser via CDP                                  | 3.4.2.1 State 15          | Phase 1 - MVP                   | Runtime UI verification        |
| SM-CG-016 | SecurityScanning State       | Semgrep, secrets, CVE, license checks                                  | 3.4.2.1 State 16          | Phase 1 - MVP                   | Security built-in              |
| SM-CG-017 | ConcurrencyValidation State  | Detect race conditions, deadlocks, data races                          | 3.4.2.1 State 17          | Phase 1 - MVP                   | Parallel code safety           |
| SM-CG-018 | FixingIssues State           | Auto-retry with fixes up to 3 attempts                                 | 3.4.2.1 State 18          | Phase 1 - MVP                   | Resilience                     |
| SM-CG-019 | FileLockRelease State        | Release locks on complete/failed (NOTE: Phase 2A with Team of Agents)  | 3.4.2.1 State 19          | Phase 2A - Post-MVP             | Resource cleanup               |
| SM-CG-020 | Complete State               | Code ready for testing, trigger Testing Machine                        | 3.4.2.1                   | Phase 1 - MVP                   | Success path                   |
| SM-CG-021 | Failed State                 | Human intervention required                                            | 3.4.2.1                   | Phase 1 - MVP                   | Escalation path                |

### 3.2 Test Intelligence State Machine

| Req #     | Requirement Name              | Requirement Description                                                                  | Specification Reference | Planned Phase | Comment                   |
| --------- | ----------------------------- | ---------------------------------------------------------------------------------------- | ----------------------- | ------------- | ------------------------- |
| SM-TI-001 | IntentSpecificationExtraction | Extract testable specs from user intent (solve Test Oracle Problem)                      | 3.4.2.2A State 1        | Phase 1 - MVP | Know what "correct" means |
| SM-TI-002 | TestOracleGeneration          | Generate verification strategies (spec-based, differential, metamorphic, contract-based) | 3.4.2.2A State 2        | Phase 1 - MVP | Proper test oracles       |
| SM-TI-003 | InputSpaceAnalysis            | Analyze input domains (boundary values, equivalence partitions, edge cases)              | 3.4.2.2A State 3        | Phase 1 - MVP | Representative test data  |
| SM-TI-004 | TestDataGeneration            | Generate test data (valid, invalid, boundary, random)                                    | 3.4.2.2A State 4        | Phase 1 - MVP | Comprehensive inputs      |
| SM-TI-005 | TestCaseGeneration            | Generate actual test code with assertions                                                | 3.4.2.2A State 5        | Phase 1 - MVP | Executable tests          |
| SM-TI-006 | AssertionStrengthAnalysis     | Verify assertions are strong (not weak like "is not None")                               | 3.4.2.2A State 6        | Phase 1 - MVP | Quality tests             |
| SM-TI-007 | TestQualityVerification       | Mutation testing for effectiveness (>80% mutation score)                                 | 3.4.2.2A State 7        | Phase 1 - MVP | Tests actually catch bugs |
| SM-TI-008 | TestSuiteOrganization         | Organize by module/feature, separate unit/integration/E2E                                | 3.4.2.2A State 8        | Phase 1 - MVP | Maintainable tests        |
| SM-TI-009 | TestImpactAnalysis            | Determine which tests affected by code changes (GNN-based)                               | 3.4.2.2A State 9        | Phase 1 - MVP | Efficient re-testing      |
| SM-TI-010 | TestUpdateGeneration          | Generate test updates when code changes (test-code co-evolution)                         | 3.4.2.2A State 10       | Phase 1 - MVP | Tests stay synchronized   |
| SM-TI-011 | Complete State                | High-quality test suite ready for execution                                              | 3.4.2.2A                | Phase 1 - MVP | Success path              |
| SM-TI-012 | Failed State                  | Unable to generate effective tests                                                       | 3.4.2.2A                | Phase 1 - MVP | Escalation                |

### 3.3 Test Execution State Machine

| Req #     | Requirement Name                | Requirement Description                                                   | Specification Reference | Planned Phase | Comment                          |
| --------- | ------------------------------- | ------------------------------------------------------------------------- | ----------------------- | ------------- | -------------------------------- |
| SM-TE-001 | EnvironmentSetup                | Create venv, install test dependencies (parallel)                         | 3.4.2.2B State 1        | Phase 1 - MVP | Ready test environment           |
| SM-TE-002 | FlakeDetectionSetup             | Configure flaky test detection (3 runs per test)                          | 3.4.2.2B State 2        | Phase 1 - MVP | Prevent non-deterministic blocks |
| SM-TE-003 | UnitTesting                     | Run pytest/jest with coverage and execution tracing (parallel 4+ workers) | 3.4.2.2B State 3        | Phase 1 - MVP | Function-level validation        |
| SM-TE-004 | IntegrationTesting              | Test API integrations, databases, contract validation (parallel)          | 3.4.2.2B State 4        | Phase 1 - MVP | System integration validation    |
| SM-TE-005 | BrowserTesting                  | Playwright E2E tests for user workflows (parallel 2-3 browsers)           | 3.4.2.2B State 5        | Phase 1 - MVP | Real user flow validation        |
| SM-TE-006 | PropertyBasedTesting            | Run hypothesis/fast-check for property tests (parallel)                   | 3.4.2.2B State 6        | Phase 1 - MVP | Invariant verification           |
| SM-TE-007 | ExecutionTraceAnalysis          | Analyze traces for failures (variable states, call stacks)                | 3.4.2.2B State 7        | Phase 1 - MVP | Actionable debugging info        |
| SM-TE-008 | FlakeDetectionAnalysis          | Identify and quarantine flaky tests (flakiness score >0.3)                | 3.4.2.2B State 8        | Phase 1 - MVP | Don't block on flaky tests       |
| SM-TE-009 | CoverageAnalysis                | Check coverage metrics (>80% target)                                      | 3.4.2.2B State 9        | Phase 1 - MVP | Adequate test coverage           |
| SM-TE-010 | SemanticCorrectnessVerification | Verify tests match intent (detect tautological assertions)                | 3.4.2.2B State 10       | Phase 1 - MVP | Meaningful tests                 |
| SM-TE-011 | ErrorClassification             | Classify failures (code bug, test bug, environmental, flaky, timeout)     | 3.4.2.2B State 11       | Phase 1 - MVP | Appropriate responses            |
| SM-TE-012 | FixingIssues                    | Auto-retry with fixes for test failures                                   | 3.4.2.2B State 12       | Phase 1 - MVP | Resilience                       |
| SM-TE-013 | TestCodeCoEvolutionCheck        | Verify tests still aligned with code after changes                        | 3.4.2.2B State 13       | Phase 1 - MVP | Prevent stale tests              |
| SM-TE-014 | Complete State                  | All tests pass, adequate coverage, semantic correctness verified          | 3.4.2.2B                | Phase 1 - MVP | Success path                     |
| SM-TE-015 | Failed State                    | Tests failed after max retries                                            | 3.4.2.2B                | Phase 1 - MVP | Escalation                       |

### 3.4 Deployment State Machine

| Req #      | Requirement Name  | Requirement Description                                          | Specification Reference | Planned Phase | Comment                  |
| ---------- | ----------------- | ---------------------------------------------------------------- | ----------------------- | ------------- | ------------------------ |
| SM-DEP-001 | PackageBuilding   | Create Docker image or build artifacts (parallel multi-stage)    | 3.4.2.3 State 1         | Phase 1 - MVP | Deployment package       |
| SM-DEP-002 | ConfigGeneration  | Generate railway.json, Dockerfile, environment config (parallel) | 3.4.2.3 State 2         | Phase 1 - MVP | Deployment configuration |
| SM-DEP-003 | RailwayUpload     | Push to Railway.app (parallel artifact/layer upload)             | 3.4.2.3 State 3         | Phase 1 - MVP | Deploy to platform       |
| SM-DEP-004 | HealthCheck       | Verify service responding (parallel endpoint checks)             | 3.4.2.3 State 4         | Phase 1 - MVP | Validate deployment      |
| SM-DEP-005 | RollbackOnFailure | Auto-rollback if health check fails                              | 3.4.2.3 State 5         | Phase 1 - MVP | Safety net               |
| SM-DEP-006 | Complete State    | Deployment successful with live URL                              | 3.4.2.3                 | Phase 1 - MVP | Success path             |
| SM-DEP-007 | Failed State      | Deployment failed, rollback triggered                            | 3.4.2.3                 | Phase 1 - MVP | Escalation               |

### 3.5 Maintenance State Machine (Phase 4)

| Req #      | Requirement Name  | Requirement Description                                                         | Specification Reference | Planned Phase      | Comment                    |
| ---------- | ----------------- | ------------------------------------------------------------------------------- | ----------------------- | ------------------ | -------------------------- |
| SM-MNT-001 | LiveMonitoring    | Continuous production monitoring (error rates, performance, API response times) | 3.4.2.4 State 1         | Phase 4 - Post-MVP | Detect issues <1s          |
| SM-MNT-002 | BrowserValidation | Real User Monitoring with session replay (parallel across sessions)             | 3.4.2.4 State 2         | Phase 4 - Post-MVP | User experience monitoring |
| SM-MNT-003 | ErrorAnalysis     | Pattern matching, severity classification (parallel patterns)                   | 3.4.2.4 State 3         | Phase 4 - Post-MVP | Root cause analysis        |
| SM-MNT-004 | IssueDetection    | Dependency graph tracing, blast radius (parallel paths)                         | 3.4.2.4 State 4         | Phase 4 - Post-MVP | Impact assessment          |
| SM-MNT-005 | AutoFixGeneration | LLM generates fix candidates (parallel candidates)                              | 3.4.2.4 State 5         | Phase 4 - Post-MVP | Automated remediation      |
| SM-MNT-006 | FixValidation     | Test in staging using CodeGen + Testing machines                                | 3.4.2.4 State 6         | Phase 4 - Post-MVP | Verify fix works           |
| SM-MNT-007 | CICDPipeline      | Automated deployment of fix (parallel deployment)                               | 3.4.2.4 State 7         | Phase 4 - Post-MVP | Push fix to production     |
| SM-MNT-008 | VerificationCheck | Confirm error rate drops (parallel regions)                                     | 3.4.2.4 State 8         | Phase 4 - Post-MVP | Verify fix effective       |
| SM-MNT-009 | LearningUpdate    | Update Yantra Codex with pattern (parallel stores)                              | 3.4.2.4 State 9         | Phase 4 - Post-MVP | Learn from incident        |
| SM-MNT-010 | Active State      | Normal monitoring, no incidents                                                 | 3.4.2.4 State 10        | Phase 4 - Post-MVP | Steady state               |
| SM-MNT-011 | Incident State    | Active incident being handled                                                   | 3.4.2.4 State 11        | Phase 4 - Post-MVP | Healing in progress        |

### 3.6 Documentation Governance State Machine (Phase 2)

| Req #      | Requirement Name      | Requirement Description                                | Specification Reference | Planned Phase      | Comment                    |
| ---------- | --------------------- | ------------------------------------------------------ | ----------------------- | ------------------ | -------------------------- |
| SM-DOC-001 | DocumentationAnalysis | Analyze what docs needed based on trigger events       | 3.4.2.5 State 1         | Phase 2 - Post-MVP | Documentation requirements |
| SM-DOC-002 | BlockIdentification   | Identify YDoc blocks to create/update                  | 3.4.2.5 State 2         | Phase 2 - Post-MVP | Target blocks              |
| SM-DOC-003 | ContentGeneration     | Generate/update documentation content using LLM        | 3.4.2.5 State 3         | Phase 2 - Post-MVP | Automated docs             |
| SM-DOC-004 | GraphLinking          | Create traceability edges in dependency graph          | 3.4.2.5 State 4         | Phase 2 - Post-MVP | Full traceability          |
| SM-DOC-005 | ConflictDetection     | Check for duplicate/conflicting docs (SSOT validation) | 3.4.2.5 State 5         | Phase 2 - Post-MVP | Single source of truth     |
| SM-DOC-006 | UserClarification     | Request user input for conflicts                       | 3.4.2.5 State 6         | Phase 2 - Post-MVP | Resolve ambiguities        |
| SM-DOC-007 | ConflictResolution    | Apply resolution or auto-resolve simple conflicts      | 3.4.2.5 State 7         | Phase 2 - Post-MVP | Fix conflicts              |
| SM-DOC-008 | Validation            | Verify documentation quality and completeness          | 3.4.2.5 State 8         | Phase 2 - Post-MVP | Quality assurance          |
| SM-DOC-009 | Complete State        | Documentation updated and synced                       | 3.4.2.5                 | Phase 2 - Post-MVP | Success path               |

---

## 4. SHARED SERVICES

### 4.1 CreationValidation Service

| Req #   | Requirement Name            | Requirement Description                                        | Specification Reference | Planned Phase | Comment                      |
| ------- | --------------------------- | -------------------------------------------------------------- | ----------------------- | ------------- | ---------------------------- |
| SVC-001 | Path Check Validation       | Check for exact file path duplicates                           | 3.4.3.1                 | Phase 1 - MVP | Prevent same path creation   |
| SVC-002 | Name Check Validation       | Check for function/class name duplicates in scope              | 3.4.3.1                 | Phase 1 - MVP | Prevent name collisions      |
| SVC-003 | Semantic Check Validation   | Detect semantic duplicates using embeddings (>0.85 similarity) | 3.4.3.1                 | Phase 1 - MVP | Content-based deduplication  |
| SVC-004 | Dependency Check Validation | Check for functional duplicates (same imports + functionality) | 3.4.3.1                 | Phase 1 - MVP | Prevent redundant code       |
| SVC-005 | De-Duplication Index        | Vector DB for similarity search with content hashes            | 3.4.3.1                 | Phase 1 - MVP | Fast duplicate detection     |
| SVC-006 | Resolution Options          | Present reuse/update/create options to user/agent              | 3.4.3.1                 | Phase 1 - MVP | Handle duplicates gracefully |

### 4.2 Browser Validation Service

| Req #   | Requirement Name       | Requirement Description                              | Specification Reference | Planned Phase | Comment                       |
| ------- | ---------------------- | ---------------------------------------------------- | ----------------------- | ------------- | ----------------------------- |
| SVC-007 | Browser Launch         | Launch Chrome/Chromium/Edge via CDP                  | 3.4.3.2                 | Phase 1 - MVP | Browser automation foundation |
| SVC-008 | Navigation             | Navigate to localhost URLs                           | 3.4.3.2                 | Phase 1 - MVP | Access application            |
| SVC-009 | Scenario Execution     | Execute test scenarios (clicks, fills, verification) | 3.4.3.2                 | Phase 1 - MVP | User flow testing             |
| SVC-010 | Console Error Capture  | Monitor and capture JavaScript errors                | 3.4.3.2                 | Phase 1 - MVP | Runtime error detection       |
| SVC-011 | Screenshot Capture     | Take screenshots for visual verification             | 3.4.3.2                 | Phase 1 - MVP | Visual testing                |
| SVC-012 | Performance Monitoring | Track load time, FPS, performance metrics            | 3.4.3.2                 | Phase 1 - MVP | Performance validation        |

### 4.3 SSOT Validation Service

| Req #   | Requirement Name         | Requirement Description                                  | Specification Reference | Planned Phase | Comment                           |
| ------- | ------------------------ | -------------------------------------------------------- | ----------------------- | ------------- | --------------------------------- |
| SVC-013 | Architecture SSOT        | Enforce single architecture document per project         | 3.4.3.3                 | Phase 1 - MVP | Prevent conflicting architectures |
| SVC-014 | Requirements SSOT        | Enforce single requirements doc per epic/feature         | 3.4.3.3                 | Phase 1 - MVP | Prevent conflicting requirements  |
| SVC-015 | API Specification SSOT   | Enforce single API spec per endpoint (OpenAPI canonical) | 3.4.3.3                 | Phase 1 - MVP | API consistency                   |
| SVC-016 | SSOT Conflict Resolution | User chooses primary when multiple found                 | 3.4.3.3                 | Phase 1 - MVP | Handle violations                 |

---

## 5. USER INTERFACE

### 5.1 Core UI Components

| Req #  | Requirement Name    | Requirement Description                             | Specification Reference | Planned Phase | Comment                    |
| ------ | ------------------- | --------------------------------------------------- | ----------------------- | ------------- | -------------------------- |
| UI-001 | File Explorer       | Browse project files with folder structure          | 3.1 Layer 5             | Phase 1 - MVP | File navigation            |
| UI-002 | Monaco Code Editor  | Syntax highlighting, autocomplete, error indicators | 3.1 Layer 5             | Phase 1 - MVP | Code editing               |
| UI-003 | Chat/Task Interface | Natural language input for tasks                    | 3.1 Layer 5             | Phase 1 - MVP | Primary interaction method |
| UI-004 | Dependency View     | Visualize dependency graph relationships            | 3.1 Layer 5             | Phase 1 - MVP | Understand connections     |
| UI-005 | Architecture View   | Display/edit architecture diagrams                  | 3.1 Layer 5, 3.1.9      | Phase 1 - MVP | Architecture visualization |
| UI-006 | Documentation View  | 4-panel layout (Features/Decisions/Changes/Tasks)   | 3.1 Layer 5, 3.1.10     | Phase 1 - MVP | Documentation access       |
| UI-007 | Browser Preview     | CDP-based live preview in system browser            | 3.1 Layer 5, 3.1.8      | Phase 1 - MVP | UI validation              |
| UI-008 | Terminal View       | Integrated terminal for command execution           | 3.1 Layer 5             | Phase 1 - MVP | Command access             |
| UI-009 | SolidJS Reactive UI | Fast, reactive UI framework                         | 3.1 Layer 5             | Phase 1 - MVP | Performance                |
| UI-010 | TailwindCSS Styling | Utility-first CSS framework                         | 3.1 Layer 5             | Phase 1 - MVP | Consistent styling         |
| UI-011 | Real-time WebSocket | Live updates from backend                           | 3.1 Layer 5             | Phase 1 - MVP | Instant feedback           |

### 5.2 Progress & Feedback

| Req #  | Requirement Name            | Requirement Description                                 | Specification Reference | Planned Phase | Comment              |
| ------ | --------------------------- | ------------------------------------------------------- | ----------------------- | ------------- | -------------------- |
| UI-012 | State Machine Progress Bars | Show progress for CodeGen, Testing, Deployment machines | 3.4.2G                  | Phase 1 - MVP | Visual feedback      |
| UI-013 | Work Visibility Indicators  | Show which files being modified by whom                 | 3.1.3                   | Phase 1 - MVP | Coordination support |
| UI-014 | Blast Radius Preview        | Show impact before executing changes                    | 3.4.2.1A                | Phase 1 - MVP | Informed decisions   |
| UI-015 | Real-time State Updates     | Show current state names and transitions                | 3.4.2G                  | Phase 1 - MVP | Transparency         |
| UI-016 | Confidence Scores           | Display confidence scores for generated code            | 3.4.1                   | Phase 1 - MVP | Quality indicator    |

---

## 6. ADVANCED FEATURES (POST-MVP)

### 6.1 Team of Agents (Phase 2A)

| Req #   | Requirement Name          | Requirement Description                                 | Specification Reference | Planned Phase       | Comment                      |
| ------- | ------------------------- | ------------------------------------------------------- | ----------------------- | ------------------- | ---------------------------- |
| ADV-001 | Lead Agent Implementation | Task decomposition and assignment coordinator           | 3.4.5.1                 | Phase 2A - Post-MVP | Team coordination            |
| ADV-002 | Specialist Agents         | Coding, Architecture, Testing, Documentation, UX agents | 3.4.5.1                 | Phase 2A - Post-MVP | Specialized roles            |
| ADV-003 | Git Coordination Branch   | Append-only event log (.yantra/coordination)            | 3.4.5.1                 | Phase 2A - Post-MVP | Coordination history         |
| ADV-004 | A2A Protocol              | Agent-to-agent communication for dependencies           | 3.4.5.1                 | Phase 2A - Post-MVP | Peer coordination            |
| ADV-005 | Agent Instruction Files   | Per-agent configuration in .yantra/agents/              | 3.4.5.1                 | Phase 2A - Post-MVP | Customizable behavior        |
| ADV-006 | Configurable Agent LLMs   | Each agent can use different LLM optimized for domain   | 3.4.5.1                 | Phase 2A - Post-MVP | Flexibility and optimization |
| ADV-007 | Parallel Agent Execution  | 3-10 agents working simultaneously                      | 3.4.5.1                 | Phase 2A - Post-MVP | 3-10x speedup                |

### 6.2 Cloud Graph Database (Phase 2B)

| Req #   | Requirement Name                 | Requirement Description                               | Specification Reference | Planned Phase       | Comment                    |
| ------- | -------------------------------- | ----------------------------------------------------- | ----------------------- | ------------------- | -------------------------- |
| ADV-008 | Tier 0 Cloud Storage             | Shared dependency graph in cloud (Redis + PostgreSQL) | 3.4.5.2                 | Phase 2B - Post-MVP | Cross-machine coordination |
| ADV-009 | Level 2 Direct Dep Detection     | Warn about direct dependency conflicts                | 3.4.5.2                 | Phase 2B - Post-MVP | Proactive warnings         |
| ADV-010 | Level 3 Transitive Dep Detection | Warn about transitive dependency conflicts            | 3.4.5.2                 | Phase 2B - Post-MVP | Deep conflict detection    |
| ADV-011 | Level 4 Semantic Dep Detection   | Warn about function signature changes                 | 3.4.5.2                 | Phase 2B - Post-MVP | API change awareness       |
| ADV-012 | Privacy-Preserving Sync          | Only sync graph structure, not code content           | 3.4.5.2                 | Phase 2B - Post-MVP | Security                   |
| ADV-013 | WebSocket/gRPC Protocol          | Low-latency communication (<50ms)                     | 3.4.5.2                 | Phase 2B - Post-MVP | Real-time updates          |
| ADV-014 | Per-Project Isolation            | Separate graph storage per project                    | 3.4.5.2                 | Phase 2B - Post-MVP | Multi-project support      |

### 6.3 Clean Code Mode (Phase 2C)

| Req #   | Requirement Name      | Requirement Description                                            | Specification Reference | Planned Phase       | Comment               |
| ------- | --------------------- | ------------------------------------------------------------------ | ----------------------- | ------------------- | --------------------- |
| ADV-015 | Dead Code Detection   | Detect unused functions, classes, imports, variables               | 3.4.5.3                 | Phase 2C - Post-MVP | Code cleanup          |
| ADV-016 | Auto-Remove Threshold | Only auto-remove with >80% confidence                              | 3.4.5.3                 | Phase 2C - Post-MVP | Safety                |
| ADV-017 | Real-Time Refactoring | Extract duplicates, simplify complex functions, rename for clarity | 3.4.5.3                 | Phase 2C - Post-MVP | Code quality          |
| ADV-018 | Component Hardening   | Security, performance, quality, dependency hardening               | 3.4.5.3                 | Phase 2C - Post-MVP | Strengthen code       |
| ADV-019 | Continuous Mode       | Background process with configurable intervals                     | 3.4.5.3                 | Phase 2C - Post-MVP | Automated maintenance |

### 6.4 Enterprise Automation (Phase 3)

| Req #   | Requirement Name          | Requirement Description                                  | Specification Reference | Planned Phase      | Comment               |
| ------- | ------------------------- | -------------------------------------------------------- | ----------------------- | ------------------ | --------------------- |
| ADV-020 | Cross-System Intelligence | Track external APIs (Stripe, Salesforce, etc.)           | 3.4.5.4                 | Phase 3 - Post-MVP | External dependencies |
| ADV-021 | Browser Automation Full   | Playwright integration for legacy systems                | 3.4.5.4                 | Phase 3 - Post-MVP | Legacy integration    |
| ADV-022 | Self-Healing Systems      | API monitoring, schema drift detection, auto-migration   | 3.4.5.4                 | Phase 3 - Post-MVP | Resilient systems     |
| ADV-023 | Multi-Language Support    | JavaScript/TypeScript parser, cross-language deps        | 3.4.5.4                 | Phase 3 - Post-MVP | Polyglot support      |
| ADV-024 | Enterprise Features       | Multitenancy, user accounts, team collaboration, billing | 3.4.5.4                 | Phase 3 - Post-MVP | Enterprise readiness  |

### 6.5 Platform Maturity (Phase 4)

| Req #   | Requirement Name         | Requirement Description                              | Specification Reference | Planned Phase      | Comment             |
| ------- | ------------------------ | ---------------------------------------------------- | ----------------------- | ------------------ | ------------------- |
| ADV-025 | Performance Optimization | GNN queries <100ms for 100k+ LOC, distributed GNN    | 3.4.5.5                 | Phase 4 - Post-MVP | Scalability         |
| ADV-026 | Advanced Refactoring     | Architectural refactoring (monolith → microservices) | 3.4.5.5                 | Phase 4 - Post-MVP | Large-scale changes |
| ADV-027 | Plugin Ecosystem         | Plugin system, marketplace, CLI, REST API, SDKs      | 3.4.5.5                 | Phase 4 - Post-MVP | Extensibility       |
| ADV-028 | Enterprise Deployment    | On-premise, air-gapped, private cloud, SLAs          | 3.4.5.5                 | Phase 4 - Post-MVP | Enterprise options  |

---

## SUMMARY BY PHASE

### Phase 1 - MVP (Requirements Count: ~200+)

- Infrastructure Layer: Language support, Dependency graph, YDoc, Context, Codex, Storage, Browser, Architecture, Security
- Agentic Layer: Four pillars, UTI, LLM orchestration
- State Machines: CodeGen (18 states), Test Intelligence (10 states), Test Execution (13 states), Deployment (6 states)
- Shared Services: CreationValidation, Browser Validation, SSOT Validation
- UI: Core components, progress indicators, feedback

### Phase 2A - Team of Agents (Requirements Count: ~30)

- Lead Agent + Specialist Agents
- Git Coordination Branch
- A2A Protocol
- File Locking Mechanism (NOTE: Deferred from Phase 1)
- Tier 2 (sled) for agent coordination

### Phase 2B - Cloud Graph Database (Requirements Count: ~20)

- Tier 0 Cloud Storage
- 4 levels of conflict detection
- Privacy-preserving sync
- Multi-machine coordination

### Phase 2C - Clean Code Mode (Requirements Count: ~15)

- Dead code detection
- Real-time refactoring
- Component hardening
- Continuous maintenance

### Phase 3 - Enterprise Automation (Requirements Count: ~20)

- Cross-system intelligence
- Browser automation (Playwright)
- Self-healing systems
- Multi-language support
- Enterprise features

### Phase 4 - Platform Maturity (Requirements Count: ~15)

- Performance optimization
- Advanced refactoring
- Plugin ecosystem
- Enterprise deployment
- Maintenance State Machine (11 states)

### Phase 2 - Documentation Governance (Requirements Count: ~10)

- Documentation Governance State Machine (9 states)
- Parallel execution with other machines
- Full traceability maintenance

---

**Total Requirements: ~300+**

**Note** : Status column intentionally left blank for tracking during implementation.S
