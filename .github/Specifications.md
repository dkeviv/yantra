# Yantra: Complete Technical Specification - Revised

| Version | Date     | Changes                                                                              |
| ------- | -------- | ------------------------------------------------------------------------------------ |
| 1.0     | Nov 15th | Specifications_old -> Archived                                                       |
| 2.0     | Dec 6th  | Revised , cleaned up Specifications                                                  |
| 3.0     | Dec 7th  | Updated Agentic workflows w.r.t better Documentation creation and governance         |
| 4.0     | Dec 8th  | Updated Ydoc system and state machines to incorporate Ydoc system                    |
| 5.0     | Dec 9th  | Conversational Memory and advanced Ydoc components                                   |
| 6.0     | Dec 9th  | Complete agentic primitives update - added all missing primitives from original spec |

# 1. Executive Summary

## 1.1 The Vision

Yantra is a fully autonomous agentic developer - an AI-powered platform that doesn't just generate code, but executes the complete software development lifecycle: from understanding requirements to deploying and monitoring production systems.

Traditional AI Code Assistants: Help developers write code faster Yantra: Replaces the entire development workflow with autonomous agents

Unlike traditional IDEs that assist developers or AI tools that suggest code, Yantra makes artificial intelligence the primary developer, with humans providing intent, oversight, and approvals only for critical decisions.

What "Fully Autonomous Agentic" Means

Not autonomous: LLM generates code → Developer manually tests → Developer fixes issues → Developer commits Partially autonomous: LLM generates code → System validates → Developer fixes issues Fully autonomous (Yantra): LLM generates code → System validates → System fixes issues → System tests → System packages → System deploys → System monitors → Repeat until perfect

Yantra handles the complete pipeline:

1. 🎯 Understand: Parse natural language requirements
2. 🔨 Build: Generate production-quality code
3. ✅ Validate: Run dependency checks, tests, security scans
4. 🔄 Fix: Auto-retry with intelligent error analysis
5. ▶️ Execute: Run the code with proper environment setup
6. 📦 Package: Build distributable artifacts (wheels, Docker images, npm packages)
7. 🚀 Deploy: Push to production (AWS, GCP, Kubernetes, Heroku)
8. 📊 Monitor: Track performance and errors in production
9. 🔧 Heal: Auto-fix production issues without human intervention

Human role: Provide intent ("Add payment processing"), review critical changes, approve deployments

## 1.2 The Problem We Solve

For Developers:

- 40-60% of development time spent debugging
- Code breaks production despite passing tests
- Integration failures when APIs change
- Repetitive coding tasks (CRUD, auth, APIs)
- Context switching between IDE, terminal, browser, deployment tools
- Manual deployment and rollback procedures
- Production firefighting and hotfix cycles

For Engineering Teams:

- Unpredictable delivery timelines
- Inconsistent code quality
- High maintenance costs
- Technical debt accumulation
- Slow time-to-market (weeks for simple features)
- DevOps bottlenecks

For Enterprises:

- Manual workflow automation (expensive, error-prone)
- Siloed systems (Slack, Salesforce, internal tools don't talk)
- Workflow tools (Zapier) can't access internal code or execute complex logic
- System breaks cascade across services
- Browser automation requires specialized developers
- No self-healing - every outage requires manual intervention

## 1.3 The Solution

Code That Never Breaks + Autonomous Execution

- AI generates code with full dependency awareness
- Automated unit + integration testing
- Security vulnerability scanning
- Browser runtime validation
- Autonomous code execution with environment setup
- Integrated terminal for command execution
- Real-time output streaming to UI
- Git integration for seamless commits

Team of Agents & Cloud Graph Database + Package/Deploy

- Team Agent Architecture - Master-Servant pattern with Git coordination branch for multi-agent parallelism
- Cloud Graph Database (Tier 0)- Shared dependency graph for proactive conflict prevention across agents and team members
  - Note: This is NOT "Cloud GNN" - the GNN (intelligence layer) runs locally. This is cloud-hosted graph database storage for coordination.
- Package building (Python wheels, Docker, npm)
- Automated deployment (AWS, GCP, Kubernetes, Heroku)
- Health checks and auto-rollback
- Generate workflows from natural language
- Scheduled jobs and event triggers
- Multi-step orchestration with error handling and retries
- CI/CD pipeline generation

Enterprise Automation & Self-Healing

- Cross-system dependency tracking
- External API monitoring and auto-healing
- Production monitoring with auto-remediation
- Browser automation for enterprise workflows
- Legacy system integration via browser control
- Multi-language support (Python + JavaScript + TypeScript)
- Infrastructure as Code generation (🆕)

Platform Maturity & Ecosystem

- Plugin ecosystem and marketplace
- Advanced refactoring and performance optimization
- Enterprise deployment (on-premise, cloud, air-gapped)
- SLA guarantees (99.9% uptime)
- Multi-tenant enterprise features

## 1.4 Competitive Advantage

| Capability                                                    | Yantra | Copilot | Cursor | Zapier | Replit Agent |
| ------------------------------------------------------------- | ------ | ------- | ------ | ------ | ------------ |
| ------------------------------------------------------------- |        |         |        |        |              |
| Dependency-aware generation                                   | ✅     | ❌      | ❌     | N/A    | ❌           |
| -                                                             | -      | -       | -      | -      | -            |
| Guaranteed no breaks                                          | ✅     | ❌      | ❌     | ❌     | ❌           |
| Truly unlimited context                                       | ✅     | ❌      | ❌     | N/A    | ❌           |
| Token-aware context                                           | ✅     | ⚠️      | ⚠️     | N/A    | ❌           |
| Automated testing                                             | ✅     | ❌      | ❌     | ❌     | ⚠️           |
| Agentic validation pipeline                                   | ✅     | ❌      | ❌     | ❌     | ❌           |
| Autonomous code execution                                     | ✅     | ❌      | ❌     | ⚪     | ✅           |
| Package building                                              | ✅     | ❌      | ❌     | ❌     | ⚠️           |
| Automated deployment                                          | ✅     | ❌      | ❌     | ⚪     | ✅           |
| Production monitoring                                         | ✅     | ❌      | ❌     | ❌     | ❌           |
| Self-healing systems                                          | ✅     | ❌      | ❌     | ❌     | ❌           |
| Network effect (failures)                                     | ✅     | ❌      | ❌     | ❌     | ❌           |
| Works with any LLM                                            | ✅     | ❌      | ⚠️     | N/A    | ❌           |
| Internal system access                                        | ✅     | ⚠️      | ⚠️     | ❌     | ⚠️           |
| Custom workflow code                                          | ✅     | ❌      | ❌     | ❌     | ⚠️           |
| Browser automation                                            | ✅     | ❌      | ❌     | ❌     | ❌           |
| Integrated terminal                                           | ✅     | ✅      | ❌     | N/A    | ✅           |
| Desktop app (native)                                          | ✅     | N/A     | ✅     | N/A    | ❌ (web)     |

## 1.5 ROADMAP

| Phase 1 - MVP                                                                                                                                                                                                                                                                                                                                                                                                                                               | Phase 2A                                                                                                                                                                                                                                                                                                                                                                                                                 | Phase 2B                                                                                                                               | Phase 3                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Dependency Graph and View1. File, Packages, Code1. No Function chaining1. Architecture View1. Editor1. Documentation System View1. Ydoc System1. Chat1. LLM integration1. LLM consulting mode1. Interaction Mode1. Agent Codegen & Testing1. State Machines - Codegen Testing, Deploy1. Local Yantra Codex (stretch)1. Agentic Deploy with Railway1. Agentic Primitives and Orchestration1. Security vulnerability scanning1. Browser runtime validation | 1. Team of Agents1. Cloud Graph DB (Tier 0)1. Cloud Yantra Codex1. Ydoc - Documentation system - Requirement to Code dependency1. Additional Agentic Deploy (AWS/GCP)1. External API dependency1. External API monitoring and MCP self healing1. CI/CD pipeline generation1. Package building (Python wheels, Docker, npm) 1. Automated deployment (AWS, GCP, Kubernetes, Heroku) 1. Infrastructure as Code generation`` | 1. Agent Monitoring primitives1. Self Healing -Production monitoring with auto-remediation 1. State Machine - Maintenance1. Clean Mode | 1. Monetization1. Plugin ecosystem and marketplace1. Advanced refactoring and performance optimization1. Enterprise deployment (on-premise, cloud, air-gapped)1. SLA guarantees (99.9% uptime)1. Multi-tenant enterprise features`` |

## 1.6 KEY DIFFERENTIATORS

Yantra Develop represents a paradigm shift in software development through eight core capabilities that distinguish it from existing IDE and AI-assisted coding tools.

### 1.6.1 Documentation Governance and Complete Traceability

Yantra maintains bidirectional traceability across the entire development lifecycle. Every requirement traces to architecture decisions, specifications, implementation code, tests, and issues through an integrated documentation system. The platform uses YDoc, a block-based documentation format with unique identifiers for each requirement, specification, or decision. These blocks link directly to code files, functions, and test cases through the dependency graph, creating an auditable chain from user intent to deployed code. When code changes, the system automatically identifies affected documentation and prompts updates, ensuring single source of truth across all artifacts.

### 1.6.2 Multi-Level Dependency Graph with Semantic Search

Beyond traditional file-level dependencies, Yantra's Graph Neural Network tracks relationships at multiple granularities: files, functions, classes, packages, external tools, and documentation blocks. The graph uses petgraph for traversal with HNSW indexing for fast semantic search. This enables impact analysis that answers questions like "what breaks if I change this function" or "what documentation needs updating for this code change" in under 100 milliseconds. The graph tracks not just import relationships but semantic connections like "implements specification SPEC-001" or "tested by test_auth_flow."

### 1.6.3 Yantra Codex Learning-Based Code Generation

Yantra Codex is a specialized GraphSAGE-based neural network trained on project-specific patterns and corrections. Unlike generic LLM code generation, the Codex learns from three sources: developer-approved code patterns, bug-fix pairs captured during maintenance, and LLM mistakes that required correction. The system builds 1024-dimensional embeddings of code patterns and problem contexts, generating code with confidence scores. When confidence drops below 0.8, the system falls back to LLM consultation rather than generating potentially incorrect code.

### 1.6.4 Multi-Source Learning from Failures

The platform captures and learns from three types of failures. First, bugs discovered during testing or production are paired with their fixes and stored as training data. Second, test failures that require code correction are analyzed to understand why generated code didn't match intent. Third, LLM mistakes where consultation mode provides incorrect solutions are logged with correct alternatives. This creates a continuously improving system where common mistakes are prevented through pattern recognition rather than repeated LLM consultation.

### 1.6.5 LLM Consulting Mode with Circuit Breakers

Yantra uses multiple LLMs strategically rather than depending on a single model. Claude Sonnet 4 serves as the primary generator for code and reasoning tasks. GPT-4 acts as a validator and fallback when Claude fails or produces low-confidence results. The system implements circuit breakers that detect repeated failures, exponential backoff for temporary issues, and model routing based on task type. This multi-LLM approach provides 99.9% availability even when individual models experience outages or degraded performance.

### 1.6.6 Integrated Deployment Pipeline

Code generation includes deployment configuration as a first-class concern. The system automatically generates Dockerfiles, Railway configuration, environment variables, and health check endpoints. Deployment happens through Railway's API with automatic rollback on failed health checks. The platform monitors deployed applications through health endpoints and can trigger self-healing workflows when issues are detected. This eliminates the gap between "code that works locally" and "code that works in production."

### 1.6.7 Self-Healing with Browser Integration

Yantra uses Chrome DevTools Protocol to launch actual browsers and validate UI behavior. The system takes screenshots, compares them to expected layouts, monitors console errors, and tests user interactions. When visual regressions or runtime errors are detected, the platform generates fixes automatically and re-validates. This creates a feedback loop where UI bugs are caught and fixed before human review, ensuring generated interfaces work correctly in real browsers.

### 1.6.8 Preventive Development Cycle

Rather than fixing bugs after they occur, Yantra validates correctness at every step. The Preventive Development Cycle includes five phases: Architect validates alignment with system design, Plan ensures task dependencies are correct, Execute validates syntax and logic before committing, Deploy confirms health checks pass before marking complete, and Maintain monitors production for issues. Each phase includes validation gates that prevent broken code from progressing. The dependency graph detects breaking changes, the browser validates UI correctness, security scanning catches vulnerabilities, and architecture validation ensures consistency.

These eight differentiators work together to create code that "never breaks" through comprehensive prevention rather than reactive debugging.

---

# Yantra: Complete Technical Specification - Part 1

Version: 4.0 Date: December 2025 Status: Final

---

## 2. COMPLETE ARCHITECTURE

### 2.1 YANTRA PLATFORM - 5 LAYER OVERVIEW

LAYER 1: INFRASTRUCTURE

1.1 Language Support ├─ Monaco Editor ├─ AST Parsing: Tree-sitter (Python, JS, Rust, Go, etc.) ├─ Multi-language Support └─ AutoCompletion

1.2 Dependency Graph (petgraph-based, misnomer "GNN" in code) ├─ Graph Structure: Nodes (files/funcs/classes/packages/tools), Edges (deps) ├─ Query Engine: <1ms dependency lookups ├─ Incremental Updates: <50ms per file change ├─ Impact Analysis: Transitive dependency traversal └─ Storage: In-memory (hot) + SQLite (persistence)

1.3 Extended Dependency Graph with Pattern/Template Nodes ├─ Code Templates: Pre-validated patterns ├─ Best Practices: Language-specific idioms ├─ Project Patterns: Learned from codebase └─ Semantic Search: <10ms retrieval for context assembly

1.4 YDoc Documentation System ├─ Block Database (canonical source) ├─ Graph-native (every block links to requirements, code, docs) ├─ Git-friendly (exportable, diffable, conflict-detectable) └─ Full traceability chain

1.5 Unlimited Context Solution ├─ Token Counting: Track context limits per LLM ├─ Hierarchical Assembly: Priority-based context inclusion (code + conversation) ├─ Compression: Summarize low-priority context ├─ Chunking: Split large operations across multiple calls ├─ Adaptive Strategies: Dynamic context based on task type └─ Conversation Integration: Recent + relevant messages from chat history

1.6 Yantra Codex (AI Code Generation) - GraphSAGE GNN (actual neural network) ├─ Neural Network: 1024-dim embeddings, 150M parameters ├─ Inference: 15ms (CPU), 5ms (GPU), ~600MB model ├─ Pattern Recognition: 978-dim problem features → code logic ├─ Confidence Scoring: 0.0-1.0 (triggers LLM review < 0.8) ├─ Continuous Learning: Learns from LLM corrections └─ Storage: SEPARATE database (.yantra/codex.db, ~500MB) [NOT in Tier 1]

1.7 Storage Architecture (Multi-Tier + Separate Codex) ├─ Tier 0: Cloud Graph DB (PostgreSQL + Redis) - Team coordination [Phase 2B] ├─ Tier 1: petgraph (in-memory) + SQLite - Dependency graph + YDoc + Conversation [MVP] ├─ Tier 2: sled - Agent coordination (local multi-agent) [Phase 2A] ├─ Tier 3: TOML files - Configuration [MVP] ├─ Tier 4: HashMap → moka - Context cache (ephemeral) [MVP] └─ Codex: SQLite + HNSW - Pattern database (SEPARATE, ~500MB) [MVP]

1.8 Storage Optimization ├─ HNSW Semantic Indexing ├─ SQLite with WAL mode └─ Connection pooling

1.9 Security Infrastructure ├─ Semgrep integration ├─ Secrets management └─ Dependency auditing

1.10 Browser Infrastructure (CDP) ├─ Launch, navigate, control └─ Real-time monitoring

1.11 Architecture View System ├─ Agent-driven architecture generation ├─ Deviation detection ├─ Version history (Rule of 3) └─ Continuous alignment monitoring

1.12 Documentation System ├─ Features tracking ├─ Decisions logging ├─ Changes tracking └─ Task management

1.13 Conversation Memory System ├─ Persistent message storage (every user/agent message) ├─ Adaptive context retrieval (recent + relevant messages) ├─ Semantic search (keyword + meaning-based) ├─ Work session linking (chat → code → tests → deploy) └─ Storage: Tier 1 SQLite (.yantra/state.db)

LAYER 2: AGENTIC FRAMEWORK

Philosophy: Four Pillars

1. PERCEIVE - Sense the environment
2. REASON - Analyze and decide
3. ACT - Execute actions
4. LEARN - Adapt from feedback

LAYER 3: AGENTIC PRIMITIVES/TOOLS

Definition: Low-level autonomous ACTIONS the agent performs

3.1 Unified Tool Interface (UTI) ├─ Protocol Router: MCP / LSP / DAP / Builtin ├─ Tool Adapters: 45+ tools, 4 protocols ├─ Consumer Abstraction: LLM Agent + Workflow Executor └─ Protocol Selection: Auto-routing by capability

LAYER 4: AGENTIC ORCHESTRATION

Definition: HOW agents coordinate primitives

4.1 LLM Orchestration ├─ Primary: Claude Sonnet 4 (code generation, reasoning) ├─ Secondary: GPT-4 Turbo (validation, fallback) ├─ Routing: Cost optimization, capability-based selection ├─ Failover: Circuit breaker, retry with exponential backoff └─ Response Caching: Redis for repeated queries

4.2 State Machine (Yantra Preventive Development Cycle) ├─ Phase Transitions: Architect → Plan → Execute → Deploy → Maintain ├─ State Persistence: SQLite with WAL mode ├─ Rollback Support: Checkpoints at phase boundaries └─ Approval Gates: Human-in-loop for critical operations

4.3 Agent Execution Intelligence ├─ Command classification ├─ Background execution ├─ Status transparency ├─ Parallel Processing ├─ Execution Mode - Clean Mode (with refactoring and hardening) └─ Interaction Modes - Guided Mode/Auto Mode

4.4 Dependency Intelligence & Environment Management ├─ Mandatory .venv isolation ├─ Dependency assessment ├─ Dry-run validation ├─ Tech Stack tracking in dependency graph ├─ Conflict detection and resolution └─ Web search for latest package info

4.5 Preventive Development Cycle (PDC) ├─ Phase 1: Architect/Design ├─ Phase 2: Plan ├─ Phase 3: Execute ├─ Phase 4: Deploy └─ Phase 5: Monitor/Maintain

LAYER 5: USER INTERFACE

5.1 Desktop Application (Tauri) 5.2 Monaco Editor 5.3 Architecture View 5.4 Documentation Panels (Features, Decisions, Changes, Tasks) 5.5 Terminal Integration 5.6 Real-time Updates (WebSocket)

---

## 3. REQUIREMENTS

### 3.1 INFRASTRUCTURE

#### 3.1.1 Language Support and Auto-Completion

Provide syntax highlighting, code completion, and language intelligence using Monaco Editor, Tree-sitter parsers, and LSP integration via Unified Tool Interface.

Supported Languages (MVP): Python, Rust with full LSP support. JavaScript/TypeScript in later phases.

Auto-Completion Sources:

1. LSP-provided completions from language servers
2. Context-aware suggestions from dependency graph
3. Yantra Codex learned patterns
4. Project-specific imports and symbols

Three-Tier Completion System:

Tier 1: Static Completions (Instant, No Network)

1. Hardcoded language snippets (20+ per language)
2. Python: if/else, for/while, try/except, class, def, list comprehension, context manager, decorator, async/await, type hints
3. JavaScript: function, const fn, class, if/else, for/while, try/catch, async/await, Promise, import/export, console methods, array methods
4. Response time: <10ms (instant), no network required, always available

Tier 2: Dependency Graph Completions (Fast, Local)

1. Project symbols (functions, classes, variables)
2. Import suggestions from project files
3. Function signatures with documentation
4. Recent symbols (last 10 files edited)
5. Response time: <50ms, 100% local (no data leaves machine), works offline

Tier 3: LLM Completions (Smart, Requires Network)

1. Context-aware multi-line completions
2. Natural language to code
3. Bug fixes and refactoring
4. Documentation generation
5. Response time: 1-3 seconds, user code sent to LLM (Claude/GPT-4), requires explicit user consent, settings option to disable

Fallback Strategy: Tier 3 fails → Tier 2 → Tier 1 (always works).

Privacy & Security: LLM completions require explicit user consent, settings option to disable, clear indication when LLM is being called. Dependency graph and static completions are 100% local with no data leaving machine.

Tree-sitter Integration: Fast incremental parsing for building dependency graph. Each language requires parser installation, query definitions for extracting imports/functions/classes, real-time parsing on file changes, AST navigation for dependency analysis.

Performance Requirements: Auto-completion latency below 100ms for files under 1000 lines, syntax highlighting updates within 50ms of keystroke, LSP response time under 200ms.

#### 3.1.2 Dependency Graph

Track all project dependencies to enable proactive conflict prevention and intelligent code generation.

CRITICAL NOTE: In codebase, this is referred to as "GNN" (Graph Neural Network) which is a MISNOMER. This is a traditional dependency graph using petgraph data structure, NOT a neural network. The actual GNN is Yantra Codex (Section 3.1.6) which uses GraphSAGE neural networks for code generation learning.

Technology Stack: petgraph library for graph data structure, HNSW indexing for semantic search, Tree-sitter for parsing source files, SQLite for persistent storage.

Comprehensive Dependency Tracking (ALL Bidirectional):

1. File ↔ File Dependencies (MVP)
2. Import relationships (which files import which)
3. Module dependencies (which modules depend on which)
4. Test file ↔ Source file relationships
5. Source file ↔ Documentation links
6. Version tracking: File content hash, last modified timestamp
7. File ↔ Code Symbol ↔ Code Symbol Dependencies (MVP)
8. Function → Function calls (caller/callee relationships)
9. Class → Class inheritance (parent/child hierarchies)
10. Method → Method invocations (cross-class calls)
11. Variable → Variable usage (data flow tracking)
12. Version tracking: Symbol signature hash, parameter changes
13. Transitive tracking across call chains
14. Package ↔ Package Dependencies (MVP)
15. Direct dependencies (package.json, requirements.txt, Cargo.toml)
16. Transitive dependencies (full dependency tree)
17. Peer dependencies and optional dependencies
18. Version tracking: Exact version (numpy==1.24.0), version range (>=1.26,<2.0), compatibility matrix
19. Version history tracking (upgraded from X to Y)

CRITICAL: Version-Level Tracking

1. Track EXACT versions for all packages (numpy==1.26.0, pandas==2.1.0, not just "numpy", "pandas")
2. Track version requirements for dependencies (requires: "numpy>=1.26,<2.0")
3. Track version history (upgraded from 1.24.0 to 1.26.0 on date)
4. WRONG: Track "numpy" as single node → cannot detect version conflicts
5. CORRECT: Track "numpy==1.24.0" and "numpy==1.26.0" as separate nodes → detect incompatibilities
6. Tool ↔ Tool Dependencies
7. Build tool chains (webpack → babel → terser)
8. Test framework dependencies (pytest → coverage → plugins)
9. Linter/formatter chains (ESLint → Prettier → plugins)
10. Version tracking: Tool version, plugin versions, config file hash
11. Package ↔ File Dependencies
12. Which files use which packages (import numpy → file.py uses numpy)
13. Unused package detection (packages installed but never imported)
14. Package-to-module mapping (numpy → specific submodules used)
15. Version tracking: Import statement location, package version used
16. Nested function tracking (numpy.random.normal, not just numpy.array)
17. User ↔ File Dependencies (MVP)
18. Active work tracking (which developer is editing which files)
19. File modification history (who last modified, when)
20. Work visibility indicators (show active sessions on files)
21. Version tracking: User session ID, file version at edit start
22. User ↔ Git Checkout Dependencies (Post-MVP)
23. Branch-to-file mapping (which files changed in which branches)
24. Merge conflict prediction (parallel edits on same files)
25. Work isolation tracking (developer workspace state)
26. Version tracking: Git commit SHA, branch name, checkout timestamp
27. File/Code ↔ External API (Post-MVP)
28. API endpoint dependencies
29. External service tracking
30. API version monitoring
31. Function ↔ Package Function (Post-MVP)
32. Which specific package functions are used
33. Function-level dependency tracking
34. Method Chain Tracking (Post-MVP)
35. Track df.groupby().agg().reset_index() level granularity
36. Complex method chain dependencies

Graph Node Types:

1. CodeFile: Represents entire source file
2. Function: Individual function/method
3. Class: Class definitions
4. Module: Package/module boundaries
5. Package: External packages with exact versions (numpy==1.24.0)
6. Tool: Build tools, linters, formatters
7. User: Developers working on project
8. YDocDocument: Documentation files
9. YDocBlock: Individual documentation blocks

Graph Edge Types:

1. Imports: File A imports File B
2. Calls: Function A calls Function B
3. Depends: Module A depends on Module B
4. UsesPackage: File uses Package@Version
5. DependsOn: Package A depends on Package B (transitive)
6. UsesTool: Project uses Tool@Version
7. TracesTo: Requirement block traces to Architecture block
8. Implements: Architecture block implements Specification block
9. RealizedIn: Specification block realized in code file
10. TestedBy: Requirement tested by test file
11. DocumentedIn: Code documented in documentation block
12. HasIssue: Code file has issue documented in Change Log
13. EditedBy: User editing file (active work tracking)

Graph Construction Process:

1. Parse all source files using Tree-sitter
2. Extract imports, function definitions, class definitions, function calls
3. Create nodes for each entity (files, functions, classes, packages with versions)
4. Create edges based on relationships (imports, calls, dependencies)
5. Index for fast querying with HNSW for semantic search
6. Track all edges bidirectionally (navigate from caller→callee AND callee→caller)

Metadata Stored:

1. Node metadata: name, type, file path, line ranges, version info, timestamps
2. Edge metadata: relationship type, confidence score, usage frequency
3. All edges bidirectional for reverse traversal

Query Capabilities:

1. Find all files importing a given file
2. Find all functions called by a function
3. Find impact of changing a function (all callers)
4. Find files implementing a requirement
5. Find documentation for code file
6. Find tests covering a requirement
7. Find all files using a specific package version
8. Find downstream dependencies for package upgrade
9. Find which developer is working on which files
10. Query: "What breaks if I upgrade package X?"

Package Tracking Scope (MVP):

1. File → Package@Version tracking (what files import what packages)
2. Package → Package transitive dependencies (from lock files)
3. Nested function tracking (numpy.random.normal, not just numpy.array)
4. Version conflict detection (simple semver-based)
5. Breaking change warnings (major version bumps)
6. Query capability: "What breaks if I upgrade X?"

Function Tracking Granularity:

1. MVP: Nested attributes (numpy.random.normal, pandas.DataFrame.groupby)
2. Deferred: Method chains (df.groupby().agg().reset_index()) - P2 Post-MVP

Cross-Language Strategy:

1. MVP: Separate graphs per language (Python graph, JavaScript graph, Rust graph)
2. Post-MVP: Unified graph with API nodes connecting languages

Storage Architecture:

1. In-memory petgraph for active session (hot path)
2. Serialized to SQLite on session end (Tier 1 persistence)
3. Reconstructed on session start with incremental updates
4. Incremental updates on file changes (reparse only changed files)

Performance Targets:

1. Graph construction for 10,000 files: under 30 seconds
2. Query response time: under 50ms for most queries (under 1ms for cached queries)
3. Incremental update on single file change: under 100ms
4. Memory footprint: under 500MB for 10,000 file project

Semantic-Enhanced Dependency Graph:

Hybrid Search Capability (Structural + Semantic):

1. Structural Dependencies (Exact):
2. Track precise code relationships
3. Imports: "UserService imports AuthContext"
4. Function Calls: "create_user() calls validate_email()"
5. Inheritance: "UserProfile extends BaseProfile"
6. Data Flow: "function returns X, passed to Y"
7. Semantic Dependencies (Fuzzy):
8. Find similar code patterns without explicit imports
9. Discover related functionality by purpose
10. Identify duplicate logic across files
11. Find alternative implementations

Embedding Model Details:

1. Model: all-MiniLM-L6-v2 (sentence-transformers)
2. Dimensions: 384 (optimal balance of speed vs quality)
3. Size: 22MB quantized ONNX model
4. Inference: <8ms per embedding on CPU (fastembed-rs)
5. Privacy: 100% local inference (no API calls)
6. Cache: In-memory LRU for frequently accessed embeddings

Semantic Search Process:

1. Generate embedding for search query (8ms)
2. HNSW index finds nearest neighbors
3. Combine with structural graph filtering
4. Return ranked results with similarity scores

Single Source of Truth:

1. Graph contains everything (structural + semantic embeddings inline)
2. Automatic sync (update node → embedding updates inline)
3. Single query (BFS traversal filters by similarity simultaneously)
4. Precision + recall (exact dependencies + fuzzy discovery)
5. Simpler architecture (no external vector DB needed)

Agent Workflow:

1. Agent should automatically create File, Package, Code dependencies based on existing files and chat
2. For new project or in-progress project, Agent creates dependencies during code generation
3. Agent creates tech stack dependencies based on requirements, architecture files, or chat
4. Agent tracks transitive dependencies automatically
5. Agent uses graph to avoid tech stack dependency issues before code creation
6. Agent uses graph to avoid changes based on change impact analysis
7. Agent uses graph to avoid merge conflicts through prevention
8. Agent uses graph to enable intelligent refactoring
9. Agent uses graph to enable semantic search

#### 3.1.3 Extended Dependency Graph

The dependency graph extends beyond code to track multi-file coordination, developer activity, and documentation relationships, enabling file locking and merge conflict prevention.

Enhanced Tracking Beyond Code:

The dependency graph tracks not just code relationships but also multi-file coordination, developer activity, and documentation relationships. This enables sophisticated features like file locking systems and merge conflict prevention.

Developer Activity Tracking:

1. File edit timestamps for recency tracking
2. Active editing sessions (who is editing what right now)
3. File lock status (locked by agent or user)
4. Agent task assignments (which agent working on which files)
5. Edit history (who modified what when)

File Locking System:

The dependency graph powers a file locking system that makes merge conflicts impossible rather than just resolvable.

Lock Acquisition Process:

1. Agent requests to edit file X
2. System queries graph for all files depending on X
3. System checks if dependent files locked by other agents
4. If locked: Agent waits or works on different file
5. If free: Lock acquired, edit proceeds
6. Lock recorded in dependency graph with timestamp

Lock Release Process:

1. Agent completes edits and commits
2. System validates no dependency violations
3. Lock released from dependency graph
4. Dependent files can now be edited
5. Lock release recorded with commit hash

Proactive Conflict Prevention:

1. No two agents can edit dependent files simultaneously
2. Edit order enforced by dependency graph structure
3. Merge conflicts become structurally impossible
4. Coordination via graph queries, not Git merge logic
5. Explicit state transitions make locking auditable

Work Visibility (MVP):

1. UI shows which developer/agent is working on which files
2. Active work indicators in file tree
3. Real-time updates via WebSocket
4. Prevents parallel edits through coordination (not enforcement)

File Locking (Post-MVP):

1. Explicit lock acquisition before edits
2. Dependency-aware locking prevents conflicts
3. Lock table enforced by system
4. Makes merge conflicts impossible (not just less likely)

#### 3.1.4 YDoc System

Unified documentation system treating documentation as first-class nodes in the dependency graph with full traceability from requirements to code.

Core Principles:

1. Block Database as canonical source (files are serialization, not storage)
2. Graph-native with every block linked to requirements, code, and other docs
3. Agent-first editing where LLM reads JSON and writes via tools
4. Git-friendly with exportable, diffable, conflict-detectable format
5. Full traceability: requirement → architecture → spec → code → test → docs

File Format:

1. Extension: .ydoc
2. Schema: ipynb-compatible (opens in VS Code, Jupyter, GitHub)
3. Custom metadata stored in cell metadata field
4. Cell types: markdown, code, raw (standard notebook types)
5. Yantra types: stored in metadata.yantra_type field

Document Types:

| Type          | Code     | Purpose                                   |
| ------------- | -------- | ----------------------------------------- |
| Requirements  | REQ      | PRD, user intent, acceptance criteria     |
| ADR           | ADR      | Architecture Decision Records             |
| Architecture  | ARCH     | System design, component diagrams, flows  |
| Tech Spec     | SPEC     | Detailed behavior specifications          |
| Project Plan  | PLAN     | Tasks, milestones, timeline               |
| Tech Guide    | TECH     | Internal technical documentation          |
| API Guide     | API      | Endpoint/interface documentation          |
| User Guide    | USER     | End-user documentation                    |
| Testing Plan  | TEST     | Test strategy, coverage plan              |
| Test Results  | RESULT   | Historical test runs (smart archived)     |
| Change Log    | CHANGE   | What changed, when, by whom               |
| Decisions Log | DECISION | Sign-offs, approvals, requirement changes |

Block Schema Fields:

1. yantra_id: Unique block identifier (UUID)
2. yantra_type: Block type (requirement, adr, architecture, spec, etc.)
3. created_by: "user" or "agent"
4. created_at: ISO-8601 timestamp
5. modified_by: Last modifier type
6. modified_at: Last modification timestamp
7. modifier_id: User ID or agent task ID
8. graph_edges: Array of links to other blocks, code, docs
9. tags: Searchable tags array
10. status: Block status (draft, review, approved, deprecated)

Block Database Tables:

Documents Table:

| Column      | Type             | Description                  |
| ----------- | ---------------- | ---------------------------- |
| id          | TEXT PRIMARY KEY | UUID                         |
| doc_type    | TEXT NOT NULL    | requirement, adr, spec, etc. |
| title       | TEXT NOT NULL    | Document title               |
| file_path   | TEXT NOT NULL    | Path to .ydoc file           |
| created_by  | TEXT NOT NULL    | user or agent                |
| created_at  | TEXT NOT NULL    | ISO-8601 timestamp           |
| modified_at | TEXT NOT NULL    | ISO-8601 timestamp           |
| version     | TEXT             | Default "1.0.0"              |
| status      | TEXT             | Default "draft"              |

Blocks Table:

| Column      | Type             | Description                  |
| ----------- | ---------------- | ---------------------------- |
| id          | TEXT PRIMARY KEY | Block UUID                   |
| doc_id      | TEXT NOT NULL    | Foreign key to documents     |
| cell_index  | INTEGER NOT NULL | Position in document         |
| cell_type   | TEXT NOT NULL    | markdown, code, raw          |
| yantra_type | TEXT NOT NULL    | requirement, spec, adr, etc. |
| content     | TEXT NOT NULL    | Actual markdown/code         |
| created_by  | TEXT NOT NULL    | user or agent                |
| created_at  | TEXT NOT NULL    | ISO-8601 timestamp           |
| modified_by | TEXT NOT NULL    | Last modifier                |
| modified_at | TEXT NOT NULL    | ISO-8601 timestamp           |
| modifier_id | TEXT NOT NULL    | user-123 or agent-task-456   |
| status      | TEXT             | Default "draft"              |

Graph Edges Table (extended from dependency graph):

| Column      | Type                | Description                                              |
| ----------- | ------------------- | -------------------------------------------------------- |
| id          | INTEGER PRIMARY KEY | Auto-increment                                           |
| source_id   | TEXT NOT NULL       | file_id, block_id, function_id                           |
| source_type | TEXT NOT NULL       | code_file, doc_block, function, class                    |
| target_id   | TEXT NOT NULL       | Target entity ID                                         |
| target_type | TEXT NOT NULL       | Target entity type                                       |
| edge_type   | TEXT NOT NULL       | traces_to, implements, realized_in, tested_by, documents |
| created_at  | TEXT NOT NULL       | ISO-8601 timestamp                                       |
| metadata    | TEXT                | JSON: {line_range, confidence, etc.}                     |

Traceability Chain Examples:

Requirement → Architecture: When agent creates architecture from requirement, create edge with source=requirement block ID, target=architecture block ID, edge_type="traces_to", metadata containing rationale for architecture decision.

Architecture → Specification: Create edge with source=architecture block ID, target=specification block ID, edge_type="implements" showing how specification implements architectural component.

Specification → Code: Create edge with source=specification block ID, target=code file path, edge_type="realized_in", metadata containing affected functions like ["authenticate_user", "refresh_token"].

Requirement → Tests: Create edge with source=requirement block ID, target=test file path, edge_type="tested_by", metadata containing test function names like ["test_oauth_login", "test_token_refresh"].

Code → Issues: When maintenance machine detects issue, create edge with source=code file path, target=issue block in Change Log, edge_type="has_issue", metadata containing severity (high) and auto_fixed status (true).

Query Examples:

Find all code implementing REQ-001:

1. Step 1: Requirement → Architecture (traverse edges with type "traces_to")
2. Step 2: Architecture → Spec (traverse edges with type "implements")
3. Step 3: Spec → Code (traverse edges with type "realized_in")
4. Result: All code files implementing that requirement

Find docs needing update if src/auth/oauth.rs changes:

1. Reverse traverse: Code → Spec (reverse "realized_in")
2. Spec → Architecture (reverse "implements")
3. Architecture → Requirement (reverse "traces_to")
4. Also find: Code → API docs (reverse "documented_in")
5. Result: All affected documentation blocks

Impact analysis for changing REQ-001:

1. Return architecture blocks (traverse "traces_to")
2. Return specification blocks (traverse "traces_to" → "implements")
3. Return code files (traverse full chain to "realized_in")
4. Return test files (traverse "tested_by")
5. Return API documentation (traverse to code → "documented_in")

Storage Location:

| Data Type                | Storage Tier                 | Rationale                               |
| ------------------------ | ---------------------------- | --------------------------------------- |
| YDoc .ydoc files         | Disk (project /ydocs folder) | Git-friendly, human-readable, portable  |
| YDoc blocks (parsed)     | Tier 1 SQLite                | Fast queries, FTS5 search, local access |
| Graph edges              | Tier 1 SQLite + petgraph     | Bidirectional queries, fast traversal   |
| Block content embeddings | Tier 2 (Vector DB)           | Semantic search across docs + code      |
| Sync metadata            | Tier 1 SQLite                | Track Confluence sync status, conflicts |

Rationale: Files on disk provide Git workflow, backup, and portability. SQLite enables fast agent queries and FTS5 full-text search. petgraph enables graph traversal for traceability chains. No duplication as file is source of truth and SQLite is indexed view.

Folder Structure:

/ydocs

/requirements

    MASTER.ydoc                 ← Requirements overview/index

    EPIC-auth.ydoc

    EPIC-payments.ydoc

/architecture

    MASTER.ydoc                 ← Architecture overview

    COMPONENT-frontend.ydoc

    COMPONENT-backend.ydoc

/specs

    MASTER.ydoc                 ← Specifications index

    FEATURE-login.ydoc

    FEATURE-checkout.ydoc

/adr

    ADR-001-use-postgres.ydoc

    ADR-002-adopt-microservices.ydoc

/guides

    /tech

    MASTER.ydoc               ← Technical guide index

    SECTION-architecture.ydoc

    SECTION-setup.ydoc

    /api

    MASTER.ydoc               ← API guide index

    MODULE-auth.ydoc

    MODULE-payments.ydoc

    /user

    MASTER.ydoc               ← User guide index

    SECTION-getting-started.ydoc

    SECTION-features.ydoc

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

Document Organization Rules:

| Doc Type     | Folder           | Naming                   | Granularity                               |
| ------------ | ---------------- | ------------------------ | ----------------------------------------- |
| Requirements | /requirements    | EPIC-{name}.ydoc         | One file per epic, blocks per requirement |
| Architecture | /architecture    | COMPONENT-{name}.ydoc    | One file per component                    |
| Tech Specs   | /specs           | FEATURE-{name}.ydoc      | One file per feature                      |
| ADR          | /adr             | ADR-{number}-{name}.ydoc | One file per decision                     |
| Tech Guide   | /guides/tech     | SECTION-{name}.ydoc      | One file per section                      |
| API Guide    | /guides/api      | MODULE-{name}.ydoc       | One file per module                       |
| User Guide   | /guides/user     | SECTION-{name}.ydoc      | One file per section                      |
| Project Plan | /plans           | SPRINT-{number}.ydoc     | One file per sprint                       |
| Testing Plan | /testing         | PLAN-{name}.ydoc         | One file per feature                      |
| Test Results | /testing/results | RESULT-{date}.ydoc       | One file per day                          |
| Change Log   | /logs            | CHANGE-LOG.ydoc          | Single file, append blocks                |
| Decision Log | /logs            | DECISION-LOG.ydoc        | Single file, append blocks                |

MASTER.ydoc Purpose: Each folder's MASTER.ydoc serves as index and overview containing: vision or high-level purpose, goals or objectives, table of contents linking to other files, status summary.

Folder Creation on Init: On project initialization, system creates all folder structure and generates MASTER.ydoc files for folders that need them (requirements, architecture, specs, guides/tech, guides/api, guides/user, plans, testing). Also creates CHANGE-LOG.ydoc and DECISION-LOG.ydoc in /logs folder.

Export on Save: When Block DB changes, serialize affected document to .ydoc JSON, write to /ydocs/{type}/{filename}.ydoc, optionally export markdown shadow to /docs/{filename}.md, stage for git.

Git Integration:

Conflict Detection Process:

1. Detect .ydoc file changed outside Yantra (git pull or external edit)
2. Compare file metadata.modified_at with Block DB timestamp
3. If external change detected: parse external .ydoc file, diff against Block DB, show user "Document changed externally"
4. Options: "Use external" (reimport to Block DB), "Keep Yantra" (overwrite file from Block DB), "Review" (show diff, manual merge)

Diff Tooling: Integrate nbstripout or similar for clean diffs. Configure .gitattributes with \*.ydoc diff=ydoc. Configure .git/config with [diff "ydoc"] and textconv = yantra ydoc-to-text. The yantra ydoc-to-text command extracts just source content (not metadata) for readable diffs.

Smart Archiving for Test Results:

Retention Policy:

1. Keep all failures indefinitely (important for debugging patterns)
2. Keep last 10 passing runs per test (recent history)
3. Keep daily summary for results older than 30 days (aggregated)
4. Delete raw results older than 90 days but keep summary only

Archive Process (Daily Job):

1. Query test results older than 30 days
2. For each result: if failure keep indefinitely, if pass check if in last 10 for that test
3. If outside last 10 and older than 30 days: create summary block, delete raw data from blocks table
4. Summary includes: date, total tests, pass/fail counts, notable failures
5. Stored as blocks in special RESULT-SUMMARY.ydoc
6. Graph edges preserved for traceability even after raw data deleted

Bidirectional Sync with External Tools:

Confluence Integration (via official MCP server):

1. User configures Confluence connection in settings
2. Agent detects YDoc updates in Yantra
3. Agent converts YDoc markdown to Confluence storage format
4. Agent pushes to Confluence via MCP server
5. On Confluence changes, MCP server notifies Yantra via webhooks
6. Yantra pulls changes and updates Block DB
7. Conflict resolution if both sides changed (agent-mediated, user decides)

Supported External Tools: Confluence (MCP server), Notion (MCP server), GitHub Wiki (Git-based sync), Markdown export to any location.

YDoc Primitives for Agent (tools accessible by agent):

| Tool                      | Purpose                      | Protocol |
| ------------------------- | ---------------------------- | -------- |
| create_ydoc_document      | Create new YDoc document     | Builtin  |
| create_ydoc_block         | Create new block in document | Builtin  |
| update_ydoc_block         | Update existing block        | Builtin  |
| link_ydoc_to_code         | Create graph edge doc → code | Builtin  |
| search_ydoc_blocks        | Search documentation blocks  | Builtin  |
| export_ydoc_to_confluence | Push doc to Confluence       | MCP      |
| import_from_confluence    | Pull doc from Confluence     | MCP      |

##### 3.1.4.1 YDocBlockEditor - Advanced UI Component

Purpose: Dedicated block-level editor for YDoc documents with real-time metadata management, traceability visualization, and collaborative editing features.

Component Architecture:

// src-ui/components/YDocBlockEditor.tsx

interface YDocBlockEditorProps {

blockId: string;

documentId: string;

initialContent: string;

metadata: YDocBlockMetadata;

onSave: (content: string, metadata: YDocBlockMetadata) => Promise`<void>`;

onDelete: () => Promise`<void>`;

onLink: (targetId: string, edgeType: string) => Promise`<void>`;

}

interface YDocBlockMetadata {

yantra_id: string;

yantra_type: 'requirement' | 'adr' | 'architecture' | 'spec' | 'plan' | 'test' | 'api' | 'user' | 'change' | 'decision';

created_by: 'user' | 'agent';

created_at: string;

modified_by: 'user' | 'agent';

modified_at: string;

modifier_id: string;

graph_edges: Array<{

    target_id: string;

    target_type: string;

    edge_type: string;

}>;

tags: string[];

status: 'draft' | 'review' | 'approved' | 'deprecated';

}

UI Layout:

┌─────────────────────────────────────────────────────────────────┐

│ YDoc Block Editor [×] │

├─────────────────────────────────────────────────────────────────┤

│ ┌─────────────────────────────────┬─────────────────────────────┐│

│ │ Content Editor (Monaco) │ Metadata Panel ││

│ │ │ ││

│ │ # Authentication System │ 📋 Block Info ││

│ │ │ ID: REQ-AUTH-001 ││

│ │ ## Requirements │ Type: Requirement ││

│ │ - User login via OAuth2 │ Status: ⚪ Draft ││

│ │ - Token refresh mechanism │ Created: 2025-01-15 ││

│ │ - Session timeout: 30 min │ Modified: 2025-01-20 ││

│ │ │ ││

│ │ [Markdown preview/live editing] │ 🏷️ Tags ││

│ │ │ + auth ││

│ │ │ + security ││

│ │ │ + oauth ││

│ │ │ ││

│ │ │ 🔗 Links (3) ││

│ │ │ → ARCH-001 (traces_to) ││

│ │ │ → SPEC-003 (implements) ││

│ │ │ → src/auth/oauth.rs ││

│ │ │ (realized_in) ││

│ │ │ ││

│ │ │ [+ Add Link] ││

│ │ │ ││

│ │ │ 👥 Collaboration ││

│ │ │ Currently editing: You ││

│ │ │ Last edit: Agent (10m ago) ││

│ └─────────────────────────────────┴─────────────────────────────┘│

│ [Save] [Cancel] [View Graph] [Export] [History] │

└─────────────────────────────────────────────────────────────────┘

Features:

1. Rich Text Editing:
   - Monaco editor with markdown syntax highlighting
   - Live markdown preview (side-by-side or toggle)
   - Auto-save every 30 seconds (configurable)
   - Undo/redo with full history (30 revisions)
   - Keyboard shortcuts: Cmd/Ctrl+S (save), Cmd/Ctrl+B (bold), etc.

2. Metadata Management Panel:

- Block Info Section:
- Display yantra_id (read-only, copy button)
- Type selector dropdown (requirement, adr, spec, etc.)
- Status indicator with color coding:
- 🔵 Draft (blue)
- 🟡 Review (yellow)
- 🟢 Approved (green)
- ⚫ Deprecated (gray)
- Created/modified timestamps with "time ago" format
- Tags Section:
- Autocomplete tag input (suggests from existing tags)
- Tag pills with remove button
- Click tag to search for related blocks
- Color-coded by category (auth=blue, security=red, etc.)
- Links Section:
- Tree view of linked items grouped by edge type
- Click link to jump to target (block or code file)
- Hover shows preview tooltip
- Bidirectional links shown (→ outgoing, ← incoming)
- [+ Add Link] button opens link picker dialog
- Collaboration Section:
- Show who's currently editing (real-time, Phase 2B)
- Last edit by user/agent with timestamp
- Conflict warning if document changed externally

3. Link Picker Dialog:

┌───────────────────────────────────────┐

│ Add Link │

├───────────────────────────────────────┤

│ Link Type: [traces_to ▼] │

│ │

│ Target: │

│ [ Search blocks/files... ] │

│ │

│ Results: │

│ □ ARCH-001: Authentication Arch │

│ □ SPEC-003: OAuth Implementation │

│ □ src/auth/oauth.rs │

│ □ tests/test_auth.py │

│ │

│ [Add Link] [Cancel] │

└───────────────────────────────────────┘

- Link type dropdown with options: traces_to, implements, realized_in, tested_by, documents, has_issue
- Search input with fuzzy matching across all blocks and code files
- Multi-select support (add multiple links at once)
- Preview pane showing target content

3. Version History:
   - Click [History] button to open timeline view
   - Shows last 30 revisions with diff visualization
   - Filter by user/agent edits
   - Restore previous version with confirmation
   - Visual diff highlighting (red=removed, green=added)

4. Export Options:

- Export to Markdown (.md)
- Export to Confluence (via MCP)
- Export to PDF (with metadata as footer)
- Copy as plain text
- Copy block ID

3. Keyboard Shortcuts:

- Cmd/Ctrl+S: Save
- Cmd/Ctrl+K: Quick search (blocks and files)
- Cmd/Ctrl+L: Add link
- Cmd/Ctrl+T: Add tag
- Cmd/Ctrl+G: View in graph
- Escape: Close editor

Implementation Details:

// Real-time auto-save

const autoSave = useCallback(

debounce(async (content: string) => {

    try {

    await invoke('save_ydoc_block', {

    blockId,

    content,

    metadata: currentMetadata

    });

    setLastSaved(new Date());

    } catch (err) {

    console.error('Auto-save failed:', err);

    showNotification('Auto-save failed', 'error');

    }

}, 30000), // 30 seconds

[blockId, currentMetadata]

);

// Tag autocomplete

const suggestTags = async (input: string) => {

const allTags = await invoke<string[]>('get_all_tags');

return allTags.filter(tag =>

    tag.toLowerCase().includes(input.toLowerCase())

);

};

// Link picker search

const searchLinkTargets = async (query: string) => {

const [blocks, files] = await Promise.all([

    invoke<YDocBlock[]>('search_ydoc_blocks', { query }),

    invoke<CodeFile[]>('search_code_files', { query })

]);

return [

    ...blocks.map(b => ({

    id: b.id,

    type: 'block',

    label:`${b.yantra_id}: ${b.title}`,

    preview: b.content.substring(0, 100)

    })),

    ...files.map(f => ({

    id: f.path,

    type: 'file',

    label: f.path,

    preview: f.preview

    }))

];

};

Integration with Backend:

// src/ydoc/block_editor.rs

#[tauri::command]

pub async fn save_ydoc_block(

    block_id: &str,

    content: &str,

    metadata: YDocBlockMetadata,

    state: State<'_, AppState>

) -> Result<(), String> {

    // 1. Update block in database

    sqlx::query!(

    "UPDATE blocks SET

    content = ?,

    modified_at = ?,

    modified_by = ?,

    status = ?

    WHERE id = ?",

    content,

    metadata.modified_at,

    metadata.modified_by,

    metadata.status,

    block_id

    )

    .execute(&state.db)

    .await

    .map_err(|e| e.to_string())?;

    // 2. Update tags (delete old, insert new)

    state.tag_service.update_tags(block_id, &metadata.tags).await?;

    // 3. Update graph edges

    state.graph_service.update_edges(block_id, &metadata.graph_edges).await?;

    // 4. Trigger document export to .ydoc file

    state.ydoc_exporter.export_document(block_id).await?;

    // 5. Notify other editors (Phase 2B - multi-user)

    state.collaboration_service

    .broadcast_block_update(block_id, content)

    .await?;

    Ok(())

}

#[tauri::command]

pub async fn get_all_tags(

    state: State<'_, AppState>

) -> Result<Vec`<String>`, String> {

    sqlx::query_scalar!(

    "SELECT DISTINCT tag FROM block_tags ORDER BY tag"

    )

    .fetch_all(&state.db)

    .await

    .map_err(|e| e.to_string())

}

Performance Targets:

- Editor load time: <200ms (including content + metadata)
- Auto-save: <100ms (non-blocking)
- Tag autocomplete: <50ms
- Link search: <100ms (FTS5 + fuzzy)
- Version history load: <300ms (last 30 revisions)
- Graph view transition: <500ms

Accessibility:

- Full keyboard navigation support
- Screen reader compatible (ARIA labels)
- High contrast mode support
- Configurable font sizes

##### 3.1.4.2 YDocTraceabilityGraph - Interactive Visualization

Purpose: Interactive D3.js/vis.js graph visualization showing full traceability chains from requirements to code to tests, with real-time updates and impact analysis.

Component Architecture:

// src-ui/components/YDocTraceabilityGraph.tsx

interface YDocTraceabilityGraphProps {

projectId: string;

focusNodeId?: string; // Optional: center on specific block/file

filterBy?: {

    types?: string[];     // Filter by node types

    edgeTypes?: string[]; // Filter by edge types

    status?: string[];    // Filter by block status

};

mode: 'full' | 'neighborhood' | 'path';

onNodeClick: (nodeId: string, nodeType: string) => void;

onEdgeClick: (edgeId: string) => void;

}

UI Layout:

┌─────────────────────────────────────────────────────────────────┐

│ YDoc Traceability Graph [Settings] [Export] │

├─────────────────────────────────────────────────────────────────┤

│ Filters: [Requirements ✓] [Architecture ✓] [Code ✓] [Tests ✓] │

│ Edge Types: [traces_to ✓] [implements ✓] [realized_in ✓] │

│ Layout: [Force ▼] | Search: [Find node... ] │

├─────────────────────────────────────────────────────────────────┤

│ │

│ ┌─────────────┐ │

│ │ REQ-001 │ │

│ │ Auth Sys │ │

│ └──────┬──────┘ │

│ │ traces_to │

│ ┌──────▼──────┐ │

│ │ ARCH-001 │ │

│ │ OAuth Arch │ │

│ └──────┬──────┘ │

│ │ implements │

│ ┌────────────┼────────────┐ │

│ │ │ │ │

│ ┌──────▼──────┐ ┌──▼────────┐ ┌▼─────────┐ │

│ │ SPEC-001 │ │ SPEC-002 │ │SPEC-003 │ │

│ │ Login │ │ Token │ │Session │ │

│ └──────┬──────┘ └──┬────────┘ └┬─────────┘ │

│ │ realized_in│ │ │

│ ┌──────▼──────┐ ┌──▼────────┐ ┌▼─────────┐ │

│ │oauth.rs │ │ jwt.rs │ │session.rs│ │

│ └──────┬──────┘ └──┬────────┘ └┬─────────┘ │

│ │ tested_by │ │ │

│ ┌──────▼───────────▼───────────▼─────────┐ │

│ │ tests/test_auth.py │ │

│ └────────────────────────────────────────┘ │

│ │

├─────────────────────────────────────────────────────────────────┤

│ Selected: REQ-001 (Requirement) │

│ Outgoing: 1 trace → ARCH-001 │

│ Impact: 3 specs, 3 files, 1 test │

│ Status: 🟢 Approved | Tags: auth, security │

└─────────────────────────────────────────────────────────────────┘

Graph Visualization Features:

1. Node Types (with distinct visual styling):
   - Requirements (🔵 Blue circles): REQ-001, REQ-002, etc.
   - Architecture (🟣 Purple diamonds): ARCH-001, ARCH-002, etc.
   - Specifications (🟢 Green hexagons): SPEC-001, SPEC-002, etc.
   - Code Files (🟠 Orange rectangles): src/auth/oauth.rs, etc.
   - Tests (🔴 Red triangles): tests/test_auth.py, etc.
   - Documentation (⚪ White rounded-rect): API-001, USER-001, etc.

2. Edge Types (with distinct colors and styles):

- traces_to (solid blue line): Requirement → Architecture
- implements (solid green line): Architecture → Specification
- realized_in (solid orange line): Specification → Code
- tested_by (dashed red line): Code → Tests
- documents (dotted purple line): Code → API Documentation
- has_issue (dashed red line, thicker): Code → Change Log

1. Interaction Modes:

   a) Full Graph Mode:

- Display all nodes and edges in the project
- Force-directed layout (D3.js force simulation)
- Zoom and pan with mouse/trackpad
- Minimap in corner for navigation

1. b) Neighborhood Mode (focus on one node):

- Show selected node + all directly connected nodes
- Depth slider: 1 hop, 2 hops, 3 hops
- Radial layout with selected node at center
- Dimmed nodes for nodes beyond focus

1. c) Path Mode (trace from A to B):

- Select source node and target node
- Highlight all possible paths between them
- Show shortest path in bold
- Dim unrelated nodes

1. Interactive Features:

   Node Interactions:

- Click: Select node, show details in bottom panel
- Double-click: Open block editor or code file
- Right-click: Context menu
- Open in editor
- Show in file tree
- Find related nodes
- Export subgraph
- Remove from graph (hide)
- Hover: Show tooltip with node metadata
- Drag: Reposition node (sticky in force layout)

1. Edge Interactions:

- Click: Highlight edge, show edge metadata in panel
- Hover: Show edge type and creation date
- Double-click: Navigate between source and target

1. Selection:

- Shift+click: Multi-select nodes
- Cmd/Ctrl+click: Add to selection
- Drag box: Select multiple nodes
- [Select All] [Clear Selection] buttons

1. Layout Algorithms:

| Layout         | Best For     | Description                                |
| -------------- | ------------ | ------------------------------------------ |
| Force-Directed | General      | D3.js force simulation, organic layout     |
| Hierarchical   | Traceability | Top-down: requirements → code → tests      |
| Radial         | Neighborhood | Selected node at center, rings by distance |
| Circular       | Cycles       | Detect and visualize circular dependencies |
| Grid           | Large graphs | Organized grid for performance             |

- Toggle between layouts with dropdown
- Smooth animations between layout changes (500ms)
- Save layout preference per project

10. Filtering and Search:

    Filter Panel:
    - Node type checkboxes (Requirements, Architecture, Code, etc.)
    - Edge type checkboxes (traces_to, implements, etc.)
    - Status filter (Draft, Review, Approved, Deprecated)
    - Tag filter (multi-select autocomplete)
    - Date range filter (created/modified)

11. Search:

- Fuzzy search by node ID, title, or content
- Regex support for advanced queries
- Search results highlighted in graph
- "Jump to" button for each result

12. Impact Analysis (right-click on node):

┌───────────────────────────────────┐

│ Impact Analysis: REQ-001 │

├───────────────────────────────────┤

│ If this requirement changes: │

│ │

│ Affected Architecture: │

│ • ARCH-001 (OAuth Architecture) │

│ │

│ Affected Specifications: │

│ • SPEC-001 (Login Flow) │

│ • SPEC-002 (Token Management) │

│ • SPEC-003 (Session Handling) │

│ │

│ Affected Code: │

│ • src/auth/oauth.rs │

│ • src/auth/jwt.rs │

│ • src/auth/session.rs │

│ │

│ Affected Tests: │

│ • tests/test_auth.py (15 tests) │

│ │

│ Affected Documentation: │

│ • API-001 (Authentication API) │

│ │

│ Total Impact: 8 artifacts │

│ │

│ [Highlight in Graph] [Export] │

└───────────────────────────────────┘

13. Real-Time Updates (Phase 2B):
    - WebSocket connection to backend
    - New nodes appear with animation
    - Deleted nodes fade out
    - Edge updates with pulse animation
    - Multi-user cursors (show who's viewing)

14. Export Options:

- Export as PNG/SVG image (current view)
- Export as JSON (graph data)
- Export as DOT file (Graphviz)
- Export subgraph (selected nodes only)
- Export traceability report (PDF)

13. Performance Optimizations:

- Level of Detail (LOD): Hide labels when zoomed out
- Culling: Don't render nodes outside viewport
- Clustering: Group nodes when >500 nodes
- WebGL Rendering: Use vis.js with WebGL for >1000 nodes
- Lazy Loading: Load graph incrementally (viewport-first)

Implementation Details:

// Graph rendering with D3.js force simulation

const GraphRenderer = ({ nodes, edges, onNodeClick }: GraphProps) => {

const svgRef = useRef`<SVGSVGElement>`(null);

const simulationRef = useRef<d3.Simulation<Node, Edge>>();

useEffect(() => {

    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);

    const width = svg.node()!.clientWidth;

    const height = svg.node()!.clientHeight;

    // Create force simulation

    const simulation = d3.forceSimulation(nodes)

    .force('link', d3.forceLink(edges)

    .id((d: any) => d.id)

    .distance(100))

    .force('charge', d3.forceManyBody().strength(-300))

    .force('center', d3.forceCenter(width / 2, height / 2))

    .force('collision', d3.forceCollide().radius(50));

    // Render edges

    const link = svg.append('g')

    .selectAll('line')

    .data(edges)

    .join('line')

    .attr('stroke', d => getEdgeColor(d.type))

    .attr('stroke-width', 2)

    .attr('stroke-dasharray', d => d.type === 'tested_by' ? '5,5' : 'none');

    // Render nodes

    const node = svg.append('g')

    .selectAll('g')

    .data(nodes)

    .join('g')

    .call(drag(simulation))

    .on('click', (event, d) => onNodeClick(d.id, d.type))

    .on('dblclick', (event, d) => openNode(d));

    // Node shapes based on type

    node.each(function(d) {

    const g = d3.select(this);

    switch (d.type) {

    case 'requirement':

    g.append('circle')

    .attr('r', 20)

    .attr('fill', '#3B82F6'); // Blue

    break;

    case 'architecture':

    g.append('polygon')

    .attr('points', '0,-20 20,0 0,20 -20,0') // Diamond

    .attr('fill', '#8B5CF6'); // Purple

    break;

    case 'spec':

    g.append('polygon')

    .attr('points', hexagonPoints(20))

    .attr('fill', '#10B981'); // Green

    break;

    case 'code':

    g.append('rect')

    .attr('x', -25)

    .attr('y', -15)

    .attr('width', 50)

    .attr('height', 30)

    .attr('fill', '#F59E0B'); // Orange

    break;

    case 'test':

    g.append('polygon')

    .attr('points', '0,-20 20,20 -20,20') // Triangle

    .attr('fill', '#EF4444'); // Red

    break;

    }

    // Node label

    g.append('text')

    .attr('dy', 35)

    .attr('text-anchor', 'middle')

    .attr('font-size', 10)

    .text(d.label);

    });

    // Update positions on simulation tick

    simulation.on('tick', () => {

    link

    .attr('x1', (d: any) => d.source.x)

    .attr('y1', (d: any) => d.source.y)

    .attr('x2', (d: any) => d.target.x)

    .attr('y2', (d: any) => d.target.y);

    node.attr('transform', (d: any) =>`translate(${d.x},${d.y})`);

    });

    simulationRef.current = simulation;

    return () => {

    simulation.stop();

    };

}, [nodes, edges]);

return <svg ref={svgRef} width="100%" height="100%" />;

};

// Backend integration

#[tauri::command]

pub async fn get_traceability_graph(

    project_id: &str,

    filter: Option`<GraphFilter>`,

    state: State<'_, AppState>

) -> Result<TraceabilityGraph, String> {

    let mut query = String::from(

    "SELECT DISTINCT

    ge.source_id, ge.source_type,

    ge.target_id, ge.target_type,

    ge.edge_type, ge.metadata

    FROM graph_edges ge"

    );

    // Apply filters

    if let Some(f) = filter {

    let mut conditions = vec![];

    if !f.node_types.is_empty() {

    conditions.push(format!(

    "(source_type IN ({}) OR target_type IN ({}))",

    f.node_types.iter().map(|t| format!("'{}'", t)).collect::<Vec_>().join(","),

    f.node_types.iter().map(|t| format!("'{}'", t)).collect::<Vec_>().join(",")

    ));

    }

    if !f.edge_types.is_empty() {

    conditions.push(format!(

    "edge_type IN ({})",

    f.edge_types.iter().map(|t| format!("'{}'", t)).collect::<Vec_>().join(",")

    ));

    }

    if !conditions.is_empty() {

    query.push_str(&format!(" WHERE {}", conditions.join(" AND ")));

    }

    }

    let edges: Vec`<GraphEdge>` = sqlx::query_as(&query)

    .fetch_all(&state.db)

    .await

    .map_err(|e| e.to_string())?;

    // Build graph structure

    let graph = build_graph_from_edges(edges).await?;

    Ok(graph)

}

Performance Targets:

- Graph load time (500 nodes): <1s
- Graph load time (5000 nodes): <5s (with clustering)
- Node interaction latency: <50ms
- Layout animation: 60 FPS
- Search: <100ms
- Impact analysis: <200ms

Integration Points:

- Block Editor: [View in Graph] button opens graph focused on current block
- Code Editor: Right-click file → "View in Traceability Graph"
- Chat: Ask agent "Show me the traceability for REQ-001" → opens graph
- State Machines: View current state machine execution as graph overlay

#### 3.1.5 Unlimited Context Solution

LLMs have finite context windows (128k-200k tokens for Claude/GPT-4), but real projects contain millions of lines of code requiring comprehensive awareness without exceeding token limits.

Solution Architecture:

Level-Based Context Strategy:

Level 0 - Always Included (1k-2k tokens): User's current message, last 3-5 conversation turns, current file being edited (truncated if too large).

Level 1 - Direct Dependencies (5k-10k tokens): Files imported by current file, files that import current file, functions called in current file, classes instantiated in current file.

Level 2 - Transitive Dependencies (10k-20k tokens): Files imported by Level 1 files (one level deep), shared utility modules, configuration files, type definitions.

Level 3 - Semantic Similarity (20k-30k tokens): Code semantically similar to current task, previously generated code for similar features, test files for similar functionality, documentation for related features.

Level 4 - Project Context (5k-10k tokens): Project README, architecture documentation, API contracts, key configuration files.

Level 5 - Yantra Codex Patterns (2k-5k tokens): Learned code generation patterns for this project, common failure patterns and fixes, testing strategies that worked, best practices from past successes.

Dynamic Context Assembly:

Step 1 - Query Dependency Graph:

1. Start with current file as root
2. Traverse import edges in both directions
3. Collect files up to depth 2
4. Track token count for each file
5. Prioritize by relevance score

Step 2 - Semantic Search:

1. Embed user's current query (8ms using all-MiniLM-L6-v2)
2. Search Yantra Codex for similar past tasks
3. Search codebase for semantically similar files
4. Rank by relevance score (cosine similarity)
5. Filter by minimum similarity threshold (>0.7)

Step 3 - Token Budget Management:

1. Total budget: 120k tokens (leaving 8k for response)
2. Allocate by level priority (Level 0 highest)
3. If budget exceeded, truncate lower levels
4. Always preserve Level 0 (current context)
5. Track actual token usage for metrics

Step 4 - Context Assembly:

1. Concatenate selected files in priority order
2. Add markers: "File: src/auth/login.py"
3. Add relationship context: "Imported by: src/api/auth_routes.py"
4. Preserve file structure (don't split mid-function)
5. Add line numbers for reference

Step 5 - Compression Techniques:

1. For large files: truncate middle (keep first/last 100 lines)
2. Summarize large documentation files (LLM-generated summary)
3. Remove comments for utility files (unless needed for understanding)
4. Deduplicate repeated imports (show once with multiple files)
5. Compress whitespace (single newlines only)

Adaptive Context: Context size and composition adapts based on task complexity.

For simple queries (bug fix, small feature):

1. Use minimal context (Level 0 + Level 1 only)
2. Total tokens: ~10k-20k
3. Fast assembly (<100ms)

For complex queries (new feature, refactoring):

1. Use full context (all levels)
2. Total tokens: up to 120k
3. Comprehensive assembly (<500ms)

For architectural changes:

1. Include all architecture docs
2. Include dependency map
3. Include affected components
4. Total tokens: up to 120k

Context Caching & Reuse:

SQLite Cache:

1. Cache compressed context by hash (file content + dependencies)
2. 24-hour TTL (time-to-live)
3. Invalidate on file changes
4. Performance gain: <50ms retrieval vs 100-500ms assembly

Shared Context Across Requests:

1. Same file referenced multiple times in session
2. Compute embedding once, reuse for all queries
3. Track with reference counting
4. Reduces redundant computation

Performance Requirements:

1. Context assembly under 500ms for typical query
2. Dependency graph traversal under 200ms
3. Semantic search under 300ms
4. Total latency under 1 second before LLM call

Context Window Utilization Metrics:

1. Track average tokens used per query
2. Track tier-by-tier token allocation
3. Identify frequently accessed but rarely useful files (candidates for exclusion)
4. A/B test context strategies for quality improvement
5. Measure impact on code generation quality

Token Counting: Accurate token counting per LLM model using tiktoken library. Track context limits per LLM (Claude: 200k, GPT-4: 128k, GPT-3.5: 16k). Warn when approaching limit (>90% usage). Adaptive compression when near limit.

Why This Enables ANY LLM (Including Qwen Coder):

The Key Insight: Most LLM failures are due to missing context, not LLM capability.

With Yantra's Context Intelligence:

1. Qwen Coder 32K (smaller model):
   - Gets 25,000 tokens of perfectly relevant context
   - Hierarchical assembly prioritizes what matters
   - Semantic enhanced dependency graph provides proven patterns
   - Known failures database prevents common mistakes
   - Result: Performs as well as GPT-4 with 100K random tokens

2. Even GPT-3.5 (16K context):

- Gets 12,000 tokens of hyper-relevant context
- Every token carefully selected
- Compression eliminates noise
- Result: Better than GPT-4 with random 100K context

Validation: Benchmark same task with GPT-4 (naive 100K context) vs Qwen Coder (optimized 25K context). Metric: Code quality, test pass rate, breaking changes. Target: Qwen performance within 5% of GPT-4.

Performance Targets:

| Operation              | MVP Target | Scale Target |
| ---------------------- | ---------- | ------------ |
| Token counting         | <10ms      | <5ms         |
| Context assembly       | <100ms     | <50ms        |
| Compression            | <50ms      | <20ms        |
| Total context pipeline | <500ms     | <200ms       |

##### 3.1.5.7 Conversation Context Assembly

Purpose: Extend hierarchical context assembly to include conversation history alongside code context, enabling the agent to maintain full conversation awareness without exceeding LLM token limits.

Integration with Code Context:

The conversation context is integrated as Level 0, taking priority before code context:

Level 0 - Conversation Context (NEW - 20,000 tokens, 17% of budget):

- Recent messages (10): Full text, always included for immediate context
- Relevant old messages (5): Semantic search finds similar past discussions
- Conversation summaries: Compressed overviews when token budget is tight

Level 1-5 - Code Context (Existing - 80,000 tokens, 66% of budget):

- Level 1: Direct dependencies (imports, function calls)
- Level 2: Transitive dependencies (one level deep)
- Level 3: Semantic similarity (similar code patterns)
- Level 4: Project context (README, architecture docs)
- Level 5: Yantra Codex patterns (learned best practices)

Reserve Buffer (20,000 tokens, 17% of budget):

- Safety margin for response generation
- Prevents context window overflow

Context Assembly Flow:

1. Load Conversation Context (parallel with code context):
   - Query conversation_messages table for recent 10 messages
   - Generate embedding for current user message
   - Perform HNSW semantic search for top-5 relevant old messages
   - Filter out duplicates (already in recent)
   - Count tokens for all conversation context

2. Load Code Context (existing flow):

- Query dependency graph for relevant files
- Perform semantic search for similar code
- Prioritize by relevance scores

1. Verify Total Budget:

- Conversation tokens + Code tokens < 120,000
- If exceeded, apply compression strategy

1. Compression Strategy (when needed):

- Priority 1: Keep recent messages full text (never compress)
- Priority 2: Keep Level 1 code (direct dependencies) full text
- Priority 3: Compress relevant old messages into summary
- Priority 4: Compress Level 2-3 code (transitive dependencies)
- Generate LLM summary: "User previously discussed X, Y, Z"

1. Assemble Final Context:

- Conversation context first (most recent information)
- Code context second (implementation details)
- Return combined context to LLM

Adaptive Strategies Extended:

Context allocation adapts based on query type:

| Task Type          | Conversation % | Code % | Use Case                                               |
| ------------------ | -------------- | ------ | ------------------------------------------------------ |
| New feature        | 25%            | 75%    | Need mostly code context, some conversation for intent |
| Bug fix            | 15%            | 85%    | Primarily code-focused, minimal conversation needed    |
| Refactoring        | 20%            | 80%    | Architecture-heavy, some conversation for rationale    |
| Follow-up question | 40%            | 60%    | Needs recent conversation heavily, moderate code       |
| Clarification      | 60%            | 40%    | Conversation-dominant, minimal code context            |
| Code review        | 30%            | 70%    | Balance conversation feedback with code details        |

Example: Follow-up Question Flow:

User (3 days ago): "Create a login system with JWT tokens"

→ Agent generates: src/auth/login.py, src/auth/jwt.py

User (today): "Add password reset to that system"

Context Assembly:

1. Conversation (40% = 48K tokens):
   - Recent: Last 10 messages (full text)
   - Relevant: Semantic search finds "login system" from 3 days ago
   - Summary: "User created JWT-based authentication system"

2. Code (60% = 72K tokens):
   - L1: Full code of src/auth/login.py, src/auth/jwt.py (20K tokens)
   - L2: Email sender functions (10K tokens)
   - L3: Database user models (15K tokens)
   - L4: Architecture overview (10K tokens)
   - L5: Pattern: Password reset workflows (17K tokens)

→ Agent understands: "that system" = JWT login from 3 days ago

→ Agent adds password reset to CORRECT files

→ Natural conversation continuity maintained

Performance Impact:

Before (Code Only):

- Context assembly: 500ms
  - Dependency graph queries: 200ms
  - Semantic search (code): 150ms
  - File loading: 100ms
  - Compression: 50ms

After (Code + Conversation):

- Context assembly: 650ms (+150ms, 30% increase)
  - Dependency graph queries: 200ms (unchanged)
  - Semantic search (code): 150ms (unchanged)
  - Conversation retrieval: 150ms (NEW)
    - Load recent (10 messages): 20ms
    - Semantic search (top-5): 30ms
    - Compression (if needed): 100ms

  - File loading: 100ms (unchanged)
  - Compression: 50ms (unchanged)

Total overhead: +150ms (acceptable for much better UX)

Token Budget Verification:

The system continuously tracks token usage:

Conversation Context:

- Recent (10 messages × 150 tokens avg) = 1,500 tokens
- Relevant (5 messages × 150 tokens avg) = 750 tokens
- Summary (if needed) = 500 tokens
- Total: ~2,750 tokens (well under 20K budget)

Code Context:

- Level 1 (direct deps): 20,000 tokens
- Level 2 (transitive): 15,000 tokens
- Level 3 (semantic): 15,000 tokens
- Level 4 (project): 10,000 tokens
- Level 5 (patterns): 15,000 tokens
- Total: ~75,000 tokens (within 80K budget)

Reserve: 20,000 tokens (response generation)

Grand Total: 97,750 tokens < 120,000 limit ✅

Integration with Existing Context Caching:

The conversation context benefits from existing Tier 4 caching:

1. Cache Key: hash(recent_messages + code_dependencies)
2. TTL: 24 hours (resets when new message arrives)
3. Invalidation: On new message, file change, or dependency update
4. Performance: Cached retrieval <50ms vs fresh assembly 150ms

Fallback Behavior:

If conversation retrieval fails (database error, corruption):

1. Log error to .yantra/logs/conversation_errors.log
2. Continue with code-only context (graceful degradation)
3. Display warning to user: "Conversation history temporarily unavailable"
4. System remains functional (no breaking changes)

#### 3.1.6 Yantra Codex

IMPORTANT DISTINCTION:

What it is: GraphSAGE Neural Network for AI code generation pattern learning. This is the ACTUAL GNN (Graph Neural Network) in the system using real machine learning.

What it is NOT: The dependency graph (which is misleadingly called "GNN" in code but is actually a petgraph data structure, not a neural network).

Purpose: Generate code from natural language using machine learning. Learn from project-specific patterns, past successes, and failures to improve future code generation.

Technology Stack: GraphSAGE neural network architecture (1024-dim embeddings, 150M parameters), vector embeddings for code and patterns, SQLite for pattern storage (SEPARATE database: .yantra/codex.db), HNSW vector index for semantic search, incremental learning on every code generation event, PyTorch for neural network training.

Storage Architecture: Yantra Codex is COMPLETELY SEPARATE from the Dependency Graph system (see Section 3.1.7 for details):

1. Location: .yantra/codex.db (separate SQLite database)
2. Why Separate: Codex stores 10,000+ patterns, queried rarely (~once per generation). Dependency graph queried constantly (<1ms requirement).
3. Size: ~500MB for full CodeContests + user patterns
4. Performance: <50ms for pattern matching via HNSW semantic search
5. Update Frequency: Periodic (on LLM corrections), not real-time like dependency graph

Why Two Systems:

| Aspect          | Dependency Graph (In code "GNN") | Yantra Codex                |
| --------------- | -------------------------------- | --------------------------- |
| Purpose         | Code relationships               | Code generation             |
| Technology      | petgraph (data structure)        | GraphSAGE (neural network)  |
| Input           | AST from tree-sitter             | Problem description         |
| Output          | Dependency queries               | Generated code              |
| Speed           | <1ms                             | 15ms                        |
| Learning        | No learning                      | Continuous learning         |
| Local/Cloud     | Both (sync structure)            | Both (sync embeddings)      |
| Storage         | Tier 1 (in-memory + SQLite)      | Separate codex.db           |
| Query Frequency | Constant (hot path)              | Rare (~once per generation) |
| Code Name       | "GNN" (misleading)               | "Yantra Codex"              |

Integration: Dependency Graph provides context → Yantra Codex generates code → Dependency Graph validates new code fits properly.

Yantra Codex: AI Pair Programming Engine (DEFAULT MODE):

Yantra Codex is a hybrid AI pair programming system that combines a specialized Graph Neural Network (GNN) with premium LLMs (Claude/ChatGPT) to generate production-quality code.

Core Innovation:

1. Yantra Codex (GNN): Fast, local, learning-focused (15ms, FREE after initial model cost)
2. Premium LLM: Review, enhance, handle edge cases (user's choice: Claude/ChatGPT)
3. Continuous Learning: Yantra learns from LLM fixes → reduces cost over time

Pair Programming Roles:

1. Yantra Codex (Junior Partner): Generates initial code, handles common patterns, learns continuously
2. LLM (Senior Partner): Reviews edge cases, adds error handling, teaches Yantra implicitly

Key Principles:

1. Hybrid Intelligence: GNN speed + LLM reasoning = superior quality
2. Cost Optimization: 90% cost reduction initially (Yantra handles most, LLM reviews selectively)
3. Continuous Learning: Yantra learns from LLM fixes → 96% cost reduction after 12 months
4. User Choice: Configure Claude Sonnet 4, GPT-4, Gemini, or other premium LLMs

Model Specifications:

GraphSAGE Neural Network (1024-dim embeddings):

1. Input: 978-dimensional problem features
2. Layers: 978 → 1536 → 1280 → 1024 (four-layer architecture)
3. Parameters: ~150M parameters
4. Model Size: ~600 MB on disk
5. Inference: 15ms on CPU, 5ms on GPU
6. Why 1024 dimensions: Sufficient capacity for multi-step logic patterns, 55-60% initial accuracy (vs 40% with 256 dims), fast inference (still feels instant), room to scale to 2048+ dims later

Learning Sources:

1. Successfully generated code (what works)
2. Bug fixes and corrections (what broke and how it was fixed)
3. Test failures and subsequent fixes (what testing caught)
4. LLM mistakes and human corrections (where AI failed)
5. Manual code edits by developers (preferred patterns)
6. Code reviews and feedback (quality signals)

Pattern Categories:

| Category             | Examples                                                  | Usage                     |
| -------------------- | --------------------------------------------------------- | ------------------------- |
| Code Structure       | File organization, naming conventions, import patterns    | Guide new file creation   |
| Error Handling       | Try/catch patterns, validation logic, error messages      | Generate robust code      |
| Testing Strategies   | Test file organization, assertion patterns, mock usage    | Generate effective tests  |
| API Patterns         | REST endpoint structure, authentication, response formats | Generate consistent APIs  |
| Database Patterns    | Query patterns, ORM usage, migration strategies           | Generate data access code |
| Security Patterns    | Input validation, sanitization, authentication flows      | Generate secure code      |
| Performance Patterns | Caching strategies, query optimization, batching          | Generate efficient code   |

Pattern Storage Schema:

Patterns Table:

| Column           | Type             | Description                             |
| ---------------- | ---------------- | --------------------------------------- |
| id               | TEXT PRIMARY KEY | Pattern UUID                            |
| category         | TEXT NOT NULL    | code_structure, error_handling, etc.    |
| pattern_type     | TEXT NOT NULL    | success, bug_fix, test_fix, llm_mistake |
| code_snippet     | TEXT NOT NULL    | The actual code pattern                 |
| context          | TEXT             | When to apply this pattern              |
| success_count    | INTEGER          | How many times successfully applied     |
| failure_count    | INTEGER          | How many times failed                   |
| confidence_score | REAL             | 0.0 to 1.0 confidence                   |
| created_at       | TEXT NOT NULL    | ISO-8601 timestamp                      |
| last_used_at     | TEXT             | Last time pattern was applied           |
| embedding        | BLOB             | Vector embedding for semantic search    |

Pattern Learning Process:

On Successful Code Generation:

1. Extract patterns from generated code (functions, classes, imports, error handling)
2. Store in Patterns table with pattern_type="success"
3. Increment success_count if pattern already exists
4. Update confidence_score based on success rate
5. Generate embedding for semantic search using all-MiniLM-L6-v2

On Bug Fix:

1. Capture original buggy code
2. Capture fixed code
3. Extract diff pattern (what changed to fix bug)
4. Store as pattern_type="bug_fix"
5. Link to original bug description
6. Use as negative example (avoid similar bugs in future)

On Test Failure Then Fix:

1. Capture failing test
2. Capture code change that made test pass
3. Store as pattern_type="test_fix"
4. Learn testing strategies that catch issues
5. Improve future test generation quality

On LLM Mistake:

1. Capture LLM-generated code that had issues
2. Capture human correction or alternative LLM response
3. Store as pattern_type="llm_mistake"
4. Learn to avoid similar mistakes
5. Improve consultation strategy

Pattern Application During Code Generation:

1. Query Yantra Codex for relevant patterns based on current task
2. Semantic search using task description embedding
3. Filter patterns by category (if generating error handling, use error_handling patterns)
4. Sort by confidence_score descending
5. Include top 5-10 patterns in LLM context as examples
6. LLM uses patterns as templates for generation

Pattern Ranking Algorithm:

score = (confidence_score × 0.5) +

    (success_count / (success_count + failure_count) × 0.3) +

    (semantic_similarity × 0.2)

Pattern Decay:

1. Patterns not used for 90 days have confidence_score reduced by 10%
2. Patterns with confidence_score below 0.3 are archived (moved to cold storage)
3. Patterns with failure_count > success_count are marked for review
4. Periodic cleanup removes truly obsolete patterns

GNN Architecture:

Graph Structure:

1. Nodes represent code patterns, functions, classes, modules
2. Edges represent relationships (pattern used in function, function calls function, module depends on module)

Node Features:

1. Code embedding vector (1024-dim)
2. Pattern category (code_structure, error_handling, etc.)
3. Success/failure metrics (counts, confidence score)
4. Temporal features (created_at, last_used_at)
5. Project-specific features (language, framework, domain)

Edge Types:

1. UsedIn: Pattern used in function
2. Calls: Function calls function
3. Depends: Module depends on module
4. SimilarTo: Patterns semantically similar
5. FixedBy: Bug fixed by pattern
6. TestsFor: Test pattern for code pattern

GNN Training:

1. Incremental learning on every code generation event
2. Update node embeddings based on success/failure
3. Propagate confidence scores through graph edges
4. Learn which patterns frequently co-occur
5. Identify anti-patterns (frequently leading to bugs)

Performance Requirements:

1. Pattern query under 100ms
2. Pattern learning under 50ms per event
3. Embedding generation under 200ms
4. GNN update under 1 second
5. Memory footprint under 1GB for 100k patterns

Cost Optimization Through Learning:

Month 1: Yantra handles 55% alone, LLM needed 45% → Cost: $9/1000 generations Month 3: Yantra handles 70% alone, LLM needed 30% → Cost: $5/1000 generations Month 6: Yantra handles 85% alone, LLM needed 15% → Cost: $3/1000 generations Month 12: Yantra handles 95% alone, LLM needed 5% → Cost: $1/1000 generations

Cost Reduction: 96% after 1 year through continuous learning!

Learning Metrics:

1. Error Handling: LLM adds try-catch → Yantra learns pattern
2. Code Refactoring: LLM improves structure → Yantra learns refactoring
3. Best Practices: LLM improves naming → Yantra learns conventions
4. Domain Patterns: LLM adds auth checks → Yantra learns domain rules

Yantra Cloud Codex (Optional, Opt-in):

Privacy-Preserving Collective Learning:

What Gets Shared:

1. Logic pattern embeddings (numbers only, not code)
2. Pattern success metrics (aggregated statistics)
3. Anonymized complexity data

What Does NOT Get Shared:

1. Actual code (never shared)
2. Variable/function names (never shared)
3. Business logic details (never shared)
4. User identity (never shared)
5. Project structure (never shared)

Network Effects:

1. 100 users × 50 requests/day = 150k patterns/month → Model v1.1 (65% accuracy)
2. 1k users × 50 requests/day = 1.5M patterns/month → Model v1.6 (80% accuracy)
3. 10k users × 50 requests/day = 15M patterns/month → Model v2.0 (90% accuracy)
4. More users = Better model = Lower LLM costs = Attracts more users (flywheel effect)

Accuracy Targets:

1. Month 1: 55-60% Yantra alone, 95%+ with LLM review (64% cost savings)
2. Month 6: 75-80% Yantra alone, 98%+ with LLM review (88% cost savings)
3. Year 2: 85%+ Yantra alone, 99%+ with LLM review (92% cost savings)
4. Year 3+: 90-95% Yantra alone, 99.5%+ with LLM review (96% cost savings)

Cross-Project Learning (Post-MVP):

1. Aggregate anonymous patterns across users
2. Identify universally successful patterns
3. Share non-proprietary best practices
4. Preserve privacy (no actual code shared, only abstract patterns)

#### 3.1.7 Storage Architecture

Multi-tier storage system optimized for different access patterns, update frequencies, and performance characteristics.

CRITICAL ARCHITECTURAL DECISION: The Yantra Codex (pattern database with 10,000+ patterns) is COMPLETELY SEPARATE from the Dependency Graph system. They have different purposes, query patterns, update frequencies, and scaling needs.

Architecture Overview:

Tier 0: Cloud (PostgreSQL + Redis) → Team Collaboration (Phase 2B)

    ↓

Tier 1: petgraph (in-memory) + SQLite → Dependency Graph + YDoc + Metadata

    ↓

Tier 2: sled → Agent Coordination (Phase 2A)

    ↓

Tier 3: TOML files → Configuration

    ↓

Tier 4: HashMap → moka → Context Cache (Ephemeral)

    ↓

Codex: SQLite + HNSW → Pattern Database (SEPARATE from all tiers)

Complete Storage Architecture Table:

| Tier   | Technology         | Purpose                                | Size     | Access Speed | Persistence | MVP      |
| ------ | ------------------ | -------------------------------------- | -------- | ------------ | ----------- | -------- |
| Tier 0 | PostgreSQL + Redis | Team collaboration (cloud)             | N/A      | <10ms        | Permanent   | Phase 2B |
| Tier 1 | petgraph + SQLite  | Dependency graph + YDoc + Conversation | 50MB-5GB | <50ms        | Permanent   | ✅ MVP   |
| Tier 2 | sled               | Agent coordination                     | <100MB   | <5ms         | Temporary   | Phase 2A |
| Tier 3 | TOML files         | Configuration                          | <1MB     | <1ms         | Permanent   | ✅ MVP   |
| Tier 4 | HashMap → moka     | Context cache                          | <500MB   | <10ms        | Ephemeral   | ✅ MVP   |
| Codex  | SQLite + HNSW      | Pattern database                       | ~500MB   | <50ms        | Permanent   | ✅ MVP   |

---

Tier 0 - Cloud Graph Database (Phase 2B - Team Collaboration):

1. Purpose: Shared dependency graph for multi-user/multi-agent coordination across machines
2. Technology: PostgreSQL + Redis + WebSocket/gRPC
3. Data Stored: SAME as Tier 1 (dependency graph, metadata, embeddings) PLUS cross-machine file locks and team presence tracking
4. Sync: Real-time bidirectional sync with local Tier 1 graph (every 30s or on change)
5. Privacy: Graph structure only (no source code), metadata embeddings only
6. Benefit: Proactive conflict prevention across team members (4 levels: same-file, direct dependency, transitive dependency, semantic dependency)
7. IMPORTANT NOTE: This is NOT "Cloud GNN". The GNN (Yantra Codex intelligence layer) runs locally. This is cloud-hosted graph database storage for coordination only.
8. Deployment: Can be self-hosted or Yantra-hosted
9. Performance: <50ms latency for conflict queries, 99.9% uptime target

Tier 1 - petgraph (In-Memory) + SQLite (Persistence) (MVP - Core System):

1. Purpose: Hot-path dependency graph queries + persistent metadata storage
2. Technology:
   - petgraph: In-memory graph structure for <1ms dependency lookups
   - SQLite: Persistent storage with WAL mode for concurrent reads
3. Data Stored:

- Dependency graph nodes/edges (files, functions, classes, packages with exact versions, tools)
- YDoc blocks database (requirements, specs, architecture, documentation)
- Metadata embeddings (384-dim vectors for semantic search, generated from metadata text NOT code)
- Agent state persistence
- Conversation history (all user/agent messages with 384-dim embeddings for semantic search)
- Known issues database (error patterns + fixes)
- Architecture snapshots

1. Storage Strategy:

- petgraph in-memory for queries (fast pointer-chasing)
- Bincode snapshots to disk every 30s
- Write-ahead log for incremental updates
- On startup: load snapshot + replay WAL

1. Database Files:

- .yantra/graph.db - Main database (dependency graph persistence)
- .yantra/ydoc.db - YDoc blocks
- .yantra/state.db - Agent state + conversation history
- Tables: conversation_sessions, conversation_messages, conversation_summaries
- Embeddings: 384-dim vectors for semantic search (fastembed, all-MiniLM-L6-v2)
- Indexes: FTS5 for keyword search, HNSW for semantic search
- Size: ~1KB per message, ~1MB per 1,000 messages

1. Performance:

- In-memory queries: <1ms for dependency lookups
- SQLite persistence: <50ms for writes
- Memory budget: ~1GB for 100k LOC projects

1. Optimization: Connection pooling (r2d2), prepared statement caching, periodic ANALYZE
2. Why Separate from Codex: Dependency graph queried constantly (hot path), Codex queried rarely (~once per generation), different update frequencies

Tier 2 - sled (Key-Value Store) (Phase 2A - Multi-Agent Coordination):

1. Purpose: Write-heavy agent coordination on local machine
2. Technology: sled (embedded LSM-tree, lock-free, pure Rust)
3. Data Stored:
   - File locks for multiple local agents (who's editing what)
   - Agent registry (active agents on this machine)
   - Agent-to-agent messages
   - Task queue
4. Key Design: Prefixed keys for namespacing (e.g., agent:codegen:state, lock:src/main.py)
5. Performance: ~100k writes/second, <5ms file lock operations
6. Scope: Local machine only (NOT cross-machine, that's Tier 0)
7. Two-Level Locking: Local (Tier 2) for multi-agent + Team-wide (Tier 0) for multi-user
8. Migration Path: MVP uses SQLite, Phase 2A migrates to sled for better write performance

Tier 3 - TOML Files (MVP - Configuration):

1. Purpose: Read-heavy user configuration
2. Technology: TOML config files on disk
3. Data Stored:
   - User preferences (LLM settings, API keys)
   - Project configuration
   - Agent instruction files (.yantra/agents/\*.toml)
4. Location: .yantra/config.toml, .yantra/agents/
5. Performance: <1ms (file system cache)
6. Access Pattern: Read on startup, rarely written

Tier 4 - HashMap → moka (MVP - Ephemeral Cache):

1. Purpose: In-memory LRU cache for performance optimization
2. Technology:
   - MVP: std::collections::HashMap (basic caching)
   - Phase 2: moka (thread-safe LRU with TTL)
3. Data Cached:

- Tokenized contexts (file_path + content_hash → tokenized full/outline/signature + embedding)
- LLM responses (repeated queries with deterministic responses)
- Dependency query results (expensive graph traversals)

1. Cache Policy:

- Max size: 500MB
- Eviction: LRU (Least Recently Used)
- TTL: 1 hour (24 hours for compressed contexts)
- Invalidation: On file save, dependency change, YDoc update

1. Performance Impact:

- Cache hit: <10ms to assemble context
- Cache miss: 100-300ms (tokenization + embedding)
- Target: >80% hit rate

1. Important: Cache misses don't break system, just slower. Purely performance optimization.
2. No Persistence: Pure in-memory, lost on restart (intentional)

---

Yantra Codex Storage (SEPARATE from all tiers above):

Why Separate from Dependency Graph:

1. Codex stores 10,000+ patterns, rarely queried during code generation (~once per generation)
2. Dependency graph queried constantly (<1ms hot path requirement)
3. Different update frequencies (Codex: periodic training, Graph: real-time updates)
4. Different scaling needs (Codex: can be large and slow, Graph: must be fast)

Technology: SQLite + HNSW vector index (separate database file)

Location: .yantra/codex.db (per-project, can sync to cloud if opted-in)

Data Stored:

1. Problem Embeddings: Problem description → 384-dim vector, maps to pattern ID
2. Pattern Templates: Logical structure (JSON), language-specific implementations (Python, JS, Rust), success/failure rates, training metadata
3. Training Examples: CodeContests dataset (10,000+ problems), user-specific examples (learned from this project), LLM-corrected examples (continuous learning)

Schema:

CREATE TABLE patterns (

    pattern_id TEXT PRIMARY KEY,

    problem_embedding BLOB,  -- 384-dim vector

    pattern_json TEXT,       -- Logical structure

    language TEXT,           -- Python, JS, Rust, etc.

    implementation TEXT,     -- Code template

    success_rate REAL,       -- 0.0 to 1.0

    usage_count INTEGER,

    created_at TEXT,

    updated_at TEXT

);

CREATE INDEX idx_patterns_language ON patterns(language);

CREATE INDEX idx_patterns_success_rate ON patterns(success_rate DESC);

HNSW Index: In-memory HNSW index for fast semantic search of problem embeddings

- M parameter: 16 (connections per node)
- ef_construction: 200 (build quality)
- ef_search: 200 (query accuracy)
- Distance metric: Cosine similarity

Access Pattern: <50ms for pattern matching (HNSW semantic search + SQL lookup)

Size: ~500MB for full CodeContests + user patterns

Update Frequency: Periodic (on LLM corrections, manual pattern additions), not real-time like dependency graph

Cloud Sync (Phase 2B - Optional Opt-in):

- User opts in to share anonymized pattern embeddings
- Cloud Codex aggregates patterns across users (network effects)
- Improved models pushed via CDN
- Storage: S3 (model weights) + PostgreSQL (metadata)
- Privacy: Only embeddings shared (numbers), never code

---

Data Flow Example - Code Generation Request:

1. User: "Implement user authentication"
2. Tier 3: Load project config (LLM settings, framework)
3. Tier 1: Query dependency graph for existing auth code
4. Tier 1: Query YDoc for authentication requirements/specs
5. Tier 4: Check if context already cached (hit/miss)
6. Codex: Search for "authentication" patterns (separate database)
7. Tier 1: Get embeddings for semantic search
8. Assemble context from all sources
9. Send to LLM
10. Tier 2: Store agent's work status (if multi-agent)
11. Tier 1: Update graph with new code + YDoc links
12. Tier 4: Cache tokenized context for reuse

---

De-duplication Index (In Tier 1 SQLite):

Purpose: Prevent duplicate files/functions/blocks across all artifact types

Table Schema:

CREATE TABLE deduplication (

    id TEXT PRIMARY KEY,             -- Hash UUID

    content_hash TEXT NOT NULL,      -- SHA-256 of content

    artifact_type TEXT NOT NULL,     -- file, function, doc_block, test

    artifact_id TEXT NOT NULL,       -- ID of actual artifact

    file_path TEXT,                  -- Path if file-based

    created_at TEXT NOT NULL,        -- ISO-8601 timestamp

    similarity_embedding BLOB        -- 384-dim vector

);

CREATE INDEX idx_dedup_hash ON deduplication(content_hash);

CREATE INDEX idx_dedup_type ON deduplication(artifact_type);

Usage: Before creating any artifact: check content_hash for exact duplicates, check similarity_embedding for semantic duplicates (>0.85 cosine similarity), if found prompt agent/user for resolution (reuse existing, update existing, create new with justification).

---

Performance Optimization:

1. SQLite WAL mode: Concurrent reads during writes
2. Connection pooling: r2d2 with max 10 connections
3. Prepared statements: Cached for repeated queries
4. ANALYZE: Periodic query optimization
5. Memory-mapped I/O: For large reads
6. Snapshot persistence: petgraph snapshots every 30s to disk

Backup Strategy:

1. Incremental backups every 15 minutes
2. Full backup daily
3. Backups in .yantra/backups/
4. Automatic cleanup after 30 days
5. Export to JSON for portability

Storage Size Estimates (per 10k LOC project):

| Data Type                     | Size                   | Location            |
| ----------------------------- | ---------------------- | ------------------- |
| Dependency Graph (in-memory)  | ~100MB                 | Tier 1 petgraph     |
| Dependency Graph (persistent) | ~50MB                  | Tier 1 SQLite       |
| YDoc Blocks                   | ~20MB                  | Tier 1 SQLite       |
| Metadata Embeddings           | ~30MB                  | Tier 1 SQLite       |
| Conversation History          | ~1MB per 1K messages   | Tier 1 SQLite       |
| Conversation Embeddings       | ~150KB per 1K messages | Tier 1 SQLite       |
| Pattern Database              | ~500MB                 | Codex SQLite        |
| Agent State                   | ~5MB                   | Tier 1 SQLite       |
| Configuration                 | <1MB                   | Tier 3 TOML         |
| Context Cache                 | <500MB                 | Tier 4 HashMap/moka |
| Total Persistent              | ~600MB                 | -                   |
| Total with Cache              | ~1.1GB                 | -                   |

Key Principles:

1. Codex is SEPARATE: Different database, different purpose, different access pattern
2. petgraph for speed: In-memory graph for <1ms queries (hot path)
3. SQLite for persistence: Reliable, battle-tested, perfect for local storage
4. sled for writes: LSM-tree optimized for write-heavy agent coordination
5. Separation of concerns: Each tier optimized for its specific use case
6. No premature optimization: Start simple (SQLite + HashMap), add complexity (sled + moka) only when needed

#### 3.1.8 Browser Integration with CDP

Enable autonomous browser automation for validation, testing, and self-healing using Chrome DevTools Protocol.

Purpose: Allow agent to launch, control, and monitor web browsers for UI validation, error capture, interactive development workflows, visual regression testing, automated testing, self-healing based on browser behavior, legacy system automation.

Approach: System Chrome + CDP (Find user's installed Chrome/Chromium/Edge, launch with remote debugging enabled, control via Chrome DevTools Protocol, zero-touch user experience).

Technology Stack: Chrome/Chromium with CDP, Rust async runtime for protocol communication, WebSocket connection to browser, JSON-RPC for CDP commands.

CDP Capabilities:

| Category           | CDP Domains                                | Use Cases                        |
| ------------------ | ------------------------------------------ | -------------------------------- |
| Page Control       | Page.navigate, Page.reload                 | Navigate to test URLs            |
| DOM Interaction    | DOM.getDocument, DOM.querySelector         | Find elements, read content      |
| Input Simulation   | Input.dispatchMouseEvent, Input.insertText | Click buttons, fill forms        |
| Network Monitoring | Network.enable, Network.responseReceived   | Monitor API calls, detect errors |
| Console Capture    | Runtime.consoleAPICalled, Log.entryAdded   | Capture JS errors, warnings      |
| Screenshots        | Page.captureScreenshot                     | Visual regression testing        |
| Performance        | Performance.getMetrics, Tracing.start      | Measure load times, FPS          |

Browser Launch Process:

1. Agent requests browser session
2. System launches Chrome with remote debugging port (default 9222)
3. Establish WebSocket connection to CDP endpoint
4. Configure browser: disable extensions, set viewport size, clear cache
5. Return session handle to agent for control

Security Sandbox:

1. Browser binds to localhost only (127.0.0.1, no external access)
2. No external network access to Yantra's internal APIs
3. Sandboxed mode enabled by default
4. No filesystem access beyond project folder
5. Random port selection to avoid conflicts

Browser Validation Workflow:

Step 1 - Deploy to Local: Agent deploys application to localhost (e.g., http://localhost:3000).

Step 2 - Launch Browser: Agent launches Chrome via CDP, navigates to localhost URL.

Step 3 - Execute Test Scenarios: Click through user flows (login, add to cart, checkout), fill forms and submit, verify expected elements appear, check for console errors, monitor network requests for failures.

Step 4 - Capture Evidence: Take screenshots at key steps, record console logs, save network HAR file (HTTP Archive), capture performance metrics (load time, FPS).

Step 5 - Detect Issues: Check for HTTP 500 errors, look for uncaught JavaScript exceptions, verify expected DOM elements exist, validate API response formats, detect broken images or missing resources.

Step 6 - Auto-Fix if Possible:

1. If API 500 error: check backend logs, identify error in code, regenerate buggy code, redeploy and retest
2. If missing DOM element: check frontend code, verify selector correct, fix React component, rebuild and retest
3. If JavaScript error: check console stack trace, identify problematic code, fix code, rebuild and retest

Step 7 - Report to User: Show validation results in UI, highlight failed scenarios with screenshots, provide screenshots of failures, link to relevant code that needs fixing.

Self-Healing Trigger:

1. Browser validation runs automatically after every deployment
2. If validation fails, agent attempts auto-fix up to 3 times
3. If auto-fix succeeds, user notified but no action needed
4. If auto-fix fails after 3 attempts, user prompted for guidance

Performance Requirements:

1. Browser launch under 2 seconds
2. CDP command latency under 100ms
3. Screenshot capture under 500ms
4. Full validation suite under 30 seconds for typical app

Browser Lifecycle Management:

1. Maximum 5 concurrent browser sessions
2. Automatic cleanup after 30 minutes idle
3. Forced cleanup on agent shutdown
4. Reuse existing session for sequential tests (avoid repeated launches)

User Privacy:

1. No telemetry sent to Yantra servers
2. All browser data stays local
3. Anonymous crash reports only (opt-in)

#### 3.1.9 Architecture View System

Autonomous agent-driven architecture visualization showing current system design with automatic generation from code, intent, or requirements.

Status: IN PROGRESS (75% Complete)

Core Principle: Architecture is agent-generated and continuously monitored. Code must align with architecture. This is an agentic platform where all architecture operations happen through the agent, not manual UI interactions.

Agent-Driven Interaction Model:

User (in chat): "Build a REST API with JWT auth" → Agent analyzes intent → Agent generates architecture diagram → Agent auto-saves to database → Architecture View shows read-only visualization → User reviews in Architecture View tab

User (in chat): "Add Redis caching layer" → Agent updates architecture → Agent auto-saves (keeps last 3 versions) → Architecture View updates visualization

Three Core Capabilities:

1. Automatic Architecture Generation - User provides specs/intent → Agent generates architecture → Agent generates code → Agent monitors for deviations
2. Deviation Detection During Implementation - Agent generates code → Checks alignment → Alerts user if deviation → User decides (update arch or fix code)
3. Continuous Alignment Monitoring - Code changes → Compare to architecture → Alert if misaligned → Enforce governance through user decision

Architecture Generation Sources:

From User Intent:

1. LLM analyzes natural language description
2. Generates components (services, databases, APIs)
3. Generates connections (REST calls, message queues, data flow)
4. Creates architecture diagram
5. Displays for user approval

From Code Analysis (Existing Project):

1. Dependency graph traverses all imports
2. Groups files into logical components by directory structure (e.g., all files in src/frontend/ become "Frontend UI" component)
3. Infers connections from imports (cross-directory imports indicate component connections)
4. Infers connection types (REST API, database connection, message queue) based on import patterns
5. Generates architecture diagram from code structure

From Requirements:

1. Parse requirements document (YDoc or external)
2. Extract system components mentioned
3. Identify component interactions from requirements
4. Generate architecture diagram

Auto-Save with Rule of 3 Versioning:

Every architecture change automatically saved with version history. Keep current version + 3 most recent past versions (total 4 versions). When 5th version created, delete oldest (version 1). Versions are immutable once created. Agent can revert to any of 3 past versions.

Version Metadata:

1. Incremental version number (1, 2, 3, 4...)
2. Full architecture state snapshot (components, connections, metadata)
3. Timestamp when created
4. Change type classification (AgentGenerated, AgentUpdated, AgentReverted, GNNSynced)
5. Agent's reasoning for the change
6. Original user message that triggered the change

Architecture Storage: SQLite table storing architecture versions. Each version contains: components array (id, name, type, layer, description, position), connections array (from/to component IDs, connection type, label), metadata (version number, timestamp, change type, user message).

Deviation Detection:

During code generation, check if generated code aligns with architecture:

1. If new component created not in architecture: alert user "New component detected: Redis Cache. Update architecture?"
2. If connection created between components with no defined connection: alert user "New connection detected: Frontend → Redis. Update architecture?"
3. If generated code violates architecture boundaries: block generation, prompt user to update architecture first

Continuous Alignment Monitoring:

1. On every code change, compare to architecture
2. Check if new imports violate component boundaries
3. Check if new files belong to existing components
4. Alert if misalignment detected
5. Enforce governance through user decision (update architecture or fix code)

Performance Targets:

1. Architecture generation from code under 5 seconds for 10k LOC
2. Architecture update and save under 500ms
3. Deviation detection under 200ms per code change
4. Architecture view rendering under 1 second

Agent Commands (Via Chat):

User: "Show me the architecture" → Agent: Opens Architecture View tab, shows current version

User: "Revert to previous architecture" → Agent: Loads version N-1, auto-saves as new version N+1 → Agent: "Reverted to version 5 (from 2 minutes ago)"

User: "Show architecture history" → Agent: Lists last 3 versions with timestamps and changes

User: "Why did you add Redis?" → Agent: Shows version history and reasoning from metadata

What's Working (✅ Implemented):

1. SQLite storage with full CRUD operations
2. Deviation detection (850 lines) with severity calculation
3. Architecture generator (from intent and code)
4. UI components (ArchitectureCanvas, HierarchicalTabs, ComponentNode, ConnectionEdge)
5. Multi-format import (JSON/MD/Mermaid/PlantUML)
6. Export functionality (agent-callable)
7. Dependency graph integration for code analysis
8. Tauri commands (17 backend APIs)
9. Read-only agent-driven UI principle
10. Impact analysis and auto-correction
11. Refactoring safety analyzer
12. Project initialization with architecture discovery

What's Pending (❌ Not Yet Implemented):

1. Rule of 3 versioning (keep 4 versions, auto-delete oldest)
2. Auto-save on every architecture change
3. Real-time deviation alerts (backend → frontend wiring)
4. Orchestrator integration (ensure blocking works in code generation flow)

#### 3.1.10 Documentation System

Comprehensive documentation management integrated with the agentic workflow, tracking features, decisions, changes, and tasks.

Status: ✅ Fully Implemented (November 23, 2025)

Purpose: Automatic extraction and structured presentation of project documentation for transparency and user guidance.

Location: src-tauri/src/documentation/mod.rs (429 lines), Frontend components

Documentation Data Types:

1. Feature: id, title, description, status (Planned/InProgress/Completed), source, timestamp
2. Decision: id, title, context, decision made, rationale, timestamp
3. Change: id, type (FileAdded/FileModified/FileDeleted), description, affected file paths, timestamp
4. Task: id, title, status (Completed/InProgress/Pending), milestone/phase, dependencies, requires user action flag, user action instructions

Data Sources:

1. Documentation extracted from markdown files in project
2. Features extracted from Features tab, requirements, architecture, technical specs
3. Decisions extracted from Decision Log, chat conversations, technical choices
4. Changes extracted from Git commits, Change Log
5. Tasks extracted from Project Plan, task decomposition

Documentation Manager:

1. Rust backend component managing documentation state
2. Workspace directory path tracking
3. Collections of features, decisions, changes, tasks in memory
4. Tauri commands for frontend access (get_features, get_decisions, get_changes, get_tasks)

Frontend Store:

1. SolidJS reactive store
2. Real-time updates via WebSocket
3. Optimistic UI updates with rollback on conflict
4. Tab-based UI showing Features, Decisions, Changes, Tasks, Plan

Four-Panel UI:

1. Features Tab: Shows what features exist (implemented, in-progress, planned)
2. Decisions Tab: Shows why decisions were made (architecture choices, tradeoffs)
3. Changes Tab: Shows what changed (file additions, modifications, deletions)
4. Tasks Tab: Shows what tasks remain (current week/phase progress)

Empty State Handling: When no documentation exists, show helpful message "No features yet. Features will appear here as Agent creates them" with guidance on what each tab shows.

Performance Targets:

1. Load last 100 items under 100ms
2. Real-time update latency under 500ms
3. Search across 1000+ items under 200ms
4. Database query optimization with proper indexes

Business Value: Creates transparency between AI agent and user, ensuring alignment on project state and next actions.

#### 3.1.11 Storage Optimizations

Optimize storage performance for agent operations requiring fast data access.

SQLite Optimizations:

1. WAL (Write-Ahead Logging) mode for concurrent reads during writes
2. Memory-mapped I/O for large reads (PRAGMA mmap_size)
3. Connection pooling with max 10 connections
4. Prepared statement caching for repeated queries
5. Periodic ANALYZE for query plan optimization
6. Automatic VACUUM on idle to reclaim space

Caching Strategy:

1. In-memory LRU cache for frequently accessed data (dependency graph nodes, recent patterns, current agent state)
2. Cache invalidation on writes (immediate)
3. Cache warming on startup (load hot data proactively)
4. Cache size limit 500MB (prevent memory bloat)

Index Strategy:

1. Create indexes on foreign keys (for JOIN performance)
2. Timestamp columns for time-based queries (for sorting)
3. Status/type columns for filtering (for WHERE clauses)
4. Composite indexes for common query patterns (multi-column queries)
5. FTS5 indexes for full-text search on documentation content

Query Optimization:

1. Use EXPLAIN QUERY PLAN to identify slow queries
2. Rewrite queries to use indexes (avoid full table scans)
3. Batch similar queries together (reduce round trips)
4. Use JOINs instead of N+1 queries (reduce total queries)
5. Limit result sets with appropriate LIMIT clauses (reduce data transfer)

Data Archiving:

1. Move old test results to Tier 4 cold storage after 90 days
2. Compress old conversation history after 6 months
3. Archive deprecated patterns after 1 year
4. Automatic cleanup of archived data older than 2 years

Monitoring:

1. Track query execution times (histogram)
2. Alert if queries exceed 500ms (performance regression)
3. Log slow queries for optimization (analyze patterns)
4. Monitor database size growth (detect anomalies)
5. Alert if disk usage exceeds 80% (prevent full disk)

# Yantra: Complete Technical Specification - Part 1B

Sections 3.1.12 through 3.4.1

---

#### 3.1.12 Security Infrastructure

Comprehensive security scanning and validation integrated into the code generation workflow.

Security Domains: Static analysis for code vulnerabilities, secrets detection in code and config, dependency vulnerability scanning against CVE database, license compliance checking, data privacy validation.

Semgrep Integration:

1. Integrate Semgrep for static security analysis
2. Use OWASP rulesets for common vulnerabilities (SQL injection, XSS, CSRF, command injection, path traversal, insecure deserialization, XXE)
3. Custom rules for project-specific security requirements
4. Parallel execution with 4 workers for performance
5. Categorize findings by severity (Critical, High, Medium, Low)

Secrets Detection:

1. Scan for hardcoded secrets using regex patterns
2. Detect: API keys, passwords, tokens, private keys, AWS credentials, database connection strings, OAuth secrets
3. Use TruffleHog patterns for comprehensive detection
4. Alert on any detected secrets before commit
5. Provide guidance on using environment variables instead
6. Suggest adding secrets to .gitignore

Dependency Vulnerability Scanning:

1. Check Python dependencies using Safety tool
2. Check npm dependencies using npm audit
3. Check Rust dependencies using cargo-audit
4. Query CVE database (NVD, Snyk) for known vulnerabilities
5. Identify severity (Critical, High, Medium, Low)
6. Suggest version upgrades to fix vulnerabilities
7. Parallel CVE checks for performance

License Compliance:

1. Scan all dependencies for licenses
2. Flag incompatible licenses (e.g., GPL in proprietary software)
3. Provide license summary report
4. Alert on license changes in dependency updates

Security Scanning Workflow:

1. Agent generates code
2. Security scan runs automatically before any commit
3. Critical and High severity issues must be fixed (blocking)
4. Medium issues generate warnings (non-blocking)
5. Low issues logged but don't block (informational)
6. Agent attempts auto-fix for common patterns
7. If auto-fix fails, prompt user for guidance

Auto-Fix Patterns:

1. SQL injection: Use parameterized queries instead of string concatenation
2. XSS: Use proper escaping/sanitization functions (e.g., DOMPurify)
3. Secrets: Move to environment variables, add .env to .gitignore
4. Path traversal: Validate and sanitize file paths, use path.join() safely
5. Command injection: Use safe subprocess calls with argument arrays (not shell=True)

Parallel Security Scanning:

1. 4 parallel workers for Semgrep rules
2. Parallel file scanning across project
3. Concurrent security rules execution (Semgrep, secrets, CVE)
4. Multiple dependency CVE lookups simultaneously
5. Performance target: scan 1000 files in O(log N) time with parallel workers

Security Integration Points:

CodeGen Phase:

1. Pre-commit security validation (before any Git commit)
2. Automatic security scan before Git commit
3. Zero critical vulnerabilities required (must be fixed before proceeding)

Testing Phase:

1. Security scan as part of validation pipeline
2. Security scan must be clean as quality gate
3. Auto-fix for critical vulnerabilities

Dependency Assessment:

1. CVE database checks during dependency assessment
2. Version vulnerability scanning
3. License compliance checks
4. Compatibility security analysis

Security Metrics & Targets:

Success Criteria:

1. Target: under 3% critical security vulnerabilities (auto-fixed)
2. Target: 95%+ generated code passes security scan
3. Target: security scan time under 10 seconds
4. Zero breaking security changes

Quality Gates:

1. All critical vulnerabilities must be fixed
2. No secrets in committed code
3. All dependencies checked for CVEs
4. Security scan clean before commit

Data Security & Privacy:

Local-Only Architecture:

1. User code never leaves machine (unless explicitly sent to LLM APIs)
2. LLM calls encrypted in transit (HTTPS)
3. No code storage on Yantra servers
4. Anonymous crash reports (opt-in)
5. Usage analytics only, no PII (opt-in)

WebSocket Security:

1. Binds to 127.0.0.1 (localhost only)
2. No external access
3. Random port selection

Chrome Sandbox:

1. Sandboxed mode (default security)
2. No filesystem access beyond project folder
3. No network access to Yantra's internal APIs

### 3.2 AGENTIC FRAMEWORK

Yantra's agentic capabilities organized into four fundamental pillars mirroring human developer capabilities:

Philosophy: The Four Pillars of Autonomous Development

🔍 PERCEIVE → 🧠 REASON → ⚡ ACT → 🔄 LEARN

Sense the environment → Analyze and Decide → Execute Action → Adapt from Feedback

PERCEIVE: Sense the environment through file operations, dependency analysis, code parsing, test results, user input, browser feedback, security scan results.

REASON: Analyze and decide using LLM-powered reasoning, dependency-aware planning, risk assessment, trade-off evaluation, pattern matching from Yantra Codex, architectural alignment checking.

ACT: Execute actions via code generation, file creation/modification, test execution, deployment operations, browser automation, Git operations, tool invocations.

LEARN: Adapt from feedback by capturing success patterns, learning from failures, updating Yantra Codex, improving test strategies, refining error handling, adjusting architectural patterns.

This framework ensures autonomous agents operate systematically: gather information, make informed decisions, take action, and continuously improve.

---

#### 3.1.13 Conversation Memory System

Purpose: Persistent storage and intelligent retrieval of all chat conversations to maintain full context across sessions and enable natural conversational continuity.

Problem Addressed: Agent forgets what was discussed after some time, users must repeat context constantly, no conversation history across app restarts, breaks natural conversation flow.

Solution Overview: Store every message permanently in Tier 1 SQLite, retrieve relevant context adaptively using hierarchical assembly + semantic search, stay within LLM token limits through smart compression.

##### 3.1.13.1 Storage Architecture (Tier 1 - SQLite)

Database Location: .yantra/state.db (Tier 1 - part of core persistence layer)

New Tables:

-- Conversation Sessions

-- One session per project chat, persists across app restarts

CREATE TABLE conversation_sessions (

    session_id TEXT PRIMARY KEY,

    project_id TEXT NOT NULL,

    user_id TEXT NOT NULL DEFAULT 'default',

    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    message_count INTEGER DEFAULT 0,

    -- Auto-generated summary for quick context (updated every 10 messages)

    summary TEXT,

    summary_updated_at TIMESTAMP,

    INDEX idx_project_session (project_id, started_at DESC),

    INDEX idx_last_message (last_message_at DESC)

);

-- Conversation Messages

-- Every message (user and agent) stored permanently

CREATE TABLE conversation_messages (

    message_id TEXT PRIMARY KEY,

    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,

    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),

    content TEXT NOT NULL,

    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Context metadata

    token_count INTEGER,

    intent_summary TEXT,  -- What user wanted (for semantic search)

    -- Link to work sessions (traceability)

    linked_codegen_session TEXT REFERENCES codegen_sessions(session_id),

    linked_test_session TEXT REFERENCES test_sessions(session_id),

    linked_deployment_session TEXT REFERENCES deployment_sessions(session_id),

    -- Semantic search (384-dim vector for HNSW indexing)

    embedding BLOB,  -- Stored as binary, generated from content + intent_summary

    INDEX idx_session_time (session_id, timestamp),

    INDEX idx_role (role),

    INDEX idx_timestamp (timestamp DESC),

    FOREIGN KEY (linked_codegen_session) REFERENCES codegen_sessions(session_id) ON DELETE SET NULL,

    FOREIGN KEY (linked_test_session) REFERENCES test_sessions(session_id) ON DELETE SET NULL,

    FOREIGN KEY (linked_deployment_session) REFERENCES deployment_sessions(session_id) ON DELETE SET NULL

);

-- Conversation Summaries (Compressed Context)

-- Generated every N messages for efficient context loading

CREATE TABLE conversation_summaries (

    summary_id TEXT PRIMARY KEY,

    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,

    message_range_start TEXT NOT NULL,  -- First message_id in range

    message_range_end TEXT NOT NULL,    -- Last message_id in range

    summary_text TEXT NOT NULL,

    token_count INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session (session_id)

);

-- Full-text search index for keyword search

CREATE VIRTUAL TABLE conversation_messages_fts USING fts5(

    message_id UNINDEXED,

    content,

    intent_summary,

    content='conversation_messages',

    content_rowid='rowid'

);

-- Trigger to keep FTS index updated

CREATE TRIGGER conversation_messages_ai AFTER INSERT ON conversation_messages BEGIN

    INSERT INTO conversation_messages_fts(rowid, message_id, content, intent_summary)

    VALUES (new.rowid, new.message_id, new.content, new.intent_summary);

END;

CREATE TRIGGER conversation_messages_ad AFTER DELETE ON conversation_messages BEGIN

    DELETE FROM conversation_messages_fts WHERE rowid = old.rowid;

END;

CREATE TRIGGER conversation_messages_au AFTER UPDATE ON conversation_messages BEGIN

    DELETE FROM conversation_messages_fts WHERE rowid = old.rowid;

    INSERT INTO conversation_messages_fts(rowid, message_id, content, intent_summary)

    VALUES (new.rowid, new.message_id, new.content, new.intent_summary);

END;

Storage Characteristics:

- Persistence: Permanent (Tier 1 SQLite with WAL mode for concurrent reads/writes)
- Size: ~1KB per message average, ~1MB per 1,000 messages, ~150KB per 1,000 embeddings
- Cleanup: User-initiated only (no auto-deletion), messages remain until explicitly deleted
- Backup: Included in standard project backup (.yantra/backups/), compressed with gzip
- Privacy: 100% local storage, never sent to cloud except when calling LLM APIs (user-controlled)

##### 3.1.13.2 Message Persistence Service

Implementation: src/conversation/store.rs (NEW module)

Core Operations:

Save User Message:

pub async fn save_user_message(

    &self,

    session_id: &str,

    content: &str,

    linked_session: Option<&str>

) -> Result`<String>` {

    // 1. Generate embedding for semantic search (384-dim vector)

    let embedding = self.embedder.embed(content).await?;

    // 2. Extract intent summary (first sentence or LLM-generated)

    let intent = self.extract_intent(content).await?;

    // 3. Count tokens using tiktoken

    let token_count = self.token_counter.count(content)?;

    // 4. Insert into database

    let message_id = Uuid::new_v4().to_string();

    sqlx::query!(

    "INSERT INTO conversation_messages

    (message_id, session_id, role, content, token_count, intent_summary,

    embedding, linked_codegen_session)

    VALUES (?, ?, 'user', ?, ?, ?, ?, ?)",

    message_id, session_id, content, token_count, intent, embedding, linked_session

    ).execute(&self.db).await?;

    // 5. Update session last_message_at and increment count

    self.update_session_timestamp(session_id).await?;

    Ok(message_id)

}

1.
2. Save Agent Message:
   - Same flow as user message but with role='assistant'
   - Embeddings generated for agent responses too (enables search)
   - Links to same work session as user message

Get Recent Messages:

pub async fn get_recent_messages(

    &self,

    session_id: &str,

    count: usize

) -> Result<Vec`<Message>`> {

    sqlx::query_as!(

    Message,

    "SELECT message_id, role, content, timestamp, token_count

    FROM conversation_messages

    WHERE session_id = ?

    ORDER BY timestamp DESC

    LIMIT ?",

    session_id, count

    ).fetch_all(&self.db).await

}

1. - Performance: <20ms for 10 messages

- Always loaded for context (highest priority)

Search Similar Messages (Semantic):

pub async fn search_similar_messages(

    &self,

    session_id: &str,

    query: &str,

    top_k: usize

) -> Result<Vec`<Message>`> {

    // 1. Generate query embedding (384-dim)

    let query_embedding = self.embedder.embed(query).await?;

    // 2. Perform HNSW semantic search (cosine similarity)

    let results = self.hnsw_search(session_id, &query_embedding, top_k).await?;

    // 3. Filter by similarity threshold (0.75)

    Ok(results.into_iter()

    .filter(|(msg, score)| *score > 0.75)

    .map(|(msg, _)| msg)

    .collect())

}

1. - Performance: <30ms for top-5 results

- Uses HNSW indexing (fast approximate nearest neighbor)

Create or Get Session:

pub async fn ensure_session(&self, project_id: &str) -> Result`<String>` {

    // Check for existing active session

    if let Some(session) = self.get_active_session(project_id).await? {

    return Ok(session.session_id);

    }

    // Create new session

    let session_id = Uuid::new_v4().to_string();

    sqlx::query!(

    "INSERT INTO conversation_sessions (session_id, project_id)

    VALUES (?, ?)",

    session_id, project_id

    ).execute(&self.db).await?;

    Ok(session_id)

}

Performance Targets:

- Save message: <50ms (includes embedding generation)
- Generate embedding: <10ms (fastembed-rs with ONNX runtime, quantized model)
- Load recent (10): <20ms (indexed query)
- Semantic search (top-5): <30ms (HNSW indexing)
- Keyword search: <50ms (SQLite FTS5)

  3.1.13.3 Adaptive Context Retrieval

Purpose: Intelligently select which messages to include in LLM context without exceeding token limits.

Implementation: src/conversation/context_retrieval.rs (NEW module)

Strategy: Hierarchical assembly with compression (similar to code context assembly)

Token Budget Allocation:

Total LLM Context: 160,000 tokens (Claude Sonnet 4)

Reserve for response: 40,000 tokens

Available for context: 120,000 tokens

Budget Breakdown:

├─ Code Context (L1-L4): 80,000 tokens (66%) - Files, dependencies, patterns

├─ Conversation Context: 20,000 tokens (17%) - Chat history

│ ├─ Recent messages (10): ~8,000 tokens (always full text, highest priority)

│ ├─ Relevant old messages (5): ~8,000 tokens (semantic search, full or compressed)

│ └─ Summaries: ~4,000 tokens (if compression needed)

└─ Reserve buffer: 20,000 tokens (17%) - Safety margin, prevents overflow

Context Assembly Process:

pub async fn assemble_conversation_context(

    &self,

    session_id: &str,

    current_message: &str,

    token_budget: usize  // e.g., 20,000 tokens for conversation

) -> Result`<ConversationContext>` {

    // ===== STEP 1: Always Include Recent Messages (Highest Priority) =====

    let recent_messages = self.store

    .get_recent_messages(session_id, 10)

    .await?;

    let recent_tokens: usize = recent_messages.iter()

    .map(|m| m.token_count)

    .sum();

    let remaining_budget = token_budget.saturating_sub(recent_tokens);

    // ===== STEP 2: Semantic Search for Relevant Old Messages =====

    let relevant_messages = if remaining_budget > 1000 {

    self.store

    .search_similar_messages(session_id, current_message, 5)

    .await?

    .into_iter()

    .filter(|msg| !recent_messages.contains(msg))  // Deduplicate

    .collect()

    } else {

    vec![]

    };

    let relevant_tokens: usize = relevant_messages.iter()

    .map(|m| m.token_count)

    .sum();

    // ===== STEP 3: Check if Compression Needed =====

    let total_tokens = recent_tokens + relevant_tokens;

    if total_tokens <= token_budget {

    // Fits! Return as-is

    return Ok(ConversationContext {

    recent_messages,

    relevant_messages,

    summaries: vec![],

    total_tokens,

    });

    }

    // ===== STEP 4: Compress Older Messages if Needed =====

    // Keep recent messages as-is (full text, never compress)

    // Summarize relevant messages to save tokens

    let compressed_relevant = self.compress_messages(&relevant_messages).await?;

    Ok(ConversationContext {

    recent_messages,  // Full text (highest priority)

    relevant_messages: vec![],  // Replaced by summary

    summaries: vec![compressed_relevant],  // Compressed version

    total_tokens: recent_tokens + compressed_relevant.token_count,

    })

}

async fn compress_messages(&self, messages: &[Message]) -> Result`<ConversationSummary>` {

    // Use LLM to generate concise summary

    let content: String = messages.iter()

    .map(|m| format!("[{}]: {}", m.role, m.content))

    .collect::<Vec_>()

    .join("\n");

    let prompt = format!(

    "Summarize this conversation history concisely (max 200 words), \

    focusing on key decisions, context, and user intent:\n\n{}",

    content

    );

    let summary = self.llm.generate(&prompt).await?;

    let token_count = self.token_counter.count(&summary)?;

    Ok(ConversationSummary {

    summary_text: summary,

    token_count,

    message_count: messages.len(),

    })

}

Adaptive Behavior:

- Short conversations (<20 messages): All messages loaded, no compression needed
- Medium conversations (20-100 messages): Recent + Relevant, minimal compression
- Long conversations (100+ messages): Recent + Summaries, aggressive compression
- Very long (1000+ messages): Recent only + high-level summary of entire conversation

##### 3.1.13.4 Conversation Search

Purpose: Enable users and agents to search past conversations for specific information.

Implementation: src/conversation/search.rs (NEW module)

Search Modes:

1. Keyword Search (Fast, exact matches):

pub async fn keyword_search(

    &self,

    session_id: &str,

    query: &str

) -> Result<Vec`<SearchResult>`> {

    sqlx::query_as!(

    SearchResult,

    "SELECT cm.message_id, cm.role, cm.content, cm.timestamp,

    snippet(conversation_messages_fts, 1, '`<mark>`', '`</mark>`', '...', 64) as snippet

    FROM conversation_messages cm

    JOIN conversation_messages_fts fts ON cm.rowid = fts.rowid

    WHERE cm.session_id = ? AND conversation_messages_fts MATCH ?

    ORDER BY cm.timestamp DESC

    LIMIT 50",

    session_id, query

    ).fetch_all(&self.db).await

}

- Uses SQLite FTS5 (full-text search)
- Performance: <50ms
- Returns: Messages with highlighted snippets

2. Semantic Search (Smart, meaning-based):

pub async fn semantic_search(

    &self,

    session_id: &str,

    query: &str,

    top_k: usize

) -> Result<Vec`<SearchResult>`> {

    // Generate query embedding

    let query_embedding = self.embedder.embed(query).await?;

    // HNSW search with cosine similarity

    let results = self.hnsw_index

    .search(&query_embedding, top_k)

    .await?;

    // Filter by similarity threshold (0.75)

    // Return with relevance scores

    Ok(results.into_iter()

    .filter(|(_, score)| *score > 0.75)

    .map(|(msg, score)| SearchResult {

    message: msg,

    relevance_score: score,

    snippet: Self::extract_snippet(&msg.content, 64),

    })

    .collect())

}

- Uses 384-dim embeddings + HNSW indexing
- Understands synonyms, related concepts
- Performance: <30ms for top-5
- Returns: Messages with relevance scores (0.0-1.0)

3. Date Range Search:

pub async fn search_by_date_range(

    &self,

    session_id: &str,

    start_date: DateTime`<Utc>`,

    end_date: DateTime`<Utc>`

) -> Result<Vec`<Message>`> {

    sqlx::query_as!(

    Message,

    "SELECT * FROM conversation_messages

    WHERE session_id = ?

    AND timestamp BETWEEN ? AND ?

    ORDER BY timestamp ASC",

    session_id, start_date, end_date

    ).fetch_all(&self.db).await

}

- Performance: <20ms
- Use case: "What did we discuss last week?"

4. Linked Work Search (Find conversations about specific code):

pub async fn search_by_linked_work(

    &self,

    work_session_id: &str

) -> Result<Vec`<Message>`> {

    sqlx::query_as!(

    Message,

    "SELECT * FROM conversation_messages

    WHERE linked_codegen_session = ?

    OR linked_test_session = ?

    OR linked_deployment_session = ?

    ORDER BY timestamp ASC",

    work_session_id, work_session_id, work_session_id

    ).fetch_all(&self.db).await

}

- Use case: "Show me the chat that led to this code generation"
- Enables full traceability: UI → Chat → Code

  3.1.13.5 Work Session Linking

Purpose: Connect conversation messages to generated code/tests/deployments for full traceability.

Linking Strategy:

When agent starts work based on user message:

pub async fn start_code_generation(

    &self,

    user_message_id: &str,

    user_intent: &str

) -> Result`<String>` {

    // 1. Create codegen session

    let codegen_session_id = self.state_machine

    .create_codegen_session(user_intent)

    .await?;

    // 2. Link conversation message to codegen session

    sqlx::query!(

    "UPDATE conversation_messages

    SET linked_codegen_session = ?

    WHERE message_id = ?",

    codegen_session_id, user_message_id

    ).execute(&self.db).await?;

    // 3. When agent responds, link response too

    let agent_message_id = self.conversation_store

    .save_agent_message(

    session_id,

    "I'll generate the code now...",

    Some(&codegen_session_id)

    ).await?;

    Ok(codegen_session_id)

}

Traceability Chain:

User Message (conversation_messages)

└─ linked_codegen_session → CodeGen Session (codegen_sessions)

├─ generated_code → Files (file_path references)

├─ linked_test_session → Test Session (test_sessions)

│ └─ test_results → Test outcomes

└─ linked_deployment_session → Deployment (deployment_sessions)

    └─ deployment_url → Live system

Querying the Chain:

pub async fn get_artifacts_from_message(

    &self,

    message_id: &str

) -> Result`<ArtifactChain>` {

    let message = self.get_message(message_id).await?;

    let mut artifacts = ArtifactChain::new();

    if let Some(codegen_id) = message.linked_codegen_session {

    let codegen = self.get_codegen_session(&codegen_id).await?;

    artifacts.code_files = codegen.generated_files;

    if let Some(test_id) = codegen.linked_test_session {

    let tests = self.get_test_session(&test_id).await?;

    artifacts.test_results = tests.results;

    }

    if let Some(deploy_id) = codegen.linked_deployment_session {

    let deployment = self.get_deployment_session(&deploy_id).await?;

    artifacts.deployment_url = deployment.url;

    }

    }

    Ok(artifacts)

}

UI Integration:

- User clicks on message in chat
- UI shows: "This message generated: src/auth/login.py, src/auth/jwt.py"
- UI shows: "Tests: 15 passed, 0 failed"
- UI shows: "Deployed to: https://myapp.railway.app"
- Full audit trail from chat to production

##### 3.1.13.6 Performance Targets

| Operation               | Target | Implementation                            | Notes                         |
| ----------------------- | ------ | ----------------------------------------- | ----------------------------- |
| Save message            | <50ms  | Direct SQLite insert with WAL             | Includes embedding generation |
| Generate embedding      | <10ms  | fastembed-rs (ONNX, quantized)            | all-MiniLM-L6-v2 model        |
| Load recent (10)        | <20ms  | Indexed query on (session_id, timestamp)  | Always fast, hot path         |
| Semantic search (top-5) | <30ms  | HNSW indexing (hnsw_rs crate)             | Approximate nearest neighbor  |
| Keyword search          | <50ms  | SQLite FTS5 full-text search              | Exact matches                 |
| Context assembly        | <200ms | Parallel: recent + semantic + compression | Total conversation + code     |
| User search UI          | <100ms | Cached results, lazy load messages        | Instant feel                  |
| Date range query        | <20ms  | Indexed query on timestamp                | Fast range scan               |
| Linked work query       | <20ms  | Foreign key indexed query                 | Fast join                     |

Scalability:

- 1,000 messages: All operations <100ms
- 10,000 messages: All operations <300ms
- 100,000 messages: All operations <1s (may need archiving strategy)

Storage Growth:

- Average message: 1KB (text + metadata)
- Average embedding: 150 bytes (384 floats \* 4 bytes, compressed)
- 1,000 messages = ~1.15MB
- 10,000 messages = ~11.5MB
- 100,000 messages = ~115MB (includes all embeddings)

##### 3.1.13.7 Privacy & Security

Data Location:

- ✅ 100% local storage (Tier 1 SQLite: .yantra/state.db)
- ✅ Never sent to cloud/servers (except when calling LLM APIs, user-controlled)
- ✅ Included in encrypted backups (.yantra/backups/, gzip compressed)

Sensitive Data Handling:

- ❌ No API keys/passwords stored in conversation (caught by secrets scanner)
- ❌ No PII in embeddings (only semantic content hashes)
- ✅ User can delete individual messages or entire sessions
- ✅ Messages cleared when project deleted
- ✅ No cross-project data leakage (scoped by project_id)

LLM Privacy: When conversation context sent to LLM APIs:

- User-initiated (agent generation requests only)
- Encrypted in transit (HTTPS/TLS)
- Not stored on LLM provider side (ephemeral, per request)
- User controls which LLM provider (OpenRouter, direct APIs)
- Desktop-first means code never leaves machine

Access Control:

- Single-user desktop app (no multi-user concerns in MVP)
- File system permissions protect .yantra/ directory
- SQLite database file readable only by user account
- Phase 2B (Cloud): Multi-user access controls, team permissions

Audit Trail:

- All conversation operations logged to .yantra/logs/conversation.log
- Includes: message saves, searches, deletions, session creation
- Retention: 30 days, auto-rotate
- Format: JSON lines for easy parsing

Data Retention:

- Conversations kept indefinitely by default
- User can manually delete old conversations
- Backup retention: 30 days (configurable)
- No automatic expiration (user controls data lifecycle)

Compliance (Enterprise, Phase 2B):

- GDPR: User can export/delete all personal data
- CCPA: Data transparency, deletion rights
- SOC 2: Audit logs, access controls (future)

##### 3.1.13.8 Migration Path (For Existing Users)

For projects created before this feature:

// Run once on app startup after upgrade

pub async fn migrate_existing_projects(&self) -> Result<()> {

    for project in self.get_all_projects().await? {

    // Check if conversation session already exists

    if self.conversation_store.has_session(&project.id).await? {

    continue; // Already migrated

    }

    // Create conversation session for project

    let session_id = self.conversation_store

    .ensure_session(&project.id)

    .await?;

    // Backfill from state machine sessions (best effort)

    for codegen_session in self.get_codegen_sessions(&project.id).await? {

    // Create synthetic conversation message from user_intent

    if let Some(intent) = codegen_session.user_intent {

    let message_id = self.conversation_store

    .save_user_message(&session_id, &intent, Some(&codegen_session.id))

    .await?;

    // Generate embedding for searchability

    self.backfill_embedding(&message_id).await?;

    }

    }

    log::info!("Migrated project {} to conversation memory", project.id);

    }

    Ok(())

}

User Communication:

- Show one-time notification: "We've added conversation memory! Your future chats will be fully preserved."
- Explain: "Previous conversations partially restored from work history where available."
- Provide: "You can now search your chat history using the search bar."

Backward Compatibility:

- Old projects work without migration (sessions created on-demand)
- No breaking changes to existing functionality
- Graceful degradation if database schema missing

---

### 3.3 AGENTIC PRIMITIVES/TOOLS

Yantra Unified Tool Interface (UTI) abstracts underlying protocol differences (LSP, MCP, DAP, built-in) and presents single consistent API for AI agent to discover, invoke, and manage tools.

#### 3.3.0 Unified Tool Interface (UTI)

The UTI provides protocol-agnostic abstraction layer enabling agent to use any tool regardless of underlying protocol.

Architecture Components: Protocol Router (routes requests to MCP/LSP/DAP/Builtin), Tool Adapters (45+ tools across 4 protocols), Consumer Abstraction (LLM Agent + Workflow Executor), Protocol Selection (auto-routing by capability).

Protocol Selection Framework:

| Question                                    | If YES →                      |
| ------------------------------------------- | ----------------------------- |
| Does editor need it real-time while typing? | LSP (Editor only)             |
| Is it core differentiator we must control?  | Builtin                       |
| Is it discrete query agent makes?           | MCP                           |
| Does it need streaming output for progress? | Builtin or MCP with streaming |
| Is there well-maintained community server?  | MCP                           |

UTI Benefits: Single API for agent (no protocol complexity), easy tool addition (implement adapter), protocol flexibility (switch protocols without changing agent code), built-in error handling and retries, automatic capability discovery.

Tool Adapter Interface: Each tool adapter implements standard interface with methods: discover_capabilities(), invoke(tool_name, parameters), stream_output(callback), handle_error(error).

Protocol Adapters:

LSP Adapter: Connects to language servers (Python LSP, Rust Analyzer), provides code completion, go-to-definition, find references, hover information, diagnostics.

MCP Adapter: Connects to MCP servers (Git, filesystem, external APIs), provides tool discovery, tool invocation, streaming responses, error handling.

DAP Adapter: Connects to debug adapters (Python debugger, Rust debugger), provides breakpoint setting, step execution, variable inspection, stack trace retrieval.

Builtin Adapter: Direct Rust implementations for core tools (dependency graph queries, Yantra Codex access, security scanning, test execution, browser control).

#### 3.3.1 PERCEIVE - Input & Sensing Layer

Tools for gathering information about project state, codebase, dependencies, and environment.

File System Operations:

| Tool             | Purpose                                     | Protocol |
| ---------------- | ------------------------------------------- | -------- |
| file_read        | Read file contents with encoding detection  | Builtin  |
| file_write       | Create/overwrite files                      | Builtin  |
| file_edit        | Surgical edits (line range, search-replace) | Builtin  |
| file_delete      | Remove files safely                         | Builtin  |
| file_move        | Rename/move files with dependency updates   | Builtin  |
| file_copy        | Duplicate files                             | Builtin  |
| directory_create | Create directories recursively              | Builtin  |
| directory_list   | List contents with filters                  | Builtin  |
| directory_tree   | Get full project structure                  | Builtin  |
| file_search      | Find files by name/pattern/glob             | Builtin  |
| file_metadata    | Get file size, modified time, permissions   | Builtin  |
| file_watch       | Monitor for changes (reactive agents)       | Builtin  |
| docx_read        | Read Word documents                         | Builtin  |
| pdf_read         | Extract text from PDFs                      | Builtin  |

Dependency Analysis:

| Tool                   | Purpose                          | Protocol |
| ---------------------- | -------------------------------- | -------- |
| query_dependencies     | Find all dependencies of file    | Builtin  |
| query_dependents       | Find all files depending on file | Builtin  |
| find_imports           | Find all imports in file         | Builtin  |
| find_callers           | Find all callers of function     | Builtin  |
| impact_analysis        | Analyze impact of changing file  | Builtin  |
| build_dependency_graph | Generate full project graph      | Builtin  |
| get_module_boundaries  | Identify architectural layers    | Builtin  |

Code Intelligence:

| Tool               | Purpose                                        | Protocol          |
| ------------------ | ---------------------------------------------- | ----------------- |
| parse_ast          | Get AST for file/snippet                       | Builtin           |
| get_symbols        | Extract functions, classes, variables, imports | Builtin           |
| get_references     | Find all usages of a symbol                    | MCP/Builtin       |
| get_definition     | Jump to definition                             | MCP/Builtin       |
| get_scope          | Get scope context for a position               | Builtin           |
| get_diagnostics    | Syntax errors, warnings                        | Builtin           |
| semantic_search    | Search code by meaning                         | Builtin           |
| get_call_hierarchy | Incoming/outgoing calls                        | Builtin           |
| get_type_hierarchy | Class inheritance chains                       | MCP/Builtin       |
| hover_info         | Get documentation for symbol                   | LSP (Editor-only) |

Note: Tree-sitter is primary for code intelligence (Builtin). MCP fallback via Pylance/rust-analyzer for advanced features. LSP is for editor UI only, not exposed to agent.

Test & Validation:

| Tool              | Purpose                        | Protocol |
| ----------------- | ------------------------------ | -------- |
| get_test_results  | Retrieve last test run results | Builtin  |
| get_coverage      | Get code coverage metrics      | Builtin  |
| get_security_scan | Get security scan results      | Builtin  |

Environment Sensing:

| Tool                   | Purpose                         | Protocol |
| ---------------------- | ------------------------------- | -------- |
| get_installed_packages | List installed dependencies     | Builtin  |
| check_environment      | Verify Python/Node/Rust version | Builtin  |
| get_git_status         | Get Git repository status       | MCP      |
| env_get                | Get environment variable        | Builtin  |
| env_set                | Set environment variable        | Builtin  |
| get_cpu_usage          | CPU metrics for optimization    | Builtin  |
| get_memory_usage       | Memory stats                    | Builtin  |
| get_disk_usage         | Disk space monitoring           | Builtin  |
| should_throttle        | Adaptive resource management    | Builtin  |

Database Operations:

| Tool       | Purpose                                  | Protocol |
| ---------- | ---------------------------------------- | -------- |
| db_connect | Establish connection with pooling        | MCP      |
| db_query   | Execute SELECT (read-only, validated)    | MCP      |
| db_execute | Execute INSERT/UPDATE/DELETE (validated) | MCP      |
| db_schema  | Get tables, columns, types, constraints  | MCP      |
| db_explain | Query execution plan                     | MCP      |
| db_migrate | Run migrations with rollback             | MCP      |
| db_seed    | Insert test data                         | MCP      |

Note: MCP primary via DB-specific MCP servers (Postgres, MySQL, MongoDB). Builtin fallback for SQLite.

API Monitoring:

| Tool                  | Purpose                       | Protocol |
| --------------------- | ----------------------------- | -------- |
| api_import_spec       | Import OpenAPI/Swagger specs  | MCP      |
| api_validate_contract | Detect breaking API changes   | MCP      |
| api_health_check      | Test endpoint availability    | Builtin  |
| api_rate_limit_check  | Track and predict rate limits | Builtin  |
| api_mock              | Create mock server from spec  | MCP      |
| api_test              | Test endpoint with assertions | MCP      |

Note: MCP primary for external APIs. Builtin HTTP client for health checks.

Browser Sensing:

| Tool               | Purpose                      | Protocol      |
| ------------------ | ---------------------------- | ------------- |
| get_console_logs   | Get browser console output   | Builtin (CDP) |
| get_network_logs   | Get browser network requests | Builtin (CDP) |
| capture_screenshot | Take browser screenshot      | Builtin (CDP) |
| get_dom_element    | Query DOM for element        | Builtin (CDP) |

#### 3.3.2 REASON - Decision-Making & Analysis Layer

Tools for analyzing information and making informed decisions.

Pattern Matching:

| Tool                  | Purpose                          | Protocol |
| --------------------- | -------------------------------- | -------- |
| search_codex_patterns | Find similar code patterns       | Builtin  |
| find_bug_fix_patterns | Find how similar bugs were fixed | Builtin  |
| find_test_strategies  | Find successful test patterns    | Builtin  |
| find_api_patterns     | Find API design patterns used    | Builtin  |

Risk Assessment:

| Tool                    | Purpose                          | Protocol |
| ----------------------- | -------------------------------- | -------- |
| analyze_blast_radius    | Assess impact of proposed change | Builtin  |
| detect_breaking_changes | Identify API breaking changes    | Builtin  |
| assess_test_coverage    | Check if change is well-tested   | Builtin  |
| check_security_risk     | Evaluate security implications   | Builtin  |

Architectural Analysis:

| Tool                         | Purpose                               | Protocol |
| ---------------------------- | ------------------------------------- | -------- |
| check_architecture_alignment | Verify code follows architecture      | Builtin  |
| detect_boundary_violations   | Find component boundary violations    | Builtin  |
| analyze_dependencies         | Check for circular dependencies       | Builtin  |
| validate_design_patterns     | Verify design patterns used correctly | Builtin  |

LLM Consultation:

| Tool                    | Purpose                             | Protocol |
| ----------------------- | ----------------------------------- | -------- |
| consult_primary_llm     | Get response from Claude Sonnet 4   | Builtin  |
| consult_secondary_llm   | Get response from GPT-4 Turbo       | Builtin  |
| consult_specialist_llm  | Get response from specialized model | Builtin  |
| aggregate_llm_responses | Combine multiple LLM responses      | Builtin  |

Confidence Scoring: agent/confidence.rs (320 lines) implemented, RiskLevel enum: Low/Medium/High/Critical.

Decision Logging: State machine persistence in SQLite, full audit trail.

Multi-LLM Orchestration: llm/multi_llm_manager.rs (13 providers) implemented.

Validation Pipeline: agent/validation.rs (412 lines) implemented.

Error Analysis: agent/orchestrator.rs::analyze_error() implemented.

Adaptive Context Assembly: Hierarchical context with dependency graph implemented.

#### 3.3.3 ACT - Execution & Action Layer

Tools for taking concrete actions: generating code, modifying files, executing tests, deploying.

Code Generation:

| Tool                   | Purpose                       | Protocol |
| ---------------------- | ----------------------------- | -------- |
| generate_code          | Generate new code from spec   | Builtin  |
| generate_tests         | Generate tests for code       | Builtin  |
| generate_documentation | Generate API documentation    | Builtin  |
| generate_boilerplate   | Generate project scaffolding  | Builtin  |
| refactor_code          | Refactor existing code        | Builtin  |
| refactor_extract       | Extract function/class/module | Builtin  |
| refactor_inline        | Inline function/variable      | Builtin  |

File Manipulation:

| Tool        | Purpose                      | Protocol |
| ----------- | ---------------------------- | -------- |
| create_file | Create new file with content | Builtin  |
| modify_file | Modify existing file         | Builtin  |
| delete_file | Delete file                  | Builtin  |
| move_file   | Move/rename file             | Builtin  |

Test Execution:

| Tool              | Purpose                          | Protocol |
| ----------------- | -------------------------------- | -------- |
| run_tests         | Execute test suite               | Builtin  |
| run_single_test   | Execute specific test            | Builtin  |
| run_coverage      | Execute tests with coverage      | Builtin  |
| run_stress_tests  | Execute concurrency stress tests | Builtin  |
| test_run_affected | Run tests for changed code only  | Builtin  |
| test_generate     | Auto-generate test cases         | Builtin  |
| e2e_run           | Browser/integration tests        | Builtin  |

Note: Testing is core to "never breaks" guarantee. Builtin exclusive with GNN integration for affected test detection.

Build & Compilation:

| Tool              | Purpose                     | Protocol |
| ----------------- | --------------------------- | -------- |
| build_project     | Full build                  | Builtin  |
| build_incremental | Changed files only          | Builtin  |
| build_check       | Type-check without emitting | Builtin  |
| build_clean       | Remove artifacts            | Builtin  |
| build_watch       | Continuous build on changes | Builtin  |
| build_optimize    | Production optimized build  | Builtin  |
| build_profile     | Build with profiling        | Builtin  |

Note: Build orchestration via terminal commands coordinated with dependency graph, testing, and deployment.

Package Management:

| Tool          | Purpose                      | Protocol |
| ------------- | ---------------------------- | -------- |
| pkg_install   | Install package(s)           | Builtin  |
| pkg_uninstall | Remove package(s)            | Builtin  |
| pkg_update    | Update package(s)            | Builtin  |
| pkg_list      | List installed packages      | Builtin  |
| pkg_search    | Search for packages          | Builtin  |
| pkg_outdated  | Check for updates            | Builtin  |
| pkg_audit     | Security vulnerability check | Builtin  |

Note: Package operations via terminal with GNN integration. Security audit integrates with vulnerability databases.

Deployment:

| Tool                | Purpose                      | Protocol |
| ------------------- | ---------------------------- | -------- |
| deploy_local        | Deploy to localhost          | Builtin  |
| deploy_railway      | Deploy to Railway            | MCP      |
| deploy_vercel       | Deploy to Vercel             | MCP      |
| deploy_aws          | Deploy to AWS                | MCP      |
| deploy_gcp          | Deploy to GCP                | MCP      |
| deploy_docker       | Build and push Docker image  | Builtin  |
| health_check        | Check deployment health      | Builtin  |
| rollback_deployment | Rollback to previous version | Builtin  |

Note: Railway is MVP focus. MCP servers for platform-specific deployments.

Browser Automation:

| Tool              | Purpose          | Protocol      |
| ----------------- | ---------------- | ------------- |
| browser_navigate  | Navigate to URL  | Builtin (CDP) |
| browser_click     | Click element    | Builtin (CDP) |
| browser_fill_form | Fill form fields | Builtin (CDP) |
| browser_submit    | Submit form      | Builtin (CDP) |
| browser_wait      | Wait for element | Builtin (CDP) |

Git Operations:

| Tool                 | Purpose                                  | Protocol    |
| -------------------- | ---------------------------------------- | ----------- |
| git_setup            | Chat-based Git configuration & auth      | Builtin     |
| git_authenticate     | Store credentials securely               | Builtin     |
| git_test_connection  | Validate authentication works            | Builtin     |
| git_status           | Current state                            | MCP/Builtin |
| git_diff             | Changes (staged, unstaged, between refs) | MCP/Builtin |
| git_log              | Commit history                           | MCP/Builtin |
| git_blame            | Line-by-line attribution                 | MCP/Builtin |
| git_commit           | Create commit with auto-messages         | MCP/Builtin |
| git_push             | Push commits to remote                   | MCP/Builtin |
| git_pull             | Pull latest changes                      | MCP/Builtin |
| git_branch           | Create/switch/list branches              | MCP/Builtin |
| git_checkout         | Checkout files/branches                  | MCP/Builtin |
| git_merge            | Merge branches                           | MCP/Builtin |
| git_stash            | Stash/pop changes                        | MCP/Builtin |
| git_reset            | Undo changes                             | MCP/Builtin |
| git_clone            | Clone repository                         | MCP/Builtin |
| git_resolve_conflict | AI-powered conflict resolution           | MCP/Builtin |

Note: MCP primary via @modelcontextprotocol/server-git. Builtin fallback via git2-rs. Chat-based setup guides users through one-time authentication with secure keychain storage.

YDoc Operations:

| Tool                 | Purpose                      | Protocol |
| -------------------- | ---------------------------- | -------- |
| create_ydoc_document | Create new YDoc document     | Builtin  |
| create_ydoc_block    | Create new block in document | Builtin  |
| update_ydoc_block    | Update existing block        | Builtin  |
| link_ydoc_to_code    | Create graph edge doc → code | Builtin  |
| search_ydoc_blocks   | Search documentation blocks  | Builtin  |

Terminal & Shell Execution:

Built-in terminal that AI agent can use to run commands autonomously. Execute code, install packages, run tests, deploy - all from within Yantra without switching to external terminals.

Implementation: src-tauri/src/terminal/executor.rs

Capabilities:

1. Command execution with real-time streaming output
2. Environment variable management
3. Working directory control
4. Background process management
5. Exit code capture and error handling

#### 3.3.4 LEARN - Feedback & Adaptation Layer

Tools for capturing outcomes and improving future performance.

Pattern Capture:

| Tool                   | Purpose                        | Protocol |
| ---------------------- | ------------------------------ | -------- |
| record_success_pattern | Store successful code pattern  | Builtin  |
| record_bug_fix         | Store bug and fix pattern      | Builtin  |
| record_test_strategy   | Store effective test pattern   | Builtin  |
| record_llm_mistake     | Store LLM error and correction | Builtin  |

Feedback Processing:

| Tool                       | Purpose                        | Protocol |
| -------------------------- | ------------------------------ | -------- |
| process_test_failure       | Analyze why test failed        | Builtin  |
| process_security_finding   | Analyze security vulnerability | Builtin  |
| process_user_feedback      | Incorporate user corrections   | Builtin  |
| process_deployment_failure | Analyze deployment issue       | Builtin  |

Codex Updates:

| Tool                            | Purpose                           | Protocol |
| ------------------------------- | --------------------------------- | -------- |
| update_pattern_confidence       | Adjust pattern confidence score   | Builtin  |
| increment_success_count         | Increment pattern success counter | Builtin  |
| increment_failure_count         | Increment pattern failure counter | Builtin  |
| archive_low_confidence_patterns | Archive patterns below threshold  | Builtin  |

Analytics:

| Tool                        | Purpose                       | Protocol |
| --------------------------- | ----------------------------- | -------- |
| track_generation_time       | Record how long code gen took | Builtin  |
| track_test_pass_rate        | Record test success rate      | Builtin  |
| track_security_scan_results | Record security findings      | Builtin  |
| track_deployment_success    | Record deployment outcomes    | Builtin  |

#### 3.3.5 Cross-Cutting Primitives

Tools that span multiple pillars and provide system-wide capabilities.

State Management:

| Tool                   | Purpose                     | Protocol |
| ---------------------- | --------------------------- | -------- |
| save_agent_state       | Persist current agent state | Builtin  |
| load_agent_state       | Restore agent state         | Builtin  |
| create_checkpoint      | Create rollback point       | Builtin  |
| rollback_to_checkpoint | Restore from checkpoint     | Builtin  |

Context Management (Enhanced with Conversation Memory):

| Tool                 | Purpose                                              | Protocol | Enhancement                       |
| -------------------- | ---------------------------------------------------- | -------- | --------------------------------- |
| assemble_context     | Build context for LLM (code + conversation)          | Builtin  | Now includes conversation history |
| compress_context     | Compress context to fit window (code + conversation) | Builtin  | Now compresses both types         |
| prioritize_context   | Rank context by relevance (code + conversation)      | Builtin  | Unified ranking                   |
| cache_context        | Cache frequently used context                        | Builtin  | Caches conversation too           |
| context_add          | Add to persistent conversation DB                    | Builtin  | Enhanced: permanent storage       |
| context_search       | Search code + conversation                           | Builtin  | Enhanced: unified search          |
| context_summarize    | Compress code + conversation                         | Builtin  | Enhanced: both types              |
| conversation_search  | Search past conversations (keyword/semantic)         | Builtin  | NEW: conversation-specific        |
| conversation_history | Retrieve conversation for context                    | Builtin  | NEW: adaptive retrieval           |
| conversation_link    | Link message to work session                         | Builtin  | NEW: traceability                 |

Enhanced Tool Details:

context_add (Enhanced):

- Before: Added to in-memory working context (ephemeral)
- After: Saves to persistent conversation_messages table (permanent)
- Usage: Automatically called when user/agent sends message
- Storage: Tier 1 SQLite with embeddings for semantic search
- Performance: <50ms including embedding generation

context_search (Enhanced):

- Before: Searched codebase only (files, functions, patterns)
- After: Unified search across code AND conversation
- Modes: code_only, conversation_only, both (default)
- Returns: Ranked results from both sources
- Performance: <100ms for unified search

context_summarize (Enhanced):

- Before: Compressed code context when token budget tight
- After: Intelligently compresses code AND conversation
- Strategy: Always keep recent conversation full-text, compress old
- LLM-based: Generates concise summaries (max 200 words)
- Performance: <100ms for compression

conversation_search (NEW):

- Purpose: Search past conversations specifically
- Modes:
  - keyword: SQLite FTS5 full-text search (<50ms)
  - semantic: HNSW embedding search (<30ms)
  - date_range: Time-based filtering (<20ms)
  - linked_work: Find chat about specific code (<20ms)
- Returns: Messages with snippets, timestamps, relevance scores
- UI: Search bar in chat interface

conversation_history (NEW):

- Purpose: Retrieve conversation messages for context assembly
- Modes:
  - recent: Last N messages (default: 10)
  - relevant: Semantically similar to current message (top-K)
  - adaptive: Recent + Relevant within token budget
  - full: Entire conversation (with compression if needed)
- Usage: Called by context assembly system automatically
- Performance: <200ms for adaptive retrieval

conversation_link (NEW):

- Purpose: Link chat message to work session (codegen/test/deploy)
- Creates: Bidirectional link for traceability
- Updates: conversation_messages.linked_codegen_session field
- Enables: "Show me the chat that created this code" feature
- UI: Click message → see generated artifacts

Communication:

| Tool                  | Purpose                 | Protocol |
| --------------------- | ----------------------- | -------- |
| send_user_message     | Send message to user    | Builtin  |
| request_user_approval | Request user decision   | Builtin  |
| show_progress         | Update progress UI      | Builtin  |
| log_event             | Log event for debugging | Builtin  |

Error Handling:

| Tool                  | Purpose                    | Protocol |
| --------------------- | -------------------------- | -------- |
| retry_with_backoff    | Retry failed operation     | Builtin  |
| fallback_to_secondary | Use backup strategy        | Builtin  |
| report_error          | Report unrecoverable error | Builtin  |
| request_human_help    | Escalate to user           | Builtin  |

### 3.4 AGENTIC ORCHESTRATION

#### 3.4.1 LLM Orchestration

Multi-LLM strategy ensuring reliability through consultation and fallback patterns.

##### 3.4.1.1 Phase 1: LLM Consulting Mode (MVP)

Single agent with multi-LLM consultation for reliability.

Primary LLM: Claude Sonnet 4 for code generation, architectural reasoning, complex analysis, test generation, documentation writing.

Secondary LLM: GPT-4 Turbo for validation, fallback on primary failure, second opinion on critical decisions, cross-validation of outputs.

Consultation Strategy:

On Primary Success: Use Claude Sonnet 4 response immediately. Log success for metrics. Update confidence in primary for this task type.

On Primary Failure: Analyze failure reason (timeout, rate limit, poor quality output). Retry with modified prompt if transient failure. If retry fails, consult GPT-4 Turbo with same prompt. Use GPT-4 response if quality acceptable. If both fail, request human guidance.

On Critical Decisions: Consult both LLMs independently. Compare responses for agreement. If agree: proceed with confidence. If disagree: present both to user for decision. Learn from user's choice for future similar decisions.

Circuit Breaker Pattern: Track failure rate for each LLM per task type. If failure rate exceeds 30% over 10 attempts, temporarily switch to alternative. Reset circuit breaker after 1 hour cooldown. Alert user if both LLMs consistently failing.

Response Caching: Cache LLM responses in Redis with 1-hour TTL. Key based on prompt hash + model name. Cache hit rate target 40%+. Reduces cost and latency for repeated queries.

Cost Optimization: Track token usage per LLM per task type. Route simple tasks to cheaper models when quality sufficient. Use expensive models only for complex reasoning. Provide transparent cost breakdown to user.

Routing Logic:

if task.complexity == "simple" and task.type in ["code_completion", "simple_refactor"]:

    use_model = "claude-sonnet-3.5"  # Cheaper, faster

elif task.requires_deep_reasoning or task.type == "architecture_design":

    use_model = "claude-sonnet-4"    # Most capable

elif task.is_validation_only:

    use_model = "gpt-4-turbo"         # Alternative perspective

Failover Sequence:

1. Try primary LLM (Claude Sonnet 4)
2. If timeout or rate limit: retry once after exponential backoff
3. If still failed: try secondary LLM (GPT-4 Turbo)
4. If both failed: check if cached response exists (even if expired)
5. If no cache: request human guidance with error context

Performance Targets: LLM response time P95 under 10 seconds. Failover latency under 2 seconds. Cache hit rate above 40%. Overall success rate above 95%.

Multi-LLM Consultation After 2 Failures:

Trigger: After 2 consecutive failures with same issue from Primary LLM.

Consultation Flow:

1. Primary LLM generates consultation prompt (meta-prompting)
2. Consultant LLM (different model) provides second opinion: identifies blind spots or framing issues, suggests alternative approaches, points out what Primary LLM might be missing
3. Primary LLM uses consultation insight for third attempt
4. If third attempt succeeds: mark consultation as helpful, update strategy for similar problems
5. If still fails: escalate to user with full context

Transparency: Show user when consultation triggered. Display which models consulted. Show consultation insights. Build trust through transparency.

##### 3.4.1.2 Phase 2: Multi-LLM Team Orchestration

Multiple specialized agents each with configurable LLM, working in parallel on different modules or tasks.

Team Structure: Lead Agent (Coordinator) with configurable LLM, Specialist Agents (Coding, Architecture, Documentation, UX, Testing) each with own configurable LLM, Module-based agents (one agent per architectural module if desired), Separate agent instruction files per agent.

Lead Agent Responsibilities: Task decomposition and assignment, conflict resolution between agents, progress tracking and reporting, approval gate management, final decision making, context assembly across agents.

Lead Agent LLM Configuration: Selectable from OpenRouter allowlist (Claude Sonnet 4, GPT-4 Turbo, Claude Opus 4, etc.). Choice based on project needs (complex projects may need Claude Opus 4, standard projects use Claude Sonnet 4). User configurable via settings.

Specialist Agent LLM Configuration:

| Agent Type          | Recommended LLM                 | Rationale                        |
| ------------------- | ------------------------------- | -------------------------------- |
| Coding Agent        | Claude Sonnet 4, Qwen Coder 72B | Code generation expertise        |
| Architecture Agent  | Claude Opus 4, Claude Sonnet 4  | Deep reasoning for system design |
| Documentation Agent | GPT-4 Turbo, Claude Sonnet 4    | Natural language generation      |
| UX Agent            | Midjourney API, Claude Sonnet 4 | Visual asset generation          |
| Testing Agent       | Claude Sonnet 4, GPT-4 Turbo    | Test strategy and generation     |

All agents have configurable LLMs: Not just Lead Agent. Each Specialist Agent can use different LLM optimized for its domain. Configuration stored in agent instruction files. Allows cost optimization (use Qwen Coder 72B for simple code, Claude for complex logic).

Cost Optimization Strategy: Use smaller open-source models like Qwen Coder 72B as primary generators. Leverage Yantra's existing consultation architecture where expensive models only called upon failure. Example: Qwen generates code, if fails validation Claude consulted for fix.

Agent Coordination: Shared workspace via Git coordination branch. Agents communicate via @mentions in shared conversation. Workspace isolation restricts agents to specific directories. Adaptive context retrieval allows configurable access to recent messages and dependency graph queries.

Parallel Execution: Multiple agents work simultaneously on different features/modules. File locking system prevents conflicts. Lead Agent tracks progress across all agents. Human approval gates maintained at each development phase.

Agent Instruction Files: Each agent has persistent instruction file in .yantra/agents/ folder. Instructions define agent behavior, preferred patterns, specific constraints. Agent must check instructions before executing each state. Instructions override default state machine behavior. Provides customization layer on top of state machines.

Phase 2 Modifications to State Machines: State machines modified to support multi-agent workflow. Key modifications: agent assignment tracking, inter-agent communication, distributed file locking, aggregated progress reporting.

Performance Targets: Parallel speedup 2-4x for multi-module projects. Inter-agent communication latency under 500ms. File lock contention under 5%. Overall team success rate above 90%.

---

[Part 1B Complete - Sections 3.1.12 through 3.4.1] [Next: Part 2 will cover State Machines 3.4.2 onwards]

# Yantra: Complete Technical Specification - Part 2

State Machines and Advanced Features

---

### 3.4.2 STATE MACHINE ARCHITECTURE

Yantra's agentic capabilities implemented through five specialized state machines plus a sixth Documentation Governance machine (running in parallel), each with focused responsibility.

Architecture Benefits:

1. Separation of Concerns: Each machine has single, clear responsibility
2. Independent Testing: Test each machine without others
3. Flexible Execution: Re-run tests without regenerating code, re-deploy without re-running tests
4. MVP Focus: Build only CodeGen, Testing, Deployment for MVP, defer Maintenance to Post-MVP
5. Parallel Execution (Future): Multiple sessions simultaneously, background test runs, continuous monitoring
6. Clear State Tracking: Separate database tables per machine, query success rates per machine, full audit trail
7. Crash Recovery: Each machine saves state independently, resume from last checkpoint, no cross-machine corruption

The Six State Machines:

1. Code Generation State Machine (MVP)
2. Responsibility: Generate production-quality code that doesn't break dependencies
3. Key States: ArchitectureGeneration → ArchitectureReview → DependencyAssessment → TaskDecomposition → DependencySequencing → ConflictCheck → PlanGeneration → BlastRadiusAnalysis → PlanReview → EnvironmentSetup → ContextAssembly → CodeGeneration → CodeValidation → DependencyValidation → BrowserValidation → SecurityScanning → ConcurrencyValidation → FixingIssues → Complete/Failed
4. Entry: User's natural language intent
5. Exit: Generated code + confidence score
6. Total States: 20 MVP (states 1-20: ArchitectureGeneration through Complete) / 21 Post-MVP (adds FileLockRelease state after Complete or Failed outcome)
7. Clarification: FileLockAcquisition (#11) is numbered in MVP sequence but only becomes active Post-MVP; CodeValidation (#14) adapts to language type (compiled: full build, interpreted: static analysis); "Failed" is a session outcome marker, not a numbered state

2A. Test Intelligence State Machine (MVP)

1. Responsibility: Generate effective, high-quality tests with proper oracles
2. Key States: IntentSpecificationExtraction → TestOracleGeneration → InputSpaceAnalysis → TestDataGeneration → TestCaseGeneration → AssertionStrengthAnalysis → TestQualityVerification → TestSuiteOrganization → TestImpactAnalysis → TestUpdateGeneration → FixingIssues → Complete/Failed
3. Entry: Generated code from CodeGen machine
4. Exit: Test suite with proper oracles + test quality report
5. Auto-Trigger: Yes (runs after CodeGen succeeds, parallel with EnvironmentSetup)
6. Total States: 12 (11 sequential + FixingIssues)

2B. Test Execution State Machine (MVP)

1. Responsibility: Execute tests with comprehensive validation and debugging feedback
2. Key States: EnvironmentSetup → FlakeDetectionSetup → UnitTesting → IntegrationTesting → BrowserTesting → PropertyBasedTesting → ExecutionTraceAnalysis → FlakeDetectionAnalysis → CoverageAnalysis → SemanticCorrectnessVerification → ErrorClassification → FixingIssues → TestCodeCoEvolutionCheck → Complete/Failed
3. Entry: Test suite from Test Intelligence machine + generated code
4. Exit: Test results + coverage report + execution traces
5. Auto-Trigger: Yes (runs after Test Intelligence completes)
6. Total States: 14 sequential states
7. Deployment State Machine (MVP - Railway Focus)
8. Responsibility: Deploy validated code to Railway.app
9. Key States: PackageBuilding → ConfigGeneration → RailwayUpload → FixingIssues → HealthCheck → RollbackOnFailure → Complete/Failed
10. Entry: Code + passing tests
11. Exit: Live Railway URL + health status
12. Auto-Trigger: No (requires user approval)
13. Total States: 7 (6 sequential + FixingIssues handles pre-deployment errors, RollbackOnFailure handles post-deployment)
14. Maintenance State Machine (Post-MVP)
15. Responsibility: Monitor production, detect issues, auto-fix, deploy patches
16. Key States: LiveMonitoring → BrowserValidation → ErrorAnalysis → IssueDetection → AutoFixGeneration → FixValidation → CICDPipeline → VerificationCheck → LearningUpdate → Active/Incident
17. Entry: Deployed application in production
18. Exit: Incident resolved or escalated
19. Auto-Trigger: Yes (automatic based on error detection)
20. Philosophy: Self-healing production systems
21. Total States: 11 total (9 sequential states + 2 runtime states: Active and Incident)
22. Documentation Governance State Machine (NEW - 6th Machine, Post-MVP)
23. Responsibility: Maintain accurate, traced documentation throughout development lifecycle
24. Key States: DocumentationAnalysis → BlockIdentification → ContentGeneration → GraphLinking → ConflictDetection → UserClarification → ConflictResolution → Validation → Complete
25. Runs: In parallel with all other machines
26. Integration: Hooks in each existing machine (CodeGen, Testing, Deploy, Maintenance)
27. Philosophy: Documentation never drifts, full traceability maintained
28. Total States: 9 sequential states

#### 3.4.2.1 Code Generation State Machine (MVP)

Responsibility: Generate production-quality code that doesn't break dependencies

Entry Point: User's natural language intent

Exit Point: Generated code + confidence score

Trigger: User submits task

Success Criteria: Code passes dependency graph validation, security scan, no breaking changes, visibility of active work

State Count: 18 MVP / 19 Post-MVP

States:

Phase 1: Architecture & Design (PDC Phase 1)

1. ArchitectureGeneration: Generate or import project architecture
2. Inputs: User intent, existing codebase (if any), requirements documents
3. Process: LLM analyzes intent and generates architecture (components, connections, layers), OR imports existing architecture from files, OR analyzes existing code to generate architecture
4. Outputs: Architecture diagram, component definitions, connection types
5. Prevention: Architecture violations, boundary violations, circular dependencies, scaling bottlenecks, security vulnerabilities by design
6. Performance Target: <10s (LLM analysis + architecture generation)
7. ArchitectureReview: Wait for user approval of architecture (human-in-the-loop)
8. ⚠️ APPROVAL GATE: User must approve architecture changes
9. Display: Visual architecture diagram, component breakdown, connection details, architectural decision rationale
10. User Actions: Approve (proceed), Modify (request changes), Reject (go back)
11. Prevention: Proceeding with wrong architecture, misaligned expectations
12. Performance Target: N/A (waits for human)

Phase 2: Planning (PDC Phase 2 - Enhanced)

3. DependencyAssessment: Assess tools/packages/techstack needed, analyze compatibility
4. Inputs: Architecture, requirements, current environment state
5. Process: Identify required packages and versions, check CVE database for vulnerabilities, validate version compatibility (semver), detect duplicate functionality, verify license compatibility, dry-run validation in temp venv (Post-MVP)
6. Web Search Integration: Agent MUST use web search for latest package info (LLM knowledge is static/outdated), query official package registries (PyPI, npm, crates.io, Maven Central), fetch latest versions/changelogs/migration guides/known issues, Tools: MCP @modelcontextprotocol/server-brave-search or @modelcontextprotocol/server-fetch
7. Semantic-Enhanced Dependency Graph Resolution: Query semantic-enhanced dependency graph with HNSW semantic search to find similar past dependency conflicts and resolutions (vector embeddings in graph nodes), retrieve cached package documentation from SQLite (Tier 3, 7-day TTL), query compatibility matrices and known breaking changes stored in graph, semantic similarity search finds similar conflict patterns (<10ms with HNSW indexing), no external vector database needed (embeddings stored in petgraph nodes)
8. Outputs: List of required packages with exact versions, compatibility report, vulnerability scan results, installation plan
9. Prevention: Incompatible library versions, vulnerable dependencies, license conflicts, deprecated dependencies
10. Performance Target: <5s (parallel CVE lookups + web search)
11. TaskDecomposition: Break feature into concrete tasks
12. Inputs: User intent, architecture, dependency assessment, existing codebase
13. Process: GNN-based feature analysis, break down into atomic implementation tasks, each task maps to specific files/components, estimate complexity for each task
14. Outputs: Task list with descriptions, file mappings, complexity scores, dependencies between tasks
15. Prevention: Missing tasks, unclear scope, unbounded work
16. Performance Target: <500ms (GNN analysis)
17. DependencySequencing: Use dependency graph to determine task order
18. Inputs: Task list from TaskDecomposition, dependency graph
19. Process: Build task dependency graph (which tasks depend on which), topological sort for execution order, identify tasks that can run in parallel
20. Outputs: Ordered task sequence, parallel task clusters, critical path
21. Prevention: Wrong task order, dependency violations, parallel conflicts
22. Performance Target: <100ms (graph traversal)
23. ConflictCheck: Check which files will be modified and show visibility
24. MVP: Display active work visibility (which developer is working on which files), query active work indicators from UI
25. Post-MVP: File locking mechanism to prevent parallel edits, query file lock table for locked files
26. Inputs: Task list, file mappings, active work data/file locks
27. Outputs: Conflict report, locked files list, recommendations for conflict avoidance
28. Prevention: Parallel edit conflicts, work coordination issues
29. Performance Target: <50ms (active work lookup - MVP) / <50ms (lock table query - Post-MVP)
30. PlanGeneration: Create executable plan with task list, time estimates, critical path
31. Inputs: Ordered tasks, complexity scores, resource availability
32. Process: Calculate time estimates based on complexity, identify critical path, assign priority to tasks, create final execution plan
33. Outputs: Complete execution plan, estimated completion time, task priorities, resource allocation
34. Prevention: Unclear scope, missing estimates, unprioritized work
35. Performance Target: <200ms (estimation algorithm)
36. BlastRadiusAnalysis: Analyze impact of proposed changes BEFORE execution
37. Purpose: Provide dependency-aware impact preview BEFORE executing changes
38. Philosophy: "Show, don't surprise" - Users should know exactly what will be affected
39. Inputs: Execution plan, dependency graph, current architecture
40. Analysis Process: Identify direct files to modify, query dependency graph for downstream dependencies (recursive), identify critical files using heuristics, find all tests covering affected files, analyze API changes (if OpenAPI spec exists), check package dependency impacts (from Tech Stack GNN), calculate risk score, estimate rollback complexity
41. Outputs: BlastRadiusAnalysis struct with: files_to_modify (direct impact), critical_files, downstream_dependencies (indirect impact), affected_tests, api_changes, breaking_changes, package_upgrades, package_conflicts, risk_level (Low/Medium/High/Critical), estimated_time, rollback_complexity, affected_user_percentage, requires_approval flag, approval_reason
42. Prevention: Unintended consequences, breaking changes without awareness, deployment risks
43. Performance Target: <2s (parallel GNN queries for dependents/tests/packages)
44. Display Logic: Show detailed view for critical files, breaking changes, large test surface (>20), wide ripple (>10 dependencies), package changes, High/Critical risk. Skip for trivial changes: single file, no dependencies, few tests (<5), no API changes, no packages, Low risk.
45. PlanReview: Optional approval gate for complex features
46. Triggered for: Features with >5 tasks OR multi-file changes OR High/Critical risk from BlastRadiusAnalysis
47. Display: Task breakdown, sequencing, estimates, complexity, blast radius preview
48. ⚠️ APPROVAL GATE (Optional): User approves or requests modifications
49. Prevention: Misaligned expectations, scope disagreements, over-complexity
50. Performance Target: N/A (waits for human if triggered)

Phase 3: Execution (PDC Phase 3)

10. EnvironmentSetup: Prepare development environment
11. Inputs: Dependency assessment results, environment requirements
12. Process: Create/validate virtual environment (.venv for Python, node_modules for Node), install dependencies (pip install, npm install, cargo build), verify versions, set environment variables, validate setup
13. Mandatory .venv Isolation: NEVER pollute system Python, ALWAYS work in .venv, auto-create if not present, validate if exists, recreate if corrupted
14. Outputs: Ready environment, installed packages list, environment validation report
15. Prevention: Environment configuration errors, missing dependencies, version mismatches
16. Performance Target: 10-30s (parallel dependency installation)
17. FileLockAcquisition (Post-MVP only - Explicit State)
18. Purpose: Acquire file locks before any edits to prevent merge conflicts
19. Inputs: Files to be modified from plan
20. Process: Query dependency graph for dependent files, check if dependent files locked by other agents, if locked wait or work on different file, if free acquire lock and record in dependency graph with timestamp
21. Outputs: Lock acquisition confirmation, locked files list
22. Prevention: Merge conflicts made structurally impossible (not just less likely)
23. Performance Target: <10ms (database transaction)
24. ContextAssembly: Load relevant code context from dependency graph + conversation history
25. Inputs: Current file, dependency graph, user intent, conversation session
26. Process:
    - Conversation Context (NEW): Query conversation_messages for recent 10 messages, semantic search for relevant old messages (top-5), allocate 20K tokens for conversation context
    - Code Context: Query dependency graph for direct dependencies (imports, callers), retrieve semantic embeddings for similar patterns, assemble context for LLM prompt using hierarchical strategy (Level 0: conversation context [20K tokens], Level 1: direct dependencies, Level 2: transitive dependencies, Level 3: semantic similarity, Level 4: project context, Level 5: Yantra Codex patterns)
    - Apply token budget management: 20K conversation + 80K code + 20K reserve = 120K total
    - Compression: If over budget, keep recent conversation full-text, compress old messages

27. Integration:

- Link current message to codegen_session_id via conversation_messages.linked_codegen_session
- Enable traceability: chat → code generation → files created

1. Outputs: Assembled context (<120k tokens), relevant files list, pattern suggestions, conversation context summary
2. Prevention: Reinventing functionality, inconsistent patterns, missing context, forgetting user intent from earlier in conversation
3. Performance Target: <200ms (with parallel GNN queries + conversation retrieval)
4. Implementation: See Section 3.1.5.7 (Conversation Context Assembly) and Section 3.1.13.3 (Adaptive Context Retrieval)
5. CodeGeneration: Generate code using Yantra Codex + Multi-LLM consultation
6. Phase A - Initial Generation: Yantra Codex (GNN) generates initial code (15ms), calculate confidence score (0.0-1.0), if confidence >= 0.8 use Yantra code directly, if confidence < 0.8 send to Primary LLM for review and enhancement, Primary LLM (Claude Sonnet 4, user's choice) reviews edge cases and adds error handling, merge Yantra + LLM suggestions, Yantra learns from LLM improvements
7. Phase B - Multi-LLM Consultation (Optional): If Yantra confidence score < 0.6 (very low confidence), trigger immediate consultation before proceeding, Consultation Flow: Primary LLM generates consultation prompt (meta-prompting), Consultant LLM (different model) provides second opinion (identifies blind spots, suggests alternative approaches), Primary LLM uses consultation insight to improve generation, use consultation to improve initial code before writing to disk
8. Transparency: Show user when consultation triggered, display which models consulted, show consultation insights, build trust through transparency
9. Phase C - File Write Operations: Determine file paths (user-specified > architecture-defined > dependency graph proximity > project conventions), validate no duplicate files (CreationValidation Service), create parent directories if needed (recursive mkdir), write generated code to filesystem (fs::write for new files, file_edit tool for modifications), handle write errors (disk full, permissions, path conflicts), update dependency graph with new file nodes and edges, invalidate context cache (Tier 4) for affected files
10. Phase D - Conversation Linkage (NEW): Create codegen_session record with session_id, user_intent, timestamps, Update conversation_messages.linked_codegen_session with codegen_session_id, Link agent response message to same codegen_session_id, Enable full traceability: user message → agent message → codegen_session → generated files
11. Multi-File Strategy: Sequential write (MVP): write files in dependency order (dependencies first, dependents second), atomic transaction (all succeed or all rollback), update graph after each successful write, Parallel write (Phase 2A): Team of Agents write independent files simultaneously, file locking prevents conflicts, dependency-ordered for related files
12. Error Recovery: File write failures trigger rollback of all writes in current batch, preserve previous file versions (git staging area), log failure details (path, error type, permissions, disk space), transition to FixingIssues state with error context, max 3 retry attempts before escalating to user
13. Outputs: Generated code (written to disk), confidence score, LLM consultation log (if triggered), list of created/modified file paths, file write operation log, updated dependency graph nodes/edges, cache invalidation list, codegen_session_id (NEW), conversation message links (NEW)
14. Prevention: Low-quality initial generation (consultation improves), file write failures, duplicate files, architectural violations in generation approach, loss of conversation traceability (NEW)
15. Performance Target: 2-5s (LLM dependent, parallel for multiple files)
16. Next State: CodeValidation (State 14) validates the generated code
17. Implementation Details: See Part 3, Section 3.4.3.3 (File Operations Service) for complete file write flow, error handling strategies, and transaction management. See Section 3.1.13.5 (Work Session Linking) for conversation linkage implementation
18. CodeValidation: Validate generated code using language-appropriate tools
19. Inputs: Generated code files (written to disk), project language/build configuration, language detection metadata
20. Language Detection: Detect project type from files and configuration (Rust: Cargo.toml, Go: go.mod, C/C++: CMakeLists.txt/Makefile, Java: pom.xml/build.gradle, TypeScript: tsconfig.json, Python: requirements.txt/pyproject.toml, JavaScript: package.json, Ruby: Gemfile), determine validation strategy based on language characteristics
21. Validation Strategy by Language Type:
    - Compiled Languages (Rust, Go, C/C++, Java, TypeScript): Run compiler/type checker (Rust: cargo check, Go: go build, C/C++: cmake/make, Java: javac/mvn compile, TypeScript: tsc --noEmit), capture compilation errors and warnings, parse output for error types (syntax, type, import, linker), map errors to source locations (file, line, column), categorize by severity (critical: fails, warnings: compiles but problematic)
    - Interpreted Languages (Python, JavaScript, Ruby): Run static analysis tools (Python: mypy for types + pylint for errors, JavaScript: eslint + flow/TypeScript check if configured, Ruby: rubocop + sorbet if configured), import validation (Python: try importing all modules, JS: check require/import statements resolve, Ruby: check require statements), syntax validation (Python: py_compile, JS: acorn/esprima parser, Ruby: ruby -c), runtime checks where applicable (Python: pytest --collect-only to validate test structure)

22. Common Validations (All Languages): LSP-based type checking (if LSP server available), linter execution (language-specific), dependency resolution check (all imports/requires exist), dead code detection, unused variable/import detection
23. Error Analysis: Extract error patterns from compiler/linter/analyzer output, query Yantra Codex for known fixes to similar errors, provide rich error context for FixingIssues state (exact error message, source location, error category, suggested fixes)
24. Outputs: Validation success/failure status, detailed error report with source locations (file, line, column), warnings list (non-blocking issues), error category (syntax, type, import, runtime, style), suggested fixes from Yantra Codex, severity level (critical, high, medium, low)
25. Performance Optimization: Incremental validation (only changed files + dependents), parallel validation (multiple files simultaneously), caching (validation results for unchanged files), early termination (stop on first critical error for fast feedback)
26. Prevention: Runtime errors in production (both compiled and interpreted), type errors (all languages with type systems), missing dependency errors, import/require errors, syntax errors missed by tree-sitter, undefined behavior, style violations that cause bugs
27. Performance Target: Compiled: <15s incremental / <60s full build, Interpreted: <5s (static analysis + import checks), Cached/unchanged: <100ms
28. Integration: On validation failure → transition to FixingIssues state with language-specific error context, on success → continue to DependencyValidation, on warnings only → continue but log warnings
29. DependencyValidation: Validate code against dependency graph
30. Inputs: Generated code, dependency graph
31. Process: Parse generated code with tree-sitter, extract new imports and dependencies, check for breaking changes (modified function signatures, removed functions, changed return types), query dependency graph for affected callers, validate no circular dependencies introduced, verify architectural boundaries not violated
32. Outputs: Validation report, breaking changes list, affected files
33. Prevention: Breaking changes, circular dependencies, architectural violations
34. Performance Target: <10ms (with parallel dependency path validation)
35. BrowserValidation: Validate UI components in browser
36. Inputs: Generated code (for web applications), localhost URL
37. Process: Launch Chrome via CDP, navigate to localhost, execute test scenarios (click flows, form fills, element verification), capture console errors, monitor network requests, take screenshots at key steps, detect issues (500 errors, JavaScript exceptions, missing elements, broken images)
38. Outputs: Browser validation report, screenshots, console logs, network HAR file
39. Prevention: UI bugs, runtime errors, broken user flows
40. Performance Target: 5-10s (single component) / 10-15s (parallel multi-component)
41. SecurityScanning: Scan for security vulnerabilities
42. Inputs: Generated code
43. Process: Semgrep static analysis (OWASP rulesets for SQL injection, XSS, CSRF, command injection, path traversal, insecure deserialization, XXE), secrets detection (API keys, passwords, tokens, credentials using TruffleHog patterns), dependency vulnerability scanning (CVE database check), license compliance checking
44. Outputs: Security scan report categorized by severity (Critical, High, Medium, Low)
45. Auto-Fix Patterns: SQL injection → use parameterized queries, XSS → use proper escaping, Secrets → move to environment variables, Path traversal → validate paths, Command injection → use safe subprocess calls
46. Blocking: Critical and High severity issues must be fixed, Medium issues generate warnings (non-blocking), Low issues logged (informational)
47. Prevention: Security vulnerabilities, hardcoded secrets, vulnerable dependencies
48. Performance Target: <10s (with 4 parallel workers for file scanning)
49. ConcurrencyValidation: Race condition and deadlock detection (NEW - Parallel Safety Check)
50. Purpose: Validate that parallel processing didn't introduce concurrency bugs
51. When: After any code generation that uses async/await, threads, or parallel execution
52. Built-in Checks (Agentic Flow - NOT just LLM reasoning): Static Analysis (Rust Clippy pedantic + thread safety lints, Python threading analyzer, JavaScript event loop analyzer), Pattern Detection (shared mutable state access, missing locks/semaphores, race-prone check-then-act patterns), GNN Analysis (identify data flow paths that could race, detect concurrent writes to same resource), Test Generation (automatically generate stress tests for concurrent code paths)
53. Auto-Fix Strategy: Level 1 Pattern-Based (No LLM - Fast): Detected shared mutable state without lock → Fix: wrap in Mutex (Rust) or threading.Lock (Python), Detected check-then-act pattern → Fix: use atomic operations or lock entire section, Level 2 LLM-Assisted Fix (With Context): Provide LLM with detected race condition explanation + code snippet + GNN data flow paths + language-specific synchronization primitives, LLM generates fixed code with proper locks/channels/atomics, Agent validates fix with stress tests, Level 3 Sequential Fallback (Safe Default): If parallel execution cannot be made safe rewrite as sequential, add TODO comment explaining why, log performance regression, Guarantee: Code always works correctly even if slower
54. Skip Conditions: No async/await keywords detected, No threading/multiprocessing imports, No parallel primitives (tokio::spawn), Pure sequential code → Skip validation (0s overhead)
55. Prevention: Race conditions, deadlocks, data races from parallel execution
56. Performance Target: <5s (static analysis + GNN race detection + test generation)
57. FixingIssues: Auto-retry with fixes if validation fails
58. Inputs: Validation failures from any previous state (code_validation, dependency, browser, security, concurrency, file_write)
59. Process: Analyze failure type (code_validation, dependency, browser, security, concurrency, file_write), retrieve relevant fixes from Yantra Codex known issues database, apply automatic fix if pattern recognized, Code Validation Error Handling: if code validation failed → parse error output (compiler/linter/analyzer), determine error category (syntax, type, import, runtime), query Codex for similar errors and fixes in this language, apply language-specific fixes (Rust: add type annotations, Python: add imports, JS: fix async/await), retry validation; File Write Error Handling: if error is disk_full → fail immediately (notify user), if error is permission_denied → try alternative paths, if error is path_invalid → sanitize and retry, if error is parent_not_found → create parent dirs and retry, if auto-fix fails request LLM to generate fix with error context, validate fix, repeat up to 3 attempts, after 3 attempts → transition to Failed state with detailed error report
60. Outputs: Fixed code or escalation to user
61. Prevention: Repeated failures, cascading issues
62. Performance Target: <10s per retry
63. Complete: Code ready for testing
64. Post-MVP: Explicit FileLockRelease state after completion
65. Outputs: Generated code, confidence score, validation reports
66. Triggers: Testing Intelligence State Machine (auto-trigger)

Failed: Human intervention required

1. Post-MVP: Explicit FileLockRelease state on failure
2. Outputs: Failure analysis, error logs, recommendations
3. User Actions: Review errors, provide guidance, retry manually

FileLockRelease (Post-MVP Only - State 21)

1. Purpose: Explicit state to release file locks when session completes or fails
2. Trigger: Automatically after reaching Complete state OR after Failed outcome (unrecoverable errors)
3. Inputs: Current agent ID, session ID, list of active file locks held by this agent/session
4. Process: Query sled (Tier 2) for all active locks held by current agent/session, release each file lock (delete from sled key-value store), notify other agents waiting for these files (pub/sub or polling), clear lock metadata from agent's state, log lock release event with duration metrics
5. Outputs: List of released file paths, lock duration for each file, list of agents notified (unblocked), lock release confirmation
6. Prevention: Stuck locks blocking other agents indefinitely, resource leaks in multi-agent environment, agent deadlocks from unreleased locks on crash, file access bottlenecks from held locks
7. Error Handling: If lock release fails log warning but continue (don't block session completion), if sled unavailable: mark locks as "stale" and rely on timeout mechanism, locks auto-expire after 1 hour regardless (safety mechanism)
8. Performance Target: <15ms total (sled operations are fast: ~1ms per lock, O(n) for n locks)
9. Integration: Runs automatically at session end, no user interaction required, transparent to developer

Performance Targets (Total Cycle):

1. MVP: <43s (Architecture: ~12s, Planning: ~6s, Execution: ~25s)
2. Post-MVP: <45s (adds explicit lock acquisition/release: ~15ms)
3. Parallel optimization: ~30-40% faster with concurrent execution

Prevention Guarantees:

1. ✅ Architecture always exists before code generation (PDC 1.1)
2. ✅ Tech stack is consistent and secure (PDC 1.2)
3. ✅ Plans are explicit and validated (PDC 2.1)
4. ✅ Work visibility (MVP) or conflict prevention (Post-MVP) (PDC 3.3)
5. ✅ Dependency-aware execution in correct order (PDC 2.1)
6. ✅ Concurrency safety validated before testing (PDC 3.4)
7. ✅ Security built-in at every phase (PDC 3.4)

#### 3.4.2.1A Blast Radius Analysis & Preview (P0 Feature - MVP)

Data Structures:

pub struct BlastRadiusAnalysis {

    // Direct Impact

    pub files_to_modify: Vec`<FileImpact>`,

    pub critical_files: Vec`<PathBuf>`,

    // Indirect Impact (from GNN downstream dependencies)

    pub downstream_dependencies: Vec`<DependencyImpact>`,

    pub affected_tests: Vec`<TestImpact>`,

    // External Impact

    pub api_changes: Vec`<ApiChange>`,

    pub breaking_changes: Vec`<BreakingChange>`,

    // Package Impact (from Tech Stack GNN)

    pub package_upgrades: Vec`<PackageUpgrade>`,

    pub package_conflicts: Vec`<PackageConflict>`,

    // Risk Metrics

    pub risk_level: RiskLevel, // Low, Medium, High, Critical

    pub estimated_time: Duration,

    pub rollback_complexity: RollbackComplexity,

    pub affected_user_percentage: f32,

    // Decision Factors

    pub requires_approval: bool,

    pub approval_reason: String,

}

pub struct FileImpact {

    pub path: PathBuf,

    pub is_critical: bool,

    pub change_type: ChangeType, // Create, Modify, Delete

    pub dependent_count: usize,

}

pub struct DependencyImpact {

    pub file: PathBuf,

    pub distance: usize, // Degrees of separation (1=direct, 2=indirect)

    pub impact_reason: String,

}

pub struct TestImpact {

    pub test_file: PathBuf,

    pub test_name: String,

    pub coverage_type: TestCoverageType, // Unit, Integration, E2E

    pub needs_update: bool,

}

pub struct ApiChange {

    pub endpoint: String,

    pub method: HttpMethod,

    pub change_type: ApiChangeType, // Added, Modified, Deprecated, Breaking

    pub breaking: bool,

}

pub struct PackageUpgrade {

    pub package: String,

    pub from_version: String,

    pub to_version: String,

    pub breaking_changes: Vec`<String>`,

}

Analysis Algorithm:

impl BlastRadiusAnalyzer {

    pub async fn analyze(&self, plan: &ExecutionPlan) -> Result<BlastRadiusAnalysis, String> {

    // 1. Identify direct files to be modified

    let files_to_modify = self.extract_files_from_plan(plan);

    // 2. Query GNN for downstream dependencies (recursive)

    let downstream = self.gnn.get_dependents_recursive(&files_to_modify).await?;

    // 3. Identify critical files using heuristics

    let critical_files = self.identify_critical_files(&files_to_modify);

    // 4. Find all tests covering affected files

    let affected_tests = self.gnn.find_tests_for_files(&files_to_modify).await?;

    // 5. Analyze API changes (if OpenAPI spec exists)

    let api_changes = if let Some(spec) = &self.openapi_spec {

    self.analyze_api_changes(spec, &files_to_modify).await?

    } else {

    vec![]

    };

    // 6. Check package dependency impacts (from Tech Stack GNN)

    let package_impacts = self.tech_stack_gnn

    .analyze_package_changes(&plan.package_changes).await?;

    // 7. Calculate risk score

    let risk_level = self.calculate_risk_level(

    &critical_files,

    &downstream,

    &api_changes,

    &package_impacts

    );

    // 8. Estimate rollback complexity

    let rollback_complexity = self.estimate_rollback_complexity(

    &plan,

    &api_changes,

    &package_impacts

    );

    Ok(BlastRadiusAnalysis {

    files_to_modify: files_to_modify.into_iter()

    .map(|f| FileImpact {

    path: f.clone(),

    is_critical: critical_files.contains(&f),

    change_type: ChangeType::Modify,

    dependent_count: self.gnn.count_dependents(&f).unwrap_or(0),

    })

    .collect(),

    critical_files,

    downstream_dependencies: downstream,

    affected_tests,

    api_changes,

    breaking_changes: api_changes.iter()

    .filter(|c| c.breaking)

    .map(|c| BreakingChange {

    description: format!("API {} {} changed", c.method, c.endpoint),

    impact: format!("Affects {}% of users", affected_user_percentage),

    })

    .collect(),

    package_upgrades: package_impacts.upgrades,

    package_conflicts: package_impacts.conflicts,

    risk_level,

    estimated_time: self.estimate_time(&plan),

    rollback_complexity,

    affected_user_percentage: self.estimate_user_impact(&api_changes),

    requires_approval: matches!(risk_level, RiskLevel::High | RiskLevel::Critical),

    approval_reason: self.generate_approval_reason(&risk_level, &critical_files, &api_changes),

    })

    }

}

pub fn should_show_detailed_blast_radius(analysis: &BlastRadiusAnalysis) -> bool {

    !analysis.critical_files.is_empty() ||

    !analysis.breaking_changes.is_empty() ||

    analysis.affected_tests.len() > 20 ||

    analysis.downstream_dependencies.len() > 10 ||

    !analysis.package_upgrades.is_empty() ||

    matches!(analysis.risk_level, RiskLevel::High | RiskLevel::Critical)

}

pub fn should_skip_blast_radius(analysis: &BlastRadiusAnalysis) -> bool {

    analysis.files_to_modify.len() == 1 &&

    analysis.downstream_dependencies.is_empty() &&

    analysis.affected_tests.len() < 5 &&

    analysis.api_changes.is_empty() &&

    analysis.package_upgrades.is_empty() &&

    matches!(analysis.risk_level, RiskLevel::Low)

}

UI Display Format:

Compact View (Small Changes):

✅ Low Risk Change

├── 2 files modified

├── 5 tests affected

└── No breaking changes

[Continue] [Details]

Detailed View (Large/Critical Changes):

📊 Blast Radius Preview

Direct Impact (Files to be Modified):

├── src/calculator.py ⚠️ CRITICAL (47 dependents)

├── src/utils.py (3 dependents)

└── tests/test_calculator.py

Indirect Impact (Downstream Dependencies):

├── 12 files import modified code:

│ ├── src/api/endpoints.py (Level 1 - direct import)

│ ├── src/services/math_service.py (Level 1)

│ ├── src/reports/generator.py (Level 2 - indirect)

│ └── ... (9 more - click to expand)

External Impact:

├── 2 API endpoints will change:

│ ├── POST /api/calculate ⚠️ BREAKING CHANGE

│ │ └── Response schema modified (added "precision" field)

│ └── GET /api/health ✅ Non-breaking (added "version")

├── 47 tests need updating:

│ ├── 23 unit tests (calculator, utils)

│ ├── 18 integration tests (API, services)

│ └── 6 E2E tests (full workflows)

Package Dependencies:

├── numpy: 1.24.0 → 1.26.0 ⚠️ UPGRADE REQUIRED

│ └── Breaking changes: numpy.array default behavior changed

└── pandas: 2.1.0 (no change)

Risk Assessment:

├── Risk Level: HIGH ⚠️

├── Breaking Changes: 1 API endpoint

├── Affected Users: ~45% of API calls (estimated)

├── Rollback Complexity: MEDIUM (DB migration needed)

└── Estimated Time: 45-60 minutes

⚠️ High-risk change detected. Manual approval required.

Reasons:

- Touches critical file: src/calculator.py (47 dependents)
- Breaking API change: POST /api/calculate
- Package upgrade with breaking changes: numpy 1.24→1.26

[Approve & Execute] [Modify Plan] [View Detailed Report] [Cancel]

Performance Targets:

1. GNN Queries: <500ms (parallel queries for dependents, tests, packages)
2. Critical File Detection: <100ms (in-memory checks + config lookup)
3. API Change Analysis: <300ms (if OpenAPI spec exists, otherwise skip)
4. Package Impact Analysis: <1s (query Tech Stack GNN + check changelogs)
5. Total Analysis Time: <2s (all operations)

#### 3.4.2.2 Testing Framework (MVP) - Multi-State Machine Architecture

Philosophy: Robust testing requires two specialized state machines working in concert:

1. Test Intelligence State Machine: Generates high-quality, effective tests with proper oracles
2. Test Execution State Machine: Executes tests with comprehensive validation and feedback

Overall Responsibility: Ensure generated code works correctly through intelligent test generation and rigorous execution

Parallel Execution (Both Machines Run Concurrently):

CodeGen Complete

    ↓

├─────────────────┐

↓ ↓

Test Intelligence Machine Environment Setup (Execution Machine)

(Generates tests) (Prepares runtime)

↓ ↓

(~37s to complete) (~15s to complete)

↓ ↓

└────────────────┘

    ↓

Test Execution Machine

(Executes generated tests)

    ↓

(~2 minutes)

    ↓

Results + Learning Data

##### 3.4.2.2A Test Intelligence State Machine (MVP)

Responsibility: Generate effective, high-quality tests that correctly verify behavior

Entry Point: Generated code + user intent from CodeGen machine

Exit Point: Comprehensive test suite with oracles + test quality report

Trigger: Automatically after CodeGen succeeds (parallel with environment setup)

Success Criteria: Mutation score >80%, Assertion strength score >5 average, Coverage potential >90%, All tests have proper oracles

State Count: 10 states (MVP)

States:

Phase 1: Test Oracle & Specification Extraction (MVP - Critical)

1. IntentSpecificationExtraction: Extract testable specifications from user's natural language intent
2. Purpose: Solve the Test Oracle Problem - determine what "correct behavior" actually means
3. Inputs: User's original intent, generated code, function signatures, docstrings, conversation context (NEW)
4. Conversation Integration (NEW): Retrieve conversation messages linked to codegen_session, Use conversation context to understand user's intent better, Search for similar past conversations about testing requirements, Helps disambiguate unclear requirements from chat history
5. Techniques: Natural language processing of user intent → formal specifications, Contract extraction (preconditions, postconditions, invariants), Expected behavior synthesis from requirements, Example-based specification (user provides examples, system generalizes)
6. Outputs: Structured behavioral specifications, expected outcomes, invariants
7. Parallel Processing: Extract specifications for multiple functions simultaneously
8. Performance Target: <2s (LLM + NLP analysis + conversation retrieval)
9. TestOracleGeneration: Generate verification strategies
10. Purpose: Create proper test oracles (not just "runs without error")
11. Oracle Types: Specification-based (verify against extracted specs), Differential (compare against reference implementation), Metamorphic (test relationships between inputs/outputs), Contract-based (verify preconditions/postconditions)
12. Inputs: Behavioral specifications from previous state
13. Outputs: Test oracle specifications, verification strategies per function
14. Parallel Processing: Generate oracles for multiple functions concurrently
15. Performance Target: <3s (LLM oracle generation)

Phase 2: Test Case Generation

3. InputSpaceAnalysis: Analyze input domains
4. Purpose: Identify representative input values
5. Techniques: Boundary value analysis, Equivalence partitioning, Edge case identification
6. Inputs: Function signatures, parameter types, domain constraints
7. Outputs: Input space partitions, boundary values, edge cases
8. Performance Target: <1s (static analysis)
9. TestDataGeneration: Generate test data
10. Strategies: Valid inputs (happy path), Invalid inputs (error handling), Boundary values (edge cases), Random values (property-based testing)
11. Inputs: Input space analysis, behavioral specifications
12. Outputs: Test data sets per function
13. Parallel Processing: Generate test data for multiple test strategies simultaneously
14. Performance Target: <5s (parallel strategies)
15. TestCaseGeneration: Generate actual test code
16. Inputs: Test data, test oracles, behavioral specifications
17. Process: Generate test functions with proper assertions, include setup/teardown code, organize by test type (unit, integration, property-based), add descriptive test names and docstrings, write test files via File Operations Service (Part 3, Section 3.4.3.3)
18. Conversation Linkage (NEW): Create test_session record with session_id and link to parent codegen_session, Update conversation_messages.linked_test_session with test_session_id, Enable traceability: user message → codegen_session → test_session → test files
19. Outputs: Complete test code files (written to disk), test_session_id (NEW), conversation message links (NEW)
20. Parallel Processing: Generate test code for multiple modules using parallel LLM calls
21. Performance Target: <5s (parallel LLM calls)

Phase 3: Test Quality Verification

6. AssertionStrengthAnalysis: Verify test assertions are strong
7. Purpose: Detect weak assertions like "assert result is not None"
8. Techniques: Assertion pattern analysis, Mutation testing simulation, Strength scoring (1-10 scale)
9. Outputs: Strength scores per assertion, weak assertions flagged
10. Performance Target: <2s
11. TestQualityVerification: Mutation testing for test effectiveness
12. Purpose: Verify tests actually catch bugs
13. Process: Generate code mutants (change operators, modify conditionals, alter return values), run tests against mutants, calculate mutation score (% of mutants killed)
14. Target: Mutation score >80%
15. Outputs: Mutation score, surviving mutants, test effectiveness report
16. Performance Target: <15s (parallel mutation testing)

Phase 4: Organization & Analysis

8. TestSuiteOrganization: Organize tests logically
9. Process: Group tests by module/feature, separate unit/integration/E2E tests, create test fixtures and helpers, generate test configuration files
10. Outputs: Organized test suite structure
11. Performance Target: <1s
12. TestImpactAnalysis: Determine which tests affected by code changes (MVP - Critical)
13. Purpose: Enable test-code co-evolution and efficient re-testing
14. Technique: Use dependency graph to map code changes → affected tests, query GNN: "Which tests call this modified function?", analyze transitive dependencies
15. Outputs: List of affected tests, impact severity (direct/indirect), recommended actions (update/rewrite/keep)
16. Parallel Processing: Analyze impact for multiple changed files simultaneously
17. Performance Target: <500ms (GNN queries with parallel lookups)
18. TestUpdateGeneration: Generate test updates when code changes (MVP - Test Co-Evolution)
19. Purpose: Keep tests synchronized with code changes
20. Triggers: Function signature changed → Update test calls, Return type changed → Update assertions, Behavior modified → Regenerate test expectations
21. Strategies: Preserve intent (keep original test purpose, update implementation), Incremental update (modify only affected parts), Staleness detection (flag tests no longer matching intent)
22. Parallel Processing: Update multiple affected tests concurrently
23. Performance Target: <3s (LLM updates with parallel calls)
24. FixingIssues: Auto-retry with fixes if test generation fails
25. Inputs: Error context from failed state (IntentSpecificationExtraction, TestOracleGeneration, TestCaseGeneration, or other generation steps), error details (syntax errors in generated tests, LLM generation failures, invalid test specifications, oracle generation errors), failure count (1-3 attempts), partial results if any
26. Process: Analyze failure type and root cause using Yantra Codex patterns, apply automatic fix based on error category: If LLM generation failed → retry with simplified prompt, use Consultant LLM (different model) for second opinion, reduce complexity of test case; If syntax errors in generated tests → parse with tree-sitter and auto-fix common patterns (missing imports, incorrect assertions, type mismatches); If invalid test oracle → regenerate with clearer specifications from intent, use example-based oracle generation; If test data generation failed → simplify input space, use boundary values only, reduce combinatorial explosion, validate fix by re-running failed generation step, increment retry counter (max 3 attempts), log each attempt with failure analysis
27. Outputs: Fixed test suite (if successful within 3 attempts), partial test suite (tests generated before failure), detailed error analysis report (if all attempts fail), recommendations for user (simplify requirements, provide examples, manual test writing)
28. Escalation: After 3 failed attempts mark session outcome as Failed and present comprehensive error report to user including: which generation step failed, why automatic fixes didn't work, partial tests generated (if any), specific recommendations for manual intervention, example test patterns to follow
29. Prevention: Test generation failures without recovery attempt, incomplete test coverage due to unhandled generation errors, cascading test quality issues from early failures, user frustration from immediate failures
30. Performance Target: <10s per retry attempt (LLM consultation + re-generation + validation)
31. Complete: High-quality test suite ready for execution

Failed: Unable to generate effective tests (escalate to human)

Performance Targets (Total Cycle): <37s

##### 3.4.2.2B Test Execution State Machine (MVP)

Responsibility: Execute tests with comprehensive validation, debugging feedback, and resilience

Entry Point: Test suite from Test Intelligence machine + generated code

Exit Point: Test results + coverage report + execution traces + learning data

Trigger: Automatically after Test Intelligence completes

Success Criteria: 100% stable tests pass (flaky tests quarantined), Coverage >80%, Semantic correctness verified, No critical errors

State Count: 13 states (MVP)

States:

Phase 1: Environment & Execution Preparation

1. EnvironmentSetup: Create virtual environment, install dependencies
2. Parallel Processing: Install multiple independent test dependencies simultaneously
3. Performance Target: 10-20s (with parallel pip/npm install)
4. FlakeDetectionSetup: Configure flaky test detection infrastructure (MVP - Critical)
5. Purpose: Prevent non-deterministic tests from blocking autonomous loop
6. Configuration: Set retry count (default: 3 runs per test), configure timeout thresholds, setup flake detection storage (SQLite)
7. Performance Target: <500ms (configuration)

Phase 2: Test Execution with Instrumentation

3. UnitTesting: Run pytest/jest for function-level tests with execution tracing
4. Execution: Run tests with coverage instrumentation, capture execution traces for failures (variable states, call stacks), run each test N times for flake detection
5. Parallel Processing: Run test files in parallel (pytest -n auto, jest --maxWorkers), execute independent test suites simultaneously, Performance: N test files × 5s = 5N sequential → ~N/4 parallel (4 workers)
6. Instrumentation: Python sys.settrace(), JavaScript source maps
7. Performance Target: <30s (parallel execution with 4+ workers)
8. IntegrationTesting: Test API integrations and data flows with contract verification
9. Execution: Test API endpoints with contract validation (OpenAPI/Pact), database integration tests, external service mocking
10. Parallel Processing: Test multiple API endpoints simultaneously, parallel database connection tests, concurrent integration scenario execution
11. Performance Target: 20-40s (with parallel API/DB testing)
12. BrowserTesting: Playwright E2E tests for user workflows
13. Execution: Launch browser via CDP, execute user scenarios (login, checkout, forms), verify UI elements and interactions, capture screenshots and console logs, monitor network requests
14. Parallel Processing: Run tests in 2-3 browsers concurrently (Chrome, Firefox, Safari), execute independent user flows simultaneously
15. Performance Target: 15-30s (parallel, 2-3 browsers)
16. PropertyBasedTesting: Run hypothesis/fast-check for property tests
17. Execution: Generate random inputs according to properties, verify invariants hold across all inputs, capture counterexamples if property violated
18. Parallel Processing: Test multiple properties concurrently
19. Performance Target: <20s (parallel property testing)

Phase 3: Analysis & Debugging

7. ExecutionTraceAnalysis: Analyze execution traces for failures
8. Purpose: Provide actionable debugging information
9. Analysis: Variable states at failure point, call stack leading to failure, input values that caused failure, comparison with successful runs
10. Outputs: Detailed failure reports with traces
11. Performance Target: <2s
12. FlakeDetectionAnalysis: Identify and quarantine flaky tests (MVP - Critical)
13. Purpose: Prevent flaky tests from blocking deployment
14. Process: Analyze test runs across N attempts (default: 3), identify tests with inconsistent results (pass sometimes, fail sometimes), calculate flakiness score (0.0-1.0), quarantine tests with flakiness score >0.3
15. Quarantine Action: Mark test as flaky in database, exclude from pass/fail criteria (don't block deployment), report flaky tests to user for investigation, allow manual override to include/exclude
16. Outputs: Flaky test list, flakiness scores, quarantine decisions
17. Performance Target: <5s (3 runs, parallel execution)
18. CoverageAnalysis: Check code coverage metrics
19. Metrics: Line coverage, branch coverage, function coverage
20. Target: >80% coverage
21. Outputs: Coverage report, uncovered lines/branches
22. Performance Target: <3s
23. SemanticCorrectnessVerification: Verify tests match intent (MVP - Critical)
24. Purpose: Detect tests that pass but don't actually verify correct behavior
25. Technique: Compare test assertions with extracted specifications, verify all specified behaviors have tests, detect tautological assertions (always true), check for meaningless assertions
26. Outputs: Semantic correctness score, meaningless test warnings
27. Performance Target: <5s

Phase 4: Error Handling & Recovery

11. ErrorClassification: Classify test failures
12. Categories: Code bug (logic error in implementation), Test bug (incorrect test), Environmental issue (setup problem), Flaky test (non-deterministic), Timeout (performance issue)
13. Outputs: Classified failures, recommended actions
14. Performance Target: <3s
15. FixingIssues: Auto-retry with fixes if tests fail
16. Process: Analyze failure type, retrieve known issue patterns from Yantra Codex, apply automatic fix if pattern recognized, if auto-fix fails request LLM to generate fix, validate fix and re-run tests
17. Performance Target: <10s per retry

Phase 5: Co-Evolution Check

13. TestCodeCoEvolutionCheck: Verify tests remain aligned with code (MVP)
14. Purpose: Ensure tests haven't become stale after code changes
15. Checks: Do tests still compile/run?, Do tests still match intent specifications?, Are assertions still appropriate for new implementation?
16. Action: If stale, trigger TestUpdateGeneration in Test Intelligence machine
17. Performance Target: <1s (GNN queries + static analysis)

Phase 6: Results & Reporting

14. Complete: All tests pass with adequate coverage and semantic correctness
15. Report: Test results, coverage metrics, mutation score, semantic correctness score, flaky tests quarantined

Failed: Tests failed after maximum retries

1. Report: Detailed failure analysis with execution traces, root causes, suggested fixes

Performance Targets (Total Cycle): <2 minutes (with parallel optimizations)

Testing Framework Integration & Workflow:

Total Testing Framework Cycle: ~2.5 minutes (with parallel optimization)

Task Tracking for Parallel Operations (MVP - Critical): When running multiple tasks in parallel, the system tracks: Task Registry (SQLite table tracking all active tasks), Context Preservation (each task maintains isolated context), Dependency Management (tasks track dependencies using GNN), Progress Monitoring (UI shows all parallel tasks with real-time updates), Failure Isolation (one task failure doesn't affect others), Context Recovery (if agent loses context, reload from task registry)

#### 3.4.2.3 Deployment State Machine (MVP - Railway Focus)

Responsibility: Deploy validated code to Railway.app

Entry Point: Code + passing tests

Exit Point: Live Railway URL + health status

Trigger: Manual (requires user approval)

Platform: Railway.app only in MVP

State Count: 6 states

States:

1. PackageBuilding: Create Docker image or build artifacts
2. Process: Build Docker image with multi-stage builds, create deployment package, optimize artifact size
3. Parallel Processing: Build multiple layers/stages simultaneously (Docker multi-stage builds)
4. Performance Target: 30-60s
5. ConfigGeneration: Generate railway.json, Dockerfile, environment config
6. Process: Generate Railway-specific configuration, create Dockerfile if not present, configure environment variables, setup build and start commands, write config files via File Operations Service (Part 3, Section 3.4.3.3), validate config syntax before proceeding
7. Parallel Processing: Generate multiple config files simultaneously
8. Performance Target: <5s
9. RailwayUpload: Push code/image to Railway
10. Process: Push Docker image to Railway registry, upload application code, trigger Railway build process
11. Conversation Linkage (NEW): Create deployment_session record with session_id and link to parent codegen_session and test_session, Update conversation_messages.linked_deployment_session with deployment_session_id, Store deployment_url in deployment_session for traceability, Enable full traceability: user message → codegen_session → test_session → deployment_session → live URL
12. Parallel Processing: Parallel upload of multiple artifacts/layers
13. Performance Target: 20-40s (depends on artifact size)
14. FixingIssues: Auto-retry with fixes if build/config/upload fails
15. Inputs: Error context from failed state (PackageBuilding, ConfigGeneration, or RailwayUpload), error type (build failure, config error, upload failure, dependency issue), error details (build logs, config validation errors, Railway API errors), failure count (1-3 attempts)
16. Process: Analyze failure type and apply appropriate recovery: If PackageBuilding failed → check for transient network errors (retry with backoff), verify Dockerfile syntax and fix common errors, check disk space and clean build cache, retry build with fresh environment; If ConfigGeneration failed → validate railway.json schema and fix, check environment variables completeness, verify build/start commands syntax, regenerate config with corrections; If RailwayUpload failed → check Railway API status and retry if transient, verify authentication tokens, check project quotas/limits, retry upload with exponential backoff, increment retry counter (max 3 attempts), log each attempt with detailed diagnostics
17. Outputs: Fixed deployment artifacts (if successful), corrected configuration files, detailed error report (if all attempts fail), Railway logs and diagnostics
18. Escalation: After 3 failed attempts mark session outcome as Failed and present error report to user including: which deployment step failed (build/config/upload), root cause analysis, Railway platform status if relevant, configuration issues found, recommendations for manual fixes (check Railway dashboard, verify environment variables, check quotas)
19. Prevention: Immediate deployment failures from transient errors, user frustration from non-retried fixable issues, wasted time on manual intervention for auto-fixable problems
20. Performance Target: <30s per retry attempt (includes rebuild/reupload time)
21. Note: FixingIssues handles pre-deployment errors; RollbackOnFailure handles post-deployment errors (after HealthCheck)
22. HealthCheck: Verify deployed service is responding
23. Process: Wait for deployment to be live, check multiple endpoints for 200 OK responses, verify expected content in responses, check for error logs
24. Parallel Processing: Check multiple endpoints simultaneously, parallel health checks for different service components, concurrent smoke tests across multiple routes
25. Performance Target: <30s
26. RollbackOnFailure: Auto-rollback if health check fails
27. Process: If health check fails, trigger automatic rollback, restore previous deployment version, notify user of failure and rollback
28. Performance Target: <30s
29. Complete: Deployment successful with live URL
30. Outputs: Live Railway URL, deployment timestamp, health check status, deployment_session_id (NEW), linked conversation messages (NEW)
31. Traceability: User can click on chat message → see generated code → see test results → see live deployment URL

Failed: Deployment failed, rollback triggered or manual intervention needed

Performance Targets (Total Cycle): <2 minutes (build + deploy + health check)

Integration with Testing: Deployment only proceeds if all tests pass (100% stable tests). Flaky tests quarantined don't block deployment.

#### 3.4.2.4 Maintenance State Machine (Post-MVP)

Responsibility: Monitor production, detect issues, auto-fix, and deploy patches

Entry Point: Deployed application in production

Exit Point: Incident resolved or escalated

Trigger: Automatic based on error detection

Philosophy: Self-healing production systems - MTTR <5 minutes for known patterns

State Count: 11 states

States:

1. LiveMonitoring: Normal monitoring state - no incidents
2. Behavior: Continuous monitoring active, alert thresholds configured, ready to transition to Incident state if issue detected
3. Monitors: Error rates, performance metrics, API response times, browser errors, user-reported issues
4. BrowserValidation: Real User Monitoring (RUM), session replay
5. Purpose: Monitor actual user behavior and detect issues
6. Monitors: JavaScript errors in production, network failures, performance degradation, user session recordings, broken user flows
7. Parallel Processing: Monitor multiple user sessions simultaneously
8. Performance Target: <1s (detection latency)
9. ErrorAnalysis: Pattern matching against known issues, root cause analysis
10. Process: Classify error type (browser, API, database, performance), search known issues database for similar patterns, analyze error frequency and impact, determine severity (Low, Medium, High, Critical)
11. Parallel Processing: Analyze multiple error patterns concurrently
12. Performance Target: <5s
13. IssueDetection: Root cause analysis, dependency graph queries
14. Process: Use dependency graph to trace error source, identify affected components, determine blast radius of issue, classify as known vs unknown issue
15. Parallel Processing: Query multiple dependency paths simultaneously
16. Performance Target: <3s
17. AutoFixGeneration: LLM generates patch based on error context
18. Inputs: Error logs, stack traces, affected code, similar past fixes from Yantra Codex
19. Process: Retrieve similar fixes from known issues database, LLM generates fix candidates, score each candidate for likelihood of success, write patch files via File Operations Service (Part 3, Section 3.4.3.3), create temporary Git branch for fix validation
20. Outputs: Fix candidate code, confidence score, impact analysis
21. Parallel Processing: Generate multiple fix candidates concurrently
22. Performance Target: 10-30s (LLM generation)
23. FixValidation: Test fix in staging using CodeGen + Testing machines
24. Process: Apply fix to staging environment, run full test suite (uses Test Intelligence + Test Execution machines), verify fix resolves the issue, ensure no new issues introduced
25. Parallel Processing: Leverages parallel processing from CodeGen and Testing machines
26. Performance Target: 2-3 minutes (full test suite with parallel execution)
27. CICDPipeline: Automated deployment of validated fix
28. Actions: Build Docker image, push to Railway/production, deploy with zero-downtime strategy (blue-green, canary), monitor deployment health
29. Parallel Processing: Leverages parallel processing from Deployment machine
30. Performance Target: 1-2 minutes (parallel deployment pipeline)
31. VerificationCheck: Confirm issue is resolved in production
32. Verification Criteria: Error rate drops to baseline, no new errors introduced, users reporting issue confirm fix, performance metrics stable, all health checks pass
33. Parallel Processing: Check error rates across multiple regions simultaneously, parallel verification across different service instances, concurrent user impact assessment
34. Performance Target: <30s (parallel checks across regions)
35. LearningUpdate: Update knowledge base with new patterns
36. What Gets Updated: Error pattern → Solution mapping in Vector DB, documentation with new known issue, metrics dashboard (MTTR for this error type), team notification/report, incident log for audit trail
37. Purpose: Learn from every incident to improve future response
38. Parallel Processing: Update multiple knowledge stores concurrently (Vector DB, documentation, metrics)
39. Performance Target: <5s
40. Active (Normal State): Normal monitoring state - no incidents
41. Incident (Active Incident State): Active incident being handled
42. Behavior: Cycles through states 3-9 (ErrorAnalysis → LearningUpdate), human can intervene at any point, escalation if auto-fix fails after N attempts

Self-Healing Loop:

1. Detect: Production error occurs (browser crash, API failure, performance degradation)
2. Analyze: Pattern matching against known issues, root cause analysis
3. Generate Fix: LLM generates patch based on error context
4. Validate Fix: Run through CodeGen + Testing machines (full validation)
5. Deploy: Automated CI/CD pipeline pushes fix to production
6. Verify: Confirm error rate drops, users unaffected
7. Learn: Update Yantra Codex, add monitoring, record in incident log

Performance Targets (Total Self-Healing Cycle): <5 minutes for known patterns

Escalation: If auto-fix fails after 3 attempts, escalate to human with full context (error logs, attempted fixes, root cause analysis)

---

[Part 2 Complete - State Machines 3.4.2.1 through 3.4.2.4] [Next: Part 3 will cover Documentation Governance State Machine, Shared Services, Supporting Sections, and Advanced Features]

# Yantra: Complete Technical Specification - Part 3

Documentation Governance, Shared Services, and Advanced Features

---

### 3.4.2.5 Documentation Governance State Machine (6th Machine - Post-MVP)

Responsibility: Maintain accurate, traced documentation throughout development lifecycle

Philosophy: Documentation is NOT an afterthought. It's generated and updated continuously as code evolves, with full traceability to requirements, architecture, and tests.

Integration: Runs in PARALLEL with all other state machines, not sequentially. Has hooks in each existing machine (CodeGen, Testing, Deploy, Maintenance).

States:

1. DocumentationAnalysis: Analyze what documentation is needed
2. Triggers: Code generation completes, Architecture changes, Test results available, Deployment completes, Maintenance fixes applied
3. Inputs: Event from triggering machine (CodeGen/Testing/Deploy/Maintenance), current project state, existing documentation
4. Process: Determine documentation type needed (requirements, architecture, API docs, change log, decision log), identify affected documentation blocks, check if SSOT (Single Source of Truth) validation needed
5. Outputs: Documentation requirements list, affected blocks
6. Performance Target: <500ms
7. BlockIdentification: Identify which YDoc blocks need creation/update
8. Inputs: Documentation requirements from DocumentationAnalysis
9. Process: Query YDoc Block DB for existing blocks, identify blocks needing updates (stale content, outdated links), identify missing blocks (new features, new APIs), map blocks to dependency graph nodes
10. Outputs: Blocks to create, blocks to update, blocks to link
11. Performance Target: <1s (database queries + GNN lookups)
12. ContentGeneration: Generate/update documentation content
13. Inputs: Block specifications from BlockIdentification, code context, test results, deployment logs, conversation context (NEW)
14. Conversation Integration (NEW): Retrieve conversation messages linked to codegen/test/deployment sessions, Use conversation context to understand user intent and requirements, Include user's original natural language intent in documentation, Link documentation blocks to originating conversation messages for audit trail
15. Process: Generate markdown content using LLM (based on code, architecture, tests, conversation), extract examples from test cases, generate API documentation from code signatures, create decision log entries from architecture changes, write YDoc block files via File Operations Service (Section 3.4.3.3)
16. Outputs: Generated content for each block, linked to conversation for traceability
17. Performance Target: <10s (LLM generation for multiple blocks)
18. GraphLinking: Create traceability edges in dependency graph
19. Inputs: Generated blocks, code files, test files, requirements
20. Process: Create edges: requirement → architecture block (traces_to), architecture → spec block (implements), spec → code file (realized_in), requirement → test file (tested_by), code → API doc block (documented_in), issue → change log block (has_issue)
21. Outputs: Graph edges for full traceability
22. Performance Target: <500ms (graph edge creation)
23. ConflictDetection: Check for conflicting documentation (SSOT validation)
24. Purpose: Ensure Single Source of Truth - no duplicate or conflicting documentation
25. Inputs: Generated blocks, existing documentation
26. Process: Check for duplicate blocks (same topic, different content), detect conflicting statements (requirement says X, code does Y), identify outdated blocks (code changed but doc didn't), verify SSOT rules (only one architecture doc, one requirements doc per topic)
27. Outputs: Conflict report, resolution recommendations
28. Performance Target: <2s (semantic similarity checks across all docs)
29. UserClarification: Request user input for conflicts
30. Triggered when: Conflicts detected in ConflictDetection state, ambiguous documentation updates needed, breaking changes to existing docs
31. Display: Show conflicting documentation side-by-side, highlight differences, suggest resolution options
32. User Actions: Choose primary source, merge content, mark as different contexts, update SSOT
33. Performance Target: N/A (waits for human)
34. ConflictResolution: Apply user's resolution or auto-resolve simple conflicts
35. Inputs: User decisions from UserClarification, conflict report
36. Process: Apply user's chosen resolution (update primary, merge, archive), auto-resolve simple conflicts (code changed → update API doc, test added → update test plan), update SSOT registry
37. Outputs: Resolved documentation, updated blocks
38. Performance Target: <1s
39. Validation: Verify documentation quality and completeness
40. Checks: All code has corresponding API documentation, all requirements have linked tests, all architecture decisions have ADRs, all breaking changes logged, traceability complete (requirement → architecture → spec → code → test → doc)
41. Outputs: Validation report, quality score
42. Performance Target: <2s (GNN traversal for traceability)
43. Complete: Documentation updated and synced

Integration Hooks in Existing Machines:

CodeGen Machine Integration:

| State                  | Documentation Hook                   | Trigger                            |
| ---------------------- | ------------------------------------ | ---------------------------------- |
| ArchitectureGeneration | Generate architecture documentation  | After architecture created/updated |
| Complete               | Generate API documentation from code | After code generation succeeds     |

Testing Machine Integration:

| State    | Documentation Hook                 | Trigger             |
| -------- | ---------------------------------- | ------------------- |
| Complete | Update test coverage documentation | After tests pass    |
| Complete | Log test results in YDoc           | After each test run |

Deployment Machine Integration:

| State             | Documentation Hook           | Trigger                   |
| ----------------- | ---------------------------- | ------------------------- |
| Complete          | Log deployment in change log | After deployment succeeds |
| RollbackOnFailure | Log rollback in change log   | After rollback            |

Maintenance Machine Integration:

| State             | Documentation Hook                   | Trigger                      |
| ----------------- | ------------------------------------ | ---------------------------- |
| AutoFixGeneration | Document issue and fix in change log | After fix generated          |
| LearningUpdate    | Update Yantra Codex documentation    | After learning from incident |

Performance Targets (Total Cycle): <20s (for full documentation update cycle)

Success Criteria: 100% requirements linked to code via traceability chain, >95% documentation accuracy (no conflicts), <24h doc freshness (documentation updated within 24h of code change)

### 3.4.3 Shared Services

These services are called by multiple state machines and provide cross-cutting functionality.

#### 3.4.3.1 CreationValidation Service (Universal De-Duplication)

Purpose: Prevent duplicate files/functions/documentation blocks/tests/configs across ALL creation operations

Called By: All state machines when creating any artifact (CodeGen creating files, Testing creating test files, Documentation creating blocks, Deployment creating configs)

Validation Steps:

Step 1: Path Check

1. Query file system for exact path match
2. If exists: Return "duplicate_path"

Step 2: Name Check

1. For functions: Query dependency graph for function with same name in same scope
2. For classes: Query for class with same name in same module
3. For documentation blocks: Query YDoc Block DB for block with same yantra_id
4. If exists: Return "duplicate_name"

Step 3: Semantic Check

1. Generate embedding for content to be created (using all-MiniLM-L6-v2, 384-dim)
2. Query de-duplication index (Vector DB) for similar embeddings
3. Calculate cosine similarity with existing artifacts
4. If similarity >0.85: Return "semantic_duplicate" with list of similar artifacts

Step 4: Dependency Check

1. Query dependency graph for artifacts with same imports/dependencies
2. If same imports + same functionality: Return "functional_duplicate"

Resolution Options (presented to user or agent):

Option 1: Reuse Existing

1. Don't create new artifact
2. Return reference to existing artifact
3. Update calling code to use existing

Option 2: Update Existing

1. Modify existing artifact with new requirements
2. Run tests to verify no breaking changes
3. Update documentation to reflect changes

Option 3: Create New (with justification)

1. Agent provides justification why new artifact needed
2. LLM validates justification
3. If valid: Create new with clear differentiation in naming
4. If invalid: Force reuse of existing

Option 4: User Clarification

1. Present both existing and proposed new to user
2. User decides: reuse, update, or create new
3. User provides guidance on differentiation

De-Duplication Index Schema:

| Column               | Type             | Description                             |
| -------------------- | ---------------- | --------------------------------------- |
| id                   | TEXT PRIMARY KEY | Hash UUID                               |
| content_hash         | TEXT NOT NULL    | SHA-256 of content                      |
| artifact_type        | TEXT NOT NULL    | file, function, doc_block, test, config |
| artifact_id          | TEXT NOT NULL    | ID of actual artifact                   |
| file_path            | TEXT             | Path if file-based artifact             |
| created_at           | TEXT NOT NULL    | ISO-8601 timestamp                      |
| similarity_embedding | BLOB             | 384-dim vector for semantic similarity  |

Performance Targets:

1. Path check: <1ms (filesystem lookup)
2. Name check: <5ms (dependency graph query)
3. Semantic check: <50ms (vector similarity search)
4. Dependency check: <10ms (GNN query)
5. Total validation: <100ms

Integration Example in CodeGen Machine:

// In CodeGeneration state, before creating file:

let validation_result = creation_validation_service

    .validate_file_creation(&file_path, &generated_code).await?;

match validation_result {

    ValidationResult::NoDuplicate => {

    // Proceed with file creation

    fs::write(&file_path, &generated_code)?;

    },

    ValidationResult::Duplicate { existing, similarity } => {

    if similarity > 0.95 {

    // Very high similarity - force reuse

    return Ok(ReuseExisting(existing));

    } else {

    // Ask agent/user for clarification

    let decision = prompt_deduplication_decision(&existing, &generated_code).await?;

    apply_decision(decision).await?;

    }

    }

}

#### 3.4.3.2 Browser Validation Service

Purpose: Validate UI components and user flows in actual browser

Called By: CodeGen Machine (BrowserValidation state), Testing Machine (BrowserTesting state), Maintenance Machine (BrowserValidation state)

Capabilities:

1. Launch browser via CDP (Chrome/Chromium/Edge)
2. Navigate to localhost URL
3. Execute test scenarios (clicks, form fills, navigation)
4. Capture console errors and network failures
5. Take screenshots for visual verification
6. Monitor performance metrics (load time, FPS)

Service Interface:

pub trait BrowserValidationService {

    async fn launch_browser(&self) -> Result`<BrowserSession>`;

    async fn navigate(&self, session: &BrowserSession, url: &str) -> Result<()>;

    async fn execute_scenario(&self, session: &BrowserSession, scenario: &TestScenario) -> Result`<ScenarioResult>`;

    async fn capture_console_errors(&self, session: &BrowserSession) -> Result<Vec`<ConsoleError>`>;

    async fn take_screenshot(&self, session: &BrowserSession, path: &Path) -> Result<()>;

    async fn close_browser(&self, session: BrowserSession) -> Result<()>;

}

Performance Targets:

1. Browser launch: <2s
2. Page navigation: <5s
3. Scenario execution: 5-30s (depends on scenario complexity)
4. Console error capture: <100ms
5. Screenshot capture: <500ms

#### 3.4.3.3 File Operations Service

Purpose: Centralized file system operations with atomic transactions, error handling, and dependency graph integration

Called By: CodeGen Machine (State 13 - CodeGeneration), Testing Machines (test file creation), Documentation Machine (YDoc file writes), Deployment Machine (config file generation)

Service Interface:

pub trait FileOperationsService {

    async fn write_file(&self, path: &Path, content: &str) -> Result`<FileWriteResult>`;

    async fn write_files_batch(&self, files: Vec<(PathBuf, String)>) -> Result`<BatchWriteResult>`;

    async fn edit_file(&self, path: &Path, edits: Vec`<Edit>`) -> Result`<FileEditResult>`;

    async fn create_directories(&self, path: &Path) -> Result<()>;

    async fn validate_path(&self, path: &Path) -> Result`<PathValidation>`;

    async fn rollback_writes(&self, transaction_id: &str) -> Result<()>;

}

File Write Operation Flow:

Step 1: Path Determination

1. Priority order: User-specified path (explicit in request) > Architecture-defined path (from architecture document) > Dependency graph proximity (near related files) > Project conventions (src/, tests/, docs/, etc.)
2. Resolve relative paths to absolute workspace paths
3. Validate path safety (no parent directory escapes, workspace-only)
4. Check path doesn't conflict with system files or .yantra/ directory

Step 2: Pre-Write Validation

1. Call CreationValidation Service (de-duplication check)
2. Check file system permissions (write access to directory)
3. Check available disk space (minimum 100MB required)
4. Verify parent directories exist or can be created
5. If file exists: determine if new file or modification

Step 3: File Write Execution

1. For new files:
   - Create parent directories recursively (fs::create_dir_all)
   - Write content atomically (fs::write with temp file + rename)
   - Set permissions (0644 for regular files, 0755 for executables)
   - Record write in transaction log
2. For file modifications:

- Use file_edit tool for surgical edits (preserves formatting)
- Backup original file (copy to .yantra/backups/)
- Apply edits line-by-line or via AST transformations
- Validate syntax after edit (tree-sitter)

Step 4: Post-Write Actions

1. Update dependency graph:
   - Add file node if new file
   - Create edges for imports/dependencies
   - Update reverse dependencies (files that import this)
2. Invalidate context cache (Tier 4):

- Remove cached contexts containing this file
- Mark related files as needing re-tokenization

1. Log to Change Log (YDoc):

- Record file creation/modification event
- Link to requirement/task that triggered change

1. Stage for Git (optional):

- git add `<filepath>`
- Prepare for eventual commit

Step 5: Transaction Commit

1. Mark transaction as successful
2. Release file locks (if Phase 2A multi-agent)
3. Return success result with file metadata

Error Handling:

Error Type: Permission Denied

1. Detection: fs::write returns ErrorKind::PermissionDenied
2. Recovery: Try alternative path in /tmp/ or user home directory, log warning about permission issue, if alternative succeeds: notify user of non-standard location, if alternative fails: escalate to user with detailed error
3. Prevention: Pre-flight permission check before write
4. Performance: <10ms (check + alternative path attempt)

Error Type: Disk Full

1. Detection: fs::write returns ErrorKind::OutOfSpace OR pre-flight disk space check
2. Recovery: Check available space on all mounted volumes, attempt write to volume with space if found, if no space available: fail immediately with clear error, do NOT retry (space unlikely to free up quickly)
3. User Notification: "Disk full. Free at least 100MB and retry."
4. Prevention: Pre-flight disk space check (fail fast)
5. Performance: <50ms (disk space query)

Error Type: Path Does Not Exist

1. Detection: fs::write returns ErrorKind::NotFound
2. Recovery: Attempt to create parent directories (fs::create_dir_all), if parent creation succeeds: retry write, if parent creation fails: check if path is valid, escalate to user if path is invalid
3. Prevention: Pre-create parent directories in Step 3
4. Performance: <20ms (mkdir + retry)

Error Type: File Already Exists

1. Detection: Should not occur (CreationValidation Service catches this)
2. Recovery: If validation was bypassed: query user (overwrite, rename, merge), log warning about validation bypass
3. Prevention: Always call CreationValidation Service before write
4. Performance: <1ms (already validated)

Error Type: Invalid Filename

1. Detection: fs::write returns ErrorKind::InvalidInput OR path validation fails
2. Recovery: Sanitize filename (remove invalid characters: <>:"/\|?\*), replace spaces with underscores, truncate to 255 characters max, retry with sanitized name
3. Prevention: Path validation in Step 1 catches most cases
4. Performance: <5ms (sanitization + retry)

Error Type: Path Too Long

1. Detection: Path length > 260 characters (Windows) or > 4096 (Unix)
2. Recovery: Shorten path components (abbreviate directory names), move file closer to workspace root, suggest shorter module/package names to user
3. Prevention: Path length check in validation
4. Performance: <5ms

Multi-File Write Strategy:

Sequential Write (MVP)

1. Write files in dependency order:
   - Dependencies first (imported files)
   - Dependents second (files that import)
   - Independent files (no order dependency)
2. Atomic transaction semantics:

- All writes succeed OR all writes rollback
- Transaction log tracks writes in current batch
- On any failure: rollback all previous writes in batch

1. Rollback procedure:

- Restore files from .yantra/backups/
- Delete newly created files
- Update dependency graph to previous state
- Invalidate all caches (full refresh)

1. Performance: 50-200ms per file (depends on size)

Parallel Write (Phase 2A - Team of Agents)

1. Identify independent files (no direct dependencies)
2. Launch parallel write operations (one per agent)
3. File locking prevents same-file conflicts:
   - Acquire file lock before write (sled Tier 2)
   - Hold lock during write operation
   - Release lock after graph update
4. Dependency-ordered for related files:

- Parallel within dependency levels
- Sequential across dependency levels

1. Performance: ~40% faster for large batches (4+ files)

Performance Targets:

1. Single file write: <50ms (validation + write + graph update)
2. Batch write (5 files): <200ms sequential / <120ms parallel
3. Rollback operation: <100ms (restore from backups)
4. Directory creation: <20ms (recursive mkdir)
5. Permission check: <10ms (filesystem stat)
6. Disk space check: <50ms (df command)

Integration Example in CodeGen State 13:

// After LLM generates code (Phase C):

let files_to_write = vec![

    (PathBuf::from("src/auth.py"), auth_code),

    (PathBuf::from("tests/test_auth.py"), test_code),

];

// Call File Operations Service

let result = file_ops_service

    .write_files_batch(files_to_write)

    .await?;

match result {

    BatchWriteResult::Success { written_files } => {

    // Update dependency graph for each file

    for file_path in written_files {

    dependency_graph.add_file_node(&file_path).await?;

    dependency_graph.extract_imports(&file_path).await?;

    }

    // Invalidate context cache

    context_cache.invalidate_files(&written_files).await?;

    // Log to Change Log

    ydoc_service.log_change(

    ChangeType::FilesCreated,

    written_files.clone()

    ).await?;

    // Transition to DependencyValidation state

    transition_to(State::DependencyValidation)?;

    },

    BatchWriteResult::PartialSuccess { written, failed } => {

    // Some files wrote, some failed

    log::warn!("Partial write success: {} ok, {} failed",

    written.len(), failed.len());

    // Rollback successful writes

    file_ops_service.rollback_writes(&transaction_id).await?;

    // Analyze failures

    let error_context = analyze_write_failures(&failed);

    // Transition to FixingIssues state

    transition_to(State::FixingIssues {

    error_context,

    retry_count: 0

    })?;

    },

    BatchWriteResult::Failure { error } => {

    // Complete failure - no files written

    log::error!("File write batch failed: {}", error);

    match error.kind {

    WriteErrorKind::DiskFull => {

    // Fail immediately - cannot recover

    notify_user("Disk full. Free space and retry.").await?;

    transition_to(State::Failed {

    reason: "Insufficient disk space".into()

    })?;

    },

    WriteErrorKind::PermissionDenied => {

    // Try alternative location

    let alt_result = try_alternative_locations(&files_to_write).await?;

    if alt_result.success {

    notify_user("Files written to alternative location.").await?;

    // Continue with alternative paths

    } else {

    // Escalate to user

    transition_to(State::Failed {

    reason: "Permission denied, no alternatives".into()

    })?;

    }

    },

    _ => {

    // Retryable error

    transition_to(State::FixingIssues {

    error_context: error.into(),

    retry_count: 0

    })?;

    }

    }

    }

}

Transaction Log Schema (Tier 1 SQLite):

| Column         | Type             | Description                             |
| -------------- | ---------------- | --------------------------------------- |
| transaction_id | TEXT PRIMARY KEY | UUID for this batch write               |
| started_at     | TEXT NOT NULL    | ISO-8601 timestamp                      |
| completed_at   | TEXT             | NULL if in progress                     |
| status         | TEXT NOT NULL    | pending, committed, rolled_back, failed |
| file_paths     | TEXT NOT NULL    | JSON array of paths in batch            |
| backup_paths   | TEXT             | JSON array of backup locations          |
| error_log      | TEXT             | JSON array of errors if failed          |

Rollback Procedure:

1. Query transaction log for transaction_id
2. For each file in backup_paths:
   - Copy backup back to original location
   - Verify file contents match backup (hash check)
3. For each newly created file:

- Delete file from filesystem
- Remove from dependency graph

1. Update transaction status to "rolled_back"
2. Clear transaction from active transactions list
3. Performance: <100ms for typical batch (3-5 files)

Extended Integration Example with Full Error Handling:

// Complete flow with error handling:

pub async fn write_generated_code(

    file_path: PathBuf,

    generated_code: String,

    services: &Services

) -> Result`<WriteCodeResult>` {

    // Step 1: Validate file creation (de-duplication)

    let validation_result = services.creation_validation

    .validate_file_creation(&file_path, &generated_code).await?;

    match validation_result {

    ValidationResult::Duplicate { existing, similarity } => {

    if similarity > 0.95 {

    return Ok(WriteCodeResult::ReuseExisting(existing));

    }

    let decision = prompt_deduplication_decision(&existing, &generated_code).await?;

    if decision == Decision::ReuseExisting {

    return Ok(WriteCodeResult::ReuseExisting(existing));

    }

    },

    ValidationResult::NoDuplicate => { /* proceed */ }

    }

    // Step 2: Write file with comprehensive error handling

    match services.file_ops.write_file(&file_path, &generated_code).await {

    Ok(write_result) => {

    // Success path

    services.dependency_graph.add_file_node(&file_path).await?;

    services.dependency_graph.extract_imports(&file_path).await?;

    services.context_cache.invalidate_file(&file_path).await?;

    services.ydoc.log_change(ChangeType::FileCreated, &file_path).await?;

    Ok(WriteCodeResult::Success {

    path: file_path,

    size: write_result.bytes_written

    })

    },

    Err(e) => {

    // Error handling for specific cases

    match e.kind {

    FileWriteErrorKind::PermissionDenied => {

    log::warn!("Permission denied for {:?}, trying alternative", file_path);

    let alt_path = find_alternative_path(&file_path)?;

    return write_generated_code(alt_path, generated_code, services).await;

    },

    FileWriteErrorKind::DiskFull => {

    return Err(anyhow!("Disk full - cannot write file"));

    },

    FileWriteErrorKind::PathTooLong => {

    let short_path = shorten_path(&file_path)?;

    return write_generated_code(short_path, generated_code, services).await;

    },

    FileWriteErrorKind::ParentNotFound => {

    services.file_ops.create_directories(&file_path.parent().unwrap()).await?;

    return write_generated_code(file_path, generated_code, services).await;

    },

    _ => {

    log::error!("File write failed: {}", e);

    return Err(e.into());

    }

    }

    }

    }

}

#### 3.4.3.4 SSOT Validation Service

Purpose: Enforce Single Source of Truth for critical project artifacts

Called By: Documentation Governance Machine (ConflictDetection state), CodeGen Machine (ArchitectureGeneration state)

SSOT Rules:

Architecture:

1. Only ONE architecture document per project
2. Location: .yantra/architecture.db (canonical) or user-specified SSOT file
3. Conflict: Multiple architecture files → User must choose primary
4. Resolution: Archive non-primary, redirect references to primary

Requirements:

1. Only ONE requirements document per epic/feature
2. Duplicate requirements → Merge or mark as sub-requirements
3. Conflicting requirements → User clarification required

API Specifications:

1. Only ONE API spec per endpoint
2. OpenAPI spec is canonical if present
3. Code-generated docs must match OpenAPI spec

Validation Process:

pub struct SSOTValidator {

    pub fn validate_ssot(&self, artifact_type: ArtifactType, project_id: &str) -> Result`<SSOTValidationResult>` {

    match artifact_type {

    ArtifactType::Architecture => {

    let arch_files = self.find_architecture_files(project_id)?;

    if arch_files.len() > 1 {

    return Ok(SSOTValidationResult::MultipleFound {

    artifacts: arch_files,

    recommendation: "Choose primary architecture source"

    });

    }

    Ok(SSOTValidationResult::Valid)

    },

    ArtifactType::Requirements => {

    // Similar logic for requirements

    },

    // ... other artifact types

    }

    }

}

#### 3.4.3.5 Conversation Memory Service (NEW)

Purpose: Provide conversation context and work session linking to all state machines

Called By: All state machines that need conversation context (CodeGen, Testing, Documentation Governance)

Service Interface:

pub trait ConversationMemoryService {

    // Save messages

    async fn save_user_message(

    &self,

    session_id: &str,

    content: &str,

    linked_session: Option<&str>

    ) -> Result`<String>`;

    async fn save_agent_message(

    &self,

    session_id: &str,

    content: &str,

    linked_session: Option<&str>

    ) -> Result`<String>`;

    // Retrieve context

    async fn get_recent_messages(

    &self,

    session_id: &str,

    count: usize

    ) -> Result<Vec`<Message>`>;

    async fn get_conversation_context(

    &self,

    session_id: &str,

    current_message: &str,

    token_budget: usize

    ) -> Result`<ConversationContext>`;

    // Search

    async fn search_similar_messages(

    &self,

    session_id: &str,

    query: &str,

    top_k: usize

    ) -> Result<Vec`<Message>`>;

    async fn search_by_date_range(

    &self,

    session_id: &str,

    start: DateTime`<Utc>`,

    end: DateTime`<Utc>`

    ) -> Result<Vec`<Message>`>;

    // Link to work sessions

    async fn link_message_to_codegen(

    &self,

    message_id: &str,

    codegen_session_id: &str

    ) -> Result<()>;

    async fn link_message_to_test(

    &self,

    message_id: &str,

    test_session_id: &str

    ) -> Result<()>;

    async fn link_message_to_deployment(

    &self,

    message_id: &str,

    deployment_session_id: &str

    ) -> Result<()>;

    // Query linked artifacts

    async fn get_artifacts_from_message(

    &self,

    message_id: &str

    ) -> Result`<ArtifactChain>`;

}

pub struct ConversationContext {

    pub recent_messages: Vec`<Message>`,      // Full text, always included

    pub relevant_messages: Vec`<Message>`,    // Semantic search results

    pub summaries: Vec`<ConversationSummary>`, // Compressed old messages

    pub total_tokens: usize,

}

pub struct ArtifactChain {

    pub code_files: Vec`<String>`,

    pub test_results: Option`<TestResults>`,

    pub deployment_url: Option`<String>`,

}

Integration Examples:

CodeGen Machine (ContextAssembly state):

// Get conversation context for LLM

let conversation_ctx = conversation_service

    .get_conversation_context(session_id, current_message, 20_000)

    .await?;

// Assemble full context: conversation + code

let full_context = ContextAssembler::new()

    .add_conversation(conversation_ctx)  // 20K tokens

    .add_code_dependencies(deps)         // 80K tokens

    .add_reserve(20_000)                 // 20K reserve

    .build()?;

// Link current message to codegen session

conversation_service

    .link_message_to_codegen(message_id, codegen_session_id)

    .await?;

Testing Machine (IntentSpecificationExtraction state):

// Retrieve conversation to understand user intent

let messages = conversation_service

    .get_recent_messages(session_id, 10)

    .await?;

// Search for similar conversations about testing

let similar = conversation_service

    .search_similar_messages(session_id, "testing requirements", 5)

    .await?;

// Use conversation context to extract better test oracles

let specifications = extract_specifications(

    user_intent,

    messages,

    similar

).await?;

Documentation Governance (ContentGeneration state):

// Get conversation context for documentation

let conversation_ctx = conversation_service

    .get_conversation_context(session_id, "doc generation", 10_000)

    .await?;

// Generate documentation including user's original intent

let doc_content = llm.generate_documentation(

    code,

    tests,

    conversation_ctx  // Include user's natural language intent

).await?;

Performance Targets:

- Save message: <50ms (includes embedding generation)
- Get recent (10): <20ms
- Semantic search (top-5): <30ms
- Context assembly: <200ms (parallel retrieval)
- Link to session: <10ms (database update)

Storage: Tier 1 SQLite (.yantra/state.db)

- Tables: conversation_sessions, conversation_messages, conversation_summaries
- Indexes: FTS5 for keyword search, HNSW for semantic search
- Size: ~1KB per message, ~150KB per 1K embeddings

Complete Specification: See Section 3.1.13 (Conversation Memory System) for full implementation details, database schema, adaptive retrieval, search capabilities, and privacy/security considerations.

### 3.4.4 Supporting Sections

#### 3.4.4.1 Parallel Processing Throughout System

Yantra implements parallel processing at three levels:

Level 1: State-Level Parallelism (Within a single state, execute independent operations concurrently)

Examples:

1. DependencyAssessment: Query multiple CVE databases simultaneously
2. SecurityScanning: Run Semgrep on multiple files concurrently (4 workers)
3. UnitTesting: Run test files in parallel (pytest -n auto)
4. BlastRadiusAnalysis: Query GNN for dependents/tests/packages in parallel

Level 2: Machine-Level Parallelism (Post-MVP - Multiple state machines running simultaneously)

Examples:

1. Test Intelligence Machine generates tests while Test Execution Machine sets up environment
2. Multiple CodeGen sessions for different features
3. Background testing while user continues editing

Level 3: Cross-Machine Parallelism (Post-MVP - Multiple developers/agents, multiple features)

Examples:

1. Team of Agents: 3-10 agents working on different modules simultaneously
2. Multiple developers with their own agent instances
3. Distributed workload across agent team

Performance Benefits:

1. O(N) sequential → O(N/K) parallel where K = number of parallel workers
2. Typical speedup: 30-40% for state-level parallelism, 3-10x for machine-level parallelism
3. Total cycle time reduction: 40-60% with full parallelization

Implementation Guidelines:

When to Parallelize:

1. Independent operations (no data dependencies)
2. I/O-bound operations (file reads, API calls, database queries)
3. CPU-bound operations with large datasets (>100 files to process)
4. Operations with >10ms individual latency

When NOT to Parallelize:

1. Sequential dependencies (operation B requires output of operation A)
2. Shared mutable state (race condition risk)
3. Small operations (overhead > benefit, <10ms operations)
4. Rate limited APIs (would trigger rate limits)
5. Stateful operations where order matters (e.g., database migrations)

#### 3.4.4.2 PDC Phases Mapped to State Machines

Complete mapping of Preventive Development Cycle phases to state machine implementation:

PDC Phase 1: ARCHITECT/DESIGN → CodeGen States: ArchitectureGeneration, ArchitectureReview, DependencyAssessment, ContextAssembly

1. Prevention: Architecture violations, boundary violations, circular dependencies, scaling bottlenecks, security vulnerabilities by design
2. Approval Gate: ArchitectureReview (mandatory human-in-loop)

PDC Phase 2: PLAN → CodeGen States: TaskDecomposition, DependencySequencing, ConflictCheck, PlanGeneration, PlanReview, EnvironmentSetup

1. Prevention: Missing tasks, wrong sequencing, scope creep, environment errors, unbounded work
2. MVP Prevention: Work visibility (show who's working on what)
3. Post-MVP Prevention: File conflicts (explicit locking)
4. Approval Gate: PlanReview (optional for >5 tasks or multi-file changes)

PDC Phase 3: EXECUTE → CodeGen States + Testing Machines

1. CodeGen States (MVP): ContextAssembly, CodeGeneration (with work visibility), DependencyValidation, BrowserValidation, SecurityScanning, ConcurrencyValidation, FixingIssues
2. Prevention: Syntax/type/logic errors, breaking changes, security issues, race conditions
3. Testing States: IntentSpecificationExtraction, TestOracleGeneration, ..., Complete
4. Prevention: Regression bugs, missing coverage, integration failures
5. Quality Gate: All tests must pass (no human approval, blocks progress)

PDC Phase 4: DEPLOY → Deployment States: PackageBuilding, ConfigGeneration, RailwayUpload, HealthCheck, RollbackOnFailure

1. Prevention: Broken deploys, environment mismatches, partial deployments
2. Approval Gate: Manual trigger for safety (human approval required)

PDC Phase 5: MONITOR/MAINTAIN (Post-MVP) → Maintenance States: LiveMonitoring, BrowserValidation, ErrorAnalysis, IssueDetection, AutoFixGeneration, FixValidation, CICDPipeline, VerificationCheck, LearningUpdate

1. Prevention: Prolonged outages, repeated incidents, manual delays
2. Self-Healing: MTTR <5 min for known patterns
3. Auto-trigger: CodeGen + Testing + Deployment for fixes

Prevention Guarantees Mapped to States:

1. ✅ Architecture is Respected (PDC 1.1) → ArchitectureGeneration + ArchitectureReview
2. ✅ Tech Stack is Consistent (PDC 1.2) → DependencyAssessment
3. ✅ Requirements are Clear (PDC 1.4) → ContextAssembly
4. ✅ Plans are Explicit (PDC 2.1) → TaskDecomposition through PlanReview
5. ✅ Code is Correct (PDC 3.2) → CodeGeneration through SecurityScanning + Testing Machines
6. ✅ Conflicts are Minimized/Impossible (PDC 3.3) → ConflictCheck + FileLockAcquisition (Post-MVP)
7. ✅ Security is Built-in (PDC 3.4) → DependencyAssessment + SecurityScanning
8. ✅ Documentation is Accurate (PDC 1.6 + 3.6) → Documentation Governance Machine
9. ✅ Deployments are Safe (PDC 4.1 + 4.2) → Deployment Machine
10. ✅ Systems Self-Heal (PDC 5.1) → Maintenance Machine
11. ✅ Concurrency is Safe (PDC 3.4) → ConcurrencyValidation

#### 3.4.4.3 State Machine Communication & Session Linking

Sequential Flow (Default):

User Intent

    ↓

│ CodeGen Machine │ → Generated Code + Confidence Score

    ↓ (auto-trigger)

│ Testing Machine │ → Test Results + Coverage

    ↓ (manual approval)

│ Deployment Machine │ → Live Railway URL

    ↓ (continuous)

│ Maintenance Machine │ → Self-Healing Loop

Session Linking: Each machine maintains references to previous sessions for full traceability:

- Conversation messages store links to work sessions (NEW): conversation_messages.linked_codegen_session, conversation_messages.linked_test_session, conversation_messages.linked_deployment_session
- Testing session stores codegen_session_id (trace back to generated code)
- Deployment session stores test_session_id (trace back to test results)
- Maintenance session stores deployment_id (trace back to what's deployed)
- Full traceability chain (NEW): User Message (chat) → CodeGen Session → Test Session → Deployment Session → Live URL → Maintenance Fixes
- Bidirectional navigation (NEW): From chat message → see generated code/tests/deployment, From code file → see originating chat conversation
- UI Integration (NEW): Click chat message → UI shows: generated files, test results, deployment URL, full audit trail with timestamps
- Database schema: See Section 3.1.13.5 (Work Session Linking) for complete conversation linkage implementation

Independent Execution: Machines can be triggered independently: Re-run tests without regenerating code (Testing machine only), Re-deploy without re-running tests (Deployment machine only), Manual fix can trigger Testing then Deployment (skip CodeGen)

#### 3.4.4.4 State Persistence & Recovery

Checkpoint Creation: Each state machine saves checkpoint after every major state transition. Checkpoints stored in .yantra/checkpoints/ with timestamp and session ID.

Checkpoint Contents:

1. Current state name and progress percentage
2. Input data for current state
3. Intermediate results from previous states
4. Context data (files being modified, dependencies, etc.)
5. Timestamp and session ID

Recovery Process:

1. On crash/restart, system checks for incomplete sessions
2. Load most recent checkpoint for that session
3. Resume from last saved state
4. User sees: "Resuming code generation from validation phase..."
5. No work lost, seamless recovery

Storage Management:

1. Location: .yantra/checkpoints/
2. Retention: Keep last 20 checkpoints by default
3. User-marked "important" checkpoints never deleted
4. Auto-compress old checkpoints (gzip)
5. Background pruning (non-blocking)
6. Disk Usage: 50-200 MB typical (20 checkpoints)

#### 3.4.4.5 Cascading Failure Protection

Problem: AI agents can enter failure loops where each attempted fix creates new problems, progressively degrading codebase.

Solution: Every modification is reversible with one click. System automatically detects cascading failures and reverts to last known working state.

Protection Mechanisms:

1. Checkpoint-Based Rollback:
2. Checkpoint before every code modification
3. If validation fails, can rollback to checkpoint in <1s
4. User sees: "Changes reverted to last working state"
5. Test-Driven Validation:
6. Run affected tests after every code change
7. If tests fail, automatic rollback
8. Up to 2 automatic retry attempts with fixes
9. After 3 failures, escalate to user
10. Cascading Detection:
11. Track failure count per session
12. If >3 failures in 10 minutes, flag as cascading
13. Alert user: "Multiple failures detected, recommend human review"
14. Offer: Rollback to last stable checkpoint, Review failure history, Manual intervention
15. User-Initiated Rollback:
16. UI button: "Undo Last Change"
17. Shows: What changed, Why it failed, Checkpoint available
18. One-click rollback to any of last 20 checkpoints

System Guarantees:

1. ✅ Reversible: Every change has checkpoint
2. ✅ Immediate Detection: Tests run after every modification
3. ✅ Auto-Recovery: Up to 2 automatic fix attempts
4. ✅ User Control: Escalation after 3 failures
5. ✅ Fast: All operations <1s except LLM generation

### 3.4.5 Advanced Post-MVP Features

#### 3.4.5.1 Phase 2A: Team of Agents Architecture (Months 3-4)

Problem: Single-agent architecture becomes bottleneck for large codebases (100k+ LOC) and teams with 5+ concurrent developers.

Solution: Transform Yantra from single agent to team of coordinating agents using Master-Lead + Specialist pattern with Git coordination branch.

Terminology Note: Changed from "Master-Servant" to "Lead Agent + Specialist Agents" for clarity.

Architecture: Lead Agent + Specialist Agents:

Lead Agent (Coordinator):

1. Receives user's natural language feature request
2. Analyzes feature using dependency graph
3. Decomposes into tasks with file mappings
4. Assigns tasks to specialist agents
5. Monitors progress via coordination branch
6. Resolves conflicts if multiple agents need same file
7. Configurable LLM (Claude Sonnet 4, GPT-4 Turbo, Claude Opus 4)

Specialist Agents (each with configurable LLM):

| Agent Type          | Recommended LLM                 | Responsibility                         |
| ------------------- | ------------------------------- | -------------------------------------- |
| Coding Agent        | Claude Sonnet 4, Qwen Coder 72B | Code generation, refactoring           |
| Architecture Agent  | Claude Opus 4, Claude Sonnet 4  | System design, architectural decisions |
| Documentation Agent | GPT-4 Turbo, Claude Sonnet 4    | Documentation generation, writing      |
| UX Agent            | Midjourney API, Claude Sonnet 4 | Visual assets, UI/UX design            |
| Testing Agent       | Claude Sonnet 4, GPT-4 Turbo    | Test generation, validation            |

ALL Agents Have Configurable LLMs: Not just Lead Agent. Each Specialist Agent can use different LLM optimized for its domain. Configuration stored in agent instruction files (.yantra/agents/). Allows cost optimization (use Qwen Coder 72B for simple code, Claude for complex logic).

Git Coordination Branch:

1. Branch name: .yantra/coordination (never merges to main)
2. Purpose: Append-only event log for feature assignments and completions
3. Each agent commits events (feature_assigned, work_started, dependency_available, feature_completed)
4. Provides: Version-controlled coordination history, human-readable audit trail, distributed operation support

Workflow:

1. Lead Agent receives feature request from user
2. Lead Agent decomposes into tasks using dependency graph
3. Lead Agent assigns tasks to specialist agents, commits assignments to coordination branch
4. Specialist agents read assignments from coordination branch
5. Each specialist agent claims files via Tier 2 (sled) file locks before modification
6. Agents work independently on their Git branches
7. Agents coordinate via A2A (Agent-to-Agent) protocol for dependencies
8. On completion, agents commit feature_completed event and create PR
9. Lead Agent monitors progress, resolves conflicts if needed

Agent Scalability:

1. Small feature (login): 3-5 files → 3-5 agents max
2. Medium feature (payments): 10-15 files → 6-10 agents optimal
3. Large feature (dashboard): 30+ files → 15-20 agents, then diminishing returns
4. System-wide practical limit: 100-200 agents

Cost Optimization: Use smaller open-source models (Qwen Coder 72B) as primary generators. Leverage Yantra's existing consultation architecture where expensive models only called upon failure.

Performance Targets:

1. Lead Agent assignment overhead: <30s per feature
2. File lock operations: <5ms (Tier 2 sled)
3. Inter-agent message latency: <100ms
4. Overall speedup: 3-10x faster feature completion

Success Metrics:

1. ✅ 3-10 agents work simultaneously on same feature
2. ✅ Zero file conflicts (all prevented proactively)
3. ✅ 3x faster feature completion (15 min vs 45 min)
4. ✅ 100+ agents supported system-wide

#### 3.4.5.2 Phase 2B: Cloud Graph Database (Tier 0 - Months 4-5)

IMPORTANT NOTE: This is NOT "Cloud GNN Intelligence". The GNN (intelligence layer) runs locally. This is cloud-hosted graph database for storage and real-time coordination across agents/users.

Problem: With Team of Agents, each agent has LOCAL dependency graph. When Agent A modifies file, Agent B doesn't know until attempting to claim same file or hitting Git merge conflict. This is reactive conflict detection (bad).

Solution: Cloud Graph Database (Tier 0) - Shared, cloud-hosted dependency graph tracking real-time file modifications across all agents and users. Enables proactive conflict prevention BEFORE work starts.

Key Innovation: Combine dependency knowledge (from local dependency graph) with activity knowledge (who's modifying what) to warn agents about potential conflicts before they occur, including transitive dependencies.

Conflict Prevention Levels:

Level 1: Same File Detection (Already implemented via Tier 2 sled)

1. Agent A claims payment.py
2. Agent B tries to claim payment.py
3. Tier 2 blocks immediately ✅

Level 2: Direct Dependency Detection (NEW with Cloud Graph DB)

1. Agent A modifies payment.py
2. Agent B wants to modify checkout.py which imports payment.py
3. Cloud Graph DB warns: "Your file depends on file being modified"

Level 3: Transitive Dependency Detection (NEW)

1. Agent A modifies database.py
2. user.py depends on database.py
3. auth.py depends on user.py
4. Agent B wants to modify auth.py
5. Cloud Graph DB traces chain: "auth.py → user.py → database.py (Agent A modifying)"

Level 4: Semantic Dependency Detection (NEW)

1. Agent A changes function signature: authenticate(username, password) → authenticate(email, password, mfa_code)
2. Cloud Graph DB knows 47 files call authenticate()
3. Any agent touching those 47 files warned: "Function you're using is being modified"

Architecture: Hybrid Local + Cloud:

Tier 0: Cloud Graph Service (Hosted or Self-Hosted):

1. Per-project isolation: project:abc123:graph, project:abc123:locks, project:abc123:agents
2. API Endpoints (WebSocket/gRPC, <50ms latency): claim_file, release_file, query_dependencies, query_conflicts, sync_graph
3. Privacy: Only graph structure synced, NOT code content

Tier 1: Local GNN (In-Memory, <1ms queries):

1. Fast local queries for code generation (hot path)
2. Syncs graph structure TO Cloud Graph DB (every 30s or on change)
3. Queries Cloud Graph DB BEFORE claiming files
4. Privacy: Only sends graph structure, not code content

Data Model (Privacy-Preserving): Only graph structure shared (file paths, node types, exports, edge types), NOT code content. File modification tracking (who's working on what, when started, estimated completion), NOT actual code changes.

Technology Stack:

1. Backend: Rust with Actix-Web or Axum
2. Database: Redis (in-memory, <50ms latency) + PostgreSQL (persistence)
3. Protocol: WebSocket (real-time) or gRPC (low latency)
4. Deployment: Fly.io, Railway, or self-hosted Docker

Performance Targets:

1. Cloud Graph DB queries: <50ms latency
2. Conflict detection: Real-time (<100ms)
3. Graph sync: Every 30s or on change
4. Uptime: 99.9%

Success Metrics:

1. ✅ <50ms latency for conflict queries
2. ✅ Zero code content leaked (only graph structure)
3. ✅ 4 levels of conflict detection working
4. ✅ 100+ agents supported simultaneously
5. ✅ Proactive conflict prevention (not reactive resolution)
6. ✅ Team collaboration enabled (multiple users, same project)

#### 3.4.5.3 Phase 2C: Clean Code Mode (Months 3-4)

Overview: Automated code maintenance system that continuously monitors, analyzes, and refactors codebases to maintain optimal code health.

Core Philosophy:

1. Zero Trust: Always validate with GNN + tests before applying changes
2. Confidence-Based: Only auto-apply changes with high confidence (>80%)
3. Non-Breaking: Never break existing functionality
4. Continuous: Runs as background process with configurable intervals

Capabilities:

1. Dead Code Detection & Removal:
2. Detects: Unused functions (zero incoming calls, not entry points), unused classes (zero instantiations), unused imports (never referenced), unused variables (assigned but never read), dead branches (unreachable code paths), commented code blocks
3. Entry Points Never Removed: main() functions, API route handlers, CLI command handlers, test functions, event handlers, exported public APIs
4. Confidence Calculation: Base confidence = 1.0 if zero calls, modifiers for recent code (×0.5), public API (×0.3), exported (×0.2)
5. Auto-Remove Threshold: 0.8 (80% confidence)
6. Real-Time Refactoring:
7. Remove unused imports (auto-apply, confidence 1.0)
8. Extract duplicate code (GNN embeddings, similarity >85%)
9. Simplify complex functions (cyclomatic complexity >10)
10. Rename for clarity (LLM suggestions)
11. Consolidate error handling
12. Optimize dependencies
13. Component Hardening (after implementation):
14. Security hardening: OWASP Top 10, language-specific vulnerabilities, secret detection, auto-fix >70% success rate
15. Performance hardening: Execution time analysis, memory profiling, N+1 query detection, API latency tracking, bottleneck identification
16. Code quality hardening: Cyclomatic complexity, code smell detection, documentation coverage, maintainability index
17. Dependency hardening: Known vulnerability check, outdated dependency detection, security score calculation

Performance Targets:

1. Dead code scan: <30s for 10k LOC
2. Refactoring analysis: <1 minute
3. Hardening scan: <2 minutes
4. Total Clean Code cycle: <5 minutes

#### 3.4.5.4 Phase 3: Enterprise Automation (Months 5-8)

Transform into enterprise workflow automation platform.

New Capabilities:

1. Cross-System Intelligence:
2. Automatic discovery of external API calls
3. Schema tracking for Stripe, Salesforce, etc.
4. Breaking change detection (API version updates)
5. End-to-end data flow validation
6. Impact analysis (what breaks if X changes?)
7. Browser Automation (Full Playwright Integration):
8. DOM interaction (click, fill, extract data)
9. Authentication handling
10. Visual regression detection
11. Legacy system integration via browser control
12. Self-Healing Systems:
13. Continuous API monitoring (every 24h)
14. Schema drift detection
15. Automatic migration code generation
16. Canary testing in sandbox
17. Auto-deploy if tests pass
18. Multi-Language Support:
19. JavaScript/TypeScript parser
20. Cross-language dependencies (Python API → React frontend)
21. Node.js + React code generation
22. Context mixing across languages
23. Enterprise Features:
24. Multitenancy (tenant isolation, per-tenant configuration)
25. User accounts & authentication (OAuth, SSO, RBAC)
26. Team collaboration (shared projects, activity feeds, code review workflows)
27. Billing & subscription (usage tracking, subscription tiers, payment integration)

#### 3.4.5.5 Phase 4: Platform Maturity (Months 9-12)

Mature platform with ecosystem and enterprise-grade reliability.

Objectives:

1. 99.9% uptime
2. Support 100k+ LOC projects
3. Plugin ecosystem
4. Enterprise deployment options

New Capabilities:

1. Performance Optimization:
2. GNN queries <100ms for 100k LOC projects
3. Distributed GNN (sharding)
4. Smart caching (LLM responses, test results)
5. Advanced Refactoring:
6. Architectural refactoring (monolith → microservices)
7. Performance optimization
8. Tech debt reduction
9. Code modernization
10. Ecosystem:
11. Plugin system (extend Yantra)
12. Marketplace (plugins, templates, workflows)
13. CLI tool (for CI/CD)
14. REST API
15. SDKs (Python, JavaScript, Go)
16. Enterprise Deployment:
17. On-premise deployment
18. Air-gapped environments
19. Private cloud
20. SLA guarantees (99.9% uptime)
21. Enterprise support

---

[Part 3 Complete - Documentation Governance, Shared Services, Supporting Sections, and Advanced Features] [Full Specification Complete Across Parts 1, 1B, 2, and 3]

\*\*
