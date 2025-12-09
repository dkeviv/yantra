# Yantra: Complete Technical Specification - Revised

Version: 2.0
Date: December 2025
Document Purpose: Complete technical blueprint for building Yantra from ground zero to enterprise platform

# 1. Executive Summary

## The Vision

Yantra is a fully autonomous agentic developer - an AI-powered platform that doesn't just generate code, but executes the complete software development lifecycle: from understanding requirements to deploying and monitoring production systems.

Traditional AI Code Assistants: Help developers write code faster Yantra: Replaces the entire development workflow with autonomous agents

Unlike traditional IDEs that assist developers or AI tools that suggest code, Yantra makes artificial intelligence the primary developer, with humans providing intent, oversight, and approvals only for critical decisions.

### What "Fully Autonomous Agentic" Means

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

### The Problem We Solve

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

### The Solution

Phase 1 (Months 1-2): Code That Never Breaks + Autonomous Execution

- AI generates code with full dependency awareness
- Automated unit + integration testing (🟡 Generation complete, execution in progress)
- Security vulnerability scanning (⚪ Post-MVP)
- Browser runtime validation (⚪ Post-MVP)
- Autonomous code execution with environment setup (🆕 Week 9-10)
- Integrated terminal for command execution (🆕 Week 9-10)
- Real-time output streaming to UI (🆕 Week 9-10)
- Git integration for seamless commits (⚪ Post-MVP)

Phase 2 (Months 3-4): Team of Agents & Cloud Graph Database + Package/Deploy

- Team Agent Architecture (🆕 Month 3-4) - Master-Servant pattern with Git coordination branch for multi-agent parallelism
- Cloud Graph Database (Tier 0) (🆕 Month 4-5) - Shared dependency graph for proactive conflict prevention across agents and team members
  - Note: This is NOT "Cloud GNN" - the GNN (intelligence layer) runs locally. This is cloud-hosted graph database storage for coordination.
- Package building (Python wheels, Docker, npm) (🆕)
- Automated deployment (AWS, GCP, Kubernetes, Heroku) (🆕)
- Health checks and auto-rollback (🆕)
- Generate workflows from natural language
- Scheduled jobs and event triggers
- Multi-step orchestration with error handling and retries
- CI/CD pipeline generation (🆕)

Phase 3 (Months 5-8): Enterprise Automation & Self-Healing

- Cross-system dependency tracking
- External API monitoring and auto-healing
- Production monitoring with auto-remediation (🆕)
- Browser automation for enterprise workflows (🆕)
- Legacy system integration via browser control (🆕)
- Multi-language support (Python + JavaScript + TypeScript)
- Infrastructure as Code generation (🆕)

Phase 4 (Months 9-12): Platform Maturity & Ecosystem

- Plugin ecosystem and marketplace
- Advanced refactoring and performance optimization
- Enterprise deployment (on-premise, cloud, air-gapped)
- SLA guarantees (99.9% uptime)
- Multi-tenant enterprise features (🆕)

### Competitive Advantage

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

### ROADMAP

| Phase 1 - MVP                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Phase 2A                                                                                                                                                                                                                                        | Phase 2B                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 1. Dependency Graph and View1. File, Packages, Code1. No Function chaining1. Architecture View1. Editor1. Documentation System View1. Chat1. LLM integration1. LLM consulting mode1. Interaction Mode1. Agent Codegen & Testin1. State Machines - Codegen Testing, Deploy1. Local Yantra Codex (stretch)1. Agentic Deploy with Railway1. Agentic primitives1. File Ops1. Terminal/Smart Terminal Management1. Database Ops1. API monitoring1. Cascading Failure Protection | 1. Team of Agents1. Cloud Dependency Graph1. Cloud Yantra Codex1. Ydoc - Documentation system - Requirement to Code dependency1. Additional Agentic Deploy (AWS/GCP)1. External API dependency1. External API monitoring and MCP self healing`` | 1. Agent Monitoring primitives1. Self Healing1. State Machine - Maintanence |

Phase 1 (Months 1-2): Code That Never Breaks + Autonomous Execution

- AI generates code with full dependency awareness (✅ COMPLETE)
- Automated unit + integration testing (🟡 Generation complete, execution in progress)
- Security vulnerability scanning (⚪ Post-MVP)
- Browser runtime validation (⚪ Post-MVP)
- Autonomous code execution with environment setup (🆕 Week 9-10)
- Integrated terminal for command execution (🆕 Week 9-10)
- Real-time output streaming to UI (🆕 Week 9-10)
- Git integration for seamless commits (⚪ Post-MVP)

Phase 2 (Months 3-4): Team of Agents & Cloud Graph Database + Package/Deploy

- Team of Agents Architecture (🆕 Month 3-4) - Master-Servant pattern with Git coordination branch for multi-agent parallelism
- Cloud Graph Database (Tier 0) (🆕 Month 4-5) - Shared dependency graph for proactive conflict prevention across agents and team members
  - Note: This is NOT "Cloud GNN" - the GNN (intelligence layer) runs locally. This is cloud-hosted graph database storage for coordination.
- Package building (Python wheels, Docker, npm) (🆕)
- Automated deployment (AWS, GCP, Kubernetes, Heroku) (🆕)
- Health checks and auto-rollback (🆕)
- Generate workflows from natural language
- Scheduled jobs and event triggers
- Multi-step orchestration with error handling and retries
- CI/CD pipeline generation (🆕)

Phase 3 (Months 5-8): Enterprise Automation & Self-Healing

- Cross-system dependency tracking
- External API monitoring and auto-healing
- Production monitoring with auto-remediation (🆕)
- Browser automation for enterprise workflows (🆕)
- Legacy system integration via browser control (🆕)
- Multi-language support (Python + JavaScript + TypeScript)
- Infrastructure as Code generation (🆕)

Phase 4 (Months 9-12): Platform Maturity & Ecosystem

- Plugin ecosystem and marketplace
- Advanced refactoring and performance optimization
- Enterprise deployment (on-premise, cloud, air-gapped)
- SLA guarantees (99.9% uptime)
- Multi-tenant enterprise features (🆕)

---

# 2. Complete Architecture

## YANTRA PLATFORM

### LAYER 1: INFRASTRUCTURE

    1.1 Language Support (10/10)

    ├─ Monaco Editor

    ├─ AST Parsing: Tree-sitter (Python, JS, Rust, Go, etc.)

    ├─ Multi-language Support

    ├─ AutoCompletion

    1.2 Dependency Graph - petgraph

    ├─ Graph Structure: Nodes (files/funcs/classes), Edges (deps)

    ├─ Query Engine: <1ms dependency lookups

    ├─ Incremental Updates: <50ms per file change

    ├─ Impact Analysis: Transitive dependency traversal

    └─ Storage: In-memory (hot) + SQLite (persistence)

    1.3 Extended GNN(Dependency Graph) with pattern/template Nodes

    ├─ Code Templates: Pre-validated patterns

    ├─ Best Practices: Language-specific idioms

    ├─ Project Patterns: Learned from codebase

    └─ Semantic Search: <10ms retrieval for context assembly

    1.4 Unlimited Context Solution

    ├─ Token Counting: Track context limits per LLM

    ├─ Hierarchical Assembly: Priority-based context inclusion

    ├─ Compression: Summarize low-priority context

    ├─ Chunking: Split large operations across multiple calls

    └─ Adaptive Strategies: Dynamic context based on task type

    1.5 Yantra Codex (AI Code Generation) - GraphSAGE GNN

    ├─ Neural Network: 1024-dim embeddings, 150M parameters

    ├─ Inference: 15ms (CPU), 5ms (GPU), ~600MB model

    ├─ Pattern Recognition: 978-dim problem features → code logic

    ├─ Confidence Scoring: 0.0-1.0 (triggers LLM review < 0.8)

    ├─ Continuous Learning: Learns from LLM corrections

    1.6 Five Tier Data Storage Architecture

    ├─ Tier 0: Cloud Storage (Optional, team coordination)

    ├─ Tier 1: In-Memory (Hot path: GNN queries, active state)

    ├─ Tier 2: Local SQLite (Persistent: graph, state, history)

    ├─ Tier 3: File System (Cold: logs, backups, large artifacts)

    └─ Tier 4: Ephemeral - LLM context (Moka)

    1.5 Storage Optimization

    ├── HNSW Semantic Indexing

    ├── Tier 3: SQLite with WAL

    └── Future: Tier 0-4

    1.6 Security Infrastructure

    ├── Semgrep integration

    ├── Secrets management

    └── Dependency auditing

    1.7 Browser Infrastructure (CDP)

    ├── Launch, navigate, control

    └── Real-time monitoring

### LAYER 2: AGENTIC FRAMEWORK

The Four Pillars Philosophy: How autonomous agents think and act

- PERCEIVE
- REASON
- ACT
- LEARN

### LAYER 3: AGENTIC PRIMITIVES/TOOLS

Definition: Low-level autonomous ACTIONS the agent performs

Criteria: Must be agent-initiated, autonomous decisions

    3.1 Unified Tool Interface (UTI) aka Agentic Primitives

    ├─ Protocol Router: MCP / LSP / DAP / Builtin

    ├─ Tool Adapters: 45+ tools, 4 protocols

    ├─ Consumer Abstraction: LLM Agent + Workflow Executor

    └─ Protocol Selection: Auto-routing by capability

### LAYER 4: AGENTIC ORCHESTRATION

Definition: HOW agents coordinate primitives (not primitives themselves)

    4.1 LLM Orchestration

    ├─ Primary: Claude Sonnet 4 (code generation, reasoning)

    ├─ Secondary: GPT-4 Turbo (validation, fallback)

    ├─ Routing: Cost optimization, capability-based selection

    ├─ Failover: Circuit breaker, retry with exponential backoff

    └─ Response Caching: Redis for repeated queries

    4.2 State Machine   (Yantra Preventive Development Cycle aka YPDC)

    ├─ Phase Transitions: Architect → Plan → Execute → Deploy

    ├─ State Persistence: SQLite with WAL mode

    ├─ Rollback Support: Checkpoints at phase boundaries

    └─ Approval Gates: Human-in-loop for critical operations

    4.3 Agent Execution Intelligence

    ├─ Command classification

    ├─ Background execution

    ├─ Status transparency

    ├─ Parallel Processing

    ├─ Execution Mode - Clean Mode  (with refactoring and hardening)

    └─ Interaction Modes - Guided Mode/Auto Mode

    4.5 Workflow Coordination

    ├─ Multi-step automation

    ├─ Conditional logic

    ├─  Error recovery

    4.6 Testing Framework

    ├─ Unit Test Generation: pytest (Python), Jest (JavaScript)

    ├─ Integration Tests: End-to-end flow validation

    ├─ Coverage Analysis: 90%+ target enforcement

    ├─ Test Execution: Parallel execution, <30s typical runtime

    ├─ Result Validation: 100% pass rate mandatory

    └─ Mock UI Testing: Component isolation testing

    4.9 Browser Validation (Chrome DevTools Protocol)

    ├─ Live Preview: Real-time UI rendering via chromiumoxide

    ├─ Visual Validation: Screenshot diffs, layout verification

    ├─ Interaction Testing: Automated user flow testing

    ├─ Console Monitoring: Runtime error detection

    └─ Performance Metrics: Core Web Vitals tracking

    4.10 Dependency Validation

    ├─ GNN-Powered: Query dependency graph for conflicts

    ├─ Breaking Change Detection: Transitive impact analysis

    ├─ Circular Dependency Prevention: Pre-commit validation

    ├─ Dry Run Validation

    └─ External API Tracking: Monitor API dependencies

    4.12 Team of Agents (Distributed Intelligence)

    ├─ Architect Agent: Design, dependency planning

    ├─ Coding Agent: Code generation, pattern application

    ├─ Testing Agent: Test creation, validation orchestration

    ├─ Security Agent: Vulnerability scanning, auto-fix

    └─ Coordination: Message bus for agent communication

### LAYER 5: USER INTERFACE (AI-FIRST)

- File Explorer, Documentation System View
- Chat/Task Interface
- Code Viewer with Monaco Editor, Dependency View, Architecture View
- Browser Preview with CDP (system browser)
- Terminal View
- Real-time WebSocket updates
- SolidJS reactive UI, TailwindCSS stylinS

## 3. REQUIREMENTS

## 3.1. INFRASTRUCTURE

### 3.1.1 Language Support and Auto-Completion

- Monaco Editor with Tree-sitter for language parsing. 10 languages will be provided for MVP (JS/TS,Rust,Go,Java,C,C++,Ruby, PHP,Swift,Kotlin)
- Basic AutoCompletion will be provided by language server as fallback to LLM

### Code Auto-completion (MVP Phase 1)

Provide intelligent code autocompletion in the Monaco editor to enhance developer productivity when reviewing or manually editing AI-generated code.

---

System Architecture

Three-Tier Strategy:

1. Primary (MVP): LLM-driven intelligent completions - context-aware, multi-line suggestions
2. Secondary (MVP): GNN-powered completions - project-aware symbol suggestions
3. Fallback (MVP): Static completions - instant keyword and snippet suggestions

Fallback Logic: LLM → GNN → Static (graceful degradation if services unavailable)

---

#### 3.1.1.1. LLM-Driven Intelligent Completions (PRIMARY - MVP)

Purpose:

Provide GitHub Copilot-like intelligent, context-aware multi line code suggestions.

Show function signatures and parameter hints when typing function calls.

When Triggered:

- User pauses typing for 300ms
- User types newline after function/class declaration
- User invokes explicitly (Ctrl+Space)

What LLM Sees:

- Current file content (full file)
- Cursor position
- Last 3 lines before cursor (immediate context)
- Imported modules (from file)
- Project context summary (from GNN): available functions, classes

LLM Prompt Template:

File: {file_path}

Language: {language}

Available symbols in project:

{gnn_symbols_summary}

Current code:

{file_content}

Cursor at line {line}, column {column}.

Complete the code starting from cursor position.

Provide only the completion, no explanations.

Response Processing:

- Parse LLM response
- Extract code completion
- Format as Monaco completion item
- Show as "ghost text" (inline suggestion)

User Actions:

- Tab/Enter: Accept suggestion
- Esc: Reject suggestion
- Continue typing: Auto-reject if diverges

Performance Requirements:

- LLM response: 1-3s (streaming preferred for faster feel)
- Cache completions for same context (5 minutes)
- Show loading indicator after 500ms

Privacy:

- User file content sent to LLM (Claude/GPT-4)
- Requires user consent in settings
- Option to disable LLM completions (falls back to GNN/static)

Cost Management:

- Debounce: Wait 300ms after typing stops
- Cache: Same context = same completion
- Limit: Max 100 completions per hour per user
- Token optimization: Send only relevant context (<2000 tokens)

---

#### 3.1.1.2. GNN-Powered Context Completions (SECONDARY - MVP)

Purpose: Suggest project-specific symbols (functions, classes, methods) based on dependency graph.

When Triggered:

- Member access: object. typed
- Import statements: from module import typed
- General context: After space or newline
- Trigger characters: ., (, , \n

What GNN Provides:

A. Import Suggestions:

Typing: import |

Suggests: calculator, utils, helpers (modules in project)

B. Imported Symbol Suggestions:

Code: from calculator import add, subtract

Typing: result = a|

Suggests: add(), subtract() (imported functions)

C. Member Access Suggestions:

Code: calc = Calculator()

Typing: calc.|

Suggests: add(), subtract(), clear() (class methods via GNN)

D. Local Symbol Suggestions:

Code in file: def helper_function(): ...

Typing: hel|

Suggests: helper_function() (defined in current file)

Context Detection Logic:

- Import context: Line starts with import or ends with import
- Member access: Text before cursor ends with .
- Function call: Text before cursor ends with (
- General: Any other typing context

Performance Requirements:

- GNN query: <200ms
- Results: Top 50 most relevant symbols
- Cache: 30 seconds for same file

Priority Ranking:

1. Local symbols (defined in current file) - highest
2. Explicitly imported symbols - high
3. Available project symbols - medium

---

#### 3.1.1.3 Static Keyword/Snippet Completions (FALLBACK - MVP)

Purpose: Instant, offline completions for language keywords and common patterns.

When Used:

- LLM unavailable or disabled
- GNN not yet built or unavailable
- Instant response needed (<10ms)
- No network connection

Python Snippets (20+):

- def → Function definition with parameters
- class → Class with **init**
- if/elif/else → Conditional blocks
- for/while → Loop structures
- try/except/finally → Error handling
- with → Context manager
- import/from...import → Import statements
- async def/await → Async patterns
- @property/@staticmethod/@classmethod → Decorators
- pass/return/yield → Control flow
- Built-in functions: print(), len(), range(), open(), etc.

JavaScript/TypeScript Snippets (20+):

- function/const fn =/() => → Function definitions
- class → Class definition
- if/else if/else → Conditionals
- for/while/for...of/for...in → Loops
- try/catch/finally → Error handling
- async/await/Promise → Async patterns
- import/export → ES6 modules
- interface/type → TypeScript types
- console.log()/console.error() → Debugging
- Array methods: map(), filter(), reduce(), etc.

Performance:

- Response time: <10ms (instant)
- No network required
- Always available

#### 3.1.1.4 Privacy & Security

LLM Completions:

- ⚠️ User code sent to LLM (Claude/GPT-4)
- Requires explicit user consent on first use
- Settings option to disable LLM completions
- Fallback to GNN/static if disabled

GNN Completions:

- 🔒 100% Local - No data leaves machine
- Works offline
- Project symbols only

Static Completions:

- 🔒 100% Local - Hardcoded snippets
- No network, no data sent anywhere

Default Settings (MVP):

- LLM completions: Enabled (user can opt-out)
- Show privacy notice on first use
- Clear indication when LLM is being called

---

### 3.1.2 Dependency Graph (GNN)

Dependency Graph is a key component to track and maintain Comprehensive Dependency Tracking (ALL Bidirectional):

1. File ↔ File Dependencies (MVP)

- Import relationships (which files import which)
- Module dependencies (which modules depend on which)
- Test file ↔ Source file relationships
- Source file ↔ Documentation
- Version tracking: File content hash, last modified timestamp

2. File ↔ Code Symbol ↔ Code Symbol Dependencies (MVP)
   - Function → Function calls (caller/callee relationships)
   - Class → Class inheritance (parent/child hierarchies)
   - Method → Method invocations (cross-class calls)
   - Variable → Variable usage (data flow tracking)
   - Version tracking: Symbol signature hash, parameter changes
   - Transitive
3. Package ↔ Package Dependencies (MVP)

- Direct dependencies (package.json, requirements.txt, Cargo.toml)
- Transitive dependencies (full dependency tree)
- Peer dependencies and optional dependencies
- Version tracking: Exact version, version range, compatibility matrix

4. Tool ↔ Tool Dependencies
   - Build tool chains (webpack → babel → terser)
   - Test framework dependencies (pytest → coverage → plugins)
   - Linter/formatter chains (ESLint → Prettier → plugins)
   - Version tracking: Tool version, plugin versions, config file hash

5. Package ↔ File Dependencies
   - Which files use which packages (import numpy → file.py uses numpy)
   - Unused package detection (packages installed but never imported)
   - Package-to-module mapping (numpy → specific submodules used)
   - Version tracking: Import statement location, package version used

6. User ↔ File Dependencies (MVP)
   - Active work tracking (which developer is editing which files)
   - File modification history (who last modified, when)
   - Work visibility indicators (show active sessions on files)
   - Version tracking: User session ID, file version at edit start

7. User ↔ Git Checkout Dependencies (Post-MVP)
   - Branch-to-file mapping (which files changed in which branches)
   - Merge conflict prediction (parallel edits on same files)
   - Work isolation tracking (developer workspace state)
   - Version tracking: Git commit SHA, branch name, checkout timestamp

8. File/code ↔ external API (post MVP)
9. Function ↔ package (post MVP)
10. Method chain tracking (post MVP)

- Agent should automatically create File, Package Code dependencies based on the existing coding/requirement files as well as chat.
- For a new project or in progress project Agent will create the dependencies during code generation for files and code.
- Agent will create tech stack dependencies in the dependency graph based on the requirements , architecture files or chat with user. Should track transitive dependencies.

This data will be used asynchronously by agent as well as user to

- Avoid tech stack dependency issue before code creation
- Avoid changes based on change impact
- Avoid merge conflicts by preventing them and better collaboration
- Enable intelligent refactoring
- Enable semantic search

Package Tracking Scope (MVP):

- ✅ File → Package@Version (track what files import what packages)
- ✅ Package → Package (transitive dependencies from lock files)
- ✅ Nested function tracking (numpy.random.normal, not just numpy.array)
- ✅ Version conflict detection (simple semver-based)
- ✅ Breaking change warnings (major version bumps)
- ✅ Query: "What breaks if I upgrade X?"

Function Tracking Granularity:

- MVP: Nested attributes (numpy.random.normal, pandas.DataFrame.groupby)
- Deferred: Method chains (df.groupby().agg().reset_index()) - P2 Post-MVP

Cross-Language Strategy:

- MVP: Separate graphs per language (Python graph, JavaScript graph, Rust graph)
- Post-MVP: Unified graph with API nodes connecting languages

### 3.1.3 Extended Dependency Graph

Semantic-Enhanced Dependency Graph (Yantra):

- Single source of truth (graph contains everything)
- Automatic sync (update node → embedding updates inline)
- Single query (BFS traversal filters by similarity simultaneously)
- Precision + recall (exact dependencies + fuzzy discovery)
- Simpler architecture (no external vector DB needed)

Embedding Model Details:

- Model: all-MiniLM-L6-v2 (sentence-transformers)
- Dimensions: 384 (optimal balance of speed vs quality)
- Size: 22MB quantized ONNX model
- Inference: <8ms per embedding on CPU (fastembed-rs)
- Privacy: 100% local inference (no API calls)
- Cache: In-memory LRU for frequently accessed embeddings

Dependency Graph with Semantic Enhancement (Code Intelligence Graph):

Hybrid Search Capability (Structural + Semantic):

1. Structural Dependencies (Exact):

// Track precise code relationships

✅ Imports: "UserService imports AuthContext"

✅ Function Calls: "create_user() calls validate_email()"

✅ Inheritance: "UserProfile extends BaseProfile"

✅ Data Flow: "function returns X, passed to Y"

2. Semantic Similarity (Fuzzy):

// Each code node can have optional embedding (384-dim vector)

pub struct CodeNode {

    id, name, type, location,        // Structural metadata

    semantic_embedding: Option<Vec`<f32>`>,  // 384-dim from all-MiniLM-L6-v2

    code_snippet: Option`<String>`,    // For generating embeddings

    docstring: Option`<String>`,       // Function/class documentation

}

3. Hybrid Search Examples:

Example 1: Intent-Driven Context Assembly

# User request: "Add email validation to user registration"

# Semantic enhancement (fuzzy discovery):

find_similar_to_intent("email validation", min_similarity=0.75)

→ validate_email() function (not called yet, but semantically relevant!)

→ validate_phone() function (similar validation pattern)

→ check_email_format() function (potential duplicate to consolidate)

# Combined result: LLM gets BOTH exact dependencies AND semantically relevant code

✅ Better code generation (finds existing validation functions)

✅ Prevent duplication (discovers similar functions)

✅ Maintains correctness (exact dependencies ensure nothing breaks)

Example 2: Refactoring Detection

# Find semantically similar functions (potential duplicates)

similar = graph.find_similar_to_node(

    "utils::validate_email",

    min_similarity=0.90,

    max_results=10

)

# Results ranked by cosine similarity:

1. validate_email_format() - 0.95 (potential duplicate!)
2. check_email() - 0.92 (potential duplicate!)
3. is_valid_email() - 0.90 (potential duplicate!)

# Action: Suggest consolidating into single function

| Operation                                                              | Performance Target | Implementation                           |
| ---------------------------------------------------------------------- | ------------------ | ---------------------------------------- |
| ---------------------------------------------------------------------- |                    |                                          |
| Graph build                                                            | <5s for 10K LOC    | Incremental parsing with tree-sitter     |
| -                                                                      | -                  | -                                        |
| Incremental update                                                     | <50ms per file     | Only reparse changed file + update edges |
| Dependency lookup                                                      | <10ms              | petgraph BFS traversal                   |
| Semantic search (MVP)                                                  | <10ms              | HNSW indexing (hnsw_rs)                  |
| Embedding generation                                                   | <10ms per node     | fastembed-rs with quantized ONNX         |
| Batch embeddings (100 nodes)                                           | <100ms             | Parallel processing                      |

### 3.1.4 Unlimited Context Solution

Yantra achieves truly unlimited context through a multi-layered approach that works with ANY LLM, including smaller models like Qwen Coder:

#### 3.1.4.1. Token-Aware Context Management

Implementation:

- Real token counting using tiktoken-rs (exact, not estimated)
- Dynamic token budget allocation based on LLM provider
- Reserve 20% of context window for response generation
- Graceful degradation when approaching limits

Token Budgets:

- Claude Sonnet 4: 160,000 context tokens (40K for response)
- GPT-4 Turbo: 100,000 context tokens (28K for response)
- Qwen Coder 32K: 25,000 context tokens (7K for response)
- Adaptive allocation per LLM capability

#### 3.1.4.2. Hierarchical Context Assembly

Strategy: Provide different levels of detail based on distance from target code:

Level 1 - Immediate Context (Full Detail):

- Complete source code of target files
- All direct dependencies (imports, function calls)
- Data structures and types referenced
- Token allocation: 40% of budget

Level 2 - Related Context (Signatures Only):

- Function signatures (no implementation)
- Class definitions (no methods)
- Type definitions and interfaces
- API contracts
- Token allocation: 30% of budget

Level 3 - Distant Context (References Only):

- Module names and imports
- High-level architecture
- Indirect dependencies (A → B → C)
- Token allocation: 20% of budget

Level 4 - Metadata (Summaries):

- Project structure overview
- Known patterns and conventions
- Relevant documentation snippets
- Token allocation: 10% of budget

#### 3.1.4.3. Intelligent Context Compression

Techniques:

A. Syntax-Aware Compression:

- Remove comments (unless directly relevant)
- Strip docstrings (keep in metadata)
- Minimize whitespace
- Remove unused imports in context
- Savings: 20-30% tokens

B. Semantic Chunking:

- Split large files into logical chunks (classes, functions)
- Include only relevant chunks
- Track chunk relationships in GNN
- Savings: 40-60% tokens for large files

C. De-duplication:

- Identical code blocks referenced multiple times
- Common utility functions
- Shared type definitions
- Savings: 10-15% tokens

#### 3.1.4.4. Semantic Search Context Retrieval

Integrated Embeddings in Dependency Graph:

Embeddings Storage:

- All function signatures and docstrings
- Common code patterns
- Failure patterns with fixes (see next section)
- Best practices and conventions
-

Semantic Search:

- Query: User intent + target code context
- Retrieve: Top-K most relevant patterns (K=10-20)
- Add to context as examples
- Cost: 2,000-5,000 tokens (high value)

#### 3.1.4.5. Adaptive Context Strategies

Based on Task Type:

| Task                                                         | Strategy                           | Token Allocation |
| ------------------------------------------------------------ | ---------------------------------- | ---------------- |
| ------------------------------------------------------------ |                                    |                  |
| New feature                                                  | Wide context (many dependencies)   | 70% L1+L2        |
| -                                                            | -                                  | -                |
| Bug fix                                                      | Deep context (full implementation) | 80% L1           |
| Refactoring                                                  | Architectural context (all usages) | 50% L1, 40% L2   |
| Testing                                                      | Target code + similar tests        | 60% L1, 30% RAG  |

Dynamic Adjustment:

- Monitor LLM confidence scores
- If low confidence → expand context
- If token limit hit → compress L2/L3
- Iterative refinement

#### 3.1.4.6. Context Caching & Reuse

SQLite Cache:

- Cache compressed context by hash (file content + dependencies)
- 24-hour TTL
- Invalidate on file changes
- Performance gain: <50ms retrieval vs 100-500ms assembly

Shared Context Across Requests:

- Same file referenced multiple times
- Compute once, reuse
- Track with reference counting

Why This Enables ANY LLM (Including Qwen Coder)

The Key Insight: Most LLM failures are due to missing context, not LLM capability.

With Yantra's Context Intelligence:

1. Qwen Coder 32K (smaller model):
   - Gets 25,000 tokens of perfectly relevant context
   - Hierarchical assembly prioritizes what matters
   - Semantic enhanced dependency graph provides proven patterns
   - Known failures database prevents common mistakes
   - Result: Performs as well as GPT-4 with 100K tokens
2. Even GPT-3.5 (16K context):

- Gets 12,000 tokens of hyper-relevant context
- Every token is carefully selected
- Compression eliminates noise
- Result: Better than GPT-4 with random 100K context

Validation:

- Benchmark: Same task with GPT-4 (naive 100K context) vs Qwen Coder (optimized 25K context)
- Metric: Code quality, test pass rate, breaking changes
- Target: Qwen performance within 5% of GPT-4

Performance Targets

| Operation                               | MVP Target | Scale Target |
| --------------------------------------- | ---------- | ------------ |
| --------------------------------------- |            |              |
| Token counting                          | <10ms      | <5ms         |
| -                                       | -          | -            |
| Context assembly                        | <100ms     | <50ms        |
| Compression                             | <50ms      | <20ms        |
| Total context pipeline                  | <500ms     | <200ms       |

### 3.1.5. Yantra Codex (AI Code Generation)

Purpose: Generate code from natural language using machine learning Implementation: GraphSAGE neural network (Python/PyTorch) Code Reference: Actual neural network for pattern recognition

Architecture:

Local Yantra Codex (Tier 1) Cloud Yantra Codex (Tier 0 - Optional)

    ───────────────────────────────────────

• GraphSAGE GNN (1024-dim) • Aggregated pattern embeddings

• 150M parameters, ~600MB • Collective learning from users

• 15ms inference time \*\* \*\* • Privacy: Embeddings only, no code

• Learns from LLM feedback • Network effects → better accuracy

• Privacy: All code stays local \*\* \*\* • Opt-in only

What it does:

- Understands problem intent (978-dim features)
- Predicts logic patterns (GraphSAGE)
- Generates code with confidence scoring
- Learns from LLM corrections
- Cost optimization (90% fewer LLM calls)

IS a Graph Neural Network - Real machine learning model trained on code patterns.

Why Two Systems?

| Aspect                                                           | Dependency Graph (In code GNN is used) | Yantra Codex               |
| ---------------------------------------------------------------- | -------------------------------------- | -------------------------- |
| ---------------------------------------------------------------- |                                        |                            |
| Purpose                                                          | Code relationships                     | Code generation            |
| -                                                                | -                                      | -                          |
| Technology                                                       | petgraph (data structure)              | GraphSAGE (neural network) |
| Input                                                            | AST from tree-sitter                   | Problem description        |
| Output                                                           | Dependency queries                     | Generated code             |
| Speed                                                            | <1ms                                   | 15ms                       |
| Learning                                                         | No learning                            | Continuous learning        |
| Local/Cloud                                                      | Both (sync structure)                  | Both (sync embeddings)     |
| Code Name                                                        | "GNN" (misleading)                     | "Yantra Codex"             |

Integration: Dependency Graph provides context → Yantra Codex generates code → Dependency Graph validates new code fits properly

---

#### 3.1.5.1 Yantra Codex: AI Pair Programming Engine (DEFAULT MODE)

Yantra Codex is a hybrid AI pair programming system that combines a specialized Graph Neural Network (GNN) with premium LLMs (Claude/ChatGPT) to generate production-quality code. This pair programming approach delivers the best of both worlds: GNN speed and learning + LLM reasoning and quality.

Core Innovation:

- Yantra Codex (GNN): Fast, local, learning-focused (15ms, FREE)
- Premium LLM: Review, enhance, handle edge cases (user's choice: Claude/ChatGPT)
- Continuous Learning: Yantra learns from LLM fixes → reduces cost over time

Pair Programming Roles:

- Yantra Codex (Junior Partner): Generates initial code, handles common patterns, learns continuously
- LLM (Senior Partner): Reviews edge cases, adds error handling, teaches Yantra implicitly

Key Principles:

1. Hybrid Intelligence: GNN speed + LLM reasoning = superior quality
2. Cost Optimization: 90% cost reduction (Yantra handles most, LLM reviews selectively)
3. Continuous Learning: Yantra learns from LLM fixes → 96% cost reduction after 12 months
4. User Choice: Configure Claude Sonnet 4, GPT-4, or other premium LLMs

---

#### 3.1.5.2 Yantra Codex Architecture

1. Model Specifications

GraphSAGE Neural Network (1024-dim embeddings):

Input: 978-dimensional problem features

Layers: 978 → 1536 → 1280 → 1024

Parameters: ~150M

Model Size: ~600 MB

Inference: 15ms (CPU), 5ms (GPU)

Why 1024 dimensions:

- Sufficient capacity for multi-step logic patterns
- 55-60% initial accuracy (vs 40% with 256 dims)
- Fast inference (still feels instant)
- Room to scale to 2048+ dims later

2. Pair Programming Workflow (Default Mode)

Step 1: Yantra Codex Generates

User Request: "Create REST API endpoint to get user by ID"

    ↓

┌───────────────────────────────────────────

│ Yantra Codex Generates │

│ - Extract 978-dim features │

│ - GNN predicts logic pattern (15ms) │

│ - Tree-sitter generates code │

│ - Calculate confidence score (0.0-1.0) │

└───────────────────────────────────────────┘

↓ Confidence >= 0.8?

Step 2: LLM Review (if confidence < 0.8)

┌───────────────────────────────────────────┐

│ LLM Review & Enhancement │

│ - Send: Yantra code + confidence issues │

│ - LLM reviews edge cases │

│ - LLM adds error handling │

│ - LLM improves code quality │

│ - User's choice: Claude/GPT-4 │

└───────────────────────────────────────────┘

Step 3: Merge & Validate

┌───────────────────────────────────────────┐

│ Merge & Validate │

│ - Merge Yantra + LLM suggestions │

│ - Run GNN dependency validation │

│ - Run automated tests │

│ - User reviews final code │

└───────────────────────────────────────────┘

Step 4: Yantra Learns

┌───────────────────────────────────────────┐

│ Yantra Learns from LLM │

│ - Extract logic pattern from final code │

│ - Store: problem → LLM-enhanced logic │

│ - Incremental GNN update │

│ - Next time: Yantra will know this! │

└───────────────────────────────────────────┘

Confidence-Based Routing:

| Confidence                                             | Routing Decision    | Rationale                            | Cost    |
| ------------------------------------------------------ | ------------------- | ------------------------------------ | ------- |
| ------------------------------------------------------ |                     |                                      |         |
| 0.9-1.0                                                | Yantra alone        | Seen pattern many times              | $0      |
| -                                                      | -                   | -                                    | -       |
| 0.8-0.9                                                | Yantra alone        | Good confidence, validate with tests | $0      |
| 0.5-0.8                                                | Yantra + LLM review | Partial knowledge, need LLM help     | ~$0.015 |
| 0.0-0.5                                                | LLM alone           | Novel pattern, Yantra can't help yet | ~$0.025 |

Learning Trajectory Example:

Week 1 (CRUD endpoint pattern):

Request 1: Yantra 0.3 → LLM review → Tests pass → Yantra learns

Request 10: Yantra 0.5 → LLM review → Tests pass → Yantra learns

Request 50: Yantra 0.75 → LLM review → Tests pass → Yantra learns

Week 4 (same pattern):

Request 200: Yantra 0.88 → No LLM needed! → Tests pass

Cost Trajectory: $0.015 → $0.010 → $0.005 → $0 (100% saved)

3. Cost & Quality Benefits

Cost Trajectory (vs LLM-only baseline $25/1000 generations):

- Month 1: $9/1000 gen (64% savings) - Yantra handles 55% alone
- Month 6: $3/1000 gen (88% savings) - Yantra handles 85% alone
- Year 1: $1/1000 gen (96% savings) - Yantra handles 95% alone

Quality Guarantee: Yantra + LLM ≥ LLM alone (pair programming is better!)

Comparison Table:

| Metric                                                              | LLM Only   | Yantra + LLM (Month 1) | Yantra + LLM (Year 1) |     |     |     |
| ------------------------------------------------------------------- | ---------- | ---------------------- | --------------------- | --- | --- | --- | --- |
| ------------------------------------------------------------------- |            |                        |                       |     |     |     |
| Cost/1000 gen                                                       | $25        | $9 (64% ↓)             | $1 (96% ↓)            |     |     |     |     |
| -                                                                   | -          | -                      | -                     |     |     |     |
| Quality                                                             | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐             | ⭐⭐⭐⭐⭐            |     |     |     |
| Speed                                                               | 3-5s       | 0.5-2s                 | 0.015-0.5s            |     |     |     |
| Learning                                                            | ❌         | ✅                     | ✅✅✅                |     |     |     |
| Privacy                                                             | ❌ (cloud) | ✅ (mostly local)      | ✅ (95% local)        |     |     |     |

#### 3.1.5.3. Multi-Language Support

Universal Logic Patterns (Learned Once):

- Input validation, error handling, data transformation
- API calls, database operations, async patterns
- Algorithm patterns, architecture patterns

Language-Specific Syntax (Tree-sitter Provides):

- Keywords and operators, type systems, standard library
- Language idioms, formatting rules

Transfer Learning:

Learn "retry with exponential backoff" in Python (1,000 examples)

↓

Automatically works in JavaScript, Rust, Go, etc. (zero additional training)

↓

Tree-sitter handles syntax differences

Supported Languages:

- Python ✅, JavaScript ✅, TypeScript ✅
- Rust, Go, Java, C++, etc. (easy to add - ~50 lines per language)

#### 3.1.5.4. Continuous Learning System

What Yantra Learns:

1. Edge Cases: LLM adds null checks → Yantra learns to add them
2. Error Handling: LLM adds try-catch → Yantra learns pattern
3. Best Practices: LLM improves naming → Yantra learns conventions
4. Domain Patterns: LLM adds auth checks → Yantra learns domain rules

Learning Metrics:

Month 1: Yantra handles 55% alone, LLM needed 45% → Cost: $9/1000 gen

Month 3: Yantra handles 70% alone, LLM needed 30% → Cost: $5/1000 gen

Month 6: Yantra handles 85% alone, LLM needed 15% → Cost: $3/1000 gen

Month 12: Yantra handles 95% alone, LLM needed 5% → Cost: $1/1000 gen

Cost Reduction: 96% after 1 year!

#### 3.1.5.5. Yantra Cloud Codex (Optional, Opt-in)

Privacy-Preserving Collective Learning:

What Gets Shared:

- ✅ Logic pattern embeddings (numbers only)
- ✅ Pattern success metrics
- ✅ Anonymized complexity data

What Does NOT Get Shared:

- ❌ Actual code
- ❌ Variable/function names
- ❌ Business logic details
- ❌ User identity
- ❌ Project structure

Network Effects:

100 users × 50 requests/day = 150k patterns/month → Model v1.1 (65% accuracy)

1k users × 50 requests/day = 1.5M patterns/month → Model v1.6 (80% accuracy)

10k users × 50 requests/day = 15M patterns/month → Model v2.0 (90% accuracy)

More users = Better model = Lower LLM costs = Attracts more users (flywheel)

Accuracy Targets

Month 1: 55-60% Yantra alone, 95%+ with LLM review (64% cost savings)

Month 6: 75-80% Yantra alone, 98%+ with LLM review (88% cost savings)

Year 2: 85%+ Yantra alone, 99%+ with LLM review (92% cost savings)

Year 3+: 90-95% Yantra alone, 99.5%+ with LLM review (96% cost savings)

Implementation Components

Core Files:

- src-python/model/graphsage.py - 1024-dim GNN model
- src-tauri/src/codex/generator.rs - Pair programming orchestrator
- src-tauri/src/codex/confidence.rs - Confidence scoring
- src-tauri/src/codex/llm_reviewer.rs - LLM review & enhancement
- src-tauri/src/codex/learner.rs - Continuous learning system

Training Pipeline:

- scripts/extract_logic_patterns.py - Extract logic patterns from CodeContests
- scripts/train_yantra_codex.py - Train GNN on problem → logic mapping
- src-python/learning/incremental_learner.py - Learn from LLM fixes

### 3.1.6 Storage Architecture (5-Tier )

#### TIER 0: Cloud (Phase 2B - Team Collaboration, Optional)

| Technology Stack:``                                                                                                                                                                                                                                                                                      | Data Stored (Same as Tier 1 + Collaboration Features):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Database:PostgreSQL 15+ with pgvector extension (for vector similarity search)Pub/Sub:Redis 7+ (for real-time updates)Protocol:WebSocket (for presence tracking)Embedding Model:Same as Tier 1 (fastembed-rs, all-MiniLM-L6-v2, 384-dim)``Database:Remote cloud server (yantra-cloud.com or self-hosted) | _ File Dependencies (Same as Tier 1)_ Code Dependencies (Same as Tier 1):_ Package/Tool Dependencies (Same as Tier 1):_ LLM-Generated Metadata & Vector Embeddings (Same as Tier 1):_ Templates & Patterns (Same as Tier 1):_ Known Issues (Same as Tier 1):_ Architecture (Same as Tier 1):_ User Presence & Locks (ONLY DIFFERENCE from Tier 1 - Team-Wide Coordination):_ User/agent identifier (name, email, agent ID)_ File paths being edited with timestamps* Lock type (soft warning vs hard blocking)* Last activity heartbeat (30-second intervals)_ Lock status (claimed, active, released, abandoned)_ Scope:Cross-machine/team-wide (Tier 2 handles local multi-agent locks)_ Purpose:Prevent conflicts between team members on different machines_ Yantra Codex Collective Learning (Opt-in):_ Aggregated pattern embeddings from all users_ Model improvement metrics |

Access: <10ms vector search via pgvector HNSW, WebSocket for real-time updates

Sync Triggers: On commit, template creation, failure discovery, file open/close, heartbeat

Privacy: NO source code, function bodies, algorithms, or business logic ever synced

MVP: Not included (single-user, single-agent only)

Phase 2B: Team collaboration with cross-machine locks

#### TIER 1: Local GNN (Hot Path - In-Memory + SQLite)

| Technology Stack:                                                                                                                                                                                                                                                                                                                                               | Data Stored:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _ Graph:petgraph (in-memory graph data structure, NOT a neural network)_ Database:SQLite 3.44+ with WAL mode (persistence)_ Parsing:tree-sitter (multi-language code parsing)_ Embedding Model:fastembed-rs (all-MiniLM-L6-v2, 384-dim vectors)\* LLM Integration:For metadata extraction (GPT-3.5 or local model)Database:.yantra/graph.db(per-project, local) | 1. File Dependencies:_ Nodes: Files, Modules, Packages, Test files_ Edges: Imports/ImportedBy, Includes, References, Tests/TestedBy* Example:auth.py → user_model.py,test_auth.py → auth.py,index.js → ./utils/helper.js* Tracks: Import relationships, module dependencies, test-source mappings* Versions:File content hash, last modified timestamp2. Code Symbol Dependencies:* Nodes: Functions, Classes, Variables, Methods* Edges: Calls/CalledBy, Uses/UsedBy, Inherits/InheritedBy, Defines/DefinedBy* Example:authenticate*user() → UserModel.find(),AdminUser extends User* Tracks: Function calls (bidirectional), class hierarchies, variable usage, data flow* Versions:Symbol signature hash, parameter changes history2. Package/Tool Dependencies:* Nodes: External packages (npm, pip, cargo), Build tools (webpack, pytest), APIs* Edges: DependsOn/RequiredBy, UsesPackage/UsedByFile, UsesTool* Example:requires: express@>=4.18.0,<5.0.0,webpack → babel → terser,auth.py uses bcrypt@4.1.2_ Tracks: Direct/transitive deps, tool chains, unused packages, version constraints* Versions:Exact version, version range, compatibility matrix* Metadata:name, type, file path, line ranges, version info, timestamps* All edges bidirectional:Navigate from caller→callee AND callee→caller4. LLM-Generated Metadata & Vector Embeddings:* Text summaries (~100 chars): "Authenticates user credentials and creates session"_ Function signatures (name + params, NO bodies):authenticate*user(username: str, password: str) -> Session* Dependencies list:["UserModel.find_by_username", "bcrypt.verify", "Session.create"]_ Purpose/keywords: "Authentication", "Security", "Session Management"5. Vector embeddings:384-dim vectors generated FROM above metadata (NOT from code)_ Purpose: Semantic search (find similar functions/patterns by meaning)_ Generation: Concatenate metadata → fastembed-rs → 384-dim vector* NO code snippets stored, only metadata + embeddings6. Templates (NodeType::Template):* Pattern metadata (name, description, use cases)_ Vector embeddings for similarity searchKnown Issues (NodeType::FailureCase):_ Failure metadata along with fixes with graph edges (CausedBy, FixedBy, SimilarTo)\_ Include LLM mistakes and fixes* Vector embeddings for finding similar failuresArchitecture (NodeType::Component/Service/Layer):* Component metadata with relationship edges* Visual metadata (optional, for diagrams)* Vector embeddings for architecture search |

Access: <5ms queries (in-memory petgraph), 1000+ reads/sec

Storage: ~500MB for 100K LOC (metadata only, NO code duplication)

Semantic Search: HNSW index in-memory, <50ms for 10K nodes, <200ms for 100K nodes

MVP: Core system (all code intelligence happens here)

NOTE: This is a dependency graph (petgraph), NOT a neural network despite being called "GNN" in codebase.

#### TIER 2: Agent Coordination (Write-Heavy)

| Technology Stack:                                                                                                                          | Data Stored:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _ MVP:SQLite 3.44+ with WAL mode_ Phase 2A:sled (lock-free LSM-tree key-value store)\* Database:.yantra/agent_state.db(per-project, local) | _ Agent State:_ Active agents (ID, status, current task, capabilities)_ Agent health and heartbeat timestamps_ File Locks (Local Multi-Agent Coordination):_ Which agent is working on which file (local machine only)_ Lock timestamps and status* Lock type (soft or hard)* Heartbeat for stale lock detection* Purpose: Prevent conflicts between multiple local agents* Task Queue:_ Pending tasks, in-progress tasks, completed tasks_ Task dependencies and priorities* Inter-Agent Messages:* Agent-to-agent coordination (local machine only) |

Access: 10K reads/sec, 100K writes/sec (with sled in Phase 2A)

Storage: <10MB (transient data)

MVP: Single-agent only (no locks needed) Phase 2B: Multi-agent coordination with local file locks

#### TIER 3: Configuration (Read-Heavy, Global)

| Technology Stack:                                                                                                                                                                                 | Data Stored:                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _ Database:SQLite 3.44+ with WAL mode_ Connection Pool:r2d2 (for efficient connection management)\* Encryption:OS keyring integration for API keyDatabase:~/.yantra/config.db(global, user-level) | _ User Configuration:_ LLM provider preferences (Claude, GPT-4, Qwen)_ API keys (encrypted at rest)_ Privacy settings (cloud sync level: strict/balanced/open)_ UI preferences (theme, layout, panel sizes)_ Global Settings:_ Default templates and patterns_ Code style preferences* Test framework preferences* Linter/formatter settings |

Access: 100:1 read:write ratio, loaded at startup

Storage: <1MB

MVP: Included (essential for configuration management)

#### TIER 4: Context Cache (Ephemeral, In-Memory)

| Technology Stack:                                                                                                                             | Data Stored:                                                                                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| _ MVP:HashMap (std::collections, basic caching)_ Phase 2+:moka (thread-safe LRU cache with TTL)Database:None (pure in-memory, no persistence) | _ Assembled Contexts:_ Hierarchical context (L1 + L2 + L3 + L4 assembled)_ Token counts (pre-calculated)_ Compressed contexts* Cache key: hash(file_content + dependencies)* LLM Response Caches:\* Repeated queries with deterministic responses (optional) |

Budget: 500MB max, automatic LRU eviction

TTL: 24 hours for compressed contexts, invalidated on file changes

Cache Miss: Re-assemble from Tier 1 (500ms vs 50ms hit), system still works

Performance: 80% hit rate expected (hot contexts cached)

Purpose: Performance optimization only, NOT a limitation on context capability

MVP: Basic HashMap caching (moka in Phase 2+)

---

#### Yantra Codex Storage (SEPARATE from Tier 1 GNN)

CRITICAL: Yantra Codex is a TRUE Graph Neural Network (GraphSAGE), completely separate from the Dependency Graph.

| Local Yantra Codex:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Cloud Yantra Codex (Phase 2B, Opt-in):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Technology Stack:_ ML Framework:PyTorch (Python)_ Architecture:GraphSAGE (3-layer GNN)_ Inference:ONNX Runtime (for Rust integration, optional)_ Training:Python scripts (offline, periodic)Storage:models/yantra*codex_v1.pth(NOT in.yantra/graph.db)Data:* Model weights: 150M parameters, 1024-dim embeddings* File size: ~600MB (.pth format)* Training data: Learned patterns from LLM corrections* NOT stored in any database tierAccess:* Inference: 15ms CPU, 5ms GPU* Training: Python/PyTorch scripts (separate process)* Loading: On Yantra startup, kept in memoryPurpose:_ Generate code from natural language intent* Pattern recognition and code completion* Learn from LLM feedback over timeSeparation Rationale:_ Different technology stack (PyTorch vs Rust petgraph)_ Different purpose (code generation vs dependency tracking)_ Different update cycle (training vs real-time graph updates)\_ Standard ML model storage (.pth file) vs graph database | Technology Stack:_ Storage:S3 (for model weights)_ Metadata:PostgreSQL (NOT in Tier 0 graph database)_ Distribution:CDN for fast model downloads_ Privacy:Differential privacy for aggregated embeddingsStorage:_ Model weights: S3 bucket (s3://yantra-codex/models/)_ Metadata: PostgreSQL (version, accuracy, training date)_ NOT in Tier 0 graph databaseData:_ Aggregated pattern embeddings from all users* NO code content, NO user data* Collective learning dataset\* Model improvement metricsPrivacy:Only embeddings shared, no source code ever transmittedNetwork Effects:More users → better accuracy → lower LLM costsDeployment:yantra-cloud.com (opt-in only) |

---

#### Storage Architecture Summary

| System                                                                                                                                                                                            | Tier           | Technology Stack                               | Data Stored                                           | Locks/Coordination                               | MVP                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------ | ----------------------------------------- |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |                |                                                |                                                       |                                                  |                                           |
| Dependency Graph                                                                                                                                                                                  | Tier 0 (Cloud) | PostgreSQL + pgvector + Redis + WebSocket      | Same as Tier 1 + Team presence/locks (cross-machine)  | Team-wide locks (Alice@machine1 ↔ Bob@machine2) | Phase 2B                                  |
| -                                                                                                                                                                                                 | -              | -                                              | -                                                     | -                                                | -                                         |
| Dependency Graph                                                                                                                                                                                  | Tier 1 (Local) | petgraph + SQLite + tree-sitter + fastembed-rs | Dependencies + LLM metadata text + 384-dim embeddings | None (read-only graph)                           | ✅ MVP                                    |
| Dependency Graph                                                                                                                                                                                  | Tier 2         | SQLite → sled                                  | Agent state, LOCAL multi-agent locks, task queue      | Local locks (Agent1 ↔ Agent2 on same machine)   | MVP: No locks `<br>`Phase 2B: Multi-agent |
| Dependency Graph                                                                                                                                                                                  | Tier 3         | SQLite + r2d2                                  | User config, API keys, preferences                    | None                                             | ✅ MVP                                    |
| Dependency Graph                                                                                                                                                                                  | Tier 4         | HashMap → moka                                 | Context cache (ephemeral)                             | None                                             | ✅ MVP                                    |
| Yantra Codex                                                                                                                                                                                      | Local Model    | PyTorch + GraphSAGE                            | 150M param weights (600MB .pth file)                  | None                                             | ✅ MVP                                    |
| Yantra Codex                                                                                                                                                                                      | Cloud Model    | S3 + PostgreSQL + CDN                          | Aggregated pattern embeddings                         | None                                             | Phase 2B                                  |

#### Key Clarifications

Embeddings Explained:

- What: Vector embeddings are 384-dimensional numeric arrays generated FROM text metadata
- Input: Concatenated text (summary + signature + dependencies + keywords)
- Process: Text → fastembed-rs (all-MiniLM-L6-v2) → 384-dim vector
- Purpose: Semantic similarity search (find related code by meaning, not keywords)
- Storage: Stored alongside metadata in database (NOT separate)
- Important: NO code snippets embedded, ONLY metadata text

Example Flow:

Code: def authenticate_user(username, password): ...

↓

LLM Extracts Metadata:

- Summary: "Authenticates user credentials and creates session"
- Signature: "authenticate_user(username: str, password: str) -> Session"
- Dependencies: ["UserModel.find_by_username", "bcrypt.verify"]
- Keywords: "Authentication", "Security"

  ↓

Concatenate: "Authenticates user credentials and creates session authenticate_user username str password str Session UserModel.find_by_username bcrypt.verify Authentication Security"

↓

fastembed-rs: [0.234, -0.891, 0.445, ..., 0.123] (384 numbers)

↓

Store: metadata text + embedding vector (NO code stored)

#### Phase Breakdown

| MVP Scope - Phase 1                                                                                                                                                                                                                                                                                                                                      | Phase 2A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _ Tier 1:Core system (dependency graph, metadata, embeddings, templates, failures, architecture)_ Tier 2:Single-agent coordination (SQLite)_ Tier 3:User configuration (SQLite)_ Tier 4:Basic context cache (HashMap)_ Yantra Codex:Local model only (models/yantra_codex_v1.pth, separate from tiers)_ Tier 0:NOT included (single-user, no cloud sync) | Tier 0:Cloud sync for team collaboration* Stores SAME data asTier 1(dependency graph, metadata, embeddings, templates, failures, architecture)* ONLY DIFFERENCE:Adds team-wide user presence tracking & cross-machine locks* Aggregates data from all team members (multi-user scale)* Collective template/failure learning across team* Cloud Yantra Codex (S3 + PostgreSQL metadata, separate from Tier 0 graph)Tier 2:Multi-agent coordination (local machine only)* File locks for multiple agents on same machine\* Prevents local agent conflicts |

Key Principles:

- Tier 0 = Tier 1 + Team Collaboration: Cloud stores identical dependency graph data as local, plus cross-machine locks
- Tier 2 = Local Multi-Agent: File locks for agents on same machine (separate from team locks)
- Two-Level Locking: Local (Tier 2) for multi-agent + Team-wide (Tier 0) for multi-user
- Embeddings = Vectorized Metadata: Not separate data, just numeric representation of text metadata
- Everything code-related in Tier 1 (unified local dependency graph)
- Metadata-based embeddings (LLM-generated, NO code duplication)
- Cloud syncs metadata only (privacy-preserving, NO source code)
- Tier 4 is performance optimization (not a limitation)
- Single source of truth: Code in files, metadata in Tier 1, cloud mirrors Tier 1
- Yantra Codex is SEPARATE: Different storage, technology, purpose (ML model vs dependency graph)

---

### 3.1.7 Browser Integration with Chrome DevTools Protocol (CDP)

Overview

Purpose: Enable Yantra to launch, control, and monitor web browsers for UI validation, error capture, and interactive development workflows.

Approach: System Chrome + CDP

- Find user's installed Chrome/Chromium/Edge
- Launch with remote debugging enabled
- Control via Chrome DevTools Protocol
- Zero-touch user experience

Phases:

- MVP (Phase 1): Core browser automation, error capture, dev server management
- Post-MVP (Phase 2): Interactive element selection, visual feedback, asset management

---

### MVP Features (Required for Launch)

#### 3.1.7.1. Chrome Discovery & Auto-Download

Requirements:

- Automatically find system Chrome installation on macOS/Windows/Linux
- Platform-specific search paths:
  - macOS: Applications folder (Chrome, Chromium, Edge, Brave)
  - Windows: Program Files (Chrome, Edge)
  - Linux: /usr/bin (google-chrome, chromium, chromium-browser, snap)
- Fallback: Download minimal Chromium (~100MB) to app data folder on first launch
- Cache Chrome path for subsequent launches
- One-time setup, transparent to user

User Experience:

- First launch: Instant if Chrome found, ~30 seconds if download needed
- Show "Setting up browser preview..." during download
- All subsequent launches: Instant (path cached)

Implementation Files

- src-tauri/src/browser/chrome_finder.rs (NEW) - Platform detection
- src-tauri/src/browser/downloader.rs (NEW) - Fallback Chromium download

---

#### 3.1.7.2. Chrome Launch with CDP

Requirements:

- Launch Chrome with remote debugging on random available port
- Use app mode for minimal UI (looks native, not browser window)
- Disable first-run prompts, default browser checks, extensions
- Configure window size (default 1280x720)
- Parse debug port from Chrome stderr output
- Establish WebSocket connection to CDP

Launch Configuration:

- --remote-debugging-port=0: Auto-select available port (no conflicts)
- --app=URL: Launch in app mode
- --no-first-run, --no-default-browser-check: Skip prompts
- --disable-extensions: Faster startup
- --disable-popup-blocking: Allow OAuth popups
- --disable-background-networking: No update checks

Implementation Files:

- src-tauri/src/browser/launcher.rs (NEW) - Chrome launching
- Update src-tauri/src/browser/cdp.rs - Replace placeholder with real launch

Technology: Use chromiumoxide crate for Rust CDP implementation

---

#### 3.1.7.3. CDP Connection & Communication

Requirements:

- Establish WebSocket connection to Chrome DevTools Protocol
- Send CDP commands (navigate, execute scripts, subscribe to events)
- Receive CDP events (console messages, errors, network events)
- Handle connection lifecycle (connect, reconnect, cleanup)
- Spawn async task to handle CDP event stream

CDP Domains to Enable:

- Runtime: Console messages, exceptions, script execution
- Network: Request/response monitoring, error capture
- Page: Navigation, lifecycle events

Implementation Files:

- Rewrite src-tauri/src/browser/cdp.rs - Replace placeholders with real CDP
- Add chromiumoxide = "0.5" to Cargo.toml

---

#### 3.1.7.4. Dev Server Management

Requirements:

- Auto-detect frontend framework (Next.js, Vite, Create React App)
- Detection criteria:
  - Next.js: presence of next.config.js/mjs
  - Vite: presence of vite.config.js/ts
  - CRA: "react-scripts" in package.json

- Start dev server with appropriate command
- Parse port from server output using regex patterns:
- Next.js: "ready on [http://localhost:3000](http://localhost:3000)"
- Vite: "Local: [http://localhost:5173](http://localhost:5173)"
- CRA: Default port 3000
- Wait for server to be ready (HTTP health check)
- 30-second timeout for server startup
- Handle port conflicts (try alternate ports)

Supported Frameworks:

- Next.js: npm run dev
- Vite: npm run dev
- Create React App: npm start
- Custom: Try common ports (3000, 3001, 5173, 8080)

Implementation Files:

- src-tauri/src/browser/dev_server.rs (NEW) - Framework detection & startup

---

#### 3.1.7.5. Runtime Injection

Requirements:

- Inject JavaScript runtime into browser before page loads
- Runtime captures errors and sends to Yantra backend
- Use CDP Page.addScriptToEvaluateOnNewDocument command
- Runtime must load before any user code executes

Runtime Capabilities:

- Connect to Yantra via WebSocket
- Override console.error to capture messages
- Listen for unhandled exceptions (window.error event)
- Listen for unhandled promise rejections
- Intercept fetch() to capture network errors
- Send all errors to Yantra with full context (message, stack, timestamp)

Implementation Files:

- src-tauri/src/browser/yantra-runtime.js (NEW) - Error capture runtime
- src-tauri/src/browser/runtime_injector.rs (NEW) - Injection logic

---

#### 3.1.7.6. Console Error Capture

Requirements:

- Subscribe to CDP Runtime domain events
- Capture console.error calls with full arguments
- Capture unhandled JavaScript exceptions with stack traces
- Include source location (URL, line number, column number)
- Send errors to agent orchestrator for auto-fix
- Support source map resolution for minified code

CDP Events to Monitor:

- Runtime.consoleAPICalled: Console API calls (log, error, warn)
- Runtime.exceptionThrown: Unhandled exceptions with details

Error Data to Capture:

- Error message
- Stack trace
- Source URL
- Line/column numbers
- Timestamp
- Error type

Implementation Files:

- src-tauri/src/browser/error_capture.rs (NEW) - CDP event subscriptions

---

#### 3.1.7.7. Network Error Capture

Requirements:

- Subscribe to CDP Network domain events
- Monitor request failures (DNS errors, timeouts, CORS)
- Capture HTTP error status codes (404, 500, etc.)
- Include full request/response context
- Send network errors to agent for analysis

CDP Events to Monitor:

- Network.loadingFailed: Failed resource loads
- Network.responseReceived: Check status codes for 4xx/5xx

Error Data to Capture:

- Request URL
- HTTP status code and text
- Error type (DNS, timeout, CORS, etc.)
- Request headers
- Response headers (if available)
- Timestamp

Implementation Files:

- src-tauri/src/browser/network_monitor.rs (NEW) - Network monitoring

---

#### 3.1.7.8. Browser Validation

Requirements:

- Launch Chrome with CDP for given URL
- Inject error capture runtime
- Subscribe to console and network events
- Navigate to URL and collect errors for timeout period (default 30s)
- Aggregate errors by severity (error, warning)
- Generate validation report with success status
- Clean up browser session after validation

Validation Flow:

1. Launch Chrome with CDP
2. Inject runtime script
3. Subscribe to console and network events
4. Navigate to URL
5. Collect errors for configured timeout
6. Classify errors by severity
7. Generate report (success boolean, error list, duration)
8. Close browser

Implementation Files:

- Update src-tauri/src/browser/validator.rs - Use real CDP instead of placeholders

Success Criteria:

- Chrome launches automatically (<2s startup)
- Dev servers start automatically for Next.js/Vite/CRA
- Console errors captured in real-time
- Network errors (404, 500, CORS) captured
- Errors sent to agent orchestrator
- Agent can auto-fix based on browser errors
- Works on macOS, Windows, Linux

---

### Post-MVP Features (Phase 2)

#### 3.1.7.9. Interactive Element Selection

Requirements:

- Enable click-to-select mode in browser preview
- Highlight elements on hover with visual feedback
- Capture element information on click:
  - Unique CSS selector
  - React component name (if applicable)
  - Source code location via source maps
  - Text content preview
  - Bounding box coordinates
  - Screenshot of element
  - Computed styles
- Send selection data to chat interface
- Allow user to reference selected element in conversation

User Flow:

1. User clicks "Select Element" button in Yantra
2. Browser enters selection mode (crosshair cursor)
3. User hovers → element highlights
4. User clicks → element info sent to chat
5. User can type commands referencing selected element
6. Agent knows exact element/file/line for precise changes

Implementation:

- Update runtime JavaScript with selection mode handlers
- Create interaction handler for selection events
- Add "Select Element" button to browser preview UI

#### 3.1.7.10. WebSocket Communication

Requirements:

- Bidirectional Browser ↔ Yantra communication channel
- WebSocket server on random localhost port
- Message types:
  - Browser → Yantra: errors, selections, hover events
  - Yantra → Browser: enable selection mode, highlight element, scroll commands
- Handle connection lifecycle
- Support multiple simultaneous browser connections

Implementation:

- Create WebSocket server module
- Update runtime to connect to WebSocket
- Define message protocol schema

#### 3.1.7.11. Source Map Integration

Requirements:

- Map browser DOM elements to source code files
- React DevTools-style fiber tree traversal
- Extract source locations from React internal properties
- Support source maps for transpiled/bundled code
- Return file name, line number, column number, component name

Benefit: Agent knows exactly which file and line to modify for visual changes

#### 3.1.7.12. Context Menu & Quick Actions

Requirements:

- Right-click menu in browser preview
- Quick actions: Replace, Edit, Remove, Duplicate, Change colors
- Context-aware actions based on element type (image, text, button, etc.)
- Trigger appropriate workflows (asset picker, color picker, text editor)
- Apply changes and refresh preview

#### 3.1.7.13. Visual Feedback Loop

Features:

- Before/After split view for proposed changes
- Visual diff highlighting
- One-click undo for any change
- Change history timeline
- Preview pending changes before application

#### 3.1.7.14. Asset Picker Integration

Features:

- Search Unsplash for free stock photos
- Generate images with DALL-E 3
- Upload local files
- Asset library for recently used images
- Direct integration with element replacement

#### 3.1.7.15 Error Handling & Edge Cases

Chrome Not Found:

- Show user-friendly message: "Downloading browser engine..."
- Download Chromium automatically (~100MB, 30-60 seconds)
- Cache for future use in app data folder
- Fallback: Ask user to install Chrome manually (rare)

Dev Server Fails to Start:

- Check for port conflicts (try next port)
- Check for missing dependencies (suggest npm install)
- Show clear error with fix suggestions
- Allow manual port specification

CDP Connection Fails:

- Retry with exponential backoff (3 attempts)
- Show user-friendly error message
- Degrade gracefully: Skip browser validation, rely on unit tests

Browser Crashes:

- Detect process exit
- Auto-restart browser
- Restore previous state (URL, tabs)
- Log crash for debugging

---

#### 3.1.7.17 Security Considerations

Local-Only Communication:

- WebSocket server binds to 127.0.0.1 (localhost only)
- No external access
- Random port selection (no fixed port conflicts)

Chrome Sandbox:

- Chrome runs in sandboxed mode (default security)
- No filesystem access beyond project folder
- No network access to Yantra's internal APIs

User Privacy:

- No telemetry sent to Yantra servers
- All browser data stays local
- Anonymous crash reports only (opt-in)

---

#### 3.1.7.18 Testing Strategy

Unit Tests:

- Chrome finder logic with mock filesystem
- Port parsing from dev server output
- Error message parsing and classification

Integration Tests:

- Launch Chrome → Connect CDP → Navigate → Capture error
- Start dev server → Detect port → Wait for ready
- Inject runtime → Trigger error → Receive via WebSocket

E2E Tests:

- Full workflow: Generate React app with error → Start server → Capture error → Agent fixes → Validation passes
- Cross-platform testing on macOS, Windows, Linux

Platform Tests:

- macOS: Chrome, Brave, Edge compatibility
- Windows: Chrome, Edge (x64 and ARM)
- Linux: Chrome, Chromium (snap and apt packages)

---

Success Metrics

MVP (Phase 1):

- 95%+ users have working browser preview (zero manual setup)
- 100% of console errors captured
- 90%+ of network errors captured
- <2 second browser startup
- <10 second dev server startup

Post-MVP (Phase 2):

- 80%+ users utilize element selection feature
- 50%+ changes made via visual interaction
- 90%+ accuracy in source map location detection

---

### 3.1.8 Architecture View System (MVP)

Overview

Status: IN PROGRESS (75% Complete - Updated Dec 4, 2025)

Business Impact: Design-first development, architecture governance

What's Working:

- ✅ SQLite storage with full CRUD operations
- ✅ Deviation detection (850 lines) with severity calculation
- ✅ Architecture generator (from intent and code)
- ✅ UI components (ArchitectureCanvas, HierarchicalTabs, ComponentNode, ConnectionEdge)
- ✅ Multi-format import (JSON/MD/Mermaid/PlantUML)
- ✅ Export functionality (agent-callable)
- ✅ GNN integration for code analysis
- ✅ Tauri commands (17 backend APIs)
- ✅ Read-only agent-driven UI principle
- ✅ Impact analysis and auto-correction
- ✅ Refactoring safety analyzer
- ✅ Project initialization with architecture discovery

What's Pending (4 features):

- ❌ Rule of 3 versioning (keep 4 versions, auto-delete oldest)
- ❌ Auto-save on every architecture change
- ❌ Real-time deviation alerts (backend → frontend wiring)
- ❌ Orchestrator integration (ensure blocking works in code generation flow)

A comprehensive architecture visualization and governance system that enables design-first development, automatic architecture generation from existing code, and bidirectional sync between conceptual architecture and implementation.

Key Principle: Architecture is automatically generated and continuously monitored. Code must align with architecture.

Three Core Capabilities:

1. Automatic Architecture Generation - User provides specs/intent → Agent generates architecture → Agent generates code → Agent monitors for deviations
2. Deviation Detection During Implementation - Agent generates code → Checks alignment → Alerts user if deviation → User decides (update arch or fix code)
3. Continuous Alignment Monitoring - Code changes → Compare to architecture → Alert if misaligned → Enforce governance through user decision

#### 3.1.8.1 Agent-Driven Architecture (Autonomous Mode)

CRITICAL DESIGN PRINCIPLE: This is an agentic platform - all architecture operations happen through the agent, not manual UI interactions.

Interaction Model

❌ NOT THIS (Manual Mode):

User clicks "Create Architecture" button

→ User drags components

→ User draws connections

→ User clicks "Save"

→ Manual diagram creation

✅ THIS (Agent-Driven Mode):

User (in chat): "Build a REST API with JWT auth"

→ Agent: Analyzes intent

→ Agent: Generates architecture diagram

→ Agent: Auto-saves to database

→ Architecture View: Shows read-only visualization

→ User: Reviews in Architecture View tab

→ User (in chat): "Add Redis caching layer"

→ Agent: Updates architecture

→ Agent: Auto-saves (keeps last 3 versions)

→ Architecture View: Updates visualization

Auto-Save with Rule of 3 Versioning (Post MVP)

Specification:

Every architecture change is automatically saved with version history following the Rule of 3.

Rule of 3 Implementation:

1. Keep current version + 3 most recent past versions (total: 4 versions)
2. When 5th version is created, delete the oldest (version 1)
3. Versions are immutable once created
4. Agent can revert to any of the 3 past versions

Version Metadata Requirements:

1. Incremental version number (1, 2, 3, 4...)
2. Full architecture state snapshot
3. Timestamp when created
4. Change type classification (AgentGenerated, AgentUpdated, AgentReverted, GNNSynced)
5. Agent's reasoning for the change
6. Original user message that triggered the change

Storage Requirements:

1. Persist only 4 versions (current + 3 past)
2. Auto-delete oldest when creating 5th version
3. Support efficient version retrieval and restoration

Agent Commands (Via Chat):

User: "Show me the architecture"

→ Agent: Opens Architecture View tab, shows current version

User: "Revert to previous architecture"

→ Agent: Loads version N-1, auto-saves as new version N+1

→ Agent: "Reverted to version 5 (from 2 minutes ago)"

User: "Show architecture history"

→ Agent: Lists last 3 versions with timestamps and changes

User: "Why did you add Redis?"

→ Agent: Shows version history and reasoning

---

#### 3.1.8.2 AI Powered Features

3.1.8.2A Architecture Generation from Intent

LLM Prompt Requirements:

- Send user's intent/description to LLM
- Request architecture diagram with components and connections
- Specify required fields: name, type, layer, description for components
- Specify connection fields: type, direction, label
- Request response in structured JSON format

Example Input: "Create a 3-tier web app with React, FastAPI, and PostgreSQL"

Expected Output Format: Structured JSON with:

- Components array containing: id, name, type (ui_component/service/database), layer (frontend/backend/database)
- Connections array containing: from/to component ids, connection type (api_call/data_flow), descriptive label

Post-Processing:

- Parse LLM JSON response
- Validate component and connection data
- Calculate initial positions for visual layout
- Store in database
- Display in Architecture View for user review

  3.1.8.8BArchitecture Generation from Code (GNN-dependency graph Analysis)

Algorithm Requirements:

1. Traverse Dependencies: Use GNN to get complete project dependency graph
2. Group Files into Components:
   - Analyze directory structure to identify logical groupings
   - Example: All files in src/frontend/ become "Frontend UI" component
   - Example: All files in src/auth/ become "Auth Service" component
3. Infer Connections from Imports:

- Analyze import statements to detect component relationships
- Cross-directory imports indicate connections
- Example: from backend.api import indicates Frontend → Backend connection

1. Detect Technology Patterns:

- Identify database components from ORM imports (sqlalchemy, prisma, mongoose)
- Identify cache components from redis/memcached imports
- Identify external services from SDK imports (stripe, twilio, sendgrid)

1. Generate Architecture JSON:

- Create component for each logical grouping
- Create connection for each import relationship
- Assign appropriate types and layers

1. Present for User Refinement:

- Display generated architecture in Architecture View
- Allow user to adjust component names, types, positions
- Allow user to add missing connections or remove incorrect ones

  3.1.8.8B Deviation Detection and Alignment

Detection Algorithm:

1. Trigger: On file save or before code generation
2. Identify Owner: Query which architecture component owns the modified file
3. Get Expected Connections: Retrieve allowed dependencies from architecture definition
4. Get Actual Dependencies: Query GNN for actual imports/function calls in the file
5. Compare and Detect Violations:
   - Calculate Deviation severity (direct violation vs indirect)
   - Generate user-friendly explanation
   - Suggest options (update arch, revert code, refactor)
6. Present alert in UI

Check Code Alignment (Reactive Detection):

Process for checking existing or modified code:

1. Load current architecture for the project
2. Identify which architecture component owns the file being checked
3. Query GNN engine for actual dependencies of the file (imports, function calls, data flows)
4. Retrieve expected dependencies from architecture component definitions
5. Compare actual vs expected dependencies to identify deviations
6. Calculate severity based on deviation type and impact
7. Return alignment result with deviations, severity, and recommendations

Monitor Code Generation (Proactive Detection):

Process for checking code before it's written:

1. Parse generated code to extract imports and dependencies (using tree-sitter)
2. Load architecture and identify target component for the file
3. Retrieve allowed dependencies for that component from architecture
4. Filter imports to find violations (dependencies not in allowed list)
5. If violations found:
   - Calculate severity from violation types
   - Generate user-friendly prompt explaining the deviation
   - Pause code generation
   - Wait for user decision
6. If no violations, allow code generation to proceed

LLM Validation Requirements:

- Send architecture definition and actual code to LLM
- Request explanation of the violation in user-friendly language
- Generate 3 options with trade-offs explained
- Example: "Architecture expects layered access, code shows direct database access"

---

#### 3.1.8.5. Implementation Requirements

Backend Module Organization

Required Modules:

- Main facade/entry point for architecture operations
- SQLite CRUD operations for persistence
- Data types and structures for components and connections
- Versioning system for snapshots and restore
- AI generator for creating architecture from user intent
- Code analyzer for generating architecture from existing code (GNN-based)
- Validator for checking code-architecture alignment
- Exporter for multiple formats (Markdown, Mermaid, JSON)
- Command handlers for frontend-backend communication

Core Data Models Requirements

Component Model:

- Unique identifier
- Name and description
- Component type classification (Service, Module, Layer, Database, External, UIComponent)
- Layer categorization (frontend, backend, database, external, infrastructure)
- Visual position (x/y coordinates)
- List of associated files
- Extensible metadata storage

Connection Model:

- Unique identifier
- Source and target component references
- Connection type classification (DataFlow, ApiCall, Event, Dependency)
- Optional label/description
- Bidirectional flag
- Extensible metadata storage

Backend API Commands

Required Backend Commands:

1. Create component
2. Update component by ID
3. Delete component by ID
4. Create connection between components
5. Retrieve complete architecture
6. Save architecture version with description
7. List all architecture versions
8. Restore specific version by ID
9. Export architecture in specified format (markdown/json/mermaid)
10. Validate code alignment against architecture

Frontend Component Organization

Required UI Components:

- Architecture canvas with graph visualization (React Flow integration)
- Custom node component for rendering architecture components
- Custom edge component for rendering connections with different styles
- Hierarchical tab navigation for filtering by layer
- Component inspector panel for viewing details
- Validation alerts for displaying misalignment warnings

State Management Requirements:

Architecture Store:

- List of all components
- List of all connections
- Currently selected component (if any)
- Current layer filter (complete, frontend, backend, etc.)
- List of alignment issues
- Loading state
- Error state

Store Actions:

- Load architecture from backend
- Create new component
- Update existing component by ID
- Delete component by ID
- Create new connection
- Validate alignment with code
- Export architecture in specified format

#### ---

#### 3.1.8.6. Data Storage Architecture

Primary Storage: SQLite Database

Storage Location:.yantra/architecture.db in user's home directory

Required Data Tables:

1. Components Table:

- Store component metadata (id, name, type, description)
- Track visual positioning (x/y coordinates, width/height)
- Support hierarchical grouping (parent-child relationships)
- Categorize by layer (frontend, backend, database, external, infrastructure)
- Store additional metadata as JSON
- Track creation and update timestamps
- Cascade delete when parent is removed

2. Connections Table:

- Store relationships between components (from/to component ids)
- Define connection types (data_flow, api_call, event, dependency)
- Support labels and descriptions
- Enable bidirectional connections
- Store additional metadata as JSON
- Track creation timestamps
- Cascade delete when components are removed

3. Component Files Table:

- Link files to components (many-to-many relationship)
- Track whether link is auto-generated or manual
- Store confidence score for auto-linked files
- Prevent duplicate file-component links
- Track creation timestamps
- Cascade delete when component is removed

4. Architecture Versions Table:

- Store version history with incremental version numbers
- Capture complete architecture state as JSON snapshot
- Record change descriptions and change types
- Preserve user intent and AI reasoning
- Track which files were changed
- Record timestamps for each version

Data Integrity Requirements:

- Enable SQLite Write-Ahead Logging (WAL) mode for reliability
- Run integrity checks on startup
- Create automatic backup before modifications
- Maintain last 10 backups in .yantra/backups/ directory
- Support foreign key constraints and cascade deletes

Secondary Storage: Git-Friendly Exports

Export Formats:

- architecture.md: Human-readable Markdown with Mermaid diagrams
- architecture.json: Machine-readable complete state for tooling

Export Triggers:

- Automatically after every architecture modification
- On-demand via agent command
- Before git commit (via git hook integration)

Recovery Strategy:

1. Primary: Use SQLite database (fast, transactional)
2. Fallback 1: Restore from architecture.json if database corrupted
3. Fallback 2: Regenerate from GNN code analysis if JSON corrupted
4. User manually reviews and approves regeneration

---

#### 3.1.8.7. Performance Targets

| Operation                                    | Target | Scale Target |
| -------------------------------------------- | ------ | ------------ |
| -------------------------------------------- |        |              |
| Load architecture from DB                    | <50ms  | <100ms       |
| -                                            | -      | -            |
| Render React Flow diagram                    | <200ms | <500ms       |
| Save component/connection                    | <10ms  | <20ms        |
| Generate architecture from intent (LLM)      | <3s    | <5s          |
| Generate architecture from code (GNN)        | <2s    | <5s          |
| Validate alignment                           | <100ms | <300ms       |
| Export to Markdown/JSON                      | <50ms  | <100ms       |
| Version snapshot                             | <20ms  | <50ms        |

---

#### 3.1.8.8. Success Metrics

Technical:

- ✅ Architecture loads and renders in <250ms
- ✅ No data loss (SQLite + JSON backup strategy)
- ✅ 100% of components linked to files
- ✅ Alignment checks complete in <100ms
- ✅ 15/15 features implemented

User Experience:

- ✅ Users can design architecture before coding
- ✅ Users understand legacy codebases via auto-generated architecture
- ✅ Misalignments detected immediately (on save)
- ✅ Users trust architecture as source of truth
- ✅ NPS >50 for architecture feature

Business Impact:

- Prevents spaghetti code through design-first approach
- Reduces onboarding time by 60% (visual architecture)
- Catches architectural violations before they become tech debt
- Differentiator: Yantra enforces architecture; other tools don't

Why This is Revolutionary

| Traditional Tools                                            | Architecture View System     |
| ------------------------------------------------------------ | ---------------------------- |
| ------------------------------------------------------------ |                              |
| Manual diagrams (always outdated)                            | Auto-synced with code        |
| -                                                            | -                            |
| No enforcement                                               | Continuous validation        |
| Static images                                                | Interactive, living diagrams |
| No code linking                                              | Files mapped to components   |
| No version history                                           | Automatic snapshots          |
| No governance                                                | Prevents misalignment        |

Key Differentiators:

- GitHub Copilot/Cursor: Generate code blindly → Result: Spaghetti code
- Yantra: Architecture-first → Code must align → Result: Clean, maintainable systems

---

### 3.1.9 Documentation System (MVP Phase 1 - IMPLEMENTED)

Overview

Status: ✅ Fully Implemented (November 23, 2025)

Purpose: Automatic extraction and structured presentation of project documentation for transparency and user guidance

Location:src-tauri/src/documentation/mod.rs (429 lines), Frontend components

The Documentation System provides a 4-panel UI that automatically extracts and displays structured project information from markdown files, enabling users to understand:

- What features exist (implemented, in-progress, planned)
- Why decisions were made (architecture choices, tradeoffs)
- What changed (file additions, modifications, deletions)
- What tasks remain (current week/phase progress)

This creates transparency between the AI agent and the user, ensuring alignment on project state and next actions.

---

Business Value

For Users:

- Transparency: See exactly what the AI has implemented
- Learning: Understand architectural decisions and rationale
- Control: Track progress and intervene when needed
- Trust: Verify AI is working on the right things

For Development:

- Single Source of Truth: Documentation extracted from markdown files
- Real-time Updates: Reflects current project state
- Context Preservation: Critical for AI agent continuity
- Debugging Aid: Track what changed when issues arise

---

#### 3.1.9.1 Architecture

Data Flow

Markdown Files (SSOT)

    ↓

DocumentationManager.load_from_files()

    ↓

Parse & Extract Structured Data

    ↓

Store in Memory (Vec `<Feature>`, Vec `<Decision>`, Vec `<Change>`, Vec `<Task>`)

    ↓

Tauri Commands (get_features, get_decisions, get_changes, get_tasks)

    ↓

Frontend documentationStore (SolidJS reactive store)

    ↓

DocumentationPanels Component (Tab-based UI)

    ↓

User Interaction → Chat Instructions

Core Components

1. Backend (Rust) - src-tauri/src/documentation/mod.rs

Documentation Data Structure Requirements:

The system must define and maintain the following core data types:

Feature Data:

- Unique identifier (string)
- Title describing the feature
- Detailed description
- Status indicator (Planned, InProgress, or Completed)
- Source attribution showing where the feature was extracted from
- ISO 8601 timestamp of extraction or last update

Decision Data:

- Unique identifier (string)
- Decision title
- Context explaining the situation requiring the decision
- The actual decision made
- Rationale explaining why this decision was chosen
- ISO 8601 timestamp

Change Data:

- Unique identifier (string)
- Change type classification (FileAdded, FileModified, FileDeleted, etc.)
- Human-readable description
- List of affected file paths
- ISO 8601 timestamp

Task Data:

- Unique identifier (string)
- Task title
- Status (Completed, InProgress, or Pending)
- Milestone or phase the task belongs to
- List of task dependencies (task IDs this task depends on)
- Flag indicating if user action is required
- Optional user action instructions (present only if user action required)

DocumentationManager State:

- Workspace directory path
- Collection of all features
- Collection of all decisions
- Collection of all changes
- Collection of all tasks

2. Frontend Store - src-ui/stores/documentationStore.ts

Documentation Store Requirements:

The frontend must maintain a reactive store with:

Store State:

- Array of features
- Array of decisions
- Array of changes
- Array of tasks
- Loading flag indicating data fetch in progress
- Error message (null if no error)

Store Initialization:

- Must initialize with empty arrays for all collections
- Must set loading flag to false
- Must set error to null

Data Loading Function:

When loading documentation, the system must:

Step 1: Set Loading State

- Set loading flag to true to indicate data fetch starting

Step 2: Parallel Data Fetching

- Must invoke backend commands for all four data types in parallel
- Must fetch features using 'get_features' command
- Must fetch decisions using 'get_decisions' command
- Must fetch changes using 'get_changes' command
- Must fetch tasks using 'get_tasks' command
- Must wait for all four fetch operations to complete

Step 3: Success Path

- Update store with all fetched data
- Set loading flag to false
- Clear any previous errors (set to null)

Step 4: Error Path

- If any fetch fails, capture the error message
- Store error message as string
- Set loading flag to false
- Preserve previously loaded data

3. UI Component - src-ui/components/DocumentationPanels.tsx

Documentation Panel UI Requirements:

The system must provide a tabbed interface for viewing documentation:

Tab State Management:

- Must track currently active tab (features, decisions, changes, or tasks)
- Must default to 'features' tab on initial load
- Must allow switching between tabs via button clicks

Lifecycle Management:

- When component mounts, must automatically trigger documentation loading

Tab Navigation UI:

- Must display four tab buttons in a horizontal row
- Each button must show:
  - Tab name (Features, Decisions, Changes, or Plan)
  - Item count in parentheses
  - Active state styling when selected
- Plan tab must show count of pending tasks only (filter out completed)

Tab Content Display:

- Must render content area below tab navigation
- Must display only the content for the currently active tab
- Must conditionally render:
  - FeaturesView component when features tab active
  - DecisionsView component when decisions tab active
  - ChangesView component when changes tab active
  - TasksView component when tasks/plan tab active
- Must pass relevant data array to each view component

---

#### 3.1.9.2 Extraction Algorithms

##### 1. Task Extraction

Pattern Recognition Requirements:

The system must extract tasks from project plan markdown files using the following algorithm:

Initialization:

- Must initialize current milestone to "MVP" as default
- Must initialize task ID counter to 0

Line-by-Line Processing:

For each line in the markdown content:

Step 1: Milestone Detection

- Must check if line contains "Week" or "Phase" keywords
- If detected:
  - Must capture the trimmed line text as the current milestone
  - All subsequent tasks belong to this milestone until next milestone header

Step 2: Task Detection

- Must check if trimmed line starts with "- [" (checkbox syntax)
- If detected, must increment task ID counter

Step 3: Status Determination

- Must examine checkbox contents and emoji indicators
- Must classify as:
  - Completed if line contains "[x]" or "[X]"
  - InProgress if line contains "🔄" emoji or "In Progress" text
  - Pending otherwise (empty checkbox)

Step 4: Title Extraction

- Must split line by "]" character
- Must take the text after the first "]" (index 1)
- Must trim whitespace
- Must remove leading "\*" characters (markdown emphasis)
- Must trim again to get clean title

Step 5: Task Creation

- Must validate title is not empty
- If valid, must create Task with:
  - ID from counter (converted to string)
  - Extracted title
  - Determined status
  - Current milestone
  - Empty dependencies list (populated later if needed)
  - requires_user_action flag set to false
  - user_action_instructions set to None
- Must add created task to tasks collection

Example Input:

## Week 1: Foundation (Nov 26 - Dec 2)

- [x] Set up Tauri + SolidJS project
- [x] Install dependencies
- [ ] 🔄 Implement GNN parser
- [ ] Add LLM integration

Extracted Output:

[

{

    "id": "1",

    "title": "Set up Tauri + SolidJS project",

    "status": "completed",

    "milestone": "Week 1"

},

{ "id": "2", "title": "Install dependencies", "status": "completed", "milestone": "Week 1" },

{ "id": "3", "title": "Implement GNN parser", "status": "in-progress", "milestone": "Week 1" },

{ "id": "4", "title": "Add LLM integration", "status": "pending", "milestone": "Week 1" }

]

##### 2. Feature Extraction

Pattern Recognition Requirements:

The system must extract features from the Features.md file using the following algorithm:

Initialization:

- Must initialize feature ID counter to 0
- Must initialize current description as empty string
- Must initialize in_feature_section flag to false

Line-by-Line Processing:

For each line in the markdown content:

Step 1: Feature Header Detection

- Must check if line starts with "###" (third-level markdown header)
- Must check if line contains status emojis: "✅" (completed) OR "🔄" (in progress) OR "⏳" (planned)
- If both conditions met:
  - Must increment feature ID counter
  - Must set in_feature_section flag to true

Step 2: Status Determination from Emoji

- Must map emoji to status:
  - "✅" → FeatureStatus::Completed
  - "🔄" → FeatureStatus::InProgress
  - "⏳" → FeatureStatus::Planned

Step 3: Title Extraction

- Must remove leading "#" characters and trim whitespace
- Must remove all status emojis ("✅", "🔄", "⏳")
- Must trim again to get clean title

Step 4: Feature Creation

- Must create new Feature with:
  - ID from counter (converted to string)
  - Extracted title
  - Empty description (populated in next steps)
  - Determined status
  - extracted_from set to "Features.md"
  - timestamp set to current UTC time in RFC3339 format
- Must add feature to features collection

Step 5: Description Capture

- If in_feature_section is true AND line is not empty AND line doesn't start with "##":
  - Must get reference to most recently added feature
  - Must append current line to feature's description
  - Must append newline character to maintain formatting

Step 6: Section End Detection

- If line starts with "##" (second-level header):
  - Must set in_feature_section to false
  - Indicates moving to next major section

Example Input:

### ✅ Dependency Graph (GNN)

Track all code dependencies using Graph Neural Networks.

Detects breaking changes automatically.

### 🔄 LLM Integration

Multi-provider support with failover.

Currently implementing OpenAI client.

### ⏳ Browser Validation

Automated UI testing in Chrome.

Planned for Week 3.

Extracted Output:

[

{

    "id": "1",

    "title": "Dependency Graph (GNN)",

    "description": "Track all code dependencies using Graph Neural Networks.\nDetects breaking changes automatically.",

    "status": "completed"

},

{

    "id": "2",

    "title": "LLM Integration",

    "description": "Multi-provider support with failover.\nCurrently implementing OpenAI client.",

    "status": "in-progress"

}

]

##### 3. Decision Extraction

Pattern Recognition Requirements:

The system must extract decisions from the Decision_Log.md file using the following algorithm:

Initialization:

- Must initialize decision ID counter to 0
- Must initialize current_decision as None (no active decision)
- Must initialize section_type as None (not in any subsection)

Line-by-Line Processing:

For each line in the markdown content:

Step 1: Decision Header Detection

- Must check if line starts with "##" (second-level markdown header)
- Must check if line does NOT contain "Decision Log" (skip the main title)
- If both conditions met:
  - Must save the previous decision (if any exists) to decisions collection
  - Must increment decision ID counter
  - Must extract title by removing "#" and trimming whitespace
  - Must create new Decision with:
    - ID from counter (converted to string)
    - Extracted title
    - Empty context, decision, and rationale (populated later)
    - timestamp set to current UTC time in RFC3339 format
  - Must set this as the current_decision

Step 2: Subsection Type Detection

- Must check for subsection markers:
  - If line starts with "Context:" or contains "Context:":
    - Set section_type to "context"

  - If line starts with "Decision:" or contains "Decision:":
  - Set section_type to "decision"
  - If line starts with "Rationale:" or contains "Rationale:":
  - Set section_type to "rationale"

Step 3: Content Capture

- If current_decision exists AND line is not empty AND line doesn't start with "\*\*":
  - Must append line to appropriate section based on section_type:
    - If section_type is "context": append to context field
    - If section_type is "decision": append to decision field
    - If section_type is "rationale": append to rationale field
  - Must append newline character after each line

Step 4: Final Decision Save

- After processing all lines, must check if current_decision exists
- If exists, must add it to decisions collection (captures last decision in file)

Example Input:

## Use SQLite for GNN Persistence

**Context:**

Need persistent storage for dependency graph between sessions.

**Decision:**

Use SQLite with schema: nodes (id, type, name, file_path), edges (from_id, to_id, edge_type).

**Rationale:**

- Zero-config (no separate database server)
- Fast queries (<10ms for typical graphs)
- ACID transactions for consistency
- Works offline

Extracted Output:

{

"id": "1",

"title": "Use SQLite for GNN Persistence",

"context": "Need persistent storage for dependency graph between sessions.",

"decision": "Use SQLite with schema: nodes (id, type, name, file_path), edges (from_id, to_id, edge_type).",

"rationale": "- Zero-config\n- Fast queries\n- ACID transactions\n- Works offline"

}

---

#### 3.1.9.3 Tauri Commands

Backend API Requirements - src-tauri/src/main.rs:

The system must expose the following Tauri commands for frontend-to-backend communication:

get_features Command:

- Must accept AppState as parameter containing the DocumentationManager
- Must acquire lock on the documentation manager (thread-safe access)
- Must retrieve the features collection from the manager
- Must return a cloned vector of all features
- Must return error string if operation fails

get_decisions Command:

- Must accept AppState as parameter
- Must acquire lock on the documentation manager
- Must retrieve the decisions collection
- Must return a cloned vector of all decisions
- Must return error string if operation fails

get_changes Command:

- Must accept AppState as parameter
- Must acquire lock on the documentation manager
- Must retrieve the changes collection
- Must return a cloned vector of all changes
- Must return error string if operation fails

get_tasks Command:

- Must accept AppState as parameter
- Must acquire lock on the documentation manager
- Must retrieve the tasks collection
- Must return a cloned vector of all tasks
- Must return error string if operation fails

reload_documentation Command:

- Must accept AppState as parameter
- Must acquire mutable lock on the documentation manager
- Must call load_from_files() to reload all documentation from disk
- Must return unit type (empty success) or error string

---

#### 3.1.9.4 UI Design

##### 4-Tab Layout

┌─────────────────────────────────────────────┐

│ [Features 15] [Decisions 8] [Changes 23] [Plan 42] \*\* \*\* \*\* \*\*│

├─────────────────────────────────────────────┤

│ \*\* \*\* \*\* \*\* \*\* \*\* │

│ ✅ Dependency Graph (GNN) \*\* \*\* \*\* \*\* │

│ Track all code dependencies using Graph \*\* \*\* \*\* \*\* │

│ Neural Networks. Detects breaking changes.\*\* \*\* \*\* \*\* │

│ Files: src-tauri/src/gnn/\*.rs \*\* \*\* \*\* \*\* \*\* \*\* │

│ Status: Completed (Nov 20, 2025) \*\* \*\* \*\* \*\* │

│ \*\* \*\* \*\* \*\* \*\* \*\* │

│ 🔄 LLM Integration \*\* \*\* \*\* \*\* \*\* \*\* │

│ Multi-provider support with failover. \*\* \*\* \*\* \*\* │

│ Files: src-tauri/src/llm/\*.rs \*\* \*\* \*\* \*\* \*\* \*\* │

│ Status: In Progress (85% complete) \*\* \*\* \*\* \*\* │

│ \*\* \*\* \*\* \*\* \*\* \*\* │

│ ⏳ Browser Validation \*\* \*\* \*\* \*\* \*\* \*\* │

│ Automated UI testing in Chrome. \*\* \*\* \*\* \*\* │

│ Planned for Week 3. \*\* \*\* \*\* \*\* \*\* \*\* │

│ \*\* \*\* \*\* \*\* \*\* \*\* │

└──────────────────────────────────────────────┘

##### Visual Indicators

Feature Status:

- ✅ Completed (green): Fully implemented and tested
- 🔄 In Progress (yellow): Actively being worked on
- ⏳ Planned (blue): Not yet started

Task Status:

- ![checked]()Completed (green checkmark)
- ![unchecked]()🔄 In Progress (yellow with spinner emoji)
- ![unchecked]()Pending (empty checkbox)

Change Types:

- 📄 FileAdded (green)
- ✏️ FileModified (yellow)
- 🗑️ FileDeleted (red)
- ➕ FunctionAdded (green)
- ➖ FunctionRemoved (red)

  3.1.9.5 Features Tab - Detailed Requirements

Purpose: Automatically extract and track features from multiple sources with accurate completion tracking.

Feature Extraction Sources:

1. Documentation Files (MVP):

- Markdown files (README.md, docs/\*.md)
- Project documentation
- Technical specifications
- Parser: CommonMark/GFM with heading detection

2. Chat Conversations (MVP):

- User requests: "Add user authentication"
- Natural language requirements
- Feature discussions
- NLP: LLM-based intent extraction

3. Code Files (MVP):

- Existing implementations
- Code comments and docstrings
- Function/class definitions
- Parser: Tree-sitter with semantic analysis

4. External Tools:

- Notion (MVP): Extract features from Notion pages via Notion API
- Confluence (Post-MVP): Extract features from Confluence spaces
- Linear (Post-MVP): Import issues and feature requests

Completion Tracking (Accurate & Automated):

Agent tracks feature completion by monitoring:

- Code generation: Files created/modified
- Test pass rates: Unit/integration tests passing
- Integration status: Dependencies satisfied
- Deployment status: Code deployed to target environment

Features automatically move through status:

- ⏳ Planned (0% complete) → 🔄 In Progress (1-99% complete) → ✅ Done (100% complete)

Completion Calculation:

pub fn calculate_feature_completion(feature: &Feature, gnn: &GNNEngine) -> f32 {

    let total_tasks = feature.required_files.len() + feature.required_tests.len();

    let completed_files = feature.required_files.iter()

    .filter(|f| gnn.file_exists(f) && gnn.file_has_no_errors(f))

    .count();

    let completed_tests = feature.required_tests.iter()

    .filter(|t| gnn.test_exists(t) && gnn.test_passes(t))

    .count();

((completed_files + completed_tests) as f32 / total_tasks as f32) \* 100.0

}

UI Display:

┌─────────────────────────────────────────────┐

│ 📋 Add User Authentication ✅ Done │

│ JWT tokens with bcrypt password hash │

│ Extracted from: Chat conversation │

│ Completion: 100% (Tests passing) │

│ Files: auth.py, test_auth.py (2/2) │

└─────────────────────────────────────────────┘

---

#### 3.1.9.6 Changes Tab - Detailed Requirements

Purpose: Provide a complete audit trail of all code changes committed during the project with file-level details and synchronization across team members.

Change Tracking Sources:

- Git commits made by Agent
- File additions, modifications, and deletions
- Code refactoring operations
- Dependency updates (package.json, requirements.txt, Cargo.toml)
- Configuration file changes
- Test file changes

MVP: Complete Change Log View

For MVP, the Changes tab serves as the Complete Change Audit Trail:

- All committed changes tracked with timestamps
- File-level granularity (what files were touched)
- Change descriptions auto-generated by Agent
- Commit hashes for full traceability
- Real-time updates across all team members
- Integration with Git history

Change Logging Workflow:

1. Agent completes implementation of a task/feature
2. Agent stages changes using Git
3. Agent generates descriptive commit message
4. Agent commits changes to repository
5. Change logged in Changes tab automatically
6. All team members see new change instantly via WebSocket
7. Change entry includes: commit hash, files changed, description, timestamp

Data Model:

pub struct ChangeLog {

    pub id: Uuid,

    pub project_id: Uuid,

    pub commit_hash: String,           // Git commit SHA

    pub branch: String,                // Git branch name

    pub description: String,           // What was changed and why

    pub files_added: Vec`<String>`,      // New files created

    pub files_modified: Vec`<String>`,   // Existing files changed

    pub files_deleted: Vec`<String>`,    // Files removed

    pub author: String,                // Agent ID or User ID

    pub timestamp: DateTime`<Utc>`,

    pub related_task_id: Option`<Uuid>`, // Link to task in Plan

    pub related_feature_id: Option`<Uuid>`, // Link to feature

    pub lines_added: usize,            // Stats for dashboard

    pub lines_deleted: usize,

    pub impact_level: ChangeImpact,    // Minor | Medium | Major

}

pub enum ChangeImpact {

    Minor,      // Small bug fixes, formatting, comments

    Medium,     // Feature additions, refactoring

    Major,      // Architecture changes, breaking changes

}

UI Display:

┌─────────────────────────────────────────────┐

│ Implement User Authentication System │

│ 🤖 Agent • Nov 28, 2025 2:45 PM │

│ Branch: feature/auth • Commit: a3f9c2e │

│ │

│ Files Added (3): │

│ • src/auth/jwt.rs │

│ • src/auth/password.rs │

│ • tests/auth_test.rs │

│ │

│ Files Modified (2): │

│ • src/main.rs (+45, -12) │

│ • Cargo.toml (+2, -0) │

│ │

│ Description: │

│ Implemented JWT-based authentication with │

│ bcrypt password hashing. Added middleware │

│ for protected routes. Includes unit tests │

│ with 95% coverage. │

│ │

│ Related: Task #12 (User Authentication) │

│ Impact: Major 🔴 │

└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐

│ Fix login endpoint CORS headers │

│ 🤖 Agent • Nov 28, 2025 1:15 PM │

│ Branch: main • Commit: b7e4d1a │

│ │

│ Files Modified (1): │

│ • src/api/auth.rs (+8, -3) │

│ │

│ Description: │

│ Added CORS headers to login endpoint to │

│ allow requests from frontend. Updated error │

│ handling for invalid credentials. │

│ │

│ Related: Bug Fix │

│ Impact: Minor 🟢 │

└─────────────────────────────────────────────┘

Filter and Search:

- By Date Range: "Last 7 days", "Last 30 days", "Custom range"
- By Impact Level: Minor | Medium | Major
- By File Type: .rs | .ts | .py | .toml | etc.
- By Author: Agent ID or User ID
- By Branch: main | feature/_ | bugfix/_
- Search: Full-text search in descriptions and file names

Integration with Git:

// src-tauri/src/documentation_panels/changes_tracker.rs

pub async fn track_commit(

    commit_hash: &str,

    branch: &str,

    files: Vec`<GitFileChange>`,

    message: &str,

) -> Result `<ChangeLog>` {

    // Extract file statistics

    let files_added = files.iter()

    .filter(|f| f.status == ChangeType::Added)

    .map(|f| f.path.clone())

    .collect();

    let files_modified = files.iter()

    .filter(|f| f.status == ChangeType::Modified)

    .map(|f| f.path.clone())

    .collect();

    let files_deleted = files.iter()

    .filter(|f| f.status == ChangeType::Deleted)

    .map(|f| f.path.clone())

    .collect();

    // Calculate impact level

    let impact = calculate_impact(&files, &message);

    // Create change log entry

    let change_log = ChangeLog {

    id: Uuid::new_v4(),

    commit_hash: commit_hash.to_string(),

    branch: branch.to_string(),

    description: message.to_string(),

    files_added,

    files_modified,

    files_deleted,

    author: "agent-1".to_string(),

    timestamp: Utc::now(),

    impact_level: impact,

    ..Default::default()

    };

    // Store in database

    store_change_log(&change_log).await?;

    // Broadcast to all connected clients via WebSocket

    broadcast_change_update(&change_log).await?;

    Ok(change_log)

}

fn calculate_impact(files: &[GitFileChange], message: &str) -> ChangeImpact {

    // Major: Architecture changes, many files, breaking changes

    if files.len() > 10 || message.contains("BREAKING") {

    return ChangeImpact::Major;

    }

    // Medium: Feature additions, multiple files

    if files.len() > 3 {

    return ChangeImpact::Medium;

    }

    // Minor: Small fixes, single file

    ChangeImpact::Minor

}

Real-Time Synchronization:

- Changes tab updates automatically when Agent commits
- WebSocket notifications push changes to all team members
- No manual refresh needed
- Optimistic UI updates with rollback on conflict

Storage:

- Database: SQLite table change_logs (Tier 3 - Read-Heavy Reference Data)
- Columns: id, project_id, commit_hash, branch, description, files_added, files_modified, files_deleted, author, timestamp, related_task_id, related_feature_id, lines_added, lines_deleted, impact_level
- Indexes: timestamp DESC (for recent changes), project_id (for filtering), commit_hash (for deduplication)

Export Capabilities:

- Export to CSV: Full change log with all details
- Export to Markdown: Formatted change log for documentation
- Export date range: Custom date filtering
- Export by author: Filter by specific agent/user

Empty State:

┌─────────────────────────────────────────────┐

│ 📝 No Changes Yet │

│ │

│ Changes will appear here automatically │

│ as Agent commits code to the repository. │

│ │

│ All committed changes are tracked with: │

│ • File-level details │

│ • Timestamps │

│ • Commit hashes │

│ • Change descriptions │

└─────────────────────────────────────────────┘

Performance Targets:

- Load last 100 changes: <100ms
- Real-time update latency: <500ms
- Search across 1000+ changes: <200ms
- Database query optimization: Indexed on timestamp and project_id

#### 3.1.9.7 Decisions Tab - Detailed Requirements

Purpose: Document all project decisions with full context to serve as an approval audit trail.

Decision Sources:

- Chat conversations with user
- Technical choices made during implementation
- Architecture decisions
- Technology selections (database, framework, libraries)
- Design pattern choices
- Trade-offs and alternatives considered

MVP: Approval Audit View

For MVP, the Decisions tab serves as the Approver Audit View:

- All critical decisions logged with timestamps
- Full context showing why decision was made
- User approval/confirmation captured
- Alternatives considered documented
- Complete audit trail for compliance/governance

Decision Logging Workflow:

1. Agent encounters decision point during implementation
2. Agent analyzes options using LLM + GNN context
3. Agent proposes recommendation with full context
4. Decision logged in Decisions tab (status: ⏳ Pending Approval)
5. User reviews in chat and approves/modifies
6. Approval captured with timestamp (status: ✅ Approved)
7. All team members see updated decision instantly via WebSocket

UI Display:

┌─────────────────────────────────────────────┐

│ Use PostgreSQL over MySQL │

│ 👤 User Approved ✅ │

│ │

│ Context: │

│ Need JSONB support for flexible schema │

│ │

│ Decision: │

│ PostgreSQL 14+ with JSONB columns │

│ │

│ Alternatives Considered: │

│ • MySQL 8.0 - Limited JSON support │

│ • MongoDB - No ACID guarantees │

│ │

│ Rationale: │

│ Better JSON performance, native support, │

│ full ACID compliance for critical data │

│ │

│ Nov 28, 2025 10:30 AM - Agent proposed │

│ Nov 28, 2025 10:32 AM - User approved │

└─────────────────────────────────────────────┘

---

#### 3.1.9.8 Plan Tab - Detailed Requirements

Purpose: Create and maintain a persistent project-level plan that Agent methodically executes and tracks.

Plan Creation Process:

1. Agent analyzes requirements from chat/docs/code
2. Agent breaks down into tasks with dependencies
3. Agent organizes tasks by milestones
4. Agent proposes plan to user for confirmation
5. User confirms/modifies milestones and prioritization
6. Agent begins execution only after user approval

Plan Persistence:

- Project-level plan stored in SQLite database
- Plan survives application restarts
- Task status preserved and updated continuously
- Historical task data maintained for audit trail
- Milestone completion tracked over time

Sub-Task Tracking:

Agent can dynamically add sub-tasks to track:

- Implementation steps (e.g., "Create database schema")
- Testing requirements (e.g., "Write unit tests for auth")
- Code review checkpoints (e.g., "Review security implementation")
- Deployment stages (e.g., "Deploy to staging")
- Documentation updates (e.g., "Update API docs")

---

#### 3.1.9.5 Integration with Chat

User Actions from Documentation Panel:

1. Click feature → Inserts into chat: "Tell me more about [Feature Name]"
2. Click pending task → Inserts: "Work on: [Task Title]"
3. Click decision → Shows context and rationale in chat
4. Click change → Shows diff and affected files

Example Flow:

User: [Clicks "⏳ Browser Validation"]

Chat: "I see you're interested in Browser Validation. This feature is planned

    for Week 3 and will enable automated UI testing in Chrome using the

    Chrome DevTools Protocol. Should I start implementing it now?"

User: "Yes, start implementing"

Agent: [Begins implementation, adds to Changes tab in real-time]

---

#### 3.1.9.6 Performance Targets

| Operation                                    | Target        | Actual    |
| -------------------------------------------- | ------------- | --------- |
| -------------------------------------------- |               |           |
| Load all documentation                       | <100ms        | ~50ms ✅  |
| -                                            | -             | -         |
| Parse Project_Plan.md                        | <10ms         | ~5ms ✅   |
| Parse Features.md                            | <10ms         | ~3ms ✅   |
| Parse Decision_Log.md                        | <10ms         | ~4ms ✅   |
| UI tab switch                                | <16ms (60fps) | ~8ms ✅   |
| Reload from disk                             | <200ms        | ~100ms ✅ |

---

#### 3.1.9.7 Source Files

| File                                                                                           | Lines | Purpose                                   | Tests  |
| ---------------------------------------------------------------------------------------------- | ----- | ----------------------------------------- | ------ |
| ---------------------------------------------------------------------------------------------- |       |                                           |        |
| src-tauri/src/documentation/mod.rs                                                             | 429   | Core backend logic, extraction algorithms | 4/4 ✅ |
| -                                                                                              | -     | -                                         | -      |
| src-ui/stores/documentationStore.ts                                                            | 198   | Reactive state management                 | N/A    |
| src-ui/components/DocumentationPanels.tsx                                                      | 248   | UI component with tabs                    | N/A    |

Total: 875 lines across 3 files

---

### 3.1.10 Storage Optimizations

#### 1. Incremental Updates

Performance Targets:

- Incremental update: <50ms per file change
- Full graph build: <5s for 10K LOC
- Dependency lookup: <10ms

Strategy:

- Only reparse changed file + update edges
- No full graph rebuild on file changes
- Use tree-sitter for incremental parsing
- petgraph BFS traversal for fast lookups

---

#### 2. Context Compression (Token Optimization)

Purpose: Fit maximum relevant context within LLM token limits

A. Syntax-Aware Compression:

- Remove comments (unless directly relevant)
- Strip docstrings (keep in metadata)
- Minimize whitespace
- Remove unused imports in context
- Savings: 20-30% tokens

B. Semantic Chunking:

- Split large files into logical chunks (classes, functions)
- Include only relevant chunks
- Track chunk relationships in Dependency Graph
- Savings: 40-60% tokens for large files

C. De-duplication:

- Identical code blocks referenced multiple times
- Common utility functions
- Shared type definitions
- Savings: 10-15% tokens

D. Hierarchical Context Assembly:

Total Potential Savings: 50-70% tokens with same/better context quality

---

#### 3. HNSW Vector Indexing (High-Performance Semantic Search)

Technology: Hierarchical Navigable Small World (HNSW) indexing

Database/Storage: In-memory petgraph CodeGraph structure (NOT a separate database)

Performance Requirements:

- 1,000 nodes: <0.1ms (meets <10ms target)
- 10,000 nodes: <2ms (meets <10ms target)
- 100,000 nodes: <5ms (meets <10ms target)

vs Linear Scan (Inadequate):

- 10,000 nodes: 50ms (5x over target - FAILS)
- 100,000 nodes: 500ms (50x over target - FAILS)

Configuration:

- M parameter: 16 (connections per node)
- max_elements: 10,000 (initial capacity)
- ef_construction: 16 (build quality)
- ef_search: 200 (query accuracy)
- Distance metric: Cosine distance

Characteristics:

- Complexity: O(log n) query time vs O(n) linear scan
- Accuracy: 99.5%+ recall
- Memory Overhead: +30-50% on embedding size (~1.5-2MB for 10k nodes)
- Index Build: O(n log n), ~1s for 10k nodes, 10s for 100k nodes
- Incremental: Supports insert/delete for graph updates

---

#### 4. Database Optimization (SQLite)

WAL Mode (Write-Ahead Logging):

- Phase 1 (NOW): Enable WAL mode immediately for Tier 3 - Read-Heavy Reference Data
- Benefit: Concurrent reads during writes
- Access Pattern: Perfect for 100:1 read:write ratio

Connection Pooling:

- Technology: r2d2 connection pooling
- Benefit: Reuse connections, reduce overhead
- Phase 1 (NOW): Implement this week

For Tier 3 (Reference Data):

- SQLite with WAL + r2d2 pooling is optimal
- No need for PostgreSQL migration
- Handles read-heavy workload perfectly

---

#### 5. Caching Strategy

Context Assembly Cache:

- Storage: SQLite
- Key: Hash of (file content + dependencies)
- TTL: 24 hours
- Invalidation: On file changes
- Performance Gain: <50ms retrieval vs 100-500ms assembly

Web Search Results Cache:

- Storage: SQLite
- TTL: 7 days for package info
- Purpose: Avoid redundant web searches for package docs
- Use Case: Dependency assessment, package compatibility

Dependency Package Cache:

- Location: ~/.yantra/cache/pip (local cache)
- Purpose: Faster installs, less bandwidth
- Benefit: Reuse downloaded packages across projects

Virtual Environment Cache:

- Location: ~/.yantra/venv-cache (global cache)
- Purpose: Shared resources across projects
- Use Case: Dry-run validations, temporary environments

Semantic Graph Cache:

- Purpose: Cache package documentation from web searches
- TTL: 7 days
- Technology: HNSW index for fast retrieval
- Storage: In Tier 3 SQLite

LRU Cache (Tier 4):

- Technology: moka crate (in-memory)
- Use Cases: Frequently accessed embeddings, token counts, LLM responses
- Budget: 500MB
- Eviction: Automatic LRU

---

#### 6. Snapshot Persistence Strategy

For Tier 1 (In-Memory Dependency Graph):

- Technology: bincode snapshots
- Frequency: On significant changes
- Format: Binary serialization (compact, fast)
- Write-Ahead Log: Track incremental changes between snapshots
- Recovery: Load last snapshot + replay WAL

Benefits:

- Fast startup (load snapshot vs rebuild)
- Crash recovery without losing work
- Minimal storage overhead (binary format)

---

### 3.1.11 Security Infrastructure

Security Infrastructure Requirements from Specifications Based on the Yantra specifications, here are the comprehensive security infrastructure requirements:

1. Security Scanning Core (SAST - Static Application Security Testing) Implementation: security/scanner.rs (512 lines) - Builtin Tool

Tools & Technologies:

Semgrep with OWASP ruleset (primary SAST tool) TruffleHog patterns for secret detection Safety (Python dependency vulnerabilities) npm audit (JavaScript dependency vulnerabilities) Gitleaks (secrets scanning) Bandit (Python security linting) Capabilities:

security_scan - SAST analysis secrets_detect - Find exposed credentials dependency_audit - CVE check on packages pkg_audit - Security vulnerability check

2. Security Validation Layers

Layer 4: Pre-Commit Security (Part of 4-Layer Validation)

Semgrep with OWASP rules Secret scanning (TruffleHog patterns) SQL injection detection XSS vulnerability detection Insecure deserialization checks Hardcoded secrets detection Performance Target: <10 seconds for security scan

3. Vulnerability Detection Types Code Vulnerabilities:

SQL injection Cross-Site Scripting (XSS) Secrets in code Insecure patterns Shell injection attempts Dependency Vulnerabilities:

CVE database checks for all dependencies Outdated package detection License conflicts Version compatibility issues Secret Detection:

API keys, tokens, passwords in code Hardcoded credentials Environment variable leaks

4. Auto-Fix Capabilities Standard Vulnerability Fixes:

SQL injection → Use parameterized queries XSS → Escape user input Pattern-based fixes for common vulnerabilities Automatic re-scan after fix Fix Strategy:

Attempt 1: Primary LLM applies fix with error context Attempt 2: Retry with accumulated context Attempt 3: Multi-LLM consultation for second opinion Escalation: Manual intervention after 3 attempts

5. Security Agent (Multi-Agent Architecture) Role: Vulnerability scanning and auto-fixing (runs in parallel)

Responsibilities:

Continuous security scanning Vulnerability detection Auto-fix generation Security validation

6. Command Execution Security Whitelist-Based Validation:

Allowed Commands:

Python: python, python3, pip, pytest, black, flake8 Node: node, npm, npx, yarn, jest Rust: cargo (build, test, run) Docker: docker (build, run, ps, stop) Git: via MCP protocol for security Cloud: aws, gcloud, kubectl, terraform, heroku Blocked Patterns:

rm -rf, sudo, su, chmod +x eval, exec, source Shell redirects to system files: > /etc/\* Network commands: curl | bash, wget | sh Shell injection: ;, |, & in arguments Security Validations:

No arbitrary command execution Argument validation (block shell injection) Path restrictions (workspace directory only) No privilege escalation Network safety Resource limits (timeout: 5 min, memory: <2GB) Audit logging (all commands logged to SQLite)

7. Data Security & Privacy Local-Only Architecture:

User code never leaves machine (unless explicitly sent to LLM APIs) LLM calls encrypted in transit (HTTPS) No code storage on Yantra servers Anonymous crash reports (opt-in) Usage analytics only, no PII (opt-in) WebSocket Security:

Binds to 127.0.0.1 (localhost only) No external access Random port selection Chrome Sandbox:

Sandboxed mode (default security) No filesystem access beyond project folder No network access to Yantra's internal APIs

8. Parallel Security Scanning Performance Optimization:

4 parallel workers for Semgrep rules Parallel file scanning Concurrent security rules execution (Semgrep, secrets, CVE) Multiple dependency CVE lookups simultaneously Performance: N files scanned in ~O(log N) time with parallel workers

9. Security Integration Points CodeGen Phase:

Pre-commit security validation Automatic security scan before Git commit Zero critical vulnerabilities required Testing Phase:

Security scan as part of validation pipeline Quality gate: Security scan must be clean Auto-fix for critical vulnerabilities Dependency Assessment:

CVE database checks Version vulnerability scanning License compliance checks Compatibility security analysis

10 Security Metrics & Targets Success Criteria:

<3% critical security vulnerabilities (auto-fixed) 95%+ generated code passes security scan <10s security scan time Zero breaking security changes Quality Gates:

All critical vulnerabilities must be fixed No secrets in committed code All dependencies checked for CVEs Security scan clean before commit

---

## 3.2. AGENTIC FRAMEWORK

### Philosophy: The Four Pillars of Autonomous Development

Yantra's agentic capabilities are organized into four fundamental pillars that mirror human developer capabilities:

🔍 PERCEIVE → 🧠 REASON → ⚡ ACT → 🔄 LEARN

Sense the environment -> Analyze and Decide -> Execute Action ->Adapt from Feedback

## 3.3 AGENTIC PRIMITIVES/TOOLS

Yantra Unified Tool Interface (UTI) abstracts away the underlying protocol differences (LSP, MCP, DAP, built-in) and presents a single, consistent API for the AI agent to discover, invoke, and manage tools.

## Protocol Selection Framework

For each capability, determine the protocol based on:

| Question                                                                  | If YES →                      |
| ------------------------------------------------------------------------- | ----------------------------- |
| ------------------------------------------------------------------------- |                               |
| Does the editor need it in real-time while typing?                        | LSP (Editor only)             |
| -                                                                         | -                             |
| Is it a core differentiator we must control?                              | Builtin                       |
| Is it a discrete query the agent makes?                                   | MCP                           |
| Does it need streaming output for progress?                               | Builtin or MCP with streaming |
| Is there a well-maintained community server?                              | MCP                           |

### 3.3.1. 🔍 PERCEIVE - Input & Sensing Layer

#### 3.3.1.1 File System Operations

Protocol:Builtin (Primary) | MCP (Fallback via @modelcontextprotocol/server-filesystem) Rationale: File operations are core to everything. Must be fast, reliable, and under our control.

| Capability                                                                                         | Tool/Terminal | Priority | Purpose                                                | Implementation                  |
| -------------------------------------------------------------------------------------------------- | ------------- | -------- | ------------------------------------------------------ | ------------------------------- |
| -------------------------------------------------------------------------------------------------- |               |          |                                                        |                                 |
| file_read                                                                                          | Tool          | P0       | Read file contents with encoding detection             | main.rs::read_file()[Builtin]   |
| -                                                                                                  | -             | -        | -                                                      | -                               |
| file_write                                                                                         | Tool          | P0       | Create/overwrite files                                 | main.rs::write_file()[Builtin]  |
| file_edit                                                                                          | Tool          | P0       | Surgical edits (line range, search-replace, AST-based) | [Builtin]                       |
| file_delete                                                                                        | Tool          | P0       | Remove files safely                                    | [Builtin]                       |
| file_move                                                                                          | Tool          | P0       | Rename/move files with dependency updates              | [Builtin]                       |
| file_copy                                                                                          | Tool          | P0       | Duplicate files                                        | [Builtin]                       |
| directory_create                                                                                   | Tool          | P0       | Create directories recursively                         | [Builtin]                       |
| directory_list                                                                                     | Tool          | P0       | List contents with filters                             | main.rs::read_dir()[Builtin]    |
| directory_tree                                                                                     | Tool          | P0       | Get full project structure                             | [Builtin]                       |
| file_search                                                                                        | Tool          | P0       | Find files by name/pattern/glob                        | [Builtin]                       |
| file_watch                                                                                         | Tool          | P2       | Monitor for changes (reactive agents)                  | [Builtin](usenotifycrate)       |
| docx_read                                                                                          | Tool          | P1       | Read Word documents                                    | [Builtin](usedocx-rs)           |
| pdf_read                                                                                           | Tool          | P1       | Extract text from PDFs                                 | [Builtin](usepdf-extract/lopdf) |

#### 3.3.1.2 Code Intelligence (Tree-sitter Powered)

Protocol:Builtin (Primary via Tree-sitter) | MCP (Secondary via Pylance/rust-analyzer) | LSP (Editor-only, not exposed to Agent) Rationale: Tree-sitter is core differentiator. Fast, multi-language, works offline. MCP for advanced features when available.

| Capability                                                                             | Tool/Terminal | Priority | Purpose                                        | Implementation                           |
| -------------------------------------------------------------------------------------- | ------------- | -------- | ---------------------------------------------- | ---------------------------------------- |
| -------------------------------------------------------------------------------------- |               |          |                                                |                                          |
| parse_ast                                                                              | Tool          | P0       | Get AST for file/snippet                       | gnn/parser.rs(tree-sitter)[Builtin]      |
| -                                                                                      | -             | -        | -                                              | -                                        |
| get_symbols                                                                            | Tool          | P0       | Extract functions, classes, variables, imports | gnn/parser.rs[Builtin]                   |
| get_references                                                                         | Tool          | P2       | Find all usages of a symbol                    | [MCP/Builtin fallback]                   |
| get_definition                                                                         | Tool          | P2       | Jump to definition                             | [MCP/Builtin fallback]                   |
| get_scope                                                                              | Tool          | P2       | Get scope context for a position               | [Builtin]                                |
| get_diagnostics                                                                        | Tool          | P0       | Syntax errors, warnings                        | Integrated in parser**[Builtin]**        |
| semantic_search                                                                        | Tool          | P1       | Search code by meaning                         | GNN semantic layer (embeddings)[Builtin] |
| get_call_hierarchy                                                                     | Tool          | P1       | Incoming/outgoing calls                        | GNN dependency tracking**[Builtin]**     |
| get_type_hierarchy                                                                     | Tool          | P2       | Class inheritance chains                       | [MCP/Builtin fallback]                   |

#### 3.3.1.3 Dependency Graph & Impact Analysis

Protocol:Builtin (Exclusive - Core Differentiator) Rationale: GNN-powered analysis is Yantra's core differentiator. petgraph data structure (not a neural network, historical naming).

| Capability                                                                                                | Tool/Terminal | Priority | Purpose                           | Implementation                              |
| --------------------------------------------------------------------------------------------------------- | ------------- | -------- | --------------------------------- | ------------------------------------------- |
| --------------------------------------------------------------------------------------------------------- |               |          |                                   |                                             |
| build_dependency_graph                                                                                    | Tool          | P0       | Generate full project graph       | gnn/engine.rs(10/10 features)[Builtin]      |
| -                                                                                                         | -             | -        | -                                 | -                                           |
| get_dependents                                                                                            | Tool          | P0       | What depends on X?                | gnn/engine.rs::get_dependents()[Builtin]    |
| get_dependencies                                                                                          | Tool          | P0       | What does X depend on?            | gnn/engine.rs::get_dependencies()[Builtin]  |
| impact_analysis                                                                                           | Tool          | P0       | If I change X, what breaks?       | architecture/deviation_detector.rs[Builtin] |
| find_cycles                                                                                               | Tool          | P1       | Detect circular dependencies      | gnn/engine.rs::detect_cycles()[Builtin]     |
| get_module_boundaries                                                                                     | Tool          | P2       | Identify architectural layers     | NEW [Builtin]                               |
| cross_repo_deps                                                                                           | Tool          | P2       | External API/service dependencies | NEW [Builtin](Phase 2)                      |

#### 3.3.1.4 Database Connections & Schema Intelligence

Protocol:MCP (Primary via DB-specific MCP servers) | Builtin (Fallback for SQLite) Rationale: Well-maintained MCP servers exist for major databases. Platform-specific, not core differentiator.

| Capability                                                                                                 | Tool/Terminal | Priority | Purpose                                  | Implementation                                |
| ---------------------------------------------------------------------------------------------------------- | ------------- | -------- | ---------------------------------------- | --------------------------------------------- |
| ---------------------------------------------------------------------------------------------------------- |               |          |                                          |                                               |
| db_connect                                                                                                 | TOOL          | P0       | Establish connection with pooling        | NEW [MCP]agent/database/connection_manager.rs |
| -                                                                                                          | -             | -        | -                                        | -                                             |
| db_query                                                                                                   | TOOL          | P0       | Execute SELECT (read-only, validated)    | NEW [MCP]                                     |
| db_execute                                                                                                 | TOOL          | P0       | Execute INSERT/UPDATE/DELETE (validated) | NEW [MCP]                                     |
| db_schema                                                                                                  | TOOL          | P0       | Get tables, columns, types, constraints  | NEW [MCP]                                     |
| db_explain                                                                                                 | TOOL          | P0       | Query execution plan                     | NEW [MCP]                                     |
| db_migrate                                                                                                 | TOOL          | P0       | Run migrations with rollback             | NEW [MCP]agent/database/migration_manager.rs  |
| db_seed                                                                                                    | TOOL          | P0       | Insert test data                         | NEW [MCP]                                     |

MCP Servers: Postgres MCP, MySQL MCP, SQLite MCP, MongoDB MCP servers

Why Tool (Not Terminal):

- Connection Pooling: Reuse connections across commands (performance)
- Credential Security: Encrypted storage, no plaintext in terminal history
- Query Validation: Prevent SQL injection, validate against schema
- Transaction Support: Atomic operations with rollback
- Schema Tracking: Update GNN when schema changes
- Cross-DB Support: Unified API for PostgreSQL, MySQL, SQLite, MongoDB

Supported Databases:

- PostgreSQL (via MCP server or tokio-postgres)
- MySQL (via MCP server or sqlx)
- SQLite (via rusqlite - already in use for GNN) [Builtin fallback]
- MongoDB (via MCP server or mongodb crate)
- Redis (via MCP server or redis crate)

#### 3.3.1.5 API Monitoring & Contract Validation

Protocol:MCP (Primary for external APIs) | Builtin (HTTP client for health checks) Rationale: API-specific operations benefit from MCP ecosystem. Builtin HTTP client for basic health checks.

| Capability                                                                                       | Tool/Terminal | Priority | Purpose                       | Implementation                             |
| ------------------------------------------------------------------------------------------------ | ------------- | -------- | ----------------------------- | ------------------------------------------ |
| ------------------------------------------------------------------------------------------------ |               |          |                               |                                            |
| api_import_spec                                                                                  | TOOL          | P0       | Import OpenAPI/Swagger specs  | MCPagent/api_monitor/spec_parser.rs        |
| -                                                                                                | -             | -        | -                             | -                                          |
| api_validate_contract                                                                            | TOOL          | P0       | Detect breaking API changes   | MCPagent/api_monitor/contract_validator.rs |
| api_health_check                                                                                 | TOOL          | P1       | Test endpoint availability    | Builtin HTTP                               |
| api_rate_limit_check                                                                             | TOOL          | P1       | Track and predict rate limits | Builtin                                    |
| api_mock                                                                                         | TOOL          | P2       | Create mock server from spec  | MCP(Phase 2)                               |
| api_test                                                                                         | TOOL          | P2       | Test endpoint with assertions | MCP](Phase 2)                              |

Why Tool (Not Terminal):

- Schema Validation: Detect breaking changes automatically
- Rate Limit Tracking: Avoid 429 errors proactively
- Contract Storage: Compare against expected schema
- GNN Integration: Update dependency graph when APIs change
- Circuit Breaker: Fail fast when API is down

#### 3.3.1.6 Environment & System Resources

Protocol:Builtin (Exclusive) Rationale: System resource monitoring is low-level, performance-critical. Must be fast and cross-platform.

| Capability                                                                                   | Tool/Terminal | Priority | Purpose                      | Implementation                      |
| -------------------------------------------------------------------------------------------- | ------------- | -------- | ---------------------------- | ----------------------------------- |
| -------------------------------------------------------------------------------------------- |               |          |                              |                                     |
| env_get/env_set                                                                              | Terminal      | P0       | Environment variables        | Via terminal commands**[Builtin]**  |
| -                                                                                            | -             | -        | -                            | -                                   |
| get_cpu_usage                                                                                | Tool          | P2       | CPU metrics for optimization | [Builtin]agent/resources/monitor.rs |
| get_memory_usage                                                                             | Tool          | P2       | Memory stats                 | [Builtin]                           |
| get_disk_usage                                                                               | Tool          | P2       | Disk space monitoring        | [Builtin]                           |
| should_throttle                                                                              | Tool          | P2       | Adaptive resource management | [Builtin]                           |

### 3.3.2. 🧠 REASON - Decision-Making & Analysis Layer

Purpose: Analyze context, make decisions, plan actions

Status: ✅ 100% COMPLETE - All reasoning capabilities implemented

| Capability                                                      | MVP Status | Implementation                                              |
| --------------------------------------------------------------- | ---------- | ----------------------------------------------------------- |
| --------------------------------------------------------------- |            |                                                             |
| Confidence Scoring                                              | ✅ DONE    | agent/confidence.rs(320 lines)                              |
| -                                                               | -          | -                                                           |
| Impact Analysis                                                 | ✅ DONE    | architecture/deviation_detector.rs::analyze_change_impact() |
| Risk Assessment                                                 | ✅ DONE    | RiskLevelenum: Low/Medium/High/Critical                     |
| Decision Logging                                                | ✅ DONE    | State machine persistence in SQLite                         |
| Multi-LLM Orchestration                                         | ✅ DONE    | llm/multi_llm_manager.rs(13 providers)                      |
| Validation Pipeline                                             | ✅ DONE    | agent/validation.rs(412 lines)                              |
| Error Analysis                                                  | ✅ DONE    | agent/orchestrator.rs::analyze_error()                      |
| Adaptive Context Assembly                                       | ✅ DONE    | Hierarchical context with GNN                               |

No additional capabilities needed - reasoning layer is complete.

### 3.3.3. ⚡ ACT - Execution & Action Layer

Purpose: Execute decisions, modify code, run commands, deploy systems

#### 3.3.3.1 Terminal & Shell Execution

What: Built-in terminal that the AI agent can use to run commands autonomously. Execute code, install packages, run tests, and deploy - all from within Yantra without switching to external terminals.

For users, there will be no restriction to use the terminal manually.

Why: Eliminate context switching between IDE, terminal, and browser. Enable fully autonomous development workflow.

Protocol:Builtin (Exclusive - Security Critical)

Rationale: Shell execution is security-critical. Must be under complete control with no third-party intermediaries.

Implementation:src/agent/terminal.rs (391 lines - IMPLEMENTED)

| Capability                                                                               | Tool/Terminal | Priority | Purpose                                          | Implementation                           |
| ---------------------------------------------------------------------------------------- | ------------- | -------- | ------------------------------------------------ | ---------------------------------------- |
| ---------------------------------------------------------------------------------------- |               |          |                                                  |                                          |
| shell_exec                                                                               | Tool          | P0       | Run command, get output                          | agent/terminal.rs(391 lines)[Builtin]    |
| -                                                                                        | -             | -        | -                                                | -                                        |
| shell_exec_streaming                                                                     | Tool          | P0       | Long-running with real-time output               | terminal/executor.rs(331 lines)[Builtin] |
| shell_background                                                                         | Tool          | P0       | Start background process                         | terminal/pty_terminal.rs[Builtin]        |
| shell_kill                                                                               | Tool          | P0       | Terminate process                                | Terminal management**[Builtin]**         |
| shell_interactive                                                                        | Tool          | P0       | Pseudo-TTY for interactive CLIs                  | PTY implementation**[Builtin]**          |
| Smart Terminal Reuse                                                                     | Tool          | P0       | Detect idle terminals, reuse before creating new | Process detection**[Builtin]**           |

Status: ✅ 100% COMPLETE - Full terminal capabilities implemented

Requirements

1. Security: Command Validation

Whitelist-Based Approach - Only allow explicitly approved commands:

| Allowed Commands:                                                                                                                                                                                                                                                                                | Restricted Patterns                                                                                        | Argument Validation: |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | -------------------- |
| Python:python,python3,pip,pytest,black,flake8,mypyJavaScript:node,npm,npx,yarn,pnpm,jest,eslintRust:cargo(build, test, run, check)Docker:docker(build, run, ps, stop, logs)Git:git(via MCP protocol for security)Cloud:aws,gcloud,kubectl,terraform,herokuDatabases:psql,mysql,redis-cli,mongo`` | Allowed with consent:System modification:rm -rf,chmod +xCode execution:eval,exec,sourceNetwork piping:curl | bash,wget            |

2. Streaming Output

Real-Time Feedback - User sees output as it happens:

- Stream stdout line-by-line to frontend
- Stream stderr line-by-line to frontend
- Distinguish between stdout and stderr with labels
- Show command execution progress
- Display spinner while command is running
- Show final exit code and execution time

Use Cases:

- Long-running builds: Show compilation progress
- Package installation: Show download and install progress
- Test execution: Show test results as they complete
- Server startup: Show initialization logs

3. Context Awareness

Environment Setup Before Execution:

- Working Directory: Set to project workspace
- Virtual Environment: Activate appropriate venv/conda/node_modules
- Environment Variables: Set PYTHONPATH, NODE_PATH, etc.
- Language Version: Use correct Python/Node/Rust version for project
- Path: Include project-specific binaries

Multi-Language Context:

- Python: Activate .venv, set PYTHONPATH
- JavaScript: Use local node_modules/.bin
- Rust: Use project's target directory
- Go: Set GOPATH to project
- Java: Set JAVA_HOME and classpath
-

4. Error Recovery

When Command Fails:

1. Capture Full Context:
   - Exit code
   - Complete stdout and stderr
   - Command that was run
   - Working directory
   - Environment variables
2. Classify Error Type:

- Import/dependency error → Missing package
- Syntax error → Code generation issue
- Runtime error → Logic bug
- Permission error → Environment setup issue
- Port conflict → Service already running
- File not found → Path issue

1. Attempt Auto-Fix:

- Missing dependency → Install it
- Permission error → Suggest fix command
- Port conflict → Find available port
- File not found → Check path and suggest correction

1. Retry Logic:

- Maximum 3 retry attempts
- Apply fix before each retry
- Different strategy for each retry
- Escalate to user if all retries fail

1. Learn from Failures:

- Store error patterns in database
- Match new errors against known fixes
- Build confidence scores for fixes
- Share fixes across projects (opt-in)

5. Smart Terminal Management

Problem: Don't interrupt user's work or waste resources

Terminal State Detection:

- Idle: No foreground process, shell showing prompt → Safe to use
- Busy: Foreground process running (build, server, etc.) → Skip this terminal
- Unknown: Can't determine state → Skip to be safe

Detection Method:

- Use tcgetpgrp (Unix) to get foreground process group
- Compare with shell's process group
- If different → Terminal is busy
- Get process name to show user what's running

Terminal Reuse Strategy:

1. Check Existing Terminals First:
   - List all open terminals
   - Check each terminal's state
   - Use first idle terminal found
2. Create New Terminal Only When Needed:

- All existing terminals are busy
- No existing terminals available
- Cap at 5 agent-managed terminals maximum

1. User Communication:

- Show which terminal is being reused: "♻️ Reusing Terminal 2"
- Explain why skipping busy terminals: "⏭️ Skipping Terminal 1 (busy: cargo build)"
- Show when creating new: "🆕 Creating new terminal (all 3 existing terminals busy)"

Use Cases:

- User running npm run dev in Terminal 1 → Agent skips it, uses Terminal 2
- User running cargo build in Terminal 2 → Agent skips it, creates Terminal 3
- All terminals idle → Agent reuses Terminal 1
- 5 terminals all busy → Agent waits or asks user

6. Agent Integration with State Machine

EnvironmentSetup State:

- Detect project type (Python, JavaScript, Rust, etc.)
- Create virtual environment if needed
- Set environment variables
- Validate environment is ready

DependencyInstallation State:

- Check for manifest file (requirements.txt, package.json, Cargo.toml)
- Run install command (pip install -r requirements.txt, npm install, etc.)
- Stream installation progress to user
- Handle installation failures with retry logic
- Update GNN after successful installation

ScriptExecution State:

- Find entry point (main.py, index.js, src/main.rs)
- Run appropriate command (python main.py, node index.js, cargo run)
- Stream execution output to user
- Capture runtime errors
- Analyze errors and attempt fixes

Testing State:

- Run test command (pytest, npm test, cargo test)
- Stream test results to user
- Parse test output for pass/fail
- Identify failing tests
- Suggest fixes for failures

7. UI Integration

Output Panel in Frontend:

- Show terminal output in dedicated panel
- Syntax highlighting for different output types
- Auto-scroll to latest output
- Ability to pause auto-scroll
- Clear output button
- Copy output to clipboard
- Filter by stdout/stderr
- Search within output

Status Indicators:

- Spinner while command running
- Success checkmark when complete
- Error icon when failed
- Execution time display
- Exit code display

8. Cross-Platform Shell Compatibility (Windows, macOS, Linux)

Problem: Windows has different shell environments (CMD, PowerShell, WSL) with incompatible command syntax compared to Unix shells (bash, zsh).

Solution: Automatic Shell Detection and Command Translation

Shell Detection:

1. Windows:

- Detect default shell: PowerShell Core (pwsh) > PowerShell 5.1 (powershell) > CMD (cmd.exe)
- Check for WSL (Windows Subsystem for Linux) availability
- Prefer PowerShell Core for modern Windows environments

2. macOS/Linux:

- Detect user's default shell: zsh, bash, fish, etc.
- Use detected shell for all command execution

Command Translation Strategy:

| Scenario                                                                                                                | Unix/macOS Command        | Windows PowerShell         | Windows CMD                | Implementation                             |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------- | -------------------------- | ------------------------------------------ |
| ----------------------------------------------------------------------------------------------------------------------- |                           |                            |                            |                                            |
| Python execution                                                                                                        | python3 script.py         | python script.py           | python script.py           | Detect Python executable (python3vspython) |
| -                                                                                                                       | -                         | -                          | -                          | -                                          |
| Environment activation                                                                                                  | source .venv/bin/activate | .venv\Scripts\Activate.ps1 | .venv\Scripts\activate.bat | Platform-specific venv paths               |
| Path separators                                                                                                         | /(forward slash)          | \or/(both work)            | \(backslash)               | Normalize withstd::path                    |
| Multiple commands                                                                                                       | cmd1 && cmd2              | cmd1; if ($?) { cmd2 }     | cmd1 && cmd2               | Shell-specific chaining                    |
| Environment variables                                                                                                   | export VAR=value          | $env:VAR='value'           | set VAR=value              | Shell-specific syntax                      |
| List files                                                                                                              | ls -la                    | Get-ChildItemorls(alias)   | dir                        | Use language tools (not shell commands)    |

8. Performance Targets

| Operation                                                  | Target | Why                          |
| ---------------------------------------------------------- | ------ | ---------------------------- |
| ---------------------------------------------------------- |        |                              |
| Command validation                                         | <1ms   | Must not slow down execution |
| -                                                          | -      | -                            |
| Command start                                              | <50ms  | Fast response to user        |
| Output streaming latency                                   | <10ms  | Real-time feel               |
| Environment setup                                          | <5s    | Quick project initialization |
| Dependency installation                                    | <30s   | Use package manager cache    |
| Terminal state check                                       | <5ms   | Fast terminal reuse decision |

#### 3.3.3.3 Code Generation & Modification

Protocol:Builtin (Primary - Core Differentiator) Rationale: Code generation orchestration with GNN, LLM, and validation pipeline is Yantra's core value. Must be under complete control.

| Capability                                                                                                | Tool/Terminal | Priority | Purpose                       | Implementation                                  |
| --------------------------------------------------------------------------------------------------------- | ------------- | -------- | ----------------------------- | ----------------------------------------------- |
| --------------------------------------------------------------------------------------------------------- |               |          |                               |                                                 |
| generate_code                                                                                             | Tool          | P0       | LLM-powered code generation   | llm/multi_llm_manager.rs+ orchestrator[Builtin] |
| -                                                                                                         | -             | -        | -                             | -                                               |
| auto_correct_code                                                                                         | Tool          | P1       | Fix architecture deviations   | architecture/deviation_detector.rs[Builtin]     |
| refactor_code                                                                                             | Tool          | P1       | Automated refactoring         | NEW [Builtin](Phase 3)                          |
| Code completion                                                                                           | Tool          | P1       | Get Completions at postiion   | [MCP/Builtin fallback]                          |
| Code hover                                                                                                | Tool          | P1       | Get hover/documentation info  | [MCP/Builtin fallback]                          |
| Code actions                                                                                              | Tool          | P1       | Get quick fix/refactorings    | [MCP/Builtin fallback]                          |
| Code rename                                                                                               | Tool          | P1       | Rename symbols across project | [MCP/Builtin fallback]                          |
| Code format                                                                                               | Tool          | P1       | Format document/selection     | [MCP/Builtin fallback]                          |
| Code imports                                                                                              | Tool          | P1       | Analyze/organize imports      | MCP pylance                                     |
| Code signature servers                                                                                    | Tool          | P1       | Function signature help       | [MCP/Builtin fallback]                          |

#### 3.3.3.3 Git & Version Control

Protocol:MCP (Primary via @modelcontextprotocol/server-git) | Builtin (Fallback via git2-rs)

Rationale: Well-maintained MCP Git server exists. Platform-specific operations benefit from MCP. Builtin fallback ensures reliability.

Overview:

Git integration allows users to connect and authenticate once through natural language chat interaction. Agent helps set up Git configuration, handle authentication, and manage version control operations seamlessly.

Chat-Based Git Setup (MVP - Priority Feature):

Users connect to Git by typing in the chat panel. Agent guides them through authentication and setup:

User Experience:

User: "Connect to GitHub"

Agent: "I'll help you connect to GitHub. To proceed, I need:

1. Your GitHub username
2. Authentication method (Personal Access Token recommended)

Which authentication method would you like to use?

- Personal Access Token (recommended for security)
- SSH key (if you have one configured)
- GitHub CLI (if installed)"

User: "Personal Access Token"

Agent: "Great! Here's how to create a Personal Access Token:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click 'Generate new token (classic)'
3. Select scopes: repo, workflow, write:packages
4. Copy the token

Once you have it, paste it here (it won't be displayed):"

User: [pastes token]

Agent: "✅ Token received. Testing connection...

✅ Successfully authenticated as @username

✅ Git configured with your email: user@example.com

You're now connected! I can commit, push, pull, and manage branches for you.

Your token is securely stored and encrypted."

One-Time Authentication:

- User authenticates once per machine
- Credentials securely stored in system keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service)
- Automatic token refresh for GitHub (when supported)
- No re-authentication needed across sessions

Agent-Assisted Setup:

Agent handles all Git configuration:

- Detects if Git is installed (installs if missing on user consent)
- Configures user.name and user.email
- Sets up credential helper for token storage
- Tests connection and validates permissions
- Guides user through any errors with clear instructions

| Capability           | Tool/Terminal | Priority | Purpose                                  | Implementation                         |
| -------------------- | ------------- | -------- | ---------------------------------------- | -------------------------------------- |
| git_setup            | Tool          | P0       | Chat-based Git configuration & auth      | [Builtin]agent/git_setup.rs            |
| git_authenticate     | Tool          | P0       | Store credentials securely               | [Builtin]Uses system keychain          |
| git_test_connection  | Tool          | P0       | Validate authentication works            | [Builtin]Test fetch/push permissions   |
| git_status           | Terminal      | P0       | Current state                            | Via terminal commands**[MCP/Builtin]** |
| git_diff             | Terminal      | P0       | Changes (staged, unstaged, between refs) | Via terminal commands**[MCP/Builtin]** |
| git_log              | Terminal      | P0       | Commit history                           | Via terminal commands**[MCP/Builtin]** |
| git_blame            | Terminal      | P0       | Line-by-line attribution                 | Via terminal commands**[MCP/Builtin]** |
| git_commit           | Terminal      | P0       | Create commit with auto-messages         | Via terminal commands**[MCP/Builtin]** |
| git_push             | Terminal      | P0       | Push commits to remote                   | Via terminal commands**[MCP/Builtin]** |
| git_pull             | Terminal      | P0       | Pull latest changes                      | Via terminal commands**[MCP/Builtin]** |
| git_branch           | Terminal      | P0       | Create/switch/list branches              | Via terminal commands**[MCP/Builtin]** |
| git_checkout         | Terminal      | P0       | Checkout files/branches                  | Via terminal commands**[MCP/Builtin]** |
| git_merge            | Terminal      | P0       | Merge branches                           | Via terminal commands**[MCP/Builtin]** |
| git_stash            | Terminal      | P0       | Stash/pop changes                        | Via terminal commands**[MCP/Builtin]** |
| git_reset            | Terminal      | P0       | Undo changes                             | Via terminal commands**[MCP/Builtin]** |
| git_clone            | Terminal      | P0       | Clone repository                         | Via terminal commands**[MCP/Builtin]** |
| git_resolve_conflict | Tool          | P2       | AI-powered conflict resolution           | [MCP/Builtin]\*\*(Post-MVP)            |

#### 3.3.3.4 GITHUB / CODE HOSTING (Post MVP)

| Capability                                                           | Tool/Terminal | Priority | Description              | Implementation |
| -------------------------------------------------------------------- | ------------- | -------- | ------------------------ | -------------- |
| -------------------------------------------------------------------- |               |          |                          |                |
| github.repos                                                         | Tool          | P3       | List/search repositories | MCP Community  |
| -                                                                    | -             | -        | -                        | -              |
| github.issues                                                        | Tool          | P3       | Manage issues            | MCP Community  |
| github.prs                                                           | Tool          | P3       | Manage pull requests     | MCP Community  |
| github.actions                                                       | Tool          | P3       | View/trigger workflows   | MCP Community  |
| github.releases                                                      | Tool          | P3       | Manage releases          | MCP Community  |
| github.gists                                                         | Tool          | P3       | Manage gists             | MCP Community  |
| github.search                                                        | Tool          | P3       | Search code/issues/PRs   | MCP Community  |

#### 3.3.3.5 Testing Execution

Protocol:Builtin (Exclusive - Core to "Never Breaks" Guarantee) Rationale: Testing is core to "never breaks" guarantee. Must integrate with GNN for affected test detection. Performance-critical.

| Capability                                                                                                | Tool/Terminal | Priority | Purpose                             | Implementation                               |
| --------------------------------------------------------------------------------------------------------- | ------------- | -------- | ----------------------------------- | -------------------------------------------- |
| --------------------------------------------------------------------------------------------------------- |               |          |                                     |                                              |
| test_run                                                                                                  | Tool          | P0       | Execute tests (file, suite, single) | testing/test_generator.rs+ executor[Builtin] |
| -                                                                                                         | -             | -        | -                                   | -                                            |
| test_run_affected                                                                                         | Tool          | P1       | Run tests for changed code only     | NEW [Builtin](use GNN)                       |
| test_coverage                                                                                             | Tool          | P0       | Get coverage report                 | pytest-cov integration**[Builtin]**          |
| test_generate                                                                                             | Tool          | P0       | Auto-generate test cases            | testing/test_generator.rs[Builtin]           |
| test_debug                                                                                                | Tool          | P2       | Run test in debug mode              | NEW [Builtin with DAP](Phase 2)              |
| test_watch                                                                                                | Tool          | P1       | Continuous test runner              | NEW [Builtin](Phase 2)                       |
| e2e_run                                                                                                   | Tool          | P0       | Browser/integration tests           | NEW [Builtin](CDP + Playwright)              |

#### 3.3.3.6 Build & Compilation

Protocol:Builtin (Primary via shell) | MCP (Optional for language-specific services) Rationale: Build orchestration needs to coordinate with dep graph, testing, deployment. Shell execution via Builtin is sufficient.

| Capability                                                                                     | Tool/Terminal | MVP Status | Purpose                     | Implementation                                    |
| ---------------------------------------------------------------------------------------------- | ------------- | ---------- | --------------------------- | ------------------------------------------------- |
| ---------------------------------------------------------------------------------------------- |               |            |                             |                                                   |
| build_project                                                                                  | Terminal      | P0         | Full build                  | Via terminal (cargo build,npm run build)[Builtin] |
| -                                                                                              | -             | -          | -                           | -                                                 |
| build_incremental                                                                              | Terminal      | P0         | Changed files only          | Via terminal**[Builtin]**                         |
| build_check                                                                                    | Terminal      | P0         | Type-check without emitting | Via terminal**[Builtin]**                         |
| build_clean                                                                                    | Terminal      | P0         | Clear artifacts             | Via terminal**[Builtin]**                         |
| lint_run                                                                                       | Tool          | P0         | Run linters                 | Security scanner includes linting**[Builtin]**    |
| lint_fix                                                                                       | Tool          | P1         | Auto-fix lint issues        | NEW [Builtin]                                     |
| format_code                                                                                    | Terminal      | P0         | Apply formatters            | Via terminal (rustfmt,prettier)[Builtin]          |

Status: ✅ Build and lint via terminal works well

#### 3.3.3.7 Package Management

Protocol:Builtin (Exclusive) Rationale: Package operations need shell execution. Audit can integrate with MCP vulnerability databases, but orchestration is Builtin.

| Capability                                                                           | Tool/Terminal | MVP Status | Purpose                      | Implementation                            |
| ------------------------------------------------------------------------------------ | ------------- | ---------- | ---------------------------- | ----------------------------------------- |
| ------------------------------------------------------------------------------------ |               |            |                              |                                           |
| pkg_install                                                                          | Tool          | P0         | Add dependency               | agent/dependencies.rs(429 lines)[Builtin] |
| -                                                                                    | -             | -          | -                            | -                                         |
| pkg_remove                                                                           | Tool          | P0         | Remove dependency            | agent/dependencies.rs[Builtin]            |
| pkg_update                                                                           | Tool          | P0         | Update dependencies          | agent/dependencies.rs[Builtin]            |
| pkg_list                                                                             | Tool          | P0         | List installed packages      | agent/dependencies.rs[Builtin]            |
| pkg_audit                                                                            | Tool          | P0         | Security vulnerability check | security/scanner.rs[Builtin]              |
| pkg_search                                                                           | Tool          | P1         | Find packages in registry    | NEW [Builtin]                             |
| pkg_lock_sync                                                                        | Tool          | P0         | Sync lockfile                | Via package manager commands**[Builtin]** |

#### 3.3.3.8 Deployment & Infrastructure

Protocol:MCP (Primary for platform-specific services) | Builtin (Container operations via shell) Rationale: Platform-specific deployments (Railway, Vercel, AWS, GCP) benefit from MCP servers. Container operations via Docker CLI (Builtin).

| Capability                                                                                            | Tool/Terminal | MVP Status | Purpose                               | Implementation                      |
| ----------------------------------------------------------------------------------------------------- | ------------- | ---------- | ------------------------------------- | ----------------------------------- |
| ----------------------------------------------------------------------------------------------------- |               |            |                                       |                                     |
| deploy_preview                                                                                        | Tool          | ✅ DONE    | Deploy to preview environment         | agent/deployment.rs(636 lines)[MCP] |
| -                                                                                                     | -             | -          | -                                     | -                                   |
| deploy_production                                                                                     | Tool          | ✅ DONE    | Deploy to prod (with confirmation)    | agent/deployment.rs[MCP]            |
| deploy_rollback                                                                                       | Tool          | ✅ DONE    | Revert deployment                     | agent/deployment.rs[MCP]            |
| deploy_status                                                                                         | Tool          | ✅ DONE    | Check deployment state                | agent/deployment.rs[MCP]            |
| deploy_logs                                                                                           | Tool          | ✅ DONE    | Fetch deployment logs                 | agent/deployment.rs[MCP]            |
| infra_provision                                                                                       | Tool          | 🔴 TODO    | Create resources (Railway, AWS, etc.) | NEW [MCP](Phase 2)                  |
| container_build                                                                                       | Terminal      | ✅ DONE    | Build Docker image                    | Viadocker build[Builtin]            |
| container_run                                                                                         | Terminal      | ✅ DONE    | Run container locally                 | Viadocker run[Builtin]              |

MCP Servers: Railway MCP (custom), Vercel MCP, AWS MCP, GCP/Azure community servers

Status: ✅ All MVP deployment complete (5/5 P0+P1 features to Railway), 🔴 Multi-cloud pending (P2 Post-MVP)

#### 3.3.3.9 Browser Automation (CDP)

Protocol:Builtin (Exclusive - Core to Browser Product) Rationale: CDP integration is core to Yantra's browser validation. Must be fast, reliable, and under complete control.

| Capability                                                                                         | Tool/Terminal | MVP Status | Purpose                 | Implementation                       |
| -------------------------------------------------------------------------------------------------- | ------------- | ---------- | ----------------------- | ------------------------------------ |
| -------------------------------------------------------------------------------------------------- |               |            |                         |                                      |
| browser_launch                                                                                     | Tool          | 🟡 PARTIAL | Start browser instance  | browser/cdp.rs(placeholder)[Builtin] |
| -                                                                                                  | -             | -          | -                       | -                                    |
| browser_navigate                                                                                   | Tool          | 🟡 PARTIAL | Go to URL               | browser/cdp.rs[Builtin]              |
| browser_click                                                                                      | Tool          | 🔴 TODO    | Click element           | NEW [Builtin]                        |
| browser_type                                                                                       | Tool          | 🔴 TODO    | Input text              | NEW [Builtin]                        |
| browser_screenshot                                                                                 | Tool          | 🔴 TODO    | Capture screen          | NEW [Builtin]                        |
| browser_select_element                                                                             | Tool          | 🔴 TODO    | Visual element picker   | NEW[Builtin](Post-MVP)               |
| browser_evaluate                                                                                   | Tool          | 🔴 TODO    | Run JS in page context  | NEW [Builtin]                        |
| browser_network                                                                                    | Tool          | 🔴 TODO    | Intercept/mock requests | NEW[Builtin](Post-MVP)               |
| browser_console                                                                                    | Tool          | 🔴 TODO    | Get console logs        | NEW [Builtin]                        |

Status: 🟡 25% complete (2/8 features) - CDP placeholder needs full implementation

#### 3.3.3.10 HTTP & API Execution

Protocol:Builtin (Primary) | MCP (Fallback for specific API integrations) Rationale: HTTP client with circuit breaker, retry, and rate limiting is infrastructure-level. Must be under control for reliability.

| Capability                                                                                                         | Tool/Terminal | MVP Status | Purpose                                    | Implementation                        |
| ------------------------------------------------------------------------------------------------------------------ | ------------- | ---------- | ------------------------------------------ | ------------------------------------- |
| ------------------------------------------------------------------------------------------------------------------ |               |            |                                            |                                       |
| http_request                                                                                                       | TOOL          | 🔴 TODO    | Make HTTP calls with retry/circuit breaker | NEW [Builtin]agent/http_client/mod.rs |
| -                                                                                                                  | -             | -          | -                                          | -                                     |
| websocket_connect                                                                                                  | Tool          | 🔴 TODO    | WebSocket client                           | NEW [Builtin](Phase 2)                |

### 3.3.4. 🔄 LEARN - Feedback & Adaptation Layer

Purpose: Learn from results, adapt behavior, improve over time

Status: ✅ 100% COMPLETE - All learning capabilities implemented

| Capability                                                  | Priority | Implementation                                         |
| ----------------------------------------------------------- | -------- | ------------------------------------------------------ |
| ----------------------------------------------------------- |          |                                                        |
| Validation Pipeline                                         | P0       | agent/validation.rs(412 lines)                         |
| -                                                           | -        | -                                                      |
| Auto-Retry with Error Analysis                              | P0       | agent/orchestrator.rs(651 lines)                       |
| Self-Correction                                             | P0       | agent/confidence.rs+ auto-retry                        |
| Confidence Score Updates                                    | P0       | Real-time confidence adjustment                        |
| Known Issues Database                                       | P0       | SQLite persistence for LLM failuresIn Cloud (Post MVP) |
| Pattern Extraction                                          | P0       | Error pattern recognition                              |
| Failure Network Effects                                     | P0       | Shared learning (privacy-preserving)                   |

### 3.3.5. 📋 Cross-Cutting Primitives

#### 3.3.5.1 Debugging

Protocol:DAP (Debug Adapter Protocol - Exclusive) Rationale: Debugging is handled by DAP (Debug Adapter Protocol), a separate protocol specifically for debugging operations.

| Capability                                                       | Tool/Terminal | Priority | Purpose                    | Implementation |
| ---------------------------------------------------------------- | ------------- | -------- | -------------------------- | -------------- |
| ---------------------------------------------------------------- |               |          |                            |                |
| debug_start                                                      | Tool          | P2       | Launch debugger            | [DAP](Phase 2) |
| -                                                                | -             | -        | -                          | -              |
| debug_breakpoint                                                 | Tool          | P2       | Set/remove breakpoints     | [DAP]          |
| debug_step                                                       | Tool          | P2       | Step over/into/out         | [DAP]          |
| debug_continue                                                   | Tool          | P2       | Resume execution           | [DAP]          |
| debug_evaluate                                                   | Tool          | P2       | Eval expression in context | [DAP]          |
| debug_stack                                                      | Tool          | P2       | Get call stack             | [DAP]          |
| debug_variables                                                  | Tool          | P2       | Inspect variables          | [DAP]          |

DAP Adapters: Python (debugpy), Node.js (node-debug2), Rust (codelldb), Go (delve)

Status: (Phase 2 - Post-MVP)

#### 3.3.5.2 Documentation

Protocol:Builtin (Primary) | MCP (Fallback for external doc fetching) Rationale: Documentation generation from code is core functionality. External doc fetching can use MCP for various sources.

| Capability                                                                                             | Tool/Terminal | Priority | Purpose                                | Implementation                    |
| ------------------------------------------------------------------------------------------------------ | ------------- | -------- | -------------------------------------- | --------------------------------- |
| ------------------------------------------------------------------------------------------------------ |               |          |                                        |                                   |
| docs_generate                                                                                          | Tool          | P0       | Generate from code (JSDoc, docstrings) | File Registry system**[Builtin]** |
| -                                                                                                      | -             | -        | -                                      | -                                 |
| docs_search                                                                                            | Tool          | P1       | Search project docs                    | [Builtin]                         |
| docs_external                                                                                          | Tool          | P2       | Fetch library documentation            | [MCP](Phase 2)                    |

#### 3.3.5.3 Security

Protocol:Builtin (Primary for scanning) | MCP (Optional for vulnerability databases)

Rationale: Security scanning is core to "never breaks" guarantee. SAST must be under control. MCP can augment with external CVE databases.

| Capability       | Tool/Terminal | MVP Status | Purpose                      | Implementation                          |
| ---------------- | ------------- | ---------- | ---------------------------- | --------------------------------------- |
| security_scan    | Tool          | P0         | SAST analysis                | security/scanner.rs(512 lines)[Builtin] |
| secrets_detect   | Tool          | P0         | Find exposed credentials     | Integrated in scanner**[Builtin]**      |
| dependency_audit | Tool          | P0         | CVE check on packages        | Integrated in scanner**[Builtin]**      |
| secrets_manager  | Tool          | P1         | Encrypted credential storage | NEW [Builtin]agent/secrets/vault.rs     |

#### 3.3.5.4 Architecture Visualization

Protocol:Builtin (Exclusive - Core Differentiator) Rationale: Architecture visualization from GNN is Yantra's core differentiator. Must be completely under control.

| Capability            | Tool/Terminal | MVP Status | Purpose                         | Implementation                              |
| --------------------- | ------------- | ---------- | ------------------------------- | ------------------------------------------- |
| arch_diagram_generate | Tool          | P0         | Create visual from code         | Architecture View System (16/16)[Builtin]   |
| arch_validate         | Tool          | P0         | Check against constraints       | architecture/deviation_detector.rs[Builtin] |
| arch_suggest          | Tool          | P0         | Recommend improvements          | Impact analysis**[Builtin]**                |
| arch_import           | Tool          | P0         | Import from MD/Mermaid/PlantUML | project_initializer.rs[Builtin]             |

Status: ✅ 100% COMPLETE - Architecture system fully implemented

#### 3.3.5.5 Context & Memory

Protocol:Builtin (Exclusive - Core Differentiator) Rationale: Context assembly with hierarchical GNN is core to Yantra's intelligence. Must be completely under control.

| Capability          | Tool/Terminal | MVP Status | Purpose                               | Implementation                         |
| ------------------- | ------------- | ---------- | ------------------------------------- | -------------------------------------- |
| context_add         | Tool          | P0         | Add to agent's working memory         | State machine persistence**[Builtin]** |
| context_search      | Tool          | P0         | Semantic search over codebase         | GNN semantic layer**[Builtin]**        |
| context_summarize   | Tool          | P0         | Compress context for token efficiency | Hierarchical assembly**[Builtin]**     |
| project_conventions | Tool          | P1         | Get coding standards/patterns         | NEW [Builtin]                          |

Status: ✅ Context management mostly complete

## 3.4 AGENTIC ORCHESTRATION

### 3.4.1 LLM Orchestration

Yantra agent will use LLM as per the user selection for code generation. Yantra will have unique multi-LLM orchestration functionality.When the primary LLM gets stuck after 2 failed attempts, Yantra automatically consults other available LLMs to get alternative perspectives, insights, or debugging help. This is collaborative, not competitive - different LLMs have different strengths and blind spots. A second opinion often breaks through barriers that cause repeated failures.

Core Philosophy:

Primary LLM stuck → Consult another LLM "What am I missing?" → Primary uses insight

Why This Works:

- LLMs have different training data and biases
- Claude: Strong reasoning, cautious on edge cases
- GPT-4: Pragmatic, compact code style
- Gemini: Different perspective, fresh approach
- When stuck, the problem is often framing or blind spot - different LLM sees differently

Cost Benefit:

- Without consultation: 5+ failed attempts → human intervention needed
- With consultation: 2 failures + 1 consultation + 1 success = 4 total calls (saves money + time)

---

#### 3.4.1.1 Feature 1: Consultation Trigger & Orchestration (MVP)

Consultation Flow:

Attempt 1: Primary LLM generates code

    ↓

    Tests fail

    ↓

Attempt 2: Primary retries with error context

    ↓

    Tests fail again (TRIGGER THRESHOLD)

    ↓

Consultation: Consultant LLM provides insight

    ↓

Attempt 3: Primary regenerates with consultant's insight

    ↓

    ✅ Success or try again

Trigger Logic:

- Threshold: After 2 consecutive failures with same primary LLM
- Consultation: Call Consultant LLM with full context
- Integration: Primary LLM regenerates with consultant's insight added to context
- Max Attempts: 3 total (2 primary alone + 1 with consultation insight)
- Escalation: After 3 failures → show user options (try different primary, manual intervention)

Backend Structure:

// File: src-tauri/src/llm/consultation.rs

Consultation Success Metrics:

- Track consultations triggered per session
- Track success rate after consultation
- Track which LLM combinations work best
- Average attempts before/after consultation

---

#### 3.4.1.2 Feature 2: Dynamic Consultation Prompt Generation (MVP)

Prompt Strategy: Instead of rigid templates, use the Primary LLM itself to generate the optimal consultation prompt based on context. This ensures the consultation request is well-framed and provides maximum value.

Two-Step Process:

Step 1: Primary LLM Creates Consultation Prompt

// Primary LLM generates the consultation request

let meta_prompt = format!(

    "You are creating a consultation request for another expert AI coding assistant.

    Context:

    - Task: {}

    - Your attempts so far: {} (all failed tests)

    - Errors encountered: {}

    Create a consultation prompt that:

    1. Asks the consultant to assume they are a top coding expert

    2. Clearly states the context and problem

    3. Includes relevant code snippets and error messages

    4. Asks for specific help resolving the issue

    Generate the consultation prompt:",

    task_description,

    attempts_summary,

    error_summary

);

let consultation_prompt = primary_llm.generate(meta_prompt).await?;

Step 2: Consultant LLM Responds

// Consultation prompt structure (generated by Primary LLM)

"You are a top coding expert consultant. Another AI attempted this task and got stuck after 2 attempts.

Task Description:

{}

Code Generated (Attempt 1):

{}

Test Failures:

{}

Code Generated (Attempt 2):

{}

Test Failures:

{}

What is being missed? What's a different approach to solve this?

Provide:

1. Root cause analysis
2. Specific fix recommendations
3. Alternative implementation approach (if applicable)"

let consultant_response = consultant_llm.generate(consultation_prompt).await?;

Consultation Prompt Template (Baseline):

If Primary LLM fails to generate consultation prompt (unlikely), fall back to this template:

You are a top-tier coding expert consultant. Another AI coding assistant attempted a task and encountered persistent failures after 2 attempts. Please provide your expert analysis.

TASK DESCRIPTION:

{task_description}

ATTEMPT 1:

Generated Code:

{code_attempt_1}

Test Results:

{test_failures_1}

ATTEMPT 2:

Generated Code:

{code_attempt_2}

Test Results:

{test_failures_2}

ANALYSIS REQUEST:

As a top coding expert, please analyze:

1. What fundamental issue is being missed?
2. What's wrong with the current approach?
3. What alternative approach would you recommend?
4. Are there edge cases not being considered?

Provide a clear, actionable response that will help resolve this issue.

Why Dynamic Prompt Generation:

- Context-aware: Adapts to specific failure patterns
- Better framing: Primary LLM knows what info is most relevant
- Flexibility: Works for any type of coding task
- No rigid templates: Consultant gets exactly the context needed

---

#### 3.4.1.3 Feature 3: LLM Settings UI - Consultation Configuration (MVP)

- User should be able to configure API keys for the LLM provider and then select Primary and Secondary aka Consultant Model from the LLM settings panel
- Consultant Model is optional can be left unselected
- LLM API providers: Claude, OpenAI, Gemini, Groq, Openrouter
- Show the top paid models and opensource models
  - Claude - Opus 4.5, Sonnet 4.5, Haiku4.5
  - OpenAI - GPT5.1, GPT Codex5.1, GPT-4 Turbo
  - Gemini - Gemini 3.0, Gemini 3.0 flash, Gemini 3.0 Pro
  - Groq - llama-3.4-70b, mixtral-8x7b
  - Openrouter - Opus 4.5, Sonnet 4.5, Haiku4.5, GPT5.1, GPT Codex5.1, GPT-4 Turbo, Gemini 3.0, Gemini 3.0 flash, Gemini 3.0 Pro, llama-3.4-70b, mixtral-8x7b, Deep Seek Coder V2, Qwen 2.4 Coder 32b, CodeLLAMA-70B

Model Selection Logic:

- Primary Model: Must be selected (required dropdown)
- Consultant Model: Optional (can be null)
  - If null in Guided Mode → Ask user after 2 failures: "Get second opinion? [Select model]"
    - Save the selection to use for future
  - If null in Auto Mode → Use same model as Primary (with different consultation prompt)
- Same Model Allowed: Yes - user can select same model for both Primary and Consultant
- Show info tooltip: "Using same model may provide limited new perspective"

---

#### 3.4.1.4 Feature 5: UI Transparency - Consultation Progress

Show user what's happening in real-time:

// Progress Panel: src-ui/components/ConsultationProgress.tsx

interface ConsultationStep {

    step: number;

    action: string;

    status: 'pending' | 'running' | 'success' | 'failed';

    model: string;

    details?: string;

}

// Example UI:

┌─────────────────────────────────────────────────────────────┐

│ Generating: User Authentication Module │

├─────────────────────────────────────────────────────────────┤

│ │

│ ✅ Attempt 1 (Claude Sonnet 4) │

│ Generated auth module with JWT │

│ ❌ Tests failed: JWT validation error │

│ │

│ ✅ Attempt 2 (Claude Sonnet 4) │

│ Fixed JWT validation logic │

│ ❌ Tests failed: Session handling edge case │

│ │

│ 🤔 Consulting GPT-4 Turbo for second opinion... │

│ 💡 Insight received: │

│ "Session lifecycle needs refresh token rotation. │

│ Current implementation doesn't handle token expiry." │

│ │

│ ✅ Attempt 3 (Claude Sonnet 4 + GPT-4 insight) │

│ Added refresh token rotation with expiry handling │

│ ✅ All tests passing ✅ │

│ │

│ 📊 Consultation saved 2+ failures & manual intervention │

└─────────────────────────────────────────────────────────────┘

Progress States:

- ⏳ Generating code...
- ✅ Code generated
- 🧪 Running tests...
- ❌ Tests failed (show summary)
- 🤔 Consulting [Model Name]...
- 💡 Insight received
- ✅ Success!
- ❌ Escalated to manual review

Transparency Benefits:

- User sees Yantra is thorough, not blindly retrying
- Builds trust in multi-LLM collaboration
- Educational: Users learn from consultation insights
- Justifies cost: Clear value of consultation vs repeated failures

#### Success Criteria (MVP)

Functional:

- ✅ Triggers consultation after exactly 2 primary failures
- ✅ Dynamically generates consultation prompt using Primary LLM
- ✅ Only shows models with valid API keys in dropdowns
- ✅ Guided Mode asks user for consultant selection when not pre-configured
- ✅ Auto Mode uses Primary model as consultant if none selected
- ✅ Allows same model for Primary and Consultant
- ✅ Integrates consultant insight into final attempt
- ✅ Shows transparent progress in UI

Performance:

- ✅ Consultation adds <500ms overhead (prompt generation + API call)
- ✅ Available models API responds in <100ms
- ✅ UI remains responsive during consultation

Metrics:

- ✅ Track consultation success rate (target: >70% resolve after consultation)
- ✅ Track cost savings vs repeated failures
- ✅ Track which model combinations work best
- ✅ User satisfaction: Consultation helpful (feedback)

---

### 3.4.2 State Machine Architecture: Separation of Concerns

Yantra's agentic capabilities are implemented through five specialized state machines with Preventive Development Cycle, each with a focused responsibility. This separation of concerns provides :

1. Clarity: Each machine has a clear, single responsibility
2. Maintainability: Changes to one machine don't affect others
3. Testability: Each machine can be tested independently
4. Scalability: Machines can run in parallel or be triggered independently
5. MVP Focus: Build only what's needed, when needed
6. Parallel Processing: States leverage concurrent execution whenever possible for performance

When to Apply Parallel Processing:

1. Independent Operations: Tasks with no data dependencies can execute simultaneously
2. Multiple Resources: Different resources (files, APIs, browser tabs) can be accessed concurrently
3. Batch Operations: Multiple similar operations (e.g., testing multiple files, scanning multiple dependencies)
4. I/O-Bound Tasks: Network requests, file reads, API calls benefit from concurrent execution

Implementation Strategy:

5. Within States: Use async/await with tokio for concurrent operations within a single state
6. Across States: Future enhancement - multiple state machines running simultaneously
7. Resource Pooling: Browser tab pool, HTTP connection pool, LLM request batching

Performance Benefits:

8. Sequential: N tasks × T seconds = N\*T total time
9. Parallel: N tasks / P workers × T seconds ≈ (N/P)\*T total time (where P = parallelism factor)
10. Example: 4 web searches: 12s sequential → 4s parallel (3x faster)

This Autonomous Development is further bolstered by “Yantra’s Preventive Development Cycle” (YPDC) which is implemented through State Machines.

Core Philosophy of YPDC is

- Problems are preventable, not inevitable
- The agent orchestrates, humans provide intent and oversight
- Zero conflicts by design, not by resolution
- Documentation is always accurate because it's generated from truth
- Quality is built-in, not inspected-in

##### Yantra Preventive Development Cycle Overview

| Phase               | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| 1. Architect/Design | Ensure solution is correct before code is written                       |
| 2. Plan             | Create executable plans that prevent implementation problems            |
| 3. Execute          | Write code that is correct, tested, secure, and conflict-free by design |
| 4. Deploy           | Ensure deployments are safe, validated, and recoverable                 |
| 5. Monitor/Maintain | Detect issues before users do and heal automatically                    |

Yantra Preventive Development Lifecycle - Consolidated Table

| 1. Architect                               |
| ------------------------------------------ |
| Sub-Phase                                  |
| 1.1 Architecture Alignment                 |
| 1.2 Tech Stack Alignment                   |
| 1.3 Existing Code Analysis                 |
| 1.4 Feature Extraction                     |
| 1.5 Architecture Sign-off                  |
| 1.6 Automated Documentation (Architecture) |
| 2. Plan                                    |
| 2.1 Execution Plan by Features             |
| 2.2 Progress Tracking                      |
| 2.3 Plan Alignment & Sign-off              |
| 3. Execute                                 |
| 3.1 Code Generation + Testing              |
| 3.2 Prevent Bugs                           |
| 3.3 Prevent Merge Conflicts                |
| 3.4 Prevent Security Issues                |
| 3.5 Auto Unit & Integration Tests          |
| 3.6 Implementation Documentation           |
| 3.7 Feature Sign-off                       |
| 4. Deploy                                  |
| 4.1 Pre-Deploy Validation                  |
| 4.2 Auto Deploy                            |
| 4.3 Deploy Sign-off                        |
| 5. Maintain                                |
| 5.1 Self-Healing                           |

##### The Five State Machines:

1. Code Generation State Machine (MVP) Responsibility: Generate production-quality code that doesn't break dependencies

Key States: Architecture Generation → Architecture Review → Dependency Assessment → Task Decomposition → Dependency Sequencing → Conflict Check → Plan Generation → Blast Radius Analysis → Plan Review → Environment Setup → Context Assembly → Code Generation → Dependency Validation → Browser Validation → Security Scanning → Concurrency Validation → Fixing Issues → Complete/Failed

Entry: User's natural language intent Exit: Generated code + confidence score Total States: 17 MVP / 19 Post-MVP

2A. Testing Intelligence State Machine (MVP) Responsibility: Ensure generated code works correctly through comprehensive testing

Key States: Test Generation → Environment Setup → Unit Testing → Browser Testing → Integration Testing → Coverage Analysis → Fixing Issues → Complete/Failed

Entry: Generated code from CodeGen machine Exit: Test results + coverage report Auto-Trigger: Yes (runs after CodeGen succeeds)

2B. Testing Execution State Machine (MVP) Responsibility: Execute tests with comprehensive validation, debugging feedback, and resilience

3.Deployment State Machine (MVP - Railway Focus) Responsibility: Deploy validated code to Railway.app

Key States: Package Building → Config Generation → Railway Upload → Health Check → Rollback On Failure → Complete/Failed

Entry: Code + passing tests Exit: Live Railway URL + health status Auto-Trigger: No (requires user approval) Platform: Railway.app only in MVP

4.Maintenance State Machine (Post-MVP) Responsibility: Monitor production, detect issues, auto-fix, and deploy patches

Key States: Live Monitoring → Browser Validation → Error Analysis → Issue Detection → Auto-Fix Generation → Fix Validation → CI/CD Pipeline → Verification Check → Learning Update → Active/Incident

Entry: Deployed application in production Exit: Incident resolved or escalated Auto-Trigger: Yes (automatic based on error detection) Philosophy: Self-healing production systems

**The Five State Machines Detailed Requirements**

##### 3.4.2.1. Code Generation State Machine (MVP)

Responsibility: Generate production-quality code that doesn't break dependencies

Entry Point: User's natural language intent Exit Point: Generated code + confidence score Trigger: User submits task Success Criteria: Code passes GNN validation, security scan, no breaking changes, visibility of active work

State Count:

11. MVP: 18 states (Architecture: 4, Planning: 6, Execution: 7 - adds ConcurrencyValidation)
12. Post-MVP: 19 states (adds explicit FileLockAcquisition and FileLockRelease states)

States:Key States: Architecture Generation → Architecture Review → Dependency Assessment → Task Decomposition → Dependency Sequencing → Conflict Check → Plan Generation → Blast Radius Analysis → Plan Review → Environment Setup → Context Assembly → Code Generation → Dependency Validation → Browser Validation → Security Scanning → Concurrency Validation → Fixing Issues → Complete/Failed

Phase 1: Architecture & Design (PDC Phase 1)

1. ArchitectureGeneration: Generate or import project architecture
2. ArchitectureReview: Wait for user approval of architecture (human-in-the-loop)

- Agent knows boundaries, won't generate violating code
- Agent will detect Circular dependencies, scaling bottlenecks
- Agent will flag single points of failure
- Agent will assess Security vulnerabilities by design - Security patterns enforced at architecture level
- ⚠️ APPROVAL GATE: User must approve architecture changes

Phase 2: Planning (PDC Phase 2 - Enhanced)

3. DependencyAssessment: Assess the tools/package/techstack needed, analyze compatibility issues with version, do dry run validation, assess vulnerabilities with CVE db
   1. Web Search Integration: Agent MUST use web search for latest package info (LLM knowledge is static/outdated)
      1. Query official package registries (PyPI, npm, crates.io, Maven Central)
      2. Fetch latest versions, changelogs, migration guides, known issues
      3. Tools: MCP @modelcontextprotocol/server-brave-search or @modelcontextprotocol/server-fetch

   2. RAG-Enhanced Resolution: Use Vector DB to find similar past dependency conflicts and resolutions
   3. Query cached package documentation (from previous web searches)
   4. Retrieve compatibility matrices, known breaking changes
   5. Provide context to LLM: "Your training data ends {date}, use web search for current info"
   6. Version-Level Tracking: Store exact versions, version constraints, compatibility status in GNN
   7. Conflict Detection: Cross-reference requested packages with existing dependencies, detect version conflicts
   8. ⚡ Parallel Processing:

- Check multiple dependencies simultaneously (parallel CVE lookups, version checks)
- Open multiple browser tabs for different package sources (PyPI, npm, docs, GitHub)
- Batch API calls to package registries
- Performance: N dependencies checked in ~O(1) time vs O(N) sequential

4. TaskDecomposition: Break feature into concrete tasks (DB changes, API endpoints, UI components, tests, docs)

- ⚡ Parallel Processing: Analyze multiple feature aspects simultaneously (data model, API surface, UI requirements)

5. DependencySequencing: Use GNN to determine task order and identify dependencies between tasks

- ⚡ Parallel Processing: Identify all independent task clusters that can execute in parallel

6. ConflictCheck: Check which files will be modified and show visibility of who's working on what

- MVP: Display active work visibility (which developer is working on which files)
- Post-MVP: File locking mechanism to prevent parallel edits
- ⚡ Parallel Processing: Check multiple file locks/status simultaneously

7. PlanGeneration: Create executable plan with task list, time estimates, critical path analysis

- ⚡ Parallel Processing: Calculate complexity estimates for multiple tasks concurrently

8. BlastRadiusAnalysis: NEW P0 - Calculate and visualize impact of planned changes

- Purpose: Show users exactly what will be affected BEFORE execution (preventive transparency)
- Data Sources: GNN dependency graph, test coverage, API specs, package metadata
- Intelligent Display Logic:
  - Always show for: critical files, breaking changes, >10 indirect deps, package upgrades
  - Show for large changes: >5 tasks, >3 files, >20 tests affected
  - Skip for small changes: single file, no dependencies, <5 tests, no breaking changes
- Output: BlastRadiusPreview (see detailed spec below)
- Performance Target: <2s (GNN queries + analysis)

9. PlanReview: User reviews and approves execution plan (optional approval gate for complex features)

- ⚠️ OPTIONAL APPROVAL GATE: Required for features >5 tasks or multi-file changes OR high blast radius
- Displays: ExecutionPlan + BlastRadiusPreview (integrated view)
- User Actions: Approve, Modify Plan, Cancel

10. EnvironmentSetup: Setup the venv with all the techstack dependencies installed. Always use venv

- ⚡ Parallel Processing: Install multiple independent packages simultaneously (parallel pip install)

Phase 3: Execution (PDC Phase 3)

11. ContextAssembly: Gather hierarchical context from GNN

- ⚡ Parallel Processing: Fetch multiple file contexts simultaneously, parallel GNN queries for different dependency levels

12. CodeGeneration: LLM generates code with full context

- MVP: Show which files are being modified in UI (active work indicator)
- Post-MVP: Explicit FileLockAcquisition state before code generation
- ⚡ Parallel Processing: Generate multiple independent files/functions simultaneously with separate LLM calls

13. DependencyValidation: GNN checks for breaking changes

- ⚡ Parallel Processing: Validate multiple dependency paths simultaneously, parallel impact analysis

14. BrowserValidation: Quick visual validation (renders, console errors, screenshots)

- ⚡ Parallel Processing:
  - Open multiple browser tabs for different pages/components
  - Test multiple viewports simultaneously (desktop, mobile, tablet)
  - Parallel screenshot capture for visual comparison
  - Performance: 3-5 components × 10s = 30-50s sequential → 10-15s parallel

15. SecurityScanning: Semgrep + OWASP rules for vulnerability detection

- ⚡ Parallel Processing:
  - Scan multiple files simultaneously
  - Run different security rules in parallel (Semgrep, secrets detection, CVE checks)
  - Parallel analysis of multiple code patterns
  - Performance: N files scanned in ~O(log N) time with parallel workers

16. ConcurrencyValidation: Race condition and deadlock detection (⚡ PARALLEL SAFETY CHECK)

- Purpose: Validate that parallel processing didn't introduce concurrency bugs
- When: After any code generation that uses async/await, threads, or parallel execution
- Built-in Checks (Agentic Flow - NOT just LLM reasoning):
  - Static Analysis: Rust Clippy pedantic + thread safety lints, Python threading analyzer, JavaScript event loop analyzer
  - Pattern Detection: Shared mutable state access, missing locks/semaphores, race-prone patterns (check-then-act)
  - GNN Analysis: Identify data flow paths that could race, detect concurrent writes to same resource
  - Test Generation: Automatically generate stress tests for concurrent code paths

- LLM Role (Supplementary):
- Explain detected race conditions to user
- Suggest fix strategies (add locks, use channels, immutable data)
- Generate fixed code with proper synchronization
- Auto-Fix Strategy:
- Level 1: Apply known patterns (e.g., add Mutex, use Arc, channel-based communication)
- Level 2: LLM generates fix with concurrency primitives
- Level 3: Sequential fallback if parallel execution is unsafe
- Performance: <5s (static analysis + GNN query)
- ⚡ Parallel Processing: Analyze multiple code paths for race conditions simultaneously

17. FixingIssues: Auto-retry with fixes if validation fails

- ⚡ Parallel Processing: Generate fixes for multiple independent issues simultaneously

18. Complete: Code ready for testing

- Post-MVP: Explicit FileLockRelease state after completion

Failed: Human intervention required

- Post-MVP: Explicit FileLockRelease state on failure

Performance Targets:

13. Task decomposition: <500ms (GNN analysis)
14. Dependency sequencing: <100ms (graph traversal)
15. Conflict check: <50ms (active work lookup - MVP) / <50ms (lock table query - Post-MVP)
16. Plan generation: <200ms (estimation algorithm)
17. File lock acquisition: <10ms (Post-MVP - database transaction)
18. Context assembly: <100ms (with parallel GNN queries)
19. Code generation: 2-5s (LLM dependent, parallel for multiple files)
20. GNN validation: <10ms (with parallel dependency path validation)
21. Browser validation: 5-10s (single component) / 10-15s (parallel multi-component)
22. Security scan: <10s (with parallel file scanning)
23. Concurrency validation: <5s (static analysis + GNN race detection)
24. File lock release: <5ms (Post-MVP)
25. Total cycle (MVP): <43s (adds concurrency validation: ~5s)
26. Total cycle (Post-MVP): <45s (adds explicit lock acquisition/release: ~15ms)
27. Parallel optimization: ~30-40% faster with concurrent execution of independent operations

New Prevention Guarantees:

28. ✅ Explicit Planning: All tasks identified before execution (PDC 2.1)
29. ✅ Work Visibility (MVP): Show which developer is working on which files (PDC 3.3)
30. ✅ Conflict Prevention (Post-MVP): File locks acquired before any edits (PDC 3.3)
31. ✅ Dependency-Aware Execution: Tasks executed in correct order (PDC 2.1)
32. ✅ Resource Coordination (Post-MVP): No two developers edit same file (PDC 3.3)
33. ✅ Concurrency Safety: Race conditions detected and auto-fixed before testing (PDC 3.4 - NEW)

---

##### 3.4.2.1A Blast Radius Analysis & Preview (P0 Feature - MVP)

Purpose: Provide dependency-aware impact preview BEFORE executing changes, enabling informed decision-making and preventing unintended consequences.

Philosophy: "Show, don't surprise" - Users should know exactly what will be affected before committing to execution.

Integration Point: Code Generation State Machine → PlanReview State (after PlanGeneration, before execution)

Data Structure

pub struct BlastRadiusAnalysis {

    // Direct Impact

    pub files_to_modify: Vec`<FileImpact>`,

    pub critical_files: Vec`<PathBuf>`,  // Core business logic identified

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

    pub risk_level: RiskLevel,  // Low, Medium, High, Critical

    pub estimated_time: Duration,

    pub rollback_complexity: RollbackComplexity,  // Low, Medium, High

    pub affected_user_percentage: f32,  // Estimated % of API calls affected

    // Decision Factors

    pub requires_approval: bool,

    pub approval_reason: String,

}

pub struct FileImpact {

    pub path: PathBuf,

    pub is_critical: bool,

    pub change_type: ChangeType,  // Create, Modify, Delete

    pub dependent_count: usize,   // How many files depend on this

}

pub struct DependencyImpact {

    pub file: PathBuf,

    pub distance: usize,  // Degrees of separation (1=direct, 2=indirect, etc.)

    pub impact_reason: String,  // "Imports modified function calculate()"

}

pub struct TestImpact {

    pub test_file: PathBuf,

    pub test_name: String,

    pub coverage_type: TestCoverageType,  // Unit, Integration, E2E

    pub needs_update: bool,  // Signature changed, needs rewrite

}

pub struct ApiChange {

    pub endpoint: String,  // "/api/calculate"

    pub method: HttpMethod,  // POST, GET, etc.

    pub change_type: ApiChangeType,  // Added, Modified, Deprecated, Breaking

    pub breaking: bool,

}

pub struct PackageUpgrade {

    pub package: String,

    pub from_version: String,

    pub to_version: String,

    pub breaking_changes: Vec`<String>`,  // Known breaking changes from changelog

}

###### Analysis Algorithm

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

    impact: format!("Affects ~{:.1}% of API calls",

    self.estimate_api_usage_percentage(c)),

    })

    .collect(),

    package_upgrades: package_impacts.upgrades,

    package_conflicts: package_impacts.conflicts,

    risk_level,

    estimated_time: self.estimate_execution_time(&plan),

    rollback_complexity,

    affected_user_percentage: self.estimate_user_impact(&api_changes),

    requires_approval: self.should_require_approval(&critical_files, &api_changes, &risk_level),

    approval_reason: self.generate_approval_reason(&critical_files, &api_changes, &risk_level),

    })

    }

    fn identify_critical_files(&self, files: &[PathBuf]) -> Vec`<PathBuf>` {

    files.iter()

    .filter(|f| {

    // Heuristic 1: Hub nodes (>10 dependents)

    self.gnn.count_dependents(f).unwrap_or(0) > 10 ||

    // Heuristic 2: Core directories

    f.to_string_lossy().contains("/core/") ||

    f.to_string_lossy().contains("/models/") ||

    f.to_string_lossy().contains("/db/") ||

    // Heuristic 3: Naming patterns

    f.file_name()

    .and_then(|n| n.to_str())

    .map(|n| n.ends_with("_service.py") ||

    n.ends_with("_controller.py") ||

    n.ends_with("_model.py"))

    .unwrap_or(false) ||

    // Heuristic 4: Explicit marking in .yantra/critical_files.json

    self.is_marked_critical(f)

    })

    .cloned()

    .collect()

    }

    fn calculate_risk_level(

    &self,

    critical_files: &[PathBuf],

    downstream: &[DependencyImpact],

    api_changes: &[ApiChange],

    package_impacts: &PackageImpactAnalysis,

    ) -> RiskLevel {

    let mut score = 0;

    // Critical files: +20 points each

    score += critical_files.len() * 20;

    // Wide ripple effect: +1 point per indirect dependency

    score += downstream.len();

    // Breaking API changes: +30 points each

    score += api_changes.iter().filter(|c| c.breaking).count() * 30;

    // Package upgrades with breaking changes: +25 points each

    score += package_impacts.upgrades.iter()

    .filter(|u| !u.breaking_changes.is_empty())

    .count() * 25;

    // Package conflicts: +40 points each (very risky)

    score += package_impacts.conflicts.len() * 40;

    match score {

    0..=20 => RiskLevel::Low,

    21..=50 => RiskLevel::Medium,

    51..=100 => RiskLevel::High,

    _ => RiskLevel::Critical,

    }

    }

    fn should_require_approval(

    &self,

    critical_files: &[PathBuf],

    api_changes: &[ApiChange],

    risk_level: &RiskLevel,

    ) -> bool {

    // Always require approval for:

    !critical_files.is_empty() ||  // Touches critical files

    api_changes.iter().any(|c| c.breaking) ||  // Breaking API changes

    matches!(risk_level, RiskLevel::High | RiskLevel::Critical) ||  // High risk

    // Size-based (existing logic from PlanReview):

    false  // Size checks done separately in PlanReview state

    }

}

###### UI Display Format

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

###### Smart Display Logic

pub fn should_show_detailed_blast_radius(analysis: &BlastRadiusAnalysis) -> bool {

    // Always show detailed view for:

    !analysis.critical_files.is_empty() ||          // Critical files

    !analysis.breaking_changes.is_empty() ||        // Breaking changes

    analysis.affected_tests.len() > 20 ||           // Large test surface

    analysis.downstream_dependencies.len() > 10 ||  // Wide ripple

    !analysis.package_upgrades.is_empty() ||        // Package changes

    matches!(analysis.risk_level, RiskLevel::High | RiskLevel::Critical)

}

pub fn should_skip_blast_radius(analysis: &BlastRadiusAnalysis) -> bool {

    // Skip for trivial changes:

    analysis.files_to_modify.len() == 1 &&          // Single file

    analysis.downstream_dependencies.is_empty() &&  // No dependencies

    analysis.affected_tests.len() < 5 &&            // Few tests

    analysis.api_changes.is_empty() &&              // No API changes

    analysis.package_upgrades.is_empty() &&         // No packages

    matches!(analysis.risk_level, RiskLevel::Low)   // Low risk

}

###### Performance Targets

- GNN Queries: <500ms (parallel queries for dependents, tests, packages)
- Critical File Detection: <100ms (in-memory checks + config lookup)
- API Change Analysis: <300ms (if OpenAPI spec exists, otherwise skip)
- Package Impact Analysis: <1s (query Tech Stack GNN + check changelogs)
- Total Analysis Time: <2s (all operations)

###### Future Enhancements (Post-MVP)

P1: Live validation during DependencyValidation state (compare planned vs actual impact)

P2: Historical blast radius tracking (learn from past changes)

P2: ML-based user impact estimation (better than heuristic %)

P3: Blast radius diff view (compare before/after architecture changes)

##### 3.4.2.2. Testing Framework (MVP) - Multi-State Machine Architecture

Philosophy: Robust testing requires two specialized state machines working in concert:

1. Test Intelligence State Machine: Generates high-quality, effective tests with proper oracles
2. Test Execution State Machine: Executes tests with comprehensive validation and feedback

Overall Responsibility: Ensure generated code works correctly through intelligent test generation and rigorous execution

---

##### 3.4.2.2A. Test Intelligence State Machine (MVP)

Responsibility: Generate effective, high-quality tests that correctly verify behavior

States:

Phase 1: Test Oracle & Specification Extraction (MVP - Critical)

1. IntentSpecificationExtraction: Extract testable specifications from user's natural language intent

- Purpose: Solve the Test Oracle Problem - determine what "correct behavior" actually means
- Inputs: User's original intent, generated code, function signatures, docstrings
- Outputs: Structured behavioral specifications, expected outcomes, invariants
- Techniques:
  - Natural language processing of user intent → formal specifications
  - Contract extraction: preconditions, postconditions, invariants
  - Expected behavior synthesis from requirements
  - Example-based specification (user provides examples, system generalizes)
- ⚡ Parallel Processing: Extract specifications for multiple functions simultaneously
- Performance Target: <2s (LLM + NLP analysis)

2. TestOracleGeneration: Create test oracles that define correct behavior

- Purpose: Generate mechanisms to verify code correctness
- Strategies:
  - Specification-based oracles: Assert against extracted specifications
  - Differential testing oracles: Compare outputs across multiple implementations
  - Metamorphic oracles: Define input transformation → output relationship properties
  - Contract verification oracles: Verify preconditions/postconditions hold
- Output: Oracle functions, assertion templates, verification strategies
- ⚡ Parallel Processing: Generate oracles for different verification strategies simultaneously
- Performance Target: <3s (parallel oracle generation)

Phase 2: Test Data Generation (MVP - Systematic)

3. InputSpaceAnalysis: Analyze function input space and constraints

- Purpose: Understand valid/invalid input domains before generating test data
- Techniques:
  - Parse function signatures for type constraints
  - Extract preconditions from docstrings and contracts
  - Analyze code paths using GNN to identify branches
  - Identify edge cases from implementation (loops, conditionals, exceptions)
- Output: Input space model, constraint sets, boundary values
- ⚡ Parallel Processing: Analyze multiple functions concurrently
- Performance Target: <1s (GNN queries + static analysis)

4. TestDataGeneration: Generate comprehensive test inputs using multiple strategies

- Purpose: Create diverse, high-quality test data covering edge cases
- Strategies:
  - Boundary Value Analysis: Generate inputs at edges (0, -1, MAX_INT, MIN_INT, empty string, None, empty list)
  - Equivalence Partitioning: Representative values from each input class
  - Constraint-Based Generation: Use Z3/SMT solver to generate inputs satisfying preconditions
  - Fuzzing Integration: Random/mutated inputs for robustness (integrated with AFL, libFuzzer)
  - Property-Based Generation: Generate inputs for property-based tests (Hypothesis-style)
  - Pairwise Combination Testing: Cover parameter interactions efficiently

- Output: Test data sets categorized by strategy, expected coverage
- ⚡ Parallel Processing:
- Run different generation strategies simultaneously (boundary + fuzzing + constraint-based)
- Generate data for multiple test cases in parallel
- Performance Target: <5s (parallel generation with 4+ strategies)

Phase 3: Test Generation & Quality Assurance (MVP)

5. TestCaseGeneration: Generate actual test functions with proper structure

- Purpose: Create executable test code using generated data and oracles
- Techniques:
  - Unit test generation (pytest, jest format)
  - Integration test generation (API workflows, database interactions)
  - E2E test generation (Playwright scenarios)
  - Property-based test generation (Hypothesis/QuickCheck)
  - Mock generation for external dependencies

- Quality Checks:
- Testability scoring of generated code
- Automatic dependency injection patterns where needed
- Mock/stub generation alongside tests
- ⚡ Parallel Processing: Generate different test types simultaneously (unit + integration + E2E)
- Performance Target: <5s (LLM generation with parallel calls)

6. AssertionStrengthAnalysis: Verify assertions are strong and precise

- Purpose: Ensure tests actually catch bugs (not just "assert result is not None")
- Scoring System:
  - Weak (Score 1): assert result (just checks truthy)
  - Medium (Score 3): assert result is not None (checks existence)
  - Strong (Score 5): assert result == 42 (precise value check)
  - Very Strong (Score 7): assert result == 42 and result_type == int and 0 <= result <= 100 (multi-constraint)
- Action: Flag weak assertions, suggest improvements, auto-strengthen where possible
- ⚡ Parallel Processing: Analyze assertions across all test files concurrently
- Performance Target: <2s (static analysis of test code)

7. TestQualityVerification: Verify generated tests are actually effective (MVP - Critical)

- Purpose: Ensure tests will catch bugs (not just pass)
- Techniques:
  - Mutation Testing: Inject bugs into code, verify tests catch them
    - Operators: Change == to !=, + to -, remove conditions, flip booleans
    - Target: 80%+ mutation score (tests kill 80% of mutants)

  - Coverage-Guided Enhancement: Generate additional tests for uncovered paths
  - Use coverage reports to identify gaps
  - Generate targeted tests for uncovered branches
  - Assertion Sufficiency: Verify each test has meaningful assertions (not just "no crash")

- Output: Test effectiveness score, mutation kill rate, coverage gaps
- ⚡ Parallel Processing:
- Run mutation testing on multiple mutants simultaneously
- Parallel coverage analysis across modules
- Performance Target: <15s (parallel mutation testing with 50+ mutants)

Phase 4: Test Suite Management (MVP)

8. TestSuiteOrganization: Organize tests into logical suites with proper structure

- Purpose: Maintain clean, organized test codebase
- Actions:
  - Group tests by module, feature, test type (unit/integration/E2E)
  - Generate test fixtures and setup/teardown functions
  - Create test configuration files (pytest.ini, jest.config.js)
  - Generate test documentation and README
- ⚡ Parallel Processing: Organize multiple test suites concurrently
- Performance Target: <1s (file operations + generation)

9. TestImpactAnalysis: Determine which tests affected by code changes (MVP - Critical for Co-Evolution)

- Purpose: Enable test-code co-evolution and efficient re-testing
- Technique: Use GNN to map code changes → affected tests
  - Query GNN: "Which tests call this modified function?"
  - Query GNN: "Which tests cover this file?"
  - Analyze transitive dependencies (function A changed → function B calls A → tests for B affected)
- Output: List of affected tests, impact severity (direct/indirect), recommended actions (update/rewrite/keep)
- ⚡ Parallel Processing: Analyze impact for multiple changed files simultaneously
- Performance Target: <500ms (GNN queries with parallel lookups)

10. TestUpdateGeneration: Generate test updates when code changes (MVP - Test Co-Evolution)

- Purpose: Keep tests synchronized with code changes
- Triggers:
  - Function signature changed → Update test calls
  - Return type changed → Update assertions
  - Behavior modified → Regenerate test expectations

- Strategies:
- Preserve intent: Keep original test purpose, update implementation
- Incremental update: Modify only affected parts, not full rewrite
- Staleness detection: Flag tests that no longer match intent
- ⚡ Parallel Processing: Update multiple affected tests concurrently
- Performance Target: <3s (LLM updates with parallel calls)

11. Complete: High-quality test suite ready for execution

Failed: Unable to generate effective tests (escalate to human)

Entry Point: Generated code + user intent from CodeGen machine Exit Point: Comprehensive test suite with oracles + test quality report Trigger: Automatically after CodeGen succeeds (parallel with environment setup) Success Criteria:

1. Mutation score >80%
2. Assertion strength score >5 average
3. Coverage potential >90%
4. All tests have proper oracles

State Count: 10 states (MVP)

Performance Targets:

1. Intent specification extraction: <2s
2. Test oracle generation: <3s
3. Input space analysis: <1s
4. Test data generation: <5s (parallel strategies)
5. Test case generation: <5s (parallel LLM calls)
6. Assertion strength analysis: <2s
7. Test quality verification: <15s (mutation testing)
8. Test suite organization: <1s
9. Test impact analysis: <500ms (GNN)
10. Test update generation: <3s
11. Total Test Intelligence cycle: <37s

---

##### 3.4.2.2B. Test Execution State Machine (MVP)

Responsibility: Execute tests with comprehensive validation, debugging feedback, and resilience

States:

Phase 1: Environment & Execution Preparation

1. EnvironmentSetup: Create virtual environment, install dependencies

- ⚡ Parallel Processing: Install multiple independent test dependencies simultaneously
- Performance Target: 10-20s (with parallel pip/npm install)

2. FlakeDetectionSetup: Configure flaky test detection infrastructure (MVP - Critical)

- Purpose: Prevent non-deterministic tests from blocking autonomous loop
- Configuration:
  - Set retry count (default: 3 runs per test)
  - Configure timeout thresholds
  - Setup flake detection storage (SQLite)
- Performance Target: <500ms (configuration)

Phase 2: Test Execution with Instrumentation

3. UnitTesting: Run pytest/jest for function-level tests with execution tracing

- Execution:
  - Run tests with coverage instrumentation
  - Capture execution traces for failures (variable states, call stacks)
  - Run each test N times for flake detection

- ⚡ Parallel Processing:
- Run test files in parallel (pytest -n auto, jest --maxWorkers)
- Execute independent test suites simultaneously
- Performance: N test files × 5s = 5N sequential → ~N/4 parallel (4 workers)
- Instrumentation: Python sys.settrace(), JavaScript source maps
- Performance Target: <30s (parallel execution with 4+ workers)

4. IntegrationTesting: Test API integrations and data flows with contract verification

- Execution:
  - Test API endpoints with contract validation (OpenAPI/Pact)
  - Database integration tests
  - External service mocking

- ⚡ Parallel Processing:
- Test multiple API endpoints simultaneously
- Parallel database connection tests
- Concurrent integration scenario execution
- Performance Target: 20-40s (with parallel API/DB testing)

5. BrowserTesting: Run E2E tests with Playwright (user interactions)

- Execution:
  - Full user workflows (login→dashboard→logout)
  - Responsive design tests (multiple viewports)
  - Accessibility tests (a11y audits)
  - Visual regression tests (snapshot comparison)

- ⚡ Parallel Processing:
- Run multiple E2E test scenarios in parallel browsers
- Test different user flows simultaneously
- Parallel execution across different browser contexts
- Performance: M scenarios × 30s = 30M sequential → ~30M/P parallel (P = parallel browsers)
- Performance Target: 30-60s (sequential) / 15-30s (parallel with 2-3 browser contexts)

6. PropertyBasedTesting: Execute property-based tests (Hypothesis/QuickCheck style) (MVP)

- Purpose: Catch edge cases automatically through property verification
- Execution:
  - Run property-based tests with random input generation
  - Shrink counterexamples to minimal failing cases
  - Verify properties hold for 100+ generated inputs

- Properties Tested:
- Invariants (e.g., sorted(list) is always ascending)
- Roundtrip properties (e.g., decode(encode(x)) == x)
- Metamorphic relations (e.g., f(2*x) == 2*f(x))
- ⚡ Parallel Processing: Run property tests for different functions concurrently
- Performance Target: <20s (100 examples per property, parallel execution)

Phase 3: Analysis & Feedback Generation

7. ExecutionTraceAnalysis: Analyze execution traces for failures (MVP - Critical)

- Purpose: Provide rich debugging context to LLM for effective fixes
- Captured Data:
  - Variable states at failure: All local/global variables at crash point
  - Execution trace: Full call stack with line numbers
  - Data flow: How values propagated through execution
  - Symbolic constraints: What constraints were violated (using symbolic execution)

- Output: Structured debug report with precise failure context
- Example Enhancement:
- Before: "Error: IndexError: list index out of range"
- After: "Error: IndexError at line 42 in calculate(). Variable 'idx'=10, but 'data' has length 8. Execution trace: main() → process() → calculate(). Root cause: Loop range exceeded list bounds."
- ⚡ Parallel Processing: Analyze traces for multiple failures concurrently
- Performance Target: <2s (trace parsing + analysis)

8. FlakeDetectionAnalysis: Identify and classify flaky tests (MVP - Critical)

- Purpose: Quarantine non-deterministic tests to prevent pipeline blocking
- Detection Logic:
  - Run each test 3 times
  - Test passes 3/3 → Stable ✅
  - Test fails 3/3 → Real failure 🔴
  - Test passes 1-2/3 → Flaky ⚠️ (quarantine)

- Flake Root Cause Analysis:
- Timing issues: Race conditions, sleep/timeout dependencies
- Ordering issues: Test execution order dependency
- State issues: Shared mutable state, improper cleanup
- External dependencies: Network, filesystem, time-based
- Action: Automatically quarantine flaky tests, flag for manual review
- ⚡ Parallel Processing: Detect flakes across test suite simultaneously
- Performance Target: <5s (3 runs per test, parallel execution)

9. CoverageAnalysis: Measure test coverage with gap identification

- Metrics:
  - Line coverage, branch coverage, path coverage
  - Function coverage, class coverage
  - Mutation coverage (from Test Intelligence phase)
- Gap Analysis: Identify uncovered code paths for targeted test generation
- ⚡ Parallel Processing: Analyze coverage for multiple modules concurrently
- Performance Target: <3s (coverage tool integration)

10. SemanticCorrectnessVerification: Verify code fulfills user intent (MVP - Architecture Gap)

- Purpose: Ensure code actually does what user asked (beyond syntax/types)
- Verification Strategies:
  - Intent-to-implementation matching: Compare generated code behavior against extracted specifications
  - Behavioral specification checking: Verify expected behaviors from intent hold
  - Regression against intent: When modifying code, ensure original intent still met
  - Differential testing: Compare behavior with reference implementations or previous versions

- Technique:
- Run tests with captured intent specifications as additional assertions
- Use LLM to reason about semantic equivalence
- Query user for confirmation on ambiguous cases
- Output: Semantic correctness score, intent alignment report, violations
- ⚡ Parallel Processing: Verify semantic correctness for multiple functions concurrently
- Performance Target: <5s (LLM reasoning + specification matching)

Phase 4: Error Classification & Learning

11. ErrorClassificationAndLearning: Categorize errors and extract fix patterns (MVP - Enhanced)
    - Purpose: Enable effective auto-retry and learn from failures
    - Structured Error Taxonomy:
      - Syntax Errors: Missing parentheses, indentation, typos
      - Type Errors: Type mismatches, attribute errors
      - Logic Errors: Off-by-one, wrong operators, incorrect conditionals
      - Runtime Errors: IndexError, KeyError, DivisionByZero
      - Integration Errors: API failures, database errors, network issues
      - Test Design Errors: Flaky tests, weak assertions, missing setup

    - Root Cause Inference (not just symptom matching):
    - Use execution traces to identify actual root cause
    - Distinguish between symptom (IndexError) and cause (incorrect loop bounds)
    - Track error propagation through call stack
    - Fix Pattern Extraction:
    - Learn from successful repairs in knowledge base
    - Extract patterns: "IndexError in loop → check loop bounds"
    - Store in Vector DB for future retrieval
    - Output: Error classification, root cause, suggested fix patterns, confidence score
    - ⚡ Parallel Processing: Classify multiple errors concurrently
    - Performance Target: <3s (pattern matching + LLM reasoning)

Phase 5: Fixing & Validation

12. FixingIssues: Apply fixes and re-run tests with learning

- Fix Strategies (prioritized):

1. Pattern-based fixes: Apply known fixes from error taxonomy
2. Trace-guided fixes: Use execution traces to generate targeted fixes
3. LLM-generated fixes: With full context (error + trace + intent)
4. Mutation-guided fixes: If mutation testing revealed weakness, strengthen tests

- Learning: Store successful fixes in knowledge base for future use
- ⚡ Parallel Processing: Fix and re-test multiple independent failures simultaneously
- Performance Target: <10s (fix generation + re-execution)

13. TestCodeCoEvolutionCheck: Verify tests remain aligned with code (MVP)

- Purpose: Ensure tests haven't become stale after code changes
- Checks:
- Do tests still compile/run?
- Do tests still match intent specifications?
- Are assertions still appropriate for new implementation?
- Action: If stale, trigger TestUpdateGeneration in Test Intelligence machine
- Performance Target: <1s (GNN queries + static analysis)

Phase 6: Results & Reporting

14. Complete: All tests pass with adequate coverage and semantic correctness

- Report: Test results, coverage metrics, mutation score, semantic correctness score, flaky tests quarantined

Failed: Tests failed after maximum retries

- Report: Detailed failure analysis with execution traces, root causes, suggested fixes

Entry Point: Test suite from Test Intelligence machine + generated code

Exit Point: Test results + coverage report + execution traces + learning data

Trigger: Automatically after Test Intelligence completes

Success Criteria:

- 100% stable tests pass (flaky tests quarantined)
- Coverage >80%
- Semantic correctness verified
- No critical errors

State Count: 13 states (MVP)

Performance Targets:

- Environment setup: 10-20s (parallel install)
- Flake detection setup: <500ms
- Unit testing: <30s (parallel, 4+ workers)
- Integration testing: 20-40s (parallel)
- Browser testing: 15-30s (parallel, 2-3 browsers)
- Property-based testing: <20s (parallel)
- Execution trace analysis: <2s
- Flake detection analysis: <5s (3 runs, parallel)
- Coverage analysis: <3s
- Semantic correctness verification: <5s
- Error classification: <3s
- Fixing issues: <10s
- Test-code co-evolution check: <1s
- Total Test Execution cycle: <2 minutes (with parallel optimizations)

---

##### 3.4.2.2.1 Testing Framework Integration & Workflow

Parallel Execution (Both Machines Run Concurrently):

CodeGen Complete

    ↓

    ├───── ─────────────────┐

    ↓                                                           ↓

Test Intelligence Machine Environment Setup (Execution Machine)

(Generates tests) (Prepares runtime)

    ↓                                 ↓

(~37s to complete) (~15s to complete)

    ↓                                 ↓

    └────────────┘

    ↓

    Test Execution Machine

    (Executes generated tests)

    ↓

    (~2 minutes)

    ↓

    Results + Learning

Total Testing Framework Cycle: ~2.5 minutes (with parallel optimization)

Task Tracking for Parallel Operations (MVP - Critical):

When running multiple tasks in parallel, the system tracks:

- Task Registry: SQLite table tracking all active tasks
  - Task ID, state machine, current state, start time, dependencies

- Context Preservation: Each task maintains isolated context
- Test suite being worked on
- Code under test
- User intent specifications
- Intermediate results
- Dependency Management: Tasks track dependencies using GNN
- Test Intelligence must complete before Test Execution
- Parallel test generations track shared resources
- Progress Monitoring: UI shows all parallel tasks with real-time updates
- Failure Isolation: One task failure doesn't affect others
- Context Recovery: If agent loses context, reload from task registry

Auto-Trigger: Yes - both machines start immediately after CodeGen completes Manual Trigger: Also available via UI button

---

##### 3.4.2.2.1 Testing Framework Data Structures

pub struct TestIntelligenceResult {

    pub test_suite: TestSuite,

    pub oracles: Vec`<TestOracle>`,

    pub test_quality_report: TestQualityReport,

    pub specifications: Vec`<BehavioralSpecification>`,

}

pub struct TestOracle {

    pub function_name: String,

    pub oracle_type: OracleType,  // Specification, Differential, Metamorphic, Contract

    pub assertions: Vec`<Assertion>`,

    pub verification_strategy: VerificationStrategy,

}

pub struct TestQualityReport {

    pub mutation_score: f32,  // % of mutants killed (target: >80%)

    pub assertion_strength: f32,  // Average strength score (target: >5)

    pub coverage_potential: f32,  // % of code covered by tests (target: >90%)

    pub property_tests: usize,

    pub weak_tests: Vec`<WeakTestReport>`,

}

pub struct ExecutionTraceReport {

    pub test_name: String,

    pub failure_point: CodeLocation,

    pub variable_states: HashMap<String, String>,

    pub call_stack: Vec`<StackFrame>`,

    pub execution_path: Vec`<CodeLocation>`,

    pub symbolic_constraints: Vec`<Constraint>`,

    pub root_cause: RootCause,

}

pub struct FlakeAnalysisReport {

    pub flaky_tests: Vec`<FlakyTest>`,

    pub quarantined_count: usize,

    pub flake_rate: f32,

}

pub struct FlakyTest {

    pub test_name: String,

    pub pass_rate: f32,  // e.g., 2/3 = 0.67

    pub root_cause: FlakeRootCause,  // Timing, Ordering, State, External

    pub recommended_fix: String,

}

pub struct SemanticCorrectnessReport {

    pub correctness_score: f32,  // 0-100%

    pub intent_alignment: IntentAlignment,

    pub violations: Vec`<IntentViolation>`,

    pub differential_results: Option`<DifferentialTestResult>`,

}

---

##### 3.4.2.2.3 Post-MVP Testing Enhancements

| Multi-Model Test Diversity | Test Intelligence | +1 state         | Test generation quality (uses 3 LLMs) |
| -------------------------- | ----------------- | ---------------- | ------------------------------------- |
| Performance Regression     | Test Execution    | +1 state         | New test type (benchmarking)          |
| API Contract Testing       | Test Execution    | Enhance existing | Adds to IntegrationTesting state      |
| Snapshot Testing           | Test Execution    | +1 state         | New test type (UI snapshots)          |
| Test Prioritization        | Test Intelligence | +1 state         | Test scheduling/ordering logic        |

Phase 2 (Months 3-4):

11. Multi-Model Test Diversity

- Issue: Tests generated by single LLM may have blind spots
- Solution: Use different LLMs for test generation (Claude, GPT-4, Gemini), merge unique cases
- Implementation: Generate tests with 3 LLMs in parallel, deduplicate, keep unique tests
- Benefit: 20-30% more edge cases caught

12. Performance Regression Testing

- Issue: No performance benchmarks currently
- Solution: Add benchmark generation and regression detection
- Metrics: Execution time, memory usage, API latency
- Detection: Flag >10% performance degradation
- State: Add PerformanceBenchmarking state to Execution machine

13. API Contract Testing

- Issue: No explicit contract validation for APIs
- Solution: Generate Pact/OpenAPI contract tests
- Verification: Request/response schemas, status codes, headers
- State: Enhance IntegrationTesting state with contract validation

14. Snapshot Testing

- Issue: Useful for UI components, not currently mentioned
- Solution: Add snapshot generation for React/SolidJS components
- Tools: Jest snapshots, Playwright screenshots
- State: Add SnapshotTesting state to Execution machine

15. Test Prioritization

- Issue: Run all tests equally, no prioritization
- Solution: Prioritize tests by change impact and historical failure rate
- Strategy:
  - Run tests for changed code first (GNN impact analysis)
  - Prioritize historically flaky/failing tests
  - Run fast tests before slow tests
  - Skip tests with no code impact (when safe)
- Benefit: 40-50% faster feedback in CI/CD

State Count (Post-MVP):

- Test Intelligence: 11 states (+2 for multi-model diversity, test prioritization)
- Test Execution: 16 states (+3 for performance, snapshots, contract testing)

##### 3.4.2.3. Deployment State Machine (MVP - Railway Focus)

Responsibility: Deploy validated code to Railway.app

States:

1. PackageBuilding: Create Docker image or build artifacts
   - ⚡ Parallel Processing: Build multiple layers/stages simultaneously (Docker multi-stage builds)
2. ConfigGeneration: Generate railway.json, Dockerfile, environment config

- ⚡ Parallel Processing: Generate multiple config files simultaneously

1. RailwayUpload: Push code/image to Railway

- ⚡ Parallel Processing: Parallel upload of multiple artifacts/layers

1. HealthCheck: Verify deployed service is responding

- ⚡ Parallel Processing:
- Check multiple endpoints simultaneously
- Parallel health checks for different service components
- Concurrent smoke tests across multiple routes

1. RollbackOnFailure: Auto-rollback if health check fails
2. Complete: Service live and healthy

Failed: Deployment failed

Entry Point: Code + passing tests Exit Point: Live Railway URL + health status Trigger: Manual (button click in UI) - requires user approval Success Criteria: Service responds to health check (HTTP 200)

MVP Scope:

- Platform: Railway.app only (no AWS, GCP, Kubernetes in MVP)
- Environments: Single production environment (no staging in MVP)
- Rollback: Manual (automatic rollback post-MVP)
- Monitoring: Basic health checks only (no metrics/logs in MVP)

Performance Targets:

- Package building: 30-60s (Docker build with multi-stage parallel builds)
- Upload to Railway: 20-40s (parallel artifact upload)
- Health check: 5-10s (parallel endpoint checks)
- Total deployment: <2 minutes
- Parallel benefit: ~20-30% faster with concurrent operations

Auto-Trigger: No - requires user approval for safety Post-MVP: Optional auto-deploy after tests pass

##### 3.4.2.4 Maintenance State Machine (Post-MVP) - Complete Details

Responsibility: Monitor production, detect issues, auto-fix, and deploy patches

Status: NOT in MVP - design complete, implementation post-MVP
Rationale: Focus MVP on working code generation/testing/deployment first

Entry Point: Deployed application in production
Exit Point: Incident resolved or escalated
Trigger: Automatic based on error detection
Success Criteria: Issue resolved without human intervention

States (11 Total)

1. LiveMonitoring

Continuous monitoring of production errors and performance

⚡ Parallel Processing:

- Monitor multiple services/endpoints simultaneously
- Parallel log aggregation from multiple sources
- Concurrent metric collection across service fleet

What it monitors:

- Error rates and types
- Performance metrics (latency, throughput)
- Resource usage (CPU, memory, disk)
- API response times
- Database query performance

2. BrowserValidation

Monitor real user browser sessions for errors

⚡ Parallel Processing:

- Track multiple user sessions concurrently
- Parallel session replay processing
- Concurrent error pattern detection across users

Purpose: "Is it still working for real users?" - Production monitoring

Tools: Sentry-like error tracking, LogRocket-like session replay

Actions:

- Real User Monitoring (RUM)
- Track browser errors in production (JavaScript errors, network failures)
- Performance degradation detection
- Session replay for debugging
- API failures affecting UI

User Experience: Invisible to users, proactive issue detection

3. ErrorAnalysis

Pattern detection in production errors

⚡ Parallel Processing:

- Analyze multiple error types simultaneously
- Parallel pattern matching across error logs
- Concurrent root cause analysis for different error classes

Techniques:

- Pattern matching against known issues (from Knowledge Base)
- Error frequency analysis
- Impact assessment (how many users affected)
- Correlation with deployments/changes

4. IssueDetection

Identify root causes using error patterns

⚡ Parallel Processing:

- Query multiple data sources simultaneously (logs, metrics, traces)

Data Sources:

- Application logs
- System metrics
- Distributed traces
- Database query logs
- Network traffic
- User reports

Output: Root cause hypothesis with confidence score

5. AutoFixGeneration

Generate fix using LLM + historical patterns

⚡ Parallel Processing:

- Generate multiple fix candidates simultaneously
- Evaluate candidates in parallel

Fix Generation Strategy:

1. Query Vector DB for similar past issues
2. LLM generates patch based on error context + historical fixes
3. Multiple fix candidates generated
4. Each candidate scored for likelihood of success

Performance Target: 10-30s (LLM, parallel candidate generation)

6. FixValidation

Test fix in staging using CodeGen + Testing machines

⚡ Parallel Processing:

- Leverages parallel processing from CodeGen and Testing machines

Validation Process:

1. Apply fix to staging environment
2. Run full test suite (uses Test Intelligence + Test Execution machines)
3. Verify fix resolves the issue
4. Ensure no new issues introduced

Performance Target: 2-3 minutes (full test suite with parallel execution)

7. CICDPipeline

Automated deployment of validated fix

⚡ Parallel Processing:

- Leverages parallel processing from Deployment machine

Actions:

1. Build Docker image
2. Push to Railway/production
3. Deploy with zero-downtime strategy (blue-green, canary)
4. Monitor deployment health

Performance Target: 1-2 minutes (parallel deployment pipeline)

8. VerificationCheck

Confirm issue is resolved in production

⚡ Parallel Processing:

- Check error rates across multiple regions simultaneously
- Parallel verification across different service instances
- Concurrent user impact assessment

Verification Criteria:

- Error rate drops to baseline
- No new errors introduced
- Users reporting issue confirm fix
- Performance metrics stable
- All health checks pass

Performance Target: <30s (parallel checks across regions)

9. LearningUpdate

Update knowledge base with new patterns

⚡ Parallel Processing:

- Update multiple knowledge stores concurrently (Vector DB, documentation, metrics)

What Gets Updated:

- Error pattern → Solution mapping in Vector DB
- Documentation with new known issue
- Metrics dashboard (MTTR for this error type)
- Team notification/report
- Incident log for audit trail

Purpose: Learn from every incident to improve future response

10. Active (Normal State)

Normal monitoring state - no incidents

Behavior:

- Continuous monitoring active
- Alert thresholds configured
- Ready to transition to Incident state if issue detected

11. Incident (Active Incident State)

Active incident being handled

Behavior:

- Cycles through states 3-9 (ErrorAnalysis → LearningUpdate)
- Human can intervene at any point
- Escalation if auto-fix fails after N attempts

###### 3.4.4.1 Self-Healing Loop:

1. **Detect**: Production error occurs (browser crash, API failure, performance degradation)
2. **Analyze**: Pattern matching against known issues, root cause analysis
3. **Generate Fix**: LLM generates patch based on error context
4. **Validate Fix**: Run through CodeGen + Testing machines (full validation)
5. **Deploy**: Automated CI/CD pipeline pushes fix to production
6. **Verify**: Confirm error rate drops, users unaffected
7. **Learn**: Update knowledge base with new error pattern + solution

Performance Targets:

- Error detection: Real-time (<1s, parallel monitoring across services)
- Pattern analysis: <5s (parallel pattern matching)
- Fix generation: 10-30s (LLM, parallel candidate generation)
- Fix validation: 2-3 minutes (full test suite with parallel execution)
- Deployment: 1-2 minutes (parallel deployment pipeline)
- Verification: <30s (parallel checks across regions)
- Total MTTR (Mean Time To Repair): <5 minutes for known patterns
- **Parallel benefit**: ~50-60% faster incident resolution with concurrent operations

---

#### 3.4.2A Parallel Processing Implementation Guide

Yantra implements parallel processing at three levels:

1. State-Level Parallelism: Within a single state, execute independent operations concurrently
2. Machine-Level Parallelism: Multiple state machines running simultaneously (Post-MVP)
3. Cross-Machine Parallelism: Different machines processing different features (Future)

##### 3.4.2A.1 State-Level Parallel Processing Patterns

Pattern 1: Parallel Resource Access (Browser Tabs, API Calls)

Example: DependencyAssessment State

├─ Tab 1: PyPI official page → Extract version, release date

├─ Tab 2: Changelog/release notes → Extract breaking changes

├─ Tab 3: Migration documentation → Extract upgrade guide

└─ Tab 4: GitHub issues → Extract known problems

Consolidation: LLM synthesizes all information into recommendation

Performance: 12s sequential → 4-5s parallel (60% reduction)

Pattern 2: Parallel Validation (Security, Tests, Health Checks)

Example: SecurityScanning State

├─ Worker 1: Semgrep scan on files 1-10

├─ Worker 2: Semgrep scan on files 11-20

├─ Worker 3: Secrets detection (all files)

└─ Worker 4: CVE database checks (all dependencies)

Consolidation: Merge results, prioritize by severity

Performance: N files × 1s = N sequential → N/4 parallel (with 4 workers)

Pattern 3: Parallel Generation (Code, Tests, Configs)

Example: CodeGeneration State

├─ LLM Call 1: Generate model.py (independent file)

├─ LLM Call 2: Generate controller.py (independent file)

├─ LLM Call 3: Generate test_model.py (independent test)

└─ LLM Call 4: Generate test_controller.py (independent test)

Dependencies: Files with dependencies generated sequentially after independent ones

Performance: 4 files × 5s = 20s sequential → ~10s parallel (2 concurrent LLM calls)

Pattern 4: Parallel Testing (Unit, E2E, Integration)

Example: UnitTesting State

├─ Test Runner 1: tests/unit/test_auth.py

├─ Test Runner 2: tests/unit/test_database.py

├─ Test Runner 3: tests/unit/test_api.py

└─ Test Runner 4: tests/unit/test_models.py

Tool: pytest -n auto (uses pytest-xdist for parallel execution)

Performance: 100 tests × 0.5s = 50s sequential → ~15s parallel (4 workers)

##### 3.4.2A.2 Implementation Technologies

Rust Async/Await (Tokio)

- Primary concurrency model for all parallel operations
- Lightweight tasks (green threads) for high concurrency
- Non-blocking I/O for network/file operations

// Example: Parallel browser tab execution

async fn fetch_dependency_info(package: &str) -> Result `<DependencyInfo>` {

    let (pypi_info, changelog, migration, issues) = tokio::join!(

    fetch_pypi_page(package),       // Tab 1

    fetch_changelog(package),        // Tab 2

    fetch_migration_guide(package),  // Tab 3

    fetch_github_issues(package),    // Tab 4

    );

    consolidate_results(pypi_info?, changelog?, migration?, issues?)

}

Connection Pooling

- HTTP connection pool for API calls (reduces connection overhead)
- Database connection pool for concurrent queries
- Browser context pool for parallel testing

Rate Limiting

- Respect API rate limits (e.g., GitHub, PyPI)
- Backoff strategy for LLM calls (avoid 429 errors)
- Circuit breaker pattern for failing services

##### 3.4.2A.3 Parallel Processing Trade-offs

| Operation Type                                                         | Sequential Time | Parallel Time | Workers      | Trade-off                 |
| ---------------------------------------------------------------------- | --------------- | ------------- | ------------ | ------------------------- |
| ---------------------------------------------------------------------- |                 |               |              |                           |
| Web Search (4 tabs)                                                    | 12s             | 4-5s          | 4 tabs       | Memory: +400MB per tab    |
| -                                                                      | -               | -             | -            | -                         |
| Security Scan (20 files)                                               | 20s             | 5s            | 4 workers    | CPU: +300% during scan    |
| Unit Tests (100 tests)                                                 | 50s             | 15s           | 4 workers    | Memory: +200MB per worker |
| E2E Tests (5 flows)                                                    | 150s            | 50s           | 3 browsers   | Memory: +1GB per browser  |
| LLM Code Gen (4 files)                                                 | 20s             | 10s           | 2 concurrent | Cost: 2x API calls/min    |

Resource Limits (MVP):

- Browser tabs: Max 5 concurrent (memory constraint)
- Test workers: Max 4 concurrent (CPU constraint)
- LLM calls: Max 3 concurrent (rate limit constraint)
- File scanners: Max 4 concurrent (I/O constraint)

Monitoring Parallel Operations:

- Track resource usage (CPU, memory, network)
- Log start/end timestamps for each parallel task
- Detect and retry failed parallel operations
- UI progress indicators show parallel task status

When NOT to Use Parallel Processing

1. Data Dependencies: Task B depends on Task A's output → Must be sequential
2. Resource Contention: Same file being read/written → Use locks or sequential access
3. Small Operations: Overhead > benefit (e.g., <10ms operations)
4. Rate Limited APIs: Would trigger rate limits → Use sequential with delays
5. Stateful Operations: Order matters (e.g., database migrations) → Sequential only

##### 3.4.2A.4 Concurrency Safety Validation (Built-in Protection)

Problem: Parallel processing can introduce race conditions, deadlocks, and data races if not properly managed.

Solution: Automated concurrency validation as part of the agentic flow (ConcurrencyValidation state).

Three-Layer Detection Strategy:

Layer 1: Static Analysis (Automated - No LLM Required)

Language-Specific Linters:

├─ Rust: cargo clippy --all -- -D warnings (thread safety built into type system)

│ └─ Detects: Send/Sync violations, missing Mutex, Arc misuse

├─ Python: threading analyzer (Python 3.12+)

│ └─ Detects: GIL contention, shared mutable state, missing locks

└─ JavaScript: ESLint concurrency rules

└─ Detects: Promise race conditions, callback hell, async/await misuse

Pattern Detection (Semgrep custom rules):

├─ Shared mutable state without locks

├─ Check-then-act patterns (time-of-check-time-of-use)

├─ Missing synchronization on concurrent writes

└─ Circular wait conditions (potential deadlocks)

Layer 2: GNN Data Flow Analysis (Automated - No LLM Required)

GNN Queries:

├─ Find all paths between concurrent writes to same variable

├─ Detect read-write conflicts across async boundaries

├─ Identify shared resources accessed from multiple threads/tasks

└─ Calculate happens-before relationships

Example Race Detection:

┌──────────────────────────────────────────────────────────┐

│ Thread 1: read counter → increment → write counter

│ Thread 2: read counter → increment → write counter │

│ │

│ GNN detects: Both threads read same variable without lock│

│ Race condition: Lost updates (counter++ not atomic) │

└──────────────────────────────────────────────────────────┘

Layer 3: Stress Test Generation (Automated - LLM Assisted)

Automatically generate tests that:

├─ Run concurrent operations 1000+ times

├─ Use different thread/task interleavings

├─ Introduce random delays to expose races

└─ Assert on invariants (e.g., counter == expected_value)

Example Generated Test (Python):

````python

def test_concurrent_counter_race():

    counter = Counter()  # Class being tested

    threads = [Thread(target=counter.increment) for _ in range(100)]

    for t in threads: t.start()

    for t in threads: t.join()

    assert counter.value == 100  # Should fail if race exists


Auto-Fix Strategy (Tiered Approach):


Level 1: Pattern-Based Auto-Fix (No LLM - Fast)

├─ Detected: Shared mutable state without lock

│ └─ Fix: Wrap in Mutex<T> (Rust) or threading.Lock (Python)

├─ Detected: Check-then-act pattern

│ └─ Fix: Use atomic operations or lock entire section

└─ Detected: Missing Arc for shared data

└─ Fix: Wrap in Arc<Mutex<T>> for cross-thread sharing


Level 2: LLM-Assisted Fix (With Context - Smart)

├─ Provide LLM with:

│ ├─ Detected race condition explanation

│ ├─ Code snippet with race

│ ├─ GNN data flow paths showing conflict

│ └─ Language-specific synchronization primitives

├─ LLM generates:

│ ├─ Fixed code with proper locks/channels/atomics

│ ├─ Explanation of fix strategy

│ └─ Test case to verify fix

└─ Agent validates fix with stress tests


Level 3: Sequential Fallback (Safe Default)

├─ If parallel execution cannot be made safe:

│ ├─ Rewrite parallel code as sequential

│ ├─ Add TODO comment explaining why

│ └─ Log performance regression for future optimization

└─ Guarantee: Code always works correctly, even if slower


**Integration with State Machine:**


CodeGeneration State ↓ (generates parallel code) ↓ DependencyValidation State ↓ BrowserValidation State ↓ SecurityScanning State ↓ ConcurrencyValidation State ← (Mandatory for parallel code) ├─ Static analysis (<1s) ├─ GNN race detection (<2s) ├─ Stress test generation (<2s) └─ Auto-fix if issues found ↓ (all validations pass) ↓ Complete State


Performance Impact:


- Static analysis: ~1s (parallel execution of linters)

- GNN analysis: ~2s (parallel data flow queries)

- Test generation: ~2s (LLM generates stress tests)

- **Total: ~5s** (acceptable overhead for safety guarantee)


Only runs if code uses async/await, threads, or parallel patterns

**Skip Conditions (Optimization):**

- No async/await keywords detected

- No threading/multiprocessing imports

- No tokio::spawn or similar parallel primitives

- Pure sequential code → Skip validation (0s overhead)


User Experience:


[Agent]: "Analyzing parallel code for race conditions..."

[Agent]: "⚠️ Detected potential race: Counter incremented without lock"

[Agent]: "🔧 Auto-fixing: Adding Mutex to protect shared state"

 [Agent]: "✅ Fix validated with 1000 concurrent operations" [Agent]: "⏱️ Concurrency validation: 4.2s"


**Why This is Agentic Flow, Not Just LLM:**


1. **Deterministic Checks**: Static analysis and GNN queries don't depend on LLM reasoning

2. **Fast**: Most issues caught by rule-based systems (<2s)

3. **Reliable**: Known patterns detected 100% of the time (not probabilistic)

4. **LLM as Assistant**: Only used for complex fixes and explanations

5. **Always Safe**: Sequential fallback guarantees correctness


**Comparison: Agentic Flow vs LLM-Only:**


| Aspect | Agentic Flow (Our Approach) | LLM-Only Approach |

|--------|----------------------------|-------------------|

| Speed | <5s (static + GNN + LLM) | 10-20s (LLM reasoning) |

| Reliability | High (rule-based detection) | Medium (LLM might miss patterns) |

| Coverage | 100% (deterministic rules) | 70-90% (LLM dependent) |

| Explainability | Precise (shows exact race) | Vague ("might have race") |

| Auto-fix | 3-tier (pattern → LLM → fallback) | LLM-only (no guaranteed fix) |

| Cost | Low (mostly static analysis) | High (large LLM prompts) |


Future Enhancements:


- Dynamic race detection (ThreadSanitizer, Valgrind integration)

- Record-replay debugging for non-deterministic races

- Formal verification for critical sections (TLA+, model checking)

- ML-based race prediction from code patterns



Future Enhancements (Post-MVP)


Machine-Level Parallelism

- CodeGen + Testing machines running simultaneously on different features

- Deploy machine running while CodeGen works on next feature

- Maintenance machine continuously monitoring while other machines work


Cross-Machine Parallelism:


- Multiple developers, multiple features, all machines working concurrently

- Team of agents architecture (master-servant pattern)

- Git coordination branch for conflict-free parallel work


**Advanced Parallel Patterns**:


- Map-reduce for large codebases (analyze 1000+ files)

- Stream processing for continuous monitoring

- Distributed execution across multiple machines




#### 3.4.2B PDC Phases to State Machines Mapping


PDC Phase 1: ARCHITECT/DESIGN

└─> CodeGen States: ArchitectureGeneration, ArchitectureReview (⚠️ approval gate), DependencyAssessment, ContextAssembly

└─> Prevents: Boundary violations, circular deps, incompatible versions, security by design

└─> Approval Gate: ArchitectureReview (mandatory human-in-loop)


PDC Phase 2: PLAN (🆕 ENHANCED - 6 states)

└─> CodeGen States: TaskDecomposition, DependencySequencing, ConflictCheck, PlanGeneration, PlanReview (⚠️ optional approval), EnvironmentSetup

└─> Prevents: Missing tasks, wrong sequencing, scope creep, environment errors

└─> MVP Prevention: Work visibility (show who's working on what files)

└─> Post-MVP Prevention: File conflicts (explicit locking mechanism)

└─> Approval Gate: PlanReview (optional for >5 tasks or multi-file changes)

└─> New Prevention: Explicit planning prevents unbounded work


PDC Phase 3: EXECUTE

└─> CodeGen States (MVP): ContextAssembly, CodeGeneration (with work visibility), DependencyValidation, BrowserValidation, SecurityScanning, FixingIssues

└─> Prevents: Syntax/type/logic errors, breaking changes, security issues

└─> Work Visibility: UI shows which files are being modified and by whom

└─> Auto-triggers Testing Machine on Complete

└─> CodeGen States (Post-MVP): FileLockAcquisition, ContextAssembly, CodeGeneration, DependencyValidation, BrowserValidation, SecurityScanning, FixingIssues, FileLockRelease

└─> Additional Prevention: Explicit file locks prevent merge conflicts before they happen

└─> Testing States: TestGeneration, EnvironmentSetup, UnitTesting, BrowserTesting, IntegrationTesting, CoverageAnalysis, FixingIssues

└─> Prevents: Regression bugs, missing coverage, integration failures

└─> Quality Gate: All tests must pass (no human approval, blocks progress)


PDC Phase 4: DEPLOY

└─> Deployment States: PackageBuilding, ConfigGeneration, RailwayUpload, HealthCheck, RollbackOnFailure

└─> Prevents: Broken deploys, environment mismatches, partial deployments

└─> Approval Gate: Manual trigger for safety (human approval required)


PDC Phase 5: MONITOR/MAINTAIN (Post-MVP)

└─> Maintenance States: LiveMonitoring, BrowserValidation, ErrorAnalysis, IssueDetection, AutoFixGeneration, FixValidation, CICDPipeline, VerificationCheck, LearningUpdate

└─> Prevents: Prolonged outages, repeated incidents, manual delays

└─> Self-Healing: MTTR <5 min for known patterns, auto-trigger CodeGen+Testing+Deployment for fixes


#### 3.4.2C PDC Prevention Guarantees → State Machine Implementation


Guarantee 1: Architecture is Respected

**PDC**: Phase 1.1 Architecture Alignment

**State Machines**:   CodeGen: **ArchitectureGeneration** + **ArchitectureReview**

- Implementation: GNN enforces boundaries, detects cycles, Agent won't generate violating code

- Approval Gate: Human-in-loop approval required before proceeding


Guarantee 2: Tech Stack is Consistent

**PDC**: Phase 1.2 Tech Stack Alignment

**State Machines**:

- CodeGen: **DependencyAssessment**

- Implementation: Check CVE database, validate versions, detect duplicates, verify licenses

- Blocks: Addition of incompatible/vulnerable dependencies before installation


Guarantee 3: Requirements are Clear

**PDC**: Phase 1.4 Feature Extraction

**State Machines**:- CodeGen: **ContextAssembly** (includes requirement analysis)

- Implementation: NLP extraction from PRDs/Notion/Slack, clarification dialogue before generation

- Prevents: Ambiguous requirements, missing edge cases, scope creep


Guarantee 3.5: Plans are Explicit and Validated

**PDC**: Phase 2.1 Execution Plan + Phase 2.3 Plan Sign-off

**State Machines**:- CodeGen: **TaskDecomposition**, **DependencySequencing**, **ConflictCheck**, **PlanGeneration**, **PlanReview** (optional approval)

- Implementation: GNN-based task breakdown, topological sorting, work visibility (MVP) or lock queries (Post-MVP), complexity estimation

- Prevents: Missing tasks, wrong task order, unbounded work, scope disagreements

- Approval Gate: Optional PlanReview for >5 tasks or multi-file changes ensures alignment before execution


Guarantee 4: Code is Correct

**PDC**: Phase 3.2 Prevent Bugs (6 layers)

**State Machines**:- CodeGen: **CodeGeneration** (tree-sitter, LSP), **DependencyValidation** (GNN), **BrowserValidation** (CDP), **SecurityScanning** (Semgrep)

- Testing: **UnitTesting**, **BrowserTesting**, **IntegrationTesting**

- Implementation: Multi-layer validation before commit, affected tests run automatically


Guarantee 5: Conflicts are Minimized (MVP) / Impossible (Post-MVP)

**PDC**: Phase 3.4 Prevent Merge Conflicts

**State Machines**:- **MVP**: **ConflictCheck** (planning) + **CodeGeneration** (with work visibility in UI)

  - Implementation: Show which developer is working on which files, active work indicators

  - Result: Developers can coordinate to avoid parallel edits on same files

- **Post-MVP**: **FileLockAcquisition** (before edits) + **FileLockRelease** (after completion) - Explicit States

  - Implementation: Dependency-aware locking via GNN, one task per person, lock table enforced by system

  - Result: Merge conflicts prevented by design, not resolved; explicit state transitions make locking auditable


Guarantee 6: Security is Built-in

**PDC**: Phase 3.4 Prevent Security Issues (5 layers)

**State Machines**:

- CodeGen: **DependencyAssessment** (CVE check), **SecurityScanning** (Semgrep, secrets detection)

- Implementation: Block vulnerable dependencies, scan code patterns, detect secrets pre-commit

- Testing: Security-specific tests generated automatically


Guarantee 7: Documentation is Accurate

**PDC**: Phase 1.6 + 3.6 Automated Documentation

**State Machines**:- CodeGen: Documentation generated post-**Complete** state

- Implementation: Generated from Architecture View, code analysis, ADRs, always current

- Prevents: Outdated docs, doc drift from reality


Guarantee 8: Deployments are Safe

**PDC**: Phase 4.1 + 4.2 Pre-Deploy Validation + Auto Deploy

**State Machines**:- Deployment: **PackageBuilding**, **ConfigGeneration**, **HealthCheck**, **RollbackOnFailure**

- Implementation: Full test suite passes, config validated, health checks, atomic rollback

- Prevents: Deploying broken code, environment mismatches, partial deployments


Guarantee 9: Systems Self-Heal

**PDC**: Phase 5.1 Self-Healing

**State Machines**:

- Maintenance (Post-MVP): **ErrorAnalysis** → **AutoFixGeneration** → **FixValidation** → **CICDPipeline** → **VerificationCheck**

- Implementation: Detect (<1s), analyze (<5s), fix (10-30s), validate (2-3min), deploy (1-2min), verify

- Result: Mean Time To Repair (MTTR) < 5 minutes for known patterns


Guarantee 10: Concurrency is Safe


**PDC**: Phase 3.4 Prevent Concurrency Bugs

**State Machines**:- CodeGen: **ConcurrencyValidation** (after SecurityScanning, before FixingIssues)

- Implementation:

  - Static analysis (Clippy, threading analyzers) detects unsafe patterns

  - GNN data flow analysis identifies race conditions

  - Stress test generation validates concurrent code paths

 - Three-tier auto-fix: Pattern-based → LLM-assisted → Sequential fallback

- Prevents: Race conditions, deadlocks, data races from parallel execution

- Guarantee: Code either passes concurrency validation OR falls back to safe sequential execution

- Performance: <5s validation overhead (only for code using parallelism)


#### 3.4.2D State Machine Communication


Sequential Flow (Default)


User Intent

 ↓

│ CodeGen Machine │ → Generated Code + Confidence Score

↓ (auto-trigger)

│ Testing Machine │ → Test Results + Coverage

 ↓ (manual approval)

│ Deployment Machine │ → Live Railway URL

↓ (continuous)

│ Maintenance Machine │ → Self-Healing Loop


With Failures and Retries


User Intent

↓

CodeGen Machine

↓ (SUCCESS)

Testing Machine

↓ (FAILED)

Testing Machine (retry with fixes)

↓ (SUCCESS)

Deployment Machine (user approves)

↓ (SUCCESS)

Maintenance Machine (continuous monitoring)

↓ (ERROR DETECTED)

Maintenance Machine (auto-fix)

↓ (trigger CodeGen + Testing)

Deployment Machine (auto-deploy fix)

↓ (VERIFIED)

Maintenance Machine (back to monitoring)


Session Linking


Each machine maintains references to previous sessions:

- **Testing session** stores `codegen_session_id` (can trace back to generated code)

- **Deployment session** stores `test_session_id` (can trace back to test results)

- **Maintenance session** stores `deployment_id` (can trace back to what's deployed)

- **Full traceability**: Maintenance error → Deployment → Tests → Code Generation → User Intent


Independent Execution

Machines can be triggered independently:

- **Re-run tests** without regenerating code (Testing machine only)

- **Re-deploy** without re-running tests (Deployment machine only)

- **Manual fix** can trigger Testing then Deployment (skip CodeGen)



PDC-to-State-Machine Visual Flow


─────────────────────────────────────────────────────────────────  PDC PHASE 1: ARCHITECT/DESIGN (Prevention: Architecture Violations) │

1.1 Architecture Alignment → [ArchitectureGeneration State] │ │

1.2 Tech Stack Alignment → [DependencyAssessment State] │ │

1.3 Existing Code Analysis → [ContextAssembly State] │ │

1.4 Feature Extraction → [ContextAssembly State] │ │

1.5 Architecture Sign-off → [ArchitectureReview State] ⚠️ APPROVAL GATE │ │

1.6 Auto Documentation → [Complete State] → Doc Generation │ ─────────────────────────────────────────────────────────────────

↓

│ CODE GENERATION STATE │ │ MACHINE 1 │

↓ ───────────────────────────────────────────────────────────────── PDC PHASE 2: PLAN (Prevention: Missing Tasks, Wrong Sequencing, Conflicts)

2.1 Execution Plan → [TaskDecomposition State]

     │ → [DependencySequencing State]

     │ → [ConflictCheck State]

     │ → [PlanGeneration State]

     │ → [EnvironmentSetup State]

2.2 Progress Tracking → [All States] → Real-time status updates │ │

2.3 Plan Sign-off → [PlanReview State] ⚠️ OPTIONAL APPROVAL GATE 🆕 │ │ (triggered for >5 tasks or multi-file changes) │ ────────────────────────────────────────────────────────────────── ↓ ─────────────────────────────────────────────────────────────────

PDC PHASE 3: EXECUTE (Prevention: Bugs, Conflicts, Security Issues) │

3.4 Prevent Conflicts → MVP: [CodeGeneration State with Work Visibility]│

   │ • UI shows which developer is working on which files (active indicators) │

   │ • Developers coordinate manually to avoid parallel edits │

   │ → Post-MVP: [FileLockAcquisition State] 🆕 │

   │ • Dependency-aware locks, GNN-based lock acquisition enforced by system │

 3.1 Code Generation + Tests → [CodeGeneration State] │

     │ • Layer 1: Syntax → Tree-sitter validation │

     │ • Layer 2: Types → LSP type checking │

     │ • Layer 3: Patterns → Linter (style, common bugs) │

3.2 Prevent Bugs → [DependencyValidation State] │

     │ • Layer 4: Logic → Generated with code │

     │ • Layer 5: Integration → To be run in Testing Machine │

     │ • Layer 6: Impact → GNN dependency graph checks │

 3.4 Prevent Security → [SecurityScanning State] │

     │ • Layer 1: Dependencies → Already checked in DependencyAssessment │

     │ • Layer 2: Code Patterns → Semgrep, OWASP rules │

     │ • Layer 3: Secrets → Gitleaks detection │

     │ • Layer 4: Security Tests → Generated with unit tests │

     │ • Layer 5: Static Analysis → Semgrep, Bandit │

 3.4 Prevent Conflicts → Post-MVP: [FileLockRelease State] 🆕 │

     │ • Release all locks on Complete or Failed, prevent resource leaks │

     │ 3.5 Auto Tests → [TestGeneration State] (next machine) │

     │ 3.6 Implementation Docs → [Complete State] → Doc Generation │

     │ 3.7 Feature Sign-off → [Complete State] ⚠️ QUALITY GATE │ ────────────────────────────────────────────────────────────────── ↓

│ TESTING STATE │ │ MACHINE 2 │

↓ ───────────────────────────────────────────────────────────────── 3.5 Auto Tests (continued) → [Testing Machine States] │

     │ • TestGeneration → LLM generates unit, integration, E2E tests │

     │ • EnvironmentSetup → Isolated test environment │

     │ • UnitTesting → pytest/jest/cargo test (Layer 4 validation) │

     │ • BrowserTesting → Playwright E2E tests (user workflows) │

     │ • IntegrationTesting → API/DB integration tests (Layer 5 validation) │

     │ • CoverageAnalysis → Ensure >80% coverage │

     │ • Complete → All tests pass ✅ │ ────────────────────────────────────────────────────────────────── ↓ ────────────────────────────────────────────────────────────────── PDC PHASE 4: DEPLOY (Prevention: Broken Deploys, Environment Mismatches) │ ────────────────────────────────────────────────────────────────── 4.1 Pre-Deploy Validation → [PackageBuilding, ConfigGeneration States] │

   │ • Test suite check → Already passed in Testing Machine │

   │ • Security scan → Already done in CodeGen Machine │

   │ • Config check → Environment variables, secrets validation │

   │ • Migration check → Database migration validation │

   │ • Canary tests → HealthCheck State │

4.2 Auto Deploy → [RailwayUpload State] │

   │ • Atomic deployment, same process every time │

   │ 4.3 Deploy Sign-off → [Complete State] ⚠️ APPROVAL GATE + AUDIT │

   │ • User approval required (manual trigger for safety) │ ────────────────────────────────────────────────────────────────── ↓

│ DEPLOYMENT STATE │ │ MACHINE 3 │

↓ ──────────────────────────────────────────────────────────────────  PDC PHASE 5: MONITOR/MAINTAIN (Prevention: Outages, Repeated Incidents) │ ────────────────────────────────────────────────────────────────── 5.1 Self-Healing → [Maintenance Machine States] (Post-MVP) │

   │ • LiveMonitoring → Detect issues in real-time (<1s) │

   │ • BrowserValidation → Real User Monitoring (RUM), session replay │

   │ • ErrorAnalysis → Pattern matching, severity classification │

   │ • IssueDetection → Root cause analysis, dependency graph queries │

   │ • AutoFixGeneration → LLM generates fix (10-30s) │

   │ • FixValidation → Run through CodeGen + Testing (2-3min) │

   │ • CICDPipeline → Automated deployment of fix (1-2min) │

   │ • VerificationCheck → Confirm error rate drops │

   │ • LearningUpdate → Update Yantra Codex, add monitoring │

   │ • Active/Incident → Continuous monitoring or active healing │ │ Result: MTTR < 5 minutes for known patterns ──────────────────────────────────────────────────────────────────


**State Machine Advantages for PDC Execution:**


1. **Granular Prevention**: Each state implements specific PDC checks

2. **Fail-Fast**: Violations detected immediately at state boundaries

3. **Rollback Points**: Can rollback to previous state if validation fails

4. **Audit Trail**: Every PDC check is logged with timestamps

5. **Parallel Execution**: Multiple PDC phases can run concurrently (future)

6. **Recovery**: Crash recovery at state boundaries, not mid-check



### Implementation Guide: PDC in State Machines


**When implementing a new state, ask these PDC questions:**

1. **What does this state prevent?** (PDC principle)

   - Example: `DependencyValidation` prevents breaking changes to dependent code

2. **Which PDC phase does this belong to?** (Architecture alignment)

   - Phases 1-2 → CodeGen Machine

   - Phase 3 → CodeGen + Testing Machines

   - Phase 4 → Deployment Machine

   - Phase 5 → Maintenance Machine

3. **What validations must pass before state exit?** (Fail-fast principle)

   - Example: `SecurityScanning` must have zero critical vulnerabilities

4. **Where are the approval gates?** (Human-in-loop)

   - Architecture changes → `ArchitectureReview` state (approval required)

   - Deployment → `Complete` state in Deployment machine (approval required)

5. **What's the rollback strategy?** (Recovery)

   - State must save checkpoint data before executing risky operations

   - State must define rollback logic if validation fails


**State Implementation Template:**


```rust


// Example: SecurityScanning state in CodeGen machine

impl State for SecurityScanning {

    fn enter(&mut self, context: &mut Context) -> Result<()> {

        // 1. What does this prevent? (PDC 3.4 Prevent Security Issues)

        context.log("Preventing: SQL injection, XSS, secrets in code, vulnerabilities");

        // 2. Checkpoint before risky operation (rollback support)

        context.save_checkpoint("before_security_scan")?;

        // 3. Execute preventive checks

        let semgrep_results = run_semgrep_scan(&context.generated_code)?;

        let secrets_found = detect_secrets(&context.generated_code)?;

        let cve_issues = check_dependencies_for_cves(&context.dependencies)?;

        // 4. Fail-fast validation

        if has_critical_issues(&semgrep_results, &secrets_found, &cve_issues) {

            context.log_error("Critical security issues found");

            return Err(StateError::SecurityValidationFailed);

        }

        // 5. Store results for audit trail

        context.store_security_results(semgrep_results, secrets_found, cve_issues);

        Ok(())

    }

    fn exit_condition(&self, context: &Context) -> Option<NextState> {

        // Proceed to FixingIssues if any issues, or Complete if clean

        if context.has_security_issues() {

            Some(NextState::FixingIssues)

        } else {

            Some(NextState::Complete)

        }

    }


    fn on_failure(&mut self, context: &mut Context, error: StateError) -> Result<NextState> {

        // Rollback to checkpoint

        context.rollback_to_checkpoint("before_security_scan")?;

        // Transition to FixingIssues with error context

        context.store_error(error);

        Ok(NextState::FixingIssues)

    }

}

When adding a new preventive check (from PDC document):

1. Identify the PDC phase and sub-phase
   * Example: PDC 1.2 Tech Stack Alignment → Check incompatible library versions
1. Find or create the appropriate state
* If check fits existing state → Add to that state's validation
* If check is substantial → Create new state (e.g., LicenseValidation)

1. Determine tool/protocol (refer to UTI)
* Builtin: Core differentiators, performance-critical
* MCP: External services, well-maintained servers
* Reference: UTI specification for protocol decisions

1. Add performance target
* Define expected execution time (e.g., <10ms, <1s, <30s)
* Add to performance targets table in state machine section

1. Update reconciliation table
* Add row to "Detailed State-by-State PDC Integration" table above
* Document what this check prevents


Example: Adding "License Conflict Detection" (PDC 1.2)


// Step 1: Identified as PDC Phase 1.2 Tech Stack Alignment

// Step 2: Fits in DependencyAssessment state

// Step 3: Builtin (license checker) per UTI specification

// Step 4: Performance target: <100ms

impl State for DependencyAssessment {

    fn enter(&mut self, context: &mut Context) -> Result<()> {

        // Existing checks...

        let version_compat = check_version_compatibility(...)?;

        let cve_issues = check_cves(...)?;

        // NEW: License conflict detection (PDC 1.2)

        let license_conflicts = self.check_license_compatibility(&context.requested_package)?;

        if !license_conflicts.is_empty() {

            context.log_warning("License conflicts detected");

            context.recommend_alternatives(&license_conflicts);

            return Err(StateError::LicenseConflict(license_conflicts));

        }

        Ok(())

    }

    fn check_license_compatibility(&self, package: &Package) -> Vec<LicenseConflict> {

        // Implementation: Check if package license conflicts with existing licenses

        // Performance: <100ms

        // Prevents: Adding GPL package to MIT project, incompatible LGPL usage

    }

}

For Users Understanding the System

What happens when I submit a task:

1. PDC Phase 1: Architect → States: ArchitectureGeneration, ArchitectureReview, DependencyAssessment
   * System validates your request against architecture
   * Checks if tech stack additions are safe
   * You approve architecture (if changes needed)
   * Prevention: Wrong patterns, circular dependencies, incompatible libraries
1. PDC Phase 2: Plan → State: EnvironmentSetup
* System creates execution plan
* Sets up isolated environment
* Prevention: Missing dependencies, environment issues

1. PDC Phase 3: Execute → States: CodeGeneration, DependencyValidation, BrowserValidation, SecurityScanning, TestGeneration, UnitTesting, BrowserTesting, IntegrationTesting
* System generates code with multi-layer validation
* Tests generated and executed automatically
* Prevention: Bugs, breaking changes, security issues, conflicts

1. PDC Phase 4: Deploy → States: PackageBuilding, ConfigGeneration, RailwayUpload, HealthCheck
* You approve deployment (manual trigger)
* System deploys with health checks
* Auto-rollback if issues detected
* Prevention: Broken deployments, environment mismatches

1. PDC Phase 5: Monitor → States: LiveMonitoring, ErrorAnalysis, AutoFixGeneration, FixValidation, CICDPipeline (Post-MVP)
* System monitors production continuously
* Auto-fixes issues before you notice
* Prevention: Prolonged outages, repeated incidents


Visual Progress: Three progress bars show state machine progress in real-time


Q: Where do approval gates exist? A: Two places:


1. ArchitectureReview state (if architecture changes proposed)
1. Deployment Complete state (manual trigger for safety)



---


#### 3.4.2E State Persistence & Recovery

Database Schema

Separate tables for each machine:

-- Code Generation Sessions

CREATE TABLE codegen_sessions (

    session_id TEXT PRIMARY KEY,

    current_phase TEXT NOT NULL,

    user_intent TEXT NOT NULL,

    generated_code TEXT,

    architecture_approved BOOLEAN DEFAULT FALSE,

    confidence_score REAL,

    attempt_count INTEGER,

    errors TEXT, -- JSON array

    browser_screenshot BLOB,

    created_at TIMESTAMP,

    updated_at TIMESTAMP

);

-- Testing Sessions

CREATE TABLE test_sessions (

    session_id TEXT PRIMARY KEY,

    codegen_session_id TEXT REFERENCES codegen_sessions,

    current_phase TEXT NOT NULL,

    total_tests INTEGER,

    passed_tests INTEGER,

    failed_tests INTEGER,


    coverage_percent REAL,

    attempt_count INTEGER,

    test_output TEXT,

    created_at TIMESTAMP,

    updated_at TIMESTAMP

);


-- Deployment Sessions


CREATE TABLE deployment_sessions (

    session_id TEXT PRIMARY KEY,

    test_session_id TEXT REFERENCES test_sessions,

    current_phase TEXT NOT NULL,

    platform TEXT, -- 'railway', 'aws', 'gcp', etc.

    railway_url TEXT,

    deployment_status TEXT,

    health_check_passed BOOLEAN,

    rollback_triggered BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP,

    updated_at TIMESTAMP

);


-- Maintenance Sessions (Post-MVP)

CREATE TABLE maintenance_sessions (

    session_id TEXT PRIMARY KEY,

    deployment_id TEXT REFERENCES deployment_sessions,

    current_phase TEXT NOT NULL,

    error_count INTEGER,

    browser_error_count INTEGER,

    auto_fixes_applied INTEGER,

    incident_severity TEXT, -- 'low', 'medium', 'high', 'critical'

    resolution_time_seconds INTEGER,

    created_at TIMESTAMP,

    updated_at TIMESTAMP

);


-- Production Errors (Post-MVP)


CREATE TABLE production_errors (

    error_id TEXT PRIMARY KEY,

    session_id TEXT REFERENCES maintenance_sessions,

    error_type TEXT, -- 'console', 'network', 'crash', 'performance'

    error_message TEXT,

    stack_trace TEXT,

    user_agent TEXT,

    url TEXT,

    browser_context TEXT, -- JSON with browser state

    fix_applied TEXT,


    resolved BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP

);


-- Auto-Generated Fixes (Post-MVP)

CREATE TABLE auto_fixes (

    fix_id TEXT PRIMARY KEY,

    error_id TEXT REFERENCES production_errors,

    fix_code TEXT,

    validation_passed BOOLEAN,

    deployed BOOLEAN,

    verified BOOLEAN,

    created_at TIMESTAMP,

    deployed_at TIMESTAMP

);

Crash Recovery

* Each machine saves state after every phase transition
* On crash/restart, machine can resume from last saved state
* User sees: "Resuming code generation from validation phase..."
* No work lost, seamless recovery

Query Benefits

* Track success rates: SELECT COUNT(*) FROM codegen_sessions WHERE current_phase = 'Complete'
* Average confidence: SELECT AVG(confidence_score) FROM codegen_sessions
* Test pass rates: SELECT AVG(passed_tests * 1.0 / total_tests) FROM test_sessions
* Deployment reliability: SELECT COUNT(*) FROM deployment_sessions WHERE health_check_passed = TRUE
* MTTR tracking: SELECT AVG(resolution_time_seconds) FROM maintenance_sessions
* Full audit trail: Link any production error back to original user intent



---


#### 3.4.1.2F Error/Mistake Tracking Integration

Problem: LLM has the tendency is repeat mistakes as it doesn’t learn dynamically.

Solution:

Any errors during any of the state machines code generation, testing, deploy and maintain should be automatically captured along with the errors under Known issues along with the fixes and should be referred to immediately after the first failure before trying auto fix.


Existing Implementation (src/gnn/known_issues.rs):


pub struct KnownIssue {

    id: Uuid,

    issue_type: IssueType,

    description: String,

    affected_files: Vec<PathBuf>,

    error_message: String,

    fix_applied: Option<String>,

    llm_used: String,

    created_at: DateTime,

}


Enhancement for Agentic Pipeline:

1. Automatic Capture:

* Every validation failure → Create KnownIssue entry
* Store LLM used, error, fix (if found)
* Link to failure pattern (for network effect)

2. Automatic Retrieval:

* Before retry → Query known issues DB
* Match by error signature + context
* If confidence >0.8 → Apply fix automatically
* Track success rate

3. Continuous Learning:

* Every successful fix → Update success_rate
* Every failed fix → Lower confidence
* Prune low-success patterns (<0.3 after 10 attempts)




---



#### 3.4.2F Browser Validation Across Three Machines

| Aspect | CodeGen | Testing | Maintenance |
| - | - | - | - |
| ---------------------------------------------------------------------- |
| Purpose | Visual preview | Functional testing | Production monitoring |
| - | - | - | - |
| Goal                                                                 | "Does it look right?" | "Does it work right?" | "Still working for users?" |
| Environment                                                          | Dev (local/headless) | Staging (automated) | Production (real users) |
| Speed                                                                | 5-10 seconds | 30-60 seconds | Continuous (real-time) |
| Scope                                                                | Single component | Full workflows | All user sessions |
| Data                                                                 | Mock data | Test data | Real user data |
| Checks                                                               | Render, console errors | User interactions, flows | Real errors, performance |
| Tools                                                                | CDP (Chrome DevTools) | Playwright/Puppeteer | Sentry/LogRocket RUM |
| Screenshot                                                           | Yes (user preview) | Optional (debugging) | Session replay |
| Failure Action                                                       | Regenerate code | Fix tests/code | Auto-fix + deploy |
| User Visible                                                         | Yes (preview shown) | No (automated) | No (silent healing) |
| MVP Status                                                           | ✅ Yes | ✅ Yes | ❌ Post-MVP |

Why Browser Validation in Multiple Machines?

CodeGen Machine:


* Developers need immediate visual feedback
* Catch UI bugs before writing tests
* "What You See Is What You Get" experience
* Fast iteration loop (5-10s)


Testing Machine:


* Comprehensive automated testing of user flows
* Prevent regressions
* Test edge cases and interactions
* Must pass before deployment (30-60s)


Maintenance Machine:


* Real user monitoring in production
* Detect issues from actual usage patterns
* Proactive self-healing
* Continuous improvement



---


#### 3.4.2G UI Representation

Three Progress Indicators

The UI shows three separate progress bars, one for each active machine:




┌─────────────────────────────────────────────────┐

│ Code Generation                                 │

│ [████████████████████] Complete ✓               │

│ Generated LoginForm.tsx with authentication     │

│ Confidence: 0.92 | No breaking changes          │

└─────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────┐

│ Testing                                         │

│ [████████████────────] Running... (8/12 tests)  │

│ Unit Tests: 5/5 ✓ | E2E Tests: 3/7 ⏳          │

│ Coverage: 87%                                   │

└─────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────┐

│ Deployment                                      │

│ [────────────────────] Waiting for approval     │

│ Ready to deploy to Railway                      │

│ [Deploy Now] [Cancel]                           │

└─────────────────────────────────────────────────┘

State Transitions Visible to User

Users see real-time updates as machines transition:


* "Assembling context..." → "Generating code..." → "Validating dependencies..." → "Scanning for vulnerabilities..."
* "Setting up test environment..." → "Running unit tests..." → "Running E2E tests..."
* "Building Docker image..." → "Deploying to Railway..." → "Running health check..."



---


Summary: Why This Architecture?

Key Benefits:


1. Separation of Concerns
   * Each machine has a single, clear responsibility
   * Changes to one machine don't affect others
   * Easier to understand and maintain
1. Independent Testing
* Test CodeGen without deploying
* Test deployment logic without code generation
* Mock dependencies between machines

1. Flexible Execution
* Re-run tests without regenerating code
* Re-deploy without re-running tests
* Skip machines when not needed

1. MVP Focus


* Build only CodeGen, Testing, Deployment for MVP
* Defer Maintenance machine to Post-MVP
* No unused states in production

5. Parallel Execution (Future)
   * Multiple code generation sessions simultaneously
   * Background test runs while user edits
   * Continuous maintenance monitoring
5. Clear State Tracking
* Separate database tables per machine
* Query success rates per machine
* Full audit trail across machines

5. Crash Recovery
* Each machine saves state independently
* Resume from last checkpoint
* No cross-machine corruption

5. User Experience
* Three progress bars show clear status
* Visual feedback per stage
* Manual approval for deployment (safety)


Trade-offs Accepted:

1. More Code: 4 state machines vs 1 monolithic machine
   * Worth it: Clearer architecture, easier maintenance
1. Database Complexity: 4 tables vs 1 unified table
* Worth it: Better queryability, clearer data model

1. Communication Overhead: Session linking between machines
* Worth it: Full traceability, flexible execution


Alternative Rejected:

Single monolithic state machine with 16+ states:

* ❌ Coupling between unrelated concerns
* ❌ Hard to test independently
* ❌ All states loaded even if unused
* ❌ Confusing which states are active
* ❌ Changes ripple across unrelated code



---


#### 3.4.2H Autonomous Agent Phases (Rust Enum)

pub enum AgentPhase {


    // ===== CODE GENERATION (✅ MVP COMPLETE) =====

    ContextAssembly,           // ✅ Gather dependencies, build hierarchical context

    CodeGeneration,            // ✅ Call LLM with context

    DependencyValidation,      // ✅ GNN check for breaking changes

    // ===== EXECUTION (🆕 WEEK 9-10) =====

    EnvironmentSetup,          // 🆕 Create venv, set env vars, docker if needed

    DependencyInstallation,    // 🆕 pip install / npm install / cargo build

    ScriptExecution,           // 🆕 Actually run the generated code

    RuntimeValidation,         // 🆕 Verify it runs without errors

    PerformanceProfiling,      // 🆕 Check performance metrics


    // ===== TESTING (🟡 PARTIAL) =====


    UnitTesting,               // 🟡 Test generation done, execution neede

    IntegrationTesting,        // ⚪ E2E test flows

    SecurityScanning,          // ⚪ Semgrep + Safety + TruffleHog

    BrowserValidation,         // ⚪ CDP for UI testing


    // ===== PACKAGING (🆕 MONTH 3) =====


    PackageConfiguration,      // 🆕 Generate setup.py, Dockerfile, package.json

    BuildExecution,            // 🆕 Build wheels, Docker images, npm packages

    AssetOptimization,         // 🆕 Minify, compress, bundle

    ArtifactGeneration,        // 🆕 Create distributable artifacts


    // ===== DEPLOYMENT (🆕 MONTH 3-4) =====

    DeploymentPrep,            // 🆕 Configure for target environment

    InfrastructureProvisioning,// 🆕 Provision cloud resources

    DatabaseMigration,         // 🆕 Run migrations safely

    ServiceDeployment,         // 🆕 Deploy to staging/prod

    HealthCheck,               // 🆕 Verify deployment success

    RollbackIfNeeded,          // 🆕 Auto-rollback on failure


    // ===== MONITORING (🆕 MONTH 5) =====

    MonitoringSetup,           // 🆕 Set up observability

    ErrorTracking,             // 🆕 Monitor production errors

    PerformanceMonitoring,     // 🆕 Track latency, throughput

    SelfHealing,               // 🆕 Auto-fix production issues


    // ===== COMMON PHASES (✅ COMPLETE) =====

    FixingIssues,              // ✅ Apply fixes based on errors

    GitCommit,                 // ⚪ Commit to version control

    Complete,                  // ✅ Success

    Failed,                    // ✅ Unrecoverable failure


}



Factors:

| Factor | Weight | Scoring |
| - | - | - |
| --------------------------------------------------- |
| LLM confidence | 30% | From LLM response metadata |
| - | - | - |
| Test pass rate                                    | 25% | % of tests passing |
| Known failure match                               | 25% | Similarity to solved issues |
| Code complexity                                   | 10% | Cyclomatic complexity |
| Dependency changes                                | 10% | # of files affected |


Thresholds:


* >0.8: High confidence → Auto-retry (up to 3 attempts)
* 0.5-0.8: Medium confidence → Auto-retry once, then escalate
* <0.5: Low confidence → Immediate human review



---


### 3.4.3 Dependency Intelligence & Environment Management (MVP Critical - Phase 1)

Problem 1: Agent does code generation without fully understanding the dependencies leading to duplicate code, wrong parameters, duplicate files etc.

Problem 2: Agent installs packages blindly without validation,  breaks environments with conflicts, has no rollback


Problem 3: Agent installs packages blindly without validating the environment and breaks the environment with conflicts, has no rollback, and doesn't track package


Solution: Comprehensive dependency intelligence with dependency assessment,  dry-run validation, mandatory .venv isolation, GNN integration, conflict detection, automatic rollback, and transparent multi-project environment management.

| # | Feature | Gap | Requirement | Priority |
| - | - | - | - | - |
| ------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.4.3.1 | Mandatory venc creation | Pollutes system Python | Always use separate venv per project.venv/bin/python, block global access | P0 |
| - | - | - | - | - |
| 3.4.3.2                                                                                                                                   | File/code/Tech stack  dependency assessment | Can't assess impact of changes | Assess the tech stack needed based on the need for new project using architecture and specifications (from chat or from the documents) or existing project before creating new code | P0 |
| 3.4.3.3                                                                                                                                   | Dry-run validation | Installs blindly, breaks production | Validate in temp venv first, detect conflicts | P0 |
| 3.4.3.4                                                                                                                                   | GNN Tech Stack Dependency Tracking | Installs without full view of the project dependencies leading to errors | GNN tracks which files/functions/packages/tools use each package and Agent checks for dependencies | P0 |
| 3.4.3.5                                                                                                                                   | Dependency conflict detection and resolution | Silent breakage, mysterious failures | Detect conflicts  and resolve before installation, suggest fixes | P0 |
| 3.4.3.5A                                                                                                                                  | Web search to resolve Dependency conflict | LLM has snapshot knowledge | Detect conflicts  and resolve before installation, suggest fixes | P0 |
| 3.4.3.6                                                                                                                                   | Rollback mechanism | Broken state on failure | Snapshot + restore on failure | P1 |
| 3.4.3.7                                                                                                                                   | Environment validation | Commands fail with unclear errors | Pre-execution checks (venv active? packages installed?) | P1 |
| 3.4.3.8                                                                                                                                   | Multi-project isolation | Cross-contamination between projects | Each project has isolated .venv, auto-switch | P1 |
| 3.4.3.9                                                                                                                                   | Usage verification | Unused dependencies pile up | Track if package is actually used after install | P2 |
| 3.4.3.10                                                                                                                                  | Dependency caching | Slow, wastes bandwidth | Local cache for faster installs | P2 |


#### 3.4.3.1: Mandatory .venv Isolation (BLOCKER)

Principle: NEVER pollute Python. ALWAYS work in .venv.

Agent should

1. Always check if venv exists, valid and activated
1. Run terminal commands only in a venv
1. If venv is not setup, then it should be setup and activated in the workspace folder with full transparency to the user
1. If venv already exists, check for validity and activate it automatically
1. If venv is corrupted, inform user and delete to recreate in guided mode. In auto mode automatically delete to recreate
1. Update Terminal Executor for Venv Enforcement:



Agent Transparency (.venv):


🔧 Ensuring Python environment is isolated...

   ✅ Found .venv at: /workspace/.venv

   🐍 Python version: 3.11.5

   📦 Installed packages: 47

   ⏱️  Created: 2 days ago

   💡 All Python commands will use .venv (never system Python)

   🛡️  Your system Python is protected!


Auto-Creation:


🔍 .venv not found in workspace

🔧 Creating isolated Python environment...

   ▶️  Running: python3 -m venv .venv

   ⏳ [2s] Creating virtual environment...

   ⏳ [4s] Installing pip, setuptools...

   ✅ Created .venv successfully!

   🐍 Python version: 3.11.5

   📦 Ready to install packages

   💡 All future Python commands will use this isolated environment

#### 3.4.3.2: Dependency Assessment and setup

Principle:Never install blindly


Dependency Assessment Workflow for Agent

1. Clarify architecture and specifications with user
1. Assess the tech stack needed based on the need for new project using architecture and specifications (from chat or from the documents)
1. Assess blast radius for the change for existing project , if code changed
1. Provide user the full transparent on the impact
1. If user skips dry-run validation, then  create lock files, requirements.txt and manifest files as needed
1. Install all the requiremets

#### 3.4.3.3: Dry-Run Validation (PO)

Principle: NEVER install without validating first. 100% accurate GNN dependency graph.


Dependency Validation Workflow for Agent

 // 1. Check current environment

 // 2. Query GNN for impact analysis

 // 3. Create isolated temp venv for dry-run

 // 4. Copy current packages to temp venv

 // 5. Dry-run installation in temp venv

 // 6. Parse dependency resolution (JSON output from pip)

 // 7. Detect conflicts

 // 8. Calculate impact

// 9. Use consultant mode if needed

// 10. Use RAG if LLM unable to resolve

// 11.User web search

 // 12. Resolve conflicts (downgrade, upgrade as needed - use LLM reasoning)

**	**12.1 Try 2 times and ask LLM consultant ask. If unresolved after  5  times ask user if they want to continue// 13 Cleanup temp venv

// 14 Create Validation report

// 15. Create lock files, requirements.txt and manifest files as needed

// 16. Create venv for the project and install all the required tools/packages in it in auto mode. In guided mode ask user and install on getting go ahead


Agent Transparency (Dry-Run):


🔍 Validating package installation...

   Package: numpy==1.26.0

   Strategy: Dry-run in isolated temp venv first

   ✅ Dry-run validation passed!

   📊 Impact Analysis:

      - Files affected: 5 (src/ml/*.py)

      - Functions affected: 12 (np.array, np.mean, etc.)

      - Conflicts: None

      - Risk: LOW

   💡 Safe to install. Proceeding with real installation in .venv


Conflict Detection Example:


❌ Validation failed: Dependency conflict detected

   Package: pandas==2.1.0

   Conflict: numpy version incompatibility

   Current environment:

   - numpy 1.24.0 (installed)

   Requested installation:

   - pandas 2.1.0 (requires numpy>=1.26.0)

   ⚠️ Risk: HIGH (numpy is used by 8 files, 25 functions)

   💡 Recommendations:

   1. [SAFE] Upgrade numpy to 1.26.0 first

      Command: pip install numpy==1.26.0

      Risk: LOW (backward compatible)

   2. [RISKY] Downgrade pandas to 2.0.x

      Command: pip install pandas<2.1

      Risk: MEDIUM (may lose features)

   3. [COMPLEX] Create separate venv for this module

      Risk: LOW but requires code isolation

   What would you like to do?



---


#### 3.4.3.4 GNN Tech Stack Dependency Tracking

Principle:Have the full view of the project dependencies to generate code without errors


Agent should TRACK  the dependencies in the dependency graph WHEN  a file is created or code is committed

* File to File ( transitive dependencies) (MVP)
* File to package version (MVP)
* Package to Package (transitive dependencies) (MVP)
* File to function, classes, imports (MVP)
* P1 Post-MVP: External API Tracking (API endpoints as nodes, track API calls across services)
* P2 Post-MVP: Method Chain Tracking (track df.groupby().agg() level granularity)
* P2 Post-MVP: Function → Package Function (which specific package functions are used)
* ✅ Track exact dependencies across files
* ✅ Detect breaking changes before commit
* ✅ Fast incremental updates (1ms)
* ✅ Cross-language support
* ✅ Transitive dependency analysis


Yantra should TRACK the user who is using the file in dependency graph

* P1 Post-MVP: User → File Tracking (team collaboration, ownership, expertise mapping)


Agent should CHECK, DETECT and RESOLVE the dependencies in the dependency graph BEFORE  a file is created or code is generated automatically

* Nested function tracking (numpy.random.normal, not just numpy.array)
*  Version conflict detection (simple semver-based)
* ✅ Breaking change warnings (major version bumps)
* ✅ Query: "What breaks if I upgrade X?"


Critical: Version-Level Tracking


* ❌ WRONG: Track "numpy" as single node → cannot detect version conflicts
* ✅ CORRECT: Track "numpy==1.24.0" and "numpy==1.26.0" as separate nodes → detect incompatibilities
* Track EXACT versions for all packages (numpy==1.26.0, pandas==2.1.0, not just "numpy", "pandas")
* Track version requirements for dependencies (requires: "numpy>=1.26,<2.0")
* Track version history (upgraded from 1.24.0 → 1.26.0 on date X)
* Enable queries: "Which files depend on numpy 1.24 specifically?" vs "Which files use any numpy?"


Metadata Sources (Priority):


Hybrid Detection Strategy

* Priority 1: Lock files (fast, reproducible)
* Priority 2: Runtime inspection (accurate, dev workflow)
* Priority 3: Manifest files (ranges only, warning)


Function Tracking Granularity:


* MVP: Nested attributes (numpy.random.normal, pandas.DataFrame.groupby)
* Deferred: Method chains (df.groupby().agg().reset_index()) - P2 Post-MVP


Cross-Language Strategy:


* MVP: Separate graphs per language (Python graph, JavaScript graph, Rust graph)
* Post-MVP: Unified graph with API nodes connecting languages


Update Triggers:


* Project open (initial scan)
* Lock file changes (watch requirements.txt, package-lock.json, Cargo.toml)
* Manual refresh (user-requested)
* Pre-code generation (ensure fresh context)


Flow:
    1. PROJECT SCAN

   ↓

2. FILE PARSING (tree-sitter)

   ├─ Python → parser.rs

   ├─ JavaScript → parser_js.rs

   ├─ Rust → parser_rust.rs

   └─ ... (13 languages)

   ↓

3. AST EXTRACTION

   ├─ Nodes: Functions, Classes, Imports

   └─ Edges: Calls, Inherits, Imports, Uses

   ↓

4. GRAPH CONSTRUCTION (petgraph)

   ├─ Add nodes to DiGraph

   ├─ Add edges between nodes

   └─ Store in HashMap for O(1) lookup

   ↓

5. INCREMENTAL TRACKING

   ├─ Cache nodes by file (1ms updates)

   ├─ Track timestamps

   └─ Dirty flag propagation

   ↓

6. PERSISTENCE (SQLite)

   ├─ Save graph to .yantra/graph.db

   └─ Load on demand

   ↓

7. QUERY API

   ├─ get_dependencies(node_id)

   ├─ get_dependents(node_id)

   └─ get_all_dependencies(node_id) [BFS]

   ↓

8. UI VISUALIZATION

   └─ Interactive graph ([Cytoscape.js](http://cytoscape.js))



Real-World Scenarios should work flawlessly


| Scenario 1:Active Development (No Lock File)** - MVP                                                                                                                                                                                                                                                                                | Scenario 2:Team Collaboration (Lock File Exists) -MVP                                                                                                                                                                                                                                | Scenario 3:Drift Detection (Lock vs Runtime Mismatch)* MVP                                                                                                                                                                                                                                                                                                                           | Scenario 4:CI/CD (Lock File Only) - Post MVP                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Developer workflow:1. pip install numpy          ← Manual install (no lock file yet)2. Write code: import numpy3. Yantra checks dependencies:├─ Lock file? ❌ None├─ Runtime inspection? ✅ numpy==1.26.3└─ Result: "numpy 1.26.3 (from pip show)"4. Yantra WARNING: "⚠️ No lock file. Run: pip freeze > requirements.txt" | Team member pulls repo:1. git pull2. npm install                ← Generates package-lock.json3. Yantra checks dependencies:├─ Lock file? ✅ package-lock.json├─ Result: "numpy@1.26.3 (from lock file)"└─ Fast: <1ms (JSON read)4. No warnings (lock file is source of truth) | Developer manually installs newer version:1. pip install numpy==1.27.0  ← Manual upgrade2. Yantra checks dependencies:├─ Lock file: numpy==1.26.3├─ Runtime: numpy==1.27.0└─ ⚠️ WARNING: "Version mismatch! Lock: 1.26.3, Installed: 1.27.0"3. Yantra suggests: "Update lock file: pip freeze > requirements.txt"Benefit: Catch version drift before it causes bugs in CI/CD. | CI pipeline:1. Checkout code2. Install from lock: pip install -r requirements.txt3. Yantra checks dependencies:├─ Lock file? ✅ requirements.txt (with exact versions from pip freeze)├─ Runtime inspection? Skip (CI uses lock file as source)└─ Result: Fast, reproducible buildsBenefit: No runtime inspection overhead in CI (lock file is sufficient).<br/> |
| Benefit: Yantra works immediately, catches breaking changes NOW, nudges toward best practices.                                                                                                                                                                                                                                      | **Benefit:** Fast, reproducible, team-aligned versions.                                                                                                                                                                                                                              | **Benefit:** Catch version drift before it causes bugs in CI/CD.                                                                                                                                                                                                                                                                                                                       | **Benefit:** No runtime inspection overhead in CI (lock file is sufficient).                                                                                                                                                                                                                                                                                       |




Agent Transparency (with Dependency graph aka GNN Integration):


📊 Updating dependency graph after numpy==1.26.0 installation...

   🔍 Scanning workspace for numpy usage...

   ✅ Found 5 files using numpy==1.26.0:

      - src/ml/model.py (np.array, np.mean, np.std)

      - src/ml/preprocessor.py (np.zeros, np.ones)

      - src/data/loader.py (np.loadtxt)

      - src/utils/math_utils.py (np.dot, np.transpose)

      - tests/test_model.py (np.testing.assert_array_equal)

   📦 Detected 12 function calls to numpy APIs

   🔗 Updated dependency graph with package→file→function mappings

   🎯 Version-Level Tracking Active:

      - Node created: pkg:numpy:1.26.0 (exact version)

      - Can now detect conflicts with numpy 1.24.0, 2.0.0, etc.

      - Version history: Upgraded from 1.24.0 → 1.26.0 (dependency resolution)

#### 3.4.3.5Conflict Detection & Resolution (MVP)


Principle: Detect conflicts and resolve BEFORE installation. Suggest intelligent fixes with risk assessment.

Step 1: Detect the Conflict

Before installing any package, the system should check:

* Identify the language/ecosystem:
* What version does the new package need? (e.g., pandas 2.1.0 needs numpy>=1.26.0)
* What version do you currently have? (e.g., numpy 1.24.0)
* Check compatibility first - Don't install anything yet
* Identify conflicts** using language-specific tools:
* Analyze the conflict:
  * What package is conflicting?
  * What version do you have now?
  * What version does the new package need?
  * Are they compatible


Step 2: Analyze the Impact

The system uses the Graph Neural Network (GNN) to understand:

* Use GNN to know **exactly which functions/methods** code uses, How many files in your project use the conflicting package?
* Query language-specific breaking change sources
* For each language use appropriate methods. Below examples are shown for Python adn JS
  * **Python:**

- GNN tracks: `import numpy`, `from flask import Flask`, `numpy.array()`

- Breaking changes from: PyPI, conda-forge, GitHub releases

- Version semantics: PEP 440 (major.minor.patch)


* **JavaScript/TypeScript:**

- GNN tracks: `import express from 'express'`, `require('lodash')`, `_.map()`

- Breaking changes from: npm registry, GitHub releases, TypeScript type definitions

- Version semantics: semver (major.minor.patch)


* Do Breaking change analysis - Cross-reference: Do breaking changes affect functions your code uses?
* Do High Confidence Analysis (When we have all the data) and Give **definitive answer** when all data is available


Breaking Change Data Sources (Per Language)


| Language   | Function Tracking                     | Breaking Change Sources                                        |
| ------------ | --------------------------------------- | ---------------------------------------------------------------- |
| Python     | numpy.array(), flask.Flask()          | PyPI changelog, conda docs, GitHub releases                    |
| JavaScript | express.Router(), _.map()             | npm registry API, GitHub releases, TypeScript breaking changes |
| Rust       | tokio::spawn(), serde::Deserialize    | crates.io API, docs.rs, Cargo CHANGELOG.md                     |
| Go         | gin.Default(), http.ListenAndServe()  | Go module proxy, GitHub releases, Go blog                      |
| Java       | SpringApplication.run(), List.of()    | Maven Central API, JavaDoc diffs, GitHub releases              |
| Ruby       | Rails.application, ActiveRecord::Base | RubyGems API, Rails upgrade guides, GitHub releases            |
| PHP        | Request::create(), Response::json()   | Packagist API, Symfony upgrade guides, GitHub releases         |
| .NET       | app.UseRouting(), LINQ.Select()       | NuGet API, Microsoft Docs breaking changes, GitHub releases    |


* If partial data available do Medium Confidence Analysis

* Have changelog but no function tracking → "Changelog shows breaking changes. Can't determine if your code is affected. LIKELY breaking."
* Have function tracking but no changelog → "Your code uses these functions. No changelog available. Based on version change, LIKELY breaking."

* If minimal data available do low confidence analysis using language-specific version semantics:
  * Python PEP 440: Major change = likely breaking
  * JavaScript semver: Major change = breaking by definition
  * Rust Cargo: Breaking change in ^1.0 dependency = likely breaking
  * Go: Major version in import path (v1 → v2) = breaking etc.

Step 3: Suggest Smart Solutions (Ranked by Safety)

The system offers 3 types of solutions, sorted from safest to riskiest:

* Upgrade the Conflicting Package
  * Show language-appropriate commands:
* Use an Older Version of the New Package
* Show language-appropriate version constraints:

* Keep your existing code exactly as-is, and create a completely separate environment for the new code.
* Show language-appropriate isolation strategies:


In guided mode, the system will show all options, sorted from lowest risk to highest risk, and explains. In auto mode, the system will decide the best option.

* What command will be run
* Why this solution works
* What's the risk (Low/Medium/High/Critical)
* What's the impact on your code
* How long it will take


Agent Transparency (Conflict Resolution):

* Should clearly show all the options and recommended option and why. Make it easily understandable for the user

#### 3.4.3.5A Web Search for Dependency Resolution:

Problem: LLM knowledge is static (training cutoff date), but packages/tools evolve constantly.

Solution: Agent uses web search to access live package documentation:

Web Search Integration (DependencyAssessment State)


When: Resolving package versions, compatibility issues, API changes

Primary Method: Browser automation via Chrome DevTools Protocol (CDP) - Built-in capability

* Agent navigates like a human user: opens browser, searches, clicks links, extracts content
* Parallel Processing: Open multiple tabs simultaneously for faster information gathering
  * Tab 1: PyPI/npm official package page (version, release date)
  * Tab 2: Package changelog/release notes (breaking changes)
  * Tab 3: Migration guide or upgrade documentation
  * Tab 4: Known issues (GitHub issues, Stack Overflow)
  * Consolidation: LLM synthesizes information from all tabs into coherent recommendation
  * Performance: ~5-8 seconds total (parallel) vs ~15-20 seconds (sequential)
* Advantages: No external dependencies, works with cached pages, full DOM access, parallel efficiency
* Trade-off: Slightly slower than API calls, but more reliable for complex pages
* Queries:
* Navigate to Google: "numpy version 2.0 breaking changes"
* Click top results (PyPI, numpy.org, Stack Overflow)
* Extract version info, changelog, migration guides from page content

* Alternative Method (Optional): MCP servers for faster API-based search
* MCP @modelcontextprotocol/server-brave-search - Brave Search API (fast, structured)
* MCP @modelcontextprotocol/server-fetch - Direct HTTP fetch (for known URLs)
* Advantages: Faster (<1 second), structured JSON responses
* Trade-off: Requires external MCP server setup, API keys, internet dependency

* Strategy:
* Default: Use browser automation with parallel tabs (built-in, reliable, no setup)
* Optimization: Use MCP servers if installed and configured (faster for simple queries)
* Hybrid: Browser automation for complex pages + MCP for structured API data
* Fallback: If both fail, use LLM knowledge with disclaimer about potential outdated info

* Processing:
* Fetch official docs (PyPI, npm, crates.io, official websites)
* Extract version compatibility matrices using tree-sitter or DOM parsing
* Consolidate information from multiple sources (official docs + community knowledge)
* Cache results in SQLite for future use (TTL: 7 days)

* Package Documentation Fetching (Post-MVP)
* Automated: When new package added, fetch + index its documentation
* Sources:
* Official docs (docs.python.org, react.dev, docs.rs)
* README files from package repositories
* Changelog/migration guides

* Method: Browser automation navigates to docs, extracts structured content
* Indexing: Store in dependency graph with semantic embeddings
* Usage: Provide as context to LLM when generating code using that package


Example Flow (Browser Automation - Default):


User: "Add numpy for matrix operations"

  ↓

DependencyAssessment State:

  → Check current numpy version in requirements.txt (1.24.0)

  → Open browser (headless Chrome via CDP)

  → Navigate to Google: "numpy latest version 2024"

  → Click first result (PyPI numpy page)

  → Extract version: 2.0.0, release date, breaking changes link

  → Navigate to numpy.org/doc/2.0/release/2.0.0-notes.html

  → Extract: Breaking changes in numpy.random module

  → Total time: ~8-12 seconds (acceptable for critical decisions)

  → Query dependency graph: Search cached migration guides

  → Retrieve: Known migration patterns from previous searches

  ↓

Agent proposes:

  "Current: numpy==1.24.0

   Latest: numpy==2.0.0 (breaking changes in numpy.random)

   Recommendation: numpy==1.26.4 (stable, compatible)

   Reason: numpy 2.0 changes numpy.random.Generator API.

   Your code uses np.random.rand() which is deprecated in 2.0.

   Version 1.26.4 is latest stable with backward compatibility.

   Source: Verified via PyPI and numpy.org (fetched 30 seconds ago)"


Implementation Notes:

* Web search results cached in SQLite with TTL (7 days for package info)
* Semantic embeddings stored in dependency graph for similarity search
* LLM prompted with: "Use web search for latest package info, your knowledge may be outdated"
* Conflict resolution prioritizes: 1) Official docs, 2) Community recommendations, 3) LLM reasoning

#### 3.4.3.6 Automatic Rollback on Failuremechanism

What: If installing packages fails, automatically restore the environment to its previous working state. Never leave the environment in a broken state.


Implementation:agent/environment_snapshot.rs (NEW, ~150 lines)

Requirements:

1. Before Installation - Create Snapshot:
   * Capture current state of environment before making any changes
   * Record all installed packages with exact versions
   * Store snapshot with unique ID and timestamp
   * Snapshot must be fast to create (<1 second)


2. During Installation - Monitor:
   * Track installation progress
   * Detect if any package fails to install
   * Capture error messages for user


3. On Failure - Auto Rollback:
   * Immediately restore environment to snapshot
   * Uninstall any partially installed packages
   * Verify environment integrity after restore
   * Report what failed and why


4. Multi-Language Support:
   * Python: Snapshot via pip freeze, restore via reinstall
   * JavaScript: Backup package.json + package-lock.json, restore via npm ci
   * Rust: Backup Cargo.lock, restore via cargo update
   * Go: Backup go.sum, restore via go mod download
   * Java: Backup dependency tree, restore via Maven/Gradle
   * Ruby: Backup Gemfile.lock, restore via bundle install
   * PHP: Backup composer.lock, restore via composer install
   * .NET: Backup package references, restore via dotnet restore


5. User Communication:
   * Show snapshot creation: "📸 Creating safety snapshot..."
   * Show rollback trigger: "❌ Installation failed: [reason]"
   * Show restore progress: "🔄 Rolling back to previous state..."
   * Confirm success: "✅ Environment restored - no changes made"


User Experience:

📸 Creating safety snapshot before installation...

✅ Snapshot created: snapshot-1701788400

Installing packages: flask==3.0.0, psycopg2==2.9.9...

  ✅ flask 3.0.0 installed

  ❌ psycopg2 2.9.9 failed: missing pg_config

❌ Installation failed!

🔄 Rolling back to previous state..

  Uninstalling flask 3.0.0...

  Restoring to snapshot-1701788400...

✅ Environment restored - no changes made

Error: psycopg2 requires PostgreSQL dev libraries

Suggestion: Install libpq-dev first, then retry

Performance Target: Rollback completes in <5 seconds


Critical: This prevents the "half-installed" state where some packages succeed and others fail, leaving the environment unusable.


#### 3.4.3.7 Pre-Execution Environment Validation (MVP)


**What:** Validate that the environment is properly set up BEFORE running any code. Prevent execution failures caused by missing dependencies, inactive virtual environments, or missing configuration.


**Implementation:** `agent/environment_validator.rs` (NEW, ~200 lines)


**Requirements:**


1. **Pre-Execution Checks** - Run these validations before executing any generated code:


   - Virtual environment is active (venv, conda, etc.)

   - All required packages are installed (query GNN for imports)

   - Environment variables are set (if code uses them)

   - Required services are running (Docker, databases, Redis, etc.)

   - Required ports are available (if starting a server)

   - File permissions are correct (if accessing protected files)

   - Disk space available (if writing large files)

2. **Smart Detection** - Analyze the code to determine requirements:


   - Parse imports to identify required packages

   - Detect environment variable usage (os.getenv, process.env)

   - Identify Docker/database connections

   - Detect port binding (server startup code)

   - Check file I/O operations

3. **Clear Reporting** - Show validation results to user:


   - List all checks performed

   - Mark each check as passed ✅ or failed ❌

   - Explain WHY each check failed

   - Provide exact commands to fix each issue

   - Estimate time to fix issues

4. **Auto-Fix Suggestions** - For each missing requirement:


   - Suggest specific fix command

   - Explain what the command does

   - Allow user to approve all fixes at once

   - Optionally auto-fix common issues (install missing packages)

5. **Blocking vs Warning** - Categorize validation failures:


   - **BLOCKING** - Must fix before execution (missing packages, inactive venv)

   - **WARNING** - Should fix but can proceed (missing optional config)

   - **INFO** - Suggestions for better setup (enable logging, add error handling)

6. **Multi-Language Support** - Validate environments for all languages:


   - Python: venv active, pip packages installed

   - JavaScript: node_modules present, npm packages installed

   - Rust: cargo dependencies resolved

   - Go: go modules downloaded

   - Java: Maven/Gradle dependencies available

   - Ruby: gems installed

   - PHP: composer packages present

   - .NET: NuGet packages restored


**User Experience:**


Before running code, show validation dialog:


🔍 Validating environment before execution...


✅ Virtual environment active (.venv)

✅ Python 3.11.5 installed

❌ Package 'flask' NOT installed

❌ Package 'psycopg2' NOT installed

✅ PostgreSQL running on port 5432

⚠️  Environment variable 'DATABASE_URL' not set (optional)


2 issues must be fixed before execution:


Fix 1: Install flask

  Command: pip install flask==3.0.0

  Why: app.py imports flask


Fix 2: Install psycopg2

  Command: pip install psycopg2-binary==2.9.9

  Why: models.py imports psycopg2

[Auto-Fix All] [Fix Manually] [Cancel Execution

**Performance Target:** Complete validation in <500ms

#### 3.4.3.8 Multi-Project Isolation (MVP - Enterprise Critical)


**What:** Each project has its own isolated virtual environment. Never share dependencies between projects. Automatically switch environments when user switches projects.


**Implementation:** `agent/project_environment_manager.rs` (NEW, ~250 lines)


**Requirements:**


1. **Per-Project Isolation:**


   - Each project gets its own virtual environment (venv, conda, node_modules, etc.)

   - Store environment in project workspace (`.venv`, `node_modules`, etc.)

   - Track which environment belongs to which project

   - Never install packages globally or in shared locations

2. **Automatic Environment Switching:**


   - Detect when user switches workspace/folder

   - Automatically activate the correct environment for that project

   - Deactivate previous environment cleanly

   - Show user which environment is now active

3. **Environment Information Tracking:**

   - Project ID and workspace path

   - Virtual environment location

   - Language version (Python 3.11.5, Node 18.0.0, etc.)

   - Installed packages count

   - Last active timestamp

   - Disk space used

4. **Multi-Language Support:**

   - Python: Separate `.venv` per project

   - JavaScript: Separate `node_modules` per project (standard behavior)

   - Rust: Separate `target` directories per project (standard behavior)

   - Go: Separate module caches per project

   - Java: Separate Maven/Gradle local repositories per project

   - Ruby: Separate gem installations per project

   - PHP: Separate vendor directories per project

   - .NET: Separate package references per project

5. **Prevent Cross-Contamination:**

   - Project A uses Flask 2.0, Project B uses Flask 3.0 → No conflicts

   - Installing package in Project A doesn't affect Project B

   - Deleting Project A's environment doesn't affect Project B

   - Each project can have different language versions

6. **Transparent Switching:**

   - Show user when environment switches

   - Display environment details (language version, package count)

   - Alert if environment needs setup (first time opening project)

   - Warn if environment is outdated or corrupted


**User Experience:**


When switching between projects:

🔄 Switching projects...


From: project-a (AI Chatbot)

└─ Python 3.11.5, 47 packages


To: project-b (ML Model)

└─ Python 3.10.8, 89 packages

✅ Environment switched successfully!

🛡️  Projects are isolated - no dependency conflicts


When opening project for first time:

🆕 New project detected: data-pipeline

Environment not found. Creating isolated environment...

✅ Created .venv with Python 3.11.5

📦 Ready to install dependencies


Enterprise Benefits:


- Work on multiple client projects without conflicts

- Different projects can use incompatible package versions

- Clean separation for security/compliance

- Easy to delete one project without affecting others

#### 3.4.3.9 Usage Verification

Track if installed packages are actually imported/used. Suggest cleanup for unused deps.

#### 3.4.3.10 Dependency Caching

Local cache (`~/.yantra/cache/pip`) for faster installs, less bandwidth.

### 3.4.4 Agent Execution Intelligence (MVP Critical - Phase 1)

Problem: Agent blocks on long-running commands (builds, tests, servers), becoming unresponsive and appearing "frozen" to users. This creates poor UX and prevents parallel task execution.


Solution: Intelligent command classification with automatic background execution strategy and transparent status reporting.


Command Classification & Execution Strategy

Agent Intelligence: Automatically detect command duration and choose optimal execution pattern.


| Command Pattern | Duration | Strategy | Agent Behavior | Example |
| - | - | - | - | - |
| ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Build commands | Long (10-60s) | Background + polling | Execute in background, poll every 2-5s, report progress | npm run build,cargo build,make |
| - | - | - | - | - |
| Test execution                                                                                                                                 | Long (5-30s) | Background + polling | Execute in background, show test progress, report results | pytest,npm test,cargo test |
| Dev servers                                                                                                                                    | Infinite | Background + monitor | Fire and forget, monitor for crashes, report when ready | npm start,python manage.py runserver |
| Package install                                                                                                                                | Medium (5-20s) | Background + polling | Execute in background, report completion | npm install,pip install -r requirements.txt |
| Quick queries                                                                                                                                  | Quick (<1s) | Synchronous | Execute immediately, return result | git status,ls,cat file.py |
| File operations                                                                                                                                | Quick (<1s) | Synchronous | Execute immediately, return result | cp,mv,rm(non-recursive) |


#### 3.4.4.1 Transparent Agent Communication

Principle: Agent must ALWAYS explain what it's doing and why. No silent operations.


Status Messages (Required for All Long-Running Commands):

Building project...

   Strategy: Background execution (expected 30s)

   Reason: Build commands block for extended periods

   Status: You can ask me anything while I monitor this!


   [2s] Still building... (checking every 2s)

   [4s] Still building...

   [6s] Build output: Compiling 47 files...

   [10s] Build complete! ✅

Transparency Requirements:

1. Start: Explain command classification
   * "Detected long-running command (npm run build)"
   * "Executing in background to keep me responsive"
1. During: Show progress updates
* Poll status every 2-5 seconds
* Show relevant output lines (errors, warnings, completion %)
* Remind user: "I'm still available for other tasks!"

1. Completion: Report results clearly
* Success: "Build completed in 23s ✅"
* Failure: "Build failed ❌ [show error excerpt]"
* Next steps: "Ready to run tests or deploy"

1. Interruption: Allow user to stop
* User: "Stop that build"
* Agent: "Terminating npm build process... Done ✅"


Implementation Architecture


Required Components

1. Command Classifier (agent/command_classifier.rs - NEW)
   * Maintain pattern database mapping commands to duration categories (Quick/Medium/Long/Infinite)
   * Match commands using exact matches first, then regex patterns
   * Default to Medium duration if command unknown
   * Provide human-readable classification explanations
1. Intelligent Executor (Update agent/orchestrator.rs)
* Classify command before execution
* Emit transparency message explaining strategy
* Execute synchronously for Quick commands
* Execute with polling for Medium/Long commands
* Execute fire-and-forget for Infinite commands (servers)
* Poll background tasks every 2 seconds and emit status updates
* Show recent output lines during execution
* Report completion with timing and exit status


3. Status Communication System
   * Emit real-time events to frontend UI
   * Support status messages, progress indicators, and error notifications
   * Enable user interruption of running processes
   * Track elapsed time for all operations
   *

UI Integration

Agent Status Panel (Real-time Updates):

┌─────────────────────────────────────────┐

│ 🤖 Agent Status: ACTIVE & AVAILABLE                                        │

├─────────────────────────────────────────┤

│ Current Tasks:                                                                                  │

│                                                                                                           │

│ 🔨 [Background] npm run build                                                       │

│    ⏱️  Running for 12s                                                                      │

│    📊 Compiling 47/150 files                                                            │

│    💬 You can ask me anything!                                                       │

│ ✅ [Complete] Generated UserService.py                                      │

│    ⏱️  Took 3s                                                                                  │

│    📝 Added to src/services/                                                            │

└─────────────────────────────────────────┘


Performance Targets

| Metric | Target | Rationale |
| - | - | - |
| --------------------------------------------------- |
| Classification time | <5ms | Must not add latency |
| - | - | - |
| Status poll interval                              | 2-5s | Balance responsiveness vs overhead |
| Status message latency                            | <50ms | Real-time feel |
| Agent response time (during background task)      | <200ms | Must remain interactive |
| Terminal pool utilization                         | >70% reuse rate | Efficient resource usage<br/> |


Success Metrics

* ✅ Agent Responsiveness: <200ms response to user queries during background tasks
* ✅ Transparency: 100% of long-running commands show status updates
* ✅ Parallel Execution: 3+ simultaneous background tasks supported
* ✅ User Confidence: Clear "what/why/when" for every operation
* ✅ No Silent Failures: All errors reported with context and suggestions
*

Benefits

1. 🚀 Perceived Speed: Agent feels 10x faster (never "frozen")
1. 💬 Continuous Availability: Always ready for new requests
1. 📊 Visibility: Users see exactly what's happening
1. 🔄 Parallel Work: Build + test + generate simultaneously
1. 🧠 Smart Resource Use: Efficient terminal pool management
1. 🎯 Trust: Transparency builds user confidence in AI agent

### ---


### 3.4.5 Test File Dependency Tracking with Coverage Analysis 🆕

Added: November 30, 2025 Status: ✅ Implemented in GNN Files:src-tauri/src/gnn/mod.rs

Overview

The GNN (Graph Neural Network) now tracks bidirectional relationships between test files and source files, enabling comprehensive test coverage analysis, impact analysis, and detection of untested code. This enhancement is critical for ensuring code quality and guiding test generation.


Architecture

New Edge Types:

1. EdgeType::Tests - Function-level test relationship
   * Represents: test_function → source_function
   * Example: test_add() Tests edge to add()
   * Used for: Fine-grained test coverage at function level


2. EdgeType::TestDependency - File-level test relationship
   * Represents: test_file → source_file
   * Example: test_calculator.py TestDependency edge to calculator.py
   * Used for: General test-to-source mapping when function mapping fails
   *

Implementation Details

1. Test File Detection - is_test_file(path: &Path) -> bool

Automatically detects test files using language-specific patterns:

Python Test Patterns:

* test_*.py (pytest convention)
* *_test.py (alternative convention)
* Files in /tests/ or /test/ directories

JavaScript/TypeScript Test Patterns:

* *.test.js, *.test.ts
* *.spec.js, *.spec.ts
* Files in /__tests__/ directories

Example Detection:

* ✅ test_calculator.py → Test file
* ✅ calculator_test.py → Test file
* ✅ tests/utils.py → Test file
* ✅ math.test.ts → Test file
* ❌ calculator.py → Source file
* ❌ utils.js → Source file


2. Test-to-Source Mapping (find_source_file_for_test)

Intelligently maps test files to their corresponding source files:

Mapping Strategy:

1. Remove test prefix/suffix: test_calculator.py → calculator.py
1. Remove test directory: tests/calculator.py → calculator.py
1. Search graph for matching source filename
1. Return matching source file if found


Example Mappings:

* test_calculator.py → calculator.py
* calculator_test.py → calculator.py
* tests/utils.py → utils.py
* math.spec.ts → math.ts


3. Edge Creation (create_test_edges)


Creates bidirectional test edges in the GNN graph:


Algorithm:

1. For each file in graph:
   * If file is a test file:
     * Find corresponding source file
     * If source found:
       * Create TestDependency edge: test_file → source_file
       * For each test function in test_file:
         * Try to infer tested function name
         * If found, create Tests edge: test_fn → source_fn
     * If source not found:
     * Log warning (test file without corresponding source)


Returns: Count of test edges created (for metrics)


Usage Examples

Scenario 1: Test Coverage Analysis

Find all tests for a given source file and identify untested files:

* Query: Get all incoming TestDependency edges for source file
* Filter: Find source files with no incoming test edges
* Result: List of untested files that need test coverage

Scenario 2: Impact Analysis

When source file changes, find affected tests:

* Query: Get incoming TestDependency edges for changed file
* Result: List of test files that need to be re-run
* Benefit: Selective test execution instead of full suite

Scenario 3: Guided Test Generation

Suggest tests for untested functions:

* Query: Get all functions in source file
* Query: Get functions with incoming Tests edges
* Compare: Identify functions without test coverage
* Result: List of untested functions needing test generation


Integration with State Machines

CodeGen State Machine:

* DependencyValidation phase: Check that new code has corresponding tests
* FixingIssues phase: Generate missing tests automatically


Testing State Machine:

* TestGeneration phase: Use untested function list to guide generation
* CoverageAnalysis phase: Calculate coverage using test edge counts
* Complete phase: Verify minimum coverage threshold (e.g., 80%)


Maintenance State Machine (Post-MVP):


* CodeChangeDetection: Identify affected tests using test edges
* TestSelection: Run only affected tests for faster feedback
* CoverageRegression: Alert if coverage drops below threshold


Performance Characteristics

Test File Detection:

* Complexity: O(1) - Simple path pattern matching
* Target: <1ms per file


Test-to-Source Mapping:

* Complexity: O(N) - Linear search through graph nodes
* Target: <10ms per test file
* Optimization: Can be improved with filename index (future enhancement)

Edge Creation:

* Complexity: O(T × F) where T = test files, F = avg functions per test
* Target: <500ms for typical project (100 test files, 10 functions each)
* One-time cost: Run during initial graph build and incremental updates

Impact Analysis:

* Complexity: O(1) - Direct edge lookup using petgraph
* Target: <5ms per query


Benefits

1. Automated Test Coverage Analysis
   * Instantly identify untested code without running tests
   * Track coverage trends over time (stored in SQLite)
1. Intelligent Test Generation
* Guide LLM to generate tests for untested functions
* Prioritize test generation based on code complexity

1. Fast Impact Analysis
* When code changes, immediately know which tests to run
* Enables selective test execution (future: only run affected tests)

1. Quality Metrics
* Calculate test-to-code ratio (# test edges / # source nodes)
* Detect test debt (source files without tests)

1. Self-Healing Tests
* When source function signature changes, identify affected test functions
* Auto-update test calls to match new signatures (future enhancement)



---



### 3.4.6 Agent Interaction Modes: Guided vs Auto Mode (MVP Phase 1 - Priority Feature)

Overview

Yantra supports two distinct interaction modes that control the level of user involvement during autonomous code generation, testing, and deployment workflows. Both modes are fully autonomous - the difference is in when and how the user is informed and consulted.


Core Principle: Yantra is an autonomous agentic platform. The agent executes the entire development lifecycle (generate → test → package → deploy → monitor). The interaction mode determines the frequency and type of user checkpoints during this autonomous execution.

#### 3.4.6.1. Auto Mode

Philosophy: "Trust the agent, work autonomously with strategic checkpoints."

The agent executes the full pipeline autonomously with minimal user interruption.


CRITICAL: Agent determines approval checkpoints at milestone boundaries and CANNOT proceed without user approval at these checkpoints.


User is consulted only for:

1. Milestone approval checkpoints (Agent-determined, feature-set boundaries - e.g., "Authentication Complete", "Payments Functional")
1. Architecture changes (adding/removing components, changing connections)
1. User action blockers (API keys needed, manual setup required, external service configuration)
1. Critical failures after 3 auto-retry attempts


Approval Checkpoint Determination (Autonomous):

Agent automatically identifies logical approval checkpoints by analyzing:

* Feature set boundaries: Grouping related features (Authentication, Payments, Security, Deployment)
* Critical integration points: External dependencies (Stripe, AWS, SendGrid)
* Security-critical milestones: Features requiring verification (Auth, Payment, Data Privacy)
* Architecture complexity: Major component additions or pattern changes
* Risk assessment: High-risk changes that need user validation


Example Checkpoints:


* ✅ Core Infrastructure Complete (Database, API setup, logging)
* ✅ Authentication Features Working (Security-critical verification)
* ✅ Payment Integration Functional (External API validation)
* ✅ Ready for Deployment (Final quality gate)


IMPORTANT: Agent will NOT proceed past any checkpoint without explicit user approval. This ensures user maintains control over project progress while benefiting from autonomous execution between checkpoints.


When Auto Mode Activates:

* User explicitly sets mode: "Switch to auto mode"
* User preference saved in .yantra/config.json
* Suitable for: Production deployments, experienced developers, standard feature implementation



Agent Behavior in Auto Mode:


User: "Add Stripe payment processing"


Agent executes autonomously:

├─ 1. Checkpoint Determination

│   └─ Analyzes feature scope

│   └─ Identifies this as "Payment Integration" milestone

│   └─ Determines approval checkpoint after Stripe integration complete

│   └─ Silent execution (logged to history)

│

├─ 2. Architecture Generation

│   └─ Creates: Gateway → Payment Service → Stripe API → Database

│   └─ Requires user approval (architecture change)

│

Agent PAUSES for architecture approval:

┌────────────────────────────────────────────────┐

│ 📐 ARCHITECTURE CHANGE REQUIRED                │

│                                                │

│ I'm planning to add:                          │

│ - Payment Service (handles Stripe)            │

│ - Payment Database (stores transactions)      │

│                                                │

│ Connections:                                   │

│ - Gateway → Payment Service (REST API)        │

│ - Payment Service → Stripe API (SDK)          │

│ - Payment Service → Payment DB (PostgreSQL)   │

│                                                │

│ Impact: No existing features affected         │

│                                                │

│ Approve? (yes/no/modify)                      │

└────────────────────────────────────────────────┘


User: "yes"

  ↓

├─ 3. Code Generation

│   ├─ Generates: payment_service.py, stripe_client.py, payment_models.py

│   ├─ GNN dependency validation

│   └─ Silent execution

│

├─ 4. Testing

│   ├─ Generates unit tests

│   ├─ Runs pytest

│   ├─ Auto-fixes failures (up to 3 attempts)

│   └─ Silent execution

│

├─ 5. Security Scanning

│   ├─ Semgrep scan

│   ├─ Auto-fixes vulnerabilities

│   └─ Silent execution

│

├─ 6. BLOCKER DETECTED! ⚠️

│   └─ Need Stripe API key

│

Agent PAUSES for user action:

┌────────────────────────────────────────────────┐

│ ⚠️  USER ACTION REQUIRED                       │

│                                                │

│ I need a Stripe API key to continue.          │

│                                                │

│ Options:                                       │

│ 1️⃣  Provide Stripe API key now                │

│ 2️⃣  Skip Stripe integration (mock it)         │

│ 3️⃣  Pause and configure later                 │

│                                                │

│ Your choice:                                   │

└────────────────────────────────────────────────┘


User: "1" + provides key

  ↓

├─ 7. Integration Testing (resumes)

│   ├─ Tests Stripe connection

│   └─ Silent execution

│

├─ 8. Git Commit

│   └─ Commits with message: "feat: Add Stripe payment processing"

│   └─ Silent execution

│

└─ 9. MILESTONE CHECKPOINT REACHED ✅

│

Agent PAUSES for milestone approval:

┌────────────────────────────────────────────────┐

│ ✅ APPROVAL CHECKPOINT: Payment Integration    │

│                                                │

│ Milestone Complete:                            │

│ ✅ Payment Service implemented                 │

│ ✅ Stripe SDK integrated                       │

│ ✅ Payment endpoints functional                │

│ ✅ 15/15 tests passing                         │

│ ✅ No security vulnerabilities                 │

│ ✅ Committed to Git (commit abc1234)           │

│                                                │

│ Review:                                        │

│ - Test payment in browser (live preview)      │

│ - Check code in Code Viewer                   │

│ - View architecture updates in Arch tab       │

│                                                │

│ Approve to proceed to next milestone?         │

│ (yes/modify/explain)                           │

└────────────────────────────────────────────────┘


User: "yes"

  ↓

Agent: "✅ Payment Integration milestone approved!



        Moving to next milestone: Testing & Security Hardening



        Estimated time: 2 days



        [Continues autonomous execution...]"


Auto Mode Logging: All actions are logged to .yantra/logs/agent_activity.log with timestamps, allowing user to review what happened without being interrupted during execution.


Auto Mode Checkpoints Summary:


1. Milestone Approval Checkpoints (Agent-determined, MANDATORY)


* Agent analyzes architecture to determine logical checkpoints
* Typically 3-5 checkpoints per project (feature-set boundaries)
* Agent CANNOT proceed without explicit user approval
* User reviews completed work before next milestone begins


2. Architecture Changes (ALWAYS requires approval)


* Adding/removing components
* Changing connections
* Modifying architectural patterns


3. User Action Blockers (PAUSES until resolved)


* API keys, credentials, tokens
* Manual configuration steps
* External service setup


4. Critical Failures (After 3 auto-retry attempts)


* Test failures that can't be auto-fixed
* Security vulnerabilities requiring manual review
* Integration errors with external services



---


#### 3.4.6.2. Guided Mode (Default for New Users)

Philosophy: "Explain everything, get user consent at each major step."


The agent explains the impact and rationale before executing each major phase. User maintains full visibility and control throughout the autonomous pipeline.


When Guided Mode Activates:

* Default for new users (first 7 days)
* User can explicitly request: "Switch to guided mode"
* Automatically activated after 3 failed attempts in Auto Mode
* Suitable for: Learning Yantra, complex refactoring, high-risk changes


Agent Behavior in Guided Mode:


User: "Add Stripe payment processing"


┌────────────────────────────────────────────────┐

│ 📋 PHASE 1: ARCHITECTURE DESIGN                │

└────────────────────────────────────────────────┘


Agent: "I'll create this architecture:


Components:


├─ Payment Service (new)

│  └─ Handles Stripe integration

│  └─ Files: payment_service.py, stripe_client.py

│

└─ Payment Database (new)

   └─ Stores transaction history

   └─ Table: payments (id, amount, status, timestamp)


Connections:


├─ Gateway → Payment Service (REST API)

├─ Payment Service → Stripe API (Stripe SDK)

└─ Payment Service → Payment DB (SQLAlchemy)


Impact Analysis (via GNN):


✅ No existing components affected

✅ Gateway already has REST API framework


⚠️  Will need Stripe API key


Estimated time: 2-3 minutes

Proceed with architecture? (yes/no/modify)"


User: "yes"


┌────────────────────────────────────────────────┐

│ 📋 PHASE 2: CODE GENERATION                    │

└────────────────────────────────────────────────┘


Agent: "I'll generate these files:


1. payment_service.py (150 lines)

   - create_payment(amount, currency)

   - get_payment_status(payment_id)

   - refund_payment(payment_id)


2. stripe_client.py (80 lines)

   - StripeClient wrapper

   - Error handling and retries

   - Webhook verification


3. payment_models.py (50 lines)

   - Payment SQLAlchemy model

   - Database migrations

Dependencies to install:


- stripe==7.0.0

- sqlalchemy==2.0.23


Impact Analysis (via GNN):


✅ No existing files will be modified

✅ No breaking changes detected


Proceed with code generation? (yes/no)"

User: "yes"


Agent: "✅ Generated 3 files (280 lines total)


┌────────────────────────────────────────────────┐

│ 📋 PHASE 3: TESTING                            │

└────────────────────────────────────────────────┘


Agent: "I'll generate and run these tests:


Unit Tests:


├─ test_payment_service.py (10 tests)

│  └─ Test payment creation, status checks, refunds

│

└─ test_stripe_client.py (8 tests)

   └─ Test Stripe API calls (mocked)


Integration Tests:


└─ test_payment_integration.py (5 tests)

   └─ End-to-end payment flow


Estimated time: 30 seconds


Run tests? (yes/no)"


User: "yes"


Agent: "Running tests..."


[Progress: ████████████████ 100%]


Agent: "✅ 23/23 tests passing


┌────────────────────────────────────────────────┐

│ 📋 PHASE 4: SECURITY SCAN                      │

└────────────────────────────────────────────────┘


Agent: "Running security scans:

- Semgrep (OWASP rules)

- Safety (dependency vulnerabilities)

- TruffleHog (secret detection)


Estimated time: 10 seconds

Run security scan? (yes/no)"


User: "yes"


Agent: "✅ No vulnerabilities found

┌────────────────────────────────────────────────┐

│ 📋 PHASE 5: GIT COMMIT                         │

└────────────────────────────────────────────────┘


Agent: "Ready to commit:


Files changed:

+ payment_service.py (150 lines)

+ stripe_client.py (80 lines)

+ payment_models.py (50 lines)

+ test_payment_service.py (60 lines)

+ test_stripe_client.py (45 lines)

+ test_payment_integration.py (35 lines)


Commit message:

'feat: Add Stripe payment processing

- Add Payment Service with create/status/refund operations

- Integrate Stripe SDK with error handling

- Add Payment database model

- 23 tests (100% passing)

- No security vulnerabilities'

Commit? (yes/no/edit message)"


User: "yes"


Agent: "✅ Committed (abc1234)

Summary:


✅ 6 files created (420 lines)

✅ 23 tests passing


✅ Architecture updated

✅ No security issues

✅ Committed to Git


Next steps:


- Add Stripe API key to .env

- Deploy to staging for integration testing

Done! 🎉"



---


#### 3.4.6.3. Guided Mode: Impact Explanation (Natural Language, Not Code Terms)

Critical Requirement: In Guided Mode, the agent must explain impact using natural language descriptions of features/components, NOT code-level details.


❌ Bad (Code Terms):


Agent: "Modifying payment_service.py will affect:


- Line 45 in gateway.py (import statement)

- Line 120 in api_routes.py (function call)

- test_gateway.py (3 tests need updates)"


✅ Good (Natural Language):


Agent: "Modifying the Payment Service will affect:


Features Impacted:


├─ 🛒 Shopping Cart Feature

│   └─ Uses Payment Service to process checkout

│   └─ Impact: Will need to update checkout flow

│

└─ 📊 Admin Dashboard

    └─ Displays payment statistics

    └─ Impact: Payment status API will have new fields


Components Impacted:


├─ Gateway (minimal changes - just API routing)

└─ Database (no schema changes)


User-Facing Changes:


✅ No breaking changes for end users

⚠️  Admin users will see new 'refund' button


Confidence: High (GNN dependency analysis shows clear boundaries)"


Implementation: Use GNN to map file dependencies → component dependencies → feature dependencies, then translate to user-friendly descriptions.



---


#### 3.4.6.4. Guided Mode: Decision Logging

Requirement: All decisions and changes must be logged with user consent.


Log Format (.yantra/logs/decisions.log):


[2024-11-28 10:23:45] ARCHITECTURE_CHANGE


User Request: "Add Stripe payment processing"

Agent Proposal: Add Payment Service, Payment Database

User Decision: APPROVED

Reasoning: "Need to support online payments"

Impact: 0 existing features affected, 2 new components added

---


[2024-11-28 10:25:30] CODE_GENERATION

Files: payment_service.py, stripe_client.py, payment_models.py

User Decision: APPROVED

GNN Analysis: No breaking changes

---


[2024-11-28 10:26:15] FEATURE_CHANGE

Feature: Shopping Cart

Change: Add payment processing to checkout flow

User Decision: APPROVED

Impact: Checkout flow will now call Payment Service

---


[2024-11-28 10:27:00] COMMIT

Files: 6 files, 420 lines

User Decision: APPROVED

Message: "feat: Add Stripe payment processing"

Commit Hash: abc1234


Log Queries:


User: "Show me all architecture changes this week"

Agent: Retrieves and displays all ARCHITECTURE_CHANGE logs from past 7 days


User: "Why did we add the Payment Service?"

Agent: "On Nov 28, you requested 'Add Stripe payment processing'.

        I proposed adding Payment Service and Payment Database.

        You approved because: 'Need to support online payments'."



---


#### 3.4.6.5. Guided Mode: Progress Status (Project-Level)

Requirement: Give regular status updates on what is implemented and what will be implemented.


Status Update Format:


Agent: "📊 PROJECT STATUS UPDATE

Completed Today:


✅ Payment Service implementation (100%)

✅ Stripe SDK integration (100%)

✅ Payment database models (100%)

✅ Unit tests (23/23 passing)

✅ Security scan (0 vulnerabilities)


In Progress:

⏳ Integration testing (60% complete)

   └─ Remaining: Test webhook handling


Planned Next:


📋 Add refund functionality

📋 Build admin dashboard for payment tracking

📋 Deploy to staging environment


Overall Project Progress:

Core Features: 75% complete

Testing: 85% complete

Documentation: 40% complete

Deployment: 0% complete

Next Milestone: Staging Deployment (Est. 2 days)"


Auto-Generated Progress Reports:

After every major phase (architecture, generation, testing), agent provides summary:


┌────────────────────────────┐

│ 📊 PROGRESS CHECKPOINT                     │

└────────────────────────────┘


Session Started: 10:15 AM

Current Time: 10:45 AM (30 minutes elapsed)


Accomplished:

✅ Architecture designed (Payment Service added)

✅ 6 files generated (420 lines)

✅ 23 tests generated and passing

✅ Security scan passed

✅ Git commit created


Remaining Today:

📋 Add Stripe API key

📋 Run integration tests

📋 Deploy to staging


Blockers:

⚠️  Need Stripe API key to proceed


Next Action:

Please provide Stripe API key or say 'skip' to continue without integration.



---


#### 3.4.6.6. Mode Switching

User Control:


User: "Switch to auto mode"

Agent: "✅ Switched to Auto Mode. I'll execute autonomously and notify you only for:


       - Architecture changes (always require approval)

       - Critical blockers (API keys, manual setup)

       - Failures after 3 retry attempts

       You can switch back anytime with 'switch to guided mode'"


User: "Switch to guided mode"

Agent: "✅ Switched to Guided Mode. I'll explain each step and ask for approval before:

       - Architecture changes

       - Code generation

       - Testing

       - Git commits

       You can switch back anytime with 'switch to auto mode'"


Automatic Mode Switching:


# Auto Mode → Guided Mode (after 3 failures)

Agent: "⚠️  I've failed 3 times trying to fix the database connection.

       Switching to Guided Mode so you can see what's happening.

       Issue: Cannot connect to PostgreSQL on localhost:5432

       Options:

       1️⃣  Help me diagnose (guided troubleshooting)

       2️⃣  Skip database for now (use SQLite)

       3️⃣  Provide database credentials

       Your choice:"



---


#### 3.4.6.7. Testing & Validation

Test Scenarios:


1. Auto Mode - Happy Path
   * User: "Add user authentication"
   * Agent executes full pipeline
   * No user prompts except architecture change
   * Completes in <3 minutes


2. Auto Mode - Blocker
   * User: "Add email sending"
   * Agent hits blocker (need SMTP credentials)
   * Pauses and prompts user
   * Resumes after user provides credentials


3. Guided Mode - Full Explanation


* User: "Add payment processing"
* Agent explains each phase
* User sees impact analysis
* All decisions logged


4. Mode Switching
   * User switches modes mid-session
   * Agent adapts behavior immediately
   * No loss of context



---


#### 3.4.6.8. User Experience Guidelines

Auto Mode UX:

* Minimal interruptions (feels fast and efficient)
* Background progress indicator (small, non-intrusive)
* Detailed logs available on demand
* Notification only for critical events


Guided Mode UX:

* Clear phase headers (Architecture, Generation, Testing, etc.)
* Impact explanations in natural language
* Visual progress bars for each phase
* Approval buttons (Yes/No/Modify)
* Undo/revert options at each checkpoint


Mode Indicator:

* Always visible in UI (top-right corner)
* Shows current mode: 🚀 Auto | 🧭 Guided
* Click to toggle or see mode explanation




---


### 3.4.7 Agent Driven Workflows

Yantra’s Agent driven worklows follows a Preventive Development Cycle (PDC) - a five-phase approach where problems are prevented at each stage rather than fixed after they occur. All workflows align to these phases and are executed through four specialized state machines.


PDC Phases:


1. Architect/Design - Create and validate system architecture before coding
1. Plan - Break down work into executable tasks with dependency awareness
1. Code Generation + Testing - Generate correct, tested code feature-by-feature
1. Deploy - Safe, validated deployments with automatic rollback
1. Maintain - Self-healing systems that detect and fix issues autonomously


Key Principle: Every workflow enforces prevention guarantees - architecture alignment, dependency validation, test coverage, security scanning, and conflict prevention.



---


#### Phase 1: Architect/Design Workflows

##### 1.1 New Project Initialization Workflow

Purpose: Establish architecture-first foundation for new projects before any code generation.


Steps:

1. Collect Requirements/Intent
   * User provides: chat description, upload file (PRD/specs), or external doc link (Notion, Confluence)
   * Agent clarifies ambiguities (auth method, database choice, framework preferences)
   * User can say "use defaults" for standard technology choices
1. Generate Architecture
* Agent creates architecture from finalized requirements using LLM
* Architecture displayed in Architecture View tab
* Components, connections, and boundaries defined
* Approval Gate: User reviews and approves (modifications allowed)
* Saved as version 1 baseline in SQLite

1. Generate Technical Specifications
* Auto-generated after architecture approval
* Includes: component specs, API contracts, data models, security requirements
* Displayed in Documentation tab
* Approval Gate: User reviews and approves

1. Extract Features
* Parsed from requirements, architecture, and technical specs
* All features marked ⏳ Planned (0% complete)
* Displayed in Features tab with descriptions and estimates


5. Create Project Plan with Milestones
   * Agent autonomously determines 3-5 approval checkpoints by analyzing:
     * Related features (e.g., Authentication, Payments, Infrastructure)
     * Security-critical features requiring validation
     * External dependencies and integration points
   * Each milestone marked with ✅ APPROVAL CHECKPOINT
   * Displayed in Plan tab with estimates and dependencies
   * Approval Gate: User approves checkpoint placement (modifications allowed)
5. Select Interaction Mode
* Choice offered AFTER plan approval:
* Guided Mode: Approve every step (architecture, code, tests, commits)
* Auto Mode: Approve only at milestones + architecture changes

* Agent CANNOT proceed past approval checkpoints without explicit user consent

5. Log All Decisions
* Record to Decision Log: architecture source, requirements source, timestamps, all approvals


Output:

* ✅ Approved architecture (Version 1)
* ✅ Technical specifications document
* ✅ Features list (all ⏳ Planned)
* ✅ Project plan with approval checkpoints
* ✅ Mode selection (Auto/Guided)
* ✅ Decision log entries


Prevention Guarantees:

* Architecture exists before code generation begins
* Requirements clarified before implementation
* Plan validated before execution
* All decisions documented with audit trail


---


##### 1.2 Existing Project Import Workflow

Purpose: Discover and document architecture for existing codebases, establish governance baseline.


Phase 1: Multi-Level Architecture Discovery

Level 1: Priority Files (Exact Paths)

* .yantra/architecture.db (100% confidence - native format)
* .yantra/architecture.json (90% confidence - Yantra export)
* architecture.md, ARCHITECTURE.md (80% confidence - root directory standard)
* architecture.json (80% confidence - root directory JSON)

Level 2: Common Documentation Locations (Pattern Match)

* docs/*.md - All markdown files recursively
* .github/*.md - System design documentation
* wiki/*.md - Wiki-based documentation
* design/*.md, architecture/*.md - Design folders
* docs/adr/*.md, docs/decisions/*.md - Architecture Decision Records


Level 3: Content-Based Detection (Deep Scan)

* Search for architecture section headers: # Architecture, ## System Architecture, etc.
* Detect diagram formats: Mermaid (graph TD, C4Context), PlantUML (@startuml), ASCII diagrams
* Keyword scoring: "component", "service", "microservice" (high-value), "authentication", "cache" (medium-value)
* Calculate confidence score: 90-100% (use immediately), 70-89% (confirm), 50-69% (present as option), <50% (request guidance)


Phase 2: Requirements/Specs Discovery

* Search for requirements and specification similar to Architecture discovery Level 2 and Level3 - fuzzy search for requirements and specifications in filename and content
* Extract and score by confidence (similar to architecture discovery)


Phase 3: Handle Discovery Results (5 Cases)


Case A: Architecture Found, No Requirements

* Import architecture to Architecture View
* Options presented to user:
  1. Generate specs from code review (recommended)
  1. Skip specs for now
  1. Analyze code for both architecture and requirements


Case B: Requirements Found, No Architecture

* Import requirements to Documentation tab
* Options presented to user:

1. Generate architecture from requirements (recommended)
1. Generate architecture from code review
1. Upload architecture file manually


Case C: Both Found (Architecture + Requirements)

* Import both to respective tabs
* Show in Architecture View + Documentation tabs
* Approval Gate: User reviews and approves imported content


Case D: Multiple Architecture Sources (50-70% confidence)

* Present all options to user with confidence scores
* User chooses: select one, combine all, or analyze code instead


Case E: Low/No Documentation (<50% or none)

* Options presented to user:

1. User describes project in chat (guided conversation)
1. Auto-analyze code (GNN-based discovery)
1. Upload/import files (Markdown, JSON, Mermaid, PlantUML)


Phase 4: Code Review Assessment (for Existing Code)

1. GNN Dependency Analysis


* Parse all source files using tree-sitter
* Build dependency graph (functions, classes, imports, calls)
* Identify components and their relationships


2. Security Scanning
   * Run Semgrep with OWASP rules (parallel processing: 4 workers)
   * Check dependencies for vulnerabilities (Safety for Python, npm audit)
   * Detect secrets (TruffleHog patterns)
   * Categorize by severity (Critical/High/Medium/Low)


3. Code Quality Assessment
   * Calculate cyclomatic complexity
   * Measure test coverage
   * Identify code smells and unused code


4. Architecture Alignment Validation
   * Compare code structure to imported/generated architecture
   * Detect: missing components, extra components, incorrect connections
   * Flag mismatches with severity levels


5. Generate Recommendations
   * Prioritize: 1) Security issues, 2) Architecture misalignment, 3) Code quality
   * Suggest: update architecture, refactor code, fix vulnerabilities
   * Provide actionable options to user


Phase 5: Feature Extraction with Status

* Extract from: requirements, architecture, specs, AND existing code
* Use GNN to detect implemented features
* Calculate completion status:
  * ✅ Done: 100% (files exist, tests pass, no errors)
  * 🔄 In Progress: 1-99% (partial implementation)
  * ⏳ Planned: 0% (not started)
* Display in Features tab with completion percentages


Phase 6: Create Plan with Current State Assessment

* Agent determines milestones based on current project state
* Prioritizes: 1) Complete in-progress features, 2) Implement pending features, 3) Harden/test
* Mark approval checkpoints
* Show estimates and current completion status


Phase 7: Mode Selection & Decision Logging

* Same as new project workflow (Guided or Auto mode)
* Log all decisions: architecture source, import method, timestamps


Output:

* ✅ Imported/generated architecture (Version 1)
* ✅ Code review report (security, quality, alignment)
* ✅ Features list with actual completion status (✅ 🔄 ⏳)
* ✅ Project plan reflecting current state
* ✅ Mode selection (Auto/Guided)
* ✅ Decision log entries


Performance Limits:

* Max files to scan: 1,000 (configurable)
* Max file size: 5 MB per file
* Timeout: 30 seconds for full scan
* Parallel processing for security scan and code analysis


Prevention Guarantees:

* Existing architecture discovered and documented
* Code quality and security assessed before new development
* Current state accurately reflected in plan
* All findings documented with severity levels



---


##### 1.3 Architecture Maintenance & Governance Workflow

Purpose: Continuously ensure code aligns with architecture during development through deviation detection and user-approved updates.


Trigger 1: During Code Generation (Proactive)


Scenario: Agent generates code that would deviate from planned architecture.

Process:

1. Agent generates code using LLM
1. Before writing to disk, parse code with tree-sitter to extract imports/dependencies
1. Load current architecture from SQLite
1. Query GNN for expected dependencies of target component
1. Compare actual imports vs allowed dependencies


On Deviation Detected:

* Pause code generation
* Calculate severity (Low/Medium/High/Critical):
  * Critical: Security breach, circular dependency
  * High: Breaks layering (e.g., UI → Database directly)
  * Medium: Skips one layer (e.g., Gateway → Database, bypassing service)
  * Low: Extra utility import (minor)
* Present to user with 3 options:
* Update Architecture: Modify architecture to allow new dependency (creates version N+1)
* Fix Code: Regenerate code using LLM to match existing architecture
* Cancel: Stop code generation entirely

* Wait for user decision
* Execute chosen action


Trigger 2: After Manual Code Edit (Reactive)


Scenario: User manually edits code, potentially breaking architecture alignment.


Process:


1. On file save, detect change via file watcher
1. Check if project has active architecture
1. Run GNN analysis on saved file to get current dependencies
1. Compare to architecture expectations for that component


On Misalignment Detected:


* Display warning notification in UI
* Add system message to chat with details:
  * File path and deviation summary (expected → actual)
  * Severity level with visual indicator
  * User-friendly explanation
  * Recommended actions
* User decides: update architecture, revert code, or refactor code


Trigger 3: New Feature Impacts Architecture (Requirement-Driven)

Scenario: User requests feature that requires architecture changes.


Process:

1. Analyze Requirement Impact
   * Parse user requirement using LLM
   * Use GNN to identify affected files and components
   * Determine if architecture changes needed
1. Preview Architecture Changes
* List new components to add
* List modified connections
* List affected existing components
* Calculate impact severity (Low/Medium/High/Breaking):
* Low: Changes within single component, no new dependencies
* Medium: Changes across 2-3 components, existing dependencies
* High: New component required
* Breaking: Fundamental pattern change (e.g., monolith → microservices)


3. Require User Approval
   * Show architecture changes BEFORE implementation
   * Display affected components visually
   * Approval Gate: Explicit approval required


4. Execute After Approval
   * Update architecture (increment version to N+1)
   * Generate code matching approved architecture
   * Validate final code aligns with architecture


Auto-Save with Rule of 3 Versioning: (Post MVP)


* Keep current version + 3 most recent past versions (total: 4 versions)
* When 5th version created, auto-delete oldest (version 1)
* Each version immutable once created
* Version metadata: incremental number, full snapshot, timestamp, change type, agent reasoning, original user message


Agent Commands (Via Chat):


* "Show me the architecture" - Opens Architecture View tab, displays current version
* "Revert to previous architecture" - Loads version N-1, auto-saves as new version N+1
* "Show architecture history" - Lists last 3 versions with timestamps and changes
* "Why did you add Redis?" - Shows version history and reasoning


Output:


* ✅ Updated architecture version (N+1)
* ✅ Deviation report (if detected)
* ✅ Code aligned with approved architecture
* ✅ Version history maintained (Rule of 3)
* ✅ Decision log entry with reasoning


Prevention Guarantees:

* Architecture never drifts from code
* All deviations detected at state boundaries (not after completion)
* Every architecture change has documented reasoning
* Can rollback to previous architecture versions
* No code generation proceeds without architecture alignment



---


#### Phase 2: Plan Workflows

##### 2.1 Task Decomposition & Sequencing Workflow

Purpose: Break down features into executable tasks with proper dependency ordering and conflict awareness.


Integrated into: Code Generation State Machine (states 5-10)


Process:

1. TaskDecomposition (<500ms)
   * GNN-based feature analysis
   * Break down into atomic implementation tasks
   * Each task maps to specific files/components
   * Prevents: missing tasks, unbounded work, unclear scope


2. DependencySequencing (<100ms)
   * GNN graph traversal to identify task dependencies
   * Topological sort for execution order
   * Prevents: wrong task order, dependency violations, parallel conflicts


3. ConflictCheck (<50ms)
   * MVP: Query active work indicators - which developers are editing which files
   * Post-MVP: Query file lock table for locked files
   * Prevents: parallel edit conflicts, work coordination issues


4. PlanGeneration (<200ms)
   * Create final execution plan with estimates
   * Complexity analysis for each task
   * Priority assignment
   * Prevents: unclear scope, missing estimates, unprioritized work


5. PlanReview (optional approval gate)
   * Triggered for: Features with >5 tasks OR multi-file changes
   * User reviews: task breakdown, sequencing, estimates, complexity
   * Approval Gate: User approves or requests modifications
   * Prevents: misaligned expectations, scope disagreements, over-complexity


6. EnvironmentSetup (automated)
   * Virtual environment creation/validation
   * Dependency assessment and installation
   * Version validation
   * Prevents: environment configuration errors, missing dependencies


Output:


* ✅ Task breakdown with file mappings
* ✅ Dependency-ordered execution sequence
* ✅ Conflict awareness (MVP: visibility, Post-MVP: locks)
* ✅ Estimates and complexity scores
* ✅ Optional plan approval (for complex features)
* ✅ Ready environment


Prevention Guarantees:


* Explicit planning prevents unbounded work
* Tasks properly sequenced based on dependencies
* Parallel work conflicts minimized (MVP) or prevented (Post-MVP)
* All tasks scoped and estimated before execution



---


#### Phase 3: Code Generation + Testing Workflows

##### 3.1 Feature Implementation Workflow (Pair Programming)

Purpose: Generate production-quality code feature-by-feature using Yantra Codex + Multi-LLM consultation.


Integrated into: Code Generation State Machine + Testing State Machine


Code Generation Process:

1. ContextAssembly (<100ms)
   * Load relevant code context from GNN
   * Retrieve semantic embeddings for similar patterns
   * Assemble context for LLM prompt
   * Prevents: reinventing functionality, inconsistent patterns, missing context


2. CodeGeneration (Pair Programming with Multi-LLM Consultation)


Phase A: Initial Generation


* Yantra Codex (GNN) generates initial code (15ms)
* Calculate confidence score (0.0-1.0)
* If confidence >= 0.8: Yantra code used directly
* If confidence < 0.8: Send to Primary LLM for review and enhancement
* Primary LLM (user's choice: Claude/GPT-4/Gemini) reviews edge cases, adds error handling
* Merge Yantra + LLM suggestions
* Yantra learns from LLM improvements for future


The above step Phase A should work with Yantra Codex as optional (may not be ready for MVP)

*


Phase B: Validation & Retry


* Run validation: tree-sitter (syntax), LSP (types), linter (style)
* Run tests automatically
* Attempt 1: If validation fails → Primary LLM retries with error context
* Attempt 2: If still fails → Primary LLM retries again


Phase C: Multi-LLM Consultation (After 2 Failures)


* Trigger: After 2 consecutive failures with same issue
* Consultation Flow:
  1. Primary LLM generates consultation prompt (meta-prompting)
  1. Consultant LLM (different model) provides second opinion:
     * Identifies blind spots or framing issues
     * Suggests alternative approaches
     * Points out what Primary LLM might be missing
  1. Primary LLM regenerates code WITH consultant's insight added to context
* Attempt 3: Primary with consultant insight (higher success rate)
* Cost Benefit: 2 failures + 1 consultation + 1 success = 4 calls (vs 5+ attempts without consultation)


Phase D: Escalation (If All Fail)


* After 3 attempts (including consultation): Escalate to user
* Options: Try different primary model, manual intervention, skip feature


Why Multi-LLM Consultation Works:


* Different LLMs have different training data and biases
* Claude: Strong reasoning, cautious on edge cases
* GPT-4: Pragmatic, compact code style
* Gemini: Different perspective, fresh approach
* When stuck, problem is often framing or blind spot - different LLM sees differently


UI Visibility:


* MVP: Shows which files are being generated (work visibility)
* Shows attempt number and which LLM is being consulted
* Example: "🤔 Consulting GPT-4 Turbo for second opinion..."


Prevents:


* Syntax errors (tree-sitter validation)
* Type errors (LSP validation)
* Style issues (linter validation)
* Getting stuck on same approach (multi-LLM brings fresh perspective)
* Excessive failed attempts (consultation after 2 failures)





3. DependencyValidation (<10ms)
   * GNN impact analysis on generated code
   * Check if changes break dependent code
   * Identify ripple effects
   * Prevents: breaking changes to dependents


4. BrowserValidation (5-10s)
   * Load generated code in headless Chrome via CDP
   * Capture screenshots, check console errors
   * Validate network requests
   * Prevents: visual regressions, runtime errors, network failures


5. SecurityScanning (parallel: 4 workers, ~5s)
   * Semgrep with OWASP rules
   * Detect: SQL injection, XSS, secrets in code
   * Check dependencies for vulnerabilities
   * Prevents: security vulnerabilities before commit


6. ConcurrencyValidation (<5s, only for parallel code)
   * Static analysis (Clippy, threading analyzers)
   * GNN data flow analysis for race conditions
   * Stress test generation for concurrent code paths
   * Auto-fix: pattern-based → LLM-assisted → sequential fallback
   * Prevents: race conditions, deadlocks, data races


7. FixingIssues (if any validation fails)
   * Auto-retry with LLM-generated fixes
   * Max 3 retries per issue
   * If still failing, escalate to user
   * Prevents: quality shortcuts


Testing Process (Auto-triggered after CodeGen Complete):

1. TestGeneration
   * LLM generates comprehensive unit tests
   * Pattern-based test templates
   * Edge case coverage
   * Prevents: missing test coverage, untested edge cases
1. EnvironmentSetup
* Virtual environment ready
* Test fixtures prepared
* Dependencies installed
* Prevents: environment inconsistencies


3. UnitTesting (parallel: 4 workers, ~8s typical)
   * pytest/jest/cargo test execution
   * Run tests for modified code + affected dependents
   * Quality Gate: ALL tests must pass (100%)
   * Prevents: logic errors, edge case failures, function behavior bugs


4. BrowserTesting (parallel: 3 browsers, ~17s typical)
   * Playwright E2E tests
   * Test user workflows (login→dashboard→logout)
   * Responsive design, accessibility tests
   * Prevents: user workflow failures, integration issues, UI bugs


5. IntegrationTesting (20-40s)
   * Component interaction tests
   * API contract validation
   * Database integration tests
   * Prevents: component interaction failures, API violations


6. CoverageAnalysis
   * pytest-cov/coverage.py execution
   * Quality Gate: >80% coverage required
   * Identify gaps in validation
   * Prevents: insufficient test coverage


7. FixingIssues (if tests fail)
   * Regenerate code with LLM
   * Re-run affected tests
   * Max 3 retries
   * If still failing, escalate to user


Output:

* ✅ Production-quality code (validated)
* ✅ Comprehensive unit tests (100% pass)
* ✅ Browser validation passed
* ✅ Integration tests passed
* ✅ >80% test coverage
* ✅ Security scan clean
* ✅ Concurrency validation passed (if applicable)
* ✅ Ready for deployment


Prevention Guarantees:

* Code correct by construction (multi-layer validation)
* All tests pass before proceeding (no exceptions)
* Security vulnerabilities caught before commit
* Concurrency bugs prevented or code falls back to safe sequential execution
* Breaking changes to dependents prevented
* Visual regressions prevented


Cost Optimization:

* Month 1: 55% handled by Yantra alone (64% cost savings)
* Month 6: 85% handled by Yantra alone (88% cost savings)
* Year 1: 95% handled by Yantra alone (96% cost savings)



---


##### 3.2 Multi-Agent Parallel Development Workflow (Post MVP)

Purpose: Enable multiple agents to work on different features simultaneously without conflicts.

Integrated into: Code Generation State Machine (ConflictCheck state) + Multi-Agent Orchestration


MVP Implementation (Work Visibility):


1. Master Decomposes Feature (30s)


* GNN analyzes dependencies
* Breaks feature into sub-features by:
  * File ownership (backend vs frontend vs tests vs docs)
  * Dependency relationships (API must complete before UI)
* Assigns sub-features to agents
* Shows which files each agent will modify


2. Agents Check Active Work (1s)


* Query active work indicators: which developers are editing which files
* UI displays: "Agent-1 working on payment.py", "You're working on checkout.tsx"
* Agents coordinate to avoid parallel edits on same files
* If conflict risk detected, agent waits or works on different files first


3. Parallel Execution (time varies)


* Agents work on independent files simultaneously
* UI shows real-time progress for each agent
* Dependent agents wait for prerequisite agents to complete


4. Sequential Completion


* Agents complete work based on dependency order
* Each agent creates separate PR
* All PRs reviewed and merged
* Prevention: Coordination through visibility, not enforcement


Post-MVP Enhancement (File Locking):


* Explicit FileLockAcquisition state before code generation
* GNN-aware dependency locking (lock file + all dependencies)
* FileLockRelease state after completion
* Merge conflicts become impossible by design


Example Flow:


User: "Add complete checkout flow with Stripe payments"

  ↓

Master: Analyzes GNN dependencies


  - Sub-feature 1: Payment API (Backend) - payment.py, api/checkout.py

  - Sub-feature 2: Checkout UI (Frontend) - Checkout.tsx

  - Sub-feature 3: Tests - test_checkout.py

  - Sub-feature 4: Docs - docs/checkout.md


  ↓

Agent-1 (Backend): Checks active work → Starts payment.py (no conflicts)

Agent-2 (Frontend): Checks dependencies → Waits for Agent-1 API completion


Agent-3 (Testing): Waits for Agent-1 and Agent-2


Agent-4 (Docs): Starts docs/checkout.md (no dependencies, parallel)


  ↓


Agent-1 completes (10 min) → Notifies Agent-2


Agent-2 starts (5 min) → Notifies Agent-3


Agent-3 starts (3 min)


Agent-4 completes (2 min, parallel)


  ↓


Total: 18 minutes (vs 45 minutes single-agent) = 2.5x faster


Output:


* ✅ Multiple features implemented simultaneously
* ✅ Zero merge conflicts (visibility-based coordination in MVP)
* ✅ Proper dependency sequencing maintained
* ✅ Separate PRs for each sub-feature


Prevention Guarantees:


* MVP: Parallel work conflicts minimized through visibility
* Post-MVP: Parallel work conflicts impossible through locking
* Dependency order preserved
* Work coordination automated



---


#### Phase 4: Deploy Workflows

##### 4.1 Safe Deployment Workflow

Purpose: Deploy validated code to production with automatic health checks and rollback capability.

Integrated into: Deployment State Machine (5 states)


Process:


1. PackageBuilding (30-60s)
   * Docker build for containerized apps
   * Artifact compilation (wheels, npm packages)
   * Asset bundling and optimization
   * Prevents: broken build artifacts, missing dependencies


2. ConfigGeneration (automated)
   * Generate railway.json deployment config
   * Create/update Dockerfile
   * Validate environment variables
   * Generate health check endpoints
   * Prevents: environment mismatches, missing configuration


3. RailwayUpload (20-40s)
   * Atomic upload to Railway platform via API
   * All-or-nothing deployment
   * Version tagging and metadata
   * Prevents: partial deployments, manual errors


4. HealthCheck (5-10s)
   * HTTP health check endpoint validation
   * Smoke tests on deployed environment
   * Response time validation
   * Database connectivity check
   * Prevents: deploying broken code, service unavailable


5. RollbackOnFailure (automatic)
   * If health check fails: instant rollback
   * Revert to last known good version
   * Alert user with failure details
   * Prevents: prolonged outages


Approval Gate:

* Manual trigger required for production deployment (human-in-loop for safety)
* User approves after reviewing: changes summary, test results, deployment plan
* Agent executes deployment automatically after approval


Output:

* ✅ Live production URL (e.g., https://app.railway.app)
* ✅ Health status confirmed
* ✅ Deployment audit trail
* ✅ Version tagged and documented


Rollback Capability:

* User can request rollback via chat: "Rollback to previous version"
* Agent executes rollback using Railway API
* Previous version restored in <2 minutes


Prevention Guarantees:

* Code fully tested before deployment (100% tests pass)
* Environment configuration validated
* Health checks prevent deploying broken code
* Automatic rollback on failure
* Zero manual deployment errors



---


#### Phase 5: Maintain Workflows

##### 5.1 Self-Healing Workflow (Post-MVP)

Purpose: Automatically detect, diagnose, fix, validate, and deploy fixes for production issues without human intervention.


Integrated into: Maintenance State Machine (11 states)


Process:


1. LiveMonitoring (continuous)


* Sentry/Datadog integration via MCP
* Real-time error tracking
* Performance monitoring
* Uptime validation
* Prevents: prolonged outages, undetected issues


2. BrowserValidation (continuous)


* Real User Monitoring (RUM)
* Session replay for errors
* Browser error tracking
* Performance degradation detection
* Prevents: silent user-facing failures


3. ErrorAnalysis (<5s)


* Pattern matching against known issues
* Severity classification (Critical/High/Medium/Low)
* Error correlation and grouping
* Prevents: repeated incidents, unknown root causes


4. IssueDetection (<5s)


* Root cause analysis
* GNN dependency graph queries for impact
* Historical pattern matching
* Prevents: misdiagnosed issues, wasted effort


5. AutoFixGeneration (10-30s)


* LLM generates fix based on error analysis
* Use pattern-based solutions from knowledge base
* Apply historical fixes for similar issues
* Prevents: manual intervention delays, human error


6. FixValidation (2-3 min)


* Run fix through full CodeGen + Testing cycle
* All tests must pass (100%)
* Security scan validation
* Browser validation
* Prevents: deploying broken fixes, introducing new bugs


7. CICDPipeline (1-2 min)


* Automated CI/CD pipeline execution
* Deployment to production
* Health checks
* Prevents: manual deploy delays, inconsistent fixes


8. VerificationCheck (ongoing)


* Monitor error rate after fix deployed
* Confirm issue resolved in production
* Performance validation
* User impact assessment
* Prevents: fix didn't work, issue persists


9. LearningUpdate (automated)


* Update Yantra Codex with new pattern
* Add monitoring for similar issues
* Create post-incident report
* Document fix in knowledge base
* Prevents: repeated incidents, knowledge loss


State Transitions:


* Active (normal operation) → Incident (on error detected)
* Incident → Auto-healing cycle → Active (issue resolved)
* Incident → EscalatedToHuman (if auto-fix fails after 3 retries)


Mean Time To Repair (MTTR):


* Known patterns: <5 minutes (detect → fix → deploy → verify)
* Novel issues: <15 minutes (includes LLM analysis and custom fix generation)
* Complex issues: Escalate to human after 3 auto-fix attempts


Output:


* ✅ Issue automatically detected
* ✅ Root cause identified
* ✅ Fix generated and validated
* ✅ Fix deployed to production
* ✅ Issue verified resolved
* ✅ Knowledge base updated
* ✅ Post-incident report created


Prevention Guarantees:


* Issues detected before users report (proactive)
* Fixes fully validated before deployment (same quality as human-written)
* No new bugs introduced by fixes
* Knowledge accumulated for future prevention
* MTTR <5 minutes for known patterns



---


##### 5.2 Proactive Monitoring & Prevention Workflow (Post-MVP)

Purpose: Detect potential issues before they become production problems.


Integrated into: Maintenance State Machine (LiveMonitoring state)


Proactive Checks:


1. Performance Degradation Detection


* Monitor response times over time
* Detect gradual slowdowns (e.g., 500ms → 2s over 1 week)
* Alert when threshold exceeded
* Auto-generate optimization recommendations


2. Error Rate Trend Analysis


* Track error rates by endpoint/feature
* Identify increasing error trends (e.g., 0.1% → 0.5% over 2 days)
* Investigate before reaching critical threshold
* Prevent: errors becoming outages


3. Dependency Version Monitoring


* Check for new package versions weekly
* Scan for security advisories
* Test compatibility in staging
* Auto-generate upgrade PR if safe


4. Resource Usage Monitoring


* Track memory/CPU/disk usage
* Predict capacity issues (e.g., disk 80% full → will fill in 3 days)
* Alert before resource exhaustion
* Prevent: out-of-memory crashes, disk full errors


5. External API Health Monitoring


* Monitor third-party API availability (Stripe, SendGrid, etc.)
* Detect degraded performance
* Switch to fallback providers if configured
* Alert user of external issues


Output:


* ✅ Potential issues detected early
* ✅ Proactive recommendations generated
* ✅ Auto-generated PRs for safe upgrades
* ✅ Capacity planning alerts
* ✅ External dependency health visibility


Prevention Guarantees:


* Issues caught before becoming critical
* Capacity planning prevents resource exhaustion
* Security vulnerabilities patched proactively
* External API issues don't surprise users



---


#### Summary: Workflows → PDC Phases → State Machines

Consolidated Workflow Mapping:


| PDC Phase | Workflows | State Machines | Approval Gates |
| - | - | - | - |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Architect/Design | 1.1 New Project Init1.2 Existing Project Import1.3 Architecture Maintenance | CodeGen (states 1-4) | Architecture, Specs, Plan |
| - | - | - | - |
| 2. Plan                                                                                                                                             | 2.1 Task Decomposition & Sequencing | CodeGen (states 5-10) | Plan Review (optional) |
| 3. Code Gen + Test                                                                                                                                  | 3.1 Feature Implementation3.2 Multi-Agent Parallel | CodeGen + Testing | None (quality gates enforce) |
| 4. Deploy                                                                                                                                           | 4.1 Safe Deployment | Deployment (states 1-5) | Manual trigger |
| 5. Maintain                                                                                                                                         | 5.1 Self-Healing5.2 Proactive Monitoring | Maintenance (states 1-11) | None (auto-heal) |


Key Principles:


* ✅ Architecture ALWAYS exists before code generation
* ✅ Plans are explicit and validated
* ✅ Code generated feature-by-feature with full testing
* ✅ Deployments are safe with automatic rollback
* ✅ Systems self-heal without human intervention
* ✅ All workflows enforce prevention guarantees


Total Cycle Time (MVP):


* New Project Init: <2 min (user input time varies)
* Existing Project Import: <1 min (+ scan time for large projects)
* Feature Implementation: <2 min per feature (code + tests)
* Deployment: <2 min (build + health check)
* Self-Healing: <5 min (detect + fix + deploy + verify)





---




### 3.4.8 Cascading Failure Protection : Cross Cutting Policy

Architecture Note: This is a policy specification defining failure recovery behavior across all state machines. For implementation details, see:


* State Persistence & Checkpoints: State Machine Architecture → Session Persistence section
* Impact Assessment: Code Generation State Machine → BlastRadiusAnalysis state
* Automated Testing: Test Intelligence + Test Execution State Machines
* Failure Recovery: Code Generation + Testing State Machines → FixingIssues state
* User Escalation: State Machine Architecture → Failed state


Problem: AI agents can enter failure loops where each attempted fix creates new problems, progressively degrading the codebase.


Solution: Every modification is reversible with one click. The system automatically detects cascading failures and reverts to the last known working state.



---


#### 3.4.4.1 Checkpoint System (Critical - MVP Foundation)

Purpose:Ensure every state change can be undone instantly.


When Checkpoints Are Created (mapped to state machines):


| Checkpoint Type | State Machine Trigger | Confidence Score Basis |
| - | - | - |
| ------------------------------------------------------------------ |
| Session | CodeGen machine initialization | 100% (clean start) |
| - | - | - |
| Feature                                                          | BeforeTaskDecompositionstate | 100% (all tests passing) |
| File                                                             | BeforeCodeGenerationstate | Based on previous test results |
| Test                                                             | Before Test Execution machine starts | Based on code validation results |
| Manual                                                           | User-triggered via UI | Current system state |


What gets saved in each checkpoint:

* File snapshots (all modified files)
* Dependency graph state (GNN serialization)
* Architecture version ID
* Test results summary
*  Confidence score (0-100% based on test pass rate)


Storage:.yantra/checkpoints/ directory, last 20 kept, gzip compressed, auto-pruned in background.


User Experience:

✅ Session Start - 10:15 AM | 8 files | 100% confidence [Restore]

🔨 Before "Add Payment" - 10:20 AM | 8 files | 100% [Restore]

⚠️ After Payment (Failed) - 10:25 AM | 11 files | 45% [Restore]

Recommendation: Restore "Before Add Payment" (100% confidence)


**Performance:** Create <500ms, Restore <1s, List <50ms

**Implementation:** See State Machine Architecture → Session Persistence


---


#### 3.4.4.2 Impact Assessment Policy

Purpose: Show users exactly what will break BEFORE making changes.


What's Analyzed:

* Dependent files (via GNN traversal)
* Affected features (GNN feature mapping)
* Risk score (30% file risk + 30% feature risk + 40% criticality)
* Test impact (which tests run, estimated failures, runtime)
*

Also see Section [3.4.3.5](https://docs.google.com/document/d/12XNkWFaxRBLZotJWu3JT0jAmBbjL6y4ODmXkWfd2n9I/edit?tab=t.0#heading=h.3n6038s1lq1r) that captures Impact Analysis under Dependency Conflict Detection & Resolution


Display Trigger Rules:


* Always show: Critical files, breaking changes, >10 dependencies, package upgrades
* Show for large changes: >5 tasks, >3 files, >20 tests
* Skip for small changes: Single file, no dependencies, <5 tests


Performance: <100ms (GNN queries)

Implementation: See Code Generation State Machine → BlastRadiusAnalysis state



---



#### 3.4.4.3. Failure Recovery Policy

Purpose: Auto-fix issues up to 3 times, then escalate to user.


Attempt Flow:


| Attempt | Action | State Transition | Recovery |
| - | - | - | - |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | Auto-fix with LLMCheck[Known Issues and fix (ection 3.1.3.4)](https://docs.google.com/document/d/12XNkWFaxRBLZotJWu3JT0jAmBbjL6y4ODmXkWfd2n9I/edit?tab=t.0#heading=h.xvpn7rld3li7)in Dependency graph for resolution with semantic search<br/> | →FixingIssuesstate | Auto-revert to last checkpoint with 95%+ confidence |
| - | - | - | - |
| 2                                                                                                                                                                                                                                                                                                                      | Use[LLM Consulting mode (Section 3.4.1.1)](https://docs.google.com/document/d/12XNkWFaxRBLZotJWu3JT0jAmBbjL6y4ODmXkWfd2n9I/edit?tab=t.0#heading=h.lne7e5lwc4fu)<br/> | →FixingIssuesstate | Auto-revert to last checkpoint with 95%+ confidence |
| 3                                                                                                                                                                                                                                                                                                                      | Escalate to user | →Failedstate | User chooses next action |


Failure Tracking:

* Counter increments on each failure
* Resets to 0 when tests pass
* Stores attempt number, timestamp, error type, error message


Performance: Revert <1s, Fix generation 2-5s (LLM dependent)

Implementation:


* See Code Generation State Machine → FixingIssues state
* See Test Execution State Machine → FixingIssues state
*

User Actions:


* Provide Info: User enters missing data → retry with new context
* Switch LLM: Try different LLM approach → retry
* Web Search: Trigger knowledge search (see next section) → retry with findings
* Skip: Mark feature as deferred → continue with other work


---


#### 3.4.4.4. Web Search Consent Policy

Purpose: Respect user privacy while enabling external knowledge access.

Web Search:

* Search Google, StackOverflow, GitHub for solutions
* Query is sent to external search APIs
* User must consent before this happens


User Consent Flow:

When agent wants to search for a solution:

1. Show Search Request:


I'd like to search for solutions to this problem:

Query: "How to handle Stripe API authentication errors in Python"

Where should I look?


1️⃣  Web search (Google, StackOverflow, GitHub)

2️⃣  Retry

4️⃣ Neither (I'll provide guidance)


Your choice:


2. Wait for User Response:
   * User selects option
   * No search happens until user responds
2. Execute Approved Search:
* If web only: Use browser automation to search
* If retry: Prompt LLM to deep dive
* If neither: ask user for guidance


Solution Display:


* Source (Local/StackOverflow/GitHub/etc.)
* Relevance score (0-100%)
* Code snippet (if available)


Implementation: See State Machine Communication → Web Search Integration



---


#### 3.4.4.5. Testing & Validation

Purpose: Detect regressions immediately after every code change.


Test Selection Strategy:


* Use GNN to find tests that cover modified files
* Don't run entire suite (too slow)
* Example: Modified payment_service.py → run only tests importing it


Regression Detection:


* Compare current results with previous run
* Flag tests that passed before but now fail
* Count: new failures vs existing failures


Trigger Recovery: New failures detected → increment failure counter → transition to FixingIssues


Performance: Test selection <10ms (GNN query), execution time varies by test count


Implementation: See Test Execution State Machine → all states



---


#### 3.4.4.6. Performance Targets

Speed Requirements:


| Operation | Target Time | Notes |
| - | - | - |
| -------------------------------------------------------- |
| Create checkpoint | Under 500ms | Snapshot 10-20 files |
| - | - | - |
| Restore checkpoint                                     | Under 1 second | Restore all files + dependency graph |
| Impact analysis                                        | Under 100ms | Dependency graph traversal |
| List checkpoints                                       | Under 50ms | Query from memory |
| Delete old checkpoints                                 | Under 200ms | Background cleanup, doesn't block work |


Why These Targets Matter:


* Checkpoints should be nearly instant so they don't slow down development
* Restore must be fast (under 1 second) for good user experience
* Impact analysis needs to be quick so user doesn't wait
* Background operations should never block the user



---


#### 3.4.4.7. Storage Management


Location:.yantra/checkpoints/


Retention Policy:

* Keep last 20 checkpoints by default
* User-marked "important" never deleted
* Auto-compress old checkpoints (gzip)
* Background pruning (non-blocking)


Disk Usage: 50-200 MB typical (20 checkpoints)

Implementation: See State Machine Architecture → Session Persistence → Storage



---


#### System Guarantees

✅ Reversible: Every change has checkpoint, one-click restore

✅ Immediate Detection: Tests run after every modification

✅ Auto-Recovery: Up to 2 automatic fix attempts

✅ User Control: Escalation after 3 failures

✅ Privacy: Web search only with explicit consent

✅ Fast: All operations <1s except LLM generation

✅ Efficient: Auto-cleanup, compressed storage


Result: Agent can try bold solutions without risk. User never loses work.




---


## Advanced Post MVP features


## Phase 2A: Team of Agents Architecture (Months 3-4)

Overview

Problem: As codebases scale beyond 100k LOC and teams grow to 5+ concurrent developers, single-agent architecture becomes a bottleneck. Developers need multiple AI agents working simultaneously on different parts of the codebase without conflicts.


Solution: Transform Yantra from a single autonomous agent to a team of coordinating agents using a Master-Servant architecture with Git coordination branch and Agent-to-Agent (A2A) protocol for proactive conflict prevention.


Key Innovation: Unlike traditional collaborative editing (which reactively resolves conflicts), Yantra uses proactive conflict prevention - agents communicate intent before making changes via GNN dependency analysis and real-time file locking.


Why Team of Agents?

Current Limitations (Single Agent):


* Only one developer can use Yantra at a time
* Large codebases (100k+ LOC) exceed single-agent capacity
* Complex features require serial execution of multiple tasks
* Bottleneck for team collaboration


Team of Agents Benefits:


* Parallelization: 3-10 agents working simultaneously on same feature
* Specialization: Dedicated agents for frontend, backend, testing, DevOps
* Scalability: Handle 100k+ LOC codebases efficiently
* Team Collaboration: Multiple developers with their own agents working on same project
* Fault Tolerance: One agent failure doesn't block others
* 3x Faster: 15 minutes instead of 45 minutes for complex features

### Architecture: Master-Servant Pattern

Core Philosophy: Master assigns work once, then agents execute independently on their own Git branches with peer-to-peer coordination.


Why Master-Servant over Peer-to-Peer?


Rejected: Pure Peer-to-Peer (P2P)


* ❌ No single source of truth - who decides work split?
* ❌ Complex consensus algorithms needed (Raft/Paxos)
* ❌ Race conditions on file writes
* ❌ Conflict resolution is reactive (after-the-fact)
* ❌ Difficult to debug - no clear authority


Chosen: Master-Servant with Git Coordination


✅ Master's Limited Role:


* Master decomposes work and assigns ONCE
* Master is not a bottleneck (not involved in ongoing work)
* Master can crash after assignment - agents continue


✅ Git Provides Code Coordination:


* Branches provide isolation (proven infrastructure)
* PRs provide integration (normal Git workflow)
* Conflicts surface at merge time (standard resolution)


✅ sled Provides Lock Coordination:


* Atomic file claims (compare-and-swap)
* Real-time file locking (proactive conflict prevention)
* Agent state tracking (progress, locks held)


✅ Mirrors Human Teams:


* Tech lead assigns features (Master)
* Developers work on branches independently (Agents)
* Developers coordinate when needed (A2A protocol)
* Team reviews PRs (human review or auto-merge)


✅ Simple State Management:


* Coordination state is append-only events (Git branch)
* File locks are key-value pairs (sled)
* Easy to debug and audit


Key Insight from Design:


Traditional master-servant has the master coordinating everything (bottleneck). Yantra's master only assigns work at the beginning, then gets out of the way. Git handles code isolation. sled handles lock coordination. Agents coordinate peer-to-peer for dependencies. This hybrid approach combines the best of centralized (clear work assignment) and decentralized (independent execution).



---


### Master Agent Responsibilities

Role: Work assignment ONLY - Master is NOT a coordinator or bottleneck


Core Principle: Master analyzes and assigns work at the beginning, then agents execute independently. Master does not approve operations, manage locks, or coordinate ongoing work. Master's job is done after assignment.


Why This Design?


* ✅ Master is not a single point of failure (if Master crashes after assignment, agents continue)
* ✅ Agents don't need Master's permission for ongoing work
* ✅ Git already solves file coordination (branches isolate, merges detect conflicts)
* ✅ Matches human team workflow (tech lead assigns features, developers work independently)


Master's Three Responsibilities:


1. Feature Decomposition:


* Receives high-level feature request from user
* Uses GNN to analyze dependencies and complexity
* Decomposes into 3-10 sub-features with minimal cross-dependencies
* Goal: Create sub-features with loose coupling to minimize inter-agent blocking
* Example: "Add Stripe payments" → Backend API, Frontend form, Testing, DevOps


2. Agent Assignment:


* Assigns each sub-feature to specialized servant agent
* Creates Git feature branches for each agent: feature/agent-1-payment-api
* Commits assignment events to .yantra/coordination branch
* Master's job is complete - can shut down or take new requests


3. User Review (Optional Advisory Role):


* Shows proposed work split to user before agents start
* User can adjust assignments if they see coupling issues
* Example: "Agent 1 (Backend), Agent 2 (Frontend) share UserService - is this split okay?"
* Master is advisory, not authoritative - assignments can be adjusted


What Master Does NOT Do:


* ❌ Approve every file operation (not a bottleneck)
* ❌ Coordinate ongoing work (agents are fully independent)
* ❌ Resolve runtime conflicts (Git branches + Tier 2 locks handle this)
* ❌ Manage file locks (Tier 2 sled handles this peer-to-peer)
* ❌ Monitor agent progress (agents self-coordinate)



---


### Servant Agent Responsibilities

Role: Fully independent execution with peer-to-peer dependency coordination


Core Principle: Agents work autonomously on their own Git branches, claim files from Tier 2 (sled) as needed, and coordinate peer-to-peer when dependencies arise. No central coordinator approval needed.


Agent Lifecycle:


1. Startup & Assignment Discovery:


* Pull .yantra/coordination branch to find assignment from Master
* Create dedicated feature branch from main: feature/agent-1-payment-api
* Register self in Tier 2 (sled): registry:agent:agent-1
* Agent is now independent - no further Master interaction needed


2. Before Modifying Files (Proactive Conflict Prevention):


* Query GNN: "What files do I need to modify for this feature?"
* Check Tier 2 locks: "Is any file I need already claimed by another agent?"
* If locked by another agent:
  * Option A: Wait and poll until released
  * Option B: Work on other parts that don't have the dependency
  * Option C: Send peer-to-peer message to coordinate approach
* If free: Claim files atomically in Tier 2: lock:src/payment/stripe.py = agent-1
* Git branch provides isolation: Even if lock fails, branch prevents corruption


3. During Work (Independent Execution):


* Modify files on own Git branch (full isolation from other agents)
* Update progress state in Tier 2: agent:agent-1:state = { phase: "CodeGen", progress: 45% }
* Release locks immediately when done with file: DELETE lock:src/payment/stripe.py
* No Master check-ins required - agent executes autonomously


4. Peer-to-Peer Dependency Coordination (A2A Protocol):


* Scenario: Agent B needs API that Agent A is building
* Agent B checks coordination branch: "Has Agent A completed user auth API?"
* If completed:
  * Agent B reads dependency_available event with commit hash
  * Agent B pulls Agent A's branch: git pull origin feature/agent-1-auth
  * Agent B continues work with Agent A's code available
* If not completed:
* Agent B sends A2A message via Tier 2: a2a:agent-2:to:agent-1:QueryDependency
* Agent A responds with ETA or commit hash when done
* Agent B works on non-dependent tasks in the meantime

* Key insight: Agents coordinate directly, no Master involvement


5. Feature Completion:


* Run full test suite on feature branch
* Create Pull Request when feature is ready
* Commit feature_completed event to coordination branch with PR number
* Release all remaining file locks
* Request human review or auto-merge if tests pass
* Agent work is complete - can take new assignment or shut down


Agent Independence Benefits:


* ✅ Fault tolerant: One agent crash doesn't block others
* ✅ Scalable: No central bottleneck for approvals
* ✅ Fast: No round-trip to Master for every operation
* ✅ Flexible: Agents can adjust strategy based on real-time conditions
* ✅ Recovery: Agents can resume from coordination branch after crashes



---


### Git Coordination Branch

Purpose: Append-only event log for LOW-FREQUENCY feature assignments and completions


Key Design Principle: Git branch for coordination state, NOT for high-frequency operations. Git provides version-controlled, auditable, human-readable coordination history that works locally and distributed.


Branch Name:.yantra/coordination (never merges to main)


Why Git Branch for Coordination?


✅ No New Infrastructure: Git is already there, no new database needed ✅ Version Controlled: Full history of assignments and completions ✅ Human Readable:git log .yantra/coordination shows all coordination events ✅ Auditable: Who assigned what to whom, when features completed ✅ Works Everywhere: Same mechanism works locally (single machine) and distributed (team with remote Git) ✅ Merge Conflicts Rare: Append-only events rarely conflict ✅ Distributed by Design: Every agent has local copy, syncs via push/pull


What Goes in Coordination Branch (Low-Frequency):


* ✅ Feature assignments by Master
* ✅ Work started events
* ✅ Dependency available events (APIs ready)
* ✅ Feature completion events with PR numbers


What Does NOT Go Here (High-Frequency):


* ❌ File lock claims/releases (Tier 2 sled handles this - 100k ops/sec)
* ❌ Real-time progress updates (Tier 2 sled handles this)
* ❌ Agent heartbeats (Tier 2 sled handles this)
* ❌ A2A messages (Tier 2 sled handles this)


Separation of Concerns:


* Coordination branch: Strategic coordination (assignments, completions) - low frequency
* Tier 2 (sled): Tactical coordination (locks, messages, state) - high frequency


Event Types:


// 1. Feature Assignment (Master assigns work)


{


  "event": "feature_assigned",


  "feature": "stripe-payment-processing",


  "agent": "agent-1",


  "branch": "feature/agent-1-payment-api",


  "files_estimate": ["src/payment/stripe.py", "src/api/checkout.py"],


  "timestamp": "2025-11-30T10:00:00Z",


  "dependencies": []


}


// 2. Work Started (Agent begins execution)


{


  "event": "work_started",


  "agent": "agent-1",


  "feature": "stripe-payment-processing",


  "timestamp": "2025-11-30T10:05:00Z"


}


// 3. Dependency Available (Agent completes API that others need)


{


  "event": "dependency_available",


  "agent": "agent-1",


  "api": "PaymentService.charge",


  "module": "src/payment/stripe.py",


  "commit_hash": "abc123def456",


  "branch": "feature/agent-1-payment-api",


  "timestamp": "2025-11-30T10:45:00Z"


}


// 4. Feature Completed (Agent finishes, creates PR)


{


  "event": "feature_completed",


  "agent": "agent-1",


  "feature": "stripe-payment-processing",


  "branch": "feature/agent-1-payment-api",


  "pr_number": 42,


  "files_modified": ["src/payment/stripe.py", "src/api/checkout.py"],


  "tests_passed": true,


  "timestamp": "2025-11-30T11:00:00Z"


}


Agent Operations:


// Agent pulls coordination state on startup


pub async fn get_my_assignment(&self) -> Result<Assignment> {


    // Pull latest events from coordination branch


    git::pull_branch(".yantra/coordination")?;


    // Read all JSON events from commit messages


    let events = git::read_json_commits(".yantra/coordination")?;


    // Find my assignment


    events.iter()


        .find(|e| e.event == "feature_assigned" && e.agent == self.id)


        .cloned()


        .ok_or_else(|| anyhow!("No assignment found for {}", self.id))


}


// Agent checks if dependency is available from another agent


pub async fn check_dependency(&self, api_name: &str) -> Result<Option<String>> {


    git::pull_branch(".yantra/coordination")?;


    let events = git::read_json_commits(".yantra/coordination")?;


    // Find most recent dependency_available event for this API


    events.iter()


        .filter(|e| e.event == "dependency_available" && e.api == api_name)


        .max_by_key(|e| e.timestamp)


        .map(|e| e.commit_hash.clone())


}


// Agent commits completion event


pub async fn mark_complete(&self, pr_number: u32) -> Result<()> {


    let event = json!({


        "event": "feature_completed",


        "agent": self.id,


        "feature": self.current_feature,


        "branch": self.branch_name,


        "pr_number": pr_number,


        "files_modified": self.modified_files,


        "tests_passed": true,


        "timestamp": Utc::now()


    });


    // Commit to coordination branch


    git::commit_to_branch(".yantra/coordination", &event)?;


    git::push_branch(".yantra/coordination")?;


    Ok(())


}


Recovery After Crashes:


One of the key benefits of the coordination branch is fault tolerance:


// Agent crashes and restarts


pub async fn resume_from_crash(&self) -> Result<()> {


    // Pull coordination branch to see what I was doing


    git::pull_branch(".yantra/coordination")?;


    let events = git::read_json_commits(".yantra/coordination")?;


    // Find my last assignment


    let assignment = events.iter()


        .filter(|e| e.event == "feature_assigned" && e.agent == self.id)


        .max_by_key(|e| e.timestamp)?;


    // Check if I already completed this


    let completed = events.iter()


        .any(|e| e.event == "feature_completed"


             && e.agent == self.id


             && e.feature == assignment.feature);


    if completed {


        // I'm done, wait for new assignment


        return Ok(());


    }


    // Resume work on my branch


    git::checkout_branch(&assignment.branch)?;


    self.continue_work(assignment)?;


    Ok(())


}


Human Inspectable:


Developers can directly inspect coordination state:


# See all coordination events


git log .yantra/coordination --oneline


# See current assignments


git show .yantra/coordination | grep "feature_assigned"


# See which dependencies are available


git log .yantra/coordination --grep="dependency_available"


# Debug why agent is blocked


git log .yantra/coordination --grep="agent-2" --since="1 hour ago"


This transparency is crucial for debugging and understanding system behavior.



---


### Architecture Design Principles

Why This Design Is Clean and Scalable:


1. Master Is Not a Single Point of Failure


Traditional multi-agent systems have a master that coordinates every operation. If the master crashes, all agents stop. Yantra's master only acts at the BEGINNING (work assignment). After assignment:


* ✅ Agents continue independently if Master crashes
* ✅ Agents don't need Master's permission for ongoing work
* ✅ Master can shut down or take new requests immediately after assignment


2. Git Already Solves Code Coordination


Rather than inventing new file coordination, Yantra uses proven Git infrastructure:


* ✅ Branches provide isolation - each agent works on their own branch
* ✅ PRs provide integration - completed work merges through normal Git flow
* ✅ Conflicts surface at merge time - handled like human team conflicts
* ✅ Proven at scale - Git handles millions of developers worldwide


3. Matches Human Team Workflow


Yantra mirrors how real development teams work:


| Human Team | Yantra Multi-Agent |
| - | - |
| ----------------------------------------------------------- |
| Tech lead assigns features | Master assigns sub-features |
| - | - |
| Developers create branches                                | Agents create feature branches |
| Developers work independently                             | Agents execute independently |
| Developers coordinate when needed                         | Agents use A2A protocol for dependencies |
| Developers submit PRs                                     | Agents create PRs with tests |
| Team reviews and merges                                   | Human reviews or auto-merge |


4. Separation of Concerns (Three Coordination Layers):


| Layer | Purpose | Frequency | Technology | Examples |
| - | - | - | - | - |
| ------------------------------------------------------------------------------------------------------------------------------------------ |
| Git Coordination Branch | Strategic assignments & completions | Low (once per feature) | Git commits | feature_assigned, feature_completed |
| - | - | - | - | - |
| Tier 2 (sled)                                                                                                                            | Tactical locks & messages | High (many per second) | sled KV store | File locks, A2A messages, agent state |
| Git Feature Branches                                                                                                                     | Code isolation | N/A | Git branches | Actual code changes |


Each layer does what it's best at. No layer tries to do everything.


5. Proactive Conflict Avoidance (Not Reactive Resolution)


Traditional Git workflow:


1. Two developers modify same file without knowing
1. Both invest time in divergent implementations
1. Git detects conflict at merge time (REACTIVE)
1. Manual resolution required, often by someone who wrote neither


Yantra workflow:


1. Agent A claims file in Tier 2 sled
1. Agent B tries to claim same file
1. Tier 2 blocks immediately (PROACTIVE)
1. Agent B either waits or works on different task
1. No wasted work, no merge conflicts


6. Event Sourcing for Recovery


The coordination branch is an event log. If an agent crashes:


// Agent reads event log on restart


let events = git::read_json_commits(".yantra/coordination")?;


// "I was assigned feature X at timestamp T"


let my_assignment = events.find(|e| e.agent == self.id)?;


// "Here's my branch name"


let my_branch = my_assignment.branch;


// Agent resumes from where it left off


git::checkout_branch(my_branch)?;


self.continue_work(my_assignment)?;


State is never lost. Agents can always resume.


7. Low-Hanging Fruit Implementation


This architecture requires minimal new infrastructure:


* ✅ Git is already there (no new database for coordination)
* ✅ sled is embedded Rust library (no server to run)
* ✅ Coordination state is version controlled and auditable
* ✅ Works locally (single machine) and distributed (team) with same mechanism
* ✅ Merge conflicts on coordination branch are rare (append-only events)



---


### Tier 2 (sled) - Real-Time File Locking

Purpose: High-frequency, low-latency coordination for file locks and agent state


Why sled (not Git)?


* Git coordination branch: Low-frequency events (assignments, completions)
* sled: High-frequency operations (file claims/releases every few seconds)
* Git: Human-readable history
* sled: Machine-optimized performance (100k writes/sec)


Data Model:


// File Locks


"lock:src/payment/stripe.py" = "agent-1"


"lock:src/components/PaymentForm.tsx" = "agent-2"


// Agent State


"agent:agent-1:state" = { phase: "CodeGen", progress: 45%, files: [...] }


"agent:agent-2:state" = { phase: "Testing", progress: 80%, files: [...] }


// Agent Registry


"registry:agent:agent-1" = { type: "Backend", started_at: "2025-11-30T10:00:00Z" }


"registry:agent:agent-2" = { type: "Frontend", started_at: "2025-11-30T10:01:00Z" }


// A2A Messages


"a2a:agent-2:to:agent-1:1" = { type: "QueryDependency", api: "PaymentService.charge" }


"a2a:agent-1:to:agent-2:1" = { type: "DependencyReady", commit: "abc123" }


File Lock API:


pub struct CoordinationStore {


    db: sled::Db, // ~/.yantra/coordination.sled


}


impl CoordinationStore {


    // Atomic file claim (returns false if already locked)


    pub fn claim_file(&self, agent_id: &str, file_path: &str) -> Result<bool> {


        let key = format!("lock:{}", file_path);


        let result = self.db.compare_and_swap(


            key.as_bytes(),


            None as Option<&[u8]>,


            Some(agent_id.as_bytes()),


        )?;


        Ok(result.is_ok())


    }


    // Release file lock


    pub fn release_file(&self, file_path: &str) -> Result<()> {


        let key = format!("lock:{}", file_path);


        self.db.remove(key.as_bytes())?;


        Ok(())


    }


    // Check which agent holds lock


    pub fn check_lock(&self, file_path: &str) -> Result<Option<String>> {


        let key = format!("lock:{}", file_path);


        if let Some(agent_id) = self.db.get(key.as_bytes())? {


            Ok(Some(String::from_utf8(agent_id.to_vec())?))


        } else {


            Ok(None)


        }


    }


    // Query GNN for dependencies and check their locks


    pub async fn check_dependencies(&self, file_path: &str, gnn: &GNN) -> Result<Vec<String>> {


        let dependencies = gnn.get_dependencies(file_path).await?;


        let mut locked_deps = Vec::new();


        for dep in dependencies {


            if let Some(agent_id) = self.check_lock(&dep)? {


                locked_deps.push(format!("{} (locked by {})", dep, agent_id));


            }


        }


        Ok(locked_deps)


    }


}



---


### Proactive Conflict Prevention Workflow

Philosophy: Prevent Conflicts BEFORE They Happen


Traditional Git workflow is REACTIVE - conflicts detected after both developers finish work:


* Developer A modifies payment.py on branch A
* Developer B modifies payment.py on branch B
* Both invest days of work
* Git detects conflict at merge time
* Manual resolution required, understanding two divergent implementations


Yantra workflow is PROACTIVE - conflicts prevented before work starts:


* Agent A claims payment.py in sled
* Agent B tries to claim payment.py
* sled blocks immediately (atomic compare-and-swap)
* Agent B works on different files or waits
* No wasted work, no merge conflicts


Git as the Coordination Layer:


Git already solves file coordination through branches and merges. Yantra leverages this:


1. Branches Provide Isolation:


* Each agent works on their own feature branch
* File changes are isolated from other agents
* No cross-agent corruption possible


2. PRs Provide Integration:


* Completed work merges through normal Git flow
* Tests run on each PR before merge
* Human review or auto-merge based on test results


3. Conflicts Surface at Merge Time:


* When they do occur (rare with file locking), handled like human team conflicts
* Git's conflict resolution tools work normally
* Developers understand this workflow already


Proactive Prevention via File Locks:


sled adds real-time lock coordination on top of Git:


* Agent claims file → lock created in sled
* Other agents check lock before claiming
* Atomic compare-and-swap prevents race conditions
* Locks released immediately when agent finishes with file


Example: E-Commerce Checkout Feature with 6 Agents


User Request: "Add complete checkout flow with Stripe payments"


Step 1: Master Decomposes Using GNN (30 seconds)


Master analyzes dependency graph to minimize coupling:


Master queries GNN for dependencies:


├─ Sub-feature 1: Payment API (Backend)


│  Files: src/payment/stripe.py, src/api/checkout.py


│  Dependencies: None (foundational)


│


├─ Sub-feature 2: Checkout UI (Frontend)


│  Files: src/components/Checkout.tsx


│  Dependencies: Payment API (needs PaymentService.charge)


│


├─ Sub-feature 3: Cart Integration (Frontend)


│  Files: src/components/Cart.tsx


│  Dependencies: None (independent UI component)


│


├─ Sub-feature 4: Integration Tests (Testing)


│  Files: tests/test_checkout.py


│  Dependencies: Payment API + Checkout UI (needs both complete)


│


├─ Sub-feature 5: Deployment Config (DevOps)


│  Files: .github/workflows/deploy.yml


│  Dependencies: None (infrastructure, runs in parallel)


│


└─ Sub-feature 6: Documentation (Docs)


   Files: docs/checkout.md


   Dependencies: None (can document as features complete)


Master assignment strategy:


- Agent 1 starts first (Backend) - others depend on this


- Agent 2 waits for Agent 1's API (Frontend has dependency)


- Agent 3, 5, 6 start immediately (no dependencies)


- Agent 4 waits for 1 & 2 (Testing needs both)


Step 2: Agents Claim Files (1 second)


// Agent 1 (Backend) - no dependencies, starts immediately


coordination_store.claim_file("agent-1", "src/payment/stripe.py")?;   // ✅ Success


coordination_store.claim_file("agent-1", "src/api/checkout.py")?;     // ✅ Success


// Agent 2 (Frontend) - checks dependencies first


let deps = gnn.get_dependencies("src/components/Checkout.tsx")?;


// deps = ["src/payment/stripe.py"] (imports PaymentService)


let locked_deps = coordination_store.check_dependencies("src/components/Checkout.tsx", &gnn)?;


// Returns: ["src/payment/stripe.py (locked by agent-1)"]


// Agent 2 decision logic:


if locked_deps.is_empty() {


    // No blocking dependencies, proceed


    coordination_store.claim_file("agent-2", "src/components/Checkout.tsx")?;


} else {


    // Dependency locked, two options:


    // Option A: Start on Cart.tsx (no dependencies)


    coordination_store.claim_file("agent-2", "src/components/Cart.tsx")?; // ✅ Success


    // Option B: Wait for Agent 1 to finish and poll


}


// Agent 3, 5, 6 proceed immediately (no dependencies)


coordination_store.claim_file("agent-3", "src/components/Cart.tsx")?;        // ❌ Already claimed by Agent 2


coordination_store.claim_file("agent-5", ".github/workflows/deploy.yml")?;   // ✅ Success (parallel)


coordination_store.claim_file("agent-6", "docs/checkout.md")?;               // ✅ Success (parallel)


Key Insight: Agent 2 detected dependency BEFORE starting work. No wasted effort.


Step 3: Parallel Execution (10 minutes)


Agent 1 (Backend):     [████████████████████] 100% - stripe.py, checkout.py ✅


                       Releases locks, commits dependency_available event


Agent 2 (Frontend):    [██████████░░░░░░░░░░] 50%  - Cart.tsx done, waiting for API


                       Polls coordination branch for Agent 1 completion


Agent 3 (Frontend):    [Blocked] - Cart.tsx claimed by Agent 2


Agent 4 (Testing):     [░░░░░░░░░░░░░░░░░░░░] 0%   - Waiting for Agent 1 & 2


Agent 5 (DevOps):      [████████████████████] 100% - deploy.yml done (parallel) ✅


Agent 6 (Docs):        [████████████████████] 100% - docs done (parallel) ✅


Step 4: Dependency Resolution via Coordination Branch (1 minute)


// Agent 1 commits completion to coordination branch


{


  "event": "dependency_available",


  "agent": "agent-1",


  "api": "PaymentService.charge",


  "module": "src/payment/stripe.py",


  "commit_hash": "abc123def456",


  "branch": "feature/agent-1-payment-api",


  "timestamp": "2025-11-30T10:45:00Z"


}


// Agent 2 polls coordination branch (every 10 seconds)


let dependency_ready = coordination_branch.check_dependency("PaymentService.charge")?;


if let Some(commit) = dependency_ready {


    // Pull Agent 1's branch to get the API


    git::fetch_branch("feature/agent-1-payment-api")?;


    git::merge_branch("feature/agent-1-payment-api")?;


    // Now continue with Checkout.tsx implementation


    coordination_store.claim_file("agent-2", "src/components/Checkout.tsx")?; // ✅ Success


    self.continue_checkout_implementation()?;


}


Agent 2 completes Checkout.tsx in 5 minutes ✅


Agent 4 sees both Agent 1 and Agent 2 complete → Starts integration tests Agent 4 completes in 3 minutes ✅


Step 5: PR Creation (30 seconds)


All agents complete → Create PRs:


├─ PR #1: Payment API (agent-1) ✅ Tests pass, 32 assertions


├─ PR #2: Checkout UI (agent-2) ✅ Tests pass, 18 assertions


├─ PR #3: Cart Integration (agent-2) ✅ Tests pass, 12 assertions


├─ PR #4: Integration Tests (agent-4) ✅ Tests pass, 45 E2E tests


├─ PR #5: Deployment (agent-5) ✅ CI pipeline validated


└─ PR #6: Documentation (agent-6) ✅ Spelling/link checks passed


Human reviews and merges (or auto-merge if confidence > 0.95)


Total Time: 15 minutes (vs 45 minutes single-agent) = 3x faster


Conflicts Prevented:


* 🚫 Agent 2 didn't modify stripe.py while Agent 1 was working (proactive detection)
* 🚫 Agent 3 didn't conflict with Agent 2 on Cart.tsx (file lock prevented)
* 🚫 Agent 4 didn't start tests until APIs were complete (dependency tracking)
* 🚫 No Git merge conflicts (each agent on own branch with lock coordination)
* ✅ All work proceeded in optimal order with minimal blocking


Why This Works:


1. GNN provides dependency knowledge - Master knows what depends on what
1. sled provides activity knowledge - Agents know who's modifying what right now
1. Combining both prevents conflicts - Check dependencies AND check locks before claiming
1. Git provides safety net - Even if locks fail, branches prevent corruption
1. Peer-to-peer coordination - Agents don't need Master after initial assignment



---


Scalability Limits


Per-Feature Parallelism:


* Optimal: 5-10 agents (sweet spot for most features)
* Maximum: 15-20 agents (diminishing returns beyond this)
* Bottleneck: File lock contention (not storage)


System-Wide Parallelism:


* Phase 2: 3-5 features simultaneously (15-50 agents system-wide)
* Phase 3: 10+ features simultaneously (50-100 agents system-wide)
* Theoretical: 1,000+ agents (sled can handle it)
* Practical: 100-200 agents (file locks become bottleneck)


Why Not More Agents Per Feature?


* Small feature (login): Only 3-5 files to modify → 3-5 agents max
* Medium feature (payments): 10-15 files → 6-10 agents optimal
* Large feature (dashboard): 30+ files → 15-20 agents, then diminishing returns


Master Agent Overhead:


* Assignment decomposition: ~30 seconds per feature
* Not a bottleneck (one-time cost, then agents run independently)



---


### Implementation Tasks (Phase 2A - 4 weeks)

Week 9: Git Coordination Branch


* Implement Git branch creation (.yantra/coordination)
* Event commit/read functions (JSON serialization)
* Agent assignment parser
* Event types: feature_assigned, work_started, dependency_available, feature_completed


Week 10: Tier 2 (sled) Migration


* Add sled dependency to Cargo.toml
* Implement CoordinationStore (file locks, agent state, registry, A2A messages)
* Migrate agent state from SQLite to sled
* File lock API: claim_file(), release_file(), check_lock()


Week 11: Master-Servant Implementation


* Master Agent: Feature decomposition using dependency graph (Note: code may call this "GNN" but it's the petgraph structure, not a neural network)
* Master Agent: Work assignment algorithm (minimize dependencies)
* Servant Agent: Assignment reading from coordination branch
* Servant Agent: File claiming before modification
* Servant Agent: State updates during work


Week 12: A2A Protocol


* A2A message types: QueryDependency, DependencyReady, IntentToModify
* Message sending/receiving via Tier 2 (sled)
* Dependency resolution workflow
* Conflict prevention logic (check locks before claiming)


Success Metrics:


* ✅ 3-10 agents can work simultaneously on same feature
* ✅ Zero file conflicts (all prevented proactively)
* ✅ 3x faster feature completion (15 min vs 45 min)
* ✅ Master assignment overhead <30 seconds
* ✅ File lock operations <5ms
* ✅ 100+ agents supported system-wide



---


## Phase 2B: Cloud Graph Database (Tier 0) - Months 4-5

IMPORTANT: This is NOT "Cloud Dependency Graph Intelligence" - the dependency analysis runs locally in Tier 1. This is a cloud-hosted graph database for storage and real-time coordination across agents/users.


Overview

Problem: With Team of Agents, each agent has a LOCAL dependency graph. When Agent A modifies a file, Agent B doesn't know until attempting to claim the same file or hitting a Git merge conflict. This is reactive conflict detection (bad).


Solution:Cloud Graph Database (Tier 0) - A shared, cloud-hosted dependency graph that tracks real-time file modifications across all agents and users working on the same project. Enables proactive conflict prevention BEFORE work starts.


Key Innovation: Combine dependency knowledge (from local dependency graph) with activity knowledge (who's modifying what) to warn agents about potential conflicts before they occur, including transitive dependencies.


Why Cloud Graph Database?

Current Problem (Local GNN Only):


* Agent A modifies payment.py
* Agent B wants to modify checkout.py which imports from payment.py
* Agent B doesn't know Agent A is changing payment.py until:
  * File lock prevents claim (good but late)
  * Git merge conflict after both finish (bad - wasted work)


Cloud Graph Database Solution:


* Agent A starts work → Marks payment.py as "in modification" in Cloud Graph DB
* Agent B queries Cloud Graph DB: "Dependencies of checkout.py?"
* Cloud Graph DB responds: "payment.py (locked by Agent A) is a dependency"
* Agent B warned BEFORE starting work: "Your file depends on payment.py being modified by Agent A. Recommend: wait 10 min, or coordinate approach."


Conflict Prevention Levels:


Level 1: Same File Detection


* Agent A claims payment.py
* Agent B tries to claim payment.py
* Tier 2 (sled) blocks immediately ✅ (already implemented)


Level 2: Direct Dependency Detection 🆕


* Agent A modifies payment.py
* Agent B wants to modify checkout.py which imports from payment.py
* Cloud Graph DB warns: "Your file depends on a file being modified"


Level 3: Transitive Dependency Detection 🆕


* Agent A modifies database.py
* user.py depends on database.py
* auth.py depends on user.py
* Agent B wants to modify auth.py
* Cloud Graph DB traces chain: "auth.py → user.py → database.py (Agent A modifying)"


Level 4: Semantic Dependency Detection 🆕


* Agent A changes signature of authenticate(username, password) → authenticate(email, password, mfa_code)
* Cloud Graph DB knows 47 files call authenticate()
* Any agent touching those 47 files gets warned: "Function you're using is being modified. Signature may change."



---


### Architecture: Hybrid Local + Cloud

Design Principle: Local GNN for speed, Cloud Graph Database for coordination


┌─────────────────────────────────────────────────────────────────┐


│ Tier 0: Cloud Graph Service (Hosted or Self-Hosted)            │


│ ────────────────────────────────────────────────────────────────│


│ Per-Project Isolation:                                          │


│ ├─ project:abc123:graph (dependency structure)                  │


│ ├─ project:abc123:locks (file modification registry)            │


│ └─ project:abc123:agents (active agent list)                    │


│                                                                  │


│ API Endpoints (WebSocket/gRPC, <50ms latency):                 │


│ ├─ claim_file(project_id, agent_id, file_path)                 │


│ ├─ release_file(project_id, file_path)                         │


│ ├─ query_dependencies(project_id, file_path) → Vec<FileDep>    │


│ ├─ query_conflicts(project_id, file_path) → Vec<Conflict>      │


│ └─ sync_graph(project_id, graph_delta) → SyncResult            │


└─────────────────────────────────────────────────────────────────┘


         ↓↑ WebSocket/gRPC (50ms sync latency)


┌─────────────────────────────────────────────────────────────────┐


│ Tier 1: Local GNN (In-Memory, <1ms queries)                     │


│ ────────────────────────────────────────────────────────────────│


│ ├─ Fast local queries (get_dependencies, find_callers)          │


│ ├─ Syncs graph structure TO Cloud Graph DB (every 30s or on change) │


│ ├─ Queries Cloud Graph DB BEFORE claiming files                 │


│ └─ Privacy: Only sends graph structure, NOT code content        │


└─────────────────────────────────────────────────────────────────┘


Hybrid Benefits:


* Local GNN: Millisecond queries for code generation (hot path)
* Cloud Graph Database: Real-time coordination across agents (warm path)
* Privacy: Only graph structure synced, not code content



---


### Data Model: Privacy-Preserving

What Gets Shared (Graph Structure Only):


pub struct CloudGraphNode {


    pub file_path: String,           // "src/payment.py"


    pub node_type: NodeType,          // File, Function, Class


    pub name: String,                 // "PaymentService"


    pub exports: Vec<String>,         // ["charge", "refund"]


}


pub struct CloudGraphEdge {


    pub source: String,               // "src/checkout.py"


    pub target: String,               // "src/payment.py"


    pub edge_type: EdgeType,          // Import, Call, Inherit


    pub symbol: String,               // "PaymentService.charge"


}


pub struct FileModification {


    pub file_path: String,            // "src/payment.py"


    pub agent_id: String,             // "agent-1"


    pub started_at: DateTime<Utc>,   // "2025-11-30T10:00:00Z"


    pub estimated_completion: Option<DateTime<Utc>>, // "2025-11-30T10:15:00Z"


}


What NEVER Gets Shared (Code Content):


// ❌ NO CODE CONTENT


pub struct PrivateData {


    pub source_code: String,          // NEVER SENT


    pub function_body: String,        // NEVER SENT


    pub variable_values: String,      // NEVER SENT


    pub api_keys: String,             // NEVER SENT


}


Example: Shared vs Private


// ✅ SHARED (structure only)


CloudGraphNode {


    file_path: "src/payment.py",


    node_type: NodeType::Function,


    name: "charge_customer",


    exports: ["charge_customer"]


}


CloudGraphEdge {


    source: "src/checkout.py",


    target: "src/payment.py",


    edge_type: EdgeType::Import,


    symbol: "charge_customer"


}


// ❌ NOT SHARED (code content)


def charge_customer(amount, card_token):


    stripe.api_key = "sk_live_ABC123"  # ← NEVER SENT TO CLOUD


    return stripe.Charge.create(...)   # ← NEVER SENT TO CLOUD



---


### Proactive Conflict Warning Workflow

Scenario: Agent B Wants to Modify File with Dependencies


Step 1: Agent B Prepares to Start Work


let file_path = "src/checkout.py";


// Query Cloud Graph DB for conflicts BEFORE claiming file


let conflicts = cloud_graph_db.query_conflicts(project_id, file_path).await?;


Step 2: Cloud Graph Database Analysis


pub async fn query_conflicts(


    &self,


    project_id: &str,


    file_path: &str


) -> Result<Vec<Conflict>> {


    let mut conflicts = Vec::new();


    // 1. Check direct dependencies


    let dependencies = self.get_dependencies(project_id, file_path)?;


    for dep in dependencies {


        if let Some(lock) = self.get_file_lock(project_id, &dep)? {


            conflicts.push(Conflict {


                level: ConflictLevel::DirectDependency,


                file: dep,


                locked_by: lock.agent_id,


                estimated_completion: lock.estimated_completion,


                message: format!("{} imports from {} (being modified by {})",


                    file_path, dep, lock.agent_id),


            });


        }


    }


    // 2. Check transitive dependencies (up to 3 levels deep)


    let transitive_deps = self.get_transitive_dependencies(project_id, file_path, 3)?;


    for dep in transitive_deps {


        if let Some(lock) = self.get_file_lock(project_id, &dep.file)? {


            conflicts.push(Conflict {


                level: ConflictLevel::TransitiveDependency,


                file: dep.file,


                locked_by: lock.agent_id,


                dependency_chain: dep.chain,


                message: format!("{} → {} → {} (being modified by {})",


                    file_path, dep.chain.join(" → "), dep.file, lock.agent_id),


            });


        }


    }


    // 3. Check reverse dependencies (what depends on this file?)


    let dependents = self.get_dependents(project_id, file_path)?;


    for dependent in dependents {


        if let Some(lock) = self.get_file_lock(project_id, &dependent)? {


            conflicts.push(Conflict {


                level: ConflictLevel::ReverseDependency,


                file: dependent,


                locked_by: lock.agent_id,


                message: format!("{} is imported by {} (being modified by {})",


                    file_path, dependent, lock.agent_id),


            });


        }


    }


    Ok(conflicts)


}


Step 3: Agent B Receives Warnings


// Cloud Graph DB returns:


vec![


    Conflict {


        level: DirectDependency,


        file: "src/payment.py",


        locked_by: "agent-1",


        estimated_completion: Some("2025-11-30T10:15:00Z"), // 10 minutes


        message: "checkout.py imports from payment.py (being modified by agent-1)"


    },


    Conflict {


        level: TransitiveDependency,


        file: "src/database.py",


        locked_by: "agent-3",


        dependency_chain: ["checkout.py", "user.py", "database.py"],


        message: "checkout.py → user.py → database.py (being modified by agent-3)"


    }


]


Step 4: Agent B Decision Matrix


match conflicts.len() {


    0 => {


        // No conflicts - proceed immediately


        agent.claim_file_and_start_work(file_path).await?;


    },


    1..=2 => {


        // Few conflicts - offer choices to user


        let choice = ui.prompt_user(&format!(


            "Conflicts detected:\n{}\n\nOptions:\n\


             1. Wait {} minutes for dependencies to complete\n\


             2. Coordinate with other agents\n\


             3. Work on different file\n\


             4. Proceed anyway (may require merge later)",


            conflicts.iter().map(|c| c.message.clone()).collect::<Vec<_>>().join("\n"),


            conflicts.iter().filter_map(|c| c.estimated_completion).map(|t| (t - Utc::now()).num_minutes()).max().unwrap_or(0)


        ))?;


        match choice {


            1 => agent.wait_for_dependencies(&conflicts).await?,


            2 => agent.send_coordination_message(&conflicts).await?,


            3 => agent.choose_alternative_task().await?,


            4 => agent.proceed_with_warning(file_path).await?,


        }


    },


    _ => {


        // Many conflicts - recommend waiting or re-planning


        ui.show_warning("High coupling detected. Recommend waiting or asking user to re-assign work.");


    }


}



---


### Cloud Graph Service Implementation

Technology Stack:


* Backend: Rust with Actix-Web or Axum
* Database: Redis (in-memory, <50ms latency) + PostgreSQL (persistence)
* Protocol: WebSocket (real-time) or gRPC (low latency)
* Deployment: Fly.io, Railway, or self-hosted Docker container


API Endpoints:


// 1. Claim file for modification


POST /api/v1/projects/{project_id}/files/claim


{


  "agent_id": "agent-1",


  "file_path": "src/payment.py",


  "estimated_duration_minutes": 15


}


Response: { "success": true } or { "error": "File already locked by agent-2" }


// 2. Release file after modification


POST /api/v1/projects/{project_id}/files/release


{


  "agent_id": "agent-1",


  "file_path": "src/payment.py"


}


Response: { "success": true }


// 3. Query dependencies


GET /api/v1/projects/{project_id}/dependencies?file=src/checkout.py


Response: {


  "dependencies": [


    { "file": "src/payment.py", "type": "import", "symbol": "PaymentService" },


    { "file": "src/user.py", "type": "import", "symbol": "User" }


  ]


}


// 4. Query conflicts BEFORE claiming


GET /api/v1/projects/{project_id}/conflicts?file=src/checkout.py


Response: {


  "conflicts": [


    {


      "level": "direct_dependency",


      "file": "src/payment.py",


      "locked_by": "agent-1",


      "estimated_completion": "2025-11-30T10:15:00Z",


      "message": "checkout.py imports from payment.py (being modified by agent-1)"


    }


  ]


}


// 5. Sync local graph structure to cloud (incremental)


POST /api/v1/projects/{project_id}/graph/sync


{


  "agent_id": "agent-1",


  "graph_delta": {


    "added_nodes": [...],


    "removed_nodes": [...],


    "added_edges": [...],


    "removed_edges": [...]


  }


}


Response: { "success": true, "server_timestamp": "2025-11-30T10:00:00Z" }


// 6. WebSocket for real-time updates


WS /api/v1/projects/{project_id}/ws


Messages:


  - FileLocked: { "file": "src/payment.py", "agent": "agent-1" }


  - FileReleased: { "file": "src/payment.py", "agent": "agent-1" }


  - GraphUpdated: { "files": ["src/payment.py"], "timestamp": "..." }


Redis Data Model:


# File locks (per project)


project:abc123:lock:src/payment.py = {


  "agent_id": "agent-1",


  "started_at": "2025-11-30T10:00:00Z",


  "estimated_completion": "2025-11-30T10:15:00Z"


}


EXPIRE project:abc123:lock:src/payment.py 3600  # Auto-release after 1 hour


# Agent registry


project:abc123:agent:agent-1 = {


  "type": "Backend",


  "started_at": "2025-11-30T10:00:00Z",


  "active": true


}


# Dependency graph (cached from PostgreSQL)


project:abc123:graph:src/payment.py:deps = ["src/database.py", "src/config.py"]


project:abc123:graph:src/checkout.py:deps = ["src/payment.py", "src/user.py"]


PostgreSQL Schema:


CREATE TABLE cloud_graph_nodes (


    id BIGSERIAL PRIMARY KEY,


    project_id UUID NOT NULL,


    file_path TEXT NOT NULL,


    node_type VARCHAR(50) NOT NULL,  -- File, Function, Class


    name TEXT NOT NULL,


    exports TEXT[],  -- Array of exported symbols


    created_at TIMESTAMPTZ DEFAULT NOW(),


    updated_at TIMESTAMPTZ DEFAULT NOW(),


    UNIQUE(project_id, file_path, name)


);


CREATE TABLE cloud_graph_edges (


    id BIGSERIAL PRIMARY KEY,


    project_id UUID NOT NULL,


    source_file TEXT NOT NULL,


    target_file TEXT NOT NULL,


    edge_type VARCHAR(50) NOT NULL,  -- Import, Call, Inherit


    symbol TEXT,


    created_at TIMESTAMPTZ DEFAULT NOW(),


    UNIQUE(project_id, source_file, target_file, edge_type, symbol)


);


CREATE INDEX idx_nodes_project ON cloud_graph_nodes(project_id);


CREATE INDEX idx_nodes_file ON cloud_graph_nodes(project_id, file_path);


CREATE INDEX idx_edges_project ON cloud_graph_edges(project_id);


CREATE INDEX idx_edges_source ON cloud_graph_edges(project_id, source_file);


CREATE INDEX idx_edges_target ON cloud_graph_edges(project_id, target_file);



---


### Local → Cloud Sync Mechanism

Sync Frequency:


* Incremental: Every 30 seconds (or on file modification)
* Full: Every 5 minutes (validate consistency)


Sync Process:


pub struct CloudGNNSync {


    local_gnn: Arc<GNNEngine>,


    cloud_client: CloudGNNClient,


    project_id: String,


    last_sync: Arc<Mutex<DateTime<Utc>>>,


}


impl CloudGNNSync {


    // Incremental sync (only changes since last sync)


    pub async fn sync_incremental(&self) -> Result<()> {


        let last_sync_time = *self.last_sync.lock().await;


        // Get changes since last sync from local GNN


        let changes = self.local_gnn.get_changes_since(last_sync_time)?;


        if changes.is_empty() {


            return Ok(()); // No changes


        }


        // Send delta to Cloud Graph DB


        let delta = GraphDelta {


            added_nodes: changes.added_nodes,


            removed_nodes: changes.removed_nodes,


            added_edges: changes.added_edges,


            removed_edges: changes.removed_edges,


        };


        self.cloud_client.sync_graph(&self.project_id, delta).await?;


        // Update last sync timestamp


        *self.last_sync.lock().await = Utc::now();


        Ok(())


    }


    // Full sync (validate consistency, recover from errors)


    pub async fn sync_full(&self) -> Result<()> {


        let local_graph = self.local_gnn.export_graph()?;


        self.cloud_client.upload_full_graph(&self.project_id, local_graph).await?;


        *self.last_sync.lock().await = Utc::now();


        Ok(())


    }


    // Auto-sync loop (runs in background)


    pub async fn start_auto_sync(&self) {


        let mut interval_incremental = tokio::time::interval(Duration::from_secs(30));


        let mut interval_full = tokio::time::interval(Duration::from_secs(300));


        loop {


            tokio::select! {


                _ = interval_incremental.tick() => {


                    if let Err(e) = self.sync_incremental().await {


                        eprintln!("Incremental sync failed: {}", e);


                    }


                }


                _ = interval_full.tick() => {


                    if let Err(e) = self.sync_full().await {


                        eprintln!("Full sync failed: {}", e);


                    }


                }


            }


        }


    }


}


Privacy Enforcement:


// Extract ONLY structure, strip code content


pub fn sanitize_graph_for_cloud(local_graph: &CodeGraph) -> CloudGraph {


    CloudGraph {


        nodes: local_graph.nodes.iter().map(|node| {


            CloudGraphNode {


                file_path: node.file_path.clone(),


                node_type: node.node_type,


                name: node.name.clone(),


                exports: node.exports.clone(),


                // ❌ NO source_code, NO function_body, NO variable_values


            }


        }).collect(),


        edges: local_graph.edges.iter().map(|edge| {


            CloudGraphEdge {


                source: edge.source.clone(),


                target: edge.target.clone(),


                edge_type: edge.edge_type,


                symbol: edge.symbol.clone(),


                // ❌ NO code snippets, NO implementation details


            }


        }).collect(),


    }


}



---


### Deployment Options

Option 1: Yantra Cloud (Hosted Service)


* URL:wss://cloud.yantra.dev/api/v1/
* Pricing:
  * Free: Single user, 1 project
  * Pro ($29/month): 3 users, 5 projects
  * Team ($99/month): Unlimited users, unlimited projects
* Benefits: Zero setup, automatic updates, 99.9% uptime SLA
* Privacy: Graph structure only (not code), encrypted in transit and at rest


Option 2: Self-Hosted (Enterprise)


* Docker Container:docker run -p 8080:8080 yantra/cloud-gnn:latest
* Requirements: 2GB RAM, 1 CPU core, Redis + PostgreSQL
* Benefits: Full control, on-premise deployment, air-gapped networks
* Setup: 15 minutes with Docker Compose


Docker Compose Example:


version: '3.8'


services:


  cloud-gnn:


    image: yantra/cloud-gnn:latest


    ports:


      - '8080:8080'


    environment:


      - DATABASE_URL=postgresql://yantra:password@postgres:5432/yantra_cloud


      - REDIS_URL=redis://redis:6379


    depends_on:


      - postgres


      - redis


  postgres:


    image: postgres:15


    environment:


      POSTGRES_DB: yantra_cloud


      POSTGRES_USER: yantra


      POSTGRES_PASSWORD: password


    volumes:


      - postgres_data:/var/lib/postgresql/data


  redis:


    image: redis:7-alpine


    volumes:


      - redis_data:/data


volumes:


  postgres_data:


  redis_data:



---


### Implementation Tasks (Phase 2B - 3 weeks)

Week 13: Cloud Graph Service (Backend)


* Actix-Web/Axum HTTP server with WebSocket support
* Redis integration for file locks and agent registry
* PostgreSQL schema and integration for graph storage
* REST API endpoints: claim_file, release_file, query_dependencies, query_conflicts
* WebSocket real-time notifications


Week 14: Local → Cloud Sync


* CloudGraphClient in Rust (WebSocket/gRPC client)
* Incremental sync (every 30s) and full sync (every 5 min)
* Privacy layer: Strip code content, send only graph structure
* Error handling and retry logic
* Background sync task (tokio::spawn)


Week 15: Conflict Detection & UI


* Implement query_conflicts() with 4 levels (same-file, direct, transitive, semantic)
* Agent B workflow: Query conflicts BEFORE claiming file
* UI warnings: Show conflicts and recommend actions
* Coordination messages: Agent-to-Agent via Tier 2
* Performance testing: <50ms conflict queries, 100+ agents


Success Metrics:


* ✅ <50ms latency for conflict queries
* ✅ Zero code content leaked (only graph structure)
* ✅ 4 levels of conflict detection working
* ✅ 100+ agents supported simultaneously
* ✅ 99.9% uptime for hosted service
* ✅ Proactive conflict prevention (not reactive resolution)
* ✅ Team collaboration enabled (multiple users, same project)



---


### Success Metrics: Team of Agents + Cloud Graph Database

Performance:


* ✅ 3-10x faster feature completion with team of agents
* ✅ <50ms conflict queries (Cloud Graph DB)
* ✅ <5ms file lock operations (Tier 2 sled)
* ✅ <30s Master assignment overhead


Scalability:


* ✅ 5-10 agents per feature (optimal)
* ✅ 100-200 agents system-wide (practical limit)
* ✅ Unlimited users per project (Cloud Graph DB enables team collaboration)


Quality:


* ✅ Zero Git merge conflicts (all prevented proactively)
* ✅ 4 levels of conflict detection (same-file, direct, transitive, semantic)
* ✅ Privacy-preserving (graph structure only, not code)


User Experience:


* ✅ Agents work independently (no Master bottleneck)
* ✅ Clear warnings BEFORE conflicts occur
* ✅ Human-readable coordination history (Git branch)
* ✅ Easy debugging (inspect coordination branch with git log)



---




## Phase 2C: Clean Code Mode (Months 3-4, Post-MVP Enhancement)

### Overview

Clean Code Mode is an automated code maintenance system that continuously monitors, analyzes, and refactors codebases to maintain optimal code health. It leverages the existing GNN dependency tracking to detect dead code, perform safe refactorings, validate changes, and harden components after implementation.


Core Philosophy:


* Zero Trust: Always validate with GNN + tests before applying changes
* Confidence-Based: Only auto-apply changes with high confidence (>80%)
* Non-Breaking: Never break existing functionality
* Continuous: Runs as background process with configurable intervals


Key Differentiators:


* Uses GNN for intelligent dead code detection (not just static analysis)
* Real-time refactoring with dependency validation
* Automated hardening after component implementation
* Test-validated changes only

### Capabilities

#### 1. Dead Code Detection & Removal

What It Detects:


* Unused Functions: Functions with zero incoming calls (not entry points)
* Unused Classes: Classes with zero instantiations
* Unused Imports: Import statements never referenced
* Unused Variables: Variables assigned but never read
* Dead Branches: Unreachable code paths
* Commented Code: Large blocks of commented-out code


Entry Points (Never Remove):


* main() functions
* API route handlers, CLI command handlers
* Test functions
* Event handlers, lifecycle hooks
* Exported public APIs


Confidence Calculation:


* Base confidence = 1.0 (if zero calls)
* Modifiers: Recent code (×0.5), Public API (×0.3), Exported (×0.2), etc.
* Auto-Remove Threshold: 0.8 (80% confidence)

#### 2. Real-Time Refactoring

Supported Refactorings:


1. Remove Unused Imports (Auto-apply: Yes, Confidence: 1.0)
1. Extract Duplicate Code (GNN embeddings, similarity >85%)
1. Simplify Complex Functions (Cyclomatic complexity > 10)
1. Rename for Clarity (LLM suggestions)
1. Consolidate Error Handling
1. Optimize Dependencies


Dependency Graph-Powered Duplicate Detection:


* Semantic similarity using code embeddings (384-dim from fastembed all-MiniLM-L6-v2)
* Cosine similarity >0.85 = duplicate
* Detects duplicates across languages (same logic, different syntax)
* Note: This uses the dependency graph's semantic embeddings, NOT Yantra Codex (which is 978-dim input for code generation)

#### 3. Component Hardening

Automated Hardening After Implementation:


Security Hardening:


* OWASP Top 10 vulnerabilities
* Language-specific vulnerabilities (eval, SQL injection, XSS)
* Secret detection (API keys, passwords)
* Auto-Fix: 70%+ success rate for critical issues


Performance Hardening:


* Execution time analysis (avg, p95, p99)
* Memory profiling
* N+1 query detection
* API latency tracking
* Bottleneck identification


Code Quality Hardening:


* Cyclomatic complexity analysis
* Code smell detection
* Documentation coverage
* Maintainability index (0-100)


Dependency Hardening:


* Known vulnerability check
* Outdated dependency detection
* Security score calculation

#### 4. Configuration System

.yantra/clean-code.toml:


[enabled]


mode = "continuous"  # continuous, daily, pre-commit, manual


[dead-code]


enabled = true


auto-remove = false


confidence-threshold = 0.8


[refactoring]


enabled = true


auto-apply = false


max-complexity = 10


duplicate-threshold = 0.85


[hardening]


enabled = true


run-after = ["component-complete", "pre-commit"]


auto-fix-security = true


[intervals]


continuous-check = "5min"


daily-cleanup = "02:00"

### Performance Targets

| Operation | Target | Rationale |
| - | - | - |
| ---------------------------------------------------------- |
| Dead code analysis (10K LOC) | < 2s | Real-time feedback |
| - | - | - |
| Duplicate detection (10K LOC)                            | < 5s | GNN embedding comparison |
| Refactoring application                                  | < 3s | Including validation |
| Component hardening                                      | < 10s | Comprehensive scan |
| Security scan                                            | < 5s | Semgrep integration |

### Success Metrics

Key Performance Indicators:


* Dead Code Reduction: < 2% dead code in healthy projects
* Refactoring Acceptance Rate: > 60% for high-confidence suggestions
* False Positive Rate: < 5%
* Security Issue Detection: 100% of OWASP Top 10
* Auto-Fix Success Rate: > 70% for critical issues
* Code Quality Improvement: +10 maintainability points after 3 months
* Developer Time Saved: 20% reduction in code review time

### Implementation Plan (5 Weeks)

Week 1: Dead Code Detection


* Implement analyzer, entry point detection, confidence scoring
* Goal: Identify dead code accurately


Week 2: Safe Removal


* Implement removal logic, GNN validation, test validation, rollback
* Goal: Remove dead code safely


Week 3: Refactoring


* Duplicate detection, complexity analysis, refactoring suggestions
* Goal: Suggest smart refactorings


Week 4: Hardening


* Security scanner integration, performance profiler, auto-fix engine
* Goal: Automated hardening


Week 5: Continuous Mode


* Background scheduler, interval-based runs, event triggers
* Goal: Automated maintenance

### Integration Points

GNN Integration:


* Leverage get_dependents(), get_incoming_edges(), feature extraction
* GNN embeddings for semantic duplicate detection


LLM Integration:


* Generate refactored code
* Suggest better names, extraction functions, documentation


Testing Integration:


* Run affected tests only, full regression, coverage tracking


Git Integration:


* Auto-commit cleaned code, create branches, descriptive commit messages



---


## Phase 3: Enterprise Automation (Months 5-8)

### Objectives

Transform into enterprise workflow automation platform:


* Cross-system dependency tracking (internal + external APIs)
* Browser automation for legacy systems
* Self-healing workflows
* Multi-language support (Python + JavaScript)
* Enterprise features: Multitenancy, user accounts, team collaboration

### New Capabilities

1. Cross-System Intelligence

* Automatic discovery of external API calls
* Schema tracking for Stripe, Salesforce, etc.
* Breaking change detection (API version updates)
* End-to-end data flow validation
* Impact analysis (what breaks if X changes?)

2. Browser Automation

* Full Playwright integration
* DOM interaction (click, fill, extract data)
* Authentication handling
* Visual regression detection

3. Self-Healing Systems

* Continuous API monitoring (every 24h)
* Schema drift detection
* Automatic migration code generation
* Canary testing in sandbox
* Auto-deploy if tests pass

4. Multi-Language Support

* JavaScript/TypeScript parser
* Cross-language dependencies (Python API → React frontend)
* Node.js + React code generation
* Context mixing across languages

5. Enterprise Features (Post-MVP)


Multitenancy:


* Tenant isolation (database, GNN, patterns)
* Per-tenant configuration
* Shared failure patterns (cross-tenant, privacy-preserved)
* Resource quotas and limits


User Accounts & Authentication:


* User registration and login (OAuth, SSO)
* Role-based access control (RBAC)
* Team workspaces
* Project sharing and permissions
* Audit logs


Team Collaboration:


* Shared projects and codebases
* Activity feeds (who generated what)
* Code review workflows
* Comment threads on generated code
* Team pattern libraries (private)


Billing & Subscription:


* Usage tracking (LLM calls, tokens)
* Subscription tiers (Free, Pro, Team, Enterprise)
* Payment integration (Stripe)
* Usage analytics and reporting

### Implementation (16 Weeks)

Weeks 17-20: External API discovery and tracking Weeks 21-24: Browser automation (Playwright) Weeks 25-28: Self-healing engine Weeks 29-32: Multi-language support Weeks 33-36: Enterprise features (multitenancy, user accounts, team collaboration)



---


## Phase 4: Platform Maturity (Months 9-12)

### Objectives

Mature platform with ecosystem and enterprise-grade reliability:


* 99.9% uptime
* Support 100k+ LOC projects
* Plugin ecosystem
* Enterprise deployment options

### New Capabilities

1. Performance Optimization

* GNN queries <100ms for 100k LOC projects
* Distributed GNN (sharding)
* Smart caching (LLM responses, test results)

2. Advanced Refactoring

* Architectural refactoring (monolith → microservices)
* Performance optimization
* Tech debt reduction
* Code modernization

3. Ecosystem

* Plugin system (extend Yantra)
* Marketplace (plugins, templates, workflows)
* CLI tool (for CI/CD)
* REST API
* SDKs (Python, JavaScript, Go)

4. Enterprise

* On-premise deployment (air-gapped)
* Custom model training
* White-label options
* 24/7 SLA support

### Implementation (16 Weeks)

Weeks 33-36: Performance & scale Weeks 37-40: Advanced refactoring Weeks 41-44: Ecosystem & marketplace Weeks 45-48: Enterprise platform


---



### User Experience


REFER TO [UX.MD](http://ux.md) FOR DETAILED UX REQUIREMENTS





---


## Getting Started (For Developers)

### Prerequisites

* Rust 1.74+ (rustup install stable)
* Node.js 18+ (nvm install 18)
* Python 3.11+ (pyenv install 3.11)
* Git
* macOS, Windows, or Linux

### Setup Development Environment

# Clone repository

git clone [https://github.com/cogumi/yantra.git](https://github.com/cogumi/yantra.git)


cd yantra

# Install Rust dependencies

cargo build

# Install frontend dependencies

cd src-ui


npm install

# Run in development mode

npm run tauri dev

### Project Structure

yantra/


├── src/ # Rust backend


│ ├── main.rs # Tauri entry point


│ ├── gnn/ # Graph Neural Network


│ ├── llm/ # LLM orchestration


│ ├── testing/ # Test generation & execution


│ ├── security/ # Security scanning


│ └── git/ # Git integration


├── src-ui/ # Frontend (SolidJS)


│ ├── components/ # UI components


│ ├── stores/ # State management


│ └── App.tsx # Main app


├── skills/ # Skill templates (future)


└── docs/ # Documentation

### Development Workflow

1. Create feature branch: git checkout -b feature/your-feature
1. Make changes
1. Run tests: cargo test && cd src-ui && npm test
1. Run linters: cargo clippy && npm run lint
1. Commit: git commit -m "feat: your feature"
1. Push and create PR

### Testing

# Run all Rust tests

cargo test

# Run frontend tests

cd src-ui && npm test

# Run integration tests

cargo test --test integration

# Run with coverage

cargo tarpaulin --out Html


**
````
