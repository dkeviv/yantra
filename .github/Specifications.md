**# Yantra: Complete Technical Specification

Version: 1.0
Date: November 2024
Document Purpose: Complete technical blueprint for building Yantra from ground zero to enterprise platform

---

## Executive Summary

### The Vision

Yantra is a **fully autonomous agentic developer** - an AI-powered platform that doesn't just generate code, but executes the complete software development lifecycle: from understanding requirements to deploying and monitoring production systems.

**Traditional AI Code Assistants:** Help developers write code faster  
**Yantra:** Replaces the entire development workflow with autonomous agents

Unlike traditional IDEs that assist developers or AI tools that suggest code, Yantra makes artificial intelligence the **primary developer**, with humans providing intent, oversight, and approvals only for critical decisions.

### What "Fully Autonomous Agentic" Means

**Not autonomous:** LLM generates code → Developer manually tests → Developer fixes issues → Developer commits  
**Partially autonomous:** LLM generates code → System validates → Developer fixes issues  
**Fully autonomous (Yantra):** LLM generates code → System validates → System fixes issues → System tests → System packages → System deploys → System monitors → Repeat until perfect

**Yantra handles the complete pipeline:**
1. 🎯 **Understand:** Parse natural language requirements
2. 🔨 **Build:** Generate production-quality code
3. ✅ **Validate:** Run dependency checks, tests, security scans
4. 🔄 **Fix:** Auto-retry with intelligent error analysis
5. ▶️ **Execute:** Run the code with proper environment setup
6. 📦 **Package:** Build distributable artifacts (wheels, Docker images, npm packages)
7. 🚀 **Deploy:** Push to production (AWS, GCP, Kubernetes, Heroku)
8. 📊 **Monitor:** Track performance and errors in production
9. 🔧 **Heal:** Auto-fix production issues without human intervention

**Human role:** Provide intent ("Add payment processing"), review critical changes, approve deployments

### The Problem We Solve

For Developers:

* 40-60% of development time spent debugging
* Code breaks production despite passing tests
* Integration failures when APIs change
* Repetitive coding tasks (CRUD, auth, APIs)
* Context switching between IDE, terminal, browser, deployment tools
* Manual deployment and rollback procedures
* Production firefighting and hotfix cycles

For Engineering Teams:

* Unpredictable delivery timelines
* Inconsistent code quality
* High maintenance costs
* Technical debt accumulation
* Slow time-to-market (weeks for simple features)
* DevOps bottlenecks

For Enterprises:

* Manual workflow automation (expensive, error-prone)
* Siloed systems (Slack, Salesforce, internal tools don't talk)
* Workflow tools (Zapier) can't access internal code or execute complex logic
* System breaks cascade across services
* Browser automation requires specialized developers
* No self-healing - every outage requires manual intervention

### The Solution

Phase 1 (Months 1-2): Code That Never Breaks + Autonomous Execution

* AI generates code with full dependency awareness (✅ COMPLETE)
* Automated unit + integration testing (🟡 Generation complete, execution in progress)
* Security vulnerability scanning (⚪ Post-MVP)
* Browser runtime validation (⚪ Post-MVP)
* **Autonomous code execution with environment setup** (🆕 Week 9-10)
* **Integrated terminal for command execution** (🆕 Week 9-10)
* **Real-time output streaming to UI** (🆕 Week 9-10)
* Git integration for seamless commits (⚪ Post-MVP)

Phase 2 (Months 3-4): Package, Deploy & Workflow Automation

* **Package building (Python wheels, Docker, npm)** (🆕)
* **Automated deployment (AWS, GCP, Kubernetes, Heroku)** (🆕)
* **Health checks and auto-rollback** (🆕)
* Generate workflows from natural language
* Scheduled jobs and event triggers
* Multi-step orchestration with error handling and retries
* **CI/CD pipeline generation** (🆕)

Phase 3 (Months 5-8): Enterprise Automation & Self-Healing

* Cross-system dependency tracking
* External API monitoring and auto-healing
* **Production monitoring with auto-remediation** (🆕)
* **Browser automation for enterprise workflows** (🆕)
* **Legacy system integration via browser control** (🆕)
* Multi-language support (Python + JavaScript + TypeScript)
* **Infrastructure as Code generation** (🆕)

Phase 4 (Months 9-12): Platform Maturity & Ecosystem

* Plugin ecosystem and marketplace
* Advanced refactoring and performance optimization
* Enterprise deployment (on-premise, cloud, air-gapped)
* SLA guarantees (99.9% uptime)
* **Distributed agent coordination** (🆕)
* **Multi-tenant enterprise features** (🆕)

### Market Opportunity

Primary Market: Developer Tools ($50B+)

* IDEs, testing tools, CI/CD platforms
* Target: Mid-market to enterprise (10-1000+ developers)

Secondary Market: Workflow Automation ($10B+)

* Replace/augment Zapier, Make, Workato
* Target: Operations teams, business analysts

Total Addressable Market: $60B+

### Competitive Advantage

| Capability                  | Yantra | Copilot | Cursor | Zapier | Replit Agent |
| --------------------------- | ------ | ------- | ------ | ------ | ------------ |
| Dependency-aware generation | ✅     | ❌      | ❌     | N/A    | ❌           |
| Guaranteed no breaks        | ✅     | ❌      | ❌     | ❌     | ❌           |
| Truly unlimited context     | ✅     | ❌      | ❌     | N/A    | ❌           |
| Token-aware context         | ✅     | ⚠️    | ⚠️   | N/A    | ❌           |
| Automated testing           | ✅     | ❌      | ❌     | ❌     | ⚠️          |
| Agentic validation pipeline | ✅     | ❌      | ❌     | ❌     | ❌           |
| **Autonomous code execution** | ✅   | ❌      | ❌     | ⚪     | ✅           |
| **Package building**        | ✅     | ❌      | ❌     | ❌     | ⚠️          |
| **Automated deployment**    | ✅     | ❌      | ❌     | ⚪     | ✅           |
| **Production monitoring**   | ✅     | ❌      | ❌     | ❌     | ❌           |
| Self-healing systems        | ✅     | ❌      | ❌     | ❌     | ❌           |
| Network effect (failures)   | ✅     | ❌      | ❌     | ❌     | ❌           |
| Works with any LLM          | ✅     | ❌      | ⚠️   | N/A    | ❌           |
| Internal system access      | ✅     | ⚠️    | ⚠️   | ❌     | ⚠️          |
| Custom workflow code        | ✅     | ❌      | ❌     | ❌     | ⚠️          |
| **Browser automation**      | ✅     | ❌      | ❌     | ❌     | ❌           |
| **Integrated terminal**     | ✅     | ❌      | ❌     | N/A    | ✅           |
| **Desktop app (native)**    | ✅     | N/A     | ✅     | N/A    | ❌ (web)     |

**Key Differentiators:**

1. **Complete Development Lifecycle**: Only platform that handles generate → run → test → package → deploy → monitor autonomously
2. **Truly Unlimited Context**: Not limited by LLM context windows through intelligent compression, chunking, and hierarchical assembly
3. **Agentic Architecture**: Fully autonomous validation pipeline with confidence scoring and auto-retry loops
4. **Enterprise-Grade Browser Automation**: Automate legacy systems, extract data, run workflows across web applications
5. **Network Effect from Failures**: Shared failure patterns (privacy-preserving) create collective intelligence that improves with every user
6. **LLM Agnostic**: Works with any LLM (Claude, GPT-4, Qwen Coder) through context enhancement, not LLM-specific features
7. **Self-Healing Production Systems**: Monitors deployed applications, detects issues, generates fixes, deploys patches automatically
8. **Desktop-First**: Native performance, local file access, no browser limitations

**vs Replit Agent:**
- Yantra: Enterprise-focused, dependency-aware, self-healing, browser automation, desktop app
- Replit: Developer sandbox, limited context, no self-healing, web-only, no enterprise features

**vs Copilot/Cursor:**
- They stop at code generation
- Yantra continues through entire deployment pipeline
- They require manual testing, packaging, deployment
- Yantra automates everything

---

## Core Architecture

### System Overview

┌──────────────────────────────────────────────────────┐

│                  AI-CODE PLATFORM                     │

├──────────────────────────────────────────────────────┤

│                                                       │

│  USER INTERFACE (AI-First)                           │

│  ┌─────────────────────────────────────────────┐    │

│  │ Chat/Task Interface (Primary - 60% screen)  │    │

│  │ Code Viewer (Secondary - 25% screen)        │    │

│  │ Browser Preview (Live - 15% screen)         │    │

│  └─────────────────────────────────────────────┘    │

│                       │                               │

│  ┌────────────────────┼─────────────────────────┐   │

│  │ ORCHESTRATION LAYER│                         │   │

│  │  Multi-LLM Manager │                         │   │

│  │  ├─ Claude Sonnet (Primary)                 │   │

│  │  ├─ GPT-4 (Secondary/Validation)            │   │

│  │  └─ Routing & Failover Logic                │   │

│  └─────────────────────────────────────────────┘   │

│                       │                               │

│  ┌────────────────────┼─────────────────────────┐   │

│  │ INTELLIGENCE LAYER │                         │   │

│  │  Graph Neural Network (GNN)                  │   │

│  │  ├─ Code Dependencies                        │   │

│  │  ├─ External API Tracking                    │   │

│  │  ├─ Data Flow Analysis                       │   │

│  │  └─ Known Issues Database (LLM Failures)     │   │

│  │                                               │   │

│  │  Vector Database (RAG)                       │   │

│  │  ├─ Code Templates                           │   │

│  │  ├─ Best Practices                           │   │

│  │  ├─ Project Patterns                         │   │

│  │  └─ Failure Pattern Library (Network Effect) │   │

│  │                                               │   │

│  │  Unlimited Context Engine                    │   │

│  │  ├─ Token Counting & Management              │   │

│  │  ├─ Context Compression & Chunking           │   │

│  │  ├─ Hierarchical Context Assembly            │   │

│  │  └─ Adaptive Context Strategies              │   │

│  └─────────────────────────────────────────────┘   │

│                       │                               │

│  ┌────────────────────┼─────────────────────────┐   │

│  │ VALIDATION LAYER   │                         │   │

│  │  ├─ Testing Engine (pytest/jest)             │   │

│  │  ├─ Security Scanner (Semgrep + custom)      │   │

│  │  ├─ Browser Integration (CDP)                │   │

│  │  ├─ Dependency Validator (GNN)               │   │

│  │  └─ Agentic Validation Pipeline              │   │

│  │                                               │   │

│  │  Agent State Machine                         │   │

│  │  ├─ Code Generation → Validation Loop        │   │

│  │  ├─ Confidence Scoring & Auto-Retry          │   │

│  │  ├─ Failure Analysis & Pattern Extraction    │   │

│  │  └─ Self-Healing with Known Issues DB        │   │

│  └─────────────────────────────────────────────┘   │

│                       │                               │

│  ┌────────────────────┼─────────────────────────┐   │

│  │ INTEGRATION LAYER  │                         │   │

│  │  ├─ Git (MCP Protocol)                       │   │

│  │  ├─ File System                              │   │

│  │  └─ External APIs (Phase 2+)                 │   │

│  └─────────────────────────────────────────────┘   │

│                                                       │

└──────────────────────────────────────────────────────┘

### Technology Stack

Desktop Framework:

* Tauri 1.5+ (Rust backend + web frontend)
* Rationale: 600KB bundle vs 150MB Electron, native performance

Frontend:

* SolidJS 1.8+ (reactive UI framework)
* Monaco Editor 0.44+ (code viewing)
* TailwindCSS 3.3+ (styling)
* WebSockets (real-time updates)

Backend (Rust):

* Tokio 1.35+ (async runtime)
* SQLite 3.44+ (GNN persistence, known issues DB)
* Reqwest 0.11+ (HTTP client)
* Serde 1.0+ (JSON serialization)
* **tiktoken-rs 0.5+ (token counting)**

GNN Implementation:

* Language: Rust (performance critical)
* Graph Library: petgraph 0.6+
* Parser: tree-sitter (Python, JS, etc.)
* Known Issues: SQLite with pattern matching

LLM Integration:

* Primary: Anthropic Claude API (claude-sonnet-4)
* Secondary: OpenAI API (gpt-4-turbo)
* Tertiary: Qwen Coder (via OpenAI-compatible API)
* Rate limiting, retry logic, circuit breaker
* **Confidence scoring from response metadata**

Context Management:

* Token counting: tiktoken-rs (exact counts)
* Compression: Syntax-aware, de-duplication
* Hierarchical assembly: 4 levels of detail
* Caching: SQLite with 24h TTL

Vector Database:

* ChromaDB (embedded mode)
* Embeddings: all-MiniLM-L6-v2 (local, lightweight)
* Storage: Code patterns, failure patterns

Testing:

* Python: pytest 7.4+, pytest-cov
* JavaScript: Jest (Phase 2+)
* Runner: Subprocess execution from Rust

Security:

* SAST: Semgrep with OWASP rules
* Dependencies: Safety (Python), npm audit
* Secrets: TruffleHog patterns

Browser:

* Protocol: Chrome DevTools Protocol (CDP)
* Library: chromiumoxide (Rust CDP client)
* Automation: Playwright (complex interactions)

Git:

* Protocol: Model Context Protocol (MCP)
* Library: git2-rs (libgit2 Rust bindings)

---

## Core Innovation: Truly Unlimited Context

### The Problem with Current AI Coding Tools

All existing AI coding tools (GitHub Copilot, Cursor, Windsurf, etc.) are fundamentally limited by LLM context windows:

* Claude Sonnet 4: 200K tokens (~150K LOC worth of context)
* GPT-4 Turbo: 128K tokens (~100K LOC worth of context)
* Qwen Coder: 32K-128K tokens depending on version

**Result:** These tools fail on large codebases, miss critical dependencies, and generate code that breaks existing functionality.

### Yantra's Solution: Context Intelligence, Not Context Limits

Yantra achieves truly unlimited context through a multi-layered approach that works with ANY LLM, including smaller models like Qwen Coder:

#### 1. Token-Aware Context Management

**Implementation:**
* Real token counting using tiktoken-rs (exact, not estimated)
* Dynamic token budget allocation based on LLM provider
* Reserve 20% of context window for response generation
* Graceful degradation when approaching limits

**Token Budgets:**
* Claude Sonnet 4: 160,000 context tokens (40K for response)
* GPT-4 Turbo: 100,000 context tokens (28K for response)
* Qwen Coder 32K: 25,000 context tokens (7K for response)
* Adaptive allocation per LLM capability

**Code:**
```rust
// src/llm/context.rs
const CLAUDE_MAX_CONTEXT_TOKENS: usize = 160_000;
const GPT4_MAX_CONTEXT_TOKENS: usize = 100_000;
const QWEN_32K_MAX_CONTEXT_TOKENS: usize = 25_000;

pub fn assemble_context_with_limit(
    gnn: &GNN,
    start_nodes: &[NodeId],
    max_tokens: usize
) -> Result<Vec<ContextItem>>
```

#### 2. Hierarchical Context Assembly

**Strategy:** Provide different levels of detail based on distance from target code:

**Level 1 - Immediate Context (Full Detail):**
* Complete source code of target files
* All direct dependencies (imports, function calls)
* Data structures and types referenced
* **Token allocation: 40% of budget**

**Level 2 - Related Context (Signatures Only):**
* Function signatures (no implementation)
* Class definitions (no methods)
* Type definitions and interfaces
* API contracts
* **Token allocation: 30% of budget**

**Level 3 - Distant Context (References Only):**
* Module names and imports
* High-level architecture
* Indirect dependencies (A → B → C)
* **Token allocation: 20% of budget**

**Level 4 - Metadata (Summaries):**
* Project structure overview
* Known patterns and conventions
* Relevant documentation snippets
* **Token allocation: 10% of budget**

**Implementation:**
```rust
pub struct HierarchicalContext {
    immediate: Vec<CodeItem>,      // Full code
    related: Vec<SignatureItem>,   // Signatures only
    distant: Vec<ReferenceItem>,   // References
    metadata: ProjectSummary,      // High-level
}
```

#### 3. Intelligent Context Compression

**Techniques:**

**A. Syntax-Aware Compression:**
* Remove comments (unless directly relevant)
* Strip docstrings (keep in metadata)
* Minimize whitespace
* Remove unused imports in context
* **Savings: 20-30% tokens**

**B. Semantic Chunking:**
* Split large files into logical chunks (classes, functions)
* Include only relevant chunks
* Track chunk relationships in GNN
* **Savings: 40-60% tokens for large files**

**C. De-duplication:**
* Identical code blocks referenced multiple times
* Common utility functions
* Shared type definitions
* **Savings: 10-15% tokens**

**Implementation:**
```rust
pub fn compress_context(
    items: Vec<ContextItem>,
    target_tokens: usize
) -> Vec<CompressedItem> {
    // 1. Remove non-essential whitespace
    // 2. Strip docstrings (keep in separate metadata)
    // 3. Deduplicate identical blocks
    // 4. Return compressed within token budget
}
```

#### 4. RAG-Enhanced Context Retrieval

**ChromaDB Integration:**

**Embeddings Storage:**
* All function signatures and docstrings
* Common code patterns
* Failure patterns with fixes (see next section)
* Best practices and conventions

**Semantic Search:**
* Query: User intent + target code context
* Retrieve: Top-K most relevant patterns (K=10-20)
* Add to context as examples
* **Cost: 2,000-5,000 tokens (high value)**

**Example:**
```
User: "Add authentication to the API endpoint"

RAG retrieves:
1. JWT authentication pattern (200 tokens)
2. Session management example (300 tokens)
3. Security best practices (150 tokens)
4. Similar endpoint with auth (400 tokens)

Total: 1,050 tokens for highly relevant context
```

#### 5. Adaptive Context Strategies

**Based on Task Type:**

| Task | Strategy | Token Allocation |
|------|----------|------------------|
| New feature | Wide context (many dependencies) | 70% L1+L2 |
| Bug fix | Deep context (full implementation) | 80% L1 |
| Refactoring | Architectural context (all usages) | 50% L1, 40% L2 |
| Testing | Target code + similar tests | 60% L1, 30% RAG |

**Dynamic Adjustment:**
* Monitor LLM confidence scores
* If low confidence → expand context
* If token limit hit → compress L2/L3
* Iterative refinement

#### 6. Context Caching & Reuse

**SQLite Cache:**
* Cache compressed context by hash (file content + dependencies)
* 24-hour TTL
* Invalidate on file changes
* **Performance gain: <50ms retrieval vs 100-500ms assembly**

**Shared Context Across Requests:**
* Same file referenced multiple times
* Compute once, reuse
* Track with reference counting

### Why This Enables ANY LLM (Including Qwen Coder)

**The Key Insight:** Most LLM failures are due to missing context, not LLM capability.

**With Yantra's Context Intelligence:**

1. **Qwen Coder 32K** (smaller model):
   * Gets 25,000 tokens of perfectly relevant context
   * Hierarchical assembly prioritizes what matters
   * RAG provides proven patterns
   * Known failures database prevents common mistakes
   * **Result: Performs as well as GPT-4 with 100K tokens**

2. **Even GPT-3.5** (16K context):
   * Gets 12,000 tokens of hyper-relevant context
   * Every token is carefully selected
   * Compression eliminates noise
   * **Result: Better than GPT-4 with random 100K context**

**Validation:**
* Benchmark: Same task with GPT-4 (naive 100K context) vs Qwen Coder (optimized 25K context)
* Metric: Code quality, test pass rate, breaking changes
* Target: Qwen performance within 5% of GPT-4

### Performance Targets

| Operation | MVP Target | Scale Target |
|-----------|------------|--------------|
| Token counting | <10ms | <5ms |
| Context assembly | <100ms | <50ms |
| Compression | <50ms | <20ms |
| RAG retrieval | <200ms | <100ms |
| Total context pipeline | <500ms | <200ms |

### Implementation Phases

**MVP (Month 1-2):**
* ✅ Token-aware context assembly (no arbitrary limits)
* ✅ BFS traversal with priority (implemented)
* ⚠️ Token counting with tiktoken-rs (add)
* ⚠️ Hierarchical context (L1 + L2) (add)
* ⚠️ Basic compression (whitespace, comments) (add)

**Post-MVP (Month 3-4):**
* Advanced compression (semantic chunking)
* RAG with ChromaDB
* Full hierarchical context (L1-L4)
* Adaptive strategies per task type
* Context caching

**Enterprise (Month 5-8):**
* Multi-language context mixing
* Cross-repository context
* Distributed context cache
* Real-time context updates

---

## Core Innovation: Fully Autonomous Agentic Architecture

### What "Fully Autonomous Agentic" Means

**Not agentic:** LLM generates code → User manually tests → User manually fixes issues → User manually commits

**Partially agentic:** LLM generates code → System validates → User fixes issues → User commits

**Fully autonomous (Yantra):** LLM generates code → System validates → System fixes → System tests → **System runs** → **System packages** → **System deploys** → **System monitors** → Repeat until perfect

**Yantra is end-to-end autonomous:** Human provides intent, AI handles entire development and deployment lifecycle.

### Complete Autonomous Agent State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│            FULLY AUTONOMOUS AGENTIC PIPELINE                     │
│         (Generate → Run → Test → Package → Deploy → Monitor)    │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   User Intent    │ (Natural language task)
    │  "Add payments"  │
    └─────────┬────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE 1: CODE GENERATION (✅ MVP COMPLETE)  │
    └─────────────────────────────────────────────┘
              │
              ├→ ContextAssembly (GNN + Hierarchical L1+L2)
              ├→ CodeGeneration (Claude/GPT-4 with context)
              ├→ DependencyValidation (GNN check, no breaks)
              └→ ConfidenceScoring (5 factors: LLM, tests, known, complexity, impact)
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE 2: EXECUTION (🆕 Week 9-10)          │
    └─────────────────────────────────────────────┘
              │
              ├→ EnvironmentSetup (venv, docker, env vars)
              ├→ DependencyInstallation (pip/npm install)
              ├→ ScriptExecution (run generated code)
              ├→ RuntimeValidation (capture output, check errors)
              └→ PerformanceProfiling (measure execution time, memory)
              │
              ├─── RUNTIME ERROR ───→ Analyze → Retry with fix
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE 3: TESTING (🟡 Partial)              │
    └─────────────────────────────────────────────┘
              │
              ├→ UnitTesting (pytest subprocess, JUnit XML)
              ├→ IntegrationTesting (E2E flows)
              ├→ SecurityScanning (Semgrep, Safety, TruffleHog)
              └→ BrowserValidation (CDP, headless Chrome)
              │
              ├─── TEST FAIL ───→ Analyze → Fix → Rerun
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE 4: PACKAGING (🆕 Month 3)            │
    └─────────────────────────────────────────────┘
              │
              ├→ PackageConfiguration (setup.py, Dockerfile, package.json)
              ├→ BuildExecution (python -m build, docker build, npm run build)
              ├→ AssetOptimization (minify, compress, bundle)
              └→ ArtifactGeneration (wheels, Docker images, npm packages)
              │
              ├─── BUILD FAIL ───→ Analyze → Fix → Rebuild
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE 5: DEPLOYMENT (🆕 Month 3-4)         │
    └─────────────────────────────────────────────┘
              │
              ├→ DeploymentPrep (config for AWS/GCP/K8s)
              ├→ InfrastructureProvisioning (terraform, CloudFormation)
              ├→ DatabaseMigration (run migrations safely)
              ├→ ServiceDeployment (deploy to staging/prod)
              ├→ HealthCheck (verify deployment success)
              └→ RollbackIfNeeded (auto-rollback on failure)
              │
              ├─── DEPLOY FAIL ───→ Rollback → Analyze → Retry
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ PHASE 6: MONITORING & HEALING (🆕 Month 5) │
    └─────────────────────────────────────────────┘
              │
              ├→ MonitoringSetup (observability tools)
              ├→ ErrorTracking (runtime errors in production)
              ├→ PerformanceMonitoring (latency, throughput)
              ├→ SelfHealing (detect issue → generate fix → deploy patch)
              └→ AlertEscalation (notify humans only for critical issues)
              │
              ▼
    ┌──────────────────────────────────────────────┐
    │          CONTINUOUS MONITORING               │
    │   (Agent stays active, monitors production)  │
    └──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RETRY & ESCALATION LOGIC                      │
└─────────────────────────────────────────────────────────────────┘

    ANY PHASE FAILS
           │
           ▼
    ┌──────────────────────┐
    │ Failure Analysis     │
    ├──────────────────────┤
    │ 1. Extract error     │
    │ 2. Check known DB    │
    │ 3. Query RAG         │
    │ 4. Score confidence  │
    └───────┬──────────────┘
            │
  ┌─────────┴──────────┐
  │                    │
  ▼                    ▼
┌──────────────┐  ┌──────────────┐
│ Known Fix    │  │ Novel Error  │
│ Conf: ≥0.5   │  │ Conf: <0.5   │
└───────┬──────┘  └───────┬──────┘
        │                 │
        │ Auto-retry      │ Escalate
        │ (up to 3x)      │ to human
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Apply Fix    │  │ Human Review │
│ + Re-test    │  │ + Learn      │
└───────┬──────┘  └───────┬──────┘
        │                 │
        └────────┬────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Update Known DB    │
        │ (Network Effect)   │
        └────────────────────┘
```

### Autonomous Agent Phases (Rust Enum)

```rust
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
    UnitTesting,               // 🟡 Test generation done, execution needed
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
```

### Why This Is Revolutionary

**Traditional Development:**
1. Developer writes code (4 hours)
2. Developer manually tests (1 hour)
3. Developer fixes bugs (2 hours)
4. Developer creates Dockerfile (30 min)
5. Developer sets up CI/CD (1 hour)
6. Developer deploys to staging (30 min)
7. Developer monitors, finds issue, hotfixes (2 hours)
**Total: 11 hours, manual work, error-prone**

**With Yantra:**
1. User: "Add payment processing"
2. Agent: Generates → Tests → Fixes → Packages → Deploys → Monitors
**Total: 10 minutes, fully automated, guaranteed no breaks**

**Time Savings: 98%**  
**Error Reduction: 99%+ (GNN prevents breaking changes)**  
**Human Role: Provide intent, approve deployments**

**Factors:**

| Factor | Weight | Scoring |
|--------|--------|---------|
| LLM confidence | 30% | From LLM response metadata |
| Test pass rate | 25% | % of tests passing |
| Known failure match | 25% | Similarity to solved issues |
| Code complexity | 10% | Cyclomatic complexity |
| Dependency changes | 10% | # of files affected |

**Thresholds:**
* **>0.8:** High confidence → Auto-retry (up to 3 attempts)
* **0.5-0.8:** Medium confidence → Auto-retry once, then escalate
* **<0.5:** Low confidence → Immediate human review

**Implementation:**
```rust
pub struct ConfidenceScore {
    llm_confidence: f32,        // 0.0-1.0
    test_pass_rate: f32,        // 0.0-1.0
    known_failure_match: f32,   // 0.0-1.0
    code_complexity: f32,       // 0.0-1.0
    dependency_impact: f32,     // 0.0-1.0
}

impl ConfidenceScore {
    pub fn overall(&self) -> f32 {
        self.llm_confidence * 0.3 +
        self.test_pass_rate * 0.25 +
        self.known_failure_match * 0.25 +
        (1.0 - self.code_complexity) * 0.1 +
        (1.0 - self.dependency_impact) * 0.1
    }
    
    pub fn should_auto_retry(&self) -> bool {
        self.overall() > 0.5
    }
}
```

---

## Terminal Integration Architecture (🆕 Week 9-10)

### Why Terminal Integration is Critical

**Problem:** Developers currently switch between:
- IDE for code
- Terminal for running scripts
- Terminal for installing dependencies
- Terminal for building packages
- Terminal for deploying
- Browser for monitoring

**Yantra Solution:** Integrated terminal with autonomous command execution.

### Design Principles

1. **Controlled Execution:** Whitelist approach, not blacklist
2. **Streaming Output:** Real-time feedback to user
3. **Security First:** No arbitrary command execution
4. **Error Recovery:** Automatic retry with intelligent analysis
5. **Context Awareness:** Commands run in project context (venv, cwd, env vars)

### Terminal Executor Module (`src/agent/terminal.rs`)

```rust
pub struct TerminalExecutor {
    workspace_path: PathBuf,
    python_env: Option<PathBuf>,    // Path to venv
    node_env: Option<PathBuf>,      // Path to node_modules
    env_vars: HashMap<String, String>,
    command_whitelist: CommandWhitelist,
}

pub enum CommandType {
    PythonRun,           // python script.py
    PythonTest,          // pytest tests/
    PythonInstall,       // pip install package
    NodeRun,             // node script.js, npm run build
    NodeTest,            // npm test, jest
    NodeInstall,         // npm install package
    RustBuild,           // cargo build, cargo test
    DockerBuild,         // docker build, docker run
    GitCommand,          // git status, git commit (via MCP)
    CloudDeploy,         // aws, gcloud, kubectl commands
}

pub struct CommandWhitelist {
    allowed_commands: HashSet<String>,    // ["python", "pip", "npm", "node", "cargo", "docker", "git", "aws", "kubectl"]
    allowed_patterns: Vec<Regex>,         // Pre-compiled regex patterns
    blocked_patterns: Vec<Regex>,         // rm -rf, sudo, etc.
}

pub struct ExecutionResult {
    command: String,
    exit_code: i32,
    stdout: String,
    stderr: String,
    execution_time: Duration,
    success: bool,
}
```

### Security: Command Validation

**Whitelist-Based Validation:**
```rust
impl TerminalExecutor {
    pub fn validate_command(&self, cmd: &str) -> Result<ValidatedCommand, SecurityError> {
        // Step 1: Extract base command
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let base_cmd = parts.first().ok_or(SecurityError::EmptyCommand)?;
        
        // Step 2: Check whitelist
        if !self.command_whitelist.allowed_commands.contains(base_cmd) {
            return Err(SecurityError::DisallowedCommand(base_cmd.to_string()));
        }
        
        // Step 3: Check blocked patterns (rm -rf, sudo, eval, etc.)
        for blocked in &self.command_whitelist.blocked_patterns {
            if blocked.is_match(cmd) {
                return Err(SecurityError::DangerousPattern);
            }
        }
        
        // Step 4: Validate arguments (no shell injection)
        for arg in &parts[1..] {
            if arg.contains(';') || arg.contains('|') || arg.contains('&') {
                return Err(SecurityError::ShellInjection);
            }
        }
        
        Ok(ValidatedCommand { command: cmd.to_string(), cmd_type: self.classify(cmd) })
    }
}
```

**Allowed Commands:**
- **Python:** `python`, `python3`, `pip`, `pytest`, `black`, `flake8`
- **Node:** `node`, `npm`, `npx`, `yarn`, `jest`
- **Rust:** `cargo` (build, test, run)
- **Docker:** `docker` (build, run, ps, stop)
- **Git:** `git` (via MCP protocol for security)
- **Cloud:** `aws`, `gcloud`, `kubectl`, `terraform`, `heroku`

**Blocked Patterns:**
- `rm -rf`, `sudo`, `su`, `chmod +x`
- `eval`, `exec`, `source`
- Shell redirects to system files: `> /etc/*`
- Network commands: `curl | bash`, `wget | sh`

### Streaming Output Implementation

```rust
pub async fn execute_with_streaming(
    &self,
    cmd: &str,
    output_sender: tokio::sync::mpsc::Sender<String>,
) -> Result<ExecutionResult> {
    let validated = self.validate_command(cmd)?;
    
    let mut child = tokio::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(&validated.command)
        .current_dir(&self.workspace_path)
        .envs(&self.env_vars)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    let stdout = child.stdout.take().ok_or(anyhow!("No stdout"))?;
    let stderr = child.stderr.take().ok_or(anyhow!("No stderr"))?;
    
    // Stream stdout in real-time
    let stdout_sender = output_sender.clone();
    let stdout_task = tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Some(line) = lines.next_line().await.ok().flatten() {
            let _ = stdout_sender.send(format!("[stdout] {}\n", line)).await;
        }
    });
    
    // Stream stderr in real-time
    let stderr_task = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Some(line) = lines.next_line().await.ok().flatten() {
            let _ = output_sender.send(format!("[stderr] {}\n", line)).await;
        }
    });
    
    // Wait for completion
    let status = child.wait().await?;
    stdout_task.await?;
    stderr_task.await?;
    
    Ok(ExecutionResult {
        command: validated.command,
        exit_code: status.code().unwrap_or(-1),
        success: status.success(),
        // ... other fields
    })
}
```

### Integration with Agent Orchestrator

**Execution Phase Flow:**
```rust
// In src/agent/orchestrator.rs

async fn handle_environment_setup(&mut self) -> Result<()> {
    // 1. Detect project type (Python, Node, Rust, etc.)
    let project_type = self.detect_project_type()?;
    
    // 2. Create virtual environment if needed
    match project_type {
        ProjectType::Python => {
            self.terminal.execute("python -m venv .venv").await?;
            self.terminal.set_python_env(PathBuf::from(".venv"))?;
        },
        ProjectType::Node => {
            // Node already uses local node_modules
        },
        _ => {}
    }
    
    // 3. Set environment variables
    self.terminal.set_env_var("PYTHONPATH", &self.workspace_path)?;
    
    self.transition_to(AgentPhase::DependencyInstallation);
    Ok(())
}

async fn handle_dependency_installation(&mut self) -> Result<()> {
    let project_type = self.state.project_type;
    
    match project_type {
        ProjectType::Python => {
            // Install from requirements.txt
            if self.workspace_path.join("requirements.txt").exists() {
                let result = self.terminal.execute("pip install -r requirements.txt").await?;
                if !result.success {
                    return self.handle_dependency_failure(result);
                }
            }
        },
        ProjectType::Node => {
            // Install from package.json
            if self.workspace_path.join("package.json").exists() {
                let result = self.terminal.execute("npm install").await?;
                if !result.success {
                    return self.handle_dependency_failure(result);
                }
            }
        },
        _ => {}
    }
    
    self.transition_to(AgentPhase::ScriptExecution);
    Ok(())
}

async fn handle_script_execution(&mut self) -> Result<()> {
    // Execute the generated code
    let entry_point = self.find_entry_point()?;
    let project_type = self.state.project_type;
    
    let command = match project_type {
        ProjectType::Python => format!("python {}", entry_point.display()),
        ProjectType::Node => format!("node {}", entry_point.display()),
        ProjectType::Rust => "cargo run".to_string(),
        _ => return Err(anyhow!("Unsupported project type")),
    };
    
    let result = self.terminal.execute(&command).await?;
    
    if !result.success {
        // Runtime error - analyze and retry
        self.handle_runtime_failure(result).await?;
    } else {
        self.transition_to(AgentPhase::RuntimeValidation);
    }
    
    Ok(())
}
```

### UI Integration: Output Panel

**Frontend (SolidJS):**
```typescript
// src-ui/components/TerminalOutput.tsx
export const TerminalOutput: Component = () => {
    const [lines, setLines] = createSignal<string[]>([]);
    const [isRunning, setIsRunning] = createSignal(false);
    
    // Listen to Tauri events for streaming output
    onMount(() => {
        listen('terminal-output', (event: Event<string>) => {
            setLines(prev => [...prev, event.payload]);
        });
        
        listen('terminal-complete', (event: Event<ExecutionResult>) => {
            setIsRunning(false);
            if (event.payload.success) {
                setLines(prev => [...prev, '\n✅ Execution successful']);
            } else {
                setLines(prev => [...prev, `\n❌ Execution failed (exit code: ${event.payload.exit_code})`]);
            }
        });
    });
    
    return (
        <div class="terminal-output">
            <div class="terminal-header">
                <span>Output</span>
                {isRunning() && <div class="spinner">Running...</div>}
            </div>
            <div class="terminal-content">
                <For each={lines()}>
                    {(line) => <div class="terminal-line">{line}</div>}
                </For>
            </div>
        </div>
    );
};
```

### Error Recovery: Runtime Failures

**When ScriptExecution Fails:**
1. **Capture Error:** Full stdout + stderr
2. **Classify Error:**
   - Import error → Missing dependency
   - Syntax error → Code generation issue
   - Runtime error → Logic issue
   - Permission error → Environment setup issue
3. **Query Known Fixes:** Check SQLite for similar errors
4. **Generate Fix:** Call LLM with error context
5. **Retry:** Up to 3 attempts

```rust
async fn handle_runtime_failure(&mut self, result: ExecutionResult) -> Result<()> {
    // Extract error message
    let error_msg = self.extract_error_message(&result.stderr)?;
    
    // Classify error type
    let error_type = self.classify_runtime_error(&error_msg)?;
    
    // Check known fixes database
    let known_fix = self.query_known_fixes(&error_msg).await?;
    
    if let Some(fix) = known_fix {
        // Apply known fix
        self.apply_fix(&fix).await?;
        self.retry_count += 1;
        
        if self.retry_count < 3 {
            // Retry execution
            self.transition_to(AgentPhase::ScriptExecution);
        } else {
            // Max retries reached, escalate
            self.transition_to(AgentPhase::Failed);
        }
    } else {
        // Novel error - ask LLM to fix
        self.transition_to(AgentPhase::FixingIssues);
    }
    
    Ok(())
}
```

### Performance Targets

| Operation | Target | Implementation |
|-----------|--------|----------------|
| Command validation | <1ms | Pre-compiled regex, HashSet lookup |
| Command execution start | <50ms | Tokio async spawn |
| Output streaming latency | <10ms | Unbuffered streaming |
| Environment setup | <5s | Cached venv creation |
| Dependency installation | <30s | Use package manager cache |
| Script execution | Varies | Depends on script |

### Week 9-10 Implementation Plan

**Week 9: Core Terminal Module**
- [ ] Implement `TerminalExecutor` struct
- [ ] Command validation with whitelist
- [ ] Subprocess execution with Tokio
- [ ] Streaming output to frontend
- [ ] Unit tests for command validation
- [ ] Integration tests for execution

**Week 10: Agent Integration**
- [ ] Add execution phases to orchestrator
- [ ] Implement environment setup logic
- [ ] Implement dependency installation
- [ ] Implement script execution with retry
- [ ] Add runtime validation
- [ ] Frontend output panel UI
- [ ] E2E test: Generate → Run → Test → Commit

### Security Considerations

1. **No Arbitrary Command Execution:** Only whitelisted commands allowed
2. **Argument Validation:** Block shell injection attempts (`;`, `|`, `&`)
3. **Path Restrictions:** Commands can only access workspace directory
4. **No Privilege Escalation:** Block `sudo`, `su`, `chmod +x`
5. **Network Safety:** Block `curl | bash`, `wget | sh` patterns
6. **Resource Limits:** Timeout after 5 minutes, kill if memory > 2GB
7. **Audit Logging:** Log all executed commands to SQLite for review

**Trade-off Analysis:**
- **Old Design:** "No shell commands for security" → Blocks full automation
- **New Design:** "Controlled command execution" → Enables full automation with security
- **Justification:** Full automation is core value prop; whitelist approach provides security without limiting functionality

---

### Known Issues Database (Network Effect)

**The Innovation:** Learn from every failure across all users (privacy-preserving)

#### Data Structure

**What We Store (Per Failure):**
```rust
pub struct KnownFailurePattern {
    // Pattern (no user code)
    pattern_id: Uuid,
    failure_type: FailureType,  // Test, Security, Browser, Dependency
    error_signature: String,     // Normalized error message
    context_pattern: ContextPattern,  // Generic context (not specific code)
    
    // Fix (generic, reusable)
    fix_strategy: FixStrategy,
    fix_code_template: String,   // Templated, not user-specific
    fix_confidence: f32,
    
    // Metadata
    llm_used: String,            // Which LLM made the mistake
    success_rate: f32,           // % of times fix worked
    occurrence_count: u32,       // How many times seen
    first_seen: DateTime,
    last_seen: DateTime,
}
```

**What We DON'T Store:**
* User code (privacy violation)
* File names or paths (identifying information)
* Variable names (user-specific)
* Business logic (proprietary)

#### Privacy-Preserving Pattern Extraction

**Example:**

**User Code (Private):**
```python
def calculate_user_discount(user_id: str, cart_total: float) -> float:
    user = database.get_user(user_id)  # NameError: database not defined
    return cart_total * user.discount_rate
```

**Extracted Pattern (Stored):**
```json
{
  "error_signature": "NameError: name '{variable}' is not defined",
  "context_pattern": {
    "ast_structure": "FunctionDef with Call to undefined variable",
    "missing_import": true,
    "suggested_imports": ["database module pattern"]
  },
  "fix_strategy": "ADD_IMPORT",
  "fix_code_template": "from {module} import {variable}",
  "success_rate": 0.95
}
```

**Result:** Pattern is reusable WITHOUT exposing user code.

#### Failure Types Tracked

**1. Test Failures:**
* Assertion errors (expected vs actual)
* Missing test fixtures
* Mock/stub configuration
* Async test issues

**2. Security Vulnerabilities:**
* SQL injection patterns
* XSS vulnerabilities
* Insecure deserialization
* Hardcoded secrets

**3. Browser Runtime Errors:**
* Console errors (JavaScript)
* Network request failures
* DOM manipulation issues
* Authentication redirects

**4. Dependency Breaks:**
* Import errors (missing modules)
* API signature mismatches
* Breaking changes in called functions
* Data type incompatibilities

#### Network Effect Mechanism

**Local First:**
* Each Yantra instance maintains local known issues DB
* Updated in real-time during usage

**Opt-In Sharing (Anonymous):**
```
User opts in → Failure patterns (only) uploaded
             → Aggregated with other users
             → Downloaded updates daily
             → Local DB enriched
```

**Privacy Guarantees:**
1. No code ever leaves user's machine (unless user opts in to pattern sharing)
2. Patterns are anonymized and generalized
3. User can review what's shared before upload
4. Can disable sharing anytime
5. Open source pattern extraction code (auditable)

**Growth Formula:**
```
Network Value = N × (Patterns per User) × (Fix Success Rate)

With 10,000 users:
- Each encounters ~100 unique failures/year
- 50% opt-in to sharing
- Total patterns: 500,000/year
- Each new user benefits from collective knowledge
```

### Validation Pipeline Details

#### 1. Dependency Validation (GNN)

**Check:**
* No breaking changes to existing function signatures
* All imports exist and are accessible
* Data types match (function args, return values)
* No circular dependencies introduced

**Implementation:**
```rust
pub fn validate_dependencies(
    gnn: &GNN,
    generated_code: &GeneratedCode
) -> ValidationResult {
    // 1. Parse generated code to AST
    // 2. Identify all function calls
    // 3. Check each call against GNN
    // 4. Verify signatures match
    // 5. Return breaks or OK
}
```

**Performance:** <10ms per validation

#### 2. Unit Test Execution

**Process:**
* Generate unit tests with LLM (separate call)
* Execute via pytest subprocess
* Parse JUnit XML results
* Track pass/fail/error counts

**Auto-Retry Logic:**
* If test fails → Analyze error
* Check known failures DB
* If match found → Apply fix automatically
* Re-run tests (up to 3 attempts)

**Performance Target:** <30s for typical project

#### 3. Integration Test Execution

**Process:**
* Generate integration tests (E2E scenarios)
* Set up test fixtures/mocks
* Execute multi-step workflows
* Validate end-to-end behavior

**Coverage:**
* API endpoints (request → response)
* Database operations (CRUD)
* External service calls (mocked)

**Performance Target:** <60s for typical project

#### 4. Security Scanning

**Tools:**
* Semgrep with OWASP ruleset
* Custom rules for common vulnerabilities
* Dependency vulnerability check (Safety, npm audit)
* Secret detection (TruffleHog patterns)

**Auto-Fix:**
* Many vulnerabilities have standard fixes
* SQL injection → Use parameterized queries
* XSS → Escape user input
* Apply fix + re-scan automatically

**Performance Target:** <10s

#### 5. Browser Validation (UI Code)

**Process:**
* Start Chrome via CDP (headless)
* Load application
* Monitor console for errors
* Execute basic user flows
* Capture network errors

**Auto-Healing:**
* Console error → Extract stack trace
* Check known issues DB
* Apply fix if confidence >0.7
* Re-test automatically

**Performance Target:** <30s for UI validation

### Agent State Persistence

**Why:** Agent may run for minutes, need to resume if interrupted

**State Stored in SQLite:**
```rust
pub struct AgentState {
    session_id: Uuid,
    current_phase: AgentPhase,  // Generation, Validation, Fixing
    attempt_count: u32,
    confidence_scores: Vec<ConfidenceScore>,
    validation_results: Vec<ValidationResult>,
    applied_fixes: Vec<AppliedFix>,
    created_at: DateTime,
    updated_at: DateTime,
}

pub enum AgentPhase {
    ContextAssembly,
    CodeGeneration,
    DependencyValidation,
    UnitTesting,
    IntegrationTesting,
    SecurityScanning,
    BrowserValidation,
    FixingIssues,
    GitCommit,
    Complete,
    Failed,
}
```

**Resume Capability:**
* If Yantra crashes → Reload state from DB
* Continue from last phase
* No re-work needed

### LLM Mistake Tracking Integration

**Existing Implementation (src/gnn/known_issues.rs):**
```rust
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
```

**Enhancement for Agentic Pipeline:**

**1. Automatic Capture:**
* Every validation failure → Create KnownIssue entry
* Store LLM used, error, fix (if found)
* Link to failure pattern (for network effect)

**2. Automatic Retrieval:**
* Before retry → Query known issues DB
* Match by error signature + context
* If confidence >0.8 → Apply fix automatically
* Track success rate

**3. Continuous Learning:**
* Every successful fix → Update success_rate
* Every failed fix → Lower confidence
* Prune low-success patterns (<0.3 after 10 attempts)

### Implementation Phases

**MVP (Month 1-2):**
* ✅ Test generation and execution (implemented)
* ✅ Known issues tracking (implemented)
* ⚠️ Confidence scoring system (add)
* ⚠️ Auto-retry logic with known fixes (add)
* ⚠️ Agent state machine (basic) (add)

**Post-MVP (Month 3-4):**
* Full validation pipeline (tests + security + browser)
* Pattern extraction from failures
* Network effect (opt-in sharing)
* Advanced confidence scoring

**Enterprise (Month 5-8):**
* Self-healing workflows
* Cross-system validation
* Distributed agent coordination
* Advanced auto-fixing (ML-based)

---

## Phase 1: MVP (Months 1-2)

### Objectives

Prove Yantra can generate production-quality code that:

1. Never breaks existing code (GNN validation)
2. Passes all tests automatically (100% pass rate)
3. Has no critical security vulnerabilities
4. Works on first deployment (no debugging needed)

### Success Metrics

* Generate working code for 10+ scenarios (auth, CRUD, APIs, etc.)
* 95% of generated code passes all tests without human intervention
* Zero breaking changes to existing code
* <3% critical security vulnerabilities (auto-fixed)
* Developer NPS >40

### Scope

In Scope: 
✅ Python codebase support (single language focus) 
✅ Internal code dependency tracking 
✅ Multi-LLM orchestration (Claude + GPT-4 + Qwen Coder support)
✅ GNN for code dependencies 
✅ **Token-aware context assembly (truly unlimited context - MVP foundation)**
✅ **Hierarchical context (L1 + L2) with compression**
✅ **Token counting with tiktoken-rs**
✅ Automated unit + integration test generation 
✅ **Confidence scoring system**
✅ **Known issues database (LLM failures + fixes)**
✅ **Basic agent state machine with auto-retry**
✅ Security vulnerability scanning 
✅ Browser integration for runtime validation 
✅ Git integration (commit/push via MCP) 
✅ Monaco editor for code viewing 
✅ Chat interface for task input

Out of Scope (Post-MVP): 
⏭️ Advanced context compression (semantic chunking) 
⏭️ Full RAG with ChromaDB 
⏭️ Pattern extraction and network effect sharing
⏭️ Full agentic validation pipeline (all 5 validations)
⏭️ Multi-language support 
⏭️ External API dependency tracking 
⏭️ Workflow automation 
⏭️ Advanced refactoring 
⏭️ Team collaboration features
⏭️ Multitenancy and user accounts

### Implementation Plan (8 Weeks)

Week 1-2: Foundation

* Tauri + SolidJS project setup
* 3-panel UI layout (chat, code, preview)
* Monaco editor integration
* File system operations
* Basic file tree component
* Project loading (select folder)

Week 3-4: GNN Engine

* tree-sitter Python parser integration
* AST extraction (functions, classes, variables)
* Graph data structures in Rust
* Dependency detection (calls, imports, data flow)
* Incremental updates
* SQLite persistence
* **Known issues database schema + storage**

Week 5-6: LLM Integration + Unlimited Context Foundation

* Claude + GPT-4 + Qwen Coder API clients
* Multi-LLM orchestrator with failover
* **tiktoken-rs integration for token counting**
* **Hierarchical context assembly (L1 + L2)**
* **Token-aware context budgeting per LLM**
* **Basic compression (whitespace, comments)**
* Prompt template system
* Code generation from natural language
* **Confidence scoring system**
* Unit test generation
* Integration test generation
* Test execution (pytest runner)

Week 7: Agentic Validation Pipeline (MVP)

* **Agent state machine (basic phases)**
* **Auto-retry logic with confidence scoring**
* **Known issues retrieval and matching**
* Dependency validation via GNN
* Unit test execution with auto-retry
* Semgrep security scanning
* Chrome DevTools Protocol integration
* Console error monitoring
* **Failure pattern capture (local only in MVP)**
* Git integration (commit with auto-messages)

Week 8: Polish & Beta

* UI/UX improvements
* Error handling and loading states
* Performance optimization
* **LLM comparison testing (GPT-4 vs Qwen Coder)**
* Documentation (getting started guide)
* Beta release to 10-20 developers
* Collect feedback

Deliverable: Desktop app (macOS, Windows, Linux) that generates, tests, validates, and commits Python code with agentic capabilities and token-aware unlimited context.

---

## Phase 2: Advanced Context + Network Effect (Months 3-4)

### Objectives

Complete unlimited context implementation and enable network effects:

* Full RAG with ChromaDB
* Advanced context compression
* Pattern extraction from failures
* Opt-in anonymous pattern sharing
* Full validation pipeline (5 validations)

### New Capabilities

1. **Complete Unlimited Context Engine**

* RAG with ChromaDB for code patterns
* Semantic search for relevant examples
* Advanced compression (semantic chunking)
* Full hierarchical context (L1-L4)
* Adaptive strategies per task type
* Context caching for performance

2. **Network Effect System**

* Privacy-preserving pattern extraction
* Anonymous failure pattern aggregation
* Opt-in pattern sharing UI
* Daily pattern database updates
* Pattern success rate tracking
* User-reviewable sharing logs

3. **Full Agentic Validation Pipeline**

* All 5 validations (dependency, unit test, integration test, security, browser)
* Advanced auto-fixing with ML patterns
* Multi-attempt retry strategies
* Escalation to human with context
* Session resumption after crashes

4. **Workflow Foundation** (Original Phase 2 content)

* Cron scheduler
* Webhook server (Axum web framework)
* Event-driven triggers
* Retry logic with exponential backoff
* Execution history and logs

### External API Integration

* API schema discovery (OpenAPI specs)
* Track API calls in GNN
* Support: Slack, SendGrid, Stripe
* Generic REST API support (via config)

3. Multi-Step Workflows

* Chain 3-5 actions
* Conditional branching (if/else)
* Error handling (try/catch)
* Data passing between steps

Example Use Case:

Webhook: Stripe payment success

→ Update database (mark order paid)

→ Send confirmation email (SendGrid)

→ Notify sales team (Slack)

→ Log to analytics

### Implementation (8 Weeks)

Weeks 9-10: Workflow definition (YAML), executor, scheduler Weeks 11-12: External API integration framework Weeks 13-14: Error handling, logging, monitoring dashboard Weeks 15-16: LLM workflow generation, beta release

---

## Phase 2B: Cluster Agents Architecture (Months 3-4, Parallel Development)

### Overview

**Problem:** As codebases scale beyond 100k LOC and teams grow to 5+ concurrent developers, single-agent architecture becomes a bottleneck. Developers need multiple AI agents working simultaneously on different parts of the codebase without conflicts.

**Solution:** Transform Yantra from a single autonomous agent to a **cluster of coordinating agents** using a Master-Servant architecture with Agent-to-Agent (A2A) protocol for proactive conflict prevention.

**Key Innovation:** Unlike traditional collaborative editing (which reactively resolves conflicts), Yantra uses **proactive conflict prevention** - agents communicate intent before making changes, preventing conflicts rather than resolving them after the fact.

### Why Cluster Agents?

**Current Limitations (Single Agent):**
- Only one developer can use Yantra at a time
- Large codebases (100k+ LOC) exceed context limits
- Complex features require serial execution of multiple tasks
- Bottleneck for team collaboration

**Cluster Agent Benefits:**
- **Parallelization:** 3-10 agents working simultaneously
- **Specialization:** Dedicated agents for frontend, backend, testing, DevOps
- **Scalability:** Handle 100k+ LOC codebases efficiently
- **Team Collaboration:** Multiple developers with their own agents
- **Fault Tolerance:** One agent failure doesn't block others

**Pricing Tiers:**
- **Starter (1 agent):** $29/month - Solo developers
- **Professional (3 agents):** $99/month - Small teams (2-5 developers)
- **Enterprise (10 agents):** $299/month - Large teams (5+ developers)

---

### Architecture: Master-Servant Pattern

**Why Master-Servant over Peer-to-Peer?**

**Rejected: Peer-to-Peer (P2P)**
- ❌ No single source of truth (coordination nightmare)
- ❌ Complex consensus algorithms (Raft/Paxos)
- ❌ Race conditions on file writes
- ❌ Conflict resolution after-the-fact

**Chosen: Master-Servant**
- ✅ Single source of truth (Master agent)
- ✅ Clear hierarchy and responsibility
- ✅ Proactive conflict prevention (not reactive resolution)
- ✅ Simple state management
- ✅ Easy to reason about and debug

---

### Master Agent Responsibilities

**1. Task Decomposition**
```
User: "Add payment processing with Stripe"

Master Agent Breakdown:
├─ Task 1: Create payment API endpoint (Backend Agent)
├─ Task 2: Add Stripe SDK integration (Backend Agent)
├─ Task 3: Build payment form UI (Frontend Agent)
├─ Task 4: Add payment success page (Frontend Agent)
├─ Task 5: Write integration tests (Testing Agent)
└─ Task 6: Update deployment config (DevOps Agent)
```

**2. Dependency Analysis**
- Build task dependency graph using GNN
- Identify parallel tasks (can run simultaneously)
- Identify sequential tasks (must wait for dependencies)
- Assign tasks to appropriate servant agents

**3. Conflict Prevention Coordination**
- Maintain file access registry (which agent is editing what)
- Approve or deny servant agent IntentToModify requests
- Prevent multiple agents from editing same file
- Track file modification history

**4. Work Validation**
- Collect completed work from servants
- Run final validation (tests, security, browser)
- Merge changes into main codebase
- Handle integration conflicts if they occur

**5. Error Recovery**
- Detect when servant agent fails
- Reassign task to another agent
- Maintain task queue and retry logic
- Escalate to human if automated recovery fails

---

### Servant Agent Responsibilities

**1. Specialized Execution**

Each servant agent specializes in specific domains:

- **Backend Agent:** API endpoints, database models, business logic
- **Frontend Agent:** UI components, state management, styling
- **Testing Agent:** Unit tests, integration tests, test fixtures
- **DevOps Agent:** Deployment, infrastructure, monitoring
- **Security Agent:** Vulnerability scanning, auto-fixing (can run in parallel)
- **Documentation Agent:** Code comments, API docs, user guides

**2. Peer Coordination via A2A Protocol**

Servants communicate directly with each other (not through master) for:
- Querying dependencies: "Does UserService.login() exist?"
- Checking interfaces: "What's the signature of PaymentAPI.charge()?"
- Sharing context: "I'm adding a new field to User model"
- Negotiating changes: "Can I modify auth.py or are you using it?"

**3. Proactive Conflict Prevention**

**Before making any change:**
1. Servant sends `IntentToModify(file_path, reason)` to Master
2. Master checks file access registry
3. If no conflict:
   - Master approves → Servant proceeds
   - Other servants notified via A2A protocol
4. If conflict detected:
   - Master denies → Servant waits or finds alternative approach
   - Master suggests: "UserService.login() is being modified by Backend Agent. Wait 30s or use auth.py instead?"

**4. Bounded Execution**

Each servant operates within defined boundaries:
- **File scope:** Only modify assigned files
- **Time limit:** Complete task within 5 minutes or report progress
- **Resource limits:** Max 500 LLM tokens per generation
- **Validation:** Local validation before submitting to Master

---

### Agent-to-Agent (A2A) Protocol

**Protocol Design Principles:**
- **Proactive, not reactive:** Prevent conflicts before they happen
- **Lightweight:** Minimal overhead for high-frequency communication
- **Stateless:** No persistent connections, message-based
- **Structured:** JSON schema for all messages

**Core Message Types:**

```json
// 1. IntentToModify - Declare intention to change file
{
  "type": "IntentToModify",
  "agent_id": "backend-agent-1",
  "file_path": "src/payment/stripe.py",
  "operation": "create|update|delete",
  "reason": "Adding Stripe payment integration",
  "estimated_duration": "2min",
  "timestamp": "2025-11-23T10:30:00Z"
}

// 2. ChangeCompleted - Notify completion
{
  "type": "ChangeCompleted",
  "agent_id": "backend-agent-1",
  "file_path": "src/payment/stripe.py",
  "changes": {
    "functions_added": ["charge_card", "refund_payment"],
    "dependencies_added": ["stripe"],
    "lines_changed": 120
  },
  "timestamp": "2025-11-23T10:32:00Z"
}

// 3. QueryDependency - Ask about code dependencies
{
  "type": "QueryDependency",
  "agent_id": "frontend-agent-1",
  "query": "Does PaymentAPI.charge() support retry_count parameter?",
  "target_file": "src/payment/stripe.py",
  "timestamp": "2025-11-23T10:33:00Z"
}

// 4. DependencyResponse - Answer dependency query
{
  "type": "DependencyResponse",
  "agent_id": "backend-agent-1",
  "query_id": "query-123",
  "answer": {
    "exists": true,
    "signature": "charge(amount: float, currency: str, retry_count: int = 3) -> PaymentResult",
    "location": "src/payment/stripe.py:45"
  },
  "timestamp": "2025-11-23T10:33:01Z"
}

// 5. ConflictNegotiation - Resolve potential conflicts
{
  "type": "ConflictNegotiation",
  "agent_id": "frontend-agent-1",
  "conflict": {
    "file_path": "src/models/user.py",
    "my_intent": "Add 'phone_number' field to User model",
    "conflicting_agent": "backend-agent-1",
    "conflicting_intent": "Refactoring User model structure"
  },
  "proposal": "I'll wait for backend-agent-1 to finish, then add my field",
  "timestamp": "2025-11-23T10:35:00Z"
}
```

**Communication Flow Example:**

```
[Frontend Agent] → [Master]: IntentToModify(payment_form.tsx)
[Master] → [Frontend Agent]: Approved (no conflicts)
[Frontend Agent] → [All Servants]: IntentToModify broadcast (via A2A)

[Frontend Agent] needs PaymentAPI info
[Frontend Agent] → [Backend Agent]: QueryDependency(PaymentAPI.charge signature)
[Backend Agent] → [Frontend Agent]: DependencyResponse(signature details)

[Frontend Agent] completes work
[Frontend Agent] → [Master]: ChangeCompleted(payment_form.tsx)
[Master] → [All Servants]: ChangeCompleted broadcast (via A2A)
```

---

### Hybrid Intelligence: Vector DB + GNN

**Why Hybrid Architecture?**

**Pure Vector DB Approach (Rejected):**
- ❌ Semantic understanding only (can't guarantee type safety)
- ❌ No structural dependency tracking
- ❌ Can't detect breaking changes (function signature changes)
- ❌ Slow for real-time dependency validation

**Pure GNN Approach (Rejected):**
- ❌ Structural dependencies only (no semantic understanding)
- ❌ Can't find similar patterns or examples
- ❌ Poor LLM context retrieval (needs exact matches)
- ❌ Doesn't capture intent or purpose

**Hybrid Vector DB + GNN (Chosen):**
- ✅ **Vector DB:** Semantic understanding (What does this function do? Find similar patterns)
- ✅ **GNN:** Structural dependencies (What imports what? Who calls this function?)
- ✅ **Combined:** Best of both worlds

---

### Vector Database Integration (ChromaDB)

**Purpose:**
- Semantic code search for LLM context
- Pattern matching for similar code
- Documentation and comment search
- Error message similarity for auto-fix

**What Gets Stored in Vector DB:**

1. **Function/Class Embeddings**
```python
# Source code
def calculate_payment_fee(amount: float, currency: str) -> float:
    """Calculate processing fee for payment.
    
    Args:
        amount: Payment amount
        currency: Currency code (USD, EUR, etc.)
    
    Returns:
        Processing fee amount
    """
    fee_rate = 0.029  # 2.9% Stripe fee
    return amount * fee_rate

# Vector DB entry
{
  "id": "func-calculate_payment_fee",
  "type": "function",
  "name": "calculate_payment_fee",
  "signature": "calculate_payment_fee(amount: float, currency: str) -> float",
  "docstring": "Calculate processing fee for payment...",
  "embedding": [0.123, -0.456, ...],  # 384-dim vector
  "file_path": "src/payment/fees.py",
  "line_start": 10,
  "line_end": 20
}
```

2. **Code Comments** (for context)
3. **Error Messages** (for pattern-based auto-fix)
4. **Test Cases** (for similar test examples)
5. **Documentation** (README, API docs)

**Vector Search Queries:**

```python
# Agent needs to implement payment refund
query = "How to refund a payment transaction?"

# Vector DB returns semantically similar code
results = vector_db.search(query, top_k=3)
# Result 1: refund_payment() function in stripe.py
# Result 2: cancel_subscription() in billing.py
# Result 3: reverse_transaction() in ledger.py

# Agent uses these as LLM context for generation
```

**Performance:**
- **Search latency:** <50ms for semantic search
- **Index update:** Real-time (on file save)
- **Storage:** ~10MB per 10k LOC (embeddings + metadata)

---

### GNN Integration for Structural Dependencies

**Purpose:**
- Track function calls, imports, inheritance
- Detect breaking changes (signature modifications)
- Validate dependencies before code execution
- Build task dependency graphs

**GNN vs Vector DB - When to Use:**

| Use Case | Tool | Reason |
|----------|------|--------|
| "Find functions that call UserService.login()" | GNN | Structural dependency |
| "Find similar authentication code" | Vector DB | Semantic similarity |
| "Will changing this function break anything?" | GNN | Impact analysis |
| "How do other projects handle OAuth?" | Vector DB | Pattern search |
| "What imports this module?" | GNN | Direct dependency |
| "Find code related to payment processing" | Vector DB | Semantic search |

**Combined Query Example:**

```python
# Agent task: "Refactor UserService.login() to support 2FA"

# Step 1: GNN - Find all callers
callers = gnn.get_function_callers("UserService.login")
# Result: ["auth_api.py:45", "login_view.py:30", "test_auth.py:20"]

# Step 2: Vector DB - Find similar 2FA implementations
patterns = vector_db.search("two-factor authentication implementation", top_k=5)
# Result: Similar code from other projects or modules

# Step 3: GNN - Validate new signature doesn't break callers
new_signature = "login(username: str, password: str, otp: str = None) -> bool"
breaking_changes = gnn.validate_signature_change(
    "UserService.login",
    new_signature
)
# Result: No breaking changes (optional parameter added)

# Agent proceeds with refactoring
```

---

### Real-Time Synchronization

**Challenge:** Multiple agents editing simultaneously - how to keep Vector DB and GNN in sync?

**Strategy: Hybrid Sync**

| Component | Sync Strategy | Latency | Reason |
|-----------|---------------|---------|--------|
| **Vector DB** | Real-time (on file save) | <100ms | Semantic search must be current |
| **GNN** | Periodic (every 2-3s) | 2-3s | Structural changes less frequent |
| **File Access Registry** | Real-time (on intent) | <10ms | Critical for conflict prevention |

**Why Periodic GNN Sync is Acceptable:**

1. **Structural changes are rare:** Most edits don't change function signatures or imports
2. **Intent-based locking:** Agents declare intent before modifying, so GNN has time to update
3. **Performance:** Full GNN rebuild for 10k LOC takes ~500ms, periodic is more efficient
4. **Validation:** Master runs final GNN validation before merging changes

**Sync Flow:**

```
[Agent A] Saves file → [Vector DB] Immediate index update (100ms)
                     ↓
[GNN Sync Thread] Checks every 2s → Detects file change → Incremental GNN update (50ms)
                     ↓
[Agent B] Queries dependency → [GNN] Returns up-to-date graph (10ms)
```

**File Access Registry (Real-Time):**

```rust
// In-memory registry for fast lookups
pub struct FileAccessRegistry {
    files: HashMap<PathBuf, FileAccess>,
}

pub struct FileAccess {
    agent_id: String,
    operation: Operation,  // Read, Write, Delete
    start_time: Instant,
    estimated_duration: Duration,
}

// Real-time operations
registry.lock_file("src/payment/stripe.py", "backend-agent-1", Write, 2min);
registry.check_conflict("src/payment/stripe.py");  // <1ms lookup
registry.unlock_file("src/payment/stripe.py", "backend-agent-1");
```

---

### Confidence-Based LLM Routing

**Problem:** Phase 1 uses perplexity-based routing (high perplexity = complex task → use better LLM). But perplexity doesn't capture task success confidence.

**New Approach: Confidence-Based Routing**

**Confidence Score Calculation:**
```rust
pub struct ConfidenceScore {
    llm_confidence: f64,      // From LLM response metadata (0-1)
    validation_score: f64,    // Tests + security + GNN validation (0-1)
    complexity_penalty: f64,  // High LOC or deep nesting reduces confidence
    historical_success: f64,  // Similar tasks success rate (0-1)
}

impl ConfidenceScore {
    pub fn calculate(&self) -> f64 {
        let base = (self.llm_confidence + self.validation_score) / 2.0;
        let adjusted = base * (1.0 - self.complexity_penalty);
        adjusted * 0.7 + self.historical_success * 0.3
    }
}
```

**Routing Strategy:**

```rust
pub enum RoutingMode {
    Adaptive {
        initial_model: String,       // "gpt-4-turbo"
        confidence_threshold: f64,   // 0.8
        max_escalations: usize,      // 2
    },
    AlwaysBest {
        model: String,               // "claude-sonnet-4"
    },
    TaskBased {
        simple_model: String,        // "gpt-3.5-turbo"
        complex_model: String,       // "claude-sonnet-4"
    },
}

impl LlmRouter {
    pub async fn route(&self, task: &Task) -> LlmClient {
        match self.mode {
            Adaptive { initial_model, confidence_threshold, max_escalations } => {
                let mut attempts = 0;
                let mut current_model = initial_model;
                
                loop {
                    let result = self.generate(current_model, task).await;
                    let confidence = result.confidence_score();
                    
                    if confidence >= confidence_threshold {
                        return result;
                    }
                    
                    if attempts >= max_escalations {
                        // Escalate to human
                        return self.request_human_help(task, result);
                    }
                    
                    // Escalate to better model
                    current_model = self.get_better_model(current_model);
                    attempts += 1;
                }
            }
            AlwaysBest { model } => {
                self.generate(model, task).await
            }
            TaskBased { simple_model, complex_model } => {
                if task.complexity() > 0.7 {
                    self.generate(complex_model, task).await
                } else {
                    self.generate(simple_model, task).await
                }
            }
        }
    }
}
```

**Configuration (TOML):**

```toml
[llm_routing]
mode = "Adaptive"

[llm_routing.adaptive]
initial_model = "gpt-4-turbo"
confidence_threshold = 0.8
max_escalations = 2

# Model hierarchy (worst to best)
escalation_chain = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "claude-sonnet-4"
]

# Tradeoffs
# Adaptive: Fast + cheap for simple tasks, escalates for complex
# AlwaysBest: Slow + expensive but highest quality
# TaskBased: Balanced, but requires complexity estimation
```

**Tradeoffs:**

| Mode | Speed | Cost | Quality | Best For |
|------|-------|------|---------|----------|
| **Adaptive** | Fast (avg) | Low-Medium | High | Production (95%+ quality, cost-optimized) |
| **AlwaysBest** | Slow | High | Highest | Critical tasks, compliance, security |
| **TaskBased** | Medium | Medium | Medium-High | Predictable workloads, tight budgets |

---

### Conflict Prevention Strategies

**1. Intent-Based Locking**

Before any file modification:
```
[Agent] → [Master]: IntentToModify(file_path)
[Master] checks registry:
  - If file unlocked → Approve + Lock file
  - If file locked by same agent → Approve
  - If file locked by different agent → Deny + Suggest alternative
```

**2. Time-Bounded Locks**

- All locks have expiration (default: 5 minutes)
- Agent must renew lock if work takes longer
- Expired locks automatically released
- Master can force-release if agent crashes

**3. Granular Locking**

Instead of locking entire file:
```python
# Option 1: File-level lock (simple, less parallelism)
lock("src/payment/stripe.py")

# Option 2: Function-level lock (complex, more parallelism)
lock("src/payment/stripe.py::charge_card")
lock("src/payment/stripe.py::refund_payment")
# Different agents can edit different functions in same file
```

**For MVP: File-level locking** (simpler, good enough for 3-10 agents)
**Future: Function-level locking** (for 10+ agents)

**4. Alternative Suggestions**

When conflict detected, Master suggests alternatives:

```
[Frontend Agent]: IntentToModify(user.py) - "Add phone_number field"
[Master]: DENIED - user.py locked by Backend Agent (refactoring)
[Master]: SUGGESTION: 
  - Wait 2 minutes (estimated time remaining)
  - Create new file user_profile.py with phone_number
  - Add phone_number to UserProfile model instead
```

---

### Git Coordination

**Challenge:** Multiple agents committing simultaneously can cause Git conflicts.

**Strategy: Centralized Commit Authority (Master Only)**

**Rules:**
1. **Servants never commit directly** - they submit work to Master
2. **Master validates all changes** before committing
3. **Master runs GNN validation** to detect integration issues
4. **Master creates single atomic commit** with all changes

**Workflow:**

```
[Agent A] Completes payment_api.py
[Agent A] → [Master]: WorkCompleted(payment_api.py)

[Agent B] Completes payment_form.tsx
[Agent B] → [Master]: WorkCompleted(payment_form.tsx)

[Master] Collects all completed work
[Master] Runs GNN validation (no breaking changes?)
[Master] Runs integration tests
[Master] Creates single commit:

git add src/payment/payment_api.py src/ui/payment_form.tsx
git commit -m "feat: Add Stripe payment processing
- Payment API endpoint (Backend Agent)
- Payment form UI (Frontend Agent)
- Integration tests passing (32/32)
- Security scan: 0 vulnerabilities"
```

**Merge Strategy:**

```rust
pub enum MergeStrategy {
    Automatic,    // Master auto-commits if all validations pass
    Manual,       // Master requests human review before commit
    Hybrid,       // Auto-commit for low-risk, manual for high-risk
}

// Risk assessment
fn assess_risk(changes: &[FileChange]) -> RiskLevel {
    if changes.iter().any(|c| c.is_critical_file()) {
        RiskLevel::High  // auth.py, database migrations, etc.
    } else if changes.len() > 10 {
        RiskLevel::Medium  // Many files changed
    } else {
        RiskLevel::Low  // Few files, non-critical
    }
}
```

**Configuration:**

```toml
[git_coordination]
merge_strategy = "Hybrid"

# Auto-commit if:
auto_commit_conditions = [
    "all_tests_pass",
    "no_security_vulnerabilities",
    "risk_level == 'Low'",
    "fewer_than_5_files_changed"
]

# Require human review if:
require_review_conditions = [
    "critical_files_modified",  # auth.py, migrations, etc.
    "more_than_10_files_changed",
    "security_vulnerabilities_detected"
]

critical_files = [
    "src/auth/**",
    "src/database/migrations/**",
    "src/security/**",
    ".github/workflows/**"
]
```

---

### Scalability & Performance

**Target Performance:**

| Metric | Single Agent | 3 Agents | 10 Agents |
|--------|--------------|----------|-----------|
| **Codebase Size** | 10k LOC | 50k LOC | 100k+ LOC |
| **Concurrent Tasks** | 1 | 3 | 10 |
| **Context Build Time** | <500ms | <800ms | <1.5s |
| **Conflict Detection** | N/A | <10ms | <20ms |
| **Commit Frequency** | Every task | Every 3 tasks | Every 10 tasks |
| **Cost per Feature** | $0.20 | $0.25 | $0.35 |

**Optimization Strategies:**

1. **Incremental GNN Updates:** Only rebuild affected subgraphs
2. **Vector DB Caching:** Cache embeddings for unchanged code
3. **Lazy Context Loading:** Only load context when needed
4. **Parallel Validation:** Run tests/security/browser checks simultaneously
5. **Batched Commits:** Combine multiple small changes into single commit

---

### Implementation Phases

**Phase 2B (Months 3-4, 8 weeks):**

**Weeks 1-2: Master-Servant Foundation**
- Implement Master agent orchestrator
- Task decomposition with GNN dependency analysis
- File access registry (in-memory)
- Basic IntentToModify approval/denial
- ✅ Milestone: Master can decompose tasks and assign to 3 servants

**Weeks 3-4: A2A Protocol**
- Implement 5 core message types (IntentToModify, ChangeCompleted, QueryDependency, DependencyResponse, ConflictNegotiation)
- WebSocket server for A2A communication
- Message routing and broadcasting
- Peer-to-peer dependency queries
- ✅ Milestone: Servants can communicate directly without Master

**Weeks 5-6: Hybrid Vector DB + GNN**
- Integrate ChromaDB for semantic search
- Embed functions, classes, comments, docs
- Real-time Vector DB updates on file save
- Periodic GNN sync (every 2s)
- Combined queries (Vector DB for semantics, GNN for structure)
- ✅ Milestone: Agents can find code semantically and validate dependencies structurally

**Weeks 7-8: Confidence-Based LLM Routing + Git Coordination**
- Implement RoutingMode enum (Adaptive, AlwaysBest, TaskBased)
- Confidence score calculation with historical success tracking
- Automatic LLM escalation on low confidence
- Centralized Git commit authority (Master only)
- Risk assessment for auto-commit vs manual review
- ✅ Milestone: System can auto-escalate LLMs and coordinate Git commits safely

**Week 9: Integration Testing**
- Multi-agent coordination tests
- Conflict prevention scenarios
- Performance benchmarks (3 vs 10 agents)
- Chaos engineering (agent failures, network issues)
- ✅ Milestone: All integration tests passing

**Week 10: Beta Release**
- Deploy to 10 beta users with Professional plan (3 agents)
- Collect feedback on conflict handling
- Measure performance gains vs single agent
- ✅ Milestone: Beta users successfully using 3 agents concurrently

---

### Success Metrics

**Technical Metrics:**
- ✅ 3 agents can work on different modules simultaneously without conflicts
- ✅ Conflict detection latency <10ms
- ✅ 95%+ of conflicts prevented proactively (not resolved reactively)
- ✅ Context build time <1s for 50k LOC codebase
- ✅ Vector DB search <50ms
- ✅ GNN incremental update <100ms per file

**Business Metrics:**
- ✅ 30% faster feature delivery with 3 agents vs 1 agent
- ✅ 10+ beta users on Professional plan (3 agents)
- ✅ <5% conflict-related errors (false positives)
- ✅ NPS >50 from beta users

**User Experience Metrics:**
- ✅ "Agents feel coordinated, not chaotic"
- ✅ "I can trust multiple agents working simultaneously"
- ✅ "Rare conflicts are resolved quickly and transparently"

---

## Phase 3: Enterprise Automation (Months 5-8)

### Objectives

Transform into enterprise workflow automation platform:

* Cross-system dependency tracking (internal + external APIs)
* Browser automation for legacy systems
* Self-healing workflows
* Multi-language support (Python + JavaScript)
* **Enterprise features: Multitenancy, user accounts, team collaboration**

### New Capabilities

1. **Cross-System Intelligence**

* Automatic discovery of external API calls
* Schema tracking for Stripe, Salesforce, etc.
* Breaking change detection (API version updates)
* End-to-end data flow validation
* Impact analysis (what breaks if X changes?)

2. **Browser Automation**

* Full Playwright integration
* DOM interaction (click, fill, extract data)
* Authentication handling
* Visual regression detection

3. **Self-Healing Systems**

* Continuous API monitoring (every 24h)
* Schema drift detection
* Automatic migration code generation
* Canary testing in sandbox
* Auto-deploy if tests pass

4. **Multi-Language Support**

* JavaScript/TypeScript parser
* Cross-language dependencies (Python API → React frontend)
* Node.js + React code generation
* Context mixing across languages

5. **Enterprise Features (Post-MVP)**

**Multitenancy:**
* Tenant isolation (database, GNN, patterns)
* Per-tenant configuration
* Shared failure patterns (cross-tenant, privacy-preserved)
* Resource quotas and limits

**User Accounts & Authentication:**
* User registration and login (OAuth, SSO)
* Role-based access control (RBAC)
* Team workspaces
* Project sharing and permissions
* Audit logs

**Team Collaboration:**
* Shared projects and codebases
* Activity feeds (who generated what)
* Code review workflows
* Comment threads on generated code
* Team pattern libraries (private)

**Billing & Subscription:**
* Usage tracking (LLM calls, tokens)
* Subscription tiers (Free, Pro, Team, Enterprise)
* Payment integration (Stripe)
* Usage analytics and reporting

### Implementation (16 Weeks)

Weeks 17-20: External API discovery and tracking 
Weeks 21-24: Browser automation (Playwright) 
Weeks 25-28: Self-healing engine 
Weeks 29-32: Multi-language support
Weeks 33-36: **Enterprise features (multitenancy, user accounts, team collaboration)**

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

## Go-to-Market Strategy

### Year 1: Developer Adoption (Free)

Strategy: Build massive user base through free access

Pricing:

* 100% Free for Year 1
* No credit card required
* Full feature access
* No usage limits

Rationale:

* Prove value before monetizing
* Build network effects
* Generate word-of-mouth
* Collect usage data to improve product
* Hook developers early

Target:

* Individual developers
* Small teams (1-10 developers)
* Early adopters and innovators
* Open source projects

Acquisition Channels:

* Product Hunt launch
* Hacker News discussions
* Dev.to and Medium articles
* YouTube demos
* GitHub showcases
* Developer conferences (talks, booths)

Success Metrics (Year 1):

* 10,000+ active users by Month 6
* 50,000+ active users by Month 12
* 80%+ retention rate
* NPS >50
* 10,000 projects created
* 1M lines of code generated

### Year 2: Freemium Transition

Strategy: Introduce paid tiers while keeping generous free tier

Pricing Tiers:

Free (Forever):

* Individual developers
* Up to 3 projects
* 100 LLM generations/month
* Community support
* Basic features

Pro ($29/month):

* Unlimited projects
* Unlimited LLM generations
* Priority LLM access (faster responses)
* Advanced features (refactoring, performance optimization)
* Email support

Team ($79/user/month):

* Everything in Pro
* Team collaboration features
* Shared dependency graphs
* Workflow automation (10 workflows)
* Admin controls
* Priority support

Enterprise (Custom pricing):

* Everything in Team
* Unlimited workflows
* On-premise deployment
* Custom model training
* SLA guarantees (99.9% uptime)
* 24/7 dedicated support
* Professional services (onboarding, training)

Target Conversion:

* 5-10% of free users to Pro ($29/mo)
* 20% of teams to Team tier ($79/user/mo)
* 50+ Enterprise customers by EOY2

Revenue Projection (Year 2):

* 50,000 users (from Year 1)
* 2,500 Pro users @ $29/mo = $72,500/mo
* 200 Team users @ $79/mo = $15,800/mo
* 50 Enterprise @ $5k/mo avg = $250,000/mo
* Total: ~$4M ARR by end of Year 2

### Year 3: Platform Play

Strategy: Expand to workflow automation market, compete with Zapier

New Revenue Streams:

* Marketplace (plugins, templates) - 30% revenue share
* Partner ecosystem (consultants) - certification programs
* Industry-specific solutions (fintech, healthcare)
* Professional services (custom workflows)

Target:

* Large enterprises (1000+ developers)
* Operations teams (workflow automation)
* Business analysts (no-code users)

Revenue Projection (Year 3):

* $15-20M ARR

---

## Appendices

### A. Development Guidelines

Code Quality Standards:

* Rust: Clippy pedantic, 80%+ test coverage, no panics in production
* Frontend: ESLint strict, Prettier formatting, TypeScript strict mode
* Generated Python: PEP 8, type hints, docstrings, error handling

Git Workflow:

* Branches: main (production), develop (integration), feature/* (features)
* Commits: Conventional Commits format
* PRs: Required reviews, CI must pass

Testing Strategy:

* Unit tests: All core logic
* Integration tests: End-to-end flows
* Performance tests: Benchmark GNN operations
* Manual testing: Weekly on all platforms

### B. Tech Stack Rationale

Why Tauri over Electron?

* 600KB vs 150MB bundle size
* Lower memory footprint (100MB vs 400MB)
* Rust backend ideal for GNN performance
* Native OS integrations

Why SolidJS over React?

* Fastest reactive framework (benchmark leader)
* Smaller bundle size
* No virtual DOM overhead
* Better TypeScript support

Why Rust for GNN?

* Memory safety without garbage collection
* Fearless concurrency (Tokio async)
* Zero-cost abstractions
* Fast graph operations (petgraph)
* Easy to parallelize

Why Multi-LLM?

* No single point of failure
* Quality improvement through consensus
* Cost optimization (route by complexity)
* Best-of-breed approach

### C. Performance Targets

MVP Targets:

* GNN graph build: <5s for 10k LOC project
* GNN incremental update: <50ms per file change
* Dependency lookup: <10ms
* Context assembly: <100ms
* Code generation: <3s (LLM dependent)
* Test execution: <30s for typical project
* Security scan: <10s
* Browser validation: <5s
* Total cycle (intent → commit): <2 minutes

Scale Targets (Month 9+):

* GNN graph build: <30s for 100k LOC project
* GNN query: <100ms for 100k LOC
* Support 1M LOC projects

### D. Security & Privacy

Data Handling:

* User code never leaves machine unless explicitly sent to LLM APIs
* LLM calls encrypted in transit (HTTPS)
* No code storage on Yantra servers (local only)
* Crash reports: Anonymous, opt-in
* Analytics: Usage only, no PII, opt-in

LLM Privacy:

* Option to use local LLM (post-MVP, Phase 2+)
* Mark sensitive files (never send to cloud LLM)
* Audit log (what was sent to cloud)
* Data retention: LLM providers' policies (typically 30 days, then deleted)

Enterprise Privacy:

* On-premise deployment (air-gapped)
* BYO LLM (use your own models)
* Encrypted at rest
* SOC2 compliance
* GDPR compliance

### E. Risk Mitigation

Technical Risks:

Risk: GNN accuracy <95% → Code still breaks Mitigation: Extensive testing, incremental rollout, fallback to manual validation

Risk: LLM hallucination → Generated code has bugs Mitigation: Multi-LLM consensus, mandatory testing, human review option

Risk: Performance degradation at scale Mitigation: Benchmarking, profiling, distributed architecture ready

Business Risks:

Risk: Low user adoption Mitigation: Free Year 1, aggressive marketing, focus on developer experience

Risk: LLM API costs too high Mitigation: Caching, smart routing, local LLM option (Phase 2+)

Risk: Competitors copy approach Mitigation: Speed of execution, network effects, proprietary GNN IP

### F. Success Criteria Summary

Month 2 (MVP):

* ✅ 20 beta users successfully generating code
* ✅ >90% of generated code passes tests
* ✅ NPS >40

Month 6:

* ✅ 10,000 active users
* ✅ >95% code success rate
* ✅ 50%+ user retention

Month 12:

* ✅ 50,000 active users
* ✅ Workflow automation live (Phase 2)
* ✅ 80%+ retention

Month 18:

* ✅ Freemium launch
* ✅ $500k ARR
* ✅ 100+ paying customers

Month 24:

* ✅ $4M ARR
* ✅ 2,500+ Pro users
* ✅ 50+ Enterprise customers

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

git clone https://github.com/cogumi/yantra.git

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

├── src/                    # Rust backend

│   ├── main.rs            # Tauri entry point

│   ├── gnn/               # Graph Neural Network

│   ├── llm/               # LLM orchestration

│   ├── testing/           # Test generation & execution

│   ├── security/          # Security scanning

│   └── git/               # Git integration

├── src-ui/                # Frontend (SolidJS)

│   ├── components/        # UI components

│   ├── stores/            # State management

│   └── App.tsx            # Main app

├── skills/                # Skill templates (future)

└── docs/                  # Documentation

### Development Workflow

1. Create feature branch: git checkout -b feature/your-feature
2. Make changes
3. Run tests: cargo test && cd src-ui && npm test
4. Run linters: cargo clippy && npm run lint
5. Commit: git commit -m "feat: your feature"
6. Push and create PR

### Testing

# Run all Rust tests

cargo test

# Run frontend tests

cd src-ui && npm test

# Run integration tests

cargo test --test integration

# Run with coverage

cargo tarpaulin --out Html

---

## Contact & Support

Project Maintainer: Vivek (Cogumi)

Repository: https://github.com/cogumi/yantra (placeholder)

Documentation: https://docs.yantra.dev (placeholder)

Community: Discord server (placeholder)

Enterprise Sales: [Placeholder]

---

End of Specification Document

This document is a living specification and will be updated as the project evolves.

**
