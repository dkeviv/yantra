**# Yantra: Complete Technical Specification

Version: 1.0
Date: November 2024
Document Purpose: Complete technical blueprint for building Yantra from ground zero to enterprise platform

---

## Executive Summary

### The Vision

Yantra is an AI-first development platform that generates production-quality code with a revolutionary guarantee: code that never breaks.

Unlike traditional IDEs that assist developers or AI tools that suggest code, Yantra makes artificial intelligence the primary developer, with humans providing intent and oversight.

### The Problem We Solve

For Developers:

* 40-60% of development time spent debugging
* Code breaks production despite passing tests
* Integration failures when APIs change
* Repetitive coding tasks (CRUD, auth, APIs)

For Engineering Teams:

* Unpredictable delivery timelines
* Inconsistent code quality
* High maintenance costs
* Technical debt accumulation

For Enterprises:

* Manual workflow automation (expensive, error-prone)
* Siloed systems (Slack, Salesforce, internal tools don't talk)
* Workflow tools (Zapier) can't access internal code
* System breaks cascade across services

### The Solution

Phase 1 (Months 1-2): Code That Never Breaks

* AI generates code with full dependency awareness
* Automated unit + integration testing
* Security vulnerability scanning
* Browser runtime validation
* Git integration for seamless commits

Phase 2 (Months 3-4): Workflow Automation

* Generate workflows from natural language
* Scheduled jobs and event triggers
* Multi-step orchestration
* Error handling and retries

Phase 3 (Months 5-8): Enterprise Platform

* Cross-system dependency tracking
* External API monitoring and auto-healing
* Browser automation for legacy systems
* Multi-language support (Python + JavaScript)

Phase 4 (Months 9-12): Platform Maturity

* Plugin ecosystem and marketplace
* Advanced refactoring and performance optimization
* Enterprise deployment (on-premise, cloud)
* SLA guarantees (99.9% uptime)

### Market Opportunity

Primary Market: Developer Tools ($50B+)

* IDEs, testing tools, CI/CD platforms
* Target: Mid-market to enterprise (10-1000+ developers)

Secondary Market: Workflow Automation ($10B+)

* Replace/augment Zapier, Make, Workato
* Target: Operations teams, business analysts

Total Addressable Market: $60B+

### Competitive Advantage

| Capability                  | Yantra | Copilot | Cursor | Zapier |
| --------------------------- | ------ | ------- | ------ | ------ |
| Dependency-aware generation | ✅     | ❌      | ❌     | N/A    |
| Guaranteed no breaks        | ✅     | ❌      | ❌     | ❌     |
| Truly unlimited context     | ✅     | ❌      | ❌     | N/A    |
| Token-aware context         | ✅     | ⚠️    | ⚠️   | N/A    |
| Automated testing           | ✅     | ❌      | ❌     | ❌     |
| Agentic validation pipeline | ✅     | ❌      | ❌     | ❌     |
| Self-healing systems        | ✅     | ❌      | ❌     | ❌     |
| Network effect (failures)   | ✅     | ❌      | ❌     | ❌     |
| Works with any LLM          | ✅     | ❌      | ⚠️   | N/A    |
| Internal system access      | ✅     | ⚠️    | ⚠️   | ❌     |
| Custom workflow code        | ✅     | ❌      | ❌     | ❌     |
| Browser automation          | ✅     | ❌      | ❌     | ❌     |

**Key Differentiators:**

1. **Truly Unlimited Context**: Not limited by LLM context windows through intelligent compression, chunking, and hierarchical assembly
2. **Agentic Architecture**: Fully autonomous validation pipeline with confidence scoring and auto-retry loops
3. **Network Effect from Failures**: Shared failure patterns (privacy-preserving) create collective intelligence that improves with every user
4. **LLM Agnostic**: Works with any LLM (Claude, GPT-4, Qwen Coder) through context enhancement, not LLM-specific features

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

## Core Innovation: Fully Agentic Architecture

### What "Fully Agentic" Means

**Not agentic:** LLM generates code → User tests it → User fixes issues

**Fully agentic:** LLM generates code → System validates → System fixes issues → System commits → Repeat until perfect

**Yantra is autonomous:** Human provides intent, AI handles entire implementation cycle.

### Agent State Machine

```
┌─────────────────────────────────────────────────────┐
│                  AGENTIC PIPELINE                    │
└─────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  User Intent │ (Natural language task)
    └───────┬──────┘
            │
            ▼
    ┌──────────────────────┐
    │ Code Generation      │ (LLM + GNN context)
    │ Confidence: 0.0-1.0  │
    └───────┬──────────────┘
            │
            ▼
    ┌──────────────────────┐
    │ Validation Pipeline  │
    ├──────────────────────┤
    │ 1. Dependency Check  │ → GNN: No breaking changes?
    │ 2. Unit Tests        │ → pytest: All pass?
    │ 3. Integration Tests │ → pytest: E2E works?
    │ 4. Security Scan     │ → Semgrep: No vulnerabilities?
    │ 5. Browser Test      │ → CDP: No runtime errors?
    └───────┬──────────────┘
            │
            ├─── ALL PASS ───┐
            │                 ▼
            │         ┌──────────────┐
            │         │ Git Commit   │
            │         │ + Push       │
            │         └──────────────┘
            │                 │
            │                 ▼
            │         ┌──────────────┐
            │         │   SUCCESS    │
            │         └──────────────┘
            │
            └─── ANY FAIL ───┐
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
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌──────────────┐          ┌──────────────┐
        │ Known Fix    │          │ Novel Error  │
        │ Conf: >0.8   │          │ Conf: <0.5   │
        └───────┬──────┘          └───────┬──────┘
                │                         │
                │ Auto-retry              │ Escalate to user
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ Apply Fix    │          │ Human Review │
        │ + Re-test    │          │ + Learn      │
        └───────┬──────┘          └───────┬──────┘
                │                         │
                └─────────┬───────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Update Known DB    │
                │ (Network Effect)   │
                └────────────────────┘
```

### Confidence Scoring System

**Purpose:** Determine if agent should auto-retry or escalate to human

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
