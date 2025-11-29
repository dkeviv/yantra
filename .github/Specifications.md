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

## Yantra Codex: AI Pair Programming Engine (DEFAULT MODE)

### Overview

Yantra Codex is a **hybrid AI pair programming system** that combines a specialized Graph Neural Network (GNN) with premium LLMs (Claude/ChatGPT) to generate production-quality code. This pair programming approach delivers the best of both worlds: GNN speed and learning + LLM reasoning and quality.

**Core Innovation**: 
- **Yantra Codex (GNN)**: Fast, local, learning-focused (15ms, FREE)
- **Premium LLM**: Review, enhance, handle edge cases (user's choice: Claude/ChatGPT)
- **Continuous Learning**: Yantra learns from LLM fixes → reduces cost over time

**Pair Programming Roles:**
- **Yantra Codex (Junior Partner)**: Generates initial code, handles common patterns, learns continuously
- **LLM (Senior Partner)**: Reviews edge cases, adds error handling, teaches Yantra implicitly

**Key Principles:**
1. **Hybrid Intelligence**: GNN speed + LLM reasoning = superior quality
2. **Cost Optimization**: 90% cost reduction (Yantra handles most, LLM reviews selectively)
3. **Continuous Learning**: Yantra learns from LLM fixes → 96% cost reduction after 12 months
4. **User Choice**: Configure Claude Sonnet 4, GPT-4, or other premium LLMs

---

### Yantra Codex Architecture

#### 1. Model Specifications

**GraphSAGE Neural Network (1024-dim embeddings):**
```
Input: 978-dimensional problem features
Layers: 978 → 1536 → 1280 → 1024
Parameters: ~150M
Model Size: ~600 MB
Inference: 15ms (CPU), 5ms (GPU)
```

**Why 1024 dimensions:**
- Sufficient capacity for multi-step logic patterns
- 55-60% initial accuracy (vs 40% with 256 dims)
- Fast inference (still feels instant)
- Room to scale to 2048+ dims later

#### 2. Pair Programming Workflow (Default Mode)

**Step 1: Yantra Codex Generates**
```
User Request: "Create REST API endpoint to get user by ID"
        ↓
┌───────────────────────────────────────────┐
│  Yantra Codex Generates                   │
│  - Extract 978-dim features               │
│  - GNN predicts logic pattern (15ms)     │
│  - Tree-sitter generates code             │
│  - Calculate confidence score (0.0-1.0)   │
└───────────────────────────────────────────┘
        ↓
    Confidence >= 0.8?
```

**Step 2: LLM Review (if confidence < 0.8)**
```
┌───────────────────────────────────────────┐
│  LLM Review & Enhancement                 │
│  - Send: Yantra code + confidence issues │
│  - LLM reviews edge cases                │
│  - LLM adds error handling               │
│  - LLM improves code quality             │
│  - User's choice: Claude/GPT-4           │
└───────────────────────────────────────────┘
```

**Step 3: Merge & Validate**
```
┌───────────────────────────────────────────┐
│  Merge & Validate                         │
│  - Merge Yantra + LLM suggestions        │
│  - Run GNN dependency validation         │
│  - Run automated tests                   │
│  - User reviews final code               │
└───────────────────────────────────────────┘
```

**Step 4: Yantra Learns**
```
┌───────────────────────────────────────────┐
│  Yantra Learns from LLM                   │
│  - Extract logic pattern from final code │
│  - Store: problem → LLM-enhanced logic   │
│  - Incremental GNN update                │
│  - Next time: Yantra will know this!     │
└───────────────────────────────────────────┘
```

**Confidence-Based Routing:**

| Confidence | Routing Decision | Rationale | Cost |
|------------|------------------|-----------|------|
| **0.9-1.0** | Yantra alone | Seen pattern many times | $0 |
| **0.8-0.9** | Yantra alone | Good confidence, validate with tests | $0 |
| **0.5-0.8** | Yantra + LLM review | Partial knowledge, need LLM help | ~$0.015 |
| **0.0-0.5** | LLM alone | Novel pattern, Yantra can't help yet | ~$0.025 |

**Learning Trajectory Example:**

```
Week 1 (CRUD endpoint pattern):
  Request 1:  Yantra 0.3 → LLM review → Tests pass → Yantra learns
  Request 10: Yantra 0.5 → LLM review → Tests pass → Yantra learns
  Request 50: Yantra 0.75 → LLM review → Tests pass → Yantra learns
  
Week 4 (same pattern):
  Request 200: Yantra 0.88 → No LLM needed! → Tests pass
  
Cost Trajectory: $0.015 → $0.010 → $0.005 → $0 (100% saved)
```

#### 3. Cost & Quality Benefits

**Cost Trajectory (vs LLM-only baseline $25/1000 generations):**
- **Month 1:** $9/1000 gen (64% savings) - Yantra handles 55% alone
- **Month 6:** $3/1000 gen (88% savings) - Yantra handles 85% alone
- **Year 1:** $1/1000 gen (96% savings) - Yantra handles 95% alone

**Quality Guarantee:** Yantra + LLM ≥ LLM alone (pair programming is better!)

**Comparison Table:**

| Metric | LLM Only | Yantra + LLM (Month 1) | Yantra + LLM (Year 1) |
|--------|----------|------------------------|----------------------|
| Cost/1000 gen | $25 | $9 (64% ↓) | $1 (96% ↓) |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | 3-5s | 0.5-2s | 0.015-0.5s |
| Learning | ❌ | ✅ | ✅✅✅ |
| Privacy | ❌ (cloud) | ✅ (mostly local) | ✅ (95% local) |

#### 4. Multi-Language Support

**Universal Logic Patterns (Learned Once):**
- Input validation, error handling, data transformation
- API calls, database operations, async patterns
- Algorithm patterns, architecture patterns

**Language-Specific Syntax (Tree-sitter Provides):**
- Keywords and operators, type systems, standard library
- Language idioms, formatting rules

**Transfer Learning:**
```
Learn "retry with exponential backoff" in Python (1,000 examples)
↓
Automatically works in JavaScript, Rust, Go, etc. (zero additional training)
↓
Tree-sitter handles syntax differences
```

**Supported Languages:**
- Python ✅, JavaScript ✅, TypeScript ✅
- Rust, Go, Java, C++, etc. (easy to add - ~50 lines per language)

#### 5. Continuous Learning System

**What Yantra Learns:**
1. **Edge Cases**: LLM adds null checks → Yantra learns to add them
2. **Error Handling**: LLM adds try-catch → Yantra learns pattern
3. **Best Practices**: LLM improves naming → Yantra learns conventions
4. **Domain Patterns**: LLM adds auth checks → Yantra learns domain rules

**Learning Metrics:**

```
Month 1:  Yantra handles 55% alone, LLM needed 45% → Cost: $9/1000 gen
Month 3:  Yantra handles 70% alone, LLM needed 30% → Cost: $5/1000 gen  
Month 6:  Yantra handles 85% alone, LLM needed 15% → Cost: $3/1000 gen
Month 12: Yantra handles 95% alone, LLM needed 5% → Cost: $1/1000 gen

Cost Reduction: 96% after 1 year!
```

#### 6. Yantra Cloud Codex (Optional, Opt-in)

**Privacy-Preserving Collective Learning:**

**What Gets Shared:**
- ✅ Logic pattern embeddings (numbers only)
- ✅ Pattern success metrics
- ✅ Anonymized complexity data

**What Does NOT Get Shared:**
- ❌ Actual code
- ❌ Variable/function names
- ❌ Business logic details
- ❌ User identity
- ❌ Project structure

**Network Effects:**
```
100 users × 50 requests/day = 150k patterns/month → Model v1.1 (65% accuracy)
1k users × 50 requests/day = 1.5M patterns/month → Model v1.6 (80% accuracy)
10k users × 50 requests/day = 15M patterns/month → Model v2.0 (90% accuracy)

More users = Better model = Lower LLM costs = Attracts more users (flywheel)
```

#### 7. Accuracy Targets

**Month 1:** 55-60% Yantra alone, 95%+ with LLM review (64% cost savings)
**Month 6:** 75-80% Yantra alone, 98%+ with LLM review (88% cost savings)
**Year 2:** 85%+ Yantra alone, 99%+ with LLM review (92% cost savings)
**Year 3+:** 90-95% Yantra alone, 99.5%+ with LLM review (96% cost savings)

#### 8. Implementation Components

**Core Files:**
- `src-python/model/graphsage.py` - 1024-dim GNN model
- `src-tauri/src/codex/generator.rs` - Pair programming orchestrator
- `src-tauri/src/codex/confidence.rs` - Confidence scoring
- `src-tauri/src/codex/llm_reviewer.rs` - LLM review & enhancement
- `src-tauri/src/codex/learner.rs` - Continuous learning system

**Training Pipeline:**
- `scripts/extract_logic_patterns.py` - Extract logic patterns from CodeContests
- `scripts/train_yantra_codex.py` - Train GNN on problem → logic mapping
- `src-python/learning/incremental_learner.py` - Learn from LLM fixes

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

## Architecture View System (MVP Phase 1 - Priority Feature)

### Overview

**Status:** 🔴 NOT STARTED  
**Priority:** ⚡ MVP REQUIRED (Implement before Pair Programming)  
**Specification:** 997 lines of detailed requirements  
**Business Impact:** Design-first development, architecture governance  
**User Request:** "Where is the visualization of architecture flow?"

A comprehensive architecture visualization and governance system that enables **design-first development**, automatic architecture generation from existing code, and bidirectional sync between conceptual architecture and implementation.

**Key Principle:** Architecture is automatically generated and continuously monitored. Code must align with architecture.

**Three Core Capabilities:**
1. **Automatic Architecture Generation** - User provides specs/intent → Agent generates architecture → Agent generates code → Agent monitors for deviations
2. **Deviation Detection During Implementation** - Agent generates code → Checks alignment → Alerts user if deviation → User decides (update arch or fix code)
3. **Continuous Alignment Monitoring** - Code changes → Compare to architecture → Alert if misaligned → Enforce governance through user decision

---

### 🤖 Agent-Driven Architecture (Autonomous Mode)

**CRITICAL DESIGN PRINCIPLE:** This is an **agentic platform** - all architecture operations happen through the agent, not manual UI interactions.

#### Interaction Model

**❌ NOT THIS (Manual Mode):**
```
User clicks "Create Architecture" button
→ User drags components
→ User draws connections
→ User clicks "Save"
→ Manual diagram creation
```

**✅ THIS (Agent-Driven Mode):**
```
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
```

#### Auto-Save with Rule of 3 Versioning

**Specification:** Every architecture change is automatically saved with version history following the Rule of 3.

**Rule of 3 Implementation:**
- Keep current version + 3 most recent past versions (total: 4 versions)
- When 5th version is created, delete the oldest (version 1)
- Versions are immutable once created
- Agent can revert to any of the 3 past versions

**Version Metadata:**
```rust
struct ArchitectureVersion {
    version_number: u32,           // Incremental: 1, 2, 3, 4...
    snapshot_json: String,         // Full architecture state
    timestamp: DateTime,           // When created
    change_type: ChangeType,       // AgentGenerated, AgentUpdated, AgentReverted
    agent_reasoning: String,       // Why this change was made
    user_intent: String,           // Original user message that triggered change
}

enum ChangeType {
    AgentGenerated,    // Agent created new architecture
    AgentUpdated,      // Agent modified existing architecture
    AgentReverted,     // Agent reverted to older version
    GNNSynced,         // Synced from code analysis
}
```

**Storage:**
```sql
-- Only keep 4 versions (current + 3 past)
-- Auto-delete oldest when creating 5th version
CREATE TABLE architecture_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_number INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_type TEXT NOT NULL,  -- 'agent_generated', 'agent_updated', 'agent_reverted', 'gnn_synced'
    agent_reasoning TEXT,
    user_intent TEXT,
    -- Constraint: Only keep 4 versions
    CHECK (version_number IN (
        SELECT version_number FROM architecture_versions 
        ORDER BY version_number DESC 
        LIMIT 4
    ))
);
```

**Agent Commands (Via Chat):**
```
User: "Show me the architecture"
→ Agent: Opens Architecture View tab, shows current version

User: "Revert to previous architecture"
→ Agent: Loads version N-1, auto-saves as new version N+1
→ Agent: "Reverted to version 5 (from 2 minutes ago)"

User: "Show architecture history"
→ Agent: Lists last 3 versions with timestamps and changes

User: "Why did you add Redis?"
→ Agent: Shows version history and reasoning
```

#### Architecture View UI (Read-Only Visualization)

**No Manual Controls:**
- ❌ No "Create Architecture" button
- ❌ No "Add Component" button
- ❌ No "Save" button
- ❌ No "Load" button
- ❌ No drag-to-create connections
- ❌ No manual component editing

**Read-Only Features:**
- ✅ Zoom and pan navigation
- ✅ Click component to see details (files, status)
- ✅ Click connection to see relationship type
- ✅ Filter by component type (Frontend/Backend/Database)
- ✅ Version history display (auto-updated)
- ✅ Export view (Markdown/Mermaid/JSON) - via agent command

**Empty State Message:**
```
┌─────────────────────────────────────────────┐
│                                             │
│              🏗️ No Architecture Yet          │
│                                             │
│  Tell me in chat what you want to build,   │
│  and I'll generate the architecture for    │
│  you automatically.                         │
│                                             │
│  Example: "Create a REST API with auth"    │
│                                             │
└─────────────────────────────────────────────┘
```

#### Agent Workflow Examples

**Example 1: Create New Architecture**
```
User: "Build a 3-tier web app with React, FastAPI, PostgreSQL"

Agent thinks:
1. Parse intent: 3-tier, React frontend, FastAPI backend, PostgreSQL
2. Generate architecture:
   - Component: React Frontend (type: UIComponent, layer: frontend)
   - Component: FastAPI Backend (type: Service, layer: backend)
   - Component: PostgreSQL (type: Database, layer: database)
   - Connection: Frontend → Backend (APICall)
   - Connection: Backend → Database (DataFlow)
3. Auto-save to database (version 1)
4. Export to architecture.json and architecture.md
5. Notify user

Agent responds: "✅ Created 3-tier architecture with 3 components and 2 connections. 
Switch to Architecture View tab to see the diagram."

Architecture View: Auto-updates to show new diagram
```

**Example 2: Update Existing Architecture**
```
User: "Add Redis caching between FastAPI and PostgreSQL"

Agent thinks:
1. Load current architecture (version 3)
2. Analyze: Need Redis component + connections
3. Generate update:
   - Add Component: Redis (type: Cache, layer: backend)
   - Add Connection: FastAPI → Redis (DataFlow)
   - Add Connection: Redis → PostgreSQL (DataFlow)
4. Auto-save (version 4, deletes version 1 per Rule of 3)
5. Store reasoning: "Added Redis for caching per user request"
6. Export updated diagram

Agent responds: "✅ Added Redis caching layer. Architecture updated to version 4."

Architecture View: Auto-updates to show Redis node and connections
```

**Example 3: Revert Architecture**
```
User: "Actually, remove Redis. Go back to the previous version."

Agent thinks:
1. Load version history (versions 2, 3, 4)
2. User wants version 3 (before Redis was added)
3. Create version 5 as copy of version 3
4. Auto-save (version 5, deletes version 2 per Rule of 3)
5. Store reasoning: "Reverted to version 3 per user request"

Agent responds: "✅ Reverted to version 3 (without Redis). Now at version 5."

Architecture View: Updates to show version 3 state (no Redis)
```

---

### 🔍 Automatic Architecture Generation & Deviation Detection

**CRITICAL WORKFLOW:** Yantra automatically generates architecture from specifications or code generation, then monitors for deviations during implementation.

#### Workflow 1: Architecture from Specifications

**Trigger:** User provides specifications or high-level requirements in chat

```
User: "Build a REST API with JWT authentication, user management, and PostgreSQL"

Agent thinks:
1. Parse requirements → Extract components:
   - API Gateway (handles routing)
   - Auth Service (JWT tokens, login/logout)
   - User Service (CRUD operations)
   - PostgreSQL Database (data persistence)

2. Infer connections:
   - Client → API Gateway (HTTPS)
   - API Gateway → Auth Service (verify JWT)
   - API Gateway → User Service (route requests)
   - Auth Service → Database (store tokens)
   - User Service → Database (store users)

3. Generate architecture diagram
4. Auto-save (version 1)
5. Show in Architecture View

Agent responds: "✅ Generated architecture with 4 components. Review in Architecture View tab."
```

**Architecture becomes the source of truth for implementation.**

#### Workflow 2: Deviation Detection During Code Generation

**Scenario:** Agent generates code that deviates from planned architecture

```
Architecture says:
  API Gateway → Auth Service → Database
  (3-tier: Gateway can't directly access Database)

Agent generates code:
  api/gateway.py:
    import psycopg2
    conn = psycopg2.connect(...)  # DIRECT DB ACCESS!

Deviation Detector triggers:
1. GNN analyzes new code in gateway.py
2. Detects: gateway.py imports psycopg2 (database library)
3. Checks architecture: Gateway should only call Auth Service
4. Identifies: ARCHITECTURAL DEVIATION
5. Calculates severity: HIGH (violates layering principle)
6. Pauses code generation
7. Alerts user with options

Agent responds: "⚠️ ARCHITECTURE DEVIATION DETECTED

I was about to add direct database access to API Gateway, but the architecture 
specifies that Gateway should only communicate with Auth Service.

Current Architecture:
  Gateway → Auth Service → Database ✓

My Code Would Create:
  Gateway → Database (direct access) ❌

This violates the 3-tier architecture principle.

What would you like me to do?

1️⃣ Update Architecture - Allow Gateway to access Database directly (simpler, but less maintainable)
2️⃣ Fix Code - Keep architecture, generate code that uses Auth Service (maintains separation)
3️⃣ Cancel - Stop code generation, let me review

Your choice (1/2/3): "

User: "2"

Agent: "✅ Fixing code to match architecture. Generating Auth Service client in Gateway..."
```

#### Workflow 3: Deviation Detection After Code Generation

**Scenario:** User manually edits code, breaking architecture alignment

```
User manually edits: src/gateway.py
  - Adds: from database import query_users
  - Calls database directly, bypassing User Service

On Save:
1. GNN detects new import: database.query_users
2. File change event → Architecture Validator runs
3. Check: gateway.py should only import user_service
4. Detect: ARCHITECTURAL MISALIGNMENT
5. Show warning in UI + chat

Agent: "⚠️ Code-Architecture Misalignment Detected

File: src/gateway.py
Change: Added direct database import

Architecture expects:
  Gateway → User Service → Database

Code now has:
  Gateway → Database (direct)

This breaks the service layer pattern.

Options:
1️⃣ Update Architecture - Remove User Service layer (architectural change)
2️⃣ Revert Code - Undo your changes to gateway.py
3️⃣ Refactor Code - Move database logic to User Service (maintain architecture)

Recommended: Option 3 (maintain clean architecture)

Your choice (1/2/3): "
```

#### Implementation Architecture

**Deviation Detection System:**

```rust
// src-tauri/src/architecture/deviation_detector.rs

pub struct DeviationDetector {
    gnn_engine: Arc<Mutex<GNNEngine>>,
    architecture_manager: ArchitectureManager,
}

impl DeviationDetector {
    /// Check if new/modified code aligns with architecture
    pub async fn check_code_alignment(
        &self,
        file_path: &Path,
        architecture_id: &str,
    ) -> Result<AlignmentResult, String> {
        // 1. Get current architecture
        let arch = self.architecture_manager.load(architecture_id)?;
        
        // 2. Find which component owns this file
        let component = arch.find_component_by_file(file_path)?;
        
        // 3. Get GNN dependencies for this file
        let actual_deps = self.gnn_engine.lock().unwrap()
            .get_file_dependencies(file_path)?;
        
        // 4. Get expected dependencies from architecture
        let expected_deps = arch.get_component_dependencies(&component.id)?;
        
        // 5. Compare actual vs expected
        let deviations = self.compare_dependencies(actual_deps, expected_deps)?;
        
        // 6. Calculate severity
        let severity = self.calculate_severity(&deviations);
        
        Ok(AlignmentResult {
            is_aligned: deviations.is_empty(),
            deviations,
            severity,
            recommendations: self.generate_recommendations(&deviations),
        })
    }
    
    /// Monitor for deviations during code generation
    pub async fn monitor_code_generation(
        &self,
        generated_code: &str,
        target_file: &Path,
        architecture_id: &str,
    ) -> Result<DeviationCheck, String> {
        // 1. Parse generated code (tree-sitter)
        let imports = self.extract_imports(generated_code)?;
        
        // 2. Get expected dependencies from architecture
        let arch = self.architecture_manager.load(architecture_id)?;
        let component = arch.find_component_by_file(target_file)?;
        let allowed_deps = arch.get_allowed_dependencies(&component.id)?;
        
        // 3. Check for violations
        let violations = imports.iter()
            .filter(|imp| !allowed_deps.contains(imp))
            .collect::<Vec<_>>();
        
        if !violations.is_empty() {
            return Ok(DeviationCheck {
                has_deviation: true,
                violations,
                severity: self.calculate_severity_from_violations(&violations),
                pause_generation: true,
                user_prompt: self.generate_user_prompt(&violations, &component),
            });
        }
        
        Ok(DeviationCheck {
            has_deviation: false,
            violations: vec![],
            severity: Severity::None,
            pause_generation: false,
            user_prompt: None,
        })
    }
}

pub enum Severity {
    None,        // No deviation
    Low,         // Minor deviation (e.g., extra utility import)
    Medium,      // Moderate (e.g., skip one layer but maintain pattern)
    High,        // Major violation (e.g., break layering completely)
    Critical,    // Catastrophic (e.g., circular dependencies)
}

pub struct AlignmentResult {
    pub is_aligned: bool,
    pub deviations: Vec<Deviation>,
    pub severity: Severity,
    pub recommendations: Vec<String>,
}

pub struct Deviation {
    pub deviation_type: DeviationType,
    pub expected: String,
    pub actual: String,
    pub affected_file: PathBuf,
    pub explanation: String,
}

pub enum DeviationType {
    UnexpectedDependency,     // Code imports something not in architecture
    MissingDependency,        // Architecture expects import, code doesn't have it
    WrongConnectionType,      // Using wrong communication pattern
    LayerViolation,           // Bypassing layers
    CircularDependency,       // Creating cycle in directed graph
}
```

**Integration Points:**

1. **During Code Generation (project_orchestrator.rs):**
```rust
// Before writing generated code
let deviation_check = deviation_detector
    .monitor_code_generation(&generated_code, &target_file, &architecture_id)
    .await?;

if deviation_check.has_deviation {
    // Pause generation
    // Show user prompt
    // Wait for user decision
    match user_decision {
        Decision::UpdateArchitecture => {
            // Modify architecture to allow new dependency
            architecture_manager.add_connection(...)?;
        },
        Decision::FixCode => {
            // Regenerate code that matches architecture
            let fixed_code = llm_orchestrator.fix_architectural_violation(...)?;
        },
        Decision::Cancel => {
            return Err("Code generation cancelled by user");
        },
    }
}

// Proceed with writing file
```

2. **After File Save (file watcher):**
```rust
// When user manually edits file
on_file_save(file_path) {
    if let Some(architecture_id) = project.active_architecture_id {
        let result = deviation_detector
            .check_code_alignment(&file_path, &architecture_id)
            .await?;
        
        if !result.is_aligned {
            // Show warning in UI
            ui.show_deviation_warning(result);
            
            // Add message to chat
            chat.add_system_message(&format!(
                "⚠️ Architecture misalignment detected in {}\n\n{}",
                file_path.display(),
                result.format_user_friendly()
            ));
        }
    }
}
```

**User Decision Flow:**

```
┌──────────────────────────────────┐
│  Deviation Detected              │
├──────────────────────────────────┤
│  Expected: A → B → C             │
│  Actual:   A → C (skips B)       │
│                                  │
│  Severity: HIGH                  │
│                                  │
│  Options:                        │
│  1️⃣ Update Architecture          │
│  2️⃣ Fix Code                     │
│  3️⃣ Cancel                       │
└──────────────────────────────────┘
           ↓
    User chooses 1️⃣
           ↓
┌──────────────────────────────────┐
│  Architecture Updated            │
├──────────────────────────────────┤
│  Added: A → C connection         │
│  Reason: User approved shortcut  │
│  Version: 5 (was 4)              │
│                                  │
│  ✅ Code now matches arch        │
└──────────────────────────────────┘
```

#### Benefits

1. **Prevents Drift:** Architecture never diverges from code
2. **Enforces Governance:** Maintains architectural decisions
3. **Documents Decisions:** Every deviation has reasoning
4. **Enables Rollback:** Can revert to previous architecture if needed
5. **Teaches Best Practices:** Users learn architectural patterns

### 1. Core Workflows

#### Workflow 1: Design-First (New Project)
```
User: "Build a REST API with JWT authentication"
  ↓
AI generates architecture diagram with components:
  - API Gateway
  - Auth Service  
  - User Service
  - PostgreSQL Database
  ↓
User refines (adds/modifies components, connections)
  ↓
User approves architecture
  ↓
AI generates code implementing this exact architecture
  ↓
Files automatically linked to components
  ↓
Result: Code matches architecture by construction
```

#### Workflow 2: Import Existing Project
```
User imports GitHub repo (156 files)
  ↓
GNN analyzes codebase structure
  ↓
AI groups files into components:
  - Frontend UI: 32 files
  - Auth Service: 6 files
  - User Service: 15 files
  ↓
AI infers connections from imports
  ↓
User refines groupings
  ↓
Architecture becomes governance layer
  ↓
Result: Legacy project now has architectural documentation
```

#### Workflow 3: Continuous Governance
```
Developer modifies api/gateway.py (adds direct DB query)
  ↓
On save, GNN detects new dependency
  ↓
System compares to architecture:
  Expected: Gateway → Service → DB
  Actual: Gateway → DB (direct)
  ↓
⚠️ Alert: "Misalignment: You're bypassing the service layer"
  ↓
Options:
  (a) Update architecture to allow direct DB access
  (b) Revert code change
  (c) Refactor code to use Service layer
  ↓
User decides → System enforces choice
  ↓
Result: Architecture never drifts from reality
```

---

### 2. Data Storage Architecture

#### Primary Storage: SQLite Database (`.yantra/architecture.db`)

**Schema:**
```sql
CREATE TABLE components (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'service', 'module', 'layer', 'database', 'external', 'ui_component'
    description TEXT,
    position_x REAL,
    position_y REAL,
    width REAL DEFAULT 200,
    height REAL DEFAULT 100,
    parent_id TEXT,  -- For hierarchical grouping
    layer TEXT,  -- 'frontend', 'backend', 'database', 'external', 'infrastructure'
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES components(id) ON DELETE CASCADE
);

CREATE TABLE connections (
    id TEXT PRIMARY KEY,
    from_component_id TEXT NOT NULL,
    to_component_id TEXT NOT NULL,
    connection_type TEXT NOT NULL,  -- 'data_flow', 'api_call', 'event', 'dependency'
    label TEXT,
    bidirectional BOOLEAN DEFAULT 0,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_component_id) REFERENCES components(id) ON DELETE CASCADE,
    FOREIGN KEY (to_component_id) REFERENCES components(id) ON DELETE CASCADE
);

CREATE TABLE component_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    auto_linked BOOLEAN DEFAULT 1,
    link_confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (component_id) REFERENCES components(id) ON DELETE CASCADE,
    UNIQUE(component_id, file_path)
);

CREATE TABLE architecture_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_number INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    change_description TEXT,
    change_type TEXT,  -- 'manual', 'ai_generated', 'auto_sync', 'import'
    user_intent TEXT,
    ai_reasoning TEXT,
    files_changed TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Corruption Protection:**
- SQLite WAL (Write-Ahead Logging) mode enabled
- Integrity check on startup
- Automatic backup before modifications
- Keep last 10 backups in `.yantra/backups/`

#### Secondary Storage: Git-Friendly Exports

**architecture.md** (Markdown + Mermaid diagrams)
**architecture.json** (Machine-readable, complete state)

**Export Triggers:**
- After every architecture modification
- On demand via command
- Before git commit (git hook)

**Recovery Strategy:**
1. Check SQLite integrity on startup
2. If corrupted: Restore from architecture.json
3. If JSON corrupted: Regenerate from GNN code analysis
4. User manually reviews and approves regeneration

---

### 3. User Interface

#### View Modes

**Architecture View** (replaces Code panel)
```
┌───────────────────────────────────────────────────────┐
│ [Complete] [Frontend ▼] [Backend ▼] [Database]       │ ← Hierarchical Tabs
├───────────────────────────────────────────────────────┤
│ ┌──────────────┐         ┌──────────────┐           │
│ │ UI Layer     │────────>│ API Client   │           │
│ │ 12 files ✓   │         │ 3 files ✓    │           │
│ └──────────────┘         └──────────────┘           │
│        │                        │                     │
│        v                        v                     │
│ ┌────────────────────────────────────┐               │
│ │       API Gateway                  │               │
│ │       5 files ✓                    │               │
│ └────────────────────────────────────┘               │
│          │                  │                         │
│          v                  v                         │
│    ┌──────────┐      ┌──────────┐                    │
│    │Auth Svc  │      │User Svc  │                    │
│    │4 files ✓ │      │6 files ✓ │                    │
│    └──────────┘      └──────────┘                    │
└───────────────────────────────────────────────────────┘
```

#### Component Visual States

- **📋 0/0 files** = Planned (gray) - Design exists, no code yet
- **🔄 2/5 files** = In Progress (yellow) - Partially implemented
- **✅ 5/5 files** = Implemented (green) - Fully coded
- **⚠️ Misaligned** (red) - Code exists but doesn't match architecture

#### Connection Types (Visual Arrows)

- **Solid arrow (→)** - Data flow
- **Dashed arrow (⇢)** - API call
- **Wavy arrow (⤳)** - Event/message
- **Dotted arrow (⋯>)** - Dependency
- **Double arrow (⇄)** - Bidirectional

#### Hierarchical Sliding Navigation

**Top-Level Tabs:**
```
[Complete] [Frontend ▼] [Backend ▼] [Database] [External]
```

**Frontend Sub-tabs** (appear when Frontend selected):
```
[UI Layer] [State Mgmt] [API Client] [Routing]
```

**Backend Sub-tabs:**
```
[API Layer] [Auth Service] [User Service] [Payment]
```

**Navigation:**
- Horizontal sliding with CSS transitions (300ms)
- Click to jump directly to any tab
- Keyboard shortcuts: `Ctrl+←/→`
- Breadcrumb trail: `Complete > Backend > Auth Service`

---

### 4. AI Integration Points

#### 4.1 Architecture Generation from Intent

**LLM Prompt:**
```
User intent: "Create a 3-tier web app with React, FastAPI, and PostgreSQL"

Generate architecture diagram with:
- Components (name, type, layer, description)
- Connections (type, direction, label)
- Format as JSON
```

**LLM Response (parsed):**
```json
{
  "components": [
    {"id": "c1", "name": "React Frontend", "type": "ui_component", "layer": "frontend"},
    {"id": "c2", "name": "FastAPI Backend", "type": "service", "layer": "backend"},
    {"id": "c3", "name": "PostgreSQL", "type": "database", "layer": "database"}
  ],
  "connections": [
    {"from": "c1", "to": "c2", "type": "api_call", "label": "REST API"},
    {"from": "c2", "to": "c3", "type": "data_flow", "label": "SQL Queries"}
  ]
}
```

#### 4.2 Architecture Generation from Code (GNN Analysis)

**Algorithm:**
```rust
1. Traverse GNN dependency graph
2. Group files by directory structure:
   - src/frontend/*.tsx → "Frontend UI" component
   - src/auth/*.py → "Auth Service" component
3. Analyze imports to infer connections:
   - "from backend.api import..." → Connection
4. Detect patterns:
   - "sqlalchemy" imports → Database component
   - "redis" imports → Cache component
5. Generate architecture JSON
6. Present to user for refinement
```

#### 4.3 Code-Architecture Alignment Validation

**Detection Algorithm:**
```rust
1. On file save, get GNN dependencies for modified file
2. Query: Which component owns this file?
3. Check expected connections from architecture
4. Compare with actual imports in code
5. If mismatch:
   - Calculate severity (direct violation vs indirect)
   - Generate user-friendly explanation
   - Suggest options (update arch, revert code, refactor)
6. Present alert in UI
```

**LLM Validation Prompt:**
```
Architecture says: API Gateway → User Service → Database
Code shows: api/gateway.py imports psycopg2 (direct DB access)

Explain the violation and suggest options:
1. Update architecture to allow direct DB access
2. Revert code change to maintain service layer
3. Refactor code to use User Service
```

---

### 5. Implementation Components (15 Features)

#### Backend (Rust - `src-tauri/src/architecture/`)

**Module Structure:**
```
architecture/
├── mod.rs           - Main facade
├── storage.rs       - SQLite CRUD operations
├── types.rs         - Component, Connection structs
├── versioning.rs    - Snapshot and restore
├── generator.rs     - AI generation from intent
├── analyzer.rs      - GNN-based generation from code
├── validator.rs     - Alignment checking
├── exporter.rs      - Markdown/Mermaid/JSON export
└── commands.rs      - Tauri command handlers
```

**Key Types:**
```rust
pub struct Component {
    pub id: String,
    pub name: String,
    pub component_type: ComponentType,
    pub layer: Layer,
    pub description: String,
    pub position: Position,
    pub files: Vec<String>,
    pub metadata: serde_json::Value,
}

pub struct Connection {
    pub id: String,
    pub from_component_id: String,
    pub to_component_id: String,
    pub connection_type: ConnectionType,
    pub label: Option<String>,
    pub bidirectional: bool,
}

pub enum ComponentType {
    Service,
    Module,
    Layer,
    Database,
    External,
    UIComponent,
}

pub enum ConnectionType {
    DataFlow,
    ApiCall,
    Event,
    Dependency,
}
```

#### Tauri Commands (10 commands)

```rust
#[tauri::command]
fn create_component(component: Component) -> Result<Component, String>;

#[tauri::command]
fn update_component(id: String, component: Component) -> Result<Component, String>;

#[tauri::command]
fn delete_component(id: String) -> Result<(), String>;

#[tauri::command]
fn create_connection(connection: Connection) -> Result<Connection, String>;

#[tauri::command]
fn get_architecture() -> Result<Architecture, String>;

#[tauri::command]
fn save_architecture_version(description: String) -> Result<i64, String>;

#[tauri::command]
fn list_versions() -> Result<Vec<ArchitectureVersion>, String>;

#[tauri::command]
fn restore_version(version_id: i64) -> Result<Architecture, String>;

#[tauri::command]
fn export_architecture(format: String) -> Result<String, String>;

#[tauri::command]
fn validate_code_alignment() -> Result<Vec<AlignmentIssue>, String>;
```

#### Frontend (SolidJS - `src-ui/components/ArchitectureView/`)

**Component Structure:**
```
ArchitectureView/
├── ArchitectureCanvas.tsx    - React Flow integration
├── ComponentNode.tsx          - Custom node component
├── ConnectionEdge.tsx         - Custom edge component
├── HierarchicalTabs.tsx       - Tab navigation
├── ComponentInspector.tsx     - Component details panel
└── ValidationAlerts.tsx       - Misalignment warnings
```

**State Management (`src-ui/stores/architectureStore.ts`):**
```typescript
export interface ArchitectureStore {
    components: Component[];
    connections: Connection[];
    selectedComponent: Component | null;
    currentLayer: string;  // 'complete', 'frontend', 'backend', etc.
    alignmentIssues: AlignmentIssue[];
    loading: boolean;
    error: string | null;
}

// Actions
export async function loadArchitecture();
export async function createComponent(component: Component);
export async function updateComponent(id: string, component: Component);
export async function deleteComponent(id: string);
export async function createConnection(connection: Connection);
export async function validateAlignment();
export async function exportArchitecture(format: 'markdown' | 'json' | 'mermaid');
```

---

### 6. Performance Targets

| Operation | Target | Scale Target |
|-----------|--------|--------------|
| Load architecture from DB | <50ms | <100ms |
| Render React Flow diagram | <200ms | <500ms |
| Save component/connection | <10ms | <20ms |
| Generate architecture from intent (LLM) | <3s | <5s |
| Generate architecture from code (GNN) | <2s | <5s |
| Validate alignment | <100ms | <300ms |
| Export to Markdown/JSON | <50ms | <100ms |
| Version snapshot | <20ms | <50ms |

---

### 7. Success Metrics

**Technical:**
- ✅ Architecture loads and renders in <250ms
- ✅ No data loss (SQLite + JSON backup strategy)
- ✅ 100% of components linked to files
- ✅ Alignment checks complete in <100ms
- ✅ 15/15 features implemented

**User Experience:**
- ✅ Users can design architecture before coding
- ✅ Users understand legacy codebases via auto-generated architecture
- ✅ Misalignments detected immediately (on save)
- ✅ Users trust architecture as source of truth
- ✅ NPS >50 for architecture feature

**Business Impact:**
- Prevents spaghetti code through design-first approach
- Reduces onboarding time by 60% (visual architecture)
- Catches architectural violations before they become tech debt
- Differentiator: Yantra enforces architecture; other tools don't

---

### 8. Why This is Revolutionary

| Traditional Tools | Architecture View System |
|-------------------|--------------------------|
| Manual diagrams (always outdated) | Auto-synced with code |
| No enforcement | Continuous validation |
| Static images | Interactive, living diagrams |
| No code linking | Files mapped to components |
| No version history | Automatic snapshots |
| No governance | Prevents misalignment |

**Key Differentiators:**
- **GitHub Copilot/Cursor**: Generate code blindly → Result: Spaghetti code
- **Yantra**: Architecture-first → Code must align → Result: Clean, maintainable systems

---

## Documentation System (MVP Phase 1 - IMPLEMENTED)

### Overview

**Status:** ✅ Fully Implemented (November 23, 2025)  
**Purpose:** Automatic extraction and structured presentation of project documentation for transparency and user guidance  
**Location:** `src-tauri/src/documentation/mod.rs` (429 lines), Frontend components  
**Tests:** 4/4 passing

The Documentation System provides a 4-panel UI that automatically extracts and displays structured project information from markdown files, enabling users to understand:
- **What features exist** (implemented, in-progress, planned)
- **Why decisions were made** (architecture choices, tradeoffs)
- **What changed** (file additions, modifications, deletions)
- **What tasks remain** (current week/phase progress)

This creates transparency between the AI agent and the user, ensuring alignment on project state and next actions.

---

### Business Value

**For Users:**
- **Transparency:** See exactly what the AI has implemented
- **Learning:** Understand architectural decisions and rationale
- **Control:** Track progress and intervene when needed
- **Trust:** Verify AI is working on the right things

**For Development:**
- **Single Source of Truth:** Documentation extracted from markdown files
- **Real-time Updates:** Reflects current project state
- **Context Preservation:** Critical for AI agent continuity
- **Debugging Aid:** Track what changed when issues arise

---

### Architecture

#### Data Flow

```
Markdown Files (SSOT)
    ↓
DocumentationManager.load_from_files()
    ↓
Parse & Extract Structured Data
    ↓
Store in Memory (Vec<Feature>, Vec<Decision>, Vec<Change>, Vec<Task>)
    ↓
Tauri Commands (get_features, get_decisions, get_changes, get_tasks)
    ↓
Frontend documentationStore (SolidJS reactive store)
    ↓
DocumentationPanels Component (Tab-based UI)
    ↓
User Interaction → Chat Instructions
```

#### Core Components

**1. Backend (Rust) - `src-tauri/src/documentation/mod.rs`**

```rust
// Core types
pub struct Feature {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: FeatureStatus, // Planned, InProgress, Completed
    pub extracted_from: String,
    pub timestamp: String,
}

pub struct Decision {
    pub id: String,
    pub title: String,
    pub context: String,
    pub decision: String,
    pub rationale: String,
    pub timestamp: String,
}

pub struct Change {
    pub id: String,
    pub change_type: ChangeType, // FileAdded, FileModified, FileDeleted, etc.
    pub description: String,
    pub files: Vec<String>,
    pub timestamp: String,
}

pub struct Task {
    pub id: String,
    pub title: String,
    pub status: TaskStatus, // Completed, InProgress, Pending
    pub milestone: String,
    pub dependencies: Vec<String>,
    pub requires_user_action: bool,
    pub user_action_instructions: Option<String>,
}

pub struct DocumentationManager {
    workspace_path: PathBuf,
    features: Vec<Feature>,
    decisions: Vec<Decision>,
    changes: Vec<Change>,
    tasks: Vec<Task>,
}
```

**2. Frontend Store (TypeScript) - `src-ui/stores/documentationStore.ts`**

```typescript
export interface DocumentationStore {
    features: Feature[];
    decisions: Decision[];
    changes: Change[];
    tasks: Task[];
    loading: boolean;
    error: string | null;
}

// Reactive SolidJS store
const [documentation, setDocumentation] = createStore<DocumentationStore>({
    features: [],
    decisions: [],
    changes: [],
    tasks: [],
    loading: false,
    error: null,
});

// Load all documentation in parallel
export async function loadDocumentation() {
    setDocumentation('loading', true);
    try {
        const [features, decisions, changes, tasks] = await Promise.all([
            invoke<Feature[]>('get_features'),
            invoke<Decision[]>('get_decisions'),
            invoke<Change[]>('get_changes'),
            invoke<Task[]>('get_tasks'),
        ]);
        setDocumentation({
            features,
            decisions,
            changes,
            tasks,
            loading: false,
            error: null,
        });
    } catch (error) {
        setDocumentation('error', String(error));
        setDocumentation('loading', false);
    }
}
```

**3. UI Component (SolidJS) - `src-ui/components/DocumentationPanels.tsx`**

```typescript
export const DocumentationPanels: Component = () => {
    const [activeTab, setActiveTab] = createSignal<'features' | 'decisions' | 'changes' | 'tasks'>('features');
    const docs = useDocumentationStore();

    onMount(() => {
        loadDocumentation();
    });

    return (
        <div class="documentation-panel">
            {/* Tab Navigation */}
            <div class="tabs">
                <button onClick={() => setActiveTab('features')} 
                        class={activeTab() === 'features' ? 'active' : ''}>
                    Features ({docs.features.length})
                </button>
                <button onClick={() => setActiveTab('decisions')} 
                        class={activeTab() === 'decisions' ? 'active' : ''}>
                    Decisions ({docs.decisions.length})
                </button>
                <button onClick={() => setActiveTab('changes')} 
                        class={activeTab() === 'changes' ? 'active' : ''}>
                    Changes ({docs.changes.length})
                </button>
                <button onClick={() => setActiveTab('tasks')} 
                        class={activeTab() === 'tasks' ? 'active' : ''}>
                    Plan ({docs.tasks.filter(t => t.status !== 'completed').length} pending)
                </button>
            </div>

            {/* Tab Content */}
            <div class="tab-content">
                <Switch>
                    <Match when={activeTab() === 'features'}>
                        <FeaturesView features={docs.features} />
                    </Match>
                    <Match when={activeTab() === 'decisions'}>
                        <DecisionsView decisions={docs.decisions} />
                    </Match>
                    <Match when={activeTab() === 'changes'}>
                        <ChangesView changes={docs.changes} />
                    </Match>
                    <Match when={activeTab() === 'tasks'}>
                        <TasksView tasks={docs.tasks} />
                    </Match>
                </Switch>
            </div>
        </div>
    );
};
```

---

### Extraction Algorithms

#### 1. Task Extraction from Project_Plan.md / IMPLEMENTATION_STATUS.md

**Pattern Recognition:**
```rust
fn extract_tasks_from_plan(&mut self, content: &str) {
    let mut current_milestone = "MVP".to_string();
    let mut task_id = 0;

    for line in content.lines() {
        // Detect milestone headers (Week X, Phase X)
        if line.contains("Week") || line.contains("Phase") {
            current_milestone = line.trim().to_string();
        }

        // Extract tasks with checkboxes: - [ ] or - [x]
        if line.trim().starts_with("- [") {
            task_id += 1;
            
            // Determine status from checkbox
            let is_completed = line.contains("[x]") || line.contains("[X]");
            let is_in_progress = line.contains("🔄") || line.contains("In Progress");
            
            let status = if is_completed {
                TaskStatus::Completed
            } else if is_in_progress {
                TaskStatus::InProgress
            } else {
                TaskStatus::Pending
            };

            // Extract title after checkbox
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
```

**Example Input:**
```markdown
## Week 1: Foundation (Nov 26 - Dec 2)

- [x] Set up Tauri + SolidJS project
- [x] Install dependencies
- [ ] 🔄 Implement GNN parser
- [ ] Add LLM integration
```

**Extracted Output:**
```json
[
    {"id": "1", "title": "Set up Tauri + SolidJS project", "status": "completed", "milestone": "Week 1"},
    {"id": "2", "title": "Install dependencies", "status": "completed", "milestone": "Week 1"},
    {"id": "3", "title": "Implement GNN parser", "status": "in-progress", "milestone": "Week 1"},
    {"id": "4", "title": "Add LLM integration", "status": "pending", "milestone": "Week 1"}
]
```

#### 2. Feature Extraction from Features.md

**Pattern Recognition:**
```rust
fn extract_features(&mut self, content: &str) {
    let mut feature_id = 0;
    let mut current_description = String::new();
    let mut in_feature_section = false;

    for line in content.lines() {
        // Detect feature headers: ### ✅ Feature Name
        if line.starts_with("###") && (line.contains("✅") || line.contains("🔄") || line.contains("⏳")) {
            feature_id += 1;
            in_feature_section = true;
            
            // Determine status from emoji
            let status = if line.contains("✅") {
                FeatureStatus::Completed
            } else if line.contains("🔄") {
                FeatureStatus::InProgress
            } else {
                FeatureStatus::Planned
            };

            // Extract title
            let title = line
                .trim_start_matches('#')
                .trim()
                .replace("✅", "")
                .replace("🔄", "")
                .replace("⏳", "")
                .trim()
                .to_string();

            self.features.push(Feature {
                id: feature_id.to_string(),
                title,
                description: String::new(),
                status,
                extracted_from: "Features.md".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }
        
        // Capture description text
        else if in_feature_section && !line.trim().is_empty() && !line.starts_with("##") {
            if let Some(last_feature) = self.features.last_mut() {
                last_feature.description.push_str(line);
                last_feature.description.push('\n');
            }
        }
        
        // End of feature section
        else if line.starts_with("##") {
            in_feature_section = false;
        }
    }
}
```

**Example Input:**
```markdown
### ✅ Dependency Graph (GNN)

Track all code dependencies using Graph Neural Networks.
Detects breaking changes automatically.

### 🔄 LLM Integration

Multi-provider support with failover.
Currently implementing OpenAI client.

### ⏳ Browser Validation

Automated UI testing in Chrome.
Planned for Week 3.
```

**Extracted Output:**
```json
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
```

#### 3. Decision Extraction from Decision_Log.md

**Pattern Recognition:**
```rust
fn extract_decisions(&mut self, content: &str) {
    let mut decision_id = 0;
    let mut current_decision: Option<Decision> = None;
    let mut section_type: Option<&str> = None;

    for line in content.lines() {
        // Detect decision headers: ## Decision Title
        if line.starts_with("##") && !line.contains("Decision Log") {
            // Save previous decision
            if let Some(decision) = current_decision.take() {
                self.decisions.push(decision);
            }
            
            decision_id += 1;
            let title = line.trim_start_matches('#').trim().to_string();
            
            current_decision = Some(Decision {
                id: decision_id.to_string(),
                title,
                context: String::new(),
                decision: String::new(),
                rationale: String::new(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }
        
        // Detect subsections
        else if line.starts_with("**Context:**") || line.contains("Context:") {
            section_type = Some("context");
        }
        else if line.starts_with("**Decision:**") || line.contains("Decision:") {
            section_type = Some("decision");
        }
        else if line.starts_with("**Rationale:**") || line.contains("Rationale:") {
            section_type = Some("rationale");
        }
        
        // Capture section content
        else if let Some(decision) = current_decision.as_mut() {
            if !line.trim().is_empty() && !line.starts_with("**") {
                match section_type {
                    Some("context") => decision.context.push_str(&format!("{}\n", line)),
                    Some("decision") => decision.decision.push_str(&format!("{}\n", line)),
                    Some("rationale") => decision.rationale.push_str(&format!("{}\n", line)),
                    _ => {}
                }
            }
        }
    }
    
    // Save last decision
    if let Some(decision) = current_decision {
        self.decisions.push(decision);
    }
}
```

**Example Input:**
```markdown
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
```

**Extracted Output:**
```json
{
    "id": "1",
    "title": "Use SQLite for GNN Persistence",
    "context": "Need persistent storage for dependency graph between sessions.",
    "decision": "Use SQLite with schema: nodes (id, type, name, file_path), edges (from_id, to_id, edge_type).",
    "rationale": "- Zero-config\n- Fast queries\n- ACID transactions\n- Works offline"
}
```

---

### Tauri Commands

**Backend API - `src-tauri/src/main.rs`:**

```rust
#[tauri::command]
fn get_features(state: State<AppState>) -> Result<Vec<Feature>, String> {
    let manager = state.documentation_manager.lock().unwrap();
    Ok(manager.get_features().to_vec())
}

#[tauri::command]
fn get_decisions(state: State<AppState>) -> Result<Vec<Decision>, String> {
    let manager = state.documentation_manager.lock().unwrap();
    Ok(manager.get_decisions().to_vec())
}

#[tauri::command]
fn get_changes(state: State<AppState>) -> Result<Vec<Change>, String> {
    let manager = state.documentation_manager.lock().unwrap();
    Ok(manager.get_changes().to_vec())
}

#[tauri::command]
fn get_tasks(state: State<AppState>) -> Result<Vec<Task>, String> {
    let manager = state.documentation_manager.lock().unwrap();
    Ok(manager.get_tasks().to_vec())
}

#[tauri::command]
fn reload_documentation(state: State<AppState>) -> Result<(), String> {
    let mut manager = state.documentation_manager.lock().unwrap();
    manager.load_from_files()
}
```

---

### UI Design

#### 4-Tab Layout

```
┌─────────────────────────────────────────────────┐
│ [Features 15] [Decisions 8] [Changes 23] [Plan 42] │
├─────────────────────────────────────────────────┤
│                                                 │
│  ✅ Dependency Graph (GNN)                      │
│     Track all code dependencies using Graph     │
│     Neural Networks. Detects breaking changes.  │
│     Files: src-tauri/src/gnn/*.rs               │
│     Status: Completed (Nov 20, 2025)            │
│                                                 │
│  🔄 LLM Integration                             │
│     Multi-provider support with failover.       │
│     Files: src-tauri/src/llm/*.rs               │
│     Status: In Progress (85% complete)          │
│                                                 │
│  ⏳ Browser Validation                          │
│     Automated UI testing in Chrome.             │
│     Planned for Week 3.                         │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### Visual Indicators

**Feature Status:**
- ✅ **Completed** (green): Fully implemented and tested
- 🔄 **In Progress** (yellow): Actively being worked on
- ⏳ **Planned** (blue): Not yet started

**Task Status:**
- [x] **Completed** (green checkmark)
- [ ] 🔄 **In Progress** (yellow with spinner emoji)
- [ ] **Pending** (empty checkbox)

**Change Types:**
- 📄 **FileAdded** (green)
- ✏️ **FileModified** (yellow)
- 🗑️ **FileDeleted** (red)
- ➕ **FunctionAdded** (green)
- ➖ **FunctionRemoved** (red)

---

### Integration with Chat

**User Actions from Documentation Panel:**

1. **Click feature** → Inserts into chat: "Tell me more about [Feature Name]"
2. **Click pending task** → Inserts: "Work on: [Task Title]"
3. **Click decision** → Shows context and rationale in chat
4. **Click change** → Shows diff and affected files

**Example Flow:**
```
User: [Clicks "⏳ Browser Validation"]
Chat: "I see you're interested in Browser Validation. This feature is planned 
       for Week 3 and will enable automated UI testing in Chrome using the 
       Chrome DevTools Protocol. Should I start implementing it now?"

User: "Yes, start implementing"
Agent: [Begins implementation, adds to Changes tab in real-time]
```

---

### Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Load all documentation | <100ms | ~50ms ✅ |
| Parse Project_Plan.md | <10ms | ~5ms ✅ |
| Parse Features.md | <10ms | ~3ms ✅ |
| Parse Decision_Log.md | <10ms | ~4ms ✅ |
| UI tab switch | <16ms (60fps) | ~8ms ✅ |
| Reload from disk | <200ms | ~100ms ✅ |

---

### Source Files

| File | Lines | Purpose | Tests |
|------|-------|---------|-------|
| `src-tauri/src/documentation/mod.rs` | 429 | Core backend logic, extraction algorithms | 4/4 ✅ |
| `src-ui/stores/documentationStore.ts` | 198 | Reactive state management | N/A |
| `src-ui/components/DocumentationPanels.tsx` | 248 | UI component with tabs | N/A |

**Total:** 875 lines across 3 files

---

### Testing

**Backend Tests - `src-tauri/src/documentation/mod.rs`:**

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_documentation_manager_creation() {
        // Verify empty state initialization
    }

    #[test]
    fn test_add_feature() {
        // Verify feature can be added programmatically
    }

    #[test]
    fn test_add_decision() {
        // Verify decision can be added programmatically
    }

    #[test]
    fn test_add_change() {
        // Verify change can be logged
    }
}
```

**Test Results:** 4/4 passing ✅

---

### Future Enhancements (Post-MVP)

1. **Real-time Updates:** Watch markdown files for changes, auto-reload UI
2. **Search & Filter:** Full-text search across all documentation
3. **Export:** Generate consolidated reports (PDF, HTML)
4. **Timeline View:** Chronological view of all changes and decisions
5. **Dependency Visualization:** Show which features depend on which decisions
6. **User Annotations:** Allow users to add notes to features/decisions
7. **Integration with Git:** Show git blame for decision timestamps
8. **AI Summarization:** Auto-generate executive summaries of changes

---

### Success Metrics

**Technical:**
- ✅ 4/4 tests passing
- ✅ <100ms load time
- ✅ Zero UI lag on tab switches
- ✅ Accurate extraction (100% of checkboxes, headers detected)

**User Experience:**
- ✅ Users can see project state at a glance
- ✅ Users understand what AI has implemented
- ✅ Users can track progress week-by-week
- ✅ Users trust the AI's work due to transparency

---

## Phase 2C: Clean Code Mode (Months 3-4, Post-MVP Enhancement)

### Overview

**Clean Code Mode** is an automated code maintenance system that continuously monitors, analyzes, and refactors codebases to maintain optimal code health. It leverages the existing GNN dependency tracking to detect dead code, perform safe refactorings, validate changes, and harden components after implementation.

**Core Philosophy:**
- **Zero Trust**: Always validate with GNN + tests before applying changes
- **Confidence-Based**: Only auto-apply changes with high confidence (>80%)
- **Non-Breaking**: Never break existing functionality
- **Continuous**: Runs as background process with configurable intervals

**Key Differentiators:**
- Uses GNN for intelligent dead code detection (not just static analysis)
- Real-time refactoring with dependency validation
- Automated hardening after component implementation
- Test-validated changes only

### Capabilities

#### 1. Dead Code Detection & Removal

**What It Detects:**
- Unused Functions: Functions with zero incoming calls (not entry points)
- Unused Classes: Classes with zero instantiations
- Unused Imports: Import statements never referenced
- Unused Variables: Variables assigned but never read
- Dead Branches: Unreachable code paths
- Commented Code: Large blocks of commented-out code

**Entry Points (Never Remove):**
- `main()` functions
- API route handlers, CLI command handlers
- Test functions
- Event handlers, lifecycle hooks
- Exported public APIs

**Confidence Calculation:**
- Base confidence = 1.0 (if zero calls)
- Modifiers: Recent code (×0.5), Public API (×0.3), Exported (×0.2), etc.
- **Auto-Remove Threshold:** 0.8 (80% confidence)

#### 2. Real-Time Refactoring

**Supported Refactorings:**
1. Remove Unused Imports (Auto-apply: Yes, Confidence: 1.0)
2. Extract Duplicate Code (GNN embeddings, similarity >85%)
3. Simplify Complex Functions (Cyclomatic complexity > 10)
4. Rename for Clarity (LLM suggestions)
5. Consolidate Error Handling
6. Optimize Dependencies

**GNN-Powered Duplicate Detection:**
- Semantic similarity using GNN embeddings (978-dim)
- Cosine similarity >0.85 = duplicate
- Detects duplicates across languages (same logic, different syntax)

#### 3. Component Hardening

**Automated Hardening After Implementation:**

**Security Hardening:**
- OWASP Top 10 vulnerabilities
- Language-specific vulnerabilities (eval, SQL injection, XSS)
- Secret detection (API keys, passwords)
- **Auto-Fix:** 70%+ success rate for critical issues

**Performance Hardening:**
- Execution time analysis (avg, p95, p99)
- Memory profiling
- N+1 query detection
- API latency tracking
- Bottleneck identification

**Code Quality Hardening:**
- Cyclomatic complexity analysis
- Code smell detection
- Documentation coverage
- Maintainability index (0-100)

**Dependency Hardening:**
- Known vulnerability check
- Outdated dependency detection
- Security score calculation

#### 4. Configuration System

**`.yantra/clean-code.toml`:**
```toml
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
```

### Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Dead code analysis (10K LOC) | < 2s | Real-time feedback |
| Duplicate detection (10K LOC) | < 5s | GNN embedding comparison |
| Refactoring application | < 3s | Including validation |
| Component hardening | < 10s | Comprehensive scan |
| Security scan | < 5s | Semgrep integration |

### Success Metrics

**Key Performance Indicators:**
- **Dead Code Reduction**: < 2% dead code in healthy projects
- **Refactoring Acceptance Rate**: > 60% for high-confidence suggestions
- **False Positive Rate**: < 5%
- **Security Issue Detection**: 100% of OWASP Top 10
- **Auto-Fix Success Rate**: > 70% for critical issues
- **Code Quality Improvement**: +10 maintainability points after 3 months
- **Developer Time Saved**: 20% reduction in code review time

### Implementation Plan (5 Weeks)

**Week 1: Dead Code Detection**
- Implement analyzer, entry point detection, confidence scoring
- Goal: Identify dead code accurately

**Week 2: Safe Removal**
- Implement removal logic, GNN validation, test validation, rollback
- Goal: Remove dead code safely

**Week 3: Refactoring**
- Duplicate detection, complexity analysis, refactoring suggestions
- Goal: Suggest smart refactorings

**Week 4: Hardening**
- Security scanner integration, performance profiler, auto-fix engine
- Goal: Automated hardening

**Week 5: Continuous Mode**
- Background scheduler, interval-based runs, event triggers
- Goal: Automated maintenance

### Integration Points

**GNN Integration:**
- Leverage `get_dependents()`, `get_incoming_edges()`, feature extraction
- GNN embeddings for semantic duplicate detection

**LLM Integration:**
- Generate refactored code
- Suggest better names, extraction functions, documentation

**Testing Integration:**
- Run affected tests only, full regression, coverage tracking

**Git Integration:**
- Auto-commit cleaned code, create branches, descriptive commit messages

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
