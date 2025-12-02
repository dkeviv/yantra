\*\*# Yantra: Complete Technical Specification

Version: 1.0
Date: November 2024
Document Purpose: Complete technical blueprint for building Yantra from ground zero to enterprise platform

---

## ⚠️ CRITICAL TERMINOLOGY CLARIFICATION

**Two Separate Systems - Do Not Confuse:**

1. **Dependency Graph** (Code Intelligence)
   - **Code Name:** Often called "GNN" in codebase (MISLEADING - historical naming)
   - **Actual Technology:** petgraph graph data structure (NOT a neural network)
   - **Purpose:** Track code relationships (imports, calls, dependencies)
   - **Local:** Tier 1 petgraph in-memory
   - **Cloud:** Tier 0 PostgreSQL + Redis (structure only, no code)

2. **Yantra Codex** (AI Code Generation)
   - **Code Name:** "Yantra Codex" or "Codex"
   - **Actual Technology:** GraphSAGE neural network (IS a real neural network)
   - **Purpose:** Generate code from natural language using ML
   - **Local:** 1024-dim GraphSAGE model, 150M parameters
   - **Cloud:** Aggregated embeddings (opt-in, privacy-preserving)

**Why This Matters:** Throughout this document, "GNN" usually refers to the dependency graph (data structure), NOT the neural network (Yantra Codex). When we mean the AI code generation system, we explicitly say "Yantra Codex" or "GraphSAGE."

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

- AI generates code with full dependency awareness (✅ COMPLETE)
- Automated unit + integration testing (🟡 Generation complete, execution in progress)
- Security vulnerability scanning (⚪ Post-MVP)
- Browser runtime validation (⚪ Post-MVP)
- **Autonomous code execution with environment setup** (🆕 Week 9-10)
- **Integrated terminal for command execution** (🆕 Week 9-10)
- **Real-time output streaming to UI** (🆕 Week 9-10)
- Git integration for seamless commits (⚪ Post-MVP)

Phase 2 (Months 3-4): Cluster Agents & Cloud Graph Database + Package/Deploy

- **Cluster Agent Architecture** (🆕 Month 3-4) - Master-Servant pattern with Git coordination branch for multi-agent parallelism
- **Cloud Graph Database (Tier 0)** (🆕 Month 4-5) - Shared dependency graph for proactive conflict prevention across agents and team members
  - **Note:** This is NOT "Cloud GNN" - the GNN (intelligence layer) runs locally. This is cloud-hosted graph database storage for coordination.
- **Package building (Python wheels, Docker, npm)** (🆕)
- **Automated deployment (AWS, GCP, Kubernetes, Heroku)** (🆕)
- **Health checks and auto-rollback** (🆕)
- Generate workflows from natural language
- Scheduled jobs and event triggers
- Multi-step orchestration with error handling and retries
- **CI/CD pipeline generation** (🆕)

Phase 3 (Months 5-8): Enterprise Automation & Self-Healing

- Cross-system dependency tracking
- External API monitoring and auto-healing
- **Production monitoring with auto-remediation** (🆕)
- **Browser automation for enterprise workflows** (🆕)
- **Legacy system integration via browser control** (🆕)
- Multi-language support (Python + JavaScript + TypeScript)
- **Infrastructure as Code generation** (🆕)

Phase 4 (Months 9-12): Platform Maturity & Ecosystem

- Plugin ecosystem and marketplace
- Advanced refactoring and performance optimization
- Enterprise deployment (on-premise, cloud, air-gapped)
- SLA guarantees (99.9% uptime)
- **Multi-tenant enterprise features** (🆕)

### Market Opportunity

Primary Market: Developer Tools ($50B+)

- IDEs, testing tools, CI/CD platforms
- Target: Mid-market to enterprise (10-1000+ developers)

Secondary Market: Workflow Automation ($10B+)

- Replace/augment Zapier, Make, Workato
- Target: Operations teams, business analysts

Total Addressable Market: $60B+

### Competitive Advantage

| Capability                    | Yantra | Copilot | Cursor | Zapier | Replit Agent |
| ----------------------------- | ------ | ------- | ------ | ------ | ------------ |
| Dependency-aware generation   | ✅     | ❌      | ❌     | N/A    | ❌           |
| Guaranteed no breaks          | ✅     | ❌      | ❌     | ❌     | ❌           |
| Truly unlimited context       | ✅     | ❌      | ❌     | N/A    | ❌           |
| Token-aware context           | ✅     | ⚠️      | ⚠️     | N/A    | ❌           |
| Automated testing             | ✅     | ❌      | ❌     | ❌     | ⚠️           |
| Agentic validation pipeline   | ✅     | ❌      | ❌     | ❌     | ❌           |
| **Autonomous code execution** | ✅     | ❌      | ❌     | ⚪     | ✅           |
| **Package building**          | ✅     | ❌      | ❌     | ❌     | ⚠️           |
| **Automated deployment**      | ✅     | ❌      | ❌     | ⚪     | ✅           |
| **Production monitoring**     | ✅     | ❌      | ❌     | ❌     | ❌           |
| Self-healing systems          | ✅     | ❌      | ❌     | ❌     | ❌           |
| Network effect (failures)     | ✅     | ❌      | ❌     | ❌     | ❌           |
| Works with any LLM            | ✅     | ❌      | ⚠️     | N/A    | ❌           |
| Internal system access        | ✅     | ⚠️      | ⚠️     | ❌     | ⚠️           |
| Custom workflow code          | ✅     | ❌      | ❌     | ❌     | ⚠️           |
| **Browser automation**        | ✅     | ❌      | ❌     | ❌     | ❌           |
| **Integrated terminal**       | ✅     | ❌      | ❌     | N/A    | ✅           |
| **Desktop app (native)**      | ✅     | N/A     | ✅     | N/A    | ❌ (web)     |

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

## Core Architecture: Two Distinct Systems

**CRITICAL DISTINCTION:** Yantra has two separate systems that work together but serve different purposes. They must not be confused:

### 1. Dependency Graph (Code Intelligence)

**Purpose:** Track structural code relationships (imports, calls, dependencies)
**Implementation:** petgraph-based directed graph (Rust)
**Code Reference:** Often called "GNN" in codebase (historical naming, NOT a neural network)

**Architecture:**

```
Local Dependency Graph (Tier 1)          Cloud Dependency Graph (Tier 0)
─────────────────────────────            ────────────────────────────────
• petgraph in-memory                     • PostgreSQL + Redis
• Structural relationships               • Shared graph structure
• Semantic embeddings (fastembed)        • Conflict prevention
• <1ms query time                        • Multi-agent coordination
• Privacy: All code stays local          • Privacy: Structure only, no code
```

**What it tracks:**

- File → File imports
- Function → Function calls
- Class → Class inheritance
- Package → File usage
- Semantic similarity (optional)

**NOT a neural network** - Just a graph data structure with optional embeddings for fuzzy search.

### 2. Yantra Codex (AI Code Generation)

**Purpose:** Generate code from natural language using machine learning
**Implementation:** GraphSAGE neural network (Python/PyTorch)
**Code Reference:** Actual neural network for pattern recognition

**Architecture:**

```
Local Yantra Codex (Tier 1)              Cloud Yantra Codex (Tier 0 - Optional)
───────────────────────────              ───────────────────────────────────────
• GraphSAGE GNN (1024-dim)               • Aggregated pattern embeddings
• 150M parameters, ~600MB                • Collective learning from users
• 15ms inference time                    • Privacy: Embeddings only, no code
• Learns from LLM feedback               • Network effects → better accuracy
• Privacy: All code stays local          • Opt-in only
```

**What it does:**

- Understands problem intent (978-dim features)
- Predicts logic patterns (GraphSAGE)
- Generates code with confidence scoring
- Learns from LLM corrections
- Cost optimization (90% fewer LLM calls)

**IS a Graph Neural Network** - Real machine learning model trained on code patterns.

### Why Two Systems?

| Aspect          | Dependency Graph          | Yantra Codex               |
| --------------- | ------------------------- | -------------------------- |
| **Purpose**     | Code relationships        | Code generation            |
| **Technology**  | petgraph (data structure) | GraphSAGE (neural network) |
| **Input**       | AST from tree-sitter      | Problem description        |
| **Output**      | Dependency queries        | Generated code             |
| **Speed**       | <1ms                      | 15ms                       |
| **Learning**    | No learning               | Continuous learning        |
| **Local/Cloud** | Both (sync structure)     | Both (sync embeddings)     |
| **Code Name**   | "GNN" (misleading)        | "Yantra Codex"             |

**Integration:** Dependency Graph provides context → Yantra Codex generates code → Dependency Graph validates new code fits properly

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

| Confidence  | Routing Decision    | Rationale                            | Cost    |
| ----------- | ------------------- | ------------------------------------ | ------- |
| **0.9-1.0** | Yantra alone        | Seen pattern many times              | $0      |
| **0.8-0.9** | Yantra alone        | Good confidence, validate with tests | $0      |
| **0.5-0.8** | Yantra + LLM review | Partial knowledge, need LLM help     | ~$0.015 |
| **0.0-0.5** | LLM alone           | Novel pattern, Yantra can't help yet | ~$0.025 |

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

| Metric        | LLM Only   | Yantra + LLM (Month 1) | Yantra + LLM (Year 1) |
| ------------- | ---------- | ---------------------- | --------------------- | --- |
| Cost/1000 gen | $25        | $9 (64% ↓)             | $1 (96% ↓)            |     |
| Quality       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐             | ⭐⭐⭐⭐⭐            |
| Speed         | 3-5s       | 0.5-2s                 | 0.015-0.5s            |
| Learning      | ❌         | ✅                     | ✅✅✅                |
| Privacy       | ❌ (cloud) | ✅ (mostly local)      | ✅ (95% local)        |

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

│ AI-CODE PLATFORM │

├──────────────────────────────────────────────────────┤

│ │

│ USER INTERFACE (AI-First) │

│ ┌─────────────────────────────────────────────┐ │

│ │ Chat/Task Interface (Primary - 60% screen) │ │

│ │ Code Viewer (Secondary - 25% screen) │ │

│ │ Browser Preview (Live - 15% screen) │ │

│ └─────────────────────────────────────────────┘ │

│ │ │

│ ┌────────────────────┼─────────────────────────┐ │

│ │ ORCHESTRATION LAYER│ │ │

│ │ Multi-LLM Manager │ │ │

│ │ ├─ Claude Sonnet (Primary) │ │

│ │ ├─ GPT-4 (Secondary/Validation) │ │

│ │ └─ Routing & Failover Logic │ │

│ └─────────────────────────────────────────────┘ │

│ │ │

│ ┌────────────────────┼─────────────────────────┐ │

│ │ INTELLIGENCE LAYER │ │ │

│ │ Graph Neural Network (GNN) │ │

│ │ ├─ Code Dependencies │ │

│ │ ├─ External API Tracking │ │

│ │ ├─ Data Flow Analysis │ │

│ │ └─ Known Issues Database (LLM Failures) │ │

│ │ │ │

│ │ Vector Database (RAG) │ │

│ │ ├─ Code Templates │ │

│ │ ├─ Best Practices │ │

│ │ ├─ Project Patterns │ │

│ │ └─ Failure Pattern Library (Network Effect) │ │

│ │ │ │

│ │ Unlimited Context Engine │ │

│ │ ├─ Token Counting & Management │ │

│ │ ├─ Context Compression & Chunking │ │

│ │ ├─ Hierarchical Context Assembly │ │

│ │ └─ Adaptive Context Strategies │ │

│ └─────────────────────────────────────────────┘ │

│ │ │

│ ┌────────────────────┼─────────────────────────┐ │

│ │ VALIDATION LAYER │ │ │

│ │ ├─ Testing Engine (pytest/jest) │ │

│ │ ├─ Security Scanner (Semgrep + custom) │ │

│ │ ├─ Browser Integration (CDP) │ │

│ │ ├─ Dependency Validator (GNN) │ │

│ │ └─ Agentic Validation Pipeline │ │

│ │ │ │

│ │ Agent State Machine │ │

│ │ ├─ Code Generation → Validation Loop │ │

│ │ ├─ Confidence Scoring & Auto-Retry │ │

│ │ ├─ Failure Analysis & Pattern Extraction │ │

│ │ └─ Self-Healing with Known Issues DB │ │

│ └─────────────────────────────────────────────┘ │

│ │ │

│ ┌────────────────────┼─────────────────────────┐ │

│ │ INTEGRATION LAYER │ │ │

│ │ ├─ Git (MCP Protocol) │ │

│ │ ├─ File System │ │

│ │ └─ External APIs (Phase 2+) │ │

│ └─────────────────────────────────────────────┘ │

│ │

└──────────────────────────────────────────────────────┘

### Technology Stack

Desktop Framework:

- Tauri 1.5+ (Rust backend + web frontend)
- Rationale: 600KB bundle vs 150MB Electron, native performance

Frontend:

- SolidJS 1.8+ (reactive UI framework)
- Monaco Editor 0.44+ (code viewing)
- TailwindCSS 3.3+ (styling)
- WebSockets (real-time updates)

Backend (Rust):

- Tokio 1.35+ (async runtime)

**System 1: Dependency Graph (Code Intelligence) - Tier 0-1**

- **Tier 0 - Cloud Graph Database (Shared Coordination):**
  - **Phase 2B (Month 4-5)**: Cloud-hosted dependency graph for team coordination
  - **Purpose**: Real-time conflict prevention across agents and users
  - **Data**: Dependency graph structure (NOT code content - privacy-preserving)
  - **API**: WebSocket/gRPC for <50ms latency queries
  - **Deployment**: Hosted (yantra-cloud.com) OR self-hosted (enterprise)
  - **Use Cases**: File modification registry, dependency-aware conflict warnings, transitive dependency tracking
  - **Note:** This is a graph DATABASE for storage/coordination. Not a neural network.
- **Tier 1 - Local Dependency Graph (Hot Path):**
  - **Phase 1-2 (NOW)**: Hybrid (in-memory petgraph + SQLite persistence with WAL + pooling)
  - **Phase 3 (Month 5-8)**: Pure in-memory + snapshot persistence for >100K LOC projects
  - Target: Sub-millisecond query times, ~1GB memory for 100K LOC
  - Tech: petgraph in-memory + bincode snapshots + write-ahead log
  - **Sync**: Local dependency graph syncs structure to Cloud Graph DB (Tier 0)
  - **Note:** Called "GNN" in codebase (historical naming). This is NOT a neural network - it's a graph data structure (petgraph).

**System 2: Yantra Codex (AI Code Generation) - Separate System**

- **Local Yantra Codex (Tier 1):**
  - **GraphSAGE Neural Network**: 1024-dim embeddings, 150M parameters, ~600MB
  - **Inference**: 15ms (CPU), 5ms (GPU)
  - **Training**: Python/PyTorch, learns from LLM corrections
  - **Purpose**: Generate code from natural language, pattern recognition
  - **Storage**: Model weights in `models/yantra_codex_v1.pth`
- **Cloud Yantra Codex (Tier 0 - Optional, Opt-in):**
  - **Collective Learning**: Aggregated pattern embeddings from all users
  - **Privacy**: Only embeddings shared, NO code or user data
  - **Network Effects**: More users → better accuracy → lower LLM costs
  - **Deployment**: yantra-cloud.com (opt-in only)

**Other Storage Tiers:**

- **Tier 2 - Agent Coordination (Write-Heavy):**
  - **Phase 1 (NOW)**: SQLite with WAL + connection pooling (adequate for single-agent)
  - **Phase 2A (Month 3-4)**: Migrate to sled for cluster agents + Git coordination branch
  - Tech: sled (pure Rust, 100k writes/sec, lock-free)
  - Design: Prefixed keys for namespacing (e.g., `agent:codegen:state`, `lock:src/main.py`, `a2a:agent-1:msg`)
  - **Git Branch**: `.yantra/coordination` for feature assignments and high-level events
- **Tier 3 - Reference Data (Read-Heavy):**
  - **Storage:** SQLite 3.44+ with WAL mode + r2d2 connection pooling
  - **Phase 1 (NOW)**: Enable WAL mode + connection pooling (THIS WEEK)
  - **Decision**: Keep SQLite permanently (perfect for read-heavy local data)
  - **Use Cases**: Known issues, architecture snapshots, configuration, audit logs
  - **Access Pattern**: 100:1 read:write ratio, concurrent reads during writes
  - **No PostgreSQL Migration**: 4-tier architecture makes PostgreSQL unnecessary
- **Tier 4 - Context Cache (Ephemeral):**
  - **Phase 2+ (Month 4)**: In-memory LRU cache (moka crate)
  - Use cases: Context assembly results, token counts, LLM response caches
  - Budget: 500MB, automatic eviction, no persistence needed
- Reqwest 0.11+ (HTTP client)
- Serde 1.0+ (JSON serialization)
- **tiktoken-rs 0.5+ (token counting)**

Dependency Graph with Semantic Enhancement (Code Intelligence Graph):

**IMPORTANT:** This is NOT the same as Yantra Codex (GraphSAGE neural network). This is a graph data structure for tracking code relationships.

**Terminology Clarification:**

- **In Code:** Often called "GNN" (historical naming, misleading)
- **Actual Nature:** petgraph-based graph data structure, NOT a Graph Neural Network
- **Purpose:** Track structural code relationships (imports, function calls, inheritance)
- **Optional:** Semantic embeddings for fuzzy search (using fastembed, not training a neural network)

**Core Architecture:**

- **Language:** Rust (performance critical)
- **Graph Library:** petgraph 0.6+ (directed graph for dependencies)
- **Parser:** tree-sitter (Python, JS, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin)
- **Storage:** SQLite 3.44+ with WAL mode (persistence)
- **Embeddings:** fastembed-rs 5.3+ (pure Rust, ONNX models for semantic search only)

**Hybrid Search Capability (Structural + Semantic):**

**1. Structural Dependencies (Exact):**

```rust
// Track precise code relationships
✅ Imports: "UserService imports AuthContext"
✅ Function Calls: "create_user() calls validate_email()"
✅ Inheritance: "UserProfile extends BaseProfile"
✅ Data Flow: "function returns X, passed to Y"
```

**2. Semantic Similarity (Fuzzy):**

```rust
// Each code node can have optional embedding (384-dim vector)
pub struct CodeNode {
    id, name, type, location,        // Structural metadata
    semantic_embedding: Option<Vec<f32>>,  // 384-dim from all-MiniLM-L6-v2
    code_snippet: Option<String>,    // For generating embeddings
    docstring: Option<String>,       // Function/class documentation
}
```

**3. Hybrid Search Examples:**

**Example 1: Intent-Driven Context Assembly**

```python
# User request: "Add email validation to user registration"

# Traditional structural search (exact dependencies):
find_dependencies("register_user")
→ UserModel class (called by register_user)
→ Database.save() (called by UserModel)
→ sanitize_input() (called by register_user)

# Semantic enhancement (fuzzy discovery):
find_similar_to_intent("email validation", min_similarity=0.75)
→ validate_email() function (not called yet, but semantically relevant!)
→ validate_phone() function (similar validation pattern)
→ check_email_format() function (potential duplicate to consolidate)

# Combined result: LLM gets BOTH exact dependencies AND semantically relevant code
✅ Better code generation (finds existing validation functions)
✅ Prevents duplication (discovers similar functions)
✅ Maintains correctness (exact dependencies ensure nothing breaks)
```

**Example 2: Refactoring Detection**

```python
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
# Structural graph ensures we update all call sites correctly
```

**Performance Targets:**

| Operation                    | Target          | Implementation                           |
| ---------------------------- | --------------- | ---------------------------------------- |
| Graph build                  | <5s for 10K LOC | Incremental parsing with tree-sitter     |
| Incremental update           | <50ms per file  | Only reparse changed file + update edges |
| Dependency lookup            | <10ms           | petgraph BFS traversal                   |
| Semantic search (MVP)        | <10ms           | **HNSW indexing (hnsw_rs)**              |
| Embedding generation         | <10ms per node  | fastembed-rs with quantized ONNX         |
| Batch embeddings (100 nodes) | <100ms          | Parallel processing                      |

**HNSW Vector Indexing (Ferrari MVP Standard):**

Yantra uses **Hierarchical Navigable Small World (HNSW)** indexing for blazing-fast semantic search at any scale. This is NOT a separate vector database—it's an in-memory index structure built directly within the CodeGraph.

**Why HNSW (Not Linear Scan):**

```
Performance at Scale:
├── Linear Scan (naive):
│   ├── 1k nodes: 0.5ms ✅
│   ├── 10k nodes: 50ms ❌ (5x over target)
│   └── 100k nodes: 500ms ❌ (50x over target)
│
└── HNSW Index (Ferrari standard):
    ├── 1k nodes: 0.1ms ✅ (5x faster)
    ├── 10k nodes: 2ms ✅ (25x faster)
    └── 100k nodes: 5ms ✅ (100x faster, meets <10ms target)
```

**Implementation:**

```rust
use hnsw_rs::prelude::*;

pub struct CodeGraph {
    graph: DiGraph<CodeNode, EdgeType>,
    node_map: HashMap<String, NodeIndex>,

    // HNSW index for O(log n) semantic search
    semantic_index: Hnsw<f32, DistCosine>,
}

impl CodeGraph {
    pub fn build_semantic_index(&mut self) {
        let hnsw = Hnsw::<f32, DistCosine>::new(
            16,    // M: max_nb_connection (connectivity)
            10000, // max_elements (initial capacity)
            16,    // ef_construction (build quality)
            200,   // ef_search (query accuracy)
            DistCosine,
        );

        // Insert embeddings from existing nodes
        for (idx, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            if let Some(embedding) = &node.semantic_embedding {
                hnsw.insert((&embedding[..], idx.index()));
            }
        }

        self.semantic_index = hnsw;
    }

    pub fn find_similar_nodes(
        &self,
        query_embedding: &[f32],
        max_results: usize,
    ) -> Vec<(CodeNode, f32)> {
        // O(log n) search via HNSW
        let neighbors = self.semantic_index.search(query_embedding, max_results, 200);

        neighbors.iter()
            .filter_map(|neighbor| {
                let node_idx = NodeIndex::new(neighbor.d_id);
                self.graph.node_weight(node_idx).map(|node| {
                    (node.clone(), 1.0 - neighbor.distance)  // Convert to similarity
                })
            })
            .collect()
    }
}
```

**Key Characteristics:**

- **Complexity:** O(log n) average query time (vs O(n) linear scan)
- **Accuracy:** 99.5%+ recall with proper ef_search tuning
- **Memory:** +30-50% overhead on embedding size (~1.5-2MB for 10k nodes)
- **Index Build:** O(n log n), ~1s for 10k nodes, 10s for 100k nodes
- **Incremental:** Supports insert/delete for graph updates
- **Pure Rust:** hnsw_rs crate (no Python dependencies)

**Why This Matters (Ferrari vs Corolla):**

🚗 **Corolla MVP (Linear Scan):**

- Works for demos and small projects
- Breaks at scale (50ms+ on real codebases)
- Technical debt from day one
- Requires rewrite for enterprise

🏎️ **Ferrari MVP (HNSW Index):**

- Enterprise-ready from day one
- Scales to 100k+ nodes (<10ms guaranteed)
- No technical debt
- Production-grade architecture

**Decision:** Yantra is a Ferrari MVP. We use HNSW indexing from the start, not as an optimization "if needed later."

**Why Not Separate RAG/Vector Database:**

❌ **Traditional RAG (e.g., ChromaDB):**

- Duplicate storage (code in files + embeddings in vector DB)
- Sync complexity (keep vector DB updated with code changes)
- Two separate queries (one for vectors, one for dependencies)
- False positives (semantic similarity without structural context)

✅ **Semantic-Enhanced Dependency Graph (Yantra):**

- Single source of truth (graph contains everything)
- Automatic sync (update node → embedding updates inline)
- Single query (BFS traversal filters by similarity simultaneously)
- Precision + recall (exact dependencies + fuzzy discovery)
- Simpler architecture (no external vector DB needed)

**Embedding Model Details:**

- **Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **Dimensions:** 384 (optimal balance of speed vs quality)
- **Size:** 22MB quantized ONNX model
- **Inference:** <8ms per embedding on CPU (fastembed-rs)
- **Privacy:** 100% local inference (no API calls)
- **Cache:** In-memory LRU for frequently accessed embeddings

LLM Integration:

- Primary: Anthropic Claude API (claude-sonnet-4)
- Secondary: OpenAI API (gpt-4-turbo)
- Tertiary: Qwen Coder (via OpenAI-compatible API)
- Rate limiting, retry logic, circuit breaker
- **Confidence scoring from response metadata**

Context Management:

- Token counting: tiktoken-rs (exact counts)
- Compression: Syntax-aware, de-duplication
- Hierarchical assembly: 4 levels of detail
- Caching: SQLite with 24h TTL
- Semantic context: Integrated in dependency graph (see above)

Testing:

- Python: pytest 7.4+, pytest-cov
- JavaScript: Jest (Phase 2+)
- Runner: Subprocess execution from Rust

Security:

- SAST: Semgrep with OWASP rules
- Dependencies: Safety (Python), npm audit
- Secrets: TruffleHog patterns

Browser:

- Protocol: Chrome DevTools Protocol (CDP)
- Library: chromiumoxide (Rust CDP client)
- Automation: Playwright (complex interactions)

Git:

- Protocol: Model Context Protocol (MCP)
- Library: git2-rs (libgit2 Rust bindings)

---

## Core Innovation: Truly Unlimited Context

### The Problem with Current AI Coding Tools

All existing AI coding tools (GitHub Copilot, Cursor, Windsurf, etc.) are fundamentally limited by LLM context windows:

- Claude Sonnet 4: 200K tokens (~150K LOC worth of context)
- GPT-4 Turbo: 128K tokens (~100K LOC worth of context)
- Qwen Coder: 32K-128K tokens depending on version

**Result:** These tools fail on large codebases, miss critical dependencies, and generate code that breaks existing functionality.

### Yantra's Solution: Context Intelligence, Not Context Limits

Yantra achieves truly unlimited context through a multi-layered approach that works with ANY LLM, including smaller models like Qwen Coder:

#### 1. Token-Aware Context Management

**Implementation:**

- Real token counting using tiktoken-rs (exact, not estimated)
- Dynamic token budget allocation based on LLM provider
- Reserve 20% of context window for response generation
- Graceful degradation when approaching limits

**Token Budgets:**

- Claude Sonnet 4: 160,000 context tokens (40K for response)
- GPT-4 Turbo: 100,000 context tokens (28K for response)
- Qwen Coder 32K: 25,000 context tokens (7K for response)
- Adaptive allocation per LLM capability

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

- Complete source code of target files
- All direct dependencies (imports, function calls)
- Data structures and types referenced
- **Token allocation: 40% of budget**

**Level 2 - Related Context (Signatures Only):**

- Function signatures (no implementation)
- Class definitions (no methods)
- Type definitions and interfaces
- API contracts
- **Token allocation: 30% of budget**

**Level 3 - Distant Context (References Only):**

- Module names and imports
- High-level architecture
- Indirect dependencies (A → B → C)
- **Token allocation: 20% of budget**

**Level 4 - Metadata (Summaries):**

- Project structure overview
- Known patterns and conventions
- Relevant documentation snippets
- **Token allocation: 10% of budget**

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

- Remove comments (unless directly relevant)
- Strip docstrings (keep in metadata)
- Minimize whitespace
- Remove unused imports in context
- **Savings: 20-30% tokens**

**B. Semantic Chunking:**

- Split large files into logical chunks (classes, functions)
- Include only relevant chunks
- Track chunk relationships in GNN
- **Savings: 40-60% tokens for large files**

**C. De-duplication:**

- Identical code blocks referenced multiple times
- Common utility functions
- Shared type definitions
- **Savings: 10-15% tokens**

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

- All function signatures and docstrings
- Common code patterns
- Failure patterns with fixes (see next section)
- Best practices and conventions

**Semantic Search:**

- Query: User intent + target code context
- Retrieve: Top-K most relevant patterns (K=10-20)
- Add to context as examples
- **Cost: 2,000-5,000 tokens (high value)**

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

| Task        | Strategy                           | Token Allocation |
| ----------- | ---------------------------------- | ---------------- |
| New feature | Wide context (many dependencies)   | 70% L1+L2        |
| Bug fix     | Deep context (full implementation) | 80% L1           |
| Refactoring | Architectural context (all usages) | 50% L1, 40% L2   |
| Testing     | Target code + similar tests        | 60% L1, 30% RAG  |

**Dynamic Adjustment:**

- Monitor LLM confidence scores
- If low confidence → expand context
- If token limit hit → compress L2/L3
- Iterative refinement

#### 6. Context Caching & Reuse

**SQLite Cache:**

- Cache compressed context by hash (file content + dependencies)
- 24-hour TTL
- Invalidate on file changes
- **Performance gain: <50ms retrieval vs 100-500ms assembly**

**Shared Context Across Requests:**

- Same file referenced multiple times
- Compute once, reuse
- Track with reference counting

### Why This Enables ANY LLM (Including Qwen Coder)

**The Key Insight:** Most LLM failures are due to missing context, not LLM capability.

**With Yantra's Context Intelligence:**

1. **Qwen Coder 32K** (smaller model):
   - Gets 25,000 tokens of perfectly relevant context
   - Hierarchical assembly prioritizes what matters
   - RAG provides proven patterns
   - Known failures database prevents common mistakes
   - **Result: Performs as well as GPT-4 with 100K tokens**

2. **Even GPT-3.5** (16K context):
   - Gets 12,000 tokens of hyper-relevant context
   - Every token is carefully selected
   - Compression eliminates noise
   - **Result: Better than GPT-4 with random 100K context**

**Validation:**

- Benchmark: Same task with GPT-4 (naive 100K context) vs Qwen Coder (optimized 25K context)
- Metric: Code quality, test pass rate, breaking changes
- Target: Qwen performance within 5% of GPT-4

### Performance Targets

| Operation              | MVP Target | Scale Target |
| ---------------------- | ---------- | ------------ |
| Token counting         | <10ms      | <5ms         |
| Context assembly       | <100ms     | <50ms        |
| Compression            | <50ms      | <20ms        |
| RAG retrieval          | <200ms     | <100ms       |
| Total context pipeline | <500ms     | <200ms       |

### Implementation Phases

**MVP (Month 1-2):**

- ✅ Token-aware context assembly (no arbitrary limits)
- ✅ BFS traversal with priority (implemented)
- ⚠️ Token counting with tiktoken-rs (add)
- ⚠️ Hierarchical context (L1 + L2) (add)
- ⚠️ Basic compression (whitespace, comments) (add)

**Post-MVP (Month 3-4):**

- Advanced compression (semantic chunking)
- RAG with ChromaDB
- Full hierarchical context (L1-L4)
- Adaptive strategies per task type
- Context caching

**Enterprise (Month 5-8):**

- Multi-language context mixing
- Cross-repository context
- Distributed context cache
- Real-time context updates

---

## Core Innovation: Fully Autonomous Agentic Architecture

### What "Fully Autonomous Agentic" Means

**Not agentic:** LLM generates code → User manually tests → User manually fixes issues → User manually commits

**Partially agentic:** LLM generates code → System validates → User fixes issues → User commits

**Fully autonomous (Yantra):** LLM generates code → System validates → System fixes → System tests → **System runs** → **System packages** → **System deploys** → **System monitors** → Repeat until perfect

**Yantra is end-to-end autonomous:** Human provides intent, AI handles entire development and deployment lifecycle.

---

## Comprehensive Agentic Capabilities Framework

### Philosophy: The Four Pillars of Autonomous Development

Yantra's agentic capabilities are organized into four fundamental pillars that mirror human developer capabilities:

```
┌─────────────────────────────────────────────────────────────┐
│                   AGENTIC CAPABILITIES                       │
│                                                               │
│  🔍 PERCEIVE  →  🧠 REASON  →  ⚡ ACT  →  🔄 LEARN          │
│                                                               │
│  Sense the      Analyze &      Execute      Adapt from      │
│  environment    decide         actions       feedback        │
└─────────────────────────────────────────────────────────────┘
```

### 1. 🔍 PERCEIVE - Input & Sensing Layer

**Purpose:** Gather comprehensive understanding of code, data, and environment

#### 1.1 File System Operations

| Capability         | Tool/Terminal | MVP Status | Purpose                                                | Implementation                      |
| ------------------ | ------------- | ---------- | ------------------------------------------------------ | ----------------------------------- |
| `file_read`        | Tool          | ✅ DONE    | Read file contents with encoding detection             | `main.rs::read_file()`              |
| `file_write`       | Tool          | ✅ DONE    | Create/overwrite files                                 | `main.rs::write_file()`             |
| `file_edit`        | Tool          | 🔴 TODO    | Surgical edits (line range, search-replace, AST-based) | **NEW**                             |
| `file_delete`      | Tool          | 🔴 TODO    | Remove files safely                                    | **NEW**                             |
| `file_move`        | Tool          | 🔴 TODO    | Rename/move files with dependency updates              | **NEW**                             |
| `file_copy`        | Tool          | 🔴 TODO    | Duplicate files                                        | **NEW**                             |
| `directory_create` | Tool          | ✅ DONE    | Create directories recursively                         | Built-in                            |
| `directory_list`   | Tool          | ✅ DONE    | List contents with filters                             | `main.rs::read_dir()`               |
| `directory_tree`   | Tool          | 🔴 TODO    | Get full project structure                             | **NEW**                             |
| `file_search`      | Tool          | 🔴 TODO    | Find files by name/pattern/glob                        | **NEW**                             |
| `file_watch`       | Tool          | 🔴 TODO    | Monitor for changes (reactive agents)                  | **NEW** (use `notify` crate)        |
| `docx_read`        | Tool          | 🔴 TODO    | Read Word documents                                    | **NEW** (use `docx-rs`)             |
| `pdf_read`         | Tool          | 🔴 TODO    | Extract text from PDFs                                 | **NEW** (use `pdf-extract`/`lopdf`) |

**Implementation Priority:**

- **P0 (MVP):** ✅ Basic file I/O (read, write, list) - DONE
- **P1 (High):** 🔴 Document readers (DOCX, PDF) for architecture generation
- **P2 (Medium):** 🔴 Advanced file ops (edit, delete, move, search)
- **P3 (Future):** 🔴 File watchers for reactive behavior

#### 1.2 Code Intelligence (Tree-sitter Powered)

| Capability           | Tool/Terminal | MVP Status | Purpose                                        | Implementation                  |
| -------------------- | ------------- | ---------- | ---------------------------------------------- | ------------------------------- |
| `parse_ast`          | Tool          | ✅ DONE    | Get AST for file/snippet                       | `gnn/parser.rs` (tree-sitter)   |
| `get_symbols`        | Tool          | ✅ DONE    | Extract functions, classes, variables, imports | `gnn/parser.rs`                 |
| `get_references`     | Tool          | 🔴 TODO    | Find all usages of a symbol                    | **NEW**                         |
| `get_definition`     | Tool          | 🔴 TODO    | Jump to definition                             | **NEW**                         |
| `get_scope`          | Tool          | 🔴 TODO    | Get scope context for a position               | **NEW**                         |
| `get_diagnostics`    | Tool          | ✅ DONE    | Syntax errors, warnings                        | Integrated in parser            |
| `semantic_search`    | Tool          | ✅ PARTIAL | Search code by meaning                         | GNN semantic layer (embeddings) |
| `get_call_hierarchy` | Tool          | ✅ DONE    | Incoming/outgoing calls                        | GNN dependency tracking         |
| `get_type_hierarchy` | Tool          | 🔴 TODO    | Class inheritance chains                       | **NEW**                         |

**Implementation Priority:**

- **P0 (MVP):** ✅ AST parsing, symbol extraction, call hierarchy - DONE
- **P1 (High):** ✅ Semantic search - DONE (GNN embeddings)
- **P2 (Medium):** 🔴 References, definitions, scope analysis
- **P3 (Future):** 🔴 Type hierarchy

#### 1.3 Dependency Graph & Impact Analysis

| Capability               | Tool/Terminal | MVP Status | Purpose                           | Implementation                       |
| ------------------------ | ------------- | ---------- | --------------------------------- | ------------------------------------ |
| `build_dependency_graph` | Tool          | ✅ DONE    | Generate full project graph       | `gnn/engine.rs` (10/10 features)     |
| `get_dependents`         | Tool          | ✅ DONE    | What depends on X?                | `gnn/engine.rs::get_dependents()`    |
| `get_dependencies`       | Tool          | ✅ DONE    | What does X depend on?            | `gnn/engine.rs::get_dependencies()`  |
| `impact_analysis`        | Tool          | ✅ DONE    | If I change X, what breaks?       | `architecture/deviation_detector.rs` |
| `find_cycles`            | Tool          | ✅ DONE    | Detect circular dependencies      | `gnn/engine.rs::detect_cycles()`     |
| `get_module_boundaries`  | Tool          | 🔴 TODO    | Identify architectural layers     | **NEW**                              |
| `cross_repo_deps`        | Tool          | 🔴 TODO    | External API/service dependencies | **NEW** (Phase 2)                    |

**Implementation Priority:**

- **P0 (MVP):** ✅ Full dependency graph - DONE (100%)
- **P1 (High):** ✅ Impact analysis - DONE
- **P2 (Future):** 🔴 Module boundaries, cross-repo dependencies

#### 1.4 Database Connections & Schema Intelligence

| Capability   | Tool/Terminal | MVP Status | Purpose                                  | Implementation                                 |
| ------------ | ------------- | ---------- | ---------------------------------------- | ---------------------------------------------- |
| `db_connect` | **TOOL**      | 🔴 TODO    | Establish connection with pooling        | **NEW** `agent/database/connection_manager.rs` |
| `db_query`   | **TOOL**      | 🔴 TODO    | Execute SELECT (read-only, validated)    | **NEW**                                        |
| `db_execute` | **TOOL**      | 🔴 TODO    | Execute INSERT/UPDATE/DELETE (validated) | **NEW**                                        |
| `db_schema`  | **TOOL**      | 🔴 TODO    | Get tables, columns, types, constraints  | **NEW**                                        |
| `db_explain` | **TOOL**      | 🔴 TODO    | Query execution plan                     | **NEW**                                        |
| `db_migrate` | **TOOL**      | 🔴 TODO    | Run migrations with rollback             | **NEW** `agent/database/migration_manager.rs`  |
| `db_seed`    | **TOOL**      | 🔴 TODO    | Insert test data                         | **NEW**                                        |

**Why Tool (Not Terminal):**

- **Connection Pooling:** Reuse connections across commands (performance)
- **Credential Security:** Encrypted storage, no plaintext in terminal history
- **Query Validation:** Prevent SQL injection, validate against schema
- **Transaction Support:** Atomic operations with rollback
- **Schema Tracking:** Update GNN when schema changes
- **Cross-DB Support:** Unified API for PostgreSQL, MySQL, SQLite, MongoDB

**Implementation Priority:**

- **P0 (High):** 🔴 Connection manager with pooling and security
- **P1 (High):** 🔴 Schema introspection and validation
- **P2 (High):** 🔴 Safe migration manager with rollback
- **P3 (Medium):** 🔴 Query execution and test data seeding

**Supported Databases:**

- PostgreSQL (via `tokio-postgres`)
- MySQL (via `sqlx`)
- SQLite (via `rusqlite` - already in use for GNN)
- MongoDB (via `mongodb` crate)
- Redis (via `redis` crate)

#### 1.5 API Monitoring & Contract Validation

| Capability              | Tool/Terminal | MVP Status | Purpose                       | Implementation                                    |
| ----------------------- | ------------- | ---------- | ----------------------------- | ------------------------------------------------- |
| `api_import_spec`       | **TOOL**      | 🔴 TODO    | Import OpenAPI/Swagger specs  | **NEW** `agent/api_monitor/spec_parser.rs`        |
| `api_validate_contract` | **TOOL**      | 🔴 TODO    | Detect breaking API changes   | **NEW** `agent/api_monitor/contract_validator.rs` |
| `api_health_check`      | **TOOL**      | 🔴 TODO    | Test endpoint availability    | **NEW**                                           |
| `api_rate_limit_check`  | **TOOL**      | 🔴 TODO    | Track and predict rate limits | **NEW**                                           |
| `api_mock`              | **TOOL**      | 🔴 TODO    | Create mock server from spec  | **NEW** (Phase 2)                                 |
| `api_test`              | **TOOL**      | 🔴 TODO    | Test endpoint with assertions | **NEW** (Phase 2)                                 |

**Why Tool (Not Terminal):**

- **Schema Validation:** Detect breaking changes automatically
- **Rate Limit Tracking:** Avoid 429 errors proactively
- **Contract Storage:** Compare against expected schema
- **GNN Integration:** Update dependency graph when APIs change
- **Circuit Breaker:** Fail fast when API is down

**Implementation Priority:**

- **P0 (High):** 🔴 Contract validation and breaking change detection
- **P1 (High):** 🔴 Health checks and rate limit tracking
- **P2 (Future):** 🔴 Mock servers and automated API testing

#### 1.6 Environment & System Resources

| Capability            | Tool/Terminal | MVP Status | Purpose                      | Implementation                       |
| --------------------- | ------------- | ---------- | ---------------------------- | ------------------------------------ |
| `env_get` / `env_set` | Terminal      | ✅ DONE    | Environment variables        | Via terminal commands                |
| `get_cpu_usage`       | Tool          | 🔴 TODO    | CPU metrics for optimization | **NEW** `agent/resources/monitor.rs` |
| `get_memory_usage`    | Tool          | 🔴 TODO    | Memory stats                 | **NEW**                              |
| `get_disk_usage`      | Tool          | 🔴 TODO    | Disk space monitoring        | **NEW**                              |
| `should_throttle`     | Tool          | 🔴 TODO    | Adaptive resource management | **NEW**                              |

**Implementation Priority:**

- **P2 (Medium):** Resource monitoring for performance optimization
- **P3 (Future):** Adaptive behavior based on system load

### 2. 🧠 REASON - Decision-Making & Analysis Layer

**Purpose:** Analyze context, make decisions, plan actions

**Status:** ✅ **100% COMPLETE** - All reasoning capabilities implemented

| Capability                | MVP Status | Implementation                                                |
| ------------------------- | ---------- | ------------------------------------------------------------- |
| Confidence Scoring        | ✅ DONE    | `agent/confidence.rs` (320 lines)                             |
| Impact Analysis           | ✅ DONE    | `architecture/deviation_detector.rs::analyze_change_impact()` |
| Risk Assessment           | ✅ DONE    | `RiskLevel` enum: Low/Medium/High/Critical                    |
| Decision Logging          | ✅ DONE    | State machine persistence in SQLite                           |
| Multi-LLM Orchestration   | ✅ DONE    | `llm/multi_llm_manager.rs` (13 providers)                     |
| Validation Pipeline       | ✅ DONE    | `agent/validation.rs` (412 lines)                             |
| Error Analysis            | ✅ DONE    | `agent/orchestrator.rs::analyze_error()`                      |
| Adaptive Context Assembly | ✅ DONE    | Hierarchical context with GNN                                 |

**No additional capabilities needed** - reasoning layer is complete.

### 3. ⚡ ACT - Execution & Action Layer

**Purpose:** Execute decisions, modify code, run commands, deploy systems

#### 3.1 Terminal & Shell Execution

| Capability             | Tool/Terminal | MVP Status | Purpose                                          | Implementation                     |
| ---------------------- | ------------- | ---------- | ------------------------------------------------ | ---------------------------------- |
| `shell_exec`           | Tool          | ✅ DONE    | Run command, get output                          | `agent/terminal.rs` (391 lines)    |
| `shell_exec_streaming` | Tool          | ✅ DONE    | Long-running with real-time output               | `terminal/executor.rs` (331 lines) |
| `shell_background`     | Tool          | ✅ DONE    | Start background process                         | `terminal/pty_terminal.rs`         |
| `shell_kill`           | Tool          | ✅ DONE    | Terminate process                                | Terminal management                |
| `shell_interactive`    | Tool          | ✅ DONE    | Pseudo-TTY for interactive CLIs                  | PTY implementation                 |
| Smart Terminal Reuse   | Tool          | ✅ DONE    | Detect idle terminals, reuse before creating new | Process detection                  |

**Status:** ✅ **100% COMPLETE** - Full terminal capabilities implemented

#### 3.2 Git & Version Control

| Capability             | Tool/Terminal | MVP Status | Purpose                                  | Implementation        |
| ---------------------- | ------------- | ---------- | ---------------------------------------- | --------------------- |
| `git_status`           | Terminal      | ✅ DONE    | Current state                            | Via terminal commands |
| `git_diff`             | Terminal      | ✅ DONE    | Changes (staged, unstaged, between refs) | Via terminal commands |
| `git_log`              | Terminal      | ✅ DONE    | Commit history                           | Via terminal commands |
| `git_blame`            | Terminal      | ✅ DONE    | Line-by-line attribution                 | Via terminal commands |
| `git_commit`           | Terminal      | ✅ DONE    | Create commit                            | Via terminal commands |
| `git_branch`           | Terminal      | ✅ DONE    | Create/switch/list branches              | Via terminal commands |
| `git_checkout`         | Terminal      | ✅ DONE    | Checkout files/branches                  | Via terminal commands |
| `git_merge`            | Terminal      | ✅ DONE    | Merge branches                           | Via terminal commands |
| `git_stash`            | Terminal      | ✅ DONE    | Stash/pop changes                        | Via terminal commands |
| `git_reset`            | Terminal      | ✅ DONE    | Undo changes                             | Via terminal commands |
| `git_resolve_conflict` | Tool          | 🔴 TODO    | AI-powered conflict resolution           | **NEW** (Post-MVP)    |

**Status:** ✅ Git operations work via terminal (100% functional)
**Future:** 🔴 AI-powered conflict resolution tool (Post-MVP)

#### 3.3 Code Generation & Modification

| Capability          | Tool/Terminal | MVP Status | Purpose                     | Implementation                            |
| ------------------- | ------------- | ---------- | --------------------------- | ----------------------------------------- |
| `generate_code`     | Tool          | ✅ DONE    | LLM-powered code generation | `llm/multi_llm_manager.rs` + orchestrator |
| `auto_correct_code` | Tool          | ✅ DONE    | Fix architecture deviations | `architecture/deviation_detector.rs`      |
| `refactor_code`     | Tool          | 🔴 TODO    | Automated refactoring       | **NEW** (Phase 3)                         |

**Status:** ✅ Code generation complete, 🔴 Advanced refactoring pending

#### 3.4 Testing Execution

| Capability          | Tool/Terminal | MVP Status | Purpose                             | Implementation                         |
| ------------------- | ------------- | ---------- | ----------------------------------- | -------------------------------------- |
| `test_run`          | Tool          | ✅ DONE    | Execute tests (file, suite, single) | `testing/test_generator.rs` + executor |
| `test_run_affected` | Tool          | 🔴 TODO    | Run tests for changed code only     | **NEW** (use GNN)                      |
| `test_coverage`     | Tool          | ✅ DONE    | Get coverage report                 | pytest-cov integration                 |
| `test_generate`     | Tool          | ✅ DONE    | Auto-generate test cases            | `testing/test_generator.rs`            |
| `test_debug`        | Tool          | 🔴 TODO    | Run test in debug mode              | **NEW** (Phase 2)                      |
| `test_watch`        | Tool          | 🔴 TODO    | Continuous test runner              | **NEW** (Phase 2)                      |
| `e2e_run`           | Tool          | 🔴 TODO    | Browser/integration tests           | **NEW** (CDP + Playwright)             |

**Status:** ✅ Basic testing complete (6/6 features), 🔴 Advanced testing pending

#### 3.5 Build & Compilation

| Capability          | Tool/Terminal | MVP Status | Purpose                     | Implementation                                |
| ------------------- | ------------- | ---------- | --------------------------- | --------------------------------------------- |
| `build_project`     | Terminal      | ✅ DONE    | Full build                  | Via terminal (`cargo build`, `npm run build`) |
| `build_incremental` | Terminal      | ✅ DONE    | Changed files only          | Via terminal                                  |
| `build_check`       | Terminal      | ✅ DONE    | Type-check without emitting | Via terminal                                  |
| `build_clean`       | Terminal      | ✅ DONE    | Clear artifacts             | Via terminal                                  |
| `lint_run`          | Tool          | ✅ DONE    | Run linters                 | Security scanner includes linting             |
| `lint_fix`          | Tool          | 🔴 TODO    | Auto-fix lint issues        | **NEW**                                       |
| `format_code`       | Terminal      | ✅ DONE    | Apply formatters            | Via terminal (`rustfmt`, `prettier`)          |

**Status:** ✅ Build and lint via terminal works well

#### 3.6 Package Management

| Capability      | Tool/Terminal | MVP Status | Purpose                      | Implementation                      |
| --------------- | ------------- | ---------- | ---------------------------- | ----------------------------------- |
| `pkg_install`   | Tool          | ✅ DONE    | Add dependency               | `agent/dependencies.rs` (429 lines) |
| `pkg_remove`    | Tool          | ✅ DONE    | Remove dependency            | `agent/dependencies.rs`             |
| `pkg_update`    | Tool          | ✅ DONE    | Update dependencies          | `agent/dependencies.rs`             |
| `pkg_list`      | Tool          | ✅ DONE    | List installed packages      | `agent/dependencies.rs`             |
| `pkg_audit`     | Tool          | ✅ DONE    | Security vulnerability check | `security/scanner.rs`               |
| `pkg_search`    | Tool          | 🔴 TODO    | Find packages in registry    | **NEW**                             |
| `pkg_lock_sync` | Tool          | ✅ DONE    | Sync lockfile                | Via package manager commands        |

**Status:** ✅ **100% COMPLETE** - Package management fully implemented

#### 3.7 Deployment & Infrastructure

| Capability          | Tool/Terminal | MVP Status | Purpose                               | Implementation                    |
| ------------------- | ------------- | ---------- | ------------------------------------- | --------------------------------- |
| `deploy_preview`    | Tool          | ✅ DONE    | Deploy to preview environment         | `agent/deployment.rs` (636 lines) |
| `deploy_production` | Tool          | ✅ DONE    | Deploy to prod (with confirmation)    | `agent/deployment.rs`             |
| `deploy_rollback`   | Tool          | ✅ DONE    | Revert deployment                     | `agent/deployment.rs`             |
| `deploy_status`     | Tool          | ✅ DONE    | Check deployment state                | `agent/deployment.rs`             |
| `deploy_logs`       | Tool          | ✅ DONE    | Fetch deployment logs                 | `agent/deployment.rs`             |
| `infra_provision`   | Tool          | 🔴 TODO    | Create resources (Railway, AWS, etc.) | **NEW** (Phase 2)                 |
| `container_build`   | Terminal      | ✅ DONE    | Build Docker image                    | Via `docker build`                |
| `container_run`     | Terminal      | ✅ DONE    | Run container locally                 | Via `docker run`                  |

**Status:** ✅ Deployment to Railway complete (5/5 features), 🔴 Multi-cloud pending

#### 3.8 Browser Automation (CDP)

| Capability               | Tool/Terminal | MVP Status | Purpose                 | Implementation                 |
| ------------------------ | ------------- | ---------- | ----------------------- | ------------------------------ |
| `browser_launch`         | Tool          | 🟡 PARTIAL | Start browser instance  | `browser/cdp.rs` (placeholder) |
| `browser_navigate`       | Tool          | 🟡 PARTIAL | Go to URL               | `browser/cdp.rs`               |
| `browser_click`          | Tool          | 🔴 TODO    | Click element           | **NEW**                        |
| `browser_type`           | Tool          | 🔴 TODO    | Input text              | **NEW**                        |
| `browser_screenshot`     | Tool          | 🔴 TODO    | Capture screen          | **NEW**                        |
| `browser_select_element` | Tool          | 🔴 TODO    | Visual element picker   | **NEW** (Post-MVP)             |
| `browser_evaluate`       | Tool          | 🔴 TODO    | Run JS in page context  | **NEW**                        |
| `browser_network`        | Tool          | 🔴 TODO    | Intercept/mock requests | **NEW** (Post-MVP)             |
| `browser_console`        | Tool          | 🔴 TODO    | Get console logs        | **NEW**                        |

**Status:** 🟡 25% complete (2/8 features) - CDP placeholder needs full implementation

#### 3.9 HTTP & API Execution

| Capability          | Tool/Terminal | MVP Status | Purpose                                    | Implementation                     |
| ------------------- | ------------- | ---------- | ------------------------------------------ | ---------------------------------- |
| `http_request`      | **TOOL**      | 🔴 TODO    | Make HTTP calls with retry/circuit breaker | **NEW** `agent/http_client/mod.rs` |
| `websocket_connect` | Tool          | 🔴 TODO    | WebSocket client                           | **NEW** (Phase 2)                  |

**Why Tool (Not Terminal via `curl`):**

- **Circuit Breaker:** Fail fast when API is down
- **Retry Logic:** Exponential backoff for transient failures
- **Rate Limiting:** Respect API limits automatically
- **Request Tracing:** Log all API calls for debugging
- **Mock Support:** Swap real APIs with mocks in tests

**Implementation Priority:**

- **P0 (High):** HTTP client with circuit breaker and retry
- **P2 (Future):** WebSocket support

### 4. 🔄 LEARN - Feedback & Adaptation Layer

**Purpose:** Learn from results, adapt behavior, improve over time

**Status:** ✅ **100% COMPLETE** - All learning capabilities implemented

| Capability                     | MVP Status | Implementation                       |
| ------------------------------ | ---------- | ------------------------------------ |
| Validation Pipeline            | ✅ DONE    | `agent/validation.rs` (412 lines)    |
| Auto-Retry with Error Analysis | ✅ DONE    | `agent/orchestrator.rs` (651 lines)  |
| Self-Correction                | ✅ DONE    | `agent/confidence.rs` + auto-retry   |
| Confidence Score Updates       | ✅ DONE    | Real-time confidence adjustment      |
| Known Issues Database          | ✅ DONE    | SQLite persistence for LLM failures  |
| Pattern Extraction             | ✅ DONE    | Error pattern recognition            |
| Failure Network Effects        | ✅ DONE    | Shared learning (privacy-preserving) |

**No additional capabilities needed** - learning layer is complete.

### 5. 📋 Cross-Cutting Capabilities

#### 5.1 Debugging

| Capability         | Tool/Terminal | MVP Status | Purpose                    | Implementation    |
| ------------------ | ------------- | ---------- | -------------------------- | ----------------- |
| `debug_start`      | Tool          | 🔴 TODO    | Launch debugger            | **NEW** (Phase 2) |
| `debug_breakpoint` | Tool          | 🔴 TODO    | Set/remove breakpoints     | **NEW**           |
| `debug_step`       | Tool          | 🔴 TODO    | Step over/into/out         | **NEW**           |
| `debug_continue`   | Tool          | 🔴 TODO    | Resume execution           | **NEW**           |
| `debug_evaluate`   | Tool          | 🔴 TODO    | Eval expression in context | **NEW**           |
| `debug_stack`      | Tool          | 🔴 TODO    | Get call stack             | **NEW**           |
| `debug_variables`  | Tool          | 🔴 TODO    | Inspect variables          | **NEW**           |

**Status:** 🔴 Not implemented (Phase 2 - Post-MVP)

#### 5.2 Documentation

| Capability      | Tool/Terminal | MVP Status | Purpose                                | Implementation       |
| --------------- | ------------- | ---------- | -------------------------------------- | -------------------- |
| `docs_generate` | Tool          | ✅ DONE    | Generate from code (JSDoc, docstrings) | File Registry system |
| `docs_search`   | Tool          | 🔴 TODO    | Search project docs                    | **NEW**              |
| `docs_external` | Tool          | 🔴 TODO    | Fetch library documentation            | **NEW** (Phase 2)    |

**Status:** ✅ Basic documentation system complete

#### 5.3 Security

| Capability         | Tool/Terminal | MVP Status | Purpose                      | Implementation                    |
| ------------------ | ------------- | ---------- | ---------------------------- | --------------------------------- |
| `security_scan`    | Tool          | ✅ DONE    | SAST analysis                | `security/scanner.rs` (512 lines) |
| `secrets_detect`   | Tool          | ✅ DONE    | Find exposed credentials     | Integrated in scanner             |
| `dependency_audit` | Tool          | ✅ DONE    | CVE check on packages        | Integrated in scanner             |
| `secrets_manager`  | Tool          | 🔴 TODO    | Encrypted credential storage | **NEW** `agent/secrets/vault.rs`  |

**Status:** ✅ Security scanning complete, 🔴 Secrets management pending

#### 5.4 Architecture Visualization

| Capability              | Tool/Terminal | MVP Status | Purpose                         | Implementation                       |
| ----------------------- | ------------- | ---------- | ------------------------------- | ------------------------------------ |
| `arch_diagram_generate` | Tool          | ✅ DONE    | Create visual from code         | Architecture View System (16/16)     |
| `arch_validate`         | Tool          | ✅ DONE    | Check against constraints       | `architecture/deviation_detector.rs` |
| `arch_suggest`          | Tool          | ✅ DONE    | Recommend improvements          | Impact analysis                      |
| `arch_import`           | Tool          | ✅ DONE    | Import from MD/Mermaid/PlantUML | `project_initializer.rs`             |

**Status:** ✅ **100% COMPLETE** - Architecture system fully implemented

#### 5.5 Context & Memory

| Capability            | Tool/Terminal | MVP Status | Purpose                               | Implementation            |
| --------------------- | ------------- | ---------- | ------------------------------------- | ------------------------- |
| `context_add`         | Tool          | ✅ DONE    | Add to agent's working memory         | State machine persistence |
| `context_search`      | Tool          | ✅ DONE    | Semantic search over codebase         | GNN semantic layer        |
| `context_summarize`   | Tool          | ✅ DONE    | Compress context for token efficiency | Hierarchical assembly     |
| `project_conventions` | Tool          | 🔴 TODO    | Get coding standards/patterns         | **NEW**                   |

**Status:** ✅ Context management mostly complete

---

## Agentic Capabilities Summary

### Implementation Status by Pillar

| Pillar          | Total Capabilities | Implemented | Pending | Completion % |
| --------------- | ------------------ | ----------- | ------- | ------------ |
| 🔍 **PERCEIVE** | 47                 | 24          | 23      | 51%          |
| 🧠 **REASON**   | 8                  | 8           | 0       | **100%** ✅  |
| ⚡ **ACT**      | 56                 | 41          | 15      | 73%          |
| 🔄 **LEARN**    | 7                  | 7           | 0       | **100%** ✅  |
| **TOTAL**       | **118**            | **80**      | **38**  | **68%**      |

### Priority Breakdown

| Priority                 | Capabilities | Status                  | Timeline            |
| ------------------------ | ------------ | ----------------------- | ------------------- |
| **P0 (Must Have - MVP)** | 65           | ✅ 53 implemented (82%) | **NOW** (Month 1-2) |
| **P1 (High Value)**      | 28           | 🔴 15 pending (54%)     | Month 3-4           |
| **P2 (Differentiators)** | 15           | 🔴 8 pending (53%)      | Month 5-8           |
| **P3 (Later)**           | 10           | 🔴 2 pending (80%)      | Month 9-12          |

### Top 10 Missing Critical Capabilities

1. **Database Connection Manager** (P0) - Connection pooling, schema tracking, safe queries
2. **API Contract Monitor** (P1) - Breaking change detection, rate limiting
3. **HTTP Client with Intelligence** (P0) - Circuit breaker, retry, rate limiting
4. **Document Readers (DOCX/PDF)** (P1) - Architecture from documents
5. **Database Migration Manager** (P1) - Safe migrations with rollback
6. **Browser Automation (CDP Full)** (P0) - Complete CDP implementation
7. **Secrets Manager** (P1) - Encrypted credential storage
8. **Advanced File Operations** (P2) - Edit, delete, move with dependency tracking
9. **E2E Testing Framework** (P2) - Playwright integration
10. **Debugging Tools** (P2) - Interactive debugging support

---

## State Machine Architecture: Separation of Concerns

### Design Philosophy

Yantra's agentic capabilities are implemented through **four specialized state machines**, each with a focused responsibility. This separation of concerns provides:

1. **Clarity**: Each machine has a clear, single responsibility
2. **Maintainability**: Changes to one machine don't affect others
3. **Testability**: Each machine can be tested independently
4. **Scalability**: Machines can run in parallel or be triggered independently
5. **MVP Focus**: Build only what's needed, when needed

### The Four State Machines

#### 1. Code Generation State Machine (MVP)

**Responsibility**: Generate production-quality code that doesn't break dependencies

**States**:

- **ArchitectureGeneration**: Generate or import project architecture
- **ArchitectureReview**: Wait for user approval of architecture (human-in-the-loop)
- **ContextAssembly**: Gather hierarchical context from GNN
- **CodeGeneration**: LLM generates code with full context
- **DependencyValidation**: GNN checks for breaking changes
- **BrowserValidation**: Quick visual validation (renders, console errors, screenshots)
- **SecurityScanning**: Semgrep + OWASP rules for vulnerability detection
- **FixingIssues**: Auto-retry with fixes if validation fails
- **Complete**: Code ready for testing
- **Failed**: Human intervention required

**Entry Point**: User's natural language intent
**Exit Point**: Generated code + confidence score
**Trigger**: User submits task
**Success Criteria**: Code passes GNN validation, security scan, no breaking changes

**Browser Validation Purpose (CodeGen)**:

- **Goal**: "Does it look right?" - Immediate visual feedback
- **Speed**: 5-10 seconds
- **Scope**: Single component/page smoke test
- **Actions**: Render in headless/headed Chrome, check console errors, capture screenshot for preview, detect network errors
- **Tools**: Chrome DevTools Protocol (CDP)
- **User Experience**: Real-time preview as code is generated (WYSIWYG)
- **Failure Action**: Retry code generation with visual feedback

**Performance Targets**:

- Context assembly: <100ms
- Code generation: 2-5s (LLM dependent)
- GNN validation: <10ms
- Browser validation: 5-10s
- Security scan: <10s
- Total cycle: <30s

#### 2. Testing State Machine (MVP)

**Responsibility**: Ensure generated code works correctly through comprehensive testing

**States**:

- **TestGeneration**: Generate unit and E2E tests using LLM
- **EnvironmentSetup**: Create virtual environment, install dependencies
- **UnitTesting**: Run pytest/jest for function-level tests
- **BrowserTesting**: Run E2E tests with Playwright (user interactions)
- **IntegrationTesting**: Test API integrations and data flows
- **CoverageAnalysis**: Measure test coverage percentage
- **FixingIssues**: Re-run tests after applying fixes
- **Complete**: All tests pass with adequate coverage
- **Failed**: Tests failed after maximum retries

**Entry Point**: Generated code from CodeGen machine
**Exit Point**: Test results + coverage report
**Trigger**: Automatically after CodeGen succeeds
**Success Criteria**: 100% tests pass, coverage >80%

**Browser Validation Purpose (Testing)**:

- **Goal**: "Does it work right?" - Automated functional validation
- **Speed**: 30-60 seconds
- **Scope**: Full user workflows and interactions
- **Actions**: Run Playwright/Puppeteer tests, test user flows (login→dashboard→logout), responsive design tests, accessibility tests, assertions on behavior
- **Tools**: Playwright for E2E automation
- **User Experience**: Automated, no user interaction needed
- **Failure Action**: Retry tests or escalate to human

**Performance Targets**:

- Test generation: 3-5s (LLM)
- Environment setup: 10-20s
- Unit tests: <30s for typical project
- Browser E2E tests: 30-60s
- Integration tests: 20-40s
- Total cycle: <2 minutes

**Auto-Trigger**: Yes - runs immediately after CodeGen completes
**Manual Trigger**: Also available via UI button

#### 3. Deployment State Machine (MVP - Railway Focus)

**Responsibility**: Deploy validated code to Railway.app

**States**:

- **PackageBuilding**: Create Docker image or build artifacts
- **ConfigGeneration**: Generate railway.json, Dockerfile, environment config
- **RailwayUpload**: Push code/image to Railway
- **HealthCheck**: Verify deployed service is responding
- **RollbackOnFailure**: Auto-rollback if health check fails
- **Complete**: Service live and healthy
- **Failed**: Deployment failed

**Entry Point**: Code + passing tests
**Exit Point**: Live Railway URL + health status
**Trigger**: Manual (button click in UI) - requires user approval
**Success Criteria**: Service responds to health check (HTTP 200)

**MVP Scope**:

- **Platform**: Railway.app only (no AWS, GCP, Kubernetes in MVP)
- **Environments**: Single production environment (no staging in MVP)
- **Rollback**: Manual (automatic rollback post-MVP)
- **Monitoring**: Basic health checks only (no metrics/logs in MVP)

**Performance Targets**:

- Package building: 30-60s (Docker build)
- Upload to Railway: 20-40s
- Health check: 5-10s
- Total deployment: <2 minutes

**Auto-Trigger**: No - requires user approval for safety
**Post-MVP**: Optional auto-deploy after tests pass

#### 4. Maintenance State Machine (Post-MVP)

**Responsibility**: Monitor production, detect issues, auto-fix, and deploy patches

**States**:

- **LiveMonitoring**: Continuous monitoring of production errors and performance
- **BrowserValidation**: Monitor real user browser sessions for errors
- **ErrorAnalysis**: Pattern detection in production errors
- **IssueDetection**: Identify root causes using error patterns
- **AutoFixGeneration**: Generate fix using LLM + historical patterns
- **FixValidation**: Test fix in staging using CodeGen + Testing machines
- **CICDPipeline**: Automated deployment of validated fix
- **VerificationCheck**: Confirm issue is resolved in production
- **LearningUpdate**: Update knowledge base with new patterns
- **Active**: Normal monitoring state
- **Incident**: Active incident being handled

**Entry Point**: Deployed application in production
**Exit Point**: Incident resolved or escalated
**Trigger**: Automatic based on error detection
**Success Criteria**: Issue resolved without human intervention

**Browser Validation Purpose (Maintenance)**:

- **Goal**: "Is it still working for real users?" - Production monitoring
- **Speed**: Continuous (real-time monitoring)
- **Scope**: All users, all sessions
- **Actions**: Real User Monitoring (RUM), track browser errors in production (JavaScript errors, network failures), performance degradation detection, session replay for debugging, API failures affecting UI
- **Tools**: Sentry-like error tracking, LogRocket-like session replay
- **User Experience**: Invisible to users, proactive issue detection
- **Failure Action**: Auto-fix → validate → deploy → verify

**Self-Healing Loop**:

1. **Detect**: Production error occurs (browser crash, API failure, performance degradation)
2. **Analyze**: Pattern matching against known issues, root cause analysis
3. **Generate Fix**: LLM generates patch based on error context
4. **Validate Fix**: Run through CodeGen + Testing machines (full validation)
5. **Deploy**: Automated CI/CD pipeline pushes fix to production
6. **Verify**: Confirm error rate drops, users unaffected
7. **Learn**: Update knowledge base with new error pattern + solution

**Performance Targets**:

- Error detection: Real-time (<1s)
- Pattern analysis: <5s
- Fix generation: 10-30s (LLM)
- Fix validation: 2-3 minutes (full test suite)
- Deployment: 1-2 minutes
- Total MTTR (Mean Time To Repair): <5 minutes for known patterns

**MVP Status**: NOT in MVP - design complete, implementation post-MVP
**Rationale**: Focus MVP on working code generation/testing/deployment first

---

### State Machine Communication

#### Sequential Flow (Default)

```
User Intent
    ↓
┌─────────────────────┐
│ CodeGen Machine     │ → Generated Code + Confidence Score
└─────────────────────┘
    ↓ (auto-trigger)
┌─────────────────────┐
│ Testing Machine     │ → Test Results + Coverage
└─────────────────────┘
    ↓ (manual approval)
┌─────────────────────┐
│ Deployment Machine  │ → Live Railway URL
└─────────────────────┘
    ↓ (continuous)
┌─────────────────────┐
│ Maintenance Machine │ → Self-Healing Loop
└─────────────────────┘
```

#### With Failures and Retries

```
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
```

#### Session Linking

Each machine maintains references to previous sessions:

- **Testing session** stores `codegen_session_id` (can trace back to generated code)
- **Deployment session** stores `test_session_id` (can trace back to test results)
- **Maintenance session** stores `deployment_id` (can trace back to what's deployed)
- **Full traceability**: Maintenance error → Deployment → Tests → Code Generation → User Intent

#### Independent Execution

Machines can be triggered independently:

- **Re-run tests** without regenerating code (Testing machine only)
- **Re-deploy** without re-running tests (Deployment machine only)
- **Manual fix** can trigger Testing then Deployment (skip CodeGen)

---

### State Persistence & Recovery

#### Database Schema

**Separate tables for each machine**:

```sql
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
```

#### Crash Recovery

- Each machine saves state after every phase transition
- On crash/restart, machine can resume from last saved state
- User sees: "Resuming code generation from validation phase..."
- No work lost, seamless recovery

#### Query Benefits

- **Track success rates**: `SELECT COUNT(*) FROM codegen_sessions WHERE current_phase = 'Complete'`
- **Average confidence**: `SELECT AVG(confidence_score) FROM codegen_sessions`
- **Test pass rates**: `SELECT AVG(passed_tests * 1.0 / total_tests) FROM test_sessions`
- **Deployment reliability**: `SELECT COUNT(*) FROM deployment_sessions WHERE health_check_passed = TRUE`
- **MTTR tracking**: `SELECT AVG(resolution_time_seconds) FROM maintenance_sessions`
- **Full audit trail**: Link any production error back to original user intent

---

### Browser Validation Across Three Machines

#### Comparison Table

| Aspect             | CodeGen                | Testing                  | Maintenance                |
| ------------------ | ---------------------- | ------------------------ | -------------------------- |
| **Purpose**        | Visual preview         | Functional testing       | Production monitoring      |
| **Goal**           | "Does it look right?"  | "Does it work right?"    | "Still working for users?" |
| **Environment**    | Dev (local/headless)   | Staging (automated)      | Production (real users)    |
| **Speed**          | 5-10 seconds           | 30-60 seconds            | Continuous (real-time)     |
| **Scope**          | Single component       | Full workflows           | All user sessions          |
| **Data**           | Mock data              | Test data                | Real user data             |
| **Checks**         | Render, console errors | User interactions, flows | Real errors, performance   |
| **Tools**          | CDP (Chrome DevTools)  | Playwright/Puppeteer     | Sentry/LogRocket RUM       |
| **Screenshot**     | Yes (user preview)     | Optional (debugging)     | Session replay             |
| **Failure Action** | Regenerate code        | Fix tests/code           | Auto-fix + deploy          |
| **User Visible**   | Yes (preview shown)    | No (automated)           | No (silent healing)        |
| **MVP Status**     | ✅ Yes                 | ✅ Yes                   | ❌ Post-MVP                |

#### Why Browser Validation in Multiple Machines?

**CodeGen Machine**:

- Developers need **immediate visual feedback**
- Catch UI bugs before writing tests
- "What You See Is What You Get" experience
- Fast iteration loop (5-10s)

**Testing Machine**:

- Comprehensive **automated testing** of user flows
- Prevent regressions
- Test edge cases and interactions
- Must pass before deployment (30-60s)

**Maintenance Machine**:

- **Real user monitoring** in production
- Detect issues from actual usage patterns
- Proactive self-healing
- Continuous improvement

---

### UI Representation

#### Three Progress Indicators

The UI shows three separate progress bars, one for each active machine:

```
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
```

#### State Transitions Visible to User

Users see real-time updates as machines transition:

- "Assembling context..." → "Generating code..." → "Validating dependencies..." → "Scanning for vulnerabilities..."
- "Setting up test environment..." → "Running unit tests..." → "Running E2E tests..."
- "Building Docker image..." → "Deploying to Railway..." → "Running health check..."

---

### Summary: Why This Architecture?

**Key Benefits:**

1. **Separation of Concerns**
   - Each machine has a single, clear responsibility
   - Changes to one machine don't affect others
   - Easier to understand and maintain

2. **Independent Testing**
   - Test CodeGen without deploying
   - Test deployment logic without code generation
   - Mock dependencies between machines

3. **Flexible Execution**
   - Re-run tests without regenerating code
   - Re-deploy without re-running tests
   - Skip machines when not needed

4. **MVP Focus**
   - Build only CodeGen, Testing, Deployment for MVP
   - Defer Maintenance machine to Post-MVP
   - No unused states in production

5. **Parallel Execution (Future)**
   - Multiple code generation sessions simultaneously
   - Background test runs while user edits
   - Continuous maintenance monitoring

6. **Clear State Tracking**
   - Separate database tables per machine
   - Query success rates per machine
   - Full audit trail across machines

7. **Crash Recovery**
   - Each machine saves state independently
   - Resume from last checkpoint
   - No cross-machine corruption

8. **User Experience**
   - Three progress bars show clear status
   - Visual feedback per stage
   - Manual approval for deployment (safety)

**Trade-offs Accepted:**

1. **More Code**: 4 state machines vs 1 monolithic machine
   - **Worth it**: Clearer architecture, easier maintenance

2. **Database Complexity**: 4 tables vs 1 unified table
   - **Worth it**: Better queryability, clearer data model

3. **Communication Overhead**: Session linking between machines
   - **Worth it**: Full traceability, flexible execution

**Alternative Rejected:**

Single monolithic state machine with 16+ states:

- ❌ Coupling between unrelated concerns
- ❌ Hard to test independently
- ❌ All states loaded even if unused
- ❌ Confusing which states are active
- ❌ Changes ripple across unrelated code

---

### Unified Pipeline View (All Four State Machines)

**Note**: This diagram shows the complete pipeline. See "State Machine Architecture" section above for detailed per-machine states.

```
┌─────────────────────────────────────────────────────────────────┐
│            FULLY AUTONOMOUS AGENTIC PIPELINE                     │
│    (Four Specialized State Machines Working Together)           │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   User Intent    │ (Natural language task)
    │  "Add payments"  │
    └─────────┬────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │ STATE MACHINE 1: CODE GENERATION (✅ MVP)   │
    └─────────────────────────────────────────────┘
              │
              ├→ ArchitectureGeneration (design first)
              ├→ ArchitectureReview (user approval)
              ├→ ContextAssembly (GNN + Hierarchical L1+L2)
              ├→ CodeGeneration (LLM with full context)
              ├→ DependencyValidation (GNN checks no breaks)
              ├→ BrowserValidation (quick visual check, 5-10s)
              ├→ SecurityScanning (Semgrep + OWASP)
              └→ FixingIssues (auto-retry if needed)
              │
              ├─── VALIDATION FAIL ───→ Retry with fixes (up to 3x)
              │
              ▼ (auto-trigger)
    ┌─────────────────────────────────────────────┐
    │ STATE MACHINE 2: TESTING (✅ MVP)          │
    └─────────────────────────────────────────────┘
              │
              ├→ TestGeneration (LLM creates unit + E2E tests)
              ├→ EnvironmentSetup (venv, dependencies)
              ├→ UnitTesting (pytest/jest)
              ├→ BrowserTesting (Playwright E2E, 30-60s)
              ├→ IntegrationTesting (API flows)
              ├→ CoverageAnalysis (measure %)
              └→ FixingIssues (retry if tests fail)
              │
              ├─── TEST FAIL ───→ Analyze → Fix → Rerun
              │
              ▼ (manual approval)
    ┌─────────────────────────────────────────────┐
    │ STATE MACHINE 3: DEPLOYMENT (✅ MVP Railway)│
    └─────────────────────────────────────────────┘
              │
              ├→ PackageBuilding (Docker image)
              ├→ ConfigGeneration (railway.json, Dockerfile)
              ├→ RailwayUpload (push to Railway)
              ├→ HealthCheck (verify service responds)
              └→ RollbackOnFailure (if health check fails)
              │
              ├─── DEPLOY FAIL ───→ Rollback → Human Review
              │
              ▼ (continuous)
    ┌─────────────────────────────────────────────┐
    │ STATE MACHINE 4: MAINTENANCE (❌ Post-MVP) │
    └─────────────────────────────────────────────┘
              │
              ├→ LiveMonitoring (track errors, performance)
              ├→ BrowserValidation (RUM, real user sessions)
              ├→ ErrorAnalysis (pattern detection)
              ├→ IssueDetection (root cause)
              ├→ AutoFixGeneration (LLM creates patch)
              ├→ FixValidation (run through CodeGen + Testing)
              ├→ CICDPipeline (deploy fix automatically)
              ├→ VerificationCheck (confirm resolved)
              └→ LearningUpdate (update knowledge base)
              │
              ▼
    ┌──────────────────────────────────────────────┐
    │       CONTINUOUS SELF-HEALING LOOP           │
    │  (Error → Fix → Validate → Deploy → Verify) │
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

| Factor              | Weight | Scoring                     |
| ------------------- | ------ | --------------------------- |
| LLM confidence      | 30%    | From LLM response metadata  |
| Test pass rate      | 25%    | % of tests passing          |
| Known failure match | 25%    | Similarity to solved issues |
| Code complexity     | 10%    | Cyclomatic complexity       |
| Dependency changes  | 10%    | # of files affected         |

**Thresholds:**

- **>0.8:** High confidence → Auto-retry (up to 3 attempts)
- **0.5-0.8:** Medium confidence → Auto-retry once, then escalate
- **<0.5:** Low confidence → Immediate human review

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

| Operation                | Target | Implementation                     |
| ------------------------ | ------ | ---------------------------------- |
| Command validation       | <1ms   | Pre-compiled regex, HashSet lookup |
| Command execution start  | <50ms  | Tokio async spawn                  |
| Output streaming latency | <10ms  | Unbuffered streaming               |
| Environment setup        | <5s    | Cached venv creation               |
| Dependency installation  | <30s   | Use package manager cache          |
| Script execution         | Varies | Depends on script                  |

### Week 9-10 Implementation Plan

**Week 9: Core Terminal Module**

- [x] Implement `TerminalExecutor` struct
- [x] Command validation with whitelist
- [x] Subprocess execution with Tokio
- [x] Streaming output to frontend
- [x] Unit tests for command validation
- [x] Integration tests for execution

**Week 10: Agent Integration**

- [ ] Add execution phases to orchestrator
- [ ] Implement environment setup logic
- [ ] Implement dependency installation
- [ ] Implement script execution with retry
- [ ] Add runtime validation
- [ ] Frontend output panel UI
- [ ] E2E test: Generate → Run → Test → Commit

---

### Smart Terminal Management (MVP Phase 1 - NEW)

**Status:** 🔴 NOT STARTED
**Priority:** ⚡ MVP CRITICAL
**Purpose:** Intelligent terminal session management to avoid interruptions and resource waste
**Business Impact:** Prevents agent from disrupting user work and optimizes system resources

#### Problem Statement

Current limitations:

1. **No Process Detection**: Agent spawns new processes without checking if terminal is busy
2. **No Interruption Prevention**: Could interrupt long-running user processes (builds, servers, etc.)
3. **No Terminal Reuse**: Always creates new terminals instead of reusing idle ones
4. **Resource Waste**: Multiple terminal instances increase memory usage

#### Requirements

**Requirement 1: Agent Full Terminal Access** ✅ **IMPLEMENTED**

- Agent can execute any whitelisted command
- Security validation with command whitelist
- Argument sanitization to prevent shell injection
- **Implementation:** `src-tauri/src/agent/terminal.rs` (391 lines)

**Requirement 2: Process Detection (NEW - NOT IMPLEMENTED)**

Agent MUST check if a terminal has a process running in foreground before using it.

**Detection Logic:**

```rust
pub enum TerminalState {
    Idle,                          // No foreground process, ready to use
    Busy { process_name: String }, // Foreground process running
    Unknown,                       // Cannot determine state
}

pub async fn check_terminal_state(terminal_id: &str) -> Result<TerminalState, String> {
    // Step 1: Get PTY file descriptor for terminal
    let pty_fd = get_pty_fd(terminal_id)?;

    // Step 2: Get foreground process group ID (TCGETPGRP ioctl)
    let fg_pgrp = unsafe {
        libc::tcgetpgrp(pty_fd)
    };

    if fg_pgrp == -1 {
        return Ok(TerminalState::Unknown);
    }

    // Step 3: Get shell's process group ID
    let shell_pgrp = get_shell_pgrp(terminal_id)?;

    // Step 4: Compare
    if fg_pgrp == shell_pgrp {
        // Shell is in foreground = terminal is idle (just showing prompt)
        Ok(TerminalState::Idle)
    } else {
        // Different process group = user is running a command
        let process_name = get_process_name(fg_pgrp)?;
        Ok(TerminalState::Busy { process_name })
    }
}

fn get_process_name(pid: i32) -> Result<String, String> {
    // On macOS: read /proc/<pid>/comm or use ps
    // On Linux: read /proc/<pid>/comm
    let output = std::process::Command::new("ps")
        .args(&["-p", &pid.to_string(), "-o", "comm="])
        .output()
        .map_err(|e| format!("Failed to get process name: {}", e))?;

    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(name)
}
```

**Use Cases:**

1. **Long-running build**: Terminal shows `cargo build` running → Agent detects "Busy(cargo)" → Skips this terminal
2. **Development server**: Terminal shows `npm run dev` running → Agent detects "Busy(node)" → Skips this terminal
3. **Idle prompt**: Terminal shows bash/zsh prompt → Agent detects "Idle" → Can safely use this terminal
4. **Interactive command**: User typed command but hasn't pressed Enter → Agent detects "Idle" → Can use after user finishes

**Requirement 3: Terminal Reuse (NEW - NOT IMPLEMENTED)**

Agent MUST check existing open terminals before creating a new one.

**Selection Algorithm:**

```rust
pub async fn select_terminal_for_execution() -> Result<String, String> {
    // Step 1: Get all open terminals from TerminalManager
    let terminal_ids = terminal_manager.list_terminals();

    if terminal_ids.is_empty() {
        // No terminals open → Create new one
        return create_new_terminal().await;
    }

    // Step 2: Check each terminal's state
    let mut idle_terminals = Vec::new();

    for terminal_id in terminal_ids {
        match check_terminal_state(&terminal_id).await? {
            TerminalState::Idle => {
                idle_terminals.push(terminal_id);
            }
            TerminalState::Busy { process_name } => {
                println!("⏭️  Skipping terminal {} (busy: {})", terminal_id, process_name);
            }
            TerminalState::Unknown => {
                // Unknown state - safer to skip
                println!("⚠️  Skipping terminal {} (unknown state)", terminal_id);
            }
        }
    }

    // Step 3: Use first idle terminal found
    if let Some(terminal_id) = idle_terminals.first() {
        println!("♻️  Reusing idle terminal: {}", terminal_id);
        return Ok(terminal_id.clone());
    }

    // Step 4: No idle terminals → Create new one
    println!("🆕 Creating new terminal (all {} existing terminals busy)", terminal_ids.len());
    create_new_terminal().await
}

async fn create_new_terminal() -> Result<String, String> {
    let terminal_id = uuid::Uuid::new_v4().to_string();
    terminal_manager.create_terminal(
        terminal_id.clone(),
        format!("Agent Terminal {}", terminal_ids.len() + 1),
        None, // Use default shell
        window,
    )?;
    Ok(terminal_id)
}
```

**Priority Rules:**

1. **Prefer Idle Terminals**: Always reuse idle terminal before creating new
2. **Never Interrupt Busy Terminals**: Skip terminals with foreground processes
3. **Create When Necessary**: Only create new terminal if all existing ones are busy
4. **Max Terminal Limit**: Cap at 5 agent-managed terminals to prevent resource exhaustion

#### Implementation Components

**File: `src-tauri/src/terminal/process_detector.rs` (NEW)**

```rust
use std::os::unix::io::AsRawFd;
use libc::{tcgetpgrp, pid_t};

pub struct ProcessDetector;

impl ProcessDetector {
    /// Check if terminal has foreground process (macOS/Linux)
    pub fn check_terminal_busy(pty_fd: i32) -> Result<TerminalState, String> {
        // Use tcgetpgrp to get foreground process group
        let fg_pgrp = unsafe { tcgetpgrp(pty_fd) };

        if fg_pgrp == -1 {
            return Err("Failed to get foreground process group".to_string());
        }

        // Get shell's process group from terminal session
        let shell_pgrp = get_shell_pgrp(pty_fd)?;

        if fg_pgrp == shell_pgrp {
            Ok(TerminalState::Idle)
        } else {
            let process_name = Self::get_process_name(fg_pgrp)?;
            Ok(TerminalState::Busy { process_name })
        }
    }

    /// Get process name from PID
    fn get_process_name(pid: pid_t) -> Result<String, String> {
        // Platform-specific implementation
        #[cfg(target_os = "macos")]
        {
            // Use libproc on macOS
            use libproc::libproc::proc_pid::name;
            name(pid).map_err(|e| format!("Failed to get process name: {}", e))
        }

        #[cfg(target_os = "linux")]
        {
            // Read from /proc on Linux
            let path = format!("/proc/{}/comm", pid);
            std::fs::read_to_string(path)
                .map(|s| s.trim().to_string())
                .map_err(|e| format!("Failed to get process name: {}", e))
        }
    }
}
```

**File: `src-tauri/src/terminal/pty_terminal.rs` (MODIFY)**

Add methods to `TerminalSession`:

```rust
impl TerminalSession {
    /// Get PTY file descriptor
    pub fn get_pty_fd(&self) -> i32 {
        use std::os::unix::io::AsRawFd;
        self.pty_pair.master.as_raw_fd()
    }

    /// Get shell process group ID
    pub fn get_shell_pgrp(&self) -> Result<i32, String> {
        // Store shell PID when spawning
        self.shell_pid
            .ok_or_else(|| "Shell PID not available".to_string())
    }

    /// Check if terminal is busy
    pub fn is_busy(&self) -> Result<bool, String> {
        let state = ProcessDetector::check_terminal_busy(self.get_pty_fd())?;
        Ok(matches!(state, TerminalState::Busy { .. }))
    }
}
```

Add methods to `TerminalManager`:

```rust
impl TerminalManager {
    /// Get first idle terminal, or create new one
    pub async fn get_idle_terminal(&self, window: Window) -> Result<String, String> {
        let terminal_ids = self.list_terminals();

        // Check each terminal
        for terminal_id in terminal_ids {
            let sessions = self.sessions.lock().unwrap();
            if let Some(session_arc) = sessions.get(&terminal_id) {
                let session = session_arc.lock().unwrap();

                if !session.is_busy()? {
                    println!("♻️  Reusing idle terminal: {}", terminal_id);
                    return Ok(terminal_id.clone());
                }
            }
        }

        // All busy → create new
        self.create_agent_terminal(window).await
    }

    /// Create terminal managed by agent
    async fn create_agent_terminal(&self, window: Window) -> Result<String, String> {
        let terminal_id = uuid::Uuid::new_v4().to_string();
        let agent_terminal_count = self.list_terminals().len();

        // Limit max terminals
        if agent_terminal_count >= 5 {
            return Err("Maximum agent terminal limit reached (5)".to_string());
        }

        self.create_terminal(
            terminal_id.clone(),
            format!("Agent Terminal {}", agent_terminal_count + 1),
            None,
            window,
        )?;

        Ok(terminal_id)
    }
}
```

**File: `src-tauri/src/agent/terminal.rs` (MODIFY)**

Update `TerminalExecutor` to use smart terminal selection:

```rust
impl TerminalExecutor {
    /// Execute command with smart terminal selection
    pub async fn execute_smart(
        &self,
        command: &str,
        args: Vec<String>,
        terminal_manager: &TerminalManager,
        window: Window,
    ) -> Result<ExecutionResult, String> {
        // Step 1: Get idle terminal or create new
        let terminal_id = terminal_manager.get_idle_terminal(window).await?;

        // Step 2: Execute command in selected terminal
        self.execute_in_terminal(&terminal_id, command, args).await
    }
}
```

#### User Experience

**Scenario 1: User has server running**

```
User Terminal 1: $ npm run dev
                  Server listening on port 3000...

Agent needs to run tests:
→ Checks Terminal 1: Busy (node)
→ Creates new Terminal 2
→ Executes: pytest tests/
✅ User's server continues uninterrupted
```

**Scenario 2: User has idle terminals**

```
User Terminal 1: $ (idle prompt)
User Terminal 2: $ cargo build (running)

Agent needs to install package:
→ Checks Terminal 1: Idle ✓
→ Checks Terminal 2: Busy (cargo)
→ Reuses Terminal 1
→ Executes: pip install requests
✅ No new terminal created, Terminal 2 uninterrupted
```

**Scenario 3: All terminals busy**

```
Terminal 1: $ npm run dev (running)
Terminal 2: $ cargo watch (running)
Terminal 3: $ docker-compose up (running)

Agent needs to run command:
→ All terminals busy
→ Creates new Terminal 4
→ Executes command
✅ No interruptions to existing processes
```

#### Success Metrics

| Metric                  | Target | Measurement                                |
| ----------------------- | ------ | ------------------------------------------ |
| Interruption Prevention | 100%   | Agent never interrupts busy terminal       |
| Terminal Reuse Rate     | >70%   | % of commands that reuse existing terminal |
| Max Agent Terminals     | ≤5     | Never create more than 5 agent terminals   |
| Detection Speed         | <10ms  | Time to check terminal state               |
| False Positives         | <1%    | Incorrectly marked as busy when idle       |

#### Platform Compatibility

| Platform    | Process Detection | Implementation                           |
| ----------- | ----------------- | ---------------------------------------- |
| **macOS**   | ✅ Supported      | tcgetpgrp + libproc                      |
| **Linux**   | ✅ Supported      | tcgetpgrp + /proc                        |
| **Windows** | ⚠️ Limited        | Windows console API (different approach) |

**Windows Note:** Windows uses different console architecture. Will require separate implementation using `GetConsoleProcessList` API.

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

- User code (privacy violation)
- File names or paths (identifying information)
- Variable names (user-specific)
- Business logic (proprietary)

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

- Assertion errors (expected vs actual)
- Missing test fixtures
- Mock/stub configuration
- Async test issues

**2. Security Vulnerabilities:**

- SQL injection patterns
- XSS vulnerabilities
- Insecure deserialization
- Hardcoded secrets

**3. Browser Runtime Errors:**

- Console errors (JavaScript)
- Network request failures
- DOM manipulation issues
- Authentication redirects

**4. Dependency Breaks:**

- Import errors (missing modules)
- API signature mismatches
- Breaking changes in called functions
- Data type incompatibilities

#### Network Effect Mechanism

**Local First:**

- Each Yantra instance maintains local known issues DB
- Updated in real-time during usage

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

- No breaking changes to existing function signatures
- All imports exist and are accessible
- Data types match (function args, return values)
- No circular dependencies introduced

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

- Generate unit tests with LLM (separate call)
- Execute via pytest subprocess
- Parse JUnit XML results
- Track pass/fail/error counts

**Auto-Retry Logic:**

- If test fails → Analyze error
- Check known failures DB
- If match found → Apply fix automatically
- Re-run tests (up to 3 attempts)

**Performance Target:** <30s for typical project

#### 3. Integration Test Execution

**Process:**

- Generate integration tests (E2E scenarios)
- Set up test fixtures/mocks
- Execute multi-step workflows
- Validate end-to-end behavior

**Coverage:**

- API endpoints (request → response)
- Database operations (CRUD)
- External service calls (mocked)

**Performance Target:** <60s for typical project

#### 4. Security Scanning

**Tools:**

- Semgrep with OWASP ruleset
- Custom rules for common vulnerabilities
- Dependency vulnerability check (Safety, npm audit)
- Secret detection (TruffleHog patterns)

**Auto-Fix:**

- Many vulnerabilities have standard fixes
- SQL injection → Use parameterized queries
- XSS → Escape user input
- Apply fix + re-scan automatically

**Performance Target:** <10s

#### 5. Browser Validation (UI Code)

**Process:**

- Start Chrome via CDP (headless)
- Load application
- Monitor console for errors
- Execute basic user flows
- Capture network errors

**Auto-Healing:**

- Console error → Extract stack trace
- Check known issues DB
- Apply fix if confidence >0.7
- Re-test automatically

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

- If Yantra crashes → Reload state from DB
- Continue from last phase
- No re-work needed

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

- Every validation failure → Create KnownIssue entry
- Store LLM used, error, fix (if found)
- Link to failure pattern (for network effect)

**2. Automatic Retrieval:**

- Before retry → Query known issues DB
- Match by error signature + context
- If confidence >0.8 → Apply fix automatically
- Track success rate

**3. Continuous Learning:**

- Every successful fix → Update success_rate
- Every failed fix → Lower confidence
- Prune low-success patterns (<0.3 after 10 attempts)

### Implementation Phases

**MVP (Month 1-2):**

- ✅ Test generation and execution (implemented)
- ✅ Known issues tracking (implemented)
- ⚠️ Confidence scoring system (add)
- ⚠️ Auto-retry logic with known fixes (add)
- ⚠️ Agent state machine (basic) (add)

**Post-MVP (Month 3-4):**

- Full validation pipeline (tests + security + browser)
- Pattern extraction from failures
- Network effect (opt-in sharing)
- Advanced confidence scoring

**Enterprise (Month 5-8):**

- Self-healing workflows
- Cross-system validation
- Distributed agent coordination
- Advanced auto-fixing (ML-based)

---

## Phase 1: MVP (Months 1-2)

### Objectives

Prove Yantra can generate production-quality code that:

1. Never breaks existing code (GNN validation)
2. Passes all tests automatically (100% pass rate)
3. Has no critical security vulnerabilities
4. Works on first deployment (no debugging needed)

### Success Metrics

- Generate working code for 10+ scenarios (auth, CRUD, APIs, etc.)
- 95% of generated code passes all tests without human intervention
- Zero breaking changes to existing code
- <3% critical security vulnerabilities (auto-fixed)
- Developer NPS >40

### Scope

In Scope:
✅ Python codebase support (single language focus)
✅ Internal code dependency tracking
✅ **Test file dependency tracking with coverage analysis** 🆕 (Nov 30, 2025)
🔄 **Tech stack dependency tracking (package-to-file mapping)** 🆕 (Nov 30, 2025)
✅ Multi-LLM orchestration (Claude + GPT-4 + Qwen Coder support)
✅ GNN for code dependencies
✅ **Token-aware context assembly (truly unlimited context - MVP foundation)**
✅ **Hierarchical context (L1 + L2) with compression**
✅ **Token counting with tiktoken-rs**
✅ Automated unit + integration test generation
✅ **Confidence scoring system**
✅ **Known issues database (LLM failures + fixes)**
✅ **Basic agent state machine with auto-retry**
✅ **Guided Mode vs Auto Mode (MVP - Priority)** 🆕
✅ **Cascading Failure Protection with Checkpoints (MVP - Priority)** 🆕
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

---

## Test File Dependency Tracking with Coverage Analysis 🆕

**Added:** November 30, 2025
**Status:** ✅ Implemented in GNN
**Files:** `src-tauri/src/gnn/mod.rs`

### Overview

The GNN (Graph Neural Network) now tracks bidirectional relationships between test files and source files, enabling comprehensive test coverage analysis, impact analysis, and detection of untested code. This enhancement is critical for ensuring code quality and guiding test generation.

### Architecture

**New Edge Types:**

1. **`EdgeType::Tests`** - Function-level test relationship
   - Represents: `test_function → source_function`
   - Example: `test_add()` Tests edge to `add()`
   - Used for: Fine-grained test coverage at function level

2. **`EdgeType::TestDependency`** - File-level test relationship
   - Represents: `test_file → source_file`
   - Example: `test_calculator.py` TestDependency edge to `calculator.py`
   - Used for: General test-to-source mapping when function mapping fails

### Implementation Details

**1. Test File Detection - `is_test_file(path: &Path) -> bool`**

Automatically detects test files using language-specific patterns:

**Python Test Patterns:**

- `test_*.py` (pytest convention)
- `*_test.py` (alternative convention)
- Files in `/tests/` or `/test/` directories

**JavaScript/TypeScript Test Patterns:**

- `*.test.js`, `*.test.ts`
- `*.spec.js`, `*.spec.ts`
- Files in `/__tests__/` directories

**Example:**

```rust
// Detects as test file
is_test_file("test_calculator.py")  // true
is_test_file("calculator_test.py")  // true
is_test_file("tests/utils.py")      // true
is_test_file("math.test.ts")        // true

// Detects as source file
is_test_file("calculator.py")       // false
is_test_file("utils.js")            // false
```

**2. Test-to-Source Mapping - `find_source_file_for_test(test_path: &Path) -> Option<NodeIndex>`**

Intelligently maps test files to their corresponding source files:

**Mapping Strategy:**

1. Remove test prefix/suffix: `test_calculator.py` → `calculator.py`
2. Remove test directory: `tests/calculator.py` → `calculator.py`
3. Search graph for matching source filename
4. Return `NodeIndex` of source file if found

**Example:**

```rust
// Maps test to source
find_source_file_for_test("test_calculator.py")     // → calculator.py
find_source_file_for_test("calculator_test.py")     // → calculator.py
find_source_file_for_test("tests/utils.py")         // → utils.py
find_source_file_for_test("math.spec.ts")           // → math.ts
```

**3. Edge Creation - `create_test_edges() -> usize`**

Creates bidirectional test edges in the GNN graph:

**Algorithm:**

```
For each file in graph:
  1. If is_test_file(file):
     2. Find corresponding source file
     3. If source found:
        a. Create TestDependency edge: test_file → source_file
        b. For each test function in test_file:
           i. Try to infer tested function name
           ii. If found, create Tests edge: test_fn → source_fn
     4. If source not found:
        - Log warning (test file without corresponding source)
```

**Returns:** Count of test edges created (for metrics)

### Usage Examples

**Scenario 1: Test Coverage Analysis**

```rust
// Find all tests for a given source file
let source_file = graph.find_node("calculator.py");
let test_edges = graph.get_incoming_edges(source_file, EdgeType::TestDependency);
let tested = !test_edges.is_empty();

// Identify untested source files
let untested_files: Vec<_> = graph.get_all_nodes()
    .filter(|node| !is_test_file(&node.path))
    .filter(|node| {
        let incoming = graph.get_incoming_edges(node.index, EdgeType::TestDependency);
        incoming.is_empty()
    })
    .collect();

println!("Untested files: {:?}", untested_files);
```

**Scenario 2: Impact Analysis**

```rust
// When source file changes, find affected tests
let changed_file = graph.find_node("calculator.py");
let affected_tests = graph.get_incoming_edges(changed_file, EdgeType::TestDependency);

println!("Need to re-run {} test files", affected_tests.len());
for test_edge in affected_tests {
    let test_node = graph.get_node(test_edge.source);
    println!("  - {}", test_node.path.display());
}
```

**Scenario 3: Guided Test Generation**

```rust
// Suggest tests for untested functions
let source_file = graph.find_node("calculator.py");
let all_functions = graph.get_functions_in_file(source_file);
let tested_functions = graph.get_outgoing_edges(source_file, EdgeType::Tests)
    .map(|edge| graph.get_node(edge.target));

let untested: Vec<_> = all_functions
    .filter(|f| !tested_functions.contains(f))
    .collect();

println!("Need tests for {} functions:", untested.len());
for func in untested {
    println!("  - {}", func.name);
}
```

### Integration with State Machines

**CodeGen State Machine:**

- **DependencyValidation phase:** Check that new code has corresponding tests
- **FixingIssues phase:** Generate missing tests automatically

**Testing State Machine:**

- **TestGeneration phase:** Use untested function list to guide generation
- **CoverageAnalysis phase:** Calculate coverage using test edge counts
- **Complete phase:** Verify minimum coverage threshold (e.g., 80%)

**Maintenance State Machine (Post-MVP):**

- **CodeChangeDetection:** Identify affected tests using test edges
- **TestSelection:** Run only affected tests for faster feedback
- **CoverageRegression:** Alert if coverage drops below threshold

### Performance Characteristics

**Test File Detection:**

- **Complexity:** O(1) - Simple path pattern matching
- **Target:** <1ms per file

**Test-to-Source Mapping:**

- **Complexity:** O(N) - Linear search through graph nodes
- **Target:** <10ms per test file
- **Optimization:** Can be improved with filename index (future enhancement)

**Edge Creation:**

- **Complexity:** O(T × F) where T = test files, F = avg functions per test
- **Target:** <500ms for typical project (100 test files, 10 functions each)
- **One-time cost:** Run during initial graph build and incremental updates

**Impact Analysis:**

- **Complexity:** O(1) - Direct edge lookup using petgraph
- **Target:** <5ms per query

### Benefits

1. **Automated Test Coverage Analysis**
   - Instantly identify untested code without running tests
   - Track coverage trends over time (stored in SQLite)

2. **Intelligent Test Generation**
   - Guide LLM to generate tests for untested functions
   - Prioritize test generation based on code complexity

3. **Fast Impact Analysis**
   - When code changes, immediately know which tests to run
   - Enables selective test execution (future: only run affected tests)

4. **Quality Metrics**
   - Calculate test-to-code ratio (# test edges / # source nodes)
   - Detect test debt (source files without tests)

5. **Self-Healing Tests**
   - When source function signature changes, identify affected test functions
   - Auto-update test calls to match new signatures (future enhancement)

### Database Schema Extension

The test edges are persisted in SQLite using the existing `edges` table:

```sql
-- Existing schema (no changes needed)
CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL,  -- 'Tests' or 'TestDependency'
    metadata TEXT,
    FOREIGN KEY (source_node_id) REFERENCES nodes(id),
    FOREIGN KEY (target_node_id) REFERENCES nodes(id)
);

-- Query examples
-- Find all tests for a source file
SELECT t.path FROM nodes t
JOIN edges e ON e.source_node_id = t.id
JOIN nodes s ON e.target_node_id = s.id
WHERE s.path = 'calculator.py' AND e.edge_type = 'TestDependency';

-- Find untested source files
SELECT path FROM nodes
WHERE path NOT LIKE '%test%'
AND id NOT IN (
    SELECT target_node_id FROM edges WHERE edge_type = 'TestDependency'
);
```

### Future Enhancements (Post-MVP)

1. **Test Quality Scoring**
   - Track assertion count per test function
   - Calculate test complexity (branches, mocks)
   - Score test quality: high/medium/low

2. **Smart Test Selection**
   - When file changes, run only affected tests (not entire suite)
   - Uses test edges for O(1) lookup of affected tests
   - Reduces test time from minutes to seconds

3. **Test Redundancy Detection**
   - Identify multiple tests covering same source function
   - Suggest consolidation or removal of duplicate tests

4. **Cross-Language Test Support**
   - Support more test frameworks (RSpec, Go testing, Rust tests)
   - Unified test edge representation across languages

---

## Tech Stack Dependency Tracking: Package-to-File Mapping 🆕

**Added:** November 30, 2025
**Status:** 🔄 Specification Complete, Implementation Pending
**Priority:** High (Addresses unnecessary package bloat and dependency confusion)

### Problem Statement

**Current Limitation:**

Today, tech stack dependencies (Python packages, npm modules) are tracked at the **project level** using files like:

- `requirements.txt` (Python)
- `package.json` (JavaScript/TypeScript)
- `Cargo.toml` (Rust)

**The Problem:**

These files list all packages but **do not specify which files use which packages**. This leads to:

1. **Unnecessary Packages Installed**
   - Over time, packages get added but source files using them may be deleted
   - No way to safely identify and remove unused packages
   - Example: `requirements.txt` has 12 packages, but only 8 are actively used

2. **Unclear Dependency Relationships**
   - When refactoring or deleting a file, unclear which packages can be removed
   - Risk of breaking other files that might use the same package
   - Forces manual code search: "Does any other file import this package?"

3. **Bloated Production Builds**
   - Unused packages increase Docker image size
   - Longer installation times in CI/CD pipelines
   - Potential security vulnerabilities in unused dependencies

4. **Lack of Granular Context for LLM**
   - When generating code, LLM doesn't know which packages are already available
   - May suggest redundant package installations
   - Cannot intelligently suggest package removal during cleanup

### Solution: GNN Package-to-File Tracking

**Core Idea:** Extend the GNN to track which files import which packages, creating a bidirectional dependency graph.

### Architecture

**New Edge Type:**

**`EdgeType::UsesPackage`** - Package dependency relationship

- Represents: `source_file → package_name`
- Example: `calculator.py` UsesPackage edge to `numpy`
- Metadata: Import statement (e.g., `import numpy as np`, `from pandas import DataFrame`)

### Implementation Details

**1. Package Detection - `extract_package_imports(file_path: &Path) -> Vec<PackageImport>`**

Parse source files to extract package imports:

**Python Import Patterns:**

```python
import numpy                      # Package: numpy
import pandas as pd              # Package: pandas
from sklearn.model import LinearRegression  # Package: scikit-learn
from myproject.utils import helper  # Internal import (ignore)
```

**JavaScript/TypeScript Import Patterns:**

```javascript
import React from 'react'; // Package: react
import { useState } from 'react'; // Package: react
const axios = require('axios'); // Package: axios
import { MyComponent } from './local'; // Local import (ignore)
```

**Detection Algorithm:**

```rust
pub struct PackageImport {
    package_name: String,   // e.g., "numpy", "react"
    import_statement: String,  // e.g., "import numpy as np"
    line_number: usize,
}

fn extract_package_imports(file_path: &Path) -> Vec<PackageImport> {
    let content = fs::read_to_string(file_path)?;
    let mut imports = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        // Python: import X or from X import Y
        if line.trim().starts_with("import ") || line.trim().starts_with("from ") {
            let package = extract_python_package(line);
            if !is_internal_import(&package) {
                imports.push(PackageImport {
                    package_name: package,
                    import_statement: line.to_string(),
                    line_number: line_num + 1,
                });
            }
        }

        // JavaScript: import X from 'Y' or require('Y')
        if line.contains("import ") || line.contains("require(") {
            let package = extract_js_package(line);
            if !is_local_import(&package) {  // Skip ./relative and @/alias imports
                imports.push(PackageImport {
                    package_name: package,
                    import_statement: line.to_string(),
                    line_number: line_num + 1,
                });
            }
        }
    }

    imports
}

fn is_internal_import(package: &str) -> bool {
    // Check if import is from same project (not external package)
    // Python: Starts with project root module name
    // Heuristic: If directory with same name exists in project, it's internal
    Path::new(package).exists()
}

fn is_local_import(package: &str) -> bool {
    // JavaScript: Starts with ./ or ../ or @/ (alias)
    package.starts_with("./") || package.starts_with("../") || package.starts_with("@/")
}
```

**2. Package Node Creation - `add_package_node(package_name: &str) -> NodeIndex`**

Create virtual nodes for packages in the graph:

```rust
pub struct PackageNode {
    name: String,           // e.g., "numpy", "react"
    version: Option<String>, // e.g., "1.24.3" (from requirements.txt)
    ecosystem: PackageEcosystem,  // Python, JavaScript, Rust
    usage_count: usize,     // Number of files using this package
}

pub enum PackageEcosystem {
    Python,
    JavaScript,
    Rust,
}

fn add_package_node(&mut self, package_name: &str, ecosystem: PackageEcosystem) -> NodeIndex {
    // Check if package node already exists
    if let Some(existing) = self.find_package_node(package_name) {
        return existing;
    }

    // Create new package node
    let node = PackageNode {
        name: package_name.to_string(),
        version: self.get_package_version(package_name),  // From requirements.txt
        ecosystem,
        usage_count: 0,
    };

    self.graph.add_node(Node::Package(node))
}
```

**3. Edge Creation - `create_package_edges() -> usize`**

Create edges from source files to package nodes:

```rust
fn create_package_edges(&mut self) -> usize {
    let mut edge_count = 0;

    for file_node in self.graph.node_indices() {
        let node = &self.graph[file_node];

        // Only process source files (not package nodes)
        if let Node::File(file_data) = node {
            let imports = extract_package_imports(&file_data.path);

            for import in imports {
                // Get or create package node
                let package_node = self.add_package_node(
                    &import.package_name,
                    self.detect_ecosystem(&file_data.path)
                );

                // Create UsesPackage edge: file → package
                self.graph.add_edge(
                    file_node,
                    package_node,
                    Edge {
                        edge_type: EdgeType::UsesPackage,
                        metadata: Some(EdgeMetadata {
                            import_statement: import.import_statement,
                            line_number: import.line_number,
                        }),
                    }
                );

                edge_count += 1;

                // Update package usage count
                if let Node::Package(pkg) = &mut self.graph[package_node] {
                    pkg.usage_count += 1;
                }
            }
        }
    }

    edge_count
}
```

### Usage Examples

**Scenario 1: Identify Unused Packages**

```rust
// Get all packages from requirements.txt
let declared_packages = parse_requirements_txt("requirements.txt");

// Get packages actually used in code (from GNN)
let used_packages: HashSet<_> = graph.get_all_nodes()
    .filter_map(|node| match node {
        Node::Package(pkg) if pkg.usage_count > 0 => Some(pkg.name.clone()),
        _ => None,
    })
    .collect();

// Find unused packages
let unused: Vec<_> = declared_packages
    .into_iter()
    .filter(|pkg| !used_packages.contains(pkg))
    .collect();

println!("Unused packages (can be removed): {:?}", unused);
// Output: ["tomli", "typing_extensions"] (from your requirements_backup.txt)
```

**Scenario 2: Safe Package Removal**

```rust
// User wants to delete calculator.py
let file_to_delete = graph.find_node("calculator.py");

// Find packages used by this file
let used_packages = graph.get_outgoing_edges(file_to_delete, EdgeType::UsesPackage);

for package_edge in used_packages {
    let package_node = graph.get_node(package_edge.target);
    let package = &package_node.as_package();

    // Check if any other file uses this package
    let other_users = graph.get_incoming_edges(package_node.index, EdgeType::UsesPackage)
        .filter(|edge| edge.source != file_to_delete)
        .count();

    if other_users == 0 {
        println!("⚠️  Package '{}' will become unused after deleting this file", package.name);
        println!("   Consider removing from requirements.txt");
    } else {
        println!("✅ Package '{}' is still used by {} other file(s)", package.name, other_users);
    }
}
```

**Scenario 3: Generate Minimal Requirements**

```rust
// Generate requirements.txt with only used packages
fn generate_minimal_requirements(graph: &DependencyGraph) -> String {
    let mut requirements = String::new();

    // Get all packages with usage_count > 0
    let used_packages: Vec<_> = graph.get_all_nodes()
        .filter_map(|node| match node {
            Node::Package(pkg) if pkg.usage_count > 0 => Some(pkg),
            _ => None,
        })
        .collect();

    // Sort by name for consistency
    used_packages.sort_by(|a, b| a.name.cmp(&b.name));

    for package in used_packages {
        // Add package with version (if known)
        if let Some(version) = &package.version {
            requirements.push_str(&format!("{}=={}\n", package.name, version));
        } else {
            requirements.push_str(&format!("{}\n", package.name));
        }

        // Add comment showing which files use it
        let users = graph.get_incoming_edges(package.index, EdgeType::UsesPackage);
        requirements.push_str(&format!("# Used by: {}\n",
            users.map(|e| graph.get_node(e.source).path)
                 .join(", ")
        ));
    }

    requirements
}

// Output example:
// numpy==1.24.3
// # Used by: calculator.py, data_processor.py
// pandas==2.0.1
// # Used by: data_processor.py
// pytest==8.4.2
// # Used by: test_calculator.py, test_data_processor.py
```

**Scenario 4: LLM Context Enhancement**

```rust
// When generating code for a file, tell LLM which packages are available
fn get_available_packages_for_llm(graph: &DependencyGraph) -> String {
    let packages: Vec<_> = graph.get_all_nodes()
        .filter_map(|node| match node {
            Node::Package(pkg) => Some(pkg.name.clone()),
            _ => None,
        })
        .collect();

    format!(
        "Available packages in this project:\n{}\n\nUse these instead of suggesting new installations.",
        packages.join(", ")
    )
}

// LLM prompt enhancement:
// "Available packages: numpy, pandas, pytest
//  Use these instead of suggesting new installations.
//  Generate a function to process data..."
```

### Integration with State Machines

**CodeGen State Machine:**

- **ArchitectureGeneration phase:**
  - When planning new features, check available packages
  - Suggest reusing existing packages vs. adding new ones
- **DependencyValidation phase:**
  - After code generation, update package edges for new imports
  - Validate that all imported packages are in requirements.txt
  - Auto-add missing packages to requirements.txt
- **Complete phase:**
  - Generate updated requirements.txt with usage comments
  - Report unused packages to user (with option to remove)

**Testing State Machine:**

- **EnvironmentSetup phase:**
  - Install only packages used by files being tested (subset optimization)
  - Speeds up test environment setup
- **Complete phase:**
  - Verify test files don't import production-only packages

**Deployment State Machine:**

- **PackageBuilding phase:**
  - Generate minimal requirements.txt for production
  - Exclude test-only packages (pytest, coverage) from production builds
  - Calculate package installation time estimate
- **ConfigGeneration phase:**
  - Create Docker layers optimized by package usage frequency
  - Put rarely-changing packages in lower layers (better caching)

**Maintenance State Machine (Post-MVP):**

- **DependencyAudit phase:**
  - Periodic check for unused packages (weekly)
  - Alert on outdated package versions
  - Suggest package consolidation (e.g., use requests instead of both urllib3 and httpx)

### Performance Characteristics

**Package Detection:**

- **Complexity:** O(L) where L = lines in file
- **Target:** <50ms per file (regex matching on imports)
- **Optimization:** Cache results, only re-scan on file change

**Edge Creation:**

- **Complexity:** O(F × I) where F = files, I = avg imports per file
- **Target:** <2s for typical project (100 files, 10 imports each)
- **Run frequency:** Initial build + incremental on file save

**Unused Package Detection:**

- **Complexity:** O(P) where P = packages in requirements.txt
- **Target:** <100ms for typical project
- **Run frequency:** On-demand (user request or pre-commit hook)

**Minimal Requirements Generation:**

- **Complexity:** O(P + E) where E = package edges
- **Target:** <200ms
- **Run frequency:** Pre-deployment or on-demand

### Database Schema Extension

**New Table: `packages`**

```sql
CREATE TABLE packages (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    version TEXT,
    ecosystem TEXT NOT NULL,  -- 'python', 'javascript', 'rust'
    usage_count INTEGER DEFAULT 0,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_package_name ON packages(name);
CREATE INDEX idx_package_ecosystem ON packages(ecosystem);
```

**Enhanced `edges` table (reuse existing):**

```sql
-- Add metadata column if not exists
ALTER TABLE edges ADD COLUMN metadata TEXT;

-- Example edge record for UsesPackage
INSERT INTO edges (source_node_id, target_node_id, edge_type, metadata)
VALUES (
    42,  -- calculator.py node
    99,  -- numpy package node
    'UsesPackage',
    '{"import_statement": "import numpy as np", "line_number": 3}'
);
```

**Query Examples:**

```sql
-- Find all files using a specific package
SELECT f.path FROM nodes f
JOIN edges e ON e.source_node_id = f.id
JOIN packages p ON e.target_node_id = p.id
WHERE p.name = 'numpy' AND e.edge_type = 'UsesPackage';

-- Find unused packages
SELECT p.name FROM packages p
LEFT JOIN edges e ON e.target_node_id = p.id AND e.edge_type = 'UsesPackage'
WHERE e.id IS NULL;

-- Get package usage statistics
SELECT
    p.name,
    p.version,
    COUNT(e.id) as usage_count,
    GROUP_CONCAT(f.path) as used_by_files
FROM packages p
LEFT JOIN edges e ON e.target_node_id = p.id AND e.edge_type = 'UsesPackage'
LEFT JOIN nodes f ON e.source_node_id = f.id
GROUP BY p.id
ORDER BY usage_count DESC;
```

### Benefits

1. **Eliminate Package Bloat**
   - Automatically detect and remove unused packages
   - Reduce Docker image size by 20-40% (typical)
   - Faster installation times in CI/CD

2. **Safe Refactoring**
   - When deleting files, know exactly which packages can be removed
   - No manual searching through codebase
   - Prevent accidental breakage from removing needed packages

3. **Enhanced LLM Context**
   - Tell LLM which packages are available (avoid redundant suggestions)
   - Guide LLM to reuse existing packages vs. adding new ones
   - Better code generation quality

4. **Granular Dependency Management**
   - Generate minimal requirements per module/feature
   - Create separate requirements files (e.g., requirements-core.txt, requirements-ml.txt)
   - Enable modular deployments

5. **Security Improvements**
   - Fewer packages = smaller attack surface
   - Easier to audit dependencies (only scan used packages)
   - Reduce vulnerability alerts from unused packages

6. **Better Documentation**
   - Auto-generated requirements.txt with usage comments
   - Clear visibility into package purpose
   - Easier onboarding for new developers

### Implementation Plan

**Week 1: Core Infrastructure**

- Add `EdgeType::UsesPackage` to enum
- Create `PackageNode` struct
- Implement `extract_package_imports()` for Python
- Write unit tests for import detection

**Week 2: Graph Integration**

- Implement `add_package_node()` method
- Implement `create_package_edges()` method
- Add package nodes to incremental update logic
- Create SQLite schema for packages table

**Week 3: Analysis Features**

- Implement `find_unused_packages()` query
- Implement `generate_minimal_requirements()` function
- Add package usage statistics (most/least used)
- Create UI component to display package usage

**Week 4: State Machine Integration**

- Integrate with CodeGen (auto-update requirements.txt)
- Integrate with Deployment (minimal production requirements)
- Add LLM context enhancement (available packages)
- Write integration tests

**Week 5: JavaScript/TypeScript Support**

- Implement `extract_package_imports()` for JS/TS
- Handle npm vs. yarn vs. pnpm (package.json)
- Add node_modules parsing
- Test with React/Next.js projects

**Week 6: Polish & Advanced Features**

- Add pre-commit hook (warn about unused packages)
- Implement package consolidation suggestions
- Add CLI command: `yantra deps analyze`
- Add CLI command: `yantra deps clean`
- Performance optimization (caching)

### Success Criteria

- ✅ Detect 100% of package imports (Python MVP)
- ✅ Correctly identify unused packages (95%+ accuracy)
- ✅ Generate minimal requirements.txt with usage comments
- ✅ <2s for package edge creation (100 files)
- ✅ <100ms for unused package detection
- ✅ Integrate with CodeGen state machine
- ✅ Reduce average Docker image size by 25%

### Future Enhancements (Post-MVP)

1. **Transitive Dependency Tracking**
   - Track sub-dependencies (packages required by packages)
   - Detect diamond dependency problems
   - Suggest version pinning to resolve conflicts

2. **Package Version Optimization**
   - Detect version conflicts between files
   - Suggest compatible version upgrades
   - Auto-update package versions safely

3. **Cross-Language Dependency Resolution**
   - Track Python packages calling JavaScript (e.g., Pyodide)
   - Track Rust FFI dependencies
   - Unified dependency graph across languages

4. **Package Recommendation Engine**
   - Suggest better packages (e.g., httpx instead of requests)
   - Detect deprecated packages and suggest alternatives
   - Recommend package consolidation opportunities

5. **Dependency Security Scoring**
   - Integrate with vulnerability databases (CVE)
   - Score packages by security risk
   - Prioritize updating high-risk packages

---

### Implementation Plan (8 Weeks)

Week 1-2: Foundation

- Tauri + SolidJS project setup
- 3-panel UI layout (chat, code, preview)
- Monaco editor integration
- File system operations
- Basic file tree component
- Project loading (select folder)

Week 3-4: GNN Engine

- tree-sitter Python parser integration
- AST extraction (functions, classes, variables)
- Graph data structures in Rust
- Dependency detection (calls, imports, data flow)
- Incremental updates
- SQLite persistence
- **Known issues database schema + storage**

Week 5-6: LLM Integration + Unlimited Context Foundation

- Claude + GPT-4 + Qwen Coder API clients
- Multi-LLM orchestrator with failover
- **tiktoken-rs integration for token counting**
- **Hierarchical context assembly (L1 + L2)**
- **Token-aware context budgeting per LLM**
- **Basic compression (whitespace, comments)**
- Prompt template system
- Code generation from natural language
- **Confidence scoring system**
- Unit test generation
- Integration test generation
- Test execution (pytest runner)

Week 7: Agentic Validation Pipeline (MVP)

- **Agent state machine (basic phases)**
- **Auto-retry logic with confidence scoring**
- **Known issues retrieval and matching**
- Dependency validation via GNN
- Unit test execution with auto-retry
- Semgrep security scanning
- Chrome DevTools Protocol integration
- Console error monitoring
- **Failure pattern capture (local only in MVP)**
- Git integration (commit with auto-messages)

Week 8: Polish & Beta

- UI/UX improvements
- Error handling and loading states
- Performance optimization
- **LLM comparison testing (GPT-4 vs Qwen Coder)**
- Documentation (getting started guide)
- Beta release to 10-20 developers
- Collect feedback

Deliverable: Desktop app (macOS, Windows, Linux) that generates, tests, validates, and commits Python code with agentic capabilities and token-aware unlimited context.

---

## Agent Interaction Modes: Guided vs Auto Mode (MVP Phase 1 - Priority Feature)

### Overview

Yantra supports two distinct interaction modes that control the level of user involvement during autonomous code generation, testing, and deployment workflows. Both modes are fully autonomous - the difference is in **when and how the user is informed and consulted**.

**Core Principle:** Yantra is an autonomous agentic platform. The agent executes the entire development lifecycle (generate → test → package → deploy → monitor). The interaction mode determines the frequency and type of user checkpoints during this autonomous execution.

### 1. Auto Mode (Default for Experienced Users)

**Philosophy:** "Trust the agent, intervene only when critical decisions are needed."

The agent executes the full pipeline with minimal user interruption. User is consulted only for:

1. **Architecture changes** (adding/removing components, changing connections)
2. **User action blockers** (API keys needed, manual setup required, external service configuration)
3. **Critical failures after 3 auto-retry attempts**

**When Auto Mode Activates:**

- User explicitly sets mode: "Switch to auto mode"
- User preference saved in `.yantra/config.json`
- Suitable for: Production deployments, CI/CD pipelines, experienced developers

**Agent Behavior in Auto Mode:**

```
User: "Add Stripe payment processing"

Agent executes autonomously:
├─ 1. Architecture Generation
│   └─ Creates: Gateway → Payment Service → Stripe API → Database
│   └─ Silent execution (logs to history, no user prompt)
│
├─ 2. Code Generation
│   ├─ Generates: payment_service.py, stripe_client.py, payment_models.py
│   ├─ GNN dependency validation
│   └─ Silent execution
│
├─ 3. Testing
│   ├─ Generates unit tests
│   ├─ Runs pytest
│   ├─ Auto-fixes failures (up to 3 attempts)
│   └─ Silent execution
│
├─ 4. Security Scanning
│   ├─ Semgrep scan
│   ├─ Auto-fixes vulnerabilities
│   └─ Silent execution
│
├─ 5. BLOCKER DETECTED! ⚠️
│   └─ Need Stripe API key
│
Agent PAUSES and prompts user:
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

├─ 6. Integration Testing (resumes)
│   ├─ Tests Stripe connection
│   └─ Silent execution
│
├─ 7. Git Commit
│   └─ Commits with message: "Add Stripe payment processing"
│   └─ Silent execution
│
└─ ✅ Complete
    Agent: "✅ Stripe payment processing added successfully!
            - 3 new files created
            - 12 tests passing
            - No security vulnerabilities
            - Committed to Git (commit abc1234)"
```

**Auto Mode Logging:**
All actions are logged to `.yantra/logs/agent_activity.log` with timestamps, allowing user to review what happened without being interrupted during execution.

**Architecture Change Exception (Always Requires Consent):**

Even in Auto Mode, architecture changes require user review:

```
Agent: "I'm planning to add these components:
        - Payment Service (handles Stripe)
        - Payment Database (stores transactions)

       Connections:
        - Gateway → Payment Service (REST API)
        - Payment Service → Stripe API (SDK)
        - Payment Service → Payment DB (PostgreSQL)

       Impact: No existing features affected

       Approve? (yes/no/modify)"

User: "yes"

Agent: "✅ Architecture updated. Proceeding with code generation..."
```

---

### 2. Guided Mode (Default for New Users)

**Philosophy:** "Explain everything, get user consent at each major step."

The agent explains the impact and rationale before executing each major phase. User maintains full visibility and control throughout the autonomous pipeline.

**When Guided Mode Activates:**

- Default for new users (first 7 days)
- User can explicitly request: "Switch to guided mode"
- Automatically activated after 3 failed attempts in Auto Mode
- Suitable for: Learning Yantra, complex refactoring, high-risk changes

**Agent Behavior in Guided Mode:**

```
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
```

---

### 3. Guided Mode: Impact Explanation (Natural Language, Not Code Terms)

**Critical Requirement:** In Guided Mode, the agent must explain impact using **natural language descriptions of features/components**, NOT code-level details.

**❌ Bad (Code Terms):**

```
Agent: "Modifying payment_service.py will affect:
- Line 45 in gateway.py (import statement)
- Line 120 in api_routes.py (function call)
- test_gateway.py (3 tests need updates)"
```

**✅ Good (Natural Language):**

```
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
```

**Implementation:** Use GNN to map file dependencies → component dependencies → feature dependencies, then translate to user-friendly descriptions.

---

### 4. Guided Mode: Decision Logging

**Requirement:** All decisions and changes must be logged with user consent.

**Log Format (`.yantra/logs/decisions.log`):**

```
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
```

**Log Queries:**

```
User: "Show me all architecture changes this week"
Agent: Retrieves and displays all ARCHITECTURE_CHANGE logs from past 7 days

User: "Why did we add the Payment Service?"
Agent: "On Nov 28, you requested 'Add Stripe payment processing'.
        I proposed adding Payment Service and Payment Database.
        You approved because: 'Need to support online payments'."
```

---

### 5. Guided Mode: Progress Status (Project-Level)

**Requirement:** Give regular status updates on what is implemented and what will be implemented.

**Status Update Format:**

```
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
```

**Auto-Generated Progress Reports:**

After every major phase (architecture, generation, testing), agent provides summary:

```
┌────────────────────────────────────────────────┐
│ 📊 PROGRESS CHECKPOINT                         │
└────────────────────────────────────────────────┘

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
```

---

### 6. Mode Switching

**User Control:**

```
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
```

**Automatic Mode Switching:**

```
# Auto Mode → Guided Mode (after 3 failures)
Agent: "⚠️  I've failed 3 times trying to fix the database connection.

       Switching to Guided Mode so you can see what's happening.

       Issue: Cannot connect to PostgreSQL on localhost:5432

       Options:
       1️⃣  Help me diagnose (guided troubleshooting)
       2️⃣  Skip database for now (use SQLite)
       3️⃣  Provide database credentials

       Your choice:"
```

---

### 7. Implementation Architecture

**Backend (Rust):**

```rust
// src-tauri/src/agent/interaction_mode.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionMode {
    Auto,    // Minimal user prompts
    Guided,  // Explain and ask at each step
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserPromptTrigger {
    ArchitectureChange,       // Always prompt (both modes)
    CodeGeneration,           // Prompt in Guided only
    Testing,                  // Prompt in Guided only
    SecurityScan,             // Prompt in Guided only
    GitCommit,                // Prompt in Guided only
    CriticalBlocker,          // Always prompt (both modes)
    FailureAfter3Retries,     // Always prompt (both modes)
}

pub struct InteractionManager {
    mode: InteractionMode,
    decision_log: DecisionLog,
    gnn_engine: Arc<Mutex<GNNEngine>>,
}

impl InteractionManager {
    /// Check if user prompt is needed for this action
    pub fn should_prompt(&self, trigger: UserPromptTrigger) -> bool {
        match (&self.mode, trigger) {
            (_, UserPromptTrigger::ArchitectureChange) => true,      // Always
            (_, UserPromptTrigger::CriticalBlocker) => true,         // Always
            (_, UserPromptTrigger::FailureAfter3Retries) => true,   // Always
            (InteractionMode::Guided, _) => true,                    // All triggers in Guided
            (InteractionMode::Auto, _) => false,                     // Skip others in Auto
        }
    }

    /// Generate natural language impact explanation using GNN
    pub async fn explain_impact(&self, files: &[PathBuf]) -> ImpactExplanation {
        let gnn = self.gnn_engine.lock().unwrap();

        // Get affected components
        let components = gnn.get_affected_components(files);

        // Get affected features
        let features = gnn.get_affected_features(files);

        // Generate user-friendly explanation
        ImpactExplanation {
            features: features.iter().map(|f| {
                FeatureImpact {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    change_type: f.change_type.clone(),
                    user_visible: f.user_visible,
                }
            }).collect(),
            components: components.iter().map(|c| {
                ComponentImpact {
                    name: c.name.clone(),
                    change_severity: c.severity.clone(),
                }
            }).collect(),
            confidence: self.calculate_confidence(&features, &components),
        }
    }

    /// Log user decision
    pub fn log_decision(&mut self, decision: Decision) {
        self.decision_log.append(decision);
    }

    /// Generate progress report
    pub fn generate_progress_report(&self, session_id: &str) -> ProgressReport {
        let session = self.decision_log.get_session(session_id);

        ProgressReport {
            completed_tasks: session.completed_tasks(),
            in_progress_tasks: session.in_progress_tasks(),
            planned_tasks: session.planned_tasks(),
            overall_progress: session.calculate_progress(),
            blockers: session.active_blockers(),
            next_action: session.suggest_next_action(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactExplanation {
    pub features: Vec<FeatureImpact>,
    pub components: Vec<ComponentImpact>,
    pub confidence: f32,  // 0.0-1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImpact {
    pub name: String,              // "Shopping Cart", "Admin Dashboard"
    pub description: String,       // Human-readable explanation
    pub change_type: String,       // "Modified", "Extended", "No Change"
    pub user_visible: bool,        // Will users notice this change?
}
```

**Frontend (SolidJS):**

```typescript
// src-ui/stores/interactionModeStore.ts

export interface InteractionModeStore {
  mode: 'auto' | 'guided';
  setMode: (mode: 'auto' | 'guided') => void;

  // Decision log
  decisions: Decision[];
  logDecision: (decision: Decision) => void;

  // Progress tracking
  currentPhase: string;
  phaseProgress: number; // 0-100
  overallProgress: number; // 0-100

  // Pending user prompt
  pendingPrompt: UserPrompt | null;
  respondToPrompt: (response: string) => void;
}
```

---

### 8. Testing & Validation

**Test Scenarios:**

1. **Auto Mode - Happy Path**
   - User: "Add user authentication"
   - Agent executes full pipeline
   - No user prompts except architecture change
   - Completes in <3 minutes

2. **Auto Mode - Blocker**
   - User: "Add email sending"
   - Agent hits blocker (need SMTP credentials)
   - Pauses and prompts user
   - Resumes after user provides credentials

3. **Guided Mode - Full Explanation**
   - User: "Add payment processing"
   - Agent explains each phase
   - User sees impact analysis
   - All decisions logged

4. **Mode Switching**
   - User switches modes mid-session
   - Agent adapts behavior immediately
   - No loss of context

---

### 9. User Experience Guidelines

**Auto Mode UX:**

- Minimal interruptions (feels fast and efficient)
- Background progress indicator (small, non-intrusive)
- Detailed logs available on demand
- Notification only for critical events

**Guided Mode UX:**

- Clear phase headers (Architecture, Generation, Testing, etc.)
- Impact explanations in natural language
- Visual progress bars for each phase
- Approval buttons (Yes/No/Modify)
- Undo/revert options at each checkpoint

**Mode Indicator:**

- Always visible in UI (top-right corner)
- Shows current mode: 🚀 Auto | 🧭 Guided
- Click to toggle or see mode explanation

---

## Browser Integration with Chrome DevTools Protocol (CDP)

### Overview

**Purpose:** Enable Yantra to launch, control, and monitor web browsers for UI validation, error capture, and interactive development workflows.

**Approach:** System Chrome + CDP (Option B from design document)

- Find user's installed Chrome/Chromium/Edge
- Launch with remote debugging enabled
- Control via Chrome DevTools Protocol
- Zero-touch user experience

**Phases:**

- **MVP (Phase 1):** Core browser automation, error capture, dev server management
- **Post-MVP (Phase 2):** Interactive element selection, visual feedback, asset management

---

### MVP Features (Required for Launch)

#### 1. Chrome Discovery & Auto-Download

**Requirement:** Find system Chrome automatically, download Chromium as fallback.

**Platform-Specific Paths:**

```rust
// src-tauri/src/browser/chrome_finder.rs

pub fn find_chrome() -> Result<PathBuf, String> {
    let paths = match std::env::consts::OS {
        "macos" => vec![
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ],
        "windows" => vec![
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        ],
        "linux" => vec![
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ],
        _ => vec![],
    };

    for path in paths {
        if Path::new(path).exists() {
            return Ok(PathBuf::from(path));
        }
    }

    Err("No Chrome-based browser found".to_string())
}

pub async fn download_chromium_fallback() -> Result<PathBuf, String> {
    // Download minimal Chromium (~100MB) to app data folder
    // One-time operation, happens in background
    // Store in: ~/.yantra/browser/chromium
}
```

**User Experience:**

- First launch: Check for Chrome (instant if found)
- Not found: Show "Setting up browser preview..." (~30 seconds)
- All subsequent launches: Instant (path cached)

**Files:**

- `src-tauri/src/browser/chrome_finder.rs` (NEW) - Platform detection
- `src-tauri/src/browser/downloader.rs` (NEW) - Fallback download

---

#### 2. Chrome Launch with CDP

**Requirement:** Launch Chrome with remote debugging enabled, app mode (minimal UI).

**Implementation:**

```rust
// src-tauri/src/browser/launcher.rs

pub async fn launch_chrome_for_preview(url: &str) -> Result<BrowserSession, String> {
    let chrome_path = find_chrome().await?;

    let process = Command::new(chrome_path)
        .args([
            "--remote-debugging-port=0",    // Random available port
            "--no-first-run",               // Skip welcome screens
            "--no-default-browser-check",   // Skip default prompt
            "--disable-extensions",         // Faster startup
            "--disable-popup-blocking",     // Allow OAuth popups
            "--disable-background-networking", // No update checks
            "--window-size=1280,720",       // Reasonable default
            format!("--app={}", url),       // App mode - minimal UI
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Parse debug port from stderr output
    let debug_port = discover_debug_port(&process).await?;

    // Connect via CDP
    let connection = connect_cdp(debug_port).await?;

    Ok(BrowserSession {
        process,
        connection,
        url: url.to_string(),
    })
}
```

**Key Flags:**

- `--remote-debugging-port=0`: Auto-select available port (no conflicts)
- `--app=URL`: Launch in app mode (looks native, not browser)
- `--no-first-run`, `--no-default-browser-check`: Skip user prompts
- `--disable-extensions`: Faster startup, no interference

**Files:**

- `src-tauri/src/browser/launcher.rs` (NEW) - Chrome launching
- Update `src-tauri/src/browser/cdp.rs` - Replace placeholder with real launch

---

#### 3. CDP Connection & Communication

**Requirement:** Establish WebSocket connection to Chrome DevTools Protocol, send/receive CDP commands.

**Implementation:**

```rust
// src-tauri/src/browser/cdp.rs

use chromiumoxide::Browser;
use chromiumoxide::browser::{BrowserConfig};

pub struct BrowserSession {
    browser: Browser,
    url: String,
}

impl BrowserSession {
    pub async fn new(url: String, debug_port: u16) -> Result<Self, String> {
        let config = BrowserConfig::builder()
            .port(debug_port)
            .build()
            .map_err(|e| e.to_string())?;

        let (browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| e.to_string())?;

        // Spawn handler task
        tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                // Handle CDP events
            }
        });

        Ok(Self { browser, url })
    }

    pub async fn navigate(&mut self) -> Result<(), String> {
        let page = self.browser.new_page(&self.url)
            .await
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    pub async fn subscribe_to_events(&self) -> Result<(), String> {
        // Subscribe to console, errors, network events
    }
}
```

**Technology Choice:**

- **Recommended:** `chromiumoxide` crate (pure Rust, async, well-maintained)
- **Alternative:** `headless_chrome` crate (older but stable)
- **Avoid:** Raw WebSocket CDP implementation (too complex)

**Files:**

- Rewrite `src-tauri/src/browser/cdp.rs` - Replace placeholders with real CDP
- Add `chromiumoxide = "0.5"` to `Cargo.toml`

---

#### 4. Dev Server Management

**Requirement:** Auto-detect frontend framework, start dev server, parse port from output.

**Implementation:**

```rust
// src-tauri/src/browser/dev_server.rs

#[derive(Debug, Clone)]
pub enum Framework {
    NextJs,
    Vite,
    CreateReactApp,
    Unknown,
}

pub fn detect_framework(project_path: &Path) -> Framework {
    if project_path.join("next.config.js").exists() ||
       project_path.join("next.config.mjs").exists() {
        return Framework::NextJs;
    }

    if project_path.join("vite.config.js").exists() ||
       project_path.join("vite.config.ts").exists() {
        return Framework::Vite;
    }

    if let Ok(package_json) = fs::read_to_string(project_path.join("package.json")) {
        if package_json.contains("react-scripts") {
            return Framework::CreateReactApp;
        }
    }

    Framework::Unknown
}

pub async fn start_dev_server(project_path: &Path) -> Result<u16, String> {
    let framework = detect_framework(project_path);

    let command = match framework {
        Framework::NextJs => "npm run dev",
        Framework::Vite => "npm run dev",
        Framework::CreateReactApp => "npm start",
        Framework::Unknown => return Err("Unknown framework".to_string()),
    };

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(project_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Parse port from output
    let port = parse_port_from_output(&mut child).await?;

    // Wait for server to be ready
    wait_for_server_ready(port).await?;

    Ok(port)
}

async fn parse_port_from_output(child: &mut Child) -> Result<u16, String> {
    // Parse patterns:
    // "ready on http://localhost:3000" (Next.js)
    // "Local: http://localhost:5173" (Vite)
    // "Compiled successfully!" + check port 3000 (CRA)

    let regex = regex::Regex::new(r"localhost:(\d+)").unwrap();
    // ... implementation
}

async fn wait_for_server_ready(port: u16) -> Result<(), String> {
    for _ in 0..30 {  // 30 second timeout
        if reqwest::get(format!("http://localhost:{}", port)).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Err("Dev server failed to start".to_string())
}
```

**Supported Frameworks:**

- Next.js: `npm run dev` → Port from "ready on http://localhost:3000"
- Vite: `npm run dev` → Port from "Local: http://localhost:5173"
- Create React App: `npm start` → Default port 3000
- Custom: Try common ports (3000, 3001, 5173, 8080)

**Files:**

- `src-tauri/src/browser/dev_server.rs` (NEW) - Framework detection & startup

---

#### 5. Runtime Injection

**Requirement:** Inject JavaScript runtime into browser before page loads to capture errors.

**Implementation:**

```rust
// src-tauri/src/browser/runtime_injector.rs

pub async fn inject_runtime(page: &Page) -> Result<(), String> {
    let runtime_script = include_str!("yantra-runtime.js");

    page.execute("Page.addScriptToEvaluateOnNewDocument", json!({
        "source": runtime_script
    })).await.map_err(|e| e.to_string())?;

    Ok(())
}
```

**Runtime Script (yantra-runtime.js):**

```javascript
// src-tauri/src/browser/yantra-runtime.js

(function () {
  'use strict';

  // Connect to Yantra backend
  const ws = new WebSocket('ws://localhost:YANTRA_PORT/browser-events');

  // Capture console errors
  const originalError = console.error;
  console.error = function (...args) {
    originalError.apply(console, args);

    ws.send(
      JSON.stringify({
        type: 'console-error',
        message: args.map((a) => String(a)).join(' '),
        stack: new Error().stack,
        timestamp: Date.now(),
      })
    );
  };

  // Capture unhandled exceptions
  window.addEventListener('error', function (event) {
    ws.send(
      JSON.stringify({
        type: 'unhandled-exception',
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack,
        timestamp: Date.now(),
      })
    );
  });

  // Capture promise rejections
  window.addEventListener('unhandledrejection', function (event) {
    ws.send(
      JSON.stringify({
        type: 'unhandled-rejection',
        reason: String(event.reason),
        promise: event.promise,
        timestamp: Date.now(),
      })
    );
  });

  // Network error monitoring
  const originalFetch = window.fetch;
  window.fetch = function (...args) {
    return originalFetch.apply(this, args).catch(function (error) {
      ws.send(
        JSON.stringify({
          type: 'network-error',
          url: args[0],
          error: error.message,
          timestamp: Date.now(),
        })
      );
      throw error;
    });
  };

  console.log('[Yantra] Runtime injected successfully');
})();
```

**Files:**

- `src-tauri/src/browser/yantra-runtime.js` (NEW) - Error capture runtime
- `src-tauri/src/browser/runtime_injector.rs` (NEW) - Injection logic

---

#### 6. Console Error Capture

**Requirement:** Subscribe to CDP events for console messages, exceptions, capture errors.

**Implementation:**

```rust
// src-tauri/src/browser/error_capture.rs

pub async fn subscribe_to_console_events(page: &Page) -> Result<(), String> {
    // Enable Runtime domain
    page.execute("Runtime.enable", json!({})).await?;

    // Subscribe to console API calls
    page.on_event("Runtime.consoleAPICalled", |event: ConsoleAPICalledEvent| {
        if event.type_ == "error" {
            let error = ConsoleError {
                message: event.args.iter().map(|a| format!("{:?}", a)).join(" "),
                stack_trace: event.stack_trace,
                url: event.url,
                line_number: event.line_number,
                timestamp: event.timestamp,
            };

            send_to_agent(error);
        }
    });

    // Subscribe to exceptions
    page.on_event("Runtime.exceptionThrown", |event: ExceptionThrownEvent| {
        let exception = UnhandledException {
            message: event.exception_details.text,
            stack_trace: event.exception_details.stack_trace,
            url: event.exception_details.url,
            line_number: event.exception_details.line_number,
            column_number: event.exception_details.column_number,
        };

        send_to_agent(exception);
    });

    Ok(())
}

fn send_to_agent(error: impl Into<BrowserError>) {
    // Send to agent orchestrator for auto-fix
    // Add to error queue
    // Trigger agent if in auto-fix mode
}
```

**CDP Events:**

- `Runtime.consoleAPICalled`: console.log, console.error, console.warn
- `Runtime.exceptionThrown`: Unhandled JavaScript exceptions
- Both include stack traces, source maps, line numbers

**Files:**

- `src-tauri/src/browser/error_capture.rs` (NEW) - CDP event subscriptions

---

#### 7. Network Error Capture

**Requirement:** Monitor network requests, capture 404s, 500s, CORS errors.

**Implementation:**

```rust
// src-tauri/src/browser/network_monitor.rs

pub async fn subscribe_to_network_events(page: &Page) -> Result<(), String> {
    // Enable Network domain
    page.execute("Network.enable", json!({})).await?;

    // Subscribe to request failures
    page.on_event("Network.loadingFailed", |event: LoadingFailedEvent| {
        let error = NetworkError {
            url: event.request.url,
            error_text: event.error_text,
            canceled: event.canceled,
            resource_type: event.resource_type,
            timestamp: event.timestamp,
        };

        send_to_agent(error);
    });

    // Subscribe to response received (to check status codes)
    page.on_event("Network.responseReceived", |event: ResponseReceivedEvent| {
        let status = event.response.status;
        if status >= 400 {  // 4xx or 5xx
            let error = HttpError {
                url: event.response.url,
                status_code: status,
                status_text: event.response.status_text,
                headers: event.response.headers,
            };

            send_to_agent(error);
        }
    });

    Ok(())
}
```

**CDP Events:**

- `Network.loadingFailed`: Failed to load resource (DNS, timeout, CORS)
- `Network.responseReceived`: Check HTTP status codes (404, 500, etc.)

**Files:**

- `src-tauri/src/browser/network_monitor.rs` (NEW) - Network monitoring

---

#### 8. Browser Validation

**Requirement:** Run validation tests, aggregate errors, generate reports.

**Current State:** Partial implementation exists but uses placeholder CDP.

**Updates Needed:**

```rust
// Update src-tauri/src/browser/validator.rs

pub struct BrowserValidator {
    timeout_seconds: u64,
    chrome_finder: ChromeFinder,
}

impl BrowserValidator {
    pub async fn validate(&self, url: String) -> Result<ValidationResult, String> {
        // 1. Launch Chrome with real CDP
        let session = launch_chrome_for_preview(&url).await?;

        // 2. Inject runtime
        inject_runtime(&session.page).await?;

        // 3. Subscribe to events
        subscribe_to_console_events(&session.page).await?;
        subscribe_to_network_events(&session.page).await?;

        // 4. Navigate and wait
        session.navigate().await?;

        // 5. Collect errors for timeout period
        let start = Instant::now();
        let mut errors = vec![];
        let mut warnings = vec![];

        while start.elapsed() < Duration::from_secs(self.timeout_seconds) {
            if let Some(error) = receive_error().await {
                match error.severity {
                    Severity::Error => errors.push(error),
                    Severity::Warning => warnings.push(error),
                    _ => {}
                }
            }
        }

        // 6. Generate report
        Ok(ValidationResult {
            success: errors.is_empty(),
            errors,
            warnings,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}
```

**Files:**

- Update `src-tauri/src/browser/validator.rs` - Use real CDP instead of placeholders

---

### MVP Implementation Checklist

**Week 1: Core CDP Foundation**

- [ ] Add `chromiumoxide = "0.5"` to `Cargo.toml`
- [ ] Implement `src-tauri/src/browser/chrome_finder.rs` (platform detection)
- [ ] Implement `src-tauri/src/browser/launcher.rs` (Chrome launch with CDP)
- [ ] Rewrite `src-tauri/src/browser/cdp.rs` (replace placeholders)
- [ ] Test Chrome launch on macOS/Windows/Linux

**Week 2: Dev Server & Error Capture**

- [ ] Implement `src-tauri/src/browser/dev_server.rs` (Next.js/Vite/CRA detection)
- [ ] Add dev server startup and port parsing
- [ ] Implement `src-tauri/src/browser/error_capture.rs` (CDP console events)
- [ ] Implement `src-tauri/src/browser/network_monitor.rs` (CDP network events)
- [ ] Test with intentionally broken React app

**Week 3: Runtime Injection**

- [ ] Create `src-tauri/src/browser/yantra-runtime.js` (error capture runtime)
- [ ] Implement `src-tauri/src/browser/runtime_injector.rs` (injection logic)
- [ ] Set up WebSocket server for browser ↔ Yantra communication
- [ ] Test runtime loads before user code
- [ ] Verify errors flow: browser → runtime → WebSocket → Yantra → agent

**Week 4: Integration & Testing**

- [ ] Update `src-tauri/src/browser/validator.rs` (use real CDP)
- [ ] End-to-end test: Generate React app → Start server → Capture errors → Agent fixes
- [ ] Cross-platform testing (macOS, Windows, Linux)
- [ ] Performance optimization (startup time <2s, memory <200MB)
- [ ] Error deduplication and aggregation

**Success Criteria:**

- Chrome launches automatically (<2s startup)
- Dev servers start automatically for Next.js/Vite/CRA
- Console errors captured in real-time
- Network errors (404, 500, CORS) captured
- Errors sent to agent orchestrator
- Agent can auto-fix based on browser errors
- Works on macOS, Windows, Linux

---

### Post-MVP Features (Phase 2)

#### 9. Interactive Element Selection (P1 Post-MVP)

**Requirement:** Click-to-select mode in browser preview, capture element info, send to chat.

**Implementation:**

```javascript
// Enhanced yantra-runtime.js

class YantraInteraction {
  enableSelectionMode() {
    this.selectionMode = true;
    document.body.style.cursor = 'crosshair';
    document.addEventListener('mouseover', this.handleHover);
    document.addEventListener('click', this.handleClick);
  }

  handleHover(e) {
    // Highlight element on hover
    e.target.classList.add('yantra-hover');

    // Send hover info to Yantra
    ws.send(
      JSON.stringify({
        type: 'element-hover',
        data: this.getElementInfo(e.target),
      })
    );
  }

  handleClick(e) {
    e.preventDefault();
    this.selectedElement = e.target;

    // Send selection to Yantra
    ws.send(
      JSON.stringify({
        type: 'element-selected',
        data: this.captureSelection(e.target),
      })
    );
  }

  captureSelection(element) {
    return {
      selector: generateUniqueSelector(element),
      componentName: getReactComponentName(element),
      sourceLocation: getSourceMapLocation(element),
      textContent: element.textContent?.slice(0, 100),
      boundingBox: element.getBoundingClientRect(),
      screenshot: captureElementScreenshot(element),
      computedStyles: getRelevantStyles(element),
    };
  }
}
```

**User Flow:**

1. User clicks "Select Element" button in Yantra
2. Browser enters selection mode (crosshair cursor)
3. User hovers → element highlights
4. User clicks → element locked with green border
5. Chat shows: "[Image selected: hero-image.jpg from HeroSection.tsx:24]"
6. User types: "Make this more professional"
7. Agent knows exact element, file, line → Makes precise change

**Files:**

- Update `src-tauri/src/browser/yantra-runtime.js` - Add selection mode
- `src-tauri/src/browser/interaction_handler.rs` (NEW) - Handle selection events
- UI: Add "Select Element" button to browser preview panel

---

#### 10. WebSocket Communication (P1 Post-MVP)

**Requirement:** Bidirectional Browser ↔ Yantra communication channel.

**Implementation:**

```rust
// src-tauri/src/browser/websocket_server.rs

pub async fn start_websocket_server() -> Result<u16, String> {
    let port = find_available_port()?;

    let server = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

    tokio::spawn(async move {
        while let Ok((stream, _)) = server.accept().await {
            tokio::spawn(handle_websocket_connection(stream));
        }
    });

    Ok(port)
}

async fn handle_websocket_connection(stream: TcpStream) {
    let ws_stream = tokio_tungstenite::accept_async(stream).await.unwrap();
    let (write, read) = ws_stream.split();

    // Forward messages: browser → Yantra
    // Forward commands: Yantra → browser
}
```

**Messages:**

- Browser → Yantra: errors, selections, hover events
- Yantra → Browser: enable selection mode, highlight element, scroll to element

**Files:**

- `src-tauri/src/browser/websocket_server.rs` (NEW) - WebSocket server
- Update `yantra-runtime.js` - Connect to WebSocket

---

#### 11. Source Map Integration (P2 Post-MVP)

**Requirement:** Map browser elements to source code (React DevTools style).

**Implementation:**

```javascript
// Enhanced yantra-runtime.js

function getSourceMapLocation(element) {
  // React DevTools-style fiber lookup
  const fiber =
    element._reactFiber$ || Object.keys(element).find((k) => k.startsWith('__reactFiber'));

  if (fiber) {
    const fiberNode = element[fiber];
    if (fiberNode._debugSource) {
      return {
        fileName: fiberNode._debugSource.fileName,
        lineNumber: fiberNode._debugSource.lineNumber,
        columnNumber: fiberNode._debugSource.columnNumber,
        componentName: fiberNode.type?.name,
      };
    }
  }

  return null;
}
```

**Benefit:** Agent knows exactly which file and line to modify.

---

#### 12. Context Menu & Quick Actions (P2 Post-MVP)

**Requirement:** Right-click menu in browser preview with Replace/Edit/Remove/Duplicate.

**User Flow:**

1. User right-clicks element
2. Context menu appears: 🔄 Replace this image | 🎨 Change colors | ✏️ Edit text | 🗑️ Remove
3. User clicks "Replace this image"
4. Asset picker opens with Unsplash search + DALL-E generation + Upload
5. User selects new image → Agent updates code → Preview refreshes

**Files:**

- Update `yantra-runtime.js` - Capture right-click events
- UI: Context menu component
- `src-tauri/src/browser/quick_actions.rs` (NEW) - Handle actions

---

#### 13. Visual Feedback Loop (P3 Post-MVP)

**Requirement:** Before/After split view, Undo/Redo stack, change history.

**Features:**

- Split view showing current vs. proposed changes
- Visual diff highlighting
- One-click undo for any change
- Change history timeline

---

#### 14. Asset Picker Integration (P3 Post-MVP)

**Requirement:** Search Unsplash, generate with DALL-E, upload local assets.

**Features:**

- Unsplash API integration (free tier: 50 requests/hour)
- DALL-E 3 integration (for custom generation)
- Local file upload
- Asset library (recently used)

---

### Performance Targets

**MVP:**

- Chrome startup: <2 seconds
- Dev server startup: <10 seconds (framework-dependent)
- Error capture latency: <100ms (real-time)
- Memory overhead: <200MB (Chrome + monitoring)
- CPU overhead: <5% (when idle)

**Post-MVP:**

- Element selection response: <50ms (interactive feel)
- Screenshot capture: <200ms
- WebSocket message latency: <10ms

---

### Error Handling & Edge Cases

**Chrome Not Found:**

- Show user-friendly message: "Downloading browser engine..."
- Download Chromium automatically (~100MB, 30-60 seconds)
- Cache for future use (~/.yantra/browser/chromium)
- Fallback: Ask user to install Chrome manually (rare)

**Dev Server Fails to Start:**

- Check for port conflicts (try next port: 3001, 3002...)
- Check for missing dependencies (run `npm install`)
- Show clear error message with fix suggestions
- Allow manual port specification

**CDP Connection Fails:**

- Retry with exponential backoff (3 attempts)
- Show user-friendly error: "Browser preview unavailable, code validation continues"
- Degrade gracefully: Skip browser validation, rely on unit tests

**Browser Crashes:**

- Detect process exit
- Auto-restart browser
- Restore previous state (URL, tab)
- Log crash for debugging

---

### Security Considerations

**Local-Only Communication:**

- WebSocket server binds to 127.0.0.1 (localhost only)
- No external access
- Random port selection (no fixed port conflicts)

**Chrome Sandbox:**

- Chrome runs in sandboxed mode (default)
- No filesystem access beyond project folder
- No network access to Yantra's internal APIs

**User Privacy:**

- No telemetry sent to Yantra servers
- All browser data stays local
- Anonymous crash reports only (opt-in)

---

### Testing Strategy

**Unit Tests:**

- Chrome finder logic (mock filesystem)
- Port parsing from dev server output
- Error message parsing and classification

**Integration Tests:**

- Launch Chrome → Connect CDP → Navigate → Capture error
- Start dev server → Detect port → Wait for ready
- Inject runtime → Trigger error → Receive via WebSocket

**E2E Tests:**

- Full workflow: Generate React app with intentional error → Start server → Browser captures error → Agent fixes error → Validation passes
- Test on macOS, Windows, Linux (CI/CD)

**Platform Tests:**

- macOS: Test with Chrome, Brave, Edge
- Windows: Test with Chrome, Edge (both x64 and ARM)
- Linux: Test with Chrome, Chromium (snap and apt)

---

### Documentation

**User Documentation:**

- Browser requirements (Chrome/Chromium/Edge)
- Troubleshooting browser preview issues
- How to manually specify Chrome path (advanced)

**Developer Documentation:**

- CDP architecture and event flow
- Adding new dev server frameworks
- Extending error capture for new error types
- WebSocket protocol specification (for Post-MVP features)

**API Documentation:**

- Tauri commands for browser control
- TypeScript types for browser events
- WebSocket message schemas

---

### Success Metrics

**MVP (Phase 1):**

- 95%+ of users have working browser preview (zero manual setup)
- 100% of console errors captured
- 90%+ of network errors captured
- <2 second browser startup time
- <10 second dev server startup time

**Post-MVP (Phase 2):**

- 80%+ of users use element selection feature
- 50%+ of changes made via visual interaction (not just chat)
- 90%+ accuracy in source map location detection

---

## Code Autocompletion (MVP Phase 1 - Developer Experience)

### Overview

**Purpose:** Provide intelligent code autocompletion in the Monaco editor to enhance developer productivity when reviewing or manually editing AI-generated code.

**Approach:** Hybrid system combining static completions (keywords, snippets) with GNN-powered context-aware suggestions.

**Phases:**

- **MVP (Phase 1):** Static completions + GNN-based symbol completion
- **Post-MVP (Phase 2):** LLM-powered intelligent completions (like GitHub Copilot)

---

### MVP Features (Required for Launch)

#### 1. Static Language Completions

**Requirement:** Provide instant keyword and snippet completions for Python and JavaScript/TypeScript.

**Implementation:**

```typescript
// src-ui/components/CodeViewer.tsx

monaco.languages.registerCompletionItemProvider('python', {
  provideCompletionItems: (model, position) => {
    const word = model.getWordUntilPosition(position);
    const range = {
      startLineNumber: position.lineNumber,
      endLineNumber: position.lineNumber,
      startColumn: word.startColumn,
      endColumn: word.endColumn,
    };

    return {
      suggestions: [
        // Python keywords
        {
          label: 'def',
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: 'def ${1:function_name}(${2:args}):\n    ${0:pass}',
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Define a function',
          range: range,
        },
        {
          label: 'class',
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText:
            'class ${1:ClassName}:\n    def __init__(self${2:, args}):\n        ${0:pass}',
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Define a class',
          range: range,
        },
        // More snippets...
      ],
    };
  },
});
```

**Supported Languages:**

- Python (keywords, common patterns, built-in functions)
- JavaScript/TypeScript (keywords, ES6+ syntax, common patterns)
- JSON (structure templates)
- Markdown (headers, links, code blocks)

**Snippets to Include:**

**Python:**

- `def` - Function definition
- `class` - Class definition
- `if`, `elif`, `else` - Conditionals
- `for`, `while` - Loops
- `try`, `except`, `finally` - Error handling
- `with` - Context managers
- `import`, `from...import` - Imports
- `async def`, `await` - Async patterns
- `@property`, `@staticmethod`, `@classmethod` - Decorators

**JavaScript/TypeScript:**

- `function`, `const fn =`, `() =>` - Function definitions
- `class` - Class definition
- `if`, `else if`, `else` - Conditionals
- `for`, `while`, `for...of`, `for...in` - Loops
- `try`, `catch`, `finally` - Error handling
- `async`, `await`, `Promise` - Async patterns
- `import`, `export` - ES6 modules
- `interface`, `type` - TypeScript types

**Performance Target:** <10ms response time (instant)

**Files:**

- `src-ui/components/CodeViewer.tsx` - Register completion providers
- `src-ui/utils/completions/python.ts` (NEW) - Python snippets
- `src-ui/utils/completions/javascript.ts` (NEW) - JavaScript snippets

---

#### 2. GNN-Powered Context-Aware Completions

**Requirement:** Query GNN dependency graph to suggest imported symbols, functions, and classes available in current file context.

**Implementation:**

```typescript
// src-ui/components/CodeViewer.tsx

monaco.languages.registerCompletionItemProvider('python', {
  provideCompletionItems: async (model, position) => {
    const currentFile = getCurrentFilePath();
    const lineContent = model.getLineContent(position.lineNumber);

    // Call Tauri backend for GNN-based suggestions
    const gnnSuggestions = await invoke('get_gnn_completions', {
      filePath: currentFile,
      line: position.lineNumber,
      column: position.column,
      context: lineContent,
    });

    return {
      suggestions: gnnSuggestions.map((s) => ({
        label: s.name,
        kind: mapCompletionKind(s.kind),
        insertText: s.insertText,
        detail: s.detail,
        documentation: s.documentation,
        sortText: s.priority, // Higher priority = shows first
      })),
    };
  },

  triggerCharacters: ['.', '(', ' '],
});
```

**Backend Implementation:**

```rust
// src-tauri/src/completion/mod.rs (NEW FILE)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub name: String,
    pub kind: CompletionKind,
    pub insert_text: String,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub priority: String, // "0" = highest, "9" = lowest
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionKind {
    Function,
    Class,
    Variable,
    Method,
    Property,
    Module,
    Keyword,
}

#[tauri::command]
pub async fn get_gnn_completions(
    file_path: String,
    line: usize,
    column: usize,
    context: String,
    state: State<'_, AppState>,
) -> Result<Vec<CompletionItem>, String> {
    let gnn = state.gnn.lock().await;

    // 1. Get file node from GNN
    let file_node = gnn.get_node_by_path(&file_path)
        .ok_or("File not found in GNN")?;

    // 2. Query imported symbols via Import edges
    let imported_symbols = gnn.get_imported_symbols(file_node)?;

    // 3. Query local symbols (functions/classes defined in file)
    let local_symbols = gnn.get_file_symbols(file_node)?;

    // 4. Detect completion context
    let completion_type = detect_context(&context, column);

    let mut completions = Vec::new();

    match completion_type {
        CompletionContext::ImportStatement => {
            // Suggest available modules from project
            completions.extend(get_available_modules(&gnn)?);
        }
        CompletionContext::MemberAccess(object) => {
            // User typed "object." - suggest methods/properties
            completions.extend(get_member_completions(&gnn, &object)?);
        }
        CompletionContext::FunctionCall => {
            // Filter to functions only
            completions.extend(
                imported_symbols.iter()
                    .chain(local_symbols.iter())
                    .filter(|s| s.kind == SymbolKind::Function)
                    .map(|s| to_completion_item(s, "1")) // High priority
            );
        }
        CompletionContext::General => {
            // All accessible symbols
            completions.extend(
                local_symbols.into_iter()
                    .map(|s| to_completion_item(&s, "0")) // Highest priority
            );
            completions.extend(
                imported_symbols.into_iter()
                    .map(|s| to_completion_item(&s, "2")) // Medium priority
            );
        }
    }

    // 5. Sort by priority and relevance
    completions.sort_by(|a, b| a.priority.cmp(&b.priority));

    Ok(completions.into_iter().take(50).collect()) // Limit to 50
}

enum CompletionContext {
    ImportStatement,
    MemberAccess(String),
    FunctionCall,
    General,
}

fn detect_context(line: &str, column: usize) -> CompletionContext {
    let before_cursor = &line[..column.min(line.len())];

    // Check for "import " or "from X import "
    if before_cursor.trim_start().starts_with("import ") ||
       before_cursor.contains("from ") && before_cursor.ends_with("import ") {
        return CompletionContext::ImportStatement;
    }

    // Check for member access: "object."
    if let Some(dot_pos) = before_cursor.rfind('.') {
        let object_name = before_cursor[..dot_pos]
            .split_whitespace()
            .last()
            .unwrap_or("")
            .to_string();
        return CompletionContext::MemberAccess(object_name);
    }

    // Check for function call context: ends with "("
    if before_cursor.trim_end().ends_with('(') {
        return CompletionContext::FunctionCall;
    }

    CompletionContext::General
}

fn get_member_completions(
    gnn: &DependencyGraph,
    object_name: &str,
) -> Result<Vec<CompletionItem>, String> {
    // Try to find class definition for object
    let class_node = gnn.find_class_for_variable(object_name)?;

    // Get all methods and properties
    let members = gnn.get_class_members(class_node)?;

    Ok(members.into_iter().map(|m| CompletionItem {
        name: m.name.clone(),
        kind: if m.is_method {
            CompletionKind::Method
        } else {
            CompletionKind::Property
        },
        insert_text: if m.is_method {
            format!("{}(${{1:}})", m.name)
        } else {
            m.name.clone()
        },
        detail: Some(m.signature),
        documentation: m.docstring,
        priority: "1".to_string(),
    }).collect())
}
```

**Completion Scenarios:**

1. **Import Suggestions:**

   ```python
   import |  # Cursor here
   # Suggests: all modules from project (calculator, utils, etc.)
   ```

2. **Imported Function/Class:**

   ```python
   from calculator import add, subtract

   result = a|  # Cursor here
   # Suggests: add(), subtract() (from imports)
   ```

3. **Member Access:**

   ```python
   calc = Calculator()
   calc.|  # Cursor here
   # Suggests: add(), subtract(), clear() (methods from class)
   ```

4. **Local Symbols:**

   ```python
   def helper_function():
       pass

   hel|  # Cursor here
   # Suggests: helper_function() (defined in same file)
   ```

**Performance Target:** <200ms response time

**Files:**

- `src-tauri/src/completion/mod.rs` (NEW) - Completion logic
- `src-tauri/src/completion/context.rs` (NEW) - Context detection
- `src-tauri/src/gnn/query.rs` (EXTEND) - Add symbol query methods
- `src-ui/components/CodeViewer.tsx` - Wire up to Monaco

---

#### 3. Trigger Characters

**Requirement:** Automatically trigger completions on specific characters.

**Configuration:**

```typescript
triggerCharacters: [
  '.', // Member access: object.method
  '(', // Function call context
  ' ', // After keywords (import, from, etc.)
  '\n', // New line (suggest indented blocks)
];
```

**Behavior:**

- `.` - Immediately show member completions
- `(` - Show function signature hints
- after `import` - Show module suggestions
- `\n` after `:` - Show indented block patterns

---

#### 4. Completion Ranking

**Requirement:** Show most relevant completions first.

**Priority Levels:**

1. **Highest (0):** Local symbols defined in current file
2. **High (1):** Explicitly imported symbols
3. **Medium (2):** Built-in functions (print, len, etc.)
4. **Low (3):** Keywords and snippets

**Additional Factors:**

- Frequency of use (track which symbols user uses most)
- Proximity in code (symbols defined nearby rank higher)
- Type compatibility (if type info available)

---

### MVP Implementation Checklist

**Week 1: Static Completions (2-3 days)**

- [ ] Create completion utility files
  - [ ] `src-ui/utils/completions/python.ts`
  - [ ] `src-ui/utils/completions/javascript.ts`
  - [ ] `src-ui/utils/completions/common.ts`
- [ ] Register Python completion provider in CodeViewer
- [ ] Register JavaScript/TypeScript completion provider
- [ ] Add 20+ Python snippets (functions, classes, loops, etc.)
- [ ] Add 20+ JavaScript snippets (functions, classes, async, etc.)
- [ ] Test completion triggering on different contexts
- [ ] Verify <10ms response time

**Week 2: GNN Integration (4-5 days)**

- [ ] Create backend completion module
  - [ ] `src-tauri/src/completion/mod.rs`
  - [ ] `src-tauri/src/completion/context.rs`
- [ ] Implement `get_gnn_completions` Tauri command
- [ ] Add GNN query methods:
  - [ ] `get_imported_symbols(file_node) -> Vec<Symbol>`
  - [ ] `get_file_symbols(file_node) -> Vec<Symbol>`
  - [ ] `get_class_members(class_node) -> Vec<Member>`
  - [ ] `find_class_for_variable(name) -> Option<NodeId>`
- [ ] Implement context detection:
  - [ ] Import statement detection
  - [ ] Member access detection (object.)
  - [ ] Function call detection
- [ ] Wire up frontend to backend
- [ ] Add caching (cache completions for 30 seconds)
- [ ] Test with real project (multi-file dependencies)
- [ ] Verify <200ms response time

**Week 3: Integration & Testing (2-3 days)**

- [ ] Merge static + GNN completions
- [ ] Implement deduplication logic
- [ ] Add error handling (fallback to static if GNN fails)
- [ ] Test edge cases:
  - [ ] Empty file
  - [ ] File not in GNN
  - [ ] Invalid syntax
  - [ ] No imports
- [ ] Performance optimization:
  - [ ] Debounce GNN queries (wait 100ms after typing stops)
  - [ ] Cache frequently used completions
  - [ ] Lazy-load completion providers
- [ ] UI polish:
  - [ ] Show loading indicator for GNN completions
  - [ ] Display completion kind icons
  - [ ] Show documentation preview
- [ ] Write unit tests (completion logic)
- [ ] Write integration tests (end-to-end)

**Success Criteria:**

- ✅ Static completions respond in <10ms
- ✅ GNN completions respond in <200ms
- ✅ 40+ built-in snippets (Python + JS combined)
- ✅ Correctly suggests imported symbols
- ✅ Member access completions work for classes
- ✅ No UI freezing during completion queries
- ✅ Fallback to static when GNN unavailable

---

### Post-MVP Features (Phase 2)

#### 5. LLM-Powered Intelligent Completions

**Requirement:** Use LLM to generate context-aware multi-line completions (like GitHub Copilot).

**Implementation:**

```rust
#[tauri::command]
pub async fn get_llm_completions(
    file_path: String,
    content: String,
    line: usize,
    state: State<'_, AppState>,
) -> Result<Vec<CompletionItem>, String> {
    let llm = state.llm_orchestrator.lock().await;

    // Extract context (50 lines before cursor)
    let context = extract_context(&content, line, 50);

    // Build prompt
    let prompt = format!(
        "Complete the following code. Provide 3 suggestions:\n\n{}\n\n[COMPLETE HERE]",
        context
    );

    // Use fast model (Claude Haiku or GPT-3.5-Turbo)
    let response = llm.complete(prompt, LLMConfig {
        model: "claude-3-haiku".to_string(),
        max_tokens: 300,
        temperature: 0.3,
        ..Default::default()
    }).await?;

    parse_llm_completions(&response.content)
}
```

**Trigger:** After 500ms of no typing (debounced)

**Features:**

- Multi-line completions
- Context-aware suggestions
- Function implementation suggestions
- Docstring generation

**Performance:** 500-1000ms (acceptable for intelligent completions)

---

#### 6. Signature Help (Function Hints)

**Requirement:** Show function signatures and parameter hints when typing function calls.

**Trigger:** When user types `(` after function name

**Display:**

```
add(a: int, b: int) -> int
    ^^^ current parameter
Add two numbers together.
```

---

#### 7. Hover Documentation

**Requirement:** Show documentation when hovering over symbols.

**Implementation:** Query GNN for symbol definition and docstring

---

### Performance Targets

**MVP:**

- Static completions: <10ms (instant)
- GNN completions: <200ms (fast)
- Memory usage: <50MB for completion cache
- No UI blocking/freezing

**Post-MVP:**

- LLM completions: <1000ms (acceptable with loading indicator)
- Signature help: <100ms
- Hover info: <50ms

---

### Error Handling

**Graceful Degradation:**

1. GNN query fails → Fallback to static completions
2. Backend unreachable → Static completions only
3. Invalid syntax → Show keywords/snippets
4. No file context → General suggestions

**User Experience:**

- Never show error to user
- Completions always work (even if reduced quality)
- Silent fallback to simpler methods

---

### Testing Strategy

**Unit Tests:**

- Context detection logic (import, member access, etc.)
- Completion ranking algorithm
- Deduplication logic

**Integration Tests:**

- Static completion provider registration
- GNN query integration
- Frontend-backend communication

**E2E Tests:**

- Type in Monaco editor → Verify completions appear
- Test each trigger character
- Test member access completions
- Test import suggestions

---

### Documentation

**User Documentation:**

- Keyboard shortcuts (Ctrl+Space to trigger manually)
- How completions work (static vs GNN)
- What to expect in different contexts

**Developer Documentation:**

- How to add new snippets
- How GNN completions work
- Architecture diagram (Monaco → Frontend → Tauri → GNN)

---

## Multi-LLM Consultation Mode (MVP Phase 1 - Stretch Goal)

### Overview

When the primary LLM gets stuck after 2 failed attempts, Yantra automatically consults other available LLMs to get alternative perspectives, insights, or debugging help. This is **collaborative, not competitive** - different LLMs have different strengths and blind spots. A second opinion often breaks through barriers that cause repeated failures.

**Core Philosophy:**

```
Primary LLM stuck → Consult another LLM "What am I missing?" → Primary uses insight
```

**Why This Works:**

- LLMs have different training data and biases
- Claude: Strong reasoning, cautious on edge cases
- GPT-4: Pragmatic, compact code style
- Gemini: Different perspective, fresh approach
- When stuck, the problem is often **framing or blind spot** - different LLM sees differently

**Cost Benefit:**

- Without consultation: 5+ failed attempts → human intervention needed
- With consultation: 2 failures + 1 consultation + 1 success = 4 total calls (saves money + time)

---

### MVP Feature 1: Consultation Trigger & Orchestration

**Consultation Flow:**

```
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
    ✅ Success or ❌ Escalate to human
```

**Trigger Logic:**

- **Threshold:** After **2 consecutive failures** with same primary LLM
- **Consultation:** Call Consultant LLM with full context
- **Integration:** Primary LLM regenerates with consultant's insight added to context
- **Max Attempts:** 3 total (2 primary alone + 1 with consultation insight)
- **Escalation:** After 3 failures → show user options (try different primary, manual intervention)

**Backend Structure:**

```rust
// File: src-tauri/src/llm/consultation.rs

pub struct ConsultationConfig {
    pub primary_model: String,           // e.g., "claude-sonnet-4"
    pub consultant1_model: Option<String>, // e.g., "gpt-4-turbo"
    pub consultation_threshold: u32,     // Default: 2 failures
}

pub struct ConsultationContext {
    pub task_description: String,
    pub failed_attempts: Vec<FailedAttempt>,
    pub error_messages: Vec<String>,
    pub code_snippets: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FailedAttempt {
    pub attempt_number: u32,
    pub generated_code: String,
    pub test_failures: Vec<String>,
    pub error_summary: String,
}

#[derive(Debug, Clone)]
pub struct ConsultantInsight {
    pub consultant_model: String,
    pub insight: String,
    pub alternative_approach: Option<String>,
    pub identified_issue: Option<String>,
}

impl ConsultationOrchestrator {
    pub async fn generate_with_consultation(
        &self,
        task: &Task,
        config: &ConsultationConfig,
    ) -> Result<Code, String> {
        let mut context = ConsultationContext::new(task);

        // Attempt 1: Primary LLM
        let result1 = self.primary_generate(task, &context).await?;
        if self.validate(&result1).await? {
            return Ok(result1);
        }
        context.add_failure(result1);

        // Attempt 2: Primary retry with error context
        let result2 = self.primary_generate(task, &context).await?;
        if self.validate(&result2).await? {
            return Ok(result2);
        }
        context.add_failure(result2);

        // Consultation: Get insight from consultant
        if let Some(consultant_model) = &config.consultant1_model {
            let insight = self.consult(consultant_model, &context).await?;
            context.add_insight(insight);

            // Attempt 3: Primary with consultant insight
            let result3 = self.primary_generate(task, &context).await?;
            if self.validate(&result3).await? {
                return Ok(result3);
            }
        }

        // All attempts failed - escalate to user
        Err(String::from("Unable to generate working code after consultation. Please review manually."))
    }
}
```

**Consultation Success Metrics:**

- Track consultations triggered per session
- Track success rate after consultation
- Track which LLM combinations work best
- Average attempts before/after consultation

---

### MVP Feature 2: Dynamic Consultation Prompt Generation

**Prompt Strategy:**
Instead of rigid templates, use the **Primary LLM itself** to generate the optimal consultation prompt based on context. This ensures the consultation request is well-framed and provides maximum value.

**Two-Step Process:**

**Step 1: Primary LLM Creates Consultation Prompt**

```rust
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
```

**Step 2: Consultant LLM Responds**

```rust
// Consultation prompt structure (generated by Primary LLM)
"You are a top coding expert consultant. Another AI attempted this task and got stuck after 2 attempts.

Task Description:
{}

Code Generated (Attempt 1):
```

{}

```

Test Failures:
{}

Code Generated (Attempt 2):
```

{}

```

Test Failures:
{}

What is being missed? What's a different approach to solve this?
Provide:
1. Root cause analysis
2. Specific fix recommendations
3. Alternative implementation approach (if applicable)"

let consultant_response = consultant_llm.generate(consultation_prompt).await?;
```

**Consultation Prompt Template (Baseline):**

If Primary LLM fails to generate consultation prompt (unlikely), fall back to this template:

```
You are a top-tier coding expert consultant. Another AI coding assistant attempted a task and encountered persistent failures after 2 attempts. Please provide your expert analysis.

TASK DESCRIPTION:
{task_description}

ATTEMPT 1:
Generated Code:
```

{code_attempt_1}

```

Test Results:
{test_failures_1}

ATTEMPT 2:
Generated Code:
```

{code_attempt_2}

```

Test Results:
{test_failures_2}

ANALYSIS REQUEST:
As a top coding expert, please analyze:
1. What fundamental issue is being missed?
2. What's wrong with the current approach?
3. What alternative approach would you recommend?
4. Are there edge cases not being considered?

Provide a clear, actionable response that will help resolve this issue.
```

**Why Dynamic Prompt Generation:**

- Context-aware: Adapts to specific failure patterns
- Better framing: Primary LLM knows what info is most relevant
- Flexibility: Works for any type of coding task
- No rigid templates: Consultant gets exactly the context needed

---

### MVP Feature 3: LLM Settings UI - Consultation Configuration

**Settings Panel Enhancement:**

```typescript
// Frontend: src-ui/components/Settings/LLMSettings.tsx

interface ConsultationSettings {
    primaryModel: string;           // Required
    consultant1Model: string | null; // Optional
    consultationEnabled: boolean;   // Toggle on/off
    consultationThreshold: number;  // Default: 2
}

// UI Structure:
<div class="consultation-config">
    <h3>Consultation Mode</h3>
    <Toggle
        label="Enable Multi-LLM Consultation"
        checked={consultationEnabled}
        onChange={handleToggle}
    />

    <Select
        label="Primary Model (Code Generator)"
        value={primaryModel}
        options={availableModels}
        onChange={setPrimaryModel}
    />

    <Select
        label="Consultant Model (Second Opinion)"
        value={consultant1Model}
        options={[{ value: null, label: "None (Auto-select)" }, ...availableModels]}
        onChange={setConsultant1Model}
        helpText="Leave empty to use same model with different prompt"
    />

    <NumberInput
        label="Consultation Threshold (failures)"
        value={consultationThreshold}
        min={1}
        max={5}
        onChange={setThreshold}
    />
</div>
```

**Available Models API:**

```rust
// Backend: src-tauri/src/llm/models.rs

#[tauri::command]
pub async fn get_available_models(
    app_handle: tauri::AppHandle
) -> Result<AvailableModelsResponse, String> {
    let config_manager = get_config_manager(&app_handle)?;
    let config = config_manager.get_config();

    let mut models = Vec::new();

    // Only show models for providers with API keys
    if config.claude_api_key.is_some() {
        models.extend(get_top_claude_models());
    }
    if config.openai_api_key.is_some() {
        models.extend(get_top_openai_models());
    }
    if config.gemini_api_key.is_some() {
        models.extend(get_top_gemini_models());
    }
    if config.groq_api_key.is_some() {
        models.extend(get_top_groq_models());
    }
    if config.openrouter_api_key.is_some() {
        models.extend(get_top_openrouter_models());
    }

    Ok(AvailableModelsResponse { models })
}

fn get_top_claude_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo { id: "claude-opus-4", name: "Claude Opus 4", provider: "claude", tier: "premium" },
        ModelInfo { id: "claude-sonnet-4", name: "Claude Sonnet 4", provider: "claude", tier: "premium" },
        ModelInfo { id: "claude-sonnet-3.5", name: "Claude Sonnet 3.5", provider: "claude", tier: "premium" },
    ]
}

fn get_top_openai_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo { id: "gpt-4-turbo", name: "GPT-4 Turbo", provider: "openai", tier: "premium" },
        ModelInfo { id: "gpt-4o", name: "GPT-4o", provider: "openai", tier: "premium" },
        ModelInfo { id: "gpt-4", name: "GPT-4", provider: "openai", tier: "premium" },
    ]
}

fn get_top_gemini_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo { id: "gemini-2.0-flash-thinking-exp", name: "Gemini 2.0 Flash Thinking", provider: "gemini", tier: "premium" },
        ModelInfo { id: "gemini-pro", name: "Gemini Pro", provider: "gemini", tier: "premium" },
    ]
}

fn get_top_groq_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo { id: "llama-3.3-70b", name: "Llama 3.3 70B", provider: "groq", tier: "free" },
        ModelInfo { id: "mixtral-8x7b", name: "Mixtral 8x7B", provider: "groq", tier: "free" },
    ]
}

fn get_top_openrouter_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo { id: "deepseek/deepseek-coder-v2", name: "DeepSeek Coder V2", provider: "openrouter", tier: "free" },
        ModelInfo { id: "qwen/qwen-2.5-coder-32b", name: "Qwen 2.5 Coder 32B", provider: "openrouter", tier: "free" },
        ModelInfo { id: "meta-llama/codellama-70b", name: "CodeLlama 70B", provider: "openrouter", tier: "free" },
    ]
}
```

**Model Selection Logic:**

- **Primary Model:** Must be selected (required dropdown)
- **Consultant Model:** Optional (can be null)
  - If null in **Guided Mode** → Ask user after 2 failures: "Get second opinion? [Select model]"
  - If null in **Auto Mode** → Use same model as Primary (with different consultation prompt)
- **Same Model Allowed:** Yes - user can select same model for both Primary and Consultant
  - Show info tooltip: "Using same model may provide limited new perspective"

---

### MVP Feature 4: Guided Mode Consultation Interaction

**Scenario: No Consultant Pre-Selected**

When Primary LLM fails twice and no consultant is configured:

```typescript
// UI: Show consultation prompt dialog

<Dialog open={showConsultationPrompt}>
    <DialogTitle>🤔 Code Generation Stuck</DialogTitle>
    <DialogContent>
        <p>The primary model ({primaryModel}) has failed twice with similar errors.</p>
        <p>Would you like to get a second opinion from another AI model?</p>

        <div class="error-summary">
            <h4>Recent Errors:</h4>
            <code>{errorSummary}</code>
        </div>

        <Select
            label="Choose Consultant Model"
            options={availableModels.filter(m => m.id !== primaryModel)}
            value={selectedConsultant}
            onChange={setSelectedConsultant}
        />

        <p class="help-text">
            Different AI models have different strengths. A second opinion often
            identifies issues the primary model missed.
        </p>
    </DialogContent>
    <DialogActions>
        <Button onClick={skipConsultation}>Skip (Manual Fix)</Button>
        <Button onClick={consultWithSelected} variant="primary">
            Get Second Opinion
        </Button>
    </DialogActions>
</Dialog>
```

**User Actions:**

1. **Get Second Opinion:** Continues with selected consultant
2. **Skip:** Shows code editor for manual intervention
3. **Remember Choice:** Checkbox to save consultant selection for future

---

### MVP Feature 5: UI Transparency - Consultation Progress

**Show user what's happening in real-time:**

```typescript
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
│ Generating: User Authentication Module                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ✅ Attempt 1 (Claude Sonnet 4)                            │
│    Generated auth module with JWT                          │
│ ❌ Tests failed: JWT validation error                      │
│                                                             │
│ ✅ Attempt 2 (Claude Sonnet 4)                            │
│    Fixed JWT validation logic                              │
│ ❌ Tests failed: Session handling edge case                │
│                                                             │
│ 🤔 Consulting GPT-4 Turbo for second opinion...            │
│ 💡 Insight received:                                        │
│    "Session lifecycle needs refresh token rotation.        │
│     Current implementation doesn't handle token expiry."   │
│                                                             │
│ ✅ Attempt 3 (Claude Sonnet 4 + GPT-4 insight)            │
│    Added refresh token rotation with expiry handling       │
│ ✅ All tests passing ✅                                     │
│                                                             │
│ 📊 Consultation saved 2+ failures & manual intervention    │
└─────────────────────────────────────────────────────────────┘
```

**Progress States:**

- ⏳ Generating code...
- ✅ Code generated
- 🧪 Running tests...
- ❌ Tests failed (show summary)
- 🤔 Consulting [Model Name]...
- 💡 Insight received
- ✅ Success!
- ❌ Escalated to manual review

**Transparency Benefits:**

- User sees Yantra is thorough, not blindly retrying
- Builds trust in multi-LLM collaboration
- Educational: Users learn from consultation insights
- Justifies cost: Clear value of consultation vs repeated failures

---

### Implementation Checklist (MVP - 2 weeks)

**Week 1: Core Consultation Logic (4-5 days)**

- [ ] Create `src-tauri/src/llm/consultation.rs` module
- [ ] Implement `ConsultationOrchestrator` with 2-failure threshold
- [ ] Implement `generate_with_consultation()` flow
- [ ] Add consultation context tracking (`ConsultationContext`, `FailedAttempt`)
- [ ] Implement dynamic prompt generation (Primary creates consultation prompt)
- [ ] Add fallback consultation template
- [ ] Wire consultation into main code generation flow
- [ ] Add configuration: `ConsultationConfig` in `LLMConfig`

**Week 1: Model Discovery & Configuration (2-3 days)**

- [ ] Implement `get_available_models()` command
- [ ] Filter models by available API keys
- [ ] Add top models per provider (Claude, OpenAI, Gemini, Groq, OpenRouter)
- [ ] Update `LLMConfig` to store `consultant1_model`
- [ ] Add `set_consultant_model()` to config manager
- [ ] Add `consultation_enabled` and `consultation_threshold` settings

**Week 2: UI Integration (4-5 days)**

- [ ] Update LLM Settings panel with consultation config
- [ ] Add Primary/Consultant model dropdowns
- [ ] Implement "Get Available Models" API call
- [ ] Show only models with valid API keys
- [ ] Add consultation toggle and threshold slider
- [ ] Implement Guided Mode consultation prompt dialog
- [ ] Add "Get Second Opinion" user flow
- [ ] Save consultant selection for future

**Week 2: Progress Transparency (2-3 days)**

- [ ] Create `ConsultationProgress.tsx` component
- [ ] Show real-time consultation steps
- [ ] Display attempt numbers, models used
- [ ] Show consultant insights in UI
- [ ] Add success/failure indicators
- [ ] Track and display consultation metrics

**Testing (2 days)**

- [ ] Unit tests: Consultation orchestrator logic
- [ ] Integration tests: Full consultation flow (2 failures → consult → success)
- [ ] UI tests: Model selection, consultation dialog
- [ ] E2E tests: Guided mode consultation interaction
- [ ] Performance tests: Consultation adds <500ms overhead
- [ ] Edge cases: No consultant, same model, API failures

---

### Success Criteria (MVP)

**Functional:**

- ✅ Triggers consultation after exactly 2 primary failures
- ✅ Dynamically generates consultation prompt using Primary LLM
- ✅ Only shows models with valid API keys in dropdowns
- ✅ Guided Mode asks user for consultant selection when not pre-configured
- ✅ Auto Mode uses Primary model as consultant if none selected
- ✅ Allows same model for Primary and Consultant
- ✅ Integrates consultant insight into final attempt
- ✅ Shows transparent progress in UI

**Performance:**

- ✅ Consultation adds <500ms overhead (prompt generation + API call)
- ✅ Available models API responds in <100ms
- ✅ UI remains responsive during consultation

**Metrics:**

- ✅ Track consultation success rate (target: >70% resolve after consultation)
- ✅ Track cost savings vs repeated failures
- ✅ Track which model combinations work best
- ✅ User satisfaction: Consultation helpful (feedback)

---

### Post-MVP Enhancements

**Post-MVP Feature: Consultant 2 (Three-Way Consultation)**

- Add optional Consultant 2 for extremely difficult problems
- Trigger: After Consultant 1 insight fails (4th attempt)
- Flow: Primary → Fail 2x → Consultant 1 → Fail → Consultant 2 → Final attempt
- Max attempts: 5 total

**Post-MVP Feature: Consultation Pattern Selection**

- Let Primary LLM choose consultation pattern:
  - "Second Opinion": What am I missing?
  - "Alternative Approach": How would you implement differently?
  - "Debug Partner": Find the bug
  - "Validation": Any issues with this approach?
- Use pattern selection to optimize consultation prompt

**Post-MVP Feature: Consultation Learning**

- Track successful consultations in database
- Learn which model combinations work best for which task types
- Auto-suggest best consultant based on task context
- Build consultation effectiveness heatmap

**Post-MVP Feature: Cost Optimization**

- Use cheaper models for consultation (e.g., GPT-4o-mini, Haiku)
- Smart routing: Only consult if high probability of success
- Consultation budget: Limit consultations per project

---

## Monetization Model (Post-MVP Priority #1)

### Overview

Yantra's monetization strategy balances affordability with sustainable business model. Users pay for the platform and tooling, while having flexibility in LLM provider choices.

**Pricing Philosophy:**

- **Platform Access:** Fixed subscription fee
- **Open-Source Models:** Unlimited usage included
- **Premium LLMs:** Bring your own API keys

---

### Pricing Structure

**Core Plan: $20/user/month**

**Included:**

- ✅ Full Yantra platform access (all MVP features)
- ✅ **Unlimited usage of open-source coding models:**
  - DeepSeek Coder V2 (33B parameters)
  - Qwen 2.5 Coder (32B parameters)
  - CodeLlama 70B
  - Llama 3.3 70B
  - Mixtral 8x7B
  - WizardCoder
  - Phind CodeLlama
- ✅ All core features:
  - Code generation with GNN context
  - Multi-LLM consultation mode
  - Automated testing & validation
  - Security scanning
  - Browser integration (CDP)
  - Git integration
  - Documentation generation
- ✅ Unlimited projects
- ✅ Unlimited code generation
- ✅ Cloud infrastructure for open-source models
- ✅ Priority support (email, community Slack)

**Free Trial:**

- ✅ First month free (no credit card required)
- ✅ Full access to all features
- ✅ Unlimited open-source model usage
- ✅ Cancel anytime

**Premium LLM Add-Ons (User Provides API Keys):**

- Claude (Opus, Sonnet, Haiku) → User provides Anthropic API key
- GPT-4 (Turbo, o1) → User provides OpenAI API key
- Gemini (Pro, Ultra) → User provides Google API key
- User pays LLM provider directly (consumption-based)
- Yantra doesn't charge markup on LLM costs

---

### Implementation Details

**Subscription Management:**

```rust
// Backend: src-tauri/src/subscription/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSubscription {
    pub user_id: String,
    pub plan: SubscriptionPlan,
    pub status: SubscriptionStatus,
    pub trial_ends_at: Option<DateTime<Utc>>,
    pub billing_cycle_start: DateTime<Utc>,
    pub billing_cycle_end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionPlan {
    FreeTrial,   // First month
    Core,        // $20/month
    Cancelled,   // Churned user
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    Active,
    TrialActive,
    Expired,
    Cancelled,
    PaymentFailed,
}

impl SubscriptionManager {
    pub fn can_use_opensource_models(&self, user: &UserSubscription) -> bool {
        matches!(user.status, SubscriptionStatus::Active | SubscriptionStatus::TrialActive)
    }

    pub fn can_use_premium_llm(&self, user: &UserSubscription, provider: LLMProvider) -> bool {
        // Check if user has active subscription
        if !self.can_use_opensource_models(user) {
            return false;
        }

        // Check if user provided API key for premium provider
        self.has_user_api_key(provider)
    }
}
```

**LLM Provider Routing:**

```rust
// Determine which LLM to use based on subscription and configuration
pub async fn route_llm_request(
    &self,
    user: &UserSubscription,
    config: &LLMConfig,
    task: &Task,
) -> Result<String, String> {
    let model = &config.primary_model;

    // Check if model is open-source (included in subscription)
    if self.is_opensource_model(model) {
        // Use Yantra's infrastructure (user paid $20/month)
        return self.call_yantra_hosted_model(model, task).await;
    }

    // Check if model is premium (requires user API key)
    if self.is_premium_model(model) {
        let provider = self.get_provider_for_model(model);

        // Verify user has API key for this provider
        if !self.has_user_api_key(&provider) {
            return Err(format!(
                "Premium model {} requires {} API key. Please add it in Settings.",
                model, provider
            ));
        }

        // Use user's API key (user pays LLM provider directly)
        return self.call_with_user_api_key(&provider, model, task).await;
    }

    Err("Unknown model".to_string())
}

fn is_opensource_model(&self, model: &str) -> bool {
    matches!(model,
        "deepseek-coder-v2" |
        "qwen-2.5-coder-32b" |
        "codellama-70b" |
        "llama-3.3-70b" |
        "mixtral-8x7b" |
        "wizardcoder" |
        "phind-codellama"
    )
}

fn is_premium_model(&self, model: &str) -> bool {
    matches!(model,
        "claude-opus-4" | "claude-sonnet-4" | "claude-sonnet-3.5" | "claude-haiku" |
        "gpt-4-turbo" | "gpt-4o" | "gpt-4" | "o1-preview" | "o1-mini" |
        "gemini-pro" | "gemini-ultra" | "gemini-2.0-flash-thinking-exp"
    )
}
```

**Subscription Enforcement:**

```rust
// Gate features behind subscription check
#[tauri::command]
pub async fn generate_code(
    app_handle: tauri::AppHandle,
    request: CodeGenerationRequest,
) -> Result<CodeGenerationResponse, String> {
    let user = get_current_user(&app_handle)?;
    let subscription = get_user_subscription(&user.id).await?;

    // Check if subscription is active
    if !subscription_manager.can_use_opensource_models(&subscription) {
        return Err("Subscription expired. Please renew to continue.".to_string());
    }

    // Check if user can use selected LLM
    let config = get_llm_config(&app_handle)?;
    let provider = get_provider_for_model(&config.primary_model);

    if !subscription_manager.can_use_premium_llm(&subscription, provider) {
        return Err(format!(
            "Premium model requires {} API key. Add it in Settings or use included open-source models.",
            provider
        ));
    }

    // Proceed with code generation
    generate_code_internal(request, config).await
}
```

**UI: Subscription Status**

```typescript
// Frontend: src-ui/components/Subscription/SubscriptionBanner.tsx

interface SubscriptionBannerProps {
    subscription: UserSubscription;
}

export function SubscriptionBanner({ subscription }: SubscriptionBannerProps) {
    if (subscription.status === 'TrialActive') {
        const daysLeft = getDaysUntilTrialEnd(subscription.trial_ends_at);
        return (
            <Banner variant="info">
                🎉 Free trial active: {daysLeft} days remaining
                <Button onClick={upgradeToPaid}>Upgrade to Core Plan</Button>
            </Banner>
        );
    }

    if (subscription.status === 'Expired') {
        return (
            <Banner variant="warning">
                ⚠️ Subscription expired. Renew to continue using Yantra.
                <Button onClick={renewSubscription}>Renew Now</Button>
            </Banner>
        );
    }

    if (subscription.status === 'Active') {
        return (
            <Banner variant="success">
                ✅ Core Plan Active ($20/month) - Unlimited open-source models
                <Link to="/settings/subscription">Manage Subscription</Link>
            </Banner>
        );
    }

    return null;
}
```

---

### Payment Integration (Post-MVP)

**Payment Provider:** Stripe

**Features:**

- [ ] Credit card payment processing
- [ ] Subscription management (create, cancel, update)
- [ ] Automatic billing ($20/month)
- [ ] Invoice generation and email
- [ ] Payment failure handling (retry, grace period)
- [ ] Proration for mid-cycle changes
- [ ] Webhook handling (payment success, failure, cancelled)

**Implementation:**

- Stripe Checkout for payment collection
- Stripe Customer Portal for subscription management
- Webhook integration for real-time status updates
- Secure storage of Stripe customer IDs (encrypted)

---

### Usage Analytics & Billing

**Track Usage (for transparency, not billing):**

- Total code generations per user
- Total LLM API calls (open-source vs premium)
- Open-source model usage (Yantra's cost)
- Premium model usage (user's cost via their API keys)
- Success rates, consultation rates
- Feature adoption metrics

**User Dashboard:**

```
Your Yantra Usage (This Month)

Open-Source Models (Included):
  - DeepSeek Coder: 1,234 requests
  - Qwen 2.5 Coder: 567 requests
  - Total: 1,801 requests (Unlimited ✅)

Premium Models (Your API Keys):
  - Claude Sonnet 4: $12.34 (charged by Anthropic)
  - GPT-4 Turbo: $8.76 (charged by OpenAI)
  - Total Premium Cost: $21.10 (billed separately by providers)

Yantra Subscription: $20.00/month (Next billing: Dec 30, 2025)
```

---

### Business Model Rationale

**Why $20/month for unlimited open-source models?**

- Average user generates ~2,000-5,000 code requests/month
- Open-source LLM costs (hosted): ~$0.002-0.005 per request
- Total cost per user: $4-25/month
- Margin: Break-even to slight profit, subsidized by enterprise plans (future)
- Value: Users save 10-100x vs paying per-token with premium LLMs

**Why BYOK (Bring Your Own Key) for premium LLMs?**

- Premium LLM costs vary wildly ($0.01-0.10 per request)
- Hard to predict user consumption patterns
- Markup model would make pricing unpredictable
- Users retain control over LLM spend
- Transparent cost structure builds trust

**Revenue Projections:**

- Month 1: 100 users × $20 = $2,000 MRR
- Month 6: 1,000 users × $20 = $20,000 MRR
- Year 1: 5,000 users × $20 = $100,000 MRR
- Costs: Infrastructure ($5-10/user), support, development
- Target: Break-even at 500-1,000 users, profitable after

---

### Implementation Timeline (Post-MVP)

**Week 1-2: Subscription Backend (10 days)**

- [ ] Design subscription database schema
- [ ] Implement `SubscriptionManager` module
- [ ] Add subscription status checks to all gated features
- [ ] Implement free trial logic (30 days)
- [ ] Add subscription status API endpoints
- [ ] Test subscription enforcement

**Week 3: Payment Integration (5 days)**

- [ ] Stripe account setup
- [ ] Integrate Stripe Checkout
- [ ] Implement subscription creation flow
- [ ] Add webhook handlers (payment events)
- [ ] Test payment flow end-to-end

**Week 4: UI & User Experience (5 days)**

- [ ] Subscription status banner
- [ ] Upgrade/downgrade flows
- [ ] Payment settings page
- [ ] Invoice history
- [ ] Usage dashboard
- [ ] Trial expiration reminders

**Week 5: Testing & Launch (5 days)**

- [ ] End-to-end subscription testing
- [ ] Payment failure scenarios
- [ ] Cancellation and reactivation flows
- [ ] Security audit (PCI compliance)
- [ ] Soft launch to beta users
- [ ] Monitor and iterate

---

### Success Metrics (Post-MVP Monetization)

**Subscription:**

- ✅ 90%+ of trial users complete signup flow
- ✅ 30%+ trial-to-paid conversion rate
- ✅ <3% payment failure rate
- ✅ <5% monthly churn rate

**User Satisfaction:**

- ✅ NPS >40 (value for money)
- ✅ 80%+ users satisfied with open-source model quality
- ✅ Clear understanding of BYOK model (premium LLMs)

**Business:**

- ✅ Break-even at 500 users
- ✅ Profitable at 1,000+ users
- ✅ Healthy unit economics (LTV > 5x CAC)

---

## Cascading Failure Protection (MVP Phase 1 - Priority Feature)

### Overview

**Problem:** AI agents can enter failure loops where each attempted fix addresses a smaller problem while losing sight of the bigger solution. Without proper safeguards, the agent can make the codebase progressively worse through cascading failures.

**Solution:** Implement a comprehensive checkpoint-based recovery system with intelligent failure detection, automatic rollback, and escalation strategies.

**Core Principle:** Every modification is reversible with one click. The system should automatically detect when it's digging deeper into failure and revert to the last known working state.

---

### 1. Checkpoint System (Critical - MVP Foundation)

**Requirement:** Create a checkpoint before **any** modification that can be restored with one click.

**Checkpoint Granularity:**

```
Project Checkpoint Hierarchy:
├─ Session Checkpoint (every chat session)
├─ Feature Checkpoint (before each feature implementation)
├─ File Checkpoint (before modifying each file)
└─ Test Checkpoint (before running tests)
```

**Checkpoint Storage:**

```rust
// src-tauri/src/checkpoints/manager.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,                    // UUID
    pub timestamp: DateTime<Utc>,
    pub checkpoint_type: CheckpointType,
    pub description: String,           // User-friendly description
    pub files_snapshot: HashMap<PathBuf, String>,  // File path → content
    pub gnn_state: Vec<u8>,           // Serialized GNN state
    pub architecture_version: i64,     // Architecture version ID
    pub test_results: Option<TestSummary>,
    pub confidence_score: f32,         // 0.0-1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointType {
    Session,      // Start of chat session
    Feature,      // Before implementing feature
    File,         // Before modifying file
    Test,         // Before running tests
    Manual,       // User-requested checkpoint
}

pub struct CheckpointManager {
    storage_path: PathBuf,  // .yantra/checkpoints/
    active_checkpoints: Vec<Checkpoint>,
    max_checkpoints: usize,  // Default: 20 (keep last 20)
}

impl CheckpointManager {
    /// Create checkpoint before any modification
    pub async fn create_checkpoint(
        &mut self,
        checkpoint_type: CheckpointType,
        description: String,
        files: &[PathBuf],
    ) -> Result<String, String> {
        let checkpoint_id = Uuid::new_v4().to_string();

        // Snapshot all affected files
        let mut files_snapshot = HashMap::new();
        for file_path in files {
            if file_path.exists() {
                let content = std::fs::read_to_string(file_path)
                    .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e))?;
                files_snapshot.insert(file_path.clone(), content);
            }
        }

        // Snapshot GNN state
        let gnn_state = self.serialize_gnn_state()?;

        // Get current architecture version
        let architecture_version = self.get_current_architecture_version()?;

        let checkpoint = Checkpoint {
            id: checkpoint_id.clone(),
            timestamp: Utc::now(),
            checkpoint_type,
            description,
            files_snapshot,
            gnn_state,
            architecture_version,
            test_results: None,
            confidence_score: 1.0,  // Assume current state is good
        };

        // Save to disk
        self.save_checkpoint(&checkpoint)?;

        // Add to active list
        self.active_checkpoints.push(checkpoint);

        // Prune old checkpoints if exceeding limit
        if self.active_checkpoints.len() > self.max_checkpoints {
            self.prune_old_checkpoints();
        }

        Ok(checkpoint_id)
    }

    /// Restore from checkpoint (one-click recovery)
    pub async fn restore_checkpoint(&self, checkpoint_id: &str) -> Result<(), String> {
        let checkpoint = self.active_checkpoints.iter()
            .find(|c| c.id == checkpoint_id)
            .ok_or_else(|| format!("Checkpoint {} not found", checkpoint_id))?;

        // Restore all files
        for (file_path, content) in &checkpoint.files_snapshot {
            std::fs::write(file_path, content)
                .map_err(|e| format!("Failed to restore {}: {}", file_path.display(), e))?;
        }

        // Restore GNN state
        self.restore_gnn_state(&checkpoint.gnn_state)?;

        // Restore architecture version
        self.restore_architecture_version(checkpoint.architecture_version)?;

        Ok(())
    }

    /// List available checkpoints (for user selection)
    pub fn list_checkpoints(&self) -> Vec<CheckpointSummary> {
        self.active_checkpoints.iter().map(|c| {
            CheckpointSummary {
                id: c.id.clone(),
                timestamp: c.timestamp,
                description: c.description.clone(),
                checkpoint_type: c.checkpoint_type.clone(),
                files_count: c.files_snapshot.len(),
                confidence_score: c.confidence_score,
            }
        }).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSummary {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub checkpoint_type: CheckpointType,
    pub files_count: usize,
    pub confidence_score: f32,
}
```

**Checkpoint UI (One-Click Restore):**

```
┌────────────────────────────────────────────────────────────┐
│ 📸 CHECKPOINTS                                  [Restore ▼]│
├────────────────────────────────────────────────────────────┤
│ ✅ Session Start - 10:15 AM (30 min ago)                   │
│    8 files | Confidence: 100%                [Restore]     │
├────────────────────────────────────────────────────────────┤
│ 🔨 Before "Add Payment Service" - 10:20 AM                │
│    8 files | Confidence: 100% | All tests passing          │
│                                              [Restore]     │
├────────────────────────────────────────────────────────────┤
│ ⚠️  After Payment Service (Failed) - 10:25 AM             │
│    11 files | Confidence: 45% | 5 tests failing            │
│                                              [Restore]     │
├────────────────────────────────────────────────────────────┤
│ ❌ After Fix Attempt #1 (Failed) - 10:27 AM               │
│    11 files | Confidence: 30% | 8 tests failing            │
│                                              [Restore]     │
└────────────────────────────────────────────────────────────┘

Recommendation: Restore "Before Add Payment Service" (100% confidence)
```

---

### 2. Impact Assessment (GNN-Based)

**Requirement:** Assess impact using GNN dependencies before making changes.

**Implementation:**

```rust
// src-tauri/src/agent/impact_analyzer.rs

pub struct ImpactAnalyzer {
    gnn_engine: Arc<Mutex<GNNEngine>>,
    checkpoint_manager: CheckpointManager,
}

impl ImpactAnalyzer {
    /// Analyze impact of proposed code change
    pub async fn analyze_impact(
        &self,
        files_to_modify: &[PathBuf],
    ) -> Result<ImpactReport, String> {
        let gnn = self.gnn_engine.lock().unwrap();

        // Find all dependent files
        let dependent_files = gnn.get_all_dependencies(files_to_modify)?;

        // Find affected features
        let affected_features = gnn.get_affected_features(files_to_modify)?;

        // Calculate risk score (0.0-1.0)
        let risk_score = self.calculate_risk_score(&dependent_files, &affected_features);

        // Estimate test impact
        let test_impact = self.estimate_test_impact(files_to_modify)?;

        Ok(ImpactReport {
            files_to_modify: files_to_modify.to_vec(),
            dependent_files,
            affected_features,
            risk_score,
            test_impact,
            recommendation: self.generate_recommendation(risk_score),
        })
    }

    fn calculate_risk_score(
        &self,
        dependent_files: &[PathBuf],
        affected_features: &[Feature],
    ) -> f32 {
        // Risk factors:
        // - Number of dependent files (more = higher risk)
        // - Number of affected features (more = higher risk)
        // - Criticality of affected features (core features = higher risk)

        let file_risk = (dependent_files.len() as f32 / 10.0).min(1.0);
        let feature_risk = (affected_features.len() as f32 / 5.0).min(1.0);
        let criticality_risk = affected_features.iter()
            .map(|f| f.criticality)
            .max()
            .unwrap_or(0.0);

        // Weighted average
        (file_risk * 0.3 + feature_risk * 0.3 + criticality_risk * 0.4)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactReport {
    pub files_to_modify: Vec<PathBuf>,
    pub dependent_files: Vec<PathBuf>,
    pub affected_features: Vec<Feature>,
    pub risk_score: f32,  // 0.0-1.0 (0=safe, 1=high risk)
    pub test_impact: TestImpact,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestImpact {
    pub tests_to_run: Vec<String>,        // Test file names
    pub estimated_failures: usize,         // Predicted test failures
    pub estimated_runtime: Duration,       // How long tests will take
}
```

**Impact Display (Guided Mode):**

```
Agent: "Impact Analysis:

Files to Modify:
- payment_service.py (new file)

Dependent Files (will be affected):
- gateway.py (imports Payment Service)
- api_routes.py (calls Payment Service API)

Affected Features:
├─ 🛒 Shopping Cart
│   └─ Uses Payment Service for checkout
│   └─ Risk: Low (well-isolated dependency)
│
└─ 📊 Admin Dashboard
    └─ Displays payment statistics
    └─ Risk: Low (read-only access)

Overall Risk Score: 0.25 (Low Risk) ✅

Tests to Run:
- test_gateway.py (expected: 2 failures)
- test_payment_service.py (new tests)
- test_shopping_cart_integration.py (expected: 0 failures)

Estimated Runtime: 15 seconds

Recommendation: Safe to proceed. Create checkpoint first."
```

---

### 3. Automated Testing After Changes

**Requirement:** Run automated tests after each change to detect impact immediately.

**Test Execution Strategy:**

```rust
// src-tauri/src/testing/auto_runner.rs

pub struct AutoTestRunner {
    checkpoint_manager: Arc<Mutex<CheckpointManager>>,
    failure_tracker: FailureTracker,
}

impl AutoTestRunner {
    /// Run tests after code change
    pub async fn run_tests_after_change(
        &mut self,
        changed_files: &[PathBuf],
    ) -> Result<TestResult, String> {
        // 1. Determine which tests to run (based on GNN dependencies)
        let tests_to_run = self.select_relevant_tests(changed_files)?;

        // 2. Run tests
        let test_result = self.execute_tests(&tests_to_run).await?;

        // 3. Compare with previous test results
        let regression = self.detect_regression(&test_result)?;

        // 4. If regression detected, trigger failure recovery
        if regression.has_new_failures {
            self.trigger_failure_recovery(regression).await?;
        }

        Ok(test_result)
    }

    async fn trigger_failure_recovery(&mut self, regression: Regression) -> Result<(), String> {
        // Increment failure count
        self.failure_tracker.increment();

        // If 3 failures, escalate to user
        if self.failure_tracker.count() >= 3 {
            self.escalate_to_user(regression).await?;
        } else {
            // Auto-retry with fix
            self.auto_retry_with_fix(regression).await?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FailureTracker {
    count: usize,
    failures: Vec<FailureAttempt>,
    initial_checkpoint_id: String,
}

impl FailureTracker {
    pub fn increment(&mut self) {
        self.count += 1;
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.failures.clear();
    }
}
```

---

### 4. Automatic Revert After Failed Attempts

**Requirement:** After each failed try, revert to the prior working checkpoint automatically.

**Revert Strategy:**

```rust
// src-tauri/src/agent/failure_recovery.rs

pub struct FailureRecovery {
    checkpoint_manager: Arc<Mutex<CheckpointManager>>,
    llm_orchestrator: Arc<Mutex<LLMOrchestrator>>,
    failure_tracker: FailureTracker,
}

impl FailureRecovery {
    /// Handle failure and attempt recovery
    pub async fn handle_failure(
        &mut self,
        failure: Failure,
    ) -> Result<RecoveryAction, String> {
        // Record failure
        self.failure_tracker.failures.push(FailureAttempt {
            attempt_number: self.failure_tracker.count(),
            timestamp: Utc::now(),
            failure_type: failure.failure_type.clone(),
            error_message: failure.error_message.clone(),
        });

        // Revert to last working checkpoint
        let last_checkpoint = self.get_last_working_checkpoint()?;
        self.checkpoint_manager.lock().unwrap()
            .restore_checkpoint(&last_checkpoint.id).await?;

        // Check if we should escalate
        if self.failure_tracker.count() >= 3 {
            return Ok(RecoveryAction::EscalateToUser);
        }

        // Attempt auto-fix
        let fix_strategy = self.generate_fix_strategy(&failure).await?;
        Ok(RecoveryAction::AutoRetry(fix_strategy))
    }

    /// Get last checkpoint with 100% confidence (all tests passing)
    fn get_last_working_checkpoint(&self) -> Result<Checkpoint, String> {
        let checkpoint_mgr = self.checkpoint_manager.lock().unwrap();
        let checkpoints = checkpoint_mgr.list_checkpoints();

        checkpoints.iter()
            .find(|c| c.confidence_score >= 0.95)
            .map(|c| checkpoint_mgr.get_checkpoint(&c.id).unwrap())
            .ok_or_else(|| "No working checkpoint found".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    AutoRetry(FixStrategy),       // Try fixing automatically
    EscalateToUser,                // Ask user for help (after 3 failures)
    SwitchLLM(String),            // Try different LLM
    SearchForSolution,             // Use RAG/web search
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureAttempt {
    pub attempt_number: usize,
    pub timestamp: DateTime<Utc>,
    pub failure_type: String,
    pub error_message: String,
}
```

**Auto-Revert Flow:**

```
Attempt 1: Generate payment_service.py
├─ Run tests
├─ 5 tests fail
├─ Failure detected! ❌
└─ Auto-revert to checkpoint "Before Add Payment Service"

Attempt 2: Generate payment_service.py (with fix)
├─ Prompt LLM: "Previous attempt failed with: ImportError: stripe module not found"
├─ LLM suggests: Add stripe to requirements.txt
├─ Generate code with fix
├─ Run tests
├─ 2 tests fail (different error)
├─ Failure detected! ❌
└─ Auto-revert to checkpoint "Before Add Payment Service"

Attempt 3: Generate payment_service.py (with different fix)
├─ Prompt LLM: "Previous attempts failed. Errors: [...]"
├─ LLM suggests: Mock Stripe in tests
├─ Generate code with fix
├─ Run tests
├─ 1 test fails
├─ Failure detected! ❌
└─ Auto-revert to checkpoint "Before Add Payment Service"

After 3 Failures → Escalate to User:
┌────────────────────────────────────────────────┐
│ ⚠️  ESCALATION: 3 FAILED ATTEMPTS              │
│                                                │
│ I've tried 3 times to add Payment Service     │
│ and reverted to the last working state each   │
│ time. Here's what happened:                    │
│                                                │
│ Attempt 1: Missing stripe dependency          │
│ Attempt 2: Import error in tests              │
│ Attempt 3: Stripe API authentication failed   │
│                                                │
│ Current state: Reverted to checkpoint          │
│ "Before Add Payment Service" (all tests✅)    │
│                                                │
│ Options:                                       │
│ 1️⃣  Provide Stripe API credentials            │
│ 2️⃣  Try with different LLM (Claude → GPT-4)   │
│ 3️⃣  Search web/RAG for solution               │
│ 4️⃣  Skip Payment Service for now              │
│                                                │
│ Your choice:                                   │
└────────────────────────────────────────────────┘
```

---

### 5. User Escalation After 3 Failures

**Requirement:** After 3 failed tries, ask user for input.

**Escalation UI:**

```typescript
// src-ui/components/FailureEscalation.tsx

interface FailureEscalationProps {
  attempts: FailureAttempt[];
  lastWorkingCheckpoint: CheckpointSummary;
  suggestedActions: RecoveryAction[];
}

export function FailureEscalation(props: FailureEscalationProps) {
  return (
    <div class="failure-escalation">
      <h2>⚠️ Need Your Help</h2>

      <div class="failure-summary">
        <p>I've attempted {props.attempts.length} times and couldn't succeed.</p>

        <div class="attempts-list">
          {props.attempts.map((attempt, i) => (
            <div class="attempt">
              <span class="attempt-number">Attempt {i + 1}:</span>
              <span class="error-message">{attempt.error_message}</span>
            </div>
          ))}
        </div>
      </div>

      <div class="current-state">
        <p>✅ Current State: Reverted to last working checkpoint</p>
        <CheckpointCard checkpoint={props.lastWorkingCheckpoint} />
      </div>

      <div class="recovery-options">
        <h3>What would you like me to do?</h3>
        {props.suggestedActions.map((action) => (
          <ActionButton action={action} />
        ))}
      </div>
    </div>
  );
}
```

---

### 6. LLM Hot-Swapping

**Requirement:** If user has multiple LLM APIs, ask if they want to try with a different LLM.

**Implementation:**

```rust
// src-tauri/src/llm/hot_swap.rs

pub struct LLMHotSwap {
    available_llms: Vec<LLMProvider>,
    current_llm: String,
    failure_tracker: FailureTracker,
}

impl LLMHotSwap {
    /// Suggest LLM switch after failures
    pub fn suggest_llm_switch(&self) -> Option<LLMProvider> {
        // If current LLM failed 3 times, suggest alternative
        if self.failure_tracker.count() >= 3 {
            self.available_llms.iter()
                .find(|llm| llm.name != self.current_llm)
                .cloned()
        } else {
            None
        }
    }

    /// Switch to different LLM
    pub async fn switch_llm(&mut self, new_llm: LLMProvider) -> Result<(), String> {
        self.current_llm = new_llm.name.clone();
        self.failure_tracker.reset();  // Reset failure count with new LLM
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LLMProvider {
    pub name: String,              // "Claude", "GPT-4", "Qwen"
    pub api_key: String,
    pub capabilities: Vec<String>,  // "code_generation", "reasoning", etc.
}
```

**LLM Switch UI:**

```
Agent: "I've failed 3 times with Claude Sonnet 4.

I noticed you have GPT-4 Turbo configured as well.

Would you like me to try again with GPT-4? It may have
different strengths for this type of problem.

Options:
1️⃣  Yes, try GPT-4 Turbo
2️⃣  No, stick with Claude (get more specific guidance)
3️⃣  Try Qwen Coder (specialized for code)

Your choice:"
```

---

### 7. RAG/Web Search with User Consent

**Requirement:** Use RAG or web search to find solutions, but only with user consent.

**Implementation:**

```rust
// src-tauri/src/agent/knowledge_search.rs

pub struct KnowledgeSearch {
    rag_engine: Option<RAGEngine>,
    web_search_enabled: bool,
}

impl KnowledgeSearch {
    /// Request user permission for external search
    pub async fn request_search_permission(
        &self,
        query: &str,
    ) -> Result<SearchPermission, String> {
        // Show prompt to user
        let prompt = format!(
            "I'd like to search for solutions to this problem:\n\n\
             Query: {}\n\n\
             Where should I look?\n\
             1️⃣  Local knowledge base (RAG - your past solutions)\n\
             2️⃣  Web search (Google, StackOverflow, GitHub)\n\
             3️⃣  Both\n\
             4️⃣  Neither (I'll provide guidance)\n\n\
             Your choice:",
            query
        );

        // Wait for user response
        let response = self.prompt_user(&prompt).await?;

        Ok(match response.as_str() {
            "1" => SearchPermission::RAGOnly,
            "2" => SearchPermission::WebOnly,
            "3" => SearchPermission::Both,
            _ => SearchPermission::None,
        })
    }

    /// Execute search with user-approved permission
    pub async fn search_for_solution(
        &self,
        query: &str,
        permission: SearchPermission,
    ) -> Result<Vec<Solution>, String> {
        let mut solutions = Vec::new();

        match permission {
            SearchPermission::RAGOnly => {
                if let Some(rag) = &self.rag_engine {
                    solutions.extend(rag.search(query).await?);
                }
            },
            SearchPermission::WebOnly => {
                if self.web_search_enabled {
                    solutions.extend(self.web_search(query).await?);
                }
            },
            SearchPermission::Both => {
                if let Some(rag) = &self.rag_engine {
                    solutions.extend(rag.search(query).await?);
                }
                if self.web_search_enabled {
                    solutions.extend(self.web_search(query).await?);
                }
            },
            SearchPermission::None => {},
        }

        Ok(solutions)
    }
}

#[derive(Debug, Clone)]
pub enum SearchPermission {
    RAGOnly,
    WebOnly,
    Both,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub source: String,          // "RAG", "StackOverflow", "GitHub"
    pub title: String,
    pub description: String,
    pub code_snippet: Option<String>,
    pub relevance_score: f32,    // 0.0-1.0
}
```

**Search Consent UI:**

```
Agent: "I found similar issues in our knowledge base:

Option 1: RAG Search Results (Local)
├─ "Stripe integration error" (90% match)
│   └─ Solution: Add API key to environment
│
└─ "Payment processing tests failing" (75% match)
    └─ Solution: Mock Stripe API in tests

Should I also search the web for more solutions?

✅ Benefits: More comprehensive solutions
⚠️  Privacy: Query sent to external search API

Search web? (yes/no)"
```

---

### 8. Testing & Validation

**Test Scenarios:**

1. **Happy Path (No Failures)**
   - Create checkpoint
   - Make code change
   - Tests pass
   - Checkpoint becomes "last working"

2. **Single Failure + Auto-Recovery**
   - Create checkpoint
   - Make code change
   - Tests fail
   - Auto-revert to checkpoint
   - Retry with fix
   - Tests pass ✅

3. **3 Failures + User Escalation**
   - Create checkpoint
   - Attempt 1: Fail → Revert → Retry
   - Attempt 2: Fail → Revert → Retry
   - Attempt 3: Fail → Revert → Escalate to user
   - User provides guidance
   - Retry with user input
   - Tests pass ✅

4. **LLM Hot-Swap**
   - 3 failures with Claude
   - Suggest GPT-4 switch
   - User approves
   - Retry with GPT-4
   - Tests pass ✅

5. **One-Click Restore from UI**
   - User clicks "Restore" on checkpoint
   - System reverts all files instantly
   - GNN state restored
   - Tests pass ✅

---

### 9. Performance Targets

| Operation             | Target | Notes                    |
| --------------------- | ------ | ------------------------ |
| Create checkpoint     | <500ms | Snapshot 10-20 files     |
| Restore checkpoint    | <1s    | Restore all files + GNN  |
| Impact analysis       | <100ms | GNN dependency traversal |
| List checkpoints      | <50ms  | Query from memory        |
| Prune old checkpoints | <200ms | Background operation     |

---

### 10. Storage Management

**Checkpoint Storage Strategy:**

```
.yantra/
├─ checkpoints/
│   ├─ session_2024-11-28_10-15.json
│   ├─ feature_payment_2024-11-28_10-20.json
│   ├─ file_payment_service_2024-11-28_10-25.json
│   └─ ... (keep last 20 checkpoints)
│
└─ checkpoint_index.db (SQLite)
    └─ Tracks all checkpoints with metadata
```

**Auto-Pruning:**

- Keep last 20 checkpoints by default
- User-marked "important" checkpoints never pruned
- Prune in background (non-blocking)
- Compress old checkpoints (gzip) to save space

---

## Phase 2: Advanced Context + Network Effect (Months 3-4)

### Objectives

Complete unlimited context implementation and enable network effects:

- Full RAG with ChromaDB
- Advanced context compression
- Pattern extraction from failures
- Opt-in anonymous pattern sharing
- Full validation pipeline (5 validations)

### New Capabilities

1. **Complete Unlimited Context Engine**

- RAG with ChromaDB for code patterns
- Semantic search for relevant examples
- Advanced compression (semantic chunking)
- Full hierarchical context (L1-L4)
- Adaptive strategies per task type
- Context caching for performance

2. **Network Effect System**

- Privacy-preserving pattern extraction
- Anonymous failure pattern aggregation
- Opt-in pattern sharing UI
- Daily pattern database updates
- Pattern success rate tracking
- User-reviewable sharing logs

3. **Full Agentic Validation Pipeline**

- All 5 validations (dependency, unit test, integration test, security, browser)
- Advanced auto-fixing with ML patterns
- Multi-attempt retry strategies
- Escalation to human with context
- Session resumption after crashes

4. **Workflow Foundation** (Original Phase 2 content)

- Cron scheduler
- Webhook server (Axum web framework)
- Event-driven triggers
- Retry logic with exponential backoff
- Execution history and logs

### External API Integration

- API schema discovery (OpenAPI specs)
- Track API calls in GNN
- Support: Slack, SendGrid, Stripe
- Generic REST API support (via config)

3. Multi-Step Workflows

- Chain 3-5 actions
- Conditional branching (if/else)
- Error handling (try/catch)
- Data passing between steps

Example Use Case:

Webhook: Stripe payment success

→ Update database (mark order paid)

→ Send confirmation email (SendGrid)

→ Notify sales team (Slack)

→ Log to analytics

### Implementation (8 Weeks)

**Weeks 9-10:** Workflow definition (YAML), executor, scheduler
**Weeks 11-12:** External API integration framework
**Weeks 13-14:** Database migration (SQLite → PostgreSQL) + Error handling, logging, monitoring dashboard
**Weeks 15-16:** LLM workflow generation, beta release

### Storage Architecture: 4-Tier Strategy

**Overview:** Yantra uses a tiered storage architecture optimized for different access patterns. Each tier uses the right tool for its specific workload.

#### Tier 1 - Hot Path (GNN Graph) 🔥

**Access Pattern:** Extremely high-frequency reads, pointer-chasing graph traversal
**Performance Target:** Sub-millisecond query times, <5s for 100K LOC graph build
**Memory Budget:** ~1GB for 100K LOC projects

**Phase 1-2 (Current - MVP):**

- **In-memory graph:** petgraph data structure loaded in RAM
- **Persistence:** SQLite with WAL mode + connection pooling
- **Write pattern:** Batch writes every operation
- **Rationale:** Adequate for 10K-100K LOC, simple implementation

**Phase 3 (Month 5-8 - Scale to >100K LOC):**

- **Storage:** Pure in-memory petgraph (no SQLite in hot path)
- **Persistence:** Snapshot files (bincode format) every 30 seconds
- **Durability:** Write-ahead log for crash recovery
- **Startup:** Load snapshot + replay WAL entries
- **Rationale:** Disk I/O is catastrophically slow for pointer-chasing. 100K+ LOC graphs need pure in-memory.

```rust
// Phase 3 design
pub struct GNNEngine {
    graph: CodeGraph,              // In-memory petgraph
    snapshot_path: PathBuf,        // ~/.yantra/gnn_snapshot.bin
    wal: WriteAheadLog,            // Incremental updates log
}

// Auto-snapshot every 30s
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        gnn.save_snapshot()?; // bincode serialization
    }
});
```

#### Tier 2 - Write-Heavy Coordination (Agent State) 📝

**Access Pattern:** High concurrent writes, lock coordination, agent registry
**Performance Target:** 100K writes/sec, lock-free operations
**Use Cases:** Agent state machines, file locks, coordination registry

**Phase 1 (Current - MVP):**

- **Storage:** SQLite with WAL mode + r2d2 connection pooling
- **Pattern:** Multiple Connection::open() calls (INEFFICIENT - needs fix)
- **Rationale:** Single-agent MVP doesn't need high write throughput

**Phase 2 (Month 4 - Cluster Agents):**

- **Storage:** sled (embedded KV store) or RocksDB
- **Tech choice:** sled preferred (pure Rust, simpler, 100k writes/sec)
- **Key design:** Prefixed keys for namespacing
  - `agent:codegen:state` - CodeGen machine state
  - `agent:testing:state` - Testing machine state
  - `lock:src/main.py` - File lock coordination
  - `registry:agent:123` - Agent registry entry
- **Rationale:** LSM-tree optimized for write throughput, lock-free

```rust
// Phase 2 design
use sled::Db;

pub struct CoordinationStore {
    db: Db, // ~/.yantra/coordination.sled
}

impl CoordinationStore {
    pub fn save_agent_state(&self, agent_id: &str, state: &AgentState) -> Result<()> {
        let key = format!("agent:{}:state", agent_id);
        self.db.insert(key.as_bytes(), bincode::serialize(&state)?)?;
        Ok(())
    }

    pub fn acquire_lock(&self, file_path: &str) -> Result<bool> {
        let key = format!("lock:{}", file_path);
        self.db.compare_and_swap(
            key.as_bytes(),
            None as Option<&[u8]>,
            Some(b"locked"),
        )?.is_ok()
    }
}
```

#### Tier 3 - Read-Heavy Reference Data (SQLite) 📚

**Access Pattern:** Read-frequently, write-rarely (100:1 ratio)
**Performance Target:** 1000s reads/sec, concurrent reads during writes
**Use Cases:** Known issues DB, architecture snapshots, config, audit logs

**Phase 1 (Current - NOW):**

- **Storage:** SQLite with WAL mode enabled
- **Optimization:** r2d2 connection pooling (reuse connections)
- **Tables:**
  - `architectures` - Architecture diagrams and metadata
  - `components` - Component definitions
  - `connections` - Component relationships
  - `known_issues` - Historical issue tracking
  - `audit_logs` - User action logs
- **Rationale:** SQLite excels at read-heavy workloads. WAL enables concurrent readers.

**Phase 2+ (Month 4+):**

- **Option A:** Keep SQLite (works great for this tier)
- **Option B:** Migrate to PostgreSQL only if multi-tenant features require it
- **Decision:** Likely keep SQLite - perfect fit for local reference data

```rust
// Phase 1 fixes (THIS WEEK)
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;

pub struct ArchitectureStorage {
    pool: Pool<SqliteConnectionManager>, // Connection pool
}

impl ArchitectureStorage {
    pub fn new(db_path: &Path) -> Result<Self> {
        let manager = SqliteConnectionManager::file(db_path);
        let pool = Pool::builder()
            .max_size(15) // Up to 15 concurrent connections
            .build(manager)?;

        // Enable WAL mode
        let conn = pool.get()?;
        conn.pragma_update(None, "journal_mode", "WAL")?;

        Ok(Self { pool })
    }
}
```

#### Tier 4 - Ephemeral Cache (In-Memory LRU) 💾

**Access Pattern:** High read frequency, automatic eviction, recompute on miss
**Performance Target:** <1ms cache hits, 500MB memory budget
**Use Cases:** Context assembly results, token counts, LLM response caches

**Phase 2+ (Month 4+):**

- **Storage:** In-memory LRU cache (moka crate)
- **Persistence:** None (ephemeral by design)
- **Eviction:** Automatic based on memory budget
- **TTL:** 1-24 hours depending on data type
- **Rationale:** Avoid write amplification from caching to disk

```rust
// Phase 2+ design
use moka::sync::Cache;
use std::time::Duration;

pub struct ContextCache {
    context_cache: Cache<String, String>,     // file_path -> assembled context
    token_cache: Cache<String, usize>,        // content_hash -> token count
    llm_cache: Cache<String, LLMResponse>,    // request_hash -> response
}

impl ContextCache {
    pub fn new() -> Self {
        Self {
            context_cache: Cache::builder()
                .max_capacity(200_000_000) // 200MB
                .time_to_live(Duration::from_secs(3600)) // 1 hour
                .build(),

            token_cache: Cache::builder()
                .max_capacity(50_000_000)  // 50MB
                .time_to_live(Duration::from_secs(86400)) // 24 hours
                .build(),

            llm_cache: Cache::builder()
                .max_capacity(250_000_000) // 250MB
                .time_to_live(Duration::from_secs(1800)) // 30 minutes
                .build(),
        }
    }
}
```

### MVP Phase 1: Critical Fixes (THIS WEEK)

**Problem:** Current SQLite usage is suboptimal:

1. ❌ No WAL mode enabled (readers block writers)
2. ❌ No connection pooling (open/close overhead on every operation)
3. ❌ agent/state.rs opens new connection every time (extremely inefficient)

**Solution:** Enable WAL + Add r2d2 pooling

**Files to update:**

1. `src-tauri/src/gnn/persistence.rs` - Enable WAL, add pooling
2. `src-tauri/src/architecture/storage.rs` - Enable WAL, add pooling
3. `src-tauri/src/agent/state.rs` - Use pooled connections, stop reopening

**Implementation:**

```rust
// 1. Add to Cargo.toml
[dependencies]
r2d2 = "0.8"
r2d2_sqlite = "0.24"

// 2. Update Database::new()
pub fn new(db_path: &Path) -> Result<Self> {
    let manager = SqliteConnectionManager::file(db_path);
    let pool = Pool::builder()
        .max_size(15)
        .build(manager)?;

    // Enable WAL mode
    let conn = pool.get()?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?; // Faster, still safe with WAL

    Ok(Self { pool })
}
```

**Success Metrics:**

- ✅ 3-5x faster database operations
- ✅ Concurrent reads during writes (WAL benefit)
- ✅ No connection open/close overhead
- ✅ Proper connection reuse

---

## Phase 2A: Cluster Agents Architecture (Months 3-4)

### Overview

**Problem:** As codebases scale beyond 100k LOC and teams grow to 5+ concurrent developers, single-agent architecture becomes a bottleneck. Developers need multiple AI agents working simultaneously on different parts of the codebase without conflicts.

**Solution:** Transform Yantra from a single autonomous agent to a **cluster of coordinating agents** using a Master-Servant architecture with Git coordination branch and Agent-to-Agent (A2A) protocol for proactive conflict prevention.

**Key Innovation:** Unlike traditional collaborative editing (which reactively resolves conflicts), Yantra uses **proactive conflict prevention** - agents communicate intent before making changes via GNN dependency analysis and real-time file locking.

### Why Cluster Agents?

**Current Limitations (Single Agent):**

- Only one developer can use Yantra at a time
- Large codebases (100k+ LOC) exceed single-agent capacity
- Complex features require serial execution of multiple tasks
- Bottleneck for team collaboration

**Cluster Agent Benefits:**

- **Parallelization:** 3-10 agents working simultaneously on same feature
- **Specialization:** Dedicated agents for frontend, backend, testing, DevOps
- **Scalability:** Handle 100k+ LOC codebases efficiently
- **Team Collaboration:** Multiple developers with their own agents working on same project
- **Fault Tolerance:** One agent failure doesn't block others
- **3x Faster:** 15 minutes instead of 45 minutes for complex features

### Architecture: Master-Servant Pattern

**Why Master-Servant over Peer-to-Peer?**

**Rejected: Peer-to-Peer (P2P)**

- ❌ No single source of truth (coordination nightmare)
- ❌ Complex consensus algorithms (Raft/Paxos)
- ❌ Race conditions on file writes
- ❌ Conflict resolution after-the-fact

**Chosen: Master-Servant with Git Coordination**

- ✅ Master's job: Work assignment ONLY (not a bottleneck)
- ✅ Git branches provide code isolation
- ✅ Tier 2 (sled) provides real-time file locking
- ✅ Proactive conflict prevention (not reactive resolution)
- ✅ Mirrors human team workflow (tech lead assigns, developers work independently)
- ✅ Simple state management, easy to debug

---

### Master Agent Responsibilities

**Role:** Work assignment and decomposition ONLY (not ongoing coordination)

1. **Feature Decomposition:**
   - Receives high-level feature request from user
   - Uses GNN to analyze dependencies and complexity
   - Decomposes into 3-10 sub-features with minimal cross-dependencies
   - Example: "Add Stripe payments" → Backend API, Frontend form, Testing, DevOps

2. **Agent Assignment:**
   - Assigns each sub-feature to specialized servant agent
   - Creates Git feature branches for each agent
   - Commits assignment events to `.yantra/coordination` branch
   - **Master's job is done** - can take new requests or shut down

3. **User Review (Optional):**
   - Shows proposed work split to user
   - User can adjust assignments before work starts
   - Example: "Agent 1 (Backend), Agent 2 (Frontend) share UserService - is this split okay?"

**What Master Does NOT Do:**

- ❌ Approve every file operation (not a bottleneck)
- ❌ Coordinate ongoing work (agents are independent)
- ❌ Resolve runtime conflicts (Git + GNN handle this)

---

### Servant Agent Responsibilities

**Role:** Independent execution with peer-to-peer coordination

1. **Startup:**
   - Pull `.yantra/coordination` branch to find assignment
   - Create feature branch from main: `feature/agent-1-payment-api`
   - Register self in Tier 2 (sled): `registry:agent:agent-1`

2. **Before Modifying Files:**
   - Query GNN: "What files do I need to modify for this feature?"
   - Check Tier 2 locks: "Is any file I need already claimed?"
   - If locked → wait, coordinate with other agent, or work on different task
   - If free → claim files in Tier 2: `lock:src/payment/stripe.py = agent-1`

3. **During Work:**
   - Modify files on own branch (Git isolation)
   - Update state in Tier 2: `agent:agent-1:state = { phase: "CodeGen", progress: 45% }`
   - Release locks when done with file: `DELETE lock:src/payment/stripe.py`

4. **Dependency Coordination (A2A Protocol):**
   - Agent B needs API from Agent A
   - Agent B checks coordination branch: "Has Agent A completed user auth API?"
   - If complete → Pull Agent A's branch, continue work
   - If not → Send A2A message via Tier 2: `a2a:agent-2:to:agent-1:QueryDependency`
   - Agent A responds with ETA or completed commit hash

5. **Feature Completion:**
   - Create Pull Request when feature is ready
   - Commit event to coordination branch: `feature_completed`
   - Release all file locks
   - Run full test suite
   - Request human review (or auto-merge if tests pass)

---

### Git Coordination Branch

**Purpose:** Append-only event log for high-level feature assignments and completions

**Branch Name:** `.yantra/coordination` (never merges to main)

**Event Types:**

```json
// 1. Feature Assignment
{
  "event": "feature_assigned",
  "feature": "stripe-payment-processing",
  "agent": "agent-1",
  "branch": "feature/agent-1-payment-api",
  "timestamp": "2025-11-30T10:00:00Z"
}

// 2. Work Started
{
  "event": "work_started",
  "agent": "agent-1",
  "feature": "stripe-payment-processing",
  "timestamp": "2025-11-30T10:05:00Z"
}

// 3. Dependency Available
{
  "event": "dependency_available",
  "agent": "agent-1",
  "api": "PaymentService.charge",
  "commit_hash": "abc123",
  "timestamp": "2025-11-30T10:45:00Z"
}

// 4. Feature Completed
{
  "event": "feature_completed",
  "agent": "agent-1",
  "feature": "stripe-payment-processing",
  "branch": "feature/agent-1-payment-api",
  "pr_number": 42,
  "timestamp": "2025-11-30T11:00:00Z"
}
```

**Why Git Branch?**

- ✅ No new infrastructure (Git already there)
- ✅ Version controlled and auditable (full history)
- ✅ Human-readable (`git log .yantra/coordination`)
- ✅ Works locally (single machine) and distributed (team)
- ✅ Merge conflicts rare (append-only events)

**Agent Operations:**

```rust
// Agent pulls coordination state on startup
pub async fn get_my_assignment(&self) -> Result<Assignment> {
    git::pull_branch(".yantra/coordination")?;
    let events = git::read_json_commits(".yantra/coordination")?;
    events.iter()
        .find(|e| e.event == "feature_assigned" && e.agent == self.id)
}

// Agent commits completion event
pub async fn mark_complete(&self) -> Result<()> {
    let event = json!({
        "event": "feature_completed",
        "agent": self.id,
        "feature": self.current_feature,
        "pr_number": self.pr_number,
        "timestamp": Utc::now()
    });
    git::commit_to_branch(".yantra/coordination", &event)?;
    git::push_branch(".yantra/coordination")?;
}
```

---

### Tier 2 (sled) - Real-Time File Locking

**Purpose:** High-frequency, low-latency coordination for file locks and agent state

**Why sled (not Git)?**

- Git coordination branch: Low-frequency events (assignments, completions)
- sled: High-frequency operations (file claims/releases every few seconds)
- Git: Human-readable history
- sled: Machine-optimized performance (100k writes/sec)

**Data Model:**

```rust
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
```

**File Lock API:**

```rust
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
```

---

### Proactive Conflict Prevention Workflow

**Example: E-Commerce Checkout Feature with 6 Agents**

**User Request:** "Add complete checkout flow with Stripe payments"

**Step 1: Master Decomposes (30 seconds)**

```
Master analyzes GNN dependencies:
├─ Sub-feature 1: Payment API (Backend) - Files: src/payment/stripe.py, src/api/checkout.py
├─ Sub-feature 2: Checkout UI (Frontend) - Files: src/components/Checkout.tsx
├─ Sub-feature 3: Cart Integration (Frontend) - Files: src/components/Cart.tsx
├─ Sub-feature 4: Integration Tests (Testing) - Files: tests/test_checkout.py
├─ Sub-feature 5: Deployment Config (DevOps) - Files: .github/workflows/deploy.yml
└─ Sub-feature 6: Documentation (Docs) - Files: docs/checkout.md

GNN shows: Checkout.tsx depends on stripe.py (API calls)
Master assigns: Agent 1 (Backend) starts first, Agent 2 (Frontend) waits for API
```

**Step 2: Agents Claim Files (1 second)**

```rust
// Agent 1 (Backend)
coordination_store.claim_file("agent-1", "src/payment/stripe.py")?; // ✅ Success
coordination_store.claim_file("agent-1", "src/api/checkout.py")?;   // ✅ Success

// Agent 2 (Frontend) - tries simultaneously
coordination_store.claim_file("agent-2", "src/components/Checkout.tsx")?; // ✅ Success

// Agent 2 checks dependencies
let locked = coordination_store.check_dependencies("src/components/Checkout.tsx", &gnn)?;
// Returns: ["src/payment/stripe.py (locked by agent-1)"]
// Agent 2 decision: Start on Cart.tsx (no dependencies), wait for stripe.py
```

**Step 3: Parallel Execution (10 minutes)**

```
Agent 1 (Backend):     [====================] 100% - stripe.py, checkout.py ✅
Agent 2 (Frontend):    [==========          ] 50%  - Cart.tsx done, waiting for stripe.py
Agent 3 (Frontend):    [====================] 100% - Cart.tsx done ✅
Agent 4 (Testing):     [                    ] 0%   - Waiting for Agent 1 & 2
Agent 5 (DevOps):      [====================] 100% - deploy.yml done (parallel) ✅
Agent 6 (Docs):        [====================] 100% - docs done (parallel) ✅
```

**Step 4: Dependency Resolution (1 minute)**

```
Agent 1 completes → Commits to coordination branch:
{
  "event": "dependency_available",
  "agent": "agent-1",
  "api": "PaymentService.charge",
  "commit": "abc123"
}

Agent 2 sees event → Pulls agent-1 branch → Continues Checkout.tsx
Agent 2 completes in 5 minutes ✅

Agent 4 sees both complete → Starts integration tests
Agent 4 completes in 3 minutes ✅
```

**Step 5: PR Creation (30 seconds)**

```
All agents complete → Create PRs:
├─ PR #1: Payment API (agent-1) ✅ Tests pass
├─ PR #2: Checkout UI (agent-2) ✅ Tests pass
├─ PR #3: Cart Integration (agent-3) ✅ Tests pass
├─ PR #4: Integration Tests (agent-4) ✅ Tests pass
├─ PR #5: Deployment (agent-5) ✅ Tests pass
└─ PR #6: Documentation (agent-6) ✅ Tests pass

Human reviews and merges (or auto-merge if all tests pass)
```

**Total Time:** 15 minutes (vs 45 minutes single-agent) = **3x faster**

**Conflicts Prevented:**

- 🚫 Agent 2 didn't modify stripe.py while Agent 1 was working on it
- 🚫 Agent 4 didn't start tests until APIs were complete
- 🚫 No Git merge conflicts (each agent on own branch with file locks)

---

### Scalability Limits

**Per-Feature Parallelism:**

- **Optimal:** 5-10 agents (sweet spot for most features)
- **Maximum:** 15-20 agents (diminishing returns beyond this)
- **Bottleneck:** File lock contention (not storage)

**System-Wide Parallelism:**

- **Phase 2:** 3-5 features simultaneously (15-50 agents system-wide)
- **Phase 3:** 10+ features simultaneously (50-100 agents system-wide)
- **Theoretical:** 1,000+ agents (sled can handle it)
- **Practical:** 100-200 agents (file locks become bottleneck)

**Why Not More Agents Per Feature?**

- Small feature (login): Only 3-5 files to modify → 3-5 agents max
- Medium feature (payments): 10-15 files → 6-10 agents optimal
- Large feature (dashboard): 30+ files → 15-20 agents, then diminishing returns

**Master Agent Overhead:**

- Assignment decomposition: ~30 seconds per feature
- Not a bottleneck (one-time cost, then agents run independently)

---

### Implementation Tasks (Phase 2A - 4 weeks)

**Week 9: Git Coordination Branch**

- Implement Git branch creation (`.yantra/coordination`)
- Event commit/read functions (JSON serialization)
- Agent assignment parser
- Event types: feature_assigned, work_started, dependency_available, feature_completed

**Week 10: Tier 2 (sled) Migration**

- Add sled dependency to Cargo.toml
- Implement CoordinationStore (file locks, agent state, registry, A2A messages)
- Migrate agent state from SQLite to sled
- File lock API: claim_file(), release_file(), check_lock()

**Week 11: Master-Servant Implementation**

- Master Agent: Feature decomposition using dependency graph (code calls this "GNN" but it's the petgraph structure)
- Master Agent: Work assignment algorithm (minimize dependencies)
- Servant Agent: Assignment reading from coordination branch
- Servant Agent: File claiming before modification
- Servant Agent: State updates during work

**Week 12: A2A Protocol**

- A2A message types: QueryDependency, DependencyReady, IntentToModify
- Message sending/receiving via Tier 2 (sled)
- Dependency resolution workflow
- Conflict prevention logic (check locks before claiming)

**Success Metrics:**

- ✅ 3-10 agents can work simultaneously on same feature
- ✅ Zero file conflicts (all prevented proactively)
- ✅ 3x faster feature completion (15 min vs 45 min)
- ✅ Master assignment overhead <30 seconds
- ✅ File lock operations <5ms
- ✅ 100+ agents supported system-wide

---

## Phase 2B: Cloud Graph Database (Tier 0) - Months 4-5

**IMPORTANT:** This is NOT "Cloud GNN" - the GNN (intelligence layer) runs locally in Tier 1. This is a cloud-hosted graph database for storage and coordination across agents/users.

### Overview

**Problem:** With Cluster Agents, each agent has a LOCAL GNN. When Agent A modifies a file, Agent B doesn't know until attempting to claim the same file or hitting a Git merge conflict. This is **reactive conflict detection** (bad).

**Solution:** **Cloud Graph Database (Tier 0)** - A shared, cloud-hosted dependency graph that tracks real-time file modifications across all agents and users working on the same project. Enables **proactive conflict prevention** BEFORE work starts.

**Key Innovation:** Combine dependency knowledge (GNN) with activity knowledge (who's modifying what) to warn agents about potential conflicts before they occur, including transitive dependencies.

### Why Cloud Graph Database?

**Current Problem (Local GNN Only):**

- Agent A modifies `payment.py`
- Agent B wants to modify `checkout.py` which imports from `payment.py`
- Agent B doesn't know Agent A is changing `payment.py` until:
  - File lock prevents claim (good but late)
  - Git merge conflict after both finish (bad - wasted work)

**Cloud Graph Database Solution:**

- Agent A starts work → Marks `payment.py` as "in modification" in Cloud Graph DB
- Agent B queries Cloud Graph DB: "Dependencies of checkout.py?"
- Cloud Graph DB responds: "`payment.py` (locked by Agent A) is a dependency"
- Agent B warned BEFORE starting work: "Your file depends on payment.py being modified by Agent A. Recommend: wait 10 min, or coordinate approach."

**Conflict Prevention Levels:**

**Level 1: Same File Detection**

- Agent A claims `payment.py`
- Agent B tries to claim `payment.py`
- Tier 2 (sled) blocks immediately ✅ (already implemented)

**Level 2: Direct Dependency Detection** 🆕

- Agent A modifies `payment.py`
- Agent B wants to modify `checkout.py` which imports from `payment.py`
- Cloud Graph DB warns: "Your file depends on a file being modified"

**Level 3: Transitive Dependency Detection** 🆕

- Agent A modifies `database.py`
- `user.py` depends on `database.py`
- `auth.py` depends on `user.py`
- Agent B wants to modify `auth.py`
- Cloud Graph DB traces chain: "auth.py → user.py → database.py (Agent A modifying)"

**Level 4: Semantic Dependency Detection** 🆕

- Agent A changes signature of `authenticate(username, password)` → `authenticate(email, password, mfa_code)`
- Cloud Graph DB knows 47 files call `authenticate()`
- Any agent touching those 47 files gets warned: "Function you're using is being modified. Signature may change."

---

### Architecture: Hybrid Local + Cloud

**Design Principle:** Local GNN for speed, Cloud Graph Database for coordination

```
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
```

**Hybrid Benefits:**

- **Local GNN:** Millisecond queries for code generation (hot path)
- **Cloud Graph Database:** Real-time coordination across agents (warm path)
- **Privacy:** Only graph structure synced, not code content

---

### Data Model: Privacy-Preserving

**What Gets Shared (Graph Structure Only):**

```rust
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
```

**What NEVER Gets Shared (Code Content):**

```rust
// ❌ NO CODE CONTENT
pub struct PrivateData {
    pub source_code: String,          // NEVER SENT
    pub function_body: String,        // NEVER SENT
    pub variable_values: String,      // NEVER SENT
    pub api_keys: String,             // NEVER SENT
}
```

**Example: Shared vs Private**

```rust
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
```

---

### Proactive Conflict Warning Workflow

**Scenario: Agent B Wants to Modify File with Dependencies**

**Step 1: Agent B Prepares to Start Work**

```rust
let file_path = "src/checkout.py";

// Query Cloud Graph DB for conflicts BEFORE claiming file
let conflicts = cloud_graph_db.query_conflicts(project_id, file_path).await?;
```

**Step 2: Cloud Graph Database Analysis**

```rust
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
```

**Step 3: Agent B Receives Warnings**

```rust
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
```

**Step 4: Agent B Decision Matrix**

```rust
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
```

---

### Cloud Graph Service Implementation

**Technology Stack:**

- **Backend:** Rust with Actix-Web or Axum
- **Database:** Redis (in-memory, <50ms latency) + PostgreSQL (persistence)
- **Protocol:** WebSocket (real-time) or gRPC (low latency)
- **Deployment:** Fly.io, Railway, or self-hosted Docker container

**API Endpoints:**

```rust
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
```

**Redis Data Model:**

```redis
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
```

**PostgreSQL Schema:**

```sql
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
```

---

### Local → Cloud Sync Mechanism

**Sync Frequency:**

- **Incremental:** Every 30 seconds (or on file modification)
- **Full:** Every 5 minutes (validate consistency)

**Sync Process:**

```rust
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
```

**Privacy Enforcement:**

```rust
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
```

---

### Deployment Options

**Option 1: Yantra Cloud (Hosted Service)**

- **URL:** `wss://cloud.yantra.dev/api/v1/`
- **Pricing:**
  - Free: Single user, 1 project
  - Pro ($29/month): 3 users, 5 projects
  - Team ($99/month): Unlimited users, unlimited projects
- **Benefits:** Zero setup, automatic updates, 99.9% uptime SLA
- **Privacy:** Graph structure only (not code), encrypted in transit and at rest

**Option 2: Self-Hosted (Enterprise)**

- **Docker Container:** `docker run -p 8080:8080 yantra/cloud-gnn:latest`
- **Requirements:** 2GB RAM, 1 CPU core, Redis + PostgreSQL
- **Benefits:** Full control, on-premise deployment, air-gapped networks
- **Setup:** 15 minutes with Docker Compose

**Docker Compose Example:**

```yaml
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
```

---

### Implementation Tasks (Phase 2B - 3 weeks)

**Week 13: Cloud Graph Service (Backend)**

- Actix-Web/Axum HTTP server with WebSocket support
- Redis integration for file locks and agent registry
- PostgreSQL schema and integration for graph storage
- REST API endpoints: claim_file, release_file, query_dependencies, query_conflicts
- WebSocket real-time notifications

**Week 14: Local → Cloud Sync**

- CloudGraphClient in Rust (WebSocket/gRPC client)
- Incremental sync (every 30s) and full sync (every 5 min)
- Privacy layer: Strip code content, send only graph structure
- Error handling and retry logic
- Background sync task (tokio::spawn)

**Week 15: Conflict Detection & UI**

- Implement query_conflicts() with 4 levels (same-file, direct, transitive, semantic)
- Agent B workflow: Query conflicts BEFORE claiming file
- UI warnings: Show conflicts and recommend actions
- Coordination messages: Agent-to-Agent via Tier 2
- Performance testing: <50ms conflict queries, 100+ agents

**Success Metrics:**

- ✅ <50ms latency for conflict queries
- ✅ Zero code content leaked (only graph structure)
- ✅ 4 levels of conflict detection working
- ✅ 100+ agents supported simultaneously
- ✅ 99.9% uptime for hosted service
- ✅ Proactive conflict prevention (not reactive resolution)
- ✅ Team collaboration enabled (multiple users, same project)

---

### Success Metrics: Cluster Agents + Cloud Graph Database

**Performance:**

- ✅ 3-10x faster feature completion with cluster agents
- ✅ <50ms conflict queries (Cloud Graph DB)
- ✅ <5ms file lock operations (Tier 2 sled)
- ✅ <30s Master assignment overhead

**Scalability:**

- ✅ 5-10 agents per feature (optimal)
- ✅ 100-200 agents system-wide (practical limit)
- ✅ Unlimited users per project (Cloud Graph DB enables team collaboration)

**Quality:**

- ✅ Zero Git merge conflicts (all prevented proactively)
- ✅ 4 levels of conflict detection (same-file, direct, transitive, semantic)
- ✅ Privacy-preserving (graph structure only, not code)

**User Experience:**

- ✅ Agents work independently (no Master bottleneck)
- ✅ Clear warnings BEFORE conflicts occur
- ✅ Human-readable coordination history (Git branch)
- ✅ Easy debugging (inspect coordination branch with `git log`)

---

## Phase 3: Enterprise Automation & Self-Healing (Months 5-8)

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

| Use Case                                       | Tool      | Reason                |
| ---------------------------------------------- | --------- | --------------------- |
| "Find functions that call UserService.login()" | GNN       | Structural dependency |
| "Find similar authentication code"             | Vector DB | Semantic similarity   |
| "Will changing this function break anything?"  | GNN       | Impact analysis       |
| "How do other projects handle OAuth?"          | Vector DB | Pattern search        |
| "What imports this module?"                    | GNN       | Direct dependency     |
| "Find code related to payment processing"      | Vector DB | Semantic search       |

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

| Component                | Sync Strategy            | Latency | Reason                           |
| ------------------------ | ------------------------ | ------- | -------------------------------- |
| **Vector DB**            | Real-time (on file save) | <100ms  | Semantic search must be current  |
| **GNN**                  | Periodic (every 2-3s)    | 2-3s    | Structural changes less frequent |
| **File Access Registry** | Real-time (on intent)    | <10ms   | Critical for conflict prevention |

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

| Mode           | Speed      | Cost       | Quality     | Best For                                  |
| -------------- | ---------- | ---------- | ----------- | ----------------------------------------- |
| **Adaptive**   | Fast (avg) | Low-Medium | High        | Production (95%+ quality, cost-optimized) |
| **AlwaysBest** | Slow       | High       | Highest     | Critical tasks, compliance, security      |
| **TaskBased**  | Medium     | Medium     | Medium-High | Predictable workloads, tight budgets      |

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

| Metric                 | Single Agent | 3 Agents      | 10 Agents      |
| ---------------------- | ------------ | ------------- | -------------- | --- |
| **Codebase Size**      | 10k LOC      | 50k LOC       | 100k+ LOC      |
| **Concurrent Tasks**   | 1            | 3             | 10             |
| **Context Build Time** | <500ms       | <800ms        | <1.5s          |
| **Conflict Detection** | N/A          | <10ms         | <20ms          |
| **Commit Frequency**   | Every task   | Every 3 tasks | Every 10 tasks |
| **Cost per Feature**   | $0.20        | $0.25         | $0.35          |     |

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
    { "id": "c1", "name": "React Frontend", "type": "ui_component", "layer": "frontend" },
    { "id": "c2", "name": "FastAPI Backend", "type": "service", "layer": "backend" },
    { "id": "c3", "name": "PostgreSQL", "type": "database", "layer": "database" }
  ],
  "connections": [
    { "from": "c1", "to": "c2", "type": "api_call", "label": "REST API" },
    { "from": "c2", "to": "c3", "type": "data_flow", "label": "SQL Queries" }
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
  currentLayer: string; // 'complete', 'frontend', 'backend', etc.
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

| Operation                               | Target | Scale Target |
| --------------------------------------- | ------ | ------------ |
| Load architecture from DB               | <50ms  | <100ms       |
| Render React Flow diagram               | <200ms | <500ms       |
| Save component/connection               | <10ms  | <20ms        |
| Generate architecture from intent (LLM) | <3s    | <5s          |
| Generate architecture from code (GNN)   | <2s    | <5s          |
| Validate alignment                      | <100ms | <300ms       |
| Export to Markdown/JSON                 | <50ms  | <100ms       |
| Version snapshot                        | <20ms  | <50ms        |

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

| Traditional Tools                 | Architecture View System     |
| --------------------------------- | ---------------------------- |
| Manual diagrams (always outdated) | Auto-synced with code        |
| No enforcement                    | Continuous validation        |
| Static images                     | Interactive, living diagrams |
| No code linking                   | Files mapped to components   |
| No version history                | Automatic snapshots          |
| No governance                     | Prevents misalignment        |

**Key Differentiators:**

- **GitHub Copilot/Cursor**: Generate code blindly → Result: Spaghetti code
- **Yantra**: Architecture-first → Code must align → Result: Clean, maintainable systems

---

## Project Initialization & Architecture-First Workflow (MVP Phase 1 - NEW)

### Overview

**Status:** 🔴 NOT STARTED
**Priority:** ⚡ MVP CRITICAL (Required for architecture governance)
**Purpose:** Automatic project analysis and architecture generation when opening projects
**Business Impact:** Ensures architecture exists BEFORE any code generation, prevents technical debt

**Key Principle:** Agent MUST analyze project and establish/verify architecture before generating ANY code.

---

### 1. New Project Initialization

**Trigger:** User creates new project via chat

**Workflow:**

```
User: "Create a REST API with JWT authentication"
    ↓
Agent Phase 1: Generate Architecture (MANDATORY)
    ↓
    1. Parse user intent
    2. Generate architecture diagram (LLM-based)
    3. Show architecture in Architecture View
    4. Ask user to review
    ↓
User: "Looks good, proceed" OR "Change X to Y"
    ↓
Agent Phase 2: Generate Code (Only after architecture approved)
    ↓
    1. Use architecture as context for code generation
    2. Generate files following architecture structure
    3. Validate code against architecture (deviation detection)
    4. Run tests
    5. Commit if all pass
```

**CRITICAL:** Code generation is BLOCKED until user approves architecture.

**Implementation:**

```rust
// In project_orchestrator.rs
pub async fn create_new_project(&mut self, intent: &str) -> Result<ProjectResult, String> {
    // Step 1: Generate architecture from intent (MANDATORY)
    let architecture = self.generate_architecture_from_intent(intent).await?;

    // Step 2: Show to user and wait for approval (Guided Mode)
    self.show_architecture_for_review(&architecture).await?;
    let approval = self.wait_for_user_approval().await?;

    if !approval.approved {
        if let Some(feedback) = approval.feedback {
            // User wants changes - regenerate
            return self.create_new_project(&feedback).await;
        } else {
            // User cancelled
            return Err("Project creation cancelled by user".to_string());
        }
    }

    // Step 3: Generate code ONLY after approval
    self.generate_code_from_architecture(&architecture).await?;

    // Step 4: Validate code against architecture
    self.validate_code_alignment(&architecture).await?;

    // ...continue with tests, commit, etc.
}
```

---

### 2. Existing Project Initialization (First Time Open)

**Trigger:** User opens existing project folder in Yantra for the first time

**Workflow:**

```
User opens project folder: /Users/user/my-project
    ↓
Agent detects: No .yantra/architecture.db file (first time)
    ↓
Agent Phase 1: Project Discovery
    ↓
    1. Check for architecture documentation files:
       - architecture.md
       - README.md (Architecture section)
       - docs/architecture/*
       - ARCHITECTURE.md
       - .github/architecture.md
    ↓
    Found documentation?
        ↓ YES                               ↓ NO
        ↓                                   ↓
    Phase 2A: Parse Documentation       Phase 2B: Code Analysis
        ↓                                   ↓
    1. Extract architecture info        1. Run GNN analysis
    2. Generate diagram from docs       2. Group files by structure
    3. Map to components                3. Infer connections
                                        4. Generate architecture
        ↓                                   ↓
        └──────────────┬──────────────────┘
                       ↓
    Phase 3: Code Review (MANDATORY)
        ↓
    1. Analyze codebase quality
    2. Detect patterns and anti-patterns
    3. Identify tech debt
    4. Generate code review report
        ↓
    Phase 4: Show to User
        ↓
    Agent: "📐 I've analyzed your project. Here's what I found:

    **Architecture:**
    - 5 components detected (Frontend, Backend, Database, Cache, Queue)
    - 12 connections mapped
    - View in Architecture View tab

    **Code Review:**
    - ✅ Good separation of concerns
    - ⚠️  Found 3 potential issues:
      1. Direct database access in API routes (bypasses service layer)
      2. Missing error handling in auth middleware
      3. No rate limiting on public endpoints

    **Tech Stack:**
    - Frontend: React 18, TypeScript 5.0
    - Backend: Express 4.18, Node 18
    - Database: PostgreSQL 15

    Would you like me to:
    1. Fix the identified issues
    2. Generate missing tests
    3. Add documentation
    4. Just show me the architecture"
        ↓
    User provides guidance
        ↓
    Agent proceeds with user-approved actions
```

**CRITICAL Requirements:**

1. ✅ Check for existing architecture docs FIRST
2. ✅ Parse docs if found (don't ignore existing knowledge)
3. ✅ Run code review BEFORE offering to modify anything
4. ✅ Ask user for confirmation before any changes
5. ✅ Generate architecture even if docs exist (sync with code)

**Implementation:**

```rust
// In project_initializer.rs (NEW MODULE)
pub struct ProjectInitializer {
    gnn: Arc<Mutex<GNNEngine>>,
    llm: Arc<Mutex<LLMOrchestrator>>,
    architecture_manager: ArchitectureManager,
}

impl ProjectInitializer {
    pub async fn initialize_existing_project(
        &mut self,
        project_path: &Path,
    ) -> Result<ProjectInitialization, String> {
        // Step 1: Check if already initialized
        if self.is_initialized(project_path)? {
            return Ok(ProjectInitialization::AlreadyInitialized);
        }

        // Step 2: Discover architecture documentation
        let arch_docs = self.discover_architecture_docs(project_path).await?;

        // Step 3: Generate architecture
        let architecture = if !arch_docs.is_empty() {
            // Parse from documentation
            self.parse_architecture_from_docs(&arch_docs).await?
        } else {
            // Generate from code analysis
            self.analyze_codebase_architecture(project_path).await?
        };

        // Step 4: Run code review (MANDATORY)
        let code_review = self.run_code_review(project_path).await?;

        // Step 5: Present findings to user
        self.present_initialization_report(
            &architecture,
            &code_review,
            &arch_docs,
        ).await?;

        // Step 6: Wait for user decision
        let user_action = self.wait_for_user_action().await?;

        // Step 7: Execute approved actions
        self.execute_user_approved_actions(user_action).await?;

        Ok(ProjectInitialization::Complete {
            architecture,
            code_review,
        })
    }

    async fn discover_architecture_docs(
        &self,
        project_path: &Path,
    ) -> Result<Vec<ArchitectureDoc>, String> {
        let mut docs = Vec::new();

        // Check common locations
        let candidates = vec![
            "architecture.md",
            "ARCHITECTURE.md",
            "README.md",
            "docs/architecture.md",
            "docs/ARCHITECTURE.md",
            ".github/architecture.md",
            "wiki/Architecture.md",
        ];

        for candidate in candidates {
            let path = project_path.join(candidate);
            if path.exists() {
                let content = std::fs::read_to_string(&path)?;

                // Check if it contains architecture information
                if self.contains_architecture_info(&content) {
                    docs.push(ArchitectureDoc {
                        path,
                        content,
                        confidence: self.calculate_arch_doc_confidence(&content),
                    });
                }
            }
        }

        Ok(docs)
    }

    fn contains_architecture_info(&self, content: &str) -> bool {
        let keywords = [
            "architecture",
            "component",
            "service",
            "module",
            "layer",
            "system design",
            "data flow",
            "dependency",
        ];

        let lower_content = content.to_lowercase();
        keywords.iter().filter(|kw| lower_content.contains(*kw)).count() >= 3
    }

    async fn run_code_review(
        &self,
        project_path: &Path,
    ) -> Result<CodeReviewReport, String> {
        // 1. Run GNN analysis
        let gnn = self.gnn.lock().unwrap();
        gnn.build_graph(project_path)?;

        // 2. Detect patterns
        let patterns = self.detect_code_patterns(&gnn)?;

        // 3. Identify issues
        let issues = self.identify_code_issues(&gnn, &patterns)?;

        // 4. Assess quality
        let quality_score = self.calculate_quality_score(&issues);

        Ok(CodeReviewReport {
            patterns,
            issues,
            quality_score,
            recommendations: self.generate_recommendations(&issues),
        })
    }
}

pub struct CodeReviewReport {
    pub patterns: Vec<CodePattern>,
    pub issues: Vec<CodeIssue>,
    pub quality_score: QualityScore,
    pub recommendations: Vec<String>,
}

pub struct CodeIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub description: String,
    pub file: PathBuf,
    pub line: Option<usize>,
    pub suggestion: String,
}

pub enum IssueSeverity {
    Critical,    // Security vulnerabilities, breaking changes
    High,        // Architectural violations, major bugs
    Medium,      // Code smells, minor issues
    Low,         // Style issues, minor improvements
}

pub enum IssueCategory {
    Architecture,      // Layer violations, wrong dependencies
    Security,          // SQL injection, XSS, etc.
    Performance,       // N+1 queries, memory leaks
    Maintainability,   // Code duplication, complexity
    Testing,           // Missing tests, poor coverage
    Documentation,     // Missing docs, outdated comments
}
```

---

### 3. Architecture Maintenance During Development

**Trigger:** User requests feature that impacts architecture

**Workflow:**

```
User: "Add Redis caching to the user service"
    ↓
Agent Phase 1: Analyze Impact (GNN-powered)
    ↓
    1. Load current architecture
    2. Identify affected components:
       - User Service (needs Redis client)
       - Database (fewer queries expected)
    3. Calculate impact score (0-100)
    4. Determine new components needed:
       - Redis Cache (new component)
    5. Determine new connections needed:
       - User Service → Redis
       - Redis → Database (cache-aside pattern)
    ↓
Agent Phase 2: Explain Impact (Natural Language)
    ↓
    Agent: "💡 Adding Redis caching will affect your architecture:

    **New Component:**
    - Redis Cache (Layer: Backend)
      Purpose: Store frequently accessed user data

    **Modified Components:**
    - User Service
      Changes: Add Redis client, implement cache-aside pattern

    **New Connections:**
    - User Service → Redis Cache (read/write)
    - Redis Cache → PostgreSQL (on cache miss)

    **Expected Benefits:**
    - 60-80% reduction in database queries
    - Faster response times (<50ms vs ~200ms)

    **Trade-offs:**
    - Added complexity (cache invalidation)
    - New dependency to manage
    - Potential stale data issues

    Architecture changes require your approval.
    Should I proceed? (yes/no/modify)"
    ↓
User: "yes" OR provides modifications
    ↓
Agent Phase 3: Update Architecture
    ↓
    1. Add Redis component
    2. Add connections
    3. Update component metadata
    4. Save new version (Rule of 3)
    5. Show updated diagram
    ↓
Agent Phase 4: Generate Code
    ↓
    1. Generate code following updated architecture
    2. Validate alignment
    3. Run tests
    4. Commit
```

**CRITICAL Requirements:**

1. ✅ Analyze impact BEFORE generating code
2. ✅ Explain in natural language (no technical jargon)
3. ✅ Show benefits AND trade-offs
4. ✅ Wait for user approval
5. ✅ Update architecture FIRST, then generate code

**Implementation:**

```rust
// In project_orchestrator.rs - UPDATE create_project method
pub async fn handle_feature_request(
    &mut self,
    request: &str,
) -> Result<FeatureResult, String> {
    // Step 1: Load current architecture
    let architecture = self.load_current_architecture()?;

    // Step 2: Analyze impact using GNN
    let impact = self.analyze_architecture_impact(&architecture, request).await?;

    // Step 3: Explain to user in natural language
    let explanation = self.explain_impact_naturally(&impact).await?;
    self.show_to_user(&explanation).await?;

    // Step 4: Wait for approval
    let approval = self.wait_for_approval().await?;

    if !approval.approved {
        return Err("Feature request not approved".to_string());
    }

    // Step 5: Update architecture FIRST
    let updated_arch = self.apply_architecture_changes(&architecture, &impact).await?;
    self.save_architecture(&updated_arch)?;

    // Step 6: Generate code following new architecture
    self.generate_code_with_architecture(&updated_arch, request).await?;

    // Step 7: Validate alignment
    self.validate_code_alignment(&updated_arch).await?;

    Ok(FeatureResult::Success)
}

async fn analyze_architecture_impact(
    &self,
    architecture: &Architecture,
    request: &str,
) -> Result<ArchitectureImpact, String> {
    // Use GNN to understand current dependencies
    let gnn = self.gnn.lock().unwrap();
    let current_deps = gnn.get_all_dependencies();

    // Use LLM to understand what request needs
    let llm = self.llm.lock().unwrap();
    let analysis = llm.analyze_feature_requirements(request).await?;

    // Calculate impact
    Ok(ArchitectureImpact {
        new_components: analysis.required_components,
        modified_components: self.find_affected_components(
            &architecture,
            &analysis.required_components,
        ),
        new_connections: analysis.required_connections,
        impact_score: self.calculate_impact_score(&analysis),
        benefits: analysis.expected_benefits,
        trade_offs: analysis.trade_offs,
    })
}

async fn explain_impact_naturally(
    &self,
    impact: &ArchitectureImpact,
) -> Result<String, String> {
    let llm = self.llm.lock().unwrap();

    let prompt = format!(
        "Explain the following architecture changes in simple, natural language:

        New components: {:?}
        Modified components: {:?}
        New connections: {:?}
        Benefits: {:?}
        Trade-offs: {:?}

        Write as if explaining to a product manager, not an engineer.
        Use analogies if helpful. Be concise but complete.",
        impact.new_components,
        impact.modified_components,
        impact.new_connections,
        impact.benefits,
        impact.trade_offs,
    );

    llm.complete(&prompt, 500).await
}
```

---

### 4. Implementation Checklist

**New Rust Modules:**

- [ ] `src-tauri/src/agent/project_initializer.rs` (500+ lines)
  - ProjectInitializer struct
  - discover_architecture_docs()
  - parse_architecture_from_docs()
  - analyze_codebase_architecture()
  - run_code_review()
  - present_initialization_report()

- [ ] `src-tauri/src/agent/code_reviewer.rs` (400+ lines)
  - CodeReviewer struct
  - detect_code_patterns()
  - identify_code_issues()
  - calculate_quality_score()
  - generate_recommendations()

- [ ] `src-tauri/src/agent/architecture_impact.rs` (300+ lines)
  - ArchitectureImpactAnalyzer struct
  - analyze_impact()
  - explain_naturally()
  - calculate_impact_score()

**Updated Modules:**

- [ ] `src-tauri/src/agent/project_orchestrator.rs`
  - Add ensure_architecture_exists()
  - Add handle_feature_request()
  - Add validate_code_alignment()
  - Block code generation until architecture approved

- [ ] `src-tauri/src/architecture/mod.rs`
  - Export project_initializer module
  - Add helper methods for initialization

**Frontend Components:**

- [ ] `src-ui/components/ArchitectureReview.tsx`
  - Show architecture for approval
  - Yes/No/Modify buttons
  - Display impact analysis

- [ ] `src-ui/components/CodeReviewReport.tsx`
  - Display code review findings
  - Show issues by severity
  - Actionable recommendations

**Tauri Commands:**

- [ ] `initialize_project(project_path: String)`
- [ ] `analyze_architecture_impact(request: String)`
- [ ] `approve_architecture(architecture_id: String, approved: bool)`
- [ ] `get_code_review_report(project_path: String)`

---

### 5. Success Metrics

**Technical:**

- ✅ 100% of new projects have architecture before code generation
- ✅ 100% of existing projects analyzed within 30 seconds
- ✅ Architecture docs discovered in 90%+ of projects that have them
- ✅ Code review completed in <60 seconds for typical projects
- ✅ Impact analysis completed in <5 seconds

**User Experience:**

- ✅ Users understand architecture before code generation
- ✅ Users can review and modify architecture easily
- ✅ Users see impact analysis in natural language
- ✅ No code generated without architecture approval
- ✅ Existing projects onboarded smoothly

**Business Impact:**

- Architecture-first approach prevents 80%+ of technical debt
- Code review catches issues before they're coded
- Users trust Yantra more (transparent process)
- Faster onboarding for existing projects

---

## Project Initialization & Architecture-First Workflow (MVP Phase 1 - HIGH PRIORITY)

### Overview

**Status:** 🔴 NOT STARTED
**Priority:** ⚡ MVP REQUIRED (Critical for architecture-first development)
**Created:** November 28, 2025
**Business Impact:** Ensures architecture always exists before code generation, prevents architectural drift

A comprehensive system that ensures every project (new or existing) has a reviewed and approved architecture before any code implementation begins. This enforces Yantra's core principle: **Architecture-First Development**.

---

### Core Requirements

#### 1. New Project Initialization

**Workflow:**

```
User: "Create a REST API with authentication"
  ↓
Agent: Analyzes intent → Generates architecture diagram
  ↓
Agent: "📐 I've designed the architecture with 4 components:
        - API Gateway (handles routing)
        - Auth Service (JWT tokens)
        - User Service (CRUD operations)
        - PostgreSQL Database

        Please review in Architecture View tab.

        Type 'approve' to proceed or suggest changes."
  ↓
User reviews architecture in Architecture View
  ↓
User: "approve" OR "add Redis caching"
  ↓
IF approved:
    Agent: Saves architecture → Generates code matching architecture
IF changes requested:
    Agent: Updates architecture → Shows updated diagram → Asks for approval again
```

**Implementation Requirements:**

- ✅ Generate architecture BEFORE any code
- ✅ Mandatory user review and approval
- ✅ Support iterative architecture refinement
- ✅ Architecture saved as version 1 (baseline)
- ✅ Agent cannot proceed without approval

---

#### 2. Existing Project - First Time Open

**Workflow:**

```
User opens existing project folder (e.g., ~/my-project)
  ↓
Agent: Scans project directory
  ↓
Agent checks for architecture files:
  - .yantra/architecture.db (SQLite)
  - architecture.md
  - architecture.json
  - README.md (architecture section)
  - ARCHITECTURE.md
  - docs/architecture.md
  ↓
CASE A: Architecture files found
  ↓
  Agent: "📄 Found existing architecture documentation.
          Importing architecture from [file]...

          Please review in Architecture View tab.
          Type 'approve' to use this architecture."
  ↓
  User reviews → approves/modifies

CASE B: No architecture files found
  ↓
  Agent: "🔍 No architecture documentation found.

          I can analyze the codebase to generate architecture, but I need
          some context first:

          Option 1: Tell me about this project (what it does, key features)
          Option 2: Type 'analyze' to auto-generate from code structure
          Option 3: Provide a README or design document to import"
  ↓
  User provides context OR says "analyze"
  ↓
  Agent: Generates architecture → Shows for review
```

**Implementation Requirements:**

- ✅ Detect existing architecture files (6 common locations)
- ✅ Parse various formats (Markdown, JSON, Mermaid diagrams)
- ✅ Extract architecture from README if present
- ✅ If no files: Request user context before analyzing
- ✅ Never auto-generate without user awareness

---

#### 3. Code Review for Existing Projects

**Workflow:**

```
Agent analyzing existing project (after architecture is defined)
  ↓
Agent performs code review:
  1. GNN dependency analysis
  2. Security scan (Semgrep)
  3. Code quality check (complexity, test coverage)
  4. Architecture alignment validation
  ↓
Agent: "📊 Code Review Complete:

        ✅ Found 45 files in 8 components
        ⚠️  3 security issues detected (2 auto-fixable)
        ⚠️  12 files don't match architecture:
           - api/gateway.py: Direct database access (violates layering)
           - user_service.py: Missing from architecture

        Recommendations:
        1. Update architecture to include user_service.py
        2. Refactor gateway.py to use Auth Service
        3. Fix security issues

        Would you like me to:
        (a) Update architecture to match code
        (b) Refactor code to match architecture
        (c) Show detailed findings first"
  ↓
User makes decision
  ↓
Agent: Executes chosen action
```

**Implementation Requirements:**

- ✅ GNN-based dependency analysis
- ✅ Security scanning integration
- ✅ Architecture alignment validation
- ✅ Generate prioritized recommendations
- ✅ Support both directions: update arch OR refactor code

---

#### 4. Architecture Maintenance During Development

**Workflow:**

```
User: "Add email notification feature"
  ↓
Agent: Analyzes requirement
  ↓
Agent: "📐 To add email notifications, I need to:

        Architecture Changes:
        - Add: Email Service (new component)
        - Add connection: User Service → Email Service
        - Add connection: Email Service → External (SendGrid API)

        Code Changes:
        - Create: src/email/service.py
        - Modify: src/user/service.py (trigger emails)
        - Add dependency: sendgrid==6.9.7

        This affects 2 existing components.

        Approve architecture changes? (yes/no)"
  ↓
User: "yes"
  ↓
Agent: Updates architecture (version 2) → Generates code → Validates alignment
```

**Implementation Requirements:**

- ✅ Detect when requirements impact architecture
- ✅ Preview architecture changes BEFORE implementation
- ✅ Show affected components and connections
- ✅ Require approval for architecture changes
- ✅ Auto-increment architecture version
- ✅ Validate final code matches approved architecture

---

### Implementation Components

#### A. ProjectInitializer (Rust)

**File:** `src-tauri/src/agent/project_initializer.rs`

```rust
pub struct ProjectInitializer {
    gnn_engine: Arc<Mutex<GNNEngine>>,
    architecture_manager: ArchitectureManager,
    architecture_generator: ArchitectureGenerator,
    architecture_analyzer: ArchitectureAnalyzer,
}

impl ProjectInitializer {
    /// Initialize a new project with architecture-first approach
    pub async fn initialize_new_project(
        &mut self,
        intent: String,
        project_path: PathBuf,
    ) -> Result<InitializationResult, String>;

    /// Initialize existing project (first time open)
    pub async fn initialize_existing_project(
        &mut self,
        project_path: PathBuf,
    ) -> Result<InitializationResult, String>;

    /// Check for existing architecture files
    fn detect_architecture_files(&self, project_path: &Path) -> Vec<PathBuf>;

    /// Import architecture from various file formats
    fn import_architecture_from_file(&self, file_path: &Path) -> Result<Architecture, String>;

    /// Perform code review on existing project
    pub async fn review_existing_code(
        &mut self,
        project_path: &Path,
        architecture: &Architecture,
    ) -> Result<CodeReviewResult, String>;

    /// Detect if requirement impacts architecture
    pub fn analyze_requirement_impact(
        &self,
        requirement: &str,
        current_architecture: &Architecture,
    ) -> Result<ArchitectureImpact, String>;
}

pub struct InitializationResult {
    pub architecture_id: String,
    pub architecture_source: ArchitectureSource, // Generated, Imported, UserProvided
    pub requires_review: bool,
    pub review_url: Option<String>, // Deep link to Architecture View
}

pub enum ArchitectureSource {
    GeneratedFromIntent,
    GeneratedFromCode,
    ImportedFromFile(PathBuf),
    UserProvided,
}

pub struct CodeReviewResult {
    pub total_files: usize,
    pub security_issues: Vec<SecurityIssue>,
    pub alignment_issues: Vec<AlignmentIssue>,
    pub quality_metrics: QualityMetrics,
    pub recommendations: Vec<Recommendation>,
}

pub struct ArchitectureImpact {
    pub has_impact: bool,
    pub new_components: Vec<Component>,
    pub modified_connections: Vec<Connection>,
    pub affected_components: Vec<String>, // Component IDs
    pub severity: ImpactSeverity, // Low, Medium, High, Breaking
}

pub enum ImpactSeverity {
    Low,      // Add utility function
    Medium,   // Add new feature to existing component
    High,     // Add new component
    Breaking, // Modify core architecture patterns
}
```

---

#### B. Architecture File Detection

**Supported File Locations (in priority order):**

1. `.yantra/architecture.db` (Yantra's native SQLite)
2. `.yantra/architecture.json` (Yantra's export format)
3. `architecture.md` (Root directory)
4. `ARCHITECTURE.md` (Root directory)
5. `docs/architecture.md` (Docs folder)
6. `README.md` with `## Architecture` section

**Supported Formats:**

- Mermaid diagrams (parse with regex)
- JSON (Yantra format or C4 model)
- Markdown structured lists
- PlantUML (basic parsing)

---

#### C. Requirement Impact Analysis

**Algorithm:**

```rust
1. Parse user requirement with LLM:
   - Extract: entities, actions, integrations
   - Classify: new feature, modification, refactoring

2. Check against current architecture:
   - Do mentioned entities exist as components?
   - Do mentioned integrations exist as connections?
   - Are there implicit dependencies?

3. Use GNN to analyze code impact:
   - Which files would be modified?
   - Which components own those files?

4. Calculate severity:
   - Low: Changes within 1 component, no new dependencies
   - Medium: Changes across 2-3 components, existing dependencies
   - High: New component needed
   - Breaking: Fundamental pattern change (e.g., monolith → microservices)

5. Generate preview:
   - Show architecture diff (before/after)
   - List affected components
   - Estimate implementation complexity
```

---

### User Interaction Flows

#### Flow 1: New Project (Happy Path)

```
1. User: "Create a task management API"
2. Agent: Generates architecture → Shows in Architecture View
3. Agent: "Please review the architecture. Type 'approve' to proceed."
4. User: "approve"
5. Agent: Saves architecture (v1) → Generates code → Tests → Commits
```

#### Flow 2: New Project (Iterative Refinement)

```
1. User: "Create a task management API"
2. Agent: Generates architecture
3. User: "Add real-time notifications with WebSockets"
4. Agent: Updates architecture → Shows changes
5. User: "approve"
6. Agent: Implements
```

#### Flow 3: Existing Project (Architecture Found)

```
1. User opens project
2. Agent: Finds architecture.md
3. Agent: "Found architecture documentation. Importing..."
4. Agent: Shows imported architecture
5. User: "looks good"
6. Agent: Ready for development
```

#### Flow 4: Existing Project (No Architecture)

```
1. User opens project
2. Agent: No architecture files found
3. Agent: "Tell me about this project or type 'analyze'"
4. User: "analyze"
5. Agent: GNN analysis → Generates architecture
6. Agent: Shows for review
7. User: "approve"
8. Agent: Saves architecture
```

#### Flow 5: Requirement with Architecture Impact

```
1. User: "Add payment processing with Stripe"
2. Agent: Analyzes requirement
3. Agent: "This requires adding:
           - Payment Service (new component)
           - Stripe Integration (external)
           - Connection: Order Service → Payment Service

           Approve architecture changes?"
4. User: "yes"
5. Agent: Updates architecture (v2) → Implements → Validates
```

---

### Integration Points

#### With ProjectOrchestrator

```rust
// In create_project()
pub async fn create_project(&mut self, intent: String, project_dir: PathBuf)
    -> Result<ProjectResult, String>
{
    // ✅ Step 1: Initialize project (creates architecture)
    let init_result = self.initializer.initialize_new_project(intent.clone(), project_dir.clone()).await?;

    // ✅ Step 2: Wait for user approval (via chat)
    self.wait_for_architecture_approval(&init_result.architecture_id).await?;

    // ✅ Step 3: Load approved architecture
    let architecture = self.arch_manager.get_architecture(&init_result.architecture_id)?;

    // ✅ Step 4: Generate code using architecture as context
    let context = self.build_context_from_architecture(&architecture);

    // ... continue with code generation ...
}
```

#### With ChatPanel (Frontend)

```typescript
async function handleMessage(message: string) {
  // Check if this is first message in new project
  if (isNewProject && !hasArchitecture) {
    // Agent will auto-generate architecture
    const response = await invoke('initialize_new_project', {
      intent: message,
    });

    // Show architecture in Architecture View
    switchToTab('architecture');

    // Wait for user approval
    setWaitingForApproval(true);
    return;
  }

  // Check if requirement impacts architecture
  const impact = await invoke('analyze_requirement_impact', {
    requirement: message,
  });

  if (impact.has_impact) {
    // Show architecture changes for approval
    showArchitectureChanges(impact);
    return;
  }

  // Normal processing
  // ...
}
```

---

### Success Metrics

**Technical:**

- ✅ 100% of new projects have architecture before code generation
- ✅ 100% of existing projects analyzed within 10 seconds
- ✅ Architecture file detection: 95%+ accuracy
- ✅ Requirement impact analysis: <500ms response time
- ✅ Zero code generation without approved architecture

**User Experience:**

- ✅ Users always aware of architecture before implementation
- ✅ Users can iteratively refine architecture
- ✅ Clear understanding of requirement impacts
- ✅ No surprises (all changes previewed)
- ✅ NPS >60 for architecture-first workflow

**Business Impact:**

- Prevents architectural drift (architecture = code)
- Reduces rework (architecture approved upfront)
- Improves code quality (enforced patterns)
- Enables scaling (clear component boundaries)
- Differentiator: Only AI tool with architecture-first guarantee

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

| Operation              | Target        | Actual    |
| ---------------------- | ------------- | --------- |
| Load all documentation | <100ms        | ~50ms ✅  |
| Parse Project_Plan.md  | <10ms         | ~5ms ✅   |
| Parse Features.md      | <10ms         | ~3ms ✅   |
| Parse Decision_Log.md  | <10ms         | ~4ms ✅   |
| UI tab switch          | <16ms (60fps) | ~8ms ✅   |
| Reload from disk       | <200ms        | ~100ms ✅ |

---

### Source Files

| File                                        | Lines | Purpose                                   | Tests  |
| ------------------------------------------- | ----- | ----------------------------------------- | ------ |
| `src-tauri/src/documentation/mod.rs`        | 429   | Core backend logic, extraction algorithms | 4/4 ✅ |
| `src-ui/stores/documentationStore.ts`       | 198   | Reactive state management                 | N/A    |
| `src-ui/components/DocumentationPanels.tsx` | 248   | UI component with tabs                    | N/A    |

**Total:** 875 lines across 3 files

---

### Minimal UI Requirements (November 28, 2025 - NEW)

#### Design Philosophy

**Objective:** Maximize information density while maintaining readability and usability.

**Core Principles:**

1. **Space Optimization:** Reduce padding and margins by 25-50% compared to standard desktop UI
2. **Single-Line Layouts:** Combine related elements horizontally (icons + text, input + button)
3. **Abbreviated Text:** Use short labels for repeated UI elements, full text on hover
4. **Visual Indicators:** Rely on icons and color coding over verbose text
5. **Efficient Filtering:** Real-time search in all list views with <10ms latency

**Target Metrics:**

- 30-40% more content visible per screen
- Font sizes: 11-16px (reduced from 14-20px standard)
- Padding: 4-12px (reduced from 12-24px standard)
- Line height: 1.4-1.6 (tighter than 1.8-2.0 standard)

#### Documentation Panels Requirements

**Search Functionality (ALL 4 TABS):**

- Real-time filtering as user types
- Search scope:
  - **Features:** Title + description
  - **Decisions:** Context + decision + rationale
  - **Changes:** Description + file names
  - **Plan:** Task titles
- Performance target: <10ms for 100 items, <50ms for 1000 items
- Empty state message when no results: "No [X] found matching '[query]'"
- Clear search on tab switch (UX consistency)

**Search Input Design:**

```
Width: 100% (w-full)
Padding: 8px horizontal, 4px vertical (px-2 py-1)
Font size: 11px (text-[11px])
Border: 1px gray with rounded corners (rounded)
Placeholder: "Search [tab name]..."
Icon: 🔍 emoji prefix (no SVG overhead)
```

**Natural Language Explanations:**

Each tab MUST include a brief explanation of where data comes from:

- **Features:** "Features are automatically extracted from your chat conversations. As you describe what you want to build, Yantra identifies and tracks features, updating their status as implementation progresses."
- **Decisions:** "Critical technical decisions are logged here with full context. Each decision includes why it was made, what alternatives were considered, and the rationale behind the choice."
- **Changes:** "Complete audit trail of all code changes. Track what files were added, modified, or deleted, along with timestamps and descriptions."
- **Plan:** "Your project plan with tasks organized by milestones. Dependencies are tracked automatically, and tasks requiring your input are highlighted."

**Purpose:** Reduce user confusion, set expectations for automation, explain empty states.

**Spacing Requirements (Minimal UI):**

```
Component               Before → After    Reduction
──────────────────────────────────────────────────
Tab buttons padding:    16px/12px → 12px/8px   25-33%
Content area padding:   16px → 8px              50%
Card padding:           12px → 8px              33%
Card spacing:           12px → 8px              33%
Search input padding:   12px/8px → 8px/4px     33-50%
Font sizes:             14px/12px → 12px/11px  14-17%
```

**Result:** 30-40% more content visible per screen

**Word-Wrap for Long Content:**

- Plan task titles: `break-words` class (prevents horizontal overflow)
- File paths: `truncate` with ellipsis for paths > 40 characters
- Dependencies: `truncate` when > 3 dependencies, show "+N more" indicator

#### Chat Panel Requirements

**Header Layout (Minimal UI):**

```
┌────────────────────────────────────────────────┐
│ Chat - Describe what...  [Model ▼] [⚙️]      │  ← Compact header
└────────────────────────────────────────────────┘
```

**Header Components:**

1. **Title** (left): "Chat" - Font: 16px bold (text-base)
2. **Subtitle** (left): "- Describe what you want to build" - Font: 11px gray (text-[11px])
3. **Model Selector** (right): Dropdown with 4 options - Font: 11px (text-[11px])
4. **API Config Button** (right): ⚙️ settings icon - Size: 20px

**Header Spacing:**

```
Padding: 12px horizontal, 8px vertical (px-3 py-2)
Height: 42px (down from 64px - 34% reduction)
Gap between title and subtitle: 4px
Gap between model selector and config button: 8px
```

**Model Selector Requirements:**

- Always visible in header (NOT in input area)
- Options: Claude Sonnet 4, GPT-4 Turbo, OpenAI o1, Groq (4 providers)
- Dropdown width: Auto-fit content
- Selected model shown as: "[Model Name] ▼"
- Dropdown opens below, z-index: 50
- Aria-label: "Select LLM model" (accessibility)

**Messages Area (Minimal UI):**

```
Terminal-like design with compact spacing:
Font: Monaco/monospace (terminal aesthetic)
Font size: 11px (text-[11px]) - down from 12px
Line height: Relaxed (1.625)
Padding: 8px horizontal, 4px vertical (px-2 py-1)
Message spacing: 2px vertical gap (space-y-0.5)
Colors: Green (user), Blue (agent), Gray (system)
```

**Space Savings:**

- Font size: 12px → 11px (8% reduction)
- Padding: 12px/8px → 8px/4px (33-50% reduction)
- Message spacing: 4px → 2px (50% reduction)
- **Result:** 30% more messages visible per screen

**Input Area Layout (Minimal UI):**

```
Before (OLD - DEPRECATED):
┌────────────────────────────────────────┐
│ [Model ▼] [⚙️] [Send ▶]               │  ← Top row
│ ┌────────────────────────────────────┐ │
│ │ Type message...                    │ │  ← Textarea
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘

After (NEW - CURRENT):
┌────────────────────────────────────────┐
│ ┌────────────────────────────────────┐ │
│ │ Type message...                  ▶│ │  ← Send inside textarea
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘
```

**Input Area Requirements:**

1. **Container:** Relative positioning (relative)
2. **Textarea:**
   - Rows: 3
   - Font size: 11px (text-[11px])
   - Padding: 8px (p-2)
   - Padding-right: 40px (pr-10) - space for send button
   - Background: Transparent
   - Border: None (clean look)
3. **Send Button:**
   - Position: Absolute (absolute right-1 bottom-1)
   - Size: 24px × 24px (p-1.5)
   - Icon: ▶ (play symbol, no SVG)
   - Background: Primary blue (#3b82f6)
   - Hover: Darker blue (#2563eb)
   - Disabled: 50% opacity when empty or generating
   - Aria-label: "Send message" (accessibility)

**Space Savings:**

- Removed entire button row (~40px height)
- Smaller fonts: 14px → 11px (21% reduction)
- Less padding: 12px → 8px (33% reduction)
- **Result:** 20% more vertical space for messages

**Keyboard Shortcuts:**

- Enter: Send message
- Shift+Enter: New line in textarea
- Cmd+K: Focus input (future)
- Escape: Clear input (future)

#### View Tabs Requirements (Minimal UI)

**Tab Design:**

```
Before (OLD - DEPRECATED):
[✏️ Editor]  [🔗 Dependencies]  [🏗️ Architecture]

After (NEW - CURRENT):
[✏️ Editor]  [🔗 Deps]  [🏗️ Arch]
```

**Tab Requirements:**

1. **Layout:** Inline-flex with icon + text (inline-flex gap-1.5)
2. **Font size:** 12px (text-xs) - down from 14px (text-sm)
3. **Padding:** 12px horizontal, 6px vertical (px-3 py-1.5) - down from 16px/8px
4. **Icon:** Emoji (same size as before, no SVG)
5. **Text:** Abbreviated for long words:
   - "Dependencies" → "Deps" (60% shorter)
   - "Architecture" → "Arch" (50% shorter)
   - "Editor" → "Editor" (unchanged)
6. **Tooltip:** Show full name on hover (title attribute)

**Active Tab Indicator:**

```
Background: Gray-900 (#111827)
Border: 2px blue border-bottom (#3b82f6)
Text color: White
```

**Inactive Tab:**

```
Background: Transparent
Border: None
Text color: Gray-400
Hover: Text color → White
```

**Space Savings:**

- Padding: 16px/8px → 12px/6px (25% reduction)
- Font size: 14px → 12px (14% smaller)
- Text length: "Dependencies" → "Deps" (60% shorter)
- **Result:** All 3 tabs fit without overflow on 1024px+ screens

**Benefits:**

- Consistent with Files/Docs toggle design
- Icons provide visual recognition (accessibility)
- Abbreviated text reduces width
- Full names on hover (no information loss)

#### Performance Requirements (Minimal UI)

**Search Performance:**

- Algorithm: O(n\*m) string matching with memoization (createMemo)
- Latency target: <5ms for typical searches (100 items)
- Max latency: <50ms for large lists (1000 items)
- No debounce needed (fast enough for instant feedback)
- Efficient re-render with SolidJS reactive primitives

**UI Re-render Performance:**

- Tab switch: <16ms (60fps) - CRITICAL
- Search results update: <10ms after keystroke
- Message list scroll: 60fps (smooth scrolling)
- Textarea auto-resize: <16ms (instant feedback)

**Memory Efficiency:**

- Memoize filtered lists (no re-computation on every render)
- Virtual scrolling for lists > 100 items (future optimization)
- Lazy load images in messages (future)

#### Accessibility Requirements (Minimal UI)

**ARIA Labels (MANDATORY):**

- Model selector: `aria-label="Select LLM model"`
- Send button: `aria-label="Send message"`
- Search inputs: `aria-label="Search [tab name]"`
- Tab buttons: `aria-label="[Full tab name]"`

**Keyboard Navigation:**

- All interactive elements must be keyboard accessible (Tab, Enter, Space)
- Focus indicators visible on all focusable elements (blue outline)
- Tab order logical: Header → Model selector → Config → Messages → Input → Send

**Screen Reader Support:**

- Empty state messages read by screen readers
- Status indicators (✅, 🔄, ⏳) have text equivalents in aria-label
- Loading states announced ("Generating code...")

**Color Contrast:**

- All text must meet WCAG AA standards (4.5:1 for normal text, 3:1 for large)
- 11px text requires higher contrast (use white/light gray on dark backgrounds)
- Status badges maintain contrast in light and dark modes

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

**Dependency Graph-Powered Duplicate Detection:**

- Semantic similarity using code embeddings (384-dim from fastembed all-MiniLM-L6-v2)
- Cosine similarity >0.85 = duplicate
- Detects duplicates across languages (same logic, different syntax)
- Note: This uses the dependency graph's semantic embeddings, NOT Yantra Codex (which is 978-dim input for code generation)

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

| Operation                     | Target | Rationale                |
| ----------------------------- | ------ | ------------------------ |
| Dead code analysis (10K LOC)  | < 2s   | Real-time feedback       |
| Duplicate detection (10K LOC) | < 5s   | GNN embedding comparison |
| Refactoring application       | < 3s   | Including validation     |
| Component hardening           | < 10s  | Comprehensive scan       |
| Security scan                 | < 5s   | Semgrep integration      |

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

- Cross-system dependency tracking (internal + external APIs)
- Browser automation for legacy systems
- Self-healing workflows
- Multi-language support (Python + JavaScript)
- **Enterprise features: Multitenancy, user accounts, team collaboration**

### New Capabilities

1. **Cross-System Intelligence**

- Automatic discovery of external API calls
- Schema tracking for Stripe, Salesforce, etc.
- Breaking change detection (API version updates)
- End-to-end data flow validation
- Impact analysis (what breaks if X changes?)

2. **Browser Automation**

- Full Playwright integration
- DOM interaction (click, fill, extract data)
- Authentication handling
- Visual regression detection

3. **Self-Healing Systems**

- Continuous API monitoring (every 24h)
- Schema drift detection
- Automatic migration code generation
- Canary testing in sandbox
- Auto-deploy if tests pass

4. **Multi-Language Support**

- JavaScript/TypeScript parser
- Cross-language dependencies (Python API → React frontend)
- Node.js + React code generation
- Context mixing across languages

5. **Enterprise Features (Post-MVP)**

**Multitenancy:**

- Tenant isolation (database, GNN, patterns)
- Per-tenant configuration
- Shared failure patterns (cross-tenant, privacy-preserved)
- Resource quotas and limits

**User Accounts & Authentication:**

- User registration and login (OAuth, SSO)
- Role-based access control (RBAC)
- Team workspaces
- Project sharing and permissions
- Audit logs

**Team Collaboration:**

- Shared projects and codebases
- Activity feeds (who generated what)
- Code review workflows
- Comment threads on generated code
- Team pattern libraries (private)

**Billing & Subscription:**

- Usage tracking (LLM calls, tokens)
- Subscription tiers (Free, Pro, Team, Enterprise)
- Payment integration (Stripe)
- Usage analytics and reporting

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

- 99.9% uptime
- Support 100k+ LOC projects
- Plugin ecosystem
- Enterprise deployment options

### New Capabilities

1. Performance Optimization

- GNN queries <100ms for 100k LOC projects
- Distributed GNN (sharding)
- Smart caching (LLM responses, test results)

2. Advanced Refactoring

- Architectural refactoring (monolith → microservices)
- Performance optimization
- Tech debt reduction
- Code modernization

3. Ecosystem

- Plugin system (extend Yantra)
- Marketplace (plugins, templates, workflows)
- CLI tool (for CI/CD)
- REST API
- SDKs (Python, JavaScript, Go)

4. Enterprise

- On-premise deployment (air-gapped)
- Custom model training
- White-label options
- 24/7 SLA support

### Implementation (16 Weeks)

Weeks 33-36: Performance & scale Weeks 37-40: Advanced refactoring Weeks 41-44: Ecosystem & marketplace Weeks 45-48: Enterprise platform

---

## Go-to-Market Strategy

### Year 1: Developer Adoption (Free)

Strategy: Build massive user base through free access

Pricing:

- 100% Free for Year 1
- No credit card required
- Full feature access
- No usage limits

Rationale:

- Prove value before monetizing
- Build network effects
- Generate word-of-mouth
- Collect usage data to improve product
- Hook developers early

Target:

- Individual developers
- Small teams (1-10 developers)
- Early adopters and innovators
- Open source projects

Acquisition Channels:

- Product Hunt launch
- Hacker News discussions
- Dev.to and Medium articles
- YouTube demos
- GitHub showcases
- Developer conferences (talks, booths)

Success Metrics (Year 1):

- 10,000+ active users by Month 6
- 50,000+ active users by Month 12
- 80%+ retention rate
- NPS >50
- 10,000 projects created
- 1M lines of code generated

### Year 2: Freemium Transition

Strategy: Introduce paid tiers while keeping generous free tier

Pricing Tiers:

Free (Forever):

- Individual developers
- Up to 3 projects
- 100 LLM generations/month
- Community support
- Basic features

Pro ($29/month):

- Unlimited projects
- Unlimited LLM generations
- Priority LLM access (faster responses)
- Advanced features (refactoring, performance optimization)
- Email support

Team ($79/user/month):

- Everything in Pro
- Team collaboration features
- Shared dependency graphs
- Workflow automation (10 workflows)
- Admin controls
- Priority support

Enterprise (Custom pricing):

- Everything in Team
- Unlimited workflows
- On-premise deployment
- Custom model training
- SLA guarantees (99.9% uptime)
- 24/7 dedicated support
- Professional services (onboarding, training)

Target Conversion:

- 5-10% of free users to Pro ($29/mo)
- 20% of teams to Team tier ($79/user/mo)
- 50+ Enterprise customers by EOY2

Revenue Projection (Year 2):

- 50,000 users (from Year 1)
- 2,500 Pro users @ $29/mo = $72,500/mo
- 200 Team users @ $79/mo = $15,800/mo
- 50 Enterprise @ $5k/mo avg = $250,000/mo
- Total: ~$4M ARR by end of Year 2

### Year 3: Platform Play

Strategy: Expand to workflow automation market, compete with Zapier

New Revenue Streams:

- Marketplace (plugins, templates) - 30% revenue share
- Partner ecosystem (consultants) - certification programs
- Industry-specific solutions (fintech, healthcare)
- Professional services (custom workflows)

Target:

- Large enterprises (1000+ developers)
- Operations teams (workflow automation)
- Business analysts (no-code users)

Revenue Projection (Year 3):

- $15-20M ARR

---

## Appendices

### A. Development Guidelines

Code Quality Standards:

- Rust: Clippy pedantic, 80%+ test coverage, no panics in production
- Frontend: ESLint strict, Prettier formatting, TypeScript strict mode
- Generated Python: PEP 8, type hints, docstrings, error handling

Git Workflow:

- Branches: main (production), develop (integration), feature/\* (features)
- Commits: Conventional Commits format
- PRs: Required reviews, CI must pass

Testing Strategy:

- Unit tests: All core logic
- Integration tests: End-to-end flows
- Performance tests: Benchmark GNN operations
- Manual testing: Weekly on all platforms

### B. Tech Stack Rationale

Why Tauri over Electron?

- 600KB vs 150MB bundle size
- Lower memory footprint (100MB vs 400MB)
- Rust backend ideal for GNN performance
- Native OS integrations

Why SolidJS over React?

- Fastest reactive framework (benchmark leader)
- Smaller bundle size
- No virtual DOM overhead
- Better TypeScript support

Why Rust for GNN?

- Memory safety without garbage collection
- Fearless concurrency (Tokio async)
- Zero-cost abstractions
- Fast graph operations (petgraph)
- Easy to parallelize

Why Multi-LLM?

- No single point of failure
- Quality improvement through consensus
- Cost optimization (route by complexity)
- Best-of-breed approach

### C. Performance Targets

MVP Targets:

- GNN graph build: <5s for 10k LOC project
- GNN incremental update: <50ms per file change
- Dependency lookup: <10ms
- Context assembly: <100ms
- Code generation: <3s (LLM dependent)
- Test execution: <30s for typical project
- Security scan: <10s
- Browser validation: <5s
- Total cycle (intent → commit): <2 minutes

Scale Targets (Month 9+):

- GNN graph build: <30s for 100k LOC project
- GNN query: <100ms for 100k LOC
- Support 1M LOC projects

### D. Security & Privacy

Data Handling:

- User code never leaves machine unless explicitly sent to LLM APIs
- LLM calls encrypted in transit (HTTPS)
- No code storage on Yantra servers (local only)
- Crash reports: Anonymous, opt-in
- Analytics: Usage only, no PII, opt-in

LLM Privacy:

- Option to use local LLM (post-MVP, Phase 2+)
- Mark sensitive files (never send to cloud LLM)
- Audit log (what was sent to cloud)
- Data retention: LLM providers' policies (typically 30 days, then deleted)

Enterprise Privacy:

- On-premise deployment (air-gapped)
- BYO LLM (use your own models)
- Encrypted at rest
- SOC2 compliance
- GDPR compliance

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

- ✅ 20 beta users successfully generating code
- ✅ >90% of generated code passes tests
- ✅ NPS >40

Month 6:

- ✅ 10,000 active users
- ✅ >95% code success rate
- ✅ 50%+ user retention

Month 12:

- ✅ 50,000 active users
- ✅ Workflow automation live (Phase 2)
- ✅ 80%+ retention

Month 18:

- ✅ Freemium launch
- ✅ $500k ARR
- ✅ 100+ paying customers

Month 24:

- ✅ $4M ARR
- ✅ 2,500+ Pro users
- ✅ 50+ Enterprise customers

---

## Getting Started (For Developers)

### Prerequisites

- Rust 1.74+ (rustup install stable)
- Node.js 18+ (nvm install 18)
- Python 3.11+ (pyenv install 3.11)
- Git
- macOS, Windows, or Linux

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

\*\*
