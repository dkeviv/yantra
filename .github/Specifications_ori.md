# Yantra: Complete Technical Specification

Version: 1.0
Date: November 2024
Document Purpose: Complete technical blueprint for building Yantra from ground zero to enterprise platform

---

## ⚠️ CRITICAL TERMINOLOGY CLARIFICATION

**Two Separate Systems - Do Not Confuse:**

1. **Dependency Graph** (Code Intelligence)

   - **Code Name:** Often called "GNN" in codebase (MISLEADING - historical naming)
   - **Reality:** This is NOT a neural network - it's a petgraph data structure for tracking code relationships
   - **Purpose:** Track code dependencies (imports, calls, relationships) for impact analysis
   - **Local:** Tier 1 petgraph in-memory + SQLite persistence
   - **Cloud:** Tier 0 PostgreSQL + Redis (metadata only, no actual code)
   - **Note:** In code, you may see "GNN" referring to Dependency Graph, not Yantra Codex
2. **Yantra Codex** (AI Code Generation) - **POST-MVP / OPTIONAL**

   - **Code Name:** "Yantra Codex" or "Codex"
   - **Actual Technology:** GraphSAGE neural network (IS a real neural network)
   - **Purpose:** Generate code from natural language using ML (speed optimization)
   - **Local:** 1024-dim GraphSAGE model, 150M parameters
   - **Cloud:** Aggregated embeddings (opt-in, privacy-preserving)
   - **MVP Status:** NOT REQUIRED - System works with LLM-only generation

**Why This Matters:** Throughout this document, we use "Dependency Graph" for the code intelligence system. The term "GNN" appears in code for historical reasons but refers to the Dependency Graph (petgraph), NOT the neural network (Yantra Codex).

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

Phase 2 (Months 3-4): Team of Agents & Cloud Graph Database + Package/Deploy

- **Team of Agents Architecture** (🆕 Month 3-4) - Master-Servant pattern with Git coordination branch for multi-agent parallelism
- **Cloud Graph Database (Tier 0)** (🆕 Month 4-5) - Shared dependency graph for proactive conflict prevention across agents and team members
  - **Note:** This is NOT a "Cloud-based Dependency Analysis System" - the dependency analysis runs locally. This is cloud-hosted graph database storage for coordination and real-time collaboration visibility.
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

| Capability                          | Yantra | Copilot | Cursor | Zapier | Replit Agent |
| ----------------------------------- | ------ | ------- | ------ | ------ | ------------ |
| Dependency-aware generation         | ✅     | ❌      | ❌     | N/A    | ❌           |
| Guaranteed no breaks                | ✅     | ❌      | ❌     | ❌     | ❌           |
| Truly unlimited context             | ✅     | ❌      | ❌     | N/A    | ❌           |
| Token-aware context                 | ✅     | ⚠️    | ⚠️   | N/A    | ❌           |
| Automated testing                   | ✅     | ❌      | ❌     | ❌     | ⚠️         |
| Agentic validation pipeline         | ✅     | ❌      | ❌     | ❌     | ❌           |
| **Autonomous code execution** | ✅     | ❌      | ❌     | ⚪     | ✅           |
| **Package building**          | ✅     | ❌      | ❌     | ❌     | ⚠️         |
| **Automated deployment**      | ✅     | ❌      | ❌     | ⚪     | ✅           |
| **Production monitoring**     | ✅     | ❌      | ❌     | ❌     | ❌           |
| Self-healing systems                | ✅     | ❌      | ❌     | ❌     | ❌           |
| Network effect (failures)           | ✅     | ❌      | ❌     | ❌     | ❌           |
| Works with any LLM                  | ✅     | ❌      | ⚠️   | N/A    | ❌           |
| Internal system access              | ✅     | ⚠️    | ⚠️   | ❌     | ⚠️         |
| Custom workflow code                | ✅     | ❌      | ❌     | ❌     | ⚠️         |
| **Browser automation**        | ✅     | ❌      | ❌     | ❌     | ❌           |
| **Integrated terminal**       | ✅     | ✅      | ❌     | N/A    | ✅           |
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

**Purpose:** Track ALL structural code relationships with bidirectional navigation and version-level granularity
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

**Comprehensive Dependency Tracking (ALL Bidirectional):**

1. **File ↔ File Dependencies**

   - Import relationships (which files import which)
   - Module dependencies (which modules depend on which)
   - Test file ↔ Source file relationships
   - **Version tracking**: File content hash, last modified timestamp
2. **Code Symbol ↔ Code Symbol Dependencies**

   - Function → Function calls (caller/callee relationships)
   - Class → Class inheritance (parent/child hierarchies)
   - Method → Method invocations (cross-class calls)
   - Variable → Variable usage (data flow tracking)
   - **Version tracking**: Symbol signature hash, parameter changes
3. **Package ↔ Package Dependencies**

   - Direct dependencies (package.json, requirements.txt, Cargo.toml)
   - Transitive dependencies (full dependency tree)
   - Peer dependencies and optional dependencies
   - **Version tracking**: Exact version, version range, compatibility matrix
4. **Tool ↔ Tool Dependencies**

   - Build tool chains (webpack → babel → terser)
   - Test framework dependencies (pytest → coverage → plugins)
   - Linter/formatter chains (ESLint → Prettier → plugins)
   - **Version tracking**: Tool version, plugin versions, config file hash
5. **Package ↔ File Dependencies**

   - Which files use which packages (`import numpy` → file.py uses numpy)
   - Unused package detection (packages installed but never imported)
   - Package-to-module mapping (numpy → specific submodules used)
   - **Version tracking**: Import statement location, package version used
6. **User ↔ File Dependencies (MVP)**

   - Active work tracking (which developer is editing which files)
   - File modification history (who last modified, when)
   - Work visibility indicators (show active sessions on files)
   - **Version tracking**: User session ID, file version at edit start
7. **User ↔ Git Checkout Dependencies (Post-MVP)**

   - Branch-to-file mapping (which files changed in which branches)
   - Merge conflict prediction (parallel edits on same files)
   - Work isolation tracking (developer workspace state)
   - **Version tracking**: Git commit SHA, branch name, checkout timestamp

**Edge Types (All Bidirectional Navigable):**

- `Calls` / `CalledBy` - Function call relationships
- `Uses` / `UsedBy` - Variable/resource usage
- `Imports` / `ImportedBy` - Import relationships
- `Inherits` / `InheritedBy` - Class inheritance
- `Defines` / `DefinedBy` - Definition relationships
- `Tests` / `TestedBy` - Test coverage relationships
- `TestDependency` / `HasTest` - Test file dependencies
- `UsesPackage` / `UsedByFile` - Package usage (Post-MVP)
- `DependsOn` / `RequiredBy` - Package dependencies (Post-MVP)
- `EditedBy` / `Edits` - User active work (MVP)
- `ModifiedBy` / `Modifies` - File modification history (MVP)
- `CheckedOut` / `ActiveIn` - Git branch relationships (Post-MVP)

**Version-Level Dependency Tracking:**

```rust
pub struct VersionedEdge {
    edge_type: EdgeType,
    source_id: String,
    target_id: String,

    // Version tracking metadata
    source_version: String,      // Hash or version number
    target_version: String,      // Hash or version number
    created_at: Timestamp,
    last_validated: Timestamp,

    // Dependency constraint (for packages)
    version_constraint: Option<String>,  // ">=1.2.0,<2.0.0"
    is_compatible: bool,

    // Change tracking
    last_modified_by: Option<String>,    // User ID
    modification_count: u32,
}
```

**Web Search & Semantic Graph for Dependency Resolution:**

**Problem:** LLM knowledge is static (training cutoff date), but packages/tools evolve constantly.

**Solution:** Agent uses web search + semantic enhanced dependency graph to access live package documentation:

1. **Web Search Integration (DependencyAssessment State)**

   - **When**: Resolving package versions, compatibility issues, API changes
   - **Primary Method**: Browser automation via Chrome DevTools Protocol (CDP) - **Built-in capability**
     - Agent navigates like a human user: opens browser, searches, clicks links, extracts content
     - **Parallel Processing**: Open multiple tabs simultaneously for faster information gathering
       - Tab 1: PyPI/npm official package page (version, release date)
       - Tab 2: Package changelog/release notes (breaking changes)
       - Tab 3: Migration guide or upgrade documentation
       - Tab 4: Known issues (GitHub issues, Stack Overflow)
       - **Consolidation**: LLM synthesizes information from all tabs into coherent recommendation
       - **Performance**: ~5-8 seconds total (parallel) vs ~15-20 seconds (sequential)
     - **Advantages**: No external dependencies, works offline (cached), full DOM access, parallel efficiency
     - **Trade-off**: Slightly slower than MCP API calls, but more reliable for complex pages
     - **Queries**:
       - Navigate to Google: "numpy version 2.0 breaking changes"
       - Click top results (PyPI, numpy.org, Stack Overflow)
       - Extract version info, changelog, migration guides from page content
   - **Alternative Method (Optional)**: MCP servers for faster API-based search
     - MCP `@modelcontextprotocol/server-brave-search` - Brave Search API (fast, structured)
     - MCP `@modelcontextprotocol/server-fetch` - Direct HTTP fetch (for known URLs)
     - **Advantages**: Faster (<1 second), structured JSON responses
     - **Trade-off**: Requires external MCP server setup, API keys, internet dependency
   - **Strategy**:
     - **Default**: Use browser automation with parallel tabs (built-in, reliable, no setup)
     - **Optimization**: Use MCP servers if installed and configured (faster for simple queries)
     - **Hybrid**: Browser automation for complex pages + MCP for structured API data
     - **Fallback**: If both fail, use LLM knowledge with disclaimer about potential outdated info
   - **Processing**:
     - Fetch official docs (PyPI, npm, crates.io, official websites)
     - Extract version compatibility matrices using tree-sitter or DOM parsing
     - Consolidate information from multiple sources (official docs + community knowledge)
     - Cache results in semantic graph for future use (TTL: 7 days)
2. **Semantic Graph-Enhanced Dependency Resolution (ConflictCheck State)**

   - **When**: Detecting version conflicts, suggesting package updates
   - **Storage**: HNSW index in CodeGraph (in-memory semantic search)
   - **Data Sources**:
     - Cached package documentation (from web search - browser or MCP)
     - Historical conflict resolutions (from past projects, stored as semantic relationships)
     - Known compatibility issues (community knowledge, embedded in graph)
     - Migration guides and changelogs (indexed semantically)
   - **Process**:
     - Query: "How to resolve numpy 1.24 vs pandas 2.1 conflict?"
     - Semantic graph retrieves: Similar past conflicts + official compatibility docs via HNSW search
     - LLM synthesizes: Recommended version upgrade path
3. **Package Documentation Fetching (Post-MVP)**

   - **Automated**: When new package added, fetch + index its documentation
   - **Sources**:
     - Official docs (docs.python.org, react.dev, docs.rs)
     - README files from package repositories
     - Changelog/migration guides
   - **Method**: Browser automation navigates to docs, extracts structured content
   - **Indexing**: Store in Vector DB with semantic embeddings
   - **Usage**: Provide as context to LLM when generating code using that package

**Example Flow (Browser Automation - Default):**

When a user requests "Add numpy for matrix operations", the system must execute the following dependency assessment workflow:

**DependencyAssessment State Requirements:**

1. Check the current numpy version in requirements.txt (example: version 1.24.0)
2. Open a headless Chrome browser instance via Chrome DevTools Protocol
3. Navigate to Google search with query "numpy latest version 2024"
4. Click the first search result (typically the PyPI numpy page)
5. Extract the latest version number, release date, and any breaking changes documentation link
6. Navigate to the breaking changes documentation page (example: numpy.org/doc/2.0/release/2.0.0-notes.html)
7. Extract relevant breaking changes information (example: changes in numpy.random module)
8. Complete the web search process within 8-12 seconds target time
9. Query the semantic dependency graph (HNSW index) for existing migration guides (example: "numpy 1.24 to 2.0 migration")
10. Retrieve any stored migration guidance from the semantic graph

**Agent Proposal Requirements:**
The agent must present a comprehensive recommendation including:

- Current version installed in the project
- Latest version available from PyPI
- Identified breaking changes that affect the codebase
- Recommended version with justification (balancing latest features with compatibility)
- Specific code patterns that would be affected by breaking changes
- Data sources and freshness timestamps for transparency (example: "Verified via PyPI and numpy.org, fetched 30 seconds ago")

Example output format: "Current version: [X], Latest version: [Y], Breaking changes: [summary], Recommendation: [Z version] because [reasoning], Affected code: [specific patterns], Source: [verification details]"

**Implementation Notes:**

- Web search results cached in SQLite with TTL (7 days for package info)
- Semantic embeddings updated incrementally (fastembed + HNSW index in CodeGraph)
- LLM prompted with: "Use web search for latest package info, your knowledge may be outdated"
- Conflict resolution prioritizes: 1) Official docs, 2) Community recommendations, 3) LLM reasoning

**NOT a neural network** - Just a graph data structure with optional embeddings for fuzzy search + web-augmented knowledge.

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
- Cost optimization (90% fewer LLM calls - Post-MVP feature)

**IS a Graph Neural Network** - Real machine learning model trained on code patterns.

**MVP Status:** Yantra Codex is POST-MVP. MVP uses LLM-only generation at standard costs.

### Why Two Systems?

| Aspect                | Dependency Graph (In code GNN is used) | Yantra Codex               |
| --------------------- | -------------------------------------- | -------------------------- |
| **Purpose**     | Code relationships                     | Code generation            |
| **Technology**  | petgraph (data structure)              | GraphSAGE (neural network) |
| **Input**       | AST from tree-sitter                   | Problem description        |
| **Output**      | Dependency queries                     | Generated code             |
| **Speed**       | <1ms                                   | 15ms                       |
| **Learning**    | No learning                            | Continuous learning        |
| **Local/Cloud** | Both (sync structure)                  | Both (sync embeddings)     |
| **Code Name**   | "GNN" (misleading)                     | "Yantra Codex"             |

**Integration:** Dependency Graph provides context → Yantra Codex generates code → Dependency Graph validates new code fits properly

---

## Yantra Codex: AI Pair Programming Engine (POST-MVP / OPTIONAL)

**MVP Status:** System fully functional WITHOUT Yantra Codex. Uses LLM-only generation for MVP. Yantra Codex is a cost optimization feature added post-MVP.

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

When a user submits a request (example: "Create REST API endpoint to get user by ID"), the Yantra Codex must:

- Extract 978-dimensional features from the request to understand the problem
- Use the GNN to predict the appropriate logic pattern (target: within 15ms)
- Generate code using tree-sitter for syntax construction
- Calculate a confidence score between 0.0 and 1.0 indicating certainty
- If confidence score is 0.8 or higher, proceed directly to validation
- If confidence score is below 0.8, proceed to LLM review step

**Step 2: LLM Review (if confidence < 0.8)**

When Yantra Codex confidence is below 0.8 threshold, the system must:

- Send the Yantra-generated code along with identified confidence issues to the LLM
- Use the user's chosen LLM (Claude Sonnet 4, GPT-4 Turbo, or Gemini) for review
- Have the LLM analyze edge cases that may not be handled
- Have the LLM add appropriate error handling mechanisms
- Have the LLM improve overall code quality and robustness
- Return the enhanced code with LLM improvements

**Step 3: Merge & Validate**

After code generation (with or without LLM review), the system must:

- Merge any suggestions from the LLM with Yantra's original code
- Run GNN dependency validation to ensure no breaking changes to dependent code
- Execute all automated tests relevant to the changes
- Present the final validated code to the user for review and approval

**Step 4: Yantra Learns**

After successful code generation and validation, Yantra Codex must:

- Extract the logic pattern from the final validated code
- Store the mapping between the problem description and the LLM-enhanced solution logic
- Perform an incremental GNN model update to learn the new pattern
- Use this learned pattern for future similar requests to improve confidence scores

**Confidence-Based Routing:**

| Confidence        | Routing Decision    | Rationale                            | Cost    |
| ----------------- | ------------------- | ------------------------------------ | ------- |
| **0.9-1.0** | Yantra alone        | Seen pattern many times              | $0      |
| **0.8-0.9** | Yantra alone        | Good confidence, validate with tests | $0      |
| **0.5-0.8** | Yantra + LLM review | Partial knowledge, need LLM help     | ~$0.015 |
| **0.0-0.5** | LLM alone           | Novel pattern, Yantra can't help yet | ~$0.025 |

**Learning Trajectory Example:**

The system must demonstrate progressive learning for repeated patterns. Using a CRUD endpoint pattern as an example:

**Week 1 - Initial Learning Phase:**

- First request: Yantra confidence score of 0.3, requires LLM review, tests pass, Yantra learns from the LLM-enhanced solution
- Tenth request: Yantra confidence improves to 0.5, still requires LLM review, tests pass, Yantra continues learning
- Fiftieth request: Yantra confidence reaches 0.75, still requires LLM review for refinement, tests pass, Yantra refines its understanding

**Week 4 - Mastery Achieved:**

- Two hundredth request: Yantra confidence reaches 0.88, no LLM needed, tests pass independently
- Cost per generation decreases progressively: $0.015 → $0.010 → $0.005 → $0 (100% cost saved for this pattern)

The system must achieve similar learning trajectories for all common patterns, with confidence scores improving through repeated exposure and LLM guidance.

#### 3. Cost & Quality Benefits (Post-MVP Feature)

**Note:** These projections apply ONLY after Yantra Codex is implemented post-MVP. MVP uses LLM-only generation at standard costs.

**Cost Trajectory (vs LLM-only baseline $25/1000 generations):**

- **MVP (Months 1-2):** $25/1000 gen (LLM-only baseline)
- **Post-MVP Month 1:** $9/1000 gen (64% savings) - Yantra handles 55% alone
- **Post-MVP Month 6:** $3/1000 gen (88% savings) - Yantra handles 85% alone
- **Post-MVP Year 1:** $1/1000 gen (96% savings) - Yantra handles 95% alone

**Quality Guarantee:** Yantra + LLM ≥ LLM alone (pair programming is better!)

**Comparison Table (Post-MVP with Yantra Codex):**

| Metric        | LLM Only (MVP)               | Yantra + LLM (Month 1 Post-MVP) | Yantra + LLM (Year 1 Post-MVP) |  |  |  |
| ------------- | ---------------------------- | ------------------------------- | ------------------------------ | - | - | - |
| Cost/1000 gen | $25            | $9 (64% ↓) | $1 (96% ↓)                     |                                |  |  |  |
| Quality       | ⭐⭐⭐⭐⭐                   | ⭐⭐⭐⭐⭐                      | ⭐⭐⭐⭐⭐                     |  |  |  |
| Speed         | 3-5s                         | 0.5-2s                          | 0.015-0.5s                     |  |  |  |
| Learning      | ❌                           | ✅                              | ✅✅✅                         |  |  |  |
| Privacy       | ❌ (cloud)                   | ✅ (mostly local)               | ✅ (95% local)                 |  |  |  |

#### 4. Multi-Language Support

**Universal Logic Patterns (Learned Once):**

- Input validation, error handling, data transformation
- API calls, database operations, async patterns
- Algorithm patterns, architecture patterns

**Language-Specific Syntax (Tree-sitter Provides):**

- Keywords and operators, type systems, standard library
- Language idioms, formatting rules

**Transfer Learning:**

The Yantra Codex system must support cross-language pattern transfer:

**Transfer Learning Requirement:**

1. When a logic pattern is learned in one programming language (example: "retry with exponential backoff" pattern learned from 1,000 Python examples)
2. The system must automatically apply that same logic pattern to code generation requests in other languages (JavaScript, Rust, Go, etc.)
3. Zero additional training should be required for the new languages
4. Tree-sitter parser must handle all language-specific syntax differences
5. The logic pattern embedding remains universal across languages, only syntax translation varies

**Supported Languages:**

- Python ✅, JavaScript ✅, TypeScript ✅
- Rust, Go, Java, C++, etc. (easy to add - ~50 lines per language)

#### 5. Continuous Learning System

**What Yantra Learns:**

1. **Edge Cases**: LLM adds null checks → Yantra learns to add them
2. **Error Handling**: LLM adds try-catch → Yantra learns pattern
3. **Best Practices**: LLM improves naming → Yantra learns conventions
4. **Domain Patterns**: LLM adds auth checks → Yantra learns domain rules

**Learning Metrics (Post-MVP with Yantra Codex):**

The Yantra Codex system must demonstrate progressive improvement in autonomy over time:

**MVP Phase (Months 1-2):**

- LLM handles 100% of code generation
- Cost: $25 per 1000 generations (baseline)

**Post-MVP Month 1:**

- Yantra Codex handles 55% of generations independently
- LLM support needed for remaining 45%
- Cost: $9 per 1000 generations

**Post-MVP Month 3:**

- Yantra Codex handles 70% of generations independently
- LLM support needed for remaining 30%
- Cost: $5 per 1000 generations

**Post-MVP Month 6:**

- Yantra Codex handles 85% of generations independently
- LLM support needed for remaining 15%
- Cost: $3 per 1000 generations

**Post-MVP Month 12:**

- Yantra Codex handles 95% of generations independently
- LLM support needed for only 5% of complex/novel cases
- Cost: $1 per 1000 generations
- Total cost reduction: 96% compared to MVP baseline

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

The Yantra Cloud Codex system must demonstrate scalable improvement based on user adoption:

**100 Users Scenario:**

- 100 users generating 50 requests per day each
- Total: 150,000 patterns collected per month
- Result: Model v1.1 with 65% baseline accuracy

**1,000 Users Scenario:**

- 1,000 users generating 50 requests per day each
- Total: 1.5 million patterns collected per month
- Result: Model v1.6 with 80% baseline accuracy

**10,000 Users Scenario:**

- 10,000 users generating 50 requests per day each
- Total: 15 million patterns collected per month
- Result: Model v2.0 with 90% baseline accuracy

**Network Effect Flywheel:**
More users contribute more patterns → Better model accuracy → Lower LLM costs for all users → More value proposition → Attracts even more users → Cycle repeats

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

## Complete Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           YANTRA PLATFORM                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 1: USER INTERFACE (AI-FIRST)                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  • Chat/Task Interface (Primary - 60% screen)                          │ │
│  │  • Code Viewer with Monaco Editor (Secondary - 25% screen)             │ │
│  │  • Browser Preview with CDP (Live - 15% screen)                        │ │
│  │  • Real-time WebSocket updates                                         │ │
│  │  • SolidJS reactive UI, TailwindCSS styling                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    ↕                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              LAYER 2: ORCHESTRATION & COORDINATION                     │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Unified Tool Interface (UTI)                                   │  │ │
│  │  │  ├─ Protocol Router: MCP / LSP / DAP / Builtin                  │  │ │
│  │  │  ├─ Tool Adapters: 45+ tools, 4 protocols                       │  │ │
│  │  │  ├─ Consumer Abstraction: LLM Agent + Workflow Executor         │  │ │
│  │  │  └─ Protocol Selection: Auto-routing by capability              │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Multi-LLM Orchestration                                        │  │ │
│  │  │  ├─ Primary: Claude Sonnet 4 (code generation, reasoning)       │  │ │
│  │  │  ├─ Secondary: GPT-4 Turbo (validation, fallback)               │  │ │
│  │  │  ├─ Routing: Cost optimization, capability-based selection      │  │ │
│  │  │  ├─ Failover: Circuit breaker, retry with exponential backoff   │  │ │
│  │  │  └─ Response Caching: Redis for repeated queries                │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  PDC State Machine                                              │  │ │
│  │  │  ├─ Phase Transitions: Architect → Plan → Execute → Deploy     │  │ │
│  │  │  ├─ State Persistence: SQLite with WAL mode                     │  │ │
│  │  │  ├─ Rollback Support: Checkpoints at phase boundaries           │  │ │
│  │  │  └─ Approval Gates: Human-in-loop for critical operations       │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Team of Agents (Distributed Intelligence)                      │  │ │
│  │  │  ├─ Architect Agent: Design, dependency planning                │  │ │
│  │  │  ├─ Coding Agent: Code generation, pattern application          │  │ │
│  │  │  ├─ Testing Agent: Test creation, validation orchestration      │  │ │
│  │  │  ├─ Security Agent: Vulnerability scanning, auto-fix            │  │ │
│  │  │  └─ Coordination: Message bus for agent communication           │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    ↕                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   LAYER 3: INTELLIGENCE & REASONING                    │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Dependency Graph (Code Intelligence) - petgraph                │  │ │
│  │  │  ├─ AST Parsing: tree-sitter (Python, JS, Rust, Go, etc.)      │  │ │
│  │  │  ├─ Graph Structure: Nodes (files/funcs/classes), Edges (deps) │  │ │
│  │  │  ├─ Query Engine: <1ms dependency lookups                       │  │ │
│  │  │  ├─ Incremental Updates: <50ms per file change                  │  │ │
│  │  │  ├─ Impact Analysis: Transitive dependency traversal            │  │ │
│  │  │  └─ Storage: In-memory (hot) + SQLite (persistence)             │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Yantra Codex (AI Code Generation) - POST-MVP                   │  │ │
│  │  │  ├─ GraphSAGE GNN: 1024-dim embeddings, 150M parameters        │  │ │
│  │  │  ├─ Inference: 15ms (CPU), 5ms (GPU), ~600MB model              │  │ │
│  │  │  ├─ Pattern Recognition: 978-dim problem features → code logic  │  │ │
│  │  │  ├─ Confidence Scoring: 0.0-1.0 (triggers LLM review < 0.8)     │  │ │
│  │  │  ├─ Continuous Learning: Learns from LLM corrections            │  │ │
│  │  │  ├─ Cost Optimization: 96% LLM call reduction after 12 months  │  │ │
│  │  │  └─ MVP: NOT INCLUDED - uses LLM-only generation                │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Semantic Enhanced Dependency Graph - fastembed + HNSW         │  │ │
│  │  │  ├─ Code Templates: Pre-validated patterns indexed semantically│  │ │
│  │  │  ├─ Best Practices: Language-specific idioms in graph          │  │ │
│  │  │  ├─ Project Patterns: Learned from codebase dependencies       │  │ │
│  │  │  ├─ Failure Library: Known issues as graph relationships       │  │ │
│  │  │  └─ Semantic Search: <10ms HNSW retrieval for context assembly │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Context Assembly Engine                                        │  │ │
│  │  │  ├─ Token Counting: Track context limits per LLM                │  │ │
│  │  │  ├─ Hierarchical Assembly: Priority-based context inclusion     │  │ │
│  │  │  ├─ Compression: Summarize low-priority context                 │  │ │
│  │  │  ├─ Chunking: Split large operations across multiple calls      │  │ │
│  │  │  └─ Adaptive Strategies: Dynamic context based on task type     │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    ↕                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 4: VALIDATION & SECURITY                      │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  5-Layer Security Framework (Prevention Stack)                  │  │ │
│  │  │  ├─ Layer 1: Pre-Generation (Intent validation, context check)  │  │ │
│  │  │  ├─ Layer 2: Generation-Time (Pattern safety, injection guards) │  │ │
│  │  │  ├─ Layer 3: Post-Generation (AST validation, syntax check)     │  │ │
│  │  │  ├─ Layer 4: Pre-Commit (Semgrep OWASP, secret scanning)        │  │ │
│  │  │  └─ Layer 5: Runtime Monitoring (Execution safety, sandboxing)  │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Testing Framework                                              │  │ │
│  │  │  ├─ Unit Test Generation: pytest (Python), Jest (JavaScript)    │  │ │
│  │  │  ├─ Integration Tests: End-to-end flow validation               │  │ │
│  │  │  ├─ Coverage Analysis: 90%+ target enforcement                  │  │ │
│  │  │  ├─ Test Execution: Parallel execution, <30s typical runtime    │  │ │
│  │  │  ├─ Result Validation: 100% pass rate mandatory                 │  │ │
│  │  │  └─ Mock UI Testing: Component isolation testing                │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Browser Integration (Chrome DevTools Protocol)                 │  │ │
│  │  │  ├─ Live Preview: Real-time UI rendering via chromiumoxide      │  │ │
│  │  │  ├─ Visual Validation: Screenshot diffs, layout verification    │  │ │
│  │  │  ├─ Interaction Testing: Automated user flow testing            │  │ │
│  │  │  ├─ Console Monitoring: Runtime error detection                 │  │ │
│  │  │  └─ Performance Metrics: Core Web Vitals tracking               │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Dependency Validator                                           │  │ │
│  │  │  ├─ GNN-Powered: Query dependency graph for conflicts           │  │ │
│  │  │  ├─ Breaking Change Detection: Transitive impact analysis       │  │ │
│  │  │  ├─ Circular Dependency Prevention: Pre-commit validation       │  │ │
│  │  │  └─ External API Tracking: Monitor API dependencies             │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Agentic Validation Pipeline                                    │  │ │
│  │  │  ├─ Code Generation → Validation Loop                           │  │ │
│  │  │  ├─ Confidence Scoring: Auto-retry logic based on confidence    │  │ │
│  │  │  ├─ Failure Analysis: Extract patterns from failures            │  │ │
│  │  │  ├─ Self-Healing: Auto-fix with Known Issues DB                 │  │ │
│  │  │  └─ Escalation: Human approval for unresolved issues            │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    ↕                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                  LAYER 5: INTEGRATION & PERSISTENCE                    │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Git Integration (MCP Protocol)                                 │  │ │
│  │  │  ├─ Repository Operations: clone, commit, push, pull, branch    │  │ │
│  │  │  ├─ Conflict Resolution: Auto-merge with GNN conflict detection │  │ │
│  │  │  ├─ Commit Strategy: Atomic commits per logical change          │  │ │
│  │  │  ├─ History Analysis: Blame, diff, log integration              │  │ │
│  │  │  └─ Branch Management: Feature branch workflow automation       │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  File System Operations                                         │  │ │
│  │  │  ├─ File Locking: SQLite-based distributed locking              │  │ │
│  │  │  ├─ CRUD Operations: Create, read, update, delete with locking  │  │ │
│  │  │  ├─ Watch Service: Real-time file change monitoring             │  │ │
│  │  │  ├─ Conflict Prevention: Lock coordination across agents        │  │ │
│  │  │  └─ Transaction Support: Rollback on validation failure         │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  5-Tier Data Storage Architecture                               │  │ │
│  │  │  ├─ Tier 0: Cloud Storage (Optional, team coordination)         │  │ │
│  │  │  ├─ Tier 1: In-Memory (Hot path: Dependency Graph, active)      │  │ │
│  │  │  ├─ Tier 2: Local SQLite/sled (Agent coordination, locks)       │  │ │
│  │  │  ├─ Tier 3: SQLite Reference (Read-heavy: config, logs)         │  │ │
│  │  │  └─ Tier 4: Cache (Ephemeral: context assembly, LRU)            │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  External Integrations (Post-MVP)                               │  │ │
│  │  │  ├─ REST APIs: HTTP client with retry logic                     │  │ │
│  │  │  ├─ WebSockets: Real-time external data streams                 │  │ │
│  │  │  ├─ Third-Party Services: Slack, SendGrid, Stripe, etc.         │  │ │
│  │  │  └─ Webhook Triggers: Event-driven workflow activation          │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

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
  - **Purpose**: Real-time visibility into which files/dependencies team members are working on
  - **Data**: Dependency graph metadata (NO actual code - privacy-preserving)
  - **Query Pattern**: Agents query Tier 0 for "Who is currently working on which files?"
  - **Sync Model**: Like Google Drive/Git - eventual consistency with conflict detection
  - **API**: WebSocket/gRPC for <50ms latency queries
  - **Deployment**: Hosted (yantra-cloud.com) OR self-hosted (enterprise)
  - **Use Cases**: File modification registry, real-time dependency tracking, transitive dependency awareness
  - **Note:** This is a graph DATABASE for storage/coordination, NOT a neural network. In code you may see "GNN" which refers to Dependency Graph, not Yantra Codex.
- **Tier 1 - Local Dependency Graph (Hot Path):**
  - **Phase 1-2 (NOW)**: Hybrid (in-memory petgraph + SQLite persistence with WAL + pooling)
  - **Phase 3 (Month 5-8)**: Pure in-memory + snapshot persistence for >100K LOC projects
  - Target: Sub-millisecond query times, ~1GB memory for 100K LOC
  - Tech: petgraph in-memory + bincode snapshots + write-ahead log
  - **Sync**: Local dependency graph syncs metadata to Cloud Graph DB (Tier 0)
  - **Note:** Called "GNN" in codebase (historical naming). This is NOT a neural network - it's a graph data structure (petgraph) for tracking code relationships.

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
  - **Phase 2A (Month 3-4)**: Migrate to sled for team of agents + Git coordination branch
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
- **In Specs:** We call this "Dependency Graph" for clarity
- **Reality:** petgraph-based graph data structure, NOT a Graph Neural Network
- **Purpose:** Track structural code relationships (imports, function calls, inheritance)
- **Optional:** Semantic embeddings for fuzzy search (using fastembed, not training a neural network)
- **Note:** In code, "GNN" refers to this Dependency Graph, not Yantra Codex

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

When a user requests "Add email validation to user registration", the system must perform both structural and semantic searches:

**Structural Search Requirements (Exact Dependencies):**
The system must find all direct dependencies of the target function "register_user":

- Identify UserModel class that is called by register_user function
- Identify Database.save() method that is called by UserModel
- Identify sanitize_input() function that is called by register_user
- Return all exact structural dependencies to ensure correctness

**Semantic Search Requirements (Fuzzy Discovery):**
The system must find semantically similar code using minimum similarity threshold of 0.75:

- Search for existing validate_email() function (semantically relevant even if not yet called)
- Discover validate_phone() function (similar validation pattern that could be reused)
- Identify check_email_format() function (potential duplicate that should be consolidated)
- Return semantically relevant code that may not have structural relationships yet

**Combined Search Benefits:**
The LLM receives both exact dependencies AND semantically relevant code, enabling:

- Better code generation by discovering existing validation functions instead of recreating them
- Duplication prevention by finding similar functions that could be reused
- Correctness maintenance through exact dependency tracking to ensure nothing breaks

**Example 2: Refactoring Detection**

The system must identify potential duplicate functions for consolidation:

**Duplicate Detection Requirements:**

1. Search for functions semantically similar to a target node (example: "utils::validate_email")
2. Use minimum similarity threshold of 0.90 to identify strong duplicates
3. Limit results to maximum of 10 candidates
4. Rank all results by cosine similarity score

**Expected Results Format:**
List candidates in descending similarity order with scores:

- First candidate with similarity >= 0.95 indicates high likelihood of duplication
- Second candidate with similarity >= 0.92 indicates strong similarity
- Third candidate with similarity >= 0.90 meets minimum threshold

**Refactoring Action Requirements:**

- Suggest consolidating similar functions into a single implementation
- Use structural graph to identify all call sites that need updating
- Ensure all callers are updated correctly to prevent breakage

**Performance Targets:**

| Operation                    | Target          | Implementation                           |
| ---------------------------- | --------------- | ---------------------------------------- |
| Graph build                  | <5s for 10K LOC | Incremental parsing with tree-sitter     |
| Incremental update           | <50ms per file  | Only reparse changed file + update edges |
| Dependency lookup            | <10ms           | petgraph BFS traversal                   |
| Semantic search (MVP)        | <10ms           | **HNSW indexing (hnsw_rs)**        |
| Embedding generation         | <10ms per node  | fastembed-rs with quantized ONNX         |
| Batch embeddings (100 nodes) | <100ms          | Parallel processing                      |

**HNSW Vector Indexing (Ferrari MVP Standard):**

Yantra uses **Hierarchical Navigable Small World (HNSW)** indexing for blazing-fast semantic search at any scale. This is NOT a separate vector database—it's an in-memory index structure built directly within the CodeGraph.

**Why HNSW (Not Linear Scan):**

The system must use HNSW indexing instead of linear scan to meet performance targets at scale:

**Performance Comparison Requirements:**

**Linear Scan Approach (Inadequate):**

- For 1,000 nodes: Achieves 0.5ms (meets target)
- For 10,000 nodes: Takes 50ms (fails - 5x over 10ms target)
- For 100,000 nodes: Takes 500ms (fails - 50x over 10ms target)
- Conclusion: Linear scan does not scale beyond small codebases

**HNSW Index Approach (Required for Ferrari Standard):**

- For 1,000 nodes: Achieves 0.1ms (5x faster than linear scan, well under target)
- For 10,000 nodes: Achieves 2ms (25x faster than linear scan, meets target)
- For 100,000 nodes: Achieves 5ms (100x faster than linear scan, meets <10ms target)
- Conclusion: HNSW indexing required to maintain sub-10ms semantic search at enterprise scale

**Implementation:**

**Semantic Search Index Requirements:**

The CodeGraph must implement HNSW (Hierarchical Navigable Small World) indexing for O(log n) semantic search performance.

**HNSW Index Configuration:**

The system must configure the HNSW semantic index with:

- M parameter: 16 (maximum number of connections per node, controls connectivity)
- max_elements: 10,000 (initial capacity for expected codebase size)
- ef_construction: 16 (build quality parameter for index construction)
- ef_search: 200 (query accuracy parameter for search operations)
- Distance metric: Cosine distance (measures similarity between embedding vectors)

**Semantic Index Building Requirements:**

When building the semantic index, the system must:

**Index Construction:**

- Create a new HNSW index instance with configured parameters
- For each node in the code graph:
  - Check if the node has a semantic embedding vector
  - If embedding exists:
    - Extract the embedding vector (array of floats)
    - Insert the embedding into the HNSW index
    - Associate the embedding with the node's index for lookup
- Store the completed HNSW index in the CodeGraph structure

**Similarity Search Requirements:**

When finding similar nodes, the system must:

**Search Execution:**

- Accept a query embedding vector from the caller
- Accept maximum number of results to return
- Execute HNSW search with ef_search parameter (200) for query accuracy
- Return list of neighbor nodes ordered by similarity

**Result Processing:**

- For each neighbor result from HNSW:
  - Convert the neighbor's ID to a NodeIndex
  - Retrieve the corresponding CodeNode from the graph
  - Convert HNSW distance to similarity score (similarity = 1.0 - distance)
  - Create result tuple containing the CodeNode and similarity score
- Filter out any invalid node references
- Return complete list of similar nodes with similarity scores

**Performance Characteristics:**

The system must achieve:

- O(log n) search complexity for finding similar nodes
- Sub-second response time for semantic searches even with 10,000+ nodes
- High recall (ability to find relevant similar nodes)
- Adjustable accuracy via ef_search parameter

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

**Why Integrated Semantic Graph (Not Separate Vector Database):**

❌ **Traditional Separate Vector Database (e.g., ChromaDB):**

- Duplicate storage (code in files + embeddings in separate vector DB)
- Sync complexity (keep vector DB manually updated with code changes)
- Two separate queries (one for vectors, one for dependencies)
- False positives (semantic similarity without structural context)
- Additional infrastructure to maintain

✅ **Semantic-Enhanced Dependency Graph (Yantra):**

- Single source of truth (graph contains structure + embeddings)
- Automatic sync (update node → embedding updates inline)
- Single query (BFS traversal filters by similarity simultaneously)
- Precision + recall (exact dependencies + fuzzy discovery)
- No external database required
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

#### 4. Semantic Graph-Enhanced Context Retrieval

**Semantic Enhanced Dependency Graph Integration:**

**Embeddings Storage in Graph:**

- All function signatures and docstrings as node properties
- Common code patterns indexed via HNSW
- Failure patterns with fixes stored as relationships
- Best practices and conventions as semantic nodes

**Semantic Search via HNSW:**

- Query: User intent + target code context
- Retrieve: Top-K most relevant patterns via HNSW index (K=10-20)
- Add to context as examples
- **Cost: 2,000-5,000 tokens (high value)**
- **Speed: <10ms query time (in-memory HNSW)**

**Example:**

```
User: "Add authentication to the API endpoint"

Semantic graph retrieves via HNSW search:
1. JWT authentication pattern from similar endpoints (200 tokens)
2. Session management example from project history (300 tokens)
3. Security best practices from documentation nodes (150 tokens)
4. Similar endpoint with auth from dependency neighbors (400 tokens)

Total: 1,050 tokens for highly relevant context
```

#### 5. Adaptive Context Strategies

**Based on Task Type:**

| Task        | Strategy                           | Token Allocation     |
| ----------- | ---------------------------------- | -------------------- |
| New feature | Wide context (many dependencies)   | 70% L1+L2            |
| Bug fix     | Deep context (full implementation) | 80% L1               |
| Refactoring | Architectural context (all usages) | 50% L1, 40% L2       |
| Testing     | Target code + similar tests        | 60% L1, 30% Semantic |

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
   - Semantic graph provides proven patterns via HNSW search
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
| Semantic search        | <200ms     | <100ms       |
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
- HNSW semantic search optimization
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

### ⚠️ CRITICAL: Protocol Architecture Reference

**IMPORTANT:** All capabilities listed below MUST be implemented according to the **Unified Tool Interface (UTI) architecture** specified in:

📄 **`docs/Research & specifications/*Yantra Unified Tool Interface.md`**

The UTI specification defines:

- **Protocol Selection:** Which capabilities use MCP, LSP, DAP, or Builtin
- **Two Consumers, Two Protocols:** Editor (Monaco) uses LSP, Agent uses MCP + Builtin
- **Comprehensive Capability Matrix:** 21+ categories with protocol decisions, fallback strategies, and implementation priorities
- **Why Builtin vs MCP:** Core differentiators (GNN, Tree-sitter, Terminal, Browser) are Builtin; ecosystem services (Git, Database, Deployment) use MCP

**Protocol Decision Framework (from UTI):**

| Protocol          | When to Use                                                   | Examples                                |
| ----------------- | ------------------------------------------------------------- | --------------------------------------- |
| **Builtin** | Core differentiator, performance-critical, security-critical  | File ops, Terminal, GNN, Tree-sitter    |
| **MCP**     | Well-maintained community servers, platform-specific services | Git, GitHub, Database, Railway, Slack   |
| **LSP**     | Real-time editor features ONLY (not exposed to Agent)         | Autocomplete, hover, diagnostics in IDE |
| **DAP**     | Debugging-specific operations                                 | Breakpoints, step debugging, variables  |

**Agent Access:** Agent uses UTI abstraction layer which routes to MCP + Builtin. LSP is used internally by Monaco editor but NOT exposed through UTI to the Agent.

**Implementation Requirement:** When implementing any capability below, developers MUST:

1. Consult UTI specification for protocol decision
2. Use UTI adapter layer (not direct protocol calls)
3. Follow fallback strategies defined in UTI
4. Respect protocol boundaries (e.g., don't expose LSP to Agent)

---

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

**UTI Protocol Reference:** All capabilities in this section use protocols as defined in UTI specification.

#### 1.1 File System Operations

**Protocol:** **Builtin** (Primary) | MCP (Fallback via @modelcontextprotocol/server-filesystem)
**Rationale:** File operations are core to everything. Must be fast, reliable, and under our control.

| Capability           | Tool/Terminal | MVP Status | Purpose                                                | Implementation                                          |
| -------------------- | ------------- | ---------- | ------------------------------------------------------ | ------------------------------------------------------- |
| `file_read`        | Tool          | ✅ DONE    | Read file contents with encoding detection             | `main.rs::read_file()` **[Builtin]**            |
| `file_write`       | Tool          | ✅ DONE    | Create/overwrite files                                 | `main.rs::write_file()` **[Builtin]**           |
| `file_edit`        | Tool          | 🔴 TODO    | Surgical edits (line range, search-replace, AST-based) | **NEW [Builtin]**                                 |
| `file_delete`      | Tool          | 🔴 TODO    | Remove files safely                                    | **NEW [Builtin]**                                 |
| `file_move`        | Tool          | 🔴 TODO    | Rename/move files with dependency updates              | **NEW [Builtin]**                                 |
| `file_copy`        | Tool          | 🔴 TODO    | Duplicate files                                        | **NEW [Builtin]**                                 |
| `directory_create` | Tool          | ✅ DONE    | Create directories recursively                         | Built-in**[Builtin]**                                   |
| `directory_list`   | Tool          | ✅ DONE    | List contents with filters                             | `main.rs::read_dir()` **[Builtin]**             |
| `directory_tree`   | Tool          | 🔴 TODO    | Get full project structure                             | **NEW [Builtin]**                                 |
| `file_search`      | Tool          | 🔴 TODO    | Find files by name/pattern/glob                        | **NEW [Builtin]**                                 |
| `file_watch`       | Tool          | 🔴 TODO    | Monitor for changes (reactive agents)                  | **NEW [Builtin]** (use `notify` crate)          |
| `docx_read`        | Tool          | 🔴 TODO    | Read Word documents                                    | **NEW [Builtin]** (use `docx-rs`)               |
| `pdf_read`         | Tool          | 🔴 TODO    | Extract text from PDFs                                 | **NEW [Builtin]** (use `pdf-extract`/`lopdf`) |

**Implementation Priority:**

- **P0 (MVP):** ✅ Basic file I/O (read, write, list) - DONE
- **P1 (High):** 🔴 Document readers (DOCX, PDF) for architecture generation
- **P2 (Medium):** 🔴 Advanced file ops (edit, delete, move, search)
- **P3 (Future):** 🔴 File watchers for reactive behavior

#### 1.2 Code Intelligence (Tree-sitter Powered)

| Capability         | Tool/Terminal | MVP Status | Purpose                                        | Implementation                  |
| ------------------ | ------------- | ---------- | ---------------------------------------------- | ------------------------------- |
| `parse_ast`      | Tool          | ✅ DONE    | Get AST for file/snippet                       | `gnn/parser.rs` (tree-sitter) |
| `get_symbols`    | Tool          | ✅ DONE    | Extract functions, classes, variables, imports | `gnn/parser.rs`               |
| `get_references` | Tool          | 🔴 TODO    | Find all usages of a symbol                    | **NEW**                   |
| `get_definition` | Tool          | 🔴 TODO    | Jump to definition                             | **NEW**                   |
| `get_scope`      | Tool          | 🔴 TODO    | Get scope context for a position               | **NEW**                   |

#### 1.2 Code Intelligence (Tree-sitter Powered)

**Protocol:** **Builtin** (Primary via Tree-sitter) | MCP (Secondary via Pylance/rust-analyzer) | LSP (Editor-only, not exposed to Agent)
**Rationale:** Tree-sitter is core differentiator. Fast, multi-language, works offline. MCP for advanced features when available.

| Capability             | Tool/Terminal | MVP Status | Purpose                                        | Implementation                                      |
| ---------------------- | ------------- | ---------- | ---------------------------------------------- | --------------------------------------------------- |
| `parse_ast`          | Tool          | ✅ DONE    | Get AST for file/snippet                       | `gnn/parser.rs` (tree-sitter) **[Builtin]** |
| `get_symbols`        | Tool          | ✅ DONE    | Extract functions, classes, variables, imports | `gnn/parser.rs` **[Builtin]**               |
| `get_references`     | Tool          | 🔴 TODO    | Find all usages of a symbol                    | **NEW [MCP/Builtin fallback]**                |
| `get_definition`     | Tool          | 🔴 TODO    | Jump to definition                             | **NEW [MCP/Builtin fallback]**                |
| `get_scope`          | Tool          | 🔴 TODO    | Get scope context for a position               | **NEW [Builtin]**                             |
| `get_diagnostics`    | Tool          | ✅ DONE    | Syntax errors, warnings                        | Integrated in parser**[Builtin]**                   |
| `semantic_search`    | Tool          | ✅ PARTIAL | Search code by meaning                         | GNN semantic layer (embeddings)**[Builtin]**  |
| `get_call_hierarchy` | Tool          | ✅ DONE    | Incoming/outgoing calls                        | GNN dependency tracking**[Builtin]**                |
| `get_type_hierarchy` | Tool          | 🔴 TODO    | Class inheritance chains                       | **NEW [MCP/Builtin fallback]**                |

**Implementation Priority:**

- **P0 (MVP):** ✅ AST parsing, symbol extraction, call hierarchy - DONE
- **P1 (High):** ✅ Semantic search - DONE (GNN embeddings)
- **P2 (Medium):** 🔴 References, definitions, scope analysis
- **P3 (Future):** 🔴 Type hierarchy

#### 1.3 Dependency Graph & Impact Analysis

**Protocol:** **Builtin** (Exclusive - Core Differentiator)
**Rationale:** GNN-powered analysis is Yantra's core differentiator. petgraph data structure (not a neural network, historical naming).

| Capability                 | Tool/Terminal | MVP Status | Purpose                           | Implementation                                             |
| -------------------------- | ------------- | ---------- | --------------------------------- | ---------------------------------------------------------- |
| `build_dependency_graph` | Tool          | ✅ DONE    | Generate full project graph       | `gnn/engine.rs` (10/10 features) **[Builtin]**     |
| `get_dependents`         | Tool          | ✅ DONE    | What depends on X?                | `gnn/engine.rs::get_dependents()` **[Builtin]**    |
| `get_dependencies`       | Tool          | ✅ DONE    | What does X depend on?            | `gnn/engine.rs::get_dependencies()` **[Builtin]**  |
| `impact_analysis`        | Tool          | ✅ DONE    | If I change X, what breaks?       | `architecture/deviation_detector.rs` **[Builtin]** |
| `find_cycles`            | Tool          | ✅ DONE    | Detect circular dependencies      | `gnn/engine.rs::detect_cycles()` **[Builtin]**     |
| `get_module_boundaries`  | Tool          | 🔴 TODO    | Identify architectural layers     | **NEW [Builtin]**                                    |
| `cross_repo_deps`        | Tool          | 🔴 TODO    | External API/service dependencies | **NEW [Builtin]** (Phase 2)                          |

**Implementation Priority:**

- **P0 (MVP):** ✅ Full dependency graph - DONE (100%)
- **P1 (High):** ✅ Impact analysis - DONE
- **P2 (Future):** 🔴 Module boundaries, cross-repo dependencies

#### 1.4 Database Connections & Schema Intelligence

**Protocol:** **MCP** (Primary via DB-specific MCP servers) | Builtin (Fallback for SQLite)
**Rationale:** Well-maintained MCP servers exist for major databases. Platform-specific, not core differentiator.

| Capability     | Tool/Terminal  | MVP Status | Purpose                                  | Implementation                                               |
| -------------- | -------------- | ---------- | ---------------------------------------- | ------------------------------------------------------------ |
| `db_connect` | **TOOL** | 🔴 TODO    | Establish connection with pooling        | **NEW [MCP]** `agent/database/connection_manager.rs` |
| `db_query`   | **TOOL** | 🔴 TODO    | Execute SELECT (read-only, validated)    | **NEW [MCP]**                                          |
| `db_execute` | **TOOL** | 🔴 TODO    | Execute INSERT/UPDATE/DELETE (validated) | **NEW [MCP]**                                          |
| `db_schema`  | **TOOL** | 🔴 TODO    | Get tables, columns, types, constraints  | **NEW [MCP]**                                          |
| `db_explain` | **TOOL** | 🔴 TODO    | Query execution plan                     | **NEW [MCP]**                                          |
| `db_migrate` | **TOOL** | 🔴 TODO    | Run migrations with rollback             | **NEW [MCP]** `agent/database/migration_manager.rs`  |
| `db_seed`    | **TOOL** | 🔴 TODO    | Insert test data                         | **NEW [MCP]**                                          |

**MCP Servers:** Postgres MCP, MySQL MCP, SQLite MCP, MongoDB MCP servers

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

- PostgreSQL (via MCP server or `tokio-postgres`)
- MySQL (via MCP server or `sqlx`)
- SQLite (via `rusqlite` - already in use for GNN) **[Builtin fallback]**
- MongoDB (via MCP server or `mongodb` crate)
- Redis (via MCP server or `redis` crate)

#### 1.5 API Monitoring & Contract Validation

**Protocol:** **MCP** (Primary for external APIs) | Builtin (HTTP client for health checks)
**Rationale:** API-specific operations benefit from MCP ecosystem. Builtin HTTP client for basic health checks.

| Capability                | Tool/Terminal  | MVP Status | Purpose                       | Implementation                                                  |
| ------------------------- | -------------- | ---------- | ----------------------------- | --------------------------------------------------------------- |
| `api_import_spec`       | **TOOL** | 🔴 TODO    | Import OpenAPI/Swagger specs  | **NEW [MCP]** `agent/api_monitor/spec_parser.rs`        |
| `api_validate_contract` | **TOOL** | 🔴 TODO    | Detect breaking API changes   | **NEW [MCP]** `agent/api_monitor/contract_validator.rs` |
| `api_health_check`      | **TOOL** | 🔴 TODO    | Test endpoint availability    | **NEW [Builtin HTTP]**                                    |
| `api_rate_limit_check`  | **TOOL** | 🔴 TODO    | Track and predict rate limits | **NEW [Builtin]**                                         |
| `api_mock`              | **TOOL** | 🔴 TODO    | Create mock server from spec  | **NEW [MCP]** (Phase 2)                                   |
| `api_test`              | **TOOL** | 🔴 TODO    | Test endpoint with assertions | **NEW [MCP]** (Phase 2)                                   |

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

**Protocol:** **Builtin** (Exclusive)
**Rationale:** System resource monitoring is low-level, performance-critical. Must be fast and cross-platform.

| Capability                | Tool/Terminal | MVP Status | Purpose                      | Implementation                                         |
| ------------------------- | ------------- | ---------- | ---------------------------- | ------------------------------------------------------ |
| `env_get` / `env_set` | Terminal      | ✅ DONE    | Environment variables        | Via terminal commands**[Builtin]**                     |
| `get_cpu_usage`         | Tool          | 🔴 TODO    | CPU metrics for optimization | **NEW [Builtin]** `agent/resources/monitor.rs` |
| `get_memory_usage`      | Tool          | 🔴 TODO    | Memory stats                 | **NEW [Builtin]**                                |
| `get_disk_usage`        | Tool          | 🔴 TODO    | Disk space monitoring        | **NEW [Builtin]**                                |
| `should_throttle`       | Tool          | 🔴 TODO    | Adaptive resource management | **NEW [Builtin]**                                |

**Implementation Priority:**

- **P2 (Medium):** Resource monitoring for performance optimization
- **P3 (Future):** Adaptive behavior based on system load

### 2. 🧠 REASON - Decision-Making & Analysis Layer

**Purpose:** Analyze context, make decisions, plan actions

**Status:** ✅ **100% COMPLETE** - All reasoning capabilities implemented

| Capability                | MVP Status | Implementation                                                  |
| ------------------------- | ---------- | --------------------------------------------------------------- |
| Confidence Scoring        | ✅ DONE    | `agent/confidence.rs` (320 lines)                             |
| Impact Analysis           | ✅ DONE    | `architecture/deviation_detector.rs::analyze_change_impact()` |
| Risk Assessment           | ✅ DONE    | `RiskLevel` enum: Low/Medium/High/Critical                    |
| Decision Logging          | ✅ DONE    | State machine persistence in SQLite                             |
| Multi-LLM Orchestration   | ✅ DONE    | `llm/multi_llm_manager.rs` (13 providers)                     |
| Validation Pipeline       | ✅ DONE    | `agent/validation.rs` (412 lines)                             |
| Error Analysis            | ✅ DONE    | `agent/orchestrator.rs::analyze_error()`                      |
| Adaptive Context Assembly | ✅ DONE    | Hierarchical context with GNN                                   |

**No additional capabilities needed** - reasoning layer is complete.

### 3. ⚡ ACT - Execution & Action Layer

**Purpose:** Execute decisions, modify code, run commands, deploy systems

#### 3.1 Terminal & Shell Execution

**Protocol:** **Builtin** (Exclusive - Security Critical)
**Rationale:** Shell execution is security-critical. Must be under complete control with no third-party intermediaries.

| Capability               | Tool/Terminal | MVP Status | Purpose                                          | Implementation                                           |
| ------------------------ | ------------- | ---------- | ------------------------------------------------ | -------------------------------------------------------- |
| `shell_exec`           | Tool          | ✅ DONE    | Run command, get output                          | `agent/terminal.rs` (391 lines) **[Builtin]**    |
| `shell_exec_streaming` | Tool          | ✅ DONE    | Long-running with real-time output               | `terminal/executor.rs` (331 lines) **[Builtin]** |
| `shell_background`     | Tool          | ✅ DONE    | Start background process                         | `terminal/pty_terminal.rs` **[Builtin]**         |
| `shell_kill`           | Tool          | ✅ DONE    | Terminate process                                | Terminal management**[Builtin]**                         |
| `shell_interactive`    | Tool          | ✅ DONE    | Pseudo-TTY for interactive CLIs                  | PTY implementation**[Builtin]**                          |
| Smart Terminal Reuse     | Tool          | ✅ DONE    | Detect idle terminals, reuse before creating new | Process detection**[Builtin]**                           |

**Status:** ✅ **100% COMPLETE** - Full terminal capabilities implemented

---

### 🆕 Agent Execution Intelligence (MVP Critical - Phase 1)

**Problem:** Agent blocks on long-running commands (builds, tests, servers), becoming unresponsive and appearing "frozen" to users. This creates poor UX and prevents parallel task execution.

**Solution:** Intelligent command classification with automatic background execution strategy and transparent status reporting.

#### Command Classification & Execution Strategy

**Agent Intelligence:** Automatically detect command duration and choose optimal execution pattern.

| Command Pattern           | Duration       | Strategy             | Agent Behavior                                            | Example                                              |
| ------------------------- | -------------- | -------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| **Build commands**  | Long (10-60s)  | Background + polling | Execute in background, poll every 2-5s, report progress   | `npm run build`, `cargo build`, `make`         |
| **Test execution**  | Long (5-30s)   | Background + polling | Execute in background, show test progress, report results | `pytest`, `npm test`, `cargo test`             |
| **Dev servers**     | Infinite       | Background + monitor | Fire and forget, monitor for crashes, report when ready   | `npm start`, `python manage.py runserver`        |
| **Package install** | Medium (5-20s) | Background + polling | Execute in background, report completion                  | `npm install`, `pip install -r requirements.txt` |
| **Quick queries**   | Quick (<1s)    | Synchronous          | Execute immediately, return result                        | `git status`, `ls`, `cat file.py`              |
| **File operations** | Quick (<1s)    | Synchronous          | Execute immediately, return result                        | `cp`, `mv`, `rm` (non-recursive)               |

#### Transparent Agent Communication

**Principle:** Agent must ALWAYS explain what it's doing and why. No silent operations.

**Status Messages (Required for All Long-Running Commands):**

```
🔨 Building project...
   Strategy: Background execution (expected 30s)
   Reason: Build commands block for extended periods
   Status: You can ask me anything while I monitor this!

   [2s] Still building... (checking every 2s)
   [4s] Still building...
   [6s] Build output: Compiling 47 files...
   [10s] Build complete! ✅
```

**Transparency Requirements:**

1. **Start:** Explain command classification

   - "Detected long-running command (npm run build)"
   - "Executing in background to keep me responsive"
2. **During:** Show progress updates

   - Poll status every 2-5 seconds
   - Show relevant output lines (errors, warnings, completion %)
   - Remind user: "I'm still available for other tasks!"
3. **Completion:** Report results clearly

   - Success: "Build completed in 23s ✅"
   - Failure: "Build failed ❌ [show error excerpt]"
   - Next steps: "Ready to run tests or deploy"
4. **Interruption:** Allow user to stop

   - User: "Stop that build"
   - Agent: "Terminating npm build process... Done ✅"

#### Implementation Architecture

**Required Components:**

1. **Command Classifier** (`agent/command_classifier.rs` - NEW)

   - Maintain pattern database mapping commands to duration categories (Quick/Medium/Long/Infinite)
   - Match commands using exact matches first, then regex patterns
   - Default to Medium duration if command unknown
   - Provide human-readable classification explanations
2. **Intelligent Executor** (Update `agent/orchestrator.rs`)

   - Classify command before execution
   - Emit transparency message explaining strategy
   - Execute synchronously for Quick commands
   - Execute with polling for Medium/Long commands
   - Execute fire-and-forget for Infinite commands (servers)
   - Poll background tasks every 2 seconds and emit status updates
   - Show recent output lines during execution
   - Report completion with timing and exit status
3. **Status Communication System**

   - Emit real-time events to frontend UI
   - Support status messages, progress indicators, and error notifications
   - Enable user interruption of running processes
   - Track elapsed time for all operations

#### UI Integration

**Agent Status Panel (Real-time Updates):**

```
┌─────────────────────────────────────────┐
│ 🤖 Agent Status: ACTIVE & AVAILABLE     │
├─────────────────────────────────────────┤
│ Current Tasks:                          │
│                                         │
│ 🔨 [Background] npm run build           │
│    ⏱️  Running for 12s                  │
│    📊 Compiling 47/150 files           │
│    💬 You can ask me anything!         │
│                                         │
│ ✅ [Complete] Generated UserService.py │
│    ⏱️  Took 3s                          │
│    📝 Added to src/services/           │
└─────────────────────────────────────────┘
```

#### Performance Targets

| Metric                                       | Target          | Rationale                          |
| -------------------------------------------- | --------------- | ---------------------------------- |
| Classification time                          | <5ms            | Must not add latency               |
| Status poll interval                         | 2-5s            | Balance responsiveness vs overhead |
| Status message latency                       | <50ms           | Real-time feel                     |
| Agent response time (during background task) | <200ms          | Must remain interactive            |
| Terminal pool utilization                    | >70% reuse rate | Efficient resource usage           |

#### Success Metrics

- ✅ **Agent Responsiveness:** <200ms response to user queries during background tasks
- ✅ **Transparency:** 100% of long-running commands show status updates
- ✅ **Parallel Execution:** 3+ simultaneous background tasks supported
- ✅ **User Confidence:** Clear "what/why/when" for every operation
- ✅ **No Silent Failures:** All errors reported with context and suggestions

#### Benefits

1. **🚀 Perceived Speed:** Agent feels 10x faster (never "frozen")
2. **💬 Continuous Availability:** Always ready for new requests
3. **📊 Visibility:** Users see exactly what's happening
4. **🔄 Parallel Work:** Build + test + generate simultaneously
5. **🧠 Smart Resource Use:** Efficient terminal pool management
6. **🎯 Trust:** Transparency builds user confidence in AI agent

**Status:** 🔴 **NOT YET IMPLEMENTED** - Critical MVP feature (P0)
**Priority:** ⚡ **P0 - MVP BLOCKER**
**Effort:** ~6-8 hours implementation + testing
**Dependencies:** Existing terminal infrastructure (already complete)

---

#### 3.2 Git & Version Control

**Protocol:** **MCP** (Primary via @modelcontextprotocol/server-git) | Builtin (Fallback via git2-rs)
**Rationale:** Well-maintained MCP Git server exists. Platform-specific operations benefit from MCP. Builtin fallback ensures reliability.

**Overview:**

Git integration allows users to connect and authenticate once through natural language chat interaction. Agent helps set up Git configuration, handle authentication, and manage version control operations seamlessly.

**Chat-Based Git Setup (MVP - Priority Feature):**

Users connect to Git by typing in the chat panel. Agent guides them through authentication and setup:

**User Experience:**

```
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
```

**One-Time Authentication:**

- User authenticates once per machine
- Credentials securely stored in system keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service)
- Automatic token refresh for GitHub (when supported)
- No re-authentication needed across sessions

**Agent-Assisted Setup:**

Agent handles all Git configuration:

- Detects if Git is installed (installs if missing on user consent)
- Configures user.name and user.email
- Sets up credential helper for token storage
- Tests connection and validates permissions
- Guides user through any errors with clear instructions

**Implementation:**

| Capability               | Tool/Terminal | MVP Status | Purpose                                  | Implementation                                      |
| ------------------------ | ------------- | ---------- | ---------------------------------------- | --------------------------------------------------- |
| `git_setup`            | Tool          | ✅ MVP     | Chat-based Git configuration & auth      | **NEW [Builtin]** `agent/git_setup.rs`      |
| `git_authenticate`     | Tool          | ✅ MVP     | Store credentials securely               | **NEW [Builtin]** Uses system keychain        |
| `git_test_connection`  | Tool          | ✅ MVP     | Validate authentication works            | **NEW [Builtin]** Test fetch/push permissions |
| `git_status`           | Terminal      | ✅ DONE    | Current state                            | Via terminal commands**[MCP/Builtin]**              |
| `git_diff`             | Terminal      | ✅ DONE    | Changes (staged, unstaged, between refs) | Via terminal commands**[MCP/Builtin]**              |
| `git_log`              | Terminal      | ✅ DONE    | Commit history                           | Via terminal commands**[MCP/Builtin]**              |
| `git_blame`            | Terminal      | ✅ DONE    | Line-by-line attribution                 | Via terminal commands**[MCP/Builtin]**              |
| `git_commit`           | Terminal      | ✅ DONE    | Create commit with auto-messages         | Via terminal commands**[MCP/Builtin]**              |
| `git_push`             | Terminal      | ✅ DONE    | Push commits to remote                   | Via terminal commands**[MCP/Builtin]**              |
| `git_pull`             | Terminal      | ✅ DONE    | Pull latest changes                      | Via terminal commands**[MCP/Builtin]**              |
| `git_branch`           | Terminal      | ✅ DONE    | Create/switch/list branches              | Via terminal commands**[MCP/Builtin]**              |
| `git_checkout`         | Terminal      | ✅ DONE    | Checkout files/branches                  | Via terminal commands**[MCP/Builtin]**              |
| `git_merge`            | Terminal      | ✅ DONE    | Merge branches                           | Via terminal commands**[MCP/Builtin]**              |
| `git_stash`            | Terminal      | ✅ DONE    | Stash/pop changes                        | Via terminal commands**[MCP/Builtin]**              |
| `git_reset`            | Terminal      | ✅ DONE    | Undo changes                             | Via terminal commands**[MCP/Builtin]**              |
| `git_clone`            | Terminal      | ✅ DONE    | Clone repository                         | Via terminal commands**[MCP/Builtin]**              |
| `git_resolve_conflict` | Tool          | 🔴 TODO    | AI-powered conflict resolution           | **NEW [Builtin]** (Post-MVP)                  |

**Implementation:**

**Git Setup Flow** (`agent/git_setup.rs` - NEW):

1. **Check Git Installation:**

   - Detect if Git is installed on the system
   - Offer to install if missing (with user consent)
2. **Authentication Detection:**

   - Check if credentials already exist in system keychain
   - Test existing connection before prompting for new credentials
3. **Guide User Through Auth:**

   - Parse user intent ("connect to github", "setup git")
   - Present available authentication methods (PAT, SSH, GitHub CLI)
   - Validate token format before attempting connection
   - Test connection with provided credentials
4. **Store Credentials Securely:**

   - Use platform-specific secure storage:
     - macOS: Keychain
     - Windows: Credential Manager
     - Linux: libsecret (GNOME Keyring/KDE Wallet)
   - Encrypt tokens before storage
5. **Configure Git:**

   - Set user.name and user.email globally
   - Configure platform-specific credential helper
   - Verify configuration is working
6. **Success Confirmation:**

   - Report successful authentication with username
   - Confirm email configuration
   - Inform user of available operations

**Status:** ✅ Git operations work via terminal (100% functional)
**MVP Addition:** ✅ Chat-based authentication and setup (Priority Feature)
**Future:** 🔴 AI-powered conflict resolution tool (Post-MVP)

---

### 🆕 Dependency Intelligence & Environment Management (MVP Critical - Phase 1)

**Problem:** Agent installs packages blindly without validation, pollutes system Python, breaks environments with conflicts, has no rollback, and doesn't track package→file→function dependencies in GNN.

**Solution:** Comprehensive dependency intelligence with dry-run validation, mandatory .venv isolation, GNN integration, conflict detection, automatic rollback, and transparent multi-project environment management.

#### Critical Gaps & Solutions

| #            | Gap                                 | Current Behavior                     | New Behavior                                            | Priority      |
| ------------ | ----------------------------------- | ------------------------------------ | ------------------------------------------------------- | ------------- |
| **1**  | No dry-run validation               | Installs blindly, breaks production  | Validate in temp venv first, detect conflicts           | 🔴 P0 BLOCKER |
| **2**  | No .venv enforcement                | Pollutes system Python               | Always use `.venv/bin/python`, block global access    | 🔴 P0 BLOCKER |
| **3**  | No package→file→function tracking | Can't assess impact of changes       | GNN tracks which files/functions use each package       | 🔴 P0 BLOCKER |
| **4**  | No conflict detection               | Silent breakage, mysterious failures | Detect conflicts before installation, suggest fixes     | 🔴 P0 BLOCKER |
| **5**  | No rollback mechanism               | Broken state on failure              | Snapshot + restore on failure                           | 🟡 P1 HIGH    |
| **6**  | No environment validation           | Commands fail with unclear errors    | Pre-execution checks (venv active? packages installed?) | 🟡 P1 HIGH    |
| **7**  | No conflict resolution AI           | User must fix manually               | Agent suggests resolutions with risk assessment         | 🟡 P1 HIGH    |
| **8**  | No multi-project isolation          | Cross-contamination between projects | Each project has isolated .venv, auto-switch            | 🟡 P1 HIGH    |
| **9**  | No usage verification               | Unused dependencies pile up          | Track if package is actually used after install         | 🟢 P2 MEDIUM  |
| **10** | No dependency caching               | Slow, wastes bandwidth               | Local cache for faster installs                         | 🟢 P2 MEDIUM  |

---

#### P0 Feature 1: Dry-Run Validation (BLOCKER)

**Principle:** NEVER install without validating first. 100% accurate GNN dependency graph.

**Implementation:** `agent/dependency_validator.rs` (NEW, ~300 lines)

**Dependency Validator Component Requirements:**

The system must implement a DependencyValidator that validates package installations before applying them to the project environment.

**Validator Configuration:**

The DependencyValidator must be configured with:

- Workspace directory path for context
- Access to the GNN (Graph Neural Network) engine for impact analysis
- Temporary virtual environment cache path for dry-run validations

**Pre-Installation Validation Requirements:**

When validating a package before installation, the system must execute the following validation workflow:

**Step 1: Current Environment Analysis**

- Must retrieve complete list of all currently installed packages with exact versions

**Step 2: Impact Analysis via GNN**

- Must query the GNN to identify all files in the codebase that import or use the target package
- Must query the GNN to identify all functions that call or depend on the target package
- Must use thread-safe access to the GNN (lock/unlock pattern)

**Step 3: Isolated Test Environment Setup**

- Must create a temporary virtual environment separate from the project's main environment
- Must use dedicated cache directory for temporary environments

**Step 4: Environment Cloning**

- Must clone all currently installed packages with exact versions to the temporary environment
- Must ensure the temporary environment matches the current project environment exactly

**Step 5: Dry-Run Installation**

- Must execute pip installation in the temporary environment with dry-run flag
- Must use pip's --report option to generate JSON output of dependency resolution
- Must capture the complete dependency tree that would be installed

**Step 6: Dependency Resolution Parsing**

- Must parse the JSON report from pip to extract all packages that would be installed
- Must capture exact versions for each package in the resolution
- Must identify the dependency chain (which package requires which)

**Step 7: Conflict Detection**

- Must compare resolved packages against currently installed packages
- Must identify version conflicts (same package, different versions)
- Must detect dependency requirement conflicts (package A needs package B v2, but v1 is installed)
- For each conflict found, must record:
  - Package name experiencing the conflict
  - Current version in the environment
  - Requested version from the new installation
  - Reason explaining why the conflict exists
  - Severity level based on version change magnitude

**Step 8: Impact Assessment**

- Must calculate the number of files that would be affected by the installation
- Must calculate the number of functions that would be affected
- Must assess overall risk level based on:
  - Number and severity of conflicts detected
  - Number of affected files
  - Type of version changes (major vs minor vs patch)

**Step 9: Cleanup**

- Must delete the temporary virtual environment after validation completes
- Must ensure no temporary artifacts remain

**Step 10: Report Generation**

- Must compile a comprehensive ValidationReport containing all analysis results

**Validation Report Structure:**

The ValidationReport must include:

- Package name being validated
- Requested version (optional)
- Safety determination (boolean: safe to install or not)
- Complete list of all conflicts detected
- Impact analysis with affected files and functions
- Complete list of file paths affected
- Complete list of function names affected
- Human-readable recommendation explaining the validation result

**Dependency Conflict Data:**

Each DependencyConflict must capture:

- Package name where conflict occurs
- Current version installed
- Requested version from new installation
- Human-readable reason explaining the conflict
- Severity classification (Low, Medium, High, or Critical)

**Conflict Severity Classification:**

The system must classify conflicts as:

- **Low:** Minor version bump (0.0.x change)
- **Medium:** Major version change (x.0.0 change)
- **High:** Incompatible version constraints
- **Critical:** Will definitely break existing code

**Conflict Detection Algorithm:**

When detecting conflicts, the system must:

**For Each Package in Resolution:**

- Compare against each currently installed package
- If package names match but versions differ:
  - Create conflict entry identifying the version mismatch
  - Set severity to High
  - Explain which package requires which version

**For Each Dependency Requirement:**

- Check if the requirement conflicts with any currently installed package
- If conflict detected, create conflict entry with full explanation

**User Feedback During Validation:**

The system must provide transparent feedback:

- Display validation start message with package name and version
- Explain validation strategy (dry-run in isolated environment)
- Display validation success/failure prominently (✅ or ❌)
- Show impact analysis results:
  - Number of files affected
  - Number of functions affected
  - List of conflicts (if any)
  - Overall risk level (LOW, MEDIUM, HIGH, CRITICAL)
- Provide recommendation for next action
- Explain whether it's safe to proceed with installation

**Agent Transparency (Dry-Run):**

```
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
```

**Conflict Detection Example:**

```
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
```

---

#### P0 Feature 2: Mandatory .venv Isolation (BLOCKER)

**Principle:** NEVER pollute system Python. ALWAYS work in `.venv`.

**Implementation:** `agent/python_environment.rs` (NEW, ~250 lines)

**Python Environment Component Requirements:**

The system must implement a PythonEnvironment component that enforces strict virtual environment isolation for all Python operations.

**Environment Configuration Data:**

The PythonEnvironment must maintain:

- Virtual environment path (always workspace/.venv)
- Python executable path (.venv/bin/python on Unix, .venv/Scripts/python.exe on Windows)
- Pip executable path (.venv/bin/pip on Unix, .venv/Scripts/pip.exe on Windows)
- Activation status flag indicating whether environment is active
- Python version string (e.g., "3.11.5")
- Creation timestamp for tracking environment age
- List of all installed packages with versions

**Virtual Environment Initialization Requirements:**

When ensuring a virtual environment exists, the system must:

**Step 1: Check for Existing Environment**

- Must construct the .venv path as workspace_directory/.venv
- Must check if the .venv directory already exists

**Step 2A: If Environment Does Not Exist**

- Must automatically create a new virtual environment
- Must call the venv creation procedure

**Step 2B: If Environment Exists**

- Must verify the existing environment is valid and uncorrupted
- Must call the venv verification procedure

**Step 3: Detect Executables (Cross-Platform)**

- On Windows:
  - Python executable must be at .venv/Scripts/python.exe
  - Pip executable must be at .venv/Scripts/pip.exe
- On Unix/macOS:
  - Python executable must be at .venv/bin/python
  - Pip executable must be at .venv/bin/pip

**Step 4: Verify Executable Existence**

- Must check that the Python executable file exists
- If not found, must return error: ".venv is invalid: {path} not found"

**Step 5: Version Detection**

- Must execute the Python executable to detect version
- Must parse and store the version string

**Step 6: Package Inventory**

- Must query pip to list all installed packages
- Must parse and store the complete package list with versions

**Step 7: Environment Object Creation**

- Must set activation status to true
- Must record current timestamp as creation time
- Must return initialized PythonEnvironment object

**Virtual Environment Creation Requirements:**

When creating a new virtual environment, the system must:

**User Notification:**

- Must display message: "🔧 Creating .venv in {workspace}..."

**Creation Execution:**

- Must use system Python3 to create the virtual environment
- Must execute command: "python3 -m venv .venv"
- Must run command in the workspace directory
- Must handle execution errors with descriptive error messages

**Success Verification:**

- Must check command exit status
- If command fails:
  - Must return error with stderr output
  - Must format error: "Failed to create .venv: {error_message}"
- If command succeeds:
  - Must display: "✅ Created .venv successfully!"

**Virtual Environment Verification Requirements:**

When verifying an existing virtual environment, the system must:

**Required File Check:**

- On Windows, must verify existence of:
  - Scripts/python.exe
  - Scripts/pip.exe
  - pyvenv.cfg
- On Unix/macOS, must verify existence of:
  - bin/python
  - bin/pip
  - pyvenv.cfg

**Corruption Detection:**

- For each required file:
  - Must check if the file exists within .venv
  - If any file is missing:
    - Must return error: ".venv is corrupted: {file} not found. Please delete .venv and I'll recreate it."

**Isolation Enforcement Requirements:**

Before executing any Python code, the system must:

**Safety Block:**

- Must check if the environment is activated (is_activated flag)
- If not activated, must block execution and return error:
  - "❌ SAFETY BLOCK: Cannot execute - .venv not activated!"
  - "💡 This prevents pollution of system Python."
  - "🔧 Please ensure .venv is properly activated."
- If activated, must allow execution to proceed

**Command Access Methods:**

The PythonEnvironment must provide:

- Method to get Python executable path for command execution
- Method to get Pip executable path for package management

**Terminal Executor Integration Requirements:**

The TerminalExecutor must enforce virtual environment usage for all Python operations:

**Python Script Execution:**

When executing Python scripts or commands, the system must:

1. Ensure .venv exists and is valid in the workspace
2. Enforce isolation safety check (block if not isolated)
3. Execute using .venv Python executable (never global Python)
4. Pass script or command using "-c" flag for inline execution

**Package Installation:**

When installing Python packages, the system must:

1. Ensure .venv exists and is valid
2. Enforce isolation safety check
3. Build pip command with "install" argument
4. Add all requested package names to arguments
5. Execute pip command using .venv pip executable

**User Feedback - Environment Check:**

When checking environment status, the system must display:

```
🔧 Ensuring Python environment is isolated...

   ✅ Found .venv at: /workspace/.venv
   🐍 Python version: 3.11.5
   📦 Installed packages: 47
   ⏱️  Created: 2 days ago

   💡 All Python commands will use .venv (never system Python)
   🛡️  Your system Python is protected!
```

**User Feedback - Auto-Creation:**

When creating a new environment, the system must display:

```
🔍 .venv not found in workspace
🔧 Creating isolated Python environment...

   ▶️  Running: python3 -m venv .venv
   ⏳ [2s] Creating virtual environment...
   ⏳ [4s] Installing pip, setuptools...
   ✅ Created .venv successfully!

   🐍 Python version: 3.11.5
   📦 Ready to install packages

   💡 All future Python commands will use this isolated environment
```

---

#### P0 Feature 3: GNN Tech Stack Dependency Tracking (BLOCKER)

**Principle:** GNN must track **version-level dependencies** (not just package names) for accurate impact analysis and conflict detection. Each version is a separate node.

**Post-MVP Priorities (Dependency Tracking Extensions):**

- **P1 Post-MVP:** External API Tracking (API endpoints as nodes, track API calls across services)
- **P1 Post-MVP:** User → File Tracking (team collaboration, ownership, expertise mapping)
- **P2 Post-MVP:** Method Chain Tracking (track `df.groupby().agg()` level granularity)
- **P2 Post-MVP:** Function → Package Function (which specific package functions are used)

**Critical: Version-Level Tracking**

- ❌ **WRONG:** Track "numpy" as single node → cannot detect version conflicts
- ✅ **CORRECT:** Track "numpy==1.24.0" and "numpy==1.26.0" as separate nodes → detect incompatibilities
- Track **EXACT versions** for all packages (numpy==1.26.0, pandas==2.1.0, not just "numpy", "pandas")
- Track **version requirements** for dependencies (requires: "numpy>=1.26,<2.0")
- Track **version history** (upgraded from 1.24.0 → 1.26.0 on date X)
- Enable queries: "Which files depend on numpy 1.24 specifically?" vs "Which files use any numpy?"

**Package Tracking Scope (MVP):**

- ✅ **File → Package@Version** (track what files import what packages)
- ✅ **Package → Package** (transitive dependencies from lock files)
- ✅ **Nested function tracking** (numpy.random.normal, not just numpy.array)
- ✅ **Version conflict detection** (simple semver-based)
- ✅ **Breaking change warnings** (major version bumps)
- ✅ **Query: "What breaks if I upgrade X?"**

**Metadata Sources (Priority):**

1. **Lock files first** (package-lock.json, Cargo.lock, poetry.lock) - exact versions
2. **Manifest files** (requirements.txt, package.json, Cargo.toml) - if no lock file
3. **Skip runtime inspection** (pip show, npm list) for MVP - adds complexity

**Function Tracking Granularity:**

- **MVP:** Nested attributes (numpy.random.normal, pandas.DataFrame.groupby)
- **Deferred:** Method chains (df.groupby().agg().reset_index()) - P2 Post-MVP

**Cross-Language Strategy:**

- **MVP:** Separate graphs per language (Python graph, JavaScript graph, Rust graph)
- **Post-MVP:** Unified graph with API nodes connecting languages

**Update Triggers:**

- Project open (initial scan)
- Lock file changes (watch requirements.txt, package-lock.json, Cargo.toml)
- Manual refresh (user-requested)
- Pre-code generation (ensure fresh context)

**Implementation:** Update `gnn/mod.rs` and `gnn/graph.rs`

**GNN Tech Stack Node Data Requirements:**

The GNN must track detailed package information in TechStackNode data structures containing:

- Package name (e.g., "numpy")
- Exact version string (e.g., "1.26.0" - EXACT version, not range)
- Programming language (Python, JavaScript, Rust, etc.)
- Installation timestamp
- List of files that use this package
- List of specific functions used from the package (e.g., "np.array", "np.mean")
- List of packages this version conflicts with (package name and version tuples)
- List of package requirements with version specifications
- Version change history tracking upgrades and downgrades

**Package Requirement Data:**

Each package requirement must capture:

- Required package name
- Version specification string (e.g., ">=1.24,<2.0" - PRECISE requirement)
- Optional flag indicating if the dependency is required or optional

**Version Change History Data:**

Each version change entry must record:

- Source version (e.g., "1.24.0")
- Target version (e.g., "1.26.0")
- Change timestamp
- Reason for the change (e.g., "Dependency conflict resolution")

**GNN Edge Type Requirements:**

The GNN must support the following edge types for tech stack tracking:

- **Import:** Code imports a module
- **Call:** Code calls a function
- **Inherit:** Class inherits from another class
- **Uses:** File uses a specific package version
- **Requires:** Package requires another package
- **ConflictsWith:** Package version conflicts with another package version

**Package Usage Tracking Requirements:**

When adding package usage to the GNN, the system must:

**Step 1: Version-Specific Node Creation**

- Must construct package node ID in format: "pkg:{package_name}:{exact_version}"
  - Example: "pkg:numpy:1.26.0"
- Must check if this exact version node already exists in the graph
- If node doesn't exist:
  - Must create new CodeNode with unique package version ID
  - Must set node name as "{package}=={version}" (e.g., "numpy==1.26.0")
  - Must initialize TechStackNode with:
    - Package name
    - Exact version (must be precise version like "1.26.0", not range like ">=1.26")
    - Programming language
    - Current timestamp as installation date
    - Empty lists for files, functions, conflicts, requirements, history
    - Provided list of functions used

**Step 2: File-to-Package Edge Creation**

- Must get the file node ID for the source file
- Must add "Uses" edge from file node to package version node
- Edge direction: File → Uses → Package@Version

**Step 3: Function Usage Tracking**

- For each function used from the package:
  - Must construct function node ID: "{pkg_node_id}::{function_name}"
    - Example: "pkg:numpy:1.26.0::np.array"
  - Must track specific function calls (e.g., "np.array", "np.mean")

**Package Query Requirements:**

**Specific Version Query:**

- When querying files using a specific package version:
  - Must construct package node ID: "pkg:{package}:{version}"
  - Must return list of file paths that have "Uses" edges to this exact version node

**Any Version Query:**

- When querying files using any version of a package:
  - Must filter all graph nodes to find TechStack nodes
  - For each node, check if package name matches (ignore version)
  - Must collect all matching package nodes regardless of version
  - For each package version found:
    - Must get all incoming edges
    - Must filter for "Uses" edge type
    - Must extract source file paths from each "Uses" edge
  - Must return aggregated list of all files using any version

**Version Update Tracking:**

- When updating a package from one version to another:
  - Must construct old and new package node IDs
  - Must locate the new version node in graph
  - Must add version change entry to new node's history:
    - Record source version
    - Record target version
    - Record current timestamp
    - Record reason for change
  - Must migrate all edges from old version node to new version node
  - Must preserve all file usage relationships

**Function Usage Query:**

- When querying which package functions are used:
  - Must analyze all "Call" edges targeting package function nodes
  - Must extract function names (e.g., "np.array", "np.mean", "pd.DataFrame")
  - Must return list of all functions actually invoked in codebase

**Post-Installation GNN Update:**

- After package installation, the system must:
  - Parse the installed package metadata
  - Add package node with exact version
  - For each dependency in metadata:
    - Add "Requires" edge from package to dependency
  - Scan entire workspace for package usage
  - Update package version information
  - Record installation timestamp

**Agent Transparency (GNN Integration):**

````
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

   💡 GNN now has 100% accurate version-level tech stack dependency information
```---

#### P0 Feature 4: Conflict Detection & Resolution (BLOCKER)

**Principle:** Detect conflicts BEFORE installation. Suggest intelligent fixes with risk assessment.

**Implementation:** `agent/conflict_resolver.rs` (NEW, ~200 lines)

**Conflict Resolution System Requirements:**

The system must implement a ConflictResolver component with access to the GNN engine for dependency analysis.

**Resolution Suggestion Requirements:**

When a dependency conflict is detected, the system must analyze the conflict and impact to generate multiple resolution options:

**Option 1: Upgrade Dependency Strategy**
- Must check if upgrading the conflicting dependency is safe based on version compatibility
- If safe, must create an upgrade resolution option with:
  - Clear strategy description (which package to upgrade and to which version)
  - Exact pip install command with version specified
  - Risk assessment based on version changes and impact analysis
  - Detailed explanation including current version, target version, and any breaking changes
  - Estimated time to complete the upgrade (typically 10 seconds)

**Option 2: Downgrade Requesting Package Strategy**
- Must search for compatible versions of the requesting package that work with current environment
- If compatible version found, must create a downgrade resolution option with:
  - Strategy description indicating use of older compatible version
  - Pip install command with version constraint
  - Risk level assessment (typically Medium)
  - Explanation of the compatibility tradeoff
  - Estimated time to complete (typically 15 seconds)

**Option 3: Environment Isolation Strategy**
- Must always provide an isolation option as a safe fallback with:
  - Strategy description for creating separate virtual environment
  - List of required commands including venv creation and code restructuring notes
  - Risk level assessment (Low - safest option)
  - Explanation that existing code remains unchanged
  - Realistic time estimate including restructuring effort (typically 5 minutes)

**Resolution Option Prioritization:**
- Must sort all resolution options by risk level (lowest risk first)
- Must return the sorted list of options to the user for selection

**Risk Assessment Requirements:**

The system must assess upgrade risk based on semantic versioning and impact analysis:

**Version Parsing:**
- Must parse current version and requested version strings
- Must handle parsing failures gracefully (treat as high risk if unparseable)

**Major Version Change Assessment:**
- If requested version has higher major version number than current
- Must classify as High risk by default
- Must upgrade to Critical risk if more than 10 files are affected
- Indicates breaking changes likely

**Minor Version Change Assessment:**
- If requested version has higher minor version number (same major version)
- Must classify as Medium risk by default
- Must downgrade to Low risk if fewer than 20 functions are affected
- Indicates feature additions with potential minor breaking changes

**Patch Version Change Assessment:**
- If only patch version increases (same major and minor versions)
- Must classify as Low risk
- Indicates bug fixes with minimal breaking change likelihood

**Unknown Version Format:**
- If version parsing fails for either current or requested version
- Must classify as High risk (be cautious when version format unknown)

**Resolution Option Data Structure Requirements:**

Each resolution option must include:
- Strategy name/description (human-readable summary)
- List of commands to execute
- Risk classification (Low, Medium, High, or Critical)
- Detailed explanation of what will happen and why
- Estimated time duration for completion

**Risk Level Definitions:**
- Low: Backward compatible changes, patch updates only
- Medium: Minor version changes with limited scope of impact
- High: Major version changes with significant potential impact
- Critical: Breaking changes affecting many files (>10) across the codebase`

**Agent Transparency (Conflict Resolution):**

```
❌ Dependency conflict detected! Let me suggest solutions...

   Conflict: pandas 2.1.0 requires numpy>=1.26.0
   Current: numpy 1.24.0
   Impact: 8 files, 25 functions affected

   💡 Resolution Options (sorted by safety):

   1. [RECOMMENDED] Upgrade numpy (RISK: LOW)
      Command: pip install numpy==1.26.0
      Reason: Backward compatible, patch release
      Impact: Minimal - numpy 1.26 is stable
      Time: ~10 seconds
      ✅ I recommend this option

   2. Use pandas 2.0.x instead (RISK: MEDIUM)
      Command: pip install pandas<2.1
      Reason: Compatible with current numpy
      Impact: You may lose pandas 2.1 features
      Time: ~15 seconds
      ⚠️  You'll miss out on newest pandas features

   3. Isolate in separate venv (RISK: LOW)
      Commands: Create workspace/module-venv/.venv
      Reason: Zero impact on existing code
      Impact: Requires code restructuring
      Time: ~5 minutes
      💡 Best for complex conflicts

   Which option would you like? (1/2/3)
```

---

#### P1 Feature 5: Automatic Rollback on Failure (HIGH)

**Implementation:** `agent/environment_snapshot.rs` (NEW, ~150 lines)

**Environment Snapshot Requirements:**

The system must implement an EnvironmentSnapshot component with the following capabilities:

**Snapshot Creation:**
- Must capture the current state of the virtual environment at a specific point in time
- Must store the virtual environment path for reference
- Must generate a unique snapshot identifier using Unix timestamp format ("snapshot-{timestamp}")
- Must record the exact timestamp when the snapshot was created
- Must execute `pip freeze` command to get complete list of installed packages with versions
- Must parse the pip freeze output into a structured list of installed packages
- Must store both the parsed package list and raw pip freeze output for redundancy

**Snapshot Restoration:**
- Must support restoring the environment to a previously captured state
- Must uninstall all currently installed packages that weren't in the snapshot
- Must reinstall all packages from the snapshot with exact versions
- Must verify the integrity of the restored environment matches the snapshot
- Must provide user feedback during the restoration process

**Rollback Integration Requirements:**

The DependencyInstaller must support automatic rollback on installation failure:

**Pre-Installation:**
- Must ensure a valid virtual environment exists in the workspace
- Must create a complete environment snapshot before attempting any package installation
- Must display snapshot confirmation message to user with snapshot identifier

**Installation Attempt:**
- Must attempt to install the requested packages
- Must track whether installation succeeds or fails

**Success Path:**
- If installation succeeds, must display success confirmation
- Must return successful installation result to caller

**Failure Path:**
- If installation fails, must capture the error message
- Must display installation failure message with error details
- Must automatically trigger environment restoration using the pre-installation snapshot
- Must display rollback confirmation after successful restoration
- Must return error result indicating both the installation failure and successful rollback
- Must ensure no partial installations remain in the environment

---

#### P1 Feature 6: Pre-Execution Environment Validation (HIGH)

**Implementation:** `agent/environment_validator.rs` (NEW, ~200 lines)

**Environment Validator Component Requirements:**

The system must implement an EnvironmentValidator that performs comprehensive validation checks before executing any code or commands.

**Validator Configuration Requirements:**

The EnvironmentValidator must be configured with:
- Workspace directory path for context-aware validation
- Access to the GNN (Graph Neural Network) engine for dependency analysis
- Thread-safe concurrent access to the GNN using Arc<Mutex>

**Validation Results Data Structure:**

The validation result must include:
- Overall validity status (boolean indicating if all checks passed)
- List of all individual validation checks performed
- List of missing requirements identified during validation
- List of actionable suggestions for resolving validation failures

**Individual Check Data Structure:**

Each validation check must capture:
- Name or description of the check being performed
- Pass/fail status of the check
- Human-readable message explaining the check result

**Pre-Execution Validation Requirements:**

Before executing any command, the system must perform the following validation sequence:

**1. Virtual Environment Activation Check:**
- Must verify that a Python virtual environment (.venv) is currently active
- Must record check result with descriptive name
- If check fails:
  - Must add ".venv activation" to missing requirements list
  - Must suggest the command: "Run: source .venv/bin/activate"
  - Must display failure message to user

**2. Required Package Installation Check:**
- Must analyze the command to detect all required Python packages using the GNN
- For each detected required package:
  - Must check if the package is currently installed in the active environment
  - Must create a validation check entry with the package name
  - If package is installed:
    - Must mark check as passed
    - Must display success message: "✅ {package} is installed"
  - If package is not installed:
    - Must mark check as failed
    - Must display failure message: "❌ {package} is NOT installed"
    - Must add package to missing requirements list
    - Must suggest installation command: "pip install {package}"

**3. Environment Variable Check (To Be Implemented):**
- Must verify that all required environment variables are set
- Must check for API keys, credentials, and configuration values
- Must suggest how to set missing environment variables

**4. Docker Service Check (If Needed):**
- Must detect if the command requires Docker
- Must verify Docker daemon is running and accessible
- Must suggest starting Docker if not running

**5. Port Availability Check (If Server/Service):**
- Must detect if the command will start a server or service
- Must check if required ports are available (not already in use)
- Must suggest which processes to stop if ports are occupied

**Validation Result Composition:**

After performing all checks, the system must:
- Set overall validity status to true only if all checks passed (no missing requirements)
- Include complete list of all validation checks performed
- Include complete list of all missing requirements
- Include complete list of all actionable suggestions for resolution
- Return the complete validation result to the caller

**User Feedback Requirements:**

The system must provide clear feedback to users:
- Display each validation check as it's performed
- Use visual indicators (✅ for pass, ❌ for fail)
- Present actionable suggestions when validation fails
- Prevent execution if validation fails until issues are resolved

---

#### P1 Feature 7: Conflict Resolution AI (HIGH)

Already covered in P0 Feature 4 - `ConflictResolver` with intelligent suggestions.

---

#### P1 Feature 8: Multi-Project Isolation (HIGH - Enterprise Critical)

**Principle:** Each project has isolated .venv. Never share dependencies. Auto-switch based on active project.

**Implementation:** `agent/project_environment_manager.rs` (NEW, ~250 lines)

**Project Environment Manager Requirements:**

The system must implement a ProjectEnvironmentManager component that maintains complete isolation between multiple projects.

**Project Environment Data Requirements:**

For each project, the system must maintain:
- Unique project identifier
- Workspace path (file system location)
- Isolated Python virtual environment (venv)
- Last active timestamp for tracking usage
- Cache of installed packages for quick reference

**Manager State Requirements:**

The ProjectEnvironmentManager must maintain:
- HashMap of all project environments indexed by project ID
- Currently active project identifier (optional, may be none)
- Global cache directory path (typically ~/.yantra/venv-cache) for shared resources

**Project Isolation Enforcement Requirements:**

When ensuring project isolation, the system must:

**New Project Setup:**
- Check if the project ID already has an associated environment
- If no environment exists:
  - Create a new isolated virtual environment in the project's workspace directory
  - Initialize a new ProjectEnvironment entry with all required metadata
  - Store the new environment in the projects HashMap
  - Set the last active timestamp to current time
  - Display confirmation message to user with project identifier
- If environment already exists, return reference to existing environment

**Project Switching Requirements:**

When switching from one project to another, the system must:

**Pre-Switch Actions:**
- Display switching message showing source and destination project names
- Update the source project's last active timestamp to current time
- Deactivate any active connections or resources for source project

**Post-Switch Actions:**
- Ensure the destination project has an isolated environment (create if needed)
- Set the destination project as the active project
- Display success confirmation with project name
- Display Python version being used in destination environment
- Display count of installed packages in destination environment

**Active Environment Access:**
- Must provide method to retrieve currently active project environment
- Must return error if no project is currently active
- Must return reference to the active ProjectEnvironment if available

**Cross-Project Isolation Guarantees:**

The system must ensure:
- Each project has completely separate package dependencies
- No package installed in one project affects another project
- No shared state between project environments
- Automatic environment switching when user changes active project
- Clear user feedback showing which environment is active

**Agent Transparency (Multi-Project):**

```
🔄 Switching projects...

   From: project-a (AI chatbot)
   └─ .venv: 47 packages, Python 3.11.5

   To: project-b (ML model)
   └─ .venv: 89 packages, Python 3.10.8

   ✅ Environment switched successfully!
   💡 Each project has isolated dependencies
   🛡️  No cross-contamination between projects
```

---

#### P2 Features (9-10) - Brief Overview

**9. Usage Verification:** Track if installed packages are actually imported/used. Suggest cleanup for unused deps.

**10. Dependency Caching:** Local cache (`~/.yantra/cache/pip`) for faster installs, less bandwidth.

---

#### Implementation Summary

| Component             | File                                   | Lines           | Priority  | Effort  |
| --------------------- | -------------------------------------- | --------------- | --------- | ------- |
| Dependency Validator  | `agent/dependency_validator.rs`        | ~300            | P0        | 4h      |
| Python Environment    | `agent/python_environment.rs`          | ~250            | P0        | 3h      |
| GNN Tech Stack        | Update `gnn/mod.rs`, `gnn/graph.rs`    | ~200            | P0        | 3h      |
| Conflict Resolver     | `agent/conflict_resolver.rs`           | ~200            | P0        | 2h      |
| Environment Snapshot  | `agent/environment_snapshot.rs`        | ~150            | P1        | 2h      |
| Environment Validator | `agent/environment_validator.rs`       | ~200            | P1        | 2h      |
| Project Env Manager   | `agent/project_environment_manager.rs` | ~250            | P1        | 3h      |
| **TOTAL**             | **7 files**                            | **~1550 lines** | **P0+P1** | **19h** |

**Status:** 🔴 **NOT YET IMPLEMENTED** - Critical MVP features (P0+P1)
**Priority:** ⚡ **P0 BLOCKERS + P1 HIGH** - Enterprise-grade dependency management
**Dependencies:** Existing GNN infrastructure, terminal executor

---

#### 3.3 Code Generation & Modification

**Protocol:** **Builtin** (Primary - Core Differentiator)
**Rationale:** Code generation orchestration with GNN, LLM, and validation pipeline is Yantra's core value. Must be under complete control.

| Capability          | Tool/Terminal | MVP Status | Purpose                     | Implementation                                          |
| ------------------- | ------------- | ---------- | --------------------------- | ------------------------------------------------------- |
| `generate_code`     | Tool          | ✅ DONE    | LLM-powered code generation | `llm/multi_llm_manager.rs` + orchestrator **[Builtin]** |
| `auto_correct_code` | Tool          | ✅ DONE    | Fix architecture deviations | `architecture/deviation_detector.rs` **[Builtin]**      |
| `refactor_code`     | Tool          | 🔴 TODO    | Automated refactoring       | **NEW [Builtin]** (Phase 3)                             |

**Status:** ✅ Code generation complete, 🔴 Advanced refactoring pending

#### 3.4 Testing Execution

**Protocol:** **Builtin** (Exclusive - Core to "Never Breaks" Guarantee)
**Rationale:** Testing is core to "never breaks" guarantee. Must integrate with GNN for affected test detection. Performance-critical.

| Capability          | Tool/Terminal | MVP Status | Purpose                             | Implementation                                       |
| ------------------- | ------------- | ---------- | ----------------------------------- | ---------------------------------------------------- |
| `test_run`          | Tool          | ✅ DONE    | Execute tests (file, suite, single) | `testing/test_generator.rs` + executor **[Builtin]** |
| `test_run_affected` | Tool          | 🔴 TODO    | Run tests for changed code only     | **NEW [Builtin]** (use GNN)                          |
| `test_coverage`     | Tool          | ✅ DONE    | Get coverage report                 | pytest-cov integration**[Builtin]**                  |
| `test_generate`     | Tool          | ✅ DONE    | Auto-generate test cases            | `testing/test_generator.rs` **[Builtin]**            |
| `test_debug`        | Tool          | 🔴 TODO    | Run test in debug mode              | **NEW [Builtin with DAP]** (Phase 2)                 |
| `test_watch`        | Tool          | 🔴 TODO    | Continuous test runner              | **NEW [Builtin]** (Phase 2)                          |
| `e2e_run`           | Tool          | 🔴 TODO    | Browser/integration tests           | **NEW [Builtin]** (CDP + Playwright)                 |

**Status:** ✅ Basic testing complete (6/6 features), 🔴 Advanced testing pending

#### 3.5 Build & Compilation

**Protocol:** **Builtin** (Primary via shell) | MCP (Optional for language-specific services)
**Rationale:** Build orchestration needs to coordinate with dep graph, testing, deployment. Shell execution via Builtin is sufficient.

| Capability          | Tool/Terminal | MVP Status | Purpose                     | Implementation                                              |
| ------------------- | ------------- | ---------- | --------------------------- | ----------------------------------------------------------- |
| `build_project`     | Terminal      | ✅ DONE    | Full build                  | Via terminal (`cargo build`, `npm run build`) **[Builtin]** |
| `build_incremental` | Terminal      | ✅ DONE    | Changed files only          | Via terminal**[Builtin]**                                   |
| `build_check`       | Terminal      | ✅ DONE    | Type-check without emitting | Via terminal**[Builtin]**                                   |
| `build_clean`       | Terminal      | ✅ DONE    | Clear artifacts             | Via terminal**[Builtin]**                                   |
| `lint_run`          | Tool          | ✅ DONE    | Run linters                 | Security scanner includes linting**[Builtin]**              |
| `lint_fix`          | Tool          | 🔴 TODO    | Auto-fix lint issues        | **NEW [Builtin]**                                           |
| `format_code`       | Terminal      | ✅ DONE    | Apply formatters            | Via terminal (`rustfmt`, `prettier`) **[Builtin]**          |

**Status:** ✅ Build and lint via terminal works well

#### 3.6 Package Management

**Protocol:** **Builtin** (Exclusive)
**Rationale:** Package operations need shell execution. Audit can integrate with MCP vulnerability databases, but orchestration is Builtin.

| Capability      | Tool/Terminal | MVP Status | Purpose                      | Implementation                                    |
| --------------- | ------------- | ---------- | ---------------------------- | ------------------------------------------------- |
| `pkg_install`   | Tool          | ✅ DONE    | Add dependency               | `agent/dependencies.rs` (429 lines) **[Builtin]** |
| `pkg_remove`    | Tool          | ✅ DONE    | Remove dependency            | `agent/dependencies.rs` **[Builtin]**             |
| `pkg_update`    | Tool          | ✅ DONE    | Update dependencies          | `agent/dependencies.rs` **[Builtin]**             |
| `pkg_list`      | Tool          | ✅ DONE    | List installed packages      | `agent/dependencies.rs` **[Builtin]**             |
| `pkg_audit`     | Tool          | ✅ DONE    | Security vulnerability check | `security/scanner.rs` **[Builtin]**               |
| `pkg_search`    | Tool          | 🔴 TODO    | Find packages in registry    | **NEW [Builtin]**                                 |
| `pkg_lock_sync` | Tool          | ✅ DONE    | Sync lockfile                | Via package manager commands**[Builtin]**         |

**Status:** ✅ **100% COMPLETE** - Package management fully implemented

#### 3.7 Deployment & Infrastructure

**Protocol:** **MCP** (Primary for platform-specific services) | Builtin (Container operations via shell)
**Rationale:** Platform-specific deployments (Railway, Vercel, AWS, GCP) benefit from MCP servers. Container operations via Docker CLI (Builtin).

| Capability          | Tool/Terminal | MVP Status | Purpose                               | Implementation                              |
| ------------------- | ------------- | ---------- | ------------------------------------- | ------------------------------------------- |
| `deploy_preview`    | Tool          | ✅ DONE    | Deploy to preview environment         | `agent/deployment.rs` (636 lines) **[MCP]** |
| `deploy_production` | Tool          | ✅ DONE    | Deploy to prod (with confirmation)    | `agent/deployment.rs` **[MCP]**             |
| `deploy_rollback`   | Tool          | ✅ DONE    | Revert deployment                     | `agent/deployment.rs` **[MCP]**             |
| `deploy_status`     | Tool          | ✅ DONE    | Check deployment state                | `agent/deployment.rs` **[MCP]**             |
| `deploy_logs`       | Tool          | ✅ DONE    | Fetch deployment logs                 | `agent/deployment.rs` **[MCP]**             |
| `infra_provision`   | Tool          | 🔴 TODO    | Create resources (Railway, AWS, etc.) | **NEW [MCP]** (Phase 2)                     |
| `container_build`   | Terminal      | ✅ DONE    | Build Docker image                    | Via `docker build` **[Builtin]**            |
| `container_run`     | Terminal      | ✅ DONE    | Run container locally                 | Via `docker run` **[Builtin]**              |

**MCP Servers:** Railway MCP (custom), Vercel MCP, AWS MCP, GCP/Azure community servers

**Status:** ✅ Deployment to Railway complete (5/5 features), 🔴 Multi-cloud pending

#### 3.8 Browser Automation (CDP)

**Protocol:** **Builtin** (Exclusive - Core to Browser Product)
**Rationale:** CDP integration is core to Yantra's browser validation. Must be fast, reliable, and under complete control.

| Capability               | Tool/Terminal | MVP Status | Purpose                 | Implementation                               |
| ------------------------ | ------------- | ---------- | ----------------------- | -------------------------------------------- |
| `browser_launch`         | Tool          | 🟡 PARTIAL | Start browser instance  | `browser/cdp.rs` (placeholder) **[Builtin]** |
| `browser_navigate`       | Tool          | 🟡 PARTIAL | Go to URL               | `browser/cdp.rs` **[Builtin]**               |
| `browser_click`          | Tool          | 🔴 TODO    | Click element           | **NEW [Builtin]**                            |
| `browser_type`           | Tool          | 🔴 TODO    | Input text              | **NEW [Builtin]**                            |
| `browser_screenshot`     | Tool          | 🔴 TODO    | Capture screen          | **NEW [Builtin]**                            |
| `browser_select_element` | Tool          | 🔴 TODO    | Visual element picker   | **NEW [Builtin]** (Post-MVP)                 |
| `browser_evaluate`       | Tool          | 🔴 TODO    | Run JS in page context  | **NEW [Builtin]**                            |
| `browser_network`        | Tool          | 🔴 TODO    | Intercept/mock requests | **NEW [Builtin]** (Post-MVP)                 |
| `browser_console`        | Tool          | 🔴 TODO    | Get console logs        | **NEW [Builtin]**                            |

**Status:** 🟡 25% complete (2/8 features) - CDP placeholder needs full implementation

#### 3.9 HTTP & API Execution

**Protocol:** **Builtin** (Primary) | MCP (Fallback for specific API integrations)
**Rationale:** HTTP client with circuit breaker, retry, and rate limiting is infrastructure-level. Must be under control for reliability.

| Capability          | Tool/Terminal | MVP Status | Purpose                                    | Implementation                               |
| ------------------- | ------------- | ---------- | ------------------------------------------ | -------------------------------------------- |
| `http_request`      | **TOOL**      | 🔴 TODO    | Make HTTP calls with retry/circuit breaker | **NEW [Builtin]** `agent/http_client/mod.rs` |
| `websocket_connect` | Tool          | 🔴 TODO    | WebSocket client                           | **NEW [Builtin]** (Phase 2)                  |

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

**Protocol:** **DAP** (Debug Adapter Protocol - Exclusive)
**Rationale:** Debugging is handled by DAP (Debug Adapter Protocol), a separate protocol specifically for debugging operations.

| Capability         | Tool/Terminal | MVP Status | Purpose                    | Implementation          |
| ------------------ | ------------- | ---------- | -------------------------- | ----------------------- |
| `debug_start`      | Tool          | 🔴 TODO    | Launch debugger            | **NEW [DAP]** (Phase 2) |
| `debug_breakpoint` | Tool          | 🔴 TODO    | Set/remove breakpoints     | **NEW [DAP]**           |
| `debug_step`       | Tool          | 🔴 TODO    | Step over/into/out         | **NEW [DAP]**           |
| `debug_continue`   | Tool          | 🔴 TODO    | Resume execution           | **NEW [DAP]**           |
| `debug_evaluate`   | Tool          | 🔴 TODO    | Eval expression in context | **NEW [DAP]**           |
| `debug_stack`      | Tool          | 🔴 TODO    | Get call stack             | **NEW [DAP]**           |
| `debug_variables`  | Tool          | 🔴 TODO    | Inspect variables          | **NEW [DAP]**           |

**DAP Adapters:** Python (debugpy), Node.js (node-debug2), Rust (codelldb), Go (delve)

**Status:** 🔴 Not implemented (Phase 2 - Post-MVP)

#### 5.2 Documentation

**Protocol:** **Builtin** (Primary) | MCP (Fallback for external doc fetching)
**Rationale:** Documentation generation from code is core functionality. External doc fetching can use MCP for various sources.

| Capability      | Tool/Terminal | MVP Status | Purpose                                | Implementation                    |
| --------------- | ------------- | ---------- | -------------------------------------- | --------------------------------- |
| `docs_generate` | Tool          | ✅ DONE    | Generate from code (JSDoc, docstrings) | File Registry system**[Builtin]** |
| `docs_search`   | Tool          | 🔴 TODO    | Search project docs                    | **NEW [Builtin]**                 |
| `docs_external` | Tool          | 🔴 TODO    | Fetch library documentation            | **NEW [MCP]** (Phase 2)           |

**Status:** ✅ Basic documentation system complete

#### 5.3 Security

**Protocol:** **Builtin** (Primary for scanning) | MCP (Optional for vulnerability databases)
**Rationale:** Security scanning is core to "never breaks" guarantee. SAST must be under control. MCP can augment with external CVE databases.

| Capability         | Tool/Terminal | MVP Status | Purpose                      | Implementation                                  |
| ------------------ | ------------- | ---------- | ---------------------------- | ----------------------------------------------- |
| `security_scan`    | Tool          | ✅ DONE    | SAST analysis                | `security/scanner.rs` (512 lines) **[Builtin]** |
| `secrets_detect`   | Tool          | ✅ DONE    | Find exposed credentials     | Integrated in scanner**[Builtin]**              |
| `dependency_audit` | Tool          | ✅ DONE    | CVE check on packages        | Integrated in scanner**[Builtin]**              |
| `secrets_manager`  | Tool          | 🔴 TODO    | Encrypted credential storage | **NEW [Builtin]** `agent/secrets/vault.rs`      |

**Status:** ✅ Security scanning complete, 🔴 Secrets management pending

#### 5.4 Architecture Visualization

**Protocol:** **Builtin** (Exclusive - Core Differentiator)
**Rationale:** Architecture visualization from GNN is Yantra's core differentiator. Must be completely under control.

| Capability              | Tool/Terminal | MVP Status | Purpose                         | Implementation                                     |
| ----------------------- | ------------- | ---------- | ------------------------------- | -------------------------------------------------- |
| `arch_diagram_generate` | Tool          | ✅ DONE    | Create visual from code         | Architecture View System (16/16)**[Builtin]**      |
| `arch_validate`         | Tool          | ✅ DONE    | Check against constraints       | `architecture/deviation_detector.rs` **[Builtin]** |
| `arch_suggest`          | Tool          | ✅ DONE    | Recommend improvements          | Impact analysis**[Builtin]**                       |
| `arch_import`           | Tool          | ✅ DONE    | Import from MD/Mermaid/PlantUML | `project_initializer.rs` **[Builtin]**             |

**Status:** ✅ **100% COMPLETE** - Architecture system fully implemented

#### 5.5 Context & Memory

**Protocol:** **Builtin** (Exclusive - Core Differentiator)
**Rationale:** Context assembly with hierarchical GNN is core to Yantra's intelligence. Must be completely under control.

| Capability            | Tool/Terminal | MVP Status | Purpose                               | Implementation                         |
| --------------------- | ------------- | ---------- | ------------------------------------- | -------------------------------------- |
| `context_add`         | Tool          | ✅ DONE    | Add to agent's working memory         | State machine persistence**[Builtin]** |
| `context_search`      | Tool          | ✅ DONE    | Semantic search over codebase         | GNN semantic layer**[Builtin]**        |
| `context_summarize`   | Tool          | ✅ DONE    | Compress context for token efficiency | Hierarchical assembly**[Builtin]**     |
| `project_conventions` | Tool          | 🔴 TODO    | Get coding standards/patterns         | **NEW [Builtin]**                      |

**Status:** ✅ Context management mostly complete

---

## Agentic Capabilities Summary

### Implementation Status by Pillar

| Pillar         | Total Capabilities | Implemented | Pending | Completion % |
| -------------- | ------------------ | ----------- | ------- | ------------ |
| 🔍**PERCEIVE** | 47                 | 24          | 23      | 51%          |
| 🧠**REASON**   | 8                  | 8           | 0       | **100%** ✅  |
| ⚡**ACT**      | 56                 | 41          | 15      | 73%          |
| 🔄**LEARN**    | 7                  | 7           | 0       | **100%** ✅  |
| **TOTAL**      | **118**            | **80**      | **38**  | **68%**      |

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

## UTI Architecture Overview for agentic capabilities: Two Consumers, Two Protocols

4.1 Overview

The Unified Tool Interface (UTI) provides the agent with access to all capabilities required for autonomous development. This section consolidates the 21+ capability categories into a single table, grouped by protocol combinations and mapped to Preventive Development Cycle phases.

**Key Design Principles:**

- **Protocol Selection:** Builtin for core differentiators and performance-critical ops; MCP for ecosystem services; DAP for debugging
- **Fallback Strategy:** Every capability has a backup protocol where feasible
- **Phase Mapping:** Each capability is mapped to the PDC phase(s) where it's primarily used

## The Need for UTI: Abstraction Layer

The Unified Tool Interface (UTI) is Yantra's solution to this complexity. It provides a **single, consistent abstraction layer** that:

1. **Normalizes Tool Access:** All capabilities exposed through a uniform API
2. **Handles Protocol Differences:** Agent doesn't care if it's MCP, LSP, DAP, or Builtin
3. **Smart Routing:** Automatically selects the right protocol/tool for each operation
4. **Unified Auth:** Single authentication system for all external services
5. **Consistent Error Handling:** Retry logic, fallbacks, circuit breakers built-in
6. **Performance Optimization:** Caching, batching, connection pooling automatic

**Source Files:**

- UTI Specification: `docs/Research & specifications/*Yantra Unified Tool Interface.md`

Visual Architecture:

```
┌──────────────────────────────────────────────────────────────────────┐
│                              YANTRA                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      MONACO EDITOR                              │ │
│  │                           │                                     │ │
│  │                      LSP Client                                 │ │
│  │                           │                                     │ │
│  │           ┌───────────────┼───────────────┐                     │ │
│  │           ▼               ▼               ▼                     │ │
│  │     Pylance(LSP)    rust-analyzer     tsserver                  │ │
│  │     [Real-time autocomplete, hover, diagnostics]                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      AI AGENT                                   │ │
│  │                           │                                     │ │
│  │              UNIFIED TOOL INTERFACE (UTI)                       │ │
│  │                           │                                     │ │
│  │           ┌───────────────┴───────────────┐                     │ │
│  │           ▼                               ▼                     │ │
│  │     MCP Adapter                    Builtin Adapter              │ │
│  │           │                               │                     │ │
│  │     ┌─────┴─────┐                   ┌─────┴─────┐               │ │
│  │     ▼           ▼                   ▼           ▼               │ │
│  │  Pylance     Git MCP             File Ops   Tree-sitter         │ │
│  │   (MCP)     Postgres             Terminal   Dep Graph (GNN)     │ │
│  │  GitHub     Railway              Browser    Code Search         │ │
│  │  Linear     Slack                Testing    Architecture View   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## Protocol Selection Framework

For each capability, UTI determines the protocol based on this decision matrix:

| Question                                              | If YES →                                    |
| ----------------------------------------------------- | ------------------------------------------- |
| Does the editor need it in real-time while typing?    | **LSP** (Editor only, not exposed to agent) |
| Is it a core differentiator we must control?          | **Builtin**                                 |
| Is it performance-critical (<10ms required)?          | **Builtin**                                 |
| Is it security-critical (command execution, secrets)? | **Builtin**                                 |
| Does it need streaming output for progress?           | **Builtin** or MCP with streaming           |
| Is there a well-maintained community MCP server?      | **MCP**                                     |
| Is it platform-specific (deployment, monitoring)?     | **MCP**                                     |
| Is it debugging-specific?                             | **DAP** (separate protocol)                 |

---

## Consolidated Capability Matrix

**Table Columns:**

- **Purpose:** CG (Code Generation), TS (Testing), DP (Deployment), MM (Monitor/Maintain)
- **Category:** Functional grouping of capabilities
- **Capabilities:** Representative capabilities in this category (not exhaustive)
- **Primary Protocol:** Main protocol used
- **Secondary Protocol:** Fallback protocol
- **Example Tool(s):** Representative implementations or MCP servers
- **PDC Phase:** Preventive Development Cycle phase(s) where used

| Purpose   | Category                          | Capabilities (representative)                                                                                               | Primary Protocol  | Secondary Protocol | Example Tool(s)                                              | PDC Phase                           |
| --------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------ | ------------------------------------------------------------ | ----------------------------------- |
| **CG**    | **File System (core ops)**        | file.read, file.write, file.edit, file.delete, file.move, directory.list, file.search, file.grep, file.watch                | **Builtin**       | MCP                | Local FS operations, @modelcontextprotocol/server-filesystem | 3 Execute                           |
| **CG**    | **AST & Parsing**                 | ast.parse, ast.query, ast.edit, ast.symbols, ast.scope, ast.diff                                                            | **Builtin**       | —                  | Tree-sitter (multi-language)                                 | 1 Architect / 3 Execute             |
| **CG/TS** | **Dependency Graph / GNN**        | depgraph.build, depgraph.query, depgraph.impact, depgraph.dependents, depgraph.cycles, depgraph.modules, depgraph.crossRepo | **Builtin**       | —                  | petgraph + GNN runtime (Yantra core)                         | 1 Architect / 2 Plan / 3 Execute    |
| **CG/DP** | **Terminal / Shell**              | shell.exec, shell.execStreaming, shell.background, shell.kill, shell.env                                                    | **Builtin**       | —                  | Local shell (Approval Queue protected)                       | 3 Execute / 4 Deploy                |
| **TS**    | **Testing (execution)**           | test.discover, test.run, test.runAffected, test.coverage, test.watch, test.generate, test.benchmark                         | **Builtin**       | —                  | pytest, jest, cargo test (invoked by Yantra)                 | 3 Execute                           |
| **TS**    | **E2E / Browser Automation**      | e2e.run, browser.launch, browser.navigate, browser.click, browser.screenshot, browser.evaluate, browser.network             | **Builtin**       | —                  | CDP via chromiumoxide                                        | 3 Execute / 5 Monitor               |
| **DP**    | **Build & Container**             | build.run, build.incremental, lint.run, lint.fix, format.run, container.build, container.push, container.run                | **Builtin**       | MCP                | Docker CLI, npm/cargo/pip (shell), Prettier/ESLint           | 3 Execute / 4 Deploy                |
| **MM**    | **Security (scanning)**           | security.scan, security.secrets, security.audit, security.permissions                                                       | **Builtin**       | MCP                | Semgrep, Gitleaks, Bandit (builtin SAST)                     | 3 Execute / 4 Deploy                |
| **CG/MM** | **Documentation & Viz**           | docs.generate, arch.diagram, arch.validate, viz.depgraph, viz.chart, viz.diff, viz.mermaid                                  | **Builtin**       | MCP                | Mermaid, Graphviz, Plotly (inline viz)                       | 1 Architect / 3 Execute / 5 Monitor |
| **CG/MM** | **Package Management**            | pkg.install, pkg.remove, pkg.audit, pkg.outdated, pkg.lockSync                                                              | **Builtin**       | MCP                | npm/pip/cargo via shell + CVE MCP                            | 2 Plan / 3 Execute                  |
| **MM**    | **Context & Memory**              | context.add, context.search, context.summarize, embeddings.generate, embeddings.search                                      | **Builtin**       | MCP                | Local embeddings (fastembed-rs), HNSW vector DB              | 2 Plan / 3 Execute                  |
| **CG**    | **Code Intelligence (symbols)**   | code.symbols, code.definition, code.references, code.completion, code.hover, code.diagnostics, code.rename, code.format     | **MCP**           | Builtin/LSP        | Pylance (MCP), rust-analyzer, tsserver; Tree-sitter fallback | 1 Architect / 3 Execute             |
| **CG**    | **Version Control (Git)**         | git.status, git.diff, git.commit, git.push, git.branch, git.merge, git.stash, git.log, git.blame                            | **MCP**           | Builtin            | @modelcontextprotocol/server-git, git2-rs fallback           | 3 Execute / 4 Deploy                |
| **CG**    | **GitHub / Code Hosting**         | github.repos, github.issues, github.prs, github.actions, github.releases, github.search                                     | **MCP**           | —                  | @modelcontextprotocol/server-github                          | 3 Execute / 4 Deploy                |
| **CG**    | **Database Access**               | db.connect, db.query, db.execute, db.schema, db.tables, db.migrate, db.seed                                                 | **MCP**           | —                  | Postgres/MySQL/SQLite/MongoDB MCP servers                    | 3 Execute / 5 Monitor               |
| **TS**    | **Test Data & Seeding**           | db.seed, test.generate, api.mock                                                                                            | **MCP + Builtin** | —                  | DB MCP (seeding) + Builtin test generator                    | 2 Plan / 3 Execute                  |
| **MM**    | **Monitoring & Observability**    | logs.tail, logs.search, metrics.query, traces.query, health.check, alerts.list, uptime.status                               | **MCP**           | Builtin            | Sentry, Prometheus, Datadog, PagerDuty MCPs                  | 5 Monitor / 4 Deploy                |
| **MM**    | **Security (CVE & scanning)**     | security.deps, security.container                                                                                           | **MCP**           | Builtin            | Snyk MCP, Trivy MCP (container scanning)                     | 3 Execute / 4 Deploy                |
| **DP**    | **Cloud Deploy & Infra**          | deploy.preview, deploy.production, deploy.rollback, deploy.status, deploy.logs, infra.provision, infra.destroy              | **MCP**           | Builtin            | Railway MCP, AWS MCP, GCP MCP, Vercel MCP                    | 4 Deploy                            |
| **MM**    | **Collaboration & Notifications** | slack.send, slack.search, email.send, notion.query, notion.update, linear.issues, jira.issues                               | **MCP**           | —                  | Slack MCP, Notion MCP, Linear MCP, Jira MCP                  | 2 Plan / 3 Execute / 5 Monitor      |
| **MM**    | **Debugging**                     | debug.launch, debug.attach, debug.breakpoint, debug.step, debug.evaluate, debug.variables, debug.stack                      | **DAP**           | —                  | debugpy (Python), node-debug2, codelldb (Rust), delve (Go)   | 3 Execute                           |
| **CG**    | **HTTP / API**                    | http.request, http.graphql, api.importSpec, api.generateClient, api.test, websocket.connect                                 | **Builtin**       | MCP                | HTTP client (builtin), OpenAPI tools                         | 3 Execute / 5 Monitor               |

### MCP Server Priority (from Section 3)

**P0 (MVP):** Git, GitHub, Slack, Railway, Linear/Jira
**P1 (High Value):** Sentry, PagerDuty, Notion, AWS/GCP
**P2 (Nice to Have):** Figma, Datadog, CircleCI, Google Docs

### Configuration

All UTI tools are configured via `yantra.tools.yaml`:

```yaml
agent:
  builtin:
    enabled: true
    features:
      depgraph: true
      treesitter: true
      browser: true

  mcp:
    enabled: true
    servers:
      git:
        package: '@modelcontextprotocol/server-git'
      github:
        package: '@modelcontextprotocol/server-github'
        config:
          token: '${GITHUB_TOKEN}'
```

---

## Preventive Development Cycle (PDC) Workflows

### Overview

Yantra follows a **Preventive Development Cycle (PDC)** - a five-phase approach where problems are prevented at each stage rather than fixed after they occur. All workflows align to these phases and are executed through four specialized state machines.

**PDC Phases:**

1. **Architect/Design** - Create and validate system architecture before coding
2. **Plan** - Break down work into executable tasks with dependency awareness
3. **Code Generation + Testing** - Generate correct, tested code feature-by-feature
4. **Deploy** - Safe, validated deployments with automatic rollback
5. **Maintain** - Self-healing systems that detect and fix issues autonomously

**Key Principle**: Every workflow enforces prevention guarantees - architecture alignment, dependency validation, test coverage, security scanning, and conflict prevention.

---

### Phase 1: Architect/Design Workflows

#### 1.1 New Project Initialization Workflow

**Purpose**: Establish architecture-first foundation for new projects before any code generation.

**Steps:**

1. **Collect Requirements/Intent**
   - User provides: chat description, upload file (PRD/specs), or external doc link (Notion, Confluence)
   - Agent clarifies ambiguities (auth method, database choice, framework preferences)
   - User can say "use defaults" for standard technology choices

2. **Generate Architecture**
   - Agent creates architecture from finalized requirements using LLM
   - Architecture displayed in Architecture View tab
   - Components, connections, and boundaries defined
   - **Approval Gate**: User reviews and approves (modifications allowed)
   - Saved as version 1 baseline in SQLite

3. **Generate Technical Specifications**
   - Auto-generated after architecture approval
   - Includes: component specs, API contracts, data models, security requirements
   - Displayed in Documentation tab
   - **Approval Gate**: User reviews and approves

4. **Extract Features**
   - Parsed from requirements, architecture, and technical specs
   - All features marked ⏳ Planned (0% complete)
   - Displayed in Features tab with descriptions and estimates

5. **Create Project Plan with Milestones**
   - Agent autonomously determines 3-5 approval checkpoints by analyzing:
     - Related features (e.g., Authentication, Payments, Infrastructure)
     - Security-critical features requiring validation
     - External dependencies and integration points
   - Each milestone marked with ✅ APPROVAL CHECKPOINT
   - Displayed in Plan tab with estimates and dependencies
   - **Approval Gate**: User approves checkpoint placement (modifications allowed)

6. **Select Interaction Mode**
   - Choice offered AFTER plan approval:
     - **Guided Mode**: Approve every step (architecture, code, tests, commits)
     - **Auto Mode**: Approve only at milestones + architecture changes
   - Agent CANNOT proceed past approval checkpoints without explicit user consent

7. **Log All Decisions**
   - Record to Decision Log: architecture source, requirements source, timestamps, all approvals

**Output:**

- ✅ Approved architecture (Version 1)
- ✅ Technical specifications document
- ✅ Features list (all ⏳ Planned)
- ✅ Project plan with approval checkpoints
- ✅ Mode selection (Auto/Guided)
- ✅ Decision log entries

**Prevention Guarantees:**

- Architecture exists before code generation begins
- Requirements clarified before implementation
- Plan validated before execution
- All decisions documented with audit trail

---

#### 1.2 Existing Project Import Workflow

**Purpose**: Discover and document architecture for existing codebases, establish governance baseline.

**Phase 1: Multi-Level Architecture Discovery**

**Level 1: Priority Files (Exact Paths)**

- `.yantra/architecture.db` (100% confidence - native format)
- `.yantra/architecture.json` (90% confidence - Yantra export)
- `architecture.md`, `ARCHITECTURE.md` (80% confidence - root directory standard)
- `architecture.json` (80% confidence - root directory JSON)

**Level 2: Common Documentation Locations (Pattern Match)**

- `docs/*.md` - All markdown files recursively
- `.github/*.md` - System design documentation
- `wiki/*.md` - Wiki-based documentation
- `design/*.md`, `architecture/*.md` - Design folders
- `docs/adr/*.md`, `docs/decisions/*.md` - Architecture Decision Records

**Level 3: Content-Based Detection (Deep Scan)**

- Search for architecture section headers: `# Architecture`, `## System Architecture`, etc.
- Detect diagram formats: Mermaid (`graph TD`, `C4Context`), PlantUML (`@startuml`), ASCII diagrams
- Keyword scoring: "component", "service", "microservice" (high-value), "authentication", "cache" (medium-value)
- Calculate confidence score: 90-100% (use immediately), 70-89% (confirm), 50-69% (present as option), <50% (request guidance)

**Phase 2: Requirements/Specs Discovery**

- Search for: `requirements.md`, `specs.md`, `PRD.md`, `docs/requirements/`, README sections
- Extract and score by confidence (similar to architecture discovery)

**Phase 3: Handle Discovery Results (5 Cases)**

**Case A: Architecture Found, No Requirements**

- Import architecture to Architecture View
- Options presented to user:
  1. Generate specs from code review (recommended)
  2. Skip specs for now
  3. Analyze code for both architecture and requirements

**Case B: Requirements Found, No Architecture**

- Import requirements to Documentation tab
- Options presented to user:
  1. Generate architecture from requirements (recommended)
  2. Generate architecture from code review
  3. Upload architecture file manually

**Case C: Both Found (Architecture + Requirements)**

- Import both to respective tabs
- Show in Architecture View + Documentation tabs
- **Approval Gate**: User reviews and approves imported content

**Case D: Multiple Architecture Sources (50-70% confidence)**

- Present all options to user with confidence scores
- User chooses: select one, combine all, or analyze code instead

**Case E: Low/No Documentation (<50% or none)**

- Options presented to user:
  1. User describes project in chat (guided conversation)
  2. Auto-analyze code (GNN-based discovery)
  3. Upload/import files (Markdown, JSON, Mermaid, PlantUML)

**Phase 4: Code Review Assessment (for Existing Code)**

1. **GNN Dependency Analysis**
   - Parse all source files using tree-sitter
   - Build dependency graph (functions, classes, imports, calls)
   - Identify components and their relationships

2. **Security Scanning**
   - Run Semgrep with OWASP rules (parallel processing: 4 workers)
   - Check dependencies for vulnerabilities (Safety for Python, npm audit)
   - Detect secrets (TruffleHog patterns)
   - Categorize by severity (Critical/High/Medium/Low)

3. **Code Quality Assessment**
   - Calculate cyclomatic complexity
   - Measure test coverage
   - Identify code smells and unused code

4. **Architecture Alignment Validation**
   - Compare code structure to imported/generated architecture
   - Detect: missing components, extra components, incorrect connections
   - Flag mismatches with severity levels

5. **Generate Recommendations**
   - Prioritize: 1) Security issues, 2) Architecture misalignment, 3) Code quality
   - Suggest: update architecture, refactor code, fix vulnerabilities
   - Provide actionable options to user

**Phase 5: Feature Extraction with Status**

- Extract from: requirements, architecture, specs, AND existing code
- Use GNN to detect implemented features
- Calculate completion status:
  - ✅ Done: 100% (files exist, tests pass, no errors)
  - 🔄 In Progress: 1-99% (partial implementation)
  - ⏳ Planned: 0% (not started)
- Display in Features tab with completion percentages

**Phase 6: Create Plan with Current State Assessment**

- Agent determines milestones based on current project state
- Prioritizes: 1) Complete in-progress features, 2) Implement pending features, 3) Harden/test
- Mark approval checkpoints
- Show estimates and current completion status

**Phase 7: Mode Selection & Decision Logging**

- Same as new project workflow (Guided or Auto mode)
- Log all decisions: architecture source, import method, timestamps

**Output:**

- ✅ Imported/generated architecture (Version 1)
- ✅ Code review report (security, quality, alignment)
- ✅ Features list with actual completion status (✅ 🔄 ⏳)
- ✅ Project plan reflecting current state
- ✅ Mode selection (Auto/Guided)
- ✅ Decision log entries

**Performance Limits:**

- Max files to scan: 1,000 (configurable)
- Max file size: 5 MB per file
- Timeout: 30 seconds for full scan
- Parallel processing for security scan and code analysis

**Prevention Guarantees:**

- Existing architecture discovered and documented
- Code quality and security assessed before new development
- Current state accurately reflected in plan
- All findings documented with severity levels

---

#### 1.3 Architecture Maintenance & Governance Workflow

**Purpose**: Continuously ensure code aligns with architecture during development through deviation detection and user-approved updates.

**Trigger 1: During Code Generation (Proactive)**

**Scenario**: Agent generates code that would deviate from planned architecture.

**Process**:

1. Agent generates code using LLM
2. Before writing to disk, parse code with tree-sitter to extract imports/dependencies
3. Load current architecture from SQLite
4. Query GNN for expected dependencies of target component
5. Compare actual imports vs allowed dependencies

**On Deviation Detected**:

- **Pause** code generation
- Calculate severity (Low/Medium/High/Critical):
  - Critical: Security breach, circular dependency
  - High: Breaks layering (e.g., UI → Database directly)
  - Medium: Skips one layer (e.g., Gateway → Database, bypassing service)
  - Low: Extra utility import (minor)
- Present to user with 3 options:
  - **Update Architecture**: Modify architecture to allow new dependency (creates version N+1)
  - **Fix Code**: Regenerate code using LLM to match existing architecture
  - **Cancel**: Stop code generation entirely
- Wait for user decision
- Execute chosen action

**Trigger 2: After Manual Code Edit (Reactive)**

**Scenario**: User manually edits code, potentially breaking architecture alignment.

**Process**:

1. On file save, detect change via file watcher
2. Check if project has active architecture
3. Run GNN analysis on saved file to get current dependencies
4. Compare to architecture expectations for that component

**On Misalignment Detected**:

- Display warning notification in UI
- Add system message to chat with details:
  - File path and deviation summary (expected → actual)
  - Severity level with visual indicator
  - User-friendly explanation
  - Recommended actions
- User decides: update architecture, revert code, or refactor code

**Trigger 3: New Feature Impacts Architecture (Requirement-Driven)**

**Scenario**: User requests feature that requires architecture changes.

**Process**:

1. **Analyze Requirement Impact**
   - Parse user requirement using LLM
   - Use GNN to identify affected files and components
   - Determine if architecture changes needed

2. **Preview Architecture Changes**
   - List new components to add
   - List modified connections
   - List affected existing components
   - Calculate impact severity (Low/Medium/High/Breaking):
     - Low: Changes within single component, no new dependencies
     - Medium: Changes across 2-3 components, existing dependencies
     - High: New component required
     - Breaking: Fundamental pattern change (e.g., monolith → microservices)

3. **Require User Approval**
   - Show architecture changes BEFORE implementation
   - Display affected components visually
   - **Approval Gate**: Explicit approval required

4. **Execute After Approval**
   - Update architecture (increment version to N+1)
   - Generate code matching approved architecture
   - Validate final code aligns with architecture

**Example Flow:**

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

**Auto-Save with Rule of 3 Versioning:**

- Keep current version + 3 most recent past versions (total: 4 versions)
- When 5th version created, auto-delete oldest (version 1)
- Each version immutable once created
- Version metadata: incremental number, full snapshot, timestamp, change type, agent reasoning, original user message

**Agent Commands (Via Chat):**

- "Show me the architecture" - Opens Architecture View tab, displays current version
- "Revert to previous architecture" - Loads version N-1, auto-saves as new version N+1
- "Show architecture history" - Lists last 3 versions with timestamps and changes
- "Why did you add Redis?" - Shows version history and reasoning

**Output:**

- ✅ Updated architecture version (N+1)
- ✅ Deviation report (if detected)
- ✅ Code aligned with approved architecture
- ✅ Version history maintained (Rule of 3)
- ✅ Decision log entry with reasoning

**Prevention Guarantees:**

- Architecture never drifts from code
- All deviations detected at state boundaries (not after completion)
- Every architecture change has documented reasoning
- Can rollback to previous architecture versions
- No code generation proceeds without architecture alignment

---

### Phase 2: Plan Workflows

#### 2.1 Task Decomposition & Sequencing Workflow

**Purpose**: Break down features into executable tasks with proper dependency ordering and conflict awareness.

**Integrated into**: Code Generation State Machine (states 5-10)

**Process:**

1. **TaskDecomposition** (<500ms)
   - GNN-based feature analysis
   - Break down into atomic implementation tasks
   - Each task maps to specific files/components
   - Prevents: missing tasks, unbounded work, unclear scope

2. **DependencySequencing** (<100ms)
   - GNN graph traversal to identify task dependencies
   - Topological sort for execution order
   - Prevents: wrong task order, dependency violations, parallel conflicts

3. **ConflictCheck** (<50ms)
   - **MVP**: Query active work indicators - which developers are editing which files
   - **Post-MVP**: Query file lock table for locked files
   - Prevents: parallel edit conflicts, work coordination issues

4. **PlanGeneration** (<200ms)
   - Create final execution plan with estimates
   - Complexity analysis for each task
   - Priority assignment
   - Prevents: unclear scope, missing estimates, unprioritized work

5. **PlanReview** (optional approval gate)
   - **Triggered for**: Features with >5 tasks OR multi-file changes
   - User reviews: task breakdown, sequencing, estimates, complexity
   - **Approval Gate**: User approves or requests modifications
   - Prevents: misaligned expectations, scope disagreements, over-complexity

6. **EnvironmentSetup** (automated)
   - Virtual environment creation/validation
   - Dependency installation
   - Version validation
   - Prevents: environment configuration errors, missing dependencies

**Output:**

- ✅ Task breakdown with file mappings
- ✅ Dependency-ordered execution sequence
- ✅ Conflict awareness (MVP: visibility, Post-MVP: locks)
- ✅ Estimates and complexity scores
- ✅ Optional plan approval (for complex features)
- ✅ Ready environment

**Prevention Guarantees:**

- Explicit planning prevents unbounded work
- Tasks properly sequenced based on dependencies
- Parallel work conflicts minimized (MVP) or prevented (Post-MVP)
- All tasks scoped and estimated before execution

---

### Phase 3: Code Generation + Testing Workflows

#### 3.1 Feature Implementation Workflow (Pair Programming)

**Purpose**: Generate production-quality code feature-by-feature using Yantra Codex + Multi-LLM consultation.

**Integrated into**: Code Generation State Machine + Testing State Machine

**Code Generation Process:**

1. **ContextAssembly** (<100ms)
   - Load relevant code context from GNN
   - Retrieve semantic embeddings for similar patterns
   - Assemble context for LLM prompt
   - Prevents: reinventing functionality, inconsistent patterns, missing context

2. **CodeGeneration (Pair Programming with Multi-LLM Consultation)**

   **Phase A: Initial Generation**

   **MVP Mode (LLM-Only - No Yantra Codex):**
   - **Primary LLM** (user's choice: Claude/GPT-4/Gemini) generates code directly
   - Uses context from GNN dependency graph
   - Standard LLM generation with prompt engineering

   **Post-MVP Mode (With Yantra Codex - Cost Optimization):**
   - **Yantra Codex (GNN)** generates initial code (15ms)
   - Calculate confidence score (0.0-1.0)
   - **If confidence >= 0.8**: Yantra code used directly
   - **If confidence < 0.8**: Send to Primary LLM for review and enhancement
   - **Primary LLM** reviews edge cases, adds error handling
   - **Merge** Yantra + LLM suggestions
   - **Yantra learns** from LLM improvements for future

   **Phase B: Validation & Retry**
   - Run validation: tree-sitter (syntax), LSP (types), linter (style)
   - Run tests automatically
   - **Attempt 1**: If validation fails → Primary LLM retries with error context
   - **Attempt 2**: If still fails → Primary LLM retries again

   **Phase C: Multi-LLM Consultation (After 2 Failures)**
   - **Trigger**: After 2 consecutive failures with same issue
   - **Consultation Flow**:
     1. Primary LLM generates consultation prompt (meta-prompting)
     2. Consultant LLM (different model) provides second opinion:
        - Identifies blind spots or framing issues
        - Suggests alternative approaches
        - Points out what Primary LLM might be missing
     3. Primary LLM regenerates code WITH consultant's insight added to context
   - **Attempt 3**: Primary with consultant insight (higher success rate)
   - **Cost Benefit**: 2 failures + 1 consultation + 1 success = 4 calls (vs 5+ attempts without consultation)

   **Phase D: Escalation (If All Fail)**
   - After 3 attempts (including consultation): Escalate to user
   - Options: Try different primary model, manual intervention, skip feature

   **Why Multi-LLM Consultation Works:**
   - Different LLMs have different training data and biases
   - Claude: Strong reasoning, cautious on edge cases
   - GPT-4: Pragmatic, compact code style
   - Gemini: Different perspective, fresh approach
   - When stuck, problem is often **framing or blind spot** - different LLM sees differently

   **UI Visibility:**
   - **MVP**: Shows which files are being generated (work visibility)
   - Shows attempt number and which LLM is being consulted
   - Example: "🤔 Consulting GPT-4 Turbo for second opinion..."

   **Prevents:**
   - Syntax errors (tree-sitter validation)
   - Type errors (LSP validation)
   - Style issues (linter validation)
   - Getting stuck on same approach (multi-LLM brings fresh perspective)
   - Excessive failed attempts (consultation after 2 failures)

3. **DependencyValidation** (<10ms)
   - GNN impact analysis on generated code
   - Check if changes break dependent code
   - Identify ripple effects
   - Prevents: breaking changes to dependents

4. **BrowserValidation** (5-10s)
   - Load generated code in headless Chrome via CDP
   - Capture screenshots, check console errors
   - Validate network requests
   - Prevents: visual regressions, runtime errors, network failures

5. **SecurityScanning** (parallel: 4 workers, ~5s)
   - Semgrep with OWASP rules
   - Detect: SQL injection, XSS, secrets in code
   - Check dependencies for vulnerabilities
   - Prevents: security vulnerabilities before commit

6. **ConcurrencyValidation** (<5s, only for parallel code)
   - Static analysis (Clippy, threading analyzers)
   - GNN data flow analysis for race conditions
   - Stress test generation for concurrent code paths
   - Auto-fix: pattern-based → LLM-assisted → sequential fallback
   - Prevents: race conditions, deadlocks, data races

7. **FixingIssues** (if any validation fails)
   - **Attempt 1**: Primary LLM retries with error context from validation
   - **Attempt 2**: Primary LLM retries again with accumulated error context
   - **Attempt 3**: If still failing after 2 attempts, trigger Multi-LLM Consultation:
     - Consultant LLM (different model) analyzes the issue
     - Provides second opinion on what might be wrong
     - Primary LLM regenerates with consultant's insight
   - **Escalation**: After 3 attempts, escalate to user for manual intervention
   - **Prevents**: Quality shortcuts, getting stuck on same failed approach

**Testing Process** (Auto-triggered after CodeGen Complete):

1. **TestGeneration**
   - LLM generates comprehensive unit tests
   - Pattern-based test templates
   - Edge case coverage
   - Prevents: missing test coverage, untested edge cases

2. **EnvironmentSetup**
   - Virtual environment ready
   - Test fixtures prepared
   - Dependencies installed
   - Prevents: environment inconsistencies

3. **UnitTesting** (parallel: 4 workers, ~8s typical)
   - pytest/jest/cargo test execution
   - Run tests for modified code + affected dependents
   - **Quality Gate**: ALL tests must pass (100%)
   - Prevents: logic errors, edge case failures, function behavior bugs

4. **BrowserTesting** (parallel: 3 browsers, ~17s typical)
   - Playwright E2E tests
   - Test user workflows (login→dashboard→logout)
   - Responsive design, accessibility tests
   - Prevents: user workflow failures, integration issues, UI bugs

5. **IntegrationTesting** (20-40s)
   - Component interaction tests
   - API contract validation
   - Database integration tests
   - Prevents: component interaction failures, API violations

6. **CoverageAnalysis**
   - pytest-cov/coverage.py execution
   - **Quality Gate**: >80% coverage required
   - Identify gaps in validation
   - Prevents: insufficient test coverage

7. **FixingIssues** (if tests fail)
   - **Attempt 1**: Primary LLM regenerates code with test failure context
   - **Attempt 2**: Primary LLM retries with accumulated failure context
   - **Attempt 3**: After 2 failures, trigger Multi-LLM Consultation for second opinion
   - Re-run affected tests after each fix attempt
   - **Escalation**: After 3 attempts, escalate to user for manual intervention

**Output:**

- ✅ Production-quality code (validated)
- ✅ Comprehensive unit tests (100% pass)
- ✅ Browser validation passed
- ✅ Integration tests passed
- ✅ >80% test coverage
- ✅ Security scan clean
- ✅ Concurrency validation passed (if applicable)
- ✅ Ready for deployment

**Prevention Guarantees:**

- Code correct by construction (multi-layer validation)
- All tests pass before proceeding (no exceptions)
- Security vulnerabilities caught before commit
- Concurrency bugs prevented or code falls back to safe sequential execution
- Breaking changes to dependents prevented
- Visual regressions prevented
- **Multi-LLM consultation prevents getting stuck** (different perspectives after 2 failures)

**Cost Optimization (Post-MVP with Yantra Codex):**

- **MVP Baseline:** 100% Primary LLM usage (standard costs)
- **Post-MVP with Yantra Codex:**
  - Month 1: 55% handled by Yantra alone (64% cost savings)
  - Month 6: 85% handled by Yantra alone (88% cost savings)
  - Year 1: 95% handled by Yantra alone (96% cost savings)
- **Multi-LLM consultation** (works with or without Yantra): 4 total calls (2 + consultation + 1) vs 5+ attempts without → saves time and money

---

#### 3.2 Multi-Agent Parallel Development Workflow

**Purpose**: Enable multiple agents to work on different features simultaneously with proactive conflict prevention using Git branches and Cloud Dependency Graph (Tier 0).

**Architecture**: Master-Agent pattern where Master assigns work then steps back, agents execute independently on Git branches, coordination through dedicated Git branch + Cloud Dependency Graph for dependency-aware conflict avoidance.

**Key Principles:**

- **Master is not a single point of failure**: Assigns work once, then agents run independently
- **Git branches provide isolation**: Each agent works on own feature branch
- **Proactive conflict avoidance**: Prevent conflicts before they happen, not resolve after
- **Cloud Dependency Graph provides real-time collaboration intelligence**: 4 levels of conflict detection

---

**Phase 1: Master Assignment (30s - One-Time)**

1. **Feature Decomposition**
   - Master receives high-level feature request
   - Dependency Graph analyzes code dependencies to minimize cross-agent coupling
   - Decomposes into sub-features with clear boundaries:
     - File ownership (backend vs frontend vs tests vs docs)
     - Dependency relationships (API must complete before UI)
     - Minimal shared code between agents
2. **Work Assignment**
   - Assigns each sub-feature to specialized agent:
     - Backend Agent: API endpoints, database models, business logic
     - Frontend Agent: UI components, state management, styling
     - Testing Agent: Unit tests, integration tests, fixtures
     - DevOps Agent: Deployment, infrastructure
     - Docs Agent: Documentation, API docs
3. **Coordination Branch Setup**
   - Create/update `.yantra/coordination` Git branch (append-only event log)
   - Commit assignment events:
     ```json
     {
       "event": "feature_assigned",
       "feature": "user-auth",
       "agent": "backend-agent-1",
       "branch": "feature/user-auth",
       "files": ["src/auth/login.py", "src/auth/session.py"],
       "timestamp": "2025-12-06T10:00:00Z"
     }
     ```
4. **User Review (Optional)**
   - Show proposed assignment: "Agent 1 handles auth, Agent 2 handles payments"
   - Highlight shared dependencies: "Both touch UserService - is this split okay?"
   - User can approve or request re-assignment
5. **Master Steps Back**
   - Master's job is done after assignment
   - Can shut down or take new requests
   - Agents continue independently without Master

---

**Phase 2: Agent Independent Execution (Parallel)**

**Before Starting Work:**

1. **Pull Coordination Branch**
   - Agent pulls `.yantra/coordination` to see assignment
   - Identifies assigned feature and files

2. **Proactive Conflict Check via Cloud GNN** (4 Levels)

   **Level 1: Same File Detection**
   - Query Cloud GNN: "Is anyone modifying files A, B, C?"
   - GNN responds with current file claims
   - If file already claimed by another agent: WARN immediately

   **Level 2: Direct Dependency Detection**
   - Query GNN: "What are dependencies of files A, B, C?"
   - GNN responds: "File B imports from File X (being modified by Agent 2)"
   - Agent warned: "Your file depends on a file being modified"

   **Level 3: Transitive Dependency Detection**
   - GNN traces dependency chains: A → B → C → X (Agent 2 modifying X)
   - Catches conflicts humans routinely miss
   - Agent warned of indirect coupling

   **Level 4: Semantic Dependency Detection**
   - GNN detects function signature changes
   - Agent 2 changing `authenticate()` signature
   - GNN warns all agents calling that function
   - "Function you're using is being modified. Signature may change."

3. **Conflict Resolution Options**
   - **No conflicts**: Proceed immediately
   - **Conflicts detected**: Choose:
     - Wait for other agent to complete (estimated time shown)
     - Coordinate approach with other agent (peer-to-peer)
     - Work on non-conflicting parts first
     - Request re-assignment from Master

4. **Claim Files in Cloud GNN**
   - Agent claims files: "Agent-1 modifying payment.py, checkout.py"
   - Real-time update visible to all agents
   - Prevents other agents from silent conflicts

5. **Create Feature Branch**
   - Agent creates Git branch from main: `feature/user-auth`
   - All work isolated to this branch
   - No impact on main or other agent branches

**During Work:**

1. **Independent Execution**
   - Agent generates code using Pair Programming workflow (3.1)
   - Runs full validation: tests, security, browser checks
   - Commits to feature branch incrementally
   - No need for Master approval during work

2. **Dependency Coordination (Peer-to-Peer)**
   - Agent B needs API from Agent A
   - Query coordination branch: "Has Agent A completed user auth API?"
   - If completed: Pull Agent A's branch, continue work
   - If not: Wait, work on other parts, or coordinate approach
3. **Commit Dependency Available Events**
   - When API/component ready for others to use:
     ```json
     {
       "event": "dependency_available",
       "agent": "backend-agent-1",
       "api": "UserService.authenticate",
       "commit": "abc123",
       "branch": "feature/user-auth"
     }
     ```
   - Other agents pull this branch to get the API

4. **Real-Time UI Updates**
   - Master UI shows progress: "Agent 1: 60% (auth API done), Agent 2: 30% (waiting for auth)"
   - Users see which agent is working on what
   - Transparency and control

**On Completion:**

1. **Release Files in Cloud GNN**
   - Agent releases file claims
   - Files now available for other agents

2. **Create Pull Request**
   - Agent creates PR for feature branch → main
   - Includes: code, tests, documentation
   - PR description links to coordination events

3. **Commit Completion Event**

   ```json
   {
     "event": "feature_completed",
     "agent": "backend-agent-1",
     "branch": "feature/user-auth",
     "pr": "PR-42",
     "timestamp": "2025-12-06T10:45:00Z"
   }
   ```

4. **PR Review & Merge**
   - Human reviews PR (or auto-merge if all tests pass)
   - Standard Git merge flow
   - Conflicts detected at merge time (rare due to proactive avoidance)

---

**Phase 3: Conflict Resolution (If Unavoidable)**

**Legitimate Conflicts:**

- Two features genuinely need to modify same code
- Not a failure - it's a coordination point

**Early Warning, Not Prevention:**

- GNN warns BEFORE either agent starts
- "Both Agent 1 and Agent 2 need to modify UserService.authenticate()"

**Resolution Options:**

1. **Single Owner**: One agent handles both changes
2. **Interface Agreement**: Agents agree on interface first, then implement
3. **Sequential Execution**: One completes, other adapts to changes

**Conflict Resolution Assistance (via GNN):**

- Which agent's changes are more extensive?
- Which changes have more downstream dependencies?
- Suggest which change should be base and which should adapt

---

**Example Flow:**

```
User: "Add complete checkout flow with Stripe payments"
  ↓
Master: Analyzes GNN dependencies (30s)
  - Sub-feature 1: Payment API (Backend) → Agent-1 → Branch: feature/payment-api
    Files: src/payment/stripe.py, src/api/checkout.py
  - Sub-feature 2: Checkout UI (Frontend) → Agent-2 → Branch: feature/checkout-ui
    Files: src/components/Checkout.tsx (depends on payment API)
  - Sub-feature 3: Tests → Agent-3 → Branch: feature/checkout-tests
    Files: tests/test_checkout.py (depends on API + UI)
  - Sub-feature 4: Docs → Agent-4 → Branch: feature/checkout-docs
    Files: docs/checkout.md (no dependencies, parallel)

Master: Commits assignments to coordination branch
Master: Steps back (job done)
  ↓
Agent-1 (Backend):
  - Pulls coordination branch → sees assignment
  - Queries Cloud GNN → No conflicts detected
  - Claims files in Cloud GNN
  - Creates branch: feature/payment-api
  - Generates code (10 min) → Tests pass
  - Commits "dependency_available" for PaymentAPI
  - Creates PR-101
  - Releases file claims
  ↓
Agent-2 (Frontend):
  - Pulls coordination branch → sees assignment
  - Queries Cloud GNN → Detects dependency on payment.py (Agent-1 working)
  - Waits for Agent-1 dependency_available event (10 min)
  - Sees event → Pulls feature/payment-api branch
  - Claims Checkout.tsx in Cloud GNN
  - Creates branch: feature/checkout-ui
  - Generates code using Agent-1's API (5 min) → Tests pass
  - Creates PR-102
  - Releases file claims
  ↓
Agent-3 (Testing):
  - Waits for Agent-1 and Agent-2 completion events
  - Pulls both branches
  - Creates branch: feature/checkout-tests
  - Generates integration tests (3 min) → Tests pass
  - Creates PR-103
  ↓
Agent-4 (Docs):
  - No dependencies - starts immediately (parallel with Agent-1)
  - Creates branch: feature/checkout-docs
  - Generates documentation (2 min)
  - Creates PR-104
  ↓
Human: Reviews 4 PRs → Merges all
Total: 18 minutes (vs 45 minutes single-agent) = 2.5x faster
Zero merge conflicts (proactive avoidance worked)
```

---

**Cloud GNN Service (Real-Time Coordination)**

**API Endpoints:**

1. **Claim File**: `POST /claim` - Agent X is modifying file Y
2. **Release File**: `POST /release` - Agent X finished with file Y
3. **Query Dependencies**: `GET /dependencies?file=Y` - What files are affected if I modify Y?
4. **Query Conflicts**: `GET /conflicts?files=A,B,C` - Is anyone modifying files that would conflict?

**Service Characteristics:**

- Lightweight, mostly in-memory with persistence
- Per-project isolation (projects don't see each other's state)
- Self-hosted (enterprise) or Yantra-hosted (convenience)
- Shares dependency graph structure (not actual code)

**Privacy:**

- Only dependency relationships shared (which files/functions depend on which)
- Actual code content stays local
- File modification state shared within project team only
- User's private projects not shared unless explicitly enabled

---

**Recovery After Crashes:**

**Agent Crash:**

- Agent restarts → Pulls coordination branch
- Reads: "I was assigned feature X, started at timestamp T, branch name B"
- Checks branch status → Resumes from where it left off
- If files still claimed in Cloud GNN → Releases stale claims

**Master Crash:**

- No problem! Master only assigns work once
- Agents continue independently on their branches
- Master can restart to assign new work

**Cloud GNN Service Crash:**

- Coordination branch still has event log (Git is durable)
- Agents fall back to coordination branch for basic coordination
- When service recovers, state rebuilt from recent events

---

**Output:**

- ✅ Multiple features implemented simultaneously (2-10x faster)
- ✅ **Proactive conflict avoidance** (not reactive resolution)
- ✅ **Zero merge conflicts** in 95%+ cases (4-level conflict detection)
- ✅ Proper dependency sequencing maintained
- ✅ Git branches provide isolation and recovery
- ✅ Coordination branch provides audit trail
- ✅ Separate PRs for each sub-feature (clean Git history)

**Prevention Guarantees:**

- **Level 1**: Same-file conflicts prevented (claim/release mechanism)
- **Level 2**: Direct dependency conflicts warned before work starts
- **Level 3**: Transitive dependency conflicts detected (humans miss these)
- **Level 4**: Semantic conflicts (function signature changes) warned
- **Fault Tolerant**: Master not a single point of failure, agents recover from crashes
- **Scalable**: Works with 2 agents (laptop) or 10 agents (team) with same mechanism
- **Understandable**: Developers already know Git branches and PRs

**Why This Is Revolutionary:**

| Traditional Multi-Agent                  | Yantra Multi-Agent                                |
| ---------------------------------------- | ------------------------------------------------- |
| Reactive conflict resolution (after PRs) | Proactive conflict avoidance (before work starts) |
| No dependency awareness                  | 4-level dependency intelligence via Cloud GNN     |
| Master bottleneck (approves every step)  | Master assigns once, then agents independent      |
| Complex locking infrastructure           | Simple Git branches + lightweight Cloud GNN       |
| Silent conflicts accumulate              | Real-time conflict warnings                       |
| Human workflow broken                    | Mirrors proven Git branch workflow                |

---

### Phase 4: Deploy Workflows

#### 4.1 Safe Deployment Workflow

**Purpose**: Deploy validated code to production with automatic health checks and rollback capability.

**Integrated into**: Deployment State Machine (5 states)

**Process:**

1. **PackageBuilding** (30-60s)
   - Docker build for containerized apps
   - Artifact compilation (wheels, npm packages)
   - Asset bundling and optimization
   - Prevents: broken build artifacts, missing dependencies

2. **ConfigGeneration** (automated)
   - Generate `railway.json` deployment config
   - Create/update `Dockerfile`
   - Validate environment variables
   - Generate health check endpoints
   - Prevents: environment mismatches, missing configuration

3. **RailwayUpload** (20-40s)
   - Atomic upload to Railway platform via API
   - All-or-nothing deployment
   - Version tagging and metadata
   - Prevents: partial deployments, manual errors

4. **HealthCheck** (5-10s)
   - HTTP health check endpoint validation
   - Smoke tests on deployed environment
   - Response time validation
   - Database connectivity check
   - Prevents: deploying broken code, service unavailable

5. **RollbackOnFailure** (automatic)
   - If health check fails: instant rollback
   - Revert to last known good version
   - Alert user with failure details
   - Prevents: prolonged outages

**Approval Gate:**

- Manual trigger required for production deployment (human-in-loop for safety)
- User approves after reviewing: changes summary, test results, deployment plan
- Agent executes deployment automatically after approval

**Output:**

- ✅ Live production URL (e.g., `https://app.railway.app`)
- ✅ Health status confirmed
- ✅ Deployment audit trail
- ✅ Version tagged and documented

**Rollback Capability:**

- User can request rollback via chat: "Rollback to previous version"
- Agent executes rollback using Railway API
- Previous version restored in <2 minutes

**Prevention Guarantees:**

- Code fully tested before deployment (100% tests pass)
- Environment configuration validated
- Health checks prevent deploying broken code
- Automatic rollback on failure
- Zero manual deployment errors

---

### Phase 5: Maintain Workflows

#### 5.1 Self-Healing Workflow (Post-MVP)

**Purpose**: Automatically detect, diagnose, fix, validate, and deploy fixes for production issues without human intervention.

**Integrated into**: Maintenance State Machine (11 states)

**Process:**

1. **LiveMonitoring** (continuous)
   - Sentry/Datadog integration via MCP
   - Real-time error tracking
   - Performance monitoring
   - Uptime validation
   - Prevents: prolonged outages, undetected issues

2. **BrowserValidation** (continuous)
   - Real User Monitoring (RUM)
   - Session replay for errors
   - Browser error tracking
   - Performance degradation detection
   - Prevents: silent user-facing failures

3. **ErrorAnalysis** (<5s)
   - Pattern matching against known issues
   - Severity classification (Critical/High/Medium/Low)
   - Error correlation and grouping
   - Prevents: repeated incidents, unknown root causes

4. **IssueDetection** (<5s)
   - Root cause analysis
   - GNN dependency graph queries for impact
   - Historical pattern matching
   - Prevents: misdiagnosed issues, wasted effort

5. **AutoFixGeneration** (10-30s)
   - LLM generates fix based on error analysis
   - Use pattern-based solutions from knowledge base
   - Apply historical fixes for similar issues
   - Prevents: manual intervention delays, human error

6. **FixValidation** (2-3 min)
   - Run fix through full CodeGen + Testing cycle
   - All tests must pass (100%)
   - Security scan validation
   - Browser validation
   - Prevents: deploying broken fixes, introducing new bugs

7. **CICDPipeline** (1-2 min)
   - Automated CI/CD pipeline execution
   - Deployment to production
   - Health checks
   - Prevents: manual deploy delays, inconsistent fixes

8. **VerificationCheck** (ongoing)
   - Monitor error rate after fix deployed
   - Confirm issue resolved in production
   - Performance validation
   - User impact assessment
   - Prevents: fix didn't work, issue persists

9. **LearningUpdate** (automated)
   - Update Yantra Codex with new pattern
   - Add monitoring for similar issues
   - Create post-incident report
   - Document fix in knowledge base
   - Prevents: repeated incidents, knowledge loss

**State Transitions:**

- **Active** (normal operation) → **Incident** (on error detected)
- **Incident** → Auto-healing cycle → **Active** (issue resolved)
- **Incident** → **EscalatedToHuman** (if auto-fix fails after 3 retries)

**Mean Time To Repair (MTTR):**

- Known patterns: <5 minutes (detect → fix → deploy → verify)
- Novel issues: <15 minutes (includes LLM analysis and custom fix generation)
- Complex issues: Escalate to human after 3 auto-fix attempts

**Output:**

- ✅ Issue automatically detected
- ✅ Root cause identified
- ✅ Fix generated and validated
- ✅ Fix deployed to production
- ✅ Issue verified resolved
- ✅ Knowledge base updated
- ✅ Post-incident report created

**Prevention Guarantees:**

- Issues detected before users report (proactive)
- Fixes fully validated before deployment (same quality as human-written)
- No new bugs introduced by fixes
- Knowledge accumulated for future prevention
- MTTR <5 minutes for known patterns

---

### Phase 5: Maintain Workflows (Continued)

#### 5.2 Proactive Monitoring & Prevention Workflow (Post-MVP)

**Purpose**: Detect potential issues before they become production problems.

**Integrated into**: Maintenance State Machine (LiveMonitoring state)

**Proactive Checks:**

1. **Performance Degradation Detection**
   - Monitor response times over time
   - Detect gradual slowdowns (e.g., 500ms → 2s over 1 week)
   - Alert when threshold exceeded
   - Auto-generate optimization recommendations

2. **Error Rate Trend Analysis**
   - Track error rates by endpoint/feature
   - Identify increasing error trends (e.g., 0.1% → 0.5% over 2 days)
   - Investigate before reaching critical threshold
   - Prevent: errors becoming outages

3. **Dependency Version Monitoring**
   - Check for new package versions weekly
   - Scan for security advisories
   - Test compatibility in staging
   - Auto-generate upgrade PR if safe

4. **Resource Usage Monitoring**
   - Track memory/CPU/disk usage
   - Predict capacity issues (e.g., disk 80% full → will fill in 3 days)
   - Alert before resource exhaustion
   - Prevent: out-of-memory crashes, disk full errors

5. **External API Health Monitoring**
   - Monitor third-party API availability (Stripe, SendGrid, etc.)
   - Detect degraded performance
   - Switch to fallback providers if configured
   - Alert user of external issues

**Output:**

- ✅ Potential issues detected early
- ✅ Proactive recommendations generated
- ✅ Auto-generated PRs for safe upgrades
- ✅ Capacity planning alerts
- ✅ External dependency health visibility

**Prevention Guarantees:**

- Issues caught before becoming critical
- Capacity planning prevents resource exhaustion
- Security vulnerabilities patched proactively
- External API issues don't surprise users

---

## Summary: Workflows → PDC Phases → State Machines

**Consolidated Workflow Mapping:**

| PDC Phase               | Workflows                                                                               | State Machines            | Approval Gates               |
| ----------------------- | --------------------------------------------------------------------------------------- | ------------------------- | ---------------------------- |
| **1. Architect/Design** | 1.1 New Project Init`<br>`1.2 Existing Project Import`<br>`1.3 Architecture Maintenance | CodeGen (states 1-4)      | Architecture, Specs, Plan    |
| **2. Plan**             | 2.1 Task Decomposition & Sequencing                                                     | CodeGen (states 5-10)     | Plan Review (optional)       |
| **3. Code Gen + Test**  | 3.1 Feature Implementation`<br>`3.2 Multi-Agent Parallel                                | CodeGen + Testing         | None (quality gates enforce) |
| **4. Deploy**           | 4.1 Safe Deployment                                                                     | Deployment (states 1-5)   | Manual trigger               |
| **5. Maintain**         | 5.1 Self-Healing`<br>`5.2 Proactive Monitoring                                          | Maintenance (states 1-11) | None (auto-heal)             |

**Key Principles:**

- ✅ Architecture ALWAYS exists before code generation
- ✅ Plans are explicit and validated
- ✅ Code generated feature-by-feature with full testing
- ✅ **Multi-LLM consultation after 2 failures** (Yantra Codex + Primary LLM + Consultant LLM)
- ✅ Deployments are safe with automatic rollback
- ✅ Systems self-heal without human intervention
- ✅ All workflows enforce prevention guarantees

**Total Cycle Time (MVP):**

- New Project Init: <2 min (user input time varies)
- Existing Project Import: <1 min (+ scan time for large projects)
- Feature Implementation: <2 min per feature (code + tests, with multi-LLM consultation if needed)
- Deployment: <2 min (build + health check)
- Self-Healing: <5 min (detect + fix + deploy + verify)

---

## State Machine Architecture: Separation of Concerns

### Design Philosophy

Yantra's agentic capabilities are implemented through **four specialized state machines with Preventive Development Cycle**, each with a focused responsibility. This separation of concerns provides :

1. **Clarity**: Each machine has a clear, single responsibility
2. **Maintainability**: Changes to one machine don't affect others
3. **Testability**: Each machine can be tested independently
4. **Scalability**: Machines can run in parallel or be triggered independently
5. **MVP Focus**: Build only what's needed, when needed
6. **Parallel Processing**: States leverage concurrent execution whenever possible for performance

### Parallel Processing Principles

**Core Philosophy**: Yantra maximizes throughput by executing independent operations concurrently within and across state machines.

**When to Apply Parallel Processing:**

- **Independent Operations**: Tasks with no data dependencies can execute simultaneously
- **Multiple Resources**: Different resources (files, APIs, browser tabs) can be accessed concurrently
- **Batch Operations**: Multiple similar operations (e.g., testing multiple files, scanning multiple dependencies)
- **I/O-Bound Tasks**: Network requests, file reads, API calls benefit from concurrent execution

**Implementation Strategy:**

- **Within States**: Use async/await with tokio for concurrent operations within a single state
- **Across States**: Future enhancement - multiple state machines running simultaneously
- **Resource Pooling**: Browser tab pool, HTTP connection pool, LLM request batching

**Performance Benefits:**

- Sequential: N tasks × T seconds = N\*T total time
- Parallel: N tasks / P workers × T seconds ≈ (N/P)\*T total time (where P = parallelism factor)
- Example: 4 web searches: 12s sequential → 4s parallel (3x faster)

### The Four State Machines

#### 1. Code Generation State Machine (MVP)

**Responsibility**: Generate production-quality code that doesn't break dependencies

**States**:

**Phase 1: Architecture & Design (PDC Phase 1)**

- **ArchitectureGeneration**: Generate or import project architecture
- **ArchitectureReview**: Wait for user approval of architecture (human-in-the-loop)
  - Agent knows boundaries, won't generate violating code
  - Agent will detect Circular dependencies, scaling bottlenecks
  - Agent will flag single points of failure
  - Agent will assess Security vulnerabilities by design - Security patterns enforced at architecture level
  - ⚠️ **APPROVAL GATE**: User must approve architecture changes

**Phase 2: Planning (PDC Phase 2 - Enhanced)**

- **DependencyAssessment**: Assess the tools/package/techstack needed, analyze compatibility issues with version, do dry run validation, assess vulnerabilities with CVE db
  - **Web Search Integration**: Agent MUST use web search for latest package info (LLM knowledge is static/outdated)
    - Query official package registries (PyPI, npm, crates.io, Maven Central)
    - Fetch latest versions, changelogs, migration guides, known issues
    - Tools: MCP `@modelcontextprotocol/server-brave-search` or `@modelcontextprotocol/server-fetch`
  - **Semantic Graph-Enhanced Resolution**: Use semantic dependency graph to find similar past dependency conflicts and resolutions
    - Query cached package documentation (from previous web searches) via HNSW index
    - Retrieve compatibility matrices, known breaking changes from graph relationships
    - Provide context to LLM: "Your training data ends {date}, use web search for current info"
  - **Version-Level Tracking**: Store exact versions, version constraints, compatibility status in GNN
  - **Conflict Detection**: Cross-reference requested packages with existing dependencies, detect version conflicts
  - **⚡ Parallel Processing**:
    - Check multiple dependencies simultaneously (parallel CVE lookups, version checks)
    - Open multiple browser tabs for different package sources (PyPI, npm, docs, GitHub)
    - Batch API calls to package registries
    - Performance: N dependencies checked in ~O(1) time vs O(N) sequential
- **TaskDecomposition**: Break feature into concrete tasks (DB changes, API endpoints, UI components, tests, docs)
  - **⚡ Parallel Processing**: Analyze multiple feature aspects simultaneously (data model, API surface, UI requirements)
- **DependencySequencing**: Use GNN to determine task order and identify dependencies between tasks
  - **⚡ Parallel Processing**: Identify all independent task clusters that can execute in parallel
- **ConflictCheck**: Check which files will be modified and show visibility of who's working on what
  - **MVP**: Display active work visibility (which developer is working on which files)
  - **Post-MVP**: File locking mechanism to prevent parallel edits
  - **⚡ Parallel Processing**: Check multiple file locks/status simultaneously
- **PlanGeneration**: Create executable plan with task list, time estimates, critical path analysis
  - **⚡ Parallel Processing**: Calculate complexity estimates for multiple tasks concurrently
- **BlastRadiusAnalysis**: **NEW P0** - Calculate and visualize impact of planned changes
  - **Purpose**: Show users exactly what will be affected BEFORE execution (preventive transparency)
  - **Data Sources**: GNN dependency graph, test coverage, API specs, package metadata
  - **Intelligent Display Logic**:
    - **Always show** for: critical files, breaking changes, >10 indirect deps, package upgrades
    - **Show for large changes**: >5 tasks, >3 files, >20 tests affected
    - **Skip for small changes**: single file, no dependencies, <5 tests, no breaking changes
  - **Output**: BlastRadiusPreview (see detailed spec below)
  - **Performance Target**: <2s (GNN queries + analysis)
- **PlanReview**: User reviews and approves execution plan (optional approval gate for complex features)
  - ⚠️ **OPTIONAL APPROVAL GATE**: Required for features >5 tasks or multi-file changes OR high blast radius
  - **Displays**: ExecutionPlan + BlastRadiusPreview (integrated view)
  - **User Actions**: Approve, Modify Plan, Cancel
- **EnvironmentSetup**: Setup the venv with all the techstack dependencies installed. Always use venv
  - **⚡ Parallel Processing**: Install multiple independent packages simultaneously (parallel pip install)

**Phase 3: Execution (PDC Phase 3)**

- **ContextAssembly**: Gather hierarchical context from GNN
  - **⚡ Parallel Processing**: Fetch multiple file contexts simultaneously, parallel GNN queries for different dependency levels
- **CodeGeneration**: LLM generates code with full context
  - **MVP**: Show which files are being modified in UI (active work indicator)
  - **Post-MVP**: Explicit FileLockAcquisition state before code generation
  - **⚡ Parallel Processing**: Generate multiple independent files/functions simultaneously with separate LLM calls
- **DependencyValidation**: GNN checks for breaking changes
  - **⚡ Parallel Processing**: Validate multiple dependency paths simultaneously, parallel impact analysis
- **BrowserValidation**: Quick visual validation (renders, console errors, screenshots)
  - **⚡ Parallel Processing**:
    - Open multiple browser tabs for different pages/components
    - Test multiple viewports simultaneously (desktop, mobile, tablet)
    - Parallel screenshot capture for visual comparison
    - Performance: 3-5 components × 10s = 30-50s sequential → 10-15s parallel
- **SecurityScanning**: Semgrep + OWASP rules for vulnerability detection
  - **⚡ Parallel Processing**:
    - Scan multiple files simultaneously
    - Run different security rules in parallel (Semgrep, secrets detection, CVE checks)
    - Parallel analysis of multiple code patterns
    - Performance: N files scanned in ~O(log N) time with parallel workers
- **ConcurrencyValidation**: Race condition and deadlock detection (⚡ **PARALLEL SAFETY CHECK**)
  - **Purpose**: Validate that parallel processing didn't introduce concurrency bugs
  - **When**: After any code generation that uses async/await, threads, or parallel execution
  - **Built-in Checks** (Agentic Flow - NOT just LLM reasoning):
    - **Static Analysis**: Rust Clippy pedantic + thread safety lints, Python threading analyzer, JavaScript event loop analyzer
    - **Pattern Detection**: Shared mutable state access, missing locks/semaphores, race-prone patterns (check-then-act)
    - **GNN Analysis**: Identify data flow paths that could race, detect concurrent writes to same resource
    - **Test Generation**: Automatically generate stress tests for concurrent code paths
  - **LLM Role** (Supplementary):
    - Explain detected race conditions to user
    - Suggest fix strategies (add locks, use channels, immutable data)
    - Generate fixed code with proper synchronization
  - **Auto-Fix Strategy**:
    - Level 1: Apply known patterns (e.g., add Mutex, use Arc, channel-based communication)
    - Level 2: LLM generates fix with concurrency primitives
    - Level 3: Sequential fallback if parallel execution is unsafe
  - **Performance**: <5s (static analysis + GNN query)
  - **⚡ Parallel Processing**: Analyze multiple code paths for race conditions simultaneously
- **FixingIssues**: Auto-retry with fixes if validation fails
  - **⚡ Parallel Processing**: Generate fixes for multiple independent issues simultaneously
- **Complete**: Code ready for testing
  - **Post-MVP**: Explicit FileLockRelease state after completion
- **Failed**: Human intervention required
  - **Post-MVP**: Explicit FileLockRelease state on failure

**Entry Point**: User's natural language intent
**Exit Point**: Generated code + confidence score
**Trigger**: User submits task
**Success Criteria**: Code passes GNN validation, security scan, no breaking changes, visibility of active work

**State Count**:

- **MVP**: 17 states (Architecture: 4, Planning: 6, Execution: 7 - adds ConcurrencyValidation)
- **Post-MVP**: 19 states (adds explicit FileLockAcquisition and FileLockRelease states)

**Browser Validation Purpose (CodeGen)**:

- **Goal**: "Does it look right?" - Immediate visual feedback
- **Speed**: 5-10 seconds
- **Scope**: Single component/page smoke test
- **Actions**: Render in headless/headed Chrome, check console errors, capture screenshot for preview, detect network errors
- **Tools**: Chrome DevTools Protocol (CDP)
- **User Experience**: Real-time preview as code is generated (WYSIWYG)
- **Failure Action**: Retry code generation with visual feedback

**Performance Targets**:

- Task decomposition: <500ms (GNN analysis)
- Dependency sequencing: <100ms (graph traversal)
- Conflict check: <50ms (active work lookup - MVP) / <50ms (lock table query - Post-MVP)
- Plan generation: <200ms (estimation algorithm)
- File lock acquisition: <10ms (Post-MVP - database transaction)
- Context assembly: <100ms (with parallel GNN queries)
- Code generation: 2-5s (LLM dependent, parallel for multiple files)
- GNN validation: <10ms (with parallel dependency path validation)
- Browser validation: 5-10s (single component) / 10-15s (parallel multi-component)
- Security scan: <10s (with parallel file scanning)
- Concurrency validation: <5s (static analysis + GNN race detection)
- File lock release: <5ms (Post-MVP)
- **Total cycle (MVP)**: <43s (adds concurrency validation: ~5s)
- **Total cycle (Post-MVP)**: <45s (adds explicit lock acquisition/release: ~15ms)
- **Parallel optimization**: ~30-40% faster with concurrent execution of independent operations

**New Prevention Guarantees**:

- ✅ **Explicit Planning**: All tasks identified before execution (PDC 2.1)
- ✅ **Work Visibility (MVP)**: Show which developer is working on which files (PDC 3.3)
- ✅ **Conflict Prevention (Post-MVP)**: File locks acquired before any edits (PDC 3.3)
- ✅ **Dependency-Aware Execution**: Tasks executed in correct order (PDC 2.1)
- ✅ **Resource Coordination (Post-MVP)**: No two developers edit same file (PDC 3.3)
- ✅ **Concurrency Safety**: Race conditions detected and auto-fixed before testing (PDC 3.4 - NEW)

#### 2. Testing State Machine (MVP)

**Responsibility**: Ensure generated code works correctly through comprehensive testing

**States**:

- **TestGeneration**: Generate unit and E2E tests using LLM
  - **⚡ Parallel Processing**: Generate tests for multiple files/modules simultaneously with parallel LLM calls
- **EnvironmentSetup**: Create virtual environment, install dependencies
  - **⚡ Parallel Processing**: Install multiple independent test dependencies simultaneously
- **UnitTesting**: Run pytest/jest for function-level tests
  - **⚡ Parallel Processing**:
    - Run test files in parallel (pytest -n auto, jest --maxWorkers)
    - Execute independent test suites simultaneously
    - Performance: N test files × 5s = 5N sequential → ~N/4 parallel (4 workers)
- **BrowserTesting**: Run E2E tests with Playwright (user interactions)
  - **⚡ Parallel Processing**:
    - Run multiple E2E test scenarios in parallel browsers
    - Test different user flows simultaneously
    - Parallel execution across different browser contexts
    - Performance: M scenarios × 30s = 30M sequential → ~30M/P parallel (P = parallel browsers)
- **IntegrationTesting**: Test API integrations and data flows
  - **⚡ Parallel Processing**:
    - Test multiple API endpoints simultaneously
    - Parallel database connection tests
    - Concurrent integration scenario execution
- **CoverageAnalysis**: Measure test coverage percentage
  - **⚡ Parallel Processing**: Analyze coverage for multiple modules concurrently
- **FixingIssues**: Re-run tests after applying fixes
  - **⚡ Parallel Processing**: Fix and re-test multiple independent failures simultaneously
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

- Test generation: 3-5s (LLM, parallel for multiple files)
- Environment setup: 10-20s (with parallel dependency installation)
- Unit tests: <30s for typical project (parallel execution with 4+ workers)
- Browser E2E tests: 30-60s (sequential) / 15-30s (parallel with 2-3 browser contexts)
- Integration tests: 20-40s (with parallel API/DB testing)
- Total cycle: <2 minutes (with parallel optimizations)
- **Parallel benefit**: ~40-50% faster with concurrent test execution

**Auto-Trigger**: Yes - runs immediately after CodeGen completes
**Manual Trigger**: Also available via UI button

#### 3. Deployment State Machine (MVP - Railway Focus)

**Responsibility**: Deploy validated code to Railway.app

**States**:

- **PackageBuilding**: Create Docker image or build artifacts
  - **⚡ Parallel Processing**: Build multiple layers/stages simultaneously (Docker multi-stage builds)
- **ConfigGeneration**: Generate railway.json, Dockerfile, environment config
  - **⚡ Parallel Processing**: Generate multiple config files simultaneously
- **RailwayUpload**: Push code/image to Railway
  - **⚡ Parallel Processing**: Parallel upload of multiple artifacts/layers
- **HealthCheck**: Verify deployed service is responding
  - **⚡ Parallel Processing**:
    - Check multiple endpoints simultaneously
    - Parallel health checks for different service components
    - Concurrent smoke tests across multiple routes
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

- Package building: 30-60s (Docker build with multi-stage parallel builds)
- Upload to Railway: 20-40s (parallel artifact upload)
- Health check: 5-10s (parallel endpoint checks)
- Total deployment: <2 minutes
- **Parallel benefit**: ~20-30% faster with concurrent operations

**Auto-Trigger**: No - requires user approval for safety
**Post-MVP**: Optional auto-deploy after tests pass

---

### Blast Radius Analysis & Preview (P0 Feature - MVP)

**Purpose**: Provide dependency-aware impact preview BEFORE executing changes, enabling informed decision-making and preventing unintended consequences.

**Philosophy**: "Show, don't surprise" - Users should know exactly what will be affected before committing to execution.

**Integration Point**: Code Generation State Machine → PlanReview State (after PlanGeneration, before execution)

#### Data Structure

```rust
pub struct BlastRadiusAnalysis {
    // Direct Impact
    pub files_to_modify: Vec<FileImpact>,
    pub critical_files: Vec<PathBuf>,  // Core business logic identified

    // Indirect Impact (from GNN downstream dependencies)
    pub downstream_dependencies: Vec<DependencyImpact>,
    pub affected_tests: Vec<TestImpact>,

    // External Impact
    pub api_changes: Vec<ApiChange>,
    pub breaking_changes: Vec<BreakingChange>,

    // Package Impact (from Tech Stack GNN)
    pub package_upgrades: Vec<PackageUpgrade>,
    pub package_conflicts: Vec<PackageConflict>,

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
    pub breaking_changes: Vec<String>,  // Known breaking changes from changelog
}
```

#### Analysis Algorithm

```rust
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

    fn identify_critical_files(&self, files: &[PathBuf]) -> Vec<PathBuf> {
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
```

#### UI Display Format

**Compact View (Small Changes):**

```
✅ Low Risk Change
├── 2 files modified
├── 5 tests affected
└── No breaking changes
[Continue] [Details]
```

**Detailed View (Large/Critical Changes):**

```
📊 Blast Radius Preview

Direct Impact (Files to be Modified):
├── src/calculator.py ⚠️ CRITICAL (47 dependents)
├── src/utils.py (3 dependents)
└── tests/test_calculator.py

Indirect Impact (Downstream Dependencies):
├── 12 files import modified code:
│   ├── src/api/endpoints.py (Level 1 - direct import)
│   ├── src/services/math_service.py (Level 1)
│   ├── src/reports/generator.py (Level 2 - indirect)
│   └── ... (9 more - click to expand)

External Impact:
├── 2 API endpoints will change:
│   ├── POST /api/calculate ⚠️ BREAKING CHANGE
│   │   └── Response schema modified (added "precision" field)
│   └── GET /api/health ✅ Non-breaking (added "version")
├── 47 tests need updating:
│   ├── 23 unit tests (calculator, utils)
│   ├── 18 integration tests (API, services)
│   └── 6 E2E tests (full workflows)

Package Dependencies:
├── numpy: 1.24.0 → 1.26.0 ⚠️ UPGRADE REQUIRED
│   └── Breaking changes: numpy.array default behavior changed
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
```

#### Smart Display Logic

```rust
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
```

#### Performance Targets

- **GNN Queries**: <500ms (parallel queries for dependents, tests, packages)
- **Critical File Detection**: <100ms (in-memory checks + config lookup)
- **API Change Analysis**: <300ms (if OpenAPI spec exists, otherwise skip)
- **Package Impact Analysis**: <1s (query Tech Stack GNN + check changelogs)
- **Total Analysis Time**: <2s (all operations)

#### Future Enhancements (Post-MVP)

**P1:** Live validation during DependencyValidation state (compare planned vs actual impact)
**P2:** Historical blast radius tracking (learn from past changes)
**P2:** ML-based user impact estimation (better than heuristic %)
**P3:** Blast radius diff view (compare before/after architecture changes)

---

### Parallel Processing Implementation Guide

#### Overview

Yantra implements parallel processing at three levels:

1. **State-Level Parallelism**: Within a single state, execute independent operations concurrently
2. **Machine-Level Parallelism**: Multiple state machines running simultaneously (Post-MVP)
3. **Cross-Machine Parallelism**: Different machines processing different features (Future)

#### State-Level Parallel Processing Patterns

**Pattern 1: Parallel Resource Access (Browser Tabs, API Calls)**

```
Example: DependencyAssessment State
├─ Tab 1: PyPI official page → Extract version, release date
├─ Tab 2: Changelog/release notes → Extract breaking changes
├─ Tab 3: Migration documentation → Extract upgrade guide
└─ Tab 4: GitHub issues → Extract known problems

Consolidation: LLM synthesizes all information into recommendation
Performance: 12s sequential → 4-5s parallel (60% reduction)
```

**Pattern 2: Parallel Validation (Security, Tests, Health Checks)**

```
Example: SecurityScanning State
├─ Worker 1: Semgrep scan on files 1-10
├─ Worker 2: Semgrep scan on files 11-20
├─ Worker 3: Secrets detection (all files)
└─ Worker 4: CVE database checks (all dependencies)

Consolidation: Merge results, prioritize by severity
Performance: N files × 1s = N sequential → N/4 parallel (with 4 workers)
```

**Pattern 3: Parallel Generation (Code, Tests, Configs)**

```
Example: CodeGeneration State
├─ LLM Call 1: Generate model.py (independent file)
├─ LLM Call 2: Generate controller.py (independent file)
├─ LLM Call 3: Generate test_model.py (independent test)
└─ LLM Call 4: Generate test_controller.py (independent test)

Dependencies: Files with dependencies generated sequentially after independent ones
Performance: 4 files × 5s = 20s sequential → ~10s parallel (2 concurrent LLM calls)
```

**Pattern 4: Parallel Testing (Unit, E2E, Integration)**

```
Example: UnitTesting State
├─ Test Runner 1: tests/unit/test_auth.py
├─ Test Runner 2: tests/unit/test_database.py
├─ Test Runner 3: tests/unit/test_api.py
└─ Test Runner 4: tests/unit/test_models.py

Tool: pytest -n auto (uses pytest-xdist for parallel execution)
Performance: 100 tests × 0.5s = 50s sequential → ~15s parallel (4 workers)
```

#### Implementation Technologies

**Rust Async/Await (Tokio)**

- Primary concurrency model for all parallel operations
- Lightweight tasks (green threads) for high concurrency
- Non-blocking I/O for network/file operations

```rust
// Example: Parallel browser tab execution
async fn fetch_dependency_info(package: &str) -> Result<DependencyInfo> {
    let (pypi_info, changelog, migration, issues) = tokio::join!(
        fetch_pypi_page(package),       // Tab 1
        fetch_changelog(package),        // Tab 2
        fetch_migration_guide(package),  // Tab 3
        fetch_github_issues(package),    // Tab 4
    );

    consolidate_results(pypi_info?, changelog?, migration?, issues?)
}
```

**Connection Pooling**

- HTTP connection pool for API calls (reduces connection overhead)
- Database connection pool for concurrent queries
- Browser context pool for parallel testing

**Rate Limiting**

- Respect API rate limits (e.g., GitHub, PyPI)
- Backoff strategy for LLM calls (avoid 429 errors)
- Circuit breaker pattern for failing services

#### Parallel Processing Trade-offs

| Operation Type           | Sequential Time | Parallel Time | Workers      | Trade-off                 |
| ------------------------ | --------------- | ------------- | ------------ | ------------------------- |
| Web Search (4 tabs)      | 12s             | 4-5s          | 4 tabs       | Memory: +400MB per tab    |
| Security Scan (20 files) | 20s             | 5s            | 4 workers    | CPU: +300% during scan    |
| Unit Tests (100 tests)   | 50s             | 15s           | 4 workers    | Memory: +200MB per worker |
| E2E Tests (5 flows)      | 150s            | 50s           | 3 browsers   | Memory: +1GB per browser  |
| LLM Code Gen (4 files)   | 20s             | 10s           | 2 concurrent | Cost: 2x API calls/min    |

**Resource Limits (MVP)**:

- Browser tabs: Max 5 concurrent (memory constraint)
- Test workers: Max 4 concurrent (CPU constraint)
- LLM calls: Max 3 concurrent (rate limit constraint)
- File scanners: Max 4 concurrent (I/O constraint)

**Monitoring Parallel Operations**:

- Track resource usage (CPU, memory, network)
- Log start/end timestamps for each parallel task
- Detect and retry failed parallel operations
- UI progress indicators show parallel task status

#### When NOT to Use Parallel Processing

1. **Data Dependencies**: Task B depends on Task A's output → Must be sequential
2. **Resource Contention**: Same file being read/written → Use locks or sequential access
3. **Small Operations**: Overhead > benefit (e.g., <10ms operations)
4. **Rate Limited APIs**: Would trigger rate limits → Use sequential with delays
5. **Stateful Operations**: Order matters (e.g., database migrations) → Sequential only

#### Concurrency Safety Validation (Built-in Protection)

**Problem**: Parallel processing can introduce race conditions, deadlocks, and data races if not properly managed.

**Solution**: Automated concurrency validation as part of the agentic flow (ConcurrencyValidation state).

**Three-Layer Detection Strategy:**

**Layer 1: Static Analysis (Automated - No LLM Required)**

```
Language-Specific Linters:
├─ Rust: cargo clippy --all -- -D warnings (thread safety built into type system)
│  └─ Detects: Send/Sync violations, missing Mutex, Arc misuse
├─ Python: threading analyzer (Python 3.12+)
│  └─ Detects: GIL contention, shared mutable state, missing locks
└─ JavaScript: ESLint concurrency rules
   └─ Detects: Promise race conditions, callback hell, async/await misuse

Pattern Detection (Semgrep custom rules):
├─ Shared mutable state without locks
├─ Check-then-act patterns (time-of-check-time-of-use)
├─ Missing synchronization on concurrent writes
└─ Circular wait conditions (potential deadlocks)
```

**Layer 2: GNN Data Flow Analysis (Automated - No LLM Required)**

```
GNN Queries:
├─ Find all paths between concurrent writes to same variable
├─ Detect read-write conflicts across async boundaries
├─ Identify shared resources accessed from multiple threads/tasks
└─ Calculate happens-before relationships

Example Race Detection:
┌──────────────────────────────────────────────────────────┐
│ Thread 1: read counter → increment → write counter       │
│ Thread 2: read counter → increment → write counter       │
│                                                           │
│ GNN detects: Both threads read same variable without lock│
│ Race condition: Lost updates (counter++ not atomic)      │
└──────────────────────────────────────────────────────────┘
```

**Layer 3: Stress Test Generation (Automated - LLM Assisted)**

````

Automatically generate tests that:
├─ Run concurrent operations 1000+ times
├─ Use different thread/task interleavings
├─ Introduce random delays to expose races
└─ Assert on invariants (e.g., counter == expected_value)

Example Generated Test (Python):

```python
def test_concurrent_counter_race():
    counter = Counter()  # Class being tested
    threads = [Thread(target=counter.increment) for _ in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert counter.value == 100  # Should fail if race exists
```

```

**Auto-Fix Strategy (Tiered Approach):**

```

Level 1: Pattern-Based Auto-Fix (No LLM - Fast)
├─ Detected: Shared mutable state without lock
│ └─ Fix: Wrap in Mutex `<T>` (Rust) or threading.Lock (Python)
├─ Detected: Check-then-act pattern
│ └─ Fix: Use atomic operations or lock entire section
└─ Detected: Missing Arc for shared data
└─ Fix: Wrap in Arc<Mutex `<T>`> for cross-thread sharing

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

```

**Integration with State Machine:**

```

CodeGeneration State
↓
(generates parallel code)
↓
DependencyValidation State
↓
BrowserValidation State
↓
SecurityScanning State
↓
ConcurrencyValidation State ← NEW (Mandatory for parallel code)
├─ Static analysis (<1s)
├─ GNN race detection (<2s)
├─ Stress test generation (<2s)
└─ Auto-fix if issues found
↓
(all validations pass)
↓
Complete State

```

**Performance Impact:**
- Static analysis: ~1s (parallel execution of linters)
- GNN analysis: ~2s (parallel data flow queries)
- Test generation: ~2s (LLM generates stress tests)
- **Total: ~5s** (acceptable overhead for safety guarantee)
- Only runs if code uses async/await, threads, or parallel patterns

**Skip Conditions (Optimization):**
- No async/await keywords detected
- No threading/multiprocessing imports
- No tokio::spawn or similar parallel primitives
- Pure sequential code → Skip validation (0s overhead)

**User Experience:**
```

[Agent]: "Analyzing parallel code for race conditions..."
[Agent]: "⚠️ Detected potential race: Counter incremented without lock"
[Agent]: "🔧 Auto-fixing: Adding Mutex to protect shared state"
[Agent]: "✅ Fix validated with 1000 concurrent operations"
[Agent]: "⏱️ Concurrency validation: 4.2s"

```

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

**Future Enhancements:**
- Dynamic race detection (ThreadSanitizer, Valgrind integration)
- Record-replay debugging for non-deterministic races
- Formal verification for critical sections (TLA+, model checking)
- ML-based race prediction from code patterns

---

#### Future Enhancements (Post-MVP)

**Machine-Level Parallelism**:

- CodeGen + Testing machines running simultaneously on different features
- Deploy machine running while CodeGen works on next feature
- Maintenance machine continuously monitoring while other machines work

**Cross-Machine Parallelism**:

- Multiple developers, multiple features, all machines working concurrently
- Team of agents architecture (master-servant pattern)
- Git coordination branch for conflict-free parallel work

**Advanced Parallel Patterns**:

- Map-reduce for large codebases (analyze 1000+ files)
- Stream processing for continuous monitoring
- Distributed execution across multiple machines

---

#### 4. Maintenance State Machine (Post-MVP)

**Responsibility**: Monitor production, detect issues, auto-fix, and deploy patches

**States**:

- **LiveMonitoring**: Continuous monitoring of production errors and performance
  - **⚡ Parallel Processing**:
    - Monitor multiple services/endpoints simultaneously
    - Parallel log aggregation from multiple sources
    - Concurrent metric collection across service fleet
- **BrowserValidation**: Monitor real user browser sessions for errors
  - **⚡ Parallel Processing**:
    - Track multiple user sessions concurrently
    - Parallel session replay processing
    - Concurrent error pattern detection across users
- **ErrorAnalysis**: Pattern detection in production errors
  - **⚡ Parallel Processing**:
    - Analyze multiple error types simultaneously
    - Parallel pattern matching across error logs
    - Concurrent root cause analysis for different error classes
- **IssueDetection**: Identify root causes using error patterns
  - **⚡ Parallel Processing**: Query multiple data sources simultaneously (logs, metrics, traces)
- **AutoFixGeneration**: Generate fix using LLM + historical patterns
  - **⚡ Parallel Processing**: Generate multiple fix candidates simultaneously, evaluate in parallel
- **FixValidation**: Test fix in staging using CodeGen + Testing machines
  - **⚡ Parallel Processing**: Leverages parallel processing from CodeGen and Testing machines
- **CICDPipeline**: Automated deployment of validated fix
  - **⚡ Parallel Processing**: Leverages parallel processing from Deployment machine
- **VerificationCheck**: Confirm issue is resolved in production
  - **⚡ Parallel Processing**:
    - Check error rates across multiple regions simultaneously
    - Parallel verification across different service instances
    - Concurrent user impact assessment
- **LearningUpdate**: Update knowledge base with new patterns
  - **⚡ Parallel Processing**: Update multiple knowledge stores concurrently (Vector DB, documentation, metrics)
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

- Error detection: Real-time (<1s, parallel monitoring across services)
- Pattern analysis: <5s (parallel pattern matching)
- Fix generation: 10-30s (LLM, parallel candidate generation)
- Fix validation: 2-3 minutes (full test suite with parallel execution)
- Deployment: 1-2 minutes (parallel deployment pipeline)
- Verification: <30s (parallel checks across regions)
- Total MTTR (Mean Time To Repair): <5 minutes for known patterns
- **Parallel benefit**: ~50-60% faster incident resolution with concurrent operations

**MVP Status**: NOT in MVP - design complete, implementation post-MVP
**Rationale**: Focus MVP on working code generation/testing/deployment first

---

---

## Preventive Development Cycle ↔ State Machine Reconciliation

### Executive Summary

The **Preventive Development Cycle (PDC)** defines Yantra's **preventive philosophy** and **what** to prevent at each phase. The **State Machines** provide the **concrete implementation architecture** that executes the PDC through specific states, transitions, and validation checkpoints.

**Reconciliation Document Reference**: `docs/Research & specifications/*Preventive Development Cycle.md` (1,244 lines)

**🆕 Recent Enhancements (December 2025):**

- Added 6 explicit Planning states for PDC Phase 2 (TaskDecomposition, DependencySequencing, ConflictCheck, PlanGeneration, PlanReview, EnvironmentSetup)
- **MVP Approach**: Work visibility for conflict minimization (show who's working on what files)
- **Post-MVP**: Explicit File Lock states (FileLockAcquisition, FileLockRelease) for conflict prevention
- Code Generation state count: MVP: 16 states, Post-MVP: 18 states (+50% from original 12)
- Added optional PlanReview approval gate for complex features (>5 tasks)
- Total cycle time: MVP: <38s, Post-MVP: <40s (planning overhead: ~850ms, locking overhead: ~15ms)
- Added prevention guarantee 3.5: Plans are Explicit and Validated
- **⚡ Parallel Processing**: All state machines leverage concurrent execution for 30-60% performance improvement
  - Web search: 4 tabs in parallel (12s → 4-5s)
  - Security scanning: 4 workers in parallel (20s → 5s)
  - Test execution: 4 workers in parallel (50s → 15s)
  - Browser E2E: 3 browsers in parallel (150s → 50s)

**Strategic Decision - MVP vs Post-MVP Locking:**

- **MVP Focus**: Show visibility of active work (which developer works on which files) via UI indicators
- **Rationale**: Visibility enables team coordination without complex locking infrastructure
- **Post-MVP**: Explicit file locking states with GNN-aware dependency locks for complete conflict prevention
- **Benefit**: Ship faster with practical conflict minimization, add enforcement layer later

**Key Integration Points:**

| Integration Layer  | PDC Element              | State Machine Element                                         | Reconciliation Strategy                                                                         |
| ------------------ | ------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Philosophy**     | Prevention over reaction | Fail-fast state transitions                                   | Violations detected at state boundaries, not after completion                                   |
| **Architecture**   | 5 PDC Phases             | 4 State Machines                                              | Phases 1-3 map to CodeGen+Testing, Phase 4 to Deployment, Phase 5 to Maintenance                |
| **Validation**     | 6-layer bug prevention   | Multi-state validation pipeline                               | Each layer maps to specific state with <10ms-60s execution time                                 |
| **Approval**       | Human-in-loop gates      | ArchitectureReview, PlanReview (optional), Complete states 🆕 | 3 approval gates at critical state boundaries (architecture, plan for complex features, deploy) |
| **Traceability**   | Full audit trail         | Session linking across machines                               | Each session stores parent session ID, full lineage from intent to production                   |
| **Recovery**       | Rollback on failure      | State persistence + checkpoints                               | SQLite persistence at state boundaries, resume from last checkpoint                             |
| **Performance** ⚡ | Fast execution           | Parallel processing within states                             | Independent operations executed concurrently (30-60% faster) using Tokio async/await            |

**Design Decision**: PDC phases don't map 1:1 to state machines. Instead, **PDC checks are distributed across states** for optimal performance and granular prevention.

**Example (MVP)**: PDC Phase 3 "Execute" spans **13 states** across 2 machines (CodeGen: 6 states, Testing: 7 states) to enable fail-fast validation and parallel execution.

---

### Quick Reference: PDC Phases → State Machines

```

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
└─> CodeGen States (MVP): ContextAssembly, CodeGeneration (with work visibility), DependencyValidation,
BrowserValidation, SecurityScanning, FixingIssues
└─> Prevents: Syntax/type/logic errors, breaking changes, security issues
└─> Work Visibility: UI shows which files are being modified and by whom
└─> Auto-triggers Testing Machine on Complete
└─> CodeGen States (Post-MVP): FileLockAcquisition, ContextAssembly, CodeGeneration, DependencyValidation,
BrowserValidation, SecurityScanning, FixingIssues, FileLockRelease
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

```

**Key Relationships:**

- 📊 **1 PDC Phase** → **1-9 States** (granular prevention) 🆕 increased from 1-7
- ⚡ **1 State** → **Multiple PDC Checks** (efficient execution)
- 🔗 **4 State Machines** → **5 PDC Phases** (distributed intelligence)
- ⚠️ **3 Approval Gates** → **Architecture + Plan (optional) + Deployment** 🆕 added Plan approval
- ✅ **10 Prevention Guarantees** → **MVP: 24 States / Post-MVP: 26 States** 🆕 increased from 17, adds ConcurrencyValidation
  - MVP: Work visibility + concurrency validation (17 CodeGen + 7 Testing = 24 non-overlapping)
  - Post-MVP: Adds FileLockAcquisition and FileLockRelease states (19 CodeGen + 7 Testing = 26 non-overlapping)

---

### Overview

The **Preventive Development Cycle (PDC)** is the conceptual framework that defines **what** Yantra prevents and **how** it prevents problems at each phase. The **State Machines** are the **implementation architecture** that executes the PDC through concrete states and transitions.

**Relationship**: PDC phases map to State Machine execution flows with specific preventive checks at each state.

### Phase-to-Machine Mapping

| PDC Phase               | State Machine(s)                                              | Purpose                                          | Key Prevention                                                                                    |
| ----------------------- | ------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **1. Architect/Design** | Code Generation (states 1-4)                                  | Validate architecture before code                | Boundary violations, circular deps, security by design                                            |
| **2. Plan**             | Code Generation (states 5-10) 🆕 Enhanced                     | Executable plan prevents implementation problems | Missing tasks, wrong sequencing, scope creep; MVP: work visibility, Post-MVP: locks               |
| **3. Execute**          | Code Generation (states 11-16 MVP / 11-18 Post-MVP) + Testing | Write correct code with tests                    | MVP: work visibility, syntax/type/logic errors; Post-MVP: adds explicit merge conflict prevention |
| **4. Deploy**           | Deployment Machine (states 1-5)                               | Safe, validated, recoverable deployments         | Broken code deploys, environment mismatches, partial deployments                                  |
| **5. Monitor/Maintain** | Maintenance Machine (states 1-11) Post-MVP                    | Detect and heal before users notice              | Prolonged outages, repeated incidents, manual delays                                              |

---

### Detailed State-by-State PDC Integration

#### Code Generation State Machine → PDC Phases 1-3

| State                                 | PDC Phase    | PDC Sub-Phase                   | Prevents                                                              | Tools/Checks                                                                                                                                                                           |
| ------------------------------------- | ------------ | ------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ArchitectureGeneration**            | 1. Architect | 1.1 Architecture Alignment      | Module boundary violations, circular dependencies                     | GNN dependency graph, Architecture View, module boundaries                                                                                                                             |
| **ArchitectureReview**                | 1. Architect | 1.5 Architecture Sign-off       | Unauthorized changes, lost decision context                           | Approval Queue (⚠️ mandatory human approval), ADR generator, audit trail                                                                                                               |
| **DependencyAssessment**              | 1. Architect | 1.2 Tech Stack Alignment        | Incompatible versions, duplicate functionality, license conflicts     | npm/pip/cargo registry (MCP), CVE database, license checker; **Web search** (latest package docs, LLM knowledge static); **Semantic graph** (HNSW index for conflict resolution, cached solutions) |
| **ContextAssembly**                   | 1. Architect | 1.3 Existing Code Analysis      | Reinventing functionality, inconsistent patterns, missing context     | Semantic embeddings, tree-sitter, GNN                                                                                                                                                  |
| **TaskDecomposition** 🆕              | 2. Plan      | 2.1 Execution Plan (tasks)      | Missing tasks, unbounded work, unclear scope                          | GNN-based feature decomposition (<500ms)                                                                                                                                               |
| **DependencySequencing** 🆕           | 2. Plan      | 2.1 Execution Plan (ordering)   | Wrong task order, dependency violations, parallel conflicts           | GNN graph traversal, topological sort (<100ms)                                                                                                                                         |
| **ConflictCheck** 🆕                  | 2. Plan      | 2.1 Execution Plan (visibility) | Parallel edit conflicts, work coordination issues                     | MVP: Active work lookup (<50ms); Post-MVP: File lock table query (<50ms)                                                                                                               |
| **PlanGeneration** 🆕                 | 2. Plan      | 2.1 Execution Plan (finalize)   | Unclear scope, no estimates, unprioritized work                       | Estimation algorithm, complexity analysis (<200ms)                                                                                                                                     |
| **PlanReview** 🆕                     | 2. Plan      | 2.3 Plan Sign-off               | Misaligned expectations, scope disagreements, over-complexity         | Approval Queue (⚠️ optional for >5 tasks or multi-file changes)                                                                                                                        |
| **EnvironmentSetup**                  | 2. Plan      | 2.1 Execution Plan (prep)       | Environment configuration errors, missing dependencies                | Virtual environment setup, dependency installer, version validator                                                                                                                     |
| **FileLockAcquisition** 🆕 (Post-MVP) | 3. Execute   | 3.3 Prevent Conflicts (before)  | Merge conflicts, parallel edit conflicts, lost work                   | Database transaction, GNN-aware lock acquisition (<10ms)                                                                                                                               |
| **CodeGeneration**                    | 3. Execute   | 3.1 Code Generation             | Syntax errors (tree-sitter), type errors (LSP), style issues (linter) | Multi-LLM, tree-sitter, LSP, pattern extractor; MVP: UI shows active work                                                                                                              |
| **DependencyValidation**              | 3. Execute   | 3.2 Prevent Bugs (Layer 6)      | Breaking changes to dependents, ripple effects                        | GNN impact analysis (<10ms)                                                                                                                                                            |
| **BrowserValidation**                 | 3. Execute   | 3.1 Code Validation             | Visual regressions, console errors, network failures                  | Chrome CDP, screenshot capture (5-10s)                                                                                                                                                 |
| **SecurityScanning**                  | 3. Execute   | 3.4 Prevent Security Issues     | SQL injection, XSS, secrets in code, vulnerabilities                  | Semgrep, OWASP rules, secrets detector (Gitleaks)                                                                                                                                      |
| **ConcurrencyValidation** 🆕          | 3. Execute   | 3.4 Prevent Concurrency Bugs    | Race conditions, deadlocks, data races from parallel execution        | Static analysis (Clippy, threading analyzer), GNN data flow analysis, stress test generation (<5s); Auto-fix with locks/atomics or sequential fallback                                 |
| **FixingIssues**                      | 3. Execute   | 3.2 Prevent Bugs (auto-retry)   | Issues from previous validations                                      | Auto-retry with LLM-generated fixes                                                                                                                                                    |
| **FileLockRelease** 🆕 (Post-MVP)     | 3. Execute   | 3.3 Prevent Conflicts (after)   | Stale locks, resource leaks, blocked parallel work                    | Lock release transaction (<5ms)                                                                                                                                                        |
| **Complete**                          | 3. Execute   | 3.7 Feature Sign-off (partial)  | Incomplete features                                                   | Confidence score, validation passed; Post-MVP: locks released                                                                                                                          |
| **Failed**                            | 3. Execute   | 3.7 Feature Sign-off (blocked)  | Quality shortcuts                                                     | Human intervention required; Post-MVP: locks released                                                                                                                                  |

#### Testing State Machine → PDC Phase 3 (continued)

| State                  | PDC Phase  | PDC Sub-Phase                     | Prevents                                                                 | Tools/Checks                                      |
| ---------------------- | ---------- | --------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- |
| **TestGeneration**     | 3. Execute | 3.5 Auto Unit & Integration Tests | Missing test coverage, untested edge cases                               | LLM test generation, pattern-based test templates |
| **EnvironmentSetup**   | 3. Execute | 3.1 Code Validation (prep)        | Environment inconsistencies, missing test dependencies                   | Virtual environment, test fixtures                |
| **UnitTesting**        | 3. Execute | 3.2 Prevent Bugs (Layer 4)        | Logic errors, edge case failures, function behavior                      | pytest/jest/cargo test (<30s)                     |
| **BrowserTesting**     | 3. Execute | 3.2 Prevent Bugs (E2E)            | User workflow failures, integration issues, UI bugs                      | Playwright E2E tests (30-60s)                     |
| **IntegrationTesting** | 3. Execute | 3.2 Prevent Bugs (Layer 5)        | Component interaction failures, API contract violations, database issues | Integration test suite (20-40s)                   |
| **CoverageAnalysis**   | 3. Execute | 3.5 Auto Tests (validation)       | Insufficient test coverage, gaps in validation                           | pytest-cov, coverage threshold (>80%)             |
| **FixingIssues**       | 3. Execute | 3.1 Code Generation (retry)       | Test failures                                                            | Re-run tests after applying fixes                 |
| **Complete**           | 3. Execute | 3.7 Feature Sign-off (tests pass) | Quality gates not met                                                    | 100% tests pass, coverage >80%                    |
| **Failed**             | 3. Execute | 3.7 Feature Sign-off (blocked)    | Tests failed after max retries                                           | Human intervention required                       |

#### Deployment State Machine → PDC Phase 4

| State                 | PDC Phase | PDC Sub-Phase                      | Prevents                                       | Tools/Checks                                           |
| --------------------- | --------- | ---------------------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| **PackageBuilding**   | 4. Deploy | 4.1 Pre-Deploy Validation (prep)   | Broken build artifacts, missing dependencies   | Docker build, artifact compilation (30-60s)            |
| **ConfigGeneration**  | 4. Deploy | 4.1 Pre-Deploy Validation (config) | Environment mismatches, missing configuration  | railway.json, Dockerfile, env vars validation          |
| **RailwayUpload**     | 4. Deploy | 4.2 Auto Deploy                    | Manual deploy errors, inconsistent deployments | Railway API (MCP), atomic upload (20-40s)              |
| **HealthCheck**       | 4. Deploy | 4.1 Pre-Deploy Validation (canary) | Deploying broken code, service unavailable     | HTTP health check (5-10s), smoke tests                 |
| **RollbackOnFailure** | 4. Deploy | 4.2 Auto Deploy (safety)           | Partial deployments, prolonged outages         | Automatic rollback to last known good version          |
| **Complete**          | 4. Deploy | 4.3 Deploy Sign-off                | Unauthorized deploys, lost history             | Live Railway URL, health status confirmed, audit trail |
| **Failed**            | 4. Deploy | 4.3 Deploy Sign-off (blocked)      | Deploy failures                                | Deployment failed, human intervention required         |

#### Maintenance State Machine → PDC Phase 5 (Post-MVP)

| State                 | PDC Phase  | PDC Sub-Phase                      | Prevents                                             | Tools/Checks                                                               |
| --------------------- | ---------- | ---------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------- |
| **LiveMonitoring**    | 5. Monitor | 5.1 Self-Healing (detect)          | Prolonged outages, undetected issues                 | Sentry (MCP), Datadog (MCP), real-time error tracking                      |
| **BrowserValidation** | 5. Monitor | 5.1 Self-Healing (user impact)     | Silent user-facing failures, performance degradation | Real User Monitoring (RUM), session replay, browser error tracking         |
| **ErrorAnalysis**     | 5. Monitor | 5.1 Self-Healing (classify)        | Repeated incidents, unknown root causes              | Pattern matching, severity classification, error correlation               |
| **IssueDetection**    | 5. Monitor | 5.1 Self-Healing (diagnose)        | Misdiagnosed issues, wasted effort                   | Root cause analysis, dependency graph queries, historical pattern matching |
| **AutoFixGeneration** | 5. Monitor | 5.1 Self-Healing (fix)             | Manual intervention delays, human error              | LLM fix generation, pattern-based solutions, historical fixes              |
| **FixValidation**     | 5. Monitor | 5.1 Self-Healing (validate)        | Deploying broken fixes, introducing new bugs         | Run through CodeGen + Testing machines (full validation, 2-3 min)          |
| **CICDPipeline**      | 5. Monitor | 5.1 Self-Healing (deploy)          | Manual deploy delays, inconsistent fixes             | Automated CI/CD pipeline (1-2 min)                                         |
| **VerificationCheck** | 5. Monitor | 5.1 Self-Healing (verify)          | Fix didn't work, issue persists                      | Error rate monitoring, production verification                             |
| **LearningUpdate**    | 5. Monitor | 5.1 Self-Healing (learn & prevent) | Repeated incidents, knowledge loss                   | Update Yantra Codex, add monitoring, create post-incident report           |
| **Active**            | 5. Monitor | Normal operation                   | Silent failures, degradation                         | Continuous monitoring, proactive detection                                 |
| **Incident**          | 5. Monitor | 5.1 Self-Healing (active)          | Prolonged outages, manual escalation                 | Active incident handling, auto-escalation if unresolved                    |

---

### PDC Prevention Guarantees → State Machine Implementation

#### Guarantee 1: Architecture is Respected

**PDC**: Phase 1.1 Architecture Alignment
**State Machines**:

- CodeGen: **ArchitectureGeneration** + **ArchitectureReview**
- Implementation: GNN enforces boundaries, detects cycles, Agent won't generate violating code
- Approval Gate: Human-in-loop approval required before proceeding

#### Guarantee 2: Tech Stack is Consistent

**PDC**: Phase 1.2 Tech Stack Alignment
**State Machines**:

- CodeGen: **DependencyAssessment**
- Implementation: Check CVE database, validate versions, detect duplicates, verify licenses
- Blocks: Addition of incompatible/vulnerable dependencies before installation

#### Guarantee 3: Requirements are Clear

**PDC**: Phase 1.4 Feature Extraction
**State Machines**:

- CodeGen: **ContextAssembly** (includes requirement analysis)
- Implementation: NLP extraction from PRDs/Notion/Slack, clarification dialogue before generation
- Prevents: Ambiguous requirements, missing edge cases, scope creep

#### Guarantee 3.5: Plans are Explicit and Validated 🆕

**PDC**: Phase 2.1 Execution Plan + Phase 2.3 Plan Sign-off
**State Machines**:

- CodeGen: **TaskDecomposition**, **DependencySequencing**, **ConflictCheck**, **PlanGeneration**, **PlanReview** (optional approval)
- Implementation: GNN-based task breakdown, topological sorting, work visibility (MVP) or lock queries (Post-MVP), complexity estimation
- Prevents: Missing tasks, wrong task order, unbounded work, scope disagreements
- Approval Gate: Optional PlanReview for >5 tasks or multi-file changes ensures alignment before execution

#### Guarantee 4: Code is Correct

**PDC**: Phase 3.2 Prevent Bugs (6 layers)
**State Machines**:

- CodeGen: **CodeGeneration** (tree-sitter, LSP), **DependencyValidation** (GNN), **BrowserValidation** (CDP), **SecurityScanning** (Semgrep)
- Testing: **UnitTesting**, **BrowserTesting**, **IntegrationTesting**
- Implementation: Multi-layer validation before commit, affected tests run automatically

#### Guarantee 5: Conflicts are Minimized (MVP) / Impossible (Post-MVP)

**PDC**: Phase 3.3 Prevent Merge Conflicts
**State Machines**:

- **MVP**: **ConflictCheck** (planning) + **CodeGeneration** (with work visibility in UI)
  - Implementation: Show which developer is working on which files, active work indicators
  - Result: Developers can coordinate to avoid parallel edits on same files
- **Post-MVP**: **FileLockAcquisition** (before edits) + **FileLockRelease** (after completion) - Explicit States
  - Implementation: Dependency-aware locking via GNN, one task per person, lock table enforced by system
  - Result: Merge conflicts prevented by design, not resolved; explicit state transitions make locking auditable

#### Guarantee 6: Security is Built-in

**PDC**: Phase 3.4 Prevent Security Issues (5 layers)
**State Machines**:

- CodeGen: **DependencyAssessment** (CVE check), **SecurityScanning** (Semgrep, secrets detection)
- Implementation: Block vulnerable dependencies, scan code patterns, detect secrets pre-commit
- Testing: Security-specific tests generated automatically

#### Guarantee 7: Documentation is Accurate

**PDC**: Phase 1.6 + 3.6 Automated Documentation
**State Machines**:

- CodeGen: Documentation generated post-**Complete** state
- Implementation: Generated from Architecture View, code analysis, ADRs, always current
- Prevents: Outdated docs, doc drift from reality

#### Guarantee 8: Deployments are Safe

**PDC**: Phase 4.1 + 4.2 Pre-Deploy Validation + Auto Deploy
**State Machines**:

- Deployment: **PackageBuilding**, **ConfigGeneration**, **HealthCheck**, **RollbackOnFailure**
- Implementation: Full test suite passes, config validated, health checks, atomic rollback
- Prevents: Deploying broken code, environment mismatches, partial deployments

#### Guarantee 9: Systems Self-Heal

**PDC**: Phase 5.1 Self-Healing
**State Machines**:

- Maintenance (Post-MVP): **ErrorAnalysis** → **AutoFixGeneration** → **FixValidation** → **CICDPipeline** → **VerificationCheck**
- Implementation: Detect (<1s), analyze (<5s), fix (10-30s), validate (2-3min), deploy (1-2min), verify
- Result: Mean Time To Repair (MTTR) < 5 minutes for known patterns

#### Guarantee 10: Concurrency is Safe 🆕

**PDC**: Phase 3.4 Prevent Concurrency Bugs
**State Machines**:

- CodeGen: **ConcurrencyValidation** (after SecurityScanning, before FixingIssues)
- Implementation:
  - Static analysis (Clippy, threading analyzers) detects unsafe patterns
  - GNN data flow analysis identifies race conditions
  - Stress test generation validates concurrent code paths
  - Three-tier auto-fix: Pattern-based → LLM-assisted → Sequential fallback
- Prevents: Race conditions, deadlocks, data races from parallel execution
- Guarantee: Code either passes concurrency validation OR falls back to safe sequential execution
- Performance: <5s validation overhead (only for code using parallelism)

---

### State Machine Communication

#### Sequential Flow (Default)

```

User Intent
↓
┌─────────────────────┐
│ CodeGen Machine │ → Generated Code + Confidence Score
└─────────────────────┘
↓ (auto-trigger)
┌─────────────────────┐
│ Testing Machine │ → Test Results + Coverage
└─────────────────────┘
↓ (manual approval)
┌─────────────────────┐
│ Deployment Machine │ → Live Railway URL
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

### PDC-to-State-Machine Visual Flow

```

┌─────────────────────────────────────────────────────────────────────────────┐
│ PREVENTIVE DEVELOPMENT CYCLE │
│ (What to Prevent + When to Prevent) │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PDC PHASE 1: ARCHITECT/DESIGN (Prevention: Architecture Violations) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1.1 Architecture Alignment → [ArchitectureGeneration State] │
│ 1.2 Tech Stack Alignment → [DependencyAssessment State] │
│ 1.3 Existing Code Analysis → [ContextAssembly State] │
│ 1.4 Feature Extraction → [ContextAssembly State] │
│ 1.5 Architecture Sign-off → [ArchitectureReview State] ⚠️ APPROVAL GATE │
│ 1.6 Auto Documentation → [Complete State] → Doc Generation │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌───────────────────────┐
│ CODE GENERATION STATE │
│ MACHINE 1 │
└───────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PDC PHASE 2: PLAN (Prevention: Missing Tasks, Wrong Sequencing, Conflicts) 🆕│
├─────────────────────────────────────────────────────────────────────────────┤
│ 2.1 Execution Plan → [TaskDecomposition State] 🆕 │
│ → [DependencySequencing State] 🆕 │
│ → [ConflictCheck State] 🆕 │
│ → [PlanGeneration State] 🆕 │
│ → [EnvironmentSetup State] │
│ 2.2 Progress Tracking → [All States] → Real-time status updates │
│ 2.3 Plan Sign-off → [PlanReview State] ⚠️ OPTIONAL APPROVAL GATE 🆕 │
│ (triggered for >5 tasks or multi-file changes) │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PDC PHASE 3: EXECUTE (Prevention: Bugs, Conflicts, Security Issues) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3.3 Prevent Conflicts → MVP: [CodeGeneration State with Work Visibility]│
│ • UI shows which developer is working on which files (active indicators) │
│ • Developers coordinate manually to avoid parallel edits │
│ → Post-MVP: [FileLockAcquisition State] 🆕 │
│ • Dependency-aware locks, GNN-based lock acquisition enforced by system │
│ 3.1 Code Generation + Tests → [CodeGeneration State] │
│ • Layer 1: Syntax → Tree-sitter validation │
│ • Layer 2: Types → LSP type checking │
│ • Layer 3: Patterns → Linter (style, common bugs) │
│ 3.2 Prevent Bugs → [DependencyValidation State] │
│ • Layer 4: Logic → Generated with code │
│ • Layer 5: Integration → To be run in Testing Machine │
│ • Layer 6: Impact → GNN dependency graph checks │
│ 3.4 Prevent Security → [SecurityScanning State] │
│ • Layer 1: Dependencies → Already checked in DependencyAssessment │
│ • Layer 2: Code Patterns → Semgrep, OWASP rules │
│ • Layer 3: Secrets → Gitleaks detection │
│ • Layer 4: Security Tests → Generated with unit tests │
│ • Layer 5: Static Analysis → Semgrep, Bandit │
│ 3.3 Prevent Conflicts → Post-MVP: [FileLockRelease State] 🆕 │
│ • Release all locks on Complete or Failed, prevent resource leaks │
│ 3.5 Auto Tests → [TestGeneration State] (next machine) │
│ 3.6 Implementation Docs → [Complete State] → Doc Generation │
│ 3.7 Feature Sign-off → [Complete State] ⚠️ QUALITY GATE │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌───────────────────────┐
│ TESTING STATE │
│ MACHINE 2 │
└───────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3.5 Auto Tests (continued) → [Testing Machine States] │
│ • TestGeneration → LLM generates unit, integration, E2E tests │
│ • EnvironmentSetup → Isolated test environment │
│ • UnitTesting → pytest/jest/cargo test (Layer 4 validation) │
│ • BrowserTesting → Playwright E2E tests (user workflows) │
│ • IntegrationTesting → API/DB integration tests (Layer 5 validation) │
│ • CoverageAnalysis → Ensure >80% coverage │
│ • Complete → All tests pass ✅ │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PDC PHASE 4: DEPLOY (Prevention: Broken Deploys, Environment Mismatches) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4.1 Pre-Deploy Validation → [PackageBuilding, ConfigGeneration States] │
│ • Test suite check → Already passed in Testing Machine │
│ • Security scan → Already done in CodeGen Machine │
│ • Config check → Environment variables, secrets validation │
│ • Migration check → Database migration validation │
│ • Canary tests → HealthCheck State │
│ 4.2 Auto Deploy → [RailwayUpload State] │
│ • Atomic deployment, same process every time │
│ 4.3 Deploy Sign-off → [Complete State] ⚠️ APPROVAL GATE + AUDIT │
│ • User approval required (manual trigger for safety) │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌───────────────────────┐
│ DEPLOYMENT STATE │
│ MACHINE 3 │
└───────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PDC PHASE 5: MONITOR/MAINTAIN (Prevention: Outages, Repeated Incidents) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5.1 Self-Healing → [Maintenance Machine States] (Post-MVP) │
│ • LiveMonitoring → Detect issues in real-time (<1s) │
│ • BrowserValidation → Real User Monitoring (RUM), session replay │
│ • ErrorAnalysis → Pattern matching, severity classification │
│ • IssueDetection → Root cause analysis, dependency graph queries │
│ • AutoFixGeneration → LLM generates fix (10-30s) │
│ • FixValidation → Run through CodeGen + Testing (2-3min) │
│ • CICDPipeline → Automated deployment of fix (1-2min) │
│ • VerificationCheck → Confirm error rate drops │
│ • LearningUpdate → Update Yantra Codex, add monitoring │
│ • Active/Incident → Continuous monitoring or active healing │
│ Result: MTTR < 5 minutes for known patterns │
└─────────────────────────────────────────────────────────────────────────────┘
↓
┌───────────────────────┐
│ MAINTENANCE STATE │
│ MACHINE 4 │
│ (❌ Post-MVP) │
└───────────────────────┘

````

**Key Symbols:**

- ⚠️ **APPROVAL GATE**: Human-in-loop approval required before proceeding
- ⚠️ **QUALITY GATE**: Automated checks must pass (no human approval, but blocks progress)
- ✅ **CHECKPOINT**: State completion milestone

**State Machine Advantages for PDC Execution:**

1. **Granular Prevention**: Each state implements specific PDC checks
2. **Fail-Fast**: Violations detected immediately at state boundaries
3. **Rollback Points**: Can rollback to previous state if validation fails
4. **Audit Trail**: Every PDC check is logged with timestamps
5. **Parallel Execution**: Multiple PDC phases can run concurrently (future)
6. **Recovery**: Crash recovery at state boundaries, not mid-check

---

---

### Implementation Guide: PDC in State Machines

#### For Developers Implementing State Machine Logic

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
````

#### For Developers Adding PDC Checks

**When adding a new preventive check (from PDC document):**

1. **Identify the PDC phase and sub-phase**
   - Example: PDC 1.2 Tech Stack Alignment → Check incompatible library versions
2. **Find or create the appropriate state**
   - If check fits existing state → Add to that state's validation
   - If check is substantial → Create new state (e.g., `LicenseValidation`)
3. **Determine tool/protocol (refer to UTI)**
   - Builtin: Core differentiators, performance-critical
   - MCP: External services, well-maintained servers
   - Reference: UTI specification for protocol decisions
4. **Add performance target**
   - Define expected execution time (e.g., <10ms, <1s, <30s)
   - Add to performance targets table in state machine section
5. **Update reconciliation table**
   - Add row to "Detailed State-by-State PDC Integration" table above
   - Document what this check prevents

**Example: Adding "License Conflict Detection" (PDC 1.2)**

```rust
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
```

#### For Users Understanding the System

**What happens when I submit a task:**

1. **PDC Phase 1: Architect** → States: ArchitectureGeneration, ArchitectureReview, DependencyAssessment
   - System validates your request against architecture
   - Checks if tech stack additions are safe
   - You approve architecture (if changes needed)
   - **Prevention**: Wrong patterns, circular dependencies, incompatible libraries
2. **PDC Phase 2: Plan** → State: EnvironmentSetup
   - System creates execution plan
   - Sets up isolated environment
   - **Prevention**: Missing dependencies, environment issues
3. **PDC Phase 3: Execute** → States: CodeGeneration, DependencyValidation, BrowserValidation, SecurityScanning, TestGeneration, UnitTesting, BrowserTesting, IntegrationTesting
   - System generates code with multi-layer validation
   - Tests generated and executed automatically
   - **Prevention**: Bugs, breaking changes, security issues, conflicts
4. **PDC Phase 4: Deploy** → States: PackageBuilding, ConfigGeneration, RailwayUpload, HealthCheck
   - You approve deployment (manual trigger)
   - System deploys with health checks
   - Auto-rollback if issues detected
   - **Prevention**: Broken deployments, environment mismatches
5. **PDC Phase 5: Monitor** → States: LiveMonitoring, ErrorAnalysis, AutoFixGeneration, FixValidation, CICDPipeline (Post-MVP)
   - System monitors production continuously
   - Auto-fixes issues before you notice
   - **Prevention**: Prolonged outages, repeated incidents

**Visual Progress**: Three progress bars show state machine progress in real-time

#### Common Questions

**Q: Why are PDC phases split across multiple states?**
A: Granular states enable fail-fast validation. Example: PDC Phase 3 "Execute" has 12 states to catch issues early (syntax errors in 5ms before attempting type checking).

**Q: Can I skip states?**
A: No, for safety. All preventive checks must run. However, you can re-trigger individual machines (e.g., re-run tests without regenerating code).

**Q: What happens if a state fails?**
A: State transitions to `FixingIssues` (CodeGen/Testing) or `RollbackOnFailure` (Deployment). System auto-retries with fixes. If auto-fix fails, human intervention required.

**Q: How do I trace a production error back to code generation?**
A: Session linking: `maintenance_session.deployment_id` → `deployment_session.test_session_id` → `test_session.codegen_session_id` → `codegen_session.user_intent`

**Q: Where do approval gates exist?**
A: Two places:

1. **ArchitectureReview** state (if architecture changes proposed)
2. **Deployment Complete** state (manual trigger for safety)

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

| Aspect                   | CodeGen                | Testing                  | Maintenance                |
| ------------------------ | ---------------------- | ------------------------ | -------------------------- |
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
    │ 3. Query Semantic    │
    │    Graph (HNSW)      │
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

- [X] Implement `TerminalExecutor` struct
- [X] Command validation with whitelist
- [X] Subprocess execution with Tokio
- [X] Streaming output to frontend
- [X] Unit tests for command validation
- [X] Integration tests for execution

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
| Max Agent Terminals     | ≤5    | Never create more than 5 agent terminals   |
| Detection Speed         | <10ms  | Time to check terminal state               |
| False Positives         | <1%    | Incorrectly marked as busy when idle       |

#### Platform Compatibility

| Platform          | Process Detection | Implementation                           |
| ----------------- | ----------------- | ---------------------------------------- |
| **macOS**   | ✅ Supported      | tcgetpgrp + libproc                      |
| **Linux**   | ✅ Supported      | tcgetpgrp + /proc                        |
| **Windows** | ⚠️ Limited      | Windows console API (different approach) |

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
- XSS → Escape user inputv
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

### LLM Mistake Tracking Integration

**Existing Implementation:** `src/gnn/known_issues.rs`

**Known Issue Data Structure Requirements:**

- Unique identifier for each issue
- Issue type classification
- Description and affected files
- Error message and stack trace
- Fix applied (if any)
- LLM used for generation
- Timestamp for tracking

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
⏭️ **Yantra Codex (GraphSAGE neural network for cost optimization - 96% LLM cost savings)**
⏭️ Advanced context compression (semantic chunking)
⏭️ Advanced HNSW semantic search optimizations
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

**Example Detection:**

- ✅ `test_calculator.py` → Test file
- ✅ `calculator_test.py` → Test file
- ✅ `tests/utils.py` → Test file
- ✅ `math.test.ts` → Test file
- ❌ `calculator.py` → Source file
- ❌ `utils.js` → Source file

**2. Test-to-Source Mapping** (`find_source_file_for_test`)

Intelligently maps test files to their corresponding source files:

**Mapping Strategy:**

1. Remove test prefix/suffix: `test_calculator.py` → `calculator.py`
2. Remove test directory: `tests/calculator.py` → `calculator.py`
3. Search graph for matching source filename
4. Return matching source file if found

**Example Mappings:**

- `test_calculator.py` → `calculator.py`
- `calculator_test.py` → `calculator.py`
- `tests/utils.py` → `utils.py`
- `math.spec.ts` → `math.ts`

**3. Edge Creation** (`create_test_edges`)

Creates bidirectional test edges in the GNN graph:

**Algorithm:**

1. For each file in graph:
   - If file is a test file:
     - Find corresponding source file
     - If source found:
       - Create TestDependency edge: test_file → source_file
       - For each test function in test_file:
         - Try to infer tested function name
         - If found, create Tests edge: test_fn → source_fn
     - If source not found:
       - Log warning (test file without corresponding source)

**Returns:** Count of test edges created (for metrics)

### Usage Examples

**Scenario 1: Test Coverage Analysis**

Find all tests for a given source file and identify untested files:

- Query: Get all incoming TestDependency edges for source file
- Filter: Find source files with no incoming test edges
- Result: List of untested files that need test coverage

**Scenario 2: Impact Analysis**

When source file changes, find affected tests:

- Query: Get incoming TestDependency edges for changed file
- Result: List of test files that need to be re-run
- Benefit: Selective test execution instead of full suite

**Scenario 3: Guided Test Generation**

Suggest tests for untested functions:

- Query: Get all functions in source file
- Query: Get functions with incoming Tests edges
- Compare: Identify functions without test coverage
- Result: List of untested functions needing test generation

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

## Agent Interaction Modes: Guided vs Auto Mode (MVP Phase 1 - Priority Feature)

### Overview

Yantra supports two distinct interaction modes that control the level of user involvement during autonomous code generation, testing, and deployment workflows. Both modes are fully autonomous - the difference is in **when and how the user is informed and consulted**.

**Core Principle:** Yantra is an autonomous agentic platform. The agent executes the entire development lifecycle (generate → test → package → deploy → monitor). The interaction mode determines the frequency and type of user checkpoints during this autonomous execution.

### 1. Auto Mode (Default for Experienced Users)

**Philosophy:** "Trust the agent, work autonomously with strategic checkpoints."

The agent executes the full pipeline autonomously with minimal user interruption.

**CRITICAL:** Agent determines approval checkpoints at milestone boundaries and CANNOT proceed without user approval at these checkpoints.

**User is consulted only for:**

1. **Milestone approval checkpoints** (Agent-determined, feature-set boundaries - e.g., "Authentication Complete", "Payments Functional")
2. **Architecture changes** (adding/removing components, changing connections)
3. **User action blockers** (API keys needed, manual setup required, external service configuration)
4. **Critical failures after 3 auto-retry attempts**

**Approval Checkpoint Determination (Autonomous):**

Agent automatically identifies logical approval checkpoints by analyzing:

- **Feature set boundaries:** Grouping related features (Authentication, Payments, Security, Deployment)
- **Critical integration points:** External dependencies (Stripe, AWS, SendGrid)
- **Security-critical milestones:** Features requiring verification (Auth, Payment, Data Privacy)
- **Architecture complexity:** Major component additions or pattern changes
- **Risk assessment:** High-risk changes that need user validation

**Example Checkpoints:**

- ✅ Core Infrastructure Complete (Database, API setup, logging)
- ✅ Authentication Features Working (Security-critical verification)
- ✅ Payment Integration Functional (External API validation)
- ✅ Ready for Deployment (Final quality gate)

**IMPORTANT:** Agent will NOT proceed past any checkpoint without explicit user approval. This ensures user maintains control over project progress while benefiting from autonomous execution between checkpoints.

**When Auto Mode Activates:**

- User explicitly sets mode: "Switch to auto mode"
- User preference saved in `.yantra/config.json`
- Suitable for: Production deployments, experienced developers, standard feature implementation

**Agent Behavior in Auto Mode:**

```
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
```

**Auto Mode Logging:**
All actions are logged to `.yantra/logs/agent_activity.log` with timestamps, allowing user to review what happened without being interrupted during execution.

**Auto Mode Checkpoints Summary:**

1. **Milestone Approval Checkpoints** (Agent-determined, MANDATORY)

   - Agent analyzes architecture to determine logical checkpoints
   - Typically 3-5 checkpoints per project (feature-set boundaries)
   - Agent CANNOT proceed without explicit user approval
   - User reviews completed work before next milestone begins
2. **Architecture Changes** (ALWAYS requires approval)

   - Adding/removing components
   - Changing connections
   - Modifying architectural patterns
3. **User Action Blockers** (PAUSES until resolved)

   - API keys, credentials, tokens
   - Manual configuration steps
   - External service setup
4. **Critical Failures** (After 3 auto-retry attempts)

   - Test failures that can't be auto-fixed
   - Security vulnerabilities requiring manual review
   - Integration errors with external services

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

## User Experience & Interface Design (MVP Phase 1)

### Overview

**Design Philosophy:** Minimal UX with AI-first interaction model

Yantra follows a **space-optimized, keyboard-first** user experience design that prioritizes content over controls. The interface is built around the principle that AI (via chat) is the primary interface, not buttons and menus.

**Reference Document:** See `UX.md` (root directory) for complete user experience guide including:

- Design philosophy and rationale
- Detailed UI component specifications
- User workflows and interactions
- Keyboard shortcuts and efficiency features
- Visual design guidelines
- Future enhancements roadmap

### Core UX Principles

1. **Space Optimization** - Every pixel counts

   - Controls take minimal space (40px top bar, inline settings)
   - Content maximized (chat, editor, terminal take 90%+ screen)
   - No unnecessary panels or toolbars
2. **Single-Line Layouts** - Inline controls where possible

   - LLM settings: provider dropdown + API key + status (one line)
   - Terminal toggle: single button with visual state
   - No dedicated settings windows unless absolutely necessary
3. **Visual Indicators** - Small, clear, unobtrusive

   - Status dots (green/red/yellow, 2px)
   - Pulsing animations for active states
   - Hover tooltips for detailed info
4. **Auto-Save** - Reduce explicit save actions

   - LLM settings auto-save on blur
   - Code auto-saves on edit (debounced)
   - Terminal history persists across sessions
5. **Keyboard-First** - Power users efficiency

   - Cmd+` toggle terminal
   - Cmd+B toggle file tree
   - All major actions have shortcuts
6. **Progressive Disclosure** - Show details on demand

   - API settings collapsed by default
   - Terminal hidden until needed
   - Dependency graph available but not intrusive

### Main Interface Layout

**3-Panel Design:**

```
┌────────────────────────────────────────────────────────────────┐
│  YANTRA          [🚀 Auto Mode]  [Provider ▼] [●] [⚙️]       │ ← 40px Top Bar
├──────────┬───────────────────────────────┬────────────────────┤
│          │                               │                    │
│  File    │         Chat Panel            │    Code Editor     │
│  Tree    │      (AI Interaction)         │  (Monaco Editor)   │
│  (256px) │        Primary UX             │   Transparency     │
│          │                               │                    │
│  [📁]    │ ┌─────────────────────────┐   │ ┌────────────────┐ │
│  [📄]    │ │ User: Add login page    │   │ │ // Generated   │ │
│  [📂]    │ │                         │   │ │ // Code here   │ │
│          │ │ Agent: I'll create...   │   │ │                │ │
│          │ │ ✅ Created auth.py      │   │ │ def login():   │ │
│          │ │ ✅ Added tests          │   │ │   ...          │ │
│          │ └─────────────────────────┘   │ └────────────────┘ │
│          │ [Type your request...]  [Send]│                    │
├──────────┴───────────────────────────────┴────────────────────┤
│  $ Terminal (Toggle: Cmd+`)                                    │ ← Hidden by default
│  > npm run dev                                                 │
│  Server running on http://localhost:3000                       │
└────────────────────────────────────────────────────────────────┘
```

### Key UI Components

#### Top Bar (40px Fixed Height)

- **YANTRA** branding (bright white, prominent)
- **Mode Indicator:** 🚀 Auto | 🧭 Guided (always visible, clickable)
- **LLM Settings:** Provider dropdown + API key input + status dot (inline, one line)
- **Settings Icon:** ⚙️ for additional configuration

#### File Tree Panel (Left, 256px default)

- Collapsible/expandable folders
- File type icons
- Real-time updates during code generation
- Drag-to-resize width
- Toggle visibility: Cmd+B

#### Chat Panel (Center, Primary)

- Primary interaction surface (60% of screen)
- Conversational AI interface
- Real-time agent activity indicators
- Task queue visibility (expandable)
- Progress indicators for long operations
- Code snippets with syntax highlighting
- Copy-to-clipboard buttons

#### Code Editor Panel (Right, 25% of screen)

- Monaco editor integration
- Syntax highlighting for multiple languages
- Real-time updates during generation
- Read-only mode with option to edit
- Tab management for multiple files
- Git diff view integration

#### Terminal Panel (Bottom, Hidden by Default)

- Toggle: Cmd+` keyboard shortcut
- Multiple terminal sessions
- Output streaming in real-time
- Command history persistence
- Background process detection
- Resize by dragging divider

### New UI Features (November 29, 2025)

#### 1. Dual-Theme System

- **Dark Blue Theme** (default): Professional coding aesthetic
- **Bright White Theme**: High contrast for accessibility
- Toggle via top bar button
- Theme persists across sessions

#### 2. Status Indicator

- Real-time agent activity visualization
- States: Idle (gray) | Thinking (yellow pulse) | Working (blue pulse) | Success (green) | Error (red)
- Click to expand task queue
- Non-intrusive (small dot in top bar)

#### 3. Task Queue Panel

- Expandable panel showing all active tasks
- Priority levels and dependencies
- Progress bars for each task
- Estimated completion times
- Cancel/pause controls

#### 4. Panel Expansion System

- Any panel can expand to full screen
- Quick toggle buttons on panel headers
- Smooth CSS animations (300ms)
- Keyboard shortcuts for expansion

#### 5. Resizable Panels

- All dividers are draggable
- Smooth cursor control during drag
- Minimum/maximum width constraints
- Double-click divider to reset to default
- Widths persist across sessions

### Visual Design System

**Colors:**

- Background: `#1a1b26` (dark blue) or `#ffffff` (bright white)
- Text: `#c0caf5` (light blue) or `#24292e` (dark gray)
- Accent: `#7aa2f7` (bright blue)
- Success: `#9ece6a` (green)
- Error: `#f7768e` (red)
- Warning: `#e0af68` (yellow)

**Typography:**

- UI Text: Inter, 14px
- Code: JetBrains Mono, 13px
- Headings: Inter Semibold

**Spacing:**

- Base unit: 8px
- Small: 4px
- Medium: 16px
- Large: 24px

**Status Indicators:**

- Size: 8px diameter
- Colors: Green (active/ready), Red (error), Yellow (warning), Gray (idle)
- Animation: Pulse effect (1.5s duration) for active states

### User Workflows

**Typical User Session:**

1. **Launch Yantra** → File tree loads, chat ready
2. **Open Project** → Click "Open Folder", select directory
3. **Wait for Analysis** → Progress indicator (5-30 seconds)
4. **Chat Interaction** → Type request in chat input
5. **Agent Works** → Status indicator shows activity
6. **Review Generated Code** → Code appears in editor panel
7. **Monitor Progress** → Terminal shows test output (if visible)
8. **Approve/Iterate** → Continue chat conversation

**Error Handling:**

- **LLM API Errors:** Red message in chat with "Retry" button
- **File System Errors:** Toast notification with actions
- **Test Failures:** Red box in chat with "Fix Automatically" button
- **Validation Errors:** Inline warnings with suggested fixes

**Progress Indicators:**

- **Long Operations:** Spinner + text + time estimate
- **Background Operations:** Non-blocking with completion toast
- **Multi-step Tasks:** Progress bar with step indicators

### Keyboard Shortcuts

| Shortcut             | Action                   |
| -------------------- | ------------------------ |
| `Cmd/Ctrl + ~`     | Toggle Terminal          |
| `Cmd/Ctrl + B`     | Toggle File Tree         |
| `Cmd/Ctrl + ,`     | Open Settings            |
| `Cmd/Ctrl + K`     | Clear Chat               |
| `Cmd/Ctrl + Enter` | Send Message             |
| `Cmd/Ctrl + ←/→` | Navigate Tabs            |
| `Cmd/Ctrl + W`     | Close Tab                |
| `Cmd/Ctrl + F`     | Find in Code             |
| `Esc`              | Cancel Current Operation |

### Accessibility Features

- High contrast themes
- Keyboard navigation for all functions
- Screen reader support
- Adjustable font sizes
- Clear focus indicators
- ARIA labels on all interactive elements

### Performance Targets

- UI interactions: <100ms response time
- Panel resize: 60fps smooth animation
- Theme switching: Instant (<50ms)
- Code rendering: <200ms for files up to 10,000 lines
- Terminal output: Real-time streaming with no lag

### Implementation Status

- ✅ 3-panel layout with resizable dividers
- ✅ Top bar with mode indicator
- ✅ LLM settings inline component
- ✅ Terminal toggle (Cmd+`)
- ✅ File tree with drag-to-resize
- ✅ Chat interface with real-time updates
- ✅ Monaco editor integration
- ✅ Dual-theme system
- ✅ Status indicator with task queue
- ✅ Panel expansion system
- ✅ Keyboard shortcuts
- 🔄 Settings modal (in progress)
- ⚪ Dependency graph visualization
- ⚪ Architecture view

### Future Enhancements

See `UX.md` for detailed roadmap including:

- Multiple terminal tabs
- Dependency graph view
- Workflow automation UI
- Plugin system UI
- Collaborative mode indicators
- Voice commands
- Advanced code visualization

**For complete UX specifications, design rationale, user flows, and detailed component documentation, refer to `UX.md` in the root directory.**

---

## Browser Integration with Chrome DevTools Protocol (CDP)

### Overview

**Purpose:** Enable Yantra to launch, control, and monitor web browsers for UI validation, error capture, and interactive development workflows.

**Approach:** System Chrome + CDP

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

**Requirements:**

- Automatically find system Chrome installation on macOS/Windows/Linux
- Platform-specific search paths:
  - macOS: Applications folder (Chrome, Chromium, Edge, Brave)
  - Windows: Program Files (Chrome, Edge)
  - Linux: /usr/bin (google-chrome, chromium, chromium-browser, snap)
- Fallback: Download minimal Chromium (~100MB) to app data folder on first launch
- Cache Chrome path for subsequent launches
- One-time setup, transparent to user

**User Experience:**

- First launch: Instant if Chrome found, ~30 seconds if download needed
- Show "Setting up browser preview..." during download
- All subsequent launches: Instant (path cached)

**Implementation Files:**

- `src-tauri/src/browser/chrome_finder.rs` (NEW) - Platform detection
- `src-tauri/src/browser/downloader.rs` (NEW) - Fallback Chromium download

---

#### 2. Chrome Launch with CDP

**Requirements:**

- Launch Chrome with remote debugging on random available port
- Use app mode for minimal UI (looks native, not browser window)
- Disable first-run prompts, default browser checks, extensions
- Configure window size (default 1280x720)
- Parse debug port from Chrome stderr output
- Establish WebSocket connection to CDP

**Launch Configuration:**

- `--remote-debugging-port=0`: Auto-select available port (no conflicts)
- `--app=URL`: Launch in app mode
- `--no-first-run`, `--no-default-browser-check`: Skip prompts
- `--disable-extensions`: Faster startup
- `--disable-popup-blocking`: Allow OAuth popups
- `--disable-background-networking`: No update checks

**Implementation Files:**

- `src-tauri/src/browser/launcher.rs` (NEW) - Chrome launching
- Update `src-tauri/src/browser/cdp.rs` - Replace placeholder with real launch

**Technology:** Use `chromiumoxide` crate for Rust CDP implementation

---

#### 3. CDP Connection & Communication

**Requirements:**

- Establish WebSocket connection to Chrome DevTools Protocol
- Send CDP commands (navigate, execute scripts, subscribe to events)
- Receive CDP events (console messages, errors, network events)
- Handle connection lifecycle (connect, reconnect, cleanup)
- Spawn async task to handle CDP event stream

**CDP Domains to Enable:**

- Runtime: Console messages, exceptions, script execution
- Network: Request/response monitoring, error capture
- Page: Navigation, lifecycle events

**Implementation Files:**

- Rewrite `src-tauri/src/browser/cdp.rs` - Replace placeholders with real CDP
- Add `chromiumoxide = "0.5"` to `Cargo.toml`

---

#### 4. Dev Server Management

**Requirements:**

- Auto-detect frontend framework (Next.js, Vite, Create React App)
- Detection criteria:
  - Next.js: presence of next.config.js/mjs
  - Vite: presence of vite.config.js/ts
  - CRA: "react-scripts" in package.json
- Start dev server with appropriate command
- Parse port from server output using regex patterns:
  - Next.js: "ready on http://localhost:3000"
  - Vite: "Local: http://localhost:5173"
  - CRA: Default port 3000
- Wait for server to be ready (HTTP health check)
- 30-second timeout for server startup
- Handle port conflicts (try alternate ports)

**Supported Frameworks:**

- Next.js: `npm run dev`
- Vite: `npm run dev`
- Create React App: `npm start`
- Custom: Try common ports (3000, 3001, 5173, 8080)

**Implementation Files:**

- `src-tauri/src/browser/dev_server.rs` (NEW) - Framework detection & startup

---

#### 5. Runtime Injection

**Requirements:**

- Inject JavaScript runtime into browser before page loads
- Runtime captures errors and sends to Yantra backend
- Use CDP `Page.addScriptToEvaluateOnNewDocument` command
- Runtime must load before any user code executes

**Runtime Capabilities:**

- Connect to Yantra via WebSocket
- Override console.error to capture messages
- Listen for unhandled exceptions (window.error event)
- Listen for unhandled promise rejections
- Intercept fetch() to capture network errors
- Send all errors to Yantra with full context (message, stack, timestamp)

**Implementation Files:**

- `src-tauri/src/browser/yantra-runtime.js` (NEW) - Error capture runtime
- `src-tauri/src/browser/runtime_injector.rs` (NEW) - Injection logic

---

#### 6. Console Error Capture

**Requirements:**

- Subscribe to CDP Runtime domain events
- Capture console.error calls with full arguments
- Capture unhandled JavaScript exceptions with stack traces
- Include source location (URL, line number, column number)
- Send errors to agent orchestrator for auto-fix
- Support source map resolution for minified code

**CDP Events to Monitor:**

- `Runtime.consoleAPICalled`: Console API calls (log, error, warn)
- `Runtime.exceptionThrown`: Unhandled exceptions with details

**Error Data to Capture:**

- Error message
- Stack trace
- Source URL
- Line/column numbers
- Timestamp
- Error type

**Implementation Files:**

- `src-tauri/src/browser/error_capture.rs` (NEW) - CDP event subscriptions

---

#### 7. Network Error Capture

**Requirements:**

- Subscribe to CDP Network domain events
- Monitor request failures (DNS errors, timeouts, CORS)
- Capture HTTP error status codes (404, 500, etc.)
- Include full request/response context
- Send network errors to agent for analysis

**CDP Events to Monitor:**

- `Network.loadingFailed`: Failed resource loads
- `Network.responseReceived`: Check status codes for 4xx/5xx

**Error Data to Capture:**

- Request URL
- HTTP status code and text
- Error type (DNS, timeout, CORS, etc.)
- Request headers
- Response headers (if available)
- Timestamp

**Implementation Files:**

- `src-tauri/src/browser/network_monitor.rs` (NEW) - Network monitoring

---

#### 8. Browser Validation

**Requirements:**

- Launch Chrome with CDP for given URL
- Inject error capture runtime
- Subscribe to console and network events
- Navigate to URL and collect errors for timeout period (default 30s)
- Aggregate errors by severity (error, warning)
- Generate validation report with success status
- Clean up browser session after validation

**Validation Flow:**

1. Launch Chrome with CDP
2. Inject runtime script
3. Subscribe to console and network events
4. Navigate to URL
5. Collect errors for configured timeout
6. Classify errors by severity
7. Generate report (success boolean, error list, duration)
8. Close browser

**Implementation Files:**

- Update `src-tauri/src/browser/validator.rs` - Use real CDP instead of placeholders

---

### MVP Implementation Checklist

**Week 1: Core CDP Foundation**

- Add `chromiumoxide = "0.5"` to Cargo.toml
- Implement Chrome finder with platform-specific paths
- Implement Chrome launcher with CDP remote debugging
- Rewrite CDP module with real implementation
- Test Chrome launch on macOS/Windows/Linux

**Week 2: Dev Server & Error Capture**

- Implement dev server detection (Next.js/Vite/CRA)
- Add dev server startup and port parsing
- Implement console error capture via CDP
- Implement network error capture via CDP
- Test with intentionally broken React app

**Week 3: Runtime Injection**

- Create error capture runtime JavaScript
- Implement runtime injection logic
- Set up WebSocket server for browser ↔ Yantra communication
- Test runtime loads before user code
- Verify error flow: browser → runtime → WebSocket → Yantra → agent

**Week 4: Integration & Testing**

- Update browser validator to use real CDP
- End-to-end test: Generate app → Start server → Capture errors → Agent fixes
- Cross-platform testing (macOS, Windows, Linux)
- Performance optimization (startup <2s, memory <200MB)
- Error deduplication and aggregation

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

#### 9. Interactive Element Selection

**Requirements:**

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

**User Flow:**

1. User clicks "Select Element" button in Yantra
2. Browser enters selection mode (crosshair cursor)
3. User hovers → element highlights
4. User clicks → element info sent to chat
5. User can type commands referencing selected element
6. Agent knows exact element/file/line for precise changes

**Implementation:**

- Update runtime JavaScript with selection mode handlers
- Create interaction handler for selection events
- Add "Select Element" button to browser preview UI

#### 10. WebSocket Communication

**Requirements:**

- Bidirectional Browser ↔ Yantra communication channel
- WebSocket server on random localhost port
- Message types:
  - Browser → Yantra: errors, selections, hover events
  - Yantra → Browser: enable selection mode, highlight element, scroll commands
- Handle connection lifecycle
- Support multiple simultaneous browser connections

**Implementation:**

- Create WebSocket server module
- Update runtime to connect to WebSocket
- Define message protocol schema

#### 11. Source Map Integration

**Requirements:**

- Map browser DOM elements to source code files
- React DevTools-style fiber tree traversal
- Extract source locations from React internal properties
- Support source maps for transpiled/bundled code
- Return file name, line number, column number, component name

**Benefit:** Agent knows exactly which file and line to modify for visual changes

#### 12. Context Menu & Quick Actions

**Requirements:**

- Right-click menu in browser preview
- Quick actions: Replace, Edit, Remove, Duplicate, Change colors
- Context-aware actions based on element type (image, text, button, etc.)
- Trigger appropriate workflows (asset picker, color picker, text editor)
- Apply changes and refresh preview

#### 13. Visual Feedback Loop

**Features:**

- Before/After split view for proposed changes
- Visual diff highlighting
- One-click undo for any change
- Change history timeline
- Preview pending changes before application

#### 14. Asset Picker Integration

**Features:**

- Search Unsplash for free stock photos
- Generate images with DALL-E 3
- Upload local files
- Asset library for recently used images
- Direct integration with element replacement

---

### Performance Targets

**MVP:**

- Chrome startup: <2 seconds
- Dev server startup: <10 seconds (framework-dependent)
- Error capture latency: <100ms (real-time)
- Memory overhead: <200MB (Chrome + monitoring)
- CPU overhead: <5% when idle

**Post-MVP:**

- Element selection response: <50ms (interactive feel)
- Screenshot capture: <200ms
- WebSocket message latency: <10ms

---

### Error Handling & Edge Cases

**Chrome Not Found:**

- Show user-friendly message: "Downloading browser engine..."
- Download Chromium automatically (~100MB, 30-60 seconds)
- Cache for future use in app data folder
- Fallback: Ask user to install Chrome manually (rare)

**Dev Server Fails to Start:**

- Check for port conflicts (try next port)
- Check for missing dependencies (suggest npm install)
- Show clear error with fix suggestions
- Allow manual port specification

**CDP Connection Fails:**

- Retry with exponential backoff (3 attempts)
- Show user-friendly error message
- Degrade gracefully: Skip browser validation, rely on unit tests

**Browser Crashes:**

- Detect process exit
- Auto-restart browser
- Restore previous state (URL, tabs)
- Log crash for debugging

---

### Security Considerations

**Local-Only Communication:**

- WebSocket server binds to 127.0.0.1 (localhost only)
- No external access
- Random port selection (no fixed port conflicts)

**Chrome Sandbox:**

- Chrome runs in sandboxed mode (default security)
- No filesystem access beyond project folder
- No network access to Yantra's internal APIs

**User Privacy:**

- No telemetry sent to Yantra servers
- All browser data stays local
- Anonymous crash reports only (opt-in)

---

### Testing Strategy

**Unit Tests:**

- Chrome finder logic with mock filesystem
- Port parsing from dev server output
- Error message parsing and classification

**Integration Tests:**

- Launch Chrome → Connect CDP → Navigate → Capture error
- Start dev server → Detect port → Wait for ready
- Inject runtime → Trigger error → Receive via WebSocket

**E2E Tests:**

- Full workflow: Generate React app with error → Start server → Capture error → Agent fixes → Validation passes
- Cross-platform testing on macOS, Windows, Linux

**Platform Tests:**

- macOS: Chrome, Brave, Edge compatibility
- Windows: Chrome, Edge (x64 and ARM)
- Linux: Chrome, Chromium (snap and apt packages)

---

### Success Metrics

**MVP (Phase 1):**

- 95%+ users have working browser preview (zero manual setup)
- 100% of console errors captured
- 90%+ of network errors captured
- <2 second browser startup
- <10 second dev server startup

**Post-MVP (Phase 2):**

- 80%+ users utilize element selection feature
- 50%+ changes made via visual interaction
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
    ✅ Success or try again
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

**Requirement:** After 2 failed try, revert to the prior working checkpoint automatically.

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
    SearchForSolution,             // Use semantic graph/web search
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
│ 3️⃣  Search web/semantic graph for solution    │
│ 4️⃣  Skip Payment Service for now              │
│                                                │
│ Your choice:                                   │
└────────────────────────────────────────────────┘
```

---

### 7. Semantic Graph/Web Search with User Consent

**Requirement:** Use semantic graph or web search to find solutions, but only with user consent.

**Implementation:**

```rust
// src-tauri/src/agent/knowledge_search.rs

pub struct KnowledgeSearch {
    semantic_graph: SemanticGraph,
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
             1️⃣  Local semantic graph (HNSW - your past solutions)\n\
             2️⃣  Web search (Google, StackOverflow, GitHub)\n\
             3️⃣  Both\n\
             4️⃣  Neither (I'll provide guidance)\n\n\
             Your choice:",
            query
        );

        // Wait for user response
        let response = self.prompt_user(&prompt).await?;

        Ok(match response.as_str() {
            "1" => SearchPermission::SemanticGraphOnly,
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
            SearchPermission::SemanticGraphOnly => {
                solutions.extend(self.semantic_graph.search_hnsw(query).await?);
            },
            SearchPermission::WebOnly => {
                if self.web_search_enabled {
                    solutions.extend(self.web_search(query).await?);
                }
            },
            SearchPermission::Both => {
                solutions.extend(self.semantic_graph.search_hnsw(query).await?);
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
    SemanticGraphOnly,
    WebOnly,
    Both,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub source: String,          // "SemanticGraph", "StackOverflow", "GitHub"
    pub title: String,
    pub description: String,
    pub code_snippet: Option<String>,
    pub relevance_score: f32,    // 0.0-1.0
}
```

**Search Consent UI:**

```
Agent: "I found similar issues in our semantic graph:

Option 1: Semantic Graph Search Results (Local)
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
2. **Two Failures + Auto-Recovery**

   - Create checkpoint
   - Make code change
   - Tests fail
   - Auto-revert to checkpoint
   - Retry with fix
   - Tests pass ✅
3. **2 Failures + LLM conulting mode**

   - Create checkpoint
   - Attempt 1: Fail → Revert → Retry
   - Attempt 2: Fail → Revert → Retry
   - LLM consulting mode
   - Retry with user input
   - Tests pass ✅
4. **One-Click Restore from UI**

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

- Advanced HNSW semantic search optimizations
- Advanced context compression
- Pattern extraction from failures
- Opt-in anonymous pattern sharing
- Full validation pipeline (5 validations)

### New Capabilities

1. **Complete Unlimited Context Engine**

- Optimized HNSW semantic search for code patterns
- Enhanced semantic search for relevant examples
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

### Storage Architecture: 5-Tier Strategy

**Overview:** Yantra uses a 5-tier storage architecture optimized for different access patterns. Each tier uses the right tool for its specific workload.

#### Tier 0 - Cloud Graph Database (Team Coordination) ☁️

**Access Pattern:** Real-time collaboration queries across machines
**Performance Target:** <50ms query latency, WebSocket updates
**Phase:** Phase 2B (Month 4-5)

**Purpose:**

- Real-time visibility: "Which files is my teammate currently editing?"
- Dependency awareness: "Who is working on dependencies I need?"
- Conflict prevention: Proactive warnings before starting work
- **NOT for code storage** - metadata only (privacy-preserving)

**Data Stored:**

- Dependency graph metadata (function signatures, relationships)
- File modification registry (who's working on what)
- User presence and locks (cross-machine coordination)
- LLM-generated summaries (no actual code)

**Technology Stack:**

- Database: PostgreSQL 15+ with pgvector extension
- Pub/Sub: Redis 7+ for real-time updates
- Protocol: WebSocket for presence tracking
- Sync Model: Like Google Drive/Git - eventual consistency

**Sync Pattern:**

- Local Tier 1 syncs metadata to Tier 0 on commit
- Agents query Tier 0: "Who's editing files A, B, C?"
- WebSocket notifications for real-time updates

#### Tier 1 - Hot Path (Dependency Graph) 🔥

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
pub struct DependencyGraphEngine {
    graph: CodeGraph,              // In-memory petgraph
    snapshot_path: PathBuf,        // ~/.yantra/depgraph_snapshot.bin
    wal: WriteAheadLog,            // Incremental updates log
}

// Auto-snapshot every 30s
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        depgraph.save_snapshot()?; // bincode serialization
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

**Phase 2 (Month 4 - Team of Agents):**

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

## Phase 2A: Team of Agents Architecture (Months 3-4)

### Overview

**Problem:** As codebases scale beyond 100k LOC and teams grow to 5+ concurrent developers, single-agent architecture becomes a bottleneck. Developers need multiple AI agents working simultaneously on different parts of the codebase without conflicts.

**Solution:** Transform Yantra from a single autonomous agent to a **team of coordinating agents** using a Master-Servant architecture with Git coordination branch and Agent-to-Agent (A2A) protocol for proactive conflict prevention.

**Key Innovation:** Unlike traditional collaborative editing (which reactively resolves conflicts), Yantra uses **proactive conflict prevention** - agents communicate intent before making changes via GNN dependency analysis and real-time file locking.

### Why Team of Agents?

**Current Limitations (Single Agent):**

- Only one developer can use Yantra at a time
- Large codebases (100k+ LOC) exceed single-agent capacity
- Complex features require serial execution of multiple tasks
- Bottleneck for team collaboration

**Team of Agents Benefits:**

- **Parallelization:** 3-10 agents working simultaneously on same feature
- **Specialization:** Dedicated agents for frontend, backend, testing, DevOps
- **Scalability:** Handle 100k+ LOC codebases efficiently
- **Team Collaboration:** Multiple developers with their own agents working on same project
- **Fault Tolerance:** One agent failure doesn't block others
- **3x Faster:** 15 minutes instead of 45 minutes for complex features

### Architecture: Master-Servant Pattern

**Core Philosophy:** Master assigns work once, then agents execute independently on their own Git branches with peer-to-peer coordination.

**Why Master-Servant over Peer-to-Peer?**

**Rejected: Pure Peer-to-Peer (P2P)**

- ❌ No single source of truth - who decides work split?
- ❌ Complex consensus algorithms needed (Raft/Paxos)
- ❌ Race conditions on file writes
- ❌ Conflict resolution is reactive (after-the-fact)
- ❌ Difficult to debug - no clear authority

**Chosen: Master-Servant with Git Coordination**

✅ **Master's Limited Role:**

- Master decomposes work and assigns ONCE
- Master is not a bottleneck (not involved in ongoing work)
- Master can crash after assignment - agents continue

✅ **Git Provides Code Coordination:**

- Branches provide isolation (proven infrastructure)
- PRs provide integration (normal Git workflow)
- Conflicts surface at merge time (standard resolution)

✅ **sled Provides Lock Coordination:**

- Atomic file claims (compare-and-swap)
- Real-time file locking (proactive conflict prevention)
- Agent state tracking (progress, locks held)

✅ **Mirrors Human Teams:**

- Tech lead assigns features (Master)
- Developers work on branches independently (Agents)
- Developers coordinate when needed (A2A protocol)
- Team reviews PRs (human review or auto-merge)

✅ **Simple State Management:**

- Coordination state is append-only events (Git branch)
- File locks are key-value pairs (sled)
- Easy to debug and audit

**Key Insight from Design:**

Traditional master-servant has the master coordinating everything (bottleneck). Yantra's master only assigns work at the beginning, then gets out of the way. Git handles code isolation. sled handles lock coordination. Agents coordinate peer-to-peer for dependencies. This hybrid approach combines the best of centralized (clear work assignment) and decentralized (independent execution).

---

### Master Agent Responsibilities

**Role:** Work assignment ONLY - Master is NOT a coordinator or bottleneck

**Core Principle:** Master analyzes and assigns work at the beginning, then agents execute independently. Master does not approve operations, manage locks, or coordinate ongoing work. **Master's job is done after assignment.**

**Why This Design?**

- ✅ Master is not a single point of failure (if Master crashes after assignment, agents continue)
- ✅ Agents don't need Master's permission for ongoing work
- ✅ Git already solves file coordination (branches isolate, merges detect conflicts)
- ✅ Matches human team workflow (tech lead assigns features, developers work independently)

**Master's Three Responsibilities:**

1. **Feature Decomposition:**

   - Receives high-level feature request from user
   - Uses GNN to analyze dependencies and complexity
   - Decomposes into 3-10 sub-features with minimal cross-dependencies
   - **Goal:** Create sub-features with loose coupling to minimize inter-agent blocking
   - Example: "Add Stripe payments" → Backend API, Frontend form, Testing, DevOps
2. **Agent Assignment:**

   - Assigns each sub-feature to specialized servant agent
   - Creates Git feature branches for each agent: `feature/agent-1-payment-api`
   - Commits assignment events to `.yantra/coordination` branch
   - **Master's job is complete** - can shut down or take new requests
3. **User Review (Optional Advisory Role):**

   - Shows proposed work split to user before agents start
   - User can adjust assignments if they see coupling issues
   - Example: "Agent 1 (Backend), Agent 2 (Frontend) share UserService - is this split okay?"
   - Master is advisory, not authoritative - assignments can be adjusted

**What Master Does NOT Do:**

- ❌ Approve every file operation (not a bottleneck)
- ❌ Coordinate ongoing work (agents are fully independent)
- ❌ Resolve runtime conflicts (Git branches + Tier 2 locks handle this)
- ❌ Manage file locks (Tier 2 sled handles this peer-to-peer)
- ❌ Monitor agent progress (agents self-coordinate)

---

### Servant Agent Responsibilities

**Role:** Fully independent execution with peer-to-peer dependency coordination

**Core Principle:** Agents work autonomously on their own Git branches, claim files from Tier 2 (sled) as needed, and coordinate peer-to-peer when dependencies arise. No central coordinator approval needed.

**Agent Lifecycle:**

1. **Startup & Assignment Discovery:**

   - Pull `.yantra/coordination` branch to find assignment from Master
   - Create dedicated feature branch from main: `feature/agent-1-payment-api`
   - Register self in Tier 2 (sled): `registry:agent:agent-1`
   - **Agent is now independent** - no further Master interaction needed
2. **Before Modifying Files (Proactive Conflict Prevention):**

   - Query GNN: "What files do I need to modify for this feature?"
   - Check Tier 2 locks: "Is any file I need already claimed by another agent?"
   - **If locked by another agent:**
     - Option A: Wait and poll until released
     - Option B: Work on other parts that don't have the dependency
     - Option C: Send peer-to-peer message to coordinate approach
   - **If free:** Claim files atomically in Tier 2: `lock:src/payment/stripe.py = agent-1`
   - **Git branch provides isolation:** Even if lock fails, branch prevents corruption
3. **During Work (Independent Execution):**

   - Modify files on own Git branch (full isolation from other agents)
   - Update progress state in Tier 2: `agent:agent-1:state = { phase: "CodeGen", progress: 45% }`
   - Release locks immediately when done with file: `DELETE lock:src/payment/stripe.py`
   - **No Master check-ins required** - agent executes autonomously
4. **Peer-to-Peer Dependency Coordination (A2A Protocol):**

   - **Scenario:** Agent B needs API that Agent A is building
   - **Agent B checks coordination branch:** "Has Agent A completed user auth API?"
   - **If completed:**
     - Agent B reads `dependency_available` event with commit hash
     - Agent B pulls Agent A's branch: `git pull origin feature/agent-1-auth`
     - Agent B continues work with Agent A's code available
   - **If not completed:**
     - Agent B sends A2A message via Tier 2: `a2a:agent-2:to:agent-1:QueryDependency`
     - Agent A responds with ETA or commit hash when done
     - Agent B works on non-dependent tasks in the meantime
   - **Key insight:** Agents coordinate directly, no Master involvement
5. **Feature Completion:**

   - Run full test suite on feature branch
   - Create Pull Request when feature is ready
   - Commit `feature_completed` event to coordination branch with PR number
   - Release all remaining file locks
   - Request human review or auto-merge if tests pass
   - **Agent work is complete** - can take new assignment or shut down

**Agent Independence Benefits:**

- ✅ Fault tolerant: One agent crash doesn't block others
- ✅ Scalable: No central bottleneck for approvals
- ✅ Fast: No round-trip to Master for every operation
- ✅ Flexible: Agents can adjust strategy based on real-time conditions
- ✅ Recovery: Agents can resume from coordination branch after crashes

---

### Git Coordination Branch

**Purpose:** Append-only event log for LOW-FREQUENCY feature assignments and completions

**Key Design Principle:** Git branch for coordination state, NOT for high-frequency operations. Git provides version-controlled, auditable, human-readable coordination history that works locally and distributed.

**Branch Name:** `.yantra/coordination` (never merges to main)

**Why Git Branch for Coordination?**

✅ **No New Infrastructure:** Git is already there, no new database needed
✅ **Version Controlled:** Full history of assignments and completions
✅ **Human Readable:** `git log .yantra/coordination` shows all coordination events
✅ **Auditable:** Who assigned what to whom, when features completed
✅ **Works Everywhere:** Same mechanism works locally (single machine) and distributed (team with remote Git)
✅ **Merge Conflicts Rare:** Append-only events rarely conflict
✅ **Distributed by Design:** Every agent has local copy, syncs via push/pull

**What Goes in Coordination Branch (Low-Frequency):**

- ✅ Feature assignments by Master
- ✅ Work started events
- ✅ Dependency available events (APIs ready)
- ✅ Feature completion events with PR numbers

**What Does NOT Go Here (High-Frequency):**

- ❌ File lock claims/releases (Tier 2 sled handles this - 100k ops/sec)
- ❌ Real-time progress updates (Tier 2 sled handles this)
- ❌ Agent heartbeats (Tier 2 sled handles this)
- ❌ A2A messages (Tier 2 sled handles this)

**Separation of Concerns:**

- **Coordination branch:** Strategic coordination (assignments, completions) - low frequency
- **Tier 2 (sled):** Tactical coordination (locks, messages, state) - high frequency

**Event Types:**

```json
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
```

**Agent Operations:**

```rust
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
```

**Recovery After Crashes:**

One of the key benefits of the coordination branch is fault tolerance:

```rust
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
```

**Human Inspectable:**

Developers can directly inspect coordination state:

```bash
# See all coordination events
git log .yantra/coordination --oneline

# See current assignments
git show .yantra/coordination | grep "feature_assigned"

# See which dependencies are available
git log .yantra/coordination --grep="dependency_available"

# Debug why agent is blocked
git log .yantra/coordination --grep="agent-2" --since="1 hour ago"
```

This transparency is crucial for debugging and understanding system behavior.

---

### Architecture Design Principles

**Why This Design Is Clean and Scalable:**

**1. Master Is Not a Single Point of Failure**

Traditional multi-agent systems have a master that coordinates every operation. If the master crashes, all agents stop. Yantra's master only acts at the BEGINNING (work assignment). After assignment:

- ✅ Agents continue independently if Master crashes
- ✅ Agents don't need Master's permission for ongoing work
- ✅ Master can shut down or take new requests immediately after assignment

**2. Git Already Solves Code Coordination**

Rather than inventing new file coordination, Yantra uses proven Git infrastructure:

- ✅ **Branches provide isolation** - each agent works on their own branch
- ✅ **PRs provide integration** - completed work merges through normal Git flow
- ✅ **Conflicts surface at merge time** - handled like human team conflicts
- ✅ **Proven at scale** - Git handles millions of developers worldwide

**3. Matches Human Team Workflow**

Yantra mirrors how real development teams work:

| Human Team                        | Yantra Multi-Agent                       |
| --------------------------------- | ---------------------------------------- |
| Tech lead assigns features        | Master assigns sub-features              |
| Developers create branches        | Agents create feature branches           |
| Developers work independently     | Agents execute independently             |
| Developers coordinate when needed | Agents use A2A protocol for dependencies |
| Developers submit PRs             | Agents create PRs with tests             |
| Team reviews and merges           | Human reviews or auto-merge              |

**4. Separation of Concerns (Three Coordination Layers):**

| Layer                             | Purpose                             | Frequency              | Technology    | Examples                              |
| --------------------------------- | ----------------------------------- | ---------------------- | ------------- | ------------------------------------- |
| **Git Coordination Branch** | Strategic assignments & completions | Low (once per feature) | Git commits   | feature_assigned, feature_completed   |
| **Tier 2 (sled)**           | Tactical locks & messages           | High (many per second) | sled KV store | File locks, A2A messages, agent state |
| **Git Feature Branches**    | Code isolation                      | N/A                    | Git branches  | Actual code changes                   |

Each layer does what it's best at. No layer tries to do everything.

**5. Proactive Conflict Avoidance (Not Reactive Resolution)**

Traditional Git workflow:

1. Two developers modify same file without knowing
2. Both invest time in divergent implementations
3. Git detects conflict at merge time (REACTIVE)
4. Manual resolution required, often by someone who wrote neither

Yantra workflow:

1. Agent A claims file in Tier 2 sled
2. Agent B tries to claim same file
3. Tier 2 blocks immediately (PROACTIVE)
4. Agent B either waits or works on different task
5. No wasted work, no merge conflicts

**6. Event Sourcing for Recovery**

The coordination branch is an event log. If an agent crashes:

```rust
// Agent reads event log on restart
let events = git::read_json_commits(".yantra/coordination")?;

// "I was assigned feature X at timestamp T"
let my_assignment = events.find(|e| e.agent == self.id)?;

// "Here's my branch name"
let my_branch = my_assignment.branch;

// Agent resumes from where it left off
git::checkout_branch(my_branch)?;
self.continue_work(my_assignment)?;
```

State is never lost. Agents can always resume.

**7. Low-Hanging Fruit Implementation**

This architecture requires minimal new infrastructure:

- ✅ Git is already there (no new database for coordination)
- ✅ sled is embedded Rust library (no server to run)
- ✅ Coordination state is version controlled and auditable
- ✅ Works locally (single machine) and distributed (team) with same mechanism
- ✅ Merge conflicts on coordination branch are rare (append-only events)

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

**Philosophy: Prevent Conflicts BEFORE They Happen**

Traditional Git workflow is REACTIVE - conflicts detected after both developers finish work:

- Developer A modifies `payment.py` on branch A
- Developer B modifies `payment.py` on branch B
- Both invest days of work
- Git detects conflict at merge time
- Manual resolution required, understanding two divergent implementations

Yantra workflow is PROACTIVE - conflicts prevented before work starts:

- Agent A claims `payment.py` in sled
- Agent B tries to claim `payment.py`
- sled blocks immediately (atomic compare-and-swap)
- Agent B works on different files or waits
- No wasted work, no merge conflicts

**Git as the Coordination Layer:**

Git already solves file coordination through branches and merges. Yantra leverages this:

1. **Branches Provide Isolation:**

   - Each agent works on their own feature branch
   - File changes are isolated from other agents
   - No cross-agent corruption possible
2. **PRs Provide Integration:**

   - Completed work merges through normal Git flow
   - Tests run on each PR before merge
   - Human review or auto-merge based on test results
3. **Conflicts Surface at Merge Time:**

   - When they do occur (rare with file locking), handled like human team conflicts
   - Git's conflict resolution tools work normally
   - Developers understand this workflow already

**Proactive Prevention via File Locks:**

sled adds real-time lock coordination on top of Git:

- Agent claims file → lock created in sled
- Other agents check lock before claiming
- Atomic compare-and-swap prevents race conditions
- Locks released immediately when agent finishes with file

**Example: E-Commerce Checkout Feature with 6 Agents**

**User Request:** "Add complete checkout flow with Stripe payments"

**Step 1: Master Decomposes Using GNN (30 seconds)**

Master analyzes dependency graph to minimize coupling:

```
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
```

**Step 2: Agents Claim Files (1 second)**

```rust
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
```

**Key Insight:** Agent 2 detected dependency BEFORE starting work. No wasted effort.

**Step 3: Parallel Execution (10 minutes)**

```
Agent 1 (Backend):     [████████████████████] 100% - stripe.py, checkout.py ✅
                       Releases locks, commits dependency_available event

Agent 2 (Frontend):    [██████████░░░░░░░░░░] 50%  - Cart.tsx done, waiting for API
                       Polls coordination branch for Agent 1 completion

Agent 3 (Frontend):    [Blocked] - Cart.tsx claimed by Agent 2

Agent 4 (Testing):     [░░░░░░░░░░░░░░░░░░░░] 0%   - Waiting for Agent 1 & 2

Agent 5 (DevOps):      [████████████████████] 100% - deploy.yml done (parallel) ✅

Agent 6 (Docs):        [████████████████████] 100% - docs done (parallel) ✅
```

**Step 4: Dependency Resolution via Coordination Branch (1 minute)**

```json
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
```

```rust
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
```

Agent 2 completes Checkout.tsx in 5 minutes ✅

Agent 4 sees both Agent 1 and Agent 2 complete → Starts integration tests
Agent 4 completes in 3 minutes ✅

**Step 5: PR Creation (30 seconds)**

```
All agents complete → Create PRs:
├─ PR #1: Payment API (agent-1) ✅ Tests pass, 32 assertions
├─ PR #2: Checkout UI (agent-2) ✅ Tests pass, 18 assertions
├─ PR #3: Cart Integration (agent-2) ✅ Tests pass, 12 assertions
├─ PR #4: Integration Tests (agent-4) ✅ Tests pass, 45 E2E tests
├─ PR #5: Deployment (agent-5) ✅ CI pipeline validated
└─ PR #6: Documentation (agent-6) ✅ Spelling/link checks passed

Human reviews and merges (or auto-merge if confidence > 0.95)
```

**Total Time:** 15 minutes (vs 45 minutes single-agent) = **3x faster**

**Conflicts Prevented:**

- 🚫 Agent 2 didn't modify stripe.py while Agent 1 was working (proactive detection)
- 🚫 Agent 3 didn't conflict with Agent 2 on Cart.tsx (file lock prevented)
- 🚫 Agent 4 didn't start tests until APIs were complete (dependency tracking)
- 🚫 No Git merge conflicts (each agent on own branch with lock coordination)
- ✅ All work proceeded in optimal order with minimal blocking

**Why This Works:**

1. **GNN provides dependency knowledge** - Master knows what depends on what
2. **sled provides activity knowledge** - Agents know who's modifying what right now
3. **Combining both prevents conflicts** - Check dependencies AND check locks before claiming
4. **Git provides safety net** - Even if locks fail, branches prevent corruption
5. **Peer-to-peer coordination** - Agents don't need Master after initial assignment

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

- Master Agent: Feature decomposition using dependency graph (Note: code may call this "GNN" but it's the petgraph structure, not a neural network)
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

**IMPORTANT:** This is NOT "Cloud Dependency Graph Intelligence" - the dependency analysis runs locally in Tier 1. This is a cloud-hosted graph database for storage and real-time coordination across agents/users.

### Overview

**Problem:** With Team of Agents, each agent has a LOCAL dependency graph. When Agent A modifies a file, Agent B doesn't know until attempting to claim the same file or hitting a Git merge conflict. This is **reactive conflict detection** (bad).

**Solution:** **Cloud Graph Database (Tier 0)** - A shared, cloud-hosted dependency graph that tracks real-time file modifications across all agents and users working on the same project. Enables **proactive conflict prevention** BEFORE work starts.

**Key Innovation:** Combine dependency knowledge (from local dependency graph) with activity knowledge (who's modifying what) to warn agents about potential conflicts before they occur, including transitive dependencies.

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

### Success Metrics: Team of Agents + Cloud Graph Database

**Performance:**

- ✅ 3-10x faster feature completion with team of agents
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

**Team of Agents Benefits:**

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

**Hybrid Semantic Graph + GNN (Chosen):**

- ✅ **Semantic Graph (HNSW):** Semantic understanding (What does this function do? Find similar patterns)
- ✅ **GNN:** Structural dependencies (What imports what? Who calls this function?)
- ✅ **Combined:** Best of both worlds - single integrated data structure

---

### Semantic Enhanced Dependency Graph with HNSW

**Purpose:**

- Semantic code search for LLM context via HNSW indexing
- Pattern matching for similar code
- Documentation and comment search
- Error message similarity for auto-fix
- All integrated within the CodeGraph (no separate database)

**What Gets Stored in Semantic Graph:**

1. **Function/Class Nodes with Embeddings**

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

# Graph node with semantic properties
{
  "id": "func-calculate_payment_fee",
  "type": "function",
  "name": "calculate_payment_fee",
  "signature": "calculate_payment_fee(amount: float, currency: str) -> float",
  "docstring": "Calculate processing fee for payment...",
  "embedding": [0.123, -0.456, ...],  # 384-dim vector (fastembed)
  "hnsw_index_id": 42,                # HNSW index reference
  "file_path": "src/payment/fees.py",
  "line_start": 10,
  "line_end": 20
}
```

2. **Code Comments** (as node properties for context)
3. **Error Patterns** (as semantic relationships for auto-fix)
4. **Test Cases** (linked as semantic neighbors for similar test examples)
5. **Documentation** (README, API docs as special nodes)

**Semantic Search via HNSW:**

```python
# Agent needs to implement payment refund
query = "How to refund a payment transaction?"

# Semantic graph HNSW search returns similar code
results = semantic_graph.search_hnsw(query, top_k=3)
# Result 1: refund_payment() function in stripe.py
# Result 2: cancel_subscription() in billing.py
# Result 3: reverse_transaction() in ledger.py

# Agent uses these as LLM context for generation
```

**Performance:**

- **Search latency:** <10ms for HNSW semantic search (in-memory)
- **Index update:** Incremental (on file save, <50ms)
- **Storage:** Embeddings stored as node properties in graph

---

### GNN Integration for Structural Dependencies

**Purpose:**

- Track function calls, imports, inheritance
- Detect breaking changes (signature modifications)
- Validate dependencies before code execution
- Build task dependency graphs

**GNN vs Semantic Search - When to Use:**

| Use Case                                       | Tool            | Reason                |
| ---------------------------------------------- | --------------- | --------------------- |
| "Find functions that call UserService.login()" | GNN             | Structural dependency |
| "Find similar authentication code"             | Semantic Search | Semantic similarity   |
| "Will changing this function break anything?"  | GNN             | Impact analysis       |
| "How do other projects handle OAuth?"          | Semantic Search | Pattern search        |
| "What imports this module?"                    | GNN             | Direct dependency     |
| "Find code related to payment processing"      | Semantic Search | Semantic search       |

**Combined Query Example:**

```python
# Agent task: "Refactor UserService.login() to support 2FA"

# Step 1: GNN - Find all callers
callers = gnn.get_function_callers("UserService.login")
# Result: ["auth_api.py:45", "login_view.py:30", "test_auth.py:20"]

# Step 2: Semantic Graph HNSW - Find similar 2FA implementations
patterns = semantic_graph.search_hnsw("two-factor authentication implementation", top_k=5)
# Result: Similar code from semantic neighbors in graph

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

| Component                      | Sync Strategy            | Latency | Reason                           |
| ------------------------------ | ------------------------ | ------- | -------------------------------- |
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

| Mode                 | Speed      | Cost       | Quality     | Best For                                  |
| -------------------- | ---------- | ---------- | ----------- | ----------------------------------------- |
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

| Metric                       | Single Agent         | 3 Agents      | 10 Agents      |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ---------------------------- | -------------------- | ------------- | -------------- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| **Codebase Size**      | 10k LOC              | 50k LOC       | 100k+ LOC      |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Concurrent Tasks**   | 1                    | 3             | 10             |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Context Build Time** | <500ms               | <800ms        | <1.5s          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Conflict Detection** | N/A                  | <10ms         | <20ms          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Commit Frequency**   | Every task           | Every 3 tasks | Every 10 tasks |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Cost per Feature**   | $0.20        | $0.25 | $0.35         |                |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

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

**Weeks 5-6: Semantic Enhanced Dependency Graph**

- Integrate HNSW indexing into CodeGraph
- Semantic embeddings as node properties (fastembed)
- Fast semantic search (<10ms in-memory)
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

**Status:** � IN PROGRESS (75% Complete - Updated Dec 4, 2025)
**Implementation:** Backend (4,876 lines) + UI (785 lines) complete, versioning workflows pending
**Priority:** ⚡ MVP REQUIRED (Implement before Pair Programming)
**Specification:** 997 lines of detailed requirements
**Business Impact:** Design-first development, architecture governance
**User Request:** "Where is the visualization of architecture flow?"

**What's Working:**

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

**What's Pending (4 features):**

- ❌ Rule of 3 versioning (keep 4 versions, auto-delete oldest)
- ❌ Auto-save on every architecture change
- ❌ Real-time deviation alerts (backend → frontend wiring)
- ❌ Orchestrator integration (ensure blocking works in code generation flow)

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

**Version Metadata Requirements:**

- Incremental version number (1, 2, 3, 4...)
- Full architecture state snapshot
- Timestamp when created
- Change type classification (AgentGenerated, AgentUpdated, AgentReverted, GNNSynced)
- Agent's reasoning for the change
- Original user message that triggered the change

**Storage Requirements:**

- Persist only 4 versions (current + 3 past)
- Auto-delete oldest when creating 5th version
- Support efficient version retrieval and restoration

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

---

### Implementation Details

**Note:** For workflows, see **Product Development Cycle (PDC) Workflows** section above. This section covers technical implementation details only.

#### Deviation Detection System Implementation

**Core Requirements:**

- Monitor code alignment with architecture using GNN
- Compare actual vs expected dependencies
- Calculate deviation severity (None/Low/Medium/High/Critical)
- Generate user-friendly recommendations

**Deviation Types:**

- Unexpected Dependency (code imports something not in architecture)
- Missing Dependency (architecture expects it, code doesn't have it)
- Wrong Connection Type (REST instead of message queue)
- Layer Violation (UI directly accessing database)
- Circular Dependency (creates cycle in dependency graph)

**Integration Points:**

**1. During Code Generation (Proactive):**

- Trigger: Before writing generated code to disk
- Run deviation detection using tree-sitter parsing
- On deviation: pause generation, present 3 options (Update Architecture/Fix Code/Cancel)
- Wait for user decision, execute chosen action

**2. After File Save (Reactive):**

- Trigger: User manually edits and saves file
- Check if project has active architecture
- Run alignment check using GNN
- On misalignment: display warning with recommendations

---

### Data Storage Architecture

#### Primary Storage: SQLite Database

**Storage Location:** `.yantra/architecture.db` in user's home directory

**Required Data Tables:**

1. **Components Table:**

   - Store component metadata (id, name, type, description)
   - Track visual positioning (x/y coordinates, width/height)
   - Support hierarchical grouping (parent-child relationships)
   - Categorize by layer (frontend, backend, database, external, infrastructure)
   - Store additional metadata as JSON
   - Track creation and update timestamps
   - Cascade delete when parent is removed
2. **Connections Table:**

   - Store relationships between components (from/to component ids)
   - Define connection types (data_flow, api_call, event, dependency)
   - Support labels and descriptions
   - Enable bidirectional connections
   - Store additional metadata as JSON
   - Track creation timestamps
   - Cascade delete when components are removed
3. **Component Files Table:**

   - Link files to components (many-to-many relationship)
   - Track whether link is auto-generated or manual
   - Store confidence score for auto-linked files
   - Prevent duplicate file-component links
   - Track creation timestamps
   - Cascade delete when component is removed
4. **Architecture Versions Table:**

   - Store version history with incremental version numbers
   - Capture complete architecture state as JSON snapshot
   - Record change descriptions and change types
   - Preserve user intent and AI reasoning
   - Track which files were changed
   - Record timestamps for each version

**Data Integrity Requirements:**

- Enable SQLite Write-Ahead Logging (WAL) mode for reliability
- Run integrity checks on startup
- Create automatic backup before modifications
- Maintain last 10 backups in `.yantra/backups/` directory
- Support foreign key constraints and cascade deletes

#### Secondary Storage: Git-Friendly Exports

**Export Formats:**

- **architecture.md**: Human-readable Markdown with Mermaid diagrams
- **architecture.json**: Machine-readable complete state for tooling

**Export Triggers:**

- Automatically after every architecture modification
- On-demand via agent command
- Before git commit (via git hook integration)

**Recovery Strategy:**

1. Primary: Use SQLite database (fast, transactional)
2. Fallback 1: Restore from architecture.json if database corrupted
3. Fallback 2: Regenerate from GNN code analysis if JSON corrupted
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

**Process:**

1. Send user's intent to LLM with structured prompt
2. LLM returns JSON with components (id, name, type, layer, description) and connections (from/to, type, label)
3. Parse and validate response
4. Calculate initial visual positions
5. Store in database
6. Display in Architecture View for user review

**Example:** "Create 3-tier web app with React, FastAPI, PostgreSQL"
→ Components: UI (React), API (FastAPI), DB (PostgreSQL)
→ Connections: UI→API (api_call), API→DB (data_flow)

#### 4.2 Architecture Generation from Code (GNN Analysis)

**Algorithm:**

1. Use GNN to get complete dependency graph
2. Group files into components by directory structure (e.g., `src/auth/` → Auth Service)
3. Infer connections from cross-directory imports
4. Detect technology patterns (ORM imports → database, SDK imports → external services)
5. Generate architecture JSON
6. Present for user refinement

#### 4.3 Code-Architecture Alignment Validation

**Process:**

1. Trigger: On file save or before code generation
2. Identify which component owns the modified file
3. Get expected connections from architecture
4. Get actual dependencies from GNN
5. Compare and detect violations (calculate severity)
6. Use LLM to generate user-friendly explanation and 3 options
7. Present alert in UI

**Options:** (a) Update architecture, (b) Revert code, (c) Refactor code

---

### 5. Implementation Components Requirements

#### Backend Module Organization

**Required Modules:**

- Main facade/entry point
- SQLite CRUD operations
- Data types (Component, Connection models)
- Versioning system (snapshots, restore)
- AI generator (architecture from intent)
- Code analyzer (architecture from GNN)
- Validator (code-architecture alignment)
- Exporter (Markdown, Mermaid, JSON)
- Command handlers (Tauri commands)

#### Core Data Models

**Component Model:**

- Fields: id, name, description, type (Service/Module/Layer/Database/External/UIComponent), layer (frontend/backend/database/external/infrastructure), position (x, y), files array, metadata JSON

**Connection Model:**

- Fields: id, from_component_id, to_component_id, type (DataFlow/ApiCall/Event/Dependency), label, bidirectional flag, metadata JSON

#### Backend API Commands

1. `create_component`, `update_component`, `delete_component`
2. `create_connection`
3. `get_architecture` (retrieve complete)
4. `save_version`, `list_versions`, `restore_version`
5. `export_architecture` (format: markdown/json/mermaid)
6. `validate_alignment`

#### Frontend Component Organization

**UI Components:**

- Architecture canvas (React Flow)
- Custom node component (component boxes)
- Custom edge component (connection arrows with different styles)
- Hierarchical tab navigation (layer filtering)
- Component inspector panel
- Validation alerts (misalignment warnings)

**State Management (Architecture Store):**

- State: components list, connections list, selected component, layer filter, alignment issues, loading/error states
- Actions: load, create/update/delete component, create connection, validate, export

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

---

## Project Initialization & Architecture-First Implementation (MVP Phase 1 - HIGH PRIORITY)

### Overview

**Status:** 🔴 NOT STARTED
**Priority:** ⚡ MVP REQUIRED (Critical for architecture-first development)
**Created:** November 28, 2025
**Business Impact:** Ensures architecture always exists before code generation, prevents architectural drift

**Note:** For detailed workflows, see the **Product Development Cycle (PDC) Workflows** section above. This section focuses on implementation components and technical requirements.

**Workflows Implemented:**

- Phase 1 (Architect/Design): New Project Initialization Workflow (1.1), Existing Project Import Workflow (1.2), Architecture Maintenance & Governance Workflow (1.3)
- These enforce Yantra's core principle: **Architecture-First Development** where every project has reviewed and approved architecture before code implementation begins.

---

### Implementation Components Requirements

#### A. Project Initializer Module

**Location:** Backend architecture module

**Required Capabilities:**

1. **Initialize New Project:**

   - Accept user intent and project path
   - Generate architecture from intent using LLM
   - Return initialization result with architecture ID
   - Mark architecture for user review
2. **Initialize Existing Project:**

   - Accept project path
   - Scan for existing architecture files
   - Import architecture from found files
   - If no architecture found, request user input
   - Return initialization result
3. **Detect Architecture Files:**

   - Search project directory for architecture documentation
   - Check 6 common locations (see list below)
   - Return list of found architecture files with priority order
4. **Import Architecture:**

   - Parse architecture from multiple file formats (Markdown, JSON, Mermaid, PlantUML)
   - Convert to internal architecture representation
   - Validate imported architecture structure
   - Return parsed architecture object
5. **Review Existing Code:**

   - Run GNN dependency analysis on project
   - Execute security scan using Semgrep
   - Validate code alignment with architecture
   - Calculate code quality metrics
   - Generate prioritized recommendations
   - Return comprehensive code review result
6. **Analyze Requirement Impact:**

   - Parse user requirement using LLM
   - Compare against current architecture
   - Use GNN to identify affected files and components
   - Calculate impact severity
   - Generate architecture change preview
   - Return impact analysis result

**Initialization Result Data:**

- Architecture unique identifier
- Source of architecture (Generated/Imported/UserProvided)
- Whether review is required before proceeding
- Deep link URL to Architecture View tab

**Architecture Sources:**

- Generated from user intent (LLM-based)
- Generated from code analysis (GNN-based)
- Imported from existing file
- Manually provided by user

**Code Review Result Data:**

- Total number of files analyzed
- List of security issues found
- List of architecture alignment issues
- Code quality metrics (complexity, test coverage)
- Prioritized list of recommendations

**Architecture Impact Data:**

- Boolean flag indicating if architecture changes needed
- List of new components to add
- List of modified connections
- List of affected existing components (by ID)
- Impact severity level (Low/Medium/High/Breaking)

**Impact Severity Levels:**

- **Low:** Changes within single component, no new dependencies
- **Medium:** Changes across 2-3 components using existing dependencies
- **High:** New component required
- **Breaking:** Fundamental architectural pattern change (e.g., monolith → microservices)

---

#### B. Architecture File Detection

**Multi-Level Search Strategy:**

**Level 1: Priority Files (Exact Paths)**
Search these first for immediate results:

1. `.yantra/architecture.db` - Yantra's native SQLite database (100% confidence)
2. `.yantra/architecture.json` - Yantra's JSON export format (90% confidence)
3. `architecture.md` / `ARCHITECTURE.md` - Root directory standard files (80% confidence)
4. `architecture.json` - Root directory JSON format (80% confidence)

**Level 2: Common Documentation Locations (Pattern Match)**
Scan these folders for architecture files:

1. **Root directory:** All `*.md` files (scan for architecture sections)
2. **docs/ folder:** All `*.md` files recursively
3. **.github/ folder:** All `*.md` files (often contains system design docs)
4. **wiki/ folder:** All `*.md` files if present
5. **design/ folder:** All `*.md` files
6. **architecture/ folder:** All `*.md` and `*.json` files
7. **ADR locations:**
   - `docs/adr/*.md`
   - `docs/decisions/*.md`
   - `adr/*.md`

**Level 3: Content-Based Detection (Deep Scan)**
For found markdown files, search for:

**Architecture Section Headers:**

- `# Architecture`
- `## Architecture`
- `# System Architecture`
- `## System Design`
- `# Technical Architecture`
- `## Component Architecture`
- `## Architecture Overview`
- `# Design`
- `## Architecture Diagram`
- `## System Overview`

**Diagram Format Detection:**

- **Mermaid:**
  - `graph TD`, `graph LR`, `flowchart`
  - `C4Context`, `C4Container`, `C4Component`
  - Keywords: "component", "service", "database", "external"
- **PlantUML:**
  - `@startuml`, `@enduml`
  - `component`, `package`, `node`, `cloud`
  - Architecture-specific keywords in diagram
- **ASCII Diagrams:**
  - Box-drawing characters: `│`, `─`, `┌`, `┐`, `└`, `┘`
  - Arrow characters: `→`, `←`, `↔`, `⇒`

**Architecture Indicators (Keyword Scoring):**

- **High-value keywords** (5 points each):
  - "component", "service", "microservice", "module", "layer"
  - "frontend", "backend", "database", "API", "gateway"
- **Medium-value keywords** (3 points each):
  - "authentication", "authorization", "cache", "queue"
  - "external", "integration", "connector", "adapter"
- **Connection keywords** (2 points each):
  - "communicates with", "depends on", "calls", "connects to"
  - "uses", "integrates with", "sends", "receives"

**Confidence Scoring Algorithm:**
Calculate score for each found file:

- **Base score:**
  - Native format (.yantra/): 100 points
  - Standard name (ARCHITECTURE.md): 80 points
  - README with section: 60 points
  - Other docs: 40 points
- **Add points for:**
  - Has diagram (+20 points)
  - Has component list (+15 points)
  - Has connections described (+10 points)
  - High keyword density (+5 to +15 points)
- **Final confidence percentage:**
  - 90-100%: Excellent (use immediately)
  - 70-89%: Good (use with confirmation)
  - 50-69%: Medium (present as option)
  - 30-49%: Low (combine with other sources)
  - <30%: Very low (request user guidance)

**Supported Import Formats:**

1. **Yantra Native:**

   - SQLite database (`.yantra/architecture.db`)
   - JSON export (`.yantra/architecture.json`)
2. **Mermaid Diagrams:**

   - Parse graph syntax for nodes and edges
   - Extract labels and connection types
   - Support flowchart, C4 diagrams
3. **JSON Formats:**

   - Yantra schema (components, connections arrays)
   - C4 model format
   - Custom JSON with components/services structure
4. **Markdown Structured:**

   - Component lists (bullet points or tables)
   - Connection descriptions
   - Embedded Mermaid/PlantUML diagrams
5. **PlantUML:**

   - Component diagrams
   - Package diagrams
   - Basic class diagrams with architecture focus
6. **Architecture Decision Records (ADRs):**

   - Aggregate multiple ADR files
   - Extract architectural decisions
   - Build component map from decisions

**Performance & Safety Limits:**

- **Max files to scan:** 1000 (configurable)
- **Max file size to read:** 5 MB per file
- **Timeout:** 30 seconds for entire scan
- **Folders to skip:**
  - `node_modules/`, `venv/`, `env/`, `.venv/`
  - `vendor/`, `target/`, `build/`, `dist/`
  - `.git/`, `.svn/`, `.hg/`
  - `__pycache__/`, `.pytest_cache/`
  - `coverage/`, `.nyc_output/`
- **Files to skip:**
  - Binary files (executables, images, videos)
  - Minified files (`*.min.js`, `*.min.css`)
  - Lock files (`package-lock.json`, `yarn.lock`)
  - Compiled files (`*.pyc`, `*.class`, `*.o`)

**Caching Strategy:**

- Cache scan results for current session
- Invalidate cache if new files detected
- Store confidence scores with results
- Remember user's selection for future sessions

---

#### C. Requirement Impact Analysis Algorithm

**Process Steps:**

1. **Parse Requirement with LLM:**

   - Extract mentioned entities (services, features, data types)
   - Identify actions (create, modify, delete, integrate)
   - Detect integrations with external systems
   - Classify change type (new feature, modification, refactoring)
2. **Compare Against Current Architecture:**

   - Check if mentioned entities exist as components
   - Check if mentioned integrations exist as connections
   - Identify implicit dependencies not explicitly mentioned
   - Detect conflicts with existing patterns
3. **Analyze Code Impact Using GNN:**

   - Identify which files would need modification
   - Determine which components own those files
   - Calculate ripple effects through dependency graph
   - Estimate number of files affected
4. **Calculate Impact Severity:**

   - **Low:** Changes within 1 component, no new dependencies
   - **Medium:** Changes across 2-3 components, existing dependencies
   - **High:** New component needed, new external dependency
   - **Breaking:** Fundamental pattern change requires architecture redesign
5. **Generate Change Preview:**

   - Create before/after architecture visualization
   - List all affected components with change descriptions
   - Estimate implementation complexity (hours/days)
   - Highlight potential risks or breaking changes

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

**Documentation Data Structure Requirements:**

The system must define and maintain the following core data types:

**Feature Data:**

- Unique identifier (string)
- Title describing the feature
- Detailed description
- Status indicator (Planned, InProgress, or Completed)
- Source attribution showing where the feature was extracted from
- ISO 8601 timestamp of extraction or last update

**Decision Data:**

- Unique identifier (string)
- Decision title
- Context explaining the situation requiring the decision
- The actual decision made
- Rationale explaining why this decision was chosen
- ISO 8601 timestamp

**Change Data:**

- Unique identifier (string)
- Change type classification (FileAdded, FileModified, FileDeleted, etc.)
- Human-readable description
- List of affected file paths
- ISO 8601 timestamp

**Task Data:**

- Unique identifier (string)
- Task title
- Status (Completed, InProgress, or Pending)
- Milestone or phase the task belongs to
- List of task dependencies (task IDs this task depends on)
- Flag indicating if user action is required
- Optional user action instructions (present only if user action required)

**DocumentationManager State:**

- Workspace directory path
- Collection of all features
- Collection of all decisions
- Collection of all changes
- Collection of all tasks

**2. Frontend Store - `src-ui/stores/documentationStore.ts`**

**Documentation Store Requirements:**

The frontend must maintain a reactive store with:

**Store State:**

- Array of features
- Array of decisions
- Array of changes
- Array of tasks
- Loading flag indicating data fetch in progress
- Error message (null if no error)

**Store Initialization:**

- Must initialize with empty arrays for all collections
- Must set loading flag to false
- Must set error to null

**Data Loading Function:**

When loading documentation, the system must:

**Step 1: Set Loading State**

- Set loading flag to true to indicate data fetch starting

**Step 2: Parallel Data Fetching**

- Must invoke backend commands for all four data types in parallel
- Must fetch features using 'get_features' command
- Must fetch decisions using 'get_decisions' command
- Must fetch changes using 'get_changes' command
- Must fetch tasks using 'get_tasks' command
- Must wait for all four fetch operations to complete

**Step 3: Success Path**

- Update store with all fetched data
- Set loading flag to false
- Clear any previous errors (set to null)

**Step 4: Error Path**

- If any fetch fails, capture the error message
- Store error message as string
- Set loading flag to false
- Preserve previously loaded data

**3. UI Component - `src-ui/components/DocumentationPanels.tsx`**

**Documentation Panel UI Requirements:**

The system must provide a tabbed interface for viewing documentation:

**Tab State Management:**

- Must track currently active tab (features, decisions, changes, or tasks)
- Must default to 'features' tab on initial load
- Must allow switching between tabs via button clicks

**Lifecycle Management:**

- When component mounts, must automatically trigger documentation loading

**Tab Navigation UI:**

- Must display four tab buttons in a horizontal row
- Each button must show:
  - Tab name (Features, Decisions, Changes, or Plan)
  - Item count in parentheses
  - Active state styling when selected
- Plan tab must show count of pending tasks only (filter out completed)

**Tab Content Display:**

- Must render content area below tab navigation
- Must display only the content for the currently active tab
- Must conditionally render:
  - FeaturesView component when features tab active
  - DecisionsView component when decisions tab active
  - ChangesView component when changes tab active
  - TasksView component when tasks/plan tab active
- Must pass relevant data array to each view component

---

### Extraction Algorithms

#### 1. Task Extraction from Project_Plan.md / IMPLEMENTATION_STATUS.md

**Pattern Recognition Requirements:**

The system must extract tasks from project plan markdown files using the following algorithm:

**Initialization:**

- Must initialize current milestone to "MVP" as default
- Must initialize task ID counter to 0

**Line-by-Line Processing:**

For each line in the markdown content:

**Step 1: Milestone Detection**

- Must check if line contains "Week" or "Phase" keywords
- If detected:
  - Must capture the trimmed line text as the current milestone
  - All subsequent tasks belong to this milestone until next milestone header

**Step 2: Task Detection**

- Must check if trimmed line starts with "- [" (checkbox syntax)
- If detected, must increment task ID counter

**Step 3: Status Determination**

- Must examine checkbox contents and emoji indicators
- Must classify as:
  - **Completed** if line contains "[x]" or "[X]"
  - **InProgress** if line contains "🔄" emoji or "In Progress" text
  - **Pending** otherwise (empty checkbox)

**Step 4: Title Extraction**

- Must split line by "]" character
- Must take the text after the first "]" (index 1)
- Must trim whitespace
- Must remove leading "\*" characters (markdown emphasis)
- Must trim again to get clean title

**Step 5: Task Creation**

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

**Pattern Recognition Requirements:**

The system must extract features from the Features.md file using the following algorithm:

**Initialization:**

- Must initialize feature ID counter to 0
- Must initialize current description as empty string
- Must initialize in_feature_section flag to false

**Line-by-Line Processing:**

For each line in the markdown content:

**Step 1: Feature Header Detection**

- Must check if line starts with "###" (third-level markdown header)
- Must check if line contains status emojis: "✅" (completed) OR "🔄" (in progress) OR "⏳" (planned)
- If both conditions met:
  - Must increment feature ID counter
  - Must set in_feature_section flag to true

**Step 2: Status Determination from Emoji**

- Must map emoji to status:
  - "✅" → FeatureStatus::Completed
  - "🔄" → FeatureStatus::InProgress
  - "⏳" → FeatureStatus::Planned

**Step 3: Title Extraction**

- Must remove leading "#" characters and trim whitespace
- Must remove all status emojis ("✅", "🔄", "⏳")
- Must trim again to get clean title

**Step 4: Feature Creation**

- Must create new Feature with:
  - ID from counter (converted to string)
  - Extracted title
  - Empty description (populated in next steps)
  - Determined status
  - extracted_from set to "Features.md"
  - timestamp set to current UTC time in RFC3339 format
- Must add feature to features collection

**Step 5: Description Capture**

- If in_feature_section is true AND line is not empty AND line doesn't start with "##":
  - Must get reference to most recently added feature
  - Must append current line to feature's description
  - Must append newline character to maintain formatting

**Step 6: Section End Detection**

- If line starts with "##" (second-level header):
  - Must set in_feature_section to false
  - Indicates moving to next major section

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

**Pattern Recognition Requirements:**

The system must extract decisions from the Decision_Log.md file using the following algorithm:

**Initialization:**

- Must initialize decision ID counter to 0
- Must initialize current_decision as None (no active decision)
- Must initialize section_type as None (not in any subsection)

**Line-by-Line Processing:**

For each line in the markdown content:

**Step 1: Decision Header Detection**

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

**Step 2: Subsection Type Detection**

- Must check for subsection markers:
  - If line starts with "**Context:**" or contains "Context:":
    - Set section_type to "context"
  - If line starts with "**Decision:**" or contains "Decision:":
    - Set section_type to "decision"
  - If line starts with "**Rationale:**" or contains "Rationale:":
    - Set section_type to "rationale"

**Step 3: Content Capture**

- If current_decision exists AND line is not empty AND line doesn't start with "\*\*":
  - Must append line to appropriate section based on section_type:
    - If section_type is "context": append to context field
    - If section_type is "decision": append to decision field
    - If section_type is "rationale": append to rationale field
  - Must append newline character after each line

**Step 4: Final Decision Save**

- After processing all lines, must check if current_decision exists
- If exists, must add it to decisions collection (captures last decision in file)

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

**Backend API Requirements - `src-tauri/src/main.rs`:**

The system must expose the following Tauri commands for frontend-to-backend communication:

**get_features Command:**

- Must accept AppState as parameter containing the DocumentationManager
- Must acquire lock on the documentation manager (thread-safe access)
- Must retrieve the features collection from the manager
- Must return a cloned vector of all features
- Must return error string if operation fails

**get_decisions Command:**

- Must accept AppState as parameter
- Must acquire lock on the documentation manager
- Must retrieve the decisions collection
- Must return a cloned vector of all decisions
- Must return error string if operation fails

**get_changes Command:**

- Must accept AppState as parameter
- Must acquire lock on the documentation manager
- Must retrieve the changes collection
- Must return a cloned vector of all changes
- Must return error string if operation fails

**get_tasks Command:**

- Must accept AppState as parameter
- Must acquire lock on the documentation manager
- Must retrieve the tasks collection
- Must return a cloned vector of all tasks
- Must return error string if operation fails

**reload_documentation Command:**

- Must accept AppState as parameter
- Must acquire mutable lock on the documentation manager
- Must call load_from_files() to reload all documentation from disk
- Must return unit type (empty success) or error string

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

- [X] **Completed** (green checkmark)
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

| File                                          | Lines | Purpose                                   | Tests  |
| --------------------------------------------- | ----- | ----------------------------------------- | ------ |
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

**Overview:**

The Documentation Panels provide real-time visibility into project features, decisions, changes, and plans. All four tabs support multi-user synchronization, automatic extraction from multiple sources, and intelligent Agent management.

**4 Tabs:**

- 📋 **Features** - What you're building (auto-extracted from docs, chat, code, external tools)
- 💡 **Decisions** - Why you chose specific approaches (approval audit trail)
- 📝 **Changes** - What files were modified (complete audit trail)
- 🎯 **Plan** - Tasks organized by milestones (persistent project plan)

**Multi-User Synchronization (ALL 4 TABS):**

When multiple users work on the same project:

- All users see the same view in real-time
- Updates synchronize instantly across all connected clients
- WebSocket-based real-time updates
- Conflict-free replicated data types (CRDTs) for eventual consistency
- Optimistic UI updates with server confirmation

**Search Functionality (ALL 4 TABS):**

- Real-time filtering as user types
- Search scope:
  - **Features:** Title + description + source attribution
  - **Decisions:** Context + decision + rationale + alternatives
  - **Changes:** Description + file names
  - **Plan:** Task titles + dependencies + milestone names
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

Each tab MUST include a brief explanation of where data comes from and key features:

- **Features:** "Features are automatically extracted from your documentation, chat conversations, and code files. Agent monitors external tools like Notion for feature updates. Status updates in real-time as implementation progresses, with accurate completion tracking. All team members see the same synchronized view."
- **Decisions:** "Critical technical decisions are logged here with full context to serve as an approval audit trail. Each decision includes why it was made, what alternatives were considered, and the rationale behind the choice. Timestamps show when Agent proposed and when user approved. All team members see the same synchronized view."
- **Changes:** "Complete audit trail of all code changes. Track what files were added, modified, or deleted, along with timestamps and descriptions. All team members see the same synchronized view."
- **Plan:** "Your project plan with tasks organized by milestones. Agent confirms milestones and prioritization before starting work. Dependencies are tracked automatically, and tasks requiring your input are highlighted. The plan persists across sessions and all team members see the same synchronized view."

**Purpose:** Reduce user confusion, set expectations for automation and multi-user sync, explain empty states, clarify Agent's role.

---

#### Features Tab - Detailed Requirements

**Purpose:** Automatically extract and track features from multiple sources with accurate completion tracking.

**Feature Extraction Sources:**

1. **Documentation Files (MVP):**

   - Markdown files (README.md, docs/\*.md)
   - Project documentation
   - Technical specifications
   - Parser: CommonMark/GFM with heading detection
2. **Chat Conversations (MVP):**

   - User requests: "Add user authentication"
   - Natural language requirements
   - Feature discussions
   - NLP: LLM-based intent extraction
3. **Code Files (MVP):**

   - Existing implementations
   - Code comments and docstrings
   - Function/class definitions
   - Parser: Tree-sitter with semantic analysis
4. **External Tools:**

   - **Notion (MVP):** Extract features from Notion pages via Notion API
   - **Confluence (Post-MVP):** Extract features from Confluence spaces
   - **Linear (Post-MVP):** Import issues and feature requests

**Completion Tracking (Accurate & Automated):**

Agent tracks feature completion by monitoring:

- **Code generation:** Files created/modified
- **Test pass rates:** Unit/integration tests passing
- **Integration status:** Dependencies satisfied
- **Deployment status:** Code deployed to target environment

Features automatically move through status:

- ⏳ **Planned** (0% complete) → 🔄 **In Progress** (1-99% complete) → ✅ **Done** (100% complete)

**Completion Calculation:**

```rust
pub fn calculate_feature_completion(feature: &Feature, gnn: &GNNEngine) -> f32 {
    let total_tasks = feature.required_files.len() + feature.required_tests.len();
    let completed_files = feature.required_files.iter()
        .filter(|f| gnn.file_exists(f) && gnn.file_has_no_errors(f))
        .count();
    let completed_tests = feature.required_tests.iter()
        .filter(|t| gnn.test_exists(t) && gnn.test_passes(t))
        .count();

    ((completed_files + completed_tests) as f32 / total_tasks as f32) * 100.0
}
```

**Data Model:**

```rust
pub struct Feature {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub source: FeatureSource, // Chat | Docs | Code | Notion | Confluence | Linear
    pub status: FeatureStatus,  // Planned | InProgress | Done
    pub completion_percentage: f32,
    pub required_files: Vec<PathBuf>,
    pub required_tests: Vec<String>,
    pub dependencies: Vec<Uuid>, // Other feature IDs
    pub extracted_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub enum FeatureSource {
    Chat { message_id: String },
    Docs { file_path: PathBuf, line_number: usize },
    Code { file_path: PathBuf, function_name: String },
    Notion { page_id: String, workspace_id: String },
    Confluence { page_id: String, space_key: String },
    Linear { issue_id: String, team_id: String },
}
```

**UI Display:**

```
┌─────────────────────────────────────────────┐
│ 📋 Add User Authentication        ✅ Done   │
│ JWT tokens with bcrypt password hash        │
│ Extracted from: Chat conversation           │
│ Completion: 100% (Tests passing)            │
│ Files: auth.py, test_auth.py (2/2)          │
└─────────────────────────────────────────────┘
```

---

#### Decisions Tab - Detailed Requirements

**Purpose:** Document all project decisions with full context to serve as an approval audit trail.

**Decision Sources:**

- Chat conversations with user
- Technical choices made during implementation
- Architecture decisions
- Technology selections (database, framework, libraries)
- Design pattern choices
- Trade-offs and alternatives considered

**MVP: Approval Audit View**

For MVP, the Decisions tab serves as the **Approver Audit View**:

- All critical decisions logged with timestamps
- Full context showing why decision was made
- User approval/confirmation captured
- Alternatives considered documented
- Complete audit trail for compliance/governance

**Decision Logging Workflow:**

1. Agent encounters decision point during implementation
2. Agent analyzes options using LLM + GNN context
3. Agent proposes recommendation with full context
4. Decision logged in Decisions tab (status: ⏳ Pending Approval)
5. User reviews in chat and approves/modifies
6. Approval captured with timestamp (status: ✅ Approved)
7. All team members see updated decision instantly via WebSocket

**Data Model:**

```rust
pub struct Decision {
    pub id: Uuid,
    pub title: String,
    pub context: String,           // Why decision needed
    pub decision: String,          // What was chosen
    pub alternatives: Vec<Alternative>, // Options considered
    pub rationale: String,         // Why this choice
    pub proposed_at: DateTime<Utc>,
    pub proposed_by: Actor,        // Agent | User
    pub approved_at: Option<DateTime<Utc>>,
    pub approved_by: Option<String>, // User ID
    pub status: DecisionStatus,    // Pending | Approved | Rejected
    pub impact: ImpactLevel,       // Low | Medium | High | Critical
}

pub struct Alternative {
    pub name: String,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
    pub reason_rejected: String,
}
```

**UI Display:**

```
┌─────────────────────────────────────────────┐
│ Use PostgreSQL over MySQL                   │
│ 👤 User Approved ✅                          │
│                                             │
│ Context:                                    │
│ Need JSONB support for flexible schema      │
│                                             │
│ Decision:                                   │
│ PostgreSQL 14+ with JSONB columns           │
│                                             │
│ Alternatives Considered:                    │
│ • MySQL 8.0 - Limited JSON support          │
│ • MongoDB - No ACID guarantees              │
│                                             │
│ Rationale:                                  │
│ Better JSON performance, native support,    │
│ full ACID compliance for critical data      │
│                                             │
│ Nov 28, 2025 10:30 AM - Agent proposed     │
│ Nov 28, 2025 10:32 AM - User approved      │
└─────────────────────────────────────────────┘
```

---

#### Changes Tab - Detailed Requirements

**Purpose:** Provide a complete audit trail of all code changes committed during the project with file-level details and synchronization across team members.

**Change Tracking Sources:**

- Git commits made by Agent
- File additions, modifications, and deletions
- Code refactoring operations
- Dependency updates (package.json, requirements.txt, Cargo.toml)
- Configuration file changes
- Test file changes

**MVP: Complete Change Log View**

For MVP, the Changes tab serves as the **Complete Change Audit Trail**:

- All committed changes tracked with timestamps
- File-level granularity (what files were touched)
- Change descriptions auto-generated by Agent
- Commit hashes for full traceability
- Real-time updates across all team members
- Integration with Git history

**Change Logging Workflow:**

1. Agent completes implementation of a task/feature
2. Agent stages changes using Git
3. Agent generates descriptive commit message
4. Agent commits changes to repository
5. Change logged in Changes tab automatically
6. All team members see new change instantly via WebSocket
7. Change entry includes: commit hash, files changed, description, timestamp

**Data Model:**

```rust
pub struct ChangeLog {
    pub id: Uuid,
    pub project_id: Uuid,
    pub commit_hash: String,           // Git commit SHA
    pub branch: String,                // Git branch name
    pub description: String,           // What was changed and why
    pub files_added: Vec<String>,      // New files created
    pub files_modified: Vec<String>,   // Existing files changed
    pub files_deleted: Vec<String>,    // Files removed
    pub author: String,                // Agent ID or User ID
    pub timestamp: DateTime<Utc>,
    pub related_task_id: Option<Uuid>, // Link to task in Plan
    pub related_feature_id: Option<Uuid>, // Link to feature
    pub lines_added: usize,            // Stats for dashboard
    pub lines_deleted: usize,
    pub impact_level: ChangeImpact,    // Minor | Medium | Major
}

pub enum ChangeImpact {
    Minor,      // Small bug fixes, formatting, comments
    Medium,     // Feature additions, refactoring
    Major,      // Architecture changes, breaking changes
}
```

**UI Display:**

```
┌─────────────────────────────────────────────┐
│ Implement User Authentication System        │
│ 🤖 Agent • Nov 28, 2025 2:45 PM            │
│ Branch: feature/auth • Commit: a3f9c2e     │
│                                             │
│ Files Added (3):                            │
│ • src/auth/jwt.rs                           │
│ • src/auth/password.rs                      │
│ • tests/auth_test.rs                        │
│                                             │
│ Files Modified (2):                         │
│ • src/main.rs (+45, -12)                    │
│ • Cargo.toml (+2, -0)                       │
│                                             │
│ Description:                                │
│ Implemented JWT-based authentication with   │
│ bcrypt password hashing. Added middleware   │
│ for protected routes. Includes unit tests   │
│ with 95% coverage.                          │
│                                             │
│ Related: Task #12 (User Authentication)     │
│ Impact: Major 🔴                            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Fix login endpoint CORS headers             │
│ 🤖 Agent • Nov 28, 2025 1:15 PM            │
│ Branch: main • Commit: b7e4d1a             │
│                                             │
│ Files Modified (1):                         │
│ • src/api/auth.rs (+8, -3)                  │
│                                             │
│ Description:                                │
│ Added CORS headers to login endpoint to     │
│ allow requests from frontend. Updated error │
│ handling for invalid credentials.           │
│                                             │
│ Related: Bug Fix                            │
│ Impact: Minor 🟢                            │
└─────────────────────────────────────────────┘
```

**Filter and Search:**

- **By Date Range:** "Last 7 days", "Last 30 days", "Custom range"
- **By Impact Level:** Minor | Medium | Major
- **By File Type:** .rs | .ts | .py | .toml | etc.
- **By Author:** Agent ID or User ID
- **By Branch:** main | feature/_ | bugfix/_
- **Search:** Full-text search in descriptions and file names

**Integration with Git:**

```rust
// src-tauri/src/documentation_panels/changes_tracker.rs

pub async fn track_commit(
    commit_hash: &str,
    branch: &str,
    files: Vec<GitFileChange>,
    message: &str,
) -> Result<ChangeLog> {
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
```

**Real-Time Synchronization:**

- Changes tab updates automatically when Agent commits
- WebSocket notifications push changes to all team members
- No manual refresh needed
- Optimistic UI updates with rollback on conflict

**Storage:**

- **Database:** SQLite table `change_logs` (Tier 3 - Read-Heavy Reference Data)
- **Columns:** id, project_id, commit_hash, branch, description, files_added, files_modified, files_deleted, author, timestamp, related_task_id, related_feature_id, lines_added, lines_deleted, impact_level
- **Indexes:** timestamp DESC (for recent changes), project_id (for filtering), commit_hash (for deduplication)

**Export Capabilities:**

- Export to CSV: Full change log with all details
- Export to Markdown: Formatted change log for documentation
- Export date range: Custom date filtering
- Export by author: Filter by specific agent/user

**Empty State:**

```
┌─────────────────────────────────────────────┐
│          📝 No Changes Yet                   │
│                                             │
│ Changes will appear here automatically      │
│ as Agent commits code to the repository.    │
│                                             │
│ All committed changes are tracked with:     │
│ • File-level details                        │
│ • Timestamps                                │
│ • Commit hashes                             │
│ • Change descriptions                       │
└─────────────────────────────────────────────┘
```

**Performance Targets:**

- Load last 100 changes: <100ms
- Real-time update latency: <500ms
- Search across 1000+ changes: <200ms
- Database query optimization: Indexed on timestamp and project_id

---

#### Plan Tab - Detailed Requirements

**Purpose:** Create and maintain a persistent project-level plan that Agent methodically executes and tracks.

**Plan Creation Process:**

1. Agent analyzes requirements from chat/docs/code
2. Agent breaks down into tasks with dependencies
3. Agent organizes tasks by milestones
4. **Agent proposes plan to user for confirmation**
5. **User confirms/modifies milestones and prioritization**
6. Agent begins execution only after user approval

**Plan Persistence:**

- **Project-level plan** stored in SQLite database
- Plan survives application restarts
- Task status preserved and updated continuously
- Historical task data maintained for audit trail
- Milestone completion tracked over time

**Sub-Task Tracking:**

Agent can dynamically add sub-tasks to track:

- Implementation steps (e.g., "Create database schema")
- Testing requirements (e.g., "Write unit tests for auth")
- Code review checkpoints (e.g., "Review security implementation")
- Deployment stages (e.g., "Deploy to staging")
- Documentation updates (e.g., "Update API docs")

**Data Model:**

```rust
pub struct Plan {
    pub id: Uuid,
    pub project_id: Uuid,
    pub milestones: Vec<Milestone>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct Milestone {
    pub id: Uuid,
    pub name: String,
    pub priority: Priority,        // High | Medium | Low
    pub tasks: Vec<Task>,
    pub depends_on: Vec<Uuid>,     // Other milestone IDs
    pub status: MilestoneStatus,   // NotStarted | InProgress | Complete
    pub completion_percentage: f32,
}

pub struct Task {
    pub id: Uuid,
    pub title: String,
    pub status: TaskStatus,        // Pending | InProgress | Complete | Blocked
    pub depends_on: Vec<Uuid>,     // Other task IDs
    pub sub_tasks: Vec<SubTask>,
    pub requires_user_action: bool,
    pub user_action_prompt: Option<String>,
    pub blocker_reason: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub struct SubTask {
    pub id: Uuid,
    pub title: String,
    pub status: TaskStatus,
}
```

**UI Display:**

```
┌─────────────────────────────────────────────┐
│ 🎯 MVP Milestone (Priority: High)           │
│ Status: 2/5 tasks complete (40%)            │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ 🔄 Implement User Authentication    🔄  │ │
│ │ In Progress: Started Nov 28, 11:00 AM   │ │
│ │ Depends on: Database layer              │ │
│ │ Sub-tasks:                              │ │
│ │   ✅ JWT token generation               │ │
│ │   🔄 Password hashing (in progress)     │ │
│ │   ⏳ Login endpoint (pending)           │ │
│ │   ⏳ Registration endpoint (pending)    │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ⏳ Add File Upload Functionality    ⏳  │ │
│ │ Depends on: Auth, S3 bucket setup       │ │
│ │ [👤 User Action Required - Click]      │ │
│ │ Action: Confirm S3 bucket configuration │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

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

**Implementation Files:**

- `src-tauri/src/documentation_panels/features_extractor.rs` (NEW, ~400 lines)
- `src-tauri/src/documentation_panels/decisions_logger.rs` (NEW, ~300 lines)
- `src-tauri/src/documentation_panels/plan_manager.rs` (NEW, ~500 lines)
- `src-tauri/src/documentation_panels/changes_tracker.rs` (EXISTS, enhance)
- `src-ui/components/DocumentationPanels.tsx` (EXISTS, enhance with multi-user sync)

---

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

Weeks 33-36: Performance & scale

Weeks 37-40:Advanced refactoring

Weeks 41-44:Ecosystem & marketplace

Weeks 45-48:Enterprise platform

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
