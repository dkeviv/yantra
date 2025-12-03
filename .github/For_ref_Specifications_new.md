# Yantra: Technical Specifications

**Version:** 2.0
**Date:** December 2, 2025
**Organization:** By Preventive Development Lifecycle
**Status:** Ferrari MVP - Production-Ready Architecture
**Target Size:** ~6,500-8,500 lines

---

## Table of Contents

1. [Executive Summary &amp; Vision](#1-executive-summary--vision)
2. [The Yantra Solution: Four Pillars of Autonomous Development](#2-the-yantra-solution-four-pillars-of-autonomous-development)
3. [Unified Tool Interface (UTI) Architecture](#3-unified-tool-interface-uti-architecture)
4. [Core Technical Systems](#4-core-technical-systems) _(includes Security Framework)_
5. [Preventive Development Lifecycle](#5-preventive-development-lifecycle)
6. [Master Feature Tables](#6-master-feature-tables) _(MVP & Post-MVP, includes Operating Modes)_
7. [Feature Details by Lifecycle Phase](#7-feature-details-by-lifecycle-phase) _(with implementation details & tool-protocol mapping)_
8. [User Experience (UX)](#8-user-experience-ux)
9. [Technology Stack](#9-technology-stack)

---

# 1. Executive Summary & Vision

## 1.1 The Vision: Code That Never Breaks

Yantra is a **fully autonomous agentic developer** - an AI-powered platform that executes the complete software development lifecycle with a revolutionary guarantee: **code that never breaks**.

**Traditional AI Code Assistants:**
LLM generates code → Developer manually tests → Developer fixes issues → Developer commits

**Yantra (Fully Autonomous):**
LLM generates code → System validates (6 layers) → System secures (5 layers) → System tests → System fixes → System deploys → System monitors → Repeat until perfect

**Human Role:** Provide intent ("Add Stripe payment integration"), approve at lifecycle gates (Plan sign-off, Deploy sign-off), provide oversight.

### What Makes Yantra Different

Unlike traditional IDEs that **assist** developers or AI tools that **suggest** code, Yantra makes artificial intelligence the **primary developer**, with humans providing:

1. **Intent** - What needs to be built ("Add payment processing")
2. **Oversight** - Approve critical decisions at lifecycle gates
3. **Domain Knowledge** - Guide edge cases and business rules

**The AI handles everything else autonomously:**

- Architecture alignment and boundary enforcement
- Dependency analysis and conflict prevention
- Code generation with full context awareness
- Multi-layer validation (syntax → types → patterns → logic → integration → impact)
- Security scanning with 5-layer prevention stack
- Automated testing with affected test detection
- Zero merge conflicts through file locking system
- Deployment with health checks
- Production monitoring and self-healing (Post-MVP)

## 1.2 Core Philosophy: Prevention Over Reaction

Yantra fundamentally shifts software development from **reactive problem-solving** to **proactive problem prevention**. Instead of finding and fixing issues after they occur, Yantra ensures issues cannot occur in the first place.

**Reactive Approach (Traditional):**

```
Write code → Find bugs → Fix bugs → Hope it works → Deploy → Production issues
```

**Preventive Approach (Yantra):**

```
Prevent by Design → Make Problems Impossible, Not Unlikely
```

**Core Principles:**

1. **Problems are preventable, not inevitable**

## 1.3 The Problem We Solve

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

**END OF SECTION 1**

---

# 2. The Yantra Solution: Four Pillars of Autonomous Development

## 2.1 Overview: How Yantra Works

Yantra is built on four foundational pillars that work together to deliver code that never breaks:

1. **Accelerated Development with Agentic Autonomous Capabilities** - AI handles the complete development lifecycle
2. **Preventive Development Cycle** - Prevention by design, not detection by inspection
3. **Fast Deployment with Auto-Deploy** - From commit to production in minutes, not hours
4. **Self-Healing Production Systems** - Detect, diagnose, and fix issues automatically (Post-MVP)

These four pillars create a continuous improvement loop: faster development → preventive quality → rapid deployment → self-healing → learning → even faster development.

---

## 2.2 Pillar 1: Accelerated Development with Agentic Autonomous Capabilities

### The Agentic Philosophy

Traditional AI coding assistants are **reactive tools** - they wait for developers to ask questions and provide suggestions. Yantra is an **autonomous agent** - it proactively executes the entire development workflow from understanding requirements to deploying code.

**What "Agentic" Means:**

```
Traditional AI Assistant (Reactive):
  Developer: "How do I add authentication?"
  AI: "Here's sample code for authentication..."
  Developer: Copies code, pastes, tests, debugs, fixes, commits

Yantra Agent (Autonomous):
  Developer: "Add authentication"
  Agent: Understands → Plans → Generates → Validates → Tests → Secures → Commits
  Developer: Reviews at approval gates only
```

### The Four Agentic Capabilities

**1. 🔍 PERCEIVE (Input & Sensing)**

The agent senses and understands the development environment:

- **Code Intelligence:** Parse and understand existing codebase via tree-sitter (Python, JavaScript, TypeScript, Rust, Go, Java, C/C++, etc.)
- **Dependency Awareness:** Track every dependency at version-level granularity (numpy==1.24.0 vs numpy==1.26.0)
- **Architecture Understanding:** Load and enforce module boundaries, layering rules, communication patterns
- **Documentation Reading:** Extract requirements from DOCX, PDF, Notion, Linear, Google Docs
- **External API Discovery:** Detect and track external API dependencies (REST, GraphQL, databases)

**Example:**

```
User: "Add Stripe payment integration"
**END OF SECTION 3**

Reference: The Preventive Development Cycle tables above are reproduced from the source file `docs/Research & specifications/*Preventive Development Cycle.md`. For detailed scenarios and flows, see that file.

---

# 4. Unified Tool Interface (UTI) Architecture — Consolidated Capability Table

Description: The UTI organizes all capabilities the agent requires. Below is a single consolidated table that replaces the prior multi-table representation. Columns: Purpose, Category, Capabilities (representative), Primary Protocol, Secondary Protocol, Example Tool(s), PDC Phase.

Notes on grouping: Rows are grouped by Primary/Secondary protocol combinations. When within a category some capabilities use different protocol combos, the category is split into multiple rows to reflect that.

Source: Consolidated from `docs/Research & specifications/*Yantra Unified Tool Interface.md` (UTI specification) and mapped to Preventive Development Cycle phases in `docs/Research & specifications/*Preventive Development Cycle.md`.

| Purpose | Category | Capabilities (representative) | Primary Protocol | Secondary Protocol | Example Tool(s) | PDC Phase |
| ------- | -------- | ----------------------------- | ---------------- | ------------------ | --------------- | --------- |
| CG | File System (core ops) | file.read, file.write, file.edit, directory.list, file.search | Builtin | MCP | Local FS, @modelcontextprotocol/server-filesystem | 3 Execute |
| CG | Code Intelligence (symbols, defs) | code.symbols, code.definition, code.references, code.completion | MCP | Builtin/LSP | Pylance (MCP), Tree-sitter fallback | 1 Architect / 3 Execute |
| CG | AST & Parsing | ast.parse, ast.query, ast.edit | Builtin | — | Tree-sitter | 1 Architect / 3 Execute |
| CG/TS | Dependency Graph / GNN | depgraph.build, depgraph.query, depgraph.impact | Builtin | — | petgraph + GNN runtime | 1 Architect / 2 Plan / 3 Execute |
| CG/DP | Terminal / Shell | shell.exec, shell.execStreaming, shell.env | Builtin | — | Local shell | 3 Execute / 4 Deploy |
| CG | Version Control (Git ops) | git.status, git.diff, git.commit, git.push | MCP | Builtin | @modelcontextprotocol/server-git, git2-rs fallback | 3 Execute / 4 Deploy |
|   |   |   |   |   |   |   |
**Practical UX (Git):** Git integration follows a VSCode-like flow: a "Connect / Sign in" control opens OAuth (GitHub, GitLab) or SSH key options; after authentication the UI lists repositories and branches and the user clicks "Clone" to instantiate the workspace. UTI surfaces OAuth and SSH methods and falls back to local git operations when MCP servers are unavailable.
| CG | GitHub / Code Hosting | github.prs, github.repos, github.actions | MCP | — | GitHub MCP | 3 Execute / 4 Deploy |
| CG | Database Access | db.connect, db.query, db.migrate | MCP | — | Postgres/MySQL MCP servers | 3 Execute / 5 Monitor |
| TS | Testing (execution) | test.discover, test.run, test.runAffected, test.coverage | Builtin | — | pytest, jest, cargo test (invoked by Yantra) | 3 Execute |
| TS | Test Data & Seeding | db.seed, test.generate | MCP + Builtin | — | DB MCP + built-in test generator | 2 Plan / 3 Execute |
| TS | E2E / Browser Automation | e2e.run, browser.launch, browser.navigate, browser.screenshot | Builtin | — | CDP (chromiumoxide) | 3 Execute / 5 Monitor |
| MM | Monitoring & Observability | logs.tail, metrics.query, health.check | MCP | Builtin | Sentry, Prometheus, Datadog | 5 Monitor / 4 Deploy |
| MM | Security (scanning) | security.scan, security.secrets, security.audit | Builtin | MCP | Semgrep (builtin), Snyk (MCP) | 3 Execute / 4 Deploy |
| DP | Build & Container | build.run, container.build, container.push | Builtin | MCP | Docker CLI (builtin), registry MCP | 4 Deploy |
| DP | Cloud Deploy & Infra | deploy.preview, deploy.production, infra.provision | MCP | Builtin | Railway MCP, AWS MCP | 4 Deploy |
| MM | Debugging | debug.launch, debug.breakpoint, debug.evaluate | DAP | — | debugpy, node-debug2, codelldb | 3 Execute |
| CG/MM | Documentation & Architecture Viz | docs.generate, arch.diagram, arch.validate | Builtin | MCP | Mermaid, Graphviz (builtin), Notion (MCP) | 1 Architect / 3 Execute |
| CG/MM | Package Management | pkg.install, pkg.audit, pkg.lockSync | Builtin | MCP | npm/pip/cargo via shell + CVE MCP | 2 Plan / 3 Execute |
| CG | Code Formatting & Lint | format.run, lint.run, lint.fix | Builtin | MCP | Prettier, ESLint, Clippy (builtin) | 3 Execute |
| MM | Context & Memory | context.add, context.search, embeddings.search | Builtin | MCP | Local embeddings, vector DB (builtin) | 2 Plan / 3 Execute |
| MM | Visualization (inline) | viz.depgraph, viz.chart, viz.diff, viz.table | Builtin | Shell | Mermaid, Plotly, Graphviz | 1 Architect / 3 Execute / 5 Monitor |
| MM | Collaboration & Notifications | slack.send, email.send, notion.update, jira.issues | MCP | — | Slack MCP, Notion MCP, Jira MCP | 2 Plan / 3 Execute |

---

Notes:

- PDC Phase mapping: I mapped capabilities to the Preventive Development Cycle phases where they predominantly apply. Several capabilities appear across phases (e.g., depgraph used in Architect & Execute), in which case I listed the most relevant phases.
- Primary/secondary protocols follow the UTI specification decision rules (performance/security/differentiator → Builtin; well-supported ecosystem services → MCP; debugging → DAP; editor concerns → LSP only).
- Example Tool(s) column shows representative implementations or MCP servers where applicable. For MCP servers, see `docs/Research & specifications/*Yantra Unified Tool Interface.md` for full MCP server names.

Reference files:

- UTI source: `docs/Research & specifications/*Yantra Unified Tool Interface.md`
- Preventive Development Cycle (PDC): `docs/Research & specifications/*Preventive Development Cycle.md`

  ├─ Load architecture view (API → Service → Data layers)
  ├─ Identify payment-related modules (none found)
  └─ Read Linear ticket description + acceptance criteria

[00:05] 🧠 REASON Phase
  ├─ Decompose into 8 sub-tasks with dependencies
  ├─ Check Stripe library compatibility (✓ Python 3.11)
  ├─ Run CVE check on stripe library (✓ No critical vulnerabilities)
  ├─ Estimate time: 5 minutes (automated)
  └─ **APPROVAL GATE: Plan Sign-Off** → Human reviews plan → Approves

[00:10] ⚡ ACT Phase (Execution)
  ├─ [Task 1] Install stripe library → pip install stripe==8.0.0
  ├─ [Task 2] Generate StripeService class (Yantra Codex)
  │   ├─ Yantra generates initial code (15ms)
  │   ├─ LLM reviews and adds error handling (2s)
  │   └─ Merged result: 150 lines, production-ready
  ├─ [Task 3] Add API endpoints /api/v1/payments
  │   ├─ POST /payments (create payment intent)
  │   ├─ POST /webhooks/stripe (handle events)
  │   └─ GET /payments/:id (payment status)
  ├─ [Task 4] Generate tests
  │   ├─ 12 unit tests (service layer)
  │   ├─ 3 integration tests (mocked Stripe API)
  │   └─ 1 E2E test (full payment flow)
  ├─ [Task 5] Run validation stack (6 layers)
  │   ├─ Layer 1: Syntax check (5ms) ✓
  │   ├─ Layer 2: Type check (200ms) ✓
  │   ├─ Layer 3: Linter (100ms) ✓
  │   ├─ Layer 4: Unit tests (8s) ✓ 15/15 passed
  │   ├─ Layer 5: Integration tests (12s) ✓ 3/3 passed
  │   └─ Layer 6: GNN affected tests (1s) ✓ 47/47 passed
  ├─ [Task 6] Run security stack (5 layers)
  │   ├─ Layer 1: CVE check ✓
  │   ├─ Layer 2: Pattern analysis ✓
  │   ├─ Layer 3: Secrets detection ✓
  │   ├─ Layer 4: Security tests ✓
  │   └─ Layer 5: SAST (Semgrep) ✓ No critical issues
  ├─ [Task 7] Commit to Git
  │   ├─ Branch: feature/stripe-payment-integration
  │   ├─ Commit: "feat: Add Stripe payment integration with webhooks"
  │   └─ Push to origin
  └─ [Task 8] Update documentation
      ├─ Notion: API documentation updated
      ├─ Linear: Ticket moved to "Ready for Review"
      └─ Slack: Notification sent to #engineering

[01:50] 🔄 LEARN Phase
  ├─ Yantra learns: "Stripe integration pattern"
  ├─ Store: API → Service separation pattern
  ├─ Store: Webhook handling with signature verification
  └─ Confidence for next Stripe task: 0.75 → 0.85 (+10%)

[02:00] ✅ COMPLETE
  └─ **APPROVAL GATE: Deploy Sign-Off** → Human reviews changes → Approves → Auto-deploys
```

**Result:** Complete feature implementation in 2 minutes (vs 4-8 hours manual), zero bugs, 95%+ test coverage, production-ready code.

### The Role of UTI (Unified Tool Interface) Architecture

To orchestrate these comprehensive agentic capabilities, Yantra employs a **Unified Tool Interface (UTI) architecture** that provides the agent with access to all necessary tools and protocols. The UTI serves as the central nervous system connecting the agent's reasoning (what to do) with execution (how to do it).

**Why UTI is Essential for Agentic Capabilities:**

Without a unified interface, the agent would need to:

- Learn different APIs for every tool (Git, databases, deployment platforms, etc.)
- Handle protocol differences (REST, GraphQL, WebSocket, CLI, etc.)
- Manage authentication for each service separately
- Deal with rate limits, retries, and failures per tool

**With UTI, the agent gets:**

- **Single consistent interface** across 21 capability categories (file system, Git, databases, testing, deployment, etc.)
- **Protocol abstraction** - Agent doesn't care if it's MCP, LSP, DAP, or Builtin
- **Smart routing** - UTI selects the right protocol/tool for each operation automatically
- **Unified error handling** - Consistent retry logic, fallbacks, circuit breakers
- **Centralized authentication** - One auth system for all external services

**Example: The Agent's Perspective**

```
WITHOUT UTI (Agent's nightmare):
  Agent needs to deploy to Railway:
  ├─ Learn Railway API (REST endpoints, authentication)
  ├─ Handle Railway-specific errors and rate limits
  ├─ Track deployment status with polling
  └─ Parse Railway-specific response formats

  Agent needs to update Linear ticket:
  ├─ Learn Linear API (GraphQL, different from Railway)
  ├─ Handle Linear-specific authentication (API key vs OAuth)
  ├─ Deal with Linear rate limits (different from Railway)
  └─ Parse Linear-specific response formats

WITH UTI (Agent's dream):
  Agent needs to deploy:
  └─ UTI.deploy(platform="railway", env="staging")
     → UTI handles protocol selection, auth, execution, error handling

  Agent needs to update ticket:
  └─ UTI.update_ticket(system="linear", ticket_id="ENG-123", status="deployed")
     → UTI handles protocol selection, auth, execution, error handling
```

**UTI enables true autonomy** by abstracting away tool-specific complexity, allowing the agent to focus on high-level reasoning (what to do) rather than low-level integration details (how to connect).

For detailed UTI architecture, protocol selection framework, and the 21 capability categories, see [Section 3: Unified Tool Interface Architecture](#3-unified-tool-interface-uti-architecture).

---

## 2.3 Pillar 2: Preventive Development Cycle - Production Work Out of the Gate

### Philosophy: Prevention Over Reaction

Traditional software development is **reactive** - write code, find bugs, fix bugs, repeat. This creates a cycle of technical debt, brittle systems, and unpredictable timelines.

Yantra's **Preventive Development Cycle** shifts from "find and fix" to "prevent by design". Each phase prevents problems in subsequent phases, creating a cascading prevention effect.

**Core Principle:** Make problems **impossible**, not unlikely.

### The Five Preventive Phases

**Visual Flow:**

```
┌────────────────────────────────────────────────────────────┐
│         PREVENTIVE DEVELOPMENT LIFECYCLE                   │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: ARCHITECT/DESIGN                                 │
│  ├─ Architecture Alignment → Prevents boundary violations  │
│  ├─ Tech Stack Validation → Prevents dependency conflicts  │
│  ├─ Feature Extraction → Prevents unclear requirements     │
│  └─ ADR Generation → Prevents undocumented decisions       │
│           ↓                                                 │
│  Phase 2: PLAN                                             │
│  ├─ Task Decomposition → Prevents unclear scope            │
│  ├─ Dependency Analysis → Prevents incorrect order         │
│  ├─ Conflict Detection → Prevents merge conflicts          │
│  └─ Plan Sign-Off (Human Gate) → Prevents wrong direction  │
│           ↓                                                 │
│  Phase 3: EXECUTE/DEVELOP                                  │
│  ├─ 6-Layer Validation → Prevents bugs                     │
│  ├─ 5-Layer Security → Prevents vulnerabilities            │
│  ├─ File Locking → Prevents merge conflicts                │
│  └─ Browser Testing → Prevents UI breaks                   │
│           ↓                                                 │
│  Phase 4: DEPLOY                                           │
│  ├─ Pre-Deploy Validation → Prevents failed deployments    │
│  ├─ Health Checks → Prevents downtime                      │
│  ├─ Deploy Sign-Off (Human Gate) → Prevents bad deploys    │
│  └─ Rollback Ready → Prevents permanent damage             │
│           ↓                                                 │
│  Phase 5: MONITOR/MAINTAIN (Post-MVP)                      │
│  ├─ Error Detection → Prevents user-discovered bugs        │
│  ├─ Auto-Rollback → Mitigates productionfailures            │
│  ├─ Root Cause Analysis → Prevents repeat incidents        │
│  └─ Self-Healing → Prevents manual intervention            │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### What "Production Work Out of the Gate" Means

**Traditional MVP Development:**

```
Build fast → Ship broken code → Find bugs in production → Hotfix cycle → Accumulate tech debt
```

**Yantra's Ferrari MVP:**

```
Build correctly from day one → Ship production-ready code → Zero production bugs → No tech debt
```

**Key Differences:**

| Aspect           | Traditional MVP                | Yantra Ferrari MVP                   |
| ---------------- | ------------------------------ | ------------------------------------ |
| **Architecture** | "We'll refactor later"         | Architecture enforced from day one   |
| **Dependencies** | Install whatever works         | Version-level validation, CVE checks |
| **Testing**      | Manual testing, low coverage   | Auto-generated tests, 90%+ coverage  |
| **Security**     | Fix vulnerabilities when found | 5-layer prevention, auto-blocked     |
| **Deployment**   | Manual, error-prone            | Automated with health checks         |
| **Quality**      | Reactive (fix bugs)            | Preventive (no bugs)                 |
| **Tech Debt**    | Accumulates rapidly            | Zero debt (prevention)               |

**Result:** Yantra's MVP is production-ready from the first commit. No "good enough for now" code, no "we'll fix it later" shortcuts. Every line of code meets production standards.

### Prevention Examples

**Example 1: Preventing Architecture Violations**

```
Traditional Approach:
  Developer adds: UserService.py directly calls PaymentService.py
  Code Review (days later): "This violates our layered architecture"
  Fix: Refactor (hours of work)

Yantra Preventive Approach:
  Agent loads architecture: API → Service → Data (no cross-layer direct calls)
  Agent attempts: UserService → PaymentService
  GNN detects: "Violation: Services cannot call services directly, use API layer"
  Agent auto-fixes: UserService → APIClient → PaymentService
  Result: Architectural correctness enforced, zero manual review needed
```

**Example 2: Preventing Merge Conflicts**

```
Traditional Approach:
  Developer A: Editing user_service.py
  Developer B: Editing user_service.py (doesn't know A is working on it)
  Git merge: CONFLICT in user_service.py
  Resolution: Manual merge (30 minutes, error-prone)

Yantra Preventive Approach:
  Agent A: Requests lock on user_service.py → Granted
  Agent B: Requests lock on user_service.py → DENIED
  Agent B: Notified "File locked by Agent A, work on different file"
  Result: Conflict impossible by design
```

**Example 3: Preventing Security Vulnerabilities**

```
Traditional Approach:
  Developer writes: query = f"SELECT * FROM users WHERE id = {user_id}"
  Code merged and deployed
  Security scan (weeks later): "SQL injection vulnerability found"
  Fix: Emergency hotfix

Yantra Preventive Approach:
  Agent generates: query = f"SELECT * FROM users WHERE id = {user_id}"
  Security Layer 2 (Pattern Analysis): "BLOCKED: SQL injection risk detected"
  Agent auto-fixes: query = "SELECT * FROM users WHERE id = %s"; execute(query, (user_id,))
  Security Layer 2: "APPROVED: Parameterized query"
  Result: Vulnerability prevented before commit
```

### The Prevention Effect

Each preventive phase creates a stronger foundation for the next phase:

**Phase 1 prevents issues in Phase 2:**

- Architecture aligned → Task decomposition knows boundaries
- Tech stack validated → No surprise dependency conflicts during planning

**Phase 2 prevents issues in Phase 3:**

- Tasks properly ordered → Development follows correct sequence
- Dependencies mapped → No missing imports during code generation

**Phase 3 prevents issues in Phase 4:**

- 6-layer validation passed → Deployment won't fail due to code issues
- Security layers passed → No vulnerabilities in production

**Phase 4 prevents issues in Phase 5:**

- Health checks passed → Application is actually running correctly
- Rollback ready → Quick recovery if issues discovered

**Phase 5 prevents issues in future cycles:**

- Root cause analysis → Same issue won't happen again
- Learning captured → Faster development next time

---

## 2.4 Pillar 3: Fast Deployment with Auto-Deploy Option

### The Deployment Problem

**Traditional Deployment Flow (Manual):**

```
Develop locally (hours) →
Run tests manually (30 min) →
Fix test failures (1 hour) →
Create PR (15 min) →
Wait for code review (hours to days) →
Merge to main (5 min) →
Wait for CI/CD pipeline (20 min) →
Manual approval (variable) →
Deploy to staging (10 min) →
Manual testing on staging (1 hour) →
Deploy to production (10 min) →
Monitor for issues (continuous)

Total Time: 1-3 days for simple feature
```

**Yantra Auto-Deploy Flow:**

```
Develop with Yantra (2 min, autonomous) →
Tests auto-run during development (included in 2 min) →
Auto-commit to feature branch (instant) →
Plan Sign-Off (human approval, 30 sec) →
Auto-deploy to Railway staging (2 min) →
Health checks (30 sec) →
Deploy Sign-Off (human approval, 30 sec) →
Auto-deploy to production (2 min) →
Auto-monitoring enabled (continuous)

Total Time: ~8 minutes for simple feature (20-40x faster)
```

### How Auto-Deploy Works

**Pre-Deployment Validation (Automatic):**

Before any deployment, Yantra runs a comprehensive validation checklist:

1. **Code Validation (6 layers - Already passed during development)**
   - ✓ Syntax correct (tree-sitter)
   - ✓ Types correct (LSP)
   - ✓ Patterns correct (linters)
   - ✓ Logic correct (unit tests)
   - ✓ Integration correct (integration tests)
   - ✓ No breaking changes (GNN affected tests)

2. **Security Validation (5 layers - Already passed during development)**
   - ✓ No critical CVEs in dependencies
   - ✓ No SQL injection, XSS, command injection patterns
   - ✓ No hardcoded secrets
   - ✓ Security tests pass
   - ✓ SAST clean (Semgrep)

3. **Deployment Readiness Checks (New validation)**
   - ✓ Health check endpoint exists (`/health` or `/api/health`)
   - ✓ All required environment variables defined
   - ✓ Database migrations ready (if applicable)
   - ✓ External API credentials valid
   - ✓ Previous deployment exists (for rollback)

**Deployment Flow (Railway MVP):**

```
Step 1: Pre-Deploy Validation
  ├─ Run validation checklist
  ├─ Generate deployment manifest
  └─ Calculate rollback plan

Step 2: Deploy Sign-Off (Human Approval Gate)
  ├─ Show: Changed files, test results, security scan
  ├─ Show: Deployment plan (staging → production)
  ├─ Human reviews and approves
  └─ Or: Reject and provide feedback

Step 3: Deploy to Staging (Auto via Railway MCP)
  ├─ Push to Railway staging environment
  ├─ Railway builds Docker image
  ├─ Railway deploys container
  └─ Railway provides deployment URL

Step 4: Health Checks (Auto)
  ├─ HTTP GET /health → Expect 200 OK
  ├─ Check response time < 2s
  ├─ Check error rate = 0%
  ├─ Run smoke tests (basic functionality)
  └─ If any check fails → Auto-rollback

Step 5: Production Deployment (Auto if staging passes)
  ├─ If health checks pass: Auto-deploy to production
  ├─ If health checks fail: Alert human, keep previous version
  ├─ Zero-downtime deployment (Railway handles)
  └─ Post-deploy health checks

Step 6: Post-Deploy Monitoring
  ├─ Track error rates (via Sentry - Post-MVP)
  ├─ Track response times
  ├─ Track deployment success/failure
  └─ Alert if issues detected
```

### Deployment Options

**Option 1: Guided Mode (Default)**

- Human approves every deployment
- Review changes before staging deployment
- Review staging results before production deployment
- Maximum control, slower

**Option 2: Auto Mode (Experienced users)**

- Auto-deploy to staging (no approval)
- Human approval required before production only
- Faster cycle, still safe (Deploy Sign-Off gate)

**Option 3: Full Auto Mode (CI/CD integration)**

- Auto-deploy to staging
- Auto-deploy to production if all checks pass
- Human notified after deployment
- Fastest cycle, requires high confidence

### Multi-Platform Support (Post-MVP)

**MVP: Railway Only**

- Easiest setup (one MCP server)
- Perfect for small projects and startups
- Zero-config deployment

**Post-MVP: Multi-Cloud**

- **AWS:** ECS, Lambda, EC2 (via aws-mcp)
- **GCP:** Cloud Run, Cloud Functions, GKE (via gcp-mcp)
- **Azure:** App Service, Functions, AKS (via azure-mcp)
- **Kubernetes:** Direct kubectl/Helm integration
- **Vercel:** Frontend deployment (Next.js, React)
- **Netlify:** Static site deployment

### Deployment Performance Targets

**MVP (Railway):**

- Staging deployment: <2 minutes
- Health checks: <30 seconds
- Production deployment: <2 minutes
- Total cycle (commit → production): <8 minutes

**Post-MVP (Multi-Cloud):**

- AWS Lambda: <30 seconds (serverless)
- AWS ECS: <3 minutes (containerized)
- GCP Cloud Run: <1 minute (serverless containers)
- Kubernetes: <5 minutes (complex orchestration)

---

## 2.5 Pillar 4: Self-Healing Production Systems (Post-MVP)

### The Vision: Zero-Touch Production Maintenance

Traditional production systems require **constant human intervention**:

- Monitoring alerts → Human investigates → Human diagnoses → Human fixes → Human deploys
- Average incident resolution time: 2-4 hours
- Average cost per incident: $5,000-$10,000

Yantra's self-healing systems **automatically detect, diagnose, and fix production issues** without human intervention:

- Monitoring detects → Agent investigates → Agent diagnoses → Agent fixes → Agent deploys → Human notified
- Average incident resolution time: 5-10 minutes
- Average cost per incident: $0 (automated)

**Note:** This is a Post-MVP feature (Month 9-12), but the foundation is built into the MVP architecture.

### The Self-Healing Cycle

```
┌──────────────────────────────────────────────────────────┐
│              SELF-HEALING PRODUCTION CYCLE                │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. DETECT (Continuous Monitoring)                       │
│     ├─ Error rate spike detected (Sentry)               │
│     ├─ Response time degradation (Railway metrics)       │
│     ├─ External API failure (HTTP monitors)              │
│     └─ Database connection pool exhaustion               │
│                      ↓                                    │
│  2. CLASSIFY (Severity Assessment)                       │
│     ├─ Critical: System down, data loss risk            │
│     ├─ High: Degraded performance, user impact          │
│     ├─ Medium: Isolated failures, workaround exists     │
│     └─ Low: Minor issues, no user impact                │
│                      ↓                                    │
│  3. DIAGNOSE (Root Cause Analysis)                       │
│     ├─ Analyze error logs and stack traces              │
│     ├─ Check recent deployments (correlation)           │
│     ├─ Query GNN for dependency changes                 │
│     ├─ Review external API status                       │
│     └─ Generate hypothesis (LLM-powered)                │
│                      ↓                                    │
│  4. DECIDE (Rollback vs Fix vs Mitigate)                │
│     ├─ If recent deployment: Rollback (instant)         │
│     ├─ If external API down: Circuit breaker (instant)  │
│     ├─ If code bug: Generate fix (5 min)                │
│     └─ If unknown: Alert human + safe mode              │
│                      ↓                                    │
│  5. HEAL (Automated Remediation)                         │
│     ├─ Execute rollback to last known good version      │
│     ├─ Or: Generate fix → Validate → Test → Deploy      │
│     ├─ Or: Enable circuit breaker/fallback              │
│     └─ Verify healing successful (health checks)        │
│                      ↓                                    │
│  6. LEARN (Prevent Recurrence)                           │
│     ├─ Store incident pattern in Known Issues DB        │
│     ├─ Update validation rules to catch earlier         │
│     ├─ Generate regression test                         │
│     └─ Share pattern with Yantra Cloud (opt-in)         │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Self-Healing Examples

**Example 1: Auto-Rollback on Deployment Failure**

```
Scenario: New deployment introduces database query that times out

[00:00] Deploy v1.2.3 to production
[00:30] Health checks pass (basic endpoint responds)
[02:00] Error rate spike detected: 15% of requests timing out
[02:05] Agent investigates:
        ├─ Recent change: v1.2.2 → v1.2.3 (2 min ago)
        ├─ Error: "DatabaseTimeout in get_user_orders()"
        ├─ Root cause: New query missing index
        └─ Severity: HIGH (user-facing errors)
[02:10] Agent decides: ROLLBACK (fastest resolution)
[02:15] Auto-rollback to v1.2.2
[02:45] Health checks pass, error rate: 0%
[03:00] Agent generates fix:
        ├─ Add database index on orders.user_id
        ├─ Generate migration
        ├─ Test migration on staging
        └─ Create PR with fix for human review
[03:30] Human notified: "Rolled back v1.2.3 due to DB timeout. Fix PR created."

Result: 3 min downtime (rollback), vs 2-4 hours manual resolution
```

**Example 2: Circuit Breaker on External API Failure**

```
Scenario: Payment provider API goes down

[00:00] Stripe API returns 503 Service Unavailable
[00:00] 50 consecutive payment failures detected
[00:05] Agent investigates:
        ├─ Error: "StripeAPIError: Service Unavailable"
        ├─ External dependency: Stripe API
        ├─ Root cause: Stripe outage (not our code)
        └─ Severity: CRITICAL (payments broken)
[00:10] Agent decides: CIRCUIT BREAKER (temporary mitigation)
[00:15] Enable circuit breaker:
        ├─ Queue payment requests instead of failing
        ├─ Return: "Payment queued, will process when service recovers"
        ├─ Monitor Stripe API every 30 seconds
        └─ User experience: Delayed payment, not failed payment
[05:00] Stripe API recovers (200 OK detected)
[05:05] Process queued payments (147 pending)
[05:30] All queued payments processed successfully
[05:35] Disable circuit breaker, resume normal operation
[06:00] Human notified: "Stripe outage detected and handled automatically. 147 payments queued and processed."

Result: Zero failed payments, vs 147 lost transactions (manual approach)
```

**Example 3: Auto-Fix for Code Bug**

```
Scenario: Edge case causes null pointer exception

[00:00] Error spike detected: NullPointerError in user_profile()
[00:05] Agent investigates:
        ├─ Error: "NoneType has no attribute 'email'"
        ├─ Affected: user_profile() function
        ├─ Frequency: 2% of requests (edge case)
        ├─ Root cause: Missing null check for users without email
        └─ Severity: MEDIUM (some users affected, workaround exists)
[00:10] Agent decides: GENERATE FIX (not critical enough for rollback)
[00:15] Agent generates fix:
        ├─ Add null check: if user.email is not None
        ├─ Add default: return "No email provided"
        ├─ Generate test case for null email scenario
        └─ Run validation (6 layers + 5 security layers)
[00:45] All validations pass
[01:00] Deploy fix to staging
[01:30] Staging health checks pass
[01:35] **Rollback Decision Gate**: Agent asks human
        "Auto-generated fix for NullPointerError. Deploy to production?"
[01:40] Human reviews and approves
[01:45] Deploy fix to production
[02:15] Error rate: 0%, issue resolved
[02:30] Regression test added to prevent recurrence

Result: 2.5 min resolution with human approval, fully tested fix
```

### Self-Healing Capabilities

**What Self-Healing Can Handle (Post-MVP):**

✅ **Automatic Rollback**

- Recent deployment causes errors → Rollback to previous version
- Fast resolution (2-3 minutes)
- Safe (returns to known-good state)

✅ **Circuit Breaker Activation**

- External API down → Queue requests, return graceful degradation
- Prevents cascade failures
- Auto-recovery when API returns

✅ **Auto-Fix Generation**

- Simple bugs (null checks, validation, error handling)
- Edge cases missed in testing
- Performance optimizations (add caching, indexes)

✅ **Configuration Changes**

- Increase connection pool size
- Adjust timeouts
- Enable/disable features flags

❌ **What Self-Healing Cannot Handle:**

- Complex business logic bugs (requires human reasoning)
- Data corruption issues (requires human judgment)
- Security incidents (requires human investigation)
- Architectural changes (requires human design)

**Human Oversight:**

Even in self-healing mode, humans stay in control:

- **Always notify** humans after automated actions
- **Rollback decision gate** for non-critical fixes
- **Human approval required** for code fixes (not rollbacks)
- **Safe mode** if agent is uncertain (alert human, wait for instructions)

---

## 2.6 The Four Pillars Working Together

The true power of Yantra emerges when all four pillars work in concert:

**Scenario: E-commerce platform needs payment integration**

```
PILLAR 1: Agentic Capabilities (Autonomous Execution)
  ├─ 🔍 PERCEIVE: Understand existing API structure, architecture boundaries
  ├─ 🧠 REASON: Decompose into tasks, analyze dependencies, check security
  ├─ ⚡ ACT: Generate code, run tests, commit, deploy
  └─ 🔄 LEARN: Store payment integration pattern for future

PILLAR 2: Preventive Development Cycle (Zero Bugs)
  ├─ Phase 1: Architecture aligned (payment service in correct layer)
  ├─ Phase 2: Tasks properly ordered (library → service → endpoints)
  ├─ Phase 3: 6-layer validation (all tests pass), 5-layer security (no vulnerabilities)
  ├─ Phase 4: Pre-deploy validation passed, health checks pass
  └─ Result: Production-ready code, zero bugs

PILLAR 3: Fast Deployment (8 Minutes)
  ├─ Auto-deploy to Railway staging (2 min)
  ├─ Health checks pass (30 sec)
  ├─ Human approval (30 sec)
  ├─ Auto-deploy to production (2 min)
  └─ Feature live in 8 minutes total

PILLAR 4: Self-Healing (Post-MVP)
  ├─ Monitor payment success rate
  ├─ Detect: Stripe API timeout spike
  ├─ Diagnose: External API issue (not our code)
  ├─ Heal: Enable circuit breaker, queue payments
  └─ Recovery: Auto-process queued payments when API returns

CONTINUOUS IMPROVEMENT LOOP:
  └─ Learning from this implementation makes next payment feature even faster
```

**Result:** Feature implemented autonomously in minutes, deployed safely, monitored continuously, self-heals if issues arise.

---

**END OF SECTION 2**

---

# 3. Preventive Development Cycle Explained

## Overview

The Preventive Development Cycle shifts software development from reactive problem-solving to proactive problem prevention. Below are the summary tables from each phase. For detailed scenarios, flows, and implementation guidance, see: `docs/Research & specifications/*Preventive Development Cycle.md`

---

## Phase 1: Architect / Design

### 1.1 Architecture Alignment

**What It Prevents:**

| Problem                            | How It's Prevented                                    |
| ---------------------------------- | ----------------------------------------------------- |
| Code violates module boundaries    | Agent knows boundaries, won't generate violating code |
| Circular dependencies introduced   | Dependency graph detects cycles before they exist     |
| Wrong patterns used                | Agent matches existing architectural patterns         |
| Scaling bottlenecks                | Architecture view flags single points of failure      |
| Security vulnerabilities by design | Security patterns enforced at architecture level      |

**Tools Required:**

| Tool                      | Type    | Purpose                                       |
| ------------------------- | ------- | --------------------------------------------- |
| Architecture View         | Builtin | Visual representation of system architecture  |
| Dependency Graph (GNN)    | Builtin | Module relationships, boundary enforcement    |
| Architecture Rules Engine | Builtin | Validates changes against defined constraints |
| Mermaid/Graphviz          | Builtin | Architecture diagram generation               |

---

### 1.2 Tech Stack Alignment

**What It Prevents:**

| Problem                       | How It's Prevented                                       |
| ----------------------------- | -------------------------------------------------------- |
| Incompatible library versions | Dependency graph checks compatibility before adding      |
| Duplicate functionality       | Agent detects existing libraries that serve same purpose |
| License conflicts             | License checker validates before dependency added        |
| Deprecated dependencies       | Version checker flags deprecated packages                |
| Framework version mismatches  | Stack definition enforces version constraints            |

**Tools Required:**

| Tool                   | Type    | Purpose                                     |
| ---------------------- | ------- | ------------------------------------------- |
| Dependency Graph       | Builtin | All dependencies and their relationships    |
| npm/pip/cargo registry | MCP     | Package metadata, versions, vulnerabilities |
| CVE Database           | MCP     | Known vulnerability lookup                  |
| License Checker        | Builtin | License compatibility validation            |
| Bundle Analyzer        | Builtin | Size impact analysis                        |

---

### 1.3 Existing Code Analysis

**What It Prevents:**

| Problem                            | How It's Prevented                              |
| ---------------------------------- | ----------------------------------------------- |
| Reinventing existing functionality | Agent searches codebase before writing new code |
| Inconsistent patterns              | Agent learns from existing code patterns        |
| Breaking existing consumers        | Impact analysis before any changes              |
| Missing context                    | Full codebase understanding via embeddings      |
| Technical debt accumulation        | Continuous refactoring recommendations          |
| Ported project issues              | Full analysis when importing from other IDEs    |

**Operating Modes:**

| Mode            | Behavior                                                                             |
| --------------- | ------------------------------------------------------------------------------------ |
| **Guided Mode** | Agent asks user before taking actions. Presents findings and waits for confirmation. |
| **Auto Mode**   | Agent performs analysis and actions automatically. Reports results after completion. |

**Tools Required:**

| Tool                   | Type    | Purpose                                           |
| ---------------------- | ------- | ------------------------------------------------- |
| Semantic Embeddings    | Builtin | Code similarity search                            |
| Tree-sitter            | Builtin | AST parsing for pattern extraction                |
| Dependency Graph (GNN) | Builtin | Usage tracking, impact analysis, module detection |
| Code Search            | Builtin | Full-text and regex search                        |
| Complexity Analyzer    | Builtin | Cyclomatic, cognitive complexity metrics          |
| Code Smell Detector    | Builtin | Anti-pattern identification                       |
| Dead Code Detector     | Builtin | Unused code identification                        |

---

### 1.4 Feature Extraction

**What It Prevents:**

| Problem                    | How It's Prevented                                    |
| -------------------------- | ----------------------------------------------------- |
| Ambiguous requirements     | Agent extracts specific acceptance criteria           |
| Missing edge cases         | Agent identifies edge cases from requirement analysis |
| Scope creep                | Clear feature boundaries defined upfront              |
| Misunderstood requirements | Agent clarifies before implementation starts          |

**Tools Required:**

| Tool           | Type    | Purpose                               |
| -------------- | ------- | ------------------------------------- |
| Google Docs    | MCP     | Read PRD documents                    |
| Notion         | MCP     | Read feature specs                    |
| Slack          | MCP     | Extract discussions                   |
| Figma          | MCP     | Design context and comments           |
| NLP Extraction | Builtin | User story and requirement extraction |

---

### 1.5 Architecture Sign-off

**What It Prevents:**

| Problem                           | How It's Prevented                   |
| --------------------------------- | ------------------------------------ |
| Unauthorized architecture changes | Explicit approval required           |
| Lost context on decisions         | Full audit trail maintained          |
| Inconsistent decision-making      | Approval criteria documented         |
| Blame games                       | Clear ownership and approval records |

**Tools Required:**

| Tool              | Type    | Purpose                           |
| ----------------- | ------- | --------------------------------- |
| Approval Queue    | Builtin | Approval workflow management      |
| Slack/Teams       | MCP     | Approver notifications            |
| ADR Generator     | Builtin | Structured decision documentation |
| Architecture View | Builtin | Visual update after approval      |

---

## Phase 2: Plan

### 2.1 Execution Plan by Features

**What It Prevents:**

| Problem             | How It's Prevented                               |
| ------------------- | ------------------------------------------------ |
| Missing tasks       | Agent decomposes feature into all required tasks |
| Wrong sequencing    | Dependency analysis ensures correct order        |
| Missed dependencies | All inter-task dependencies identified           |
| Unclear scope       | Each task has specific deliverables              |
| Unbounded work      | Time estimates based on complexity analysis      |

**Tools Required:**

| Tool                | Type    | Purpose                         |
| ------------------- | ------- | ------------------------------- |
| Dependency Graph    | Builtin | Task dependency analysis        |
| Work Tracker        | Builtin | Current work-in-progress status |
| Complexity Analyzer | Builtin | Time estimation                 |
| Jira/Linear         | MCP     | External task management sync   |

---

### 2.2 Progress Tracking

**What It Prevents:**

| Problem                  | How It's Prevented                        |
| ------------------------ | ----------------------------------------- |
| Unknown project status   | Real-time progress visibility             |
| Blocked work not noticed | Blockers surfaced immediately             |
| Estimation drift         | Continuous re-estimation based on actuals |
| Silent delays            | Automatic alerts when behind schedule     |

**Tools Required:**

| Tool                | Type    | Purpose                          |
| ------------------- | ------- | -------------------------------- |
| Activity Monitor    | Builtin | File changes, commits, test runs |
| Progress Calculator | Builtin | % complete based on deliverables |
| Velocity Tracker    | Builtin | Compare estimate vs actual       |
| Alert System        | Builtin | Notify on issues                 |
| Jira/Linear         | MCP     | Sync status to external trackers |
| Slack               | MCP     | Alert notifications              |

---

### 2.3 Plan Alignment & Sign-off

**What It Prevents:**

| Problem                 | How It's Prevented                        |
| ----------------------- | ----------------------------------------- |
| Misaligned expectations | Stakeholders review plan before execution |
| Resource conflicts      | Resource availability verified            |
| Timeline surprises      | Delivery dates agreed upfront             |
| Scope disagreements     | Explicit scope sign-off                   |

**Tools Required:**

| Tool               | Type    | Purpose                  |
| ------------------ | ------- | ------------------------ |
| Approval Queue     | Builtin | Approval workflow        |
| Slack/Teams        | MCP     | Notifications            |
| Calendar           | MCP     | Schedule review meetings |
| Document Generator | Builtin | Plan documentation       |

---

## Phase 3: Execute

### 3.1 Code Generation + Testing

**What It Prevents:**

| Problem               | How It's Prevented                |
| --------------------- | --------------------------------- |
| Syntax errors         | Tree-sitter validates before save |
| Type errors           | LSP checks continuously           |
| Missing tests         | Tests generated alongside code    |
| Broken existing code  | Affected tests run before commit  |
| Style inconsistencies | Auto-formatted, linted            |

**Tools Required:**

| Tool                    | Type            | Purpose                      |
| ----------------------- | --------------- | ---------------------------- |
| Tree-sitter             | Builtin         | Syntax validation            |
| LSP (Pylance, tsserver) | MCP             | Type checking                |
| ESLint/Ruff/Clippy      | Builtin (Shell) | Linting                      |
| Pytest/Jest/Cargo test  | Builtin (Shell) | Test execution               |
| Dependency Graph        | Builtin         | Affected test identification |
| Git                     | MCP             | Version control              |
| Work Tracker            | Builtin         | Lock management              |

---

### 3.2 Prevent Bugs (6-Layer Validation)

**Preventive Layers:**

| Layer | Tool        | Speed   | What It Prevents          |
| ----- | ----------- | ------- | ------------------------- |
| 1     | Tree-sitter | ~5ms    | Syntax errors             |
| 2     | LSP         | ~200ms  | Type mismatches           |
| 3     | Linter      | ~100ms  | Common bug patterns       |
| 4     | Unit Tests  | ~2-10s  | Logic errors, edge cases  |
| 5     | Integration | ~10-60s | Component interaction     |
| 6     | Dep Graph   | ~1s     | Ripple effects, conflicts |

---

### 3.3 Prevent Merge Conflicts (File Locking)

**What It Prevents:**

| Problem                      | How It's Prevented            |
| ---------------------------- | ----------------------------- |
| Two people editing same file | File locking system           |
| Related file conflicts       | Dependency-aware locking      |
| Context switching conflicts  | One task per person at a time |
| Stale branch conflicts       | Continuous rebase             |

**Tools Required:**

| Tool                   | Type    | Purpose                             |
| ---------------------- | ------- | ----------------------------------- |
| Dependency Graph (GNN) | Builtin | File relationships, impact analysis |
| Work Tracker           | Builtin | Lock management, active tasks       |
| Git                    | MCP     | Branch status, rebase operations    |
| Notification System    | Builtin | Conflict warnings                   |

---

### 3.4 Prevent Security Issues (5-Layer Security)

**Prevention Layers:**

| Layer | When               | Tool         | What It Prevents         |
| ----- | ------------------ | ------------ | ------------------------ |
| 1     | Before Adding      | CVE Database | Vulnerable dependencies  |
| 2     | During Development | Agent        | SQL injection, XSS       |
| 3     | Before Commit      | Gitleaks     | Secrets in code          |
| 4     | With Unit Tests    | Agent        | Security test generation |
| 5     | Before PR          | Semgrep      | SAST analysis            |

**Tools Required:**

| Tool                  | Type            | Purpose                          |
| --------------------- | --------------- | -------------------------------- |
| Snyk                  | MCP             | Vulnerability database, scanning |
| npm audit / pip-audit | Builtin (Shell) | Dependency vulnerabilities       |
| Semgrep               | Builtin (Shell) | Code pattern security scanning   |
| Bandit                | Builtin (Shell) | Python security linter           |
| Gitleaks              | Builtin (Shell) | Secrets detection                |
| Trivy                 | Builtin (Shell) | Container security scanning      |

---

### 3.5 Auto Unit & Integration Tests

**Test Generation Matrix:**

| Code Type             | Unit Tests Generated                       | Integration Tests Generated        |
| --------------------- | ------------------------------------------ | ---------------------------------- |
| Pure function         | Input/output for normal, edge, error cases | N/A                                |
| Class/method          | Method behavior, state changes             | Cross-method interactions          |
| API endpoint          | N/A                                        | Request/response, auth, validation |
| Database operation    | Mocked DB calls                            | Actual DB with test data           |
| External service call | Mocked service responses                   | Contract tests                     |

---

### 3.6 Feature Sign-off

**Sign-off Checklist:**

| Check                         | Automated           | Manual Review |
| ----------------------------- | ------------------- | ------------- |
| All tests passing             | ✅ Yes              |               |
| Test coverage meets threshold | ✅ Yes              |               |
| No security vulnerabilities   | ✅ Yes              |               |
| Documentation updated         | ✅ Yes              |               |
| Code reviewed                 |                     | ✅ Yes        |
| Acceptance criteria met       |                     | ✅ Yes        |
| Performance acceptable        | ✅ Yes (benchmarks) |               |

---

## Phase 4: Deploy

### 4.1 Pre-Deploy Validation

**What It Prevents:**

| Problem                | How It's Prevented             |
| ---------------------- | ------------------------------ |
| Deploying broken code  | Full test suite must pass      |
| Environment mismatches | Configuration validated        |
| Missing migrations     | Migration status checked       |
| Incompatible versions  | Version compatibility verified |

**Tools Required:**

| Tool                      | Type            | Purpose             |
| ------------------------- | --------------- | ------------------- |
| GitHub Actions / CircleCI | MCP             | CI/CD pipeline      |
| Pytest/Jest               | Builtin (Shell) | Test execution      |
| Snyk/Trivy                | Builtin (Shell) | Security scanning   |
| Environment Validator     | Builtin         | Config verification |
| Approval Queue            | Builtin         | Deploy approvals    |

---

### 4.2 Auto Deploy

**Deployment Platforms:**

| Platform               | Type | Use Case                       |
| ---------------------- | ---- | ------------------------------ |
| Railway                | MCP  | Rapid deployment, auto-scaling |
| GCP (Cloud Run, GKE)   | MCP  | Google Cloud workloads         |
| AWS (ECS, Lambda, EKS) | MCP  | Amazon Web Services            |
| Azure                  | MCP  | Microsoft Azure                |
| Kubernetes (generic)   | MCP  | Self-managed K8s               |

**Deployment Strategies:**

| Strategy     | When to Use            | How It Works                 |
| ------------ | ---------------------- | ---------------------------- |
| Rolling      | Standard deployments   | Replace instances gradually  |
| Blue-Green   | Zero downtime required | Switch traffic all at once   |
| Canary       | High-risk changes      | Deploy to small % first      |
| Feature Flag | Gradual rollout        | Deploy code, enable via flag |

---

## Phase 5: Monitor / Maintain

### 5.1 Self-Healing (Post-MVP)

**What It Prevents:**

| Problem                    | How It's Prevented                 |
| -------------------------- | ---------------------------------- |
| Prolonged outages          | Automatic rollback on failure      |
| Repeated incidents         | Root cause analysis and fix        |
| Alert fatigue              | Smart deduplication and escalation |
| Manual intervention delays | Automated first response           |

**Self-Healing Capabilities:**

| Scenario                      | Automatic Response                |
| ----------------------------- | --------------------------------- |
| Error rate spike after deploy | Auto-rollback to previous version |
| Memory leak detected          | Restart affected pods/containers  |
| Database connection exhausted | Scale connection pool, alert      |
| Third-party service timeout   | Enable fallback, circuit breaker  |
| Certificate expiring          | Auto-renew, alert if fails        |
| Disk space low                | Clean old logs, alert             |
| Rate limit approaching        | Throttle non-critical traffic     |

**Monitoring Tools:**

| Tool               | Type | Purpose                            |
| ------------------ | ---- | ---------------------------------- |
| Sentry             | MCP  | Error tracking, stack traces       |
| Datadog            | MCP  | Metrics, APM, logs                 |
| New Relic          | MCP  | Application performance monitoring |
| PagerDuty          | MCP  | On-call management, escalation     |
| Opsgenie           | MCP  | Incident management                |
| Prometheus/Grafana | MCP  | Metrics and dashboards             |
| Jira               | MCP  | Issue tracking                     |
| Slack              | MCP  | Notifications                      |

---

## Tool Summary by Type

| Category              | Builtin | MCP    | Shell (via Builtin) |
| --------------------- | ------- | ------ | ------------------- |
| Architecture & Design | 4       | 5      | 1                   |
| Planning              | 3       | 4      | 0                   |
| Code Intelligence     | 2       | 3      | 0                   |
| Testing               | 1       | 0      | 3                   |
| Security              | 1       | 1      | 3                   |
| Version Control       | 0       | 1      | 0                   |
| Deployment            | 1       | 6      | 0                   |
| Monitoring            | 0       | 8      | 0                   |
| Documentation         | 2       | 3      | 1                   |
| **Total**             | **14**  | **31** | **8**               |

---

## MCP Server Priority

### Priority 0 (Must Have for MVP)

| Tool        | Reason                                  |
| ----------- | --------------------------------------- |
| Git/GitHub  | Version control is fundamental          |
| Jira/Linear | Issue tracking integration              |
| Slack       | Team notifications                      |
| Railway     | Deployment platform for rapid iteration |

### Priority 1 (High Value)

| Tool               | Reason                                    |
| ------------------ | ----------------------------------------- |
| Sentry             | Error tracking essential for self-healing |
| PagerDuty/Opsgenie | Incident management                       |
| Confluence/Notion  | Documentation sync                        |
| GCP/AWS            | Enterprise cloud deployment               |

### Priority 2 (Nice to Have)

| Tool              | Reason              |
| ----------------- | ------------------- |
| Figma             | Design handoff      |
| Datadog/New Relic | Advanced monitoring |
| CircleCI          | Alternative CI/CD   |
| Google Docs       | Document reading    |

---

**END OF SECTION 3**

**Reference:** For detailed scenarios, flows, and implementation guidance, see: `docs/Research & specifications/*Preventive Development Cycle.md`

---

# 4. Unified Tool Interface (UTI) Architecture — Consolidated Capability Table

## 4.1 Overview

The Unified Tool Interface (UTI) provides the agent with access to all capabilities required for autonomous development. This section consolidates the 21+ capability categories into a single table, grouped by protocol combinations and mapped to Preventive Development Cycle phases.

**Key Design Principles:**

- **Protocol Selection:** Builtin for core differentiators and performance-critical ops; MCP for ecosystem services; DAP for debugging
- **Fallback Strategy:** Every capability has a backup protocol where feasible
- **Phase Mapping:** Each capability is mapped to the PDC phase(s) where it's primarily used

**Source Files:**

- UTI Specification: `docs/Research & specifications/*Yantra Unified Tool Interface.md`
- Preventive Development Cycle: `docs/Research & specifications/*Preventive Development Cycle.md`

---

## 4.2 The Need for UTI: Abstraction Layer

The Unified Tool Interface (UTI) is Yantra's solution to this complexity. It provides a **single, consistent abstraction layer** that:

1. **Normalizes Tool Access:** All capabilities exposed through a uniform API
2. **Handles Protocol Differences:** Agent doesn't care if it's MCP, LSP, DAP, or Builtin
3. **Smart Routing:** Automatically selects the right protocol/tool for each operation
4. **Unified Auth:** Single authentication system for all external services
5. **Consistent Error Handling:** Retry logic, fallbacks, circuit breakers built-in
6. **Performance Optimization:** Caching, batching, connection pooling automatic

**Benefits for the Agent:**

```
WITHOUT UTI (Complex):
  Agent wants to deploy to Railway:
  ├─ Import Railway SDK
  ├─ Handle Railway authentication (API key)
  ├─ Learn Railway-specific API endpoints
  ├─ Parse Railway response formats
  ├─ Implement retry logic for Railway rate limits
  ├─ Handle Railway-specific errors
  └─ Poll for deployment status

  Agent wants to update Linear ticket:
  ├─ Import Linear GraphQL client (different from Railway)
  ├─ Handle Linear authentication (OAuth, different from Railway)
  ├─ Learn Linear GraphQL schema
  ├─ Parse GraphQL responses (different format)
  ├─ Implement retry logic for GraphQL errors
  └─ Handle Linear-specific errors

WITH UTI (Simple):
  Agent wants to deploy:
  └─ UTI.deploy(platform="railway", env="staging")

  Agent wants to update ticket:
  └─ UTI.update_ticket(system="linear", ticket_id="ENG-123", status="deployed")

  UTI handles: Protocol selection, auth, execution, retries, errors, parsing
```

**Benefits for Yantra:**

1. **Add new tools easily:** Implement UTI interface once, available everywhere
2. **Switch protocols:** Move from MCP to Builtin (or vice versa) without changing agent code
3. **A/B test providers:** Try different MCP servers without rewriting integrations
4. **Centralized monitoring:** Track all tool usage through single point
5. **Cost optimization:** Route expensive operations to cheaper alternatives

---

## 4.3 UTI Architecture Overview: Two Consumers, Two Protocols

Modern language tools like Pylance expose **both** LSP and MCP interfaces because they serve different consumers with different needs:

| Consumer            | Protocol          | Characteristics                                            | Use Cases                                                                               |
| ------------------- | ----------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Editor (Monaco)** | **LSP**           | Real-time, position-aware, streaming, tied to editor state | Autocomplete as you type, hover documentation, live diagnostics, syntax highlighting    |
| **AI Agent**        | **MCP + Builtin** | Discrete request/response, stateless, batch-capable        | "Find all functions in this file", "Get dependencies of module X", "Run affected tests" |

**Key Design Decision:** The UTI exposes **MCP + Built-in** to the agent. LSP is used internally for editor features but not directly exposed through UTI.

**Visual Architecture:**

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

**Why Two Adapters?**

**MCP Adapter:**

- Connects to external MCP servers (community or custom)
- Examples: Git, GitHub, databases, deployment platforms
- Benefit: Leverage community-maintained servers
- Trade-off: External dependency, network latency

**Builtin Adapter:**

- Direct Rust implementations within Yantra
- Examples: File operations, terminal, dependency graph, browser CDP
- Benefit: Maximum performance, no external dependencies, full control
- Trade-off: We maintain the code

---

## 4.4 Protocol Selection Framework

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

### Protocol Decision Examples

**Example 1: File Operations → Builtin**

- ✅ Core capability (needed for everything)
- ✅ Performance-critical (<1ms required)
- ✅ No external dependency needed
- ❌ No suitable MCP server
- **Decision:** Builtin (with MCP fallback if needed)

**Example 2: Git Operations → MCP**

- ❌ Not core differentiator (standard Git operations)
- ❌ Not performance-critical (100ms acceptable)
- ✅ Well-maintained MCP server exists (`@modelcontextprotocol/server-git`)
- ✅ Builtin fallback available (git2-rs for direct access)
- **Decision:** MCP primary, Builtin fallback

**Example 3: Dependency Graph → Builtin**

- ✅ Core differentiator (Yantra's killer feature)
- ✅ Performance-critical (<10ms queries)
- ✅ Tight integration with GNN needed
- ❌ No external solution exists
- **Decision:** Builtin only

**Example 4: Database Operations → MCP**

- ❌ Not core differentiator
- ✅ Well-maintained MCP servers exist (postgres, mysql, sqlite, mongodb)
- ✅ Platform-specific (different per database)
- ❌ No need for Yantra-specific implementation
- **Decision:** MCP only

---

## 4.5 Consolidated Capability Matrix

**Table Columns:**

- **Purpose:** CG (Code Generation), TS (Testing), DP (Deployment), MM (Monitor/Maintain)
- **Category:** Functional grouping of capabilities
- **Capabilities:** Representative capabilities in this category (not exhaustive)
- **Primary Protocol:** Main protocol used
- **Secondary Protocol:** Fallback protocol
- **Example Tool(s):** Representative implementations or MCP servers
- **PDC Phase:** Preventive Development Cycle phase(s) where used

**Grouping:** Rows are grouped by Primary/Secondary protocol combinations. When capabilities within a category use different protocols, the category appears in multiple rows.

---

### UTI Capability Table

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

---

## 4.6 Protocol Distribution Summary

| Protocol          | Total Categories | Primary Use Cases                                             | Maintainer                |
| ----------------- | ---------------- | ------------------------------------------------------------- | ------------------------- |
| **Builtin**       | 11               | Core differentiators, performance-critical, security-critical | Yantra                    |
| **MCP**           | 8                | Ecosystem services, platform integrations, external tools     | Community + Vendors       |
| **DAP**           | 1                | Debugging operations                                          | Debug adapter maintainers |
| **MCP + Builtin** | 1                | Hybrid (test data seeding)                                    | Mixed                     |

**Total Tool Count:**

- Builtin: ~95 individual tools (includes file ops, terminal, tree-sitter, GNN, browser, viz)
- MCP: ~45 tools across multiple servers
- DAP: ~10 debugging tools
- **Grand Total: ~160 agent-accessible tools**

---

## 4.7 Implementation Notes

### Protocol Selection Rules

| Question                                     | If YES →    |
| -------------------------------------------- | ----------- |
| Is it a core differentiator we must control? | **Builtin** |
| Is performance critical (<10ms required)?    | **Builtin** |
| Is it security-critical (shell, secrets)?    | **Builtin** |
| Well-maintained community MCP server exists? | **MCP**     |
| Platform-specific (deployment, monitoring)?  | **MCP**     |
| Debugging-specific?                          | **DAP**     |

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

**END OF SECTION 4**

**Reference:** For full capability details, protocol decision matrix, and tool definitions, see: `docs/Research & specifications/*Yantra Unified Tool Interface.md`

---

## 5. Complete System Architecture

### 5.1 Architecture Overview

Yantra's architecture is designed around the **Preventive Development Cycle (PDC)** with **Unified Tool Interface (UTI)** as the abstraction layer enabling seamless tool orchestration across protocols. The system operates as an AI-first development platform with five distinct layers working together to deliver code that never breaks.

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
│  │  │  Yantra Codex (AI Code Generation) - GraphSAGE GNN              │  │ │
│  │  │  ├─ Neural Network: 1024-dim embeddings, 150M parameters        │  │ │
│  │  │  ├─ Inference: 15ms (CPU), 5ms (GPU), ~600MB model              │  │ │
│  │  │  ├─ Pattern Recognition: 978-dim problem features → code logic  │  │ │
│  │  │  ├─ Confidence Scoring: 0.0-1.0 (triggers LLM review < 0.8)     │  │ │
│  │  │  ├─ Continuous Learning: Learns from LLM corrections            │  │ │
│  │  │  └─ Cost Optimization: 90% LLM call reduction (96% after 12mo) │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Vector Database (RAG) - fastembed + redb                       │  │ │
│  │  │  ├─ Code Templates: Pre-validated patterns                      │  │ │
│  │  │  ├─ Best Practices: Language-specific idioms                    │  │ │
│  │  │  ├─ Project Patterns: Learned from codebase                     │  │ │
│  │  │  ├─ Failure Library: Known issues, LLM failure patterns         │  │ │
│  │  │  └─ Semantic Search: <10ms retrieval for context assembly       │  │ │
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
│  │  │  4-Tier Data Storage Architecture                               │  │ │
│  │  │  ├─ Tier 0: Cloud Storage (Optional, team coordination)         │  │ │
│  │  │  ├─ Tier 1: In-Memory (Hot path: GNN queries, active state)     │  │ │
│  │  │  ├─ Tier 2: Local SQLite (Persistent: graph, state, history)    │  │ │
│  │  │  └─ Tier 3: File System (Cold: logs, backups, large artifacts)  │  │ │
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

### 5.2 Two Critical Systems: Dependency Graph vs Yantra Codex

**IMPORTANT DISTINCTION:** Yantra has two separate intelligence systems with different purposes. Understanding this distinction is critical for architecture comprehension:

#### System 1: Dependency Graph (Code Intelligence)

**Purpose:** Track structural code relationships (imports, calls, dependencies)
**Implementation:** petgraph-based directed graph (Rust)
**Code Reference:** Often called "GNN" in codebase (historical naming, NOT a neural network)

**Architecture:**

```
Local Dependency Graph (Tier 1)          Cloud Dependency Graph (Tier 0 - Phase 2B)
─────────────────────────────            ────────────────────────────────────────
• petgraph in-memory                     • PostgreSQL + Redis (optional)
• Structural relationships               • Shared graph structure for teams
• Semantic embeddings (fastembed)        • Conflict prevention coordination
• <1ms query time                        • Multi-agent synchronization
• Privacy: All code stays local          • Privacy: Structure only, no code content
```

**What It Tracks:**

- File → File imports
- Function → Function calls
- Class → Class inheritance
- Package → File usage
- Semantic similarity (optional embeddings)

**NOT a Neural Network** - This is a graph data structure (petgraph) with optional embeddings for fuzzy search.

#### System 2: Yantra Codex (AI Code Generation)

**Purpose:** Generate code from natural language using machine learning
**Implementation:** GraphSAGE neural network (Python/PyTorch) + Tree Sitter
**Code Reference:** Actual neural network for pattern recognition

**Architecture:**

```
Local Yantra Codex (Tier 1)              Cloud Yantra Codex (Tier 0 - Optional)
───────────────────────────              ────────────────────────────────────
• GraphSAGE GNN (1024-dim)               • Aggregated pattern embeddings
• 150M parameters, ~600MB                • Collective learning from users
• 15ms inference (CPU), 5ms (GPU)        • Privacy: Embeddings only, no code
• Learns from LLM feedback               • Network effects → better accuracy
• Privacy: All code stays local          • Opt-in only
```

**What It Does:**

- Understands problem intent (978-dim features)
- Predicts logic patterns (GraphSAGE neural network) and use Tree Sitter for context and generate codes
- Generates code with confidence scoring
- Learns from LLM corrections over time
- Cost optimization (90% fewer LLM calls)

**IS a Graph Neural Network** - Real machine learning model trained on code patterns.

#### Why Two Systems?

| Aspect          | Dependency Graph                | Yantra Codex               |
| --------------- | ------------------------------- | -------------------------- |
| **Purpose**     | Code relationships              | Code generation            |
| **Technology**  | petgraph (data structure)       | GraphSAGE (neural network) |
| **Input**       | AST from tree-sitter            | Problem description        |
| **Output**      | Dependency queries              | Generated code             |
| **Speed**       | <1ms                            | 15ms                       |
| **Learning**    | No learning                     | Continuous learning        |
| **Local/Cloud** | Both (sync structure)           | Both (sync embeddings)     |
| **Codebase**    | Often called "GNN" (misleading) | "Yantra Codex"             |

**Integration Flow:**
Dependency Graph provides context → Yantra Codex generates code → Dependency Graph validates new code fits properly

### 5.3 Yantra Codex: Hybrid AI Pair Programming

**Core Innovation:** Yantra Codex acts as a **junior AI developer** paired with a **senior LLM reviewer** (Claude/GPT-4), combining GNN speed with LLM reasoning.

#### Pair Programming Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: User Intent                                        │
│  "Create REST API endpoint to get user by ID"               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Yantra Codex Generates (Junior Developer)         │
│  • Extract 978-dimensional problem features                 │
│  • GraphSAGE predicts logic pattern (15ms)                  │
│  • tree-sitter generates code structure                     │
│  • Calculate confidence score (0.0-1.0)                     │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                 Confidence >= 0.8?
                         │
        ┌────────────────┴────────────────┐
        │ YES (80% of cases)              │ NO (20% of cases)
        ↓                                 ↓
┌───────────────────────┐    ┌────────────────────────────────┐
│  STEP 3a: Direct Use  │    │  STEP 3b: LLM Review (Senior)  │
│  • Yantra code used   │    │  • Send: Yantra code + issues  │
│  • Fast (15ms)        │    │  • LLM reviews edge cases      │
│  • Free (no LLM cost) │    │  • Adds error handling         │
│                       │    │  • Improves quality            │
│                       │    │  • User choice: Claude/GPT-4   │
└───────────┬───────────┘    └───────────┬────────────────────┘
            │                            │
            └────────────┬───────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Validation Pipeline (Mandatory)                    │
│  • Unit tests (pytest/jest)                                 │
│  • Security scan (Semgrep)                                  │
│  • Dependency validation (GNN)                              │
│  • Browser preview (CDP, if UI)                             │
│  • 100% pass rate required                                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                  All tests pass?
                         │
        ┌────────────────┴────────────────┐
        │ YES                             │ NO
        ↓                                 ↓
┌───────────────────────┐    ┌────────────────────────────────┐
│  STEP 5a: Commit      │    │  STEP 5b: Feedback Loop        │
│  • Git commit via MCP │    │  • Extract failure patterns    │
│  • Update GNN graph   │    │  • Yantra learns from failure  │
│  • Log to history     │    │  • LLM generates fixed code    │
│                       │    │  • Retry validation pipeline   │
└───────────────────────┘    └────────────┬───────────────────┘
                                          │
                                          └──► Back to STEP 4
```

#### Model Specifications

**GraphSAGE Neural Network:**

- **Architecture:** 978 → 1536 → 1280 → 1024 dimensions
- **Parameters:** ~150M
- **Model Size:** ~600 MB
- **Inference Time:** 15ms (CPU), 5ms (GPU)
- **Training:** Python/PyTorch, continuous learning from LLM feedback

**Why 1024 Dimensions:**

- Sufficient capacity for multi-step logic patterns
- 55-60% initial accuracy (vs 40% with 256 dims)
- Fast inference (feels instant to users)
- Scalable to 2048+ dims in future versions

#### Learning Loop

```
Generated Code → Validation → Pass/Fail → Pattern Extraction → Training Update
     ↑                                                               ↓
     └───────────────────────── Model Improvement ─────────────────┘
```

**Continuous Improvement:**

- **Week 1:** 55% direct use, 45% LLM review
- **Month 3:** 70% direct use, 30% LLM review
- **Month 12:** 96% direct use, 4% LLM review
- **Cost Reduction:** 90% → 96% over 12 months

### 5.4 UTI Integration in Architecture

The **Unified Tool Interface (UTI)** sits at Layer 2 (Orchestration) and serves as the **central nervous system** for tool coordination. It abstracts protocol differences and enables both the **LLM Agent** and **Workflow Executor** to invoke tools without protocol awareness.

#### UTI Position in Stack

```
Layer 1: User Interface
         ↕
Layer 2: ┌─────────────────────────────────────┐
         │  UTI (Protocol Router & Adapters)   │ ← Central abstraction layer
         │  ├─ MCP Tools (31 tools via MCP)    │
         │  ├─ LSP Tools (2 tools via LSP)     │
         │  ├─ DAP Tools (1 tool via DAP)      │
         │  └─ Builtin Tools (14 native Rust)  │
         └─────────────────────────────────────┘
         ↕                          ↕
Multi-LLM Orchestration    PDC State Machine
Team of Agents             File Locking System
         ↕
Layer 3: Intelligence (GNN, Yantra Codex, Vector DB)
         ↕
Layer 4: Validation (Testing, Security, Browser CDP)
         ↕
Layer 5: Integration (Git MCP, File System, Storage)
```

#### Protocol Distribution Across Layers

| Layer | Component              | Protocol Used | UTI Role                           |
| ----- | ---------------------- | ------------- | ---------------------------------- |
| 1     | Monaco Editor          | Builtin       | Direct Rust API                    |
| 2     | LLM Agent              | Via UTI       | Protocol-agnostic tool invocation  |
| 2     | Workflow Executor      | Via UTI       | Protocol-agnostic tool invocation  |
| 2     | Team of Agents         | Via UTI       | Coordinated tool access            |
| 3     | Dependency Graph (GNN) | Builtin       | Direct petgraph/tree-sitter access |
| 3     | Yantra Codex           | Builtin       | Direct PyTorch model inference     |
| 3     | Vector DB (RAG)        | Builtin       | Direct fastembed/redb access       |
| 4     | Testing Framework      | Builtin       | Direct pytest/jest execution       |
| 4     | Security Scanner       | Builtin       | Direct Semgrep API                 |
| 4     | Browser CDP            | Builtin       | Direct chromiumoxide binding       |
| 5     | Git Operations         | **MCP**       | Via UTI → MCP adapter              |
| 5     | GitHub API             | **MCP**       | Via UTI → MCP adapter              |
| 5     | File System            | Builtin       | Direct Rust std::fs                |

**Key Insight:** UTI is exclusively used by **Layer 2 components** (orchestration) to coordinate external tools and services. Layers 3-5 use **direct Builtin APIs** for performance-critical operations.

### 5.5 Multi-LLM Orchestration

**Architecture Pattern:** Circuit breaker + retry with exponential backoff + intelligent routing

#### Supported LLM Providers

| Provider         | Primary Use Case        | Rate Limit Strategy    | Failover Target |
| ---------------- | ----------------------- | ---------------------- | --------------- |
| Claude Sonnet 4  | Code generation, review | 5 req/min (Tier 1)     | GPT-4 Turbo     |
| GPT-4 Turbo      | Validation, fallback    | 10 req/min             | Claude Sonnet 4 |
| DeepSeek V3      | Cost-efficient tasks    | 20 req/min             | GPT-4 Turbo     |
| Gemini 2.0 Flash | Fast prototyping        | 15 req/min             | Claude Sonnet 4 |
| Groq (Llama 3.3) | Sub-second responses    | 30 req/min (free tier) | DeepSeek V3     |

#### Routing Logic

```rust
fn route_llm_request(task_type: TaskType, priority: Priority) -> LLMProvider {
    match (task_type, priority) {
        (TaskType::CodeGeneration, Priority::High) => LLMProvider::ClaudeSonnet4,
        (TaskType::Validation, _) => LLMProvider::GPT4Turbo,
        (TaskType::Refactoring, Priority::Low) => LLMProvider::DeepSeekV3,
        (TaskType::QuickReview, _) => LLMProvider::Groq,
        (TaskType::Prototyping, _) => LLMProvider::GeminiFlash,
        _ => LLMProvider::ClaudeSonnet4, // Default
    }
}
```

#### Circuit Breaker Pattern

```
┌────────────────────────────────────────────────┐
│  Request → Check Circuit State                 │
│              ↓                                  │
│     ┌────────┴────────┐                        │
│     │ CLOSED (Normal) │                        │
│     └────────┬────────┘                        │
│              ↓                                  │
│         Success/Failure?                       │
│              ↓                                  │
│     Failure Rate > 50%?                        │
│              ↓ YES                             │
│     ┌────────┴────────┐                        │
│     │ OPEN (Fail-Fast)│                        │
│     └────────┬────────┘                        │
│              ↓                                  │
│     Wait 30s → Try 1 request                   │
│              ↓                                  │
│     ┌────────┴────────────┐                    │
│     │ HALF-OPEN (Testing) │                    │
│     └────────┬────────────┘                    │
│              ↓                                  │
│         Success?                               │
│      ↓ YES        ↓ NO                         │
│   CLOSED       OPEN (60s wait)                 │
└────────────────────────────────────────────────┘
```

#### Response Caching

- **Cache Storage:** Redis (in-memory) + SQLite (persistence)
- **Cache Key:** Hash of (LLM provider + model + prompt + parameters)
- **TTL:** 24 hours for code generation, 1 hour for validation
- **Invalidation:** On codebase changes detected by GNN
- **Hit Rate Target:** 40%+ (reduces cost and latency)

### 5.6 PDC State Machine

The **Preventive Development Cycle (PDC)** is implemented as a state machine with five phases. Each phase has specific entry/exit conditions and approval gates.

#### State Transition Diagram

```
                    ┌──────────────────┐
                    │  IDLE (Start)    │
                    └────────┬─────────┘
                             ↓
                    ┌────────────────────────────┐
                    │  PHASE 1: ARCHITECT/DESIGN │
                    │  • Analyze intent          │
                    │  • Design solution         │
                    │  • Plan dependencies       │
                    └────────┬───────────────────┘
                             ↓ [Design approved]
                    ┌────────────────────────────┐
                    │  PHASE 2: PLAN             │
                    │  • Create task breakdown   │
                    │  • Estimate complexity     │
                    │  • Allocate tools          │
                    └────────┬───────────────────┘
                             ↓ [Plan approved]
                    ┌────────────────────────────┐
                    │  PHASE 3: EXECUTE          │
                    │  • Generate code           │
                    │  • Run tests               │
                    │  • Security scan           │
                    │  • Validate dependencies   │
                    └────────┬───────────────────┘
                             ↓ [All tests pass]
                    ┌────────────────────────────┐
                    │  PHASE 4: DEPLOY           │
                    │  • Commit to Git           │
                    │  • Update GNN graph        │
                    │  • Browser preview         │
                    └────────┬───────────────────┘
                             ↓ [Deployed]
                    ┌────────────────────────────┐
                    │  PHASE 5: MONITOR          │
                    │  • Runtime validation      │
                    │  • Log issues              │
                    │  • Feedback loop           │
                    └────────┬───────────────────┘
                             ↓
                    ┌──────────────────┐
                    │  COMPLETE         │
                    └───────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
         [New Task]    [Rollback]    [Iterate]
```

#### State Persistence

**Storage:** SQLite with Write-Ahead Logging (WAL)

```sql
CREATE TABLE pdc_state (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL,
    current_phase TEXT NOT NULL, -- 'architect' | 'plan' | 'execute' | 'deploy' | 'monitor'
    phase_data JSON NOT NULL,    -- Phase-specific context
    approval_status TEXT,         -- 'pending' | 'approved' | 'rejected'
    checkpoint_data BLOB,         -- Serialized state for rollback
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Approval Gates

| Phase | Approval Required?           | Approval Trigger                       |
| ----- | ---------------------------- | -------------------------------------- |
| 1     | Yes (Guided/Auto modes)      | Design deviates from existing patterns |
| 2     | Optional (user configurable) | High complexity score (>8/10)          |
| 3     | No (automated validation)    | Tests + security scans must pass       |
| 4     | Yes (before Git commit)      | User confirms changes                  |
| 5     | No (passive monitoring)      | Runtime errors trigger alerts          |

**Clean Code Mode:** All approval gates auto-approved if code passes all validations.

### 5.7 Team of Agents Architecture

Yantra uses a **multi-agent system** where specialized agents collaborate on different aspects of development. Agents communicate via a message bus and coordinate through shared state.

#### Agent Roles

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MESSAGE BUS (In-Memory)                     │
│                    Event-Driven Agent Communication                  │
└────┬────────────────┬────────────────┬────────────────┬─────────────┘
     │                │                │                │
     ↓                ↓                ↓                ↓
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│  ARCHITECT  │ │   CODING    │ │   TESTING    │ │  SECURITY    │
│    AGENT    │ │    AGENT    │ │    AGENT     │ │    AGENT     │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘
│ • Design    │ │ • Generate  │ │ • Create     │ │ • Scan for   │
│   solution  │ │   code      │ │   tests      │ │   vulns      │
│ • Plan deps │ │ • Apply     │ │ • Execute    │ │ • Auto-fix   │
│ • Select    │ │   patterns  │ │   tests      │ │   critical   │
│   tools     │ │ • Refactor  │ │ • Coverage   │ │ • Report     │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘
      ↓                ↓                ↓                ↓
      └────────────────┴────────────────┴────────────────┘
                           ↓
              ┌────────────────────────┐
              │   SHARED STATE         │
              │  (SQLite + In-Memory)  │
              │  • Task queue          │
              │  • File locks          │
              │  • Agent status        │
              └────────────────────────┘
```

#### Agent Communication Protocol

**Message Structure:**

```rust
struct AgentMessage {
    id: Uuid,
    sender: AgentType,        // Architect | Coding | Testing | Security
    receiver: AgentType,      // Or Broadcast
    message_type: MessageType, // Request | Response | Event | Error
    payload: serde_json::Value,
    priority: Priority,       // High | Medium | Low
    timestamp: DateTime<Utc>,
}
```

**Example Flow:**

1. **Architect Agent** → Broadcast: "Design complete, dependencies identified"
2. **Coding Agent** → Architect: "Request dependency details for file X"
3. **Architect Agent** → Coding: "Dependencies: [list]"
4. **Coding Agent** → Broadcast: "Code generated, ready for testing"
5. **Testing Agent** → Coding: "Tests failed, error details attached"
6. **Coding Agent** → Testing: "Fixed code, re-run tests"
7. **Testing Agent** → Broadcast: "All tests passed"
8. **Security Agent** → Broadcast: "Security scan complete, 0 issues"

#### Coordination via File Locking

**Problem:** Multiple agents modifying same file simultaneously
**Solution:** SQLite-based distributed file locking

```sql
CREATE TABLE file_locks (
    file_path TEXT PRIMARY KEY,
    locked_by TEXT NOT NULL,      -- Agent ID
    lock_type TEXT NOT NULL,       -- 'read' | 'write'
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);
```

**Lock Acquisition:**

```rust
async fn acquire_write_lock(file_path: &str, agent_id: &str) -> Result<FileLock> {
    // Try to acquire lock (expires in 30s)
    let lock = db.execute(
        "INSERT INTO file_locks (file_path, locked_by, lock_type, expires_at)
         VALUES (?, ?, 'write', datetime('now', '+30 seconds'))
         ON CONFLICT DO NOTHING",
        params![file_path, agent_id]
    )?;

    // Check if lock acquired
    if lock.rows_affected() > 0 {
        Ok(FileLock { file_path, agent_id })
    } else {
        Err(Error::LockContention)
    }
}
```

**Auto-Release:** Locks expire after 30 seconds or are explicitly released when agent completes operation.

### 5.8 Security Framework: 5-Layer Prevention Stack

Yantra implements security as **layers of prevention** rather than post-generation scanning. Each layer catches different vulnerability classes.

#### Layer-by-Layer Security

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: PRE-GENERATION (Intent Validation)                   │
│  ├─ Validate user intent against security policies             │
│  ├─ Block known malicious patterns (e.g., "delete all users")  │
│  ├─ Check permissions for sensitive operations                 │
│  └─ Cost: <1ms, Catches: 5% of issues                          │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: GENERATION-TIME (Pattern Safety)                     │
│  ├─ Safe-by-default code generation templates                  │
│  ├─ SQL injection prevention (parameterized queries)           │
│  ├─ XSS prevention (auto-escape user inputs)                   │
│  ├─ CSRF token inclusion for state-changing operations         │
│  └─ Cost: ~5ms, Catches: 40% of issues                         │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: POST-GENERATION (AST Validation)                     │
│  ├─ Parse generated code with tree-sitter                      │
│  ├─ Check for dangerous function calls (eval, exec, etc.)      │
│  ├─ Validate control flow (no infinite loops)                  │
│  ├─ Ensure error handling exists                               │
│  └─ Cost: ~10ms, Catches: 30% of issues                        │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: PRE-COMMIT (Static Analysis)                         │
│  ├─ Semgrep with OWASP Top 10 rules                            │
│  ├─ Secret scanning (TruffleHog patterns)                      │
│  ├─ Dependency vulnerability check (Safety for Python, npm)    │
│  ├─ License compliance verification                            │
│  └─ Cost: ~10s, Catches: 20% of issues                         │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: RUNTIME MONITORING (Execution Safety)                │
│  ├─ Sandbox execution for untrusted code                       │
│  ├─ Resource limits (CPU, memory, network)                     │
│  ├─ Network traffic monitoring                                 │
│  ├─ Anomaly detection (unexpected behavior)                    │
│  └─ Cost: Continuous, Catches: 5% of issues                    │
└─────────────────────────────────────────────────────────────────┘
```

**Coverage:** 5 + 40 + 30 + 20 + 5 = **100% layered coverage**

**Auto-Fix Capability:**

- Layers 1-2: Block and regenerate with safety constraints
- Layer 3: Auto-fix AST issues (add error handling, remove unsafe calls)
- Layer 4: Auto-fix critical vulnerabilities (e.g., parameterize SQL)
- Layer 5: Isolate and alert (manual review required)

**Escalation Policy:**

- Critical: Auto-fix attempted, human approval required if fix fails
- High: Auto-fix attempted, proceed with warning if fix fails
- Medium/Low: Log and proceed (user notified)

### 5.9 Browser Integration via Chrome DevTools Protocol

Yantra uses **chromiumoxide** (Rust CDP bindings) to control a headless Chrome browser for UI validation and testing.

#### CDP Architecture

```
┌──────────────────────────────────────────────────────┐
│  Yantra (Rust Backend)                               │
│  ┌────────────────────────────────────────────────┐  │
│  │  chromiumoxide (CDP Client)                    │  │
│  │  ├─ Page.navigate()                            │  │
│  │  ├─ Page.captureScreenshot()                   │  │
│  │  ├─ Runtime.evaluate()                         │  │
│  │  ├─ Network.enable()                           │  │
│  │  └─ DOM.querySelector()                        │  │
│  └────────────────┬───────────────────────────────┘  │
│                   │ WebSocket                        │
│                   ↓                                   │
│  ┌────────────────────────────────────────────────┐  │
│  │  Chrome Headless (Browser Process)             │  │
│  │  ├─ Renders UI in memory                       │  │
│  │  ├─ Executes JavaScript                        │  │
│  │  ├─ Monitors console errors                    │  │
│  │  └─ Reports network activity                   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

#### Use Cases

1. **Live Preview:** Render UI changes in real-time as code is generated
2. **Visual Validation:** Screenshot comparison (before vs after)
3. **Interaction Testing:** Simulate user clicks, form submissions
4. **Console Monitoring:** Detect JavaScript runtime errors
5. **Network Inspection:** Verify API calls are made correctly
6. **Performance Metrics:** Measure Core Web Vitals (LCP, FID, CLS)

#### CDP Workflow Example

```rust
async fn validate_ui_change(url: &str) -> Result<ValidationResult> {
    let browser = Browser::default().await?;
    let page = browser.new_page(url).await?;

    // Wait for page to load
    page.wait_for_navigation().await?;

    // Capture screenshot
    let screenshot = page.screenshot(ScreenshotParams::default()).await?;

    // Check for console errors
    let errors = page.evaluate("console.log.calls").await?;

    // Measure performance
    let metrics = page.metrics().await?;

    Ok(ValidationResult {
        screenshot,
        console_errors: errors,
        lcp: metrics.largest_contentful_paint,
        fid: metrics.first_input_delay,
        cls: metrics.cumulative_layout_shift,
    })
}
```

### 5.10 Data Storage: 4-Tier Architecture

Yantra uses a tiered storage strategy optimized for different access patterns and data persistence requirements.

#### Tier Breakdown

| Tier | Storage Type       | Purpose                      | Access Speed | Persistence | Size Limit |
| ---- | ------------------ | ---------------------------- | ------------ | ----------- | ---------- |
| 0    | Cloud (PostgreSQL) | Team coordination (opt-in)   | 50-200ms     | Permanent   | Unlimited  |
| 1    | In-Memory (Rust)   | Hot path (GNN, active state) | <1ms         | Volatile    | 1-2 GB     |
| 2    | SQLite (WAL)       | Persistent local data        | 1-10ms       | Permanent   | 100 GB     |
| 3    | File System        | Cold storage (logs, backups) | 10-100ms     | Permanent   | Unlimited  |

#### Data Placement Strategy

**Tier 0 (Cloud):**

- Dependency graph structure (no code content)
- File modification registry for conflict prevention
- Team agent coordination state
- Usage: Optional, opt-in for team features

**Tier 1 (In-Memory):**

- Dependency graph (petgraph in-memory)
- Active PDC state for current task
- LLM response cache (Redis)
- File lock registry
- Agent message bus

**Tier 2 (SQLite):**

- Dependency graph persistence (snapshot)
- PDC state history
- File locks
- Code generation history
- Security scan results
- Test results

**Tier 3 (File System):**

- Generated code files
- Test output logs
- Security scan reports
- Browser screenshots
- Backup snapshots

#### SQLite Schema (Key Tables)

```sql
-- Dependency graph persistence
CREATE TABLE graph_nodes (
    id INTEGER PRIMARY KEY,
    node_type TEXT NOT NULL, -- 'file' | 'function' | 'class'
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    metadata JSON
);

CREATE TABLE graph_edges (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL, -- 'imports' | 'calls' | 'inherits'
    FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
    FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
);

-- PDC state persistence (from Section 5.6)
CREATE TABLE pdc_state (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL,
    current_phase TEXT NOT NULL,
    phase_data JSON NOT NULL,
    approval_status TEXT,
    checkpoint_data BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- File locking (from Section 5.7)
CREATE TABLE file_locks (
    file_path TEXT PRIMARY KEY,
    locked_by TEXT NOT NULL,
    lock_type TEXT NOT NULL,
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

-- Code generation history
CREATE TABLE generation_history (
    id INTEGER PRIMARY KEY,
    task_description TEXT NOT NULL,
    generated_code TEXT NOT NULL,
    llm_provider TEXT NOT NULL,
    confidence_score REAL,
    validation_status TEXT, -- 'passed' | 'failed' | 'pending'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Security scan results
CREATE TABLE security_scans (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    scan_type TEXT NOT NULL, -- 'semgrep' | 'secrets' | 'dependencies'
    issues JSON NOT NULL,
    severity TEXT NOT NULL, -- 'critical' | 'high' | 'medium' | 'low'
    auto_fixed BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Performance Optimizations

- **SQLite WAL Mode:** Write-Ahead Logging for concurrent reads
- **Connection Pooling:** 10 connections max, reuse across operations
- **Prepared Statements:** Pre-compile frequent queries
- **Indexes:** On foreign keys, frequently queried columns
- **Batch Writes:** Group multiple writes into single transaction
- **Async I/O:** Use tokio::fs for non-blocking file operations

---

**END OF SECTION 5**

**Reference:** For detailed component implementations and code references, see: `/Technical_Guide.md`

---
