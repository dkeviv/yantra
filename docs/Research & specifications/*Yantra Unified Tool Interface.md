# Yantra Unified Tool Interface (UTI) Specification v2

## Executive Summary

The Unified Tool Interface abstracts away the underlying protocol differences (LSP, MCP, DAP, built-in) and presents a single, consistent API for the AI agent to discover, invoke, and manage tools. This revision clarifies the separation between **Editor concerns** (LSP) and **Agent concerns** (MCP + Builtin), and provides a comprehensive capability matrix for building a fully agentic development platform.

---

## 1. Key Insight: Two Consumers, Two Protocols

Modern language tools like Pylance expose **both** LSP and MCP interfaces because they serve different consumers:

| Consumer            | Protocol | Characteristics                                            |
| ------------------- | -------- | ---------------------------------------------------------- |
| **Editor (Monaco)** | LSP      | Real-time, position-aware, streaming, tied to editor state |
| **AI Agent**        | MCP      | Discrete request/response, stateless, batch-capable        |

**Design Decision:** The UTI exposes **MCP + Builtin** to the agent. LSP is used internally for editor features but not exposed through UTI.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              YANTRA                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      MONACO EDITOR                                 │ │
│  │                           │                                        │ │
│  │                      LSP Client                                    │ │
│  │                           │                                        │ │
│  │           ┌───────────────┼───────────────┐                        │ │
│  │           ▼               ▼               ▼                        │ │
│  │     Pylance(LSP)    rust-analyzer     tsserver                     │ │
│  │     [Real-time autocomplete, hover, diagnostics]                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      AI AGENT                                      │ │
│  │                           │                                        │ │
│  │              UNIFIED TOOL INTERFACE (UTI)                          │ │
│  │                           │                                        │ │
│  │           ┌───────────────┴───────────────┐                        │ │
│  │           ▼                               ▼                        │ │
│  │     MCP Adapter                    Builtin Adapter                 │ │
│  │           │                               │                        │ │
│  │     ┌─────┴─────┐                   ┌─────┴─────┐                  │ │
│  │     ▼           ▼                   ▼           ▼                  │ │
│  │  Pylance     Git MCP             File Ops   Tree-sitter            │ │
│  │   (MCP)     Postgres             Terminal   Dep Graph (GNN)        │ │
│  │  GitHub     Railway              Browser    Code Search            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Protocol Selection Framework

For each capability, determine the protocol based on:

| Question                                           | If YES →                      |
| -------------------------------------------------- | ----------------------------- |
| Does the editor need it in real-time while typing? | LSP (Editor only)             |
| Is it a core differentiator we must control?       | Builtin                       |
| Is it a discrete query the agent makes?            | MCP                           |
| Does it need streaming output for progress?        | Builtin or MCP with streaming |
| Is there a well-maintained community server?       | MCP                           |

### Protocol Decision Matrix

| Capability Type    | Editor (Monaco) | Agent (UTI) | Reasoning                                      |
| ------------------ | --------------- | ----------- | ---------------------------------------------- |
| Autocomplete       | LSP             | MCP         | Editor needs real-time; agent needs discrete   |
| Diagnostics        | LSP             | MCP         | Editor streams; agent queries batch            |
| Hover info         | LSP             | MCP         | Editor needs position; agent asks about symbol |
| Go to definition   | LSP             | MCP         | Both need it, different contexts               |
| File operations    | —               | Builtin     | Core capability, no external dependency        |
| Terminal           | —               | Builtin     | Core capability, security critical             |
| Git operations     | —               | MCP         | Well-maintained servers exist                  |
| Database           | —               | MCP         | Well-maintained servers exist                  |
| Dependency graph   | —               | Builtin     | Core differentiator (GNN)                      |
| Browser automation | —               | Builtin     | Core capability (CDP)                          |
| Deployment         | —               | MCP         | Platform-specific (Railway, Vercel)            |

---

## 3. Comprehensive Capability Matrix

### Legend

| Symbol | Meaning                              |
| ------ | ------------------------------------ |
| ●      | Primary protocol for this capability |
| ○      | Secondary/fallback protocol          |
| —      | Not applicable                       |

### Purpose Categories

| Code   | Purpose                                                     |
| ------ | ----------------------------------------------------------- |
| **CG** | Code Generation - Writing, editing, refactoring code        |
| **TS** | Testing - Unit, integration, E2E, coverage                  |
| **DP** | Deployment - Build, deploy, infrastructure                  |
| **MM** | Monitor/Maintain - Logging, debugging, performance, updates |

---

### 3.1 FILE SYSTEM OPERATIONS

| Capability       | Purpose | Builtin | MCP | LSP | Description                                 | Maintainer |
| ---------------- | ------- | ------- | --- | --- | ------------------------------------------- | ---------- |
| file.read        | CG      | ●       | ○   | —   | Read file contents                          | Yantra     |
| file.write       | CG      | ●       | ○   | —   | Write/create files                          | Yantra     |
| file.edit        | CG      | ●       | —   | —   | Surgical edits (line range, search-replace) | Yantra     |
| file.delete      | CG      | ●       | ○   | —   | Delete files                                | Yantra     |
| file.move        | CG      | ●       | ○   | —   | Move/rename files                           | Yantra     |
| file.copy        | CG      | ●       | ○   | —   | Copy files                                  | Yantra     |
| file.search      | CG      | ●       | ○   | —   | Search files by name/pattern                | Yantra     |
| file.grep        | CG      | ●       | —   | —   | Search file contents (regex)                | Yantra     |
| file.watch       | MM      | ●       | —   | —   | Monitor file changes                        | Yantra     |
| directory.create | CG      | ●       | ○   | —   | Create directories                          | Yantra     |
| directory.list   | CG      | ●       | ○   | —   | List directory contents                     | Yantra     |
| directory.tree   | CG      | ●       | —   | —   | Get project structure                       | Yantra     |

**Why Builtin Primary:** File operations are core to everything. Must be fast, reliable, and under our control. MCP fallback via `@modelcontextprotocol/server-filesystem` if needed.

---

### 3.2 CODE INTELLIGENCE

| Capability         | Purpose | Builtin | MCP | LSP | Description                            | Maintainer        |
| ------------------ | ------- | ------- | --- | --- | -------------------------------------- | ----------------- |
| code.symbols       | CG      | ○       | ●   | ●   | Get symbols (functions, classes, vars) | Language servers  |
| code.definition    | CG      | ○       | ●   | ●   | Go to definition                       | Language servers  |
| code.references    | CG      | —       | ●   | ●   | Find all references                    | Language servers  |
| code.completion    | CG      | —       | ●   | ●   | Get completions at position            | Language servers  |
| code.hover         | CG      | —       | ●   | ●   | Get hover/documentation info           | Language servers  |
| code.diagnostics   | CG, MM  | —       | ●   | ●   | Get errors/warnings                    | Language servers  |
| code.actions       | CG      | —       | ●   | ●   | Get quick fixes, refactorings          | Language servers  |
| code.rename        | CG      | —       | ●   | ●   | Rename symbol across project           | Language servers  |
| code.format        | CG      | —       | ●   | ●   | Format document/selection              | Language servers  |
| code.imports       | CG      | —       | ●   | —   | Analyze/organize imports               | Pylance MCP, etc. |
| code.signature     | CG      | —       | ●   | ●   | Function signature help                | Language servers  |
| code.callHierarchy | CG      | —       | ●   | ●   | Incoming/outgoing calls                | Language servers  |
| code.typeHierarchy | CG      | —       | ●   | ●   | Class inheritance                      | Language servers  |

**Protocol Split:**

- **LSP (Editor):** Real-time autocomplete, hover on mouse, diagnostics as-you-type
- **MCP (Agent):** "What functions are in this file?", "Find all usages of X", batch diagnostics

**Builtin Fallback:** Tree-sitter for basic symbol extraction when LSP/MCP unavailable.

---

### 3.3 AST & PARSING (Tree-sitter)

| Capability  | Purpose | Builtin | MCP | LSP | Description                      | Maintainer |
| ----------- | ------- | ------- | --- | --- | -------------------------------- | ---------- |
| ast.parse   | CG      | ●       | —   | —   | Parse file to AST                | Yantra     |
| ast.query   | CG      | ●       | —   | —   | Run Tree-sitter queries          | Yantra     |
| ast.symbols | CG      | ●       | —   | —   | Extract symbols (fallback)       | Yantra     |
| ast.scope   | CG      | ●       | —   | —   | Get scope at position            | Yantra     |
| ast.edit    | CG      | ●       | —   | —   | AST-aware code transformations   | Yantra     |
| ast.diff    | CG      | ●       | —   | —   | Structural diff between versions | Yantra     |

**Why Builtin Only:** Tree-sitter is a core differentiator. Fast, multi-language, works offline. Powers the dependency graph.

---

### 3.4 DEPENDENCY GRAPH (GNN)

| Capability            | Purpose | Builtin | MCP | LSP | Description                         | Maintainer |
| --------------------- | ------- | ------- | --- | --- | ----------------------------------- | ---------- |
| depgraph.build        | CG, MM  | ●       | —   | —   | Build full project dependency graph | Yantra     |
| depgraph.query        | CG      | ●       | —   | —   | Query graph relationships           | Yantra     |
| depgraph.dependents   | CG      | ●       | —   | —   | What depends on X?                  | Yantra     |
| depgraph.dependencies | CG      | ●       | —   | —   | What does X depend on?              | Yantra     |
| depgraph.impact       | CG, TS  | ●       | —   | —   | Impact analysis for changes         | Yantra     |
| depgraph.cycles       | MM      | ●       | —   | —   | Detect circular dependencies        | Yantra     |
| depgraph.modules      | CG      | ●       | —   | —   | Identify module boundaries          | Yantra     |
| depgraph.hotspots     | MM      | ●       | —   | —   | Find high-coupling areas            | Yantra     |
| depgraph.crossRepo    | CG      | ●       | —   | —   | Track cross-system dependencies     | Yantra     |

**Why Builtin Only:** This is Yantra's core differentiator. GNN-powered analysis that competitors don't have.

---

### 3.5 TERMINAL / SHELL

| Capability          | Purpose    | Builtin | MCP | LSP | Description                     | Maintainer |
| ------------------- | ---------- | ------- | --- | --- | ------------------------------- | ---------- |
| shell.exec          | CG, TS, DP | ●       | —   | —   | Execute command, get output     | Yantra     |
| shell.execStreaming | CG, TS, DP | ●       | —   | —   | Execute with real-time output   | Yantra     |
| shell.background    | DP, MM     | ●       | —   | —   | Start background process        | Yantra     |
| shell.kill          | MM         | ●       | —   | —   | Terminate process               | Yantra     |
| shell.interactive   | MM         | ●       | —   | —   | Pseudo-TTY for interactive CLIs | Yantra     |
| shell.env           | CG, DP     | ●       | —   | —   | Get/set environment variables   | Yantra     |

**Why Builtin Only:** Security critical. Must control what commands can run, implement approval queue.

---

### 3.6 VERSION CONTROL (Git)

| Capability    | Purpose | Builtin | MCP | LSP | Description              | Maintainer    |
| ------------- | ------- | ------- | --- | --- | ------------------------ | ------------- |
| git.status    | CG      | ○       | ●   | —   | Get current status       | MCP Community |
| git.diff      | CG      | ○       | ●   | —   | Get diff of changes      | MCP Community |
| git.log       | CG      | ○       | ●   | —   | Commit history           | MCP Community |
| git.blame     | CG      | ○       | ●   | —   | Line-by-line attribution | MCP Community |
| git.commit    | CG      | ○       | ●   | —   | Create commit            | MCP Community |
| git.branch    | CG      | ○       | ●   | —   | Branch operations        | MCP Community |
| git.checkout  | CG      | ○       | ●   | —   | Checkout files/branches  | MCP Community |
| git.merge     | CG      | ○       | ●   | —   | Merge branches           | MCP Community |
| git.stash     | CG      | ○       | ●   | —   | Stash operations         | MCP Community |
| git.reset     | CG      | ○       | ●   | —   | Undo changes             | MCP Community |
| git.push      | DP      | ○       | ●   | —   | Push to remote           | MCP Community |
| git.pull      | CG      | ○       | ●   | —   | Pull from remote         | MCP Community |
| git.clone     | CG      | ○       | ●   | —   | Clone repository         | MCP Community |
| git.conflicts | CG      | ○       | ●   | —   | Get conflict information | MCP Community |

**MCP Server:** `@modelcontextprotocol/server-git`

**Builtin Fallback:** Shell commands (`git status`, `git diff`, etc.) if MCP fails.

---

### 3.7 GITHUB / CODE HOSTING

| Capability      | Purpose | Builtin | MCP | LSP | Description              | Maintainer    |
| --------------- | ------- | ------- | --- | --- | ------------------------ | ------------- |
| github.repos    | CG      | —       | ●   | —   | List/search repositories | MCP Community |
| github.issues   | MM      | —       | ●   | —   | Manage issues            | MCP Community |
| github.prs      | CG      | —       | ●   | —   | Manage pull requests     | MCP Community |
| github.actions  | DP, MM  | —       | ●   | —   | View/trigger workflows   | MCP Community |
| github.releases | DP      | —       | ●   | —   | Manage releases          | MCP Community |
| github.gists    | CG      | —       | ●   | —   | Manage gists             | MCP Community |
| github.search   | CG      | —       | ●   | —   | Search code/issues/PRs   | MCP Community |

**MCP Server:** `@modelcontextprotocol/server-github`

---

### 3.8 DATABASE

| Capability | Purpose | Builtin | MCP | LSP | Description                | Maintainer    |
| ---------- | ------- | ------- | --- | --- | -------------------------- | ------------- |
| db.connect | CG, MM  | —       | ●   | —   | Establish connection       | MCP Community |
| db.query   | CG, MM  | —       | ●   | —   | Execute SELECT (read-only) | MCP Community |
| db.execute | CG      | —       | ●   | —   | Execute mutations          | MCP Community |
| db.schema  | CG      | —       | ●   | —   | Get schema information     | MCP Community |
| db.tables  | CG      | —       | ●   | —   | List tables                | MCP Community |
| db.explain | MM      | —       | ●   | —   | Query execution plan       | MCP Community |
| db.migrate | DP      | —       | ●   | —   | Run migrations             | MCP Community |
| db.seed    | TS      | —       | ●   | —   | Insert test data           | MCP Community |
| db.backup  | MM      | —       | ●   | —   | Backup database            | MCP Community |

**MCP Servers:**

- PostgreSQL: `@modelcontextprotocol/server-postgres`
- MySQL: `@modelcontextprotocol/server-mysql`
- SQLite: `@modelcontextprotocol/server-sqlite`
- MongoDB: `@modelcontextprotocol/server-mongodb`

---

### 3.9 TESTING

| Capability       | Purpose | Builtin | MCP | LSP | Description                     | Maintainer |
| ---------------- | ------- | ------- | --- | --- | ------------------------------- | ---------- |
| test.discover    | TS      | ●       | —   | —   | Find all tests in project       | Yantra     |
| test.run         | TS      | ●       | —   | —   | Run tests (file, suite, single) | Yantra     |
| test.runAffected | TS      | ●       | —   | —   | Run tests for changed code      | Yantra     |
| test.watch       | TS      | ●       | —   | —   | Continuous test runner          | Yantra     |
| test.coverage    | TS      | ●       | —   | —   | Get coverage report             | Yantra     |
| test.generate    | TS      | ●       | —   | —   | Auto-generate test cases        | Yantra     |
| test.debug       | TS      | ●       | ○   | —   | Run test in debug mode          | Yantra     |
| test.benchmark   | TS, MM  | ●       | —   | —   | Run performance benchmarks      | Yantra     |
| test.mutation    | TS      | ●       | —   | —   | Mutation testing                | Yantra     |
| e2e.run          | TS      | ●       | —   | —   | Run E2E/integration tests       | Yantra     |
| e2e.record       | TS      | ●       | —   | —   | Record browser interactions     | Yantra     |

**Why Builtin Primary:** Testing is core to "never breaks" guarantee. Must integrate with dependency graph for affected test detection.

---

### 3.10 BUILD & COMPILATION

| Capability        | Purpose | Builtin | MCP | LSP | Description                 | Maintainer |
| ----------------- | ------- | ------- | --- | --- | --------------------------- | ---------- |
| build.run         | DP      | ●       | —   | —   | Full build                  | Yantra     |
| build.incremental | DP      | ●       | —   | —   | Changed files only          | Yantra     |
| build.check       | CG      | ●       | —   | ○   | Type-check without emitting | Yantra     |
| build.clean       | DP      | ●       | —   | —   | Clear artifacts             | Yantra     |
| build.watch       | CG      | ●       | —   | —   | Watch mode                  | Yantra     |
| lint.run          | CG      | ●       | ○   | —   | Run linters                 | Yantra     |
| lint.fix          | CG      | ●       | ○   | —   | Auto-fix lint issues        | Yantra     |
| format.run        | CG      | ●       | ○   | ○   | Apply formatters            | Yantra     |

**Why Builtin Primary:** Build orchestration needs to coordinate with dep graph, testing, deployment.

---

### 3.11 PACKAGE MANAGEMENT

| Capability   | Purpose | Builtin | MCP | LSP | Description                  | Maintainer |
| ------------ | ------- | ------- | --- | --- | ---------------------------- | ---------- |
| pkg.install  | CG      | ●       | —   | —   | Add dependency               | Yantra     |
| pkg.remove   | CG      | ●       | —   | —   | Remove dependency            | Yantra     |
| pkg.update   | MM      | ●       | —   | —   | Update dependencies          | Yantra     |
| pkg.list     | CG      | ●       | —   | —   | List installed packages      | Yantra     |
| pkg.outdated | MM      | ●       | —   | —   | Check for updates            | Yantra     |
| pkg.audit    | MM      | ●       | ○   | —   | Security vulnerability check | Yantra     |
| pkg.search   | CG      | ●       | —   | —   | Search package registry      | Yantra     |
| pkg.info     | CG      | ●       | —   | —   | Get package details          | Yantra     |
| pkg.lockSync | CG      | ●       | —   | —   | Sync lockfile                | Yantra     |

**Why Builtin Primary:** Package operations need shell exec. Audit can use MCP for vulnerability databases.

---

### 3.12 DEBUGGING (DAP)

| Capability       | Purpose | Builtin | MCP | LSP | DAP | Description            | Maintainer   |
| ---------------- | ------- | ------- | --- | --- | --- | ---------------------- | ------------ |
| debug.launch     | TS, MM  | —       | —   | —   | ●   | Start debugger         | DAP Adapters |
| debug.attach     | MM      | —       | —   | —   | ●   | Attach to process      | DAP Adapters |
| debug.breakpoint | TS, MM  | —       | —   | —   | ●   | Set/remove breakpoints | DAP Adapters |
| debug.step       | TS, MM  | —       | —   | —   | ●   | Step over/into/out     | DAP Adapters |
| debug.continue   | TS, MM  | —       | —   | —   | ●   | Resume execution       | DAP Adapters |
| debug.pause      | MM      | —       | —   | —   | ●   | Pause execution        | DAP Adapters |
| debug.evaluate   | TS, MM  | —       | —   | —   | ●   | Evaluate expression    | DAP Adapters |
| debug.variables  | TS, MM  | —       | —   | —   | ●   | Inspect variables      | DAP Adapters |
| debug.stack      | TS, MM  | —       | —   | —   | ●   | Get call stack         | DAP Adapters |
| debug.threads    | MM      | —       | —   | —   | ●   | List threads           | DAP Adapters |

**Note:** DAP (Debug Adapter Protocol) is a separate protocol specifically for debugging. Agent uses DAP through a dedicated adapter.

**DAP Adapters:**

- Python: `debugpy`
- Node.js: `node-debug2`
- Rust: `codelldb`
- Go: `delve`

---

### 3.13 DEPLOYMENT

| Capability        | Purpose | Builtin | MCP | LSP | Description               | Maintainer   |
| ----------------- | ------- | ------- | --- | --- | ------------------------- | ------------ |
| deploy.preview    | DP      | —       | ●   | —   | Deploy to preview/staging | Platform MCP |
| deploy.production | DP      | —       | ●   | —   | Deploy to production      | Platform MCP |
| deploy.rollback   | DP      | —       | ●   | —   | Revert deployment         | Platform MCP |
| deploy.status     | DP, MM  | —       | ●   | —   | Check deployment state    | Platform MCP |
| deploy.logs       | MM      | —       | ●   | —   | Fetch deployment logs     | Platform MCP |
| deploy.scale      | DP      | —       | ●   | —   | Scale instances           | Platform MCP |
| deploy.env        | DP      | —       | ●   | —   | Manage env variables      | Platform MCP |
| infra.provision   | DP      | —       | ●   | —   | Create cloud resources    | Platform MCP |
| infra.destroy     | DP      | —       | ●   | —   | Destroy resources         | Platform MCP |
| container.build   | DP      | ●       | ○   | —   | Build Docker image        | Yantra       |
| container.push    | DP      | ●       | ○   | —   | Push to registry          | Yantra       |
| container.run     | DP, TS  | ●       | ○   | —   | Run container locally     | Yantra       |

**MCP Servers (Platform-specific):**

- Railway: `railway-mcp` (if available) or custom
- Vercel: `vercel-mcp`
- AWS: `@modelcontextprotocol/server-aws`
- GCP: Community server
- Azure: Community server

**Builtin:** Container operations via shell (docker CLI).

---

### 3.14 BROWSER AUTOMATION (CDP)

| Capability            | Purpose | Builtin | MCP | LSP | Description             | Maintainer |
| --------------------- | ------- | ------- | --- | --- | ----------------------- | ---------- |
| browser.launch        | TS, MM  | ●       | —   | —   | Start browser instance  | Yantra     |
| browser.close         | TS      | ●       | —   | —   | Close browser           | Yantra     |
| browser.navigate      | TS      | ●       | —   | —   | Go to URL               | Yantra     |
| browser.click         | TS      | ●       | —   | —   | Click element           | Yantra     |
| browser.type          | TS      | ●       | —   | —   | Input text              | Yantra     |
| browser.select        | TS      | ●       | —   | —   | Select dropdown option  | Yantra     |
| browser.screenshot    | TS, MM  | ●       | —   | —   | Capture screen          | Yantra     |
| browser.pdf           | TS      | ●       | —   | —   | Export to PDF           | Yantra     |
| browser.selectElement | CG      | ●       | —   | —   | Visual element picker   | Yantra     |
| browser.evaluate      | TS      | ●       | —   | —   | Run JS in page context  | Yantra     |
| browser.network       | TS, MM  | ●       | —   | —   | Intercept/mock requests | Yantra     |
| browser.console       | MM      | ●       | —   | —   | Get console logs        | Yantra     |
| browser.errors        | MM      | ●       | —   | —   | Get runtime errors      | Yantra     |
| browser.performance   | MM      | ●       | —   | —   | Performance metrics     | Yantra     |
| browser.accessibility | TS      | ●       | —   | —   | Accessibility audit     | Yantra     |

**Why Builtin Only:** CDP integration is core to Yantra's browser product. Must be fast and reliable.

---

### 3.15 HTTP / API

| Capability         | Purpose | Builtin | MCP | LSP | Description            | Maintainer |
| ------------------ | ------- | ------- | --- | --- | ---------------------- | ---------- |
| http.request       | CG, TS  | ●       | ○   | —   | Make HTTP calls        | Yantra     |
| http.graphql       | CG, TS  | ●       | —   | —   | GraphQL queries        | Yantra     |
| api.importSpec     | CG      | ●       | —   | —   | Import OpenAPI/Swagger | Yantra     |
| api.generateClient | CG      | ●       | —   | —   | Generate API client    | Yantra     |
| api.mock           | TS      | ●       | —   | —   | Create mock server     | Yantra     |
| api.test           | TS      | ●       | —   | —   | Test endpoint          | Yantra     |
| websocket.connect  | MM      | ●       | —   | —   | WebSocket client       | Yantra     |
| websocket.send     | MM      | ●       | —   | —   | Send message           | Yantra     |

**Why Builtin Primary:** HTTP is fundamental, needs to be fast and support all auth methods.

---

### 3.16 SECURITY

| Capability           | Purpose | Builtin | MCP | LSP | Description                | Maintainer      |
| -------------------- | ------- | ------- | --- | --- | -------------------------- | --------------- |
| security.scan        | TS, MM  | ●       | ○   | —   | SAST analysis              | Yantra          |
| security.secrets     | MM      | ●       | ○   | —   | Detect exposed credentials | Yantra          |
| security.deps        | MM      | ○       | ●   | —   | CVE check on dependencies  | Snyk MCP, etc.  |
| security.container   | MM      | —       | ●   | —   | Container image scanning   | Trivy MCP, etc. |
| security.permissions | MM      | ●       | —   | —   | Check file permissions     | Yantra          |
| security.audit       | MM      | ●       | —   | —   | Full security audit        | Yantra          |

**MCP Servers:**

- Snyk: Community MCP for vulnerability scanning
- Trivy: Container scanning

---

### 3.17 MONITORING & OBSERVABILITY

| Capability         | Purpose | Builtin | MCP | LSP | Description                      | Maintainer   |
| ------------------ | ------- | ------- | --- | --- | -------------------------------- | ------------ |
| logs.tail          | MM      | ●       | ○   | —   | Tail log files                   | Yantra       |
| logs.search        | MM      | ●       | ○   | —   | Search logs                      | Yantra       |
| logs.aggregate     | MM      | —       | ●   | —   | Query log aggregation service    | Platform MCP |
| metrics.query      | MM      | —       | ●   | —   | Query metrics (Prometheus, etc.) | Platform MCP |
| metrics.dashboard  | MM      | —       | ●   | —   | Get dashboard data               | Platform MCP |
| traces.query       | MM      | —       | ●   | —   | Query distributed traces         | Platform MCP |
| alerts.list        | MM      | —       | ●   | —   | List alerts                      | Platform MCP |
| alerts.acknowledge | MM      | —       | ●   | —   | Acknowledge alert                | Platform MCP |
| health.check       | MM      | ●       | ○   | —   | Service health checks            | Yantra       |
| uptime.status      | MM      | —       | ●   | —   | Uptime monitoring                | Platform MCP |

---

### 3.18 DOCUMENTATION

| Capability     | Purpose | Builtin | MCP | LSP | Description                      | Maintainer    |
| -------------- | ------- | ------- | --- | --- | -------------------------------- | ------------- |
| docs.generate  | CG      | ●       | —   | —   | Generate from code (JSDoc, etc.) | Yantra        |
| docs.search    | CG      | ○       | ●   | —   | Search project docs              | MCP Community |
| docs.external  | CG      | —       | ●   | —   | Fetch library documentation      | MCP Community |
| docs.readme    | CG      | ●       | —   | —   | Generate/update README           | Yantra        |
| docs.changelog | CG      | ●       | —   | —   | Generate changelog               | Yantra        |

---

### 3.19 ARCHITECTURE & VISUALIZATION

| Capability    | Purpose | Builtin | MCP | LSP | Description                      | Maintainer |
| ------------- | ------- | ------- | --- | --- | -------------------------------- | ---------- |
| arch.diagram  | CG, MM  | ●       | —   | —   | Generate architecture diagram    | Yantra     |
| arch.validate | MM      | ●       | —   | —   | Validate against constraints     | Yantra     |
| arch.suggest  | CG      | ●       | —   | —   | Suggest improvements             | Yantra     |
| arch.compare  | MM      | ●       | —   | —   | Compare versions                 | Yantra     |
| arch.export   | CG      | ●       | —   | —   | Export (Mermaid, PlantUML, etc.) | Yantra     |

**Why Builtin Only:** Tied to dependency graph, core differentiator.

---

### 3.20 CONTEXT & MEMORY

| Capability          | Purpose | Builtin | MCP | LSP | Description                         | Maintainer |
| ------------------- | ------- | ------- | --- | --- | ----------------------------------- | ---------- |
| context.add         | CG      | ●       | —   | —   | Add to agent working memory         | Yantra     |
| context.search      | CG      | ●       | —   | —   | Semantic search over codebase       | Yantra     |
| context.summarize   | CG      | ●       | —   | —   | Compress context (token efficiency) | Yantra     |
| context.conventions | CG      | ●       | —   | —   | Get project coding standards        | Yantra     |
| context.history     | CG      | ●       | —   | —   | Get relevant past interactions      | Yantra     |
| embeddings.generate | CG      | ●       | —   | —   | Generate code embeddings            | Yantra     |
| embeddings.search   | CG      | ●       | —   | —   | Vector similarity search            | Yantra     |

---

### 3.21 COLLABORATION & NOTIFICATIONS

| Capability    | Purpose | Builtin | MCP | LSP | Description          | Maintainer    |
| ------------- | ------- | ------- | --- | --- | -------------------- | ------------- |
| slack.send    | MM      | —       | ●   | —   | Send Slack message   | MCP Community |
| slack.search  | MM      | —       | ●   | —   | Search Slack history | MCP Community |
| email.send    | MM      | —       | ●   | —   | Send email           | MCP Community |
| notion.query  | CG      | —       | ●   | —   | Query Notion pages   | MCP Community |
| notion.update | CG      | —       | ●   | —   | Update Notion        | MCP Community |
| linear.issues | MM      | —       | ●   | —   | Manage Linear issues | MCP Community |
| jira.issues   | MM      | —       | ●   | —   | Manage Jira issues   | MCP Community |

**MCP Servers:**

- Slack: `@modelcontextprotocol/server-slack`
- Notion: Community server
- Linear: Community server
- Jira: Community server

---

## 4. Summary by Purpose

### 4.1 Code Generation (CG)

| Category            | Key Capabilities                        | Primary Protocol       |
| ------------------- | --------------------------------------- | ---------------------- |
| File Operations     | read, write, edit, search               | Builtin                |
| Code Intelligence   | symbols, definition, completion, rename | MCP (Language servers) |
| AST/Parsing         | parse, query, transform                 | Builtin (Tree-sitter)  |
| Dependency Analysis | graph, impact, modules                  | Builtin (GNN)          |
| Version Control     | status, diff, commit, branch            | MCP (Git)              |
| Documentation       | generate, search                        | Builtin + MCP          |

### 4.2 Testing (TS)

| Category         | Key Capabilities            | Primary Protocol         |
| ---------------- | --------------------------- | ------------------------ |
| Test Execution   | run, watch, coverage        | Builtin                  |
| Smart Testing    | affected tests, mutation    | Builtin (uses dep graph) |
| E2E Testing      | browser automation          | Builtin (CDP)            |
| Debugging        | breakpoints, step, evaluate | DAP                      |
| Test Data        | DB seeding, mocking         | MCP + Builtin            |
| Security Testing | SAST, dependency audit      | Builtin + MCP            |

### 4.3 Deployment (DP)

| Category       | Key Capabilities              | Primary Protocol        |
| -------------- | ----------------------------- | ----------------------- |
| Build          | compile, bundle, check        | Builtin                 |
| Containers     | build, push, run              | Builtin (shell)         |
| Cloud Deploy   | preview, production, rollback | MCP (Platform-specific) |
| Infrastructure | provision, destroy, scale     | MCP (Platform-specific) |
| Environment    | env vars, secrets             | MCP                     |

### 4.4 Monitor/Maintain (MM)

| Category           | Key Capabilities             | Primary Protocol |
| ------------------ | ---------------------------- | ---------------- |
| Logging            | tail, search, aggregate      | Builtin + MCP    |
| Metrics            | query, dashboard             | MCP              |
| Debugging          | attach, evaluate, stack      | DAP              |
| Health             | checks, uptime               | Builtin + MCP    |
| Security           | scan, audit, CVE check       | Builtin + MCP    |
| Alerts             | list, acknowledge            | MCP              |
| Browser Monitoring | console, errors, performance | Builtin (CDP)    |

### 4.5 Visualization (VZ) — NEW

| Category               | Key Capabilities                          | Primary Protocol         |
| ---------------------- | ----------------------------------------- | ------------------------ |
| **Diagrams**           | mermaid, graphviz, plantuml               | Builtin + Shell          |
| **Code Visualization** | depgraph, callgraph, typegraph, ast       | Builtin                  |
| **Data Charts**        | line, bar, pie, scatter, heatmap, treemap | Builtin (Plotly/ECharts) |
| **Tables**             | interactive tables, pivot, matrix         | Builtin                  |
| **Temporal**           | timeline, gantt, calendar                 | Builtin                  |
| **Comparison**         | diff, schemaDiff, compare                 | Builtin                  |
| **Interactive**        | network, force graph, json explorer       | Builtin                  |
| **Export**             | SVG, PNG, PDF, HTML, React                | Builtin + Shell          |
| **Python/R**           | matplotlib, seaborn, ggplot, bokeh        | Shell (wrapped)          |

**Visualization serves ALL purposes:**

| Purpose | Visualization Use Cases                                                       |
| ------- | ----------------------------------------------------------------------------- |
| **CG**  | Architecture diagrams, dependency graphs, diff views, implementation planning |
| **TS**  | Test coverage heatmaps, performance charts, failure timelines                 |
| **DP**  | Deployment timelines, infrastructure diagrams, resource usage charts          |
| **MM**  | Metrics dashboards, log analysis, alert timelines, performance monitoring     |

---

## 5. Updated Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  YANTRA                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         MONACO EDITOR (UI)                              ││
│  │                               │                                         ││
│  │                          LSP Client                                     ││
│  │                               │                                         ││
│  │              ┌────────────────┼────────────────┐                        ││
│  │              ▼                ▼                ▼                        ││
│  │        Pylance (LSP)   rust-analyzer     tsserver                       ││
│  │                                                                         ││
│  │        [Real-time: autocomplete, hover, diagnostics-as-you-type]        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                            AI AGENT                                     ││
│  │                               │                                         ││
│  │                  ┌────────────┴────────────┐                            ││
│  │                  ▼                         ▼                            ││
│  │        AgentToolkit API            Approval Queue                       ││
│  │                  │                         │                            ││
│  │                  ▼                         ▼                            ││
│  │  ┌──────────────────────────────────────────────────────────────────┐  ││
│  │  │              UNIFIED TOOL INTERFACE (UTI)                        │  ││
│  │  ├──────────────────────────────────────────────────────────────────┤  ││
│  │  │  Registry │ Router │ Executor │ Monitor │ Security               │  ││
│  │  └──────────────────────────────────────────────────────────────────┘  ││
│  │                               │                                         ││
│  │           ┌───────────────────┼───────────────────┐                     ││
│  │           ▼                   ▼                   ▼                     ││
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           ││
│  │  │ BUILTIN ADAPTER │ │  MCP ADAPTER    │ │  DAP ADAPTER    │           ││
│  │  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤           ││
│  │  │ • File Ops      │ │ • Pylance (MCP) │ │ • debugpy       │           ││
│  │  │ • Terminal      │ │ • Git           │ │ • node-debug    │           ││
│  │  │ • Tree-sitter   │ │ • GitHub        │ │ • codelldb      │           ││
│  │  │ • Dep Graph     │ │ • Postgres      │ │ • delve         │           ││
│  │  │ • Browser (CDP) │ │ • Railway       │ └─────────────────┘           ││
│  │  │ • HTTP          │ │ • Slack         │                               ││
│  │  │ • Testing       │ │ • Snyk          │                               ││
│  │  │ • Build         │ │ • Notion        │                               ││
│  │  │ • Context/RAG   │ │ • Prometheus    │                               ││
│  │  │ • VISUALIZATION │ │                 │                               ││
│  │  └─────────────────┘ └─────────────────┘                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Tool Count Summary

| Protocol              | Tool Count  | Maintained By                         |
| --------------------- | ----------- | ------------------------------------- |
| **Builtin**           | ~95 tools   | Yantra team (includes ~30 viz tools)  |
| **MCP**               | ~45 tools   | Community + Vendors                   |
| **DAP**               | ~10 tools   | Debug adapter maintainers             |
| **Shell (wrapped)**   | ~10 tools   | External tools (Graphviz, matplotlib) |
| **LSP (Editor only)** | ~15 methods | Language server maintainers           |

**Total Agent-accessible tools: ~160**

---

## 7. Core Design Principles (Updated)

| Principle                | Description                                                                   |
| ------------------------ | ----------------------------------------------------------------------------- |
| **MCP-First for Agent**  | Agent uses MCP for language intelligence, not LSP                             |
| **LSP for Editor Only**  | LSP powers Monaco real-time features, not exposed to agent                    |
| **Builtin for Core**     | Differentiators (dep graph, tree-sitter, CDP,**visualization** ) are built-in |
| **Builtin for Security** | Shell, file ops under our control for safety                                  |
| **MCP for Ecosystem**    | Leverage community for Git, DB, platforms                                     |
| **DAP for Debugging**    | Dedicated protocol for debug operations                                       |
| **Fallback Chains**      | Every capability has backup (MCP → Builtin → Shell)                           |
| **Viz-First Analysis**   | Agent shows visualizations inline before/during code generation               |
| **Multi-Output Viz**     | Every visualization can render to chat, panel, or file                        |

---

## 8. Implementation Priority

### Phase 1: Foundation (Weeks 1-4)

- [ ] UTI core: Registry, Router, Executor
- [ ] Builtin Adapter: File ops, Terminal, Tree-sitter
- [ ] Basic MCP Adapter: Git
- [ ] Approval Queue
- [ ] **Basic Visualization: Mermaid, Tables, JSON explorer (inline chat)**

### Phase 2: Code Intelligence (Weeks 5-8)

- [ ] MCP integration: Pylance, tsserver MCPs
- [ ] Builtin: Dependency graph (GNN)
- [ ] Builtin fallback: Tree-sitter symbols
- [ ] **Visualization: Dependency graph, Call graph, AST tree rendering**

### Phase 3: Testing & Debug (Weeks 9-12)

- [ ] Builtin: Test runner, coverage
- [ ] Builtin: Browser automation (CDP)
- [ ] DAP Adapter: Python, Node.js debugging
- [ ] **Visualization: Charts (Plotly/ECharts), Diff view, Timeline**

### Phase 4: Deployment & Monitoring (Weeks 13-16)

- [ ] MCP: Database servers
- [ ] MCP: Platform deployment (Railway)
- [ ] MCP: Monitoring integrations
- [ ] Builtin: Container operations
- [ ] **Visualization: Heatmaps, Network graphs, Git graphs**

### Phase 5: Advanced Viz & Export (Weeks 17-20)

- [ ] **Export: PNG, PDF, HTML, React component generation**
- [ ] **Python integration: matplotlib, seaborn, plotly**
- [ ] **External tools: Graphviz, PlantUML, ggplot**
- [ ] Visualization Panel with tabs and full interactivity
- [ ] Plugin system for custom visualizations

---

## 9. Configuration Schema

```yaml
# yantra.tools.yaml

# Editor configuration (LSP for Monaco)
editor:
  lsp:
    python:
      command: 'pylance-langserver'
      args: ['--stdio']
    rust:
      command: 'rust-analyzer'
    typescript:
      command: 'typescript-language-server'
      args: ['--stdio']

# Agent configuration (UTI)
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
      # Code Intelligence (via MCP, not LSP)
      pylance:
        enabled: true # Uses Pylance's MCP interface

      # Version Control
      git:
        package: '@modelcontextprotocol/server-git'

      github:
        package: '@modelcontextprotocol/server-github'
        config:
          token: '${GITHUB_TOKEN}'

      # Database
      postgres:
        package: '@modelcontextprotocol/server-postgres'
        config:
          connectionString: '${DATABASE_URL}'

      # Deployment
      railway:
        package: 'railway-mcp'
        config:
          token: '${RAILWAY_TOKEN}'

  dap:
    enabled: true
    adapters:
      python:
        command: 'python'
        args: ['-m', 'debugpy.adapter']
      node:
        command: 'node'
        args: ['node_modules/node-debug2/out/src/nodeDebug.js']

# Security
security:
  approvalRequired:
    - 'builtin.shell.*'
    - 'builtin.file.write'
    - 'builtin.file.delete'
    - 'mcp.git.commit'
    - 'mcp.git.push'
    - 'mcp.db.execute'
    - 'mcp.deploy.*'

  autoApprove:
    - pattern: 'builtin.file.read'
    - pattern: 'mcp.git.status'
    - pattern: 'builtin.test.run'

  blocked:
    - pattern: 'DROP TABLE'
      tools: ['mcp.db.*']
    - pattern: 'rm -rf /'
      tools: ['builtin.shell.*']
```

---

## 10. Visualization System

Yantra is not just a code IDE — it's a full development and data analysis platform. The agent must be able to create visualizations for ad-hoc analysis, show them inline in chat, and export them as files.

### 10.1 Visualization Modes

| Mode            | Description                           | Use Case                         |
| --------------- | ------------------------------------- | -------------------------------- |
| **Inline Chat** | Render directly in conversation       | Quick analysis, exploration      |
| **Panel View**  | Open in dedicated visualization panel | Detailed inspection, interaction |
| **File Export** | Save as file (SVG, PNG, PDF, HTML)    | Reports, sharing, documentation  |

### 10.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YANTRA UI                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          CHAT PANEL                                    │ │
│  │                                                                        │ │
│  │  User: "Show me the login trends for the last month"                   │ │
│  │                                                                        │ │
│  │  Agent: "Here's the login trend analysis:"                             │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │  📊 INLINE VISUALIZATION                                         │ │ │
│  │  │  ┌────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  │                    ╭─────╮                                 │  │ │ │
│  │  │  │               ╭───╯     ╰───╮                              │  │ │ │
│  │  │  │          ╭───╯              ╰──╮                           │  │ │ │
│  │  │  │     ╭───╯                      ╰───╮                       │  │ │ │
│  │  │  │  ───╯                              ╰───                    │  │ │ │
│  │  │  │  Nov 1    Nov 8    Nov 15    Nov 22    Nov 29              │  │ │ │
│  │  │  └────────────────────────────────────────────────────────────┘  │ │ │
│  │  │  [🔍 Expand] [💾 Save as PNG] [📋 Copy SVG] [📁 Open in Panel]   │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  Agent: "Peak activity was Nov 15 with 2,340 logins..."               │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     VISUALIZATION PANEL (Optional)                     │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ [Tab: Dep Graph] [Tab: Query Results] [Tab: Login Trends] ←active│ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │                                                                  │ │ │
│  │  │     Full interactive visualization with zoom, pan, filters       │ │ │
│  │  │                                                                  │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Visualization Capability Matrix

| Capability               | Purpose | Builtin | MCP | Shell | Description                                    |
| ------------------------ | ------- | ------- | --- | ----- | ---------------------------------------------- |
| **DIAGRAMS**             |         |         |     |       |                                                |
| viz.mermaid              | CG, MM  | ●       | —   | —     | Flowcharts, sequence, ERD, class, state, gantt |
| viz.graphviz             | CG, MM  | ○       | —   | ●     | DOT language graphs (complex layouts)          |
| viz.plantuml             | CG      | —       | ○   | ●     | UML diagrams                                   |
| viz.excalidraw           | CG      | ●       | —   | —     | Hand-drawn style diagrams                      |
| **CODE VISUALIZATION**   |         |         |     |       |                                                |
| viz.depgraph             | CG, MM  | ●       | —   | —     | Dependency graph (uses GNN)                    |
| viz.callgraph            | CG      | ●       | —   | —     | Function call hierarchy                        |
| viz.typegraph            | CG      | ●       | —   | —     | Type/class hierarchy                           |
| viz.astTree              | CG      | ●       | —   | —     | AST visualization                              |
| viz.gitGraph             | CG      | ●       | —   | —     | Git branch visualization                       |
| **DATA CHARTS**          |         |         |     |       |                                                |
| viz.chart                | MM      | ●       | —   | —     | Line, bar, pie, scatter, area                  |
| viz.histogram            | MM      | ●       | —   | —     | Distribution visualization                     |
| viz.heatmap              | MM      | ●       | —   | —     | 2D density visualization                       |
| viz.boxplot              | MM      | ●       | —   | —     | Statistical distribution                       |
| viz.treemap              | MM      | ●       | —   | —     | Hierarchical data                              |
| viz.sankey               | MM      | ●       | —   | —     | Flow visualization                             |
| viz.radar                | MM      | ●       | —   | —     | Multi-dimensional comparison                   |
| viz.candlestick          | MM      | ●       | —   | —     | Financial data                                 |
| viz.geo                  | MM      | ●       | —   | —     | Geographic/map visualizations                  |
| **TABLES & GRIDS**       |         |         |     |       |                                                |
| viz.table                | MM      | ●       | —   | —     | Interactive sortable table                     |
| viz.pivot                | MM      | ●       | —   | —     | Pivot table analysis                           |
| viz.matrix               | MM      | ●       | —   | —     | Matrix/correlation view                        |
| **TEMPORAL**             |         |         |     |       |                                                |
| viz.timeline             | MM      | ●       | —   | —     | Events over time                               |
| viz.gantt                | CG      | ●       | —   | —     | Project timeline (via Mermaid)                 |
| viz.calendar             | MM      | ●       | —   | —     | Calendar heatmap                               |
| **COMPARISON**           |         |         |     |       |                                                |
| viz.diff                 | CG      | ●       | —   | —     | Code/text diff view                            |
| viz.schemaDiff           | CG      | ●       | —   | —     | Database schema diff                           |
| viz.compare              | MM      | ●       | —   | —     | Side-by-side comparison                        |
| **INTERACTIVE**          |         |         |     |       |                                                |
| viz.explorer             | MM      | ●       | —   | —     | JSON/object tree explorer                      |
| viz.network              | MM      | ●       | —   | —     | Interactive network graph                      |
| viz.force                | MM      | ●       | —   | —     | Force-directed graph                           |
| **EXPORT/RENDER**        |         |         |     |       |                                                |
| viz.toSvg                | CG      | ●       | —   | —     | Export any viz to SVG                          |
| viz.toPng                | CG      | ●       | —   | ○     | Export to PNG (via Puppeteer/CDP)              |
| viz.toPdf                | CG      | ●       | —   | ○     | Export to PDF                                  |
| viz.toHtml               | CG      | ●       | —   | —     | Export as standalone HTML                      |
| viz.toReact              | CG      | ●       | —   | —     | Generate React component                       |
| **PYTHON/R INTEGRATION** |         |         |     |       |                                                |
| viz.matplotlib           | MM      | —       | —   | ●     | Python matplotlib plots                        |
| viz.seaborn              | MM      | —       | —   | ●     | Python seaborn statistical plots               |
| viz.plotly               | MM      | ●       | —   | ○     | Plotly.js or Python plotly                     |
| viz.ggplot               | MM      | —       | —   | ●     | R ggplot2 visualizations                       |
| viz.bokeh                | MM      | —       | —   | ●     | Python Bokeh interactive plots                 |

### 10.4 Visualization Tool Definitions

#### 10.4.1 Inline Chat Rendering (Option 1)

```typescript
// Core inline visualization tools
const inlineVizTools: Tool[] = [
  {
    id: 'builtin.viz.render',
    name: 'Render Visualization',
    description: 'Render visualization inline in chat or in panel',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        type: {
          enum: [
            'mermaid',
            'chart',
            'table',
            'diff',
            'json',
            'markdown',
            'depgraph',
            'callgraph',
            'timeline',
            'network',
            'heatmap',
          ],
        },
        spec: { type: 'object', description: 'Visualization specification' },
        target: {
          enum: ['chat', 'panel', 'both'],
          default: 'chat',
          description: 'Where to render',
        },
        title: { type: 'string' },
        width: { type: 'integer', description: 'Width in pixels' },
        height: { type: 'integer', description: 'Height in pixels' },
        interactive: { type: 'boolean', default: true },
      },
      required: ['type', 'spec'],
    },
    outputSchema: {
      type: 'object',
      properties: {
        vizId: { type: 'string', description: 'ID for referencing this viz' },
        rendered: { type: 'boolean' },
        panelId: { type: 'string', description: 'Panel ID if opened in panel' },
      },
    },
  },

  // Mermaid diagrams
  {
    id: 'builtin.viz.mermaid',
    name: 'Mermaid Diagram',
    description: 'Create flowchart, sequence, ERD, class, state, or gantt diagram',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'Mermaid diagram code' },
        theme: { enum: ['default', 'dark', 'forest', 'neutral'], default: 'default' },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['code'],
    },
  },

  // Charts
  {
    id: 'builtin.viz.chart',
    name: 'Create Chart',
    description: 'Create interactive chart from data',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        type: {
          enum: [
            'line',
            'bar',
            'pie',
            'scatter',
            'area',
            'histogram',
            'boxplot',
            'heatmap',
            'radar',
            'treemap',
            'sankey',
            'candlestick',
          ],
        },
        data: {
          type: 'array',
          description: 'Array of data points or objects',
        },
        x: { type: 'string', description: 'X-axis field name' },
        y: { type: 'string', description: 'Y-axis field or array of fields' },
        color: { type: 'string', description: 'Color/category field' },
        title: { type: 'string' },
        xLabel: { type: 'string' },
        yLabel: { type: 'string' },
        legend: { type: 'boolean', default: true },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['type', 'data'],
    },
  },

  // Data table
  {
    id: 'builtin.viz.table',
    name: 'Data Table',
    description: 'Interactive sortable, filterable data table',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        data: { type: 'array', description: 'Array of row objects' },
        columns: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              key: { type: 'string' },
              label: { type: 'string' },
              sortable: { type: 'boolean' },
              format: { enum: ['text', 'number', 'date', 'currency', 'percent'] },
            },
          },
        },
        sortable: { type: 'boolean', default: true },
        filterable: { type: 'boolean', default: true },
        pagination: { type: 'boolean', default: true },
        pageSize: { type: 'integer', default: 20 },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['data'],
    },
  },

  // Dependency graph
  {
    id: 'builtin.viz.depgraph',
    name: 'Dependency Graph',
    description: 'Visualize code dependencies from GNN analysis',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        target: { type: 'string', description: 'File, folder, or symbol' },
        direction: { enum: ['dependents', 'dependencies', 'both'], default: 'both' },
        depth: { type: 'integer', default: 3 },
        layout: { enum: ['hierarchical', 'force', 'radial', 'dagre'], default: 'dagre' },
        highlight: { type: 'array', items: { type: 'string' }, description: 'Nodes to highlight' },
        filter: { type: 'string', description: 'Filter pattern' },
        target: { enum: ['chat', 'panel', 'both'], default: 'panel' },
      },
      required: ['target'],
    },
  },

  // Call graph
  {
    id: 'builtin.viz.callgraph',
    name: 'Call Graph',
    description: 'Visualize function call relationships',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        function: { type: 'string', description: 'Function name or path:function' },
        direction: { enum: ['incoming', 'outgoing', 'both'], default: 'both' },
        depth: { type: 'integer', default: 3 },
        includeExternal: { type: 'boolean', default: false },
        target: { enum: ['chat', 'panel', 'both'], default: 'panel' },
      },
      required: ['function'],
    },
  },

  // Diff view
  {
    id: 'builtin.viz.diff',
    name: 'Diff View',
    description: 'Side-by-side or unified diff visualization',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        left: { type: 'string', description: 'Left content, file path, or git ref' },
        right: { type: 'string', description: 'Right content, file path, or git ref' },
        mode: { enum: ['sideBySide', 'unified', 'inline'], default: 'sideBySide' },
        language: { type: 'string', description: 'Syntax highlighting language' },
        wordWrap: { type: 'boolean', default: false },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['left', 'right'],
    },
  },

  // Timeline
  {
    id: 'builtin.viz.timeline',
    name: 'Timeline',
    description: 'Visualize events over time',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        events: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              timestamp: { type: 'string', description: 'ISO timestamp' },
              label: { type: 'string' },
              category: { type: 'string' },
              description: { type: 'string' },
              duration: { type: 'integer', description: 'Duration in ms' },
            },
            required: ['timestamp', 'label'],
          },
        },
        groupBy: { type: 'string', description: 'Field to group by' },
        scale: { enum: ['minute', 'hour', 'day', 'week', 'month', 'year'], default: 'day' },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['events'],
    },
  },

  // Network/Graph
  {
    id: 'builtin.viz.network',
    name: 'Network Graph',
    description: 'Interactive network/graph visualization',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        nodes: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              label: { type: 'string' },
              group: { type: 'string' },
              size: { type: 'number' },
            },
            required: ['id'],
          },
        },
        edges: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              source: { type: 'string' },
              target: { type: 'string' },
              label: { type: 'string' },
              weight: { type: 'number' },
            },
            required: ['source', 'target'],
          },
        },
        layout: { enum: ['force', 'hierarchical', 'radial', 'circular'], default: 'force' },
        directed: { type: 'boolean', default: true },
        target: { enum: ['chat', 'panel', 'both'], default: 'panel' },
      },
      required: ['nodes', 'edges'],
    },
  },

  // JSON Explorer
  {
    id: 'builtin.viz.json',
    name: 'JSON Explorer',
    description: 'Interactive tree view for JSON/objects',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        data: { type: ['object', 'array'] },
        expandDepth: { type: 'integer', default: 2 },
        searchable: { type: 'boolean', default: true },
        copyable: { type: 'boolean', default: true },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['data'],
    },
  },

  // Heatmap
  {
    id: 'builtin.viz.heatmap',
    name: 'Heatmap',
    description: '2D heatmap visualization',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          description: '2D array or array of {x, y, value} objects',
        },
        xLabels: { type: 'array', items: { type: 'string' } },
        yLabels: { type: 'array', items: { type: 'string' } },
        colorScale: { enum: ['viridis', 'plasma', 'blues', 'reds', 'greens'], default: 'viridis' },
        showValues: { type: 'boolean', default: true },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['data'],
    },
  },

  // Git graph
  {
    id: 'builtin.viz.gitGraph',
    name: 'Git Graph',
    description: 'Visualize git branch history',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        branch: { type: 'string', description: 'Branch to visualize (default: current)' },
        maxCommits: { type: 'integer', default: 50 },
        showAll: { type: 'boolean', default: false, description: 'Show all branches' },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
    },
  },

  // Markdown with embedded visualizations
  {
    id: 'builtin.viz.markdown',
    name: 'Rich Markdown',
    description: 'Render markdown with embedded diagrams, math, and code',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        content: { type: 'string' },
        renderMermaid: { type: 'boolean', default: true },
        renderMath: { type: 'boolean', default: true },
        renderCode: { type: 'boolean', default: true },
        target: { enum: ['chat', 'panel', 'both'], default: 'chat' },
      },
      required: ['content'],
    },
  },
];
```

#### 10.4.2 File Export (Option 2)

```typescript
const exportVizTools: Tool[] = [
  {
    id: 'builtin.viz.export',
    name: 'Export Visualization',
    description: 'Export visualization to file',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        vizId: { type: 'string', description: 'ID of rendered visualization' },
        format: { enum: ['svg', 'png', 'pdf', 'html', 'json'] },
        path: { type: 'string', description: 'Output file path' },
        width: { type: 'integer', description: 'Width in pixels (for raster)' },
        height: { type: 'integer', description: 'Height in pixels (for raster)' },
        scale: { type: 'number', default: 2, description: 'Scale factor for PNG' },
        background: { type: 'string', default: 'white' },
      },
      required: ['vizId', 'format', 'path'],
    },
    outputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string' },
        size: { type: 'integer' },
        format: { type: 'string' },
      },
    },
    sideEffects: true,
  },

  // Direct SVG generation
  {
    id: 'builtin.viz.toSvg',
    name: 'Generate SVG',
    description: 'Generate SVG from visualization spec',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        type: { enum: ['mermaid', 'chart', 'network', 'depgraph', 'callgraph'] },
        spec: { type: 'object' },
        path: {
          type: 'string',
          description: 'Output path (optional, returns content if not provided)',
        },
      },
      required: ['type', 'spec'],
    },
    outputSchema: {
      type: 'object',
      properties: {
        svg: { type: 'string', description: 'SVG content if no path provided' },
        path: { type: 'string', description: 'File path if saved' },
      },
    },
  },

  // PNG generation (via CDP/Puppeteer)
  {
    id: 'builtin.viz.toPng',
    name: 'Generate PNG',
    description: 'Render visualization to PNG image',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        type: { enum: ['mermaid', 'chart', 'network', 'depgraph', 'callgraph', 'html'] },
        spec: { type: 'object' },
        path: { type: 'string' },
        width: { type: 'integer', default: 1200 },
        height: { type: 'integer', default: 800 },
        scale: { type: 'number', default: 2, description: 'Retina scale' },
        background: { type: 'string', default: 'white' },
      },
      required: ['type', 'spec', 'path'],
    },
    sideEffects: true,
  },

  // PDF generation
  {
    id: 'builtin.viz.toPdf',
    name: 'Generate PDF',
    description: 'Export visualization or report to PDF',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        content: {
          type: ['string', 'array'],
          description: 'HTML/Markdown content or array of vizIds',
        },
        path: { type: 'string' },
        pageSize: { enum: ['A4', 'Letter', 'Legal'], default: 'A4' },
        orientation: { enum: ['portrait', 'landscape'], default: 'portrait' },
        margins: {
          type: 'object',
          properties: {
            top: { type: 'string', default: '1cm' },
            bottom: { type: 'string', default: '1cm' },
            left: { type: 'string', default: '1cm' },
            right: { type: 'string', default: '1cm' },
          },
        },
        header: { type: 'string' },
        footer: { type: 'string' },
      },
      required: ['content', 'path'],
    },
    sideEffects: true,
  },

  // Standalone HTML
  {
    id: 'builtin.viz.toHtml',
    name: 'Generate HTML',
    description: 'Export as standalone HTML file with embedded visualization',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        vizId: { type: 'string', description: 'Visualization ID' },
        path: { type: 'string' },
        title: { type: 'string' },
        includeInteractivity: { type: 'boolean', default: true },
        selfContained: { type: 'boolean', default: true, description: 'Inline all assets' },
      },
      required: ['vizId', 'path'],
    },
    sideEffects: true,
  },

  // React component generation
  {
    id: 'builtin.viz.toReact',
    name: 'Generate React Component',
    description: 'Generate reusable React component from visualization',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        vizId: { type: 'string' },
        path: { type: 'string' },
        componentName: { type: 'string' },
        typescript: { type: 'boolean', default: true },
        includeData: {
          type: 'boolean',
          default: false,
          description: 'Embed data or accept as prop',
        },
      },
      required: ['vizId', 'path', 'componentName'],
    },
    sideEffects: true,
  },
];
```

#### 10.4.3 External Tools via Shell/MCP (Option 3)

```typescript
const externalVizTools: Tool[] = [
  // Graphviz
  {
    id: 'builtin.viz.graphviz',
    name: 'Graphviz',
    description: 'Generate graph using Graphviz DOT language',
    category: 'viz',
    protocol: 'builtin', // Wraps shell
    inputSchema: {
      type: 'object',
      properties: {
        dot: { type: 'string', description: 'DOT language graph definition' },
        engine: { enum: ['dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi'], default: 'dot' },
        format: { enum: ['svg', 'png', 'pdf'], default: 'svg' },
        path: { type: 'string', description: 'Output path (optional for inline)' },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
      },
      required: ['dot'],
    },
  },

  // PlantUML
  {
    id: 'builtin.viz.plantuml',
    name: 'PlantUML',
    description: 'Generate UML diagrams using PlantUML',
    category: 'viz',
    protocol: 'builtin', // Wraps shell or MCP
    inputSchema: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'PlantUML code' },
        format: { enum: ['svg', 'png'], default: 'svg' },
        path: { type: 'string' },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
      },
      required: ['code'],
    },
  },

  // Python Matplotlib
  {
    id: 'builtin.viz.matplotlib',
    name: 'Matplotlib Plot',
    description: 'Execute Python matplotlib code and capture output',
    category: 'viz',
    protocol: 'builtin', // Wraps shell
    inputSchema: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'Python code that uses matplotlib' },
        data: { type: 'object', description: 'Data to pass to script as JSON' },
        format: { enum: ['png', 'svg', 'pdf'], default: 'png' },
        path: { type: 'string', description: 'Output path' },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
        dpi: { type: 'integer', default: 150 },
        figsize: {
          type: 'array',
          items: { type: 'number' },
          default: [10, 6],
          description: '[width, height] in inches',
        },
      },
      required: ['code'],
    },
  },

  // Python Seaborn
  {
    id: 'builtin.viz.seaborn',
    name: 'Seaborn Plot',
    description: 'Create statistical visualizations with Seaborn',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        plotType: {
          enum: [
            'scatter',
            'line',
            'bar',
            'box',
            'violin',
            'heatmap',
            'pairplot',
            'jointplot',
            'regplot',
            'catplot',
          ],
        },
        data: { type: 'array', description: 'Data as array of objects' },
        x: { type: 'string' },
        y: { type: 'string' },
        hue: { type: 'string' },
        style: { enum: ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'], default: 'whitegrid' },
        format: { enum: ['png', 'svg'], default: 'png' },
        path: { type: 'string' },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
      },
      required: ['plotType', 'data'],
    },
  },

  // Python Plotly (can also use builtin.viz.chart with plotly backend)
  {
    id: 'builtin.viz.plotlyPython',
    name: 'Plotly (Python)',
    description: 'Create interactive Plotly visualizations via Python',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'Python code using plotly' },
        data: { type: 'object' },
        format: { enum: ['html', 'png', 'svg', 'json'], default: 'html' },
        path: { type: 'string' },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
      },
      required: ['code'],
    },
  },

  // R ggplot2
  {
    id: 'builtin.viz.ggplot',
    name: 'ggplot2',
    description: 'Create R ggplot2 visualizations',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'R code using ggplot2' },
        data: { type: 'array', description: 'Data to pass to R script' },
        format: { enum: ['png', 'svg', 'pdf'], default: 'png' },
        path: { type: 'string' },
        width: { type: 'number', default: 10 },
        height: { type: 'number', default: 6 },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
      },
      required: ['code'],
    },
  },

  // Vega-Lite (declarative)
  {
    id: 'builtin.viz.vegalite',
    name: 'Vega-Lite',
    description: 'Create visualization from Vega-Lite JSON spec',
    category: 'viz',
    protocol: 'builtin',
    inputSchema: {
      type: 'object',
      properties: {
        spec: { type: 'object', description: 'Vega-Lite specification' },
        format: { enum: ['svg', 'png', 'canvas'], default: 'svg' },
        path: { type: 'string' },
        target: { enum: ['chat', 'panel', 'file'], default: 'chat' },
      },
      required: ['spec'],
    },
  },
];
```

### 10.5 Visualization Library Stack

| Category           | Primary Library     | Fallback             | Use Case                    |
| ------------------ | ------------------- | -------------------- | --------------------------- |
| **Diagrams**       | Mermaid.js          | Graphviz (shell)     | Flowcharts, sequence, ERD   |
| **Charts**         | Plotly.js           | ECharts, Chart.js    | All chart types             |
| **Tables**         | TanStack Table      | AG Grid              | Interactive data tables     |
| **Networks**       | React Flow          | Cytoscape.js, Vis.js | Dependency graphs, networks |
| **Diff**           | Monaco Diff Editor  | diff2html            | Code comparison             |
| **JSON**           | react-json-view     | —                    | Object exploration          |
| **Timeline**       | vis-timeline        | —                    | Event timelines             |
| **Maps**           | Leaflet + D3        | —                    | Geographic data             |
| **Export PNG/PDF** | Puppeteer (CDP)     | html2canvas          | Raster export               |
| **Python plots**   | matplotlib, seaborn | —                    | Statistical analysis        |

### 10.6 Example Agent Workflows

#### Data Analysis Workflow

```typescript
// User: "Analyze the login patterns from our database"

// 1. Query data
const loginData = await toolkit.call('mcp.postgres.query', {
  query: `
    SELECT 
      date_trunc('hour', created_at) as hour,
      count(*) as logins,
      count(distinct user_id) as unique_users
    FROM logins 
    WHERE created_at > now() - interval '7 days'
    GROUP BY 1 ORDER BY 1
  `,
});

// 2. Show line chart inline in chat
await toolkit.call('builtin.viz.chart', {
  type: 'line',
  data: loginData.rows,
  x: 'hour',
  y: ['logins', 'unique_users'],
  title: 'Login Activity - Last 7 Days',
  target: 'chat',
});

// 3. Show hourly heatmap
await toolkit.call('builtin.viz.heatmap', {
  data: processedData, // reshape to day x hour
  xLabels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
  yLabels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
  title: 'Login Heatmap by Day and Hour',
  target: 'chat',
});

// 4. Export to PDF for report
await toolkit.call('builtin.viz.toPdf', {
  content: [vizId1, vizId2], // Reference previous visualizations
  path: 'reports/login-analysis.pdf',
  title: 'Weekly Login Analysis Report',
});
```

#### Code Architecture Analysis

```typescript
// User: "Show me the architecture of the auth module"

// 1. Generate dependency graph
await toolkit.call('builtin.viz.depgraph', {
  target: 'src/auth/',
  depth: 4,
  layout: 'hierarchical',
  target: 'chat', // Show inline first
});

// 2. Generate call graph for main function
await toolkit.call('builtin.viz.callgraph', {
  function: 'src/auth/index.ts:authenticate',
  direction: 'outgoing',
  depth: 3,
  target: 'chat',
});

// 3. Create sequence diagram for the flow
await toolkit.call('builtin.viz.mermaid', {
  code: `
    sequenceDiagram
      participant C as Client
      participant A as Auth API
      participant D as Database
      participant T as Token Service
    
      C->>A: POST /login
      A->>D: Find user
      D-->>A: User record
      A->>A: Verify password
      A->>T: Generate JWT
      T-->>A: Token
      A-->>C: 200 + Token
  `,
  target: 'chat',
});

// 4. Export all as HTML documentation
await toolkit.call('builtin.viz.toHtml', {
  vizId: 'all', // Export all visualizations
  path: 'docs/auth-architecture.html',
  title: 'Auth Module Architecture',
  selfContained: true,
});
```

#### Planning Implementation

```typescript
// User: "Help me plan the refactoring of the payment module"

// 1. Show current structure
await toolkit.call('builtin.viz.depgraph', {
  target: 'src/payments/',
  highlight: ['src/payments/legacy.ts'], // Highlight problem areas
  target: 'chat',
});

// 2. Create comparison table
await toolkit.call('builtin.viz.table', {
  data: [
    { component: 'PaymentProcessor', currentLOC: 450, proposedLOC: 200, complexity: 'High' },
    { component: 'Validator', currentLOC: 200, proposedLOC: 150, complexity: 'Medium' },
    { component: 'Gateway', currentLOC: 300, proposedLOC: 180, complexity: 'Medium' },
  ],
  columns: [
    { key: 'component', label: 'Component' },
    { key: 'currentLOC', label: 'Current LOC', format: 'number' },
    { key: 'proposedLOC', label: 'Proposed LOC', format: 'number' },
    { key: 'complexity', label: 'Complexity' },
  ],
  target: 'chat',
});

// 3. Create implementation timeline
await toolkit.call('builtin.viz.mermaid', {
  code: `
    gantt
      title Payment Module Refactoring
      dateFormat YYYY-MM-DD
      section Phase 1
        Extract interfaces     :a1, 2025-01-01, 3d
        Create new processor   :a2, after a1, 5d
      section Phase 2
        Migrate validators     :b1, after a2, 4d
        Update tests           :b2, after a2, 3d
      section Phase 3
        Deploy & monitor       :c1, after b1, 2d
  `,
  target: 'chat',
});
```

#### Statistical Analysis with Python

```typescript
// User: "Do a statistical analysis of response times"

// 1. Get the data
const metrics = await toolkit.call('mcp.prometheus.query', {
  query: 'http_request_duration_seconds_bucket{handler="/api/v1/users"}[24h]',
});

// 2. Use Python seaborn for statistical visualization
await toolkit.call('builtin.viz.seaborn', {
  plotType: 'violin',
  data: metrics,
  x: 'endpoint',
  y: 'duration',
  hue: 'status_code',
  style: 'whitegrid',
  target: 'chat',
});

// 3. Create correlation heatmap
await toolkit.call('builtin.viz.matplotlib', {
  code: `
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data)
corr = df[['response_time', 'payload_size', 'concurrent_users']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
  `,
  data: metrics,
  target: 'chat',
});

// 4. Save the analysis
await toolkit.call('builtin.viz.toPng', {
  type: 'matplotlib',
  spec: { code: '...', data: metrics },
  path: 'analysis/response-time-correlation.png',
  dpi: 300,
});
```

### 10.7 Chat Visualization Rendering

The chat interface handles visualization rendering:

```typescript
interface ChatVisualization {
  vizId: string;
  type: string;
  rendered: ReactNode; // The actual rendered component

  // Actions available on each visualization
  actions: {
    expand: () => void; // Open in panel
    export: (format: string) => void;
    copy: (format: string) => void;
    edit: () => void; // Open spec editor
  };

  // Metadata
  title?: string;
  description?: string;
  dataSource?: string;
}

// Chat message can contain multiple visualizations
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  visualizations?: ChatVisualization[];
}
```

### 10.8 Visualization Panel Integration

```typescript
interface VisualizationPanel {
  // Panel management
  open(vizId: string): void;
  close(panelId: string): void;

  // Tabs
  createTab(viz: ChatVisualization): string;
  switchTab(tabId: string): void;

  // Interactivity
  onNodeClick: (nodeId: string) => void; // For graphs
  onZoom: (level: number) => void;
  onFilter: (filter: string) => void;

  // Export from panel
  exportCurrent(format: string, path?: string): Promise<void>;
}
```

---

## 11. Agent API (Updated)

```typescript
interface AgentToolkit {
  // Discovery
  listTools(filter?: ToolFilter): Tool[];
  getTool(id: string): Tool | null;
  searchTools(query: string): Tool[];

  // Execution - Agent never knows the underlying protocol
  call<T = unknown>(toolId: string, params: Record<string, unknown>): Promise<T>;
  callStreaming(toolId: string, params: Record<string, unknown>): AsyncIterable<ToolStreamChunk>;

  // Context
  setWorkspace(path: string): void;
  setCurrentFile(path: string): void;
}

// Usage examples
const toolkit = getToolkit();

// File operation (Builtin)
const content = await toolkit.call('builtin.file.read', { path: 'src/main.rs' });

// Code intelligence (MCP - routed to Pylance MCP)
const symbols = await toolkit.call('mcp.code.symbols', { path: 'src/main.py' });

// Git operation (MCP)
const status = await toolkit.call('mcp.git.status', {});

// Debug operation (DAP)
await toolkit.call('dap.debug.setBreakpoint', { file: 'main.py', line: 42 });

// === VISUALIZATION EXAMPLES ===

// Inline chart in chat
await toolkit.call('builtin.viz.chart', {
  type: 'line',
  data: salesData,
  x: 'month',
  y: 'revenue',
  title: 'Monthly Revenue',
  target: 'chat', // Renders inline in conversation
});

// Dependency graph in panel
await toolkit.call('builtin.viz.depgraph', {
  target: 'src/api/',
  depth: 3,
  layout: 'hierarchical',
  target: 'panel', // Opens in visualization panel
});

// Mermaid diagram inline
await toolkit.call('builtin.viz.mermaid', {
  code: `flowchart LR
    A[User] --> B[API]
    B --> C[Database]`,
  target: 'chat',
});

// Export visualization to file
await toolkit.call('builtin.viz.toPng', {
  type: 'chart',
  spec: { type: 'bar', data: reportData },
  path: 'reports/quarterly-chart.png',
  width: 1200,
  height: 800,
});

// Python matplotlib for statistical analysis
await toolkit.call('builtin.viz.matplotlib', {
  code: `
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.boxplot(data=df, x='category', y='value')
    plt.title('Distribution by Category')
  `,
  data: analysisData,
  target: 'chat',
});

// Export full report as PDF
await toolkit.call('builtin.viz.toPdf', {
  content: [vizId1, vizId2, vizId3],
  path: 'reports/analysis-report.pdf',
  title: 'Q4 Analysis Report',
});

// Agent doesn't care about protocol - UTI handles routing!
```

---

## 12. Tool Count Summary (Updated)

| Protocol              | Tool Count  | Categories                                                           |
| --------------------- | ----------- | -------------------------------------------------------------------- |
| **Builtin**           | ~95 tools   | File, Terminal, Tree-sitter, GNN, Browser, Testing,**Visualization** |
| **MCP**               | ~45 tools   | Code Intelligence, Git, Database, Deployment, Collaboration          |
| **DAP**               | ~10 tools   | Debugging                                                            |
| **Shell (wrapped)**   | ~10 tools   | Graphviz, PlantUML, matplotlib, ggplot                               |
| **LSP (Editor only)** | ~15 methods | Real-time autocomplete, hover, diagnostics                           |

**Total Agent-accessible tools: ~160**

---

## 13. Implementation Priority (Updated)

### Phase 1: Foundation (Weeks 1-4)

- [ ] UTI core: Registry, Router, Executor
- [ ] Builtin Adapter: File ops, Terminal, Tree-sitter
- [ ] Basic MCP Adapter: Git
- [ ] Approval Queue
- [ ] **Basic Visualization: Mermaid, Tables (inline chat)**

### Phase 2: Code Intelligence (Weeks 5-8)

- [ ] MCP integration: Pylance, tsserver MCPs
- [ ] Builtin: Dependency graph (GNN)
- [ ] Builtin fallback: Tree-sitter symbols
- [ ] **Visualization: Dependency graph, Call graph rendering**

### Phase 3: Testing & Debug (Weeks 9-12)

- [ ] Builtin: Test runner, coverage
- [ ] Builtin: Browser automation (CDP)
- [ ] DAP Adapter: Python, Node.js debugging
- [ ] **Visualization: Charts (Plotly), Diff view**

### Phase 4: Deployment & Monitoring (Weeks 13-16)

- [ ] MCP: Database servers
- [ ] MCP: Platform deployment (Railway)
- [ ] MCP: Monitoring integrations
- [ ] Builtin: Container operations
- [ ] **Visualization: Timeline, Heatmaps, Network graphs**

### Phase 5: Advanced Viz & Export (Weeks 17-20)

- [ ] **Export: PNG, PDF, HTML generation**
- [ ] **Python integration: matplotlib, seaborn**
- [ ] **External tools: Graphviz, PlantUML**
- [ ] Visualization Panel with tabs
- [ ] Plugin system for custom visualizations
