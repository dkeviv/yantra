# Yantra - Decision Log

**Purpose:** Track all significant design and architecture decisions  
**Last Updated:** November 20, 2025

---

## How to Use This Document

When making significant design or architecture decisions:

1. Add a new entry with the date
2. Describe the decision clearly
3. Explain the rationale and alternatives considered
4. Note the impact and affected components
5. Include who made the decision

---

## Decision Format

```
## [Date] - [Decision Title]

**Status:** [Proposed | Accepted | Superseded]
**Deciders:** [Names]
**Impact:** [High | Medium | Low]

### Context
What circumstances led to this decision?

### Decision
What did we decide to do?

### Rationale
Why did we make this decision?

### Alternatives Considered
What other options were evaluated?

### Consequences
What are the implications (positive and negative)?

### Related Decisions
Links to related decision entries
```

---

## Decisions

### 🆕 November 22, 2025 - Add Terminal Integration for Full Automation

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** CRITICAL - Enables complete autonomous development lifecycle

#### Context
During Session 5 brainstorming, the true vision for Yantra was clarified: Not just a code generator, but a **fully autonomous agentic developer** that handles the complete software development lifecycle from understanding requirements to deploying and monitoring production systems.

**Original Design Assumption:** "No shell command execution for security reasons" → Yantra would only generate code and validate it, but users would manually run, test, package, and deploy.

**Reality Check:** This assumption fundamentally limits Yantra's value proposition. The vision is **full automation**: Generate → Run → Test → Package → Deploy → Monitor → Heal. Without terminal integration, Yantra cannot:
- Run generated code to verify it works
- Execute tests in subprocess
- Install missing dependencies automatically
- Build distributable packages (Docker, wheels)
- Deploy to cloud platforms
- Monitor production and auto-fix issues

**The Question:** "Can the lack of terminal integration be perceived as a limitation by developers?"  
**Answer:** YES - It's not just a limitation, it's a showstopper for the autonomous vision.

#### Decision
**Add comprehensive terminal integration with secure command execution capabilities.**

Implement a `TerminalExecutor` module that:
1. Executes whitelisted commands in controlled subprocess environment
2. Streams real-time output to UI via async channels
3. Validates commands using whitelist + regex patterns
4. Blocks dangerous commands (rm -rf, sudo, eval, shell injection)
5. Maintains execution context (venv, env vars, working directory)
6. Logs all commands to SQLite for audit trail
7. Implements resource limits (timeout, memory)

**Shift in Philosophy:**
- **OLD:** "No shell commands → Security through prohibition"
- **NEW:** "Controlled command execution → Security through validation"

#### Rationale

**1. Full Automation is the Core Value Proposition**
- **Competitive Moat:** Yantra's unique value is being the ONLY platform that handles the complete development lifecycle autonomously
- **vs Copilot/Cursor:** They stop at code generation. Yantra continues through deployment.
- **vs Replit Agent:** Yantra adds enterprise features (self-healing, browser automation, desktop app)
- **Time Savings:** 98% reduction in development time (10 min vs 11 hours for full feature deployment)

**2. Developer Expectations**
- Modern AI coding tools (Replit Agent, Devin) already execute code
- Developers expect automation, not just code suggestions
- Terminal integration is table stakes, not a luxury

**3. Verification Requires Execution**
- **Truth:** The only way to know if code works is to run it
- Static analysis and tests are insufficient without actual execution
- Runtime errors can only be detected by running the code
- Dependency issues surface during installation, not before

**4. Security Through Smart Design**
- Whitelist approach is proven (sandboxes, containers use this)
- Command validation prevents injection attacks
- Audit logging provides forensics
- Resource limits prevent resource exhaustion
- Much safer than allowing users to run arbitrary external terminal commands

**5. User Experience**
- **No Context Switching:** Everything in one window (Yantra)
- **Full Transparency:** Users see exactly what commands are executed
- **Real-Time Feedback:** Watch progress as it happens
- **Learning Tool:** Understand what commands Yantra uses
- **Trust Building:** Verify Yantra's actions in real-time

#### Alternatives Considered

**Alternative 1: No Terminal Integration (Original Design)**
- **Pros:** 
  - Simpler security model
  - No command injection risks
  - Smaller attack surface
- **Cons:** ❌
  - Fundamentally limits Yantra to code generation only
  - Cannot verify generated code works
  - Cannot auto-install dependencies
  - Cannot build packages or deploy
  - User must switch to external terminal (poor UX)
  - **Conclusion:** Defeats the purpose of autonomous development

**Alternative 2: User Provides Terminal Commands**
- **Pros:**
  - Security responsibility on user
  - No command validation needed
- **Cons:** ❌
  - User still has to think about commands (not autonomous)
  - Security risk if user provides malicious commands
  - Doesn't reduce developer workload
  - **Conclusion:** Not truly autonomous

**Alternative 3: Limited Command Set (Hardcoded)**
- **Pros:**
  - Simple implementation
  - Very secure (no dynamic commands)
- **Cons:** ❌
  - Too restrictive for real-world use cases
  - Cannot handle custom build tools
  - Cannot adapt to different tech stacks
  - **Conclusion:** Not flexible enough

**Alternative 4: Full Shell Access (Unsafe)**
- **Pros:**
  - Maximum flexibility
  - No command restrictions
- **Cons:** ❌ UNACCEPTABLE
  - Major security vulnerability
  - Allows arbitrary command execution
  - No audit trail
  - Could delete files, install malware, etc.
  - **Conclusion:** Irresponsible design

**Alternative 5: Whitelist + Validation (CHOSEN)**
- **Pros:** ✅
  - Secure yet flexible
  - Supports all necessary automation
  - Full audit trail
  - Blocks dangerous patterns
  - Adapts to different tech stacks
  - Best balance of security and functionality
- **Cons:**
  - More complex implementation
  - Requires ongoing maintenance of whitelist
  - **Conclusion:** Best approach for production system

#### Consequences

**Positive:**
1. **Enables Full Automation**
   - Complete generate → run → test → package → deploy pipeline
   - True autonomous development (human provides intent only)
   - 98% time savings (10 min vs 11 hours for complete feature)

2. **Competitive Differentiation**
   - Only platform with complete development lifecycle automation
   - Stronger moat vs Copilot, Cursor, Windsurf
   - Comparable to Replit Agent but with enterprise features

3. **Better User Experience**
   - No context switching between tools
   - Real-time feedback and transparency
   - Learning tool (see what commands are used)
   - Trust building through visibility

4. **Verification & Quality**
   - Code verified by actual execution
   - Runtime errors caught automatically
   - Dependencies validated by installation
   - Tests run in real environment

5. **Enterprise Features Enabled**
   - Package building (Docker, wheels, npm)
   - Automated deployment (AWS, GCP, K8s)
   - Production monitoring & self-healing
   - CI/CD pipeline generation

**Negative:**
1. **Implementation Complexity**
   - Need to build secure command executor
   - Regex patterns for validation
   - Streaming output infrastructure
   - Error handling and recovery
   - **Mitigation:** Well-documented architecture, comprehensive tests

2. **Security Risks (Mitigated)**
   - Command injection → Blocked by argument validation
   - Dangerous commands → Blocked by pattern matching
   - Resource exhaustion → Timeout and memory limits
   - Privilege escalation → Block sudo, su, chmod +x
   - **Mitigation:** Multiple layers of security

3. **Maintenance Burden**
   - Whitelist needs updates for new tools
   - Patterns need refinement over time
   - **Mitigation:** Community contributions, automated pattern updates

4. **Platform Differences**
   - Commands differ across OS (Windows/Mac/Linux)
   - Shell syntax variations
   - **Mitigation:** Detect OS, adapt commands accordingly

**Trade-offs Accepted:**
- **Simplicity ↔ Functionality:** Accept complexity for automation
- **Strict Security ↔ Flexibility:** Balance via whitelist approach
- **Fast Implementation ↔ Robustness:** Invest time in proper security

#### Implementation Details

**Security Measures:**
1. **Command Whitelist** (HashSet for O(1) lookup)
   - Python: `python`, `python3`, `pip`, `pytest`, `black`, `flake8`
   - Node: `node`, `npm`, `npx`, `yarn`, `jest`
   - Rust: `cargo`
   - Docker: `docker` (build, run, ps, stop only)
   - Git: `git` (via MCP protocol for extra security)
   - Cloud: `aws`, `gcloud`, `kubectl`, `terraform`, `heroku`

2. **Blocked Patterns** (Pre-compiled Regex)
   - File operations: `rm -rf`, `chmod +x`
   - Privilege escalation: `sudo`, `su`
   - Code execution: `eval`, `exec`, `source`
   - Shell injection: `;`, `|`, `&`, `` ` ``, `$(`, `{`, `}`
   - Network attacks: `curl | bash`, `wget | sh`
   - System file access: `> /etc/*`, `> /sys/*`

3. **Argument Validation**
   - Check each argument for shell metacharacters
   - Reject commands with suspicious patterns
   - Validate file paths are within workspace

4. **Resource Limits**
   - Timeout: 5 minutes per command
   - Memory: Kill if exceeds 2GB
   - CPU: No hard limit (local execution)

5. **Audit Logging**
   - Log all commands to SQLite
   - Include: timestamp, command, exit code, output, user intent
   - Enable forensics and debugging

**Architecture:**
```rust
// src/agent/terminal.rs

pub struct TerminalExecutor {
    workspace_path: PathBuf,
    python_env: Option<PathBuf>,
    env_vars: HashMap<String, String>,
    command_whitelist: CommandWhitelist,
}

pub struct CommandWhitelist {
    allowed_commands: HashSet<String>,
    allowed_patterns: Vec<Regex>,
    blocked_patterns: Vec<Regex>,
}

impl TerminalExecutor {
    // 1. Validate command (whitelist + pattern check)
    pub fn validate_command(&self, cmd: &str) -> Result<ValidatedCommand>
    
    // 2. Execute with streaming output
    pub async fn execute_with_streaming(
        &self,
        cmd: &str,
        output_sender: mpsc::Sender<String>,
    ) -> Result<ExecutionResult>
    
    // 3. Environment setup
    pub fn setup_environment(&mut self, project_type: ProjectType) -> Result<()>
}
```

**Integration with Agent:**
- Add 5 new phases to orchestrator:
  1. `EnvironmentSetup` - Create venv, set env vars
  2. `DependencyInstallation` - pip install, npm install
  3. `ScriptExecution` - Run generated code
  4. `RuntimeValidation` - Verify execution success
  5. `PerformanceProfiling` - Measure execution time

**UI Component:**
- Bottom terminal panel (30% height, resizable)
- Real-time streaming output (<10ms latency)
- Color-coded: stdout (white), stderr (red), success (green)
- Features: Auto-scroll, copy, clear, search, timestamps

#### Performance Targets
- Command validation: <1ms
- Subprocess spawn: <50ms
- Output streaming latency: <10ms per line
- Environment setup: <5s (venv creation)
- Dependency installation: <30s (with caching)
- Full execution cycle: <3 minutes (generate → run → test → commit)

#### Timeline
- **Week 9-10:** Terminal executor, test runner, dependency installer, output panel UI
- **Month 3-4:** Package building, deployment automation
- **Month 5:** Monitoring & self-healing

#### Related Decisions
- Use Tokio for async subprocess execution (enables streaming)
- Use mpsc channels for output streaming (real-time updates)
- Use SQLite for audit logging (existing infrastructure)
- Add orchestrator execution phases (extends state machine)

#### Lessons Learned
1. **Early assumptions need validation:** "No shell commands" was premature optimization for security
2. **Vision drives architecture:** Clarifying the autonomous vision changed everything
3. **Security through design, not prohibition:** Whitelist approach is secure AND flexible
4. **User expectations matter:** Modern AI tools execute code, Yantra must too
5. **Verification requires execution:** Static analysis is insufficient without running code

---

### November 20, 2025 - Use Tauri Over Electron

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
Need to choose a desktop application framework for Yantra that provides cross-platform support, good performance, and reasonable bundle size.

#### Decision
Use Tauri 1.5+ as the desktop application framework.

#### Rationale
- **Bundle Size:** Tauri produces 600KB bundles vs Electron's 150MB
- **Memory Footprint:** Tauri uses ~100MB vs Electron's ~400MB
- **Performance:** Rust backend provides better performance for GNN operations
- **Native Integration:** Better OS integration and native feel
- **Security:** Rust's memory safety provides additional security guarantees

#### Alternatives Considered
1. **Electron**
   - Pros: Mature ecosystem, widely used, extensive documentation
   - Cons: Large bundle size, high memory usage, slower startup
   
2. **Native Apps (per platform)**
   - Pros: Best performance, fully native
   - Cons: Need to maintain 3 separate codebases, much higher development cost

#### Consequences
- **Positive:**
  - Smaller download size attracts more users
  - Better performance for GNN operations
  - Lower memory usage = better user experience
  - Rust backend enables better security
  
- **Negative:**
  - Smaller community compared to Electron
  - Fewer ready-made components
  - Team needs Rust knowledge

#### Related Decisions
- Use SolidJS for frontend (Nov 20, 2025)

---

### November 20, 2025 - Use SolidJS Over React

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** Medium

#### Context
Need to choose a frontend framework for the Tauri application that is fast, lightweight, and has good TypeScript support.

#### Decision
Use SolidJS 1.8+ as the frontend framework.

#### Rationale
- **Performance:** Fastest reactive framework in benchmarks
- **Bundle Size:** Smaller than React
- **No Virtual DOM:** Direct DOM updates are faster
- **TypeScript Support:** First-class TypeScript support
- **Reactivity:** Better reactivity model than React

#### Alternatives Considered
1. **React**
   - Pros: Huge ecosystem, most developers know it, extensive libraries
   - Cons: Larger bundle size, virtual DOM overhead, slower
   
2. **Vue**
   - Pros: Good performance, nice API, growing ecosystem
   - Cons: Smaller community than React, less TypeScript support
   
3. **Svelte**
   - Pros: Compiles away, small bundle, good performance
   - Cons: Smaller ecosystem, less mature

#### Consequences
- **Positive:**
  - Best performance for UI updates
  - Smaller bundle contributes to overall app size goals
  - Modern reactive paradigm
  
- **Negative:**
  - Smaller community = fewer resources
  - Team needs to learn SolidJS
  - Fewer third-party components

#### Related Decisions
- Use Tauri for desktop (Nov 20, 2025)

---

### November 20, 2025 - Use Rust for GNN Implementation

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
The Graph Neural Network (GNN) is performance-critical, handling code dependency analysis for potentially 100k+ lines of code.

#### Decision
Implement the GNN engine in Rust.

#### Rationale
- **Performance:** Native performance without garbage collection
- **Memory Safety:** No null pointers, no data races
- **Concurrency:** Fearless concurrency with Tokio
- **Zero-Cost Abstractions:** High-level code with C-level performance
- **Integration:** Already using Rust for Tauri backend
- **petgraph:** Excellent graph library available

#### Alternatives Considered
1. **Python**
   - Pros: Easier to write, NetworkX library available
   - Cons: Too slow for 100k LOC projects, GIL limits concurrency
   
2. **TypeScript/JavaScript**
   - Pros: Frontend team already knows it
   - Cons: Not fast enough, no memory safety guarantees
   
3. **C++**
   - Pros: Maximum performance, Boost Graph Library
   - Cons: Manual memory management, harder to maintain, more bugs

#### Consequences
- **Positive:**
  - Meets performance targets (<5s for 10k LOC)
  - Can handle scale (100k LOC)
  - Memory safe = fewer bugs
  - Concurrent processing possible
  
- **Negative:**
  - Steeper learning curve
  - Longer development time initially
  - Fewer developers know Rust

#### Related Decisions
- Use petgraph for graph operations (Nov 20, 2025)
- Use SQLite for persistence (Nov 20, 2025)

---

### November 20, 2025 - Use petgraph for Graph Operations

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** Medium

#### Context
Need a graph data structure library for implementing the GNN.

#### Decision
Use petgraph 0.6+ for graph data structures and algorithms.

#### Rationale
- **Mature:** Well-tested and stable library
- **Performance:** Optimized graph algorithms
- **Flexible:** Supports directed/undirected, weighted/unweighted graphs
- **Algorithms:** Includes BFS, DFS, shortest path, etc.
- **Zero-Cost:** Generic implementation with no runtime overhead

#### Alternatives Considered
1. **Custom Implementation**
   - Pros: Full control, optimized for our use case
   - Cons: High development cost, need to implement all algorithms, testing burden
   
2. **graph-rs**
   - Pros: Another Rust graph library
   - Cons: Less mature, smaller community, fewer features

#### Consequences
- **Positive:**
  - Save development time
  - Battle-tested algorithms
  - Good documentation
  
- **Negative:**
  - Dependency on external library
  - Need to learn petgraph API

#### Related Decisions
- Use Rust for GNN (Nov 20, 2025)

---

### November 20, 2025 - Use SQLite for GNN Persistence

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** Medium

#### Context
Need to persist the GNN graph between application sessions and support incremental updates.

#### Decision
Use SQLite 3.44+ for GNN persistence.

#### Rationale
- **Embedded:** No separate database server needed
- **Fast:** Excellent performance for local storage
- **ACID:** Transaction support ensures data integrity
- **Portable:** Single file database
- **Mature:** Battle-tested and reliable
- **Query Support:** SQL for complex queries

#### Alternatives Considered
1. **File-based (JSON/Binary)**
   - Pros: Simpler, no database dependency
   - Cons: No query support, need to load entire graph, no transactions
   
2. **PostgreSQL/MySQL**
   - Pros: More powerful, better for multi-user
   - Cons: Requires separate server, overkill for desktop app, more complex setup
   
3. **RocksDB/LevelDB**
   - Pros: Fast key-value store
   - Cons: No SQL, harder to query, less mature Rust bindings

#### Consequences
- **Positive:**
  - Fast incremental updates
  - Query support for complex lookups
  - Transaction support
  - No additional installation needed
  
- **Negative:**
  - Need to design schema carefully
  - SQLite dependency

#### Related Decisions
- Use Rust for GNN (Nov 20, 2025)

---

### November 20, 2025 - Use Multi-LLM Orchestration (Claude + GPT-4)

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
LLMs are critical for code generation quality, but single LLM has reliability and quality limitations.

#### Decision
Use multiple LLMs with intelligent orchestration:
- Claude Sonnet 4 as primary
- GPT-4 Turbo as secondary/fallback

#### Rationale
- **Reliability:** No single point of failure
- **Quality:** Can use consensus for critical operations
- **Cost Optimization:** Route simple tasks to cheaper model
- **Best-of-Breed:** Use each LLM's strengths
- **Failover:** If Claude is down, fall back to GPT-4

#### Alternatives Considered
1. **Single LLM (Claude only)**
   - Pros: Simpler implementation, lower cost
   - Cons: Single point of failure, no consensus option
   
2. **Single LLM (GPT-4 only)**
   - Pros: Simpler implementation, very capable
   - Cons: Higher cost, single point of failure
   
3. **Open Source LLMs Only**
   - Pros: No API costs, full control
   - Cons: Lower quality, requires expensive GPU, deployment complexity

#### Consequences
- **Positive:**
  - Higher reliability (99%+ uptime)
  - Better code quality through consensus
  - Optimized costs
  - Flexibility to add more LLMs later
  
- **Negative:**
  - More complex implementation
  - Need to manage multiple API keys
  - Higher development cost initially
  - Need smart routing logic

#### Related Decisions
- Implement circuit breaker pattern (Nov 20, 2025)
- Add response caching (Nov 20, 2025)

---

### November 20, 2025 - Use tree-sitter for Code Parsing

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
Need to parse Python code to build the GNN and understand code structure.

#### Decision
Use tree-sitter with tree-sitter-python for code parsing.

#### Rationale
- **Fast:** Incremental parsing is very fast
- **Accurate:** Produces proper AST
- **Error-Tolerant:** Can parse incomplete code
- **Incremental:** Only reparse changed sections
- **Multi-Language:** Can add JS/TS support later
- **Rust Bindings:** Good Rust support

#### Alternatives Considered
1. **Python's AST Module**
   - Pros: Native Python support, official
   - Cons: Requires Python runtime, not incremental, not usable from Rust
   
2. **Custom Parser**
   - Pros: Full control
   - Cons: Huge development effort, error-prone, hard to maintain
   
3. **ANTLR**
   - Pros: Powerful parser generator
   - Cons: Slower than tree-sitter, more complex, larger dependency

#### Consequences
- **Positive:**
  - Fast incremental parsing
  - Can meet <50ms update target
  - Supports future multi-language needs
  - Error-tolerant
  
- **Negative:**
  - Tree-sitter dependency
  - Learning curve for tree-sitter query language

#### Related Decisions
- Use Rust for GNN (Nov 20, 2025)

---

### November 20, 2025 - Use Monaco Editor for Code Viewing

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** Low

#### Context
Need a code editor component for displaying generated code to users.

#### Decision
Use Monaco Editor 0.44+ for the code viewer panel.

#### Rationale
- **Industry Standard:** Same editor as VS Code
- **Feature-Rich:** Syntax highlighting, IntelliSense, minimap
- **Well-Maintained:** Active development by Microsoft
- **Familiar:** Developers already know it
- **TypeScript:** Great TypeScript support

#### Alternatives Considered
1. **CodeMirror**
   - Pros: Lightweight, modular
   - Cons: Less feature-rich, smaller ecosystem
   
2. **Ace Editor**
   - Pros: Mature, widely used
   - Cons: Less active development, older architecture
   
3. **Custom Editor**
   - Pros: Full control, lightweight
   - Cons: Huge development effort, reinventing wheel

#### Consequences
- **Positive:**
  - Professional code viewing experience
  - Familiar to developers
  - Rich features out of the box
  
- **Negative:**
  - Large bundle size (~5MB)
  - Need to bundle separately

#### Related Decisions
- Use SolidJS for frontend (Nov 20, 2025)

---

### November 20, 2025 - Focus on Python Only for MVP

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
Need to choose initial language(s) to support in MVP. Multi-language support is complex.

#### Decision
Support Python only for MVP (Phase 1). Add JavaScript/TypeScript in Phase 3.

#### Rationale
- **Focus:** Allows us to perfect the experience for one language
- **Faster MVP:** Reduces scope and complexity
- **Market:** Python is very popular for backend, data science, AI
- **Testing:** Single language test generation is simpler
- **GNN:** Can optimize GNN for Python patterns

#### Alternatives Considered
1. **Python + JavaScript in MVP**
   - Pros: Broader market, full-stack support
   - Cons: 2x the complexity, delays MVP, harder to perfect
   
2. **JavaScript Only**
   - Pros: Huge market, web-focused
   - Cons: Less compelling for backend-heavy projects

#### Consequences
- **Positive:**
  - Faster MVP delivery
  - Better quality for Python support
  - Clear target audience
  - Simpler testing
  
- **Negative:**
  - Misses full-stack developers initially
  - No frontend code generation in MVP
  - Need to add languages later

#### Related Decisions
- Plan JavaScript/TypeScript for Phase 3 (Nov 20, 2025)

---

### November 20, 2025 - Use Model Context Protocol (MCP) for Git

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** Medium

#### Context
Need to integrate with Git for committing generated code.

#### Decision
Use Model Context Protocol (MCP) for Git operations via git2-rs.

#### Rationale
- **Standardized:** MCP is emerging standard for tool integration
- **Native:** git2-rs provides native Git operations (libgit2)
- **No Shell:** Avoid shell command execution
- **Cross-Platform:** Works consistently across OS

#### Alternatives Considered
1. **Shell Commands**
   - Pros: Simple, familiar
   - Cons: Security risk, platform-specific, output parsing issues
   
2. **Direct libgit2**
   - Pros: Full control
   - Cons: Lower-level API, more code to write

#### Consequences
- **Positive:**
  - Safe Git operations
  - Cross-platform consistency
  - Future-proof with MCP standard
  
- **Negative:**
  - Need to learn MCP and git2-rs APIs

#### Related Decisions
- None yet

---

### November 20, 2025 - Use Horizontal Slices for Implementation

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
Need to decide implementation strategy: build full layers vs complete features.

#### Decision
Implement in horizontal slices (complete features) rather than vertical slices (full layers).

#### Rationale
- **Ship Faster:** Users get working features sooner
- **Feedback:** Can get user feedback on complete features
- **Motivation:** Team sees working features, stays motivated
- **Pivot:** Easier to pivot based on user feedback
- **MVP Spirit:** Aligns with MVP methodology

#### Alternatives Considered
1. **Vertical Slices (Layer-by-Layer)**
   - Pros: Clean architecture, complete layers
   - Cons: Nothing works until all layers complete, no early feedback

#### Consequences
- **Positive:**
  - Working features every sprint
  - Early user validation
  - Better demos
  - Reduced risk
  
- **Negative:**
  - Some code might need refactoring
  - Architecture emerges gradually

#### Related Decisions
- None yet

---

### November 20, 2025 - Mandatory 100% Test Pass Rate

**Status:** Accepted  
**Deciders:** Project Team  
**Impact:** High

#### Context
Need to define testing standards and what to do with failing tests.

#### Decision
100% of tests MUST pass. Never change test conditions to make tests pass. Fix the underlying issues instead.

#### Rationale
- **Quality:** Ensures generated code actually works
- **Trust:** Users trust "code that never breaks" promise
- **Discipline:** Forces proper implementation
- **No Technical Debt:** No accumulation of "known failures"

#### Alternatives Considered
1. **Allow Some Failures**
   - Pros: Faster development
   - Cons: Violates core promise, accumulates technical debt
   
2. **Skip Tests**
   - Pros: Even faster development
   - Cons: Defeats purpose of automated testing

#### Consequences
- **Positive:**
  - High code quality
  - User trust
  - Validates core value proposition
  
- **Negative:**
  - Slower development initially
  - Must fix all issues before proceeding
  - No shortcuts allowed

#### Related Decisions
- 90%+ coverage required (Nov 20, 2025)

---

### November 20, 2025 - LLM Mistake Tracking & Learning System

**Status:** Accepted  
**Deciders:** Product Team (User Request)  
**Impact:** High

#### Context
LLMs (Claude, GPT-4) tend to make repeated coding mistakes even after being corrected. Without a learning mechanism, the system will continuously regenerate the same buggy patterns, reducing code quality and user trust.

**Problem:**
- Same mistakes repeated across sessions
- No memory of previous corrections
- Each model has unique error patterns
- Manual tracking not scalable

#### Decision
Implement an **automated LLM Mistake Tracking & Learning System** with hybrid storage:

1. **Vector Database (ChromaDB)** for semantic mistake patterns
2. **SQLite** for structured error metadata
3. **Automatic Detection** from test failures, security scans, and chat monitoring
4. **Pre-Generation Review** injects known issues into LLM context

#### Rationale

**Why Vector DB for Patterns:**
- Semantic similarity: "forgot await" matches "async without await"
- Fast k-NN search for relevant past mistakes
- Store code snippets with natural language descriptions
- Embeddings capture context better than exact matching

**Why Hybrid Storage:**
- SQLite: Fast queries for model-specific stats, frequency, timestamps
- Vector DB: Semantic search for similar issues across different wordings
- Complementary strengths

**Why Automatic Detection:**
- Manual logging won't scale
- Test failures indicate code issues
- Security scans reveal vulnerability patterns
- Chat monitoring detects repeated user corrections

#### Implementation Architecture

**Components:**

1. **Mistake Detector Module** (`src/learning/detector.rs`)
   - Monitors: Test failures, security scan results, chat corrections
   - Extracts: Error signature, context, model used, code snippet
   - Creates: Mistake pattern entries

2. **Pattern Storage** (`src/learning/storage.rs`)
   - SQLite schema:
     ```sql
     CREATE TABLE mistake_patterns (
       id INTEGER PRIMARY KEY,
       model_name TEXT,
       error_signature TEXT,
       frequency INTEGER,
       severity TEXT,
       first_seen TIMESTAMP,
       last_seen TIMESTAMP,
       fix_applied BOOLEAN
     );
     ```
   - ChromaDB collections:
     - `llm_mistakes`: Embedded mistake descriptions
     - `successful_fixes`: Embedded fix patterns

3. **Pattern Retrieval** (`src/learning/retrieval.rs`)
   - Query vector DB for top-K similar mistakes
   - Filter by model name and recency
     - Inject into LLM system prompt before generation

4. **Chat Monitor** (`src-ui/components/ChatPanel.tsx`)
   - Detect correction patterns: "no that's wrong", "fix the bug"
   - Extract what was wrong from conversation
   - Send to mistake detector

#### Workflow

**Code Generation Flow with Learning:**
```
User Request
  ↓
Query Vector DB (top-5 similar past mistakes for this model)
  ↓
Inject into System Prompt:
  "Common mistakes to avoid:
   1. [Mistake pattern 1]
   2. [Mistake pattern 2]
   ..."
  ↓
Generate Code (Claude/GPT-4)
  ↓
Run Tests
  ↓
If Test Fails:
  → Extract error pattern
  → Store in Vector DB + SQLite
  → Retry generation with mistake context
```

**Chat Correction Flow:**
```
User: "Fix that async bug"
  ↓
Parse conversation for error context
  ↓
Extract: What was wrong, what model generated it
  ↓
Store pattern in Vector DB
  ↓
Regenerate with mistake context injected
```

#### Alternatives Considered

1. **Prompt Engineering Only**
   - Pros: Simple, no storage needed
   - Cons: Can't learn from past, no model-specific patterns, limited context window

2. **SQL Database Only**
   - Pros: Fast exact matching
   - Cons: Can't find semantically similar issues, requires exact error text

3. **Manual Issue Tracking (copilot-instructions.md)**
   - Pros: Human-curated, high quality
   - Cons: Doesn't scale, no automation, stale quickly

4. **Fine-tuning Models**
   - Pros: Permanent learning
   - Cons: Expensive, requires retraining, can't fine-tune Claude API

#### Consequences

**Positive:**
- **Continuous Improvement:** System learns from every mistake
- **Model-Specific:** Track patterns unique to Claude vs GPT-4
- **Scalable:** Automatic detection and storage
- **Context-Aware:** Semantic search finds similar issues
- **Reduced Errors:** Known patterns prevented before generation
- **Better UX:** Fewer regeneration cycles, faster success

**Negative:**
- **Additional Complexity:** New module to maintain
- **Storage Growth:** Vector DB size increases over time
- **False Positives:** May inject irrelevant patterns
- **Performance:** Extra vector search adds latency (~50-100ms)
- **Privacy:** Must ensure mistake patterns don't leak sensitive code

#### Implementation Timeline

- **Week 5-6 (MVP):** Basic detection from test failures
- **Week 7 (MVP):** Vector DB integration, pattern storage
- **Week 8 (MVP):** Pre-generation pattern injection
- **Post-MVP:** Chat monitoring, advanced pattern extraction

#### Performance Targets

- Pattern retrieval: <100ms for top-K search
- Storage: <1MB per 100 patterns
- Injection: <50ms to add to prompt
- Max patterns per generation: 5-10 (context limit)

#### Related Decisions
- Use ChromaDB for Vector DB (Nov 20, 2025)
- Multi-LLM Orchestration (Nov 20, 2025)
- GNN for Dependency Tracking (Nov 20, 2025)

---

### November 20, 2025 - Implement Circuit Breaker Pattern for LLM Calls

**Status:** Accepted (Implemented)  
**Deciders:** Technical Team  
**Impact:** High

#### Context
LLM API calls can fail due to rate limits, network issues, or service outages. Without proper resilience patterns, the system would:
- Keep trying failed providers indefinitely
- Waste user time and API quota
- Provide poor user experience with long timeouts
- Risk cascading failures

#### Decision
Implement a **Circuit Breaker Pattern** for each LLM provider with three states:

1. **Closed (Normal):** Requests pass through normally
2. **Open (Failing):** Fast-fail without attempting request
3. **HalfOpen (Testing):** Try one request to test recovery

**Parameters:**
- Failure Threshold: 3 consecutive failures
- Cooldown Period: 60 seconds
- State stored with atomic operations for thread-safety

#### Rationale

**Why Circuit Breaker:**
- Prevents system from continuously calling failing services
- Fast-fail provides immediate feedback to user
- Automatic recovery testing after cooldown
- Industry-standard resilience pattern (Netflix Hystrix, etc.)

**Why These Parameters:**
- 3 failures: Balance between quick detection and avoiding false positives
- 60s cooldown: Typical for API rate limit resets
- HalfOpen state: Graceful recovery without overwhelming service

**Why Per-Provider:**
- Claude failure shouldn't affect OpenAI availability
- Independent monitoring and recovery
- Better observability

#### Implementation Details

**State Machine:**
```
Closed → (3 failures) → Open
Open → (60s timeout) → HalfOpen
HalfOpen → (success) → Closed
HalfOpen → (failure) → Open
```

**Code Location:**
- `src/llm/orchestrator.rs`: CircuitBreaker struct with state tracking
- Uses Rust atomics for lock-free state reads
- RwLock for state modifications
- Integrated with retry logic

**Interaction with Retry:**
- Retries happen within a single circuit breaker attempt
- 3 retries with exponential backoff (100ms, 200ms, 400ms)
- Circuit opens only after all retries exhausted

#### Alternatives Considered

1. **Simple Timeout Without Circuit Breaker**
   - Pros: Simpler implementation
   - Cons: Keeps trying failing service, wastes time

2. **Bulkhead Pattern**
   - Pros: Isolates failures, resource limits
   - Cons: More complex, overkill for 2 providers

3. **Retry Only**
   - Pros: Simple, no state management
   - Cons: Slow to detect persistent failures

#### Consequences

**Positive:**
- Fast-fail improves responsiveness (no 30s timeouts)
- Automatic recovery without manual intervention
- Better resource utilization (don't waste quota on failing provider)
- Clear observability of provider health

**Negative:**
- Additional complexity in orchestrator
- Must tune threshold and cooldown parameters
- Risk of false positives during temporary glitches
- State management adds memory overhead (~100 bytes per provider)

#### Performance Impact
- Circuit state check: <1ms (atomic read)
- No impact on successful requests
- Saves 30s timeout on fast-fail

#### Testing
- Unit tests for all state transitions (4 tests)
- Recovery testing after cooldown
- Concurrent access testing with RwLock

#### Related Decisions
- Multi-LLM Orchestration (Nov 20, 2025)
- Exponential Backoff Retry (Nov 20, 2025)

---

### November 20, 2025 - Configuration Management with JSON Persistence

**Status:** Accepted (Implemented)  
**Deciders:** Technical Team  
**Impact:** Medium

#### Context
Users need to configure LLM providers and API keys without editing code or environment variables. Configuration must:
- Persist across application restarts
- Be secure (API keys not exposed to frontend)
- Be easy to change via UI
- Support multiple environments (dev/prod)

#### Decision
Implement **JSON-based configuration persistence** with:
- Storage in OS-specific config directory (`~/.config/yantra/llm_config.json`)
- LLMConfigManager for all config operations
- Tauri commands for frontend access
- Sanitized config (boolean flags instead of actual keys)

**Config Structure:**
```json
{
  "primary_provider": "Claude",
  "claude_api_key": "sk-ant-...",
  "openai_api_key": "sk-proj-...",
  "max_retries": 3,
  "timeout_seconds": 30
}
```

#### Rationale

**Why JSON:**
- Human-readable for debugging
- Easy to edit manually if needed
- Native Rust serde support
- No additional dependencies

**Why OS Config Directory:**
- Standard location: `~/.config/yantra/` (macOS/Linux), `%APPDATA%\yantra\` (Windows)
- Proper permissions (user-only readable)
- Survives app reinstalls
- OS handles cleanup on user removal

**Why Sanitized Config:**
- Never send actual API keys to frontend
- Send boolean flags: `has_claude_key`, `has_openai_key`
- Frontend shows "✓ Configured" vs "Not configured"
- Security: keys only in backend memory

#### Implementation Details

**LLMConfigManager Methods:**
- `new()`: Load existing config or create default
- `set_primary_provider()`: Switch between Claude/OpenAI
- `set_claude_key()`: Store Claude API key
- `set_openai_key()`: Store OpenAI API key
- `clear_api_key()`: Remove specific key
- `get_sanitized_config()`: Return safe config for frontend
- `save()`: Persist to JSON file

**Tauri Commands:**
- `get_llm_config`: Retrieve sanitized config
- `set_llm_provider`: Change primary provider
- `set_claude_key`: Update Claude key
- `set_openai_key`: Update OpenAI key
- `clear_llm_key`: Remove key
- `set_llm_retry_config`: Adjust retry/timeout

**Frontend Integration:**
- TypeScript API wrapper in `src-ui/api/llm.ts`
- SolidJS settings component in `src-ui/components/LLMSettings.tsx`
- Password-masked input fields
- Real-time validation and feedback

#### Alternatives Considered

1. **Environment Variables**
   - Pros: Standard practice, secure
   - Cons: Requires restart to change, not user-friendly

2. **SQLite Database**
   - Pros: Structured queries, encryption
   - Cons: Overkill for simple config, added dependency

3. **TOML Configuration**
   - Pros: More expressive, comments
   - Cons: Less common, requires toml crate

4. **Encrypted Configuration**
   - Pros: Maximum security
   - Cons: Key management complexity, overkill for local app

#### Consequences

**Positive:**
- User-friendly configuration via UI
- No need to restart app when changing providers
- Secure (keys never leave backend)
- Standard location familiar to users
- Easy to backup/restore

**Negative:**
- Config file readable by user (but that's their machine)
- No encryption at rest (acceptable for local desktop app)
- Must handle file corruption gracefully
- Migration needed if config format changes

#### Security Considerations
- File permissions: 600 (owner read/write only)
- Keys never logged or sent to frontend
- Validation of input before saving
- Graceful handling of missing/corrupt config

#### Testing
- Config creation and loading (1 test)
- Provider switching (2 tests)
- API key management (1 test)
- Sanitization (1 test)
- Total: 4 tests passing

#### Related Decisions
- Multi-LLM Orchestration (Nov 20, 2025)
- Use Tauri for Desktop Framework (Nov 20, 2025)

---

## Decision Process

### When to Create a Decision Entry

Create a decision entry when:
- Choosing between significant architectural approaches
- Selecting major dependencies or technologies
- Changing existing decisions
- Making trade-offs with substantial impact
- Establishing project-wide standards or practices

### When NOT to Create a Decision Entry

Don't create entries for:
- Minor implementation details
- Obvious choices with no alternatives
- Temporary workarounds
- Personal coding preferences

---

## Decision Status Values

- **Proposed:** Under consideration, not yet decided
- **Accepted:** Decision made and being implemented
- **Superseded:** Replaced by a newer decision
- **Rejected:** Considered but not chosen

---

**Last Updated:** November 20, 2025  
**Next Update:** As decisions are made
