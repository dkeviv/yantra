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
