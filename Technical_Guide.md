# Yantra - Technical Guide

**Version:** MVP 1.0  
**Last Updated:** November 20, 2025  
**Audience:** Developers and Technical Contributors

---

## Overview

This guide provides detailed technical information about Yantra's architecture, implementation details, algorithms, and methodologies. It serves as a comprehensive reference for developers maintaining and extending the platform.

---

## Architecture Overview

### System Components

Yantra follows a layered architecture with five main components:

1. **User Interface Layer** - AI-first chat interface with code viewer and browser preview
2. **Orchestration Layer** - Multi-LLM management and routing
3. **Intelligence Layer** - GNN for dependency tracking and RAG for templates
4. **Validation Layer** - Testing, security scanning, and browser validation
5. **Integration Layer** - Git, file system, and external API connections

### Technology Stack

**Desktop Framework:**
- Tauri 1.5+ for cross-platform desktop application
- Chosen for: 600KB bundle size (vs 150MB Electron), lower memory footprint, Rust backend performance

**Frontend:**
- SolidJS 1.8+ for reactive UI
- Monaco Editor 0.44+ for code viewing
- TailwindCSS 3.3+ for styling
- Chosen for: Fastest reactive framework, smaller bundle, no virtual DOM overhead

**Backend (Rust):**
- Tokio 1.35+ for async runtime
- SQLite 3.44+ for GNN persistence
- petgraph 0.6+ for graph operations
- tree-sitter for code parsing
- Chosen for: Memory safety, fearless concurrency, zero-cost abstractions

---

## Component Implementation Details

### 1. Graph Neural Network (GNN) Engine

**Status:** 🔴 Not Implemented (Week 3-4)

#### Purpose
Track all code dependencies to ensure generated code never breaks existing functionality.

#### Implementation Approach

**Graph Structure:**
- **Nodes:** Functions, classes, variables, imports
- **Edges:** Calls, uses, imports, inherits, data flow
- **Storage:** SQLite with adjacency list representation

**Why This Approach:**
- Adjacency lists provide O(1) edge lookup
- SQLite enables persistence and incremental updates
- petgraph provides efficient graph algorithms

**Algorithm Overview:**

1. **Initial Graph Build**
   - Parse all Python files using tree-sitter
   - Extract AST nodes (functions, classes, variables)
   - Create graph nodes for each symbol
   - Detect relationships (calls, imports, inheritance)
   - Create graph edges for relationships
   - Store in SQLite

2. **Incremental Updates**
   - Watch for file changes
   - Parse only modified files
   - Update affected subgraph
   - Propagate changes to dependents
   - Target: <50ms per file change

3. **Dependency Lookup**
   - Query graph for symbol
   - Traverse edges (BFS/DFS)
   - Return all dependents
   - Target: <10ms

**Reference Files:**
- `src/gnn/mod.rs` - Main GNN module
- `src/gnn/parser.rs` - tree-sitter Python parser
- `src/gnn/graph.rs` - Graph data structures
- `src/gnn/persistence.rs` - SQLite integration
- `src/gnn/incremental.rs` - Incremental update logic

**Performance Targets:**
- Initial build: <5s for 10k LOC
- Incremental update: <50ms per file
- Dependency lookup: <10ms
- Memory usage: <100MB for 100k LOC

---

### 2. Multi-LLM Orchestration

**Status:** � 40% Complete (Week 5-6) - Foundation Ready ✅

#### Purpose
Coordinate multiple LLM providers (Claude Sonnet 4, GPT-4 Turbo) for code generation with automatic failover and circuit breaker protection.

#### Implementation Status

**Completed:**
- ✅ Claude Sonnet 4 API client with full HTTP integration
- ✅ OpenAI GPT-4 Turbo client with deterministic settings
- ✅ Multi-LLM orchestrator with state management
- ✅ Circuit breaker pattern with state machine
- ✅ Configuration management with JSON persistence
- ✅ Retry logic with exponential backoff
- ✅ Tauri commands for configuration
- ✅ Frontend UI for LLM settings
- ✅ 14 unit tests passing

**Pending:**
- 🔄 Context assembly from GNN
- 🔄 Code generation Tauri command
- 🔄 Response caching
- 🔄 Token usage tracking

#### Implementation Details

**Why Multi-LLM:**
- No single point of failure
- Quality improvement through validation
- Cost optimization (configurable primary provider)
- Best-of-breed approach

**Architecture:**

1. **API Clients** (`claude.rs`, `openai.rs`)
   - HTTP clients using reqwest with async/await
   - System prompt builder for coding tasks
   - User prompt builder with context injection
   - Code block extraction with language detection
   - Response parsing and error handling
   - Token usage tracking

2. **Circuit Breaker Pattern** (`orchestrator.rs`)
   - Three states: Closed (normal), Open (failing), HalfOpen (testing recovery)
   - Failure threshold: 3 consecutive failures
   - Cooldown period: 60 seconds
   - State transitions tracked with atomic operations
   - Thread-safe with Arc<RwLock<>>
   
   **State Machine:**
   ```
   Closed → (3 failures) → Open
   Open → (60s timeout) → HalfOpen
   HalfOpen → (success) → Closed
   HalfOpen → (failure) → Open
   ```

3. **Failover Mechanism** (`orchestrator.rs`)
   - Primary LLM: Configurable (Claude or OpenAI)
   - Secondary LLM: Automatic failover
   - Retry with exponential backoff: 100ms, 200ms, 400ms
   - Max retries: 3 per provider
   - If primary fails → try secondary
   - If both fail → return error to user

4. **Configuration System** (`config.rs`)
   - JSON persistence to OS-specific config directory
   - `~/.config/yantra/llm_config.json` on macOS/Linux
   - Secure API key storage (never exposed to frontend)
   - Sanitized config for UI (boolean flags only)
   - Settings: primary_provider, claude_api_key, openai_api_key, max_retries, timeout_seconds

5. **Frontend Integration**
   - `src-ui/api/llm.ts`: TypeScript API wrapper
   - `src-ui/components/LLMSettings.tsx`: Full settings UI
   - Provider selection toggle (Claude ↔ OpenAI)
   - Password-masked API key inputs
   - Status indicators (✓ Configured / Not configured)
   - Save/clear operations with validation

**Algorithm Flow:**

```
1. User requests code generation
2. Orchestrator checks circuit breaker state
3. If Closed → try primary LLM
4. If Open → skip to secondary LLM
5. If HalfOpen → test with current request
6. On failure → exponential backoff retry (3 attempts)
7. If still failing → update circuit breaker state
8. If primary exhausted → failover to secondary
9. Return result or comprehensive error
```

**Reference Files:**
- `src/llm/mod.rs` (105 lines) - Core types and module root
- `src/llm/claude.rs` (300+ lines) - Claude Sonnet 4 client
- `src/llm/openai.rs` (200+ lines) - OpenAI GPT-4 Turbo client
- `src/llm/orchestrator.rs` (280+ lines) - Circuit breaker + orchestration
- `src/llm/config.rs` (180+ lines) - Configuration management
- `src/llm/context.rs` (20 lines) - Placeholder for GNN context assembly
- `src/llm/prompts.rs` (10 lines) - Placeholder for prompt templates
- `src-ui/api/llm.ts` (60 lines) - TypeScript API bindings
- `src-ui/components/LLMSettings.tsx` (230+ lines) - Settings UI

**Performance Metrics:**
- Circuit breaker decision: <1ms (atomic operations)
- HTTP request timeout: 30s configurable
- Exponential backoff delays: 100ms, 200ms, 400ms
- Failover latency: ~100ms (immediate retry on secondary)
- Target response time: <3s (LLM API dependent)

**Configuration Options:**
```json
{
  "primary_provider": "Claude",
  "claude_api_key": "sk-ant-...",
  "openai_api_key": "sk-proj-...",
  "max_retries": 3,
  "timeout_seconds": 30
}
```

**Testing Coverage:**
- Circuit breaker state transitions (4 tests)
- API client prompt building (2 tests)
- Code block extraction (1 test)
- Configuration persistence (4 tests)
- Provider switching (2 tests)
- API key management (1 test)
- Total: 14 unit tests passing ✅

---

### 3. Code Generation Pipeline

**Status:** 🔴 Not Implemented (Week 5-6)

#### Purpose
Generate production-quality Python code from natural language with full dependency awareness.

#### Implementation Approach

**Pipeline Steps:**

1. **Intent Understanding**
   - Parse user input
   - Extract: action, target, constraints
   - Example: "Add user auth" → {action: add, target: auth, method: JWT}

2. **Context Assembly**
   - Query GNN for relevant code
   - Gather existing patterns
   - Collect related functions/classes
   - Build context window (<100ms)

3. **Prompt Construction**
   - Template: system prompt + context + user request
   - Include: code style guide, test requirements, security rules
   - Inject GNN context for dependency awareness

4. **Code Generation**
   - Call LLM with constructed prompt
   - Parse response
   - Extract generated code
   - Validate syntax

5. **Dependency Validation**
   - Check against GNN
   - Ensure no breaking changes
   - Verify all dependencies exist

**Why This Approach:**
- Context assembly ensures dependency awareness
- Structured prompts improve output quality
- GNN validation prevents breaking changes

**Reference Files:**
- `src/llm/generator.rs` - Code generation logic
- `src/llm/prompts.rs` - Prompt templates
- `src/llm/context.rs` - Context assembly
- `src/gnn/validator.rs` - Dependency validation

**Use Cases:**

**Use Case 1: Generate REST API**
```
Input: "Create GET /users/:id endpoint"
Context: Existing User model, Flask patterns, error handling style
Output: Endpoint code + tests (integration with existing codebase)
```

**Use Case 2: Add Business Logic**
```
Input: "Calculate shipping cost based on weight and distance"
Context: Existing shipping module, rate calculation patterns
Output: Function with type hints, docstrings, tests
```

---

### 4. Automated Testing Engine

**Status:** 🔴 Not Implemented (Week 5-6)

#### Purpose
Automatically generate and execute comprehensive unit and integration tests.

#### Implementation Approach

**Test Generation:**

1. **Analyze Generated Code**
   - Extract functions, classes, methods
   - Identify inputs and outputs
   - Detect edge cases
   - Find dependencies

2. **Generate Test Cases**
   - Happy path tests
   - Edge case tests (empty, null, max values)
   - Error condition tests
   - Integration tests for dependencies
   - Target: 90%+ coverage

3. **Test Execution**
   - Write tests to temporary file
   - Run pytest via subprocess
   - Parse output (JUnit XML)
   - Report results

**Why This Approach:**
- Comprehensive testing ensures quality
- Subprocess isolation prevents interference
- pytest is industry standard for Python

**Workflow:**

1. Code generated
2. Tests auto-generated
3. Tests executed (<30s)
4. Results parsed
5. If failures → regenerate code or tests
6. Repeat until 100% pass rate

**Reference Files:**
- `src/testing/mod.rs` - Main testing module
- `src/testing/generator.rs` - Test generation logic
- `src/testing/runner.rs` - pytest runner
- `src/testing/parser.rs` - Result parser

**Performance Targets:**
- Test generation: <5s
- Test execution: <30s for typical project
- Coverage: >90%
- Pass rate: 100% (mandatory)

---

### 5. Security Scanning

**Status:** 🔴 Not Implemented (Week 7)

#### Purpose
Automatically scan generated code for security vulnerabilities and auto-fix critical issues.

#### Implementation Approach

**Scanning Tools:**

1. **Semgrep with OWASP Rules**
   - Static analysis for security issues
   - SQL injection, XSS, CSRF, etc.
   - Custom rules for Python patterns

2. **Safety (Python Dependencies)**
   - Check for known vulnerabilities in packages
   - CVE database lookup

3. **TruffleHog Patterns**
   - Scan for hardcoded secrets
   - API keys, passwords, tokens
   - Regex patterns for common secrets

**Why This Approach:**
- Semgrep is fast and accurate
- OWASP rules cover common vulnerabilities
- Multi-layer scanning catches more issues

**Workflow:**

1. Code generated
2. Run Semgrep scan (<10s)
3. Run Safety check (<5s)
4. Run secret scan (<2s)
5. Parse results
6. Categorize by severity (critical, high, medium, low)
7. Auto-fix critical issues
8. Report to user

**Auto-Fix Examples:**

```python
# SQL Injection
Before: f"SELECT * FROM users WHERE id = {user_id}"
After:  "SELECT * FROM users WHERE id = ?", (user_id,)

# Hardcoded Secret
Before: api_key = "sk-1234567890"
After:  api_key = os.getenv("API_KEY")
```

**Reference Files:**
- `src/security/mod.rs` - Main security module
- `src/security/semgrep.rs` - Semgrep integration
- `src/security/safety.rs` - Safety checker
- `src/security/secrets.rs` - Secret scanner
- `src/security/autofix.rs` - Auto-fix logic

**Performance Targets:**
- Total scan time: <10s
- Auto-fix rate: >80% for critical issues
- False positive rate: <5%

---

### 6. Browser Integration (Chrome DevTools Protocol)

**Status:** 🔴 Not Implemented (Week 7)

#### Purpose
Validate UI code runs correctly in browser with no console errors.

#### Implementation Approach

**Technology:**
- Chrome DevTools Protocol (CDP)
- chromiumoxide Rust library
- Headless Chrome

**Why This Approach:**
- CDP provides full browser control
- Headless mode is fast and lightweight
- Can capture console errors and network issues

**Workflow:**

1. **Launch Browser**
   - Start headless Chrome
   - Connect via CDP

2. **Load Code**
   - Create temporary HTML file
   - Include generated JavaScript/CSS
   - Navigate browser to file

3. **Monitor Console**
   - Listen for console messages
   - Capture errors, warnings
   - Track JavaScript exceptions

4. **Capture Results**
   - Screenshot for visual validation
   - Console output
   - Network errors
   - Performance metrics

5. **Generate Fixes**
   - If errors found → send to LLM
   - LLM generates fix
   - Reload and validate
   - Repeat until clean

**Reference Files:**
- `src/browser/mod.rs` - Main browser module
- `src/browser/cdp.rs` - CDP client
- `src/browser/monitor.rs` - Console monitoring
- `src/browser/validator.rs` - Validation logic

**Use Cases:**

**Use Case 1: Form Validation**
```
Generated: Login form with validation
Browser test:
- Form renders ✓
- Input validation works ✓
- Submit button functions ✓
- No console errors ✓
```

**Use Case 2: API Integration**
```
Generated: Fetch and display user data
Browser test:
- API call succeeds ✓
- Data displays correctly ✓
- Error handling works ✓
- No CORS issues ✓
```

---

### 7. Git Integration (Model Context Protocol)

**Status:** 🔴 Not Implemented (Week 7)

#### Purpose
Automatically commit validated, tested code to Git repository.

#### Implementation Approach

**Technology:**
- Model Context Protocol (MCP) for Git operations
- git2-rs (libgit2 Rust bindings)

**Why This Approach:**
- MCP provides standardized Git interface
- libgit2 is fast and reliable
- Native Git operations without shell commands

**Workflow:**

1. **Pre-commit Validation**
   - Ensure all tests pass
   - Ensure security scan clean
   - Ensure GNN validation complete

2. **Generate Commit Message**
   - Use Conventional Commits format
   - LLM generates descriptive message
   - Format: `type(scope): description`
   - Examples:
     - `feat(auth): Add JWT authentication`
     - `fix(reports): Correct date formatting`
     - `refactor(user): Implement dependency injection`

3. **Commit**
   - Stage changed files
   - Create commit with message
   - Handle conflicts if any

4. **Push**
   - Push to remote (optional)
   - Report success/failure

**Reference Files:**
- `src/git/mod.rs` - Main Git module
- `src/git/mcp.rs` - MCP integration
- `src/git/commit.rs` - Commit logic
- `src/git/message.rs` - Message generation

**Commit Message Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: feat, fix, refactor, test, docs, chore

---

### 8. User Interface Implementation

**Status:** 🔴 Not Implemented (Week 1-2)

#### Purpose
Provide AI-first interface for user interaction with code viewing and browser preview.

#### Implementation Approach

**Technology:**
- SolidJS 1.8+ for reactive UI
- Monaco Editor for code viewing
- TailwindCSS for styling
- WebSockets for real-time updates

**Why This Approach:**
- SolidJS is fastest reactive framework
- Monaco is VS Code's editor (industry standard)
- TailwindCSS enables rapid UI development
- WebSockets provide instant updates

**Component Structure:**

1. **Main Layout** (App.tsx)
   - 3-panel grid layout
   - Responsive design
   - Panel resizing

2. **Chat Panel** (60% width)
   - Message list (chat history)
   - Input field
   - Send button
   - Loading indicators
   - Progress updates

3. **Code Viewer** (25% width)
   - Monaco editor instance
   - Python syntax highlighting
   - Line numbers
   - Read-only mode
   - File tabs

4. **Browser Preview** (15% width)
   - Iframe for preview
   - Refresh button
   - Console output display
   - Error highlighting

**State Management:**
- SolidJS stores for global state
- Signals for reactive updates
- Context for shared data

**Reference Files:**
- `src-ui/App.tsx` - Main application
- `src-ui/components/ChatPanel.tsx` - Chat interface
- `src-ui/components/CodeViewer.tsx` - Monaco integration
- `src-ui/components/BrowserPreview.tsx` - Preview pane
- `src-ui/stores/appStore.ts` - Application state
- `src-ui/styles/index.css` - TailwindCSS styles

---

## Data Flow

### Complete User Interaction Flow

```
User Input (Chat)
    ↓
Intent Parsing
    ↓
GNN Context Query ←→ Graph Database (SQLite)
    ↓
Prompt Construction
    ↓
LLM Orchestrator → Claude API / GPT-4 API
    ↓
Code Generation
    ↓
Dependency Validation ←→ GNN
    ↓
Test Generation
    ↓
Test Execution (pytest)
    ↓
Security Scan (Semgrep/Safety)
    ↓
Browser Validation (CDP) ←→ Headless Chrome
    ↓
Git Commit ←→ Git Repository
    ↓
Success Response (Chat)
```

---

## Database Schema

### SQLite Schema for GNN

**Status:** 🔴 Not Implemented (Week 3-4)

```sql
-- Nodes table
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,  -- 'function', 'class', 'variable', 'import'
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    signature TEXT,
    docstring TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edges table
CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    type TEXT NOT NULL,  -- 'calls', 'uses', 'imports', 'inherits'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

-- Indexes for performance
CREATE INDEX idx_nodes_name ON nodes(name);
CREATE INDEX idx_nodes_file ON nodes(file_path);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
```

---

## Performance Optimization Strategies

### GNN Performance

**Target:** <5s for 10k LOC, <50ms incremental updates

**Strategies:**

1. **Incremental Updates**
   - Parse only changed files
   - Update only affected subgraph
   - Use file watching for triggers

2. **Efficient Queries**
   - Index all common lookups
   - Use adjacency list for O(1) edge lookup
   - Cache frequently accessed paths

3. **Parallel Processing**
   - Parse multiple files concurrently
   - Use Tokio for async operations
   - Leverage Rust's fearless concurrency

### LLM Performance

**Target:** <3s response time

**Strategies:**

1. **Response Caching**
   - Cache based on prompt hash
   - 24-hour TTL
   - ~40% cache hit rate expected

2. **Context Optimization**
   - Send only relevant code
   - Limit context to 4000 tokens
   - Prioritize recent and related code

3. **Streaming Responses**
   - Stream LLM output
   - Show progress to user
   - Faster perceived performance

---

## Error Handling Strategy

### Graceful Degradation

1. **LLM Failures**
   - Fallback to secondary LLM
   - If both fail → show error, allow manual intervention

2. **Test Failures**
   - Show failed tests
   - Regenerate code
   - Max 3 attempts, then ask user

3. **Security Issues**
   - Auto-fix critical issues
   - Report non-fixable issues
   - Block commit if critical issues remain

4. **GNN Errors**
   - Rebuild graph if corrupted
   - Use cached version if available
   - Proceed without GNN as last resort (warn user)

---

## Testing Strategy

### Unit Tests (90%+ Coverage)

**Reference Files:**
- `src/gnn/tests.rs` - GNN unit tests
- `src/llm/tests.rs` - LLM unit tests
- `src/testing/tests.rs` - Testing module tests
- `src/security/tests.rs` - Security tests

### Integration Tests

**Reference Files:**
- `tests/integration/gnn_integration_test.rs`
- `tests/integration/llm_integration_test.rs`
- `tests/integration/end_to_end_test.rs`

### Performance Tests

**Reference Files:**
- `benches/gnn_benchmark.rs`
- `benches/llm_benchmark.rs`

---

## Security Considerations

### Data Privacy

- User code stays on local machine
- Only prompts/code sent to LLM APIs
- No telemetry without consent
- Encrypted API communication (HTTPS)

### Dependency Security

- Regular dependency audits
- Auto-update security patches
- Minimal dependency footprint

### Code Execution Safety

- Sandboxed test execution
- No arbitrary code execution
- File system access limited to project folder

---

## Deployment Architecture

### Desktop Application

**Packaging:**
- Tauri bundler for platform-specific builds
- macOS: .dmg installer
- Windows: .exe installer
- Linux: .AppImage

**Auto-updates:**
- Built-in Tauri updater
- Check for updates on launch
- Background download
- User approval required

---

## Future Technical Improvements (Post-MVP)

### Phase 2 (Months 3-4)
- Workflow execution runtime
- External API schema tracking
- Webhook server

### Phase 3 (Months 5-8)
- Multi-language parsers (JavaScript, TypeScript)
- Cross-language dependency tracking
- Playwright browser automation

### Phase 4 (Months 9-12)
- Distributed GNN (sharding)
- Advanced caching strategies
- Plugin architecture

---

## Development Setup

### Prerequisites

```bash
# Rust 1.74+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js 18+
nvm install 18

# Python 3.11+ (for testing)
pyenv install 3.11
```

### Build Commands

```bash
# Install dependencies
cargo build
cd src-ui && npm install

# Run in development
npm run tauri dev

# Run tests
cargo test
cd src-ui && npm test

# Build production
npm run tauri build
```

---

## Monitoring and Observability

### Metrics to Track

1. **Performance Metrics**
   - GNN build time
   - GNN query time
   - LLM response time
   - Test execution time

2. **Quality Metrics**
   - Test pass rate (target: 100%)
   - Code coverage (target: >90%)
   - Security scan results

3. **Usage Metrics**
   - Code generation requests
   - Test execution count
   - Security issues found/fixed

---

### 7. LLM Mistake Tracking & Learning System

**Status:** 🟡 Specified, Not Implemented (Week 7-8)

#### Purpose
Automatically detect, store, and learn from LLM coding mistakes to prevent repeated errors across sessions and improve code generation quality over time.

#### Problem Statement
LLMs make repeated mistakes:
- Same async/await errors across sessions
- Model-specific patterns (Claude vs GPT-4 have different failure modes)
- No memory of previous corrections
- Manual tracking doesn't scale

#### Implementation Approach

**Architecture: Hybrid Storage System**

**1. Vector Database (ChromaDB) - Semantic Pattern Storage**

Store mistake patterns with embeddings for semantic similarity search:

```rust
// Mistake pattern in Vector DB
struct MistakePattern {
    description: String,          // "Forgot to add 'await' keyword for async functions"
    code_snippet: String,         // Example of buggy code
    context: String,              // "When generating FastAPI endpoints"
    fix_description: String,      // How to fix it
    model_name: String,           // "claude-sonnet-4"
    embedding: Vec<f32>,          // Generated by all-MiniLM-L6-v2
}
```

**Collections:**
- `llm_mistakes`: Embedded mistake descriptions with code examples
- `successful_fixes`: Embedded fix patterns that worked

**Why Vector DB:**
- Semantic search: "forgot await" matches "async without await"
- Find similar issues even with different wording
- Store code context with natural language descriptions
- Fast k-NN search (<100ms for top-K)

**2. SQLite - Structured Metadata**

Store mistake metadata for filtering and statistics:

```sql
CREATE TABLE mistake_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vector_db_id TEXT NOT NULL,        -- Reference to ChromaDB entry
    model_name TEXT NOT NULL,          -- 'claude-sonnet-4' or 'gpt-4-turbo'
    error_signature TEXT NOT NULL,     -- Hash of error type + context
    category TEXT NOT NULL,            -- 'syntax', 'async', 'type', 'security', etc.
    severity TEXT NOT NULL,            -- 'critical', 'major', 'minor'
    frequency INTEGER DEFAULT 1,       -- How many times seen
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fix_applied BOOLEAN DEFAULT FALSE,
    user_corrections INTEGER DEFAULT 0, -- How many times user corrected
    test_failures INTEGER DEFAULT 0,   -- How many times caught by tests
    INDEX idx_model_category (model_name, category),
    INDEX idx_frequency (frequency DESC),
    INDEX idx_last_seen (last_seen DESC)
);

CREATE TABLE mistake_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER NOT NULL,
    project_path TEXT,                 -- Which project
    file_path TEXT,                    -- Which file
    generated_code TEXT,               -- What code was generated
    error_message TEXT,                -- Error from test/scan
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES mistake_patterns(id)
);
```

**3. Automatic Detection Sources**

**A. Test Failure Detection**
```rust
// In testing engine
async fn on_test_failure(test_result: TestResult, generated_code: &str, model: &str) {
    let pattern = extract_mistake_pattern(
        test_result.error_message,
        test_result.failed_assertion,
        generated_code,
        model
    );
    
    mistake_tracker.record(pattern).await;
}
```

**Detection triggers:**
- pytest assertion errors
- Runtime exceptions
- Type errors
- Import errors

**B. Security Scan Detection**
```rust
// In security scanner
async fn on_security_issue(finding: SecurityFinding, code: &str, model: &str) {
    if finding.severity == "critical" || finding.severity == "high" {
        let pattern = MistakePattern {
            description: format!("Security: {}", finding.description),
            code_snippet: finding.vulnerable_code,
            category: "security",
            severity: finding.severity,
            model_name: model.to_string(),
            ...
        };
        
        mistake_tracker.record(pattern).await;
    }
}
```

**Detection triggers:**
- SQL injection patterns
- XSS vulnerabilities
- Hardcoded secrets
- Unsafe deserialization

**C. Chat Correction Monitoring**
```typescript
// In ChatPanel.tsx
async function detectUserCorrection(messages: Message[]) {
    const correctionPatterns = [
        /no,? that'?s? wrong/i,
        /fix (the|that) (bug|error|issue)/i,
        /you forgot to/i,
        /should be .+ not .+/i,
        /don'?t use .+, use .+ instead/i
    ];
    
    for (const msg of messages) {
        if (msg.role === 'user' && containsCorrection(msg.content)) {
            // Extract what was wrong from conversation context
            const previousCode = getPreviousGeneratedCode(messages, msg);
            const correction = extractCorrection(msg.content);
            
            await api.recordMistake({
                description: correction.description,
                code_snippet: previousCode,
                model: correction.model,
                source: 'user_correction'
            });
        }
    }
}
```

**4. Pre-Generation Pattern Injection**

**Workflow:**
```
User Request: "Create FastAPI endpoint for /users"
    ↓
Query Vector DB: Get top-5 similar mistakes for "claude-sonnet-4" + "FastAPI"
    ↓
Retrieved Patterns:
    1. "Forgot async/await in endpoint functions" (similarity: 0.92)
    2. "Missing Pydantic model validation" (similarity: 0.87)
    3. "No error handling for database operations" (similarity: 0.81)
    ↓
Construct Enhanced System Prompt:
    """
    Generate FastAPI endpoint code following best practices.
    
    CRITICAL: Avoid these common mistakes:
    
    1. ALWAYS use 'async def' for endpoint functions and 'await' for database calls
       Bad:  def get_user(user_id: int):
                   return db.query(User).get(user_id)
       Good: async def get_user(user_id: int):
                   return await db.query(User).get(user_id)
    
    2. ALWAYS validate request bodies with Pydantic models
       Bad:  @app.post("/users")
             def create_user(data: dict):
       Good: @app.post("/users")
             async def create_user(data: UserCreate):
    
    3. ALWAYS wrap database operations in try/except
       ...
    """
    ↓
Send to LLM (Claude)
    ↓
Generate Code (with mistake context)
```

**Implementation:**
```rust
// In LLM generator
async fn generate_code_with_learning(
    request: &CodeGenRequest,
    model: &str
) -> Result<GeneratedCode> {
    // 1. Query vector DB for similar past mistakes
    let relevant_mistakes = mistake_retrieval::query(
        &request.description,
        model,
        top_k = 5,
        min_similarity = 0.75
    ).await?;
    
    // 2. Build mistake context for prompt
    let mistake_context = build_mistake_context(&relevant_mistakes);
    
    // 3. Inject into system prompt
    let enhanced_prompt = format!(
        "{}\n\n{}\n\n{}",
        BASE_SYSTEM_PROMPT,
        mistake_context,
        request.user_prompt
    );
    
    // 4. Generate code with enhanced prompt
    let code = llm_client.generate(&enhanced_prompt).await?;
    
    Ok(code)
}

fn build_mistake_context(mistakes: &[MistakePattern]) -> String {
    let mut context = String::from("CRITICAL: Avoid these common mistakes:\n\n");
    
    for (i, mistake) in mistakes.iter().enumerate() {
        context.push_str(&format!(
            "{}. {}\n   Frequency: {} occurrences\n   Example:\n{}\n\n",
            i + 1,
            mistake.description,
            mistake.frequency,
            indent_code(&mistake.code_snippet, 3)
        ));
    }
    
    context
}
```

**5. Learning Loop**

```
Code Generation
    ↓
Run Tests
    ↓
Test Fails? → Extract Error Pattern → Store in Vector DB + SQLite
    ↓                                    ↓
Test Passes                         Increment Frequency
    ↓                                    ↓
Security Scan                       Update Last Seen
    ↓                                    ↓
Vulnerability? → Extract Pattern → Store in Vector DB + SQLite
    ↓                                    ↓
Clean Code                          Increment Frequency
    ↓
Commit to Git
```

**6. Pattern Maintenance**

```rust
// Periodic cleanup and optimization
async fn maintain_patterns() {
    // Archive old patterns (>6 months, frequency < 3)
    db.execute("
        UPDATE mistake_patterns 
        SET archived = TRUE 
        WHERE last_seen < date('now', '-6 months') 
        AND frequency < 3
    ").await?;
    
    // Merge similar patterns
    let duplicates = find_similar_patterns(similarity_threshold = 0.95).await?;
    for (pattern1, pattern2) in duplicates {
        merge_patterns(pattern1, pattern2).await?;
    }
    
    // Recompute embeddings for updated patterns
    reindex_vector_db().await?;
}
```

#### Why This Approach

**Vector DB for Semantic Search:**
- Captures meaning, not just keywords
- "Forgot to close database connection" matches "connection not closed properly"
- Handles variations in error descriptions
- Stores code context with natural language

**SQLite for Metadata:**
- Fast filtering by model, category, frequency
- Track statistics (occurrences, corrections)
- Efficient indexing for queries
- Relational data (patterns → occurrences)

**Automatic Detection:**
- Scales better than manual tracking
- Real-time learning from failures
- No human annotation needed
- Catches all error types

**Pre-Generation Injection:**
- Prevents mistakes before they happen
- Context window more efficient than fine-tuning
- Model-agnostic (works with any LLM)
- Immediate effect (no retraining)

#### Reference Files

**Backend (Rust):**
- `src/learning/mod.rs` - Main learning module
- `src/learning/detector.rs` - Mistake detection logic
- `src/learning/storage.rs` - SQLite operations
- `src/learning/vector_db.rs` - ChromaDB integration
- `src/learning/retrieval.rs` - Pattern retrieval and ranking
- `src/learning/maintenance.rs` - Pattern cleanup and optimization

**Frontend (SolidJS):**
- `src-ui/components/ChatPanel.tsx` - Chat monitoring for corrections
- `src-ui/stores/mistakeStore.ts` - Mistake tracking state
- `src-ui/components/MistakeDashboard.tsx` - View tracked patterns (admin)

**Database:**
- `migrations/007_mistake_patterns.sql` - SQLite schema
- `chroma_collections/llm_mistakes/` - ChromaDB storage

#### Performance Targets

- **Pattern Retrieval:** <100ms for top-K vector search
- **Storage:** <1MB per 100 patterns (with embeddings)
- **Injection Overhead:** <50ms to build mistake context
- **Max Patterns per Generation:** 5-10 (balance context vs tokens)
- **Detection Latency:** <200ms to extract and store pattern

#### Use Cases

**Use Case 1: Async/Await Pattern**
```python
# LLM generates (Claude):
def fetch_user(user_id: int):
    return db.query(User).filter_by(id=user_id).first()

# Test fails: RuntimeWarning: coroutine 'query' was never awaited
# Pattern extracted and stored

# Next generation for similar task:
# System prompt now includes:
# "ALWAYS use 'async def' and 'await' for database operations"

# LLM generates (Claude):
async def fetch_user(user_id: int):
    return await db.query(User).filter_by(id=user_id).first()
# ✅ Test passes
```

**Use Case 2: SQL Injection**
```python
# LLM generates (GPT-4):
query = f"SELECT * FROM users WHERE email = '{email}'"

# Security scan detects SQL injection
# Pattern extracted and stored

# Next generation for similar task:
# System prompt includes:
# "NEVER use f-strings for SQL queries. Use parameterized queries."

# LLM generates (GPT-4):
query = "SELECT * FROM users WHERE email = ?"
cursor.execute(query, (email,))
# ✅ Security scan passes
```

**Use Case 3: User Correction**
```
User: "Create a function to upload files"
AI: [generates sync code]
User: "No that's wrong, it should be async"
AI: [regenerates with async]

# Chat monitor detects correction
# Pattern stored: "File upload functions should be async"

# Next file upload task:
# System prompt includes the learned pattern
# ✅ Generates async code on first try
```

#### Privacy & Security Considerations

**Data to Store:**
- ✅ Error patterns (sanitized)
- ✅ Code structure (no business logic)
- ✅ Model name and metadata
- ❌ Sensitive data (credentials, API keys)
- ❌ Complete files (only snippets)
- ❌ User identifiable information

**Sanitization:**
```rust
fn sanitize_code_snippet(code: &str) -> String {
    let sanitized = code
        .replace_credentials()      // Remove API keys, passwords
        .replace_business_logic()   // Replace with placeholders
        .truncate_to_relevant();    // Keep only error context
    
    sanitized
}
```

**User Control:**
- Opt-out of mistake tracking
- Clear stored patterns
- Export patterns for review
- Disable specific pattern categories

#### Future Enhancements (Post-MVP)

1. **Cross-Project Learning:** Share patterns across user's projects
2. **Community Patterns:** Opt-in sharing of anonymized patterns
3. **Pattern Marketplace:** Download common patterns from community
4. **LLM Fine-Tuning:** Export patterns as fine-tuning dataset
5. **Confidence Scoring:** Weight patterns by success rate
6. **Temporal Decay:** Reduce weight of old patterns
7. **Active Learning:** Ask user to validate uncertain patterns

---

## Contributing Guidelines

### Code Style

**Rust:**
- Run `cargo clippy --` for linting
- Run `cargo fmt` for formatting
- Follow Rust API guidelines

**Frontend:**
- Run `npm run lint` for ESLint
- Run `npm run format` for Prettier
- Follow SolidJS patterns

### Testing Requirements

- All new code must have tests
- Maintain >90% coverage
- All tests must pass

### Documentation

- Update Technical_Guide.md for architectural changes
- Update File_Registry.md for new files
- Update Decision_Log.md for design decisions

---

**Last Updated:** November 20, 2025  
**Next Major Update:** After Week 2 (Foundation Complete)
