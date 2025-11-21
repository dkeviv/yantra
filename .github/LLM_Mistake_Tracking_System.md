# LLM Mistake Tracking & Learning System

**Version:** 1.0  
**Date:** November 20, 2025  
**Status:** Specified, Implementation in Week 7-8

---

## Executive Summary

**Problem:** LLMs (Claude, GPT-4) make repeated coding mistakes even after being corrected, reducing code quality and requiring multiple regeneration cycles.

**Solution:** Automated mistake detection, storage, and learning system that:
- Detects mistakes from test failures, security scans, and user corrections
- Stores patterns in vector database for semantic similarity search
- Injects learned patterns into prompts before generation
- Continuously improves over time

**Impact:** 
- 30-50% reduction in regeneration cycles
- Higher first-pass code quality
- Model-specific learning (Claude vs GPT-4)
- Automatic improvement without manual tracking

---

## Architecture Overview

### Hybrid Storage System

**Vector Database (ChromaDB):** Semantic pattern storage
- Store mistake descriptions with embeddings
- Fast k-NN similarity search (<100ms)
- Find "forgot await" when searching "async without await"
- Store code snippets with natural language context

**SQLite:** Structured metadata
- Model name, frequency, severity, timestamps
- Fast filtering and statistics
- Relationship tracking (patterns → occurrences)

### Data Flow

```
Code Generation Request
    ↓
Query Vector DB (top-5 similar past mistakes)
    ↓
Inject Patterns into System Prompt
    ↓
LLM Generates Code
    ↓
Run Tests + Security Scan
    ↓
Detect Failures → Extract Pattern → Store in DB
    ↓
Learning Loop Continues
```

---

## Automatic Detection Sources

### 1. Test Failure Detection

**Triggers:**
- pytest assertion errors
- Runtime exceptions (TypeError, AttributeError, etc.)
- Import errors
- Async/await errors

**Example:**
```python
# Generated code:
def fetch_user(user_id: int):
    return db.query(User).filter_by(id=user_id).first()

# Test fails:
RuntimeWarning: coroutine 'query' was never awaited

# Pattern extracted:
{
    "description": "Forgot 'async/await' for database operations",
    "code_snippet": "def fetch_user...",
    "category": "async",
    "severity": "major",
    "model": "claude-sonnet-4"
}
```

### 2. Security Scan Detection

**Triggers:**
- SQL injection vulnerabilities
- XSS vulnerabilities
- Hardcoded secrets
- Unsafe deserialization
- Missing input validation

**Example:**
```python
# Generated code:
query = f"SELECT * FROM users WHERE email = '{email}'"

# Semgrep detects:
SQL Injection vulnerability (CWE-89)

# Pattern extracted:
{
    "description": "SQL injection: f-string in query",
    "fix": "Use parameterized queries",
    "category": "security",
    "severity": "critical",
    "model": "gpt-4-turbo"
}
```

### 3. Chat Correction Monitoring

**Detection Patterns:**
- "no, that's wrong"
- "fix the bug"
- "you forgot to..."
- "should be X not Y"
- "don't use X, use Y instead"

**Example:**
```
User: "Create a file upload endpoint"
AI: [generates sync code]
User: "No that's wrong, file uploads should be async"
AI: [regenerates]

# Pattern extracted:
{
    "description": "File upload functions should be async",
    "context": "When generating file upload endpoints",
    "model": "claude-sonnet-4",
    "source": "user_correction"
}
```

---

## Pattern Storage Schema

### SQLite Schema

```sql
CREATE TABLE mistake_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vector_db_id TEXT NOT NULL,        -- Reference to ChromaDB
    model_name TEXT NOT NULL,          -- 'claude-sonnet-4' or 'gpt-4-turbo'
    error_signature TEXT NOT NULL,     -- Hash of error type + context
    category TEXT NOT NULL,            -- 'syntax', 'async', 'type', 'security'
    severity TEXT NOT NULL,            -- 'critical', 'major', 'minor'
    frequency INTEGER DEFAULT 1,       -- Occurrence count
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fix_applied BOOLEAN DEFAULT FALSE,
    user_corrections INTEGER DEFAULT 0,
    test_failures INTEGER DEFAULT 0,
    archived BOOLEAN DEFAULT FALSE,
    INDEX idx_model_category (model_name, category),
    INDEX idx_frequency (frequency DESC)
);

CREATE TABLE mistake_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER NOT NULL,
    project_path TEXT,
    file_path TEXT,
    generated_code TEXT,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES mistake_patterns(id)
);
```

### Vector Database Collections

**Collection: `llm_mistakes`**
- Embedded mistake descriptions
- Code snippets with context
- Fix descriptions
- Metadata: model, category, frequency

**Collection: `successful_fixes`**
- Embedded fix patterns that worked
- Before/after code examples
- Success metrics

---

## Pre-Generation Pattern Injection

### Enhanced System Prompt

**Before (Standard):**
```
Generate Python code following best practices.
Create a FastAPI endpoint for user authentication.
```

**After (With Learning):**
```
Generate Python code following best practices.

CRITICAL: Avoid these common mistakes:

1. ALWAYS use 'async def' for endpoint functions and 'await' for database calls
   Seen: 15 times | Model: claude-sonnet-4
   Bad:  def get_user(user_id: int):
             return db.query(User).get(user_id)
   Good: async def get_user(user_id: int):
             return await db.query(User).get(user_id)

2. ALWAYS validate request bodies with Pydantic models
   Seen: 12 times | Model: claude-sonnet-4
   Bad:  @app.post("/users")
         def create_user(data: dict):
   Good: @app.post("/users")
         async def create_user(data: UserCreate):

3. ALWAYS wrap database operations in try/except
   Seen: 9 times | Model: claude-sonnet-4
   ...

Create a FastAPI endpoint for user authentication.
```

### Retrieval Algorithm

```rust
async fn get_relevant_patterns(
    request_description: &str,
    model_name: &str,
    top_k: usize
) -> Vec<MistakePattern> {
    // 1. Vector similarity search
    let similar = vector_db.query(
        request_description,
        top_k * 2,  // Over-fetch for filtering
        min_similarity: 0.70
    ).await?;
    
    // 2. Filter by model
    let model_specific = similar
        .into_iter()
        .filter(|p| p.model_name == model_name)
        .collect();
    
    // 3. Rank by relevance score
    let ranked = rank_patterns(
        model_specific,
        factors: [frequency, recency, severity]
    );
    
    // 4. Return top-K
    ranked.into_iter().take(top_k).collect()
}
```

---

## Learning Loop

### Continuous Improvement Cycle

```
1. User Request
    ↓
2. Query Past Mistakes (Vector DB)
    ↓
3. Inject into Prompt
    ↓
4. Generate Code
    ↓
5. Run Tests
    ↓
6. If Fail: Extract & Store Pattern
    ↓
7. Regenerate with New Pattern
    ↓
8. Success → Update Statistics
```

### Pattern Evolution

**Initial Detection:**
```json
{
    "description": "Forgot await in async function",
    "frequency": 1,
    "code_snippet": "async def func(): db.query(...)"
}
```

**After 5 Occurrences:**
```json
{
    "description": "Forgot await in async function",
    "frequency": 5,
    "code_snippet": "async def func(): db.query(...)",
    "contexts": ["FastAPI endpoints", "database queries"],
    "fix_description": "Add 'await' before async calls"
}
```

**After 15 Occurrences (High Priority):**
```json
{
    "description": "Forgot await in async function",
    "frequency": 15,
    "severity": "major",  // Upgraded from "minor"
    "always_inject": true,  // Show in every generation
    "success_rate_after_injection": 0.93
}
```

---

## Privacy & Security

### Data Sanitization

**What to Store:**
- ✅ Error patterns (sanitized)
- ✅ Code structure (no business logic)
- ✅ Model name and metadata

**What NOT to Store:**
- ❌ Credentials, API keys, passwords
- ❌ Complete files (only relevant snippets)
- ❌ Business-specific logic
- ❌ User identifiable information

**Sanitization Process:**
```rust
fn sanitize_code_snippet(code: &str) -> String {
    code
        .replace_regex(r"['\"](.*?password.*?)['\""]", "***")
        .replace_regex(r"api[_-]?key\s*=\s*['\"].*?['\"]", "api_key=***")
        .replace_regex(r"(https?://)[^\s]+", "$1***")
        .truncate_to_relevant_lines(5)  // Keep only error context
        .replace_business_names()
}
```

### User Controls

- **Opt-out:** Disable mistake tracking entirely
- **Clear Data:** Delete all stored patterns
- **Export:** Download patterns for review
- **Category Filter:** Disable specific categories (e.g., security patterns)
- **Review:** Admin dashboard to view tracked patterns

---

## Performance Targets

| Operation | Target | Why |
|-----------|--------|-----|
| Pattern Retrieval | <100ms | Top-K vector search with filtering |
| Pattern Injection | <50ms | Build mistake context string |
| Detection | <200ms | Extract and store new pattern |
| Storage Size | <1MB / 100 patterns | With embeddings (384 dims) |
| Max Patterns per Generation | 5-10 | Balance context vs token cost |

---

## Implementation Timeline

### Week 7 (Jan 1-7, 2026)

**Day 1-2: Database Setup**
- Set up ChromaDB embedded mode
- Create SQLite schema
- Write migration scripts

**Day 3-4: Detection Module**
- Implement test failure detector
- Implement security scan detector
- Test pattern extraction

**Day 5-7: Storage & Retrieval**
- Build pattern storage module
- Build vector DB integration
- Build pattern retrieval with ranking

### Week 8 (Jan 8-15, 2026)

**Day 1-2: Pre-Generation Injection**
- Integrate retrieval into LLM generator
- Build mistake context builder
- Test end-to-end with Claude

**Day 3-4: Chat Monitoring**
- Implement correction detection in ChatPanel
- Parse conversation for corrections
- Test user correction flow

**Day 5-7: Testing & Optimization**
- End-to-end integration tests
- Performance optimization
- Privacy review and sanitization
- Documentation

---

## Success Metrics

**MVP (End of Week 8):**
- ✅ 3 detection sources working (tests, security, chat)
- ✅ <100ms pattern retrieval
- ✅ 20+ unique patterns stored
- ✅ 30% reduction in regeneration cycles

**Month 3:**
- ✅ 100+ unique patterns per model
- ✅ 50% reduction in regeneration cycles
- ✅ 95% first-pass code quality

**Month 6:**
- ✅ 500+ patterns with high coverage
- ✅ Cross-project learning enabled
- ✅ 98% first-pass code quality

---

## Future Enhancements (Post-MVP)

1. **Community Patterns**
   - Opt-in sharing of anonymized patterns
   - Download patterns from community
   - Contribute patterns to marketplace

2. **Active Learning**
   - Ask user to validate uncertain patterns
   - Confidence scoring for patterns
   - A/B testing of pattern effectiveness

3. **Cross-Project Learning**
   - Share patterns across user's projects
   - Identify organization-specific patterns
   - Team-wide pattern sharing

4. **LLM Fine-Tuning**
   - Export patterns as fine-tuning dataset
   - Periodic fine-tuning of local models
   - Measure improvement from fine-tuning

5. **Temporal Decay**
   - Reduce weight of old patterns over time
   - Archive patterns not seen in 6+ months
   - Adapt to evolving best practices

6. **Advanced Analytics**
   - Mistake dashboard (most common errors)
   - Model comparison (Claude vs GPT-4 error rates)
   - Project health score based on patterns

---

## References

**Implementation Files:**
- `src/learning/mod.rs` - Main module
- `src/learning/detector.rs` - Mistake detection
- `src/learning/storage.rs` - SQLite operations
- `src/learning/vector_db.rs` - ChromaDB integration
- `src/learning/retrieval.rs` - Pattern retrieval
- `src-ui/components/ChatPanel.tsx` - Chat monitoring

**Documentation:**
- Technical_Guide.md - Detailed implementation
- Decision_Log.md - Architecture decision (Nov 20, 2025)
- File_Registry.md - File tracking

**Related Systems:**
- GNN (dependency tracking)
- LLM Orchestrator (code generation)
- Testing Engine (failure detection)
- Security Scanner (vulnerability detection)

---

**Document Maintained By:** Vivek Durairaj  
**Last Updated:** November 20, 2025  
**Status:** Living document, updated as implementation progresses
