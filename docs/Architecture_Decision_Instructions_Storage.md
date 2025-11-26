# Architecture Decision: GNN vs Vector DB for Project Instructions

**Date:** November 24, 2025  
**Decision Type:** Core Architecture  
**Status:** 🔴 Proposed - Needs Final Decision  
**Impact:** HIGH - Affects performance, scalability, and feature capabilities

---

## The Question

> "For project instructions should we use GNN or vector db? Explain the architecture decision"

---

## TL;DR - Recommendation

**Use BOTH in a Hybrid Architecture:**
- **GNN (Primary):** For scope-based matching and structural relationships
- **Vector DB (Secondary):** For semantic similarity and learning from violations

**Why Hybrid?** Each solves different problems that project instructions require.

---

## Deep Analysis

### Problem Space: What Project Instructions Need

Project instructions have **two distinct dimensions**:

1. **Structural/Scope-Based Matching** (WHERE rules apply)
   - "This security rule applies to all files in `auth/` directory"
   - "This code style applies to all API endpoints"
   - "This test requirement applies to all public functions"
   - **Nature:** Deterministic, graph-based, exact matching

2. **Semantic/Content-Based Matching** (WHAT rules are relevant)
   - "User is adding authentication → Security rules about passwords are relevant"
   - "Code violates 'no plaintext passwords' → Find similar past violations"
   - "Suggest new instruction based on repeated validation failures"
   - **Nature:** Probabilistic, similarity-based, fuzzy matching

---

## Option 1: GNN Only (Pure Structural Approach)

### How It Would Work

```rust
// Instructions as nodes in GNN
Instruction Node {
  id: "sec-001",
  type: SecurityRule,
  rule: "No plaintext passwords",
  scope: Global,
}

// Relationships as edges
Instruction --[AppliesTo]--> FileNode("auth.py")
Instruction --[ViolatedBy]--> FunctionNode("register_user")
Instruction --[RelatedTo]--> Instruction("use-bcrypt")
```

**Context Injection:**
```rust
// When generating code for auth.py:
// 1. Find FileNode for auth.py
// 2. Traverse AppliesTo edges backward
// 3. Get all connected Instruction nodes
// 4. Inject into prompt
```

### Advantages ✅

1. **Guaranteed Injection** - Graph traversal ensures no instruction is missed
2. **Fast Scope Matching** - O(1) to O(log n) lookups with proper indexing
3. **Structural Relationships** - Can model instruction dependencies
4. **Zero False Positives** - Exact scope matching (no semantic ambiguity)
5. **Existing Infrastructure** - Already have GNN for code dependencies
6. **Deterministic** - Same input always produces same output

### Disadvantages ❌

1. **No Semantic Understanding** - Can't find "similar" instructions
   - "No plaintext passwords" won't match "Never store passwords unencrypted"
   
2. **Rigid Scope Matching** - Must predefine all scopes
   - What if user asks about "security in login flow"? No graph path exists.
   
3. **Learning is Hard** - Can't automatically suggest new instructions
   - Would need manual rule creation for every pattern
   
4. **Violation Similarity** - Can't find similar past violations
   - "password = input()" won't match "user.pwd = request.data"
   
5. **Context Awareness Limited** - Relies on predefined relationships
   - Can't infer that "JWT token" task needs "security" instructions

### Performance

- **Scope lookup:** <10ms (GNN graph traversal)
- **Memory:** ~100 bytes per instruction node
- **Scalability:** Excellent (10,000+ instructions no problem)

---

## Option 2: Vector DB Only (Pure Semantic Approach)

### How It Would Work

```python
# Instructions embedded in ChromaDB
{
  "id": "sec-001",
  "rule": "Never store passwords in plain text. Always use bcrypt or argon2.",
  "examples": ["password = input()", "user.password = form.data"],
  "embedding": [0.123, -0.456, ...],  # 384-dim vector
  "metadata": {
    "type": "SecurityRule",
    "severity": "Error",
    "violation_count": 5
  }
}
```

**Context Injection:**
```python
# When generating code for auth.py with intent "add user registration":
query_embedding = embed("add user registration with password")
similar_instructions = vector_db.search(query_embedding, top_k=5)
# Returns: password security rules, validation rules, etc.
```

### Advantages ✅

1. **Semantic Matching** - Finds relevant instructions even with different wording
   - "add login" → finds password security rules automatically
   
2. **Learning from Patterns** - Can cluster similar violations
   - Automatically group "forgot await" errors across different contexts
   
3. **Fuzzy Scope** - No need to predefine exact scopes
   - "security in authentication flow" finds relevant rules via similarity
   
4. **Suggestion Engine** - Can suggest new instructions based on embeddings
   - Cluster violations → generate instruction candidate
   
5. **Natural Language Queries** - User intent directly maps to instructions
   - "What security rules apply here?" → semantic search

### Disadvantages ❌

1. **No Guaranteed Coverage** - Might miss instructions due to poor embeddings
   - If "password" and "credential" embed differently, rules could be missed
   
2. **False Positives** - Semantic similarity can be misleading
   - "API rate limiting" might match "API authentication" incorrectly
   
3. **Performance Overhead** - Embedding generation + vector search
   - 50-100ms per search vs <10ms for GNN
   
4. **Non-Deterministic** - Same query can return different results
   - Embedding models can drift, top-K can vary
   
5. **Scope Enforcement Weak** - Hard to enforce "ONLY files in auth/ directory"
   - Similarity search doesn't understand hierarchical file structures
   
6. **Requires Embedding Model** - Additional dependency
   - Need to ship all-MiniLM-L6-v2 or similar (80MB model)

### Performance

- **Semantic search:** 50-100ms (embedding + k-NN search)
- **Memory:** ~1KB per instruction (embedding + metadata)
- **Scalability:** Good (1,000-10,000 instructions, slower beyond)

---

## Option 3: Hybrid Architecture (RECOMMENDED)

### The Key Insight

**Instructions have TWO orthogonal dimensions:**

1. **Scope (Structure)** → GNN handles this perfectly
2. **Semantic Relevance** → Vector DB handles this perfectly

**Don't choose one - use both where they're strongest!**

### Architecture Design

```rust
pub struct InstructionStore {
    gnn: GNNEngine,           // For scope-based matching
    vector_db: ChromaDB,      // For semantic similarity
    sqlite: SQLite,           // For metadata and stats
}

pub struct ProjectInstruction {
    id: String,
    rule: String,
    
    // GNN properties (structural)
    scope: InstructionScope,  // Global, Directory, FilePattern, Module
    applies_to: Vec<NodeId>,  // Graph nodes this applies to
    
    // Vector DB properties (semantic)
    embedding: Option<Vec<f32>>,
    semantic_tags: Vec<String>,  // ["security", "password", "authentication"]
    
    // Metadata (SQLite)
    priority: u8,
    violation_count: usize,
    compliance_count: usize,
}
```

### Workflow: Context Assembly

```rust
async fn find_applicable_instructions(
    engine: &InstructionStore,
    context: &CodeGenerationContext,
) -> Vec<ProjectInstruction> {
    // PHASE 1: GNN - Scope-based filtering (guaranteed coverage)
    let scope_matched = engine.gnn.find_instructions_by_scope(
        context.file_path,
        context.target_node,
    );
    // Result: All instructions that MUST apply based on scope
    // Time: <10ms
    
    // PHASE 2: Vector DB - Semantic enhancement (relevance ranking)
    let query = format!(
        "{}. File: {}. Context: {}",
        context.intent,
        context.file_path,
        context.description
    );
    let semantic_matched = engine.vector_db.search(
        &query,
        top_k: 20,
    );
    // Result: Instructions semantically similar to user intent
    // Time: ~50ms
    
    // PHASE 3: Merge and deduplicate
    let mut all_instructions = scope_matched;
    all_instructions.extend(
        semantic_matched.into_iter()
            .filter(|inst| !all_instructions.contains(inst))
    );
    
    // PHASE 4: Prioritize by violation history + relevance
    all_instructions.sort_by(|a, b| {
        let priority_cmp = b.priority.cmp(&a.priority);
        if priority_cmp != Ordering::Equal {
            return priority_cmp;
        }
        
        // Frequently violated rules get higher priority
        let violation_ratio_a = a.violation_count as f64 / (a.compliance_count + 1) as f64;
        let violation_ratio_b = b.violation_count as f64 / (b.compliance_count + 1) as f64;
        violation_ratio_b.partial_cmp(&violation_ratio_a).unwrap()
    });
    
    // Return top-N to fit token budget (30% of context)
    all_instructions.truncate(compute_max_instructions(token_budget));
    all_instructions
}
```

### When to Use Each Component

| Use Case | Primary Tool | Secondary Tool | Reason |
|----------|--------------|----------------|--------|
| **"Find rules for auth/ directory"** | GNN | - | Exact scope matching |
| **"Find rules related to 'login'"** | Vector DB | - | Semantic similarity |
| **"Find rules for UserService.login()"** | GNN | Vector DB | Node-specific + semantic context |
| **"Suggest new rule from violations"** | Vector DB | - | Clustering similar patterns |
| **"Validate rule applies to file"** | GNN | - | Graph relationship check |
| **"Find similar past violations"** | Vector DB | - | Semantic similarity search |
| **"Get all Global rules"** | GNN | - | Scope query |
| **"What security rules apply here?"** | GNN (scope) | Vector DB (semantic) | Both dimensions needed |

### Complete Example: Authentication Endpoint

```rust
// User request: "Add user registration endpoint with password"
// File: src/api/auth.py
// Function: register_user()

// STEP 1: GNN scope matching (guaranteed coverage)
let gnn_results = gnn.find_instructions(&[
    InstructionScope::Global,                    // All global rules
    InstructionScope::Directory("src/api"),      // All API rules
    InstructionScope::FilePattern("**/auth.py"), // All auth file rules
    InstructionScope::CodeType(NodeType::Function), // All function rules
]);
// Results: [
//   "100% test coverage" (Global),
//   "API endpoints need rate limiting" (Directory: src/api),
//   "Auth files require extra security review" (FilePattern: auth.py),
//   "All functions need docstrings" (CodeType: Function)
// ]
// Time: 8ms

// STEP 2: Vector DB semantic search (relevance ranking)
let embedding = embed("user registration endpoint with password authentication");
let vector_results = vector_db.search(embedding, top_k=10);
// Results: [
//   "Never store passwords in plain text" (similarity: 0.92),
//   "Use bcrypt with 12+ rounds" (similarity: 0.89),
//   "Validate email format" (similarity: 0.78),
//   "Rate limit registration endpoints" (similarity: 0.75),
//   ... (6 more)
// ]
// Time: 52ms

// STEP 3: Merge (GNN results are mandatory, Vector DB adds relevance)
let merged = merge_and_deduplicate(gnn_results, vector_results);
// Results: [
//   "Never store passwords in plain text" (GNN: No, Vector: Yes, Priority: 9),
//   "Use bcrypt with 12+ rounds" (GNN: No, Vector: Yes, Priority: 8),
//   "100% test coverage" (GNN: Yes, Vector: No, Priority: 8),
//   "API endpoints need rate limiting" (GNN: Yes, Vector: Yes, Priority: 7),
//   "Auth files require extra security review" (GNN: Yes, Vector: No, Priority: 6),
//   "Validate email format" (GNN: No, Vector: Yes, Priority: 5),
//   ... (more)
// ]

// STEP 4: Token budget allocation (top-N to fit 30% of context)
let final_instructions = allocate_token_budget(merged, max_tokens: 15000);
// Results: Top 6-8 instructions that fit in budget
// Inject into LLM prompt

// TOTAL TIME: ~60ms (8ms GNN + 52ms Vector DB + overhead)
```

### Learning Loop with Hybrid

```rust
// After code generation and validation:

// GNN updates (structural)
if validation_failed {
    gnn.add_edge(instruction, violated_function, EdgeType::ViolatedBy);
    instruction.violation_count += 1;
    instruction.priority = min(instruction.priority + 1, 10);
    
    // Update scope if pattern emerges
    if instruction.violation_count > 10 {
        // Check if all violations in same directory
        let violation_nodes = gnn.get_violations(instruction.id);
        let common_dir = find_common_directory(violation_nodes);
        if common_dir.is_some() {
            // Suggest scope refinement
            suggest_scope_update(instruction, common_dir);
        }
    }
}

// Vector DB updates (semantic)
if validation_failed {
    // Store violation pattern
    let violation_pattern = ViolationPattern {
        instruction_id: instruction.id,
        code_snippet: generated_code,
        context: generation_context,
        description: format!("Violated: {}", instruction.rule),
    };
    
    // Embed and store for future similarity search
    vector_db.add(
        collection: "violations",
        document: violation_pattern.description,
        metadata: violation_pattern,
    );
}

// Future generations can query both:
// GNN: "What instructions were violated in similar code structures?"
// Vector DB: "What violations are semantically similar to this intent?"
```

### Advantages of Hybrid ✅

1. **Best of Both Worlds**
   - GNN: Guaranteed scope coverage (no false negatives)
   - Vector DB: Semantic relevance (better context)

2. **Performance Optimized**
   - GNN for fast exact lookups (<10ms)
   - Vector DB only when semantic ranking needed (~50ms)
   - Total: ~60ms (acceptable for code generation pipeline)

3. **Learning on Both Dimensions**
   - GNN learns structural patterns (which scopes need rules)
   - Vector DB learns semantic patterns (which violations cluster together)

4. **Fallback Strategy**
   - If Vector DB is slow/unavailable → Use GNN only (still works)
   - If GNN is empty → Use Vector DB only (semantic fallback)

5. **Scalability**
   - GNN handles structural queries efficiently (10K+ instructions)
   - Vector DB handles semantic queries efficiently (1K-5K instructions)
   - SQLite handles metadata and stats (unlimited)

### Implementation Complexity

**Moderate** - But we already have most pieces:

- ✅ GNN infrastructure exists (src/gnn/)
- ✅ SQLite persistence exists
- ⏳ Need to add ChromaDB integration (planned for LLM mistake tracking)
- ⏳ Need to design hybrid query logic

**Estimated Effort:** 2-3 weeks
- Week 1: GNN instruction node types + scope matching
- Week 2: Vector DB integration + embedding generation
- Week 3: Hybrid query logic + learning loop

---

## Decision Matrix

| Criterion | GNN Only | Vector DB Only | Hybrid (GNN + Vector) |
|-----------|----------|----------------|----------------------|
| **Scope Matching** | ✅ Perfect | ❌ Weak | ✅ Perfect |
| **Semantic Relevance** | ❌ None | ✅ Excellent | ✅ Excellent |
| **Guaranteed Coverage** | ✅ Yes | ❌ No | ✅ Yes |
| **Learning Capability** | ⚠️ Limited | ✅ Strong | ✅ Strong |
| **Performance** | ✅ <10ms | ⚠️ 50-100ms | ⚠️ ~60ms |
| **False Positives** | ✅ Zero | ⚠️ Possible | ✅ Minimal (GNN filters) |
| **False Negatives** | ⚠️ Possible | ⚠️ Possible | ✅ Minimal (hybrid coverage) |
| **Implementation Complexity** | ✅ Low | ⚠️ Medium | ❌ High |
| **Deterministic** | ✅ Yes | ❌ No | ⚠️ Partially |
| **Scalability** | ✅ Excellent | ⚠️ Good | ✅ Excellent |
| **Memory Usage** | ✅ Low | ⚠️ Medium | ⚠️ Medium |

---

## Recommendation

### Use Hybrid Architecture with Phased Implementation

**Phase 1 (Week 9): GNN Only - MVP**
- Get instructions working quickly
- Scope-based matching is 80% of the value
- Guarantees no instructions are missed
- Fast performance (<10ms)

**Phase 2 (Week 10-11): Add Vector DB - Enhanced Relevance**
- Integrate ChromaDB (already planned for mistake tracking)
- Add semantic similarity search
- Improve relevance ranking
- Enable learning from violations

**Phase 3 (Week 12): Hybrid Optimization**
- Combine GNN + Vector DB results
- Optimize token budget allocation
- Learning loop on both dimensions
- Suggestion engine

### Why This Approach?

1. **Incremental Value** - GNN alone is already valuable
2. **Risk Mitigation** - If Vector DB integration has issues, GNN still works
3. **Reuse Infrastructure** - ChromaDB needed for mistake tracking anyway
4. **Performance Tuning** - Can optimize hybrid queries after seeing usage patterns

---

## Architecture Diagrams

### Hybrid Query Flow

```
User Request: "Add user login with OAuth"
  ↓
┌─────────────────────────────────────────┐
│  PHASE 1: GNN Scope Matching (10ms)     │
├─────────────────────────────────────────┤
│  Query: file_path, node_type, scope     │
│  Returns: All scope-matching rules      │
│  Result: 8 instructions (guaranteed)    │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  PHASE 2: Vector Semantic Search (50ms) │
├─────────────────────────────────────────┤
│  Query: "user login OAuth authentication"│
│  Returns: Top-10 semantically similar   │
│  Result: 10 instructions (relevance)    │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  PHASE 3: Merge & Deduplicate (2ms)     │
├─────────────────────────────────────────┤
│  Union of GNN + Vector DB results       │
│  Remove duplicates                       │
│  Result: 14 unique instructions         │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  PHASE 4: Prioritize (1ms)              │
├─────────────────────────────────────────┤
│  Sort by:                                │
│  1. Priority (user-defined)             │
│  2. Violation ratio (learned)           │
│  3. Semantic relevance (Vector DB score)│
│  Result: 14 instructions (ordered)      │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  PHASE 5: Token Budget Fit (1ms)        │
├─────────────────────────────────────────┤
│  Allocate 30% of token budget           │
│  Fit top-N instructions                  │
│  Result: Top 7 instructions (fit budget)│
└─────────────────────────────────────────┘
  ↓
Inject into LLM Prompt

TOTAL TIME: ~64ms (acceptable)
```

### Data Storage Architecture

```
┌──────────────────────────────────────────────┐
│              HYBRID STORAGE                  │
├──────────────────────────────────────────────┤
│                                              │
│  ┌────────────┐  ┌─────────────┐  ┌───────┐│
│  │    GNN     │  │  Vector DB  │  │ SQLite││
│  │  (petgraph)│  │  (ChromaDB) │  │       ││
│  └────────────┘  └─────────────┘  └───────┘│
│       │                 │              │    │
│       ▼                 ▼              ▼    │
│  Scope Matching    Semantic Search   Stats │
│  - Global          - Embeddings      - Count│
│  - Directory       - Similarity      - Date │
│  - FilePattern     - Clustering      - Rate │
│  - Module          - Suggestions     - Meta │
│  - CodeType        - Learning                │
│                                              │
└──────────────────────────────────────────────┘
         │                 │              │
         └─────────────────┴──────────────┘
                           │
                           ▼
               InstructionStore API
                  (Unified Interface)
```

---

## Performance Targets

| Operation | GNN Only | Vector DB Only | Hybrid |
|-----------|----------|----------------|--------|
| **Scope Query** | <10ms | N/A | <10ms |
| **Semantic Query** | N/A | 50-100ms | 50-100ms |
| **Hybrid Query** | N/A | N/A | ~60ms |
| **Validation** | <5ms | <50ms | <50ms |
| **Learning Update** | <5ms | <20ms | <25ms |
| **Suggestion** | N/A | 100-200ms | 100-200ms |

**Target for Hybrid Query:** <100ms (acceptable within code generation flow)

---

## Migration Path

### Start with GNN (Simple)
```rust
// Week 9: GNN only
let instructions = gnn.find_by_scope(scope);
```

### Add Vector DB (Enhanced)
```rust
// Week 10: Add Vector DB
let instructions = vector_db.search(intent, top_k=10);
```

### Combine (Optimal)
```rust
// Week 11: Hybrid
let scope_instructions = gnn.find_by_scope(scope);
let semantic_instructions = vector_db.search(intent, top_k=10);
let combined = merge_and_prioritize(scope_instructions, semantic_instructions);
```

---

## Conclusion

**Final Recommendation: Hybrid Architecture**

Use **GNN for scope-based matching** (structural, guaranteed coverage) and **Vector DB for semantic relevance** (learning, similarity). Implement in phases:

1. **Week 9:** GNN only (MVP, 80% value)
2. **Week 10-11:** Add Vector DB (enhanced relevance)
3. **Week 12:** Optimize hybrid queries

This approach:
- ✅ Leverages existing GNN infrastructure
- ✅ Reuses ChromaDB (planned for mistake tracking anyway)
- ✅ Guarantees no false negatives (GNN scope matching)
- ✅ Enables learning and suggestions (Vector DB semantic)
- ✅ Provides fallback (either component can work alone)
- ✅ Scales well (10K+ instructions)

**Why not pure GNN?** Loses semantic understanding and learning capability.  
**Why not pure Vector DB?** No guaranteed coverage, harder to enforce scopes.  
**Why hybrid?** Best of both worlds with acceptable complexity trade-off.

---

## References

- **GNN Implementation:** `src/gnn/mod.rs`, `src/gnn/graph.rs`
- **Vector DB Plan:** Decision Log (LLM Mistake Tracking, Nov 20, 2025)
- **Specifications:** `.github/Specifications.md` (Hybrid Intelligence section)
- **Similar Architecture:** Cluster Agents (Phase 2B uses GNN + Vector DB hybrid)

---

**Status:** Architecture decided, ready for phased implementation  
**Next Step:** Update Project_Instructions_System.md with hybrid approach  
**Timeline:** 4 weeks (Week 9: GNN, Weeks 10-12: Hybrid optimization)
