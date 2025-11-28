# Archive: Partial Yantra Codex Documents

**Date Archived:** November 26, 2025  
**Reason:** Superseded by comprehensive implementation plan  
**Replacement Document:** `../Yantra_Codex_Implementation_Plan.md`

---

## Why These Were Archived

These three documents were created on November 24, 2025, and represented early explorations of the Yantra Codex architecture. However, each document covered only **partial aspects** of the full vision, leading to confusion and missing the complete picture.

### Archived Documents

#### 1. `Yantra_Codex_GNN.md`
- **Focus:** Quick win analysis, test generation use case
- **What it covered:** Benefits of real GNN, specific use cases
- **What it missed:** 
  - Tree-sitter's role in code generation
  - GNN predicts AST structure (not code text)
  - Complete implementation details
  - Two-phase architecture (local + cloud)

#### 2. `Yantra_Codex_Multi_Tier_Architecture.md`
- **Focus:** Cloud collective learning infrastructure (Phase 2)
- **What it covered:** Federated learning, network effects, privacy
- **What it missed:**
  - Phase 1: Local GNN + Tree-sitter architecture
  - AST prediction mechanism
  - Bootstrap with curated datasets (jumped straight to cloud)
  - Implementation roadmap for local learning first

#### 3. `Yantra_Codex_GraphSAGE_Knowledge_Distillation.md`
- **Focus:** LLM teacher-student distillation
- **What it covered:** Knowledge distillation techniques, soft labels
- **What it missed:**
  - **Bootstrap with curated CodeContests first** (not LLM)
  - Tree-sitter parsers already implemented
  - GNN outputs embeddings, Tree-sitter generates code
  - On-the-go learning approach
- **Why it caused confusion:** Made it seem like LLM distillation is the primary approach, when actually we start with curated datasets

---

## The Complete Picture (Now in Implementation Plan)

The new **`Yantra_Codex_Implementation_Plan.md`** provides:

### ✅ Two-Phase Architecture
- **Phase 1:** Local GNN + Tree-sitter (bootstrap with CodeContests)
- **Phase 2:** Cloud collective learning (aggregate from all users)

### ✅ Complete Technical Understanding
- GNN predicts AST structure (embeddings)
- Tree-sitter generates code text from AST
- On-the-go learning from every generation
- Experience replay and adaptive thresholds

### ✅ Implementation Roadmap
- Week 1: Extract AST patterns from CodeContests
- Week 2: Train GraphSAGE on real data
- Week 3: Code generation pipeline
- Week 4: On-the-go learning system
- Month 3-6: Cloud infrastructure

### ✅ What We Already Have
- Tree-sitter parsers: `parser.rs` (Python), `parser_js.rs` (JS/TS)
- CodeContests dataset: 6,508 training examples
- GraphSAGE model: Needs retraining on real AST patterns

---

## Key Insights That Were Missing

### 1. GNN Cannot Generate Code Text
The old documents didn't clearly explain that:
- GNN outputs **embeddings** (256-dimensional vectors)
- These represent **AST structure patterns**
- Tree-sitter converts AST → actual code text

### 2. Tree-sitter Already Implemented
The old documents didn't mention that tree-sitter parsers are complete and ready:
- `src-tauri/src/gnn/parser.rs` (278 lines)
- `src-tauri/src/gnn/parser_js.rs` (306 lines)
- Extract functions, classes, imports, calls
- Create CodeNode and CodeEdge structures

### 3. Bootstrap Strategy
The distillation document made it seem like LLM is primary:
- ❌ Old approach: Start with LLM distillation
- ✅ Actual approach: Start with CodeContests curated dataset
- LLM is fallback/supplement, not foundation

### 4. On-the-Go Learning
The old documents discussed offline training:
- ❌ Old: Separate training phases
- ✅ Actual: Learn from every generation (no separate training)
- Experience replay, adaptive thresholds
- Continuous improvement: 40% → 85% → 95%

---

## Historical Value

These documents remain valuable for:
- Understanding the evolution of thinking
- Specific use cases (test generation, bug prediction)
- Cloud architecture details (federated learning)
- Knowledge distillation techniques

But for **implementation**, use the new comprehensive plan.

---

## References

**Current Document (Use This):**
- `../Yantra_Codex_Implementation_Plan.md` - Complete two-phase architecture with implementation code

**Related Documentation:**
- `../Decision_Log.md` - Track of architecture decisions
- `../Technical_Guide.md` - How components work
- `../.github/Session_Handoff.md` - Session continuity context
