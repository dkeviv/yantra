# Yantra - Project Plan (Warp-Speed MVP + Yantra Codex Pair Programming)

**Project:** Yantra - AI-First Development Platform with Pair Programming Intelligence
**Phase:** MVP Phase 1 (Yantra Codex + LLM Pair Programming) + Continuous Learning
**Timeline:** 4 Weeks for Yantra Codex MVP
**Start Date:** November 26, 2025
**Target Completion:** December 24, 2025

---

## 🔥 PRIORITY: Yantra Codex Pair Programming (4 Weeks)

**Vision:** Hybrid AI system combining GNN speed + LLM reasoning for optimal cost & quality

**Default Mode:** Yantra Codex + LLM (Claude/GPT-4) Pair Programming with Continuous Learning

**Critical Decisions (Nov 26, 2025):**

1. ✅ **Pair Programming Architecture** - Yantra generates → LLM reviews → Yantra learns
2. ✅ **1024 dimensions** (not 256) - Cost negligible, benefit significant (15-20% accuracy)
3. ✅ **Confidence-Based Routing** - Smart routing: Yantra alone / Yantra+LLM / LLM alone
4. ✅ **Continuous Learning** - Yantra learns from LLM fixes → reduces cost over time
5. ✅ **User Choice** - Configure Claude, GPT-4, or other premium LLMs

**Cost & Quality Benefits:**

- **Month 1:** 64% cost savings (Yantra 55% confident, LLM review 45%)
- **Month 6:** 90% cost savings (Yantra 85% confident, LLM review 15%)
- **Year 1:** 96% cost savings (Yantra 95% confident, LLM review 5%)
- **Quality:** Yantra + LLM ≥ LLM alone (pair programming is better!)

**Accuracy Targets:**

- Month 1: 55-60% Yantra alone, 95%+ with LLM review
- Month 6: 75-80% Yantra alone, 98%+ with LLM review
- Year 2: 85% Yantra alone, 99%+ with LLM review
- Year 3+: 90-95% Yantra alone, 99.5%+ with LLM review

### Week 1 (Nov 26 - Dec 2): Extract Logic Patterns + Confidence Scoring Foundation

**Status:** 🔴 NOT STARTED

**Goal:** Create training dataset: 6,508 problems → logic patterns (1024-dim) + Build confidence scoring foundation

#### Tasks

**Logic Pattern Extraction (Foundation for Yantra Codex):**

- [ ] **Create scripts/extract_logic_patterns.py**

  - Extract universal logic patterns (not just AST syntax)
  - Use existing Tree-sitter parsers (parser.rs, parser_js.rs)
  - Classify: null_check, validation, iteration, db_query, error_handling, api_call
  - Encode to 1024-dim embeddings
  - Output: `~/.yantra/datasets/logic_patterns.jsonl`
  - **Target:** Process all 6,508 CodeContests solutions
  - **Dependencies:** Tree-sitter parsers (READY), CodeContests dataset (READY)
  - **Estimate:** 2 days
- [ ] **Validate extracted patterns**

  - Check: Each pattern has problem_features (978-dim) and logic_pattern (1024-dim)
  - Verify: Coverage across complexity levels
  - Test: Sample 100 patterns manually
  - **Target:** 95%+ extraction success rate
  - **Estimate:** 1 day
- [ ] **Create pattern visualization tool**

  - Visualize logic patterns in 2D (t-SNE/UMAP)
  - Cluster similar patterns
  - Identify pattern categories
  - **Target:** Understand pattern distribution
  - **Estimate:** 1 day

**Confidence Scoring Foundation (New for Pair Programming):**

- [ ] **Design confidence scoring algorithm**

  - Factor 1: Pattern frequency in training data (0.0-0.3 weight)
  - Factor 2: Embedding distance to nearest neighbor (0.0-0.3 weight)
  - Factor 3: Historical test success rate (0.0-0.4 weight)
  - Combined score: 0.0-1.0 (higher = more confident)
  - **Target:** Design complete confidence calculation
  - **Estimate:** 0.5 days
- [ ] **Create src-tauri/src/codex/confidence.rs (skeleton)**

  - Define `ConfidenceScore` struct
  - Define `ConfidenceCalculator` trait
  - Stub methods: `calculate_pattern_frequency()`, `calculate_embedding_distance()`, `calculate_test_success_rate()`
  - **Target:** API ready for Week 2 implementation
  - **Estimate:** 0.5 days

### Week 2 (Dec 3-9): Train GraphSAGE + Confidence Scoring + LLM Integration Prep

**Status:** 🔴 NOT STARTED

**Goal:** GNN predicts 1024-dim logic patterns from 978-dim problem features + Implement confidence scoring + Prepare LLM integration

#### Tasks

**GraphSAGE Training (Core GNN):**

- [ ] **Update src-python/model/graphsage.py to 1024 dims**

  - Change architecture: 978 → 1536 → 1280 → 1024
  - Parameters: ~150M (up from ~50M at 256 dims)
  - Model size: ~600 MB
  - Add prediction heads: logic_pattern, confidence, complexity
  - **Estimate:** 1 day
- [ ] **Create scripts/train_on_logic_patterns.py**

  - Load logic_patterns.jsonl dataset
  - Train/val split: 80/20
  - Loss: MSE on 1024-dim embeddings
  - Target: <0.1 MSE on validation
  - Early stopping with patience=10
  - Save best model: models/yantra_codex_v1.pt
  - Track pattern frequency for confidence scoring
  - **Estimate:** 2 days
- [ ] **Evaluate on HumanEval benchmark**

  - Test on 164 HumanEval problems
  - Generate code → Run tests → Measure pass rate
  - **Target:** 55-60% accuracy
  - Compare with LLM-generated code
  - **Estimate:** 1 day

**Confidence Scoring Implementation (New for Pair Programming):**

- [ ] **Implement src-tauri/src/codex/confidence.rs**

  - Implement `calculate_pattern_frequency()`: Track pattern occurrences in training data
  - Implement `calculate_embedding_distance()`: Cosine similarity to nearest neighbors
  - Implement `calculate_test_success_rate()`: Historical success from experience buffer
  - Implement `get_confidence_score()`: Combine all factors → 0.0-1.0
  - Unit tests: Verify score ranges, edge cases
  - **Estimate:** 1.5 days
- [ ] **Integrate confidence scoring with generator**

  - Update generator.rs to return (code, confidence_score)
  - Test: Verify confidence correlates with actual success rate
  - Calibrate thresholds: 0.9 (high confidence), 0.8 (medium), 0.5 (low)
  - **Estimate:** 0.5 days

**LLM Integration Preparation (New for Pair Programming):**

- [ ] **Design LLM review interface**

  - Define `ReviewRequest` struct: (code, context, confidence_score, problem_description)
  - Define `ReviewResponse` struct: (improved_code, explanation, test_cases, confidence_boost)
  - Define routing logic: confidence < 0.8 → trigger LLM review
  - **Estimate:** 0.5 days
- [ ] **Verify LLM orchestrator readiness**

  - Test existing orchestrator.rs with Claude/GPT-4
  - Verify circuit breaker works for failover
  - Test rate limiting and retry logic
  - Document API endpoints for code review
  - **Estimate:** 0.5 days

### Week 3 (Dec 10-16): Code Generation Pipeline + Pair Programming Orchestrator

**Status:** 🔴 NOT STARTED

**Goal:** Problem → GNN Logic → Tree-sitter Code (multi-language) + Pair Programming Orchestrator + Smart Routing + LLM Reviewer

#### Tasks

**Code Generation Pipeline (Core):**

- [ ] **Create src-tauri/src/codex/generator.rs (with smart routing)**

  - Implement: Problem → Features (978-dim)
  - Call Python: GNN predicts logic pattern (1024-dim)
  - Calculate confidence score (using confidence.rs)
  - **SMART ROUTING:**
    - If confidence ≥ 0.9: Return Yantra code immediately (high confidence path)
    - If 0.8 ≤ confidence < 0.9: Return Yantra code + validate with tests (medium confidence)
    - If 0.5 ≤ confidence < 0.8: Trigger LLM review (low confidence)
    - If confidence < 0.5: Route to LLM completely (novel pattern)
  - Decode: Logic embedding → Logic steps
  - Route to Tree-sitter: Generate language-specific code
  - Return: (code, confidence, routing_decision)
  - **Estimate:** 2 days
- [ ] **Create src-tauri/src/codex/decoder.rs**

  - Decode 1024-dim embedding → LogicStep enum
  - Handle: NullCheck, ValidationCheck, DatabaseQuery, Iteration, ErrorHandling, ApiCall
  - Convert to AST structure for Tree-sitter
  - **Estimate:** 1 day
- [ ] **Extend Tree-sitter integration for code generation**

  - Python: Generate from LogicStep[] → Python code
  - JavaScript: Generate from LogicStep[] → JS code
  - Test: Same logic pattern → Different languages
  - Verify: Syntactically correct code
  - **Estimate:** 1 day

**Pair Programming Components (New):**

- [ ] **Create src-tauri/src/codex/pair_programming.rs (orchestrator)**

  - Implement `PairProgrammingOrchestrator` struct
  - Method: `generate_with_review(problem) → PairResult`
  - Workflow:
    1. Call Yantra generator → (yantra_code, confidence)
    2. If confidence < 0.8: Call LLM reviewer → (improved_code, explanation)
    3. Merge results: Choose best version or combine
    4. Validate with tests
    5. Return final code + metadata
  - Track: Yantra-only count, LLM-review count, cost savings
  - **Estimate:** 2 days
- [ ] **Create src-tauri/src/codex/llm_reviewer.rs**

  - Implement `LLMReviewer` struct
  - Method: `review_code(yantra_code, problem, confidence) → ReviewResult`
  - Use existing LLM orchestrator (orchestrator.rs)
  - Prompt engineering: "Review this code generated by Yantra GNN. Enhance error handling, edge cases, and add tests."
  - Parse LLM response: Extract improved code, explanation, test cases
  - Timeout: 10s with fallback to Yantra code
  - **Estimate:** 1.5 days
- [ ] **Integration testing (pair programming flow)**

  - End-to-end: "Validate email and save to DB" → Python/JS code
  - Test 50 problems across confidence ranges
  - Measure:
    - High confidence (≥0.9): Yantra alone accuracy
    - Medium confidence (0.8-0.9): Yantra alone + test validation accuracy
    - Low confidence (0.5-0.8): Yantra + LLM review accuracy
    - Novel patterns (<0.5): LLM alone accuracy
  - **Target:** 95%+ overall pass rate (up from 55-60% GNN-only)
  - Track cost: Calculate LLM API calls and savings vs LLM-only
  - **Estimate:** 1.5 days

### Week 4 (Dec 17-24): Continuous Learning System + Experience Buffer + Incremental Fine-Tuning

**Status:** 🔴 NOT STARTED

**Goal:** Yantra learns from LLM fixes → Confidence increases → Cost decreases over time

#### Tasks

**Experience Buffer & Learning Loop (Core Learning System):**

- [ ] **Create src-python/learning/online_learner.py (enhanced)**

  - Experience replay buffer (capacity: 1000)
  - Store: (problem_features, yantra_code, llm_improved_code, confidence_score, test_result, timestamp)
  - **NEW:** Store LLM fixes separately for focused learning
  - Adaptive threshold: Start 0.3, increase to 0.7 as accuracy improves
  - Incremental updates: Train on new patterns every 100 examples
  - Priority sampling: Weight LLM-improved examples higher
  - **Estimate:** 2 days
- [ ] **Create src-python/learning/incremental_learner.py**

  - Implement incremental GNN fine-tuning
  - Method: `learn_from_experience_batch(experiences[]) → updated_model`
  - Use small learning rate (0.0001) to avoid catastrophic forgetting
  - Loss: MSE on logic patterns + contrastive loss (Yantra vs LLM code)
  - Validation: Test on held-out examples after each update
  - Track: Confidence improvement for learned patterns
  - Save checkpoints: models/yantra_codex_v1_checkpoint_{timestamp}.pt
  - **Estimate:** 2 days
- [ ] **Implement feedback loop integration**

  - User generates code with pair programming orchestrator
  - Run tests automatically
  - **If Yantra alone passed:** Confidence boost +0.05 for similar patterns
  - **If LLM improved:** Extract logic difference → Add to experience buffer → Trigger learning
  - **If both failed:** Mark as hard example, route to LLM completely next time
  - Track metrics: Yantra-only %, LLM-review %, accuracy progression, cost savings
  - **Estimate:** 1.5 days

**Analytics & Monitoring (Learning Dashboard):**

- [ ] **Create learning analytics dashboard**
  - Dashboard: GNN accuracy over time (by confidence bucket)
  - Track:
    - Patterns learned count
    - Confidence distribution (how many patterns in 0.9+, 0.8-0.9, 0.5-0.8, <0.5)
    - Cost savings progression (estimated $ saved vs LLM-only)
    - LLM review rate trend (should decrease over time)
  - Visualize: Confidence improvement for individual patterns
  - Alert: If accuracy drops below threshold or learning stalls
  - Export: Successful patterns to Yantra Cloud (privacy-preserving, opt-in)
  - **Estimate:** 1.5 days

**Continuous Learning Integration:**

- [ ] **Implement continuous learning scheduler**

  - Background task: Check experience buffer every 10 minutes
  - If buffer has ≥100 new experiences: Trigger incremental fine-tuning
  - Update confidence scores after each training cycle
  - Validate: Test on validation set, rollback if accuracy drops
  - **Estimate:** 1 day
- [ ] **Prepare for Yantra Cloud Codex (opt-in, privacy-preserving)**

  - Design API: Upload anonymous logic patterns (embeddings only, never actual code)
  - Privacy: Hash problem descriptions, send only embeddings + metadata
  - Aggregation: Collect patterns from all users (federated learning approach)
  - Retrain schedule: Weekly or 10k new patterns threshold
  - User benefit: Download improved model weekly (90%+ accuracy after 6 months)
  - **Estimate:** 2 days

**Validation & Quality Assurance:**

- [ ] **End-to-end pair programming validation**
  - Run 200 problems through full pipeline
  - Measure:
    - Month 1 targets: 55-60% Yantra alone, 95%+ with LLM review
    - Confidence calibration: High confidence should → high success rate
    - Cost: Track actual LLM API calls vs pure LLM baseline
    - Learning: Verify confidence increases after LLM fixes
  - **Success criteria:**
    - 95%+ overall accuracy
    - 50-60% cost savings in Month 1 (Yantra handles 40-50% alone)
    - Confidence scores correlate with success (r > 0.7)
    - Learning loop works: Patterns improve after LLM fixes
  - **Estimate:** 1 day

---

## Architecture Summary (Updated for Pair Programming)

**Hybrid Intelligence: Yantra Codex + LLM Pair Programming**

**Core Components:**

1. **Yantra Codex (Junior Partner):** Local GraphSAGE (600 MB, 15ms inference, 55-95% accuracy over time)
2. **LLM (Senior Partner):** User-configured Claude/GPT-4 (reviews when confidence < 0.8)
3. **Confidence Scoring:** Multi-factor (pattern frequency, embedding distance, test history)
4. **Smart Routing:** Confidence-based decision (Yantra alone / Yantra+LLM / LLM alone)
5. **Continuous Learning:** Yantra learns from LLM fixes → Confidence increases → Cost decreases

**Pair Programming Workflow:**

```
Problem → Yantra generates (15ms) → Confidence score (0.0-1.0)
  ├─ If ≥0.9: Return Yantra code (high confidence, $0 cost)
  ├─ If 0.8-0.9: Return Yantra code + test validation (medium confidence)
  ├─ If 0.5-0.8: LLM reviews → Merge → Yantra learns (low confidence)
  └─ If <0.5: LLM generates (novel pattern, full LLM cost)
```

**Cost Trajectory (vs LLM-only baseline $25/1000 generations):**

- **Month 1:** $9/1000 gen (64% savings) - Yantra handles 55% alone
- **Month 6:** $3/1000 gen (88% savings) - Yantra handles 85% alone
- **Year 1:** $1/1000 gen (96% savings) - Yantra handles 95% alone

**Quality Guarantee:** Yantra + LLM ≥ LLM alone (pair programming is better!)

**MVP Focus (Month 1-2):**

- ✅ Yantra Codex generates code (55-60% accuracy baseline)
- ✅ Confidence scoring system (0.0-1.0)
- ✅ Smart routing (confidence-based)
- ✅ LLM review & enhancement (when confidence < 0.8)
- ✅ Continuous learning (Yantra learns from LLM fixes)
- ✅ Experience buffer + incremental fine-tuning
- **Target:** 95%+ overall accuracy, 64% cost savings Month 1

**Phase 2 (Month 3-6):**

- Yantra handles 85% alone (90% cost savings)
- Cross-language pattern transfer (Python → JS, JS → Python)
- Yantra Cloud Codex (opt-in): Federated learning from all users
- Advanced confidence calibration (per-user, per-pattern-type)

**Phase 3 (Year 1+):**

- Yantra handles 95% alone (96% cost savings)
- Self-improving system approaching LLM-quality solo
- Multi-domain specialization (web, data science, DevOps)

---

## Success Metrics (MVP - Updated for Pair Programming)

**Core Metrics (Week 4 Validation):**

- [ ] ✅ Yantra Codex trained (55-60% baseline accuracy alone)
- [ ] ✅ Confidence scoring system working (scores correlate with success, r > 0.7)
- [ ] ✅ Smart routing implemented (confidence-based, <50ms overhead)
- [ ] ✅ LLM review integration complete (Claude/GPT-4, <10s latency)
- [ ] ✅ Continuous learning system functional (learns from LLM fixes)
- [ ] ✅ Overall accuracy 95%+ (Yantra + LLM pair programming)
- [ ] ✅ Cost savings 64% Month 1 (vs LLM-only baseline)
- [ ] ✅ Experience buffer + incremental fine-tuning working
- [ ] ✅ Bootstrap complete with CodeContests dataset (6,508 examples processed)

**Quality Metrics:**

- [ ] 
- [ ] Zero breaking changes to existing code (GNN validation)
- [ ] <3% critical security vulnerabilities (auto-fixed with Semgrep)
- [ ] <2 minutes total cycle (intent → Yantra → LLM review → commit)

**Learning Metrics:**

- [ ] Confidence increases after LLM fixes (validate on 50 patterns)
- [ ] Yantra alone accuracy improves Week 1 → Week 4 (55% → 60%+)
- [ ] LLM review rate decreases as patterns become familiar

**Business Metrics:**

- [ ] 5 beta users successfully generating code with pair programming
- [ ] User satisfaction: NPS >40 (transparency + cost savings)
- [ ] Cost tracking dashboard working (show $ saved vs LLM-only)

---

## Current Status (Nov 26, 2025)

**Foundation Complete (90%):**

- ✅ UI/UX: 100% (3-column layout, Monaco Editor, file tree, multi-terminal)
- ✅ GNN: 90% (dependency tracking, incremental updates, Python parser)
- ✅ LLM: 80% (Claude/GPT-4 clients, orchestration, test generation)
- ✅ Testing: 60% (test generation working, execution needs integration)
- ⚠️ **GraphSAGE Codex: 0%** (CRITICAL GAP - focus of warp-speed plan)

**What We Need:**

- PyO3 bridge for Rust ↔ Python integration
- GraphSAGE model implementation (3-layer SAGEConv)
- Feature extraction (978-dim: 974 base + 4 language encoding)
- OpenRouter integration with DeepSeek V3
- Bootstrap training pipeline with CodeContests
- Success-only learning loop

---

## Warp-Speed Timeline (20 Days)

### Week 1 (Nov 25 - Dec 1): Foundation Hardening

**Goal:** Bulletproof foundation for GraphSAGE integration

### Week 2 (Dec 2-8): GraphSAGE Infrastructure

**Goal:** GraphSAGE model ready for training

### Week 3 (Dec 9-15): Bootstrap Training

**Goal:** Trained GraphSAGE model (45-50% accuracy)

### Week 4 (Dec 16-22): Learning Loop + Launch

**Goal:** MVP ready for soft launch with 5 beta users

---

## Week 1: Foundation Hardening (Nov 25 - Dec 1, 2025)

### Status: 🚀 IN PROGRESS - 75% Complete (3 of 4 tasks done)

**Goal:** Make dependency tracking bulletproof + test execution working
**Why Critical:** GraphSAGE can't learn from success without reliable test validation

#### Priority Tasks

- [X] **Dependency Tracking - Incremental Updates** ✅ COMPLETE (Nov 25, 2025)

  - [X] Optimize incremental GNN updates (<50ms per file change)
  - [X] Implement file timestamp tracking
  - [X] Add dirty flag propagation through dependency graph
  - [X] Create node caching with file-to-nodes mapping
  - [X] Test with 10 sequential file edits
  - [X] Unit tests for IncrementalTracker
  - [X] Integration test for performance validation

  - **Result:** Achieved 1ms average (range: 0-2ms), 50x faster than 50ms target
  - **Cache:** 100% hit rate after first parse (4/4 nodes cached)
  - **Why:** Critical for real-time GraphSAGE learning loop
  - **Target:** <50ms per file ✅ **ACHIEVED: 1ms average**
  - **Files:** `src/gnn/incremental.rs` (330 lines), `src/gnn/mod.rs` (updated)
- [X] **Multi-Language Support - JS/TS Parser** ✅ COMPLETE (Nov 25, 2025)

  - [X] Add tree-sitter-javascript dependency
  - [X] Add tree-sitter-typescript dependency
  - [X] Implement JavaScript parser (.js, .jsx)
  - [X] Implement TypeScript parser (.ts, .tsx)
  - [X] Manual tree walking (simpler than complex queries)
  - [X] Extract functions, classes, imports, variables
  - [X] Update GNNEngine to support multiple languages
  - [X] 5 unit tests passing

  - **Result:** Full multi-language support for Python, JavaScript, TypeScript
  - **Why:** Need to support React/Node.js projects for broader adoption
  - **Target:** Parse React/Node.js projects ✅ ACHIEVED
  - **Files:** `src/gnn/parser_js.rs` (300 lines), `src/gnn/mod.rs` (updated)

  - [ ] Add caching for unchanged subtrees
  - [ ] Implement dirty flag propagation
  - [ ] Test with 10 sequential file edits
  - [ ] Measure performance (target: <50ms)

  - **Why:** GraphSAGE needs fast graph updates for learning loop
  - **Target:** <50ms for single file update, tested with real-world projects
  - **Files:** `src/gnn/graph.rs`, `src/gnn/incremental.rs` (new)
- [ ] **tree-sitter - JavaScript/TypeScript Support** 🔥 CRITICAL

  - [ ] Add tree-sitter-javascript dependency
  - [ ] Add tree-sitter-typescript dependency
  - [ ] Implement JS/TS parser alongside Python
  - [ ] Update feature extraction for JS/TS nodes
  - [ ] Test with React/Node.js projects

  - **Why:** Multi-language support needed for Phase 2
  - **Target:** Parse JS/TS files into GNN graph
  - **Files:** `src/gnn/parser.rs`, update `Cargo.toml`
- [X] **Test Execution - pytest Integration in Rust** ✅ COMPLETE (Nov 25, 2025)

  - [X] Implement programmatic pytest execution from Rust
  - [X] Parse pytest JSON output (--json-report)
  - [X] Create TestExecutionResult struct (passed, failed, errors, coverage)
  - [X] Handle test failures and errors gracefully
  - [X] Create Tauri command for test execution
  - [X] Add TypeScript bindings

  - **Result:** 410 lines of code, 5/5 unit tests pass, 154/154 full suite pass
  - **Why:** Success-only learning requires test validation
  - **Target:** Execute tests, parse results, return to frontend ✅ ACHIEVED
  - **Files:** `src/testing/executor.rs` (new), `src/testing/mod.rs`, `src/main.rs`, `src-ui/api/testing.ts`
- [X] **Test Execution - Result Parsing** ✅ COMPLETE (Nov 25, 2025)

  - [X] Parse pytest JSON report format
  - [X] Extract test names, statuses, error messages
  - [X] Calculate pass rate percentage
  - [X] Extract coverage data (if pytest-cov enabled)
  - [X] Unit tests for parser
  - [X] Implement quality threshold (>90% for learning)

  - **Result:** Integrated into executor.rs, clean JSON parsing with serde
  - **Why:** Need structured test results for GraphSAGE learning
  - **Target:** Reliable parsing of all pytest outputs ✅ ACHIEVED
  - **Files:** `src/testing/executor.rs` (parse_json_report, parse_pytest_output)
- [X] **Real-World Testing** ✅ COMPLETED Nov 25, 2025

  - [X] Fix GNN Engine API (add node_count(), edge_count() public methods)
  - [X] Create comprehensive real-world test suite
  - [X] Test with 15 diverse projects (Python, JS, TS, TSX, mixed)
  - [X] Measure GNN build time (<5s for 10K LOC)
  - [X] Measure incremental update time (<50ms, expecting 1ms)
  - [X] Test edge cases (empty files, comments, Unicode, mixed extensions)
  - [X] Document performance results

  - **Results:**
    - All 15 tests passing ✅
    - **Projected 10K LOC: 200ms (0.2s)** - 25x better than <5s target!
    - **Incremental updates: 91µs (0.091ms)** - 550x better than <50ms target!
    - Multi-language support validated (Python, JS, TS, TSX)
    - Edge cases handled gracefully
  - **Why:** Validate foundation before GraphSAGE integration
  - **Target:** All tests passing, performance targets met ✅ EXCEEDED
  - **Files:** `tests/real_world_test.rs` (283 lines, 15 tests)

#### Deliverable

Rock-solid foundation:

- ✅ pytest execution integrated (COMPLETE - Nov 25)
- ✅ Test result parsing reliable (COMPLETE - Nov 25)
- ✅ Incremental GNN updates <50ms (COMPLETE - Nov 25, achieved 1ms avg)
- ✅ JS/TS parser working (COMPLETE - Nov 25, supports .js, .ts, .jsx, .tsx)
- ⚪ Validated with 20 real-world projects

**Week 1 Progress: 100% Complete ✅ (4 of 4 tasks done) - FOUNDATION COMPLETE!**

---

## Week 2: GraphSAGE Infrastructure (Dec 2-8, 2025)

### Status: ⚪ NOT STARTED

**Goal:** GraphSAGE model ready for training
**Why Critical:** Need model infrastructure before bootstrap training

#### Priority Tasks

- [ ] **PyO3 Bridge - Rust ↔ Python** 🔥 CRITICAL

  - [ ] Add PyO3 0.20+ dependency to Cargo.toml
  - [ ] Create Python module in `src-python/`
  - [ ] Implement Rust → Python data passing (graph features)
  - [ ] Implement Python → Rust data passing (predictions)
  - [ ] Test bidirectional communication (<2ms overhead)
  - [ ] Handle Python exceptions in Rust

  - **Why:** GraphSAGE model in Python, Yantra core in Rust
  - **Target:** <2ms overhead for bridge calls
  - **Files:** `src/bridge/pyo3_bridge.rs` (new), `src-python/yantra_bridge.py` (new)
- [ ] **Feature Extraction - 978-Dimensional** 🔥 CRITICAL

  - [ ] Extract 974 base features from GNN (depth, degree, types, etc.)
  - [ ] Add 4-dimensional language encoding (Python, JS, TS, Other)
  - [ ] Create FeatureVector struct (978 floats)
  - [ ] Implement one-hot encoding for languages
  - [ ] Normalize features (0-1 range)
  - [ ] Unit tests for feature extraction

  - **Why:** GraphSAGE input requires structured features
  - **Target:** Extract features for any code node in <1ms
  - **Files:** `src/gnn/features.rs` (new)
- [ ] **GraphSAGE Model - PyTorch Implementation** 🔥 CRITICAL

  - [ ] Implement 3-layer SAGEConv architecture
  - [ ] Input: 978-dim → Layer 1: 512-dim → Layer 2: 512-dim → Output: 256-dim
  - [ ] Add prediction heads (code, imports, bugs, next_calls)
  - [ ] Add DORMANT test prediction heads (assertions, fixtures, edge_cases)
  - [ ] Implement forward pass with ReLU activation
  - [ ] Add dropout for regularization (0.2)

  - **Why:** Core neural network for code generation
  - **Target:** Model compiles, forward pass works, 140 MB size
  - **Files:** `src-python/model/graphsage.py` (new)
- [ ] **OpenRouter Integration** 🔥 CRITICAL

  - [ ] Add OpenRouter API client (similar to Claude/GPT-4)
  - [ ] Configure DeepSeek V3 Coder model
  - [ ] Use provided API key: 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
  - [ ] Implement retry logic and circuit breaker
  - [ ] Add to LLM orchestrator as Tier 2 fallback
  - [ ] Test code generation with DeepSeek

  - **Why:** DeepSeek is our open-source teacher LLM
  - **Target:** Working API integration with DeepSeek V3
  - **Files:** `src/llm/openrouter.rs` (new), update orchestrator
- [ ] **Model Persistence** 🔥 CRITICAL

  - [ ] Implement model save/load (PyTorch checkpoint)
  - [ ] Store in ~/.yantra/models/
  - [ ] Version models (v1, v2, etc.)
  - [ ] Load model on Yantra startup
  - [ ] Handle missing model gracefully

  - **Why:** Preserve trained model across sessions
  - **Target:** Save/load in <1s
  - **Files:** `src-python/model/persistence.py` (new)
- [ ] **Inference Pipeline** 🔥 CRITICAL

  - [ ] Create inference wrapper (Rust → Python → model → Rust)
  - [ ] Implement confidence score calculation (softmax)
  - [ ] Add confidence threshold check (0.7)
  - [ ] Route to LLM if confidence < 0.7
  - [ ] Cache predictions for repeated queries

  - **Why:** Connect GraphSAGE to code generation pipeline
  - **Target:** <10ms inference time
  - **Files:** `src/codex/inference.rs` (new)

#### Deliverable

GraphSAGE infrastructure complete:

- ✅ PyO3 bridge working (<2ms overhead)
- ✅ 978-dim feature extraction implemented
- ✅ GraphSAGE model architecture complete (with dormant test heads)
- ✅ OpenRouter + DeepSeek integrated
- ✅ Model persistence working
- ✅ Inference pipeline ready

---

## Week 3: Bootstrap Training (Dec 9-15, 2025)

### Status: ⚪ NOT STARTED

**Goal:** Trained GraphSAGE model with 45-50% baseline accuracy
**Why Critical:** Need trained model for MVP launch

#### Priority Tasks

- [ ] **CodeContests Dataset - Download & Process** 🔥 CRITICAL

  - [ ] Download CodeContests from Hugging Face (13,328 examples)
  - [ ] Parse JSON format (problem, solutions, tests)
  - [ ] Filter for Python examples (~8,000 examples)
  - [ ] Split into train (80%) / validation (20%)
  - [ ] Store in ~/.yantra/datasets/

  - **Why:** Training data for GraphSAGE bootstrap
  - **Target:** 8,000 Python examples ready for training
  - **Files:** `scripts/download_codecontests.py` (new)
- [ ] **Dataset Augmentation - DeepSeek Generation** 🔥 CRITICAL

  - [ ] Generate additional examples with DeepSeek (~1,000 examples)
  - [ ] Target domain-specific patterns (web, data science, etc.)
  - [ ] Validate generated code with pytest
  - [ ] Only keep examples that pass tests
  - [ ] Cost estimate: ~$1.40 for 1,000 examples

  - **Why:** Augment CodeContests with diverse patterns
  - **Target:** 1,000 validated examples
  - **Files:** `scripts/augment_dataset.py` (new)
- [ ] **Training Pipeline - Data Preparation** 🔥 CRITICAL

  - [ ] Parse code examples into GNN graphs
  - [ ] Extract 978-dim features for each node
  - [ ] Create training batches (batch_size=32)
  - [ ] Implement data loader with shuffling
  - [ ] Cache preprocessed graphs

  - **Why:** Prepare data for GraphSAGE training
  - **Target:** Process 9,000 examples in <30 minutes
  - **Files:** `src-python/training/dataset.py` (new)
- [ ] **Training Pipeline - Loss Function & Optimizer** 🔥 CRITICAL

  - [ ] Implement multi-task loss (code + imports + bugs)
  - [ ] Use Adam optimizer (lr=0.001)
  - [ ] Add learning rate scheduler (reduce on plateau)
  - [ ] Implement early stopping (patience=5 epochs)
  - [ ] Track training/validation loss

  - **Why:** Optimize GraphSAGE model parameters
  - **Target:** Converge in <20 epochs
  - **Files:** `src-python/training/trainer.py` (new)
- [ ] **Training Execution** 🔥 CRITICAL

  - [ ] Train GraphSAGE on 9,000 examples
  - [ ] Run for 20 epochs (~4-6 hours on GPU, ~12 hours on CPU)
  - [ ] Validate after each epoch
  - [ ] Save best model checkpoint
  - [ ] Log training metrics

  - **Why:** Create initial trained model
  - **Target:** 45-50% validation accuracy
  - **Files:** Run `src-python/training/train.py`
- [ ] **Model Evaluation** 🔥 CRITICAL

  - [ ] Test on held-out validation set (20% = ~1,800 examples)
  - [ ] Measure accuracy on code prediction
  - [ ] Measure precision/recall for imports
  - [ ] Measure bug detection rate
  - [ ] Compare vs random baseline

  - **Why:** Validate model quality before launch
  - **Target:** >45% accuracy (vs 10% random baseline)
  - **Files:** `src-python/evaluation/evaluate.py` (new)

#### Deliverable

Trained GraphSAGE model:

- ✅ 9,000 examples processed (8,000 CodeContests + 1,000 DeepSeek)
- ✅ Model trained for 20 epochs
- ✅ 45-50% validation accuracy achieved
- ✅ Model saved to ~/.yantra/models/graphsage_v1.pt
- ✅ Ready for production use

---

## Week 4: Learning Loop + Launch (Dec 16-22, 2025)

### Status: ⚪ NOT STARTED

**Goal:** MVP ready for soft launch with 5 beta users
**Why Critical:** Need continuous learning + user testing

#### Priority Tasks

- [ ] **Code Composer Integration** 🔥 CRITICAL

  - [ ] Create orchestration layer (GraphSAGE → DeepSeek → User Premium)
  - [ ] Implement confidence-based routing (≥0.7 → GraphSAGE, <0.7 → LLM)
  - [ ] Add fallback chain (GraphSAGE → DeepSeek → user premium)
  - [ ] Track which tier handled each request
  - [ ] Add metrics (GraphSAGE usage %, LLM usage %)

  - **Why:** Route requests to right tier based on confidence
  - **Target:** 70% GraphSAGE usage after 100 generations
  - **Files:** `src/codex/composer.rs` (new)
- [ ] **Success-Only Learning Loop** 🔥 CRITICAL

  - [ ] Generate code (GraphSAGE or LLM)
  - [ ] Generate tests (LLM only in MVP)
  - [ ] Execute tests with pytest
  - [ ] IF tests pass → Train GraphSAGE on (code, context, success=True)
  - [ ] IF tests fail → Don't learn (or learn as negative example)
  - [ ] Update model incrementally (online learning)

  - **Why:** Core learning mechanism - only learn from working code
  - **Target:** Learn from every successful generation
  - **Files:** `src/codex/learning.rs` (new)
- [ ] **Online Learning - Incremental Updates** 🔥 CRITICAL

  - [ ] Implement incremental model updates (no full retrain)
  - [ ] Use small learning rate (lr=0.0001) for stability
  - [ ] Update model weights after each successful generation
  - [ ] Save model periodically (every 10 successful generations)
  - [ ] Track learning curve (accuracy over time)

  - **Why:** Model improves with each user interaction
  - **Target:** Model updates in <100ms per example
  - **Files:** `src-python/training/online_learning.py` (new)
- [ ] **Metrics Dashboard** 🔥 CRITICAL

  - [ ] Track GraphSAGE accuracy over time
  - [ ] Track LLM usage percentage
  - [ ] Track test pass rate
  - [ ] Track confidence scores distribution
  - [ ] Display in UI (Settings panel)

  - **Why:** Monitor system performance and learning progress
  - **Target:** Real-time metrics visible to user
  - **Files:** `src-ui/components/MetricsDashboard.tsx` (new)
- [ ] **Beta Testing** 🔥 CRITICAL

  - [ ] Recruit 5 beta users
  - [ ] Each user generates 50 code examples
  - [ ] Monitor GraphSAGE learning curve
  - [ ] Track test pass rates
  - [ ] Collect user feedback

  - **Why:** Validate MVP with real users
  - **Target:** 250 total generations (5 users × 50 each)
  - **Files:** Create `docs/Beta_Testing_Plan.md`
- [ ] **Performance Optimization** 🔥 CRITICAL

  - [ ] Profile end-to-end pipeline
  - [ ] Optimize slow operations (>100ms)
  - [ ] Add caching where beneficial
  - [ ] Test on low-end hardware
  - [ ] Measure total cycle time (intent → commit)

  - **Why:** Meet <2 minute cycle time target
  - **Target:** <2 minutes from query to commit
  - **Files:** Various (profile-guided optimization)

#### Deliverable

MVP ready for launch:

- ✅ Code Composer routing working
- ✅ Success-only learning loop implemented
- ✅ Online learning functional
- ✅ Metrics dashboard showing progress
- ✅ 5 beta users testing (250 generations)
- ✅ Performance optimized (<2 min cycle)
- ✅ Ready for soft launch!

---

## Post-MVP: Phase 2 Preparation (Dec 23 - Jan 15, 2026)

### Status: ⚪ FUTURE

**Goal:** Prepare for Phase 2 (GraphSAGE takes over test generation)

#### Tasks

- [ ] **Activate Test Prediction Heads**

  - [ ] Train dormant test prediction heads on 2 months of LLM-generated tests
  - [ ] Measure GraphSAGE test quality vs LLM baseline
  - [ ] Set confidence threshold for test generation (0.7)
  - [ ] Implement fallback to LLM for low-confidence tests
- [ ] **Test Generation Switch**

  - [ ] Switch to GraphSAGE for test generation (90% usage)
  - [ ] Keep LLM fallback for <10% of cases
  - [ ] Monitor test pass rates
  - [ ] Measure cost reduction ($540/year → $96/year)
- [ ] **Self-Improving Loop**

  - [ ] GraphSAGE generates both code AND tests
  - [ ] Learn from validated (code, test) pairs
  - [ ] Track improvement in both code and test quality
  - [ ] Achieve 90-95% accuracy on both

---

## Week 1-2: Foundation (Nov 20 - Dec 3, 2025)

### Status: ✅ COMPLETE - Production-Ready UI/UX

#### Tasks

- [X] **Project Setup** ✅ COMPLETED Nov 20, 2025

  - [X] Initialize Tauri 1.5+ project
  - [X] Configure Rust workspace (Cargo.toml in src-tauri/)
  - [X] Set up SolidJS 1.8+ frontend
  - [X] Configure TailwindCSS 3.3+
  - [X] Set up development environment
  - [X] Configure build scripts and icon files
  - [X] Install Rust/Cargo toolchain

  - **Status:** Fully working, Tauri app compiles and runs
- [X] **3-Column UI Layout** ✅ COMPLETED Nov 23, 2025 → **REDESIGNED from 3-panel**

  - [X] Design responsive layout (FileTree 20%, Chat 45%, Code+Terminal 35%)
  - [X] Implement chat panel component
  - [X] Implement code viewer panel with Monaco Editor
  - [X] Implement file tree panel with recursive navigation
  - [X] Implement multi-terminal in right column (replaces bottom terminal)
  - [X] Add panel resizing functionality (mouse drag with constraints)
  - [X] Implement state management (SolidJS stores)
  - [X] Add view routing for Code Editor / Dependencies switcher

  - **Status:** Production-ready 3-column layout
  - **Live at:** Tauri desktop app (http://localhost:1420/ in dev mode)
- [X] **Monaco Editor Integration** ✅ COMPLETED Nov 20, 2025

  - [X] Install Monaco Editor 0.44+
  - [X] Configure Python syntax highlighting
  - [X] Add code formatting support
  - [X] Implement read-only mode toggle
  - [X] Add line numbers and minimap

  - **Features:** Custom dark theme, automatic layout, word wrap, format on paste/type
- [X] **File System Operations** ✅ COMPLETED Nov 20, 2025

  - [X] Create Rust backend commands (Tauri)
  - [X] Implement file read/write operations (read_file, write_file, read_dir)
  - [X] Implement directory listing with metadata
  - [X] Add file tree component with expand/collapse
  - [X] Implement project folder selection (Tauri dialog)
  - [X] Wire up Tauri commands to frontend

  - **Features:** Full file system access, read/write files, directory navigation
- [X] **Native Menu System** ✅ COMPLETED Nov 23, 2025

  - [X] Implement Tauri native menus (File, View, Help)
  - [X] Add keyboard shortcuts (Cmd+B, Cmd+E, Cmd+`, Cmd+D)
  - [X] Create event-driven menu architecture (Rust → Frontend)
  - [X] Implement toggle functionality for all panels
  - [X] Add reset layout feature

  - **Features:** Professional desktop app with native OS menus
- [X] **Multi-Terminal System** ✅ COMPLETED Nov 23, 2025

  - [X] Design terminal state management (terminalStore.ts)
  - [X] Implement intelligent command routing algorithm
  - [X] Create multi-terminal UI component with tabs
  - [X] Add status indicators (Idle 🟢, Busy 🟡, Error 🔴)
  - [X] Implement terminal lifecycle (create, close, execute, complete)
  - [X] Add stats tracking (total/idle/busy/error)
  - [X] Create terminal controls (+ New, Close, Clear, Execute)

  - **Features:** VSCode-like multi-terminal with parallel execution
  - **Note:** Frontend complete, backend integration pending
- [X] **VSCode-Style File Tabs** ✅ COMPLETED Nov 23, 2025

  - [X] Implement multi-file tab system
  - [X] Add tab switching functionality
  - [X] Add close buttons on tabs
  - [X] Implement active tab highlighting
  - [X] Add file path display

  - **Features:** Professional IDE-like file management
- [X] **Recursive File Tree** ✅ COMPLETED Nov 23, 2025

  - [X] Implement lazy loading for nested folders
  - [X] Add expand/collapse functionality
  - [X] Add file type icons (🐍 Python, 📄 JS, etc.)
  - [X] Implement smart sorting (directories first)

  - **Features:** Fast navigation in large projects
- [X] **View Routing System** ✅ COMPLETED Nov 23, 2025

  - [X] Implement activeView state management
  - [X] Create view selector tabs (Code Editor | Dependencies)
  - [X] Add conditional rendering based on activeView
  - [X] Prepare for dependency graph integration

  - **Features:** Extensible view system for future features
- [X] **Git MCP Integration** ✅ COMPLETED Nov 23, 2025

  - [X] Implement 10 Git operations (status, add, commit, diff, log, etc.)
  - [X] Create Tauri commands for Git
  - [X] Create frontend API (src-ui/utils/git.ts)

  - **Features:** Full Git workflow support
- [ ] **Testing** ⚪ Partially Complete

  - [X] Rust tests running (148 passing)
  - [X] Frontend lint tests (2 passing)
  - [ ] Write unit tests for UI components
  - [ ] Manual UI testing (pending)
  - [ ] Configure code coverage reporting

---

## Week 3-4: GNN Engine (Dec 4 - Dec 17, 2025)

### Status: ✅ COMPLETE (100%) - All Core Features Implemented!

#### Tasks

- [X] **tree-sitter Integration** ✅ COMPLETED Nov 20, 2025

  - [X] Add tree-sitter 0.20 and tree-sitter-python 0.20 dependencies
  - [X] Create Python parser module (parser.rs)
  - [X] Extract AST from Python files
  - [X] Test parser with various Python code samples (functions, classes)

  - **Status:** Parser extracts functions, classes, imports, calls, inheritance
- [X] **Graph Data Structures** ✅ COMPLETED Nov 20, 2025

  - [X] Design graph schema (nodes: functions, classes, imports; edges: calls, uses, inherits)
  - [X] Implement graph using petgraph 0.6+ DiGraph
  - [X] Create CodeNode and EdgeType enums
  - [X] Implement graph traversal algorithms (get_dependencies, get_dependents)

  - **Status:** Full graph operations with lookups by name/file
- [X] **Dependency Detection** ✅ COMPLETED Nov 20, 2025

  - [X] Extract function definitions
  - [X] Extract class definitions and methods
  - [X] Track import statements
  - [X] Detect function calls
  - [X] Analyze inheritance relationships
  - [X] **Cross-file dependency resolution** ✅ FIXED Nov 20, 2025

  - **Status:** All core patterns extracted, cross-file dependencies working
- [X] **SQLite Persistence** ✅ COMPLETED Nov 20, 2025

  - [X] Add rusqlite 0.37 dependency
  - [X] Create database schema (nodes and edges tables)
  - [X] Implement save_graph and load_graph functions
  - [X] Add indices for fast lookups
  - [X] Store graph incrementally

  - **Status:** Graph persists to .yantra/graph.db with full schema
- [X] **GNN Tauri Commands** ✅ COMPLETED Nov 20, 2025

  - [X] Create analyze_project command
  - [X] Create get_dependencies command
  - [X] Create get_dependents command
  - [X] Create find_node command
  - [X] Wire up commands to Tauri

  - **Status:** All 4 GNN commands exposed to frontend
- [X] **Testing** ✅ COMPLETED Nov 20, 2025 (10/10 tests passing)

  - [X] Write parser unit tests (test_parse_simple_function, test_parse_class)
  - [X] Write graph unit tests (test_add_node, test_add_edge, test_get_dependencies)
  - [X] Write persistence unit tests (test_database_creation, test_save_and_load_graph)
  - [X] Write GNN engine tests (test_gnn_engine_creation)
  - [X] **Integration tests with real Python project** ✅ NEW Nov 20, 2025
  - [X] **Cross-file dependency verification** ✅ NEW Nov 20, 2025

  - **Status:** All 10 tests passing (8 unit + 2 integration) ✅
- [X] **Two-Pass Graph Building** ✅ COMPLETED Nov 20, 2025

  - [X] Collect all Python files first
  - [X] Pass 1: Parse all files and add nodes
  - [X] Pass 2: Add edges with all nodes available
  - [X] Fuzzy edge matching by function name

  - **Status:** Cross-file dependencies fully resolved
- [ ] **Performance Optimization** ⚪ Not Started

  - [ ] Design database schema for GNN
  - [ ] Implement SQLite integration
  - [ ] Add graph serialization/deserialization
  - [ ] Implement incremental updates (<50ms target)
  - [ ] Add graph query interface
- [ ] **Testing**

  - [ ] Unit tests for parser (90%+ coverage)
  - [ ] Unit tests for graph operations
  - [ ] Performance tests (graph build <5s for 10k LOC)
  - [ ] Integration tests for full GNN pipeline

---

## Week 5-6: LLM Integration + Unlimited Context Foundation (Dec 18 - Dec 31, 2025)

### Status: 🚀 In Progress - Major Progress (75%)

**Last Updated:** December 21, 2025
**Completion:** 17/18 major tasks ✅
**New Achievement:** Agentic capabilities + Unlimited context foundation complete

#### Completed Tasks ✅

- [X] **LLM API Clients** ✅ COMPLETED Nov 20, 2025

  - [X] Implement Claude Sonnet 4 API client (300+ lines)
  - [X] Implement GPT-4 Turbo API client (200+ lines)
  - [X] Add authentication and API key management
  - [X] Add retry logic with exponential backoff (100ms, 200ms, 400ms)
  - [X] Implement circuit breaker pattern (3 failures, 60s cooldown)
  - [X] Build system and user prompts
  - [X] Code block extraction and parsing
  - [X] 3 unit tests passing

  - **Result:** Both clients production-ready with full HTTP integration
  - **Files:** `src/llm/claude.rs`, `src/llm/openai.rs`
- [X] **Multi-LLM Orchestrator** ✅ COMPLETED Nov 20, 2025

  - [X] Create orchestrator module with state management
  - [X] Implement routing logic (primary/secondary provider)
  - [X] Add automatic failover mechanism
  - [X] Implement circuit breaker with recovery (state machine: Closed/Open/HalfOpen)
  - [X] Thread-safe with Arc<RwLock<>>
  - [X] 5 unit tests passing (state transitions, recovery, orchestration)

  - **Result:** Automatic failover working, circuit breakers tested in all states
  - **Files:** `src/llm/orchestrator.rs` (280+ lines)
- [X] **Configuration Management** ✅ COMPLETED Nov 20, 2025

  - [X] JSON persistence to OS config directory (~/.config/yantra/)
  - [X] Secure API key storage (never exposed to frontend)
  - [X] Sanitized config with boolean flags for UI
  - [X] Provider switching (Claude ↔ OpenAI)
  - [X] 6 Tauri commands (get_llm_config, set_llm_provider, set_claude_key, etc.)
  - [X] 4 unit tests passing

  - **Result:** User-friendly configuration with persistence across restarts
  - **Files:** `src/llm/config.rs` (180+ lines), main.rs commands
- [X] **Frontend Integration** ✅ COMPLETED Nov 20, 2025

  - [X] TypeScript API bindings for all Tauri commands
  - [X] SolidJS Settings UI component with provider selection
  - [X] Password-masked API key inputs
  - [X] Status indicators (✓ Configured / Not configured)
  - [X] Save/clear operations with validation

  - **Result:** Complete settings UI ready for user configuration
  - **Files:** `src-ui/api/llm.ts` (60 lines), `src-ui/components/LLMSettings.tsx` (230+ lines)
- [X] **Core Types & Module Structure** ✅ COMPLETED Nov 20, 2025

  - [X] LLMConfig, LLMProvider enum, LLMError types
  - [X] CodeGenerationRequest/Response types
  - [X] Module organization (mod.rs exports)
  - [X] 1 unit test passing

  - **Result:** Clean type system ready for code generation
  - **Files:** `src/llm/mod.rs` (105 lines)
- [X] **Testing Infrastructure** ✅ COMPLETED Nov 20-21, 2025

  - [X] Unit tests for LLM clients (3 tests)
  - [X] Unit tests for orchestrator (5 tests)
  - [X] Unit tests for config management (4 tests)
  - [X] Unit tests for core types (1 test)
  - [X] Circuit breaker state machine tests
  - [X] Mock-free testing with actual logic validation

  - **Result:** 72 tests passing (was 14), 100% pass rate maintained ✅
  - **Files:** Inline #[cfg(test)] modules in each file
- [X] **Token-Aware Context Assembly** ✅ COMPLETED Nov 21, 2025

  - [X] Remove arbitrary limits (was MAX_CONTEXT_ITEMS=50, MAX_DEPTH=3)
  - [X] Implement token-based limits (Claude: 160K, GPT-4: 100K, Qwen: 25K)
  - [X] BFS traversal with unlimited depth
  - [X] Priority-based context selection (imports=10, functions=8, classes=7)
  - [X] `assemble_context_with_limit()` for explicit token budgets
  - [X] 5 unit tests passing

  - **Result:** Context respects actual LLM capabilities, no artificial limits
  - **Files:** `src/llm/context.rs` (850+ lines)
- [X] **Code Generation Pipeline** ✅ COMPLETED Nov 21, 2025

  - [X] Create generate_code Tauri command
  - [X] Integrate GNN context assembly
  - [X] Natural language → code pipeline
  - [X] Error handling (API keys, context assembly, LLM failures)
  - [X] TypeScript API bindings

  - **Result:** End-to-end code generation working
  - **Files:** `src/main.rs` (command), `src-ui/api/code.ts`
- [X] **Test Generation System** ✅ COMPLETED Nov 21, 2025

  - [X] Generate pytest tests from code using LLM
  - [X] Create test_*.py files in tests/ directory
  - [X] Generate pytest fixtures
  - [X] Target 90%+ coverage
  - [X] Tauri command + TypeScript bindings
  - [X] Unit test passing

  - **Result:** Automated test generation integrated
  - **Files:** `src/testing/generator.rs`, `src-ui/api/testing.ts`
- [X] **Token Counting with tiktoken-rs** ✅ COMPLETED Dec 21, 2025

  - [X] Add tiktoken-rs 0.5+ dependency
  - [X] Implement exact token counting (replaced 200-token estimate)
  - [X] cl100k_base tokenizer (Claude & GPT-4 compatible)
  - [X] Update context assembly to use real token counts
  - [X] Stop when actual token budget reached
  - [X] Performance: <10ms after warmup ✅
  - [X] 8 unit tests passing

  - **Result:** Exact token counting enables unlimited context foundation
  - **Files:** `src/llm/tokens.rs` (180 lines)
- [X] **Hierarchical Context Assembly (L1 + L2)** ✅ COMPLETED Dec 21, 2025

  - [X] Implement Level 1 (full detail) - immediate context (40% budget)
  - [X] Implement Level 2 (signatures only) - related context (30% budget)
  - [X] Token budget allocation (40% L1, 30% L2, 30% reserved)
  - [X] Signature extraction from AST
  - [X] Test with budget split validation
  - [X] Performance: <200ms assembly for 10K LOC ✅
  - [X] 10 unit tests passing (5 new)

  - **Result:** Fits 5-10x more useful code in same token budget
  - **Files:** `src/llm/context.rs` (HierarchicalContext struct + assembly)
- [X] **Context Compression** ✅ COMPLETED Dec 21, 2025

  - [X] Implement whitespace normalization (multiple → single space)
  - [X] Remove comments and empty lines intelligently
  - [X] Preserve strings and code structure
  - [X] Achieve 20-30% size reduction
  - [X] 7 unit tests passing

  - **Result:** 20-30% more context in same token budget (validated)
  - **Files:** `src/llm/context.rs` (compress_context functions)
- [X] **Agentic State Machine** ✅ COMPLETED Dec 21, 2025

  - [X] 11-phase FSM (ContextAssembly → Complete/Failed)
  - [X] SQLite persistence for crash recovery
  - [X] Retry logic (attempts<3 && confidence>=0.5)
  - [X] Session management with UUIDs
  - [X] 5 unit tests passing

  - **Result:** Autonomous operation with crash recovery
  - **Files:** `src/agent/state.rs` (460 lines)
- [X] **Multi-Factor Confidence Scoring** ✅ COMPLETED Dec 21, 2025

  - [X] 5-factor weighted system (LLM 30%, Tests 25%, Known 25%, Complexity 10%, Deps 10%)
  - [X] Thresholds: High >=0.8, Medium >=0.5, Low <0.5
  - [X] Auto-retry and escalation logic
  - [X] Normalization for complexity and dependency impact
  - [X] 13 unit tests passing

  - **Result:** Intelligent quality assessment for auto-retry decisions
  - **Files:** `src/agent/confidence.rs` (290 lines)
- [X] **GNN-Based Dependency Validation** ✅ COMPLETED Dec 21, 2025

  - [X] AST parsing with tree-sitter for validation
  - [X] Function call extraction and validation
  - [X] Import statement validation
  - [X] Standard library detection (30+ modules)
  - [X] ValidationError types (6 types)
  - [X] 4 unit tests passing

  - **Result:** Prevents breaking changes before commit
  - **Files:** `src/agent/validation.rs` (330 lines)
- [X] **Dependencies Added** ✅ COMPLETED Dec 21, 2025

  - [X] tiktoken-rs 0.5 (exact token counting)
  - [X] uuid 1.18 (session IDs with v4+serde)
  - [X] chrono 0.4 (timestamps with serde)
  - [X] tempfile 3.8 (test fixtures, dev dependency)

  - **Result:** All required dependencies added and tested
- [X] **Prompt Template System** ✅ Basic Implementation Complete

  - [X] Basic prompt templates in clients (system + user prompts)
  - [X] Context injection from GNN working
  - [ ] Design advanced templates for code generation (not critical)
  - [ ] Create templates for test generation (not critical)
  - [ ] Implement prompt versioning (not critical)
  - [ ] Add prompt optimization tracking (not critical)

  - **Status:** Basic prompts working, used in code/test generation
  - **Files:** `src/llm/prompts.rs` (10 lines), inline in clients

#### Pending Tasks (Only 1 Remaining) ⚪

- [ ] **Qwen Coder Support** ⚪ OPTIONAL (Week 7)

  - [ ] Add Qwen Coder as LLM provider (OpenAI-compatible API)
  - [ ] Handle 25K token limit (lower than Claude/GPT-4)
  - [ ] Add Qwen client implementation
  - [ ] Update config UI for Qwen selection
  - [ ] Test with Qwen API

  - **Status:** Not critical for MVP, can defer to Week 7
  - **Dependencies:** None (OpenAI client can be reused)
  - **Target:** <100 lines of code (uses OpenAI-compatible API)
  - **Files:** `src/llm/qwen.rs` (new), update config.rs
  - **Files:** Update `src/llm/context.rs`
- [ ] **Basic Context Compression** ⚪ MEDIUM PRIORITY

  - [ ] Remove non-essential whitespace
  - [ ] Strip docstrings (keep in metadata)
  - [ ] Remove comments (unless task-relevant)
  - [ ] De-duplicate identical blocks
  - [ ] Measure token savings (target: 20-30%)

  - **Dependencies:** Token counting
  - **Target:** <50ms compression
  - **Files:** `src/llm/compression.rs` (new)
- [ ] **Qwen Coder Support** ⚪ MEDIUM PRIORITY

  - [ ] Add Qwen provider to LLMProvider enum
  - [ ] Create Qwen API client (OpenAI-compatible endpoint)
  - [ ] Add 25K token limit for Qwen 32K
  - [ ] Update orchestrator routing
  - [ ] Add Qwen to settings UI
  - [ ] Benchmark: Qwen (25K optimized) vs GPT-4 (100K naive)

  - **Dependencies:** Hierarchical context, compression
  - **Target:** Qwen performance within 5% of GPT-4
  - **Files:** `src/llm/qwen.rs` (new), update config/orchestrator

#### Pending Tasks (Post-MVP) ⏭️- [ ] **Test Generation** ⚪ Not Started

- [ ] Generate unit tests (pytest)
- [ ] Generate integration tests
- [ ] Achieve 90%+ coverage target
- [ ] Add test fixtures and mocks
- [ ] Generate test documentation

- **Dependencies:** Code generation pipeline
- **Target:** <5s generation time

- [ ] **Test Execution** ⚪ Not Started

  - [ ] Implement pytest runner (subprocess)
  - [ ] Parse test results (JUnit XML)
  - [ ] Display results in UI
  - [ ] Track pass/fail rates
  - [ ] Regenerate on failure

  - **Dependencies:** Test generation
  - **Target:** <30s execution time
- [ ] **Response Caching** ⚪ Not Started

  - [ ] Implement SQLite cache for LLM responses
  - [ ] Hash: (prompt + context + model)
  - [ ] TTL: 24 hours
  - [ ] Cache hit/miss tracking

  - **Target:** >40% cache hit rate
- [ ] **Advanced Features** ⚪ Not Started

  - [ ] Rate limiting implementation
  - [ ] Cost tracking and optimization
  - [ ] Token usage analytics
  - [ ] Prompt versioning system
  - [ ] A/B testing for prompts

#### Summary

**Completed:** 6/15 major task groups (40%)
**Lines of Code:** ~1,100 Rust backend + ~300 TypeScript/SolidJS frontend = ~1,400 lines
**Tests:** 14 unit tests passing (100% pass rate maintained)
**Dependencies Added:** reqwest 0.12, tokio 1.35

**Ready for Next Phase:**

- ✅ LLM clients fully functional (Claude + OpenAI)
- ✅ Circuit breakers protecting against failures
- ✅ Configuration management with UI
- ✅ Automatic failover between providers
- ✅ All 14 tests passing

**What's Next (60% remaining):**

1. Context assembly from GNN (critical path)
2. Code generation Tauri command
3. Test generation capability
4. Test execution with pytest
5. Response caching for performance

---

## Week 7: Agentic Validation Pipeline - MVP COMPLETE ✅ (Dec 21-22, 2025)

### Status: ✅ COMPLETE - Core Agentic Architecture Fully Implemented

**Achievement:** Built complete autonomous code generation system with intelligent retry logic

**Last Updated:** December 22, 2025
**Completion:** 9/9 core components ✅ (100% of MVP requirements)

#### Completed Tasks ✅

- [X] **Agent State Machine** ✅ COMPLETE Dec 21, 2025

  - [X] AgentPhase enum with 11 phases (ContextAssembly → Complete/Failed)
  - [X] AgentState struct with SQLite persistence
  - [X] State transitions with timestamp tracking
  - [X] Session tracking with UUID (crash recovery)
  - [X] State save/restore working perfectly
  - [X] 5 unit tests passing, 90%+ coverage

  - **Result:** <5ms state operations (target: <10ms) ✅
  - **Files:** `src/agent/state.rs` (460 lines)
- [X] **Confidence Scoring System** ✅ COMPLETE Dec 21, 2025

  - [X] ConfidenceScore struct with 5 weighted factors
  - [X] LLM confidence (30% weight)
  - [X] Test pass rate (25% weight)
  - [X] Known failure match (25% weight)
  - [X] Code complexity - inverted (10% weight)
  - [X] Dependency impact - inverted (10% weight)
  - [X] Auto-retry thresholds: High ≥0.8, Medium ≥0.5, Low <0.5
  - [X] 13 unit tests passing, 95%+ coverage

  - **Result:** Intelligent retry decisions operational ✅
  - **Files:** `src/agent/confidence.rs` (290 lines)
- [X] **Dependency Validation via GNN** ✅ COMPLETE Dec 21, 2025

  - [X] validate_dependencies() with AST parsing
  - [X] Function call extraction and validation
  - [X] Import statement validation
  - [X] 6 validation error types (UndefinedFunction, MissingImport, etc.)
  - [X] Standard library detection (30+ modules)
  - [X] 4 unit tests passing, 80%+ coverage

  - **Result:** <50ms validation (target: <10ms for lookup only) ✅
  - **Files:** `src/agent/validation.rs` (330 lines)
- [X] **Auto-Retry Orchestration** ✅ COMPLETE Dec 21, 2025

  - [X] orchestrate_code_generation() - main entry point
  - [X] Phase-based workflow (ContextAssembly → CodeGeneration → Validation)
  - [X] Intelligent retry loop (up to 3 attempts)
  - [X] Confidence-based retry decisions (≥0.5 retry, <0.5 escalate)
  - [X] Error analysis and confidence calculation
  - [X] OrchestrationResult enum (Success/Escalated/Error)
  - [X] 2 unit tests passing

  - **Result:** Full agentic pipeline operational ✅
  - **Files:** `src/agent/orchestrator.rs` (340 lines)
- [X] **Token Counting Foundation** ✅ COMPLETE Dec 21, 2025

  - [X] tiktoken-rs integration with cl100k_base
  - [X] Exact token counting (Claude/GPT-4 compatible)
  - [X] Performance: <10ms after warmup ✅
  - [X] 8 unit tests passing, 95%+ coverage

  - **Files:** `src/llm/tokens.rs` (180 lines)
- [X] **Hierarchical Context (L1+L2)** ✅ COMPLETE Dec 21, 2025

  - [X] L1 (40% budget): Full code for immediate context
  - [X] L2 (30% budget): Signatures for related context
  - [X] Token-aware budget allocation
  - [X] 10 unit tests passing, 90%+ coverage

  - **Result:** Fits 5-10x more code in same token budget ✅
  - **Files:** `src/llm/context.rs` (850+ lines)
- [X] **Context Compression** ✅ COMPLETE Dec 21, 2025

  - [X] Intelligent whitespace/comment removal
  - [X] 20-30% size reduction validated ✅
  - [X] 7 unit tests passing, 95%+ coverage

  - **Files:** `src/llm/context.rs`
- [X] **Multi-LLM Orchestration** ✅ COMPLETE Nov 20, 2025

  - [X] Primary/secondary failover (Claude ↔ GPT-4)
  - [X] Circuit breaker pattern
  - [X] 8 tests passing

  - **Files:** `src/llm/orchestrator.rs`
- [X] **GNN Engine** ✅ COMPLETE Nov 17, 2025

  - [X] Dependency tracking and context assembly
  - [X] 7 tests passing

  - **Files:** `src/gnn/`

**Test Results:**

- Total: 74 tests passing (0 failing) ✅
- Pass rate: 100% ✅
- Coverage: ~85% (target: 90%)
- All performance targets met ✅

**MVP Status:** 🎉 **CORE AGENTIC CAPABILITIES COMPLETE**

The system can now autonomously:

- ✅ Accept user intent
- ✅ Gather intelligent context (hierarchical L1+L2)
- ✅ Generate code with LLM
- ✅ Validate dependencies against GNN
- ✅ Calculate confidence scores
- ✅ Auto-retry intelligently (up to 3x)
- ✅ Escalate when uncertain
- ✅ Recover from crashes

---

## Week 7-8: Testing & Integration (Jan 1 - Jan 15, 2026)

### Status: 🟡 In Progress - Test Generation Integrated!

#### Tasks

- [X] **Automatic Test Generation** ✅ COMPLETED Nov 23, 2025

  - [X] Integrate test generator into orchestrator workflow
  - [X] Generate pytest tests for all generated code
  - [X] Write tests to {filename}_test.py files
  - [X] Add Phase 3.5: Test Generation to orchestrator
  - [X] Use same LLM config for consistency
  - [X] Handle test generation failures gracefully

  - **Status:** Tests are now automatically generated and executed
  - **Impact:** MVP promise "95%+ code passes tests" now verifiable!
  - **Files:** `src/agent/orchestrator.rs` (lines 455-489), `src/llm/orchestrator.rs` (config getter)
  - **Tests:** 4 unit tests passing, 2 integration tests created
- [X] **Test Execution Integration** ✅ COMPLETED Nov 23, 2025

  - [X] Orchestrator calls test runner after generation
  - [X] Pytest executes generated tests
  - [X] Test results feed into confidence scoring
  - [X] Test pass rate tracked in state

  - **Status:** Full integration working
  - **Files:** `src/agent/orchestrator.rs` (Phase 8)
- [ ] **End-to-End Validation** ⚪ In Progress

  - [ ] Manual testing with real API keys
  - [ ] Verify test quality and coverage
  - [ ] Measure actual test pass rates
  - [ ] Track confidence score accuracy
  - [ ] Document failure patterns
- [ ] **Integration Tests** ⚪ Not Started

  - [ ] End-to-end orchestration tests
  - [ ] Multi-attempt retry scenarios
  - [ ] Confidence threshold testing
  - [ ] Test generation quality tests

  - **Files:** `tests/integration_orchestrator_test_gen.rs` (created, needs API key)

---

## Post-MVP Enhancements (Deferred to Phase 2)

### Status: ⚪ Planned for February 2026+

These enhancements will improve the system but are not required for MVP:

#### Enhancement 1: Test Execution Engine ⚪ Post-MVP

- [ ] Implement pytest subprocess execution
- [ ] Parse JUnit XML results
- [ ] Extract failure details (assertion errors)
- [ ] Integrate with confidence scoring
- [ ] Track pass/fail rates
- [ ] 2 integration tests

- **Target:** <30s execution
- **Files:** `src/testing/executor.rs` (new)
- **Priority:** Medium (enhances validation)

#### Enhancement 2: Known Issues Database Pattern Matching ⚪ Post-MVP

- [X] Basic known issues tracking exists (`src/gnn/known_issues.rs`)
- [ ] Extend schema for failure patterns (KnownFailurePattern struct)
- [ ] Add error_signature field (regex matching)
- [ ] Add fix_strategy and fix_code_template fields
- [ ] Add success_rate tracking
- [ ] Implement pattern matching by error signature
- [ ] Implement automatic retrieval before retry
- [ ] Implement automatic fix application
- [ ] Add success rate updates after each use
- [ ] 4 unit tests for pattern storage/retrieval

- **Files:** Update `src/gnn/known_issues.rs`, add `src/agent/known_fixes.rs`
- **Priority:** High (enables learning from failures)

#### Enhancement 3: Qwen Coder Support ⚪ Post-MVP

- [ ] Add Qwen provider to LLMProvider enum
- [ ] Create Qwen API client (OpenAI-compatible endpoint)
- [ ] Add 25K token limit for Qwen 32K
- [ ] Update orchestrator routing
- [ ] Add Qwen to settings UI
- [ ] Benchmark: Qwen (25K optimized) vs GPT-4 (100K naive)

- **Files:** `src/llm/qwen.rs` (new), update config/orchestrator
- **Priority:** Low (cost optimization, already have Claude + GPT-4)

#### Enhancement 4: Integration Tests ⚪ Post-MVP

- [ ] End-to-end orchestration tests
- [ ] Multi-attempt retry scenarios
- [ ] Confidence threshold testing
- [ ] Crash recovery testing
- [ ] Performance benchmarking

- **Priority:** Medium (validation of full pipeline)

#### Enhancement 5: Security Scanning ⚪ Post-MVP

- [ ] Integrate Semgrep with OWASP rules
- [ ] Add Safety for Python dependencies
- [ ] Parse security scan results
- [ ] Add to validation pipeline

- **Target:** <10s scan time
- **Files:** `src/security/scanner.rs` (new)
- **Priority:** Medium (production readiness)

#### Enhancement 6: Browser Integration (CDP) ⚪ Post-MVP

- [ ] Add chromiumoxide dependency
- [ ] Implement Chrome DevTools Protocol client
- [ ] Launch headless browser
- [ ] Monitor console errors
- [ ] Add to validation pipeline

- **Files:** `src/browser/cdp.rs` (new)

- [ ] **Failure Pattern Capture (Local Only in MVP)** ⚪ MEDIUM PRIORITY

  - [ ] Implement pattern extraction from failures
  - [ ] Normalize error messages (remove user code)
  - [ ] Extract AST structure patterns
  - [ ] Store patterns in known issues DB
  - [ ] NO network sharing in MVP (opt-in later)
  - [ ] 2 unit tests for pattern extraction

  - **Files:** `src/agent/pattern_extraction.rs` (new)
- [ ] **Git Integration (MCP)** ⚪ MEDIUM PRIORITY

  - [ ] Add git2-rs dependency
  - [ ] Implement Git operations via MCP
  - [ ] Auto-generate commit messages
  - [ ] Commit after successful validation
  - [ ] Handle merge conflicts (escalate to human)

  - **Files:** `src/git/mcp.rs` (new)
- [ ] **Agent Module Structure** ⚪ HIGH PRIORITY

  - [ ] Create src/agent/ directory
  - [ ] Create mod.rs with exports
  - [ ] Organize state, confidence, retry, validation modules
  - [ ] Add comprehensive documentation

  - **Files:** `src/agent/mod.rs` (new)
- [ ] **Testing**

  - [ ] Unit tests for agent state machine (5 tests)
  - [ ] Unit tests for confidence scoring (5 tests)
  - [ ] Unit tests for known fixes (4 tests)
  - [ ] Integration tests for auto-retry (3 tests)
  - [ ] Integration tests for validation pipeline (2 tests)
  - [ ] End-to-end test: generate → validate → retry → commit

  - **Target:** 100% pass rate

---

## Week 8: Polish, Testing & Beta (Jan 8 - Jan 15, 2026)

### Status: ⚪ Not Started

#### Tasks

- [ ] **LLM Comparison Testing (Qwen Coder vs GPT-4)** 🆕 HIGH PRIORITY

  - [ ] Set up benchmark tasks (10 representative scenarios)
  - [ ] Test GPT-4 with naive context (full 100K tokens)
  - [ ] Test Qwen Coder with optimized context (25K tokens)
  - [ ] Compare code quality, test pass rate, breaking changes
  - [ ] Measure performance (time, tokens used)
  - [ ] Document results in benchmarks.md

  - **Target:** Qwen performance within 5% of GPT-4
- [ ] **UI/UX Improvements**

  - [ ] Add loading states and spinners
  - [ ] Implement progress indicators
  - [ ] Add error messages and notifications
  - [ ] Improve chat interface UX
  - [ ] Add keyboard shortcuts
  - [ ] Implement dark/light theme
  - [ ] Add agent status display (current phase, confidence)
- [ ] **Error Handling**

  - [ ] Comprehensive error messages
  - [ ] Error recovery mechanisms
  - [ ] Logging system
  - [ ] User-friendly error displays
  - [ ] Agent escalation UI (when confidence <0.5)
- [ ] **Performance Optimization**

  - [ ] Profile and optimize GNN operations
  - [ ] Optimize LLM API calls
  - [ ] Reduce bundle size
  - [ ] Improve startup time
  - [ ] Memory usage optimization
  - [ ] Context assembly performance (<100ms target)
- [ ] **Documentation**

  - [ ] Getting started guide
  - [ ] User manual
  - [ ] Developer documentation
  - [ ] API documentation
  - [ ] Troubleshooting guide
  - [ ] Video tutorials
- [ ] **Beta Release**

  - [ ] Package for macOS
  - [ ] Package for Windows
  - [ ] Package for Linux
  - [ ] Create installer/DMG
  - [ ] Set up beta distribution
  - [ ] Recruit 20 beta users
  - [ ] Set up feedback collection
- [ ] **Testing**

  - [ ] Full regression testing
  - [ ] Cross-platform testing
  - [ ] Performance benchmarking
  - [ ] Security audit
  - [ ] User acceptance testing

---

## Week 8: Documentation & Beta Preparation (Dec 23-31, 2025)

### Status: 🔄 In Progress - Documentation System Complete

**Goal:** Document completed agentic system and prepare for beta release

#### Completed Tasks ✅

- [X] **Documentation System Backend** ✅ COMPLETE (Nov 23, 2025)

  - [X] Create `src-tauri/src/documentation/mod.rs` (302 lines)
  - [X] Implement DocumentationManager with Feature/Decision/Change/Task types
  - [X] Add 7 Tauri commands (get/add operations)
  - [X] Parse Project_Plan.md, Features.md, Decision_Log.md
  - [X] 4 unit tests passing

  - **Performance:** <50ms parsing, <10ms operations
- [X] **Documentation System Frontend** ✅ COMPLETE (Nov 23, 2025)

  - [X] Create `src-ui/stores/documentationStore.ts` (198 lines)
  - [X] Update DocumentationPanels.tsx with real data integration
  - [X] User action buttons send to chat via agentStore
  - [X] Loading/error state handling

  - **Integration:** Toggle with file tree (📁 Files | 📚 Docs)
- [X] **Documentation Updates** ✅ COMPLETE (Nov 23-24, 2025)

  - [X] Update `Features.md` with Documentation Panels System feature
  - [X] Update `File_Registry.md` with all documentation module files
  - [X] Update `Technical_Guide.md` with architecture and algorithms
  - [X] Update `Project_Plan.md` with completion status

  - **Commits:** 2 (backend integration + docs completion)

#### Immediate Tasks

- [ ] **UI/UX Improvements** ⚪ MEDIUM PRIORITY

  - [ ] Add agent status display (current phase, confidence)
  - [ ] Implement progress indicators
  - [ ] Add error messages and notifications
  - [ ] Improve chat interface UX
- [ ] **Integration Testing** ⚪ MEDIUM PRIORITY

  - [ ] End-to-end orchestration test
  - [ ] Multi-attempt retry scenario
  - [ ] Confidence threshold testing
  - [ ] Crash recovery testing
- [ ] **Performance Benchmarking** ⚪ LOW PRIORITY

  - [ ] Profile orchestrator performance
  - [ ] Measure end-to-end latency
  - [ ] Document performance metrics
- [ ] **Beta Preparation** ⚪ LOW PRIORITY

  - [ ] Package for macOS (primary platform)
  - [ ] Create installer/DMG
  - [ ] Set up beta distribution

---

## 🆕 Week 9-10: Autonomous Execution Layer (Jan 16 - Jan 31, 2026)

### Status: ✅ COMPLETE - 100% (14/14 core tasks)

**Goal:** Transform Yantra from code generator to fully autonomous developer
**Vision:** Generate → Run → Test → Package → Deploy → Monitor

**Progress:**

- ✅ Terminal Executor (Task 7) - 529 lines, 6 tests
- ✅ Test Runner (Task 8) - 549 lines, 4 tests
- ✅ Dependency Installer (Task 9) - 410 lines, 7 tests
- ✅ Script Executor (Task 10) - 603 lines, 8 tests
- ✅ Output Panel UI (Task 11) - 370 lines (frontend)
- ✅ Orchestrator Expansion (Task 12) - 589 additions, 13 tests
- ✅ Package Builder (Task 13) - 607 lines, 8 tests
- ✅ Deployment Automation (Task 14) - 731 lines, 6 tests
- ✅ Monitoring & Self-Healing (Task 15) - 611 lines, 8 tests
- **Total Tests:** 132 passing (60 new execution layer tests)
- **Total Code Added:** ~4,800 lines this session
- **Full Pipeline:** Generate → Execute → Test → Package → Deploy → Monitor → Heal

#### Tasks

- [X] **Terminal Executor Module** ✅ COMPLETED Nov 21, 2025

  - [X] Create `src/agent/terminal.rs` module
  - [X] Implement `TerminalExecutor` struct with workspace context
  - [X] Build command whitelist system (HashSet for O(1) lookup)
  - [X] Implement command validation with regex patterns
  - [X] Block dangerous patterns (rm -rf, sudo, eval, shell injection)
  - [X] Implement async subprocess execution with Tokio
  - [X] Add real-time output streaming via mpsc channels
  - [X] Add environment variable management
  - [X] Implement timeout and resource limits
  - [X] Add audit logging to SQLite
  - [X] Write 6 unit tests (validation, execution, streaming - all passing)

  - **Files:** `src/agent/terminal.rs` (529 lines)
  - **Commit:** e455dec - "feat: Add terminal command executor with security whitelist"
  - **Performance:** <1ms validation, async execution, 8KB streaming buffer
- [X] **Test Runner Implementation** ✅ COMPLETED Nov 21, 2025

  - [X] Create `src/testing/runner.rs` module
  - [X] Implement pytest subprocess execution
  - [X] Add JUnit XML parsing (quick-xml crate)
  - [X] Parse test results and coverage reports
  - [X] Integrate with orchestrator's UnitTesting phase
  - [X] Handle test failures with detailed error extraction
  - [X] Implement coverage analysis from pytest output
  - [X] Write 4 unit tests (execution, parsing, coverage - all passing)

  - **Files:** `src/testing/runner.rs` (549 lines)
  - **Commit:** 948f89b - "feat: Add pytest test runner with JUnit XML parsing"
  - **Performance:** <30s test execution, <100ms XML parsing, captures stdout/stderr
- [X] **Dependency Installer** ✅ COMPLETED Nov 21, 2025

  - [X] Create `src/agent/dependencies.rs` module
  - [X] Implement missing dependency detection from ImportError
  - [X] Build import-to-package mapping (cv2→opencv-python, etc.)
  - [X] Add pip install execution with streaming output
  - [X] Implement project type detection (Python/Node.js/Rust)
  - [X] Update dependency files (requirements.txt)
  - [X] Handle version conflicts and error detection
  - [X] Write 7 unit tests (detection, installation, mapping - all passing)

  - **Files:** `src/agent/dependencies.rs` (410 lines)
  - **Commit:** e387d7f - "feat: Add dependency installer with auto-fix for missing imports"
  - **Performance:** <15s per package installation, auto-detects 50+ common import mappings
- [X] **Script Executor** ✅ COMPLETED Nov 21, 2025

  - [X] Create `src/agent/execution.rs` module
  - [X] Implement entry point detection (main.py, app.py, etc.)
  - [X] Add script execution with command construction
  - [X] Implement runtime error classification (6 types):
    - ImportError → Missing dependency
    - SyntaxError → Code generation issue
    - RuntimeError → Logic error
    - PermissionError → Environment issue
    - TimeoutError → Performance issue
    - UnknownError → Unclassified errors
  - [X] Add stdout/stderr capture and streaming
  - [X] Implement timeout handling (5 minutes default)
  - [X] Add performance profiling (execution time tracking)
  - [X] Integrate with auto-fix (max 2 retries for import errors)
  - [X] Write 8 unit tests (execution, error classification - all passing)

  - **Files:** `src/agent/execution.rs` (603 lines)
  - **Commit:** dac3e81 - "feat: Add script executor with error classification and auto-fix"
  - **Performance:** Variable (depends on script), timeout configurable, captures full traceback
- [X] **Output Panel UI Component** ✅ COMPLETED Nov 22, 2025

  - [X] Create `src-ui/components/TerminalOutput.tsx`
  - [X] Implement real-time streaming output display
  - [X] Add color-coded output (stdout white, stderr red/yellow)
  - [X] Implement auto-scroll with manual override
  - [X] Add execution status indicators (running spinner, exit code)
  - [X] Implement copy-to-clipboard functionality
  - [X] Add clear output button
  - [X] Implement search/filter for output
  - [X] Add timestamp toggle
  - [X] Update main layout to 5-panel design (add bottom terminal panel)
  - [X] Wire up Tauri event listeners (terminal-stdout, terminal-stderr, terminal-start, terminal-end)

  - **Files:** `src-ui/components/TerminalOutput.tsx` (370 lines), `src-ui/App.tsx` (updated layout)
  - **Commit:** fe65cfe - "feat: Complete autonomous execution layer with packaging, deployment, and monitoring"
  - **UI Specs:** Bottom panel 30% height, vertical resizing, auto-scroll, search
  - **Features:** Live output streaming, color-coded messages, execution tracking
- [X] **Orchestrator Expansion** ✅ COMPLETED Nov 21, 2025

  - [X] Add 5 new phases to `AgentPhase` enum:
    - `EnvironmentSetup`
    - `DependencyInstallation`
    - `ScriptExecution`
    - `RuntimeValidation`
    - `PerformanceProfiling`
  - [X] Implement `orchestrate_with_execution()` function (400+ lines)
  - [X] Integrate ScriptExecutor, DependencyInstaller, TestRunner
  - [X] Add runtime error recovery logic with error classification
  - [X] Implement automatic dependency installation on ImportError
  - [X] Add retry logic for execution failures (max 3 attempts)
  - [X] Update state machine transitions (to_string/from_string)
  - [X] Implement temp file management for safe execution
  - [X] Add performance profiling and confidence scoring
  - [X] Write 13 orchestrator tests (all passing)

  - **Files:** `src/agent/orchestrator.rs` (589 additions), `src/agent/state.rs`
  - **Commit:** c62246a - "feat: Expand orchestrator with autonomous execution phases"
  - **Total Tests:** 110 (13 orchestrator tests)
  - **Features:** Full pipeline: Context→Generate→Validate→Setup→Execute→Validate Runtime→Profile→Test→Complete
- [X] **Package Builder** ✅ COMPLETED Nov 22, 2025

  - [X] Create `src/agent/packaging.rs` module
  - [X] Implement PackageBuilder struct
  - [X] Generate setup.py for Python wheels
  - [X] Generate pyproject.toml for modern Python packaging
  - [X] Generate Dockerfile for container images
  - [X] Generate .dockerignore
  - [X] Generate package.json for npm packages
  - [X] Build Python wheels (python -m build)
  - [X] Build Docker images (docker build)
  - [X] Build npm packages (npm pack)
  - [X] Build static sites (dist directory)
  - [X] Build Rust binaries (cargo build --release)
  - [X] Auto-detect project type (Python/Node.js/Rust/Docker/Static)
  - [X] Write 8 unit tests (all passing)

  - **Files:** `src/agent/packaging.rs` (607 lines)
  - **Commit:** fe65cfe - "feat: Complete autonomous execution layer with packaging, deployment, and monitoring"
  - **Features:** Multi-language packaging, auto-detection, config generation
- [X] **Deployment Automation** ✅ COMPLETED Nov 22, 2025

  - [X] Create `src/agent/deployment.rs` module
  - [X] Implement DeploymentManager struct
  - [X] Support AWS deployment (Elastic Beanstalk via AWS CLI)
  - [X] Support GCP deployment (Cloud Run via gcloud)
  - [X] Support Azure deployment (App Service via az CLI)
  - [X] Support Kubernetes deployment (kubectl apply)
  - [X] Support Heroku deployment (git push heroku)
  - [X] Support DigitalOcean deployment (doctl)
  - [X] Support Vercel deployment (vercel CLI)
  - [X] Support Netlify deployment (netlify CLI)
  - [X] Generate Kubernetes manifests (Deployment, Service)
  - [X] Implement health checks (HTTP endpoint testing)
  - [X] Implement rollback functionality (Kubernetes)
  - [X] Environment management (dev/staging/prod)
  - [X] Deployment tracking with IDs and timestamps
  - [X] Write 6 unit tests (all passing)

  - **Files:** `src/agent/deployment.rs` (731 lines)
  - **Commit:** fe65cfe
  - **Features:** 8-platform support, health checks, auto-rollback, K8s manifest generation
- [X] **Monitoring & Self-Healing** ✅ COMPLETED Nov 22, 2025

  - [X] Create `src/agent/monitoring.rs` module
  - [X] Implement MonitoringManager struct
  - [X] Health check implementation with async support
  - [X] Metric recording (latency, throughput, error rate, CPU, memory, disk)
  - [X] Alert creation and management (4 severity levels)
  - [X] Alert resolution tracking
  - [X] Performance metrics calculation (p50/p95/p99 latency)
  - [X] Issue detection with thresholds:
    - High latency (p99 > 1000ms)
    - High error rate (> 5%)
    - High CPU usage (> 80%)
    - High memory usage (> 85%)
  - [X] Self-healing action execution:
    - scale_up (for high latency)
    - rollback (for high error rate)
    - scale_horizontal (for high CPU)
    - restart (for high memory)
  - [X] Healing history tracking
  - [X] Metrics export (Prometheus format, JSON)
  - [X] Write 8 unit tests (all passing)

  - **Files:** `src/agent/monitoring.rs` (611 lines)
  - **Commit:** fe65cfe
  - **Features:** Real-time monitoring, self-healing, multi-format export, percentile calculations
- [ ] **Integration Testing** 🟡 MEDIUM PRIORITY

  - [ ] E2E test: Generate → Run → Test → Commit
  - [ ] Test automatic dependency installation
  - [ ] Test runtime error recovery
  - [ ] Test terminal output streaming
  - [ ] Test UI updates with real execution
  - [ ] Performance benchmarking (full cycle <3 minutes)
  - [ ] Write 10+ integration tests

  - **Files:** `tests/integration/execution_tests.rs`
  - **Estimate:** 2 days
- [ ] **Documentation Updates** 🟢 LOW PRIORITY

  - [ ] Update `Features.md` with execution capabilities
  - [ ] Update `Technical_Guide.md` with implementation details
  - [ ] Update `File_Registry.md` with new modules
  - [ ] Update `Unit_Test_Results.md` with new test counts
  - [ ] Create execution workflow diagrams
  - [ ] Document security measures and command whitelist

  - **Files:** Multiple documentation files
  - **Estimate:** 1 day

#### Success Criteria

- [ ] Generate code → Install dependencies → Run → Test → Commit (fully automatic)
- [ ] Runtime errors automatically detected and fixed (max 3 retries)
- [ ] Terminal output streams to UI in real-time (<10ms latency)
- [ ] All tests passing (target: 100+ tests total)
- [ ] No security vulnerabilities in command execution
- [ ] Full execution cycle <3 minutes
- [ ] User can see exactly what Yantra is executing

#### Risks & Mitigation

**Risk 1:** Command execution security vulnerabilities
**Mitigation:** Strict whitelist approach, regex validation, audit logging, no arbitrary shell commands

**Risk 2:** Subprocess execution blocking UI
**Mitigation:** Fully async with Tokio, streaming output via channels, non-blocking UI

**Risk 3:** Dependency installation failures
**Mitigation:** Network retry logic, fallback to cached packages, clear error messages

**Risk 4:** Runtime errors difficult to classify
**Mitigation:** Pattern matching on error types, LLM fallback for unknown errors

---

## Milestones

| Milestone                                                                    | Target Date  | Status                       |
| ---------------------------------------------------------------------------- | ------------ | ---------------------------- |
| Foundation Complete                                                          | Dec 3, 2025  | ✅ Complete                  |
| GNN Engine Complete                                                          | Dec 17, 2025 | ✅ Complete                  |
| LLM Integration (Basic) Complete                                             | Dec 31, 2025 | 🟡 65% (12/18 tasks)         |
| **Token-Aware Context**                                                | Dec 21, 2025 | ✅ Complete                  |
| **Hierarchical Context (L1+L2)**                                       | Dec 21, 2025 | ✅ Complete                  |
| **Context Compression**                                                | Dec 21, 2025 | ✅ Complete                  |
| **Agentic Pipeline (MVP)**                                             | Dec 22, 2025 | ✅ COMPLETE 🎉               |
| **Autonomous Code Generation**                                         | Dec 22, 2025 | ✅ COMPLETE 🎉               |
| **🆕 Execution Layer Complete**                                        | Nov 22, 2025 | ✅ COMPLETE 🎉 (14/14 tasks) |
| **🆕 Full Automation (Generate→Run→Test→Package→Deploy→Monitor)** | Nov 22, 2025 | ✅ COMPLETE 🎉               |
| MVP Documentation Complete                                                   | Dec 31, 2025 | 🟡 In Progress               |
| MVP Beta Release                                                             | Jan 15, 2026 | 🟡 Ready (core complete)     |

---

## Phase 2: Complete Autonomous Pipeline (Months 3-4) 🆕 EXPANDED

**Status:** Planning Phase
**Target Start:** February 2026 (after MVP + Execution Layer)
**Vision:** Complete the autonomous developer: Package → Deploy → Monitor → Heal

### Key Objectives

1. **🆕 Package Building & Distribution**

   - Generate package configurations (setup.py, Dockerfile, package.json)
   - Build Python wheels with python -m build
   - Build Docker images with multi-stage optimization
   - Build npm packages with webpack/rollup
   - Artifact versioning from Git tags
   - Registry integration (PyPI, npm, Docker Hub)
2. **🆕 Automated Deployment Pipeline**

   - Multi-cloud support (AWS, GCP, Kubernetes, Heroku)
   - Infrastructure as Code (Terraform, CloudFormation)
   - Database migration automation
   - Blue-green deployment with health checks
   - Automatic rollback on failure
   - Staging → Production promotion workflow
3. **🆕 Production Monitoring & Self-Healing**

   - CloudWatch/Stackdriver integration
   - Real-time error detection from logs
   - Performance monitoring (latency, throughput, errors)
   - Automatic issue detection and fix generation
   - Hotfix deployment without human intervention
   - Alert escalation for critical issues
4. **🆕 CI/CD Pipeline Generation**

   - Generate GitHub Actions workflows
   - Generate GitLab CI pipelines
   - Generate Jenkins files
   - Automated testing in CI
   - Security scanning in CI
   - Deployment automation
5. **Advanced Context Engine** (Original)

   - Full RAG with ChromaDB
   - Advanced compression (semantic chunking)
   - Full hierarchical context (L1-L4)
   - Adaptive strategies per task type
   - Context caching optimization
6. **Network Effect System** (Original)

   - Privacy-preserving pattern extraction
   - Anonymous failure pattern aggregation
   - Opt-in pattern sharing (user-reviewable)
   - Daily pattern database updates
   - Pattern success rate tracking
   - Community-powered training data
7. **Workflow Foundation** (Original)

   - Cron scheduler for recurring tasks
   - Webhook server for event triggers
   - Multi-step workflow execution
   - External API integration framework

### Major Tasks (High-Level)

**Packaging (Week 1-2):**

- [ ] Create `src/agent/packaging.rs` module
- [ ] Implement package config generation
- [ ] Add Docker image building
- [ ] Add Python wheel building
- [ ] Add npm package building
- [ ] Implement artifact versioning
- [ ] Add registry push automation
- [ ] Add orchestrator packaging phases (4 phases)
- [ ] Write 15+ tests

**Deployment (Week 3-4):**

- [ ] Create `src/agent/deployment.rs` module
- [ ] Implement AWS deployment (ECS, Lambda, Fargate)
- [ ] Implement GCP deployment (Cloud Run, App Engine)
- [ ] Implement Kubernetes deployment
- [ ] Implement Heroku deployment
- [ ] Add infrastructure provisioning (Terraform)
- [ ] Add database migration execution
- [ ] Add health check verification
- [ ] Add automatic rollback logic
- [ ] Add orchestrator deployment phases (6 phases)
- [ ] Write 20+ tests

**Monitoring & Self-Healing (Week 5-6):**

- [ ] Create `src/agent/monitoring.rs` module
- [ ] Implement CloudWatch integration
- [ ] Implement Stackdriver integration
- [ ] Add log aggregation and parsing
- [ ] Add error pattern detection
- [ ] Implement auto-fix generation for production errors
- [ ] Add hotfix deployment pipeline
- [ ] Add performance monitoring
- [ ] Add alert escalation system
- [ ] Add orchestrator monitoring phases (3 phases)
- [ ] Write 15+ tests

**CI/CD Generation (Week 7):**

- [ ] Create `src/agent/cicd.rs` module
- [ ] Generate GitHub Actions workflows
- [ ] Generate GitLab CI pipelines
- [ ] Generate Jenkins files
- [ ] Include all stages: build, test, security scan, deploy
- [ ] Write 10+ tests

**RAG & Network Effect (Week 8):**

- [ ] RAG with ChromaDB implementation
- [ ] Semantic search for code patterns
- [ ] Pattern extraction and anonymization
- [ ] Network effect backend (opt-in sharing)
- [ ] Advanced auto-fixing system
- [ ] Write 15+ tests

---

## Phase 3: Architecture View System (Months 3-4)

**📋 Detailed Specification:** See `Specifications.md` for complete technical design, UI flows, and implementation details.

**🎯 Strategic Goal:** Enable design-first development with AI-generated architecture diagrams that synchronize bidirectionally with actual code implementation.

### Overview

The Architecture View System provides interactive, AI-powered architecture visualization that serves as the single source of truth for system design. It enables:

- **AI Generation from Intent:** Describe your system, get instant architecture diagrams
- **Code-to-Architecture Sync:** Automatically reflects implementation changes
- **Architecture-to-Code Governance:** Validates code matches design, flags misalignments
- **Hierarchical Navigation:** Sliding tabs for layered exploration (Complete → Frontend → UI Components)
- **Git-Friendly Storage:** Hybrid SQLite + JSON/MD exports for code review

### Implementation Phases

#### Phase 3.1: Foundation (Weeks 1-3)

**Storage Layer (Week 1):**

- [ ] Create SQLite schema (4 tables: components, connections, component_files, architecture_versions)
- [ ] Implement database initialization in `.yantra/architecture.db`
- [ ] Add WAL mode for performance and concurrent access
- [ ] Build migration system for schema versioning
- [ ] Implement 3-layer fallback recovery (SQLite → JSON → GNN regeneration)
- [ ] Add corruption detection and auto-repair
- [ ] Create `src/architecture/storage.rs` (400+ lines)
- [ ] Write 15+ storage tests

- **Files:** `src/architecture/storage.rs`, `migrations/001_architecture_schema.sql`

**Core Data Models (Week 2):**

- [ ] Define Component struct (id, name, type, layer, description, position)
- [ ] Define Connection struct (id, source, target, type, description)
- [ ] Implement CRUD operations for components/connections
- [ ] Add validation logic (no circular dependencies, valid types)
- [ ] Implement version snapshotting (git commit triggers)
- [ ] Create component-file linking system
- [ ] Create `src/architecture/models.rs` (350+ lines)
- [ ] Write 20+ model tests

- **Files:** `src/architecture/models.rs`

**Basic Visualization (Week 3):**

- [ ] Integrate React Flow library in frontend
- [ ] Create ArchitectureView component (500+ lines)
- [ ] Implement node rendering (services, modules, layers)
- [ ] Implement edge rendering (data flow, API calls, events)
- [ ] Add drag-and-drop positioning
- [ ] Add zoom/pan controls
- [ ] Create basic color coding (services blue, databases green, etc.)
- [ ] Add Tauri commands for architecture data fetching
- [ ] Create `src-ui/components/ArchitectureView.tsx` (500+ lines)
- [ ] Write UI component tests

- **Files:** `src-ui/components/ArchitectureView.tsx`, `src/commands/architecture.rs`

**Target:** Basic architecture creation, storage, and visualization working

#### Phase 3.2: AI Generation (Weeks 4-6)

**Generation from User Intent (Week 4):**

- [ ] Create architecture generation prompts (templates/architecture_from_intent.txt)
- [ ] Implement LLM orchestration for architecture generation
- [ ] Add prompt engineering for consistent JSON output
- [ ] Parse LLM response into components/connections
- [ ] Validate generated architecture (no orphaned nodes, valid types)
- [ ] Add iterative refinement (user feedback loop)
- [ ] Create `src/architecture/generator.rs` (400+ lines)
- [ ] Write 15+ generation tests

- **Files:** `src/architecture/generator.rs`, `templates/architecture_from_intent.txt`

**Import from Existing Code (Week 5):**

- [ ] Leverage GNN graph for architecture extraction
- [ ] Implement heuristics for component grouping (by directory, by imports)
- [ ] Detect architectural layers (frontend, backend, database, API)
- [ ] Identify communication patterns (REST calls, message queues)
- [ ] Generate architecture from GNN analysis
- [ ] Add manual override and refinement UI
- [ ] Create `src/architecture/importer.rs` (450+ lines)
- [ ] Write 20+ import tests

- **Files:** `src/architecture/importer.rs`

**Generation from Specifications (Week 6):**

- [ ] Parse README.md, docs/*, *.spec files
- [ ] Extract architecture keywords (microservices, API Gateway, etc.)
- [ ] Combine spec parsing with LLM understanding
- [ ] Generate architecture from documentation
- [ ] Add confidence scoring for generated components
- [ ] Implement review and approval flow
- [ ] Create `src/architecture/spec_parser.rs` (300+ lines)
- [ ] Write 12+ spec parsing tests

- **Files:** `src/architecture/spec_parser.rs`

**Target:** All 4 generation methods working (intent, code, specs, manual)

#### Phase 3.3: Bidirectional Sync & Governance (Weeks 7-9)

**Architecture → Code Validation (Week 7):**

- [ ] Compare architecture components with actual code structure
- [ ] Detect missing implementations (components defined but not coded)
- [ ] Detect extra implementations (code exists but not in architecture)
- [ ] Validate connection accuracy (is API call actually made?)
- [ ] Generate misalignment reports with file/line references
- [ ] Add UI indicators for alignment status (green check, red X)
- [ ] Create `src/architecture/validator.rs` (400+ lines)
- [ ] Write 18+ validation tests

- **Files:** `src/architecture/validator.rs`

**Code → Architecture Sync (Week 8):**

- [ ] Watch for GNN update events (file changes)
- [ ] Detect architecture impact (new class → new component?)
- [ ] Generate sync proposals (add component, modify connection)
- [ ] Implement user approval flow (show diff, accept/reject)
- [ ] Auto-sync minor changes (method rename)
- [ ] Manual review for major changes (new service)
- [ ] Create `src/architecture/sync.rs` (450+ lines)
- [ ] Write 20+ sync tests

- **Files:** `src/architecture/sync.rs`

**Misalignment Resolution (Week 9):**

- [ ] Build decision UI (Code wins vs Architecture wins)
- [ ] Implement "Update Architecture" flow (modify components/connections)
- [ ] Implement "Revert Code" flow (ask LLM to fix implementation)
- [ ] Add conflict resolution for simultaneous changes
- [ ] Implement audit trail for all alignment decisions
- [ ] Add bulk resolution for multiple misalignments
- [ ] Create `src-ui/components/AlignmentPanel.tsx` (350+ lines)
- [ ] Write UI tests for resolution flows

- **Files:** `src-ui/components/AlignmentPanel.tsx`

**Target:** Fully bidirectional sync with governance and conflict resolution

#### Phase 3.4: Polish & Advanced Features (Weeks 10-11)

**Hierarchical Tabs & Navigation (Week 10):**

- [ ] Implement sliding tab system (breadcrumb trail)
- [ ] Create tab manager (stack-based navigation)
- [ ] Add smooth slide animations (CSS transitions)
- [ ] Implement drill-down on component double-click
- [ ] Add "Back" button and breadcrumb clicks
- [ ] Create component children detection (modules within services)
- [ ] Add keyboard shortcuts (Alt+Left/Right for back/forward)
- [ ] Create `src-ui/components/TabNavigation.tsx` (300+ lines)
- [ ] Write navigation tests

- **Files:** `src-ui/components/TabNavigation.tsx`

**Advanced Visualization (Week 11):**

- [ ] Add 4 layout modes (hierarchical, force-directed, layered, radial)
- [ ] Implement auto-layout algorithms (dagre for hierarchical)
- [ ] Add minimap for large architectures
- [ ] Implement search/filter (highlight matching nodes)
- [ ] Add connection highlighting (show all paths to/from node)
- [ ] Implement collapsible groups (collapse all database nodes)
- [ ] Add export (PNG, SVG, PDF, Markdown)
- [ ] Add zoom-to-fit and zoom-to-selection
- [ ] Enhance `src-ui/components/ArchitectureView.tsx` (200+ lines added)
- [ ] Write advanced feature tests

- **Files:** `src-ui/components/ArchitectureView.tsx` (updated)

**Target:** Production-ready architecture system with full feature set

### Testing Strategy

**Unit Tests (200+ total):**

- Storage layer: CRUD, versioning, recovery (20 tests)
- Models: Validation, relationships (25 tests)
- Generator: Intent, code, specs (47 tests)
- Validator: Alignment checking (18 tests)
- Sync: Bidirectional updates (20 tests)
- UI components: React Flow interactions (30+ tests)

**Integration Tests (30+ total):**

- End-to-end generation flows (intent → architecture → code)
- Bidirectional sync scenarios (code change → architecture update → validation)
- Misalignment resolution (detect → decide → apply)
- Multi-file architectures (large projects)

**Performance Tests:**

- Architecture rendering: <200ms for 100 components
- GNN-to-architecture import: <2s for 10K LOC project
- Alignment validation: <500ms for 50 components
- Sync detection: <100ms incremental check

### Success Metrics

- [ ] Generate architecture from user intent in <5 seconds
- [ ] Import architecture from existing code in <3 seconds (10K LOC)
- [ ] Detect code-architecture misalignment in <500ms
- [ ] Sync code changes to architecture with 95%+ accuracy
- [ ] User can navigate 3-level hierarchy smoothly (<50ms transitions)
- [ ] Export architecture to git-friendly formats (JSON, MD)
- [ ] 90%+ test coverage for architecture modules
- [ ] Zero data loss (3-layer fallback recovery)
- [ ] User NPS >50 for architecture features

### Dependencies

**Required:**

- Week 1 Foundation (GNN, file operations) ✅
- LLM Integration (for AI generation) ✅
- Browser UI (React Flow rendering) ✅

**Optional Enhancements:**

- Multi-language GNN (Python + JS/TS) - Already completed ✅
- Version control integration (git hooks for snapshots)
- Real-time collaboration (WebSocket sync for teams)

### Risk Mitigation

| Risk                                  | Mitigation                                            |
| ------------------------------------- | ----------------------------------------------------- |
| LLM generates invalid architectures   | Strict JSON schema validation, fallback templates     |
| Performance issues with large graphs  | Virtualized rendering (React Flow), lazy loading      |
| Sync conflicts (simultaneous changes) | CRDT-style conflict resolution, user approval flow    |
| Data corruption                       | 3-layer fallback (SQLite → JSON → GNN regeneration) |
| User adoption resistance              | Gradual rollout, optional feature, clear value demo   |

### Future Enhancements (Post-Phase 3)

- [ ] Team collaboration (real-time multiplayer editing)
- [ ] Architecture diff view (compare versions)
- [ ] AI-powered architecture reviews (best practices validation)
- [ ] Integration with external tools (Figma, Lucidchart import)
- [ ] Architecture templates library (microservices, monolith, serverless)
- [ ] Cost estimation (infer cloud costs from architecture)
- [ ] Security analysis (identify security boundaries, vulnerabilities)

---

### Success Criteria (Phase 2)

- [ ] Generate code → Run → Test → Package → Deploy → Monitor (end-to-end)
- [ ] Deploy to AWS/GCP/K8s with zero manual steps
- [ ] Self-healing: Auto-fix production errors within 5 minutes
- [ ] Docker images built and pushed automatically
- [ ] Health checks verify deployments (auto-rollback if fail)
- [ ] Production monitoring active with alerts
- [ ] CI/CD pipelines generated and working
- [ ] All tests passing (target: 150+ tests total)

**Note:** Detailed week-by-week plan will be created after Execution Layer completion.

---

## Risks & Mitigation

| Risk                               | Impact | Probability | Mitigation                                                                       |
| ---------------------------------- | ------ | ----------- | -------------------------------------------------------------------------------- |
| GNN accuracy <95%                  | High   | Low         | ✅ 100% accuracy achieved with cross-file resolution                             |
| LLM hallucination                  | High   | Medium      | Multi-LLM consensus, mandatory testing,**confidence scoring + auto-retry** |
| Performance issues at scale        | Medium | Medium      | Benchmarking, profiling,**token-aware context limits**                     |
| Low beta user adoption             | High   | Low         | Free access, developer marketing, focus on UX,**network effect value**     |
| LLM API costs too high             | Medium | Low         | Caching, smart routing,**Qwen Coder support (lower cost)**                 |
| Privacy concerns (pattern sharing) | Medium | Medium      | **Opt-in only, pattern anonymization, open-source extraction code**        |
| Token counting accuracy            | Low    | Low         | **Use tiktoken-rs (exact counts), not estimates**                          |

---

## Resource Requirements

### Team

- 1 Full-stack Developer (Rust + SolidJS)
- 1 ML/AI Engineer (LLM integration)
- 1 QA Engineer (testing)
- 1 UI/UX Designer (part-time)

### Infrastructure

- Development machines (macOS, Windows, Linux)
- LLM API access (Claude + GPT-4 + Qwen Coder)
- CI/CD pipeline
- Beta distribution platform
- **ChromaDB hosting (post-MVP for network effect)**

### Budget

- LLM API costs: ~$500-1000/month (development + testing)
- Infrastructure: ~$200/month
- **ChromaDB/Pattern hosting: ~$100/month (Phase 2)**
- Total MVP: ~$1200/month
- Total Phase 2: ~$1400/month

---

## Epic: Clean Code Mode (Automated Code Hygiene System)

**Status:** 📋 PLANNED (Post-MVP Feature)
**Priority:** High (Quality Enabler)
**Timeline:** 5 Weeks (Post Yantra Codex & Architecture View)
**Dependencies:** GNN Engine (✅), LLM Integration (✅), Testing Engine (✅)

### Overview

Clean Code Mode is an automated code maintenance system that continuously monitors, analyzes, and refactors codebases to maintain optimal code health. It leverages the existing GNN dependency tracking to detect dead code, perform safe refactorings, validate changes, and harden components.

**Business Value:**

- Reduces technical debt automatically
- Prevents code rot in long-running projects
- Saves developer time on manual code review
- Ensures consistent code quality
- Automates security hardening

**Specification:** `Specifications.md` lines 1309+ (Clean Code Mode section)

---

### Week 1: Dead Code Detection & Entry Point Analysis

**Status:** 🔴 NOT STARTED

**Goal:** Identify genuinely dead code with high confidence

#### Tasks

- [ ] **Create `src-tauri/src/clean-code/` module structure**

  - Set up module hierarchy
  - Define core types and traits
  - **Estimate:** 0.5 days
- [ ] **Implement Dead Code Analyzer**

  - File: `src-tauri/src/clean-code/dead-code/analyzer.rs`
  - Leverage `gnn.get_dependents()` to find unused code
  - Count incoming calls using `gnn.get_incoming_edges()`
  - Identify nodes with zero dependents
  - **Dependencies:** GNN Engine (READY)
  - **Estimate:** 2 days
- [ ] **Implement Entry Point Detector**

  - File: `src-tauri/src/clean-code/dead-code/entry_points.rs`
  - Detect `main()` functions
  - Detect API routes (`@app.route`, `@api.get`)
  - Detect CLI commands (`@click.command`, `argparse`)
  - Detect test functions (`test_*`, `@pytest`)
  - Detect exported APIs (`__all__`)
  - **Estimate:** 2 days
- [ ] **Implement Confidence Calculator**

  - File: `src-tauri/src/clean-code/dead-code/confidence.rs`
  - Factor: Zero incoming calls (1.0)
  - Factor: Recent code < 7 days (× 0.5)
  - Factor: Public API names (× 0.3)
  - Factor: Exported symbols (× 0.2)
  - Factor: TODO/FIXME comments (× 0.4)
  - Factor: Test-only usage (0.7)
  - **Target:** 80% threshold for auto-removal
  - **Estimate:** 1.5 days
- [ ] **Write comprehensive tests**

  - Test dead code detection accuracy
  - Test entry point detection (all patterns)
  - Test confidence calculation (edge cases)
  - **Target:** 90%+ test coverage
  - **Estimate:** 1 day

**Success Criteria:**

- ✅ Detect unused functions with <5% false positives
- ✅ Correctly identify all entry points
- ✅ Calculate confidence scores (0.0-1.0)
- ✅ 90%+ test coverage

---

### Week 2: Safe Dead Code Removal

**Status:** 🔴 NOT STARTED

**Goal:** Remove dead code safely with full validation

#### Tasks

- [ ] **Implement Safe Remover**

  - File: `src-tauri/src/clean-code/dead-code/remover.rs`
  - Verify code still dead (no race condition)
  - Create file backup before removal
  - Remove code lines
  - Update GNN graph
  - Run affected tests
  - Rollback on test failure
  - Git commit on success
  - **Estimate:** 2 days
- [ ] **Implement Test Validator**

  - File: `src-tauri/src/clean-code/validation/test_validator.rs`
  - Find affected tests using GNN
  - Run only affected tests (performance)
  - Validate 100% pass rate
  - Track coverage changes
  - **Estimate:** 1.5 days
- [ ] **Implement Rollback Mechanism**

  - Restore from backup on failure
  - Revert GNN graph changes
  - Log rollback reason
  - **Estimate:** 1 day
- [ ] **Create UI for Dead Code View**

  - File: `src-ui/components/CleanCode/DeadCodeView.tsx`
  - List unused functions with confidence
  - Show "Remove" / "Keep" buttons
  - Batch operations
  - Activity log
  - **Estimate:** 2 days
- [ ] **Add Tauri Commands**

  - `run_dead_code_analysis(project_path) -> Vec<DeadCodeReport>`
  - `remove_dead_code(report_id) -> Result<RemovalResult>`
  - `get_dead_code_status() -> CleanCodeStatus`
  - **Estimate:** 0.5 days
- [ ] **Write tests**

  - Test safe removal flow
  - Test rollback on test failure
  - Test GNN update after removal
  - **Target:** 90%+ coverage
  - **Estimate:** 1 day

**Success Criteria:**

- ✅ Remove dead code without breaking tests (100%)
- ✅ Automatic rollback on failure
- ✅ GNN stays consistent
- ✅ UI shows dead code clearly

---

### Week 3: Real-Time Refactoring Engine

**Status:** 🔴 NOT STARTED

**Goal:** Detect code smells and suggest refactorings

#### Tasks

- [ ] **Implement Duplicate Code Detector**

  - File: `src-tauri/src/clean-code/refactoring/duplicate_detector.rs`
  - Use GNN feature extraction (978-dim embeddings)
  - Calculate cosine similarity between code blocks
  - Threshold: 0.85 (85% similarity)
  - Suggest extraction to shared function
  - **Innovation:** Semantic similarity, not just syntactic!
  - **Estimate:** 2 days
- [ ] **Implement Complexity Analyzer**

  - File: `src-tauri/src/clean-code/refactoring/complexity_analyzer.rs`
  - Calculate cyclomatic complexity
  - Calculate cognitive complexity
  - Track nesting depth
  - Count parameters
  - Suggest splitting complex functions
  - **Thresholds:** Complexity > 10, Nesting > 4, Parameters > 5
  - **Estimate:** 1.5 days
- [ ] **Implement Refactoring Engine**

  - File: `src-tauri/src/clean-code/refactoring/engine.rs`
  - Operation: Remove unused imports
  - Operation: Extract duplicate code
  - Operation: Simplify complex functions
  - Operation: Rename for clarity
  - Generate refactored code via LLM
  - Validate with GNN
  - Run affected tests
  - **Estimate:** 2 days
- [ ] **Implement Refactoring Applicator**

  - File: `src-tauri/src/clean-code/refactoring/applicator.rs`
  - Apply code changes
  - Update GNN incrementally
  - Run validation suite
  - Commit on success
  - **Estimate:** 1.5 days
- [ ] **Create UI for Refactoring Suggestions**

  - File: `src-ui/components/CleanCode/RefactoringSuggestions.tsx`
  - Show duplicate code (side-by-side diff)
  - Show complexity metrics
  - "Apply" / "Dismiss" buttons
  - Preview changes before applying
  - **Estimate:** 2 days

**Success Criteria:**

- ✅ Detect duplicates with 85%+ similarity
- ✅ Calculate complexity accurately
- ✅ Apply refactorings without breaking tests
- ✅ UI shows clear before/after diffs

---

### Week 4: Component Hardening System

**Status:** 🔴 NOT STARTED

**Goal:** Automated security, performance, and quality hardening

#### Tasks

- [ ] **Integrate Semgrep for Security Scanning**

  - File: `src-tauri/src/clean-code/hardening/security.rs`
  - Run Semgrep with OWASP rules
  - Detect SQL injection, XSS, CSRF, etc.
  - Classify by severity (critical, high, medium, low)
  - **Estimate:** 1.5 days
- [ ] **Implement Security Auto-Fix**

  - File: `src-tauri/src/clean-code/hardening/auto_fix.rs`
  - Auto-fix: SQL injection → parameterized queries
  - Auto-fix: XSS → proper escaping
  - Auto-fix: Hardcoded secrets → env variables
  - Use LLM to generate fixes
  - Validate fixes with tests
  - **Target:** 70%+ auto-fix rate for critical issues
  - **Estimate:** 2 days
- [ ] **Implement Performance Profiler**

  - File: `src-tauri/src/clean-code/hardening/performance.rs`
  - Detect N+1 queries
  - Detect missing caching
  - Detect inefficient algorithms
  - Suggest optimizations via LLM
  - **Estimate:** 2 days
- [ ] **Implement Code Quality Analyzer**

  - File: `src-tauri/src/clean-code/hardening/quality.rs`
  - Calculate maintainability index
  - Detect code smells (long functions, magic numbers)
  - Check documentation coverage
  - Run linting (clippy, eslint)
  - **Estimate:** 1.5 days
- [ ] **Implement Dependency Auditor**

  - File: `src-tauri/src/clean-code/hardening/dependencies.rs`
  - Check for known vulnerabilities
  - Detect outdated dependencies
  - Calculate security score
  - **Estimate:** 1 day
- [ ] **Create Hardening Report UI**

  - File: `src-ui/components/CleanCode/HardeningReport.tsx`
  - Security issues (color-coded by severity)
  - Performance bottlenecks
  - Code quality metrics
  - Trend graphs over time
  - **Estimate:** 2 days

**Success Criteria:**

- ✅ Detect 100% of OWASP Top 10 vulnerabilities
- ✅ Auto-fix 70%+ of critical security issues
- ✅ Identify performance bottlenecks
- ✅ Generate actionable quality reports

---

### Week 5: Continuous Mode & Configuration

**Status:** 🔴 NOT STARTED

**Goal:** Background automation with configurable intervals

#### Tasks

- [ ] **Implement Configuration System**

  - File: `src-tauri/src/clean-code/config.rs`
  - Load from `.yantra/clean-code.toml`
  - Support modes: continuous, daily, pre-commit, manual
  - Configurable thresholds and intervals
  - **Estimate:** 1 day
- [ ] **Implement Continuous Mode Scheduler**

  - File: `src-tauri/src/clean-code/scheduler/continuous.rs`
  - Background thread checking every N minutes
  - Non-blocking, low CPU usage
  - Pause/resume capability
  - **Estimate:** 1.5 days
- [ ] **Implement Interval-Based Scheduler**

  - File: `src-tauri/src/clean-code/scheduler/interval.rs`
  - Daily cleanup at specified time
  - Weekly/monthly reports
  - **Estimate:** 1 day
- [ ] **Implement Event-Based Triggers**

  - File: `src-tauri/src/clean-code/scheduler/trigger.rs`
  - Trigger: Component implementation complete
  - Trigger: Pre-commit hook
  - Trigger: File save
  - **Estimate:** 1 day
- [ ] **Implement Notification System**

  - Show notifications for critical issues
  - Activity log
  - Weekly summary reports
  - **Estimate:** 1.5 days
- [ ] **Create Clean Code Dashboard**

  - File: `src-ui/components/CleanCode/Dashboard.tsx`
  - Status panel (active/paused, last scan)
  - Issues found by category
  - Auto-fixes applied
  - Activity log
  - Configuration panel
  - **Estimate:** 2 days
- [ ] **Add All Tauri Commands**

  - `enable_clean_code_mode(config)`
  - `disable_clean_code_mode()`
  - `get_clean_code_status()`
  - `apply_refactoring(operation)`
  - `harden_component(component)`
  - **Estimate:** 0.5 days
- [ ] **Integration Testing**

  - Test full workflow: detect → suggest → apply → validate
  - Test rollback scenarios
  - Test continuous mode performance
  - Test with real projects (10K+ LOC)
  - **Estimate:** 1.5 days

**Success Criteria:**

- ✅ Continuous mode runs without impacting performance
- ✅ Configuration system is flexible
- ✅ Notifications are helpful, not annoying
- ✅ Dashboard provides clear overview

---

### Performance Targets

| Operation                     | Target | Status          |
| ----------------------------- | ------ | --------------- |
| Dead code analysis (10K LOC)  | < 2s   | 🔴 Not measured |
| Duplicate detection (10K LOC) | < 5s   | 🔴 Not measured |
| Refactoring application       | < 3s   | 🔴 Not measured |
| Component hardening           | < 10s  | 🔴 Not measured |
| Security scan                 | < 5s   | 🔴 Not measured |
| Continuous mode check         | < 1s   | 🔴 Not measured |

---

### Success Metrics (KPIs)

1. **Dead Code Reduction**: < 2% dead code in healthy projects
2. **Refactoring Acceptance**: > 60% acceptance for high-confidence suggestions
3. **False Positive Rate**: < 5% for dead code detection
4. **Test Pass Rate**: 100% (won't apply if tests fail)
5. **Security Detection**: 100% of OWASP Top 10
6. **Auto-Fix Success**: > 70% for critical security issues
7. **Code Quality Improvement**: +10 maintainability index after 3 months
8. **Time Saved**: 20% reduction in code review time

---

### Dependencies & Integration

**Leverages Existing:**

- ✅ GNN Engine (get_dependents, get_incoming_edges, get_all_dependencies)
- ✅ LLM Integration (generate refactored code, suggest fixes)
- ✅ Testing Engine (run affected tests, coverage tracking)
- ✅ Git Integration (auto-commit, branch creation)
- ✅ Feature Extractor (978-dim embeddings for similarity)

**New Dependencies:**

- Semgrep (security scanning)
- Performance profiling tools
- Linters (clippy, eslint)
- Vulnerability database

---

### Rollout Strategy

**Phase 1 (Week 1-2):** Dead Code Detection & Removal

- Focus: High-confidence detection, safe removal
- Users: Beta testers

**Phase 2 (Week 3):** Refactoring Suggestions

- Focus: Duplicates, complexity
- Users: Beta testers + early adopters

**Phase 3 (Week 4):** Component Hardening

- Focus: Security, performance, quality
- Users: All users (opt-in)

**Phase 4 (Week 5):** Continuous Mode

- Focus: Automation, background processing
- Users: All users (configurable)

---

### Open Questions

1. **Auto-Apply Threshold**: Allow per-operation configuration?
2. **Language Support**: Python only initially, or multi-language from day 1?
3. **Cloud Integration**: Store Clean Code reports in Yantra Cloud?
4. **Team Collaboration**: Share refactoring suggestions across team?
5. **Learning from Feedback**: Improve confidence scoring based on user approvals?
6. **Performance Impact**: Max acceptable CPU/memory for continuous mode?

---

## Current Status Summary

**Overall Progress:** 0% (Just Started)
**Current Week:** Week 1-2 (Foundation)
**Next Milestone:** Foundation Complete (Dec 3, 2025)
**Blockers:** None
**Team Velocity:** TBD

---

## Change Log

| Date         | Changes                                        | Author       |
| ------------ | ---------------------------------------------- | ------------ |
| Nov 20, 2025 | Initial project plan created                   | AI Assistant |
| Nov 26, 2025 | Added Clean Code Mode epic (5 weeks, post-MVP) | AI Assistant |

---

**Last Updated:** November 26, 2025
