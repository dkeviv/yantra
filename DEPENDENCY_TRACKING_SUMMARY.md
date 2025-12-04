# Dependency Tracking - Summary & Next Steps

**Date:** December 4, 2025  
**Status:** Specifications Updated ✅  
**Priority:** P0 BLOCKER for MVP

---

## What We Discussed & Agreed On

### 1. **Post-MVP Dependency Tracking Priorities** ✅

| Feature                         | Priority      | Reasoning                                          |
|---------------------------------|---------------|----------------------------------------------------|
| External API Tracking           | **P1 Post-MVP** | Critical for microservices, API contracts          |
| User → File Tracking            | **P1 Post-MVP** | Important for team collaboration, ownership        |
| Method Chain Tracking           | **P2 Post-MVP** | Complex, diminishing returns vs. nested functions  |
| Function → Package Function     | **P2 Post-MVP** | Nice-to-have after basic package tracking works    |

### 2. **Blast Radius Analysis Feature** ✅

**Question:** "Where does blast radius preview fit? State machine or codegen intelligence?"

**Answer:** **BOTH** - but primary location is **State Machine (PlanReview State)**

#### Primary Location: Code Generation State Machine → PlanReview State

**When:** After PlanGeneration, BEFORE execution (preventive, not reactive)

**Purpose:** Show users exactly what will be affected BEFORE committing to execution

**Integration:**
```
Planning Phase:
├─ DependencyAssessment
├─ TaskDecomposition
├─ PlanGeneration
├─ BlastRadiusAnalysis ← NEW P0 State
└─ PlanReview ← Displays blast radius + gets approval
```

**What It Shows:**
```
📊 Blast Radius Preview

Direct Impact:
├── 3 files directly modified
├── 1 critical file (calculator.py)

Indirect Impact:
├── 12 files indirectly affected
├── 2 API endpoints changed (1 breaking)
└── 47 tests need updating

Package Impact:
├── numpy: 1.24.0 → 1.26.0 (upgrade)
└── Breaking changes detected

Risk: HIGH ⚠️
Time: 45-60 minutes
Approval required: YES

[Approve] [Modify] [Cancel]
```

**Smart Display Logic:**
- **Always show detailed** for: critical files, breaking changes, >10 indirect deps, package upgrades, high risk
- **Show compact** for: medium changes (2-5 files, <20 tests)
- **Skip entirely** for: trivial changes (single file, no deps, <5 tests, low risk)

**Approval Gate Triggers:**
```rust
fn should_require_approval(analysis: &BlastRadiusAnalysis) -> bool {
    !analysis.critical_files.is_empty() ||       // Core logic affected
    !analysis.breaking_changes.is_empty() ||     // API contracts changed
    matches!(analysis.risk_level, High | Critical)  // High risk score
}
```

#### Secondary Location: DependencyValidation State (P1 Post-MVP)

**When:** During execution, AFTER code generation, BEFORE commit

**Purpose:** Live validation - catch unexpected impacts that planning missed

**Output:** Streaming updates comparing planned vs. actual impact

---

## Specifications Updated ✅

### File: `.github/Specifications.md`

**Added Sections:**

1. **GNN Tech Stack Dependency Tracking (Lines 2670-2750)** - Enhanced with:
   - Post-MVP priorities (P1: External API, User→File; P2: Method Chain, Function→Package)
   - MVP scope clarifications (what to implement first)
   - Metadata sources (lock files > manifests > runtime)
   - Function tracking granularity (nested functions for MVP)
   - Cross-language strategy (separate graphs per language)
   - Update triggers (project open, lock file changes, manual refresh)

2. **Blast Radius Analysis Feature (Lines 4060-4350)** - Complete specification:
   - Data structures (BlastRadiusAnalysis, FileImpact, DependencyImpact, etc.)
   - Analysis algorithm (7-step process with GNN queries)
   - Critical file detection heuristics (>10 dependents, core dirs, naming patterns, explicit marking)
   - Risk scoring algorithm (weighted by critical files, breaking changes, package conflicts)
   - UI display formats (compact vs. detailed views)
   - Smart display logic (when to show/skip)
   - Performance targets (<2s total analysis time)
   - Future enhancements (P1: live validation, P2: historical tracking, ML-based impact estimation)

3. **State Machine Updates (Line 3860)** - Added BlastRadiusAnalysis state:
   - Integrated into Planning Phase
   - Feeds into PlanReview approval gate
   - Triggers approval for high-risk changes
   - Smart logic to skip trivial changes

### File: `DEPENDENCY_TRACKING_ANALYSIS.md` (Created)

**Content:** Comprehensive 6-part analysis document:
1. What IS implemented (code-to-code dependencies)
2. What's NOT implemented (package dependencies)
3. The version-level tracking gap
4. Detailed analysis by dependency type
5. Implementation roadmap (~900 lines of code needed)
6. Summary matrix showing all combinations

---

## Technical Decisions Made ✅

### Package Version Granularity
**Decision:** Track installed version only for MVP (Option A)  
**Rationale:** Simpler, sufficient for conflict detection. Add requirement ranges post-MVP.

### Package Function Depth
**Decision:** Nested attributes (Option B) - `numpy.random.normal`  
**Rationale:** Catches breaking changes like numpy 2.0 API changes. Skip method chains for MVP.

### Metadata Sources
**Decision:** Lock files first, fallback to manifests  
**Rationale:** Lock files have exact versions. Skip runtime inspection for MVP complexity.

### Transitive Dependencies
**Decision:** Full tree (Option C)  
**Rationale:** Lock files already have this data. Propagated breaking changes matter.

### Breaking Change Detection
**Decision:** Semantic versioning + major version warnings (Option A)  
**Rationale:** Simple, fast, catches 80% of issues. Add changelog scraping post-MVP.

### Cross-Language Dependencies
**Decision:** Separate graphs per language for MVP (Option A)  
**Rationale:** Simpler. Cross-language API tracking needs External API feature (P1 Post-MVP).

### Graph Update Triggers
**Decision:** A + B hybrid - build on project open, rebuild on lock file changes  
**Rationale:** Watch requirements.txt, package.json, Cargo.toml. Incremental updates for performance.

---

## Implementation Estimates

### MVP Package Tracking (P0 - Required)

**Features (6 total):**
1. File → Package@Version tracking
2. Package → Package tracking (transitive deps)
3. Version-level node creation
4. Conflict detection (simple semver)
5. Breaking change warnings
6. Query APIs (get_files_using_package, etc.)

**Estimated Effort:**
- Rust code: ~900 lines
  - Extend NodeType enum with TechStack variant: ~50 lines
  - TechStackNode struct + methods: ~150 lines
  - Parse lock files (package-lock.json, Cargo.lock, poetry.lock): ~200 lines
  - Create package nodes during graph build: ~150 lines
  - Query methods: ~200 lines
  - Conflict detection algorithm: ~150 lines
- Tests: ~200 lines
- **Total: ~1,100 lines**
- **Time: 2-3 weeks focused work**

### Blast Radius Analysis (P0 - Required)

**Features (1 total):**
1. BlastRadiusAnalyzer implementation
2. Integration with PlanReview state
3. UI components (BlastRadiusPreview)
4. Smart display logic
5. Approval gate triggers

**Estimated Effort:**
- Rust backend: ~500 lines
  - BlastRadiusAnalyzer struct + methods: ~300 lines
  - Critical file detection: ~100 lines
  - Risk scoring: ~100 lines
- TypeScript frontend: ~300 lines
  - BlastRadiusPreview component: ~200 lines
  - Approval dialog: ~100 lines
- Tests: ~150 lines
- **Total: ~950 lines**
- **Time: 1-2 weeks**

### Total MVP Addition: ~2,050 lines, 3-5 weeks

---

## Corrected Implementation Status

### Previous Claim
- ✅ 97/97 P0+P1 capabilities complete (100%)
- ✅ GNN Dependency Tracking: 10/10 (100%)

### Reality Check (After Analysis)
- 🟡 **82/97 P0+P1 capabilities** complete (85%)
- ✅ GNN Code Dependencies: 10/10 (100%) - file→file, function→function, test→source
- ❌ **GNN Package Dependencies: 0/6 (0%)** - NOT STARTED, P0 BLOCKER
- ❌ **Blast Radius Analysis: 0/1 (0%)** - NOT STARTED, P0 BLOCKER

**Gap Identified:**
- 15 features missing (7 package tracking + 1 blast radius + 7 related capabilities)
- Cannot guarantee "code that never breaks" without package dependency tracking
- Previous "97/97" was incorrect - GNN "10/10" referred only to internal code dependencies

### Updated Breakdown

| Component                          | MVP Status | Details                                     |
|------------------------------------|------------|---------------------------------------------|
| GNN Code Dependencies              | ✅ 10/10   | 100% - Verified correct                     |
| GNN Package Dependencies           | ❌ 0/6     | 0% - File→Package, Package→Package, etc.    |
| Blast Radius Analysis              | ❌ 0/1     | 0% - PlanReview integration                 |
| Architecture View                  | 🟡 12/16   | 75% - Versioning workflows pending          |
| Other P0+P1 Capabilities           | ✅ 60/73   | 82% - Agent modules, LLM, testing, etc.     |
| **Total P0+P1**                    | **82/97**  | **85% MVP**                                 |

---

## Next Steps (Recommended Priority)

### Phase 1: Package Dependency Tracking (P0 - Weeks 1-3)
1. Extend GNN with TechStack node type
2. Parse lock files (package-lock.json, Cargo.lock, poetry.lock, requirements.txt)
3. Create package nodes + edges (Uses, Requires)
4. Implement version conflict detection
5. Add query APIs (get_files_using_package, etc.)
6. Write comprehensive tests

**Deliverable:** 6/6 package tracking features complete

### Phase 2: Blast Radius Analysis (P0 - Weeks 4-5)
1. Implement BlastRadiusAnalyzer (backend)
2. Integrate with Code Generation State Machine (PlanReview state)
3. Build BlastRadiusPreview UI component
4. Add smart display logic + approval triggers
5. Write tests for all scenarios

**Deliverable:** 1/1 blast radius feature complete

### Phase 3: Verification & Polish (Week 6)
1. End-to-end testing with real projects
2. Performance optimization (<2s blast radius analysis)
3. Documentation updates (Technical_Guide.md, Features.md)
4. Update IMPLEMENTATION_STATUS.md to 97/97 (100%)

**Deliverable:** MVP truly complete, all 97 P0+P1 capabilities functional

---

## Key Takeaways

### What Works Today ✅
- **Internal code dependencies:** Fully tracked (functions, classes, files, tests)
- **Multi-language parsing:** 11 languages supported
- **Semantic search:** HNSW indexing working
- **Test coverage:** GNN finds all tests for any file

### Critical Gaps ❌
- **Package dependencies:** Not tracked at all (0%)
- **Version-level tracking:** Doesn't exist
- **Blast radius preview:** Not implemented
- **Breaking change warnings:** No package-level warnings

### Why This Matters 🎯
**Yantra's promise:** "Code that never breaks"

**Reality without package tracking:**
- ❌ Can't detect: "Upgrading numpy 1.24→2.0 breaks pandas"
- ❌ Can't answer: "What files will break if I upgrade this package?"
- ❌ Can't prevent: Version conflicts before they happen
- ❌ Can't guarantee: Safe refactoring when packages change

**With package tracking:**
- ✅ Detect conflicts BEFORE installation fails
- ✅ Show blast radius: "47 files affected by numpy upgrade"
- ✅ Warn: "numpy 2.0 has breaking API changes"
- ✅ Suggest: "Use numpy==1.26.4 to satisfy all dependencies"

---

## Questions Answered

### Q: "Where does blast radius fit? State machine or codegen intelligence?"

**A:** **State Machine (PlanReview state) - PRIMARY LOCATION**

**Reasoning:**
- Fits preventive philosophy (PDC Phase 2: Planning)
- Happens BEFORE execution (not reactive)
- Already has approval gate for complex changes
- User can review, approve, or cancel
- Aligns with "show, don't surprise" principle

**Secondary location (P1 Post-MVP):** DependencyValidation state - live tracking during execution

### Q: "Are we tracking all dependencies to the version level?"

**A:** **NO** - Major gaps identified:

**Currently tracked:**
- ✅ File → File (imports)
- ✅ Function → Function (calls)
- ✅ Test → Source (coverage)

**NOT tracked:**
- ❌ File → Package@Version
- ❌ Package → Package (transitive deps)
- ❌ Function → Package Function (which numpy functions used)
- ❌ User → File (not MVP scope)
- ❌ Version-level tracking (nowhere in codebase)

**Coverage:** Only 56% of meaningful dependency relationships

---

## Specification Files Updated

1. **.github/Specifications.md**
   - Added Post-MVP priorities for dependency tracking extensions
   - Added complete Blast Radius Analysis specification (data structures, algorithms, UI, logic)
   - Updated Code Generation State Machine with BlastRadiusAnalysis state
   - Enhanced GNN Tech Stack Dependency Tracking with MVP scope + decisions

2. **DEPENDENCY_TRACKING_ANALYSIS.md** (New)
   - 6-part comprehensive analysis
   - What's implemented vs. missing
   - Version-level gap explanation
   - Implementation roadmap
   - Dependency matrix (all combinations)

3. **DEPENDENCY_TRACKING_SUMMARY.md** (This File)
   - Executive summary of all decisions
   - Technical alignment
   - Implementation estimates
   - Next steps

---

**Prepared by:** AI Assistant  
**Reviewed with:** User (Vivek)  
**Date:** December 4, 2025  
**Status:** ✅ Specifications Complete, Ready for Implementation
