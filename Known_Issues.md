# Yantra - Known Issues

**Purpose:** Track all bugs, issues, and their fixes  
**Last Updated:** November 23, 2025

---

## Active Issues

### Issue #1: Integration Tests Require API Keys for Execution

**Status:** Open (By Design)  
**Severity:** Low  
**Reported:** November 23, 2025  
**Component:** Testing  
**Assigned:** N/A (Manual testing required)

#### Description
The integration tests for automatic test generation (`tests/integration_orchestrator_test_gen.rs`) require an `ANTHROPIC_API_KEY` environment variable to run the full E2E flow with real LLM calls.

**Impact:**
- Tests skip in CI environment when API key not present
- Cannot verify test generation quality without manual testing
- MVP blocker fix validated structurally but not end-to-end

#### Steps to Reproduce
1. Run `cargo test integration_orchestrator_test_gen`
2. Without `ANTHROPIC_API_KEY` set, tests print "Skipping test: ANTHROPIC_API_KEY not set"
3. Tests pass (via skip) but don't validate actual behavior

#### Root Cause
- Integration tests need real LLM API to generate code and tests
- Cannot mock LLM responses realistically for this test
- API keys should not be committed to repository

#### Solution
**Current Approach (Acceptable for MVP):**
- Tests skip gracefully when API key unavailable
- Manual testing with real API key required before releases
- Documentation updated to note manual testing requirement

**Future Enhancement (Post-MVP):**
- Add mock LLM responses for integration tests
- Or: Use recorded LLM responses (VCR pattern)
- Or: Set up secure CI environment with encrypted API keys

#### Workaround
**For Manual Testing:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
cargo test integration_orchestrator_test_gen --test integration_orchestrator_test_gen
```

**Expected Output:**
- test_orchestrator_generates_tests_for_code: PASS (~15-20s)
- test_orchestrator_runs_generated_tests: PASS (~15-20s)

#### Fixed In
N/A - By design, will remain as manual testing requirement for MVP

---

## Issue Format

```
## Issue #[Number]: [Short Title]

**Status:** [Open | In Progress | Fixed | Won't Fix]
**Severity:** [Critical | High | Medium | Low]
**Reported:** [Date]
**Component:** [GNN | LLM | UI | Testing | Security | Browser | Git]
**Assigned:** [Person]

### Description
Clear description of the issue

### Steps to Reproduce
1. Step 1
2. Step 2
3. Expected vs Actual

### Root Cause
What's causing the issue

### Fix
How it was fixed (or planned fix)

### Fixed In
Version/commit where fixed
```

---

## Resolved Issues

*No resolved issues yet.*

---

## Won't Fix

*No "won't fix" issues yet.*

---

## Common Patterns

*As issues are discovered and fixed, common patterns will be documented here to prevent recurrence.*

### Pattern Categories

#### GNN Issues
*To be populated as issues are discovered*

#### LLM Issues
*To be populated as issues are discovered*

#### UI Issues
*To be populated as issues are discovered*

#### Testing Issues
*To be populated as issues are discovered*

#### Security Issues
*To be populated as issues are discovered*

#### Browser Issues
*To be populated as issues are discovered*

#### Git Issues
*To be populated as issues are discovered*

---

## Issue Statistics

| Category | Open | In Progress | Fixed | Total |
|----------|------|-------------|-------|-------|
| Critical | 0 | 0 | 0 | 0 |
| High | 0 | 0 | 0 | 0 |
| Medium | 0 | 0 | 0 | 0 |
| Low | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **0** |

---

**Last Updated:** November 20, 2025  
**Next Update:** When issues are discovered
