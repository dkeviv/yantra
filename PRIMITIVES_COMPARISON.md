# Agentic Primitives Comparison: Specifications_revised vs Specifications.md

**Date:** December 9, 2025  
**Purpose:** Comprehensive comparison to identify missing primitives in current Specifications.md

---

## 🔍 EXECUTIVE SUMMARY

**Current Specifications.md (Section 3.3):** ~50 primitives  
**Original Specifications_revised-for_ref.md (Section 3.3):** ~120+ primitives  
**Missing from current Specifications.md:** ~70 primitives

### Critical Missing Sections:

1. ✅ **File System Operations** - Partially in current spec (4 primitives) → Should have 13 primitives
2. ❌ **Code Intelligence (Tree-sitter)** - MISSING entirely → Should have 9 primitives
3. ❌ **Dependency Graph & Impact Analysis** - MISSING entirely → Should have 7 primitives (overlaps with existing GNN section)
4. ❌ **Database Operations** - MISSING entirely → Should have 7 primitives
5. ❌ **API Monitoring** - MISSING entirely → Should have 6 primitives
6. ❌ **Environment & System Resources** - MISSING entirely → Should have 5 primitives
7. ⚠️ **Git Operations** - Severely incomplete (4 primitives) → Should have 17 primitives
8. ❌ **Testing Execution** - MISSING entirely → Should have 7 primitives
9. ❌ **Build & Compilation** - MISSING entirely → Should have 7 primitives
10. ❌ **Package Management** - MISSING entirely → Should have 7 primitives
11. ❌ **Deployment** - Incomplete (5 primitives) → Should have 8 primitives
12. ❌ **Code Generation** - Incomplete (4 primitives) → Should have 7 primitives
13. ✅ **Browser Automation** - Complete (5 primitives) ✅
14. ✅ **YDoc Operations** - Complete (5 primitives) ✅ (NEW - not in original)
15. ✅ **Terminal & Shell** - Complete (5 capabilities) ✅

---

## 📊 DETAILED SECTION-BY-SECTION COMPARISON

### 1️⃣ PERCEIVE - Input & Sensing Layer

#### 1.1 File System Operations

**Specifications_revised-for_ref.md (Original):**

```
13 primitives total:
1. file_read [P0] - Read file contents with encoding detection [Builtin]
2. file_write [P0] - Create/overwrite files [Builtin]
3. file_edit [P0] - Surgical edits (line range, search-replace, AST-based) [Builtin]
4. file_delete [P0] - Remove files safely [Builtin]
5. file_move [P0] - Rename/move files with dependency updates [Builtin]
6. file_copy [P0] - Duplicate files [Builtin]
7. directory_create [P0] - Create directories recursively [Builtin]
8. directory_list [P0] - List contents with filters [Builtin]
9. directory_tree [P0] - Get full project structure [Builtin]
10. file_search [P0] - Find files by name/pattern/glob [Builtin]
11. file_watch [P2] - Monitor for changes (reactive agents) [Builtin] (use notify crate)
12. docx_read [P1] - Read Word documents [Builtin] (use docx-rs)
13. pdf_read [P1] - Extract text from PDFs [Builtin] (use pdf-extract/lopdf)
```

**Current Specifications.md (Section 3.3.1):**

```
4 primitives only:
1. read_file - Read file contents [Builtin]
2. list_files - List files in directory [Builtin]
3. search_files - Search for files by name/pattern [Builtin]
4. file_metadata - Get file size, modified time, permissions [Builtin]
```

**❌ MISSING (9 primitives):**

- file_write
- file_edit
- file_delete
- file_move
- file_copy
- directory_create
- directory_tree
- docx_read
- pdf_read

**✅ ACTION:** Add missing 9 file operations to Specifications.md Section 3.3.1

---

#### 1.2 Code Intelligence (Tree-sitter Powered)

**Specifications_revised-for_ref.md (Original):**

```
9 primitives total [Builtin + MCP fallback]:
1. parse_ast [P0] - Get AST for file/snippet [Builtin via tree-sitter]
2. get_symbols [P0] - Extract functions, classes, variables, imports [Builtin]
3. get_references [P2] - Find all usages of a symbol [MCP/Builtin fallback]
4. get_definition [P2] - Jump to definition [MCP/Builtin fallback]
5. get_scope [P2] - Get scope context for a position [Builtin]
6. get_diagnostics [P0] - Syntax errors, warnings [Builtin]
7. semantic_search [P1] - Search code by meaning [Builtin GNN]
8. get_call_hierarchy [P1] - Incoming/outgoing calls [Builtin GNN]
9. get_type_hierarchy [P2] - Class inheritance chains [MCP/Builtin fallback]
```

**Current Specifications.md (Section 3.3.1):**

```
5 primitives (but listed as LSP, NOT Builtin/tree-sitter):
1. parse_file - Parse file into AST [Builtin] ⚠️
2. find_symbols - Find all functions/classes in file [LSP] ❌
3. go_to_definition - Jump to symbol definition [LSP] ❌
4. find_references - Find all uses of symbol [LSP] ❌
5. hover_info - Get documentation for symbol [LSP] ❌
```

**⚠️ CRITICAL ISSUE:** Current spec labels these as LSP, but original spec says they're **Builtin via Tree-sitter** with optional MCP fallback. LSP is "Editor-only, not exposed to Agent" per original spec.

**✅ ACTION:**

1. Correct protocol to **[Builtin via tree-sitter]** (Primary) + **[MCP]** (Optional fallback)
2. Add missing primitives: get_scope, get_diagnostics, semantic_search, get_call_hierarchy, get_type_hierarchy
3. Clarify that LSP is for editor UI only, NOT for agent

---

#### 1.3 Dependency Analysis

**Specifications_revised-for_ref.md (Original):**

```
7 primitives under "Dependency Graph & Impact Analysis" [Builtin Exclusive]:
1. build_dependency_graph [P0] - Generate full project graph [Builtin]
2. get_dependents [P0] - What depends on X? [Builtin]
3. get_dependencies [P0] - What does X depend on? [Builtin]
4. impact_analysis [P0] - If I change X, what breaks? [Builtin]
5. find_cycles [P1] - Detect circular dependencies [Builtin]
6. get_module_boundaries [P2] - Identify architectural layers [NEW Builtin]
7. cross_repo_deps [P2] - External API/service dependencies [NEW Builtin Phase 2]
```

**Current Specifications.md (Section 3.3.1):**

```
7 primitives (matches original):
1. query_dependencies - Find all dependencies of file [Builtin]
2. query_dependents - Find all files depending on file [Builtin]
3. find_imports - Find all imports in file [Builtin]
4. find_callers - Find all callers of function [Builtin]
5. impact_analysis - Analyze impact of changing file [Builtin]
6. build_dependency_graph - Generate full project graph [Builtin]
7. get_module_boundaries - Identify architectural layers [Builtin]
```

**✅ STATUS:** Mostly complete! Current spec has equivalent coverage.

**⚠️ MINOR:** Original has `find_cycles`, current doesn't explicitly list it.

---

#### 1.4 Database Connections & Schema Intelligence

**Specifications_revised-for_ref.md (Original):**

```
7 primitives [MCP Primary, Builtin fallback for SQLite]:
1. db_connect [P0] - Establish connection with pooling [MCP]
2. db_query [P0] - Execute SELECT (read-only, validated) [MCP]
3. db_execute [P0] - Execute INSERT/UPDATE/DELETE (validated) [MCP]
4. db_schema [P0] - Get tables, columns, types, constraints [MCP]
5. db_explain [P0] - Query execution plan [MCP]
6. db_migrate [P0] - Run migrations with rollback [MCP]
7. db_seed [P0] - Insert test data [MCP]

MCP Servers: Postgres MCP, MySQL MCP, SQLite MCP, MongoDB MCP
```

**Current Specifications.md (Section 3.3.1):**

```
❌ COMPLETELY MISSING - No database primitives listed
```

**✅ ACTION:** Add entire Database Operations section (7 primitives) to Specifications.md under PERCEIVE

---

#### 1.5 API Monitoring & Contract Validation

**Specifications_revised-for_ref.md (Original):**

```
6 primitives [MCP Primary for external APIs, Builtin HTTP]:
1. api_import_spec [P0] - Import OpenAPI/Swagger specs [MCP]
2. api_validate_contract [P0] - Detect breaking API changes [MCP]
3. api_health_check [P1] - Test endpoint availability [Builtin HTTP]
4. api_rate_limit_check [P1] - Track and predict rate limits [Builtin]
5. api_mock [P2] - Create mock server from spec [MCP Phase 2]
6. api_test [P2] - Test endpoint with assertions [MCP Phase 2]
```

**Current Specifications.md (Section 3.3.1):**

```
❌ COMPLETELY MISSING - No API monitoring primitives listed
```

**✅ ACTION:** Add entire API Monitoring section (6 primitives) to Specifications.md under PERCEIVE

---

#### 1.6 Environment & System Resources

**Specifications_revised-for_ref.md (Original):**

```
5 primitives [Builtin Exclusive]:
1. env_get/env_set [P0] - Environment variables [Terminal/Builtin]
2. get_cpu_usage [P2] - CPU metrics for optimization [Builtin]
3. get_memory_usage [P2] - Memory stats [Builtin]
4. get_disk_usage [P2] - Disk space monitoring [Builtin]
5. should_throttle [P2] - Adaptive resource management [Builtin]
```

**Current Specifications.md (Section 3.3.1):**

```
5 primitives (matches!):
1. get_installed_packages - List installed dependencies [Builtin]
2. check_environment - Verify Python/Node/Rust version [Builtin]
3. get_git_status - Get Git repository status [MCP]
4. env_get - Get environment variable [Builtin]
5. env_set - Set environment variable [Builtin]
```

**⚠️ DISCREPANCY:** Current spec includes package/git operations mixed in. Original has pure system resources (CPU, memory, disk).

**✅ ACTION:** Clarify Environment Sensing vs System Resources as separate subcategories

---

#### 1.7 Test & Validation

**Specifications_revised-for_ref.md (Original):**

```
No explicit primitives in PERCEIVE layer - Testing is in ACT layer
```

**Current Specifications.md (Section 3.3.1):**

```
3 primitives:
1. get_test_results - Retrieve last test run results [Builtin]
2. get_coverage - Get code coverage metrics [Builtin]
3. get_security_scan - Get security scan results [Builtin]
```

**✅ STATUS:** Current spec adds test sensing primitives (good addition)

---

#### 1.8 Browser Sensing

**Current Specifications.md only:**

```
4 primitives [Builtin CDP]:
1. get_console_logs - Get browser console output [Builtin CDP]
2. get_network_logs - Get browser network requests [Builtin CDP]
3. capture_screenshot - Take browser screenshot [Builtin CDP]
4. get_dom_element - Query DOM for element [Builtin CDP]
```

**✅ STATUS:** Current spec adds browser sensing (not in original under PERCEIVE)

---

### 2️⃣ REASON - Decision-Making & Analysis Layer

**Both specs agree:** REASON layer is complete with 8 capabilities (confidence scoring, impact analysis, risk assessment, decision logging, multi-LLM orchestration, validation pipeline, error analysis, adaptive context assembly).

**✅ STATUS:** No changes needed

---

### 3️⃣ ACT - Execution & Action Layer

#### 3.1 Terminal & Shell Execution

**Specifications_revised-for_ref.md (Original):**

```
6 capabilities [Builtin Exclusive]:
1. shell_exec [P0] - Run command, get output [Builtin]
2. shell_exec_streaming [P0] - Long-running with real-time output [Builtin]
3. shell_background [P0] - Start background process [Builtin]
4. shell_kill [P0] - Terminate process [Builtin]
5. shell_interactive [P0] - Pseudo-TTY for interactive CLIs [Builtin]
6. Smart Terminal Reuse [P0] - Detect idle terminals, reuse before creating new [Builtin]
```

**Current Specifications.md (Section 3.3.3):**

```
5 capabilities (slightly different naming):
1. execute_command - Execute shell command with streaming [Builtin]
2. manage_environment_vars - Environment variable management [Builtin]
3. control_working_directory - Working directory control [Builtin]
4. manage_background_processes - Background process management [Builtin]
5. capture_exit_codes - Exit code capture and error handling [Builtin]
```

**✅ STATUS:** Equivalent coverage with different naming convention

---

#### 3.2 Git & Version Control

**Specifications_revised-for_ref.md (Original):**

```
17 primitives [MCP/Builtin]:
1. git_setup [P0 Tool] - Chat-based Git configuration & auth [Builtin] ⭐ NEW
2. git_authenticate [P0 Tool] - Store credentials securely [Builtin] ⭐ NEW
3. git_test_connection [P0 Tool] - Validate authentication works [Builtin] ⭐ NEW
4. git_status [P0 Terminal] - Current state [MCP/Builtin]
5. git_diff [P0 Terminal] - Changes (staged, unstaged, between refs) [MCP/Builtin]
6. git_log [P0 Terminal] - Commit history [MCP/Builtin]
7. git_blame [P0 Terminal] - Line-by-line attribution [MCP/Builtin]
8. git_commit [P0 Terminal] - Create commit with auto-messages [MCP/Builtin]
9. git_push [P0 Terminal] - Push commits to remote [MCP/Builtin]
10. git_pull [P0 Terminal] - Pull latest changes [MCP/Builtin]
11. git_branch [P0 Terminal] - Create/switch/list branches [MCP/Builtin]
12. git_checkout [P0 Terminal] - Checkout files/branches [MCP/Builtin]
13. git_merge [P0 Terminal] - Merge branches [MCP/Builtin]
14. git_stash [P0 Terminal] - Stash/pop changes [MCP/Builtin]
15. git_reset [P0 Terminal] - Undo changes [MCP/Builtin]
16. git_clone [P0 Terminal] - Clone repository [MCP/Builtin]
17. git_resolve_conflict [P2 Tool] - AI-powered conflict resolution [MCP/Builtin] (Post-MVP)

Protocol: MCP (Primary via @modelcontextprotocol/server-git) | Builtin (Fallback via git2-rs)
Note: Via terminal commands = [MCP/Builtin] (both available)
```

**Current Specifications.md (Section 3.3.3):**

```
4 primitives only [MCP]:
1. git_commit - Commit changes [MCP]
2. git_push - Push to remote [MCP]
3. git_branch - Create branch [MCP]
4. git_merge - Merge branches [MCP]
```

**❌ CRITICAL GAP:** Missing 13 Git primitives!

**✅ ACTION:** Add all 17 Git primitives with proper [MCP/Builtin] designation

---

#### 3.3 Testing Execution

**Specifications_revised-for_ref.md (Original):**

```
7 primitives [Builtin Exclusive]:
1. test_run [P0] - Execute tests (file, suite, single) [Builtin]
2. test_run_affected [P1] - Run tests for changed code only [NEW Builtin, use GNN]
3. test_coverage [P0] - Get coverage report [Builtin pytest-cov integration]
4. test_generate [P0] - Auto-generate test cases [Builtin]
5. test_debug [P2] - Run test in debug mode [NEW Builtin with DAP Phase 2]
6. test_watch [P1] - Continuous test runner [NEW Builtin Phase 2]
7. e2e_run [P0] - Browser/integration tests [NEW Builtin CDP + Playwright]
```

**Current Specifications.md (Section 3.3.3):**

```
4 primitives:
1. run_tests - Execute test suite [Builtin]
2. run_single_test - Execute specific test [Builtin]
3. run_coverage - Execute tests with coverage [Builtin]
4. run_stress_tests - Execute concurrency stress tests [Builtin]
```

**⚠️ PARTIAL:** Current has basics, missing test_run_affected, test_generate, test_debug, test_watch, e2e_run

**✅ ACTION:** Add 3 missing test primitives (test_run_affected, test_generate, e2e_run)

---

#### 3.4 Build & Compilation

**Specifications_revised-for_ref.md (Original):**

```
7 primitives [Builtin via Terminal]:
1. build_project [P0] - Full build [Terminal]
2. build_incremental [P0] - Changed files only [Terminal]
3. build_check [P0] - Type-check without emitting [Terminal]
4. build_clean [P0] - Remove artifacts [Terminal]
5. build_watch [P1] - Continuous build on file changes [Terminal]
6. build_optimize [P1] - Production optimized build [Terminal]
7. build_profile [P2] - Build with profiling enabled [Terminal]
```

**Current Specifications.md (Section 3.3.3):**

```
❌ COMPLETELY MISSING - No build primitives listed
```

**✅ ACTION:** Add entire Build & Compilation section (7 primitives)

---

#### 3.5 Package Management

**Specifications_revised-for_ref.md (Original):**

```
7 primitives [Builtin via Terminal]:
1. pkg_install [P0] - Install package(s) [Terminal]
2. pkg_uninstall [P0] - Remove package(s) [Terminal]
3. pkg_update [P0] - Update package(s) [Terminal]
4. pkg_list [P0] - List installed packages [Terminal]
5. pkg_search [P1] - Search for packages [Terminal]
6. pkg_outdated [P1] - Check for updates [Terminal]
7. pkg_audit [P1] - Security vulnerability check [Terminal]
```

**Current Specifications.md (Section 3.3.3):**

```
❌ COMPLETELY MISSING - No package management primitives listed
```

**✅ ACTION:** Add entire Package Management section (7 primitives)

---

#### 3.6 Deployment

**Specifications_revised-for_ref.md (Original):**

```
8 primitives [MCP + Builtin]:
1. deploy_local [P0] - Deploy to localhost [Builtin]
2. deploy_railway [P0] - Deploy to Railway [MCP]
3. deploy_vercel [P1] - Deploy to Vercel [MCP]
4. deploy_aws [P2] - Deploy to AWS [MCP]
5. deploy_gcp [P2] - Deploy to GCP [MCP]
6. deploy_docker [P1] - Build and push Docker image [Builtin]
7. health_check [P0] - Check deployment health [Builtin]
8. rollback_deployment [P1] - Rollback to previous version [Builtin]
```

**Current Specifications.md (Section 3.3.3):**

```
5 primitives:
1. deploy_local - Deploy to localhost [Builtin]
2. deploy_railway - Deploy to Railway [MCP]
3. deploy_aws - Deploy to AWS [MCP]
4. health_check - Check deployment health [Builtin]
5. rollback_deployment - Rollback to previous version [Builtin]
```

**⚠️ PARTIAL:** Missing deploy_vercel, deploy_gcp, deploy_docker

**✅ ACTION:** Add 3 missing deployment primitives

---

#### 3.7 Code Generation

**Specifications_revised-for_ref.md (Original):**

```
7 primitives [Builtin]:
1. generate_code [P0] - Generate new code from spec [Builtin]
2. generate_tests [P0] - Generate tests for code [Builtin]
3. generate_documentation [P0] - Generate API documentation [Builtin]
4. generate_boilerplate [P1] - Generate project scaffolding [Builtin]
5. refactor_code [P1] - Refactor existing code [Builtin]
6. refactor_extract [P1] - Extract function/class/module [Builtin]
7. refactor_inline [P1] - Inline function/variable [Builtin]
```

**Current Specifications.md (Section 3.3.3):**

```
4 primitives:
1. generate_code - Generate new code from spec [Builtin]
2. generate_tests - Generate tests for code [Builtin]
3. generate_documentation - Generate API documentation [Builtin]
4. refactor_code - Refactor existing code [Builtin]
```

**⚠️ PARTIAL:** Missing generate_boilerplate, refactor_extract, refactor_inline

**✅ ACTION:** Add 3 missing code generation primitives

---

#### 3.8 File Manipulation

**Both specs have 4 primitives - equivalent**

**✅ STATUS:** Complete

---

#### 3.9 Browser Automation

**Both specs have 5 primitives - equivalent**

**✅ STATUS:** Complete

---

#### 3.10 YDoc Operations

**Current Specifications.md only:**

```
5 primitives [Builtin]:
1. create_ydoc_document - Create new YDoc document [Builtin]
2. create_ydoc_block - Create new block in document [Builtin]
3. update_ydoc_block - Update existing block [Builtin]
4. link_ydoc_to_code - Create graph edge doc → code [Builtin]
5. search_ydoc_blocks - Search documentation blocks [Builtin]
```

**✅ STATUS:** New addition (not in original spec) - KEEP

---

### 4️⃣ LEARN - Feedback & Adaptation Layer

**Both specs agree:** LEARN layer primitives are all Yantra Codex-related (Phase 2/Stretch goal)

**✅ STATUS:** Complete (but not implemented - expected)

---

### 5️⃣ Cross-Cutting Primitives

**Current spec has comprehensive cross-cutting primitives for State Management, Context Management, Communication, Error Handling**

**✅ STATUS:** Complete

---

## 🎯 FINAL RECOMMENDATIONS

### Immediate Actions (Add to current Specifications.md):

1. **File System Operations** → Add 9 missing primitives (file_write, file_edit, file_delete, file_move, file_copy, directory_create, directory_tree, docx_read, pdf_read)

2. **Code Intelligence** → Fix protocol designation (change LSP to Builtin/tree-sitter) + Add 4 missing (get_scope, get_diagnostics, semantic_search, get_call_hierarchy, get_type_hierarchy)

3. **Database Operations** → Add entire section (7 primitives)

4. **API Monitoring** → Add entire section (6 primitives)

5. **Git Operations** → Expand from 4 to 17 primitives with [MCP/Builtin] designation

6. **Testing Execution** → Add 3 missing (test_run_affected, test_generate, e2e_run)

7. **Build & Compilation** → Add entire section (7 primitives)

8. **Package Management** → Add entire section (7 primitives)

9. **Deployment** → Add 3 missing (deploy_vercel, deploy_gcp, deploy_docker)

10. **Code Generation** → Add 3 missing (generate_boilerplate, refactor_extract, refactor_inline)

### Where to Add in Specifications.md:

**Section 3.3.1 PERCEIVE** should include:

- File System Operations (expand from 4 to 13)
- Code Intelligence (fix protocol + expand from 5 to 9)
- Dependency Analysis (keep as-is, already complete)
- **NEW:** Database Operations (7 primitives)
- **NEW:** API Monitoring (6 primitives)
- Environment & System Resources (clarify categorization)
- Test & Validation (keep as-is)
- Browser Sensing (keep as-is)

**Section 3.3.3 ACT** should include:

- Terminal & Shell (keep as-is)
- Git & Version Control (expand from 4 to 17)
- Testing Execution (expand from 4 to 7)
- **NEW:** Build & Compilation (7 primitives)
- **NEW:** Package Management (7 primitives)
- Deployment (expand from 5 to 8)
- Code Generation (expand from 4 to 7)
- File Manipulation (keep as-is)
- Browser Automation (keep as-is)
- YDoc Operations (keep as-is - NEW addition)

---

## 📈 STATISTICS

**Total Primitives:**

- Original Specifications_revised-for_ref.md: ~120 primitives
- Current Specifications.md: ~50 primitives
- **Gap: ~70 missing primitives**

**Completion Rate:** 42% (50/120)

**Priority Breakdown of Missing Primitives:**

- P0 (Critical): ~35 missing
- P1 (High): ~20 missing
- P2 (Medium): ~15 missing
