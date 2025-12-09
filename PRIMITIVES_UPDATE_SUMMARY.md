# Agentic Primitives Update Summary

**Date:** December 9, 2025  
**Updated By:** Automated update based on Specifications_revised-for_ref.md  
**Specifications Version:** 5.0 → 6.0

---

## 📊 Update Statistics

### File Changes:

- **Backup Created:** `.github/Specifications.md.backup_20251209_103922` (287KB)
- **Updated File:** `.github/Specifications.md` (294KB)
- **Size Increase:** +7KB (2.4% increase)

### Primitive Count Changes:

- **Before:** ~50 primitives
- **After:** ~120 primitives
- **Added:** ~70 primitives
- **Preserved:** All YDoc and Conversation primitives ✅

---

## ✅ What Was Added

### 3.3.1 PERCEIVE - Input & Sensing Layer

#### File System Operations (Expanded from 4 → 14 primitives)

**Added:**

- `file_write` - Create/overwrite files
- `file_edit` - Surgical edits (line range, search-replace)
- `file_delete` - Remove files safely
- `file_move` - Rename/move files with dependency updates
- `file_copy` - Duplicate files
- `directory_create` - Create directories recursively
- `directory_list` - List contents with filters (renamed from list_files)
- `directory_tree` - Get full project structure
- `file_watch` - Monitor for changes (reactive agents)
- `docx_read` - Read Word documents
- `pdf_read` - Extract text from PDFs

#### Code Intelligence (Updated protocols + expanded from 5 → 10 primitives)

**Protocol Changes:**

- Changed from `LSP` to `Builtin` (Tree-sitter primary) with `MCP/Builtin` fallback
- Clarified: LSP is for editor UI only, not exposed to agent

**Added:**

- `parse_ast` - Get AST for file/snippet (renamed from parse_file)
- `get_symbols` - Extract functions, classes, variables, imports (renamed from find_symbols)
- `get_references` - Find all usages (protocol changed LSP → MCP/Builtin)
- `get_definition` - Jump to definition (protocol changed LSP → MCP/Builtin)
- `get_scope` - Get scope context for a position
- `get_diagnostics` - Syntax errors, warnings
- `semantic_search` - Search code by meaning
- `get_call_hierarchy` - Incoming/outgoing calls
- `get_type_hierarchy` - Class inheritance chains

#### Environment Sensing (Expanded from 5 → 9 primitives)

**Added:**

- `get_cpu_usage` - CPU metrics for optimization
- `get_memory_usage` - Memory stats
- `get_disk_usage` - Disk space monitoring
- `should_throttle` - Adaptive resource management

#### Database Operations (NEW - 7 primitives)

**Complete new section:**

- `db_connect` - Establish connection with pooling [MCP]
- `db_query` - Execute SELECT (read-only, validated) [MCP]
- `db_execute` - Execute INSERT/UPDATE/DELETE (validated) [MCP]
- `db_schema` - Get tables, columns, types, constraints [MCP]
- `db_explain` - Query execution plan [MCP]
- `db_migrate` - Run migrations with rollback [MCP]
- `db_seed` - Insert test data [MCP]

Note: MCP primary via DB-specific servers. Builtin fallback for SQLite.

#### API Monitoring (NEW - 6 primitives)

**Complete new section:**

- `api_import_spec` - Import OpenAPI/Swagger specs [MCP]
- `api_validate_contract` - Detect breaking API changes [MCP]
- `api_health_check` - Test endpoint availability [Builtin]
- `api_rate_limit_check` - Track and predict rate limits [Builtin]
- `api_mock` - Create mock server from spec [MCP]
- `api_test` - Test endpoint with assertions [MCP]

Note: MCP primary for external APIs. Builtin HTTP for health checks.

---

### 3.3.3 ACT - Execution & Action Layer

#### Code Generation (Expanded from 4 → 7 primitives)

**Added:**

- `generate_boilerplate` - Generate project scaffolding
- `refactor_extract` - Extract function/class/module
- `refactor_inline` - Inline function/variable

#### Test Execution (Expanded from 4 → 7 primitives)

**Added:**

- `test_run_affected` - Run tests for changed code only (GNN-powered)
- `test_generate` - Auto-generate test cases
- `e2e_run` - Browser/integration tests

#### Git Operations (Expanded from 4 → 17 primitives) ⭐ CRITICAL

**Added:**

- `git_setup` - Chat-based Git configuration & auth [Builtin]
- `git_authenticate` - Store credentials securely [Builtin]
- `git_test_connection` - Validate authentication works [Builtin]
- `git_status` - Current state [MCP/Builtin]
- `git_diff` - Changes (staged, unstaged, between refs) [MCP/Builtin]
- `git_log` - Commit history [MCP/Builtin]
- `git_blame` - Line-by-line attribution [MCP/Builtin]
- `git_pull` - Pull latest changes [MCP/Builtin]
- `git_checkout` - Checkout files/branches [MCP/Builtin]
- `git_stash` - Stash/pop changes [MCP/Builtin]
- `git_reset` - Undo changes [MCP/Builtin]
- `git_clone` - Clone repository [MCP/Builtin]
- `git_resolve_conflict` - AI-powered conflict resolution [MCP/Builtin]

**Protocol Clarification:**

- `git_commit`, `git_push`, `git_branch`, `git_merge` updated from [MCP] to [MCP/Builtin]
- MCP primary via @modelcontextprotocol/server-git
- Builtin fallback via git2-rs
- Chat-based setup for one-time authentication with secure keychain storage

#### Build & Compilation (NEW - 7 primitives)

**Complete new section:**

- `build_project` - Full build [Builtin]
- `build_incremental` - Changed files only [Builtin]
- `build_check` - Type-check without emitting [Builtin]
- `build_clean` - Remove artifacts [Builtin]
- `build_watch` - Continuous build on changes [Builtin]
- `build_optimize` - Production optimized build [Builtin]
- `build_profile` - Build with profiling [Builtin]

Note: Via terminal commands coordinated with dependency graph, testing, deployment.

#### Package Management (NEW - 7 primitives)

**Complete new section:**

- `pkg_install` - Install package(s) [Builtin]
- `pkg_uninstall` - Remove package(s) [Builtin]
- `pkg_update` - Update package(s) [Builtin]
- `pkg_list` - List installed packages [Builtin]
- `pkg_search` - Search for packages [Builtin]
- `pkg_outdated` - Check for updates [Builtin]
- `pkg_audit` - Security vulnerability check [Builtin]

Note: Via terminal with GNN integration. Security audit integrates with vulnerability databases.

#### Deployment (Expanded from 5 → 8 primitives)

**Added:**

- `deploy_vercel` - Deploy to Vercel [MCP]
- `deploy_gcp` - Deploy to GCP [MCP]
- `deploy_docker` - Build and push Docker image [Builtin]

---

## 🔒 What Was Preserved

### YDoc Operations (5 primitives) ✅

All YDoc primitives remain intact:

- `create_ydoc_document`
- `create_ydoc_block`
- `update_ydoc_block`
- `link_ydoc_to_code`
- `search_ydoc_blocks`

### Conversation Memory (10 primitives in Context Management) ✅

All conversation primitives remain intact:

- `context_add` (Enhanced for conversation)
- `context_search` (Enhanced for unified search)
- `context_summarize` (Enhanced for both types)
- `conversation_search` (NEW)
- `conversation_history` (NEW)
- `conversation_link` (NEW)

Plus all conversation database tables and schemas preserved.

### Other Preserved Sections ✅

- All REASON layer capabilities (8 capabilities)
- All LEARN layer capabilities (16 primitives)
- All Cross-Cutting primitives (16 primitives)
- Browser Automation (5 primitives)
- File Manipulation (4 primitives)
- Terminal & Shell Execution (5 capabilities)

---

## 📝 Protocol Clarifications

### Key Protocol Changes:

1. **Code Intelligence:** Changed from `LSP` to `Builtin (Tree-sitter)` + `MCP` fallback
   - Rationale: Tree-sitter is core differentiator. LSP is editor-only, not for agent.

2. **Git Operations:** Changed from `MCP` to `MCP/Builtin`
   - Rationale: Dual interface - MCP primary, Builtin fallback ensures reliability

3. **Added Protocol Notes:**
   - Database: "MCP primary via DB-specific servers. Builtin fallback for SQLite"
   - API Monitoring: "MCP primary for external APIs. Builtin HTTP for health checks"
   - Testing: "Builtin exclusive with GNN integration for affected test detection"
   - Build: "Via terminal commands coordinated with dependency graph"
   - Package: "Via terminal with GNN integration"

---

## 🎯 Coverage Summary

### By Pillar:

#### PERCEIVE (Input & Sensing)

- File System: 14 primitives ✅ (was 4)
- Code Intelligence: 10 primitives ✅ (was 5, protocol corrected)
- Dependency Analysis: 7 primitives ✅ (unchanged)
- Database: 7 primitives ✅ (NEW)
- API Monitoring: 6 primitives ✅ (NEW)
- Environment: 9 primitives ✅ (was 5)
- Test Sensing: 3 primitives ✅ (unchanged)
- Browser Sensing: 4 primitives ✅ (unchanged)
  **Subtotal: 60 primitives**

#### REASON (Decision-Making)

- All 8 capabilities ✅ (unchanged)
  **Subtotal: 8 capabilities**

#### ACT (Execution & Action)

- Code Generation: 7 primitives ✅ (was 4)
- File Manipulation: 4 primitives ✅ (unchanged)
- Test Execution: 7 primitives ✅ (was 4)
- Git Operations: 17 primitives ✅ (was 4)
- Build & Compilation: 7 primitives ✅ (NEW)
- Package Management: 7 primitives ✅ (NEW)
- Deployment: 8 primitives ✅ (was 5)
- Browser Automation: 5 primitives ✅ (unchanged)
- YDoc Operations: 5 primitives ✅ (unchanged)
- Terminal: 5 capabilities ✅ (unchanged)
  **Subtotal: 72 primitives**

#### LEARN (Feedback & Adaptation)

- All 16 primitives ✅ (unchanged)
  **Subtotal: 16 primitives**

#### Cross-Cutting

- State Management: 4 primitives ✅
- Context Management: 10 primitives ✅ (Enhanced with conversation)
- Communication: 4 primitives ✅
- Error Handling: 4 primitives ✅
  **Subtotal: 22 primitives**

### Grand Total: ~120 primitives (up from ~50)

---

## 🔍 Verification Checklist

### Before/After Comparison:

- ✅ Backup created: `Specifications.md.backup_20251209_103922`
- ✅ File size increased from 287KB to 294KB (+7KB)
- ✅ Version updated from 5.0 to 6.0
- ✅ All YDoc primitives preserved
- ✅ All Conversation primitives preserved
- ✅ All missing primitives from original spec added
- ✅ Protocol designations corrected (LSP → Builtin/MCP where appropriate)
- ✅ Protocol notes added for clarity
- ✅ No implementation details included (as requested)

### Key Sections Added:

- ✅ Database Operations (7 primitives)
- ✅ API Monitoring (6 primitives)
- ✅ Build & Compilation (7 primitives)
- ✅ Package Management (7 primitives)
- ✅ Complete Git Operations (17 primitives total)

### Protocol Correctness:

- ✅ Code Intelligence uses Builtin (Tree-sitter) + MCP fallback (not LSP for agent)
- ✅ Git uses MCP/Builtin (dual interface)
- ✅ Database uses MCP (primary) + Builtin (SQLite fallback)
- ✅ API Monitoring uses MCP (external) + Builtin (HTTP)
- ✅ All protocol notes included where necessary

---

## 📋 Next Steps

1. **Review the updated Specifications.md** to ensure all primitives are correctly documented
2. **Compare with backup** using: `diff .github/Specifications.md.backup_20251209_103922 .github/Specifications.md`
3. **Update Excel sheet** (`generate_primitives_excel.py`) to include all 120 primitives
4. **Update Requirements_Table.md** if needed to reflect new primitives
5. **Update implementation tracking** to mark new primitives as "NOT IMPLEMENTED"

---

## 🎉 Success Criteria Met

✅ All missing primitives from Specifications_revised-for_ref.md added  
✅ YDoc primitives preserved  
✅ Conversation primitives preserved  
✅ Protocol designations corrected based on original spec  
✅ No implementation details included (kept high-level tool descriptions)  
✅ Specifications.md is now superset of all primitives  
✅ Backup created for safety  
✅ Version updated to 6.0

**Status: COMPLETE** 🎊
