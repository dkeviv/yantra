Here's the comprehensive tool architecture for a full agentic IDE:

---

## 1. File System Operations

| Tool               | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| `file_read`        | Read file contents (with encoding detection)           |
| `file_write`       | Create/overwrite files                                 |
| `file_edit`        | Surgical edits (line range, search-replace, AST-based) |
| `file_delete`      | Remove files                                           |
| `file_move`        | Rename/move files                                      |
| `file_copy`        | Duplicate files                                        |
| `directory_create` | Create directories                                     |
| `directory_list`   | List contents (with filters, recursion)                |
| `directory_tree`   | Get project structure                                  |
| `file_search`      | Find files by name/pattern/glob                        |
| `file_watch`       | Monitor for changes (for reactive agents)              |

---

## 2. Code Intelligence (Tree-sitter Powered)

| Tool                 | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| `parse_ast`          | Get AST for file/snippet                       |
| `get_symbols`        | Extract functions, classes, variables, imports |
| `get_references`     | Find all usages of a symbol                    |
| `get_definition`     | Jump to definition                             |
| `get_scope`          | Get scope context for a position               |
| `get_diagnostics`    | Syntax errors, warnings                        |
| `semantic_search`    | Search code by meaning (embeddings)            |
| `get_call_hierarchy` | Incoming/outgoing calls                        |
| `get_type_hierarchy` | Class inheritance chains                       |

---

## 3. Dependency Graph (Your GNN Layer)

| Tool                     | Purpose                           |
| ------------------------ | --------------------------------- |
| `build_dependency_graph` | Generate full project graph       |
| `get_dependents`         | What depends on X?                |
| `get_dependencies`       | What does X depend on?            |
| `impact_analysis`        | If I change X, what breaks?       |
| `find_cycles`            | Detect circular dependencies      |
| `get_module_boundaries`  | Identify architectural layers     |
| `cross_repo_deps`        | External API/service dependencies |

---

## 4. Terminal / Shell

| Tool                   | Purpose                            |
| ---------------------- | ---------------------------------- |
| `shell_exec`           | Run command, get output            |
| `shell_exec_streaming` | Long-running with real-time output |
| `shell_background`     | Start background process           |
| `shell_kill`           | Terminate process                  |
| `shell_interactive`    | Pseudo-TTY for interactive CLIs    |
| `env_get`/`env_set`    | Environment variables              |

---

## 5. Git / Version Control

| Tool                   | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `git_status`           | Current state                            |
| `git_diff`             | Changes (staged, unstaged, between refs) |
| `git_log`              | Commit history                           |
| `git_blame`            | Line-by-line attribution                 |
| `git_commit`           | Create commit                            |
| `git_branch`           | Create/switch/list branches              |
| `git_checkout`         | Checkout files/branches                  |
| `git_merge`            | Merge branches                           |
| `git_stash`            | Stash/pop changes                        |
| `git_reset`            | Undo changes                             |
| `git_resolve_conflict` | Conflict resolution helper               |

---

## 6. Database

| Tool         | Purpose                                  |
| ------------ | ---------------------------------------- |
| `db_connect` | Establish connection (connection string) |
| `db_query`   | Execute SELECT (read-only)               |
| `db_execute` | Execute INSERT/UPDATE/DELETE             |
| `db_schema`  | Get tables, columns, types, constraints  |
| `db_explain` | Query execution plan                     |
| `db_migrate` | Run migrations                           |
| `db_seed`    | Insert test data                         |

---

## 7. Testing

| Tool                | Purpose                             |
| ------------------- | ----------------------------------- |
| `test_run`          | Execute tests (file, suite, single) |
| `test_run_affected` | Run tests for changed code only     |
| `test_coverage`     | Get coverage report                 |
| `test_generate`     | Auto-generate test cases            |
| `test_debug`        | Run test in debug mode              |
| `test_watch`        | Continuous test runner              |
| `e2e_run`           | Browser/integration tests           |

---

## 8. Build / Compilation

| Tool                | Purpose                     |
| ------------------- | --------------------------- |
| `build_project`     | Full build                  |
| `build_incremental` | Changed files only          |
| `build_check`       | Type-check without emitting |
| `build_clean`       | Clear artifacts             |
| `lint_run`          | Run linters                 |
| `lint_fix`          | Auto-fix lint issues        |
| `format_code`       | Apply formatters            |

---

## 9. Package Management

| Tool            | Purpose                      |
| --------------- | ---------------------------- |
| `pkg_install`   | Add dependency               |
| `pkg_remove`    | Remove dependency            |
| `pkg_update`    | Update dependencies          |
| `pkg_list`      | List installed packages      |
| `pkg_audit`     | Security vulnerability check |
| `pkg_search`    | Find packages in registry    |
| `pkg_lock_sync` | Sync lockfile                |

---

## 10. Debugging

| Tool               | Purpose                    |
| ------------------ | -------------------------- |
| `debug_start`      | Launch debugger            |
| `debug_breakpoint` | Set/remove breakpoints     |
| `debug_step`       | Step over/into/out         |
| `debug_continue`   | Resume execution           |
| `debug_evaluate`   | Eval expression in context |
| `debug_stack`      | Get call stack             |
| `debug_variables`  | Inspect variables          |

---

## 11. Deployment / Infrastructure

| Tool                | Purpose                               |
| ------------------- | ------------------------------------- |
| `deploy_preview`    | Deploy to preview environment         |
| `deploy_production` | Deploy to prod (with confirmation)    |
| `deploy_rollback`   | Revert deployment                     |
| `deploy_status`     | Check deployment state                |
| `deploy_logs`       | Fetch deployment logs                 |
| `infra_provision`   | Create resources (Railway, AWS, etc.) |
| `container_build`   | Build Docker image                    |
| `container_run`     | Run container locally                 |

---

## 12. Browser Automation (CDP)

| Tool                     | Purpose                 |
| ------------------------ | ----------------------- |
| `browser_launch`         | Start browser instance  |
| `browser_navigate`       | Go to URL               |
| `browser_click`          | Click element           |
| `browser_type`           | Input text              |
| `browser_screenshot`     | Capture screen          |
| `browser_select_element` | Visual element picker   |
| `browser_evaluate`       | Run JS in page context  |
| `browser_network`        | Intercept/mock requests |
| `browser_console`        | Get console logs        |

---

## 13. API / HTTP

| Tool                | Purpose                |
| ------------------- | ---------------------- |
| `http_request`      | Make HTTP calls        |
| `api_import_spec`   | Import OpenAPI/Swagger |
| `api_mock`          | Create mock server     |
| `api_test`          | Test endpoint          |
| `websocket_connect` | WebSocket client       |

---

## 14. Documentation

| Tool            | Purpose                                |
| --------------- | -------------------------------------- |
| `docs_generate` | Generate from code (JSDoc, docstrings) |
| `docs_search`   | Search project docs                    |
| `docs_external` | Fetch library documentation            |

---

## 15. Security

| Tool               | Purpose                  |
| ------------------ | ------------------------ |
| `security_scan`    | SAST analysis            |
| `secrets_detect`   | Find exposed credentials |
| `dependency_audit` | CVE check on packages    |

---

## 16. Architecture Visualization

| Tool                    | Purpose                   |
| ----------------------- | ------------------------- |
| `arch_diagram_generate` | Create visual from code   |
| `arch_validate`         | Check against constraints |
| `arch_suggest`          | Recommend improvements    |

---

## 17. Context / Memory

| Tool                  | Purpose                               |
| --------------------- | ------------------------------------- |
| `context_add`         | Add to agent's working memory         |
| `context_search`      | Semantic search over codebase         |
| `context_summarize`   | Compress context for token efficiency |
| `project_conventions` | Get coding standards/patterns         |

---

## Priority Tiers for MVP

| Tier                     | Tools                                                    |
| ------------------------ | -------------------------------------------------------- |
| **P0 (Must have)**       | File ops, Shell, Git, Code intelligence, Test run, Build |
| **P1 (High value)**      | Dependency graph, Database, Package mgmt, Lint/format    |
| **P2 (Differentiators)** | Browser automation, Debugging, Deployment, Security scan |
| **P3 (Later)**           | Architecture viz, Docs generation, Advanced context      |

---

**Usage:** This was a **high-usage response** (~1500 tokens). Want me to dive deeper into interface schemas for any specific tier/category?
