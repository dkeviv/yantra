# Yantra - File Registry

**Version:** MVP 1.0  
**Last Updated:** November 20, 2025 - 5:30 PM  
**Purpose:** Track all project files, their purposes, implementations, and dependencies

---

## Documentation Files

### Root Level Documentation

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `README.md` | ⚪ To be created | Project overview and quick start | None | - |
| `Specifications.md` | ✅ Exists | Complete technical specification | None | Nov 20, 2025 |
| `Project_Plan.md` | ✅ Created | Task tracking and timeline | None | Nov 20, 2025 |
| `Features.md` | ✅ Created | Feature documentation with use cases | None | Nov 20, 2025 |
| `UX.md` | ✅ Created | User flows and experience guide | None | Nov 20, 2025 |
| `Technical_Guide.md` | ✅ Created | Developer technical reference | None | Nov 20, 2025 |
| `File_Registry.md` | ✅ Created | This file - tracks all files | None | Nov 20, 2025 |
| `Decision_Log.md` | ✅ Created | Architecture and design decisions | None | Nov 20, 2025 |
| `Known_Issues.md` | ✅ Created | Bug tracking and fixes | None | Nov 20, 2025 |
| `Unit_Test_Results.md` | ✅ Created | Unit test results tracking | None | Nov 20, 2025 |
| `Integration_Test_Results.md` | ✅ Created | Integration test results | None | Nov 20, 2025 |
| `Regression_Test_Results.md` | ✅ Created | Regression test results | None | Nov 20, 2025 |

---

## Configuration Files

### Root Level Configuration

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `Cargo.toml` | ⚪ To be created | Rust workspace configuration | None | - |
| `Cargo.lock` | ⚪ Auto-generated | Rust dependency lock file | Cargo.toml | - |
| `package.json` | ✅ Created | Node.js project configuration | None | Nov 20, 2025 |
| `package-lock.json` | ✅ Auto-generated | Node.js dependency lock file | package.json | Nov 20, 2025 |
| `tauri.conf.json` | ✅ Created | Tauri application configuration | None | Nov 20, 2025 (in src-tauri/) |
| `.gitignore` | ✅ Created | Git ignore patterns | None | Nov 20, 2025 |
| `.eslintrc.json` | ✅ Created | ESLint configuration | None | Nov 20, 2025 |
| `.prettierrc` | ✅ Created | Prettier formatting configuration | None | Nov 20, 2025 |
| `tsconfig.json` | ✅ Created | TypeScript configuration | None | Nov 20, 2025 |
| `tsconfig.node.json` | ✅ Created | TypeScript Node configuration | tsconfig.json | Nov 20, 2025 |
| `tailwind.config.js` | ✅ Created | TailwindCSS configuration | None | Nov 20, 2025 |
| `postcss.config.js` | ✅ Created | PostCSS configuration | None | Nov 20, 2025 |
| `vite.config.ts` | ✅ Created | Vite build configuration | None | Nov 20, 2025 |
| `index.html` | ✅ Created | Main HTML entry point | None | Nov 20, 2025 |

---

## Source Files (Rust Backend)

### Main Application

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-tauri/src/main.rs` | ✅ Updated | Tauri app with file system and GNN commands | tauri, serde, std::fs, gnn module | Nov 20, 2025 |
| `src-tauri/build.rs` | ✅ Created | Tauri build script | tauri-build | Nov 20, 2025 |
| `src-tauri/Cargo.toml` | ✅ Updated | Rust dependencies with GNN deps | tree-sitter, petgraph, rusqlite | Nov 20, 2025 |
| `src-tauri/tauri.conf.json` | ✅ Created | Tauri app configuration | None | Nov 20, 2025 |
| `src-tauri/icons/*.png` | ✅ Created | Application icons (placeholder) | None | Nov 20, 2025 |
| `src/lib.rs` | ⚪ To be created | Library root | All modules | - |

**Main.rs Commands:**
- File System: read_file, write_file, read_dir, path_exists, get_file_info
- GNN: analyze_project, get_dependencies, get_dependents, find_node

### GNN Module (Week 3-4)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-tauri/src/gnn/mod.rs` | ✅ Created | GNN engine with CodeNode, CodeEdge types, main GNNEngine struct | parser, graph, persistence | Nov 20, 2025 |
| `src-tauri/src/gnn/parser.rs` | ✅ Created | tree-sitter Python parser, extracts functions/classes/imports/calls | tree-sitter, tree-sitter-python | Nov 20, 2025 |
| `src-tauri/src/gnn/graph.rs` | ✅ Created | petgraph CodeGraph with dependency tracking | petgraph | Nov 20, 2025 |
| `src-tauri/src/gnn/persistence.rs` | ✅ Created | SQLite database for graph persistence | rusqlite | Nov 20, 2025 |
| `src/gnn/incremental.rs` | ⚪ To be created | Incremental graph update logic | graph.rs, persistence.rs | - |
| `src/gnn/validator.rs` | ⚪ To be created | Dependency validation logic | graph.rs | - |

**Implementation Details:**
- **mod.rs (167 lines)**: Main GNN engine with parse_file(), build_graph(), scan_directory(), persist(), load(), get_dependencies(), get_dependents(), find_node()
- **parser.rs (278 lines)**: Python AST parser using tree-sitter, extracts function definitions, class definitions, imports, function calls, inheritance
- **graph.rs (232 lines)**: Directed graph using petgraph DiGraph, nodes (functions/classes/imports), edges (calls/uses/inherits), with export/import for serialization
- **persistence.rs (225 lines)**: SQLite schema with nodes and edges tables, indices for fast lookups, save_graph/load_graph methods
- **Tests**: 8 unit tests passing (parser, graph, persistence, engine creation)
- **Tauri Commands**: analyze_project, get_dependencies, get_dependents, find_node

### LLM Module (Week 5-6) - 40% Complete ✅

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src/llm/mod.rs` | ✅ Complete | LLM module root with core types | All LLM submodules | Nov 20 |
| `src/llm/claude.rs` | ✅ Complete | Claude Sonnet 4 API client | reqwest, serde, tokio | Nov 20 |
| `src/llm/openai.rs` | ✅ Complete | OpenAI GPT-4 Turbo client | reqwest, serde, tokio | Nov 20 |
| `src/llm/orchestrator.rs` | ✅ Complete | Multi-LLM orchestration + circuit breaker | claude.rs, openai.rs | Nov 20 |
| `src/llm/config.rs` | ✅ Complete | Configuration management with persistence | serde, tokio | Nov 20 |
| `src/llm/context.rs` | ⚪ Placeholder | Context assembly from GNN | GNN module | Nov 20 |
| `src/llm/prompts.rs` | ⚪ Placeholder | Prompt template system | None | Nov 20 |

**Implementation Details:**
- **mod.rs (105 lines)**: Core types - LLMConfig, LLMProvider enum (Claude/OpenAI), CodeGenerationRequest/Response, LLMError
- **claude.rs (300+ lines)**: Full HTTP client with Messages API, system/user prompt building, code block extraction, response parsing
- **openai.rs (200+ lines)**: Chat completions client with temperature 0.2 for deterministic code, similar structure to Claude
- **orchestrator.rs (280+ lines)**: CircuitBreaker state machine (Closed/Open/HalfOpen), retry with exponential backoff (100ms-400ms), automatic failover (Claude → OpenAI)
- **config.rs (180+ lines)**: JSON persistence to OS config dir, secure API key storage, sanitized config for frontend (boolean flags only)
- **context.rs (20 lines)**: Placeholder for smart context assembly from GNN
- **prompts.rs (10 lines)**: Placeholder for version-controlled templates
- **Tests**: 14 unit tests passing (circuit breaker states, recovery, orchestrator, config management, API clients)
- **Tauri Commands**: get_llm_config, set_llm_provider, set_claude_key, set_openai_key, clear_llm_key, set_llm_retry_config

**Frontend Integration:**
- `src-ui/api/llm.ts` (60 lines): TypeScript API wrapper for all LLM config Tauri commands
- `src-ui/components/LLMSettings.tsx` (230+ lines): Full-featured SolidJS settings UI with provider selection, API key inputs, status indicators

### Testing Module (Week 5-6)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src/testing/mod.rs` | ⚪ To be created | Testing module root | All testing submodules | - |
| `src/testing/generator.rs` | ⚪ To be created | Test generation logic | LLM module | - |
| `src/testing/runner.rs` | ⚪ To be created | pytest subprocess runner | tokio | - |
| `src/testing/parser.rs` | ⚪ To be created | Test result parser (JUnit XML) | None | - |
| `src/testing/tests.rs` | ⚪ To be created | Testing module unit tests | All testing modules | - |

### Security Module (Week 7)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src/security/mod.rs` | ⚪ To be created | Security module root | All security submodules | - |
| `src/security/semgrep.rs` | ⚪ To be created | Semgrep integration | tokio | - |
| `src/security/safety.rs` | ⚪ To be created | Python Safety checker | tokio | - |
| `src/security/secrets.rs` | ⚪ To be created | Secret scanning (TruffleHog patterns) | regex | - |
| `src/security/autofix.rs` | ⚪ To be created | Auto-fix generation logic | LLM module | - |
| `src/security/tests.rs` | ⚪ To be created | Security module unit tests | All security modules | - |

### Browser Module (Week 7)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src/browser/mod.rs` | ⚪ To be created | Browser module root | All browser submodules | - |
| `src/browser/cdp.rs` | ⚪ To be created | Chrome DevTools Protocol client | chromiumoxide | - |
| `src/browser/monitor.rs` | ⚪ To be created | Console monitoring logic | cdp.rs | - |
| `src/browser/validator.rs` | ⚪ To be created | Browser validation logic | cdp.rs, monitor.rs | - |
| `src/browser/tests.rs` | ⚪ To be created | Browser module unit tests | All browser modules | - |

### Git Module (Week 7)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src/git/mod.rs` | ⚪ To be created | Git module root | All git submodules | - |
| `src/git/mcp.rs` | ⚪ To be created | Model Context Protocol integration | git2 | - |
| `src/git/commit.rs` | ⚪ To be created | Commit logic | git2 | - |
| `src/git/message.rs` | ⚪ To be created | Commit message generation | LLM module | - |
| `src/git/tests.rs` | ⚪ To be created | Git module unit tests | All git modules | - |

### Learning Module (Week 7-8) - LLM Mistake Tracking

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src/learning/mod.rs` | ⚪ To be created | Learning module root | All learning submodules | - |
| `src/learning/detector.rs` | ⚪ To be created | Automatic mistake detection | Testing, Security modules | - |
| `src/learning/storage.rs` | ⚪ To be created | SQLite operations for patterns | SQLite | - |
| `src/learning/vector_db.rs` | ⚪ To be created | ChromaDB integration | ChromaDB | - |
| `src/learning/retrieval.rs` | ⚪ To be created | Pattern retrieval and ranking | vector_db.rs, storage.rs | - |
| `src/learning/maintenance.rs` | ⚪ To be created | Pattern cleanup and optimization | vector_db.rs, storage.rs | - |
| `src/learning/sanitizer.rs` | ⚪ To be created | Code sanitization for privacy | None | - |
| `src/learning/tests.rs` | ⚪ To be created | Learning module unit tests | All learning modules | - |

---

## Frontend Files (SolidJS)

### Application Root

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-ui/index.tsx` | ✅ Created | Application entry point | App.tsx | Nov 20, 2025 |
| `src-ui/App.tsx` | ✅ Created | Main app with 3-panel layout | All components, appStore | Nov 20, 2025 |
| `src-ui/styles/index.css` | ✅ Created | Global styles and Tailwind imports | TailwindCSS | Nov 20, 2025 |

### Components (Week 1-2)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-ui/components/ChatPanel.tsx` | ✅ Updated | Chat interface with mock code generation | stores/appStore.ts | Nov 20, 2025 |
| `src-ui/components/CodeViewer.tsx` | ✅ Updated | Monaco Editor with Python highlighting | stores/appStore.ts, monaco-editor | Nov 20, 2025 |
| `src-ui/components/BrowserPreview.tsx` | ✅ Created | Browser preview placeholder | None | Nov 20, 2025 |
| `src-ui/components/FileTree.tsx` | ✅ Created | File tree for project navigation | stores/appStore.ts, utils/tauri.ts | Nov 20, 2025 |
| `src-ui/components/MessageList.tsx` | ⚪ To be created | Chat message list | None | - |
| `src-ui/components/MessageInput.tsx` | ⚪ To be created | Chat input field | None | - |
| `src-ui/components/LoadingIndicator.tsx` | ⚪ To be created | Loading spinner component | None | - |
| `src-ui/components/ErrorDisplay.tsx` | ⚪ To be created | Error message display | None | - |

### State Management (Week 1-2)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-ui/stores/appStore.ts` | ✅ Updated | Global app state with signals | SolidJS | Nov 20, 2025 |
| `src-ui/stores/chatStore.ts` | ⚪ To be created | Chat state management | SolidJS | - |
| `src-ui/stores/fileStore.ts` | ⚪ To be created | File system state | SolidJS | - |
| `src-ui/stores/codeStore.ts` | ⚪ To be created | Code editor state | SolidJS | - |

### Styles (Week 1-2)

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-ui/styles/index.css` | ✅ Created | Main stylesheet with Tailwind imports | TailwindCSS | Nov 20, 2025 |
| `src-ui/styles/chat.css` | ⚪ To be created | Chat panel styles | None | - |
| `src-ui/styles/code.css` | ⚪ To be created | Code viewer styles | None | - |

### Utilities

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-ui/monaco-setup.ts` | ✅ Created | Monaco Editor worker configuration | monaco-editor | Nov 20, 2025 |
| `src-ui/utils/tauri.ts` | ✅ Created | Tauri API wrapper for file operations | @tauri-apps/api | Nov 20, 2025 |
| `src-ui/utils/formatting.ts` | ⚪ To be created | Text formatting utilities | None | - |
| `src-ui/utils/validation.ts` | ⚪ To be created | Input validation utilities | None | - |

---

## Test Files

### Integration Tests

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `tests/integration/gnn_integration_test.rs` | ⚪ To be created | GNN end-to-end integration tests | GNN module | - |
| `tests/integration/llm_integration_test.rs` | ⚪ To be created | LLM integration tests | LLM module | - |
| `tests/integration/end_to_end_test.rs` | ⚪ To be created | Complete pipeline test | All modules | - |

### Performance Tests

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `benches/gnn_benchmark.rs` | ⚪ To be created | GNN performance benchmarks | criterion, GNN module | - |
| `benches/llm_benchmark.rs` | ⚪ To be created | LLM performance benchmarks | criterion, LLM module | - |

### Frontend Tests

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `src-ui/components/__tests__/ChatPanel.test.tsx` | ⚪ To be created | ChatPanel component tests | Jest, Testing Library | - |
| `src-ui/components/__tests__/CodeViewer.test.tsx` | ⚪ To be created | CodeViewer component tests | Jest, Testing Library | - |

---

## GitHub Configuration

### Workflows

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `.github/workflows/ci.yml` | ⚪ To be created | CI/CD pipeline | None | - |
| `.github/workflows/release.yml` | ⚪ To be created | Release automation | None | - |

### Templates

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `.github/ISSUE_TEMPLATE/bug_report.md` | ⚪ To be created | Bug report template | None | - |
| `.github/ISSUE_TEMPLATE/feature_request.md` | ⚪ To be created | Feature request template | None | - |

### Documentation

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `.github/copilot-instructions.md` | ✅ Created | GitHub Copilot instructions | None | Nov 20, 2025 |
| `.github/prompts/copilot instructions.prompt.md` | ✅ Exists | Copilot instructions source | None | Nov 20, 2025 |
| `.github/Session_Handoff.md` | ⚪ To be created | Session continuity document | None | - |

---

## Database Files

| File | Status | Purpose | Dependencies | Last Updated |
|------|--------|---------|--------------|--------------|
| `yantra.db` | ⚪ Runtime created | SQLite database (GNN + cache) | None | - |
| `.gitignore` | ⚪ To be created | Includes yantra.db to not commit | None | - |

---

## Build Artifacts (Not Committed)

These files are generated during build and should be in `.gitignore`:

| Path | Purpose | Generated By |
|------|---------|--------------|
| `target/` | Rust build artifacts | cargo |
| `node_modules/` | Node.js dependencies | npm |
| `dist/` | Vite build output | vite |
| `src-tauri/target/` | Tauri build artifacts | tauri |
| `*.db` | SQLite database files | runtime |
| `*.log` | Log files | runtime |

---

## Deprecated Files

*No files are deprecated yet. When files become obsolete, they will be listed here with strikethrough.*

Example format:
- ~~`old_file.rs`~~ - Replaced by `new_file.rs` on [date]

---

## File Creation Guidelines

### Before Creating a File

1. **Check this registry** to see if file exists
2. **Check for similar functionality** in existing files
3. **Update this registry** after creating the file

### File Header Template

All source files should include a header comment:

```rust
// File: src/module/file.rs
// Purpose: Brief description of what this file does
// Dependencies: List of main dependencies
// Last Updated: Date
```

---

## Change Log

| Date | File | Change | Author |
|------|------|--------|--------|
| Nov 20, 2025 | File_Registry.md | Initial creation | AI Assistant |
| Nov 20, 2025 | Project_Plan.md | Created | AI Assistant |
| Nov 20, 2025 | Features.md | Created | AI Assistant |
| Nov 20, 2025 | UX.md | Created | AI Assistant |
| Nov 20, 2025 | Technical_Guide.md | Created | AI Assistant |
| Nov 20, 2025 | .github/copilot-instructions.md | Created | AI Assistant |

---

**Last Updated:** November 20, 2025  
**Next Update:** After each file creation/modification
