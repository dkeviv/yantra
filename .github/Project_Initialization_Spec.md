# Project Initialization & Architecture-First Workflows

**Created:** November 28, 2025  
**Status:** 🔴 NOT IMPLEMENTED  
**Priority:** ⚡ MVP CRITICAL - Foundation for all project work

---

## Overview

This specification defines how Yantra handles project initialization for both new and existing projects, ensuring architecture is ALWAYS created and reviewed BEFORE any code implementation.

**Core Principle:** Architecture-first, code-second. No code generation without approved architecture.

---

## 1. New Project Workflow

### User Intent
```
User: "Create a REST API with JWT authentication"
```

### Agent Workflow

#### Phase 1: Architecture Generation (MANDATORY)
```rust
1. Agent detects new project intent
2. Agent generates architecture using ArchitectureGenerator
   → Components: API Gateway, Auth Service, User Service, Database
   → Connections: Gateway → Services, Services → DB
3. Agent saves architecture (version 1) to SQLite
4. Agent exports to architecture.md + architecture.json
5. Agent switches UI to Architecture View tab
```

#### Phase 2: User Review & Approval (MANDATORY)
```
Agent: "📐 I've generated an architecture with 4 components and 5 connections.

Components:
  • API Gateway (Entry point for all requests)
  • Auth Service (JWT token generation/validation)  
  • User Service (User CRUD operations)
  • PostgreSQL Database (Data persistence)

Architecture:
  Client → API Gateway → Auth Service → Database
  Client → API Gateway → User Service → Database

Please review in the Architecture View tab.

Would you like me to:
1️⃣ Proceed with this architecture
2️⃣ Modify components (add/remove/change)
3️⃣ Regenerate with different approach

Your choice (1/2/3):"
```

**CRITICAL:** Agent MUST wait for user approval. No code generation until user confirms.

#### Phase 3: Code Generation (After Approval)
```rust
// Only after user approves (choice 1)
1. Agent uses architecture as context for code generation
2. Agent generates files per architecture plan
3. Agent validates each file against architecture
4. Agent runs tests
5. Agent commits with reference to architecture
```

---

## 2. Existing Project Workflow (First Time Open)

### User Action
```
User opens existing project folder in Yantra
→ /path/to/my-existing-project
```

### Agent Workflow

#### Step 1: Check for Architecture Files
```rust
fn check_architecture_files(project_path: &Path) -> ArchitectureSource {
    // Priority 1: Check for architecture.json (Yantra's format)
    if project_path.join("architecture.json").exists() {
        return ArchitectureSource::YantraFile;
    }
    
    // Priority 2: Check for architecture.md
    if project_path.join("architecture.md").exists() {
        return ArchitectureSource::MarkdownFile;
    }
    
    // Priority 3: Check for common architecture files
    let arch_files = [
        "docs/architecture.md",
        "ARCHITECTURE.md",
        "docs/design.md",
        "README.md",  // May contain architecture section
    ];
    
    for file in arch_files {
        if project_path.join(file).exists() {
            return ArchitectureSource::DocumentationFile(file.to_string());
        }
    }
    
    ArchitectureSource::None
}
```

#### Step 2: Architecture Discovery

**Scenario A: Architecture Files Found**
```
Agent: "🔍 I found architecture documentation in docs/architecture.md

Would you like me to:
1️⃣ Import this architecture into Yantra
2️⃣ Analyze the codebase and generate new architecture (ignore existing docs)
3️⃣ Compare existing docs with actual code structure

Recommendation: Option 3 (verify docs match code)

Your choice (1/2/3):"
```

**Scenario B: No Architecture Files**
```
Agent: "📂 I opened your project with 156 files.

I don't see any architecture documentation. 

Before I can help with code changes, I need to understand the architecture.

Would you like me to:
1️⃣ Analyze the codebase and auto-generate architecture (recommended)
2️⃣ You'll describe the architecture, and I'll create it
3️⃣ Skip architecture (not recommended - may break things)

Your choice (1/2/3):"
```

#### Step 3: Code Analysis & Review (If Option 1 Selected)

```rust
async fn analyze_existing_project(
    project_path: &Path,
    gnn: &GNNEngine,
    analyzer: &ArchitectureAnalyzer,
) -> ProjectAnalysisResult {
    // 1. Scan file structure
    let files = scan_directory(project_path)?;
    
    // 2. Build GNN graph
    gnn.build_graph(project_path)?;
    
    // 3. Generate architecture from code
    let architecture = analyzer.generate_from_code(project_path)?;
    
    // 4. Code quality analysis
    let quality_report = analyze_code_quality(&files, gnn)?;
    
    // 5. Dependency analysis
    let dependencies = analyze_dependencies(gnn)?;
    
    ProjectAnalysisResult {
        architecture,
        files_count: files.len(),
        quality_report,
        dependencies,
        complexity_score: calculate_complexity(gnn)?,
    }
}
```

**Agent Report:**
```
Agent: "✅ Analysis complete!

📊 Project Overview:
  • 156 files analyzed
  • 8 components identified
  • 23 inter-component connections
  • Complexity: Medium (6.2/10)

🏗️ Generated Architecture:
  • Frontend UI (React, 45 files)
  • API Gateway (Express, 12 files)
  • Auth Service (6 files)
  • User Service (15 files)
  • Payment Service (8 files)
  • Database Layer (PostgreSQL, 5 files)
  • External APIs (Stripe, SendGrid)
  • Redis Cache (3 files)

⚠️ Code Quality Issues Found:
  • 12 files with cyclic dependencies
  • 5 services accessing database directly (bypassing service layer)
  • 3 components with >500 LOC (consider splitting)

📐 Architecture saved. Please review in Architecture View tab.

Shall I:
1️⃣ Proceed with this architecture
2️⃣ Show me the issues in detail
3️⃣ Regenerate with different grouping

Your choice:"
```

#### Step 4: User Review & Refinement
```
User can:
- View architecture in Architecture View tab
- Ask questions: "Why did you group these files?"
- Request changes: "Split Frontend into UI Components and State Management"
- Approve: "Looks good, use this"
```

#### Step 5: Save as Project Baseline
```rust
// After user approval
1. Save architecture to .yantra/architecture.db
2. Export to architecture.json (version control)
3. Export to architecture.md (human-readable)
4. Mark as "baseline" version
5. Enable architecture validation for future changes
```

---

## 3. Architecture Maintenance (Ongoing)

### Scenario: User Requests Feature That Affects Architecture

**User Request:**
```
User: "Add Redis caching for user sessions"
```

**Agent Analysis:**
```rust
async fn analyze_architecture_impact(
    intent: &str,
    current_architecture: &Architecture,
    gnn: &GNNEngine,
) -> ArchitectureImpact {
    // 1. Parse intent using LLM
    let changes = llm.parse_architecture_changes(intent).await?;
    
    // 2. Identify affected components
    let affected = find_affected_components(changes, current_architecture);
    
    // 3. Calculate impact score (GNN-based)
    let impact_score = gnn.calculate_impact_score(&affected)?;
    
    // 4. Generate change preview
    let preview = generate_architecture_diff(current_architecture, &changes);
    
    ArchitectureImpact {
        new_components: changes.components_to_add,
        modified_components: affected,
        new_connections: changes.connections_to_add,
        impact_score,
        preview,
    }
}
```

**Agent Response:**
```
Agent: "🔍 Analyzing architecture impact...

📊 Impact Analysis:
  • New component needed: Redis Cache
  • Modified components: Auth Service (2 files)
  • New connections: Auth Service → Redis → Session Store
  • Impact score: LOW (isolated change, no breaking changes)

📐 Proposed Architecture Change (v2 → v3):

  BEFORE:
    Auth Service → Database (store sessions)
    
  AFTER:
    Auth Service → Redis Cache → Database (fallback)
    
  Benefits:
    • Faster session lookups (<5ms vs 50ms)
    • Reduced database load
    • Better scalability
    
  Trade-offs:
    • Additional infrastructure (Redis)
    • Session state now distributed
    • Cache invalidation complexity

🤔 Shall I:
1️⃣ Update architecture and implement
2️⃣ Show me the code changes first
3️⃣ Skip Redis, use in-memory cache instead

Your choice:"
```

### Architecture Alignment Validation

**During Code Generation:**
```rust
// Before writing each file
let deviation = deviation_detector.check_alignment(
    &generated_code,
    &target_file,
    &architecture
).await?;

if deviation.has_violations {
    // PAUSE and ask user
    prompt_user_for_decision(deviation);
}
```

**Example Violation:**
```
Agent: "⚠️ ARCHITECTURE VIOLATION DETECTED

I was about to write code that violates the architecture:

Planned Architecture:
  Auth Service → Redis → Database
  
My Generated Code:
  Auth Service → Database (direct, skipping Redis)
  
Issue: Bypassing Redis cache defeats the purpose.

Options:
1️⃣ Fix my code (use Redis as planned)
2️⃣ Update architecture (make Redis optional)
3️⃣ Cancel this change

Recommendation: Option 1

Your choice:"
```

---

## 4. Implementation Components

### Backend (Rust)

#### New File: `src-tauri/src/agent/project_initializer.rs`
```rust
pub struct ProjectInitializer {
    gnn: Arc<Mutex<GNNEngine>>,
    llm: Arc<Mutex<LLMOrchestrator>>,
    arch_manager: ArchitectureManager,
    analyzer: ArchitectureAnalyzer,
    generator: ArchitectureGenerator,
}

impl ProjectInitializer {
    /// Initialize new project with architecture-first workflow
    pub async fn initialize_new_project(
        &mut self,
        intent: &str,
        project_path: &Path,
    ) -> Result<InitializationResult, String>;
    
    /// Initialize existing project (first-time open)
    pub async fn initialize_existing_project(
        &mut self,
        project_path: &Path,
    ) -> Result<InitializationResult, String>;
    
    /// Check for existing architecture files
    fn check_architecture_files(&self, path: &Path) -> ArchitectureSource;
    
    /// Analyze codebase and generate architecture
    async fn analyze_and_generate_architecture(
        &mut self,
        path: &Path,
    ) -> Result<Architecture, String>;
    
    /// Import architecture from existing documentation
    async fn import_from_documentation(
        &mut self,
        file_path: &Path,
    ) -> Result<Architecture, String>;
    
    /// Analyze code quality
    fn analyze_code_quality(
        &self,
        files: &[PathBuf],
    ) -> CodeQualityReport;
    
    /// Wait for user approval
    async fn wait_for_user_approval(
        &self,
        architecture: &Architecture,
    ) -> ApprovalResult;
}
```

#### Enhanced: `src-tauri/src/agent/project_orchestrator.rs`
```rust
impl ProjectOrchestrator {
    /// Create project with architecture-first workflow
    pub async fn create_project_with_architecture(
        &mut self,
        intent: &str,
        project_path: &Path,
    ) -> Result<ProjectResult, String> {
        // 1. Initialize and generate architecture
        let initializer = ProjectInitializer::new(...);
        let init_result = initializer.initialize_new_project(intent, project_path).await?;
        
        // 2. Wait for user approval
        if !init_result.user_approved {
            return Err("User did not approve architecture".to_string());
        }
        
        // 3. Use approved architecture for code generation
        let architecture = init_result.architecture;
        self.generate_code_from_architecture(&architecture, project_path).await
    }
    
    /// Open existing project with architecture initialization
    pub async fn open_existing_project(
        &mut self,
        project_path: &Path,
    ) -> Result<OpenProjectResult, String> {
        // 1. Check if already initialized (.yantra/architecture.db exists)
        if self.is_initialized(project_path) {
            return self.load_existing_architecture(project_path);
        }
        
        // 2. First-time open - initialize
        let initializer = ProjectInitializer::new(...);
        initializer.initialize_existing_project(project_path).await
    }
    
    /// Check architecture alignment before code generation
    async fn ensure_architecture_alignment(
        &self,
        intent: &str,
        architecture: &Architecture,
    ) -> Result<ArchitectureImpact, String>;
}
```

#### Enhanced: `src-tauri/src/architecture/deviation_detector.rs`
```rust
impl DeviationDetector {
    /// Check if requirement aligns with current architecture
    pub async fn check_requirement_alignment(
        &self,
        requirement: &str,
        architecture: &Architecture,
    ) -> Result<AlignmentCheck, String> {
        // Parse requirement to identify components/connections
        let parsed = self.parse_requirement(requirement).await?;
        
        // Check if existing architecture supports this
        let missing_components = self.find_missing_components(&parsed, architecture);
        let missing_connections = self.find_missing_connections(&parsed, architecture);
        
        if missing_components.is_empty() && missing_connections.is_empty() {
            return Ok(AlignmentCheck::Aligned);
        }
        
        // Generate architecture impact
        Ok(AlignmentCheck::RequiresArchitectureUpdate {
            new_components: missing_components,
            new_connections: missing_connections,
            impact_score: self.calculate_impact_score(...),
        })
    }
}
```

### Frontend (TypeScript)

#### New Store: `src-ui/stores/projectInitializationStore.ts`
```typescript
export interface ProjectInitializationState {
    status: 'idle' | 'analyzing' | 'generating' | 'awaiting_approval' | 'approved' | 'rejected';
    currentArchitecture: Architecture | null;
    analysisReport: ProjectAnalysisReport | null;
    pendingApproval: boolean;
    userChoice: number | null;
}

export async function initializeNewProject(intent: string, projectPath: string);
export async function initializeExistingProject(projectPath: string);
export async function approveArchitecture(choice: number);
export async function requestArchitectureModification(changes: string);
```

#### Enhanced: `src-ui/components/ChatPanel.tsx`
```typescript
// Detect project initialization
if (message.includes('create') || message.includes('build')) {
    // Trigger architecture-first workflow
    await initializeNewProject(message, projectPath);
    
    // Wait for user approval in chat
    setAwaitingApproval(true);
    return;
}

// Handle user approval responses
if (awaitingApproval) {
    const choice = parseInt(message);
    if (choice >= 1 && choice <= 3) {
        await approveArchitecture(choice);
        setAwaitingApproval(false);
    }
}
```

---

## 5. Tauri Commands

```rust
#[tauri::command]
async fn initialize_new_project(
    intent: String,
    project_path: String,
) -> Result<InitializationResult, String>;

#[tauri::command]
async fn initialize_existing_project(
    project_path: String,
) -> Result<InitializationResult, String>;

#[tauri::command]
async fn approve_architecture(
    architecture_id: String,
    choice: u8,
) -> Result<ApprovalResult, String>;

#[tauri::command]
async fn check_requirement_alignment(
    requirement: String,
    architecture_id: String,
) -> Result<AlignmentCheck, String>;

#[tauri::command]
async fn analyze_architecture_impact(
    requirement: String,
    architecture_id: String,
) -> Result<ArchitectureImpact, String>;
```

---

## 6. User Flows

### Flow 1: New Project (Happy Path)
```
1. User: "Create a REST API with auth"
2. Agent: Generates architecture → Shows in UI
3. Agent: "Please review and approve (1/2/3)"
4. User: "1" (approve)
5. Agent: Generates code following architecture
6. Agent: Validates each file against architecture
7. Agent: Runs tests
8. Agent: Commits with architecture reference
9. Done ✅
```

### Flow 2: New Project (Modification)
```
1. User: "Create a REST API"
2. Agent: Generates architecture
3. Agent: "Please approve (1/2/3)"
4. User: "2" (modify)
5. Agent: "What changes would you like?"
6. User: "Add GraphQL support"
7. Agent: Updates architecture → Shows changes
8. Agent: "Approve updated architecture? (1/2/3)"
9. User: "1"
10. Agent: Proceeds with updated architecture
```

### Flow 3: Existing Project (Auto-generate)
```
1. User: Opens project folder
2. Agent: Checks for architecture files → None found
3. Agent: "Analyze codebase? (1/2/3)"
4. User: "1" (analyze)
5. Agent: Analyzes 156 files, builds GNN graph
6. Agent: Generates architecture, identifies 8 components
7. Agent: Shows quality report (issues found)
8. Agent: "Approve this architecture? (1/2/3)"
9. User: "2" (show issues)
10. Agent: Lists cyclic dependencies, violations
11. User: "1" (approve anyway)
12. Agent: Saves as baseline
13. Done ✅
```

### Flow 4: Requirement with Architecture Impact
```
1. User: "Add Redis caching"
2. Agent: Checks current architecture
3. Agent: Analyzes impact → New component needed
4. Agent: Shows architecture diff (v2 → v3)
5. Agent: "Approve change? (1/2/3)"
6. User: "1"
7. Agent: Updates architecture (saves v3)
8. Agent: Generates code following new architecture
9. Agent: Validates alignment
10. Done ✅
```

---

## 7. Success Criteria

### MVP Requirements:
- ✅ NO code generation without approved architecture
- ✅ Architecture generated for 100% of new projects
- ✅ Architecture generated for 100% of existing projects (first open)
- ✅ User approval required before code generation
- ✅ Architecture impact shown for requirements
- ✅ Deviation detection prevents misalignment
- ✅ Architecture versioning tracks all changes

### Performance Targets:
- Architecture generation (new): <5 seconds
- Architecture generation (existing, 100 files): <10 seconds
- Impact analysis: <2 seconds
- Deviation check: <100ms per file

### User Experience:
- Clear prompts for approval
- Understandable architecture descriptions
- Visual architecture in UI
- Easy modification workflow
- Non-technical users can understand

---

## 8. Implementation Priority

**Phase 1 (Week 1):**
1. ✅ Create `project_initializer.rs` module
2. ✅ Implement new project workflow
3. ✅ Implement user approval mechanism
4. ✅ Integrate with project_orchestrator.rs

**Phase 2 (Week 2):**
5. ✅ Implement existing project analysis
6. ✅ Add architecture file detection
7. ✅ Add code quality analysis
8. ✅ Frontend approval UI in ChatPanel

**Phase 3 (Week 3):**
9. ✅ Implement requirement alignment checking
10. ✅ Add architecture impact analysis
11. ✅ Integrate with deviation detector
12. ✅ Add architecture update workflow

**Phase 4 (Week 4):**
13. ✅ Testing and refinement
14. ✅ Documentation updates
15. ✅ User testing feedback
16. ✅ Performance optimization

---

## 9. Files to Create/Modify

### New Files:
1. `src-tauri/src/agent/project_initializer.rs` (~500 lines)
2. `src-ui/stores/projectInitializationStore.ts` (~200 lines)

### Modified Files:
1. `src-tauri/src/agent/project_orchestrator.rs` (add architecture-first methods)
2. `src-tauri/src/architecture/deviation_detector.rs` (add requirement alignment)
3. `src-ui/components/ChatPanel.tsx` (add approval workflow)
4. `src-tauri/src/main.rs` (register new commands)

### Total New Code: ~1,000 lines
### Total Modified Code: ~300 lines

---

**Status:** Ready for implementation  
**Next Step:** Create project_initializer.rs module
