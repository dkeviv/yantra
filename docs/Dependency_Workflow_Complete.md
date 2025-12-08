# Complete Dependency Workflow - New & Existing Projects

**Created:** December 5, 2025  
**Status:** Specification for Implementation  
**Related:** `.github/Specifications.md` (State Machines), `Dependency_Detection_Strategy.md`

---

## Overview

This document defines the **complete dependency tracking workflow** integrated with Yantra's state machines, covering:

1. **New Project Creation** - From requirements to deployed code
2. **Existing Project Modification** - Safe code changes with dependency validation
3. **GNN Dependency Graph** - Central source of truth for all dependency decisions

---

## Architecture Principles

### 1. **GNN as Single Source of Truth**

```
All dependency decisions flow through GNN:

┌─────────────────────────────────────────────────────────┐
│                     GNN Dependency Graph                │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Packages  │  │    Files     │  │  Functions   │ │
│  │  (External) │  │  (Internal)  │  │  (Internal)  │ │
│  └─────────────┘  └──────────────┘  └──────────────┘ │
│         │                │                  │          │
│         └────────────────┴──────────────────┘          │
│              Edges: Uses, Requires, Calls              │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ Query/Update
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼─────┐    ┌──────▼──────┐   ┌─────▼──────┐
   │ CodeGen  │    │   Testing   │   │ Deployment │
   │  Machine │    │   Machine   │   │  Machine   │
   └──────────┘    └─────────────┘   └────────────┘
```

### 2. **Dependency States**

```rust
pub enum DependencyState {
    /// Not yet assessed (new project)
    Unknown,

    /// Assessed, ready to install (dry run passed)
    Validated {
        packages: Vec<PackageInfo>,
        conflicts: Vec<ConflictInfo>,
        total_size: u64,
        install_time_estimate: Duration,
    },

    /// Installed and tracked in GNN
    Installed {
        packages: Vec<InstalledPackage>,
        lock_file: Option<PathBuf>,
        gnn_nodes: Vec<String>, // Node IDs in graph
    },

    /// Drift detected (lock vs runtime mismatch)
    Drifted {
        mismatches: Vec<VersionMismatch>,
    },

    /// Conflict detected (incompatible versions)
    Conflicted {
        conflicts: Vec<ConflictInfo>,
    },
}

pub struct PackageInfo {
    pub name: String,
    pub version: String,              // Exact version: "1.26.3"
    pub source: DependencySource,     // LockFile, Runtime, Manifest
    pub language: Language,           // Python, JavaScript, Rust
    pub required_by: Vec<String>,     // File paths that import this
    pub requires: Vec<PackageRequirement>, // Transitive dependencies
    pub cve_vulnerabilities: Vec<CVE>,
}

pub struct InstalledPackage {
    pub info: PackageInfo,
    pub gnn_node_id: String,          // "pkg:numpy:1.26.3"
    pub installation_date: SystemTime,
    pub used_functions: Vec<String>,  // np.array, np.mean
}
```

### 3. **Hybrid Detection Strategy** (From Dependency_Detection_Strategy.md)

```
Priority Order:
1. Lock File (if exists & recent)     → Fast (<1ms), exact, reproducible
2. Runtime Inspection (fallback)      → Accurate (100-500ms), dev workflow
3. Manifest File (warning only)       → Version ranges, imprecise
```

---

## Flow 1: New Project Creation

### **Scenario:** User says "Build a Flask API with PostgreSQL"

### **State Machine:** Code Generation Machine

```
┌──────────────────────────────────────────────────────────────────┐
│              NEW PROJECT DEPENDENCY WORKFLOW                     │
└──────────────────────────────────────────────────────────────────┘

Phase 1: Architecture & Design
────────────────────────────────
State: ArchitectureGeneration
├─ Input: User intent ("Flask API with PostgreSQL")
├─ GNN Query: Check for existing similar projects (templates)
├─ LLM: Generate architecture plan
│   └─ Output: {
│        files: ["app.py", "models.py", "config.py"],
│        dependencies: ["flask", "psycopg2", "python-dotenv"],
│        database: "PostgreSQL",
│        architecture: "MVC pattern"
│      }
└─ GNN Update: Create architecture nodes (NOT in graph yet)

State: ArchitectureReview
├─ Human Approval: User reviews architecture
└─ ⚠️ APPROVAL GATE (blocks until approved)


Phase 2: Planning (DEPENDENCY CRITICAL)
────────────────────────────────────────
State: DependencyAssessment ⭐ NEW/ENHANCED
├─ Input: Required packages from architecture
│
├─ Step 1: Package Discovery
│   ├─ Parse architecture plan for dependencies
│   ├─ Extract import statements from templates/examples
│   └─ Query: packages = ["flask", "psycopg2", "python-dotenv"]
│
├─ Step 2: Version Resolution (Hybrid Strategy)
│   ├─ Check lock file: ❌ None (new project)
│   ├─ Check runtime: ❌ None (venv not created yet)
│   ├─ Check manifest: ❌ None (new project)
│   └─ Fetch latest stable versions:
│       ├─ PyPI API: flask → 3.0.0 (latest stable)
│       ├─ PyPI API: psycopg2 → 2.9.9 (latest stable)
│       └─ PyPI API: python-dotenv → 1.0.0 (latest stable)
│
├─ Step 3: Dry Run Validation ⭐ CRITICAL
│   ├─ Check version compatibility:
│   │   ├─ flask 3.0.0 requires python>=3.8
│   │   ├─ psycopg2 2.9.9 requires python>=3.7
│   │   └─ ✅ All compatible with python 3.11
│   │
│   ├─ Check for conflicts:
│   │   ├─ flask requires Werkzeug>=3.0.0
│   │   ├─ Werkzeug 3.0.3 available
│   │   └─ ✅ No conflicts detected
│   │
│   ├─ Simulate pip install (dry run):
│   │   └─ $ pip install --dry-run flask psycopg2 python-dotenv
│   │       Output: Would install 12 packages (flask + deps)
│   │       Total size: 45.3 MB
│   │       Estimated time: 8s
│   │
│   └─ Check CVE database:
│       ├─ Query: vulnerabilities in [flask, psycopg2, python-dotenv]
│       └─ ✅ No critical vulnerabilities found
│
├─ Step 4: Conflict Resolution (if any)
│   └─ If conflicts detected:
│       ├─ Find compatible versions via SAT solver
│       ├─ Suggest alternatives (e.g., asyncpg instead of psycopg2)
│       └─ Escalate to user if unresolvable
│
├─ Step 5: Create requirements.txt (manifest)
│   └─ Generate:
│       # requirements.txt
│       flask==3.0.0
│       psycopg2==2.9.9
│       python-dotenv==1.0.0
│
├─ Output: DependencyPlan {
│     packages: [
│       { name: "flask", version: "3.0.0", source: "PyPI" },
│       { name: "psycopg2", version: "2.9.9", source: "PyPI" },
│       { name: "python-dotenv", version: "1.0.0", source: "PyPI" },
│     ],
│     conflicts: [],
│     vulnerabilities: [],
│     total_size: 45.3 MB,
│     install_time: 8s,
│     manifest_file: "requirements.txt"
│   }
│
└─ State Transition: → TaskDecomposition

State: TaskDecomposition
├─ Break architecture into tasks
└─ State Transition: → EnvironmentSetup


Phase 2.5: Environment Setup (EXECUTION PREP)
──────────────────────────────────────────────
State: EnvironmentSetup ⭐ CRITICAL
├─ Step 1: Create virtual environment
│   ├─ Check if .venv exists: ❌ None
│   ├─ Run: python3 -m venv .venv
│   └─ ✅ Virtual environment created
│
├─ Step 2: Install dependencies (ACTUAL INSTALLATION)
│   ├─ Activate venv: source .venv/bin/activate
│   ├─ Run: pip install -r requirements.txt
│   │   └─ Progress: flask (8.2MB) ▓▓▓▓▓▓▓▓░░ 80%
│   └─ ✅ All packages installed
│
├─ Step 3: Generate lock file (REPRODUCIBILITY)
│   ├─ Run: pip freeze > requirements.lock.txt
│   └─ Output: requirements.lock.txt with EXACT versions
│       flask==3.0.0
│       Werkzeug==3.0.3
│       Jinja2==3.1.2
│       ... (all transitive deps with exact versions)
│
├─ Step 4: Create dependency nodes in GNN ⭐ GRAPH CREATION
│   ├─ For each installed package:
│   │   └─ GNN.add_package_node(
│   │        id: "pkg:flask:3.0.0",
│   │        name: "flask",
│   │        version: "3.0.0",
│   │        language: Python,
│   │        installation_date: now(),
│   │        source: "pip install",
│   │      )
│   │
│   └─ Create transitive dependency edges:
│       ├─ Edge: "pkg:flask:3.0.0" --Requires--> "pkg:Werkzeug:3.0.3"
│       ├─ Edge: "pkg:flask:3.0.0" --Requires--> "pkg:Jinja2:3.1.2"
│       └─ ... (full dependency tree)
│
└─ Output: DependencyState::Installed {
      packages: [...],
      lock_file: Some("requirements.lock.txt"),
      gnn_nodes: ["pkg:flask:3.0.0", "pkg:psycopg2:2.9.9", ...]
    }


Phase 3: Execution (CODE GENERATION)
────────────────────────────────────
State: ContextAssembly
├─ Query GNN for available packages:
│   └─ GNN.get_installed_packages() → [flask:3.0.0, psycopg2:2.9.9]
│
├─ Build context:
│   └─ {
│       packages: [flask, psycopg2, python-dotenv],
│       architecture: "MVC",
│       database: "PostgreSQL",
│       examples: [...] // Code patterns from GNN
│     }
│
└─ State Transition: → CodeGeneration

State: CodeGeneration
├─ LLM generates code WITH dependency awareness:
│   └─ Prompt includes:
│       "Available packages: flask==3.0.0, psycopg2==2.9.9
│        Do NOT use packages outside this list.
│        Use import flask, not from flask import *"
│
├─ Generated code:
│   └─ app.py:
│       from flask import Flask, request
│       import psycopg2
│       from dotenv import load_dotenv
│       ...
│
└─ State Transition: → DependencyValidation

State: DependencyValidation ⭐ GRAPH UPDATE
├─ Parse generated code for imports:
│   ├─ Tree-sitter AST: find all import statements
│   └─ Found: [flask, psycopg2, dotenv, os, json]
│
├─ Validate imports against GNN:
│   ├─ flask: ✅ In GNN (pkg:flask:3.0.0)
│   ├─ psycopg2: ✅ In GNN (pkg:psycopg2:2.9.9)
│   ├─ dotenv: ✅ In GNN (pkg:python-dotenv:1.0.0)
│   ├─ os: ✅ Built-in (skip)
│   └─ json: ✅ Built-in (skip)
│
├─ Create file-to-package edges in GNN:
│   ├─ Edge: "app.py" --Uses--> "pkg:flask:3.0.0"
│   ├─ Edge: "app.py" --Uses--> "pkg:psycopg2:2.9.9"
│   └─ Edge: "app.py" --Uses--> "pkg:python-dotenv:1.0.0"
│
├─ Track function-level usage (DETAILED):
│   └─ Parse: from flask import Flask, request
│       ├─ Update GNN node: pkg:flask:3.0.0
│       │   └─ used_functions: ["Flask", "request"]
│       └─ This enables: "Which files use flask.request?"
│
└─ State Transition: → BrowserValidation

[Remaining states: BrowserValidation, SecurityScanning, FixingIssues, Complete]
```

---

## Flow 2: Existing Project Modification

### **Scenario:** User says "Add authentication to Flask API"

### **State Machine:** Code Generation Machine

```
┌──────────────────────────────────────────────────────────────────┐
│         EXISTING PROJECT DEPENDENCY WORKFLOW                     │
└──────────────────────────────────────────────────────────────────┘

Phase 0: Pre-Check (GNN Load)
──────────────────────────────
├─ Load GNN from .yantra/graph.db
├─ Query existing dependencies:
│   └─ GNN.get_installed_packages() → [
│        { name: "flask", version: "3.0.0", gnn_id: "pkg:flask:3.0.0" },
│        { name: "psycopg2", version: "2.9.9", gnn_id: "pkg:psycopg2:2.9.9" },
│      ]
│
└─ Check for drift:
    ├─ Read lock file: requirements.lock.txt → flask==3.0.0
    ├─ Runtime inspection: pip show flask → 3.0.0
    └─ ✅ No drift detected


Phase 1: Architecture & Design
────────────────────────────────
State: ArchitectureGeneration
├─ Input: "Add authentication"
├─ GNN Query: Load existing architecture
│   └─ Current: ["app.py", "models.py", "config.py"]
│
├─ LLM: Generate architecture changes
│   └─ Output: {
│        new_files: ["auth.py"],
│        modified_files: ["app.py"],
│        new_dependencies: ["flask-jwt-extended", "bcrypt"],
│        removed_dependencies: []
│      }
│
└─ State Transition: → ArchitectureReview

State: ArchitectureReview
└─ ⚠️ User approves adding flask-jwt-extended


Phase 2: Planning (DEPENDENCY UPDATE)
──────────────────────────────────────
State: DependencyAssessment ⭐ INCREMENTAL UPDATE
├─ Input: New dependencies ["flask-jwt-extended", "bcrypt"]
│
├─ Step 1: Check if already installed
│   ├─ GNN Query: get_package("flask-jwt-extended") → ❌ Not found
│   ├─ GNN Query: get_package("bcrypt") → ❌ Not found
│   └─ Need to install: ["flask-jwt-extended", "bcrypt"]
│
├─ Step 2: Version Resolution (Hybrid Strategy)
│   ├─ Check lock file: requirements.lock.txt
│   │   └─ ❌ Neither package in lock file
│   │
│   ├─ Check runtime: pip show flask-jwt-extended
│   │   └─ ❌ Not installed
│   │
│   └─ Fetch latest compatible versions:
│       ├─ PyPI: flask-jwt-extended → 4.6.0
│       │   └─ Requires: flask>=2.0, PyJWT>=2.0
│       └─ PyPI: bcrypt → 4.1.2
│           └─ Requires: python>=3.7
│
├─ Step 3: Compatibility Check (CRITICAL)
│   ├─ Check against existing packages:
│   │   ├─ flask-jwt-extended 4.6.0 requires flask>=2.0
│   │   ├─ Current: flask==3.0.0 ✅ Compatible
│   │   └─ ✅ No conflicts
│   │
│   ├─ Dry run validation:
│   │   └─ $ pip install --dry-run flask-jwt-extended bcrypt
│   │       Would install 3 packages:
│   │       - flask-jwt-extended==4.6.0
│   │       - bcrypt==4.1.2
│   │       - PyJWT==2.8.0 (new transitive dep)
│   │       Total size: 12.1 MB
│   │       Estimated time: 3s
│   │
│   └─ CVE check:
│       └─ ✅ No vulnerabilities
│
├─ Step 4: Update requirements.txt
│   └─ Append:
│       flask-jwt-extended==4.6.0
│       bcrypt==4.1.2
│
└─ State Transition: → EnvironmentSetup


Phase 2.5: Environment Setup (INCREMENTAL INSTALL)
───────────────────────────────────────────────────
State: EnvironmentSetup ⭐ INSTALL NEW DEPS
├─ Step 1: Activate existing venv
│   └─ source .venv/bin/activate ✅
│
├─ Step 2: Install NEW dependencies only
│   ├─ Run: pip install flask-jwt-extended==4.6.0 bcrypt==4.1.2
│   └─ ✅ Installed (3s)
│
├─ Step 3: Update lock file
│   ├─ Run: pip freeze > requirements.lock.txt
│   └─ ✅ Lock file updated with new packages
│
├─ Step 4: Add NEW dependency nodes to GNN ⭐ GRAPH UPDATE
│   ├─ GNN.add_package_node(
│   │    id: "pkg:flask-jwt-extended:4.6.0",
│   │    name: "flask-jwt-extended",
│   │    version: "4.6.0",
│   │    installation_date: now(),
│   │  )
│   │
│   ├─ GNN.add_package_node(
│   │    id: "pkg:bcrypt:4.1.2",
│   │    name: "bcrypt",
│   │    version: "4.1.2",
│   │  )
│   │
│   └─ Create transitive edges:
│       └─ "pkg:flask-jwt-extended:4.6.0" --Requires--> "pkg:PyJWT:2.8.0"
│
└─ State Transition: → ContextAssembly


Phase 3: Execution (CODE GENERATION WITH DEPENDENCY AWARENESS)
───────────────────────────────────────────────────────────────
State: ContextAssembly
├─ Query GNN for ALL dependencies (existing + new):
│   └─ GNN.get_installed_packages() → [
│        flask:3.0.0,
│        psycopg2:2.9.9,
│        flask-jwt-extended:4.6.0, ⬅ NEW
│        bcrypt:4.1.2,                ⬅ NEW
│      ]
│
├─ Query GNN for existing code that uses flask:
│   └─ GNN.get_dependents("pkg:flask:3.0.0") → ["app.py"]
│
└─ Build context:
    └─ {
        packages: [...all packages],
        existing_files: {
          "app.py": {
            imports: ["flask", "psycopg2"],
            functions: ["create_app", "index"],
            uses_auth: false  ⬅ From GNN analysis
          }
        },
        new_packages: ["flask-jwt-extended", "bcrypt"]
      }

State: CodeGeneration
├─ LLM generates code with dependency awareness:
│   └─ Prompt:
│       "Existing: app.py uses flask, psycopg2
│        New packages available: flask-jwt-extended, bcrypt
│        Modify app.py to add JWT authentication
│        Create auth.py for authentication logic"
│
├─ Generated code:
│   ├─ app.py (MODIFIED):
│   │   from flask import Flask, request
│   │   from flask_jwt_extended import JWTManager  ⬅ NEW IMPORT
│   │   from auth import authenticate, hash_password ⬅ NEW IMPORT
│   │   ...
│   │
│   └─ auth.py (NEW FILE):
│       from flask_jwt_extended import create_access_token
│       import bcrypt
│       ...
│
└─ State Transition: → DependencyValidation

State: DependencyValidation ⭐ VALIDATE & UPDATE GRAPH
├─ Parse generated code for imports:
│   ├─ app.py imports: [flask, flask_jwt_extended, psycopg2, auth]
│   └─ auth.py imports: [flask_jwt_extended, bcrypt]
│
├─ Validate NEW imports against GNN:
│   ├─ flask_jwt_extended: ✅ In GNN (pkg:flask-jwt-extended:4.6.0)
│   ├─ bcrypt: ✅ In GNN (pkg:bcrypt:4.1.2)
│   └─ auth: ✅ Internal module (will be created)
│
├─ Detect BREAKING CHANGES (CRITICAL):
│   ├─ Query GNN: "Which files depend on app.py?"
│   │   └─ Result: ["test_app.py", "models.py"]
│   │
│   ├─ Analyze changes to app.py:
│   │   ├─ Function create_app(): Modified (added JWT init)
│   │   ├─ Function index(): Unchanged
│   │   └─ ⚠️ New dependency: flask_jwt_extended
│   │
│   ├─ Impact analysis:
│   │   ├─ test_app.py: ⚠️ May need updates (imports app.create_app)
│   │   └─ models.py: ✅ No impact (doesn't import create_app)
│   │
│   └─ Warn user:
│       "⚠️ Modifying app.py may affect 2 files:
│        - test_app.py (imports create_app)
│        - Consider updating tests for JWT authentication"
│
├─ Update GNN with NEW file-to-package edges:
│   ├─ Edge: "app.py" --Uses--> "pkg:flask-jwt-extended:4.6.0"
│   ├─ Edge: "auth.py" --Uses--> "pkg:flask-jwt-extended:4.6.0"
│   └─ Edge: "auth.py" --Uses--> "pkg:bcrypt:4.1.2"
│
├─ Track function-level usage:
│   └─ Update pkg:flask-jwt-extended:4.6.0:
│       └─ used_functions: ["JWTManager", "create_access_token"]
│
└─ State Transition: → BrowserValidation

[Remaining states continue with validation and testing]
```

---

## Flow 3: Dependency Drift Detection & Resolution

### **Scenario:** Developer manually upgrades package

```
┌──────────────────────────────────────────────────────────────────┐
│              DEPENDENCY DRIFT WORKFLOW                           │
└──────────────────────────────────────────────────────────────────┘

Trigger: User runs "pip install --upgrade flask"
────────────────────────────────────────────────────────────────────

1. File Watcher Detects Change
   ├─ Monitor: .venv/lib/python*/site-packages/
   └─ Event: flask updated 3.0.0 → 3.1.0

2. Drift Detection (Automatic Background Process)
   ├─ Query GNN: get_package("flask") → version: "3.0.0"
   ├─ Runtime inspection: pip show flask → version: "3.1.0"
   └─ ⚠️ DRIFT DETECTED: Lock file (3.0.0) ≠ Installed (3.1.0)

3. Impact Analysis
   ├─ Query GNN: get_dependents("pkg:flask:3.0.0")
   │   └─ Files: ["app.py", "auth.py"]
   │
   ├─ Query GNN: get_packages_requiring("flask")
   │   └─ Packages: ["flask-jwt-extended:4.6.0"]
   │
   ├─ Check compatibility:
   │   ├─ flask-jwt-extended 4.6.0 requires flask>=2.0
   │   ├─ flask 3.1.0 ✅ Satisfies requirement
   │   └─ ✅ No conflicts detected
   │
   └─ Check for breaking changes:
       ├─ Query changelog: flask 3.0.0 → 3.1.0
       ├─ Found: Deprecated flask.json module (use flask.json directly)
       └─ Scan code: grep -r "flask.json" app.py
           └─ ❌ Not found (no impact)

4. User Notification (In-App Toast)
   ┌─────────────────────────────────────────────────────────┐
   │  ⚠️ Dependency Drift Detected                          │
   │                                                         │
   │  Package: flask                                         │
   │  Lock file: 3.0.0                                       │
   │  Installed: 3.1.0                                       │
   │                                                         │
   │  Impact: 2 files affected (app.py, auth.py)           │
   │  Breaking changes: None detected                        │
   │                                                         │
   │  [Update Lock File] [Revert to 3.0.0] [Dismiss]       │
   └─────────────────────────────────────────────────────────┘

5. Resolution (User Choice: Update Lock File)
   ├─ Step 1: Update requirements.txt
   │   └─ Change: flask==3.0.0 → flask==3.1.0
   │
   ├─ Step 2: Update lock file
   │   └─ Run: pip freeze > requirements.lock.txt
   │
   ├─ Step 3: Update GNN (Version History Tracking)
   │   ├─ Create new node: "pkg:flask:3.1.0"
   │   ├─ Copy edges from old node to new node
   │   ├─ Add version history:
   │   │   └─ VersionChange {
   │   │        from: "3.0.0",
   │   │        to: "3.1.0",
   │   │        changed_at: now(),
   │   │        reason: "Manual upgrade by user"
   │   │      }
   │   └─ Update file edges:
   │       ├─ app.py: "pkg:flask:3.0.0" → "pkg:flask:3.1.0"
   │       └─ auth.py: "pkg:flask:3.0.0" → "pkg:flask:3.1.0"
   │
   └─ Step 4: Trigger testing
       └─ Auto-run tests for affected files (app.py, auth.py)
```

---

## Complete State Machine Integration

### **Code Generation Machine States with Dependency Focus**

```rust
pub enum CodeGenState {
    // Phase 1: Architecture & Design
    ArchitectureGeneration,
    ArchitectureReview,  // ⚠️ APPROVAL GATE

    // Phase 2: Planning
    DependencyAssessment,  // ⭐ NEW/ENHANCED
    TaskDecomposition,
    DependencySequencing,
    ConflictCheck,
    PlanGeneration,
    PlanReview,  // Optional approval

    // Phase 2.5: Environment Setup
    EnvironmentSetup,  // ⭐ CRITICAL - Install deps + Update GNN

    // Phase 3: Execution
    ContextAssembly,
    CodeGeneration,
    DependencyValidation,  // ⭐ CRITICAL - Validate + Update GNN
    BrowserValidation,
    SecurityScanning,

    // Phase 4: Fix/Complete
    FixingIssues,
    Complete,
    Failed,
}
```

### **DependencyAssessment State (Detailed)**

```rust
pub struct DependencyAssessmentState {
    pub required_packages: Vec<String>,  // From architecture/code
    pub resolved_versions: HashMap<String, String>,  // name → version
    pub conflicts: Vec<ConflictInfo>,
    pub vulnerabilities: Vec<CVE>,
    pub total_size: u64,
    pub install_time_estimate: Duration,
    pub gnn_query_results: GNNQueryResults,
}

impl DependencyAssessmentState {
    pub async fn execute(&mut self) -> Result<StateTransition> {
        // Step 1: Discover required packages
        self.discover_packages().await?;

        // Step 2: Check if already installed (GNN query)
        let installed = self.check_gnn_for_installed().await?;
        self.required_packages.retain(|p| !installed.contains(p));

        // Step 3: Resolve versions (hybrid strategy)
        self.resolve_versions().await?;

        // Step 4: Dry run validation
        self.validate_compatibility().await?;

        // Step 5: Check CVE database
        self.check_vulnerabilities().await?;

        // Step 6: Conflict resolution (if needed)
        if !self.conflicts.is_empty() {
            self.resolve_conflicts().await?;
        }

        // Step 7: Generate manifest file
        self.generate_requirements_txt()?;

        Ok(StateTransition::ToEnvironmentSetup)
    }

    async fn check_gnn_for_installed(&self) -> Result<Vec<String>> {
        // Query GNN: "Which packages are already in the graph?"
        let installed_nodes = self.gnn.query(
            "SELECT name FROM nodes WHERE type = 'Package'"
        )?;

        Ok(installed_nodes.into_iter()
            .map(|n| n.name)
            .collect())
    }

    async fn resolve_versions(&mut self) -> Result<()> {
        for package in &self.required_packages {
            // Hybrid strategy (from Dependency_Detection_Strategy.md)
            let version = if let Some(v) = self.read_lock_file(package)? {
                v  // Fast path: lock file exists
            } else if let Some(v) = self.runtime_inspection(package)? {
                v  // Dev workflow: check what's installed
            } else {
                self.fetch_latest_stable(package).await?  // New install
            };

            self.resolved_versions.insert(package.clone(), version);
        }
        Ok(())
    }
}
```

### **EnvironmentSetup State (Detailed)**

```rust
pub struct EnvironmentSetupState {
    pub venv_path: PathBuf,
    pub packages_to_install: Vec<PackageInfo>,
    pub installed_packages: Vec<InstalledPackage>,
    pub lock_file_path: PathBuf,
    pub gnn_update_log: Vec<String>,
}

impl EnvironmentSetupState {
    pub async fn execute(&mut self) -> Result<StateTransition> {
        // Step 1: Create/activate venv
        self.ensure_venv().await?;

        // Step 2: Install dependencies
        self.install_packages().await?;

        // Step 3: Generate/update lock file
        self.generate_lock_file().await?;

        // Step 4: Update GNN with NEW dependency nodes ⭐ CRITICAL
        self.update_gnn_graph().await?;

        Ok(StateTransition::ToContextAssembly)
    }

    async fn update_gnn_graph(&mut self) -> Result<()> {
        for package in &self.installed_packages {
            // Create package node
            let node_id = format!("pkg:{}:{}", package.name, package.version);

            self.gnn.add_node(CodeNode {
                id: node_id.clone(),
                node_type: NodeType::Package,
                name: format!("{}=={}", package.name, package.version),
                metadata: PackageMetadata {
                    version: package.version.clone(),
                    language: package.language,
                    installation_date: SystemTime::now(),
                    source: package.source.clone(),
                },
            })?;

            // Create transitive dependency edges
            for dep in &package.requires {
                let dep_node_id = format!("pkg:{}:{}", dep.package, dep.version);

                self.gnn.add_edge(CodeEdge {
                    source_id: node_id.clone(),
                    target_id: dep_node_id,
                    edge_type: EdgeType::Requires,
                })?;
            }

            self.gnn_update_log.push(format!(
                "Added package node: {} (with {} dependencies)",
                node_id, package.requires.len()
            ));
        }

        // Persist GNN to disk
        self.gnn.persist()?;

        Ok(())
    }
}
```

### **DependencyValidation State (Detailed)**

```rust
pub struct DependencyValidationState {
    pub generated_code: String,
    pub file_path: String,
    pub detected_imports: Vec<ImportStatement>,
    pub validation_errors: Vec<ValidationError>,
    pub graph_updates: Vec<GraphUpdate>,
}

impl DependencyValidationState {
    pub async fn execute(&mut self) -> Result<StateTransition> {
        // Step 1: Parse code for imports
        self.detect_imports()?;

        // Step 2: Validate against GNN
        self.validate_imports_against_gnn().await?;

        // Step 3: Breaking change analysis
        self.analyze_breaking_changes().await?;

        // Step 4: Update GNN with file-to-package edges
        self.update_gnn_edges().await?;

        // Step 5: Track function-level usage
        self.track_function_usage().await?;

        if self.validation_errors.is_empty() {
            Ok(StateTransition::ToBrowserValidation)
        } else {
            Ok(StateTransition::ToFixingIssues)
        }
    }

    async fn validate_imports_against_gnn(&mut self) -> Result<()> {
        for import in &self.detected_imports {
            // Check if package exists in GNN
            let query = format!(
                "SELECT id FROM nodes WHERE type = 'Package' AND name LIKE '{}%'",
                import.package
            );

            let results = self.gnn.query(&query)?;

            if results.is_empty() {
                self.validation_errors.push(ValidationError {
                    severity: ErrorSeverity::Error,
                    message: format!(
                        "Package '{}' not found in dependency graph.
                         Did you forget to add it to requirements.txt?",
                        import.package
                    ),
                    suggestion: format!(
                        "Run: pip install {} && pip freeze > requirements.lock.txt",
                        import.package
                    ),
                });
            }
        }
        Ok(())
    }

    async fn analyze_breaking_changes(&mut self) -> Result<()> {
        // Query GNN: "Which files/functions depend on this file?"
        let dependents = self.gnn.get_dependents(&self.file_path)?;

        if !dependents.is_empty() {
            self.validation_errors.push(ValidationError {
                severity: ErrorSeverity::Warning,
                message: format!(
                    "Modifying {} may affect {} other files",
                    self.file_path,
                    dependents.len()
                ),
                affected_files: dependents.iter()
                    .map(|d| d.file_path.clone())
                    .collect(),
            });
        }

        Ok(())
    }

    async fn update_gnn_edges(&mut self) -> Result<()> {
        for import in &self.detected_imports {
            // Find package node in GNN
            let package_node_id = self.gnn.find_package_node(
                &import.package,
                None  // Any version
            )?;

            // Create file → package edge
            self.gnn.add_edge(CodeEdge {
                source_id: self.file_path.clone(),
                target_id: package_node_id.clone(),
                edge_type: EdgeType::Uses,
                metadata: EdgeMetadata {
                    import_statement: import.raw_statement.clone(),
                    line_number: import.line_number,
                },
            })?;

            self.graph_updates.push(GraphUpdate {
                edge_type: "Uses",
                source: self.file_path.clone(),
                target: package_node_id,
            });
        }

        // Persist updates
        self.gnn.persist()?;

        Ok(())
    }
}
```

---

## GNN Query API for Dependency Management

```rust
impl GNNEngine {
    /// Get all installed packages (with versions)
    pub fn get_installed_packages(&self) -> Result<Vec<InstalledPackage>> {
        let nodes = self.graph.query_nodes(|node| {
            node.node_type == NodeType::Package
        })?;

        Ok(nodes.into_iter()
            .map(|n| InstalledPackage::from_node(n))
            .collect())
    }

    /// Check if package is installed (returns exact version)
    pub fn get_package_version(&self, name: &str) -> Result<Option<String>> {
        let node = self.graph.find_node(|n| {
            n.node_type == NodeType::Package && n.name.starts_with(name)
        })?;

        Ok(node.map(|n| n.metadata.version.clone()))
    }

    /// Get all files that use a package
    pub fn get_files_using_package(&self, package: &str, version: Option<&str>)
        -> Result<Vec<String>> {
        let package_id = if let Some(v) = version {
            format!("pkg:{}:{}", package, v)
        } else {
            // Find any version
            self.find_package_node(package, None)?
        };

        let dependents = self.graph.get_dependents(&package_id);

        Ok(dependents.into_iter()
            .filter(|n| n.node_type == NodeType::File)
            .map(|n| n.file_path.clone())
            .collect())
    }

    /// Get all packages used by a file
    pub fn get_packages_used_by_file(&self, file_path: &str)
        -> Result<Vec<PackageInfo>> {
        let dependencies = self.graph.get_dependencies(file_path);

        Ok(dependencies.into_iter()
            .filter(|n| n.node_type == NodeType::Package)
            .map(|n| PackageInfo::from_node(n))
            .collect())
    }

    /// Impact analysis: What breaks if I upgrade this package?
    pub fn analyze_upgrade_impact(&self, package: &str,
                                   from_version: &str,
                                   to_version: &str)
        -> Result<UpgradeImpact> {
        let old_id = format!("pkg:{}:{}", package, from_version);
        let affected_files = self.get_files_using_package(package, Some(from_version))?;

        // Check for breaking changes (query external API or local database)
        let breaking_changes = self.check_breaking_changes(
            package, from_version, to_version
        )?;

        Ok(UpgradeImpact {
            package: package.to_string(),
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            affected_files,
            breaking_changes,
            risk_level: self.calculate_risk_level(&affected_files, &breaking_changes),
        })
    }

    /// Find unused packages (not referenced by any file)
    pub fn find_unused_packages(&self) -> Result<Vec<String>> {
        let all_packages = self.get_installed_packages()?;
        let mut unused = Vec::new();

        for package in all_packages {
            let users = self.get_files_using_package(&package.name, Some(&package.version))?;
            if users.is_empty() {
                unused.push(format!("{}=={}", package.name, package.version));
            }
        }

        Ok(unused)
    }
}
```

---

## Error Handling & Recovery

### **Common Scenarios**

```rust
pub enum DependencyError {
    /// Package not found in any source
    PackageNotFound { name: String },

    /// Version conflict detected
    VersionConflict {
        package: String,
        required_by: Vec<(String, String)>,  // (file, version_spec)
    },

    /// CVE vulnerability detected
    VulnerabilityDetected {
        package: String,
        version: String,
        cve_ids: Vec<String>,
        severity: Severity,
    },

    /// Import not in dependency graph
    UnrecognizedImport {
        import: String,
        file: String,
        suggestion: String,
    },

    /// Drift detected
    VersionDrift {
        package: String,
        lock_version: String,
        installed_version: String,
    },
}

impl DependencyError {
    pub fn recovery_action(&self) -> RecoveryAction {
        match self {
            DependencyError::PackageNotFound { name } => {
                RecoveryAction::InstallPackage {
                    package: name.clone(),
                    prompt_user: true,
                }
            }

            DependencyError::VersionConflict { package, required_by } => {
                RecoveryAction::ResolveConflict {
                    use_sat_solver: true,
                    fallback_to_user: true,
                }
            }

            DependencyError::VulnerabilityDetected { severity, .. } => {
                if severity == Severity::Critical {
                    RecoveryAction::BlockAndAlert
                } else {
                    RecoveryAction::WarnUser
                }
            }

            DependencyError::UnrecognizedImport { suggestion, .. } => {
                RecoveryAction::SuggestFix {
                    suggestion: suggestion.clone(),
                    auto_fix_available: true,
                }
            }

            DependencyError::VersionDrift { .. } => {
                RecoveryAction::PromptUpdate {
                    options: vec![
                        "Update lock file to match installed",
                        "Revert to lock file version",
                        "Ignore (not recommended)"
                    ],
                }
            }
        }
    }
}
```

---

## Performance Targets

| Operation                     | Target | Current | Status |
| ----------------------------- | ------ | ------- | ------ |
| GNN package query             | <10ms  | 2ms     | ✅     |
| Dry run validation            | <5s    | 3s      | ✅     |
| Lock file read                | <1ms   | 0.5ms   | ✅     |
| Runtime inspection (pip show) | <500ms | 250ms   | ✅     |
| Package install (small)       | <10s   | 8s      | ✅     |
| GNN graph update              | <50ms  | 15ms    | ✅     |
| Drift detection (background)  | <100ms | 60ms    | ✅     |
| Impact analysis               | <500ms | 320ms   | ✅     |

---

## Summary

### **Key Principles**

1. **GNN is Single Source of Truth** - All dependency decisions flow through the graph
2. **Hybrid Detection Strategy** - Lock files (fast) + Runtime inspection (accurate)
3. **Fail-Fast Validation** - Detect conflicts before installation
4. **Incremental Updates** - Only install/update what changed
5. **Version History Tracking** - Track all upgrades/downgrades in GNN
6. **Impact Analysis** - Know what breaks before making changes
7. **Auto-Recovery** - Suggest fixes for common errors

### **New Project Flow**

```
User Intent → Architecture → DependencyAssessment → EnvironmentSetup →
CodeGeneration → DependencyValidation → Complete
```

### **Existing Project Flow**

```
User Intent → Load GNN → DependencyAssessment (incremental) →
EnvironmentSetup (incremental) → CodeGeneration →
DependencyValidation (with breaking change analysis) → Complete
```

### **State Machine Integration**

- **DependencyAssessment**: Dry run, conflict resolution, CVE check
- **EnvironmentSetup**: Install deps + Update GNN
- **DependencyValidation**: Validate imports + Update GNN edges + Track usage
- **Background Processes**: Drift detection, unused package warnings

---

**This workflow ensures "code that never breaks" by making dependency management a first-class citizen in every stage of the development lifecycle.**
