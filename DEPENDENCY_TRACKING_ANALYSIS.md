# Dependency Tracking Analysis - All Combinations

**Date:** December 4, 2025  
**Analysis:** Complete inventory of what's implemented vs. what's specified

---

## Quick Summary: What's Tracked vs. What's Missing

| Dependency Type                 | Version-Level?     | Status                 | Implementation       |
| ------------------------------- | ------------------ | ---------------------- | -------------------- |
| **File → File**                 | ❌ No versions     | ✅ **DONE**            | Imports, Uses edges  |
| **File → Function**             | ❌ No versions     | ✅ **DONE**            | Calls, Defines edges |
| **Function → Function**         | ❌ No versions     | ✅ **DONE**            | Calls edges          |
| **Test → Source (file)**        | ❌ No versions     | ✅ **DONE**            | TestDependency edges |
| **Test → Source (function)**    | ❌ No versions     | ✅ **DONE**            | Tests edges          |
| **File → Package**              | ❌ **NO VERSIONS** | ❌ **NOT IMPLEMENTED** | Missing entirely     |
| **Package → Package**           | ❌ **NO VERSIONS** | ❌ **NOT IMPLEMENTED** | Missing entirely     |
| **Function → Package Function** | ❌ **NO VERSIONS** | ❌ **NOT IMPLEMENTED** | Missing entirely     |
| **User → File**                 | N/A                | ❌ **NOT IMPLEMENTED** | No user tracking     |

**CRITICAL GAP:** No version-level tracking anywhere. No package dependency tracking at all.

---

## Part 1: What IS Implemented (Code-to-Code Dependencies)

### 1.1 Current Node Types ✅

```rust
// Location: src-tauri/src/gnn/mod.rs lines 56-62
pub enum NodeType {
    Function,    // ✅ Functions tracked
    Class,       // ✅ Classes tracked
    Variable,    // ✅ Variables tracked
    Import,      // ✅ Import statements tracked
    Module,      // ✅ Modules/files tracked
}
```

**Missing Node Types:**

- ❌ `Package` (e.g., numpy==1.26.0)
- ❌ `PackageFunction` (e.g., numpy.array)
- ❌ `User` (for user activity tracking)
- ❌ `ExternalAPI` (for API dependencies)

### 1.2 Current Edge Types ✅

```rust
// Location: src-tauri/src/gnn/mod.rs lines 78-87
pub enum EdgeType {
    Calls,           // ✅ Function → Function calls
    Uses,            // ✅ Generic usage (variable, etc.)
    Imports,         // ✅ File → File imports
    Inherits,        // ✅ Class → Class inheritance
    Defines,         // ✅ File defines Function/Class
    Tests,           // ✅ Test function → Source function
    TestDependency,  // ✅ Test file → Source file
}
```

**Missing Edge Types:**

- ❌ `UsesPackage` (File → Package)
- ❌ `Requires` (Package → Package dependency)
- ❌ `ConflictsWith` (Package → Package conflict)
- ❌ `ModifiedBy` (User → File edit tracking)
- ❌ `CallsAPI` (Function → External API)

### 1.3 What Works: Internal Code Dependencies ✅

#### File → File (Imports)

```rust
// Example: calculator.py imports numpy
CodeEdge {
    edge_type: EdgeType::Imports,
    source_id: "file:calculator.py",
    target_id: "file:utils.py",
}
```

**Status:** ✅ Fully working  
**Version tracking:** ❌ No  
**Evidence:** parser.rs line 173 creates Import nodes

#### Function → Function (Calls)

```rust
// Example: main() calls calculate()
CodeEdge {
    edge_type: EdgeType::Calls,
    source_id: "func:main",
    target_id: "func:calculate",
}
```

**Status:** ✅ Fully working  
**Version tracking:** ❌ No  
**Evidence:** All 11 parsers create Call edges

#### Test → Source (Coverage)

```rust
// Example: test_add() tests add()
CodeEdge {
    edge_type: EdgeType::Tests,
    source_id: "func:test_add",
    target_id: "func:add",
}
```

**Status:** ✅ Fully working  
**Version tracking:** ❌ No  
**Evidence:** mod.rs lines 386-428 `create_test_edges()`

---

## Part 2: What's NOT Implemented (Package Dependencies)

### 2.1 File → Package ❌

**What Should Exist:**

```rust
// Example: calculator.py uses numpy==1.26.0
CodeEdge {
    edge_type: EdgeType::UsesPackage,
    source_id: "file:calculator.py",
    target_id: "pkg:numpy:1.26.0",  // VERSION-SPECIFIC!
}
```

**Status:** ❌ **COMPLETELY MISSING**

**What's Missing:**

1. ❌ No package nodes in graph
2. ❌ No import statement parsing for external packages
3. ❌ No version extraction from requirements.txt/package.json
4. ❌ No UsesPackage edge type
5. ❌ Cannot answer: "Which files use numpy?"
6. ❌ Cannot answer: "Which files use numpy 1.24 vs 1.26?"

**Impact:**

- Cannot detect unused packages
- Cannot safely remove packages after deleting files
- Cannot identify which files break after package upgrade
- Cannot generate minimal requirements.txt

### 2.2 Package → Package (Transitive Dependencies) ❌

**What Should Exist:**

```rust
// Example: pandas==2.1.0 requires numpy>=1.24,<2.0
CodeEdge {
    edge_type: EdgeType::Requires,
    source_id: "pkg:pandas:2.1.0",
    target_id: "pkg:numpy:1.26.0",
    metadata: VersionRequirement {
        spec: ">=1.24,<2.0",
        satisfied_by: "1.26.0",
    }
}
```

**Status:** ❌ **COMPLETELY MISSING**

**What's Missing:**

1. ❌ No package-to-package edges
2. ❌ No version requirement tracking
3. ❌ No conflict detection
4. ❌ Cannot answer: "What does pandas depend on?"
5. ❌ Cannot answer: "Will upgrading numpy break pandas?"
6. ❌ Cannot detect circular dependencies

**Impact:**

- Cannot detect version conflicts before installation
- Cannot explain why package installation fails
- Cannot suggest compatible versions
- No transitive dependency awareness

### 2.3 Function → Package Function ❌

**What Should Exist:**

```rust
// Example: calculate() calls numpy.array()
CodeEdge {
    edge_type: EdgeType::Calls,
    source_id: "func:calculate",
    target_id: "pkgfunc:numpy:1.26.0::array",  // Versioned!
}
```

**Status:** ❌ **COMPLETELY MISSING**

**What's Missing:**

1. ❌ No tracking of which package functions are used
2. ❌ Cannot distinguish between `np.array()` vs `np.mean()`
3. ❌ Cannot answer: "Which functions from numpy are actually used?"
4. ❌ Cannot suggest lightweight alternatives
5. ❌ No deprecation warnings (e.g., numpy 1.x → 2.0 breaking changes)

**Impact:**

- Cannot identify minimal API surface
- Cannot detect breaking API changes after upgrade
- Cannot suggest package alternatives
- No granular usage analysis

### 2.4 User → File (Activity Tracking) ❌

**What Should Exist:**

```rust
// Example: user@email.com modified calculator.py
CodeEdge {
    edge_type: EdgeType::ModifiedBy,
    source_id: "user:vivek@example.com",
    target_id: "file:calculator.py",
    metadata: Modification {
        timestamp: SystemTime::now(),
        change_type: "edit",
        lines_added: 10,
        lines_deleted: 5,
    }
}
```

**Status:** ❌ **NOT IMPLEMENTED** (Not in MVP scope)

**What's Missing:**

1. ❌ No user nodes in graph
2. ❌ No activity tracking
3. ❌ Cannot answer: "Who last modified this file?"
4. ❌ Cannot answer: "What files does User X work on?"
5. ❌ No collaboration insights

**Impact:**

- No team collaboration features
- No expertise mapping (who knows what)
- No code ownership tracking
- Cannot suggest reviewers

---

## Part 3: The Critical Gap - Version-Level Tracking

### Current State: Version-Agnostic ❌

**Everything tracked is version-agnostic:**

```rust
// Current CodeNode - NO VERSION INFO
pub struct CodeNode {
    pub id: String,              // "file:calculator.py"
    pub node_type: NodeType,     // Function, Class, etc.
    pub name: String,            // "calculate"
    pub file_path: String,       // "src/calculator.py"
    pub line_start: usize,
    pub line_end: usize,
    // ❌ NO VERSION FIELD
    // ❌ NO PACKAGE REFERENCE
    // ❌ NO TIMESTAMP
}
```

### What's Needed: Version-Aware Nodes

**Specification says (lines 2670-2850):**

```rust
// REQUIRED but NOT IMPLEMENTED
pub enum NodeType {
    Function,
    Class,
    // ... existing types ...

    // ❌ MISSING:
    TechStack(TechStackNode),  // Package with version
}

pub struct TechStackNode {
    pub package_name: String,        // "numpy"
    pub version: String,             // "1.26.0" - EXACT version
    pub language: Language,          // Python, JavaScript, etc.
    pub used_by_files: Vec<PathBuf>,
    pub used_functions: Vec<String>, // ["array", "mean", "std"]
    pub conflicts_with: Vec<(String, String)>,
    pub requires: Vec<PackageRequirement>,
    pub version_history: Vec<VersionChange>,
}
```

**Why This Matters:**

1. **Version Conflicts:** Different files might need different numpy versions

   ```
   file_a.py needs numpy==1.24.0 (legacy code)
   file_b.py needs numpy==1.26.0 (new features)
   → Currently: Cannot detect this conflict
   → With version tracking: Alert before installation fails
   ```

2. **Safe Upgrades:** Know impact before upgrading

   ```
   Query: "Which files use numpy 1.24 specifically?"
   → Currently: Cannot answer
   → With version tracking: List all affected files
   ```

3. **Minimal Dependencies:** Generate exact requirements
   ```
   requirements.txt has: numpy, pandas, scipy, sklearn
   Code actually uses: numpy, pandas
   → Currently: Cannot identify unused packages
   → With version tracking: Auto-generate minimal requirements
   ```

---

## Part 4: Detailed Gap Analysis by Dependency Type

### 4.1 File → Package Dependencies ❌

**Specification Requirements (lines 7065-7400):**

**Detection:**

```rust
// SHOULD parse import statements and extract packages
fn extract_package_imports(file_path: &Path) -> Vec<PackageImport> {
    // Parse: import numpy as np
    // Extract: package="numpy", version from requirements.txt
    // Return: PackageImport { package: "numpy", version: "1.26.0", line: 5 }
}
```

**Node Creation:**

```rust
// SHOULD create package nodes
fn add_package_node(&mut self, package: &str, version: &str) -> NodeIndex {
    let node_id = format!("pkg:{}:{}", package, version);
    // Create: NodeType::TechStack(TechStackNode { ... })
}
```

**Edge Creation:**

```rust
// SHOULD create UsesPackage edges
fn create_package_edges(&mut self) -> usize {
    // For each file: extract_package_imports()
    // Create edge: file → package@version
}
```

**Queries SHOULD Support:**

```rust
// Which files use numpy?
get_files_using_package("numpy")

// Which files use numpy 1.24 specifically?
get_files_using_package_version("numpy", "1.24.0")

// What packages does calculator.py use?
get_packages_used_by_file("calculator.py")

// Can I safely remove pandas?
is_package_unused("pandas")
```

**Current Reality:**

- ❌ `extract_package_imports()` - NOT IMPLEMENTED
- ❌ `add_package_node()` - NOT IMPLEMENTED
- ❌ `create_package_edges()` - NOT IMPLEMENTED
- ❌ All queries - NOT IMPLEMENTED

### 4.2 Package → Package Dependencies ❌

**Specification Requirements (lines 2710-2780):**

**What's Needed:**

```rust
// Track package dependencies from package metadata
pub struct PackageRequirement {
    pub package: String,           // "numpy"
    pub version_spec: String,      // ">=1.24,<2.0"
    pub optional: bool,
}

// Create Requires edges
CodeEdge {
    edge_type: EdgeType::Requires,
    source_id: "pkg:pandas:2.1.0",
    target_id: "pkg:numpy:1.26.0",
}

// Create ConflictsWith edges
CodeEdge {
    edge_type: EdgeType::ConflictsWith,
    source_id: "pkg:tensorflow:2.14",
    target_id: "pkg:numpy:2.0.0",  // TF doesn't support numpy 2.x yet
}
```

**Queries SHOULD Support:**

```rust
// What does pandas depend on?
get_package_dependencies("pandas", "2.1.0")

// Will upgrading numpy break anything?
check_upgrade_impact("numpy", "1.24.0" → "2.0.0")

// Find compatible versions
find_compatible_versions(&["pandas==2.1.0", "numpy==?"])

// Detect circular dependencies
detect_circular_dependencies()
```

**Current Reality:**

- ❌ PackageRequirement struct - NOT DEFINED
- ❌ EdgeType::Requires - NOT IMPLEMENTED
- ❌ EdgeType::ConflictsWith - NOT IMPLEMENTED
- ❌ All queries - NOT IMPLEMENTED

### 4.3 Function → Package Function ❌

**Specification Says:**

```rust
// Track specific function usage within packages
// Example: calculate() uses numpy.array(), not numpy.mean()

// Create versioned package function nodes
let node_id = "pkgfunc:numpy:1.26.0::array";

// Track which functions are actually used
pub struct TechStackNode {
    pub used_functions: Vec<String>,  // ["array", "zeros", "dot"]
}

// Query: What numpy functions does calculator.py use?
get_package_functions_used_by_file("calculator.py", "numpy")
// → ["array", "mean", "std"]  (not the entire numpy API)
```

**Why This Matters:**

- Identify minimal API surface for lightweight alternatives
- Detect breaking API changes: numpy 1.x `np.sum(axis=0)` vs 2.x different default
- Suggest micro-libraries instead of heavy packages

**Current Reality:**

- ❌ No package function tracking
- ❌ Cannot distinguish between different numpy functions
- ❌ Cannot suggest alternatives like `tinynumpy` if only using basic functions

### 4.4 User → File (Out of MVP Scope, but Documented)

**Not Implemented** - This is for team collaboration features (post-MVP)

Would track:

- Who modified which files
- Ownership and expertise
- Review suggestions
- Activity patterns

---

## Part 5: Implementation Roadmap to Close Gaps

### Priority 1: File → Package (BLOCKER for MVP)

**Required Implementation:**

1. **Extend NodeType enum:**

   ```rust
   pub enum NodeType {
       Function,
       Class,
       // ... existing ...
       Package(TechStackNode),  // NEW
   }
   ```

2. **Add to parser.rs:**

   ```rust
   fn extract_package_imports(file: &Path) -> Vec<(String, String)> {
       // Parse import numpy, import pandas
       // Look up versions in requirements.txt
       // Return: [("numpy", "1.26.0"), ("pandas", "2.1.0")]
   }
   ```

3. **Parse requirements.txt:**

   ```rust
   fn parse_requirements_txt(path: &Path) -> HashMap<String, String> {
       // Read requirements.txt
       // Parse: numpy==1.26.0
       // Return: {"numpy": "1.26.0", ...}
   }
   ```

4. **Create package nodes during graph build:**

   ```rust
   pub fn build_graph(&mut self, project_path: &Path) -> Result<(), String> {
       // 1. Parse all source files (existing)
       // 2. Parse requirements.txt (NEW)
       // 3. Create package nodes (NEW)
       // 4. Create UsesPackage edges (NEW)
   }
   ```

5. **Add query methods:**
   ```rust
   pub fn get_files_using_package(&self, package: &str) -> Vec<PathBuf>;
   pub fn get_packages_used_by_file(&self, file: &Path) -> Vec<String>;
   pub fn is_package_unused(&self, package: &str) -> bool;
   ```

**Estimated Effort:** ~300 lines of code

### Priority 2: Package → Package

**Required Implementation:**

1. **Parse package metadata:**

   ```rust
   // From pip show pandas or package-lock.json
   fn get_package_dependencies(pkg: &str, version: &str) -> Vec<PackageRequirement>
   ```

2. **Create Requires edges:**

   ```rust
   // pandas requires numpy>=1.24,<2.0
   ```

3. **Conflict detection:**
   ```rust
   pub fn check_version_conflicts(&self) -> Vec<Conflict>;
   ```

**Estimated Effort:** ~400 lines of code

### Priority 3: Function → Package Function

**Required Implementation:**

1. **Track import aliases:**

   ```rust
   // import numpy as np
   // Store: {"np": "numpy:1.26.0"}
   ```

2. **Parse function calls:**

   ```rust
   // np.array() → "numpy:1.26.0::array"
   ```

3. **Create CallsPackageFunc edges**

**Estimated Effort:** ~200 lines of code

---

## Part 6: Summary - What You Asked For vs. What Exists

### Your Question: "Are we tracking all dependencies... to the version level?"

**Answer:** ❌ **NO - Major Gaps Exist**

| What You Asked    | Status     | Details                                                  |
| ----------------- | ---------- | -------------------------------------------------------- |
| Package → Package | ❌ **NO**  | No package nodes, no Requires edges, no version tracking |
| File → Package    | ❌ **NO**  | Imports parsed but external packages not tracked         |
| File → File       | ✅ **YES** | But no versions (files don't have versions)              |
| User → File       | ❌ **NO**  | No user tracking (out of MVP scope)                      |
| **VERSION LEVEL** | ❌ **NO**  | No version tracking anywhere in codebase                 |

### Critical Findings:

1. **GNN tracks internal code perfectly** (function → function, file → file, test → source)
2. **GNN does NOT track external dependencies** (packages)
3. **NO version-level tracking exists** (neither for code nor packages)
4. **Specifications are detailed** (997 lines for package tracking) but **implementation is 0%**

### What Works:

- ✅ 7,134 lines of GNN code for internal dependencies
- ✅ 11 language parsers
- ✅ Test coverage tracking
- ✅ Semantic search
- ✅ Query interface

### What's Missing:

- ❌ 0 lines of package dependency tracking
- ❌ No version-level tracking
- ❌ No package nodes in graph
- ❌ No UsesPackage edges
- ❌ ~900 lines of code needed to implement

---

## Recommendation

**Implement Package Tracking as Priority P0:**

The Specifications clearly mark this as "**BLOCKER**" (line 2670). Without it:

- Cannot answer: "Can I safely remove this package?"
- Cannot answer: "What will break if I upgrade numpy?"
- Cannot generate minimal requirements.txt
- Cannot detect version conflicts before they fail

**Next Steps:**

1. Implement File → Package (Priority 1) - ~300 lines
2. Implement Package → Package (Priority 2) - ~400 lines
3. Add version tracking to all nodes (architectural change)
4. Implement query APIs for package dependencies

**Total Estimated Effort:** ~900 lines of Rust code + 200 lines of tests

---

**Prepared by:** AI Assistant  
**Date:** December 4, 2025  
**Based on:** Code inspection (7,134 lines) + Specifications.md (lines 2670-2850, 7065-7400)
