# Dependency Detection Strategy - Hybrid Approach

**Created:** December 5, 2025  
**Status:** Specification for Implementation

---

## Problem Statement

**Question:** "If no lock file, then manifest file is not that reliable during dev time. So would npm show or pip show needed?"

**Answer:** YES! Manifest files with version ranges (`>=1.20.0`, `^2.0.0`) don't tell us what's ACTUALLY installed.

---

## The Three Sources of Truth

### 1. **Lock Files** (Production Truth)

```json
// package-lock.json
{
  "numpy": {
    "version": "1.26.3", // ✅ EXACT version
    "resolved": "https://...",
    "integrity": "sha512-..."
  }
}
```

**Pros:**

- Exact versions
- Committed to repo (reproducible)
- Fast to read (JSON parse)
- Includes transitive dependencies

**Cons:**

- May not exist in dev workflow
- Becomes stale if not updated
- Not present in early project setup

**When to Use:** CI/CD, production builds, team collaboration

---

### 2. **Runtime Inspection** (Development Truth)

```bash
$ pip show numpy
Name: numpy
Version: 1.26.3              # ✅ EXACT version (what's installed NOW)
Location: /venv/lib/python3.11/site-packages

$ npm list numpy --depth=0
numpy@1.26.3                 # ✅ EXACT version (what's installed NOW)
```

**Pros:**

- Always accurate (reflects current environment)
- Includes manually installed packages
- Shows transitive dependencies
- Works during active development

**Cons:**

- Slower (100-500ms per package)
- Requires environment activation (venv, node_modules)
- Not reproducible across machines
- May differ from committed code

**When to Use:** Active development, debugging, "what breaks NOW?"

---

### 3. **Manifest Files** (Intent, Not Truth)

```python
# requirements.txt
numpy>=1.20.0                # ❌ Version RANGE, not exact
pandas^2.0.0                 # ❌ Could be 2.0.0 or 2.1.4
requests*                    # ❌ Could be ANYTHING
```

**Pros:**

- Committed to repo
- Shows developer intent
- Fast to read

**Cons:**

- **NOT EXACT VERSIONS** (ranges only)
- Missing transitive dependencies
- Often stale during development
- Can't detect breaking changes from ">=1.20.0"

**When to Use:** Only as warning ("Please run pip freeze / npm install")

---

## Proposed Hybrid Strategy

### **Decision Tree:**

```
Get Package Version:
│
├─ Lock file exists?
│  ├─ YES: Read from lock file (fast, exact)
│  └─ NO: ↓
│
├─ Virtual environment active?
│  ├─ YES: Use runtime inspection (pip show / npm list)
│  └─ NO: ↓
│
└─ Manifest file exists?
   ├─ YES: Parse manifest, WARN user about ranges
   └─ NO: ERROR - No dependency info available
```

### **Implementation Priority:**

```rust
pub fn get_package_version(package_name: &str, language: Language) -> Result<PackageInfo> {
    // Priority 1: Lock file (if exists and recent)
    if let Some(version) = read_from_lock_file(package_name, language)? {
        return Ok(PackageInfo {
            version,
            source: DependencySource::LockFile,
            exact: true,
        });
    }

    // Priority 2: Runtime inspection (actual installed version)
    if let Some(version) = runtime_inspection(package_name, language)? {
        // WARN user if no lock file
        if !lock_file_exists(language) {
            warn!("No lock file found. Run 'pip freeze > requirements.txt' or 'npm install'");
        }

        return Ok(PackageInfo {
            version,
            source: DependencySource::Runtime,
            exact: true,
        });
    }

    // Priority 3: Manifest file (show range as warning)
    if let Some(range) = read_from_manifest(package_name, language)? {
        return Ok(PackageInfo {
            version: range.clone(),
            source: DependencySource::Manifest,
            exact: false,  // ⚠️ NOT EXACT - show warning in UI
        });
    }

    Err("Package not found in any source")
}
```

---

## Real-World Scenarios

### **Scenario 1: Active Development (No Lock File)**

```
Developer workflow:
1. pip install numpy          ← Manual install (no lock file yet)
2. Write code: import numpy
3. Yantra checks dependencies:
   ├─ Lock file? ❌ None
   ├─ Runtime inspection? ✅ numpy==1.26.3
   └─ Result: "numpy 1.26.3 (from pip show)"

4. Yantra WARNING: "⚠️ No lock file. Run: pip freeze > requirements.txt"
```

**Benefit:** Yantra works immediately, catches breaking changes NOW, nudges toward best practices.

---

### **Scenario 2: Team Collaboration (Lock File Exists)**

```
Team member pulls repo:
1. git pull
2. npm install                ← Generates package-lock.json
3. Yantra checks dependencies:
   ├─ Lock file? ✅ package-lock.json
   ├─ Result: "numpy@1.26.3 (from lock file)"
   └─ Fast: <1ms (JSON read)

4. No warnings (lock file is source of truth)
```

**Benefit:** Fast, reproducible, team-aligned versions.

---

### **Scenario 3: Drift Detection (Lock vs Runtime Mismatch)**

```
Developer manually installs newer version:
1. pip install numpy==1.27.0  ← Manual upgrade
2. Yantra checks dependencies:
   ├─ Lock file: numpy==1.26.3
   ├─ Runtime: numpy==1.27.0
   └─ ⚠️ WARNING: "Version mismatch! Lock: 1.26.3, Installed: 1.27.0"

3. Yantra suggests: "Update lock file: pip freeze > requirements.txt"
```

**Benefit:** Catch version drift before it causes bugs in CI/CD.

---

### **Scenario 4: CI/CD (Lock File Only)**

```
CI pipeline:
1. Checkout code
2. Install from lock: pip install -r requirements.txt
3. Yantra checks dependencies:
   ├─ Lock file? ✅ requirements.txt (with exact versions from pip freeze)
   ├─ Runtime inspection? Skip (CI uses lock file as source)
   └─ Result: Fast, reproducible builds
```

**Benefit:** No runtime inspection overhead in CI (lock file is sufficient).

---

## Performance Comparison

| Method                 | Speed                 | Accuracy                | Use Case                       |
| ---------------------- | --------------------- | ----------------------- | ------------------------------ |
| **Lock File**          | <1ms (JSON read)      | 100% (exact versions)   | Production, CI/CD, team collab |
| **Runtime Inspection** | 100-500ms per package | 100% (actual installed) | Active development, debugging  |
| **Manifest File**      | <1ms (text parse)     | 0% (ranges only)        | Warning/fallback only          |

---

## Updated Specification

### **Metadata Sources (REVISED Priority):**

1. **Lock files first** (package-lock.json, Cargo.lock, poetry.lock)
   - Exact versions, fast, reproducible
   - Use when available and recent

2. **Runtime inspection** (pip show, npm list)
   - Fallback when no lock file
   - Primary during active development
   - Warn user to create lock file

3. **Manifest files** (requirements.txt, package.json)
   - Last resort (version ranges only)
   - Display as warning: "⚠️ Imprecise - version range detected"
   - Prompt user to generate lock file

### **Implementation Files:**

- `src-tauri/src/agent/dependency_manager.rs` (140-220)
  - Already implements runtime inspection ✅
  - Need to add lock file reading
  - Need to add drift detection

- `src-tauri/src/gnn/mod.rs`
  - Add `TechStackNode` with exact versions
  - Track dependency source (lock vs runtime vs manifest)

---

## Breaking Change Detection Examples

### **With Exact Versions (Lock/Runtime):**

```
numpy 1.26.3 → 1.27.0
├─ Check API changes in 1.27.0
├─ Search codebase for deprecated functions
└─ WARN: "numpy.function_x() removed in 1.27.0"
```

### **With Version Ranges (Manifest):**

```
numpy>=1.20.0
├─ ⚠️ Cannot determine breaking changes
├─ Could be 1.20.0 or 1.26.3 or 1.27.0
└─ WARN: "Run pip freeze to get exact version"
```

---

## Recommendation

✅ **Keep current implementation** (`pip show`/`npm list` in dependency_manager.rs)  
✅ **Add lock file reading** as optimization  
✅ **Add drift detection** (lock vs runtime mismatch)  
✅ **Update spec** to reflect hybrid approach (DONE)  
✅ **Add warnings** to nudge users toward lock files

**Result:** Best of both worlds - works during development, promotes best practices, catches drift.

---

## Next Steps

1. ✅ Update specification (Completed December 5, 2025)
2. ⏳ Implement lock file readers:
   - `read_package_lock_json()` for Node.js
   - `read_poetry_lock()` for Python
   - `read_cargo_lock()` for Rust
3. ⏳ Add drift detection:
   - Compare lock file vs runtime versions
   - Warn on mismatch
4. ⏳ Update UI to show dependency source:
   - "numpy 1.26.3 (from lock file)" ✅
   - "numpy 1.26.3 (from pip show)" ⚠️ + warning
   - "numpy >=1.20.0 (from requirements.txt)" ❌ + error

---

**Conclusion:** Runtime inspection (`pip show`/`npm list`) is **essential** during development. Lock files are **ideal** but not always present. The hybrid approach gives us accuracy during development while promoting reproducibility.
