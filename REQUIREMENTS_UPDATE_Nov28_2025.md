# Requirements Update - November 28, 2025

**Purpose:** Document new requirements and changes requested by user  
**Status:** Pending implementation in Specifications.md and IMPLEMENTATION_STATUS.md  
**Priority:** MVP - High Priority

---

## 1. Technical Guide Clarity ✅ IMPLEMENTED

**Requirement:** Technical_Guide.md should only contain current implementation, not historical information.

**Changes Made:**
- ✅ Updated `.github/copilot-instructions.md` with explicit requirement
- ✅ Updated `Technical_Guide.md` header to clarify purpose
- ✅ Removed "What's Complete" milestone section with dates

**Key Points:**
- Document ONLY current implementation
- Must always reflect code as currently implemented
- Remove any deprecated/historical sections immediately when code changes
- Historical information tracked in git history and Decision_Log.md

---

## 2. Explicit Feature Extraction with Confirmation

**Requirement:** Make feature extraction explicit and confirmable.

### Workflow

**After user describes intent:**
1. Yantra responds with structured summary:
   ```
   "I understand you want:
   (1) User authentication with JWT
   (2) Password reset flow via email
   (3) Rate limiting on login attempts
   
   Is this correct? (yes/no/modify)"
   ```

2. User confirms or clarifies
3. Confirmed summary becomes SSOT entry in Features.md

### Bidirectional Linking

**Requirements → Code:**
- Each extracted feature links to implementing files
- Track: `Feature → [file1.py, file2.py, ...]`

**Code → Requirements:**
- Each generated file links back to spawning requirement
- Track: `file.py → [Feature1, Feature3, ...]`

### Requirements-Based Validation

**Test Reporting Format:**
```
"Requirement 1 (JWT auth): 4/4 tests passing ✅
 Requirement 2 (Password reset): 2/3 tests passing ⚠️
   - Failed test: test_email_sending
 Requirement 3 (Rate limiting): 6/6 tests passing ✅"
```

**Implementation Files:**
- `src-tauri/src/documentation/feature_tracker.rs` - Feature tracking with bidirectional links
- `src-tauri/src/testing/requirements_validator.rs` - Map test results to requirements
- `src-ui/components/FeatureConfirmation.tsx` - UI for feature confirmation

**Priority:** HIGH - MVP Feature

---

## 3. Remove Dependencies Tab, Add Tech Stack Tab

### Current State (TO BE REMOVED)
- Dependencies tab shows code dependencies graph
- Has "Reset" and "Export" functions (to be removed)

### New Requirements

#### 3.1 Remove Reset and Export Functions
- ❌ Remove "Reset" function from Dependencies tab
- ❌ Remove "Export" function from Dependencies tab

#### 3.2 Create Tech Stack Tab

**Purpose:** Show all tool dependencies with versions and their relationships

**Tab Name:** "Tech Stack" (replaces "Dependencies")

**Content Structure:**
```
┌────────────────────────────────────────────────┐
│ 🛠️ Tech Stack                                  │
├────────────────────────────────────────────────┤
│                                                │
│ Frontend Dependencies:                         │
│ ├─ SolidJS 1.8.7              ✅ Installed    │
│ │  └─ Dependencies:                           │
│ │      ├─ solid-js-router 0.8.4              │
│ │      └─ vite 4.4.9                         │
│ │                                             │
│ ├─ TailwindCSS 3.3.5          ✅ Installed    │
│ └─ Monaco Editor 0.44.0       ✅ Installed    │
│                                                │
│ Backend Dependencies (Rust):                   │
│ ├─ Tauri 1.5.4                ✅ Installed    │
│ │  └─ Dependencies:                           │
│ │      ├─ tokio 1.35.0                       │
│ │      ├─ serde 1.0.193                      │
│ │      └─ reqwest 0.11.23                    │
│ │                                             │
│ ├─ petgraph 0.6.4             ✅ Installed    │
│ └─ tree-sitter 0.20.10        ✅ Installed    │
│                                                │
│ Python Dependencies (AI/ML):                   │
│ ├─ PyTorch 2.1.0              ✅ Installed    │
│ │  └─ Dependencies:                           │
│ │      ├─ numpy 1.26.2                       │
│ │      └─ cuda-toolkit 12.1 (optional)       │
│ │                                             │
│ ├─ tree-sitter 0.20.4         ✅ Installed    │
│ └─ pytest 7.4.3               ✅ Installed    │
│                                                │
│ LLM APIs:                                      │
│ ├─ Anthropic Claude API       ⚙️ Configured   │
│ └─ OpenAI API                 ⚙️ Configured   │
│                                                │
│ [Search Tools]  [Refresh]                     │
└────────────────────────────────────────────────┘
```

**Features:**
1. **Hierarchical Display:**
   - Group by category (Frontend, Backend, Python, APIs)
   - Show version numbers
   - Show installation status (✅ Installed, ⚠️ Outdated, ❌ Missing)

2. **Dependency Graph:**
   - If Tool A requires Tool B, show indented under A
   - Example: `SolidJS → solid-js-router, vite`
   - Visual tree structure with └─ and ├─ characters

3. **Searchable:**
   - Search box: "Search tools..."
   - Filter by name, version, or category
   - Highlight matching results

4. **Actionable:**
   - Click tool name → Show details (changelog, documentation link)
   - "Refresh" button → Re-scan dependencies
   - No "Reset" or "Export" (removed per requirements)

**Data Source:**
- Parse `package.json` for npm dependencies
- Parse `Cargo.toml` for Rust dependencies
- Parse `requirements.txt` / `pyproject.toml` for Python
- Query LLM config for API status

**Implementation Files:**
- `src-tauri/src/dependencies/tech_stack.rs` - Backend scanner
- `src-ui/components/TechStackPanel.tsx` - UI component
- Update `App.tsx` view tabs: "Dependencies" → "Tech Stack"

**Priority:** HIGH - MVP Feature

---

## 4. Enhanced Dependencies Graph (Code Dependencies)

**Context:** The Dependencies tab currently shows code dependency graph. With Tech Stack tab added, this needs enhancement.

### New Requirements for Code Dependencies Graph

#### 4.1 Readable by Default
- Graph should be readable without zooming
- Use appropriate node sizes and spacing
- Auto-layout algorithm (dagre, elk, or similar)
- Default zoom level: fit-to-screen

#### 4.2 Scrollable for Large Graphs
- If graph exceeds viewport, enable scrolling
- Maintain smooth pan and zoom
- Mini-map in corner for navigation (optional)

#### 4.3 Searchable Navigation
**User can search for:**
- Function names: "getUserById"
- Class names: "UserService"
- File names: "auth_service.py"

**Behavior:**
- Search box at top: "Search dependencies..."
- As user types, highlight matching nodes
- Press Enter or click result → Zoom to node
- Show only matching subgraph (filter mode)

**Example:**
```
User types: "auth"
→ Highlights: auth_service.py, auth_middleware.py, authenticate()
→ User clicks auth_service.py
→ Graph zooms to show auth_service.py and its direct dependencies
```

#### 4.4 Agent-Driven Navigation

**User can ask agent in chat:**
```
User: "Show me the dependencies for auth_service.py"
Agent: Opens Dependencies tab
       → Searches for "auth_service.py"
       → Zooms to node
       → Highlights direct dependencies
       
Agent responds: "Here are the dependencies for auth_service.py:
- Depends on: database.py, jwt_utils.py, config.py
- Used by: api_routes.py, middleware.py
[View in Dependencies Tab]"
```

**Implementation:**
- Agent can send command: `navigate_to_dependency(file_name)`
- UI receives command via Tauri event
- Automatically switches to Dependencies tab
- Searches and focuses on requested node

**Chat Commands:**
- "Show dependencies for X"
- "What depends on X?"
- "Find calls to function Y"
- "Show me the graph around X"

**Implementation Files:**
- `src-ui/components/DependencyGraph.tsx` - Enhanced graph component
- `src-ui/stores/graphNavigationStore.ts` - Navigation state management
- `src-tauri/src/agent/graph_navigator.rs` - Agent commands for navigation

**Priority:** HIGH - MVP Feature

---

## 5. UI Tab Structure Update

### Current Tabs (Right Panel)
- Editor
- Dependencies
- Architecture

### Updated Tabs (Right Panel)
- Editor
- **Tech Stack** (new - replaces Dependencies for tool/library tracking)
- **Dependencies** (enhanced - code dependency graph with search)
- Architecture

**Rationale:**
- Tech Stack: Show what tools/libraries are used (npm, cargo, pip packages)
- Dependencies: Show how code files depend on each other (function calls, imports)
- Clear separation of concerns

**Alternative: Keep 3 Tabs**
If space is limited, keep 3 tabs:
- Editor
- Dependencies (with tabs inside: "Code Dependencies" | "Tech Stack")
- Architecture

**User Preference:** To be decided during implementation

---

## 6. Implementation Priority

| Priority | Feature | Effort | MVP |
|----------|---------|--------|-----|
| 1 | Explicit Feature Extraction | 3 days | ✅ Yes |
| 2 | Tech Stack Tab | 2 days | ✅ Yes |
| 3 | Enhanced Dependencies Navigation | 3 days | ✅ Yes |
| 4 | Bidirectional Requirements Linking | 2 days | ✅ Yes |
| 5 | Requirements-Based Test Validation | 2 days | ✅ Yes |

**Total Effort:** ~12 days (2.5 weeks)

**Phasing:**
- **Week 1:** Feature Extraction + Tech Stack Tab
- **Week 2:** Enhanced Dependencies Navigation
- **Week 3:** Bidirectional Linking + Requirements Validation

---

## 7. Specifications.md Updates Required

### Sections to Add/Update

1. **Add New Section:** "Explicit Feature Extraction & Requirements Management"
   - Location: After "Documentation System" section (line ~5694)
   - Content: Full specification as outlined in #2 above
   - ~300 lines

2. **Update Section:** "Architecture View System"
   - Add subsection: "Tech Stack Tab"
   - Add subsection: "Enhanced Dependencies Navigation"
   - Location: Update existing Architecture View section (lines 4224-5150)
   - ~150 lines of additions

3. **Update Section:** "UI Design" (if exists)
   - Update tab structure: Editor | Tech Stack | Dependencies | Architecture
   - Or: Editor | Dependencies (with sub-tabs) | Architecture

---

## 8. IMPLEMENTATION_STATUS.md Updates Required

### Add New MVP Features

**Section 12: UI/Frontend (Basic + Minimal UI)** - Update with:

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 11.5 | Tech Stack Tab | 🔴 TODO | `src-ui/components/TechStackPanel.tsx` | - | Tool dependencies with versions |
| 11.6 | Enhanced Dependencies Navigation | 🔴 TODO | `src-ui/components/DependencyGraph.tsx` | - | Search + agent navigation |

**New Section 13: Feature Extraction & Requirements** - Add:

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 13.1 | Explicit Feature Extraction | 🔴 TODO | `src-tauri/src/documentation/feature_tracker.rs` | - | Confirmation workflow |
| 13.2 | Bidirectional Linking | 🔴 TODO | Feature tracker + GNN integration | - | Features ↔ Code |
| 13.3 | Requirements-Based Validation | 🔴 TODO | `src-tauri/src/testing/requirements_validator.rs` | - | Test results per requirement |
| 13.4 | Feature Confirmation UI | 🔴 TODO | `src-ui/components/FeatureConfirmation.tsx` | - | Approval interface |

**Update Totals:**
- MVP Features: 52/93 → 52/99 (56% → 52%)
- Add 6 new MVP features
- All marked as 🔴 TODO, HIGH priority

---

## 9. Action Items

### For Copilot/Agent:
1. ✅ Update `.github/copilot-instructions.md` - DONE
2. ✅ Update `Technical_Guide.md` header - DONE
3. ⏳ Add "Explicit Feature Extraction" section to Specifications.md
4. ⏳ Add "Tech Stack Tab" requirements to Specifications.md
5. ⏳ Add "Enhanced Dependencies" requirements to Specifications.md
6. ⏳ Update IMPLEMENTATION_STATUS.md with new MVP features
7. ⏳ Update decision log if architectural decisions made

### For Development Team:
1. Review this document and approve requirements
2. Prioritize implementation order
3. Assign tasks to sprints
4. Update project timeline if needed

---

## 10. Success Criteria

**Feature Extraction:**
- ✅ 100% of features explicitly confirmed before implementation
- ✅ <5% feature misinterpretation rate
- ✅ 100% of generated files linked to features

**Tech Stack Tab:**
- ✅ All dependencies visible with versions
- ✅ Dependency graph shows tool relationships
- ✅ Search finds tools in <100ms
- ✅ Status indicators (installed/missing) accurate

**Enhanced Dependencies:**
- ✅ Graph readable by default (no zoom required for <50 nodes)
- ✅ Search finds functions/classes/files in <100ms
- ✅ Agent can navigate to any node via chat command
- ✅ User can scroll/pan/zoom smoothly (60fps)

---

**Last Updated:** November 28, 2025  
**Author:** GitHub Copilot (based on user requirements)  
**Status:** Requirements documented, pending Specifications.md and IMPLEMENTATION_STATUS.md updates
