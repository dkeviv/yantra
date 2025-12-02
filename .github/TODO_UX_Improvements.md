# Yantra - UX Improvements & Bug Fixes TODO

**Created:** November 29, 2025  
**Total Tasks:** 24  
**Priority:** HIGH (fixes + UX enhancements)

---

## 🚨 CRITICAL FIXES (Priority 1 - Do First)

### 1. Architecture View - Fix Send Trait Issues ⚠️ BLOCKING
**Tasks:** #1, #2  
**Status:** Not Started  
**Blocks:** All AI architecture features

**Problem:**
6 async Tauri commands have Send trait issues because they hold `MutexGuard` across await points:
- `generate_architecture_from_intent`
- `generate_architecture_from_code`
- `initialize_new_project`
- `initialize_existing_project`
- `review_existing_code`
- `analyze_requirement_impact`

**Solution:**
1. **Task 1:** Fix Send trait issues
   - Drop `MutexGuard` before await points
   - Restructure code to avoid holding locks across awaits
   - Clone necessary data before dropping guard
   - Files: `src-tauri/src/architecture/commands.rs`
   
2. **Task 2:** Re-enable commented commands
   - Uncomment the 2 disabled async commands in main.rs
   - Ensure proper error handling and trait bounds
   - Test all 6 commands work correctly
   - Files: `src-tauri/src/main.rs`, `src-tauri/src/architecture/commands.rs`

**Acceptance Criteria:**
- All 6 async commands compile without errors
- Commands are registered in Tauri invoke_handler
- Manual testing shows commands execute successfully
- No MutexGuard held across await points

---

## 🖥️ TERMINAL MANAGEMENT (Priority 2 - Infrastructure)

### 2. Smart Terminal Management
**Tasks:** #3, #4  
**Status:** Not Started  
**Estimated Time:** 4-6 hours

**Requirements:**

#### Task 3: Process Detection
- Detect if a process is running in foreground in a terminal
- Check process state before executing new commands
- Platform-specific implementations:
  - **macOS:** Use `ps` or `lsof` commands
  - **Linux:** Check `/proc` filesystem
  - **Windows:** Use `tasklist` or WMI queries
- Files: `src-tauri/src/terminal/executor.rs` or new module
- Return: `bool` indicating if terminal is busy

#### Task 4: Terminal Reuse Logic
- Smart terminal management: Check if idle before creating new
- Reuse existing idle terminals for new commands
- Track terminal states: `HashMap<terminal_id, TerminalState>`
- States: `Idle`, `Busy`, `Closed`
- Files: `src-tauri/src/terminal/executor.rs`

**Implementation Plan:**
```rust
// New data structure
struct TerminalState {
    id: String,
    status: TerminalStatus,
    last_command: String,
    created_at: SystemTime,
}

enum TerminalStatus {
    Idle,
    Busy,
    Closed,
}

// New methods
fn is_terminal_busy(terminal_id: &str) -> Result<bool, String>
fn get_idle_terminal() -> Option<String>
fn reuse_or_create_terminal() -> String
```

**Acceptance Criteria:**
- Process detection works on all platforms
- Idle terminals are reused instead of creating new ones
- Terminal state tracked accurately
- Reduced terminal creation overhead

---

## 🎨 THEME SYSTEM (Priority 3 - Visual)

### 3. Dark Blue & Bright White Themes
**Tasks:** #5, #6, #7, #8  
**Status:** Not Started  
**Estimated Time:** 6-8 hours

**Requirements:**

#### Task 5: Dark Blue Theme Design
- Primary background: Dark blue (#0B1437 or similar navy)
- Secondary background: Lighter blue (#1A2849)
- Text primary: White/off-white (#E5E7EB)
- Text secondary: Light gray (#9CA3AF)
- Accent: Bright blue (#3B82F6)
- Border: Dark blue-gray (#1F2937)
- Files: `src-ui/index.css` or new `theme.css`

#### Task 6: Bright White Theme Design
- Primary background: White (#FFFFFF)
- Secondary background: Light gray (#F3F4F6)
- Text primary: Black/dark gray (#1A1A1A)
- Text secondary: Medium gray (#6B7280)
- Accent: Blue (#2563EB)
- Border: Light gray (#E5E7EB)
- Ensure WCAG AA contrast: 4.5:1 minimum

#### Task 7: Theme Toggle Component
- Small toggle button in Yantra title section (top-left)
- Visual: Moon icon (dark) / Sun icon (light) or toggle switch
- Click to switch between dark blue and bright white
- Store preference: `localStorage.setItem('theme', 'dark' | 'light')`
- Files: New `src-ui/components/ThemeToggle.tsx`, update `App.tsx`

#### Task 8: Apply Theme to All Components
- Use CSS variables for dynamic theming
- Apply to: ChatPanel, FileExplorer, Editor, LLMSettings, all panels
- Test all UI elements in both themes
- Ensure no color hardcoding
- Files: `src-ui/index.css`, all component `.tsx` files

**Color Palette Reference:**

| Element | Dark Blue Theme | Bright White Theme |
|---------|----------------|-------------------|
| Primary BG | #0B1437 | #FFFFFF |
| Secondary BG | #1A2849 | #F3F4F6 |
| Text Primary | #E5E7EB | #1A1A1A |
| Text Secondary | #9CA3AF | #6B7280 |
| Accent | #3B82F6 | #2563EB |
| Border | #1F2937 | #E5E7EB |
| Success | #10B981 | #059669 |
| Error | #EF4444 | #DC2626 |
| Warning | #F59E0B | #D97706 |

**Acceptance Criteria:**
- Toggle switches themes instantly (no reload)
- All components styled correctly in both themes
- Theme preference persists across sessions
- No visual glitches or color conflicts

---

## 🤖 AGENT PANEL & STATUS (Priority 4 - Core UX)

### 4. Agent Panel Rename & Status System
**Tasks:** #9, #10, #11, #12, #13, #14, #15, #16, #17  
**Status:** Not Started  
**Estimated Time:** 10-12 hours

**Requirements:**

#### Phase 1: Rename Chat to Agent (Tasks 9-10)
- **Task 9:** Replace all "Chat" with "Agent"
  - Component: `ChatPanel.tsx` → `AgentPanel.tsx` (optional)
  - UI labels, titles, tooltips, comments
  - Files: `src-ui/components/ChatPanel.tsx`, `src-ui/App.tsx`
  
- **Task 10:** Remove "Describe what you want" placeholder
  - Replace with empty or better text: "Type your task here..."
  - Files: Agent input field

#### Phase 2: Status Indicator (Task 11)
- Small icon near "Agent" label (16-20px)
- Two states:
  - **Running:** Animated spinner/pulse (blue)
  - **Idle:** Static checkmark/circle (green)
- Clear visual distinction
- Files: New `src-ui/components/StatusIndicator.tsx`

#### Phase 3: Task Queue Backend (Tasks 12-14)
- **Task 12:** Backend system
  - Data structure: Queue with task objects
  - Fields: `id`, `description`, `status` (pending/in-progress/completed)
  - Persistence: Save to `task_queue.json`
  - Files: New `src-tauri/src/agent/task_queue.rs`
  
- **Task 13:** Tauri commands
  - `get_task_queue() -> Vec<Task>`
  - `get_current_task() -> Option<Task>`
  - `add_task(description: String) -> Task`
  - `update_task_status(id: String, status: TaskStatus)`
  - `complete_task(id: String)`
  - Register in main.rs invoke_handler
  
- **Task 14:** Integrate with Plan items
  - Align with documentation extractor's plan items
  - Bidirectional sync: Plan ↔ Task queue
  - Files: `src-tauri/src/agent/task_queue.rs`, `src-tauri/src/documentation/extractor.rs`

#### Phase 4: Task Panel UI (Tasks 15-17)
- **Task 15:** Create panel component
  - Overlay panel (not fixed in layout)
  - Appears below/beside status indicator
  - Lists: Current task (highlighted) + Upcoming tasks
  - Status badges: Pending/In Progress/Done
  - Files: New `src-ui/components/TaskPanel.tsx`
  
- **Task 16:** Expand/Collapse logic
  - Click status indicator → Panel expands
  - Click anywhere else → Panel collapses (click-away listener)
  - Smooth animations (CSS transitions)
  - Z-index management
  - Files: `StatusIndicator.tsx`, `TaskPanel.tsx`
  
- **Task 17:** Connect to backend
  - Fetch task queue via Tauri commands
  - Real-time updates: Poll every 2-3 seconds OR use Tauri events
  - Display current + upcoming tasks
  - Files: `TaskPanel.tsx`, new `src-ui/api/tasks.ts`

**Data Structures:**
```rust
// Backend (Rust)
struct Task {
    id: String,
    description: String,
    status: TaskStatus,
    created_at: SystemTime,
    started_at: Option<SystemTime>,
    completed_at: Option<SystemTime>,
}

enum TaskStatus {
    Pending,
    InProgress,
    Completed,
}
```

```typescript
// Frontend (TypeScript)
interface Task {
  id: string;
  description: string;
  status: 'pending' | 'in-progress' | 'completed';
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}
```

**Acceptance Criteria:**
- "Agent" replaces "Chat" everywhere
- Status indicator shows correct state (running/idle)
- Clicking status opens task panel
- Task panel shows current + upcoming tasks
- Tasks persist across app restarts
- Tasks aligned with Plan items
- Panel collapses on click-away

---

## 📐 PANEL MANAGEMENT (Priority 5 - Layout)

### 5. Panel Expansion & Resizing
**Tasks:** #18, #19, #20, #21, #22  
**Status:** Not Started  
**Estimated Time:** 6-8 hours

**Requirements:**

#### Tasks 18-20: Panel Expand Toggles
- Add expand/maximize button to each panel header:
  - **Task 18:** File Explorer
  - **Task 19:** Agent (Chat) Panel
  - **Task 20:** Editor Panel

- **Behavior:**
  - Click expand → Panel fills entire window (other panels hidden)
  - Click again → Restore original layout
  - Toggle icon: Expand arrows (⤢) / Compress arrows (⤡)
  - Files: Individual panel components + `App.tsx`

#### Task 21: Shared Expansion Logic
- Create reusable hook: `usePanelExpansion.ts` or store: `layoutStore.ts`
- Only one panel expanded at a time
- Smooth CSS transitions (300ms)
- State: Track which panel is expanded (`null | 'fileExplorer' | 'agent' | 'editor'`)
- Files: New `src-ui/hooks/usePanelExpansion.ts` or `src-ui/stores/layoutStore.ts`

#### Task 22: File Explorer Width Adjustment
- Add vertical drag handle between File Explorer and Agent panel
- Drag left/right to resize File Explorer width
- Similar to existing Chat panel width adjustment
- Constraints: Min 200px, Max 500px
- Store width in localStorage
- Files: `src-ui/App.tsx` or new `src-ui/components/ResizablePanels.tsx`

**Implementation Example:**
```typescript
// usePanelExpansion.ts
export function usePanelExpansion() {
  const [expandedPanel, setExpandedPanel] = createSignal<string | null>(null);
  
  const toggleExpand = (panelId: string) => {
    setExpandedPanel(expandedPanel() === panelId ? null : panelId);
  };
  
  const isExpanded = (panelId: string) => expandedPanel() === panelId;
  
  return { expandedPanel, toggleExpand, isExpanded };
}
```

**Acceptance Criteria:**
- Each panel has expand toggle button
- Only one panel expands at a time
- Smooth expand/collapse animations
- File Explorer width adjustable via drag
- Divider behaves like Chat panel divider
- Layout doesn't break on resize

---

## ✅ TESTING & DOCUMENTATION (Priority 6 - Final)

### 6. Integration Testing & Documentation
**Tasks:** #23, #24  
**Status:** Not Started  
**Estimated Time:** 4-6 hours

#### Task 23: End-to-End Testing
Test all new features together:

1. **Theme System:**
   - Toggle between dark blue and bright white
   - Verify all components styled correctly
   - Check localStorage persistence

2. **Agent Panel:**
   - Verify "Agent" replaces "Chat"
   - Status indicator shows correct state
   - Task panel opens/closes correctly

3. **Task Queue:**
   - Add tasks via backend
   - Verify tasks appear in panel
   - Check persistence across restarts

4. **Panel Management:**
   - Expand each panel to full window
   - Verify only one expands at a time
   - Test File Explorer width adjustment

5. **Cross-Component:**
   - Theme toggle + expanded panel
   - Task panel + expanded panels
   - Resize dividers + theme toggle

#### Task 24: Update Documentation
Update following files:

1. **Features.md:**
   - Add "Theme System" section
   - Add "Agent Status & Task Queue" section
   - Add "Panel Management" section
   - Include use cases and screenshots

2. **UX.md:**
   - Update user flows with new UX
   - Document theme switching workflow
   - Document task queue interaction

3. **Technical_Guide.md:**
   - Add "Theme System Architecture" section
   - Add "Task Queue System" section
   - Add "Panel Management" section
   - Document implementation details

4. **IMPLEMENTATION_STATUS.md:**
   - Mark all completed features as DONE
   - Update percentages
   - Add new feature rows

**Acceptance Criteria:**
- All features work individually
- All features work together without conflicts
- No layout breaks or visual glitches
- All documentation updated and accurate

---

## 📊 TASK SUMMARY

### By Priority

| Priority | Category | Tasks | Estimated Time | Blocks Others |
|----------|----------|-------|----------------|---------------|
| 1 | **Critical Fixes** | #1, #2 | 3-4 hours | YES (AI features) |
| 2 | **Terminal Management** | #3, #4 | 4-6 hours | NO |
| 3 | **Theme System** | #5-8 | 6-8 hours | NO |
| 4 | **Agent & Status** | #9-17 | 10-12 hours | NO |
| 5 | **Panel Management** | #18-22 | 6-8 hours | NO |
| 6 | **Testing & Docs** | #23, #24 | 4-6 hours | NO |
| **TOTAL** | | **24 tasks** | **33-44 hours** | |

### Recommended Sequence

**Week 1 (Focus: Fixes & Infrastructure)**
1. ✅ Tasks 1-2: Fix Architecture Send trait issues (CRITICAL)
2. ✅ Tasks 3-4: Terminal management
3. ✅ Tasks 5-8: Theme system

**Week 2 (Focus: Agent Features)**
4. ✅ Tasks 9-10: Rename Chat to Agent
5. ✅ Tasks 11-14: Status indicator + Task queue backend
6. ✅ Tasks 15-17: Task panel UI

**Week 3 (Focus: Polish & Testing)**
7. ✅ Tasks 18-22: Panel expansion & resizing
8. ✅ Tasks 23-24: Testing & documentation

### Dependencies

```
Task 1 (Fix Send traits) → Task 2 (Re-enable commands)
                         → ALL AI features unblocked

Task 3 (Process detection) → Task 4 (Terminal reuse)

Task 5 (Dark theme) → Task 7 (Toggle)
Task 6 (Light theme) → Task 7 (Toggle) → Task 8 (Apply to all)

Task 9 (Rename) → Task 10 (Placeholder)
Task 11 (Status indicator) → Task 16 (Expand/collapse)
Task 12 (Task queue backend) → Task 13 (Tauri commands)
                             → Task 14 (Plan integration)
                             → Task 17 (Connect UI)
Task 15 (Task panel UI) → Task 16 (Expand/collapse)
                       → Task 17 (Connect backend)

Task 18 (File Explorer expand) → Task 21 (Shared logic)
Task 19 (Agent expand) → Task 21 (Shared logic)
Task 20 (Editor expand) → Task 21 (Shared logic)

Task 1-22 (All features) → Task 23 (Testing)
                         → Task 24 (Documentation)
```

---

## 🎯 ACCEPTANCE CRITERIA CHECKLIST

### Critical Fixes
- [ ] All 6 architecture async commands compile without Send trait errors
- [ ] Commands are registered and callable from frontend
- [ ] Manual testing shows commands execute successfully

### Terminal Management
- [ ] Process detection works on macOS, Linux, Windows
- [ ] Idle terminals are reused before creating new ones
- [ ] Terminal state tracked accurately (Idle/Busy/Closed)

### Theme System
- [ ] Dark blue theme looks professional and consistent
- [ ] Bright white theme meets WCAG AA contrast standards
- [ ] Toggle switches themes instantly without reload
- [ ] Theme preference persists across app restarts
- [ ] All components styled correctly in both themes

### Agent & Status
- [ ] "Agent" replaces "Chat" everywhere in UI
- [ ] Status indicator shows correct state (running/idle with clear icons)
- [ ] Task panel opens on status click, closes on click-away
- [ ] Tasks displayed: Current (highlighted) + Upcoming (list)
- [ ] Tasks persist across app restarts
- [ ] Tasks aligned with Plan items from documentation

### Panel Management
- [ ] Each panel (File Explorer, Agent, Editor) has expand button
- [ ] Only one panel expands at a time
- [ ] Expand/collapse animations smooth (300ms transition)
- [ ] File Explorer width adjustable via drag (200-500px range)
- [ ] Layout doesn't break on any resize operation

### Testing & Documentation
- [ ] All features tested individually
- [ ] All features tested together (no conflicts)
- [ ] No layout breaks, visual glitches, or Z-index issues
- [ ] Features.md updated with new sections
- [ ] UX.md updated with new workflows
- [ ] Technical_Guide.md updated with implementation details
- [ ] IMPLEMENTATION_STATUS.md updated with completed features

---

## 📝 NOTES

### Code Quality Standards
- Follow Rust best practices (no panics, proper error handling)
- Use SolidJS reactive patterns (createSignal, createEffect)
- TypeScript strict mode (all types defined)
- Accessibility: Keyboard navigation, ARIA labels, focus management
- Performance: Debounce expensive operations, lazy load when possible

### Testing Strategy
- Unit tests for backend logic (task queue, terminal detection)
- Integration tests for Tauri commands
- Manual testing for UI/UX features
- Cross-browser testing (Chrome, Firefox, Safari)
- Platform testing (macOS, Linux, Windows)

### Documentation Updates
- Update after each feature completion (not batch at end)
- Include code examples where helpful
- Add screenshots for visual features
- Update File_Registry.md with new files

---

**Last Updated:** November 29, 2025  
**Tracking:** This TODO list is synced with Copilot's manage_todo_list  
**Status:** All tasks in "not-started" state, ready for implementation
