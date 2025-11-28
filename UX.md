# Yantra - User Experience Guide

**Version:** MVP 1.0  
**Last Updated:** November 28, 2025  
**Audience:** End Users and Administrators

---

## Design Philosophy

**Updated:** November 28, 2025

Yantra follows a **Minimal UX** design philosophy focused on maximizing content space and minimizing control overhead:

### Core Principles

1. **Space Optimization** - Every pixel counts
   - Controls take minimal space (top bar, inline settings)
   - Content maximized (chat, editor, terminal take 90%+ screen)
   - No unnecessary panels or toolbars

2. **Single-Line Layouts** - Inline controls where possible
   - LLM settings: provider dropdown + API key + status (one line)
   - Terminal toggle: single button with visual state
   - No dedicated settings windows unless absolutely necessary

3. **Visual Indicators** - Small, clear, unobtrusive
   - Status dots (green/red/yellow, 2px)
   - Pulsing animations for active states
   - Hover tooltips for detailed info

4. **Auto-Save** - Reduce explicit save actions
   - LLM settings auto-save on blur
   - Code auto-saves on edit (debounced)
   - Terminal history persists across sessions

5. **Keyboard-First** - Power users efficiency
   - Cmd+` toggle terminal
   - Cmd+B toggle file tree
   - All major actions have shortcuts

6. **Progressive Disclosure** - Show details on demand
   - API settings collapsed by default
   - Terminal hidden until needed
   - Dependency graph available but not intrusive

### Design Rationale

**Why Minimal UX?**
- Yantra is a development tool, not a GUI app
- Developers need code visibility, not buttons
- Chat is the primary interface (AI-first)
- Screen real estate is precious on laptops
- Faster workflows with fewer clicks

**What This Means:**
- Top bar is 40px (not 60-80px like traditional apps)
- Settings are inline (not separate windows)
- Panels collapse/expand (not always visible)
- One flex row > multiple rows with sections
- Tooltips explain > visible labels everywhere

---

## Getting Started

### Status: ✅ Implemented

### Installation

1. **Download Yantra**
   - Visit yantra.dev (coming soon)
   - Download for your platform (macOS, Windows, Linux)
   - Run the installer

2. **First Launch**
   - Open Yantra application
   - You'll see "YANTRA" in bright white at the top
   - 3-panel interface: File Tree (left), Chat (center), Code Editor (right)
   - Terminal hidden by default (toggle with Cmd+`)

3. **Load Your Project**
   - Click "Open Project Folder" in File Tree
   - Select your project folder
   - Yantra analyzes your codebase (takes 5-30 seconds)
   - File tree populates with your project structure
   - You're ready to start!

---

## Main User Interface (November 28, 2025)

### Top Bar (40px Fixed Height)

**Design:** Minimal space, maximum functionality

```
┌──────────────────────────────────────────────────────────────────┐
│  YANTRA (bright white)              [Terminal: Hide ▼]    [Settings] │
└──────────────────────────────────────────────────────────────────┘
```

**Components:**
- **YANTRA Logo** - Left side, bright white (#FFFFFF), 18px font, bold
- **Terminal Toggle** - Right side, shows "Show" or "Hide" based on state
- **Settings Button** - Far right (future)

**Implementation:**
- Fixed 40px height
- Background: `#1e1e1e` (dark gray)
- Flex layout: justify-between, items-center
- No padding waste, compact design

### 3-Panel Layout

```
┌──────────┬────────────────────────┬──────────────────────────────┐
│          │                        │                              │
│ FILE     │     CHAT PANEL         │     CODE EDITOR             │
│ TREE     │    (Full Height)       │    (Resizable)              │
│          │                        │                              │
│ 📁 src   │  💬 What do you want   │  [file1.py] [file2.py]      │
│  🐍 app  │     to build?          │                              │
│  🐍 util │                        │  def calculate_total():      │
│ 📁 tests │  Provider: [Claude▼]   │    return sum(items)         │
│  🐍 test │  API Key: [••••••] 🟢  │                              │
│          │                        │  # Auto-validated            │
│ 256px    │        Flexible        │        Flexible              │
│  Fixed   │      (30-70%)          │      (30-70%)                │
├──────────┴────────────────────────┴──────────────────────────────┤
│                        ◄──────► Resize Handle                     │
├───────────────────────────────────────────────────────────────────┤
│  TERMINAL (Toggleable, 0-30% height)                             │
│  [Terminal 1] $ npm run dev                                       │
│  Server running on http://localhost:3000                          │
└───────────────────────────────────────────────────────────────────┘
```

### Panel Descriptions

**File Tree (256px Fixed Width):**
- **Purpose:** Navigate project structure
- **Design:** Fixed width, left-aligned, dark background
- **Features:**
  - Recursive folder navigation
  - Click folders to expand/collapse
  - Click files to open in editor
  - File type icons (🐍 .py, 📄 .js, etc.)
  - Smart sorting (directories first, alphabetical)
- **Toggle:** Cmd+B to show/hide (future)

**Chat Panel (Center, Flexible Width):**
- **Purpose:** Primary AI interaction area
- **Design:** Minimalist, focus on conversation
- **Features:**
  - Natural language input at bottom
  - Conversation history with auto-scroll
  - Progress updates during generation
  - LLM settings inline (collapsed by default)
  - API config button (⚙️ icon) in input area
- **Constraints:** 30-70% of available width (minus FileTree)

**Code Editor (Right, Flexible Width):**
- **Purpose:** View and edit generated/existing code
- **Design:** Monaco editor (VS Code engine)
- **Features:**
  - Syntax highlighting for multiple languages
  - Multi-file tabs (VSCode-style)
  - File path in header
  - Line numbers, minimap, bracket matching
  - Close buttons on tabs
- **Constraints:** 30-70% of available width (minus FileTree)

**Terminal (Bottom, Toggleable):**
- **Purpose:** Run commands, see output, debug
- **Design:** Hidden by default, slides up when shown
- **Features:**
  - Standard terminal emulator
  - Command history (↑/↓ keys)
  - Multiple terminal sessions (future)
  - Drag vertical divider to resize (0-30% height)
- **Toggle Methods:**
  1. Top bar button: "Terminal: Show/Hide"
  2. Keyboard: Cmd+`
  3. Menu: View → Toggle Terminal
  4. Drag divider: Shows terminal when dragged up

---

## Resizable Dividers (November 28, 2025)

### Design: Smooth, No Visual Offset

**Vertical Divider (Chat ↔ Editor):**
- **Purpose:** Adjust space between chat and code editor
- **Visual:** 6px gray bar between panels
- **Behavior:**
  - Hover: Cursor changes to `↔` (col-resize)
  - Click-drag: Both panels resize in real-time
  - Range: Chat 30-70%, Editor 30-70% (balanced view)
  - **Fix Applied (Nov 28):** Cursor now perfectly aligned with divider, no offset

**Horizontal Divider (Editor ↔ Terminal):**
- **Purpose:** Adjust space between editor and terminal
- **Visual:** 4px gray bar between panels
- **Behavior:**
  - Hover: Cursor changes to `↕` (row-resize)
  - Click-drag: Terminal height adjusts
  - Range: Terminal 0-30% of window height
  - Dragging up from 0 shows terminal

### Technical Implementation (Global Cursor Control)

**Problem Solved (November 28):**
- **Issue:** Cursor appeared offset to the right of divider during drag
- **Root Cause:** FileTree width (256px) not accounted for in mouse calculations
- **Solution:** 
  1. Adjusted mouse position: `mouseXRelative = e.clientX - 256`
  2. Global CSS cursor override with `!important`
  3. Prevented text selection during drag

**CSS Classes:**
```css
/* Force cursor during resize - overrides all other cursors */
body.dragging-horizontal * {
  cursor: col-resize !important;
}
body.dragging-vertical * {
  cursor: row-resize !important;
}
```

**Result:** Smooth, flicker-free dragging with cursor perfectly aligned on divider

---

## LLM Settings (November 28, 2025)

### Design: Minimal Inline Component

**Philosophy:** Single-line layout, auto-save, visual status

**Layout:**
```
[Provider ▼]  [API Key Input ••••••••]  🟢
```

**Components:**

1. **Provider Dropdown** (128px width)
   - Options: Claude, OpenAI, Qwen
   - Auto-switches API key placeholder
   - Clears input when changed

2. **API Key Input** (Flexible width)
   - Type: Password (hidden characters)
   - Placeholder: "Enter API Key" (unconfigured) or "••••••••" (configured)
   - Auto-save: On blur (when you click away)
   - Security: Clears input after successful save

3. **Status Indicator** (2px dot)
   - 🟢 Green: API key configured and valid
   - 🔴 Red: No API key configured
   - 🟡 Yellow (pulsing): Saving in progress

**Location:**
- In Chat Panel, below input area
- Collapsed by default
- Click ⚙️ (API config button) to expand/collapse

**Behavior:**
1. Click API config button (⚙️) in chat input area
2. Settings expand below with single-line layout
3. Select provider from dropdown
4. Type API key in input field
5. Click away or press Tab → auto-saves
6. Status dot updates to green ✅
7. Input clears for security
8. Settings remain expanded until you click ⚙️ again

**Why This Design?**
- **Space Efficient:** One line vs. previous ~200px panel
- **Fast:** No "Save" button, auto-saves on blur
- **Clear:** Status dot shows configuration state instantly
- **Secure:** Input clears after save (can't see key again)
- **Accessible:** Tooltips explain each component

---

## Menu System (November 28, 2025)

### Design: Custom Items, No macOS Bloat

**Menu Bar:**
```
Yantra  File  Edit  View  Help
```

**Yantra Menu:**
- About Yantra
- Check for Updates...
- Settings... (Cmd+,)
- Quit Yantra (Cmd+Q)

**File Menu:**
- New File (Cmd+N)
- New Folder
- Open... (Cmd+O)
- Save (Cmd+S)
- Close (Cmd+W)

**Edit Menu** (Clean - No Native macOS Items):
- Undo (Cmd+Z)
- Redo (Cmd+Shift+Z)
- ---
- Cut (Cmd+X)
- Copy (Cmd+C)
- Paste (Cmd+V)
- Select All (Cmd+A)
- ---
- Find (Cmd+F)
- Replace (Cmd+Option+F)

**Fix Applied (November 28):**
- **Problem:** macOS native items appearing (Writing Tools, AutoFill, Start Dictation, Emojis & Symbols)
- **Solution:** Replaced all `MenuItem::` with `CustomMenuItem::` for full control
- **Result:** Clean edit menu with only intended items

**View Menu:**
- Toggle Terminal (Cmd+`)
- Toggle File Tree (Cmd+B)
- ---
- Reset Layout

**Help Menu:**
- Documentation
- Report Issue
- About

### Keyboard Shortcuts

| Shortcut | Action | Category |
|----------|--------|----------|
| **Cmd+`** | Toggle Terminal | View |
| **Cmd+B** | Toggle File Tree | View |
| Cmd+N | New File | File |
| Cmd+O | Open | File |
| Cmd+S | Save | File |
| Cmd+W | Close | File |
| Cmd+Z | Undo | Edit |
| Cmd+Shift+Z | Redo | Edit |
| Cmd+X | Cut | Edit |
| Cmd+C | Copy | Edit |
| Cmd+V | Paste | Edit |
| Cmd+A | Select All | Edit |
| Cmd+F | Find | Edit |
| Cmd+Option+F | Replace | Edit |
| Cmd+, | Settings | Yantra |
| Cmd+Q | Quit | Yantra |

**Note:** Bold shortcuts are newly implemented in Session 2 (November 28)

---

## User Workflows

### Workflow 1: Start New Project

**Steps:**
1. Launch Yantra
2. In Chat Panel, type: "Create a FastAPI project with authentication"
3. Agent asks clarifying questions (optional)
4. Agent generates project structure:
   - Creates all files (backend, frontend, tests, docs)
   - Writes code with proper imports and dependencies
   - Generates unit tests automatically
   - Runs tests and shows results in chat
   - Validates security (Semgrep scan)
   - Commits to git with descriptive message
5. Files appear in File Tree on left
6. Click any file to view code in Editor
7. Terminal shows test execution output (if terminal visible)
8. Review code, ask for changes in chat

**Time:** ~2-5 minutes for complete project (depending on complexity)

### Workflow 2: Add Feature to Existing Project

**Steps:**
1. Open your project folder (File → Open)
2. Wait for Yantra to analyze codebase (5-30 seconds)
3. In Chat Panel, describe feature: "Add user profile endpoint with validation"
4. Agent analyzes dependencies using GNN (Graph Neural Network)
5. Agent generates code:
   - Identifies files to modify/create
   - Adds new endpoint with proper imports
   - Updates tests
   - Validates no breaking changes
6. Code appears in Editor with tabs for each modified file
7. Tests run automatically, results in chat
8. If tests pass → auto-commits to git
9. If tests fail → agent shows error and offers to fix

**Time:** ~1-3 minutes per feature

### Workflow 3: Configure LLM Provider

**Steps:**
1. In Chat Panel, click ⚙️ (API config button)
2. LLM Settings expand below (single line)
3. Select provider from dropdown: [Claude ▼]
4. Type API key: `sk-ant-...`
5. Click away or press Tab
6. Status dot turns yellow (saving...)
7. After 1-2 seconds, dot turns green ✅
8. Input clears for security
9. Ready to use! Start chatting

**Time:** ~10 seconds

### Workflow 4: View Terminal Output

**Steps:**
1. Press **Cmd+`** (or click "Terminal: Show" button in top bar)
2. Terminal slides up from bottom
3. See command output (tests, builds, server logs)
4. Run manual commands if needed: `npm run dev`
5. Drag horizontal divider to resize terminal height
6. Press **Cmd+`** again to hide terminal

**Time:** Instant toggle

### Workflow 5: Adjust Panel Sizes

**Steps:**
1. **Chat ↔ Editor:** Hover over vertical gray bar between panels
2. Cursor changes to ↔
3. Click and drag left/right to resize
4. Panels adjust in real-time
5. **Editor ↔ Terminal:** Hover over horizontal gray bar
6. Cursor changes to ↕
7. Click and drag up/down to resize
8. Terminal height adjusts (0-30% range)

**Constraints:**
- Chat panel: 30-70% of available width
- Editor panel: 30-70% of available width
- Terminal: 0-30% of window height

---

## Status Indicators

### Top Bar
- **Terminal Button:** "Terminal: Show" or "Terminal: Hide" indicates current state

### Chat Panel
- **Agent Status:** Working/Idle/Error messages appear in conversation
- **LLM Settings Status Dot:**
  - 🟢 Green: API key configured
  - �� Red: No API key
  - 🟡 Yellow pulsing: Saving

### File Tree
- File type icons: 🐍 .py, 📄 .js, 📘 .ts, 📗 .jsx, 📙 .tsx, 📋 .json, 📝 .md

### Code Editor
- **Tab States:** Active tab highlighted, inactive tabs dimmed
- **Unsaved Changes:** Dot (•) next to filename in tab

### Terminal
- **Command Status:** Exit codes shown (0 = success, >0 = error)
- **Long-running:** Process indicators (spinner, ellipsis)

---

## Performance & Responsiveness

### Load Times
- **Application Launch:** <2 seconds to window visible
- **Project Analysis:** 5-30 seconds (depends on codebase size)
- **File Open:** <100ms to display in editor
- **Chat Response:** 1-5 seconds (LLM dependent)
- **Code Generation:** 3-30 seconds (depends on complexity)

### Smooth Interactions
- **Panel Resize:** 60 FPS dragging, real-time updates
- **Terminal Toggle:** Instant show/hide animation
- **File Tree:** Lazy loading, no lag with 1000+ files
- **Chat Scroll:** Auto-scroll smooth, manual scroll preserved
- **Editor:** Monaco engine performance (same as VS Code)

### Resource Usage
- **Memory:** ~200-500 MB typical usage
- **CPU:** Minimal when idle (<5%), spikes during code generation
- **Disk:** SQLite database for history, GNN graph, file cache

---

## Accessibility

### Keyboard Navigation
- **Tab:** Move between input fields
- **Shift+Tab:** Move backward
- **Arrow Keys:** Navigate file tree
- **Cmd+Shortcuts:** All major actions accessible

### Screen Reader Support
- **Labels:** All buttons have aria-labels
- **Status:** ARIA live regions for chat updates
- **Tooltips:** Descriptive text for icons

### Visual
- **Contrast:** WCAG AA compliant (4.5:1 minimum)
- **Font Size:** Readable 13-14px base, scalable
- **Colors:** Red/green indicators supplemented with icons

---

## Error Handling & User Feedback

### Error States

**LLM API Errors:**
- **Display:** Red message in chat with error details
- **Actions:** "Retry" button, link to settings
- **Example:** "API key invalid. Click ⚙️ to update."

**File System Errors:**
- **Display:** Toast notification (top-right)
- **Actions:** "Retry" or "Cancel"
- **Example:** "Cannot write to file (permission denied)"

**Test Failures:**
- **Display:** Red box in chat with test output
- **Actions:** "Fix Automatically" or "Show Code"
- **Example:**
  ```
  ❌ 3 tests failed
  test_user_auth: AssertionError on line 42
  [Fix Automatically] [Show Code]
  ```

### Progress Indicators

**Long Operations:**
- **Spinner:** Animated icon in chat
- **Text:** "Analyzing codebase..." / "Generating code..." / "Running tests..."
- **Time Estimate:** "~30 seconds remaining" (when available)

**Background Operations:**
- **Non-Blocking:** Agent continues responding during git commits, test runs
- **Notifications:** Toast when background task completes

---

## Tips & Best Practices

### For Best Results

1. **Be Specific:** "Add user authentication with JWT tokens" > "Add login"
2. **Provide Context:** Mention existing files if adding to codebase
3. **Iterative:** Start small, add features incrementally
4. **Review Code:** Always check generated code in Editor
5. **Use Terminal:** Monitor test output, build logs
6. **Keep API Keys Updated:** Green dot = ready to go

### Space Optimization

1. **Hide Terminal:** Use Cmd+` to toggle when not needed
2. **Resize Panels:** Give chat more space when reading responses
3. **Close Tabs:** Close editor tabs you're not using
4. **Collapse Settings:** LLM settings collapse after configuration

### Keyboard Efficiency

1. **Learn Shortcuts:** Cmd+` (terminal), Cmd+B (file tree)
2. **Tab Navigation:** Tab through input fields
3. **Arrow Keys:** Navigate file tree without mouse

---

## Troubleshooting

### "Terminal not responding"
- Check if terminal is visible (Cmd+` to toggle)
- Resize terminal divider (might be collapsed to 0 height)
- Reset layout: View → Reset Layout

### "API key not working"
- Click ⚙️ in chat input area
- Verify provider dropdown matches your key type
- Re-enter API key, wait for green dot
- Check API key validity on provider website

### "File tree not loading"
- Verify project folder has read permissions
- Check file count (<10,000 files recommended)
- Try reloading: File → Open (select same folder)

### "Divider cursor offset / flickering"
- **Fixed in November 28 update!**
- If still occurring, report issue via Help → Report Issue

### "Edit menu shows unwanted items"
- **Fixed in November 28 update!**
- macOS native items (Writing Tools, etc.) removed
- If still occurring, restart application

---

## Future Enhancements (Roadmap)

### Planned UX Improvements

**Phase 2 (Next 2 Months):**
- Settings window (Cmd+,) - minimal modal dialog
- File tree toggle (Cmd+B) - show/hide left panel
- Multiple terminal tabs - manage multiple sessions
- Dependency graph view - visualize code relationships
- Theme customization - light mode, custom colors

**Phase 3 (Months 5-8):**
- Browser preview panel - live web app preview (Chrome DevTools Protocol)
- Workflow automation UI - visual workflow builder
- Plugin system - community extensions
- Collaborative mode - multi-user editing

**Phase 4 (Months 9-12):**
- Voice commands - "Yantra, add a login page"
- Mobile companion app - monitor builds, get notifications
- Advanced code visualization - heatmaps, complexity graphs

---

## Changelog

### November 28, 2025
- ✅ **Added:** Minimal UX design philosophy section (space optimization, single-line layouts)
- ✅ **Added:** Top bar with YANTRA branding (bright white, 40px)
- ✅ **Added:** Terminal toggle button in top bar (show/hide state)
- ✅ **Added:** Keyboard shortcut Cmd+` for terminal toggle
- ✅ **Fixed:** Edit menu now clean - removed unwanted macOS items (Writing Tools, AutoFill, Dictation)
- ✅ **Fixed:** Vertical divider cursor alignment - no more offset or flicker
- ✅ **Improved:** Mouse position calculations account for FileTree width (256px)
- ✅ **Improved:** Global cursor control during drag with CSS `!important` classes
- ✅ **Redesigned:** LLM Settings to minimal inline component (single line: dropdown + input + status dot)
- ✅ **Added:** Auto-save for LLM settings (on blur)
- ✅ **Added:** Visual status indicators (green/red/yellow dots)
- ✅ **Added:** View menu with Toggle Terminal and Toggle File Tree options
- ✅ **Updated:** All documentation to reflect minimal design philosophy

### November 23, 2025
- 🎯 Initial UX documentation with 3-panel layout
- 📋 Multi-terminal interface description
- ⌨️ Keyboard shortcuts defined
- 🔄 User workflows documented

---

## Summary

Yantra's UX is designed around the principle of **minimal controls, maximum content**. The interface prioritizes:

1. **Chat-First Interaction** - AI is the primary interface, not buttons
2. **Space Optimization** - Every pixel used efficiently
3. **Keyboard Efficiency** - All actions accessible via shortcuts
4. **Visual Clarity** - Status clear at a glance (dots, colors, text)
5. **Auto-Save & Smart Defaults** - Reduce explicit user actions
6. **Progressive Disclosure** - Show details only when needed

This philosophy enables developers to focus on building, not navigating UI. The result: **faster workflows, less cognitive load, more productivity**.

---

**For technical implementation details, see:**
- Technical_Guide.md - Component architecture, algorithms
- Features.md - Feature specifications and use cases
- File_Registry.md - File purposes and dependencies

**For development guidelines, see:**
- .github/copilot-instructions.md - Coding standards and requirements
