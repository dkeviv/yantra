# Yantra - User Experience Guide

**Version:** MVP 1.1
**Last Updated:** December 9, 2025
**Audience:** End Users and Administrators

---

## Design Philosophy

**Updated:** December 9, 2025

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

**File Tree (Resizable, Responsive Width):**

- **Purpose:** Navigate project structure
- **Design:** Resizable width, left-aligned, dark background
- **Width Constraints:**
  - **Responsive:** 10%-60% of window width
  - **Minimum:** 150px (or 10% of window, whichever is larger)
  - **Maximum:** 600px (or 60% of window, whichever is smaller)
  - **Auto-close:** Automatically closes when dragged below 10% of window width
- **Features:**
  - Recursive folder navigation
  - Click folders to expand/collapse
  - Click files to open in editor
  - File type icons (🐍 .py, 📄 .js, etc.)
  - Smart sorting (directories first, alphabetical)
  - Drag right edge to resize width
- **Toggle:** View menu to reopen after auto-close

**Chat Panel (Center, Flexible Width):**

- **Purpose:** Primary AI interaction area
- **Design:** Minimalist, focus on conversation
- **Features:**
  - Natural language input at bottom
  - Conversation history with auto-scroll
  - Progress updates during generation
  - LLM settings inline (collapsed by default)
  - API config button (⚙️ icon) in input area
- **Constraints:** Flexible width (shares space with Code Editor)

**Code Editor (Right, Flexible Width):**

- **Purpose:** View and edit generated/existing code
- **Design:** Monaco editor (VS Code engine)
- **Features:**
  - Syntax highlighting for multiple languages
  - Multi-file tabs (VSCode-style)
  - File path in header
  - Line numbers, minimap, bracket matching
  - Close buttons on tabs
  - Switch between Editor, Dependencies, and Architecture views
- **Constraints:** Flexible width, auto-closes when dragged below 10% of window width
- **Auto-close:** Can be reopened from View menu

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

## Resizable Dividers (December 9, 2025)

### Design: Smooth, Thin, Responsive

**Updated:** December 9, 2025 - All dividers reduced for cleaner UI

**Vertical Divider (Chat ↔ Editor):**

- **Purpose:** Adjust space between chat and code editor
- **Visual:** 3px thin gray bar between panels (reduced from 6px)
- **Behavior:**
  - Hover: Cursor changes to `↔` (col-resize), divider highlights
  - Click-drag: Both panels resize in real-time
  - **Auto-close:** Code Editor closes when dragged below 10% of window width
  - Can reopen from View menu after auto-close
  - Smooth transition on hover (background color changes)

**Vertical Divider (File Tree ↔ Chat):**

- **Purpose:** Adjust File Tree width
- **Visual:** 2px ultra-thin gray bar (reduced from 4px)
- **Behavior:**
  - Hover: Cursor changes to `↔` (col-resize)
  - Click-drag: File Tree resizes (10%-60% of window)
  - **Auto-close:** File Tree closes when dragged below 10% of window width
  - Responsive constraints prevent overly narrow or wide panels

**Horizontal Divider (Editor ↔ Terminal):**

- **Purpose:** Adjust space between editor and terminal
- **Visual:** 2px thin gray bar (reduced from 4px)
- **Behavior:**
  - Hover: Cursor changes to `↕` (row-resize)
  - Click-drag: Terminal height adjusts
  - Range: Terminal 0-30% of window height
  - Dragging up from 0 shows terminal

### Technical Implementation (Global Cursor Control)

**Problem Solved (November 28):**

- **Issue:** Cursor appeared offset to the right of divider during drag
- **Root Cause:** FileTree width not accounted for in mouse calculations
- **Solution:**
  1. Adjusted mouse position: `mouseXRelative = e.clientX - fileTreeWidth`
  2. Global CSS cursor override with `!important`
  3. Prevented text selection during drag

**New Features (December 9, 2025):**

1. **Auto-close at 10% threshold:**
   - Panels automatically close when dragged below 10% of window width
   - Prevents unusably narrow panels
   - Clean UX with View menu to reopen

2. **Thinner dividers (2-3px):**
   - Reduced visual clutter
   - Smoother, more modern appearance
   - Still easy to grab and drag

3. **Responsive width constraints:**
   - Minimum and maximum widths scale with window size
   - No hard-coded pixel limits
   - Better multi-monitor support

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

## Typography and Font Consistency (December 9, 2025)

### Design: Consistent 11px Font Across All Tabs

**Updated:** December 9, 2025 - Standardized font sizes for visual consistency

**Philosophy:**

- Uniform typography creates a cohesive, professional appearance
- Consistent font sizes reduce cognitive load
- Clear hierarchy with primary (11px) and secondary (10px) text

**Typography Hierarchy:**

1. **Primary Text (11px):**
   - Tab labels (Code Dependencies, Traceability)
   - Button labels (Files, Functions, Classes, Reset, Export)
   - Help text at bottom of tabs
   - Graph node labels
   - Selected node details
   - Filter chips and controls

2. **Secondary Text (10px):**
   - Edge labels in Architecture view
   - Tooltips
   - Metadata text

**Implementation:**

**Dependencies Tab:**

- Filter buttons: 11px (consistent with other tabs)
- Selected node details: 11px
- Loading/error messages: 11px
- Graph node labels: 11px (reduced from 12px)
- Button styling uses theme variables for consistency

**Traceability Tab:**

- Already using 11px (reference standard)
- Help text: 11px
- Filter controls: 11px
- Graph labels: 11px

**Architecture Tab:**

- Node labels: 11px (reduced from 12px)
- Edge labels: 10px (appropriate for smaller text)
- Component details: 11px

**Button Styling Consistency:**

All tabs now use unified button styling:

```css
font-size: 11px;
background-color: active ? var(--accent-primary) : var(--bg-tertiary);
color: active ? var(--text-on-accent) : var(--text-secondary);
border: active ? none : 1px solid var(--border-secondary);
padding: 4px 10px;
border-radius: 3px;
```

**Benefits:**

1. **Visual Consistency:** All tabs look cohesive
2. **Better Readability:** 11px is optimal for UI elements
3. **Theme Compatibility:** Works well in both Dark Blue and Bright White themes
4. **Professional Polish:** Uniform typography signals attention to detail

**Testing:**

- Switch between Dependencies, Traceability, and Architecture tabs
- All text should appear consistent in size and style
- Help text at bottom should match across tabs
- Buttons should have identical styling

---

## New UI Features (November 29, 2025)

### 1. Dual-Theme System - Switch Between Dark Blue & Bright White

**Status:** ✅ Fully Implemented

**Location:** Top bar, right side (next to YANTRA logo)

**Design:** Toggle button with Sun/Moon icon

**Themes:**

1. **Dark Blue Theme (Default):**
   - Primary: #0B1437 (deep navy)
   - Background: #0E1726 (dark slate)
   - Accent: #4E7DD9 (professional blue)
   - Text: #E2E8F0 (soft white)
   - Best for: Low-light environments, night work

2. **Bright White Theme:**
   - Primary: #FFFFFF (pure white)
   - Background: #F8FAFC (light gray)
   - Accent: #3B82F6 (vibrant blue)
   - Text: #1E293B (dark slate)
   - Best for: Daylight work, high ambient light
   - WCAG AA contrast compliant

**User Interaction:**

1. Click **Sun icon** (☀️) in top bar to switch to Bright White theme
2. Click **Moon icon** (🌙) in top bar to switch to Dark Blue theme
3. Theme preference saved to localStorage
4. Smooth 0.3s transition between themes
5. Persists across sessions

**Visual Feedback:**

- Icon changes based on current theme
- Hover effect: Icon scales 1.1x
- Active state: Icon scales 0.95x (click feedback)
- Smooth color transitions across entire UI

**Use Cases:**

- Switch to Dark Blue for evening work (reduce eye strain)
- Switch to Bright White for office/daylight environment
- Accessibility: High-contrast bright theme for better readability
- Personal preference: Choose theme that matches workflow

---

### 2. Status Indicator - Real-Time Agent Activity

**Status:** ✅ Fully Implemented

**Location:** Agent panel header (next to "Agent" title)

**Design:** Small visual indicator (16px)

**States:**

1. **Idle (○):**
   - Static circular dot (6px diameter)
   - Color: Theme-aware primary color
   - Tooltip: "Agent is idle"

2. **Running (◌):**
   - Animated spinning circle
   - Clockwise rotation, 1-second duration
   - Color: Theme-aware primary color
   - Tooltip: "Agent is running..."

**User Interaction:**

- **No interaction required** - Automatically updates
- Hover to see tooltip explaining state
- Visual cue without being intrusive

**Visual Feedback:**

- Instant state updates when AI starts/stops processing
- Smooth CSS animation for running state
- Small size (unobtrusive, doesn't block content)
- Positioned in panel header (out of main work area)

**Use Cases:**

- Monitor code generation progress
- Know when AI is thinking vs. idle
- Debug hanging issues (stuck on running = check logs)
- Multi-task while AI works (see running state)

---

### 3. Task Queue Panel - Complete Visibility Into Agent Work

**Status:** ✅ Fully Implemented

**Toggle:** Top bar button "📋 Show Tasks" (right side)

**Design:** Slide-in overlay panel from right (320px width)

**Layout:**

```
┌─────────────────────────────┐
│  📋 Task Queue         [✕]   │ <- Header
├─────────────────────────────┤
│  📊 Statistics               │
│  ⏳ Pending: 5               │
│  🔄 In Progress: 1           │
│  ✅ Completed: 15            │
│  ❌ Failed: 2                │
├─────────────────────────────┤
│  🔵 Current Task             │ <- Highlighted
│  Generate authentication API  │
│  Priority: High              │
│  Started: 2 minutes ago      │
├─────────────────────────────┤
│  📝 Task List                │
│                              │
│  🟡 Pending - Medium         │
│  Create database models      │
│  Created: 5 minutes ago      │
│                              │
│  🟢 Completed - High         │
│  Write unit tests            │
│  Completed: 10 minutes ago   │
│                              │
│  🔴 Failed - Critical        │
│  Deploy to production        │
│  Failed: 1 hour ago          │
│  Error: Connection timeout   │
└─────────────────────────────┘
```

**Components:**

1. **Header:**
   - Title: "📋 Task Queue"
   - Close button (✕) in top right

2. **Statistics Dashboard:**
   - Pending count (yellow)
   - In-progress count (blue)
   - Completed count (green)
   - Failed count (red)

3. **Current Task Highlight:**
   - Blue background
   - Shows currently executing task
   - Priority badge
   - Time since started

4. **Task List:**
   - All tasks with status badges
   - Priority indicators
   - Timestamps (relative, e.g., "5 minutes ago")
   - Error messages for failed tasks

**Status Badges:**

- 🔵 **In Progress** - Blue badge with spinning icon
- 🟡 **Pending** - Yellow badge
- 🟢 **Completed** - Green badge with checkmark
- 🔴 **Failed** - Red badge with X

**Priority Badges:**

- 🔴 **Critical** - Red badge
- 🟠 **High** - Orange badge
- 🟡 **Medium** - Yellow badge
- 🟢 **Low** - Green badge

**User Interaction:**

1. Click "📋 Show Tasks" button in top bar
2. Panel slides in from right (smooth 0.3s animation)
3. View statistics, current task, and task list
4. Auto-refreshes every 5 seconds
5. Click backdrop (outside panel) to close
6. Click ✕ button to close

**Visual Feedback:**

- Smooth slide-in/out animation
- Auto-scroll to current task
- Backdrop darkens when panel open
- Hover effects on close button

**Use Cases:**

- Monitor multi-step workflow progress
- Review completed work history
- Debug failed tasks with error messages
- Understand what AI is doing at any moment
- Prioritize work by viewing pending tasks

---

### 4. Panel Expansion System - Focus on Any Panel

**Status:** ✅ Fully Implemented

**Location:** Expand buttons in each panel header

**Panels with Expansion:**

1. **File Explorer** - Expand button in header
2. **Agent Panel** - Expand button in header
3. **Code Editor** - Expand button in header (shows when no file tabs)

**Design:** Expand button with arrow icon

**Icons:**

- **◀ (Left Arrow):** Panel can expand (not currently expanded)
- **▶ (Right Arrow):** Panel is expanded (click to collapse)

**Behavior:**

- **Default Layout:**
  - File Explorer: 20% width
  - Agent Panel: 30% width
  - Code Editor: 50% width

- **When Panel Expanded:**
  - Expanded panel: 70% width
  - Other two panels: 15% width each
  - Smooth 0.3s CSS transition

- **Only One Panel Expanded at a Time:**
  - Expanding a panel automatically collapses others
  - Clicking expand button on expanded panel collapses all to default

**User Interaction:**

1. Click **◀** button in any panel header
2. That panel expands to 70% width
3. Other panels collapse to 15% each
4. Click **▶** button to collapse back to default layout
5. Click expand on different panel to switch focus

**Visual Feedback:**

- Icon changes: ◀ → ▶ when expanded
- Smooth width transitions (0.3s ease-in-out)
- No layout shift or jank
- Hover effect: Button background lightens

**Use Cases:**

- **Expand File Explorer:** See deeply nested file paths, long filenames
- **Expand Agent Panel:** Read long AI conversations, view more context
- **Expand Code Editor:** Focus on code, see more lines/columns
- **Switch Focus:** Expand file tree → find file → expand editor → review code

---

### 5. File Explorer Width Adjustment - Drag to Resize

**Status:** ✅ Fully Implemented

**Location:** Right edge of File Explorer panel

**Design:** 1px drag handle (expands to 4px on hover)

**Visual:**

- Thin vertical line on right edge of File Explorer
- Color: Theme-aware primary color
- Cursor: `col-resize` (↔) on hover
- Width: 1px default, 4px on hover

**Behavior:**

- **Drag to Adjust Width:**
  - Click and drag handle left/right
  - File Explorer width adjusts in real-time
  - Range: 200px (minimum) to 500px (maximum)
  - Smooth 60fps updates during drag

- **Persistence:**
  - Width saved to localStorage
  - Remembered across sessions
  - Default: 280px if no preference

- **Visibility:**
  - Only visible when File Explorer is NOT expanded
  - Hidden when any panel is expanded (70% layout)
  - Returns when all panels collapse to default

**User Interaction:**

1. Hover over right edge of File Explorer
2. Cursor changes to ↔ (resize cursor)
3. Drag handle visible (4px width)
4. Click and drag left/right
5. File Explorer width adjusts smoothly
6. Release mouse to finish
7. Width saved automatically

**Visual Feedback:**

- Handle width increases on hover (1px → 4px)
- Cursor changes to col-resize
- Smooth drag with no lag (60fps)
- Width clamped to 200-500px range

**Use Cases:**

- **Widen for Long Filenames:** Drag to 400px to see full paths
- **Narrow for More Code Space:** Drag to 220px to gain editor space
- **Custom Workspace:** Adjust to personal preference
- **Reset to Default:** Drag back to ~280px for balanced layout

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

### Universal Model Selection (November 29, 2025) - NEW

**Status:** ✅ Fully Implemented

**Location:** LLM Settings section (always visible, no toggle button)

**Design:** Checkboxes for each model per provider

**Supported Providers & Models:**

1. **🟣 Claude (Anthropic):**
   - ☑ claude-3-5-sonnet-20241022 (default)
   - ☐ claude-3-5-haiku-20241022
   - ☐ claude-3-opus-20240229

2. **🟢 OpenAI:**
   - ☑ gpt-4o (default)
   - ☐ gpt-4o-mini
   - ☐ gpt-4-turbo
   - ☐ gpt-3.5-turbo

3. **🔵 OpenRouter:**
   - ☑ deepseek-r1 (default)
   - ☐ qwen-2.5-72b
   - ☐ llama-3.3-70b
   - (Various other models)

4. **🟠 Groq:**
   - ☑ llama-3.3-70b (default)
   - ☐ llama-3.1-70b
   - ☐ mixtral-8x7b

5. **🔴 Gemini (Google):**
   - ☑ gemini-2.0-flash-exp (default)
   - ☐ gemini-1.5-pro
   - ☐ gemini-1.5-flash

**User Interaction:**

1. LLM Settings section now **always visible** (no toggle button)
2. Scroll to "Model Selection" subsection
3. Check/uncheck models for each provider
4. Selected models auto-save to localStorage
5. Agent panel filters chat to show only selected models

**Visual Feedback:**

- Checkboxes styled with theme-aware colors
- Provider sections with provider icons
- Model names in monospace font for clarity
- Hover effects on checkboxes (background color change)

**Chat Panel Filtering:**

- Only messages from **selected models** appear in chat history
- Deselecting a model hides its messages from chat
- Empty selection shows all messages (fallback behavior)
- Instant filtering (no page refresh needed)

**Use Cases:**

1. **Use Only Best Models:**
   - Select only GPT-4o and Claude Sonnet 3.5
   - Chat shows only these 2 models' responses
   - Reduces clutter, focuses on quality

2. **Compare Multiple Models:**
   - Select 3 models: GPT-4o, Claude Sonnet, Gemini Flash
   - Send same prompt to all 3
   - Compare responses side-by-side in chat

3. **Filter Out Mini Models:**
   - Uncheck gpt-4o-mini, claude-haiku, gemini-flash
   - Chat hides all "mini" model responses
   - See only full-size model outputs

4. **Cost-Conscious Development:**
   - Select only cheaper models (mini, haiku, flash)
   - Agent uses these for generation
   - Save 80% on API costs while developing

**Default Selection:**

- All models selected by default
- User can deselect to filter
- Preference saved to localStorage: `yantra-selected-models`

**Theme Integration:**

- Checkboxes use CSS variables: `var(--color-text)`, `var(--color-border)`
- Adapts to both Dark Blue and Bright White themes
- Provider icons use theme-aware colors

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

---

- Cut (Cmd+X)
- Copy (Cmd+C)
- Paste (Cmd+V)
- Select All (Cmd+A)

---

- Find (Cmd+F)
- Replace (Cmd+Option+F)

**Fix Applied (November 28):**

- **Problem:** macOS native items appearing (Writing Tools, AutoFill, Start Dictation, Emojis & Symbols)
- **Solution:** Replaced all `MenuItem::` with `CustomMenuItem::` for full control
- **Result:** Clean edit menu with only intended items

**View Menu:**

- Toggle Terminal (Cmd+`)
- Toggle File Tree (Cmd+B)

---

- Reset Layout

**Help Menu:**

- Documentation
- Report Issue
- About

### Keyboard Shortcuts

| Shortcut  | Action           | Category |
| --------- | ---------------- | -------- |
| **Cmd+`** | Toggle Terminal  | View     |
| **Cmd+B** | Toggle File Tree | View     |
| Cmd+N     | New File         | File     |
| Cmd+O     | Open             | File     |
| Cmd+S     | Save             | File     |
| Cmd+W     | Close            | File     |
| Cmd+Z     | Undo             | Edit     |

---

## Documentation Panels (November 28, 2025 - Updated)

### Overview

**Status:** ✅ Fully Enhanced with Search & Minimal UI
**Location:** Left sidebar (toggles with File Tree)
**Width:** Fixed 256px (w-64)

The Documentation Panels automatically track your project's features, decisions, changes, and plan. Everything is extracted from your chat conversations and markdown files—no manual data entry required.

### Tab Structure

**4 Tabs (Minimal UI Design):**

- 📋 **Features** - What you're building (auto-extracted from docs, chat, code)
- 💡 **Decisions** - Why you chose specific approaches (approval audit trail)
- 📝 **Changes** - What files were modified (complete audit trail)
- 🎯 **Plan** - Tasks organized by milestones (persistent project plan)

**Tab Design (Compact):**

```
Font size: 12px (text-xs)
Padding: 12px horizontal, 8px vertical (px-3 py-2)
Border: 2px blue underline for active tab
Background: Gray-700 for active, transparent for inactive
```

**Multi-User Synchronization:**

All four tabs synchronize in real-time across users working on the same project. When one user makes changes, all team members see the same view instantly.

### Search Functionality (NEW - November 28)

**Every tab now has search:**

```
┌─────────────────────────────────────────────┐
│  Features are automatically extracted...    │
│  [🔍 Search features...                   ] │
│                                             │
│  📋 Add User Authentication               │
│     Status: ✅ Done                         │
│     JWT tokens with bcrypt password hash    │
│     Extracted from: Chat conversation       │
│                                             │
│  🔄 Implement File Upload                  │
│     Status: 🔄 In Progress                  │
│     S3 storage with presigned URLs          │
│     Extracted from: User request            │
└─────────────────────────────────────────────┘
```

**Search Behavior:**

- Real-time filtering as you type
- Searches in title AND description (Features)
- Searches in context, decision, rationale (Decisions)
- Searches in description and file names (Changes)
- Searches in task titles (Plan)
- Empty state message: "No X found matching 'query'"

**Performance:**

- <5ms latency for typical searches
- Instant feedback (no debounce needed for small lists)
- Efficient with createMemo memoization

### Natural Language Explanations (Updated - December 3, 2025)

**Each tab now explains itself with detailed context:**

**Features Tab:**

> "Features are automatically extracted from your documentation, chat conversations, and code files. Agent monitors external tools like Notion for feature updates. Status updates in real-time as implementation progresses, with accurate completion tracking. All team members see the same synchronized view."

**Decisions Tab:**

> "Critical technical decisions are logged here with full context to serve as an approval audit trail. Each decision includes why it was made, what alternatives were considered, and the rationale behind the choice. Timestamps show when Agent proposed and when user approved. All team members see the same synchronized view."

**Changes Tab:**

> "Complete audit trail of all code changes. Track what files were added, modified, or deleted, along with timestamps and descriptions. All team members see the same synchronized view."

**Plan Tab:**

> "Your project plan with tasks organized by milestones. Agent confirms milestones and prioritization before starting work. Dependencies are tracked automatically, and tasks requiring your input are highlighted. The plan persists across sessions and all team members see the same synchronized view."

> "Your project plan with tasks organized by milestones. Dependencies are tracked automatically, and tasks requiring your input are highlighted."

**Why Explanations Matter:**

- Users understand where data comes from (docs, chat, code, external tools)
- Sets expectations for automation and multi-user synchronization
- Reduces confusion about empty states
- Explains system behavior in natural language
- Clarifies Agent's role in managing each tab
- Highlights approval audit and persistence features

### Minimal UI Updates (November 28)

**Reduced Spacing:**

```
Before → After (Space Savings)
─────────────────────────────────
Content padding:    16px → 8px   (50% reduction)
Card padding:       12px → 8px   (33% reduction)
Vertical spacing:   12px → 8px   (33% reduction)
Tab padding:     16px/12px → 12px/8px (25-33%)
```

**Result:** ~40% more content visible per screen

**Reduced Font Sizes:**

```
Before → After (Density Improvement)
────────────────────────────────────
Tab labels:      14px → 12px  (14% smaller)
Card titles:     14px → 12px  (14% smaller)
Card content:    12px → 11px  (8% smaller)
Timestamps:      12px → 10px  (17% smaller)
Search input:    14px → 11px  (21% smaller)
```

**Result:** Improved information density while maintaining readability

**Word-Wrap for Long Content:**

- Plan task titles: break-words prevents overflow
- File paths: truncate with ellipsis (...)
- All text: proper line wrapping

### Features Tab

**Purpose:** Automatically extract and track features from multiple sources with accurate completion tracking.

**Feature Extraction Sources:**

1. **Documentation Files:**
   - Markdown files (README.md, docs/\*.md)
   - Project documentation
   - Technical specifications

2. **Chat Conversations:**
   - User requests: "Add user authentication"
   - Natural language requirements
   - Feature discussions

3. **Code Files:**
   - Existing implementations
   - Code comments and docstrings
   - Function/class definitions

4. **External Tools (Post-MVP):**
   - **Notion:** Extract features from Notion pages
   - **Confluence:** Extract features from Confluence spaces (Post-MVP)
   - **Linear:** Import issues and feature requests

**Multi-User Synchronization:**

When multiple users work on the same project, all users see the same Features view in real-time. Feature status updates are synchronized across all connected clients instantly.

**Completion Tracking:**

Agent accurately tracks feature completion by monitoring:

- Code generation and implementation
- Test pass rates
- Integration status
- Deployment status

Features automatically move through status:

- ⏳ **Planned** → 🔄 **In Progress** → ✅ **Done**

**What You See:**

```
┌─────────────────────────────────────────────┐
│ Features are automatically extracted from   │
│ your documentation, chat conversations, and │
│ code files. Status updates in real-time.    │
│                                             │
│ [🔍 Search features...                    ] │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ 📋 Add User Authentication        ✅ Done│ │
│ │ JWT tokens with bcrypt password hash    │ │
│ │ Extracted from: Chat conversation       │ │
│ │ Completion: 100% (Tests passing)        │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ 🔄 Implement File Upload  🔄 In Progress│ │
│ │ S3 storage with presigned URLs          │ │
│ │ Extracted from: README.md               │ │
│ │ Completion: 60% (4/7 files done)        │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ⏳ Add Email Notifications      ⏳ Planned│ │
│ │ SendGrid integration with templates     │ │
│ │ Extracted from: Notion roadmap          │ │
│ │ Completion: 0% (Not started)            │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Status Indicators:**

- ✅ Done - Green badge with checkmark (100% complete)
- 🔄 In Progress - Yellow badge with arrows (1-99% complete)
- ⏳ Planned - Gray badge with hourglass (0% complete)

**Information Shown:**

- Feature title (bold, 12px)
- Status badge (right-aligned)
- Description (11px, gray)
- Source attribution (10px, italic)
- Completion percentage with details

**Interaction:**

- Click search to filter features
- Scroll through feature list
- Real-time updates as Agent works
- Synchronized across all team members
- Read-only display (Agent manages status)

### Decisions Tab

**Purpose:** Document all project decisions with full context to serve as an approval audit trail.

**Decision Documentation:**

Agent automatically documents decisions based on:

- Chat conversations with user
- Technical choices made during implementation
- Architecture decisions
- Technology selections
- Design pattern choices
- Trade-offs and alternatives considered

**Multi-User Synchronization:**

When multiple users work on the same project, all users see the same Decisions view in real-time. New decisions are synchronized across all connected clients instantly.

**MVP: Approval Audit View**

For MVP, the Decisions tab serves as the **Approver Audit View**:

- All critical decisions logged with timestamps
- Full context showing why decision was made
- User approval/confirmation captured
- Alternatives considered documented
- Complete audit trail for compliance

**What You See:**

```
┌─────────────────────────────────────────────┐
│ Critical technical decisions are logged     │
│ here with full context. Serves as approval  │
│ audit trail for all project choices.        │
│                                             │
│ [🔍 Search decisions...                   ] │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Use PostgreSQL over MySQL               │ │
│ │ 👤 User Approved ✅                      │ │
│ │                                         │ │
│ │ Context:                                │ │
│ │ Need JSONB support for flexible schema  │ │
│ │                                         │ │
│ │ Decision:                               │ │
│ │ PostgreSQL 14+ with JSONB columns       │ │
│ │                                         │ │
│ │ Alternatives Considered:                │ │
│ │ • MySQL 8.0 - Limited JSON support      │ │
│ │ • MongoDB - No ACID guarantees          │ │
│ │                                         │ │
│ │ Rationale:                              │ │
│ │ Better JSON performance, native support,│ │
│ │ full ACID compliance for critical data  │ │
│ │                                         │ │
│ │ Nov 28, 2025 10:30 AM - Agent proposed │ │
│ │ Nov 28, 2025 10:32 AM - User approved  │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Choose FastAPI over Django              │ │
│ │ 👤 User Approved ✅                      │ │
│ │                                         │ │
│ │ Context:                                │ │
│ │ Building REST API with async endpoints  │ │
│ │                                         │ │
│ │ Decision:                               │ │
│ │ FastAPI with Pydantic validation        │ │
│ │                                         │ │
│ │ Alternatives Considered:                │ │
│ │ • Django REST Framework - Heavier      │ │
│ │ • Flask - Manual async handling         │ │
│ │                                         │ │
│ │ Rationale:                              │ │
│ │ Native async, automatic OpenAPI docs,   │ │
│ │ type safety with Pydantic               │ │
│ │                                         │ │
│ │ Nov 28, 2025 09:15 AM - Agent proposed │ │
│ │ Nov 28, 2025 09:18 AM - User approved  │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Information Shown:**

- Decision title (bold, 12px)
- Approval status badge (✅ Approved / ⏳ Pending)
- Context section (11px, gray) - Why decision needed
- Decision section (11px, white, bold) - What was chosen
- Alternatives section (11px, gray) - Options considered
- Rationale section (11px, gray) - Why this choice
- Dual timestamps (10px, light gray):
  - When Agent proposed
  - When User approved

**Decision Logging Workflow:**

1. Agent encounters decision point during implementation
2. Agent analyzes options and proposes recommendation
3. Decision logged with full context in Decisions tab
4. User reviews in chat and approves/modifies
5. Approval captured with timestamp
6. All team members see updated decision instantly

**Interaction:**

- Search to filter decisions by keyword
- Scroll through decision history
- Click to expand full details (if truncated)
- Real-time updates as new decisions logged
- Synchronized across all team members
- Read-only display (Agent manages logging)

### Changes Tab

**What You See:**

```
┌─────────────────────────────────────────────┐
│ [🔍 Search changes...                     ] │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ file-added     Nov 28, 2025 11:15 AM    │ │
│ │ Created authentication service          │ │
│ │ 📄 src/auth/service.py                  │ │
│ │ 📄 src/auth/__init__.py                 │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ file-modified  Nov 28, 2025 11:20 AM    │ │
│ │ Added user registration endpoint        │ │
│ │ 📄 src/api/routes.py                    │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Change Types (Color-Coded Badges):**

- file-added - Green (10px badge)
- file-modified - Blue (10px badge)
- file-deleted - Red (10px badge)
- refactored - Purple (10px badge)

**Information Shown:**

- Change type badge (left)
- Timestamp (right, 10px)
- Description (12px, white)
- File list (11px, truncated paths)

### Plan Tab

**Purpose:** Create and maintain a persistent project-level plan that Agent methodically executes and tracks.

**Plan Creation:**

Agent creates project plan by:

- Breaking down user requirements into tasks
- Identifying dependencies between tasks
- Organizing tasks by milestones
- **Confirming milestones and prioritization with user**
- Adding sub-tasks as needed to track granular work

**Plan Persistence:**

- **Project-level plan** persists across sessions
- Plan survives application restarts
- Task status preserved and updated continuously
- Historical task data maintained for audit trail

**Multi-User Synchronization:**

When multiple users work on the same project, all users see the same Plan view in real-time. Task status updates, new tasks, and milestone changes are synchronized across all connected clients instantly.

**Milestone Confirmation:**

Before starting work, Agent must:

1. Propose milestones and task breakdown
2. Present prioritization to user
3. Wait for user confirmation/modification
4. Adjust plan based on user feedback
5. Only then begin execution

**Sub-Task Tracking:**

Agent can dynamically add sub-tasks to track:

- Implementation steps
- Testing requirements
- Code review checkpoints
- Deployment stages
- Documentation updates

**What You See:**

```
┌─────────────────────────────────────────────┐
│ Your project plan with tasks organized by   │
│ milestones. Agent confirms milestones and   │
│ prioritization before starting work.        │
│                                             │
│ [🔍 Search plan...                        ] │
│                                             │
│ 🎯 MVP Milestone (Priority: High)           │
│ Status: 2/5 tasks complete (40%)            │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ✅ Setup Project Structure          ✅  │ │
│ │ Completed: Nov 28, 2025 09:00 AM        │ │
│ │ Depends on: None                        │ │
│ │ Sub-tasks: 3/3 complete                 │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ✅ Implement Database Layer         ✅  │ │
│ │ Completed: Nov 28, 2025 10:15 AM        │ │
│ │ Depends on: Project Structure           │ │
│ │ Sub-tasks:                              │ │
│ │   ✅ Design schema                      │ │
│ │   ✅ Create migrations                  │ │
│ │   ✅ Add connection pooling             │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ 🔄 Implement User Authentication    🔄  │ │
│ │ In Progress: Started Nov 28, 11:00 AM   │ │
│ │ Depends on: Database layer              │ │
│ │ Sub-tasks:                              │ │
│ │   ✅ JWT token generation               │ │
│ │   🔄 Password hashing (in progress)     │ │
│ │   ⏳ Login endpoint (pending)           │ │
│ │   ⏳ Registration endpoint (pending)    │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ⏳ Add File Upload Functionality    ⏳  │ │
│ │ Depends on: Auth, S3 bucket setup       │ │
│ │ [👤 User Action Required - Click]      │ │
│ │ Action: Confirm S3 bucket configuration │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ⏳ Setup Email Notifications        ⏳  │ │
│ │ Depends on: SendGrid API key            │ │
│ │ Blocked: Waiting for API key from user  │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ 🎯 Phase 2 Milestone (Priority: Medium)     │
│ Status: 0/3 tasks complete (0%)             │
│ Starts after: MVP Milestone                 │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ⏳ Implement Real-time Features     ⏳  │ │
│ │ Depends on: MVP completion              │ │
│ │ [⏱️ Waiting on: MVP milestone]         │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Task Status:**

- ✅ Completed - Green checkmark badge
- 🔄 In Progress - Yellow arrows badge with sub-task breakdown
- ⏳ Pending - Gray hourglass badge
- 🚫 Blocked - Red stop badge with blocker reason

**Milestone Status:**

- Progress percentage (X/Y tasks complete)
- Priority indicator (High/Medium/Low)
- Dependency on other milestones
- Estimated completion (based on velocity)

**Information Shown:**

- Milestone header (12px, bold, white) with status
- Task title (12px, white, word-wrapped)
- Status badge (right-aligned, 10px)
- Timestamps (completion time or start time)
- Dependencies (11px, gray, truncated if long)
- Sub-tasks with individual status (11px, indented)
- User action button (when confirmation needed)
- Blocker reason (when task blocked)

**User Action Workflow:**

1. Task requires user input/confirmation
2. "👤 User Action Required" button appears
3. Click button → instructions sent to chat
4. User reviews and provides input in chat
5. Agent confirms understanding
6. Agent executes the action
7. Task status updates automatically
8. All team members see update instantly

**Plan Tracking:**

- Agent methodically works through tasks in order
- Respects dependencies (won't start dependent tasks early)
- Adds sub-tasks as needed for granular tracking
- Updates status in real-time
- Highlights blockers and user action items
- Calculates milestone completion percentage
- Provides velocity metrics

**Interaction:**

- Click search to filter tasks by keyword
- Scroll through plan hierarchy
- Click "User Action Required" for instructions
- Real-time updates as Agent works
- Synchronized across all team members
- Read-only display (Agent manages plan)

### Toggle Between Files and Docs

**Toggle Buttons:**

```
[Files]  [Docs]
 (active) (inactive)
```

**Behavior:**

- Click to switch between File Tree and Documentation Panels
- One panel visible at a time (maximize space)
- State persists during session
- Defaults to Files on application start

**Design (Minimal):**

```
Font size: 12px (text-sm)
Padding: 12px horizontal, 8px vertical (px-3 py-2)
Width: 50% each (flex-1)
Active: Gray-700 background
Inactive: Transparent with gray text
```

---

## Chat Panel (November 28, 2025 - Major Update)

### Overview

**Status:** ✅ Fully Enhanced with Minimal UI
**Location:** Center of screen
**Width:** Dynamic (45% default, resizable)

The Chat Panel is your primary interface to Yantra. Describe what you want to build, and the AI agent generates code, runs tests, and commits to git—all autonomously.

### Header (NEW Layout - November 28)

**Before (Old Design):**

```
┌──────────────────────────────────────────┐
│ Chat - Describe what you want to build  │  ← Large header (24px padding)
└──────────────────────────────────────────┘
```

**After (Minimal UI):**

```
┌──────────────────────────────────────────┐
│ Chat - Describe...  [Claude ▼] [⚙️]    │  ← Compact header (12px padding)
└──────────────────────────────────────────┘
```

**Header Components:**

1. **Title** (left): "Chat" (16px bold)
2. **Subtitle** (left): "- Describe what you want to build" (11px gray)
3. **Model Selector** (right): Dropdown with Claude/GPT-4/etc. (11px)
4. **API Config Button** (right): ⚙️ settings icon

**Space Savings:**

- Header height: 64px → 42px (34% reduction)
- Padding: 24px/16px → 12px/8px (50% reduction)
- Font sizes: 20px/14px → 16px/11px (20-21% smaller)

**Benefits:**

- Model selection always visible (not hidden in input area)
- More vertical space for messages
- Cleaner visual hierarchy
- Easier to change models mid-conversation

### Messages Area (NEW Design - November 28)

**Terminal-Like Design (Further Minimized):**

```
You › Create a REST API with authentication
Yantra › Generated 5 files, all tests passing ✅
You › Add user profile endpoint
Yantra › Generating code... (animated)
```

**Message Style:**

```
Font: Monaco/monospace (terminal aesthetic)
Font size: 11px (was 12px)
Line height: Relaxed (1.625)
Padding: 8px horizontal, 4px vertical (was 12px/8px)
Spacing: 2px between messages (was 4px)
Colors: Green (user), Blue (agent), Gray (system)
```

**Space Savings:**

- Font size: 12px → 11px (8% smaller)
- Padding: 12px/8px → 8px/4px (33-50% reduction)
- Message spacing: 4px → 2px (50% reduction)
- Result: ~30% more messages visible per screen

**Scroll Behavior:**

- Auto-scrolls to latest message
- Smooth scrolling animation
- Preserves scroll position when typing
- Infinite scroll for history (future)

### Input Area (NEW Design - November 28)

**Before (Old Design):**

```
┌────────────────────────────────────────────┐
│ [Model Dropdown ▼] [⚙️] [Send Button ▶]  │  ← Top row
│ ┌────────────────────────────────────────┐ │
│ │ Type your message...                   │ │  ← Textarea below
│ └────────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

**After (Minimal UI):**

```
┌────────────────────────────────────────────┐
│ ┌────────────────────────────────────────┐ │
│ │ Type your message...                 ▶│ │  ← Send button INSIDE
│ │                                        │ │
│ └────────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

**Key Changes:**

1. **Model selector moved to header** (top right)
2. **Send button inside textarea** (bottom right, absolute position)
3. **Single-row layout** (no separate button row)

**Textarea Design:**

```
Rows: 3
Font size: 11px (was 14px)
Padding: 8px (was 12px)
Padding-right: 40px (space for send button)
Background: Transparent (blends with container)
Border: None (clean look)
```

**Send Button Design:**

```
Position: absolute (right-1 bottom-1)
Size: 24px × 24px (p-1.5)
Icon: ▶ (play symbol)
Background: Primary blue (#3b82f6)
Hover: Darker blue (#2563eb)
Disabled: 50% opacity when empty or generating
```

**Space Savings:**

- Removed entire button row (~40px)
- Smaller fonts: 14px → 11px (21% reduction)
- Less padding: 12px → 8px (33% reduction)
- Result: ~20% more vertical space

**Keyboard Shortcuts:**

- Enter: Send message
- Shift+Enter: New line in textarea
- Escape: Clear input (future)
- Cmd+K: Focus input (future)

### LLM Settings Panel (Improved - November 28)

**Trigger:** Click ⚙️ (settings icon) in header

**Panel Design (Modal Overlay):**

```
┌──────────────────────────────────────────────┐
│                                              │
│     ┌─────────────────────────────────┐     │
│     │ API Configuration               │     │
│     │                                 │     │
│     │ Current Provider:               │     │
│     │ • Claude (Anthropic) ✓         │     │
│     │                                 │     │
│     │ Configuration Status:           │     │
│     │ ✓ Claude: Configured            │     │
│     │ ○ OpenAI: Not configured        │     │
│     │ ○ OpenRouter: Coming soon       │     │
│     │ ○ Groq: Coming soon             │     │
│     │                                 │     │
│     │ [Select Provider ▼]            │     │
│     │ [Enter API Key      ]          │     │
│     │ [Save]                         │     │
│     │                                 │     │
│     └─────────────────────────────────┘     │
│                                              │
└──────────────────────────────────────────────┘
```

**Components:**

1. **Current Provider Display** - Shows active LLM
2. **Configuration Status** - All 4 providers with ✓/○ indicators
3. **Provider Dropdown** - Select from 4 options
4. **API Key Input** - Password field (hidden characters)
5. **Save Button** - Saves and validates API key

**Behavior:**

1. Click ⚙️ in chat header
2. Modal overlay appears
3. Select provider from dropdown
4. Enter API key
5. Click Save
6. System validates key
7. Success message appears
8. Modal auto-closes after 2 seconds

**Security:**

- Input field clears after save
- API keys never logged or displayed
- Stored securely in OS keychain
- Password type prevents copy-paste visibility

---

## View Tabs (November 28, 2025 - Updated)

### Overview

**Status:** ✅ Enhanced with Minimal UI
**Location:** Right panel header
**Tabs:** Editor, Dependencies, Architecture

### Tab Design (NEW - November 28)

**Before (Old Design):**

```
[✏️ Editor]  [🔗 Dependencies]  [🏗️ Architecture]  ← Full text labels
```

**After (Minimal UI):**

```
[✏️ Editor]  [🔗 Deps]  [🏗️ Arch]  ← Abbreviated text
```

**Tab Style:**

```
Font size: 12px (was 14px)
Padding: 12px horizontal, 6px vertical (was 16px/8px)
Icon: Emoji (same size)
Text: Abbreviated for long words
Gap: 6px between icon and text (inline-flex)
```

**Space Savings:**

- Padding: 16px/8px → 12px/6px (25% reduction)
- Font size: 14px → 12px (14% smaller)
- Text length: "Dependencies" → "Deps" (60% shorter)
- Text length: "Architecture" → "Arch" (50% shorter)
- Result: All 3 tabs fit without overflow

**Benefits:**

- All tabs visible on smaller screens
- Icons provide visual recognition
- Abbreviated text reduces width
- Tooltip shows full name on hover

**Active Tab Indicator:**

```
Background: Gray-900 (darker)
Border: 2px blue border-bottom
Text: White
```

**Inactive Tab:**

```
Background: Transparent
Border: None
Text: Gray-400
Hover: Text becomes white
```

**Architecture View UI (Read-Only Visualization)**

No Manual Controls:

1. ❌ No "Create Architecture" button
2. ❌ No "Add Component" button
3. ❌ No "Save" button
4. ❌ No "Load" button
5. ❌ No drag-to-create connections
6. ❌ No manual component editing

Read-Only Features:

1. ✅ Zoom and pan navigation
2. ✅ Click component to see details (files, status)
3. ✅ Click connection to see relationship type
4. ✅ Filter by component type (Frontend/Backend/Database)
5. ✅ Version history display (auto-updated)
6. ✅ Export view (Markdown/Mermaid/JSON) - via agent command

Empty State Message:

🏗️ No Architecture Yet

Tell me in chat what you want to build,

and I'll generate the architecture for

you automatically.

Example: "Create a REST API with auth"

---

#### View Modes

**Architecture View** (replaces Code panel)

```
┌───────────────────────────────────────────────────────┐
│ [Complete] [Frontend ▼] [Backend ▼] [Database]       │ ← Hierarchical Tabs
├───────────────────────────────────────────────────────┤
│ ┌──────────────┐         ┌──────────────┐           │
│ │ UI Layer     │────────>│ API Client   │           │
│ │ 12 files ✓   │         │ 3 files ✓    │           │
│ └──────────────┘         └──────────────┘           │
│        │                        │                     │
│        v                        v                     │
│ ┌────────────────────────────────────┐               │
│ │       API Gateway                  │               │
│ │       5 files ✓                    │               │
│ └────────────────────────────────────┘               │
│          │                  │                         │
│          v                  v                         │
│    ┌──────────┐      ┌──────────┐                    │
│    │Auth Svc  │      │User Svc  │                    │
│    │4 files ✓ │      │6 files ✓ │                    │
│    └──────────┘      └──────────┘                    │
└───────────────────────────────────────────────────────┘
```

#### Component Visual States

- **📋 0/0 files** = Planned (gray) - Design exists, no code yet
- **🔄 2/5 files** = In Progress (yellow) - Partially implemented
- **✅ 5/5 files** = Implemented (green) - Fully coded
- **⚠️ Misaligned** (red) - Code exists but doesn't match architecture

#### Connection Types (Visual Arrows)

- **Solid arrow (→)** - Data flow
- **Dashed arrow (⇢)** - API call
- **Wavy arrow (⤳)** - Event/message
- **Dotted arrow (⋯>)** - Dependency
- **Double arrow (⇄)** - Bidirectional

#### Hierarchical Sliding Navigation

**Top-Level Tabs:**

```
[Complete] [Frontend ▼] [Backend ▼] [Database] [External]
```

**Frontend Sub-tabs** (appear when Frontend selected):

```
[UI Layer] [State Mgmt] [API Client] [Routing]
```

**Backend Sub-tabs:**

```
[API Layer] [Auth Service] [User Service] [Payment]
```

**Navigation:**

- Horizontal sliding with CSS transitions (300ms)
- Click to jump directly to any tab
- Keyboard shortcuts: `Ctrl+←/→`
- Breadcrumb trail: `Complete > Backend > Auth Service`

---

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

## YDoc Documentation System UX

### Status: ✅ Backend Complete | 🚧 UI Integration In Progress (Dec 9, 2025)

### Overview

YDoc is Yantra's integrated documentation system that treats documentation as first-class code artifacts. Every requirement, architecture decision, specification, and test result lives in the dependency graph alongside your code.

### Design Philosophy

**Documentation as Code:**

- Documentation files (`.ydoc`) are treated like code files
- Open `.ydoc` files in the main editor (same workflow as opening `.py` or `.rs`)
- Full traceability: Requirements → Architecture → Code → Tests
- Bidirectional navigation: Code ↔ Documentation

**Two Graph Views:**

1. **Code Dependencies** - Shows file/function/class relationships
2. **Traceability Graph** - Shows requirement/spec/code/test chains

These are **separate but related** - different nodes, different edges, different purposes.

---

### YDoc File Structure

**Project Structure:**

```
project-root/
├── ydocs/                      # All documentation lives here
│   ├── requirements/
│   │   ├── MASTER.ydoc        # Requirements overview
│   │   ├── EPIC-auth.ydoc     # Authentication requirements
│   │   └── EPIC-payment.ydoc  # Payment requirements
│   ├── architecture/
│   │   ├── MASTER.ydoc        # Architecture overview
│   │   ├── COMPONENT-api.ydoc # API component design
│   │   └── COMPONENT-db.ydoc  # Database design
│   ├── specs/
│   │   ├── MASTER.ydoc        # Specifications index
│   │   ├── FEATURE-login.ydoc # Login feature spec
│   │   └── FEATURE-oauth.ydoc # OAuth spec
│   ├── adr/
│   │   ├── ADR-001-postgres.ydoc
│   │   └── ADR-002-rust.ydoc
│   ├── guides/
│   │   ├── tech/              # Technical guides
│   │   ├── api/               # API documentation
│   │   └── user/              # User guides
│   ├── testing/
│   │   ├── MASTER.ydoc
│   │   ├── PLAN-auth.ydoc
│   │   └── results/           # Test result archives
│   └── logs/
│       ├── CHANGE-LOG.ydoc    # What changed
│       └── DECISION-LOG.ydoc  # Key decisions
└── src/                        # Your code
```

---

### Main Editor Integration

**Opening YDoc Files:**

```
┌─────────────────────────────────────────────────────────────────┐
│ File Tree              │ Main Editor (YDoc Block Editor)        │
├────────────────────────┼────────────────────────────────────────┤
│ 📁 ydocs/             │ ┌────────────────────┬─────────────────┐│
│   📁 requirements/    │ │ Content Editor     │ Metadata Panel  ││
│   📁 architecture/    │ │                    │                 ││
│   📄 EPIC-auth.ydoc ← │ │ # Authentication   │ 📋 Block Info   ││
│   📁 specs/           │ │                    │ ID: REQ-001     ││
│   📁 adr/             │ │ ## Requirements    │ Type: Requirement││
│   📁 guides/          │ │ - OAuth2 login     │ Status: ✅ Approved││
│   📁 testing/         │ │ - Token refresh    │                 ││
│   📁 logs/            │ │ - 30 min timeout   │ 🏷️ Tags        ││
│                       │ │                    │ • auth          ││
│ 📁 src/               │ │ [Monaco editor     │ • security      ││
│   📄 main.rs          │ │  with markdown     │                 ││
│   📄 auth.rs          │ │  highlighting]     │ 🔗 Links (3)    ││
│                       │ │                    │ → ARCH-001      ││
│                       │ │                    │ → src/auth.rs   ││
│                       │ │                    │ → test_auth.py  ││
│                       │ └────────────────────┴─────────────────┘│
│                       │ [Save] [View Graph] [Export] [History]  │
└───────────────────────┴─────────────────────────────────────────┘
```

**How It Works:**

1. Double-click `EPIC-auth.ydoc` in file tree
2. Main editor switches to YDoc Block Editor mode
3. Left side: Monaco editor with markdown
4. Right side: Metadata panel (tags, links, status)
5. Edit documentation like code (same workflow)

**File Type Detection:**

- `.ydoc` files → YDocBlockEditor component
- `.md` files → Markdown editor
- `.py`, `.rs`, etc. → Monaco code editor
- Same editor pane, different modes

---

### Graph Views (Tabbed Interface)

**Unified Graph Viewer with Tabs:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Graph View:  ┌───────────────┐  ┌──────────────┐              │
│              │ Code Deps     │  │ Traceability │              │
│              │ (Active)      │  └──────────────┘              │
│              └───────────────┘                                 │
├─────────────────────────────────────────────────────────────────┤
│ [Show: Files ✓] [Functions ✓] [Classes ✓] [Imports ✓]         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     Currently Showing: Code Dependencies                        │
│                                                                 │
│     ┌──────────┐                                               │
│     │ main.rs  │────────imports──────→ ┌──────────┐            │
│     │  (file)  │                       │ auth.rs  │            │
│     └──────────┘                       │  (file)  │            │
│          │                             └──────────┘            │
│          │                                  │                  │
│       calls                               calls                │
│          │                                  │                  │
│          ↓                                  ↓                  │
│   ┌──────────────┐                  ┌──────────────┐          │
│   │ main()       │                  │ login()      │          │
│   │ (function)   │                  │ (function)   │          │
│   └──────────────┘                  └──────────────┘          │
│                                                                 │
│     Legend: 🔵 Files  🟢 Functions  🟠 Classes  🟣 Imports    │
└─────────────────────────────────────────────────────────────────┘
```

**Switch to Traceability Tab:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Graph View:  ┌──────────────┐  ┌───────────────┐              │
│              │ Code Deps    │  │ Traceability  │              │
│              └──────────────┘  │ (Active)      │              │
│                                └───────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│ [Show: Requirements ✓] [Specs ✓] [Code ✓] [Tests ✓]           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     Currently Showing: Traceability Chain                       │
│                                                                 │
│   ┌──────────────┐                                             │
│   │ REQ-001      │─────traces_to────→ ┌──────────────┐        │
│   │ OAuth Login  │                    │ ARCH-001     │        │
│   │ (requirement)│                    │ Auth System  │        │
│   └──────────────┘                    │ (architecture)│       │
│                                       └──────────────┘        │
│                                             │                  │
│                                        implements              │
│                                             │                  │
│                                             ↓                  │
│                                      ┌──────────────┐          │
│                                      │ SPEC-003     │          │
│                                      │ OAuth Flow   │          │
│                                      │ (spec)       │          │
│                                      └──────────────┘          │
│                                             │                  │
│                                       realized_in              │
│                                             │                  │
│                                             ↓                  │
│                                      ┌──────────────┐          │
│                                      │ src/auth.rs  │          │
│                                      │ (code)       │          │
│                                      └──────────────┘          │
│                                             │                  │
│                                        tested_by               │
│                                             │                  │
│                                             ↓                  │
│                                      ┌──────────────┐          │
│                                      │ test_auth.py │          │
│                                      │ (test)       │          │
│                                      └──────────────┘          │
│                                                                 │
│   Legend: 🔴 Requirements  🔵 Architecture  🟡 Specs           │
│           🟢 Code  🟣 Tests                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Why Two Separate Graphs?**

| Aspect      | Code Dependencies                  | Traceability                                   |
| ----------- | ---------------------------------- | ---------------------------------------------- |
| **Purpose** | "What breaks if I change this?"    | "What requirements does this implement?"       |
| **Nodes**   | Files, functions, classes, imports | Requirements, architecture, specs, code, tests |
| **Edges**   | calls, imports, uses, inherits     | traces_to, implements, realized_in, tested_by  |
| **Size**    | 10,000+ nodes (large projects)     | 100-500 nodes (curated docs)                   |
| **Usage**   | Refactoring, dependency analysis   | Requirements traceability, compliance          |

**Navigation Between Graphs:**

- Right-click code file in Code Deps → "View Requirements" → switches to Traceability tab
- Right-click requirement in Traceability → "View Implementation" → switches to Code Deps tab
- Context actions create seamless cross-navigation

---

### YDoc Block Editor Features

**Content Editing (Left Panel):**

- Monaco editor with markdown syntax highlighting
- Live preview toggle (side-by-side or single view)
- Auto-save every 30 seconds
- Undo/redo with 30-revision history
- Keyboard shortcuts: Cmd+S (save), Cmd+B (bold), Cmd+K (search)

**Metadata Panel (Right Panel):**

```
┌─────────────────────────────────┐
│ 📋 Block Info                   │
├─────────────────────────────────┤
│ ID: REQ-AUTH-001                │
│ Type: [Requirement ▼]           │
│ Status: ✅ Approved             │
│ Created: Jan 15, 2025           │
│ Modified: Jan 20, 2025 (10m ago)│
├─────────────────────────────────┤
│ 🏷️ Tags                         │
├─────────────────────────────────┤
│ [+ Add tag...]                  │
│ × auth   × security   × oauth   │
├─────────────────────────────────┤
│ 🔗 Links (3)                    │
├─────────────────────────────────┤
│ → ARCH-001 (traces_to)          │
│ → SPEC-003 (implements)         │
│ → src/auth/oauth.rs (realized_in)│
│ ← test_auth.py (tested_by)      │
│                                 │
│ [+ Add Link]                    │
├─────────────────────────────────┤
│ 👥 Collaboration                │
├─────────────────────────────────┤
│ Currently editing: You          │
│ Last edit: Agent (10m ago)      │
└─────────────────────────────────┘
```

**Status Indicators:**

- 🔵 Draft (in progress)
- 🟡 Review (needs approval)
- 🟢 Approved (finalized)
- ⚫ Deprecated (outdated)

**Tag Autocomplete:**

- Type to search existing tags
- Suggests: auth, security, oauth, api, database, etc.
- Click tag to search for related blocks
- Color-coded by category

**Link Picker Dialog:**

```
┌───────────────────────────────────────┐
│ Add Link                              │
├───────────────────────────────────────┤
│ Link Type: [traces_to ▼]             │
│                                       │
│ Target: [Search blocks/files...    ] │
│                                       │
│ Results:                              │
│ □ ARCH-001: Authentication Arch       │
│ □ SPEC-003: OAuth Implementation      │
│ □ src/auth/oauth.rs                   │
│ □ tests/test_auth.py                  │
│                                       │
│ [Add Link] [Cancel]                   │
└───────────────────────────────────────┘
```

---

### Test Result Archive Management

**Purpose:** Keep database lean by archiving old test results (>30 days) while preserving summary statistics.

**Archive Panel (in YDoc Browser):**

```
┌─────────────────────────────────────────────────────────────────┐
│ YDoc Browser                                                    │
├─────────────────────────────────────────────────────────────────┤
│ 📚 Documents                                                    │
│   📁 Requirements (5)                                           │
│   📁 Architecture (3)                                           │
│   📁 Specifications (8)                                         │
│   📁 Testing (2)                                                │
│   📁 Test Results (45) ← [Archive]                             │
│                                                                 │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ 📦 Test Result Archive                                    │  │
│ ├───────────────────────────────────────────────────────────┤  │
│ │ Archive test results older than: [30 ▼] days             │  │
│ │ [Archive Now]  Last archived: 2 days ago                  │  │
│ │                                                           │  │
│ │ 📊 Archived Summaries (12)                               │  │
│ │                                                           │  │
│ │ ▼ 2025-01-15 (30 days ago)                               │  │
│ │   • Test Suite: auth_tests                               │  │
│ │   • Total: 45 tests                                      │  │
│ │   • Passed: 43 ✓  Failed: 2 ✗                           │  │
│ │   • Duration: 2.3s                                       │  │
│ │   [View Details]                                         │  │
│ │                                                           │  │
│ │ ▼ 2025-01-10 (35 days ago)                               │  │
│ │   • Test Suite: payment_tests                            │  │
│ │   • Total: 67 tests                                      │  │
│ │   • Passed: 67 ✓  Failed: 0 ✗                           │  │
│ │   • Duration: 4.1s                                       │  │
│ │   [View Details]                                         │  │
│ │                                                           │  │
│ │ Settings:                                                │  │
│ │ Keep archive for: [365 ▼] days                          │  │
│ │ [Cleanup Old Archives]                                   │  │
│ └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Archive Process:**

1. Click [Archive Now] button
2. Backend queries test results older than threshold (30 days default)
3. Creates summary statistics: total/passed/failed counts, duration
4. Deletes raw test data, keeps summary
5. Archive displayed in collapsible list
6. Graph edges preserved (traceability intact)

**Cleanup:**

- Keep archives for 1 year by default (configurable)
- Manual cleanup via [Cleanup Old Archives] button
- Automatic cleanup runs weekly

---

### Cross-Navigation Features

**From Code to Documentation:**

```
┌─────────────────────────────────────┐
│ Code Editor (src/auth/oauth.rs)    │
├─────────────────────────────────────┤
│ fn login(user: &str) -> Result {   │
│     // implementation               │ ← Right-click
│ }                                   │
│                                     │
│ Context Menu:                       │
│ ─────────────                       │
│ Cut                                 │
│ Copy                                │
│ Paste                               │
│ ─────────────                       │
│ 📄 View Requirements                │ ← Shows REQ-001, REQ-005
│ 📐 View Architecture                │ ← Shows ARCH-001
│ 📋 View Specifications              │ ← Shows SPEC-003
│ 🧪 View Tests                       │ ← Shows test_auth.py
│ 📊 View in Traceability Graph       │ ← Opens graph view
└─────────────────────────────────────┘
```

**From Documentation to Code:**

```
┌─────────────────────────────────────┐
│ YDoc Editor (REQ-001)               │
├─────────────────────────────────────┤
│ # REQ-001: User Authentication      │
│                                     │ ← Right-click
│ Users must authenticate via OAuth2  │
│                                     │
│ Context Menu:                       │
│ ─────────────                       │
│ Cut                                 │
│ Copy                                │
│ Paste                               │
│ ─────────────                       │
│ 💻 View Implementation              │ ← Opens src/auth/oauth.rs
│ 🧪 View Tests                       │ ← Opens test_auth.py
│ 📊 View in Traceability Graph       │ ← Opens graph view
│ 🔗 Add Link...                      │ ← Opens link picker
└─────────────────────────────────────┘
```

---

### Keyboard Shortcuts (YDoc)

| Shortcut       | Action                      |
| -------------- | --------------------------- |
| **Editor**     |                             |
| Cmd+S          | Save block                  |
| Cmd+K          | Quick search blocks/files   |
| Cmd+L          | Add link                    |
| Cmd+T          | Add tag                     |
| Cmd+G          | View in graph               |
| Cmd+P          | Toggle preview              |
| Escape         | Close editor                |
| **Navigation** |                             |
| Cmd+Shift+F    | Search all YDoc             |
| Cmd+Shift+G    | Open graph view             |
| Cmd+Shift+B    | Open YDoc browser           |
| **Graph**      |                             |
| Cmd+1          | Switch to Code Dependencies |
| Cmd+2          | Switch to Traceability      |
| Cmd+F          | Filter nodes                |
| Cmd+R          | Reset view                  |
| Cmd+E          | Export graph                |

---

### Visual Design

**Color Scheme (Unified):**

**Code Graph:**

- 🔵 Files: Blue (#3b82f6)
- 🟢 Functions: Green (#10b981)
- 🟠 Classes: Orange (#f59e0b)
- 🟣 Imports: Purple (#8b5cf6)

**Traceability Graph:**

- 🔴 Requirements: Red (#ef4444)
- 🔵 Architecture: Blue (#3b82f6)
- 🟡 Specs: Yellow (#facc15)
- 🟢 Code: Green (#10b981)
- 🟣 Tests: Purple (#8b5cf6)

**Status Colors:**

- 🔵 Draft: Blue
- 🟡 Review: Yellow
- 🟢 Approved: Green
- ⚫ Deprecated: Gray

**Hover Effects:**

- Code file hover → Highlight linked YDoc blocks in sidebar
- YDoc block hover → Highlight linked code files in tree
- Tooltip shows: "Implements REQ-001, SPEC-003"

---

### User Workflows

**1. Creating a New Requirement:**

```
1. Open File Tree → ydocs/requirements/
2. Right-click → "New YDoc Document"
3. Enter title: "User Profile Management"
4. Select type: "Requirements"
5. Document opens in editor
6. Write requirement markdown
7. Add tags: profile, user, crud
8. Add links to architecture/specs (if exist)
9. Auto-saves every 30s
10. Close editor (Cmd+W)
```

**2. Tracing Requirement to Code:**

```
1. Open REQ-001 in editor
2. Click [View in Graph] button
3. Graph switches to Traceability tab
4. REQ-001 is centered and highlighted
5. Follow edges: REQ-001 → ARCH-001 → SPEC-003 → src/auth.rs
6. Right-click src/auth.rs → "View Implementation"
7. Code editor opens src/auth.rs
8. Visual indicator shows "Implements REQ-001"
```

**3. Finding Documentation for Code:**

```
1. Editing src/auth/oauth.rs
2. Right-click in editor → "View Requirements"
3. Sidebar shows: REQ-001, REQ-005
4. Click REQ-001 → Opens in YDoc editor
5. See full requirement chain
6. Make documentation updates
7. Changes sync to graph automatically
```

**4. Archiving Old Test Results:**

```
1. Open YDoc Browser panel
2. Navigate to Test Results section
3. See "45 test results" with [Archive] button
4. Click [Archive]
5. Dialog: "Archive results older than [30] days?"
6. Click [Archive Now]
7. Backend creates summaries (2.3s)
8. Raw results deleted, summaries shown
9. Database size reduced by ~70%
10. Traceability links preserved
```

---

### Performance Expectations

**Editor Operations:**

- YDoc file open: <200ms
- Block save: <100ms (non-blocking)
- Tag autocomplete: <50ms
- Link search: <100ms
- Version history load: <300ms

**Graph Operations:**

- Graph render (code deps): <1s for 1000 nodes
- Graph render (traceability): <500ms for 500 nodes
- Graph filter: <100ms
- Graph zoom/pan: 60fps smooth
- Tab switch: <300ms

**Archive Operations:**

- Archive 100 test results: <2s
- View summaries: <100ms
- Cleanup old archives: <1s

---

### Best Practices

**Document Organization:**

1. Use MASTER.ydoc files as indexes
2. One epic per requirements file
3. One component per architecture file
4. One feature per spec file
5. Keep ADRs atomic (one decision per file)

**Tagging Strategy:**

1. Use consistent tag vocabulary
2. Max 3-5 tags per block
3. Tag categories: feature, component, priority, team
4. Example: `auth`, `high-priority`, `backend`, `sprint-5`

**Linking Strategy:**

1. Always link requirements → architecture
2. Always link architecture → specs
3. Always link specs → code
4. Always link requirements → tests
5. Use correct edge types (traces_to, implements, etc.)

**Archive Strategy:**

1. Archive test results monthly
2. Keep failures indefinitely
3. Keep last 10 passing runs
4. Review archive before cleanup
5. Export important results before archiving

---

## Troubleshooting

### YDoc Issues

**"YDoc file not opening in editor"**

- Verify file has `.ydoc` extension
- Check file permissions (read/write)
- Try closing and reopening the file
- Check console for parse errors

**"Links not showing in metadata panel"**

- Ensure graph edges exist in database
- Run "Rebuild Graph" from Tools menu
- Check if targets still exist (files not deleted)
- Verify edge types are correct

**"Archive not working"**

- Check test results exist in `/ydocs/testing/results/`
- Verify results are older than threshold (30 days)
- Check database permissions
- View logs: Help → View Logs

**"Graph not showing nodes"**

- Switch between Code Deps / Traceability tabs
- Check filter settings (might hide all nodes)
- Click "Reset View" button
- Refresh graph: Cmd+Shift+G

**"Traceability chain incomplete"**

- Verify all required documents exist
- Check graph edges in database
- Use link picker to add missing links
- Run "Validate Traceability" from Tools menu

### General Issues

**"Terminal not responding"**

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

### "Panel won't stay open / auto-closes"

- **New feature (December 9):** Panels auto-close at 10% of window width
- This is intentional to prevent unusably narrow panels
- Reopen from View menu after auto-close
- Drag panel wider to keep it open (above 10% threshold)

### "Dividers too thin / hard to grab"

- **Updated (December 9):** Dividers reduced to 2-3px for cleaner UI
- Hover over divider to see highlight effect
- Cursor changes to resize indicator (↔ or ↕)
- If still difficult, report via Help → Report Issue

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

### December 9, 2025 - UX Improvements Update

- ✅ **Improved:** Responsive panel widths with auto-close at 10% threshold
- ✅ **Removed:** Fixed minimum width restrictions on File Explorer (was 200px)
- ✅ **Added:** File Explorer now responsive (10%-60% of window, min 150px, max 600px)
- ✅ **Added:** Auto-close functionality when panels dragged below 10% of window width
- ✅ **Added:** View menu reopening for closed panels
- ✅ **Improved:** Divider widths reduced for smoother UI
  - Chat-Code divider: 6px → 3px
  - FileTree-Chat divider: 4px → 2px
  - Terminal divider: 4px → 2px
  - FileExplorer resize handle: 4px → 2px
- ✅ **Standardized:** Font sizes across all tabs to 11px
  - Dependencies tab buttons and text: 12px → 11px
  - Traceability tab: Already 11px (reference standard)
  - Architecture tab node labels: 12px → 11px
- ✅ **Unified:** Button styling across all tabs using theme variables
- ✅ **Added:** Typography hierarchy section in documentation
- ✅ **Updated:** Panel descriptions with new responsive behavior
- ✅ **Updated:** Resizable dividers section with new measurements
- 📝 **Files Modified:** layoutStore.ts, App.tsx, GraphViewer.css, DependencyGraph.tsx, ArchitectureCanvas.tsx
- 🎯 **Result:** Cleaner, more consistent, and responsive UI

### December 9, 2025 - YDoc System

- ✅ **Added:** Comprehensive YDoc Documentation System UX section
- ✅ **Added:** YDoc file structure and organization guidelines
- ✅ **Added:** Main editor integration for `.ydoc` files (YDocBlockEditor mode)
- ✅ **Added:** Dual graph view design (Code Dependencies + Traceability tabs)
- ✅ **Added:** YDoc Block Editor features documentation (content + metadata panel)
- ✅ **Added:** Test Result Archive Management UI design
- ✅ **Added:** Cross-navigation features (code ↔ documentation)
- ✅ **Added:** Keyboard shortcuts for YDoc operations
- ✅ **Added:** Visual design specifications (color scheme, hover effects)
- ✅ **Added:** User workflows for YDoc (create, trace, find, archive)
- ✅ **Added:** Performance expectations for YDoc operations
- ✅ **Added:** Best practices for documentation organization
- ✅ **Added:** YDoc-specific troubleshooting section
- 🚧 **Implementation:** Backend complete, UI integration in progress

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
