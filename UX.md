# sYantra - User Experience Guide

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

| Shortcut        | Action           | Category |
| --------------- | ---------------- | -------- |
| **Cmd+`** | Toggle Terminal  | View     |
| **Cmd+B** | Toggle File Tree | View     |
| Cmd+N           | New File         | File     |
| Cmd+O           | Open             | File     |
| Cmd+S           | Save             | File     |
| Cmd+W           | Close            | File     |
| Cmd+Z           | Undo             | Edit     |

---

## Documentation Panels (November 28, 2025 - Updated)

### Overview

**Status:** ✅ Fully Enhanced with Search & Minimal UI
**Location:** Left sidebar (toggles with File Tree)
**Width:** Fixed 256px (w-64)

The Documentation Panels automatically track your project's features, decisions, changes, and plan. Everything is extracted from your chat conversations and markdown files—no manual data entry required.

### Tab Structure

**4 Tabs (Minimal UI Design):**

- 📋 **Features** - What you're building
- 💡 **Decisions** - Why you chose specific approaches
- 📝 **Changes** - What files were modified
- 🎯 **Plan** - Tasks organized by milestones

**Tab Design (Compact):**

```
Font size: 12px (text-xs)
Padding: 12px horizontal, 8px vertical (px-3 py-2)
Border: 2px blue underline for active tab
Background: Gray-700 for active, transparent for inactive
```

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

### Natural Language Explanations (NEW - November 28)

**Each tab now explains itself:**

**Features Tab:**

> "Features are automatically extracted from your chat conversations. As you describe what you want to build, Yantra identifies and tracks features, updating their status as implementation progresses."

**Decisions Tab:**

> "Critical technical decisions are logged here with full context. Each decision includes why it was made, what alternatives were considered, and the rationale behind the choice."

**Changes Tab:**

> "Complete audit trail of all code changes. Track what files were added, modified, or deleted, along with timestamps and descriptions."

**Plan Tab:**

> "Your project plan with tasks organized by milestones. Dependencies are tracked automatically, and tasks requiring your input are highlighted."

**Why Explanations Matter:**

- Users understand where data comes from
- Sets expectations for automation
- Reduces confusion about empty states
- Explains system behavior in natural language

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

**What You See:**

```
┌─────────────────────────────────────────────┐
│ [🔍 Search features...                    ] │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ 📋 Add User Authentication        ✅ Done│ │
│ │ JWT tokens with bcrypt password hash    │ │
│ │ Extracted from: Chat conversation       │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ 🔄 Implement File Upload  🔄 In Progress│ │
│ │ S3 storage with presigned URLs          │ │
│ │ Extracted from: User request            │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ ⏳ Add Email Notifications      ⏳ Planned│ │
│ │ SendGrid integration with templates     │ │
│ │ Extracted from: Product roadmap         │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Status Indicators:**

- ✅ Done - Green badge with checkmark
- 🔄 In Progress - Yellow badge with arrows
- ⏳ Planned - Gray badge with hourglass

**Information Shown:**

- Feature title (bold, 12px)
- Status badge (right-aligned)
- Description (11px, gray)
- Source attribution (10px, italic)

**Interaction:**

- Click search to filter
- Scroll through features
- No editing (read-only display)

### Decisions Tab

**What You See:**

```
┌─────────────────────────────────────────────┐
│ [🔍 Search decisions...                   ] │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Use PostgreSQL over MySQL               │ │
│ │                                         │ │
│ │ Context:                                │ │
│ │ Need JSONB support for flexible schema  │ │
│ │                                         │ │
│ │ Decision:                               │ │
│ │ PostgreSQL 14+ with JSONB columns       │ │
│ │                                         │ │
│ │ Rationale:                              │ │
│ │ Better JSON performance, native support │ │
│ │                                         │ │
│ │ Nov 28, 2025 10:30 AM                   │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Information Shown:**

- Decision title (bold, 12px)
- Context section (11px, gray)
- Decision section (11px, white, bold)
- Rationale section (11px, gray)
- Timestamp (10px, light gray)

**Decision Logging:**

- Automatically captured from chat
- Includes AI reasoning
- Shows alternatives considered
- Explains trade-offs made

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

**What You See:**

```
┌─────────────────────────────────────────────┐
│ [🔍 Search plan...                        ] │
│                                             │
│ 🎯 MVP Milestone                            │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Implement User Authentication       ✅  │ │
│ │ Depends on: Database setup              │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Add File Upload Functionality       🔄  │ │
│ │ Depends on: Auth, S3 bucket             │ │
│ │ [👤 User Action Required - Click]      │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Setup Email Notifications           ⏳  │ │
│ │ Depends on: SendGrid API key            │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Task Status:**

- ✅ Completed - Green checkmark badge
- 🔄 In Progress - Yellow arrows badge
- ⏳ Pending - Gray hourglass badge

**Information Shown:**

- Milestone header (12px, bold, white)
- Task title (12px, white, word-wrapped)
- Status badge (right-aligned, 10px)
- Dependencies (11px, gray, truncated)
- User action button (when needed)

**User Actions:**

- Click "👤 User Action Required" button
- Task instructions sent to chat
- You review and confirm
- Agent executes the action
- Task status updates automatically

**Plan Panel Overflow Fix (November 28):**

- Added break-words to task titles
- Truncate long dependency lists
- pr-2 padding on titles prevents badge overlap
- All tasks now visible without horizontal scroll

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
