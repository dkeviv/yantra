# Yantra - Known Issues

**Purpose:** Track all bugs, issues, and their fixes  
**Last Updated:** November 28, 2025

---

## Active Issues

### Issue #1: Integration Tests Require API Keys for Execution

**Status:** Open (By Design)  
**Severity:** Low  
**Reported:** November 23, 2025  
**Component:** Testing  
**Assigned:** N/A (Manual testing required)

#### Description
The integration tests for automatic test generation (`tests/integration_orchestrator_test_gen.rs`) require an `ANTHROPIC_API_KEY` environment variable to run the full E2E flow with real LLM calls.

**Impact:**
- Tests skip in CI environment when API key not present
- Cannot verify test generation quality without manual testing
- MVP blocker fix validated structurally but not end-to-end

#### Steps to Reproduce
1. Run `cargo test integration_orchestrator_test_gen`
2. Without `ANTHROPIC_API_KEY` set, tests print "Skipping test: ANTHROPIC_API_KEY not set"
3. Tests pass (via skip) but don't validate actual behavior

#### Root Cause
- Integration tests need real LLM API to generate code and tests
- Cannot mock LLM responses realistically for this test
- API keys should not be committed to repository

#### Solution
**Current Approach (Acceptable for MVP):**
- Tests skip gracefully when API key unavailable
- Manual testing with real API key required before releases
- Documentation updated to note manual testing requirement

**Future Enhancement (Post-MVP):**
- Add mock LLM responses for integration tests
- Or: Use recorded LLM responses (VCR pattern)
- Or: Set up secure CI environment with encrypted API keys

#### Workaround
**For Manual Testing:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
cargo test integration_orchestrator_test_gen --test integration_orchestrator_test_gen
```

**Expected Output:**
- test_orchestrator_generates_tests_for_code: PASS (~15-20s)
- test_orchestrator_runs_generated_tests: PASS (~15-20s)

#### Fixed In
N/A - By design, will remain as manual testing requirement for MVP

---

## Issue Format

```
## Issue #[Number]: [Short Title]

**Status:** [Open | In Progress | Fixed | Won't Fix]
**Severity:** [Critical | High | Medium | Low]
**Reported:** [Date]
**Component:** [GNN | LLM | UI | Testing | Security | Browser | Git]
**Assigned:** [Person]

### Description
Clear description of the issue

### Steps to Reproduce
1. Step 1
2. Step 2
3. Expected vs Actual

### Root Cause
What's causing the issue

### Fix
How it was fixed (or planned fix)

### Fixed In
Version/commit where fixed
```

---

## Resolved Issues

### Issue #5: Component Tests Failing Due to Missing CSS Classes and Mock Issues

**Status:** ✅ Fixed  
**Severity:** High  
**Reported:** November 30, 2025  
**Component:** Testing / Frontend Components  
**Fixed By:** Session 3 (Nov 30, 2025)

#### Description
After migrating from Vitest to Jest for component testing, 52 out of 76 component tests were failing (32% pass rate). Tests were hanging indefinitely and failing due to missing CSS classes, incorrect mock data, and implementation mismatches.

#### Steps to Reproduce
1. Run `npm run test:components`
2. Observe: Tests hang for very long time, must be manually cancelled
3. When completed: Only 24/76 tests passing

#### Root Causes

**1. Test Hanging (Most Critical):**
- Tauri mock function `jest.fn()` returns `undefined` instead of Promises
- TaskPanel's `onMount()` calls `await invoke('get_task_queue')`
- `await undefined` → never resolves → infinite wait
- Each test waits until timeout → 76 tests × ~5s timeout = 6+ minutes

**2. Missing CSS Classes:**
- StatusIndicator: Missing `.status-indicator`, `.idle`, `.running`, size classes
- ThemeToggle: Wrong theme names ('dark-blue' vs 'dark'), wrong localStorage key
- TaskPanel: Missing `.backdrop`, `.task-panel`, `.close-button`, badge classes

**3. Design Mismatches:**
- Statistics labels: "Active" vs expected "In Progress", "Done" vs "Completed"
- Missing Failed count in statistics display
- Timestamps showing formatted dates instead of relative times
- Error messages not displaying for failed tasks

**4. Size and Color Issues:**
- StatusIndicator not applying explicit pixel dimensions (16px, 24px, 32px)
- CSS variables not being used (used other variable names)

#### Fixes Applied

**1. Created Tauri Module Mock (`src-ui/__mocks__/@tauri-apps/api/tauri.js`):**
```javascript
export const invoke = jest.fn((cmd) => {
  switch (cmd) {
    case 'get_task_queue': return Promise.resolve([...tasks...]);
    case 'get_current_task': return Promise.resolve({...task...});
    case 'get_task_stats': return Promise.resolve({...stats...});
    default: return Promise.resolve(null);
  }
});
```
✅ Tests now complete in <1 second instead of hanging

**2. Fixed StatusIndicator:**
- Added `.status-indicator` class to container
- Added dynamic status classes (`.idle`, `.running`)
- Added dynamic size classes (`.small`, `.medium`, `.large`)
- Changed default size from 'medium' to 'small'
- Added explicit pixel dimensions: `width: sizePixels[size()]`
- Changed colors to use `var(--color-primary)`

**3. Fixed ThemeToggle:**
- Changed theme type: `'dark-blue'|'bright-white'` → `'dark'|'bright'`
- Changed localStorage key: `'yantra-theme'` → `'theme'`
- Replaced SVG icons with emoji (🌙 and ☀️)
- Added try-catch for localStorage errors (jsdom compatibility)

**4. Fixed TaskPanel:**
- Added structural classes: `.backdrop`, `.task-panel`, `.close-button`, `.current-task`
- Added badge classes: `.badge-pending`, `.badge-in-progress`, `.badge-completed`, `.badge-failed`
- Added priority classes: `.priority-critical`, `.priority-high`, `.priority-medium`, `.priority-low`
- Changed statistics labels: "Active" → "In Progress", "Done" → "Completed"
- Added 5th statistic: Failed count
- Implemented relative time formatting: `formatDate()` returns "2 minutes ago"
- Ensured error messages display for failed tasks

**5. Fixed Test Data:**
- Added `error` field to failed tasks in test mock data
- Added `total` field to mockStats
- Fixed auto-refresh test expectations (2 calls → 3 calls for queue + current + stats)
- Fixed rapid clicking test expectation (10 clicks from 'dark' → 'dark', not 'bright')

#### Files Changed
1. `src-ui/__mocks__/@tauri-apps/api/tauri.js` - Created Tauri API mock
2. `src-ui/components/StatusIndicator.tsx` - Added CSS classes, dimensions, colors
3. `src-ui/components/ThemeToggle.tsx` - Fixed theme names, localStorage, error handling
4. `src-ui/components/TaskPanel.tsx` - Added CSS classes, fixed labels, timestamps, stats
5. `src-ui/components/__tests__/TaskPanel.test.tsx` - Updated mock data and expectations
6. `src-ui/components/__tests__/ThemeToggle.test.tsx` - Fixed rapid clicking test, added waitFor import
7. `jest.setup.cjs` - Added complementary Tauri mock

#### Results
- **Before:** 24/76 tests passing (32%) - tests hung indefinitely
- **After:** 74/76 tests passing (97%) - tests complete in <1 second
- **Improvement:** +50 tests fixed (+65 percentage points)

**Remaining 2 Failures:**
- StatusIndicator dimension test - jsdom limitation (getComputedStyle returns empty string)
- StatusIndicator CSS variables test - jsdom limitation (computed styles not available)

These 2 tests would pass in a real browser but fail in jsdom due to technical limitations of the test environment.

#### Test Suite Summary
| Component | Tests Passing | Total Tests | Pass Rate |
|-----------|---------------|-------------|-----------|
| StatusIndicator | 18/20 | 20 | 90% |
| ThemeToggle | 25/25 | 25 | 100% |
| TaskPanel | 31/31 | 31 | 100% |
| **Total** | **74/76** | **76** | **97%** |

#### Fixed In
Multiple commits (November 30, 2025)

---

### Issue #2: Divider Cursor Offset ~100px to the Right

**Status:** ✅ Fixed  
**Severity:** Medium  
**Reported:** November 28, 2025  
**Component:** UI  
**Fixed By:** Session 2 (Nov 28, 2025)

#### Description
When dragging the vertical divider between Chat and Code Editor panels, the cursor appeared approximately 100px to the right of the divider itself, creating a confusing UX where the cursor and divider were visually separated.

#### Steps to Reproduce
1. Open Yantra application
2. Hover over vertical divider between Chat and Code Editor
3. Click and drag to resize
4. Observe: Cursor appears ~100px to the right of the actual divider position

#### Root Cause
The mouse position calculation used a hardcoded FileTree width of 256px:
```typescript
const fileTreeWidth = appStore.showFileTree() ? 256 : 0;
```

However, due to browser rendering, padding, borders, or zoom levels, the actual rendered width could differ slightly, causing the calculated mouse position to be offset.

#### Fix
Changed to dynamically measure the actual FileTree width using `getBoundingClientRect()`:
```typescript
const fileTreeElement = document.querySelector('.w-64');
const fileTreeWidth = fileTreeElement ? fileTreeElement.getBoundingClientRect().width : 0;
```

This ensures the mouse position calculation always uses the exact rendered width, eliminating any offset.

#### Files Changed
- `src-ui/App.tsx` - Lines 59-61: Updated handleMouseMove to use getBoundingClientRect()

#### Result
✅ Cursor now perfectly aligns with divider during drag  
✅ No visual offset or flicker  
✅ Smooth, intuitive resizing experience

#### Fixed In
Commit: 4401f6b (November 28, 2025)

---

### Issue #3: macOS Native Menu Items Appearing in Edit Menu

**Status:** ✅ Fixed  
**Severity:** Medium  
**Reported:** November 28, 2025  
**Component:** UI / Menu System  
**Fixed By:** Session 2 (Nov 28, 2025)

#### Description
macOS was automatically injecting native menu items into the "Edit" menu:
- "Writing Tools"
- "AutoFill"
- "Start Dictation"
- "Emoji & Symbols"

These items appeared even though the menu was defined with only custom items (Undo, Redo, Cut, Copy, Paste, etc.), creating visual clutter and contradicting the minimal UX design philosophy.

#### Steps to Reproduce
1. Launch Yantra on macOS
2. Open the "Edit" menu from menu bar
3. Observe unwanted native macOS items appearing below custom items

#### Root Cause
macOS automatically recognizes the "Edit" menu name and injects standard system menu items regardless of the custom menu definition. This is a Tauri v1 limitation where:
1. Menus named "Edit" trigger macOS system behavior
2. Using `MenuItem::Separator` (native items) can also trigger additional injections
3. No way to disable this behavior in Tauri v1

#### Fix
**Solution 1:** Renamed "Edit" to "Actions"
```rust
let edit_menu = Submenu::new(
    "Actions",  // Changed from "Edit"
    Menu::new()
        .add_item(CustomMenuItem::new("undo", "Undo").accelerator("Cmd+Z"))
        // ... rest of items
);
```

**Solution 2:** Replaced all `MenuItem::Separator` with custom disabled separators
```rust
.add_item(CustomMenuItem::new("separator1", "───────────────").disabled())
```

#### Files Changed
- `src-tauri/src/main.rs` - Lines 896-909: Renamed Edit menu to Actions
- `src-tauri/src/main.rs` - Lines 885-930: Replaced native separators with custom separators

#### Result
✅ Menu now shows "Actions" instead of "Edit"  
✅ No macOS native items appear  
✅ Clean, minimal menu with only intended items  
✅ All keyboard shortcuts still work (Cmd+Z, Cmd+C, etc.)

#### Fixed In
Commit: 4401f6b (November 28, 2025)

---

### Issue #4: Qwen Provider Showing in LLM Settings (Not Implemented)

**Status:** ✅ Fixed  
**Severity:** Low  
**Reported:** November 28, 2025  
**Component:** UI / LLM Settings  
**Fixed By:** Session 2 (Nov 28, 2025)

#### Description
The LLM Settings dropdown showed "Qwen" as a provider option, but Qwen is not implemented in the backend. Selecting it would not work, creating user confusion.

#### Steps to Reproduce
1. Open Yantra application
2. In Chat Panel, click ⚙️ (API config button)
3. LLM Settings expand showing provider dropdown
4. Observe: "Claude", "OpenAI", and "Qwen" options
5. Select "Qwen" and try to save API key
6. Result: Nothing happens (not implemented)

#### Root Cause
Frontend LLMSettings component included Qwen in the provider dropdown, but backend only supports Claude and OpenAI:
```typescript
type ProviderType = 'claude' | 'openai' | 'qwen';  // Qwen not implemented
```

Backend has no commands for:
- `setQwenKey()`
- Qwen provider configuration

#### Fix
Removed Qwen from frontend:
```typescript
type ProviderType = 'claude' | 'openai';  // Only implemented providers

// Removed from dropdown
<select>
  <option value="claude">Claude</option>
  <option value="openai">OpenAI</option>
  {/* <option value="qwen">Qwen</option> - REMOVED */}
</select>
```

Also removed Qwen-related logic from:
- `getProviderStatus()` - Removed 'qwen' case
- `handleBlur()` - Removed Qwen save logic

#### Files Changed
- `src-ui/components/LLMSettings.tsx` - Lines 9, 23-25, 31-40, 63-80, 103-104: Removed Qwen references

#### Result
✅ Only Claude and OpenAI show in provider dropdown  
✅ Both providers work correctly  
✅ No confusion about unsupported providers

#### Future Enhancement
When OpenRouter and Groq are implemented:
1. Add backend support in `src-tauri/src/llm/mod.rs`
2. Add Tauri commands for API key management
3. Update TypeScript API bindings in `src-ui/api/llm.ts`
4. Add to LLMSettings dropdown

#### Fixed In
Commit: 4401f6b (November 28, 2025)

---

### Issue #5: Close Folder Menu Item Not Working

**Status:** ✅ Fixed  
**Severity:** High  
**Reported:** November 28, 2025  
**Component:** UI / File Management  
**Fixed By:** Session 2 (Nov 28, 2025)

#### Description
Clicking "File → Close Folder" did not properly close the project:
- Project path was cleared but shown as empty string instead of null
- File tree remained visible with previous project files
- Open file tabs remained in editor
- Code editor still showed previous file content

#### Steps to Reproduce
1. Open a project folder (File → Open Folder)
2. Observe files in file tree, open some files
3. Click File → Close Folder
4. Result: Message appears but file tree and editor still show project

#### Root Cause
The `menu-close-folder` event handler only cleared the project path and showed a message:
```typescript
appStore.setProjectPath('');  // Should be null, not empty string
appStore.addMessage('system', 'Closed project folder');
// Missing: Clear file tree, open files, editor content
```

Additionally:
- FileTree component had no way to clear its internal state (rootPath, treeNodes)
- No event communication between App and FileTree
- Open files array was not cleared
- Active file index was not reset

#### Fix
**Part 1: Enhanced App.tsx handler**
```typescript
const unlistenMenuCloseFolder = listen('menu-close-folder', () => {
  appStore.setProjectPath(null);  // Use null, not empty string
  appStore.setOpenFiles([]);      // Clear all open files
  appStore.setActiveFileIndex(-1); // Reset active file
  appStore.setCurrentCode('# Your generated code will appear here\n'); // Clear editor
  window.dispatchEvent(new CustomEvent('close-project')); // Notify FileTree
  appStore.addMessage('system', '✅ Project folder closed. Open a new project to get started.');
});
```

**Part 2: FileTree event listener**
```typescript
onMount(() => {
  const handleCloseProject = () => {
    setRootPath(null);    // Clear root path
    setTreeNodes([]);     // Clear file tree
    setError(null);       // Clear any errors
  };
  
  window.addEventListener('close-project', handleCloseProject);
  onCleanup(() => window.removeEventListener('close-project', handleCloseProject));
});
```

#### Files Changed
- `src-ui/App.tsx` - Lines 157-165: Enhanced close folder handler
- `src-ui/components/FileTree.tsx` - Lines 1-33: Added close-project event listener

#### Result
✅ Project path cleared (null)  
✅ File tree completely empty  
✅ All open file tabs closed  
✅ Editor shows placeholder text  
✅ Clear confirmation message  
✅ Ready for new project

#### Fixed In
Commit: 4401f6b (November 28, 2025)

---

## Won't Fix

*No "won't fix" issues yet.*

---

## Common Patterns

*As issues are discovered and fixed, common patterns will be documented here to prevent recurrence.*

### Pattern Categories

#### GNN Issues
*To be populated as issues are discovered*

#### LLM Issues
*To be populated as issues are discovered*

#### UI Issues
*To be populated as issues are discovered*

#### Testing Issues
*To be populated as issues are discovered*

#### Security Issues
*To be populated as issues are discovered*

#### Browser Issues
*To be populated as issues are discovered*

#### Git Issues
*To be populated as issues are discovered*

---

## Issue Statistics

| Category | Open | In Progress | Fixed | Total |
|----------|------|-------------|-------|-------|
| Critical | 0 | 0 | 0 | 0 |
| High | 0 | 0 | 0 | 0 |
| Medium | 0 | 0 | 0 | 0 |
| Low | 0 | 0 | 0 | 0 |
| **Total** | **0** | **0** | **0** | **0** |

---

## Issue #3: Dual Test System Required (Vitest + Jest)

**Status:** Resolved (Workaround Implemented)  
**Severity:** Medium  
**Reported:** December 2024  
**Component:** Testing Infrastructure  
**Assigned:** N/A

### Description
SolidJS component tests cannot run in vitest due to JSX transformation issues. Required implementing a dual test system using both vitest and Jest.

**Problem:**
- Vitest failed to resolve `solid-js/jsx-dev-runtime` for component tests
- Root cause: Version conflicts between vitest's bundled Vite and vite-plugin-solid
- Multiple attempted fixes failed (aliases, different JSX modes, plugin configurations)

**Solution Implemented:**
- **Vitest**: Store and utility tests (49 tests, 100% passing)
- **Jest**: Component tests (76 tests, 24 passing, 52 failing)

### Technical Details

**Failed Attempts:**
1. ❌ Alias jsx-dev-runtime to dev.js
2. ❌ Use vite-plugin-solid in vitest.config.ts
3. ❌ Different JSX transform modes
4. ❌ Merge vite and vitest configs

**Working Solution:**

**Vitest Configuration** (`vitest.config.ts`):
```typescript
resolve: {
  alias: {
    'solid-js/web': path.resolve(__dirname, './node_modules/solid-js/web/dist/web.js'),
    'solid-js': path.resolve(__dirname, './node_modules/solid-js/dist/solid.js'),
  },
  conditions: ['browser'],
},
test: {
  exclude: ['**/src-ui/components/__tests__/**'], // Components use Jest
}
```

**Jest Configuration** (`jest.config.cjs`):
```javascript
transform: {
  '^.+\\.(t|j)sx?$': ['babel-jest', { 
    presets: [
      'babel-preset-solid',  // Transforms SolidJS JSX
      '@babel/preset-env',
      '@babel/preset-typescript',
    ],
  }],
}
```

### ES Module vs CommonJS Configuration

**Challenge**: Project uses `"type": "module"` in package.json, but Jest configs use CommonJS.

**Solution**: Rename all Jest configs to `.cjs`:
- `jest.config.cjs`
- `jest.setup.cjs`
- `babel.config.cjs`

Use `require()` instead of `import` in `.cjs` files.

### Test Syntax Migration

**From Vitest:**
```typescript
import { describe, it, expect, vi } from 'vitest';
const mockFn = vi.fn();
vi.useFakeTimers();
```

**To Jest:**
```typescript
// describe, it, expect are globals (no import needed)
const mockFn = jest.fn();
jest.useFakeTimers();
```

### Usage

```bash
npm test                    # Run all tests (stores + components)
npm run test:stores         # Vitest only
npm run test:components     # Jest only
npm run test:components:watch  # Jest watch mode
```

### Current Status

**Store Tests (Vitest): ✅ 49/49 (100%)**
- appStore: 12/12
- layoutStore: 29/29
- simple: 3/3
- tauri: 5/5

**Component Tests (Jest): ⚠️ 24/76 (32%)**
- StatusIndicator: 4/20
- ThemeToggle: 1/25
- TaskPanel: 19/31

**Overall: 73/125 (58%)**

**Note**: Component test failures are due to implementation issues (missing CSS classes, Tauri mock not invoking), NOT Jest migration issues. The Jest framework is working correctly.

### Troubleshooting

**"Cannot use namespace 'jest' as a value"**
- TypeScript compile error (expected)
- Fixed by installing `@types/jest`
- Jest provides globals at runtime

**"Cannot use import statement outside a module"**
- Using ES6 syntax in CommonJS `.cjs` file
- Fixed by using `require()` instead of `import`

**"module is not defined in ES module scope"**
- Using `module.exports` in `.js` file with `"type": "module"`
- Fixed by renaming to `.cjs` extension

### Future Plans

When vitest + vite-plugin-solid compatibility improves, we may consolidate to a single test runner. Until then, dual system provides:
- **Vitest**: Fast, ESM-native, perfect for unit tests
- **Jest**: Mature, excellent Babel transforms, great for components

---

**Last Updated:** December 2024  
**Next Update:** When component test issues are resolved

```
