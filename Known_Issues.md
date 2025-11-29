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

**Last Updated:** November 20, 2025  
**Next Update:** When issues are discovered
