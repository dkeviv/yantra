# Browswer Integration

**Option A: Bundled Chromium + CDP**

Ship Chromium with Yantra. Control via CDP. +150MB install size. Zero external dependencies. Guaranteed identical behavior everywhere. User never thinks about browsers.

**Option B: System Chrome + CDP (Seleected)**

Find user's installed Chrome. Control via CDP. ~0MB added install size. Requires user to have Chrome (most developers do). Fallback: download Chromium on first use if no Chrome.

**Option C: System WebView (Limited) (Not enough for Yantra)**

Use OS WebView (like VS Code Simple Browser). No CDP, limited control. Can't capture console, can't automate interactions. Not suitable for Yantra's needs.

---

## Real Choice: Option A vs Option B

| Factor          | Option A (Bundled)   | Option B (System Chrome)     |
| --------------- | -------------------- | ---------------------------- |
| Install size    | +150MB               | +0MB (or +100MB fallback)    |
| Setup           | Zero touch           | Zero touch (usually)         |
| Reliability     | 100% works           | 99% works (Chrome common)    |
| Version control | Yantra controls      | User's Chrome version        |
| Updates         | Yantra ships updates | Chrome updates independently |
| Offline install | Yes                  | Maybe (if fallback needed)   |

---

## Recommendation

**Option B with Smart Fallback**

Most developers have Chrome. Use it, zero added size. If no Chrome found, download Chromium automatically. Download happens once, in background. User barely notices.

**Why Not Bundle (Option A)**

150MB is significant for installer. Developers already have Chrome. Unnecessary duplication.

**When to Reconsider Option A**

If user complaints about "Chrome not found" are frequent. If Chrome version incompatibilities cause issues. For enterprise "air-gapped" deployments.

---

## Zero-Touch Flow with Option B

**First Launch**

```
Yantra starts
    ↓
Check for Chrome/Chromium/Edge
    ↓
Found → Store path, done
    ↓
Not found → Show "Downloading browser engine..."
    ↓
Download minimal Chromium (~100MB) to app data
    ↓
Done, never ask again
```

**Every Subsequent Launch**

```
Yantra starts
    ↓
Browser path already known
    ↓
Ready instantly
```

**During Development**

```
Agent generates frontend code
    ↓
Yantra starts dev server
    ↓
Yantra launches Chrome via CDP (hidden from user)
    ↓
Preview appears in Yantra panel
    ↓
Errors flow to agent automatically
```

User touches nothing. User configures nothing. Browser integration just works.

## Zero-Touch CDP Setup

**What User Experiences**

Opens Yantra. Opens project. Agent generates frontend code. Browser preview appears with running app. That's it.

**What Happens Behind the Scenes**

Yantra detects frontend project. Yantra starts dev server. Yantra finds or downloads Chrome. Yantra launches Chrome with CDP enabled. Yantra connects via CDP. Yantra injects runtime. Preview appears in Yantra panel. Agent receives error stream.

User touches nothing.

---

## Chrome Discovery (Zero Install)

**Step 1: Find Existing Chrome**

Most developers already have Chrome installed.

```
macOS: /Applications/Google Chrome.app
Windows: C:\Program Files\Google\Chrome\Application\chrome.exe
Linux: /usr/bin/google-chrome or /usr/bin/chromium
```

Yantra checks these paths on startup. If found, use it. No download needed.

**Step 2: Fallback to Other Browsers**

No Chrome? Check for:

```
Chromium (same CDP protocol)
Microsoft Edge (Chromium-based, supports CDP)
Brave (Chromium-based, supports CDP)
```

Any Chromium-based browser works.

**Step 3: Last Resort - Auto Download**

No Chromium browser found. Yantra downloads minimal Chromium (~100MB). Downloads to Yantra's app data folder. One-time, happens in background. User sees: "Setting up browser preview..." then done.

**User Never**

Installs Chrome manually for Yantra. Configures browser paths. Manages browser versions. Thinks about CDP.

---

## Chrome Launch (Invisible)

**How Yantra Launches Chrome**

```rust
// Simplified - actual implementation
fn launch_chrome_for_preview(url: &str) -> Result<ChromeConnection> {
    let chrome_path = find_chrome()?;
  
    let process = Command::new(chrome_path)
        .args([
            "--remote-debugging-port=0",    // Random available port
            "--no-first-run",               // Skip welcome screens
            "--no-default-browser-check",   // Skip default browser prompt
            "--disable-extensions",          // Faster startup
            "--disable-popup-blocking",      // Allow popups for OAuth
            "--window-size=1280,720",       // Reasonable default
            "--app={}",                      // App mode - minimal UI
            url
        ])
        .spawn()?;
  
    let debug_port = discover_debug_port(&process)?;
    let connection = connect_cdp(debug_port)?;
  
    Ok(connection)
}
```

**App Mode**

`--app=URL` launches Chrome without address bar, tabs, bookmarks. Looks like native window. Feels integrated, not "external browser."

**Random Debug Port**

`--remote-debugging-port=0` picks available port. No conflicts with other Chrome instances. Yantra discovers port from Chrome's output.

---

## Dev Server Management

**Auto-Start Dev Server**

Yantra detects framework. Yantra knows the command:

```
Next.js → npm run dev
Vite → npm run dev
CRA → npm start
```

Yantra runs command in background. Yantra waits for server to be ready. Yantra detects port from output.

**Port Detection**

Parse dev server output:

```
"ready on http://localhost:3000" → port 3000
"Local: http://localhost:5173" → port 5173
```

Or try common ports: 3000, 3001, 5173, 8080.

**Server Health Check**

Yantra pings `http://localhost:{port}` until it responds. Then launches browser. No race conditions.

---

## Runtime Injection via CDP

**Inject Script Before Page Loads**

CDP allows injecting JavaScript that runs before page scripts.

```rust
async fn inject_runtime(cdp: &CdpConnection) -> Result<()> {
    // Inject Yantra runtime on every page load
    cdp.send("Page.addScriptToEvaluateOnNewDocument", json!({
        "source": include_str!("yantra-runtime.js")
    })).await?;
  
    Ok(())
}
```

**No Proxy Needed**

CDP injects directly. No proxy server to run. No port conflicts. Cleaner than proxy approach.

**Runtime Script Is Embedded**

`yantra-runtime.js` compiled into Yantra binary. No external files to manage. No CDN dependencies.

---

## Error Capture via CDP

**Console Messages**

```rust
async fn listen_for_errors(cdp: &CdpConnection) -> Result<()> {
    cdp.subscribe("Runtime.consoleAPICalled", |event| {
        if event.type_ == "error" {
            send_to_agent(ConsoleError {
                message: event.args,
                stack_trace: event.stack_trace,
                url: event.url,
                line: event.line_number,
            });
        }
    }).await?;
  
    cdp.subscribe("Runtime.exceptionThrown", |event| {
        send_to_agent(UnhandledException {
            message: event.exception_details.text,
            stack_trace: event.exception_details.stack_trace,
            // ...
        });
    }).await?;
  
    Ok(())
}
```

**Network Failures**

```rust
cdp.subscribe("Network.loadingFailed", |event| {
    send_to_agent(NetworkError {
        url: event.request.url,
        error: event.error_text,
    });
}).await?;
```

**CDP Gives Everything**

Console logs, errors, warnings. Unhandled exceptions with stack traces. Network requests and failures.


---


**Current Flow**

Agent generates UI. Browser shows preview. User sees something they want to change. User types in chat: "Change the hero image to something more professional." Agent guesses which element user means. Often wrong. Frustrating back-and-forth.

**Ideal Flow**

User clicks element directly in browser preview. Selection highlighted. User says: "Make this more professional." Agent knows exactly what element. Precise change. No guessing.

---

## Interactive Browser Architecture

**Bidirectional Communication**

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Yantra    │◄───────►│   Yantra    │◄───────►│   Browser   │
│    Chat     │         │   Backend   │   CDP   │   Preview   │
└─────────────┘         └─────────────┘         └─────────────┘
                              │
                              │ WebSocket
                              ▼
                        ┌─────────────┐
                        │   Yantra    │
                        │   Runtime   │
                        │  (injected) │
                        └─────────────┘
```

**Three Communication Channels**

Yantra → Browser (via CDP): Navigate, click, execute scripts.
Browser → Yantra (via Runtime): User interactions, selections, events.
Yantra ↔ Chat: User intent, agent responses.

---

## Injected Runtime Capabilities

**Selection Mode**

User clicks "Select Element" in Yantra. Runtime activates selection mode. User hovers → elements highlight. User clicks → element selected. Selection info sent to Yantra.

**Selection Data Captured**

```javascript
// yantra-runtime.js
function captureSelection(element) {
    return {
        // Identity
        selector: generateUniqueSelector(element),
        componentName: getReactComponentName(element),
        sourceLocation: getSourceMapLocation(element),
      
        // Visual
        tagName: element.tagName,
        className: element.className,
        boundingBox: element.getBoundingClientRect(),
        screenshot: captureElementScreenshot(element),
      
        // Content
        textContent: element.textContent?.slice(0, 100),
        src: element.src,  // For images
        href: element.href, // For links
      
        // Context
        parentComponent: getParentComponentName(element),
        siblingCount: element.parentElement?.children.length,
      
        // Asset info (if applicable)
        assetType: detectAssetType(element), // 'icon', 'image', 'illustration'
        assetSource: element.dataset.yantraAsset, // 'lucide', 'unsplash', etc.
    };
}
```

**Source Map Integration**

```javascript
function getSourceMapLocation(element) {
    // React DevTools-style fiber lookup
    const fiber = element._reactFiber || element.__reactFiber;
    if (fiber) {
        const source = fiber._debugSource;
        return {
            fileName: source.fileName,
            lineNumber: source.lineNumber,
            columnNumber: source.columnNumber,
            componentName: fiber.type?.name
        };
    }
    return null;
}
```

Agent knows exactly which file and line to modify.

---

## User Interaction Flows

**Flow 1: Select and Describe**

```
User: [Clicks "Select" mode in Yantra]
User: [Clicks hero image in browser]
        ↓
Browser highlights image, sends selection to Yantra
        ↓
Chat shows: [Image element selected: hero-image.jpg from HeroSection.tsx:24]
        ↓
User types: "Make this more energetic and colorful"
        ↓
Agent knows:
  - Exact element (img.hero-image)
  - Source file (HeroSection.tsx, line 24)
  - Current asset (Unsplash photo)
  - What to change
        ↓
Agent: Searches Unsplash for "energetic colorful team"
       OR generates with DALL-E if no match
       Updates code, browser refreshes
        ↓
User sees new image instantly
```

**Flow 2: Quick Actions Menu**

```
User: [Right-clicks element in browser]
        ↓
Context menu appears:
┌────────────────────────┐
│ 🔄 Replace this image  │
│ 🎨 Change colors       │
│ ✏️ Edit text           │
│ 🗑️ Remove element      │
│ 📋 Duplicate           │
│ 💬 Describe change...  │
└────────────────────────┘
        ↓
User clicks "Replace this image"
        ↓
Asset picker opens:
┌─────────────────────────────────────────────┐
│ Replace Hero Image                    [×]   │
├─────────────────────────────────────────────┤
│ Current: team-meeting.jpg (Unsplash)        │
│                                             │
│ Search: [energetic team collaboration    ]  │
│                                             │
│ Results:                                    │
│ ┌─────┬─────┬─────┬─────┐                  │
│ │     │     │     │     │                  │
│ │     │     │     │     │                  │
│ └─────┴─────┴─────┴─────┘                  │
│                                             │
│ [Search Unsplash] [Generate Custom] [Upload]│
└─────────────────────────────────────────────┘
        ↓
User clicks alternative
        ↓
Image swapped, code updated, preview refreshes
```

**Flow 3: Natural Language with Selection**

```
User: [Clicks icon in navigation]
Browser: Highlights icon, shows tooltip "Settings icon (Lucide)"
        ↓
User types: "This should be a gear, not a cog"
        ↓
Agent understands:
  - Element: Settings icon
  - Current: lucide:cog
  - Request: Change to gear variant
        ↓
Agent: Swaps to lucide:settings-2 (gear style)
Code updated, preview refreshes
```

**Flow 4: Multi-Select**

```
User: [Shift+clicks multiple elements]
Browser: Highlights all selected
        ↓
User types: "Make all these icons blue"
        ↓
Agent: Updates className for all selected icons
       Adds text-blue-500 to each
       Preview refreshes
```

---

## Selection Mode UI

**Yantra Toolbar:**

```
┌─────────────────────────────────────────────┐
│ Preview                    [🖱️ Select] [↻]  │
├─────────────────────────────────────────────┤
│                                             │
│           Browser Preview                   │
│                                             │
└─────────────────────────────────────────────┘
```

**When Select Mode Active:**

```
┌─────────────────────────────────────────────┐
│ Preview               [🖱️ Selecting...] [×] │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │                                         │ │
│ │    ┌─────────────────────┐              │ │
│ │    │   [Hovered Element] │ ← Blue border│ │
│ │    │    highlighted      │              │ │
│ │    └─────────────────────┘              │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Click element to select, ESC to cancel      │
└─────────────────────────────────────────────┘
```

**After Selection:**

```
┌─────────────────────────────────────────────┐
│ Preview                    [🖱️ Select] [↻]  │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │                                         │ │
│ │    ╔═════════════════════╗              │ │
│ │    ║   Selected Element  ║ ← Green border│
│ │    ║      (locked)       ║              │ │
│ │    ╚═════════════════════╝              │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Selected: <img> in HeroSection.tsx:24       │
│ [Replace] [Edit] [Remove] [Deselect]        │
└─────────────────────────────────────────────┘
```

---

## Chat Integration

**Selection Appears in Chat:**

```
┌─────────────────────────────────────────────┐
│ Chat                                        │
├─────────────────────────────────────────────┤
│                                             │
│ You selected:                               │
│ ┌─────────────────────────────────────────┐ │
│ │ [thumbnail]  Hero Image                 │ │
│ │              HeroSection.tsx:24         │ │
│ │              Source: Unsplash           │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ What would you like to change?              │
│                                             │
│ Quick actions:                              │
│ [Replace Image] [Resize] [Add Effect]       │
│                                             │
│ Or describe what you want...                │
│                                             │
│ [Type message...]                           │
└─────────────────────────────────────────────┘
```

**User Types Change:**

```
User: "Make it darker and add a gradient overlay"

Agent: I'll update the hero image:
       1. Apply brightness filter (darker)
       2. Add gradient overlay from transparent to black
     
       [Shows code diff]
     
       [Apply] [Modify] [Cancel]
```

---

## Runtime Implementation

**Injected Script:**

```javascript
// yantra-runtime.js

class YantraInteraction {
    constructor() {
        this.selectionMode = false;
        this.selectedElement = null;
        this.ws = new WebSocket('ws://localhost:YANTRA_PORT');
    }
  
    enableSelectionMode() {
        this.selectionMode = true;
        document.body.style.cursor = 'crosshair';
        document.addEventListener('mouseover', this.handleHover);
        document.addEventListener('click', this.handleClick);
        document.addEventListener('keydown', this.handleKeydown);
    }
  
    handleHover = (e) => {
        if (!this.selectionMode) return;
      
        // Remove previous highlight
        document.querySelectorAll('.yantra-hover').forEach(el => {
            el.classList.remove('yantra-hover');
        });
      
        // Add highlight to current
        e.target.classList.add('yantra-hover');
      
        // Send hover info to Yantra (for tooltip)
        this.ws.send(JSON.stringify({
            type: 'element-hover',
            data: this.getElementInfo(e.target)
        }));
    }
  
    handleClick = (e) => {
        if (!this.selectionMode) return;
      
        e.preventDefault();
        e.stopPropagation();
      
        this.selectedElement = e.target;
        this.selectedElement.classList.add('yantra-selected');
      
        // Send selection to Yantra
        this.ws.send(JSON.stringify({
            type: 'element-selected',
            data: this.captureSelection(e.target)
        }));
      
        this.disableSelectionMode();
    }
  
    handleContextMenu = (e) => {
        e.preventDefault();
      
        const elementInfo = this.getElementInfo(e.target);
      
        // Send to Yantra to show context menu
        this.ws.send(JSON.stringify({
            type: 'context-menu',
            data: {
                element: elementInfo,
                position: { x: e.clientX, y: e.clientY }
            }
        }));
    }
  
    getElementInfo(element) {
        return {
            tagName: element.tagName,
            className: element.className,
            id: element.id,
            textContent: element.textContent?.slice(0, 50),
            isImage: element.tagName === 'IMG',
            isIcon: this.detectIcon(element),
            src: element.src,
            rect: element.getBoundingClientRect()
        };
    }
  
    captureSelection(element) {
        return {
            ...this.getElementInfo(element),
            selector: this.generateSelector(element),
            sourceLocation: this.getSourceLocation(element),
            screenshot: this.captureScreenshot(element),
            computedStyles: this.getRelevantStyles(element),
            assetInfo: this.getAssetInfo(element)
        };
    }
  
    generateSelector(element) {
        // Generate unique CSS selector
        if (element.id) return `#${element.id}`;
      
        // Use data-testid if available
        if (element.dataset.testid) {
            return `[data-testid="${element.dataset.testid}"]`;
        }
      
        // Use React component name if available
        const componentName = this.getReactComponentName(element);
        if (componentName) {
            return `[data-component="${componentName}"]`;
        }
      
        // Fallback to path-based selector
        return this.getPathSelector(element);
    }
  
    getSourceLocation(element) {
        // Try React DevTools approach
        const fiber = element._reactFiber$ || 
                      Object.keys(element).find(k => k.startsWith('__reactFiber'));
      
        if (fiber) {
            const fiberNode = element[fiber];
            if (fiberNode._debugSource) {
                return {
                    file: fiberNode._debugSource.fileName,
                    line: fiberNode._debugSource.lineNumber,
                    component: fiberNode.type?.name
                };
            }
        }
      
        return null;
    }
  
    // Listen for commands from Yantra
    handleCommand(command) {
        switch (command.type) {
            case 'enable-selection':
                this.enableSelectionMode();
                break;
            case 'disable-selection':
                this.disableSelectionMode();
                break;
            case 'highlight-element':
                this.highlightElement(command.selector);
                break;
            case 'scroll-to-element':
                this.scrollToElement(command.selector);
                break;
        }
    }
}

// Initialize
window.yantra = new YantraInteraction();

// Inject styles
const style = document.createElement('style');
style.textContent = `
    .yantra-hover {
        outline: 2px solid #3B82F6 !important;
        outline-offset: 2px;
    }
    .yantra-selected {
        outline: 3px solid #10B981 !important;
        outline-offset: 2px;
    }
`;
document.head.appendChild(style);
```

---

## Yantra Backend Handling

**WebSocket Handler:**

```rust
// src-tauri/src/browser/interaction.rs

pub async fn handle_browser_message(msg: BrowserMessage, state: &AppState) {
    match msg {
        BrowserMessage::ElementHover { data } => {
            // Update UI tooltip with element info
            state.emit("element-hover", data);
        }
      
        BrowserMessage::ElementSelected { data } => {
            // Store selection
            state.selected_element = Some(data.clone());
          
            // Update chat with selection context
            state.emit("element-selected", SelectionContext {
                element: data,
                quick_actions: get_quick_actions(&data),
                source_location: data.source_location,
            });
        }
      
        BrowserMessage::ContextMenu { data } => {
            // Show native context menu or custom UI
            state.emit("show-context-menu", ContextMenuData {
                element: data.element,
                position: data.position,
                actions: get_context_actions(&data.element),
            });
        }
    }
}

fn get_quick_actions(element: &ElementData) -> Vec<QuickAction> {
    let mut actions = vec![];
  
    if element.is_image {
        actions.push(QuickAction::ReplaceImage);
        actions.push(QuickAction::AddFilter);
        actions.push(QuickAction::Resize);
    }
  
    if element.is_icon {
        actions.push(QuickAction::SwapIcon);
        actions.push(QuickAction::ChangeColor);
        actions.push(QuickAction::ChangeSize);
    }
  
    if element.text_content.is_some() {
        actions.push(QuickAction::EditText);
    }
  
    actions.push(QuickAction::ChangeStyle);
    actions.push(QuickAction::Remove);
    actions.push(QuickAction::Duplicate);
  
    actions
}
```

---

## Agent Context Enhancement

**When User Sends Message with Selection:**

```rust
fn build_agent_context(message: &str, selection: Option<&ElementSelection>) -> AgentContext {
    let mut context = AgentContext::new(message);
  
    if let Some(sel) = selection {
        context.add_context(format!(
            "User has selected an element:
            - Type: {}
            - Location: {}:{} 
            - Current content/src: {}
            - CSS classes: {}
            - Component: {}
          
            User's request applies to this specific element.",
            sel.tag_name,
            sel.source_location.file,
            sel.source_location.line,
            sel.content_or_src,
            sel.class_name,
            sel.component_name
        ));
      
        // Include relevant source code
        let source_snippet = read_source_lines(
            &sel.source_location.file,
            sel.source_location.line - 5,
            sel.source_location.line + 10
        );
        context.add_context(format!("Relevant source code:\n{}", source_snippet));
    }
  
    context
}
```

**Agent Response:**

Agent knows exactly what element. Agent knows exact file and line. Agent makes precise change. No guessing, no back-and-forth.

---

## Advanced: Visual Feedback Loop

**Before/After Preview:**

```
User selects image, asks for change
        ↓
Agent generates new version
        ↓
Preview shows split view:
┌─────────────────────────────────────────────┐
│         Before          │       After       │
│    [Current Image]      │   [New Image]     │
│                         │                   │
└─────────────────────────────────────────────┘

[← Keep Original] [Apply Change →] [Try Different]
```

**Undo/Redo Stack:**

Every change recorded. User can undo any change. Visual history of modifications.

```
Change History:
├── 3. Changed hero image (2 min ago) [Undo]
├── 2. Updated icon color (5 min ago) [Undo]
└── 1. Initial generation (10 min ago)
```

---

## Summary

**What We Built**

Injected runtime enables element selection. WebSocket connects browser to Yantra. CDP provides full control. Chat integrates with selections. Agent gets precise context.

**User Experience**

Click element → describe change → see result. No guessing which element. No describing location. Direct manipulation + natural language.

**Technical Flow**

```
User clicks element in browser
        ↓
Runtime captures: selector, source location, asset info
        ↓
WebSocket sends to Yantra backend
        ↓
Chat shows selection with quick actions
        ↓
User describes change
        ↓
Agent has full context: element + source + user intent
        ↓
Agent makes precise code change
        ↓
Browser hot-reloads
        ↓
User sees result instantly
```

This transforms the browser from passive preview to interactive design surface. User points and speaks. Agent understands and executes.
