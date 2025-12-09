// File: src-ui/App.tsx
// Purpose: Main application component with resizable 5-panel layout
// Dependencies: solid-js, stores/appStore, components
// Last Updated: November 23, 2025

import { Component, createSignal, onMount, Show } from 'solid-js';
import { appStore } from './stores/appStore';
import { terminalStore } from './stores/terminalStore';
import { layoutStore } from './stores/layoutStore';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/tauri';
import FileTree from './components/FileTree';
import ChatPanel from './components/ChatPanel';
import CodeViewer from './components/CodeViewer';
import Terminal from './components/Terminal';
import GraphViewer from './components/GraphViewer';
import { AgentStatus } from './components/AgentStatus';
import { Notifications } from './components/Notifications';
import DocumentationPanels from './components/DocumentationPanels';
import ArchitectureView from './components/ArchitectureView';
import ThemeToggle from './components/ThemeToggle';
import TaskPanel from './components/TaskPanel';

const App: Component = () => {
  const [isDragging, setIsDragging] = createSignal<number | null>(null);
  const [terminalHeight, setTerminalHeight] = createSignal(0); // Terminal hidden by default
  const [showDocsPanels, setShowDocsPanels] = createSignal(false); // Toggle between Files and Docs
  const [showTaskPanel, setShowTaskPanel] = createSignal(false);
  const [isResizingFileExplorer, setIsResizingFileExplorer] = createSignal(false);

  // Handle panel resizing
  const handleMouseDown = (panelIndex: number) => (e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(panelIndex);

    // Add dragging class to body for global cursor control
    if (panelIndex === 3) {
      document.body.classList.add('dragging-vertical');
    } else {
      document.body.classList.add('dragging-horizontal');
    }
  };

  // Handle File Explorer resize
  const handleFileExplorerResize = (e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizingFileExplorer(true);
    document.body.classList.add('dragging-horizontal');
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging() === null && !isResizingFileExplorer()) return;

    e.preventDefault();

    // Handle File Explorer width resizing
    if (isResizingFileExplorer()) {
      const newWidth = e.clientX;
      const windowWidth = window.innerWidth;
      const tenPercent = windowWidth * 0.1;

      // Auto-close if dragged to less than 10% of window width
      if (newWidth < tenPercent) {
        appStore.setShowFileTree(false);
        setIsResizingFileExplorer(false);
        document.body.classList.remove('dragging-horizontal');
        return;
      }

      layoutStore.updateFileExplorerWidth(newWidth);
      return;
    }

    if (isDragging() === 3) {
      // Dragging terminal horizontal divider
      const topBarHeight = 40; // Top bar with YANTRA title
      const containerHeight = window.innerHeight - topBarHeight;
      const mouseY = e.clientY - topBarHeight;
      const percentage = (mouseY / containerHeight) * 100;
      const newMainHeight = Math.min(Math.max(percentage, 50), 85); // Main area: 50-85%
      const newTerminalHeight = 100 - newMainHeight;

      setTerminalHeight(newTerminalHeight);
      return;
    }

    // For vertical dividers (0 and 1), we need to account for FileTree fixed width
    // Get the actual FileTree element width to be precise
    const fileTreeElement = document.querySelector('.w-64');
    const fileTreeWidth = fileTreeElement ? fileTreeElement.getBoundingClientRect().width : 0;
    const availableWidth = window.innerWidth - fileTreeWidth;
    const mouseXRelative = e.clientX - fileTreeWidth; // Mouse position relative to resizable area

    if (isDragging() === 0) {
      // Dragging FileTree-Chat divider (not used since FileTree is fixed width)
      // This shouldn't happen, but keeping for completeness
      return;
    }

    if (isDragging() === 1) {
      // Dragging chat-code divider
      const windowWidth = window.innerWidth;
      const tenPercent = windowWidth * 0.1;

      // Calculate percentage of available width (excluding FileTree)
      const percentage = (mouseXRelative / availableWidth) * 100;

      // Check if chat or code should auto-close
      const chatAbsoluteWidth = (availableWidth * percentage) / 100;
      const codeAbsoluteWidth = availableWidth - chatAbsoluteWidth;

      // Auto-close chat if less than 10% of total window
      if (chatAbsoluteWidth < tenPercent) {
        // Close chat panel - not implementing yet, would need UI control
        console.log('Chat panel would auto-close here');
      }

      // Auto-close code editor if less than 10% of total window
      if (codeAbsoluteWidth < tenPercent) {
        appStore.setShowCode(false);
        setIsDragging(null);
        document.body.classList.remove('dragging-horizontal');
        return;
      }

      const newChatWidth = Math.min(Math.max(percentage, 20), 80); // Chat: 20-80% of available width
      const newCodeWidth = 100 - newChatWidth; // Code gets the rest

      appStore.setChatWidth(newChatWidth);
      appStore.setCodeWidth(newCodeWidth);
      appStore.setPreviewWidth(0); // Not used in current layout
    } else if (isDragging() === 2) {
      // Dragging code-preview divider (if needed in future)
      const chatW = appStore.chatWidth();
      const remaining = 100 - chatW;
      const percentage = (mouseXRelative / availableWidth) * 100;
      const codeFromStart = percentage - chatW;
      const newCodeWidth = Math.min(Math.max(codeFromStart, remaining * 0.3), remaining * 0.7);

      appStore.setCodeWidth(newCodeWidth);
      appStore.setPreviewWidth(remaining - newCodeWidth);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(null);
    setIsResizingFileExplorer(false);
    // Remove dragging classes from body
    document.body.classList.remove('dragging-horizontal', 'dragging-vertical');
  };

  onMount(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Listen for auto-close event from layoutStore
    const handleCloseFileExplorer = () => {
      appStore.setShowFileTree(false);
    };
    window.addEventListener('close-file-explorer', handleCloseFileExplorer);

    // Keyboard shortcut for terminal toggle (Cmd+`)
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey && e.key === '`') {
        e.preventDefault();
        setTerminalHeight(terminalHeight() > 0 ? 0 : 30);
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    // Initialize terminal event listeners
    terminalStore.initializeEventListeners();

    // Listen for menu events
    const unlistenMenuNewFile = listen('menu-new-file', async () => {
      // Create a new untitled file
      appStore.addMessage('system', 'Creating new file...');
      // TODO: Implement new file creation
    });

    const unlistenMenuNewFolder = listen('menu-new-folder', async () => {
      appStore.addMessage('system', 'Creating new folder...');
      // TODO: Implement new folder creation
    });

    const unlistenMenuOpenFolder = listen('menu-open-folder', async () => {
      try {
        const { open } = await import('@tauri-apps/api/dialog');
        const selected = await open({
          directory: true,
          multiple: false,
        });

        if (selected && typeof selected === 'string') {
          appStore.setProjectPath(selected);
          appStore.addMessage('system', `Opened project: ${selected}`);
          // Trigger project analysis
          await invoke('analyze_project', { projectPath: selected });
        }
      } catch (error) {
        appStore.addMessage('system', `Error opening folder: ${error}`);
      }
    });

    const unlistenMenuSave = listen('menu-save', async () => {
      appStore.addMessage('system', 'Saving current file...');
      // TODO: Implement save current file
    });

    const unlistenMenuSaveAll = listen('menu-save-all', async () => {
      appStore.addMessage('system', 'Saving all files...');
      // TODO: Implement save all files
    });

    const unlistenMenuCloseFolder = listen('menu-close-folder', () => {
      // Clear project path
      appStore.setProjectPath(null);
      // Clear all open files
      appStore.setOpenFiles([]);
      appStore.setActiveFileIndex(-1);
      // Clear current code
      appStore.setCurrentCode('# Your generated code will appear here\n');
      // Emit event to FileTree to clear its state
      window.dispatchEvent(new CustomEvent('close-project'));
      // Add confirmation message
      appStore.addMessage('system', '✅ Project folder closed. Open a new project to get started.');
    });

    const unlistenMenuFind = listen('menu-find', () => {
      appStore.addMessage('system', 'Find functionality - coming soon');
      // TODO: Implement find dialog
    });

    const unlistenMenuReplace = listen('menu-replace', () => {
      appStore.addMessage('system', 'Replace functionality - coming soon');
      // TODO: Implement replace dialog
    });

    const unlistenTogglePanel = listen<string>('toggle-panel', (event) => {
      const panel = event.payload;
      if (panel === 'fileTree') {
        appStore.setShowFileTree(!appStore.showFileTree());
      } else if (panel === 'codeEditor') {
        appStore.setShowCode(!appStore.showCode());
      } else if (panel === 'terminal') {
        // Toggle terminal visibility
        setTerminalHeight(terminalHeight() > 0 ? 0 : 30);
      }
    });

    const unlistenShowView = listen<string>('show-view', (event) => {
      const view = event.payload;
      if (view === 'dependencies') {
        appStore.setActiveView('dependencies');
      }
    });

    const unlistenResetLayout = listen('reset-layout', () => {
      appStore.setShowFileTree(true);
      appStore.setShowCode(true);
      setTerminalHeight(30);
      appStore.setChatWidth(45);
      appStore.setCodeWidth(35);
    });

    const unlistenMenuCopyCode = listen('menu-copy-code', async () => {
      try {
        const content = appStore.currentCode() || '';
        await navigator.clipboard.writeText(content);
        appStore.addMessage('system', 'Code copied to clipboard');
      } catch (error) {
        appStore.addMessage('system', `Error copying code: ${error}`);
      }
    });

    const unlistenMenuToggleFileTree = listen('menu-toggle-file-tree', () => {
      appStore.setShowFileTree(!appStore.showFileTree());
    });

    const unlistenMenuToggleTerminal = listen('menu-toggle-terminal', () => {
      setTerminalHeight(terminalHeight() > 0 ? 0 : 30);
    });

    const unlistenMenuToggleDependencies = listen('menu-toggle-dependencies', () => {
      appStore.setActiveView('dependencies');
    });

    const unlistenMenuResetLayout = listen('menu-reset-layout', () => {
      appStore.setShowFileTree(true);
      appStore.setShowCode(true);
      setTerminalHeight(30);
      appStore.setChatWidth(45);
      appStore.setCodeWidth(35);
    });

    const unlistenMenuAbout = listen('menu-about', () => {
      appStore.addMessage('system', 'Yantra - AI-First Development Platform\nVersion 0.1.0');
    });

    const unlistenMenuSettings = listen('menu-settings', () => {
      appStore.addMessage('system', 'Settings functionality - coming soon');
      // TODO: Open settings panel
    });

    const unlistenMenuCheckUpdates = listen('menu-check-updates', () => {
      appStore.addMessage('system', 'Checking for updates...');
      // TODO: Implement update check
    });

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('close-file-explorer', handleCloseFileExplorer);
      unlistenMenuNewFile.then((fn) => fn());
      unlistenMenuNewFolder.then((fn) => fn());
      unlistenMenuOpenFolder.then((fn) => fn());
      unlistenMenuSave.then((fn) => fn());
      unlistenMenuSaveAll.then((fn) => fn());
      unlistenMenuCloseFolder.then((fn) => fn());
      unlistenMenuFind.then((fn) => fn());
      unlistenMenuReplace.then((fn) => fn());
      unlistenTogglePanel.then((fn) => fn());
      unlistenShowView.then((fn) => fn());
      unlistenResetLayout.then((fn) => fn());
      unlistenMenuCopyCode.then((fn) => fn());
      unlistenMenuToggleFileTree.then((fn) => fn());
      unlistenMenuToggleTerminal.then((fn) => fn());
      unlistenMenuToggleDependencies.then((fn) => fn());
      unlistenMenuResetLayout.then((fn) => fn());
      unlistenMenuAbout.then((fn) => fn());
      unlistenMenuSettings.then((fn) => fn());
      unlistenMenuCheckUpdates.then((fn) => fn());
    };
  });

  return (
    <div
      class="h-screen w-screen overflow-hidden flex flex-col"
      style={{
        'background-color': 'var(--bg-primary)',
        color: 'var(--text-primary)',
      }}
    >
      {/* Notifications Overlay */}
      <Notifications />

      {/* Top Bar - YANTRA Title */}
      <div
        class="h-10 flex items-center justify-between px-4 flex-shrink-0"
        style={{
          'background-color': 'var(--bg-elevated)',
          'border-bottom': '1px solid var(--border-primary)',
        }}
      >
        <div class="flex items-center gap-3">
          <div class="text-xl font-bold tracking-wider" style={{ color: 'var(--text-primary)' }}>
            YANTRA
          </div>
          <ThemeToggle />
        </div>
        <div class="flex items-center gap-2">
          {/* Task Panel Toggle Button */}
          <button
            onClick={() => setShowTaskPanel(!showTaskPanel())}
            class="px-3 py-1 text-xs rounded transition-colors"
            style={{
              'background-color': showTaskPanel() ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
            }}
            title="Toggle Task Panel"
          >
            {showTaskPanel() ? '📋 Hide Tasks' : '📋 Show Tasks'}
          </button>

          {/* Terminal Toggle Button */}
          <button
            onClick={() => setTerminalHeight(terminalHeight() > 0 ? 0 : 30)}
            class="px-3 py-1 text-xs rounded transition-colors"
            style={{
              'background-color':
                terminalHeight() > 0 ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
            }}
            title="Toggle Terminal (Cmd+`)"
          >
            {terminalHeight() > 0 ? '🖥️ Hide Terminal' : '🖥️ Show Terminal'}
          </button>
        </div>
      </div>

      {/* Main Layout - 3 Column Design */}
      <div class="flex flex-1 overflow-hidden">
        {/* Left Column - File Tree OR Documentation Panels */}
        <Show when={appStore.showFileTree()}>
          <div
            class="flex flex-col relative"
            style={{
              width: layoutStore.isExpanded('fileExplorer')
                ? '70%'
                : `${layoutStore.fileExplorerWidth()}px`,
              transition: 'width 0.3s ease',
              'background-color': 'var(--bg-secondary)',
              'border-right': '1px solid var(--border-primary)',
            }}
          >
            {/* Toggle Buttons */}
            <div class="flex" style={{ 'border-bottom': '1px solid var(--border-primary)' }}>
              <button
                onClick={() => setShowDocsPanels(false)}
                class="flex-1 px-3 py-2 text-sm font-medium transition-colors"
                style={{
                  'background-color': !showDocsPanels() ? 'var(--bg-tertiary)' : 'transparent',
                  color: !showDocsPanels() ? 'var(--text-primary)' : 'var(--text-tertiary)',
                  'border-bottom': !showDocsPanels()
                    ? '2px solid var(--accent-primary)'
                    : '2px solid transparent',
                }}
              >
                📁 Files
              </button>
              <button
                onClick={() => setShowDocsPanels(true)}
                class="flex-1 px-3 py-2 text-sm font-medium transition-colors"
                style={{
                  'background-color': showDocsPanels() ? 'var(--bg-tertiary)' : 'transparent',
                  color: showDocsPanels() ? 'var(--text-primary)' : 'var(--text-tertiary)',
                  'border-bottom': showDocsPanels()
                    ? '2px solid var(--accent-primary)'
                    : '2px solid transparent',
                }}
              >
                📚 Docs
              </button>
            </div>

            {/* Content */}
            <div class="flex-1 overflow-y-auto">
              <Show when={!showDocsPanels()}>
                <FileTree />
              </Show>
              <Show when={showDocsPanels()}>
                <DocumentationPanels />
              </Show>
            </div>
            {/* Agent Status at bottom */}
            <AgentStatus />

            {/* Resize Handle for FileExplorer width - only when not expanded */}
            <Show when={!layoutStore.isExpanded('fileExplorer')}>
              <div
                class="absolute top-0 right-0 w-0.5 h-full cursor-col-resize hover:bg-primary-500 transition-colors"
                style={{ 'background-color': 'var(--border-primary)' }}
                onMouseDown={handleFileExplorerResize}
                title="Drag to resize"
              />
            </Show>
          </div>
        </Show>

        {/* Resize Handle FileTree-Chat */}
        <Show when={appStore.showFileTree()}>
          <div
            class="w-0.5 cursor-col-resize transition-colors flex-shrink-0"
            style={{
              'background-color': 'var(--border-primary)',
              'z-index': '10',
              position: 'relative',
            }}
            onMouseDown={handleMouseDown(0)}
          />
        </Show>

        {/* Center Column - Chat Panel */}
        <div
          class="flex-1 flex flex-col"
          style={{
            width: layoutStore.isExpanded('agent')
              ? '70%'
              : layoutStore.isExpanded('fileExplorer') || layoutStore.isExpanded('editor')
                ? '15%'
                : `${appStore.chatWidth()}%`,
            transition: 'width 0.3s ease',
          }}
        >
          <ChatPanel />
        </div>

        {/* Resize Handle Chat-Code */}
        <Show when={appStore.showCode()}>
          <div
            class="cursor-col-resize transition-colors flex-shrink-0"
            style={{
              'background-color': 'var(--border-primary)',
              'z-index': '100',
              position: 'relative',
              'min-width': '3px',
              width: '3px',
              cursor: 'col-resize',
              'user-select': 'none',
            }}
            onMouseDown={handleMouseDown(1)}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--accent-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--border-primary)';
            }}
          />
        </Show>

        {/* Right Column - Code + Terminal Stack */}
        <Show when={appStore.showCode()}>
          <div
            class="flex flex-col"
            style={{
              width: layoutStore.isExpanded('editor')
                ? '70%'
                : layoutStore.isExpanded('fileExplorer') || layoutStore.isExpanded('agent')
                  ? '15%'
                  : `${appStore.codeWidth()}%`,
              transition: 'width 0.3s ease',
            }}
          >
            {/* View Selector Tabs */}
            {/* View tabs - Minimal UI with inline icons */}
            <div
              class="flex"
              style={{
                'background-color': 'var(--bg-secondary)',
                'border-bottom': '1px solid var(--border-primary)',
              }}
            >
              <button
                class="px-3 py-1.5 text-xs transition-colors cursor-pointer"
                style={{
                  'background-color':
                    appStore.activeView() === 'editor' ? 'var(--bg-primary)' : 'transparent',
                  color:
                    appStore.activeView() === 'editor'
                      ? 'var(--text-primary)'
                      : 'var(--text-tertiary)',
                  'border-bottom':
                    appStore.activeView() === 'editor'
                      ? '2px solid var(--accent-primary)'
                      : '2px solid transparent',
                  'font-weight': appStore.activeView() === 'editor' ? '600' : '400',
                  'pointer-events': 'auto',
                }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('Editor tab clicked, current view:', appStore.activeView());
                  appStore.setActiveView('editor');
                  console.log('After setActiveView, view is:', appStore.activeView());
                }}
                title="Editor"
              >
                <span class="inline-flex items-center gap-1.5">
                  <span>✏️</span>
                  <span>Editor</span>
                </span>
              </button>
              <button
                class="px-3 py-1.5 text-xs transition-colors cursor-pointer"
                style={{
                  'background-color':
                    appStore.activeView() === 'dependencies' ? 'var(--bg-primary)' : 'transparent',
                  color:
                    appStore.activeView() === 'dependencies'
                      ? 'var(--text-primary)'
                      : 'var(--text-tertiary)',
                  'border-bottom':
                    appStore.activeView() === 'dependencies'
                      ? '2px solid var(--accent-primary)'
                      : '2px solid transparent',
                  'font-weight': appStore.activeView() === 'dependencies' ? '600' : '400',
                  'pointer-events': 'auto',
                }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('Dependencies tab clicked, current view:', appStore.activeView());
                  appStore.setActiveView('dependencies');
                  console.log('After setActiveView, view is:', appStore.activeView());
                }}
                title="Dependencies"
              >
                <span class="inline-flex items-center gap-1.5">
                  <span>🔗</span>
                  <span>Deps</span>
                </span>
              </button>
              <button
                class="px-3 py-1.5 text-xs transition-colors cursor-pointer"
                style={{
                  'background-color':
                    appStore.activeView() === 'architecture' ? 'var(--bg-primary)' : 'transparent',
                  color:
                    appStore.activeView() === 'architecture'
                      ? 'var(--text-primary)'
                      : 'var(--text-tertiary)',
                  'border-bottom':
                    appStore.activeView() === 'architecture'
                      ? '2px solid var(--accent-primary)'
                      : '2px solid transparent',
                  'font-weight': appStore.activeView() === 'architecture' ? '600' : '400',
                  'pointer-events': 'auto',
                }}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('Architecture tab clicked, current view:', appStore.activeView());
                  appStore.setActiveView('architecture');
                  console.log('After setActiveView, view is:', appStore.activeView());
                }}
                title="Architecture"
              >
                <span class="inline-flex items-center gap-1.5">
                  <span>🏗️</span>
                  <span>Arch</span>
                </span>
              </button>
            </div>

            {/* Code Viewer - resizable height */}
            <div class="flex-1 overflow-hidden" style={{ height: `${100 - terminalHeight()}%` }}>
              <Show when={appStore.activeView() === 'editor'}>
                <CodeViewer />
              </Show>
              <Show when={appStore.activeView() === 'dependencies'}>
                <GraphViewer />
              </Show>
              <Show when={appStore.activeView() === 'architecture'}>
                <ArchitectureView />
              </Show>
            </div>

            {/* Terminal Output - in same column as code */}
            <Show when={terminalHeight() > 0}>
              {/* Horizontal Resize Handle for Terminal */}
              <div
                class="h-0.5 transition-colors select-none"
                style={{
                  cursor: 'row-resize',
                  'background-color': 'var(--border-primary)',
                }}
                onMouseDown={handleMouseDown(3)}
              />

              <div class="overflow-hidden" style={{ height: `${terminalHeight()}%` }}>
                <Terminal terminalId="terminal-1" name="Terminal 1" />
              </div>
            </Show>
          </div>
        </Show>
      </div>

      {/* Task Panel Overlay */}
      <TaskPanel isOpen={showTaskPanel()} onClose={() => setShowTaskPanel(false)} />
    </div>
  );
};

export default App;
