// File: src-ui/App.tsx
// Purpose: Main application component with resizable 5-panel layout
// Dependencies: solid-js, stores/appStore, components
// Last Updated: November 23, 2025

import { Component, createSignal, onMount, Show } from 'solid-js';
import { appStore } from './stores/appStore';
import { terminalStore } from './stores/terminalStore';
import { listen } from '@tauri-apps/api/event';
import FileTree from './components/FileTree';
import ChatPanel from './components/ChatPanel';
import CodeViewer from './components/CodeViewer';
import MultiTerminal from './components/MultiTerminal';
import DependencyGraph from './components/DependencyGraph';
import { AgentStatus } from './components/AgentStatus';
import { Notifications } from './components/Notifications';
import DocumentationPanels from './components/DocumentationPanels';

const App: Component = () => {
  const [isDragging, setIsDragging] = createSignal<number | null>(null);
  const [terminalHeight, setTerminalHeight] = createSignal(30); // Terminal height in %
  const [showDocsPanels, setShowDocsPanels] = createSignal(false); // Toggle between Files and Docs

  // Handle panel resizing
  const handleMouseDown = (panelIndex: number) => (e: MouseEvent) => {
    e.preventDefault();
    setIsDragging(panelIndex);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging() === null) return;

    if (isDragging() === 3) {
      // Dragging terminal vertical divider
      const containerHeight = window.innerHeight - 56; // Subtract top bar height
      const mouseY = e.clientY - 56; // Adjust for top bar
      const percentage = (mouseY / containerHeight) * 100;
      const newMainHeight = Math.min(Math.max(percentage, 50), 85); // Main area: 50-85%
      const newTerminalHeight = 100 - newMainHeight;
      
      setTerminalHeight(newTerminalHeight);
      return;
    }

    const containerWidth = window.innerWidth;
    const mouseX = e.clientX;
    const percentage = (mouseX / containerWidth) * 100;

    if (isDragging() === 1) {
      // Dragging chat-code divider
      const newChatWidth = Math.min(Math.max(percentage, 40), 75);
      const remaining = 100 - newChatWidth;
      const codePercentage = (appStore.codeWidth() / (appStore.codeWidth() + appStore.previewWidth())) * remaining;
      
      appStore.setChatWidth(newChatWidth);
      appStore.setCodeWidth(codePercentage);
      appStore.setPreviewWidth(remaining - codePercentage);
    } else if (isDragging() === 2) {
      // Dragging code-preview divider
      const chatW = appStore.chatWidth();
      const remaining = 100 - chatW;
      const codeFromStart = percentage - chatW;
      const newCodeWidth = Math.min(Math.max(codeFromStart, remaining * 0.3), remaining * 0.7);
      
      appStore.setCodeWidth(newCodeWidth);
      appStore.setPreviewWidth(remaining - newCodeWidth);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(null);
  };

  onMount(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Initialize terminal event listeners
    terminalStore.initializeEventListeners();

    // Listen for menu events
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

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      unlistenTogglePanel.then(fn => fn());
      unlistenShowView.then(fn => fn());
      unlistenResetLayout.then(fn => fn());
    };
  });

  return (
    <div class="h-screen w-screen bg-gray-900 text-white overflow-hidden">
      {/* Notifications Overlay */}
      <Notifications />
      
      {/* Top Bar - Removed redundant Yantra branding and Open Project button (agent-driven) */}
      <div class="h-14 bg-gray-800 border-b border-gray-700 flex items-center px-6">
        <div class="text-sm text-gray-400">
          {appStore.projectPath() ? (
            <span class="text-primary-400">Project: {appStore.projectPath()}</span>
          ) : (
            <span>Use "open project [path]" command to load a project</span>
          )}
        </div>
      </div>

      {/* Main Layout - 3 Column Design */}
      <div class="flex h-[calc(100vh-3.5rem)]">
        {/* Left Column - File Tree OR Documentation Panels (20% width) */}
        <Show when={appStore.showFileTree()}>
          <div class="w-64 flex flex-col bg-gray-800 border-r border-gray-700">
            {/* Toggle Buttons */}
            <div class="flex border-b border-gray-700">
              <button
                onClick={() => setShowDocsPanels(false)}
                class={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                  !showDocsPanels()
                    ? 'bg-gray-700 text-white border-b-2 border-primary-500'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                📁 Files
              </button>
              <button
                onClick={() => setShowDocsPanels(true)}
                class={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                  showDocsPanels()
                    ? 'bg-gray-700 text-white border-b-2 border-primary-500'
                    : 'text-gray-400 hover:text-white'
                }`}
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
          </div>
        </Show>

        {/* Resize Handle FileTree-Chat */}
        <Show when={appStore.showFileTree()}>
          <div class="w-1 bg-gray-700 hover:bg-primary-500 transition-colors cursor-col-resize" />
        </Show>

        {/* Center Column - Chat Panel (Full Height, 45% default) */}
        <div class="flex-1 flex flex-col" style={{ width: `${appStore.chatWidth()}%` }}>
          <ChatPanel />
        </div>

        {/* Resize Handle Chat-Code */}
        <Show when={appStore.showCode()}>
          <div
            class="w-1 resize-handle cursor-col-resize hover:bg-primary-500 transition-colors bg-gray-700"
            onMouseDown={handleMouseDown(1)}
          />
        </Show>

        {/* Right Column - Code + Terminal Stack (35% default) */}
        <Show when={appStore.showCode()}>
          <div class="flex flex-col" style={{ width: `${appStore.codeWidth()}%` }}>
            {/* View Selector Tabs */}
            <div class="flex bg-gray-800 border-b border-gray-700">
              <button
                class={`px-4 py-2 text-sm transition-colors ${
                  appStore.activeView() === 'editor'
                    ? 'bg-gray-900 text-white border-b-2 border-primary-500'
                    : 'text-gray-400 hover:text-white'
                }`}
                onClick={() => appStore.setActiveView('editor')}
              >
                📝 Editor
              </button>
              <button
                class={`px-4 py-2 text-sm transition-colors ${
                  appStore.activeView() === 'dependencies'
                    ? 'bg-gray-900 text-white border-b-2 border-primary-500'
                    : 'text-gray-400 hover:text-white'
                }`}
                onClick={() => appStore.setActiveView('dependencies')}
              >
                🔗 Dependencies
              </button>
            </div>

            {/* Code Viewer - resizable height */}
            <div class="flex-1 overflow-hidden" style={{ height: `${100 - terminalHeight()}%` }}>
              <Show when={appStore.activeView() === 'editor'}>
                <CodeViewer />
              </Show>
              <Show when={appStore.activeView() === 'dependencies'}>
                <DependencyGraph />
              </Show>
            </div>

            {/* Horizontal Resize Handle for Terminal */}
            <div
              class="h-1 resize-handle cursor-row-resize hover:bg-primary-500 transition-colors bg-gray-700"
              onMouseDown={handleMouseDown(2)}
            />

            {/* Terminal Output - in same column as code */}
            <Show when={terminalHeight() > 0}>
              <div class="overflow-hidden" style={{ height: `${terminalHeight()}%` }}>
                <MultiTerminal />
              </div>
            </Show>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default App;
