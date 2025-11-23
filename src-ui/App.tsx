// File: src-ui/App.tsx
// Purpose: Main application component with resizable 5-panel layout
// Dependencies: solid-js, stores/appStore, components
// Last Updated: November 23, 2025

import { Component, createSignal, onMount, Show } from 'solid-js';
import { appStore } from './stores/appStore';
import FileTree from './components/FileTree';
import ChatPanel from './components/ChatPanel';
import CodeViewer from './components/CodeViewer';
import TerminalOutput from './components/TerminalOutput';
import { AgentStatus } from './components/AgentStatus';
import { Notifications } from './components/Notifications';

const App: Component = () => {
  const [isDragging, setIsDragging] = createSignal<number | null>(null);
  const [terminalHeight, setTerminalHeight] = createSignal(30); // Terminal height in %

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

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  });

  return (
    <div class="h-screen w-screen bg-gray-900 text-white overflow-hidden">
      {/* Notifications Overlay */}
      <Notifications />
      
      {/* Top Bar */}
      <div class="h-14 bg-gray-800 border-b border-gray-700 flex items-center px-6">
        <div class="flex items-center space-x-3">
          <svg
            class="w-8 h-8 text-primary-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
            />
          </svg>
          <h1 class="text-xl font-bold">Yantra</h1>
        </div>
        <div class="ml-auto flex items-center space-x-4">
          <button
            class="px-4 py-2 text-sm bg-gray-700 rounded hover:bg-gray-600 transition-colors"
            onClick={() => {
              // TODO: Implement project loading
              appStore.loadProject('/path/to/project');
            }}
          >
            Open Project
          </button>
          <div class="text-sm text-gray-400">
            {appStore.projectPath() ? (
              <span class="text-primary-400">Project loaded</span>
            ) : (
              'No project'
            )}
          </div>
        </div>
      </div>

      {/* Main Layout - 3 Column Design */}
      <div class="flex h-[calc(100vh-3.5rem)]">
        {/* Left Column - File Tree (20% width) */}
        <Show when={appStore.showFileTree()}>
          <div class="w-64 flex flex-col bg-gray-800 border-r border-gray-700">
            <div class="flex-1 overflow-y-auto">
              <FileTree />
            </div>
            {/* Agent Status at bottom */}
            <div class="border-t border-gray-700 p-2">
              <AgentStatus />
            </div>
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
            {/* Code Viewer - resizable height */}
            <div class="flex-1 overflow-hidden" style={{ height: `${100 - terminalHeight()}%` }}>
              <CodeViewer />
            </div>

            {/* Horizontal Resize Handle for Terminal */}
            <div
              class="h-1 resize-handle cursor-row-resize hover:bg-primary-500 transition-colors bg-gray-700"
              onMouseDown={handleMouseDown(2)}
            />

            {/* Terminal Output - in same column as code */}
            <div class="overflow-hidden" style={{ height: `${terminalHeight()}%` }}>
              <TerminalOutput />
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default App;
