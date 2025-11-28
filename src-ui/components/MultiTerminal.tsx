// File: src-ui/components/MultiTerminal.tsx
// Purpose: Multi-terminal component with VSCode-style interface
// Dependencies: solid-js, terminalStore
// Last Updated: November 28, 2025

import { Component, For, Show, createSignal } from 'solid-js';
import { terminalStore } from '../stores/terminalStore';

const MultiTerminal: Component = () => {
  const [commandInput, setCommandInput] = createSignal('');
  const [showTerminalDropdown, setShowTerminalDropdown] = createSignal(false);

  const handleExecuteCommand = async () => {
    const command = commandInput().trim();
    if (!command) return;

    const terminalId = await terminalStore.executeCommand(command);
    if (terminalId) {
      // Simulate command execution (in reality, this would call Tauri)
      // For now, just mark as complete after 2 seconds
      setTimeout(() => {
        terminalStore.completeCommand(terminalId, `Command executed: ${command}`, true);
      }, 2000);
      
      setCommandInput('');
    } else {
      alert('All terminals are busy. Please wait or create a new terminal.');
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleExecuteCommand();
    }
  };

  return (
    <div class="flex flex-col h-full bg-gray-900 font-mono">
      {/* Title Bar with Controls */}
      <div class="flex items-center justify-between bg-gray-800 border-b border-gray-700 px-3 py-1">
        <div class="flex items-center gap-2">
          {/* Terminal selector dropdown */}
          <div class="relative">
            <button
              onClick={() => setShowTerminalDropdown(!showTerminalDropdown())}
              class="flex items-center gap-1 px-2 py-1 text-xs text-gray-300 hover:bg-gray-700 rounded transition-colors"
            >
              <span>{terminalStore.getActiveTerminal()?.name || 'Terminal'}</span>
              <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {/* Dropdown menu */}
            <Show when={showTerminalDropdown()}>
              <div class="absolute top-full left-0 mt-1 bg-gray-800 border border-gray-700 rounded shadow-lg z-10 min-w-[150px]">
                <For each={terminalStore.terminals()}>
                  {(terminal) => (
                    <button
                      onClick={() => {
                        terminalStore.setActiveTerminal(terminal.id);
                        setShowTerminalDropdown(false);
                      }}
                      class={`w-full text-left px-3 py-2 text-xs flex items-center justify-between hover:bg-gray-700 transition-colors ${
                        terminalStore.activeTerminalId() === terminal.id ? 'bg-gray-700' : ''
                      }`}
                    >
                      <span class="text-gray-300">{terminal.name}</span>
                      <span
                        class={`w-1.5 h-1.5 rounded-full ${
                          terminal.status === 'idle'
                            ? 'bg-green-500'
                            : terminal.status === 'busy'
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                      />
                    </button>
                  )}
                </For>
              </div>
            </Show>
          </div>

          {/* Terminal count indicator */}
          <span class="text-xs text-gray-500">
            ({terminalStore.terminals().length})
          </span>
        </div>

        {/* Action buttons */}
        <div class="flex items-center gap-1">
          {/* New terminal */}
          <button
            onClick={() => terminalStore.createTerminal()}
            class="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
            title="New Terminal"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
          </button>

          {/* Delete terminal */}
          <Show when={terminalStore.terminals().length > 1}>
            <button
              onClick={() => {
                const terminal = terminalStore.getActiveTerminal();
                if (terminal) {
                  terminalStore.closeTerminal(terminal.id);
                }
              }}
              class="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Close Terminal"
            >
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </Show>

          {/* Clear output */}
          <button
            onClick={() => {
              const terminal = terminalStore.getActiveTerminal();
              if (terminal) {
                terminalStore.clearOutput(terminal.id);
              }
            }}
            class="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
            title="Clear Output"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Terminal Body */}
      <div class="flex-1 overflow-y-auto p-2 text-sm bg-black">
        <Show
          when={terminalStore.getActiveTerminal()}
          fallback={<div class="text-gray-500">No terminal active</div>}
        >
          {(terminal) => (
            <div class="space-y-1">
              {/* Output lines */}
              <For each={terminal().output}>
                {(line) => (
                  <div class="text-gray-300">{line}</div>
                )}
              </For>

              {/* Current command indicator */}
              <Show when={terminal().currentCommand}>
                <div class="text-yellow-400">
                  <span class="text-green-400">$ </span>
                  {terminal().currentCommand}
                  <span class="animate-pulse">_</span>
                </div>
              </Show>

              {/* Input prompt */}
              <div class="flex items-center">
                <span class="text-green-400 mr-2">$</span>
                <input
                  type="text"
                  value={commandInput()}
                  onInput={(e) => setCommandInput(e.currentTarget.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type command here..."
                  aria-label="Terminal command input"
                  class="flex-1 bg-transparent text-gray-300 outline-none border-none"
                  disabled={!terminalStore.canExecuteCommand()}
                  autofocus
                />
              </div>
            </div>
          )}
        </Show>
      </div>
    </div>
  );
};

export default MultiTerminal;
