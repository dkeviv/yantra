// File: src-ui/components/MultiTerminal.tsx
// Purpose: Multi-terminal component with tabs and intelligent execution
// Dependencies: solid-js, terminalStore
// Last Updated: November 23, 2025

import { Component, For, Show, createSignal } from 'solid-js';
import { terminalStore } from '../stores/terminalStore';

const MultiTerminal: Component = () => {
  const [commandInput, setCommandInput] = createSignal('');

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
      handleExecuteCommand();
    }
  };

  return (
    <div class="flex flex-col h-full bg-gray-900">
      {/* Terminal Tabs */}
      <div class="flex bg-gray-800 border-b border-gray-700 overflow-x-auto">
        <For each={terminalStore.terminals()}>
          {(terminal) => (
            <div
              class={`flex items-center px-4 py-2 border-r border-gray-700 cursor-pointer transition-colors ${
                terminalStore.activeTerminalId() === terminal.id
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
              onClick={() => terminalStore.setActiveTerminal(terminal.id)}
            >
              <span class="text-sm mr-2">{terminal.name}</span>
              <span
                class={`w-2 h-2 rounded-full mr-2 ${
                  terminal.status === 'idle'
                    ? 'bg-green-500'
                    : terminal.status === 'busy'
                    ? 'bg-yellow-500 animate-pulse'
                    : 'bg-red-500'
                }`}
              />
              <Show when={terminalStore.terminals().length > 1}>
                <button
                  class="text-gray-500 hover:text-white ml-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    terminalStore.closeTerminal(terminal.id);
                  }}
                >
                  ×
                </button>
              </Show>
            </div>
          )}
        </For>
        
        {/* New Terminal Button */}
        <button
          class="px-4 py-2 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
          onClick={() => terminalStore.createTerminal()}
          title="New Terminal"
        >
          + New
        </button>
      </div>

      {/* Terminal Stats Bar */}
      <div class="px-4 py-2 bg-gray-800 border-b border-gray-700 text-xs text-gray-400">
        <span class="mr-4">
          📊 Total: {terminalStore.getStats().total}
        </span>
        <span class="mr-4 text-green-400">
          ✓ Idle: {terminalStore.getStats().idle}
        </span>
        <span class="mr-4 text-yellow-400">
          ⚡ Busy: {terminalStore.getStats().busy}
        </span>
        <Show when={terminalStore.getStats().error > 0}>
          <span class="text-red-400">
            ✗ Error: {terminalStore.getStats().error}
          </span>
        </Show>
      </div>

      {/* Active Terminal Output */}
      <div class="flex-1 overflow-y-auto p-4 font-mono text-sm">
        <Show
          when={terminalStore.getActiveTerminal()}
          fallback={<div class="text-gray-500">No terminal active</div>}
        >
          {(terminal) => (
            <>
              <div class="text-gray-400 mb-2">
                {terminal().name} - Status: {terminal().status}
              </div>
              <Show when={terminal().currentCommand}>
                <div class="text-yellow-400 mb-2">
                  Running: {terminal().currentCommand}
                </div>
              </Show>
              <For each={terminal().output}>
                {(line) => (
                  <div class="text-gray-300">
                    {line}
                  </div>
                )}
              </For>
            </>
          )}
        </Show>
      </div>

      {/* Command Input */}
      <div class="border-t border-gray-700 p-4 bg-gray-800">
        <div class="flex items-center">
          <span class="text-primary-400 mr-2">$</span>
          <input
            type="text"
            value={commandInput()}
            onInput={(e) => setCommandInput(e.currentTarget.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter command..."
            class="flex-1 bg-gray-900 text-white px-3 py-2 rounded border border-gray-700 focus:border-primary-500 focus:outline-none"
            disabled={!terminalStore.canExecuteCommand()}
          />
          <button
            onClick={handleExecuteCommand}
            disabled={!commandInput().trim() || !terminalStore.canExecuteCommand()}
            class="ml-2 px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
          >
            Execute
          </button>
          <button
            onClick={() => {
              const terminal = terminalStore.getActiveTerminal();
              if (terminal) {
                terminalStore.clearOutput(terminal.id);
              }
            }}
            class="ml-2 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
          >
            Clear
          </button>
        </div>
        <div class="text-xs text-gray-500 mt-2">
          Press Enter to execute • {terminalStore.getIdleTerminals().length} terminal(s) available
        </div>
      </div>
    </div>
  );
};

export default MultiTerminal;
