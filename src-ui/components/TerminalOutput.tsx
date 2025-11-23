// File: src-ui/components/TerminalOutput.tsx
// Purpose: Terminal output panel with real-time streaming and command execution
// Dependencies: solid-js, @tauri-apps/api/event
// Last Updated: November 22, 2025

import { Component, createSignal, onMount, onCleanup, For, Show } from 'solid-js';
import { listen } from '@tauri-apps/api/event';

interface OutputLine {
  id: number;
  text: string;
  type: 'stdout' | 'stderr' | 'command' | 'info' | 'error' | 'success';
  timestamp: Date;
}

interface ExecutionStatus {
  isRunning: boolean;
  command: string | null;
  exitCode: number | null;
  startTime: Date | null;
  endTime: Date | null;
}

const TerminalOutput: Component = () => {
  const [outputLines, setOutputLines] = createSignal<OutputLine[]>([]);
  const [executionStatus, setExecutionStatus] = createSignal<ExecutionStatus>({
    isRunning: false,
    command: null,
    exitCode: null,
    startTime: null,
    endTime: null,
  });
  const [autoScroll, setAutoScroll] = createSignal(true);
  const [searchTerm, setSearchTerm] = createSignal('');
  const [showTimestamps, setShowTimestamps] = createSignal(false);
  
  let outputContainer: HTMLDivElement | undefined;
  let lineCounter = 0;

  // Add output line
  const addLine = (text: string, type: OutputLine['type']) => {
    const line: OutputLine = {
      id: lineCounter++,
      text,
      type,
      timestamp: new Date(),
    };
    
    setOutputLines((prev) => [...prev, line]);
    
    // Auto-scroll to bottom if enabled
    if (autoScroll() && outputContainer) {
      setTimeout(() => {
        outputContainer!.scrollTop = outputContainer!.scrollHeight;
      }, 10);
    }
  };

  // Listen for terminal events from Tauri
  onMount(async () => {
    // Listen for stdout
    const unlistenStdout = await listen<string>('terminal-stdout', (event) => {
      addLine(event.payload, 'stdout');
    });

    // Listen for stderr
    const unlistenStderr = await listen<string>('terminal-stderr', (event) => {
      addLine(event.payload, 'stderr');
    });

    // Listen for command start
    const unlistenStart = await listen<{ command: string }>('terminal-start', (event) => {
      addLine(`> ${event.payload.command}`, 'command');
      setExecutionStatus({
        isRunning: true,
        command: event.payload.command,
        exitCode: null,
        startTime: new Date(),
        endTime: null,
      });
    });

    // Listen for command completion
    const unlistenEnd = await listen<{ exit_code: number }>('terminal-end', (event) => {
      const status = executionStatus();
      const endTime = new Date();
      const duration = status.startTime 
        ? ((endTime.getTime() - status.startTime.getTime()) / 1000).toFixed(2)
        : '0.00';
      
      if (event.payload.exit_code === 0) {
        addLine(`✓ Command completed successfully (${duration}s)`, 'success');
      } else {
        addLine(`✗ Command failed with exit code ${event.payload.exit_code} (${duration}s)`, 'error');
      }
      
      setExecutionStatus({
        isRunning: false,
        command: status.command,
        exitCode: event.payload.exit_code,
        startTime: status.startTime,
        endTime,
      });
    });

    // Cleanup listeners
    onCleanup(() => {
      unlistenStdout();
      unlistenStderr();
      unlistenStart();
      unlistenEnd();
    });
  });

  // Clear output
  const clearOutput = () => {
    setOutputLines([]);
    lineCounter = 0;
  };

  // Copy all output to clipboard
  const copyToClipboard = async () => {
    const text = outputLines()
      .map((line) => line.text)
      .join('\n');
    
    try {
      await navigator.clipboard.writeText(text);
      addLine('✓ Output copied to clipboard', 'info');
    } catch (err) {
      addLine('✗ Failed to copy to clipboard', 'error');
    }
  };

  // Filter lines by search term
  const filteredLines = () => {
    const term = searchTerm().toLowerCase();
    if (!term) return outputLines();
    
    return outputLines().filter((line) =>
      line.text.toLowerCase().includes(term)
    );
  };

  // Format timestamp
  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // Get line color class
  const getLineColorClass = (type: OutputLine['type']) => {
    switch (type) {
      case 'stdout':
        return 'text-gray-200';
      case 'stderr':
        return 'text-red-400';
      case 'command':
        return 'text-blue-400 font-semibold';
      case 'info':
        return 'text-cyan-400';
      case 'error':
        return 'text-red-500 font-semibold';
      case 'success':
        return 'text-green-400 font-semibold';
      default:
        return 'text-gray-200';
    }
  };

  // Handle scroll to detect manual scrolling
  const handleScroll = () => {
    if (!outputContainer) return;
    
    const { scrollTop, scrollHeight, clientHeight } = outputContainer;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    
    if (!isAtBottom && autoScroll()) {
      setAutoScroll(false);
    }
  };

  // Scroll to bottom button
  const scrollToBottom = () => {
    if (outputContainer) {
      outputContainer.scrollTop = outputContainer.scrollHeight;
      setAutoScroll(true);
    }
  };

  return (
    <div class="h-full flex flex-col bg-gray-900">
      {/* Header */}
      <div class="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4 justify-between">
        <div class="flex items-center space-x-3">
          <svg class="w-5 h-5 text-primary-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <h3 class="text-sm font-semibold text-gray-200">Terminal Output</h3>
          
          {/* Execution Status */}
          <Show when={executionStatus().isRunning}>
            <div class="flex items-center space-x-2">
              <div class="animate-spin h-4 w-4 border-2 border-primary-500 border-t-transparent rounded-full" />
              <span class="text-xs text-gray-400">Running...</span>
            </div>
          </Show>
          
          <Show when={!executionStatus().isRunning && executionStatus().exitCode !== null}>
            <div class="flex items-center space-x-2">
              <Show when={executionStatus().exitCode === 0}>
                <svg class="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
                <span class="text-xs text-green-500">Success</span>
              </Show>
              <Show when={executionStatus().exitCode !== 0}>
                <svg class="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                </svg>
                <span class="text-xs text-red-500">Failed (exit {executionStatus().exitCode})</span>
              </Show>
            </div>
          </Show>
        </div>

        {/* Actions */}
        <div class="flex items-center space-x-2">
          {/* Search */}
          <input
            type="text"
            placeholder="Search output..."
            class="px-3 py-1 text-xs bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-primary-500 text-gray-200"
            value={searchTerm()}
            onInput={(e) => setSearchTerm(e.currentTarget.value)}
          />
          
          {/* Timestamps Toggle */}
          <button
            class={`px-2 py-1 text-xs rounded transition-colors ${
              showTimestamps() ? 'bg-primary-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
            onClick={() => setShowTimestamps(!showTimestamps())}
            title="Toggle timestamps"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>

          {/* Copy Button */}
          <button
            class="px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600 transition-colors text-gray-300"
            onClick={copyToClipboard}
            title="Copy to clipboard"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </button>

          {/* Clear Button */}
          <button
            class="px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600 transition-colors text-gray-300"
            onClick={clearOutput}
            title="Clear output"
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      {/* Output Container */}
      <div
        ref={outputContainer}
        class="flex-1 overflow-y-auto p-4 font-mono text-sm"
        onScroll={handleScroll}
      >
        <Show
          when={filteredLines().length > 0}
          fallback={
            <div class="flex flex-col items-center justify-center h-full text-gray-500">
              <svg class="w-16 h-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p class="text-sm">No output yet</p>
              <p class="text-xs mt-2">Command output will appear here</p>
            </div>
          }
        >
          <For each={filteredLines()}>
            {(line) => (
              <div class="flex items-start space-x-3 mb-1 hover:bg-gray-800 px-2 py-1 rounded">
                <Show when={showTimestamps()}>
                  <span class="text-xs text-gray-500 flex-shrink-0">
                    {formatTimestamp(line.timestamp)}
                  </span>
                </Show>
                <span class={`flex-1 whitespace-pre-wrap break-words ${getLineColorClass(line.type)}`}>
                  {line.text}
                </span>
              </div>
            )}
          </For>
        </Show>
      </div>

      {/* Scroll to Bottom Button */}
      <Show when={!autoScroll() && outputLines().length > 0}>
        <div class="absolute bottom-4 right-4">
          <button
            class="px-3 py-2 bg-primary-600 text-white rounded-full shadow-lg hover:bg-primary-700 transition-colors flex items-center space-x-2"
            onClick={scrollToBottom}
          >
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            <span class="text-xs">Scroll to bottom</span>
          </button>
        </div>
      </Show>
    </div>
  );
};

export default TerminalOutput;
