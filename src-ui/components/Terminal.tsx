// File: src-ui/components/Terminal.tsx
// Purpose: Real PTY-based terminal using xterm.js
// Dependencies: solid-js, xterm, @tauri-apps/api
// Last Updated: November 28, 2025

import { Component, onMount, onCleanup, createSignal } from 'solid-js';
import { invoke } from '@tauri-apps/api/tauri';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { Terminal as XTerm } from 'xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import 'xterm/css/xterm.css';

interface TerminalProps {
  terminalId: string;
  name: string;
}

export const Terminal: Component<TerminalProps> = (props) => {
  let terminalRef: HTMLDivElement | undefined;
  let xterm: XTerm | undefined;
  let fitAddon: FitAddon | undefined;
  const [isReady, setIsReady] = createSignal(false);
  
  let unlistenData: UnlistenFn | undefined;
  let unlistenClosed: UnlistenFn | undefined;

  onMount(async () => {
    if (!terminalRef) return;

    // Create xterm instance
    xterm = new XTerm({
      fontFamily: '"Cascadia Code", Menlo, Monaco, "Courier New", monospace',
      fontSize: 13,
      theme: {
        background: '#000000',
        foreground: '#ffffff',
        cursor: '#ffffff',
      },
      cursorBlink: true,
      cursorStyle: 'block',
      scrollback: 10000,
      tabStopWidth: 4,
    });

    // Add addons
    fitAddon = new FitAddon();
    xterm.loadAddon(fitAddon);
    xterm.loadAddon(new WebLinksAddon());

    // Open terminal in DOM
    xterm.open(terminalRef);
    fitAddon.fit();

    // Create PTY terminal on backend
    try {
      await invoke('create_pty_terminal', {
        terminalId: props.terminalId,
        name: props.name,
        shell: null, // Use default shell
      });
      setIsReady(true);
    } catch (error) {
      console.error('Failed to create PTY terminal:', error);
      xterm.writeln(`\x1b[31mError: Failed to create terminal: ${error}\x1b[0m`);
      return;
    }

    // Handle user input
    xterm.onData(async (data) => {
      if (!isReady()) return;
      
      try {
        // Convert string to base64 for transmission
        const encoded = btoa(data);
        await invoke('write_pty_input', {
          terminalId: props.terminalId,
          data: encoded,
        });
      } catch (error) {
        console.error('Failed to write to terminal:', error);
      }
    });

    // Listen for terminal output
    unlistenData = await listen<{ terminal_id: string; data: string }>(
      'terminal-data',
      (event) => {
        if (event.payload.terminal_id === props.terminalId && xterm) {
          // Decode base64 data and write to terminal
          const decoded = atob(event.payload.data);
          xterm.write(decoded);
        }
      }
    );

    // Listen for terminal close
    unlistenClosed = await listen<{ terminal_id: string }>(
      'terminal-closed',
      (event) => {
        if (event.payload.terminal_id === props.terminalId && xterm) {
          xterm.writeln('\r\n\x1b[33m[Process completed]\x1b[0m');
          setIsReady(false);
        }
      }
    );

    // Handle terminal resize
    const resizeObserver = new ResizeObserver(() => {
      if (fitAddon && xterm) {
        fitAddon.fit();
        // Notify backend of new size
        const dims = fitAddon.proposeDimensions();
        if (dims) {
          invoke('resize_pty_terminal', {
            terminalId: props.terminalId,
            rows: dims.rows,
            cols: dims.cols,
          }).catch(console.error);
        }
      }
    });

    if (terminalRef) {
      resizeObserver.observe(terminalRef);
    }

    onCleanup(() => {
      resizeObserver.disconnect();
    });
  });

  onCleanup(async () => {
    // Clean up event listeners
    if (unlistenData) unlistenData();
    if (unlistenClosed) unlistenClosed();

    // Close terminal on backend
    try {
      await invoke('close_pty_terminal', {
        terminalId: props.terminalId,
      });
    } catch (error) {
      console.error('Failed to close terminal:', error);
    }

    // Dispose xterm
    if (xterm) {
      xterm.dispose();
    }
  });

  return (
    <div
      ref={terminalRef}
      class="w-full h-full bg-black"
    />
  );
};

export default Terminal;
