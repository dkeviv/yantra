// File: src-ui/stores/terminalStore.ts
// Purpose: Multi-terminal management with intelligent command execution
// Dependencies: solid-js
// Last Updated: November 23, 2025

import { createSignal } from 'solid-js';

export interface Terminal {
  id: string;
  name: string;
  status: 'idle' | 'busy' | 'error';
  currentCommand: string | null;
  output: string[];
  createdAt: Date;
  lastUsed: Date;
}

// Terminal instances
const [terminals, setTerminals] = createSignal<Terminal[]>([
  {
    id: 'terminal-1',
    name: 'Terminal 1',
    status: 'idle',
    currentCommand: null,
    output: ['Welcome to Yantra Terminal 1'],
    createdAt: new Date(),
    lastUsed: new Date(),
  },
]);

const [activeTerminalId, setActiveTerminalId] = createSignal<string>('terminal-1');

export const terminalStore = {
  // Getters
  terminals,
  activeTerminalId,

  // Get active terminal
  getActiveTerminal: () => {
    const id = activeTerminalId();
    return terminals().find(t => t.id === id);
  },

  // Get terminal by ID
  getTerminal: (id: string) => {
    return terminals().find(t => t.id === id);
  },

  // Find idle terminal
  findIdleTerminal: () => {
    return terminals().find(t => t.status === 'idle');
  },

  // Get all idle terminals
  getIdleTerminals: () => {
    return terminals().filter(t => t.status === 'idle');
  },

  // Get all busy terminals
  getBusyTerminals: () => {
    return terminals().filter(t => t.status === 'busy');
  },

  // Create new terminal
  createTerminal: (name?: string) => {
    const terminalCount = terminals().length;
    const newId = `terminal-${terminalCount + 1}`;
    const newTerminal: Terminal = {
      id: newId,
      name: name || `Terminal ${terminalCount + 1}`,
      status: 'idle',
      currentCommand: null,
      output: [`Welcome to ${name || `Terminal ${terminalCount + 1}`}`],
      createdAt: new Date(),
      lastUsed: new Date(),
    };

    setTerminals([...terminals(), newTerminal]);
    setActiveTerminalId(newId);
    return newId;
  },

  // Close terminal
  closeTerminal: (id: string) => {
    const updated = terminals().filter(t => t.id !== id);
    setTerminals(updated);

    // Switch to another terminal if active was closed
    if (activeTerminalId() === id && updated.length > 0) {
      setActiveTerminalId(updated[0].id);
    }
  },

  // Set active terminal
  setActiveTerminal: (id: string) => {
    if (terminals().find(t => t.id === id)) {
      setActiveTerminalId(id);
    }
  },

  // Update terminal status
  setTerminalStatus: (id: string, status: 'idle' | 'busy' | 'error') => {
    const updated = terminals().map(t =>
      t.id === id
        ? { ...t, status, lastUsed: new Date() }
        : t
    );
    setTerminals(updated);
  },

  // Set current command
  setCurrentCommand: (id: string, command: string | null) => {
    const updated = terminals().map(t =>
      t.id === id
        ? { ...t, currentCommand: command, lastUsed: new Date() }
        : t
    );
    setTerminals(updated);
  },

  // Add output to terminal
  addOutput: (id: string, output: string) => {
    const updated = terminals().map(t =>
      t.id === id
        ? { ...t, output: [...t.output, output] }
        : t
    );
    setTerminals(updated);
  },

  // Clear terminal output
  clearOutput: (id: string) => {
    const updated = terminals().map(t =>
      t.id === id
        ? { ...t, output: [] }
        : t
    );
    setTerminals(updated);
  },

  // Intelligent command execution
  // Returns terminal ID where command will be executed, or null if all busy
  executeCommand: async (command: string, preferredTerminalId?: string): Promise<string | null> => {
    // Check if preferred terminal is available
    if (preferredTerminalId) {
      const terminal = terminals().find(t => t.id === preferredTerminalId);
      if (terminal && terminal.status === 'idle') {
        terminalStore.setTerminalStatus(preferredTerminalId, 'busy');
        terminalStore.setCurrentCommand(preferredTerminalId, command);
        terminalStore.addOutput(preferredTerminalId, `$ ${command}`);
        return preferredTerminalId;
      }
    }

    // Find any idle terminal
    const idleTerminal = terminalStore.findIdleTerminal();
    if (idleTerminal) {
      terminalStore.setTerminalStatus(idleTerminal.id, 'busy');
      terminalStore.setCurrentCommand(idleTerminal.id, command);
      terminalStore.addOutput(idleTerminal.id, `$ ${command}`);
      return idleTerminal.id;
    }

    // All terminals busy - create new one
    const newId = terminalStore.createTerminal();
    terminalStore.setTerminalStatus(newId, 'busy');
    terminalStore.setCurrentCommand(newId, command);
    terminalStore.addOutput(newId, `$ ${command}`);
    return newId;
  },

  // Complete command execution
  completeCommand: (id: string, output: string, success: boolean = true) => {
    terminalStore.addOutput(id, output);
    terminalStore.setTerminalStatus(id, success ? 'idle' : 'error');
    terminalStore.setCurrentCommand(id, null);
  },

  // Check if command can be executed (at least one terminal available)
  canExecuteCommand: () => {
    return terminals().some(t => t.status === 'idle') || terminals().length < 10;
  },

  // Get terminal statistics
  getStats: () => {
    const allTerminals = terminals();
    return {
      total: allTerminals.length,
      idle: allTerminals.filter(t => t.status === 'idle').length,
      busy: allTerminals.filter(t => t.status === 'busy').length,
      error: allTerminals.filter(t => t.status === 'error').length,
    };
  },
};
