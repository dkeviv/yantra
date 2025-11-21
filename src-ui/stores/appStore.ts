// File: src-ui/stores/appStore.ts
// Purpose: Global application state management
// Dependencies: solid-js
// Last Updated: November 20, 2025

import { createSignal } from 'solid-js';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export interface AppState {
  messages: Message[];
  currentCode: string;
  projectPath: string | null;
  isGenerating: boolean;
}

// Chat messages
const [messages, setMessages] = createSignal<Message[]>([
  {
    id: '1',
    role: 'system',
    content: 'Welcome to Yantra! I\'m your AI coding assistant. Tell me what you want to build, and I\'ll generate production-quality Python code that never breaks.',
    timestamp: new Date(),
  },
]);

// Current code in the editor
const [currentCode, setCurrentCode] = createSignal<string>('# Your generated code will appear here\n');

// Currently loaded project path
const [projectPath, setProjectPath] = createSignal<string | null>(null);

// Loading/generating state
const [isGenerating, setIsGenerating] = createSignal<boolean>(false);

// Panel widths (percentages)
const [chatWidth, setChatWidth] = createSignal<number>(60);
const [codeWidth, setCodeWidth] = createSignal<number>(25);
const [previewWidth, setPreviewWidth] = createSignal<number>(15);

export const appStore = {
  // Getters
  messages,
  currentCode,
  projectPath,
  isGenerating,
  chatWidth,
  codeWidth,
  previewWidth,

  // Setters
  setMessages,
  setCurrentCode,
  setProjectPath,
  setIsGenerating,
  setChatWidth,
  setCodeWidth,
  setPreviewWidth,

  // Actions
  addMessage: (role: 'user' | 'assistant' | 'system', content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      role,
      content,
      timestamp: new Date(),
    };
    setMessages([...messages(), newMessage]);
  },

  clearMessages: () => {
    setMessages([]);
  },

  updateCode: (code: string) => {
    setCurrentCode(code);
  },

  loadProject: (path: string) => {
    setProjectPath(path);
  },
};
