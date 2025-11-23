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

// Panel visibility
const [showCode, setShowCode] = createSignal<boolean>(true);
const [showPreview, setShowPreview] = createSignal<boolean>(true);
const [showFileTree, setShowFileTree] = createSignal<boolean>(true);

// Open files management
const [openFiles, setOpenFiles] = createSignal<Array<{path: string, name: string, content: string}>>([]);
const [activeFileIndex, setActiveFileIndex] = createSignal<number>(-1);

export const appStore = {
  // Getters
  messages,
  currentCode,
  projectPath,
  isGenerating,
  chatWidth,
  codeWidth,
  previewWidth,
  showCode,
  showPreview,
  showFileTree,
  openFiles,
  activeFileIndex,

  // Setters
  setMessages,
  setCurrentCode,
  setProjectPath,
  setIsGenerating,
  setShowCode,
  setShowPreview,
  setShowFileTree,
  setOpenFiles,
  setActiveFileIndex,
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

  // File management actions
  openFile: (path: string, name: string, content: string) => {
    const files = openFiles();
    const existingIndex = files.findIndex(f => f.path === path);
    
    if (existingIndex >= 0) {
      // File already open, just switch to it
      setActiveFileIndex(existingIndex);
      setCurrentCode(files[existingIndex].content);
    } else {
      // Add new file
      const newFiles = [...files, { path, name, content }];
      setOpenFiles(newFiles);
      setActiveFileIndex(newFiles.length - 1);
      setCurrentCode(content);
    }
  },

  closeFile: (index: number) => {
    const files = openFiles();
    const newFiles = files.filter((_, i) => i !== index);
    setOpenFiles(newFiles);
    
    const activeIdx = activeFileIndex();
    if (index === activeIdx) {
      // Closing active file, switch to previous or next
      const newActiveIdx = Math.max(0, Math.min(activeIdx, newFiles.length - 1));
      setActiveFileIndex(newFiles.length > 0 ? newActiveIdx : -1);
      if (newFiles.length > 0) {
        setCurrentCode(newFiles[newActiveIdx].content);
      } else {
        setCurrentCode('# No file open\n');
      }
    } else if (index < activeIdx) {
      // Adjust active index if closing file before it
      setActiveFileIndex(activeIdx - 1);
    }
  },

  switchToFile: (index: number) => {
    const files = openFiles();
    if (index >= 0 && index < files.length) {
      setActiveFileIndex(index);
      setCurrentCode(files[index].content);
    }
  },
};
