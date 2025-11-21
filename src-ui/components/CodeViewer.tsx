// File: src-ui/components/CodeViewer.tsx
// Purpose: Code viewer component with Monaco Editor for syntax highlighting
// Dependencies: solid-js, appStore, monaco-editor
// Last Updated: November 20, 2025

import { Component, onMount, createEffect, onCleanup } from 'solid-js';
import { appStore } from '../stores/appStore';
import { monaco } from '../monaco-setup';

const CodeViewer: Component = () => {
  let editorContainer: HTMLDivElement | undefined;
  let editor: monaco.editor.IStandaloneCodeEditor | undefined;

  onMount(() => {
    if (!editorContainer) return;

    // Configure Monaco Editor theme
    monaco.editor.defineTheme('yantra-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [],
      colors: {
        'editor.background': '#111827', // gray-900
        'editor.lineHighlightBackground': '#1f2937', // gray-800
      },
    });

    // Create the editor instance
    editor = monaco.editor.create(editorContainer, {
      value: appStore.currentCode() || '# Write or generate Python code here\n',
      language: 'python',
      theme: 'yantra-dark',
      automaticLayout: true,
      fontSize: 14,
      lineNumbers: 'on',
      minimap: {
        enabled: true,
      },
      scrollBeyondLastLine: false,
      wordWrap: 'on',
      formatOnPaste: true,
      formatOnType: true,
      tabSize: 4,
      insertSpaces: true,
      readOnly: false,
    });

    // Listen to content changes
    editor.onDidChangeModelContent(() => {
      if (editor) {
        appStore.updateCode(editor.getValue());
      }
    });
  });

  // Update editor when code changes externally
  createEffect(() => {
    const code = appStore.currentCode();
    if (editor && code !== editor.getValue()) {
      editor.setValue(code);
    }
  });

  // Cleanup on unmount
  onCleanup(() => {
    editor?.dispose();
  });

  const handleCopy = () => {
    if (editor) {
      const code = editor.getValue();
      navigator.clipboard.writeText(code);
    }
  };

  const handleSave = () => {
    if (editor) {
      const code = editor.getValue();
      // TODO: Implement save to file via Tauri command
      console.log('Save file:', code);
    }
  };

  return (
    <div class="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div class="px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold text-white">Code</h2>
        <p class="text-sm text-gray-400 mt-1">Generated Python code</p>
      </div>

      {/* Monaco Editor */}
      <div ref={editorContainer} class="flex-1" />

      {/* Footer */}
      <div class="px-6 py-3 border-t border-gray-700 flex justify-between items-center">
        <span class="text-sm text-gray-400">
          {appStore.projectPath() || 'No project loaded'}
        </span>
        <div class="flex space-x-2">
          <button 
            onClick={handleCopy}
            class="px-4 py-2 text-sm bg-gray-800 text-gray-200 rounded hover:bg-gray-700 transition-colors"
          >
            Copy
          </button>
          <button 
            onClick={handleSave}
            class="px-4 py-2 text-sm bg-gray-800 text-gray-200 rounded hover:bg-gray-700 transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default CodeViewer;
