// File: src-ui/components/CodeViewer.tsx
// Purpose: Code viewer component with Monaco Editor and file tabs
// Dependencies: solid-js, appStore, monaco-editor
// Last Updated: November 23, 2025

import { Component, onMount, createEffect, onCleanup, For, Show } from 'solid-js';
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
      fontSize: 12,
      lineNumbers: 'on',
      lineNumbersMinChars: 3,
      glyphMargin: false,
      folding: true,
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
      padding: {
        top: 4,
        bottom: 4,
      },
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

  return (
    <div class="flex flex-col h-full bg-gray-900">
      {/* File Tabs */}
      <Show when={appStore.openFiles().length > 0}>
        <div class="flex bg-gray-800 border-b border-gray-700 overflow-x-auto">
          <For each={appStore.openFiles()}>
            {(file, index) => (
              <div
                class={`flex items-center px-4 py-2 border-r border-gray-700 cursor-pointer transition-colors ${
                  index() === appStore.activeFileIndex()
                    ? 'bg-gray-900 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
                onClick={() => appStore.switchToFile(index())}
              >
                <span class="text-sm mr-2">{file.name}</span>
                <button
                  class="text-gray-500 hover:text-white"
                  onClick={(e) => {
                    e.stopPropagation();
                    appStore.closeFile(index());
                  }}
                >
                  ×
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>

      {/* Header */}
        {/* Editor */}      {/* Monaco Editor */}
      <div ref={editorContainer} class="flex-1" />
    </div>
  );
};

export default CodeViewer;
