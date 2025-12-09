// File: src-ui/components/CodeViewer.tsx
// Purpose: Multi-modal code viewer with Monaco Editor for code files and YDocBlockEditor for .ydoc files
// Dependencies: solid-js, appStore, monaco-editor, Tree-sitter completion provider, YDocBlockEditor
// Last Updated: December 9, 2025

import { Component, onMount, createEffect, onCleanup, For, Show } from 'solid-js';
import { appStore } from '../stores/appStore';
import { layoutStore } from '../stores/layoutStore';
import { monaco, registerCompletionProvider } from '../monaco-setup';
import { YDocBlockEditor } from './YDocBlockEditor';

const CodeViewer: Component = () => {
  let editorContainer: HTMLDivElement | undefined;
  let editor: monaco.editor.IStandaloneCodeEditor | undefined;
  let completionRegistered = false;

  // Helper to check if the active file is a .ydoc file
  const isYDocFile = () => {
    const files = appStore.openFiles();
    const activeIdx = appStore.activeFileIndex();
    if (activeIdx >= 0 && activeIdx < files.length) {
      const activePath = files[activeIdx].path;
      return activePath.endsWith('.ydoc');
    }
    return false;
  };

  onMount(() => {
    if (!editorContainer) return;

    // Get colors from CSS variables
    const bgPrimary = getComputedStyle(document.documentElement)
      .getPropertyValue('--bg-primary')
      .trim();
    const bgSecondary = getComputedStyle(document.documentElement)
      .getPropertyValue('--bg-secondary')
      .trim();

    // Configure Monaco Editor theme
    monaco.editor.defineTheme('yantra-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [],
      colors: {
        'editor.background': bgPrimary,
        'editor.lineHighlightBackground': bgSecondary,
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
      // Enable advanced completion features
      suggest: {
        showKeywords: true,
        showSnippets: true,
        showFunctions: true,
        showClasses: true,
        showModules: true,
      },
      quickSuggestions: {
        other: true,
        comments: false,
        strings: false,
      },
    });

    // Register Tree-sitter + GNN completion provider
    // Try to get project path from workspace or use current directory
    const projectPath = appStore.projectPath() || '/tmp/yantra-project';
    if (!completionRegistered) {
      registerCompletionProvider('python', projectPath);
      registerCompletionProvider('javascript', projectPath);
      registerCompletionProvider('typescript', projectPath);
      registerCompletionProvider('rust', projectPath);
      completionRegistered = true;
    }

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
    <div class="flex flex-col h-full" style={{ 'background-color': 'var(--bg-primary)' }}>
      {/* File Tabs */}
      <Show when={appStore.openFiles().length > 0}>
        <div
          class="flex overflow-x-auto"
          style={{
            'background-color': 'var(--bg-secondary)',
            'border-bottom': '1px solid var(--border-primary)',
          }}
        >
          <For each={appStore.openFiles()}>
            {(file, index) => (
              <div
                class="flex items-center px-4 py-2 cursor-pointer transition-colors"
                style={{
                  'background-color':
                    index() === appStore.activeFileIndex()
                      ? 'var(--bg-primary)'
                      : 'var(--bg-secondary)',
                  color:
                    index() === appStore.activeFileIndex()
                      ? 'var(--text-primary)'
                      : 'var(--text-tertiary)',
                  'border-right': '1px solid var(--border-primary)',
                }}
                onClick={() => appStore.switchToFile(index())}
              >
                <span class="text-sm mr-2">{file.name}</span>
                <button
                  class="hover:opacity-80"
                  style={{ color: 'var(--text-tertiary)' }}
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

      {/* Header with expand button when no file tabs */}
      <Show when={appStore.openFiles().length === 0}>
        <div
          class="px-3 py-2 flex items-center justify-end"
          style={{
            'background-color': 'var(--bg-secondary)',
            'border-bottom': '1px solid var(--border-primary)',
          }}
        >
          <button
            onClick={() => layoutStore.togglePanelExpansion('editor')}
            class="px-1.5 py-0.5 text-xs rounded hover:opacity-70 transition-opacity"
            style={{
              'background-color': layoutStore.isExpanded('editor')
                ? 'var(--accent-primary)'
                : 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
            }}
            title={layoutStore.isExpanded('editor') ? 'Collapse panel' : 'Expand panel'}
          >
            {layoutStore.isExpanded('editor') ? '◀' : '▶'}
          </button>
        </div>
      </Show>

      {/* Monaco Editor or YDoc Editor */}
      <Show
        when={!isYDocFile()}
        fallback={
          <Show when={appStore.activeFileIndex() >= 0}>
            <YDocBlockEditor
              docId={appStore.openFiles()[appStore.activeFileIndex()]?.path || ''}
              initialContent={appStore.currentCode()}
              onSave={(content, metadata) => {
                appStore.updateCode(content);
                // TODO: Save metadata through appStore or API
                console.log('YDoc saved:', { content, metadata });
              }}
              onCancel={() => {
                // Optional: Close the file or revert changes
                console.log('YDoc edit cancelled');
              }}
            />
          </Show>
        }
      >
        <div ref={editorContainer} class="flex-1" />
      </Show>
    </div>
  );
};

export default CodeViewer;
