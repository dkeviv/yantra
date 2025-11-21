// File: src-ui/components/FileTree.tsx
// Purpose: File tree component for project navigation
// Dependencies: solid-js, tauri utils
// Last Updated: November 20, 2025

import { Component, createSignal, For, Show } from 'solid-js';
import { FileEntry, readDir, selectFolder } from '../utils/tauri';
import { appStore } from '../stores/appStore';

const FileTree: Component = () => {
  const [entries, setEntries] = createSignal<FileEntry[]>([]);
  const [expandedDirs, setExpandedDirs] = createSignal<Set<string>>(new Set());
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const loadDirectory = async (path: string) => {
    try {
      setLoading(true);
      setError(null);
      const dirEntries = await readDir(path);
      setEntries(dirEntries);
      appStore.loadProject(path);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load directory');
      console.error('Error loading directory:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenFolder = async () => {
    try {
      const folder = await selectFolder();
      if (folder) {
        await loadDirectory(folder);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open folder');
    }
  };

  const toggleDirectory = async (entry: FileEntry) => {
    const expanded = expandedDirs();
    const newExpanded = new Set(expanded);
    
    if (expanded.has(entry.path)) {
      newExpanded.delete(entry.path);
    } else {
      newExpanded.add(entry.path);
    }
    
    setExpandedDirs(newExpanded);
  };

  const handleFileClick = async (entry: FileEntry) => {
    if (entry.is_directory) {
      await toggleDirectory(entry);
    } else {
      // Load file content into editor
      try {
        const { readFile } = await import('../utils/tauri');
        const content = await readFile(entry.path);
        appStore.updateCode(content);
        appStore.addMessage('system', `Loaded file: ${entry.name}`);
      } catch (err) {
        appStore.addMessage('system', `Failed to load file: ${err}`);
      }
    }
  };

  const getFileIcon = (entry: FileEntry) => {
    if (entry.is_directory) {
      return expandedDirs().has(entry.path) ? '📂' : '📁';
    }
    
    const ext = entry.name.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'py': return '🐍';
      case 'js': case 'jsx': case 'ts': case 'tsx': return '📜';
      case 'json': return '📋';
      case 'md': return '📝';
      case 'txt': return '📄';
      default: return '📄';
    }
  };

  return (
    <div class="flex flex-col h-full bg-gray-900 border-r border-gray-700">
      {/* Header */}
      <div class="px-4 py-3 border-b border-gray-700">
        <button
          onClick={handleOpenFolder}
          class="w-full px-4 py-2 text-sm bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
          disabled={loading()}
        >
          {loading() ? 'Loading...' : 'Open Project Folder'}
        </button>
      </div>

      {/* Error Display */}
      <Show when={error()}>
        <div class="px-4 py-2 bg-red-900/20 border-b border-red-700/50">
          <p class="text-sm text-red-400">{error()}</p>
        </div>
      </Show>

      {/* Project Path */}
      <Show when={appStore.projectPath()}>
        <div class="px-4 py-2 border-b border-gray-700">
          <p class="text-xs text-gray-400 truncate" title={appStore.projectPath() || ''}>
            {appStore.projectPath()}
          </p>
        </div>
      </Show>

      {/* File List */}
      <div class="flex-1 overflow-y-auto">
        <Show
          when={entries().length > 0}
          fallback={
            <div class="flex items-center justify-center h-full text-gray-500 text-sm px-4 text-center">
              {loading() ? 'Loading files...' : 'Open a project folder to see files'}
            </div>
          }
        >
          <ul class="py-2">
            <For each={entries()}>
              {(entry) => (
                <li>
                  <button
                    onClick={() => handleFileClick(entry)}
                    class="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-gray-800 transition-colors flex items-center gap-2"
                  >
                    <span class="text-base">{getFileIcon(entry)}</span>
                    <span class="flex-1 truncate">{entry.name}</span>
                    <Show when={!entry.is_directory && entry.size}>
                      <span class="text-xs text-gray-500">
                        {(entry.size! / 1024).toFixed(1)}KB
                      </span>
                    </Show>
                  </button>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </div>
    </div>
  );
};

export default FileTree;
