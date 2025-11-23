// File: src-ui/components/FileTree.tsx
// Purpose: File tree component with recursive folder navigation
// Dependencies: solid-js, tauri utils
// Last Updated: November 23, 2025

import { Component, createSignal, For, Show, type JSX } from 'solid-js';
import { FileEntry, readDir, selectFolder, readFile } from '../utils/tauri';
import { appStore } from '../stores/appStore';

interface TreeNode extends FileEntry {
  children?: TreeNode[];
  isExpanded?: boolean;
}

const FileTree: Component = () => {
  const [rootPath, setRootPath] = createSignal<string | null>(null);
  const [treeNodes, setTreeNodes] = createSignal<TreeNode[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const loadDirectory = async (path: string): Promise<TreeNode[]> => {
    try {
      const dirEntries = await readDir(path);
      // Sort: directories first, then files, both alphabetically
      return dirEntries
        .sort((a, b) => {
          if (a.is_directory !== b.is_directory) {
            return a.is_directory ? -1 : 1;
          }
          return a.name.localeCompare(b.name);
        })
        .map(entry => ({
          ...entry,
          children: entry.is_directory ? [] : undefined,
          isExpanded: false,
        }));
    } catch (err) {
      console.error('Error loading directory:', err);
      throw err;
    }
  };

  const handleOpenFolder = async () => {
    try {
      setLoading(true);
      setError(null);
      const folder = await selectFolder();
      if (folder) {
        setRootPath(folder);
        const nodes = await loadDirectory(folder);
        setTreeNodes(nodes);
        appStore.loadProject(folder);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open folder');
    } finally {
      setLoading(false);
    }
  };

  const toggleDirectory = async (node: TreeNode, path: number[]) => {
    if (!node.is_directory) return;

    const newTree = [...treeNodes()];
    let current: TreeNode[] = newTree;
    
    // Navigate to the parent node
    for (let i = 0; i < path.length - 1; i++) {
      current = current[path[i]].children!;
    }
    
    const nodeIndex = path[path.length - 1];
    const targetNode = current[nodeIndex];

    if (targetNode.isExpanded) {
      // Collapse
      targetNode.isExpanded = false;
    } else {
      // Expand - load children if not loaded
      if (!targetNode.children || targetNode.children.length === 0) {
        try {
          targetNode.children = await loadDirectory(targetNode.path);
        } catch (err) {
          setError(`Failed to load ${targetNode.name}: ${err}`);
          return;
        }
      }
      targetNode.isExpanded = true;
    }

    setTreeNodes(newTree);
  };

  const handleFileClick = async (node: TreeNode, path: number[]) => {
    if (node.is_directory) {
      await toggleDirectory(node, path);
    } else {
      // Load file content
      try {
        const content = await readFile(node.path);
        appStore.openFile(node.path, node.name, content);
      } catch (err) {
        appStore.addMessage('system', `Failed to load file: ${err}`);
      }
    }
  };

  const getFileIcon = (node: TreeNode) => {
    if (node.is_directory) {
      return node.isExpanded ? '📂' : '📁';
    }
    
    const ext = node.name.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'py': return '🐍';
      case 'js': case 'jsx': case 'ts': case 'tsx': return '📜';
      case 'json': return '📋';
      case 'md': return '📝';
      case 'txt': return '📄';
      case 'html': return '🌐';
      case 'css': case 'scss': return '🎨';
      case 'rs': return '🦀';
      case 'go': return '🐹';
      default: return '📄';
    }
  };

  const renderTree = (nodes: TreeNode[], path: number[] = [], depth: number = 0): JSX.Element => {
    return (
      <For each={nodes}>
        {(node, index) => {
          const currentPath = [...path, index()];
          return (
            <>
              <li>
                <button
                  onClick={() => handleFileClick(node, currentPath)}
                  class="w-full px-2 py-1 text-left text-sm text-gray-300 hover:bg-gray-800 transition-colors flex items-center gap-2"
                  style={{ 'padding-left': `${depth * 16 + 8}px` }}
                >
                  <span class="text-base">{getFileIcon(node)}</span>
                  <span class="flex-1 truncate">{node.name}</span>
                  <Show when={!node.is_directory && node.size}>
                    <span class="text-xs text-gray-500">
                      {(node.size! / 1024).toFixed(1)}KB
                    </span>
                  </Show>
                </button>
              </li>
              <Show when={node.is_directory && node.isExpanded && node.children}>
                {renderTree(node.children!, currentPath, depth + 1)}
              </Show>
            </>
          );
        }}
      </For>
    );
  };

  return (
    <div class="flex flex-col h-full bg-gray-800">
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
      <Show when={rootPath()}>
        <div class="px-4 py-2 border-b border-gray-700">
          <p class="text-xs text-gray-400 truncate" title={rootPath() || ''}>
            {rootPath()}
          </p>
        </div>
      </Show>

      {/* File Tree */}
      <div class="flex-1 overflow-y-auto">
        <Show
          when={treeNodes().length > 0}
          fallback={
            <div class="flex items-center justify-center h-full text-gray-500 text-sm px-4 text-center">
              {loading() ? 'Loading files...' : 'Open a project folder to see files'}
            </div>
          }
        >
          <ul class="py-2">
            {renderTree(treeNodes())}
          </ul>
        </Show>
      </div>
    </div>
  );
};

export default FileTree;
