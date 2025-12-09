// File: src-ui/components/FileTree.tsx
// Purpose: File tree component with recursive folder navigation and .ydoc file support
// Dependencies: solid-js, tauri utils
// Last Updated: December 9, 2025

import { Component, createSignal, For, Show, onMount, onCleanup, type JSX } from 'solid-js';
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

  // Listen for close-project event to clear file tree
  onMount(() => {
    const handleCloseProject = () => {
      setRootPath(null);
      setTreeNodes([]);
      setError(null);
    };

    window.addEventListener('close-project', handleCloseProject);

    onCleanup(() => {
      window.removeEventListener('close-project', handleCloseProject);
    });
  });

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
        .map((entry) => ({
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
        // Notify chat that project is loaded
        appStore.addMessage(
          'system',
          `✅ Project opened: ${folder}\n\nI'm ready to help you build. What would you like to create?`
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open folder');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateNewProject = async () => {
    try {
      setLoading(true);
      setError(null);

      // Let user select where to create the project
      const parentFolder = await selectFolder();
      if (!parentFolder) {
        setLoading(false);
        return;
      }

      // Ask for project name via chat
      appStore.addMessage(
        'system',
        '📁 Creating new project...\n\nPlease tell me:\n1. What should we name this project?\n2. What type of project? (e.g., Python web app, React frontend, API server, etc.)\n3. What will it do?'
      );

      // For now, just open the folder - the agent will guide through chat
      setRootPath(parentFolder);
      const nodes = await loadDirectory(parentFolder);
      setTreeNodes(nodes);
      appStore.loadProject(parentFolder);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
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
      return { text: '', isFolder: true, expanded: node.isExpanded, color: '' };
    }

    const ext = node.name.split('.').pop()?.toLowerCase();
    const iconMap: Record<string, { text: string; color: string }> = {
      // JavaScript/TypeScript
      js: { text: 'JS', color: '#f7df1e' },
      jsx: { text: 'JSX', color: '#61dafb' },
      ts: { text: 'TS', color: '#3178c6' },
      tsx: { text: 'TSX', color: '#3178c6' },
      mjs: { text: 'MJS', color: '#f7df1e' },

      // Python
      py: { text: 'PY', color: '#3776ab' },
      pyw: { text: 'PY', color: '#3776ab' },

      // Rust
      rs: { text: 'RS', color: '#ce422b' },
      toml: { text: 'TOML', color: '#9c4221' },

      // Web
      html: { text: 'HTML', color: '#e34c26' },
      htm: { text: 'HTM', color: '#e34c26' },
      css: { text: 'CSS', color: '#1572b6' },
      scss: { text: 'SCSS', color: '#cc6699' },
      sass: { text: 'SASS', color: '#cc6699' },
      less: { text: 'LESS', color: '#1d365d' },

      // Data/Config
      json: { text: 'JSON', color: '#fbca04' },
      yaml: { text: 'YAML', color: '#cb171e' },
      yml: { text: 'YML', color: '#cb171e' },
      xml: { text: 'XML', color: '#e34c26' },
      md: { text: 'MD', color: '#083fa1' },
      txt: { text: 'TXT', color: '#6b7280' },
      csv: { text: 'CSV', color: '#0f9d58' },

      // YDoc files (Yantra Documentation)
      ydoc: { text: '📄', color: '#8b5cf6' },

      // Other languages
      go: { text: 'GO', color: '#00add8' },
      java: { text: 'JAVA', color: '#007396' },
      c: { text: 'C', color: '#555555' },
      cpp: { text: 'CPP', color: '#00599c' },
      h: { text: 'H', color: '#555555' },
      sh: { text: 'SH', color: '#4eaa25' },
      bash: { text: 'BASH', color: '#4eaa25' },
      sql: { text: 'SQL', color: '#e38c00' },
      php: { text: 'PHP', color: '#777bb4' },
      rb: { text: 'RB', color: '#cc342d' },
      swift: { text: 'SWIFT', color: '#f05138' },
      kt: { text: 'KT', color: '#7f52ff' },

      // Build/Config files
      lock: { text: 'LOCK', color: '#6b7280' },
      env: { text: 'ENV', color: '#ecd53f' },
      gitignore: { text: 'GIT', color: '#f05033' },
      dockerfile: { text: 'DOCK', color: '#2496ed' },
    };

    const icon = iconMap[ext || ''] || { text: ext?.toUpperCase() || 'FILE', color: '#6b7280' };
    return { ...icon, isFolder: false, expanded: false };
  };

  const renderTree = (nodes: TreeNode[], path: number[] = [], depth: number = 0): JSX.Element => {
    return (
      <For each={nodes}>
        {(node, index) => {
          const currentPath = [...path, index()];
          const icon = getFileIcon(node);
          return (
            <>
              <li>
                <button
                  onClick={() => handleFileClick(node, currentPath)}
                  class="w-full px-2 py-0.5 text-left text-xs transition-colors flex items-center gap-1.5"
                  style={{
                    'padding-left': `${depth * 12 + 8}px`,
                    'background-color':
                      appStore.openFiles().find((f) => f.path === node.path) &&
                      appStore.activeFileIndex() ===
                        appStore.openFiles().findIndex((f) => f.path === node.path)
                        ? 'var(--bg-tertiary)'
                        : 'transparent',
                    color: 'var(--text-primary)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--bg-hover)';
                  }}
                  onMouseLeave={(e) => {
                    const isActive =
                      appStore.openFiles().find((f) => f.path === node.path) &&
                      appStore.activeFileIndex() ===
                        appStore.openFiles().findIndex((f) => f.path === node.path);
                    e.currentTarget.style.backgroundColor = isActive
                      ? 'var(--bg-tertiary)'
                      : 'transparent';
                  }}
                >
                  {/* Icon/Badge */}
                  <Show
                    when={icon.isFolder}
                    fallback={
                      <span
                        class="text-[7px] font-bold px-0.5 py-0.5 rounded"
                        style={{
                          color: icon.color,
                          border: `1px solid ${icon.color}`,
                          'min-width': '20px',
                          'max-width': '20px',
                          'text-align': 'center',
                          opacity: '1',
                          filter: 'brightness(1.4)',
                        }}
                      >
                        {icon.text}
                      </span>
                    }
                  >
                    <span class="text-sm" style={{ filter: 'brightness(1.3)' }}>
                      {icon.expanded ? '📂' : '📁'}
                    </span>
                  </Show>

                  <span class="flex-1 truncate">{node.name}</span>
                  <Show when={!node.is_directory && node.size}>
                    <span class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
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
    <div class="flex flex-col h-full" style={{ 'background-color': 'var(--bg-secondary)' }}>
      {/* Header - Show different buttons based on project state */}
      <div class="px-4 py-3" style={{ 'border-bottom': '1px solid var(--border-primary)' }}>
        <Show
          when={rootPath()}
          fallback={
            <div class="space-y-2">
              <button
                onClick={handleOpenFolder}
                class="w-full px-4 py-2 text-sm rounded hover:opacity-90 transition-all flex items-center justify-center gap-2"
                style={{
                  'background-color': 'var(--accent-primary)',
                  color: 'var(--text-inverse)',
                }}
                disabled={loading()}
              >
                <span>📁</span>
                <span>{loading() ? 'Loading...' : 'Open Existing Project'}</span>
              </button>
              <button
                onClick={handleCreateNewProject}
                class="w-full px-4 py-2 text-sm rounded hover:opacity-90 transition-all flex items-center justify-center gap-2"
                style={{
                  'background-color': 'var(--status-success)',
                  color: 'var(--text-inverse)',
                }}
                disabled={loading()}
              >
                <span>✨</span>
                <span>{loading() ? 'Loading...' : 'Create New Project'}</span>
              </button>
            </div>
          }
        >
          {/* Project is open - show minimal info */}
          <div class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
            <div class="font-medium mb-1" style={{ color: 'var(--text-primary)' }}>
              Project Open
            </div>
            <div class="truncate" title={rootPath() || ''}>
              {rootPath()}
            </div>
          </div>
        </Show>
      </div>

      {/* Error Display */}
      <Show when={error()}>
        <div
          class="px-4 py-2"
          style={{
            'background-color': 'rgba(239, 68, 68, 0.15)',
            'border-bottom': '1px solid var(--status-error)',
          }}
        >
          <p class="text-sm" style={{ color: 'var(--status-error)' }}>
            {error()}
          </p>
        </div>
      </Show>

      {/* File Tree */}
      <div class="flex-1 overflow-y-auto">
        <Show
          when={treeNodes().length > 0}
          fallback={
            <div
              class="flex items-center justify-center h-full text-sm px-4 text-center"
              style={{ color: 'var(--text-tertiary)' }}
            >
              {loading() ? 'Loading files...' : 'Open or create a project to get started'}
            </div>
          }
        >
          <ul class="py-1">{renderTree(treeNodes())}</ul>
        </Show>
      </div>
    </div>
  );
};

export default FileTree;
