/**
 * DependencyGraph.tsx
 * 
 * Purpose: Interactive dependency graph visualization using cytoscape.js
 * Shows file dependencies, function calls, and parameter flow from GNN
 * 
 * Features:
 * - Interactive graph with zoom/pan
 * - Node click for details
 * - Color-coded by type (file, function, class)
 * - Filtering by dependency type
 * - Export to PNG
 * 
 * Created: November 23, 2025
 */

import { createSignal, onMount, onCleanup, Show } from 'solid-js';
import cytoscape, { Core, ElementDefinition } from 'cytoscape';
import { invoke } from '@tauri-apps/api/tauri';

interface DependencyNode {
  id: string;
  label: string;
  type: 'file' | 'function' | 'class' | 'import';
  file_path?: string;
}

interface DependencyEdge {
  source: string;
  target: string;
  type: 'calls' | 'imports' | 'uses' | 'inherits';
}

interface GraphData {
  nodes: DependencyNode[];
  edges: DependencyEdge[];
}

export default function DependencyGraph() {
  const [cy, setCy] = createSignal<Core | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [selectedNode, setSelectedNode] = createSignal<DependencyNode | null>(null);
  const [filterType, setFilterType] = createSignal<string>('all');
  
  let containerRef: HTMLDivElement | undefined;

  onMount(async () => {
    if (!containerRef) return;

    try {
      // Query GNN for dependencies
      const data = await invoke<GraphData>('get_graph_dependencies');
      
      if (!data || !data.nodes || data.nodes.length === 0) {
        setError('No dependencies found. Open a project folder to see the dependency graph.');
        setLoading(false);
        return;
      }

      // Transform to cytoscape format
      const elements: ElementDefinition[] = [
        // Nodes
        ...data.nodes.map(node => ({
          data: {
            id: node.id,
            label: node.label,
            type: node.type,
            file_path: node.file_path,
          },
        })),
        // Edges
        ...data.edges.map(edge => ({
          data: {
            id: `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            type: edge.type,
          },
        })),
      ];

      // Initialize cytoscape
      const cyInstance = cytoscape({
        container: containerRef,
        elements,
        style: [
          // Node styles
          {
            selector: 'node',
            style: {
              'background-color': '#666',
              'label': 'data(label)',
              'color': '#fff',
              'text-outline-color': '#000',
              'text-outline-width': 2,
              'font-size': '12px',
              'width': 30,
              'height': 30,
            },
          },
          {
            selector: 'node[type="file"]',
            style: {
              'background-color': '#3b82f6', // Blue for files
              'shape': 'roundrectangle',
            },
          },
          {
            selector: 'node[type="function"]',
            style: {
              'background-color': '#10b981', // Green for functions
              'shape': 'ellipse',
            },
          },
          {
            selector: 'node[type="class"]',
            style: {
              'background-color': '#f59e0b', // Orange for classes
              'shape': 'diamond',
            },
          },
          {
            selector: 'node[type="import"]',
            style: {
              'background-color': '#8b5cf6', // Purple for imports
              'shape': 'hexagon',
            },
          },
          {
            selector: 'node:selected',
            style: {
              'border-width': 3,
              'border-color': '#ef4444',
            },
          },
          // Edge styles
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#999',
              'target-arrow-color': '#999',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
            },
          },
          {
            selector: 'edge[type="calls"]',
            style: {
              'line-color': '#10b981',
              'target-arrow-color': '#10b981',
            },
          },
          {
            selector: 'edge[type="imports"]',
            style: {
              'line-color': '#8b5cf6',
              'target-arrow-color': '#8b5cf6',
            },
          },
          {
            selector: 'edge[type="uses"]',
            style: {
              'line-color': '#3b82f6',
              'target-arrow-color': '#3b82f6',
            },
          },
          {
            selector: 'edge[type="inherits"]',
            style: {
              'line-color': '#f59e0b',
              'target-arrow-color': '#f59e0b',
              'line-style': 'dashed',
            },
          },
        ],
        layout: {
          name: 'cose', // Force-directed layout
          animate: true,
          animationDuration: 500,
          nodeRepulsion: 8000,
          idealEdgeLength: 100,
        },
      });

      // Node click handler
      cyInstance.on('tap', 'node', (evt) => {
        const node = evt.target;
        const data = node.data() as DependencyNode;
        setSelectedNode(data);
      });

      // Background click handler (deselect)
      cyInstance.on('tap', (evt) => {
        if (evt.target === cyInstance) {
          setSelectedNode(null);
        }
      });

      setCy(cyInstance);
      setLoading(false);
    } catch (err) {
      console.error('Failed to load dependencies:', err);
      setError(err instanceof Error ? err.message : 'Failed to load dependency graph');
      setLoading(false);
    }
  });

  onCleanup(() => {
    const cyInstance = cy();
    if (cyInstance) {
      cyInstance.destroy();
    }
  });

  // Filter nodes by type
  const applyFilter = (type: string) => {
    const cyInstance = cy();
    if (!cyInstance) return;

    if (type === 'all') {
      cyInstance.elements().style('display', 'element');
    } else {
      cyInstance.elements().style('display', 'none');
      cyInstance.nodes(`[type="${type}"]`).style('display', 'element');
      // Show edges connected to visible nodes
      cyInstance.nodes(`[type="${type}"]`).connectedEdges().style('display', 'element');
      // Show nodes connected to those edges
      cyInstance.nodes(`[type="${type}"]`).connectedEdges().connectedNodes().style('display', 'element');
    }
    
    setFilterType(type);
  };

  // Reset zoom/position
  const resetView = () => {
    const cyInstance = cy();
    if (cyInstance) {
      cyInstance.fit();
      cyInstance.center();
    }
  };

  // Export to PNG
  const exportPNG = () => {
    const cyInstance = cy();
    if (cyInstance) {
      const png = cyInstance.png({ full: true });
      const link = document.createElement('a');
      link.href = png;
      link.download = 'dependency-graph.png';
      link.click();
    }
  };

  return (
    <div class="h-full w-full flex flex-col bg-gray-900">
      {/* Header with controls */}
      <div class="p-3 border-b border-gray-700 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <h2 class="text-sm font-semibold text-white">Dependency Graph</h2>
          
          {/* Filter buttons */}
          <div class="flex gap-1">
            <button
              onClick={() => applyFilter('all')}
              class={`px-2 py-1 text-xs rounded ${
                filterType() === 'all'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              All
            </button>
            <button
              onClick={() => applyFilter('file')}
              class={`px-2 py-1 text-xs rounded ${
                filterType() === 'file'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📄 Files
            </button>
            <button
              onClick={() => applyFilter('function')}
              class={`px-2 py-1 text-xs rounded ${
                filterType() === 'function'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              ⚡ Functions
            </button>
            <button
              onClick={() => applyFilter('class')}
              class={`px-2 py-1 text-xs rounded ${
                filterType() === 'class'
                  ? 'bg-orange-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              🏛️ Classes
            </button>
          </div>
        </div>

        {/* Action buttons */}
        <div class="flex gap-2">
          <button
            onClick={resetView}
            class="px-3 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600"
            disabled={loading()}
          >
            🔄 Reset View
          </button>
          <button
            onClick={exportPNG}
            class="px-3 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600"
            disabled={loading()}
          >
            💾 Export PNG
          </button>
        </div>
      </div>

      {/* Legend */}
      <div class="px-3 py-2 bg-gray-800 border-b border-gray-700 flex items-center gap-4 text-xs text-gray-400">
        <span class="flex items-center gap-1">
          <span class="w-3 h-3 rounded-sm bg-blue-500"></span> Files
        </span>
        <span class="flex items-center gap-1">
          <span class="w-3 h-3 rounded-full bg-green-500"></span> Functions
        </span>
        <span class="flex items-center gap-1">
          <span class="w-3 h-3 rotate-45 bg-orange-500"></span> Classes
        </span>
        <span class="flex items-center gap-1">
          <span class="w-3 h-3 rounded bg-purple-500"></span> Imports
        </span>
        <span class="ml-auto">
          Zoom: Mouse wheel | Pan: Click and drag | Select: Click node
        </span>
      </div>

      {/* Graph container */}
      <div class="flex-1 relative">
        <Show when={loading()}>
          <div class="absolute inset-0 flex items-center justify-center bg-gray-900">
            <div class="text-center">
              <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p class="text-gray-400">Loading dependencies...</p>
            </div>
          </div>
        </Show>

        <Show when={error()}>
          <div class="absolute inset-0 flex items-center justify-center bg-gray-900">
            <div class="text-center max-w-md px-4">
              <div class="text-yellow-500 text-5xl mb-4">⚠️</div>
              <p class="text-gray-300 mb-2">{error()}</p>
              <p class="text-gray-500 text-sm">
                Make sure you have opened a project folder using the file tree panel.
              </p>
            </div>
          </div>
        </Show>

        <div
          ref={containerRef}
          class="w-full h-full"
          classList={{ hidden: loading() || error() !== null }}
        />
      </div>

      {/* Selected node details panel */}
      <Show when={selectedNode()}>
        <div class="p-3 border-t border-gray-700 bg-gray-800">
          <div class="text-sm">
            <div class="flex items-center justify-between mb-2">
              <h3 class="font-semibold text-white">Selected Node</h3>
              <button
                onClick={() => setSelectedNode(null)}
                class="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div class="space-y-1 text-gray-300">
              <div>
                <span class="text-gray-500">Type:</span>{' '}
                <span class="capitalize">{selectedNode()?.type}</span>
              </div>
              <div>
                <span class="text-gray-500">Name:</span>{' '}
                <span>{selectedNode()?.label}</span>
              </div>
              <Show when={selectedNode()?.file_path}>
                <div>
                  <span class="text-gray-500">File:</span>{' '}
                  <span class="text-xs font-mono">{selectedNode()?.file_path}</span>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
