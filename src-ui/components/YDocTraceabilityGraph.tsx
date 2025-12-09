// File: src-ui/components/YDocTraceabilityGraph.tsx
// Purpose: Interactive traceability graph visualization for YDoc with layouts, filtering, and context menus
// Dependencies: solid-js, api/ydoc
// Last Updated: December 9, 2025

import { createSignal, createEffect, onMount, onCleanup, Component } from 'solid-js';
import { getTraceabilityChain, getCoverageStats, TraceabilityChain } from '../api/ydoc';
import './YDocTraceabilityGraph.css';

interface GraphNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  visible: boolean;
}

interface GraphEdge {
  source: string;
  target: string;
  type: string;
  visible: boolean;
}

type LayoutType = 'force-directed' | 'hierarchical' | 'circular' | 'tree';

interface FilterState {
  nodeTypes: Set<string>;
  edgeTypes: Set<string>;
}

interface ContextMenu {
  x: number;
  y: number;
  nodeId: string;
  nodeType: string;
}

interface YDocTraceabilityGraphProps {
  blockId?: string;
  onNodeClick: (nodeId: string, nodeType: string) => void;
}

export const YDocTraceabilityGraph: Component<YDocTraceabilityGraphProps> = (props) => {
  const [chain, setChain] = createSignal<TraceabilityChain | null>(null);
  const [stats, setStats] = createSignal<Record<string, number> | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);
  const [zoom, setZoom] = createSignal(1);
  const [pan, setPan] = createSignal({ x: 0, y: 0 });
  let canvasRef: HTMLCanvasElement | undefined;
  let containerRef: HTMLDivElement | undefined;
  const [nodes, setNodes] = createSignal<GraphNode[]>([]);
  const [edges, setEdges] = createSignal<GraphEdge[]>([]);
  let animationRef: number | undefined;

  // NEW: Layout, filtering, context menu states
  const [layout, setLayout] = createSignal<LayoutType>('force-directed');
  const [showFilters, setShowFilters] = createSignal(false);
  const [filters, setFilters] = createSignal<FilterState>({
    nodeTypes: new Set([
      'doc_block',
      'code_file',
      'function',
      'class',
      'test_file',
      'api_endpoint',
    ]),
    edgeTypes: new Set(['forward', 'backward']),
  });
  const [contextMenu, setContextMenu] = createSignal<ContextMenu | null>(null);
  const availableNodeTypes = [
    'doc_block',
    'code_file',
    'function',
    'class',
    'test_file',
    'api_endpoint',
  ];
  const availableEdgeTypes = ['forward', 'backward'];

  onMount(() => {
    loadCoverageStats();
  });

  createEffect(() => {
    if (props.blockId) {
      loadTraceabilityChain(props.blockId);
    }
  });

  createEffect(() => {
    const currentChain = chain();
    if (currentChain) {
      buildGraph(currentChain);
    }
  });

  createEffect(() => {
    if (nodes().length > 0) {
      startAnimation();
    }
    onCleanup(() => {
      if (animationRef !== undefined) {
        cancelAnimationFrame(animationRef);
      }
    });
  });

  const loadCoverageStats = async () => {
    try {
      const coverageStats = await getCoverageStats();
      setStats(coverageStats);
    } catch (err) {
      console.error('Failed to load coverage stats:', err);
    }
  };

  const loadTraceabilityChain = async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const traceChain = await getTraceabilityChain(id);
      setChain(traceChain);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load traceability chain');
      console.error('Traceability error:', err);
    } finally {
      setLoading(false);
    }
  };

  const buildGraph = (traceChain: TraceabilityChain) => {
    const graphNodes: GraphNode[] = [];
    const graphEdges: GraphEdge[] = [];
    const nodeMap = new Map<string, GraphNode>();
    const currentFilters = filters();

    // Helper to add node
    const addNode = (id: string, label: string, type: string, layerX: number, layerY: number) => {
      if (!nodeMap.has(id)) {
        const node: GraphNode = {
          id,
          label: label || id,
          type,
          x: layerX + Math.random() * 100 - 50,
          y: layerY,
          vx: 0,
          vy: 0,
          visible: currentFilters.nodeTypes.has(type),
        };
        nodeMap.set(id, node);
        graphNodes.push(node);
      }
    };

    // Center root node
    const rootId = traceChain.root.entity_id;
    addNode(rootId, traceChain.root.label || rootId, traceChain.root.entity_type, 400, 300);

    // Forward chain (going right)
    traceChain.forward_chain.forEach((entity, idx) => {
      const layerX = 600 + idx * 200;
      const layerY = 300 + (idx % 2 === 0 ? -50 : 50);
      addNode(
        entity.entity_id,
        entity.label || entity.entity_id,
        entity.entity_type,
        layerX,
        layerY
      );

      // Add edge from previous
      const sourceId = idx === 0 ? rootId : traceChain.forward_chain[idx - 1].entity_id;
      graphEdges.push({
        source: sourceId,
        target: entity.entity_id,
        type: 'forward',
        visible: currentFilters.edgeTypes.has('forward'),
      });
    });

    // Backward chain (going left)
    traceChain.backward_chain.forEach((entity, idx) => {
      const layerX = 200 - idx * 200;
      const layerY = 300 + (idx % 2 === 0 ? 50 : -50);
      addNode(
        entity.entity_id,
        entity.label || entity.entity_id,
        entity.entity_type,
        layerX,
        layerY
      );

      // Add edge to previous
      const targetId = idx === 0 ? rootId : traceChain.backward_chain[idx - 1].entity_id;
      graphEdges.push({
        source: entity.entity_id,
        target: targetId,
        type: 'backward',
        visible: currentFilters.edgeTypes.has('backward'),
      });
    });

    setNodes(graphNodes);
    setEdges(graphEdges);

    // Apply initial layout
    applyLayout(layout(), graphNodes);
  };

  // NEW: Layout algorithms
  const applyLayout = (layoutType: LayoutType, layoutNodes: GraphNode[]) => {
    if (!canvasRef) return;

    const width = canvasRef.width;
    const height = canvasRef.height;
    const centerX = width / 2;
    const centerY = height / 2;

    switch (layoutType) {
      case 'hierarchical':
        applyHierarchicalLayout(layoutNodes, centerX, centerY);
        break;
      case 'circular':
        applyCircularLayout(layoutNodes, centerX, centerY);
        break;
      case 'tree':
        applyTreeLayout(layoutNodes, centerX, centerY);
        break;
      case 'force-directed':
      default:
        // Already positioned in buildGraph
        break;
    }

    setNodes([...layoutNodes]);
  };

  const applyHierarchicalLayout = (layoutNodes: GraphNode[], centerX: number, centerY: number) => {
    // Group nodes by type hierarchy
    const layers: Record<string, GraphNode[]> = {
      requirements: [],
      architecture: [],
      code: [],
      tests: [],
    };

    layoutNodes.forEach((node) => {
      if (node.type === 'doc_block') layers.requirements.push(node);
      else if (node.type === 'api_endpoint') layers.architecture.push(node);
      else if (node.type === 'code_file' || node.type === 'function' || node.type === 'class')
        layers.code.push(node);
      else if (node.type === 'test_file') layers.tests.push(node);
    });

    const layerKeys = Object.keys(layers).filter((k) => layers[k].length > 0);
    const layerSpacing = 200;
    const nodeSpacing = 150;

    layerKeys.forEach((layerKey, layerIndex) => {
      const layerNodes = layers[layerKey];
      const layerY = centerY - (layerKeys.length * layerSpacing) / 2 + layerIndex * layerSpacing;

      layerNodes.forEach((node, nodeIndex) => {
        const layerX = centerX - (layerNodes.length * nodeSpacing) / 2 + nodeIndex * nodeSpacing;
        node.x = layerX;
        node.y = layerY;
        node.vx = 0;
        node.vy = 0;
      });
    });
  };

  const applyCircularLayout = (layoutNodes: GraphNode[], centerX: number, centerY: number) => {
    const radius = Math.min(centerX, centerY) * 0.6;
    const angleStep = (2 * Math.PI) / layoutNodes.length;

    layoutNodes.forEach((node, index) => {
      const angle = index * angleStep;
      node.x = centerX + radius * Math.cos(angle);
      node.y = centerY + radius * Math.sin(angle);
      node.vx = 0;
      node.vy = 0;
    });
  };

  const applyTreeLayout = (layoutNodes: GraphNode[], centerX: number, centerY: number) => {
    if (layoutNodes.length === 0) return;

    // Find root (node with most connections)
    const connectionCounts = new Map<string, number>();
    const currentEdges = edges();
    currentEdges.forEach((edge) => {
      connectionCounts.set(edge.source, (connectionCounts.get(edge.source) || 0) + 1);
      connectionCounts.set(edge.target, (connectionCounts.get(edge.target) || 0) + 1);
    });

    const root = layoutNodes.reduce((max, node) =>
      (connectionCounts.get(node.id) || 0) > (connectionCounts.get(max.id) || 0) ? node : max
    );

    // BFS layout
    const visited = new Set<string>();
    const queue: Array<{ node: GraphNode; level: number; position: number }> = [];
    queue.push({ node: root, level: 0, position: 0 });
    visited.add(root.id);

    const levelNodes: GraphNode[][] = [];

    while (queue.length > 0) {
      const { node, level, position } = queue.shift()!;

      if (!levelNodes[level]) levelNodes[level] = [];
      levelNodes[level].push(node);

      // Find children
      const children = edges()
        .filter((e) => e.source === node.id && !visited.has(e.target))
        .map((e) => layoutNodes.find((n) => n.id === e.target))
        .filter((n) => n !== undefined) as GraphNode[];

      children.forEach((child, idx) => {
        visited.add(child.id);
        queue.push({ node: child, level: level + 1, position: position * children.length + idx });
      });
    }

    // Position nodes
    const levelSpacing = 150;
    const nodeSpacing = 120;

    levelNodes.forEach((levelNodeList, levelIndex) => {
      const levelY = centerY - (levelNodes.length * levelSpacing) / 2 + levelIndex * levelSpacing;

      levelNodeList.forEach((node, nodeIndex) => {
        const levelX = centerX - (levelNodeList.length * nodeSpacing) / 2 + nodeIndex * nodeSpacing;
        node.x = levelX;
        node.y = levelY;
        node.vx = 0;
        node.vy = 0;
      });
    });
  };

  const startAnimation = () => {
    const animate = () => {
      updatePhysics();
      drawGraph();
      animationRef = requestAnimationFrame(animate);
    };
    animate();
  };

  const updatePhysics = () => {
    if (!canvasRef || nodes().length === 0) return;

    const width = canvasRef.width;
    const height = canvasRef.height;
    const currentNodes = nodes();
    const currentEdges = edges();

    // Simple force-directed layout
    const damping = 0.9;
    const repulsion = 5000;
    const attraction = 0.01;

    // Apply forces between nodes (repulsion)
    for (let i = 0; i < currentNodes.length; i++) {
      for (let j = i + 1; j < currentNodes.length; j++) {
        const dx = currentNodes[j].x - currentNodes[i].x;
        const dy = currentNodes[j].y - currentNodes[i].y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = repulsion / (distance * distance);

        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;

        currentNodes[i].vx -= fx;
        currentNodes[i].vy -= fy;
        currentNodes[j].vx += fx;
        currentNodes[j].vy += fy;
      }
    }

    // Apply spring forces for edges (attraction)
    currentEdges.forEach((edge) => {
      const source = currentNodes.find((n) => n.id === edge.source);
      const target = currentNodes.find((n) => n.id === edge.target);
      if (!source || !target) return;

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = attraction * distance;

      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;

      source.vx += fx;
      source.vy += fy;
      target.vx -= fx;
      target.vy -= fy;
    });

    // Update positions with damping
    currentNodes.forEach((node) => {
      node.vx *= damping;
      node.vy *= damping;
      node.x += node.vx;
      node.y += node.vy;

      // Keep nodes in bounds
      node.x = Math.max(50, Math.min(width - 50, node.x));
      node.y = Math.max(50, Math.min(height - 50, node.y));
    });

    setNodes([...currentNodes]);
  };

  const drawGraph = () => {
    const canvas = canvasRef;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    const currentPan = pan();
    const currentZoom = zoom();
    ctx.translate(currentPan.x, currentPan.y);
    ctx.scale(currentZoom, currentZoom);

    const currentNodes = nodes();
    const currentEdges = edges();
    const currentSelectedNode = selectedNode();

    // Draw edges (only visible ones)
    currentEdges
      .filter((edge) => edge.visible)
      .forEach((edge) => {
        const source = currentNodes.find((n) => n.id === edge.source);
        const target = currentNodes.find((n) => n.id === edge.target);
        if (!source || !target || !source.visible || !target.visible) return;

        ctx.strokeStyle = edge.type === 'forward' ? '#4ec9b0' : '#569cd6';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();

        // Draw arrow
        const angle = Math.atan2(target.y - source.y, target.x - source.x);
        const arrowSize = 10;
        ctx.fillStyle = edge.type === 'forward' ? '#4ec9b0' : '#569cd6';
        ctx.beginPath();
        ctx.moveTo(target.x, target.y);
        ctx.lineTo(
          target.x - arrowSize * Math.cos(angle - Math.PI / 6),
          target.y - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          target.x - arrowSize * Math.cos(angle + Math.PI / 6),
          target.y - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
      });

    // Draw nodes (only visible ones)
    currentNodes
      .filter((node) => node.visible)
      .forEach((node) => {
        const isSelected = currentSelectedNode === node.id;
        const radius = isSelected ? 35 : 30;

        // Node circle
        ctx.fillStyle = getNodeColor(node.type);
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
        ctx.fill();

        if (isSelected) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        // Node label
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const shortLabel =
          node.label.length > 15 ? node.label.substring(0, 12) + '...' : node.label;
        ctx.fillText(shortLabel, node.x, node.y);

        // Node type
        ctx.fillStyle = '#cccccc';
        ctx.font = '10px Arial';
        ctx.fillText(node.type, node.x, node.y + radius + 15);
      });

    ctx.restore();
  };

  const getNodeColor = (type: string): string => {
    const colors: Record<string, string> = {
      doc_block: '#c586c0',
      code_file: '#4ec9b0',
      function: '#dcdcaa',
      class: '#569cd6',
      test_file: '#9cdcfe',
      api_endpoint: '#ce9178',
    };
    return colors[type] || '#808080';
  };

  const handleCanvasClick = (e: MouseEvent) => {
    const canvas = canvasRef;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const currentPan = pan();
    const currentZoom = zoom();
    const currentNodes = nodes();
    const x = (e.clientX - rect.left - currentPan.x) / currentZoom;
    const y = (e.clientY - rect.top - currentPan.y) / currentZoom;

    // Find clicked node
    for (const node of currentNodes.filter((n) => n.visible)) {
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
      if (distance < 30) {
        setSelectedNode(node.id);
        props.onNodeClick(node.id, node.type);
        setContextMenu(null); // Close context menu on click
        return;
      }
    }

    setSelectedNode(null);
    setContextMenu(null);
  };

  // NEW: Context menu handler
  const handleCanvasContextMenu = (e: MouseEvent) => {
    e.preventDefault();
    const canvas = canvasRef;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const currentPan = pan();
    const currentZoom = zoom();
    const currentNodes = nodes();
    const x = (e.clientX - rect.left - currentPan.x) / currentZoom;
    const y = (e.clientY - rect.top - currentPan.y) / currentZoom;

    // Find right-clicked node
    for (const node of currentNodes.filter((n) => n.visible)) {
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
      if (distance < 30) {
        setContextMenu({
          x: e.clientX,
          y: e.clientY,
          nodeId: node.id,
          nodeType: node.type,
        });
        return;
      }
    }

    setContextMenu(null);
  };

  // NEW: Context menu actions
  const handleOpenInEditor = () => {
    const menu = contextMenu();
    if (menu) {
      // TODO: Integrate with file opening
      console.log('Opening in editor:', menu.nodeId);
      props.onNodeClick(menu.nodeId, menu.nodeType);
      setContextMenu(null);
    }
  };

  const handleCopyId = () => {
    const menu = contextMenu();
    if (menu) {
      navigator.clipboard.writeText(menu.nodeId);
      setContextMenu(null);
    }
  };

  const handleFindReferences = () => {
    const menu = contextMenu();
    if (menu) {
      // TODO: Integrate with search
      console.log('Finding references for:', menu.nodeId);
      setContextMenu(null);
    }
  };

  // NEW: Filter handlers
  const toggleNodeTypeFilter = (nodeType: string) => {
    const currentFilters = filters();
    const newNodeTypes = new Set(currentFilters.nodeTypes);
    if (newNodeTypes.has(nodeType)) {
      newNodeTypes.delete(nodeType);
    } else {
      newNodeTypes.add(nodeType);
    }

    const newFilters = {
      ...currentFilters,
      nodeTypes: newNodeTypes,
    };
    setFilters(newFilters);

    // Update node visibility
    const currentNodes = nodes();
    setNodes(
      currentNodes.map((node) => ({
        ...node,
        visible: newNodeTypes.has(node.type),
      }))
    );
  };

  const toggleEdgeTypeFilter = (edgeType: string) => {
    const currentFilters = filters();
    const newEdgeTypes = new Set(currentFilters.edgeTypes);
    if (newEdgeTypes.has(edgeType)) {
      newEdgeTypes.delete(edgeType);
    } else {
      newEdgeTypes.add(edgeType);
    }

    const newFilters = {
      ...currentFilters,
      edgeTypes: newEdgeTypes,
    };
    setFilters(newFilters);

    // Update edge visibility
    const currentEdges = edges();
    setEdges(
      currentEdges.map((edge) => ({
        ...edge,
        visible: newEdgeTypes.has(edge.type),
      }))
    );
  };

  const clearAllFilters = () => {
    const allNodeTypes = new Set(availableNodeTypes);
    const allEdgeTypes = new Set(availableEdgeTypes);

    setFilters({
      nodeTypes: allNodeTypes,
      edgeTypes: allEdgeTypes,
    });

    const currentNodes = nodes();
    const currentEdges = edges();
    setNodes(currentNodes.map((node) => ({ ...node, visible: true })));
    setEdges(currentEdges.map((edge) => ({ ...edge, visible: true })));
  };

  // NEW: Keyboard shortcuts
  createEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const currentLayout = layout();
      const currentNodes = nodes();
      // F - Toggle filters
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        setShowFilters((prev) => !prev);
      }
      // L - Cycle layouts
      else if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        const layouts: LayoutType[] = ['force-directed', 'hierarchical', 'circular', 'tree'];
        const currentIndex = layouts.indexOf(currentLayout);
        const nextLayout = layouts[(currentIndex + 1) % layouts.length];
        setLayout(nextLayout);
        applyLayout(nextLayout, currentNodes);
      }
      // R - Reset view
      else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReset();
      }
      // + - Zoom in
      else if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        handleZoomIn();
      }
      // - - Zoom out
      else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        handleZoomOut();
      }
      // C - Clear filters
      else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        clearAllFilters();
      }
      // Escape - Close context menu
      else if (e.key === 'Escape') {
        setContextMenu(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    onCleanup(() => window.removeEventListener('keydown', handleKeyDown));
  });

  // Close context menu on outside click
  createEffect(() => {
    const menu = contextMenu();
    const handleClickOutside = () => setContextMenu(null);
    if (menu) {
      document.addEventListener('click', handleClickOutside);
      onCleanup(() => document.removeEventListener('click', handleClickOutside));
    }
  });

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.2, 3));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.2, 0.5));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    if (props.blockId) {
      loadTraceabilityChain(props.blockId);
    }
  };

  onMount(() => {
    const canvas = canvasRef;
    const container = containerRef;
    if (!canvas || !container) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
  });

  return (
    <div class="ydoc-traceability-graph">
      <div class="graph-header">
        <div class="graph-controls">
          {/* Layout selector */}
          <select
            class="layout-select"
            value={layout()}
            onChange={(e) => {
              const newLayout = e.target.value as LayoutType;
              setLayout(newLayout);
              applyLayout(newLayout, nodes());
            }}
            title="Graph Layout (L)"
          >
            <option value="force-directed">Force-Directed</option>
            <option value="hierarchical">Hierarchical</option>
            <option value="circular">Circular</option>
            <option value="tree">Tree</option>
          </select>

          {/* Filter toggle */}
          <button
            class={`control-btn ${showFilters() ? 'active' : ''}`}
            onClick={() => setShowFilters(!showFilters())}
            title="Toggle Filters (F)"
          >
            {showFilters() ? 'Hide' : 'Show'} Filters
          </button>

          <button class="control-btn" onClick={handleZoomIn} title="Zoom In (+)">
            +
          </button>
          <button class="control-btn" onClick={handleZoomOut} title="Zoom Out (-)">
            −
          </button>
          <button class="control-btn" onClick={handleReset} title="Reset View (R)">
            Reset
          </button>
        </div>
      </div>

      {/* Filter Panel */}
      {showFilters() && (
        <div class="filter-panel">
          <div class="filter-section">
            <h4>Node Types</h4>
            <div class="filter-chips">
              {availableNodeTypes.map((type) => (
                <button
                  class={`filter-chip ${filters().nodeTypes.has(type) ? 'active' : ''}`}
                  onClick={() => toggleNodeTypeFilter(type)}
                >
                  {type.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>

          <div class="filter-section">
            <h4>Edge Types</h4>
            <div class="filter-chips">
              {availableEdgeTypes.map((type) => (
                <button
                  class={`filter-chip ${filters().edgeTypes.has(type) ? 'active' : ''}`}
                  onClick={() => toggleEdgeTypeFilter(type)}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          <button class="clear-filters-btn" onClick={clearAllFilters} title="Clear All Filters (C)">
            Clear All Filters
          </button>
        </div>
      )}

      {stats() && (
        <div class="coverage-stats">
          <div class="stat-item">
            <span class="stat-label">Requirements:</span>
            <span class="stat-value">{stats()!.total_requirements || 0}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Implemented:</span>
            <span class="stat-value implemented">{stats()!.implemented_requirements || 0}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Tested:</span>
            <span class="stat-value tested">{stats()!.tested_requirements || 0}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Coverage:</span>
            <span class="stat-value coverage">
              {((stats()!.coverage_percentage || 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {error() && (
        <div class="error-message">
          <span>⚠️ {error()}</span>
        </div>
      )}

      {loading() && (
        <div class="loading-overlay">
          <div class="spinner"></div>
          <p>Loading traceability graph...</p>
        </div>
      )}

      {!props.blockId && !loading() && (
        <div class="empty-state">
          <p>Select a block to view its traceability chain</p>
        </div>
      )}

      <div ref={containerRef} class="graph-container">
        <canvas
          ref={canvasRef}
          class="graph-canvas"
          onClick={handleCanvasClick}
          onContextMenu={handleCanvasContextMenu}
        />
      </div>

      {/* Context Menu */}
      {contextMenu() && (
        <div
          class="context-menu"
          style={{
            position: 'fixed',
            left: `${contextMenu()!.x}px`,
            top: `${contextMenu()!.y}px`,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <button class="context-menu-item" onClick={handleOpenInEditor}>
            📝 Open in Editor
          </button>
          <button class="context-menu-item" onClick={handleCopyId}>
            📋 Copy ID
          </button>
          <button class="context-menu-item" onClick={handleFindReferences}>
            🔍 Find References
          </button>
        </div>
      )}

      {chain() && (
        <div class="graph-legend">
          <div class="legend-item">
            <div class="legend-color forward"></div>
            <span>Forward (implements, traces to)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color backward"></div>
            <span>Backward (documented by, tests)</span>
          </div>
          <div class="keyboard-shortcuts">
            <strong>Shortcuts:</strong> F=Filters | L=Layout | R=Reset | +/- =Zoom | C=Clear
          </div>
        </div>
      )}
    </div>
  );
};
