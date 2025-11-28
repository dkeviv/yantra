/**
 * ArchitectureCanvas - Main React Flow canvas for architecture visualization
 * 
 * Features:
 * - Interactive drag-and-drop component positioning
 * - Zoom and pan controls
 * - Minimap for navigation
 * - Custom node and edge rendering
 * - Real-time updates from store
 */

import { createEffect, createMemo, For, Show } from 'solid-js';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  Connection,
  ConnectionMode,
  MarkerType,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { architectureState, addConnection, updateComponent, getFilteredComponents } from '../../stores/architectureStore';
import type { Component as ArchComponent, ConnectionType } from '../../api/architecture';
import * as archAPI from '../../api/architecture';
import ComponentNode from './ComponentNode';
import ConnectionEdge from './ConnectionEdge';

// Custom node types
const nodeTypes = {
  component: ComponentNode,
};

// Custom edge types
const edgeTypes = {
  connection: ConnectionEdge,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Convert architecture component to React Flow node
 */
function componentToNode(component: ArchComponent): Node {
  return {
    id: component.id,
    type: 'component',
    position: component.position,
    data: {
      component,
      isSelected: architectureState.selectedComponentId === component.id,
    },
    draggable: true,
  };
}

/**
 * Convert architecture connection to React Flow edge
 */
function connectionToEdge(connection: any): Edge {
  const color = archAPI.getConnectionColor(connection.connection_type);
  const strokeStyle = archAPI.getConnectionStrokeStyle(connection.connection_type);

  return {
    id: connection.id,
    source: connection.source_id,
    target: connection.target_id,
    type: 'connection',
    animated: connection.connection_type === 'Event', // Animate event connections
    style: {
      stroke: color,
      ...strokeStyle,
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color,
    },
    data: {
      connection,
      isSelected: architectureState.selectedConnectionId === connection.id,
    },
    label: connection.label,
  };
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function ArchitectureCanvas() {
  // Convert architecture data to React Flow format
  const nodes = createMemo(() => {
    const components = getFilteredComponents();
    return components.map(componentToNode);
  });

  const edges = createMemo(() => {
    if (!architectureState.current) return [];
    return architectureState.current.connections.map(connectionToEdge);
  });

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  /**
   * Handle node drag end - update component position
   */
  const onNodeDragStop = async (_event: any, node: Node) => {
    try {
      await updateComponent(node.id, {
        position: node.position,
      });
    } catch (error) {
      console.error('Failed to update component position:', error);
    }
  };

  /**
   * Handle new connection creation
   */
  const onConnect = async (connection: Connection) => {
    if (!connection.source || !connection.target) return;

    try {
      // Default to DataFlow connection type
      // TODO: Show modal to select connection type
      await addConnection(
        connection.source,
        connection.target,
        'DataFlow' as ConnectionType,
        undefined
      );
    } catch (error) {
      console.error('Failed to create connection:', error);
    }
  };

  /**
   * Handle node click - select component
   */
  const onNodeClick = (_event: any, node: Node) => {
    // Selection is handled in ComponentNode
  };

  /**
   * Handle edge click - select connection
   */
  const onEdgeClick = (_event: any, edge: Edge) => {
    // Selection is handled in ConnectionEdge
  };

  /**
   * Handle canvas click - deselect all
   */
  const onPaneClick = () => {
    // Deselect component/connection when clicking canvas
    // This is handled by the store
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div class="w-full h-full bg-gray-900">
      <Show
        when={architectureState.current && !architectureState.isLoading}
        fallback={
          <div class="flex items-center justify-center h-full">
            <Show
              when={architectureState.isLoading}
              fallback={
                <div class="text-gray-400">
                  <p class="text-xl mb-2">No Architecture Loaded</p>
                  <p class="text-sm">Create or load an architecture to get started</p>
                </div>
              }
            >
              <div class="text-gray-400">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4" />
                <p>Loading architecture...</p>
              </div>
            </Show>
          </div>
        }
      >
        <ReactFlow
          nodes={nodes()}
          edges={edges()}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodeDragStop={onNodeDragStop}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneClick={onPaneClick}
          connectionMode={ConnectionMode.Loose}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          className="bg-gray-900"
          minZoom={0.1}
          maxZoom={2}
        >
          {/* Background grid */}
          <Background
            variant={BackgroundVariant.Dots}
            gap={20}
            size={1}
            color="#374151"
          />

          {/* Controls (zoom, fit view, etc.) */}
          <Controls
            className="bg-gray-800 border border-gray-700"
            style={{ button: { backgroundColor: '#1f2937', borderColor: '#374151' } }}
          />

          {/* Minimap for navigation */}
          <MiniMap
            className="bg-gray-800 border border-gray-700"
            maskColor="rgba(0, 0, 0, 0.3)"
            nodeColor={(node) => {
              const component = node.data?.component;
              if (!component) return '#6b7280';
              return archAPI.getStatusColor(component.status);
            }}
          />
        </ReactFlow>

        {/* Error display */}
        <Show when={architectureState.error}>
          <div class="absolute bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg">
            {architectureState.error}
          </div>
        </Show>

        {/* Architecture info overlay */}
        <div class="absolute top-4 left-4 bg-gray-800 border border-gray-700 rounded-lg shadow-lg px-4 py-3">
          <h2 class="text-white font-semibold text-lg mb-1">
            {architectureState.current?.name}
          </h2>
          <Show when={architectureState.current?.description}>
            <p class="text-gray-400 text-sm mb-2">
              {architectureState.current?.description}
            </p>
          </Show>
          <div class="flex gap-4 text-sm">
            <span class="text-gray-400">
              Components: <span class="text-white">{nodes().length}</span>
            </span>
            <span class="text-gray-400">
              Connections: <span class="text-white">{edges().length}</span>
            </span>
          </div>
        </div>
      </Show>
    </div>
  );
}
