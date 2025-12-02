/**
 * ArchitectureCanvas - Main canvas for architecture visualization using Cytoscape
 * 
 * Features:
 * - Interactive drag-and-drop component positioning
 * - Zoom and pan controls
 * - Custom node and edge rendering
 * - Real-time updates from store
 */

import { createEffect, onMount, onCleanup, Show } from 'solid-js';
import cytoscape, { Core, ElementDefinition } from 'cytoscape';

import { 
  architectureState, 
  updateComponent, 
  getFilteredComponents,
  selectComponent,
  selectConnection 
} from '../../stores/architectureStore';
import type { Component as ArchComponent } from '../../api/architecture';
import * as archAPI from '../../api/architecture';

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Convert architecture component to Cytoscape node
 */
function componentToNode(component: ArchComponent): ElementDefinition {
  const statusColor = archAPI.getStatusColor(component.component_type);
  const statusIndicator = archAPI.getStatusIndicator(component.component_type);
  const statusText = archAPI.getStatusText(component.component_type);
  const isSelected = component.id === architectureState.selectedComponentId;
  
  return {
    group: 'nodes',
    data: {
      id: component.id,
      label: `${statusIndicator} ${component.name}\n${component.category}\n${statusText}`,
      'background-color': statusColor,
      'border-color': isSelected ? '#3b82f6' : statusColor,
    },
    position: component.position,
    classes: isSelected ? 'selected' : '',
  };
}

/**
 * Convert architecture connection to Cytoscape edge
 */
function connectionToEdge(connection: any): ElementDefinition {
  const color = archAPI.getConnectionColor(connection.connection_type);
  const isSelected = architectureState.selectedConnectionId === connection.id;
  
  return {
    group: 'edges',
    data: {
      id: connection.id,
      source: connection.source_id,
      target: connection.target_id,
      label: connection.description || connection.connection_type,
      'line-color': color,
    },
    classes: isSelected ? 'selected' : '',
  };
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function ArchitectureCanvas() {
  let containerRef: HTMLDivElement | undefined;
  let cy: Core | undefined;

  /**
   * Initialize Cytoscape instance
   */
  onMount(() => {
    if (!containerRef) return;

    // Initialize cytoscape
    cy = cytoscape({
      container: containerRef,
      
      style: [
        // Node styles
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'data(background-color)',
            'border-width': 2,
            'border-color': 'data(border-color)',
            'width': 180,
            'height': 80,
            'font-size': '12px',
            'color': '#ffffff',
            'text-wrap': 'wrap',
            'text-max-width': '160px',
            'shape': 'roundrectangle',
          },
        },
        {
          selector: 'node.selected',
          style: {
            'border-width': 4,
            'border-color': '#3b82f6',
          },
        },
        // Edge styles
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': 'data(line-color)',
            'target-arrow-color': 'data(line-color)',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'color': '#9ca3af',
            'text-background-color': '#1f2937',
            'text-background-opacity': 1,
            'text-background-padding': '3px',
          },
        },
        {
          selector: 'edge.selected',
          style: {
            'width': 3,
            'opacity': 1,
            'line-color': '#3b82f6',
            'target-arrow-color': '#3b82f6',
          },
        },
      ],

      layout: {
        name: 'preset',
      },

      minZoom: 0.3,
      maxZoom: 2,
      wheelSensitivity: 0.2,
    });

    // Node click handler - select component
    cy.on('tap', 'node', (event) => {
      const nodeId = event.target.id();
      selectComponent(nodeId);
      updateCytoscape();
    });

    // Edge click handler - select connection
    cy.on('tap', 'edge', (event) => {
      const edgeId = event.target.id();
      selectConnection(edgeId);
      updateCytoscape();
    });

    // Canvas click handler - deselect all
    cy.on('tap', (event) => {
      if (event.target === cy) {
        selectComponent(null);
        selectConnection(null);
        updateCytoscape();
      }
    });

    // Node drag end handler - update component position
    cy.on('dragfree', 'node', async (event) => {
      const node = event.target;
      const position = node.position();
      
      try {
        await updateComponent(node.id(), {
          position: { x: position.x, y: position.y },
        });
      } catch (error) {
        console.error('Failed to update component position:', error);
      }
    });

    // Double-click edge to create connection (future enhancement)
    // For now, connections are created via toolbar

    updateCytoscape();
  });

  /**
   * Update Cytoscape with current architecture data
   */
  const updateCytoscape = () => {
    if (!cy) return;

    const components = getFilteredComponents();
    const connections = architectureState.current?.connections || [];

    // Build elements array
    const elements: ElementDefinition[] = [
      ...components.map(componentToNode),
      ...connections.map(connectionToEdge),
    ];

    // Update graph
    cy.elements().remove();
    cy.add(elements);

    // Fit view if this is the first render
    if (components.length > 0) {
      cy.fit(undefined, 50);
    }
  };

  /**
   * Effect: Update graph when architecture changes
   */
  createEffect(() => {
    // Watch for changes in current architecture or filter
    architectureState.current;
    architectureState.selectedComponentId;
    architectureState.selectedConnectionId;
    updateCytoscape();
  });

  /**
   * Cleanup on unmount
   */
  onCleanup(() => {
    if (cy) {
      cy.destroy();
    }
  });

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div class="w-full h-full bg-gray-900 relative">
      <Show
        when={architectureState.current && !architectureState.isLoading}
        fallback={
          <div class="flex items-center justify-center h-full">
            <Show
              when={architectureState.isLoading}
              fallback={
                <div class="text-gray-400 text-center max-w-md">
                  <div class="text-6xl mb-6">🏗️</div>
                  <p class="text-xl mb-3 font-semibold">No Architecture Yet</p>
                  <p class="text-sm mb-4 leading-relaxed">
                    Tell me in <span class="text-blue-400 font-medium">chat</span> what you want to build, 
                    and I'll generate the architecture diagram for you automatically.
                  </p>
                  <div class="bg-gray-800/50 border border-gray-700 rounded-lg p-4 text-left">
                    <div class="text-xs text-gray-500 mb-2">Example prompts:</div>
                    <div class="space-y-2 text-sm">
                      <div class="flex items-start gap-2">
                        <span class="text-blue-400">💬</span>
                        <span>"Create a REST API with JWT authentication"</span>
                      </div>
                      <div class="flex items-start gap-2">
                        <span class="text-blue-400">💬</span>
                        <span>"Build a 3-tier web app with React and FastAPI"</span>
                      </div>
                      <div class="flex items-start gap-2">
                        <span class="text-blue-400">💬</span>
                        <span>"Add Redis caching to my architecture"</span>
                      </div>
                    </div>
                  </div>
                </div>
              }
            >
              <div class="text-gray-400 text-center">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4 mx-auto" />
                <p>Loading architecture...</p>
              </div>
            </Show>
          </div>
        }
      >
        {/* Cytoscape container */}
        <div ref={containerRef} class="w-full h-full" />

        {/* Error display */}
        <Show when={architectureState.error}>
          <div class="absolute bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg">
            {architectureState.error}
          </div>
        </Show>

        {/* Architecture info overlay */}
        <div class="absolute top-4 left-4 bg-gray-800/90 backdrop-blur border border-gray-700 rounded-lg shadow-lg px-4 py-3">
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
              Components: <span class="text-white">{getFilteredComponents().length}</span>
            </span>
            <span class="text-gray-400">
              Connections: <span class="text-white">{architectureState.current?.connections.length || 0}</span>
            </span>
          </div>
        </div>

        {/* Controls overlay */}
        <div class="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur border border-gray-700 rounded-lg shadow-lg p-2 flex flex-col gap-2">
          <button
            onClick={() => cy?.fit(undefined, 50)}
            class="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
            title="Fit to view"
          >
            🎯 Fit
          </button>
          <button
            onClick={() => cy?.zoom(cy.zoom() * 1.2)}
            class="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
            title="Zoom in"
          >
            ➕
          </button>
          <button
            onClick={() => cy?.zoom(cy.zoom() / 1.2)}
            class="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
            title="Zoom out"
          >
            ➖
          </button>
        </div>
      </Show>
    </div>
  );
}
