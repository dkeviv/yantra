/**
 * ConnectionEdge - Custom React Flow edge for architecture connections
 * 
 * Displays:
 * - Connection type with appropriate arrow style
 * - Color-coded by connection type
 * - Label showing connection purpose
 * - Delete button on hover
 */

import { Show } from 'solid-js';
import { BaseEdge, EdgeLabelRenderer, EdgeProps, getBezierPath } from 'reactflow';
import * as archAPI from '../../api/architecture';
import { selectConnection, deleteConnection } from '../../stores/architectureStore';
import type { Connection } from '../../api/architecture';

interface ConnectionEdgeData {
  connection: Connection;
  isSelected: boolean;
}

export default function ConnectionEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps<ConnectionEdgeData>) {
  const { connection, isSelected } = data;

  // Calculate bezier path
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // Get connection styling
  const color = archAPI.getConnectionColor(connection.connection_type);
  const arrow = archAPI.getConnectionArrow(connection.connection_type);

  // Handle click to select
  const handleClick = (e: MouseEvent) => {
    e.stopPropagation();
    selectConnection(connection.id);
  };

  // Handle delete
  const handleDelete = async (e: MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Delete connection?`)) {
      try {
        await deleteConnection(connection.id);
      } catch (error) {
        console.error('Failed to delete connection:', error);
        alert('Failed to delete connection');
      }
    }
  };

  return (
    <>
      {/* Base edge path */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: color,
          strokeWidth: isSelected ? 3 : 2,
          opacity: isSelected ? 1 : 0.8,
        }}
      />

      {/* Edge label with connection info */}
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: 'all',
          }}
          class="nodrag nopan"
        >
          <div
            onClick={handleClick}
            class={`
              px-3 py-1 rounded-full text-xs font-medium
              bg-gray-800 border-2 shadow-lg
              transition-all duration-200
              hover:scale-110 cursor-pointer
              ${isSelected ? 'border-blue-500 ring-2 ring-blue-500/50' : 'border-gray-700'}
            `}
            style={{
              color: color,
              'border-color': isSelected ? '#3b82f6' : color,
            }}
          >
            {/* Connection type arrow */}
            <span class="mr-1">{arrow}</span>

            {/* Connection label or type name */}
            <span>
              {connection.label || connection.connection_type}
            </span>

            {/* Delete button (shown on hover or when selected) */}
            <Show when={isSelected}>
              <button
                onClick={handleDelete}
                class="ml-2 text-gray-400 hover:text-red-500 transition-colors inline-flex items-center"
                title="Delete connection"
              >
                <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </Show>
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
