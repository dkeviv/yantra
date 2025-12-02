/**
 * ComponentNode - Custom React Flow node for architecture components
 * 
 * Displays:
 * - Component name and type
 * - Status indicator with emoji (📋🔄✅⚠️)
 * - File count (2/5 files implemented)
 * - Description
 * - Edit and delete buttons
 */

import { Show } from 'solid-js';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import * as archAPI from '../../api/architecture';
import { selectComponent, deleteComponent } from '../../stores/architectureStore';
import type { Component } from '../../api/architecture';

interface ComponentNodeData {
  component: Component;
  isSelected: boolean;
}

export default function ComponentNode({ data }: NodeProps<ComponentNodeData>) {
  const { component, isSelected } = data;

  // Get status text from component_type
  const statusText = archAPI.getStatusText(component.component_type);
  
  // Calculate status color
  const statusColor = archAPI.getStatusColor(component.component_type);
  const statusIndicator = archAPI.getStatusIndicator(component.component_type);

  // Get file count text
  const fileCount = component.files.length;
  const fileCountText = () => {
    if (component.component_type.type === 'Planned') {
      return 'No files yet';
    } else if (component.component_type.type === 'InProgress') {
      const { completed, total } = component.component_type;
      return `${completed}/${total} files (in progress)`;
    } else if (component.component_type.type === 'Implemented') {
      return `${fileCount} file${fileCount !== 1 ? 's' : ''}`;
    } else {
      return `${fileCount} file${fileCount !== 1 ? 's' : ''} (misaligned)`;
    }
  };

  // Handle click to select
  const handleClick = (e: MouseEvent) => {
    e.stopPropagation();
    selectComponent(component.id);
  };

  // Handle delete
  const handleDelete = async (e: MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Delete component "${component.name}"?`)) {
      try {
        await deleteComponent(component.id);
      } catch (error) {
        console.error('Failed to delete component:', error);
        alert('Failed to delete component');
      }
    }
  };

  return (
    <div
      onClick={handleClick}
      class={`
        min-w-[200px] max-w-[300px]
        bg-gray-800 border-2 rounded-lg shadow-lg
        transition-all duration-200
        hover:shadow-xl hover:scale-105
        cursor-pointer
        ${isSelected ? 'border-blue-500 ring-2 ring-blue-500/50' : 'border-gray-700'}
      `}
      style={{ 'border-color': isSelected ? '#3b82f6' : statusColor }}
    >
      {/* Connection handles */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-blue-500"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-green-500"
      />

      {/* Header with status */}
      <div
        class="px-4 py-2 border-b border-gray-700 flex items-center justify-between"
        style={{ 'background-color': statusColor + '20' }}
      >
        <div class="flex items-center gap-2">
          <span class="text-2xl">{statusIndicator}</span>
          <div>
            <div class="text-white font-semibold">{component.name}</div>
            <div class="text-xs text-gray-400">{statusText}</div>
          </div>
        </div>
        <button
          onClick={handleDelete}
          class="text-gray-400 hover:text-red-500 transition-colors p-1"
          title="Delete component"
        >
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div class="px-4 py-3">
        {/* File count */}
        <div class="text-sm text-gray-400 mb-2">
          <svg class="w-4 h-4 inline mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
            />
          </svg>
          {fileCountText()}
        </div>

        {/* Description */}
        <Show when={component.description}>
          <div class="text-sm text-gray-300 mt-2 line-clamp-2">
            {component.description}
          </div>
        </Show>

        {/* Files list (if any) */}
        <Show when={component.files.length > 0}>
          <div class="mt-2 text-xs text-gray-500">
            <div class="font-semibold mb-1">Files:</div>
            <ul class="space-y-0.5 max-h-20 overflow-y-auto">
              {component.files.slice(0, 3).map((file) => (
                <li class="truncate" title={file}>
                  • {file.split('/').pop()}
                </li>
              ))}
              {component.files.length > 3 && (
                <li class="text-gray-600">
                  + {component.files.length - 3} more...
                </li>
              )}
            </ul>
          </div>
        </Show>
      </div>

      {/* Footer with timestamp */}
      <div class="px-4 py-2 border-t border-gray-700 text-xs text-gray-500">
        Updated: {new Date(component.updated_at).toLocaleDateString()}
      </div>
    </div>
  );
}
