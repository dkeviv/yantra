/**
 * ArchitectureView - Main wrapper component
 * 
 * Combines:
 * - HierarchicalTabs for filtering
 * - ArchitectureCanvas for visualization
 * - Toolbar for actions (create, load, save, export, undo/redo)
 */

import { Show } from 'solid-js';
import ArchitectureCanvas from './ArchitectureCanvas';
import HierarchicalTabs from './HierarchicalTabs';
import {
  architectureState,
  saveVersion,
  exportArchitecture,
  undo,
  redo,
  canUndo,
  canRedo,
  addComponent,
} from '../../stores/architectureStore';
import type { ComponentCategory } from '../../api/architecture';

export default function ArchitectureView() {
  // Handle save version
  const handleSaveVersion = async () => {
    const description = prompt('Version description (optional):');

    try {
      await saveVersion(description || undefined);
      alert('Version saved successfully!');
    } catch (error) {
      console.error('Failed to save version:', error);
      alert('Failed to save version');
    }
  };

  // Handle export
  const handleExport = async (format: 'markdown' | 'mermaid' | 'json') => {
    try {
      const exportedData = await exportArchitecture(format);

      // Copy to clipboard
      await navigator.clipboard.writeText(exportedData);
      alert(`Exported as ${format.toUpperCase()} and copied to clipboard!`);
    } catch (error) {
      console.error('Failed to export architecture:', error);
      alert('Failed to export architecture');
    }
  };

  // Handle add component
  const handleAddComponent = async () => {
    const name = prompt('Component name:');
    if (!name) return;

    const type = prompt('Component type (Backend/Frontend/Database/External/Utility):') as ComponentCategory;
    if (!type) return;

    const description = prompt('Description (optional):');

    try {
      await addComponent(
        name,
        type,
        { x: Math.random() * 400, y: Math.random() * 300 }, // Random position
        description || undefined
      );
    } catch (error) {
      console.error('Failed to add component:', error);
      alert('Failed to add component');
    }
  };

  return (
    <div class="flex flex-col h-full">
      {/* Toolbar - only show when architecture is loaded */}
      <Show when={architectureState.current}>
        <div class="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center justify-between">
          <div class="flex items-center gap-2">
            {/* Component actions */}
            <button
              onClick={handleAddComponent}
              class="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition-colors"
            >
              ➕ Add Component
            </button>
            <button
              onClick={handleSaveVersion}
              class="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded text-sm transition-colors"
            >
              💾 Save Version
            </button>
          </div>

          {/* Right side actions */}
          <div class="flex items-center gap-2">
            {/* Undo/Redo */}
            <button
              onClick={() => undo()}
              disabled={!canUndo()}
              class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Undo"
            >
              ↶
            </button>
            <button
              onClick={() => redo()}
              disabled={!canRedo()}
              class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Redo"
            >
              ↷
            </button>

            {/* Export dropdown */}
            <div class="ml-4 pl-4 border-l border-gray-700 flex items-center gap-2">
              <button
                onClick={() => handleExport('markdown')}
                class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
                title="Export as Markdown"
              >
                📝 MD
              </button>
              <button
                onClick={() => handleExport('mermaid')}
                class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
                title="Export as Mermaid diagram"
              >
                📊 Mermaid
              </button>
              <button
                onClick={() => handleExport('json')}
                class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
                title="Export as JSON"
              >
                📦 JSON
              </button>
            </div>
          </div>
        </div>

        {/* Hierarchical tabs for filtering */}
        <HierarchicalTabs />
      </Show>

      {/* Main canvas */}
      <div class="flex-1">
        <ArchitectureCanvas />
      </div>
    </div>
  );
}
