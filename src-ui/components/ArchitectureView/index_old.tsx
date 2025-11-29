/**
 * ArchitectureView - Main wrapper component (READ-ONLY, Agent-Driven)
 * 
 * DESIGN PRINCIPLE: This is an agentic platform. All architecture operations
 * happen through the agent via chat, not through manual UI controls.
 * 
 * This component only VISUALIZES agent-generated architecture.
 * 
 * Features:
 * - HierarchicalTabs for filtering
 * - ArchitectureCanvas for visualization
 * - Version history display (read-only)
 * - Export functionality (via agent commands)
 * 
 * NO manual create/edit/save/delete operations!
 */

import { Show } from 'solid-js';
import ArchitectureCanvas from './ArchitectureCanvas';
import HierarchicalTabs from './HierarchicalTabs';
import {
  architectureState,
  exportArchitecture,
} from '../../stores/architectureStore';

export default function ArchitectureView() {
  // Handle export (called by agent commands, not user buttons)
  const handleExport = async (format: 'markdown' | 'mermaid' | 'json') => {
    try {
      const exportedData = await exportArchitecture(format);
      
      // Copy to clipboard
      await navigator.clipboard.writeText(exportedData);
      
      // Could also show a toast notification here
      console.log(`Exported architecture as ${format.toUpperCase()}`);
    } catch (error) {
      console.error('Failed to export architecture:', error);
    }
  };

  return (
    <div class="flex flex-col h-full">
      {/* Hierarchical tabs for filtering - only show when architecture exists */}
      <Show when={architectureState.current && !architectureState.isLoading}>
        <HierarchicalTabs />
      </Show>

      {/* Main canvas (handles its own empty state) */}
      <div class="flex-1">
        <ArchitectureCanvas />
      </div>

      {/* Version info overlay - only show when architecture exists */}
      <Show when={architectureState.current && !architectureState.isLoading}>
        <div class="absolute bottom-4 right-4 bg-gray-800/90 backdrop-blur border border-gray-700 rounded-lg shadow-lg px-3 py-2 text-xs text-gray-400">
          Version: {architectureState.current?.id.slice(0, 8) || 'N/A'}
          <span class="mx-2">•</span>
          Last updated: {new Date(architectureState.current?.updated_at || Date.now()).toLocaleTimeString()}
        </div>
      </Show>
    </div>
  );
}
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
