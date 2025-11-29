/**
 * ArchitectureView - Main wrapper component (READ-ONLY, Agent-Driven)
 * 
 * DESIGN PRINCIPLE: This is an agentic platform. All architecture operations
 * happen through the agent via chat, not through manual UI controls.
 * 
 * This component only VISUALIZES agent-generated architecture.
 * 
 * Features:
 * - HierarchicalTabs for filtering by component type
 * - ArchitectureCanvas for visualization
 * - Version history display (read-only)
 * - Export functionality (called by agent commands)
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
  // Handle export (called by agent commands, not directly by user)
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

  // Expose for agent to call
  (window as any).__yantraArchitectureExport = handleExport;

  return (
    <div class="flex flex-col h-full bg-gray-900">
      {/* Hierarchical tabs for filtering - only show when architecture exists */}
      <Show when={architectureState.current && !architectureState.isLoading}>
        <HierarchicalTabs />
      </Show>

      {/* Main canvas (handles its own empty state) */}
      <div class="flex-1 relative">
        <ArchitectureCanvas />
        
        {/* Version info overlay - only show when architecture exists */}
        <Show when={architectureState.current && !architectureState.isLoading}>
          <div class="absolute bottom-4 right-4 bg-gray-800/90 backdrop-blur border border-gray-700 rounded-lg shadow-lg px-3 py-2 text-xs text-gray-400">
            <div class="flex items-center gap-2">
              <span>Version: {architectureState.current?.id.slice(0, 8) || 'N/A'}</span>
              <span>•</span>
              <span>Updated: {new Date(architectureState.current?.updated_at || Date.now()).toLocaleTimeString()}</span>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}
