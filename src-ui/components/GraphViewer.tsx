/**
 * GraphViewer.tsx
 *
 * Purpose: Unified graph viewer with tabs for Code Dependencies and Traceability
 * Provides seamless switching between different graph visualizations
 *
 * Features:
 * - Tabbed interface (Code Deps / Traceability)
 * - Persistent tab selection
 * - Shared layout and styling
 *
 * Created: December 9, 2025
 */

import { createSignal, Show } from 'solid-js';
import DependencyGraph from './DependencyGraph';
import { YDocTraceabilityGraph } from './YDocTraceabilityGraph';
import './GraphViewer.css';

export default function GraphViewer() {
  // Load last selected tab from localStorage, default to 'code'
  const savedTab = localStorage.getItem('graphViewer_activeTab') || 'code';
  const [activeTab, setActiveTab] = createSignal<'code' | 'traceability'>(
    savedTab as 'code' | 'traceability'
  );
  const [selectedBlockId, setSelectedBlockId] = createSignal<string | undefined>(undefined);

  const switchTab = (tab: 'code' | 'traceability') => {
    setActiveTab(tab);
    localStorage.setItem('graphViewer_activeTab', tab);
  };

  const handleNodeClick = (nodeId: string, nodeType: string) => {
    console.log('Node clicked:', nodeId, nodeType);
    setSelectedBlockId(nodeId);
  };

  return (
    <div class="graph-viewer">
      {/* Tab Bar */}
      <div class="graph-tabs">
        <button
          class={`tab-button ${activeTab() === 'code' ? 'active' : ''}`}
          onClick={() => switchTab('code')}
          title="View code dependencies (files, functions, classes)"
        >
          <span class="tab-label">Code Dependencies</span>
        </button>
        <button
          class={`tab-button ${activeTab() === 'traceability' ? 'active' : ''}`}
          onClick={() => switchTab('traceability')}
          title="View traceability chain (requirements, specs, code, tests)"
        >
          <span class="tab-label">Traceability</span>
        </button>
      </div>

      {/* Graph Content */}
      <div class="graph-content">
        <Show when={activeTab() === 'code'}>
          <DependencyGraph />
        </Show>

        <Show when={activeTab() === 'traceability'}>
          <YDocTraceabilityGraph blockId={selectedBlockId()} onNodeClick={handleNodeClick} />
        </Show>
      </div>

      {/* Quick Help */}
      <div class="graph-help">
        <Show when={activeTab() === 'code'}>
          <div class="help-text">
            <strong>Code Dependencies:</strong> Shows file/function/class relationships. Use filters
            above to toggle visibility. Click nodes for details.
          </div>
        </Show>
        <Show when={activeTab() === 'traceability'}>
          <div class="help-text">
            <strong>Traceability:</strong> Shows requirement → architecture → spec → code → test
            chains. Select a YDoc block to view its traceability.
          </div>
        </Show>
      </div>
    </div>
  );
}
