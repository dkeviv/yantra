// File: src-ui/components/DocumentationPanels.tsx
// Purpose: Display Features, Decisions, Changes, and Plan extracted from chat
// Features:
//   - Searchable content across all tabs
//   - Natural language explanations (not code-centric)
//   - Decisions displayed from decision_log with timestamp
//   - Reduced padding for minimal UI philosophy
// Dependencies: solid-js, documentationStore, agentStore
// Last Updated: November 28, 2025

import { Component, createSignal, For, Show, onMount, createMemo } from 'solid-js';
import { documentationStore, type Task } from '../stores/documentationStore';
import { agentStore } from '../stores/agentStore';

type PanelType = 'features' | 'decisions' | 'changes' | 'plan';

export const DocumentationPanels: Component = () => {
  const [activePanel, setActivePanel] = createSignal<PanelType>('features');
  const [searchQuery, setSearchQuery] = createSignal('');

  // Load documentation on mount
  onMount(() => {
    documentationStore.loadDocumentation();
  });

  // Filtered content based on search query
  const filteredFeatures = createMemo(() => {
    const query = searchQuery().toLowerCase();
    if (!query) return documentationStore.features();
    return documentationStore.features().filter(f =>
      f.title.toLowerCase().includes(query) ||
      f.description.toLowerCase().includes(query)
    );
  });

  const filteredDecisions = createMemo(() => {
    const query = searchQuery().toLowerCase();
    if (!query) return documentationStore.decisions();
    return documentationStore.decisions().filter(d =>
      d.title.toLowerCase().includes(query) ||
      d.context.toLowerCase().includes(query) ||
      d.decision.toLowerCase().includes(query) ||
      d.rationale.toLowerCase().includes(query)
    );
  });

  const filteredChanges = createMemo(() => {
    const query = searchQuery().toLowerCase();
    if (!query) return documentationStore.changes();
    return documentationStore.changes().filter(c =>
      c.description.toLowerCase().includes(query) ||
      c.files.some(f => f.toLowerCase().includes(query))
    );
  });

  const filteredTasks = createMemo(() => {
    const query = searchQuery().toLowerCase();
    if (!query) return documentationStore.tasks();
    return documentationStore.tasks().filter(t =>
      t.title.toLowerCase().includes(query)
    );
  });

  const handleUserActionClick = (task: Task) => {
    if (task.userActionInstructions) {
      // Send user action instructions to chat
      agentStore.executeCommand(task.userActionInstructions);
    }
  };

  return (
    <div class="h-full flex flex-col bg-gray-800">
      {/* Tab Navigation - Reduced padding for minimal UI */}
      <div class="flex border-b border-gray-700">
        <button
          onClick={() => { setActivePanel('features'); setSearchQuery(''); }}
          class={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activePanel() === 'features'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          📋 Features
        </button>
        <button
          onClick={() => { setActivePanel('decisions'); setSearchQuery(''); }}
          class={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activePanel() === 'decisions'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          💡 Decisions
        </button>
        <button
          onClick={() => { setActivePanel('changes'); setSearchQuery(''); }}
          class={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activePanel() === 'changes'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          📝 Changes
        </button>
        <button
          onClick={() => { setActivePanel('plan'); setSearchQuery(''); }}
          class={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activePanel() === 'plan'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          🎯 Plan
        </button>
      </div>

      {/* Content Area - Reduced padding for minimal UI */}
      <div class="flex-1 overflow-y-auto p-2">
        {/* Features Panel */}
        <Show when={activePanel() === 'features'}>
          <div class="space-y-2">
            {/* Natural language explanation */}
            <div class="text-[11px] text-gray-400 italic mb-2 px-1">
              Features are automatically extracted from your chat conversations. As you describe what you want to build, Yantra identifies and tracks features, updating their status as implementation progresses.
            </div>
            
            {/* Search bar */}
            <div class="mb-2">
              <input
                type="text"
                placeholder="Search features..."
                value={searchQuery()}
                onInput={(e) => setSearchQuery(e.currentTarget.value)}
                class="w-full px-2 py-1 text-[11px] bg-gray-700 text-white rounded border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>

            <Show when={documentationStore.loading()}>
              <div class="text-xs text-gray-400">Loading features...</div>
            </Show>
            <Show when={documentationStore.error()}>
              <div class="text-xs text-red-400">{documentationStore.error()}</div>
            </Show>
            <Show when={!documentationStore.loading() && !documentationStore.error()}>
              <For each={filteredFeatures()}>
                {(feature) => (
                  <div class="bg-gray-700 rounded-lg p-2">
                    <div class="flex items-start justify-between mb-1">
                      <h4 class="text-xs font-medium text-white flex-1 break-words pr-2">{feature.title}</h4>
                      <span class={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${
                        feature.status === 'completed' ? 'bg-green-600 text-white' :
                        feature.status === 'in-progress' ? 'bg-yellow-600 text-white' :
                        'bg-gray-600 text-gray-300'
                      }`}>
                        {feature.status === 'completed' ? '✅ Done' :
                         feature.status === 'in-progress' ? '🔄 In Progress' :
                         '⏳ Planned'}
                      </span>
                    </div>
                    <p class="text-[11px] text-gray-300 mb-1">{feature.description}</p>
                    <p class="text-[10px] text-gray-500 italic">{feature.extractedFrom}</p>
                  </div>
                )}
              </For>
              <Show when={filteredFeatures().length === 0 && searchQuery()}>
                <div class="text-xs text-gray-400 text-center py-4">No features found matching "{searchQuery()}"</div>
              </Show>
            </Show>
          </div>
        </Show>

        {/* Decisions Panel */}
        <Show when={activePanel() === 'decisions'}>
          <div class="space-y-2">
            {/* Natural language explanation */}
            <div class="text-[11px] text-gray-400 italic mb-2 px-1">
              Critical technical decisions are logged here with full context. Each decision includes why it was made, what alternatives were considered, and the rationale behind the choice.
            </div>
            
            {/* Search bar */}
            <div class="mb-2">
              <input
                type="text"
                placeholder="Search decisions..."
                value={searchQuery()}
                onInput={(e) => setSearchQuery(e.currentTarget.value)}
                class="w-full px-2 py-1 text-[11px] bg-gray-700 text-white rounded border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>

            <Show when={!documentationStore.loading() && !documentationStore.error()}>
              <For each={filteredDecisions()}>
                {(decision) => (
                  <div class="bg-gray-700 rounded-lg p-2">
                    <h4 class="text-xs font-medium text-white mb-1.5 break-words">{decision.title}</h4>
                    <div class="space-y-1.5 text-[11px]">
                      <div>
                        <span class="text-gray-400 font-medium">Context:</span>
                        <p class="text-gray-300 mt-0.5">{decision.context}</p>
                      </div>
                      <div>
                        <span class="text-gray-400 font-medium">Decision:</span>
                        <p class="text-white font-medium mt-0.5">{decision.decision}</p>
                      </div>
                      <div>
                        <span class="text-gray-400 font-medium">Rationale:</span>
                        <p class="text-gray-300 mt-0.5">{decision.rationale}</p>
                      </div>
                      <div class="text-[10px] text-gray-500">
                        {new Date(decision.timestamp).toLocaleString()}
                      </div>
                    </div>
                  </div>
                )}
              </For>
              <Show when={filteredDecisions().length === 0 && searchQuery()}>
                <div class="text-xs text-gray-400 text-center py-4">No decisions found matching "{searchQuery()}"</div>
              </Show>
            </Show>
          </div>
        </Show>

        {/* Changes Panel */}
        <Show when={activePanel() === 'changes'}>
          <div class="space-y-2">
            {/* Natural language explanation */}
            <div class="text-[11px] text-gray-400 italic mb-2 px-1">
              Complete audit trail of all code changes. Track what files were added, modified, or deleted, along with timestamps and descriptions.
            </div>
            
            {/* Search bar */}
            <div class="mb-2">
              <input
                type="text"
                placeholder="Search changes..."
                value={searchQuery()}
                onInput={(e) => setSearchQuery(e.currentTarget.value)}
                class="w-full px-2 py-1 text-[11px] bg-gray-700 text-white rounded border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>

            <Show when={!documentationStore.loading() && !documentationStore.error()}>
              <For each={filteredChanges()}>
                {(change) => (
                  <div class="bg-gray-700 rounded-lg p-2">
                    <div class="flex items-start justify-between mb-1">
                      <span class={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${
                        change.changeType === 'file-added' ? 'bg-green-600 text-white' :
                        change.changeType === 'file-modified' ? 'bg-blue-600 text-white' :
                        change.changeType === 'file-deleted' ? 'bg-red-600 text-white' :
                        'bg-purple-600 text-white'
                      }`}>
                        {change.changeType}
                      </span>
                      <span class="text-[10px] text-gray-500">
                        {new Date(change.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p class="text-xs text-white mb-1.5 break-words">{change.description}</p>
                    <div class="text-[11px] text-gray-400">
                      <For each={change.files}>
                        {(file) => <div class="truncate">📄 {file}</div>}
                      </For>
                    </div>
                  </div>
                )}
              </For>
              <Show when={filteredChanges().length === 0 && searchQuery()}>
                <div class="text-xs text-gray-400 text-center py-4">No changes found matching "{searchQuery()}"</div>
              </Show>
            </Show>
          </div>
        </Show>

        {/* Plan Panel */}
        <Show when={activePanel() === 'plan'}>
          <div class="space-y-2">
            {/* Natural language explanation */}
            <div class="text-[11px] text-gray-400 italic mb-2 px-1">
              Your project plan with tasks organized by milestones. Dependencies are tracked automatically, and tasks requiring your input are highlighted.
            </div>
            
            {/* Search bar */}
            <div class="mb-2">
              <input
                type="text"
                placeholder="Search plan..."
                value={searchQuery()}
                onInput={(e) => setSearchQuery(e.currentTarget.value)}
                class="w-full px-2 py-1 text-[11px] bg-gray-700 text-white rounded border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>

            <Show when={!documentationStore.loading() && !documentationStore.error()}>
              {/* Group by milestone */}
              <div class="space-y-2">
                <div>
                  <h4 class="text-xs font-semibold text-white mb-1.5 flex items-center">
                    <span class="mr-1.5">🎯</span> MVP Milestone
                  </h4>
                  <div class="space-y-1.5">
                    <For each={filteredTasks().filter((t: Task) => t.milestone === 'MVP')}>
                      {(task) => (
                        <div class="bg-gray-700 rounded-lg p-2">
                          <div class="flex items-start justify-between mb-1">
                            <h5 class="text-xs font-medium text-white flex-1 break-words pr-2">{task.title}</h5>
                            <span class={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${
                              task.status === 'completed' ? 'bg-green-600 text-white' :
                              task.status === 'in-progress' ? 'bg-yellow-600 text-white' :
                              'bg-gray-600 text-gray-300'
                            }`}>
                              {task.status === 'completed' ? '✅' :
                               task.status === 'in-progress' ? '🔄' :
                               '⏳'}
                            </span>
                          </div>
                          
                          <Show when={task.dependencies.length > 0}>
                            <div class="text-[11px] text-gray-400 mb-1 truncate">
                              Depends on: {task.dependencies.join(', ')}
                            </div>
                          </Show>
                          
                          <Show when={task.requiresUserAction}>
                            <button
                              onClick={() => handleUserActionClick(task)}
                              class="mt-1.5 w-full text-[11px] px-2 py-1.5 bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors flex items-center justify-center"
                            >
                              <span class="mr-1.5">👤</span>
                              User Action Required - Click for Instructions
                            </button>
                          </Show>
                        </div>
                      )}
                    </For>
                    <Show when={filteredTasks().filter((t: Task) => t.milestone === 'MVP').length === 0 && searchQuery()}>
                      <div class="text-xs text-gray-400 text-center py-4">No tasks found matching "{searchQuery()}"</div>
                    </Show>
                  </div>
                </div>
              </div>
            </Show>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default DocumentationPanels;
