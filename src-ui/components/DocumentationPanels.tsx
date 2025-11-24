// File: src-ui/components/DocumentationPanels.tsx
// Purpose: Display Features, Decisions, Changes, and Plan extracted from chat
// Dependencies: solid-js, appStore
// Last Updated: November 23, 2025

import { Component, createSignal, For, Show } from 'solid-js';

type PanelType = 'features' | 'decisions' | 'changes' | 'plan';

interface Feature {
  id: string;
  title: string;
  description: string;
  status: 'planned' | 'in-progress' | 'completed';
  extractedFrom: string; // chat message or file
}

interface Decision {
  id: string;
  title: string;
  context: string;
  decision: string;
  rationale: string;
  timestamp: string;
}

interface Change {
  id: string;
  type: 'file-added' | 'file-modified' | 'file-deleted' | 'function-added' | 'function-removed';
  description: string;
  files: string[];
  timestamp: string;
}

interface Task {
  id: string;
  title: string;
  status: 'completed' | 'in-progress' | 'pending';
  milestone: string; // 'MVP', 'Phase 1', 'Phase 2', etc.
  dependencies: string[]; // IDs of other tasks or user actions
  requiresUserAction: boolean;
  userActionInstructions?: string;
}

export const DocumentationPanels: Component = () => {
  const [activePanel, setActivePanel] = createSignal<PanelType>('features');

  // Mock data - will be populated from backend
  const features = (): Feature[] => [
    {
      id: '1',
      title: 'Agent-first UI Control',
      description: 'All UI actions controlled via natural language chat commands',
      status: 'completed',
      extractedFrom: 'Chat: "Every action should be agent-controlled"'
    },
    {
      id: '2',
      title: 'LLM Configuration Panel',
      description: 'Model selection and API key configuration',
      status: 'completed',
      extractedFrom: 'Chat: "Add LLM model selector"'
    }
  ];

  const decisions = (): Decision[] => [
    {
      id: '1',
      title: 'Remove Confidence from UI',
      context: 'Confidence scores shown to users',
      decision: 'Hide confidence scores - internal agent metric only',
      rationale: 'Users want results, not implementation details. Agent handles quality internally.',
      timestamp: new Date().toISOString()
    }
  ];

  const changes = (): Change[] => [
    {
      id: '1',
      type: 'file-modified',
      description: 'Removed confidence display from AgentStatus',
      files: ['src-ui/components/AgentStatus.tsx'],
      timestamp: new Date().toISOString()
    }
  ];

  const tasks = (): Task[] => [
    {
      id: '1',
      title: 'Implement Agent Command System',
      status: 'completed',
      milestone: 'MVP',
      dependencies: [],
      requiresUserAction: false
    },
    {
      id: '2',
      title: 'Configure API Keys',
      status: 'pending',
      milestone: 'MVP',
      dependencies: ['1'],
      requiresUserAction: true,
      userActionInstructions: 'Please configure your API keys for Claude and GPT-4 in the LLM settings panel.'
    }
  ];

  const handleUserActionClick = (task: Task) => {
    if (task.userActionInstructions) {
      // TODO: Send instructions to chat
      console.log('User action required:', task.userActionInstructions);
    }
  };

  return (
    <div class="h-full flex flex-col bg-gray-800">
      {/* Tab Navigation */}
      <div class="flex border-b border-gray-700">
        <button
          onClick={() => setActivePanel('features')}
          class={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activePanel() === 'features'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          📋 Features
        </button>
        <button
          onClick={() => setActivePanel('decisions')}
          class={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activePanel() === 'decisions'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          💡 Decisions
        </button>
        <button
          onClick={() => setActivePanel('changes')}
          class={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activePanel() === 'changes'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          📝 Changes
        </button>
        <button
          onClick={() => setActivePanel('plan')}
          class={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activePanel() === 'plan'
              ? 'bg-gray-700 text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          🎯 Plan
        </button>
      </div>

      {/* Content Area */}
      <div class="flex-1 overflow-y-auto p-4">
        {/* Features Panel */}
        <Show when={activePanel() === 'features'}>
          <div class="space-y-3">
            <div class="text-sm text-gray-400 mb-4">
              Features automatically extracted from chat conversations and requirements
            </div>
            <For each={features()}>
              {(feature) => (
                <div class="bg-gray-700 rounded-lg p-3">
                  <div class="flex items-start justify-between mb-2">
                    <h4 class="text-sm font-medium text-white">{feature.title}</h4>
                    <span class={`text-xs px-2 py-1 rounded ${
                      feature.status === 'completed' ? 'bg-green-600 text-white' :
                      feature.status === 'in-progress' ? 'bg-yellow-600 text-white' :
                      'bg-gray-600 text-gray-300'
                    }`}>
                      {feature.status === 'completed' ? '✅ Done' :
                       feature.status === 'in-progress' ? '🔄 In Progress' :
                       '⏳ Planned'}
                    </span>
                  </div>
                  <p class="text-xs text-gray-300 mb-2">{feature.description}</p>
                  <p class="text-xs text-gray-500 italic">{feature.extractedFrom}</p>
                </div>
              )}
            </For>
          </div>
        </Show>

        {/* Decisions Panel */}
        <Show when={activePanel() === 'decisions'}>
          <div class="space-y-3">
            <div class="text-sm text-gray-400 mb-4">
              Critical decisions made during development
            </div>
            <For each={decisions()}>
              {(decision) => (
                <div class="bg-gray-700 rounded-lg p-3">
                  <h4 class="text-sm font-medium text-white mb-2">{decision.title}</h4>
                  <div class="space-y-2 text-xs">
                    <div>
                      <span class="text-gray-400">Context:</span>
                      <p class="text-gray-300 mt-1">{decision.context}</p>
                    </div>
                    <div>
                      <span class="text-gray-400">Decision:</span>
                      <p class="text-white font-medium mt-1">{decision.decision}</p>
                    </div>
                    <div>
                      <span class="text-gray-400">Rationale:</span>
                      <p class="text-gray-300 mt-1">{decision.rationale}</p>
                    </div>
                    <div class="text-gray-500 text-xs mt-2">
                      {new Date(decision.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              )}
            </For>
          </div>
        </Show>

        {/* Changes Panel */}
        <Show when={activePanel() === 'changes'}>
          <div class="space-y-3">
            <div class="text-sm text-gray-400 mb-4">
              Change log for audit and transparency
            </div>
            <For each={changes()}>
              {(change) => (
                <div class="bg-gray-700 rounded-lg p-3">
                  <div class="flex items-start justify-between mb-2">
                    <span class={`text-xs px-2 py-1 rounded ${
                      change.type === 'file-added' ? 'bg-green-600 text-white' :
                      change.type === 'file-modified' ? 'bg-blue-600 text-white' :
                      change.type === 'file-deleted' ? 'bg-red-600 text-white' :
                      'bg-purple-600 text-white'
                    }`}>
                      {change.type}
                    </span>
                    <span class="text-xs text-gray-500">
                      {new Date(change.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p class="text-sm text-white mb-2">{change.description}</p>
                  <div class="text-xs text-gray-400">
                    <For each={change.files}>
                      {(file) => <div>📄 {file}</div>}
                    </For>
                  </div>
                </div>
              )}
            </For>
          </div>
        </Show>

        {/* Plan Panel */}
        <Show when={activePanel() === 'plan'}>
          <div class="space-y-4">
            <div class="text-sm text-gray-400 mb-4">
              Tasks, milestones, and dependencies
            </div>
            
            {/* Group by milestone */}
            <div class="space-y-4">
              <div>
                <h4 class="text-sm font-semibold text-white mb-3 flex items-center">
                  <span class="mr-2">🎯</span> MVP Milestone
                </h4>
                <div class="space-y-2">
                  <For each={tasks().filter(t => t.milestone === 'MVP')}>
                    {(task) => (
                      <div class="bg-gray-700 rounded-lg p-3">
                        <div class="flex items-start justify-between mb-2">
                          <h5 class="text-sm font-medium text-white flex-1">{task.title}</h5>
                          <span class={`text-xs px-2 py-1 rounded ml-2 ${
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
                          <div class="text-xs text-gray-400 mb-2">
                            Depends on: {task.dependencies.join(', ')}
                          </div>
                        </Show>
                        
                        <Show when={task.requiresUserAction}>
                          <button
                            onClick={() => handleUserActionClick(task)}
                            class="mt-2 w-full text-xs px-3 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors flex items-center justify-center"
                          >
                            <span class="mr-2">👤</span>
                            User Action Required - Click for Instructions
                          </button>
                        </Show>
                      </div>
                    )}
                  </For>
                </div>
              </div>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default DocumentationPanels;
