// File: src-ui/components/ChatPanel.tsx
// Purpose: Agent interface for user-agent interaction with minimal UI design
// Features:
//   - Model selection in header (top right)
//   - Reduced font sizes and padding for minimal UI
//   - Send button inside textarea container
//   - Terminal-like message display
// Dependencies: solid-js, appStore, agentStore, llmStore
// Last Updated: November 29, 2025

import { Component, For, createSignal, Show, createEffect, onMount } from 'solid-js';
import { appStore } from '../stores/appStore';
import { llmApi, type ChatMessage, type ModelInfo } from '../api/llm';
import { terminalStore } from '../stores/terminalStore';
import { layoutStore } from '../stores/layoutStore';
import LLMSettings from './LLMSettings';
import StatusIndicator from './StatusIndicator';

const ChatPanel: Component = () => {
  const [input, setInput] = createSignal('');
  const [selectedModel, setSelectedModel] = createSignal('');
  const [availableModels, setAvailableModels] = createSignal<ModelInfo[]>([]);
  const [showApiConfig, setShowApiConfig] = createSignal(false);
  const [currentProvider, setCurrentProvider] = createSignal<string>('');

  // Load available models when component mounts or provider changes
  onMount(async () => {
    try {
      const config = await llmApi.getConfig();
      const provider = config.primary_provider.toLowerCase();
      setCurrentProvider(provider);
      await loadModelsForProvider(provider);
    } catch (error) {
      console.error('Failed to load initial config:', error);
    }
  });

  // Function to load models for a specific provider
  const loadModelsForProvider = async (provider: string) => {
    try {
      // Get all available models
      const allModels = await llmApi.getAvailableModels(provider as any);

      // Get user's selected models
      const selectedIds = await llmApi.getSelectedModels();

      // Filter: Show selected models only, or all if no selection
      const modelsToShow =
        selectedIds.length > 0 ? allModels.filter((m) => selectedIds.includes(m.id)) : allModels;

      setAvailableModels(modelsToShow);

      // Set default model if no model is selected
      if (!selectedModel() && modelsToShow.length > 0) {
        const defaultModel = await llmApi.getDefaultModel(provider as any);
        // Use default if it's in the filtered list, otherwise use first
        const modelToUse = modelsToShow.find((m) => m.id === defaultModel)
          ? defaultModel
          : modelsToShow[0].id;
        setSelectedModel(modelToUse);
      }
    } catch (error) {
      console.error(`Failed to load models for ${provider}:`, error);
      setAvailableModels([]);
    }
  };

  // Watch for provider changes (when user changes provider in settings)
  createEffect(() => {
    // Re-check config periodically or on certain events
    const interval = setInterval(async () => {
      try {
        const config = await llmApi.getConfig();
        const provider = config.primary_provider.toLowerCase();
        if (provider !== currentProvider()) {
          setCurrentProvider(provider);
          await loadModelsForProvider(provider);
        }
      } catch (error) {
        // Silently fail - user might not have configured provider yet
      }
    }, 2000); // Check every 2 seconds

    return () => clearInterval(interval);
  });

  const handleSend = async () => {
    const message = input().trim();
    if (!message) return;

    // Add user message
    appStore.addMessage('user', message);
    setInput('');
    appStore.setIsGenerating(true);

    try {
      // Prepare conversation history for LLM
      const conversationHistory: ChatMessage[] = appStore
        .messages()
        .filter((msg) => msg.role !== 'system')
        .map((msg) => ({
          role: msg.role,
          content: msg.content,
        }));

      // Call natural language chat API
      const response = await llmApi.chat(message, conversationHistory);

      // Handle the response
      appStore.addMessage('assistant', response.response);

      // Execute any detected actions
      if (response.action) {
        await executeDetectedAction(response.action);
      }
    } catch (error) {
      appStore.addMessage('assistant', `❌ Error: ${error}`);
    } finally {
      appStore.setIsGenerating(false);
    }
  };

  const executeDetectedAction = async (action: any) => {
    try {
      switch (action.action_type) {
        case 'run_command': {
          const command = action.parameters.command;
          if (command) {
            const result = await terminalStore.executeCommand(command);
            appStore.addMessage('assistant', `✅ Command executed: ${command}\nResult: ${result}`);
          }
          break;
        }
        case 'toggle_panel': {
          const panel = action.parameters.panel;
          if (panel === 'dependencies') {
            appStore.setActiveView('dependencies');
          } else if (panel === 'terminal') {
            // Toggle terminal
          } else if (panel === 'filetree') {
            appStore.setShowFileTree(!appStore.showFileTree());
          }
          break;
        }
        case 'generate_code': {
          // Check if this is a full project creation request
          const intent = action.parameters.intent || '';
          const lowerIntent = intent.toLowerCase();

          // Detect project creation keywords
          const isProjectCreation =
            lowerIntent.includes('create a project') ||
            lowerIntent.includes('create an app') ||
            lowerIntent.includes('build a') ||
            lowerIntent.includes('generate a project') ||
            (lowerIntent.includes('create') &&
              (lowerIntent.includes('api') ||
                lowerIntent.includes('application') ||
                lowerIntent.includes('website') ||
                lowerIntent.includes('service')));

          if (isProjectCreation) {
            // E2E project creation
            appStore.addMessage('assistant', '� Starting autonomous project creation...');

            // Determine template from intent
            let template: string | undefined;
            if (
              lowerIntent.includes('express') ||
              (lowerIntent.includes('rest') && lowerIntent.includes('api'))
            ) {
              template = 'express-api';
            } else if (lowerIntent.includes('react')) {
              template = 'react-app';
            } else if (lowerIntent.includes('fastapi')) {
              template = 'fastapi-service';
            } else if (lowerIntent.includes('cli')) {
              template = 'node-cli';
            }

            // Ask for project directory if not in workspace
            const projectDir = appStore.projectPath() || '/tmp/yantra-project';

            try {
              const result = await llmApi.createProjectAutonomous(intent, projectDir, template);

              if (result.success) {
                let response = `✅ Project created successfully!\n\n`;
                response += `📁 Location: ${result.project_dir}\n`;
                response += `📝 Generated ${result.generated_files.length} files\n`;

                if (result.test_results) {
                  response += `🧪 Tests: ${result.test_results.passed}/${result.test_results.total} passed`;
                  if (result.test_results.coverage_percent) {
                    response += ` (${result.test_results.coverage_percent.toFixed(1)}% coverage)`;
                  }
                  response += '\n';
                }

                response += `\nFiles created:\n${result.generated_files.map((f) => `  - ${f}`).join('\n')}`;

                appStore.addMessage('assistant', response);
              } else {
                let response = `❌ Project creation encountered issues:\n\n`;
                response += result.errors.map((e) => `  - ${e}`).join('\n');
                response += `\n\n${result.generated_files.length} files were created before errors occurred.`;
                appStore.addMessage('assistant', response);
              }
            } catch (error) {
              appStore.addMessage('assistant', `❌ Project creation failed: ${error}`);
            }
          } else {
            // Single file generation (existing behavior)
            appStore.addMessage(
              'assistant',
              '🔧 Single-file code generation will be available soon!'
            );
          }
          break;
        }
      }
    } catch (error) {
      appStore.addMessage('assistant', `⚠️ Action failed: ${error}`);
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div class="flex flex-col h-full" style={{ 'background-color': 'var(--bg-primary)' }}>
      {/* Header - Model selection in top right */}
      <div
        class="px-3 py-2 flex items-center justify-between"
        style={{
          'background-color': 'var(--bg-secondary)',
          'border-bottom': '1px solid var(--border-primary)',
        }}
      >
        <div class="flex items-center gap-2">
          <h2 class="text-base font-bold" style={{ color: 'var(--text-primary)' }}>
            Agent
          </h2>
          <StatusIndicator status={appStore.isGenerating() ? 'running' : 'idle'} size="small" />
          {/* Expand Button */}
          <button
            onClick={() => layoutStore.togglePanelExpansion('agent')}
            class="ml-1 px-1.5 py-0.5 text-xs rounded hover:opacity-70 transition-opacity"
            style={{
              'background-color': layoutStore.isExpanded('agent')
                ? 'var(--accent-primary)'
                : 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
            }}
            title={layoutStore.isExpanded('agent') ? 'Collapse panel' : 'Expand panel'}
          >
            {layoutStore.isExpanded('agent') ? '◀' : '▶'}
          </button>
        </div>
        <div class="flex items-center gap-2">
          {/* Model Selection */}
          <select
            value={selectedModel()}
            onChange={(e) => setSelectedModel(e.currentTarget.value)}
            class="text-[11px] px-2 py-1 rounded focus:outline-none"
            style={{
              'background-color': 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-secondary)',
            }}
            title="Select LLM model"
            aria-label="Select LLM model"
            disabled={availableModels().length === 0}
          >
            <Show when={availableModels().length === 0}>
              <option value="">No models available</option>
            </Show>
            <Show when={availableModels().length > 0}>
              <For each={availableModels()}>
                {(model) => (
                  <option value={model.id} title={model.description}>
                    {model.name}
                  </option>
                )}
              </For>
            </Show>
          </select>
          {/* API Config Button */}
          <button
            onClick={() => setShowApiConfig(!showApiConfig())}
            class="p-1 transition-colors"
            style={{
              color: 'var(--text-tertiary)',
            }}
            title="API Configuration"
          >
            ⚙️
          </button>
        </div>
      </div>

      {/* Messages - Terminal-like with reduced font and padding */}
      <div class="flex-1 overflow-y-auto px-2 py-1 space-y-0.5">
        <For each={appStore.messages()}>
          {(message) => (
            <div class="font-mono text-[11px] leading-relaxed">
              <span
                style={{
                  color:
                    message.role === 'user' ? 'var(--status-success)' : 'var(--accent-primary)',
                }}
              >
                {message.role === 'user' ? 'You' : 'Yantra'}
              </span>
              <span style={{ color: 'var(--text-tertiary)' }} class="mx-1">
                ›
              </span>
              <span style={{ color: 'var(--text-secondary)' }}>{message.content}</span>
            </div>
          )}
        </For>

        {appStore.isGenerating() && (
          <div class="font-mono text-[11px] leading-relaxed">
            <span style={{ color: 'var(--accent-primary)' }}>Yantra</span>
            <span style={{ color: 'var(--text-tertiary)' }} class="mx-1">
              ›
            </span>
            <span style={{ color: 'var(--text-tertiary)' }} class="animate-pulse">
              Generating...
            </span>
          </div>
        )}
      </div>

      {/* API Config Modal */}
      <Show when={showApiConfig()}>
        <div
          class="absolute inset-0 flex items-center justify-center z-50"
          style={{ 'background-color': 'rgba(0, 0, 0, 0.5)' }}
        >
          <div
            class="rounded-lg p-4 max-w-lg w-full mx-4"
            style={{ 'background-color': 'var(--bg-secondary)' }}
          >
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                API Configuration
              </h3>
              <button
                onClick={() => setShowApiConfig(false)}
                class="transition-colors"
                style={{ color: 'var(--text-tertiary)' }}
              >
                ✕
              </button>
            </div>
            <LLMSettings />
          </div>
        </div>
      </Show>

      {/* Input Area - Send button inside textarea */}
      <div class="px-3 py-2" style={{ 'border-top': '1px solid var(--border-primary)' }}>
        <div class="rounded-lg p-2" style={{ 'background-color': 'var(--bg-secondary)' }}>
          {/* Textarea with inline send button */}
          <div class="relative">
            <textarea
              value={input()}
              onInput={(e) => setInput(e.currentTarget.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
              class="w-full bg-transparent text-[11px] focus:outline-none resize-none pr-10"
              style={{
                color: 'var(--text-primary)',
              }}
              rows="3"
            />
            {/* Send button inside textarea container */}
            <button
              onClick={handleSend}
              disabled={!input().trim() || appStore.isGenerating()}
              class="absolute right-1 bottom-1 p-1.5 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              style={{
                'background-color': 'var(--accent-primary)',
                color: 'var(--text-inverse)',
              }}
              title="Send message (Enter)"
            >
              ▶
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;
