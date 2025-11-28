// File: src-ui/components/ChatPanel.tsx
// Purpose: Chat interface component for user interaction with agent-first commands
// Dependencies: solid-js, appStore, agentStore
// Last Updated: November 28, 2025

import { Component, For, createSignal, Show } from 'solid-js';
import { appStore } from '../stores/appStore';
import { llmApi, type ChatMessage } from '../api/llm';
import { terminalStore } from '../stores/terminalStore';
import LLMSettings from './LLMSettings';

const ChatPanel: Component = () => {
  const [input, setInput] = createSignal('');
  const [selectedModel, setSelectedModel] = createSignal('claude-sonnet-4');
  const [showApiConfig, setShowApiConfig] = createSignal(false);

  const handleSend = async () => {
    const message = input().trim();
    if (!message) return;

    // Add user message
    appStore.addMessage('user', message);
    setInput('');
    appStore.setIsGenerating(true);

    try {
      // Prepare conversation history for LLM
      const conversationHistory: ChatMessage[] = appStore.messages()
        .filter(msg => msg.role !== 'system')
        .map(msg => ({
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
            (lowerIntent.includes('create') && (
              lowerIntent.includes('api') ||
              lowerIntent.includes('application') ||
              lowerIntent.includes('website') ||
              lowerIntent.includes('service')
            ));

          if (isProjectCreation) {
            // E2E project creation
            appStore.addMessage('assistant', '� Starting autonomous project creation...');
            
            // Determine template from intent
            let template: string | undefined;
            if (lowerIntent.includes('express') || (lowerIntent.includes('rest') && lowerIntent.includes('api'))) {
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
                
                response += `\nFiles created:\n${result.generated_files.map(f => `  - ${f}`).join('\n')}`;
                
                appStore.addMessage('assistant', response);
              } else {
                let response = `❌ Project creation encountered issues:\n\n`;
                response += result.errors.map(e => `  - ${e}`).join('\n');
                response += `\n\n${result.generated_files.length} files were created before errors occurred.`;
                appStore.addMessage('assistant', response);
              }
            } catch (error) {
              appStore.addMessage('assistant', `❌ Project creation failed: ${error}`);
            }
          } else {
            // Single file generation (existing behavior)
            appStore.addMessage('assistant', '🔧 Single-file code generation will be available soon!');
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
    <div class="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div class="px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold text-white inline-block">Chat</h2>
        <span class="text-sm text-gray-400 ml-3">- Describe what you want to build</span>
      </div>

      {/* Messages - Terminal-like, immersive */}
      <div class="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        <For each={appStore.messages()}>
          {(message) => (
            <div class="font-mono text-xs leading-relaxed">
              <span class={message.role === 'user' ? 'text-green-400' : 'text-blue-400'}>
                {message.role === 'user' ? 'You' : 'Yantra'}
              </span>
              <span class="text-gray-500 mx-1">›</span>
              <span class="text-gray-200">{message.content}</span>
            </div>
          )}
        </For>

        {appStore.isGenerating() && (
          <div class="font-mono text-xs leading-relaxed">
            <span class="text-blue-400">Yantra</span>
            <span class="text-gray-500 mx-1">›</span>
            <span class="text-gray-400 animate-pulse">Generating...</span>
          </div>
        )}
      </div>

      {/* Input Area - Single Immersive Panel */}
      <div class="px-6 py-4 border-t border-gray-700">
        {/* Unified Input Panel */}
        <div class="bg-gray-800 rounded-lg p-3">
          {/* Top row: Model selector and API config */}
          <div class="flex items-center justify-between mb-2 pb-2 border-b border-gray-700">
            <div class="flex items-center gap-2">
              {/* Model Selection */}
              <select
                value={selectedModel()}
                onChange={(e) => setSelectedModel(e.currentTarget.value)}
                class="bg-gray-700 text-white text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary-500"
                title="Select LLM model"
              >
                <option value="claude-sonnet-4">Claude Sonnet 4</option>
                <option value="gpt-4-turbo">GPT-4 Turbo</option>
                <option value="claude-opus">Claude Opus</option>
                <option value="gpt-4">GPT-4</option>
              </select>

              {/* API Config Button */}
              <button
                onClick={() => setShowApiConfig(!showApiConfig())}
                class="p-1 text-gray-400 hover:text-white transition-colors"
                title="Configure API settings"
              >
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>

            {/* Send button on the right - Minimal */}
            <button
              onClick={handleSend}
              disabled={!input().trim() || appStore.isGenerating()}
              class="p-1.5 text-primary-400 hover:text-primary-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="Send (Enter)"
            >
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>

          {/* Textarea */}
          <textarea
            value={input()}
            onInput={(e) => setInput(e.currentTarget.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything... (e.g., 'create a sample program', 'run npm test', 'show dependencies')"
            class="w-full bg-transparent text-white focus:outline-none resize-none"
            rows="3"
            disabled={appStore.isGenerating()}
          />

          {/* API Configuration - Full LLM Settings */}
          <Show when={showApiConfig()}>
            <div class="mt-3 pt-3 border-t border-gray-700">
              <LLMSettings />
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;
