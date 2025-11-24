// File: src-ui/components/ChatPanel.tsx
// Purpose: Chat interface component for user interaction with agent-first commands
// Dependencies: solid-js, appStore, agentStore
// Last Updated: November 23, 2025

import { Component, For, createSignal, Show } from 'solid-js';
import { appStore } from '../stores/appStore';
import { agentStore, getCommandSuggestions } from '../stores/agentStore';

const ChatPanel: Component = () => {
  const [input, setInput] = createSignal('');
  const [showSuggestions, setShowSuggestions] = createSignal(false);
  const [selectedModel, setSelectedModel] = createSignal('claude-sonnet-4');
  const [showApiConfig, setShowApiConfig] = createSignal(false);

  const handleSend = async () => {
    const message = input().trim();
    if (!message) return;

    // Add user message
    appStore.addMessage('user', message);
    setInput('');
    setShowSuggestions(false);

    // Check if it's an agent command (starts with action words)
    const commandKeywords = ['open', 'close', 'show', 'hide', 'run', 'execute', 'create', 'list', 'reset', 'maximize'];
    const isCommand = commandKeywords.some(keyword => message.toLowerCase().startsWith(keyword));

    if (isCommand) {
      // Execute as agent command
      appStore.setIsGenerating(true);
      
      try {
        const result = await agentStore.executeCommand(message);
        appStore.addMessage('assistant', result);
      } catch (error) {
        appStore.addMessage('assistant', `❌ Error: ${error}`);
      } finally {
        appStore.setIsGenerating(false);
      }
    } else {
      // Regular chat message - send to LLM (TODO: implement LLM integration)
      appStore.setIsGenerating(true);

      // Mock response for now
      setTimeout(() => {
        appStore.addMessage(
          'assistant',
          'I can help you with code generation. Try commands like:\n' +
          '• "run npm test" - Execute commands in terminal\n' +
          '• "show dependencies" - View dependency graph\n' +
          '• "open new terminal" - Create a new terminal\n' +
          '• "list files" - See open files'
        );
        appStore.setIsGenerating(false);
      }, 500);
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

      {/* Messages */}
      <div class="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        <For each={appStore.messages()}>
          {(message) => (
            <div
              class={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                class={`max-w-[80%] rounded-lg px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-primary-600 text-white'
                    : message.role === 'system'
                    ? 'bg-gray-700 text-gray-200'
                    : 'bg-gray-800 text-gray-100'
                }`}
              >
                <div class="text-[10px] font-medium mb-1">
                  {message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'Yantra'}
                </div>
                <div class="whitespace-pre-wrap text-[10px]">{message.content}</div>
                <div class="text-[10px] opacity-60 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          )}
        </For>

        {appStore.isGenerating() && (
          <div class="flex justify-start">
            <div class="bg-gray-800 text-gray-100 rounded-lg px-4 py-3">
              <div class="flex items-center space-x-2">
                <div class="animate-pulse">●</div>
                <span>Generating...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div class="px-6 py-4 border-t border-gray-700">
        {/* Command Suggestions */}
        <Show when={showSuggestions() && input().trim()}>
          <div class="mb-2 flex flex-wrap gap-2">
            <For each={getCommandSuggestions()}>
              {(suggestion) => (
                <button
                  onClick={() => {
                    setInput(suggestion);
                    setShowSuggestions(false);
                  }}
                  class="px-3 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
                >
                  {suggestion}
                </button>
              )}
            </For>
          </div>
        </Show>

        {/* Input with icon send button */}
        <div class="flex space-x-2 mb-3">
          <textarea
            value={input()}
            onInput={(e) => {
              setInput(e.currentTarget.value);
              setShowSuggestions(true);
            }}
            onKeyPress={handleKeyPress}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            placeholder="Type a command or message... (e.g., 'open new terminal', 'show dependencies')"
            class="flex-1 bg-gray-800 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            rows="3"
            disabled={appStore.isGenerating()}
          />
        </div>

        {/* LLM Configuration Panel */}
        <div class="flex items-center justify-between bg-gray-800 rounded-lg px-3 py-2">
          <div class="flex items-center space-x-3 flex-1">
            {/* Model Selection */}
            <div class="flex items-center space-x-2">
              <label class="text-xs text-gray-400">Model:</label>
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
            </div>

            {/* API Config Button */}
            <button
              onClick={() => setShowApiConfig(!showApiConfig())}
              class="text-xs text-gray-400 hover:text-white transition-colors"
              title="Configure API settings"
            >
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
          </div>

          {/* Send Icon Button */}
          <button
            onClick={handleSend}
            disabled={!input().trim() || appStore.isGenerating()}
            class="p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Send message (Enter)"
          >
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>

        {/* API Configuration Modal */}
        <Show when={showApiConfig()}>
          <div class="mt-2 bg-gray-800 rounded-lg p-3 space-y-2">
            <div>
              <label class="text-xs text-gray-400 block mb-1">API Key</label>
              <input
                type="password"
                placeholder="Enter your API key..."
                class="w-full bg-gray-700 text-white text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary-500"
              />
            </div>
            <div class="flex justify-end space-x-2">
              <button
                onClick={() => setShowApiConfig(false)}
                class="text-xs px-3 py-1 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  // TODO: Save API key
                  setShowApiConfig(false);
                }}
                class="text-xs px-3 py-1 bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default ChatPanel;
