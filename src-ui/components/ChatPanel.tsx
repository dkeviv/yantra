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

      {/* Input */}
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

        <div class="flex space-x-2">
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
          <button
            onClick={handleSend}
            disabled={!input().trim() || appStore.isGenerating()}
            class="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;
