// File: src-ui/components/LLMSettings.tsx
// Purpose: LLM configuration UI component
// Last Updated: November 20, 2025

import { Component, createSignal, onMount, Show } from 'solid-js';
import { llmApi, LLMConfig } from '../api/llm';

const LLMSettings: Component = () => {
  const [config, setConfig] = createSignal<LLMConfig | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [message, setMessage] = createSignal('');
  const [claudeKey, setClaudeKey] = createSignal('');
  const [openaiKey, setOpenaiKey] = createSignal('');

  // Load config on mount
  onMount(async () => {
    try {
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
    } catch (error) {
      console.error('Failed to load LLM config:', error);
      setMessage('Failed to load configuration');
    }
  });

  const handleSetProvider = async (provider: 'claude' | 'openai') => {
    setLoading(true);
    setMessage('');
    try {
      await llmApi.setProvider(provider);
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      setMessage(`Primary provider set to ${provider}`);
    } catch (error: any) {
      setMessage(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveClaudeKey = async () => {
    if (!claudeKey()) {
      setMessage('Please enter a Claude API key');
      return;
    }
    
    setLoading(true);
    setMessage('');
    try {
      await llmApi.setClaudeKey(claudeKey());
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      setClaudeKey(''); // Clear input for security
      setMessage('Claude API key saved');
    } catch (error: any) {
      setMessage(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveOpenAIKey = async () => {
    if (!openaiKey()) {
      setMessage('Please enter an OpenAI API key');
      return;
    }
    
    setLoading(true);
    setMessage('');
    try {
      await llmApi.setOpenAIKey(openaiKey());
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      setOpenaiKey(''); // Clear input for security
      setMessage('OpenAI API key saved');
    } catch (error: any) {
      setMessage(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClearKey = async (provider: 'claude' | 'openai') => {
    setLoading(true);
    setMessage('');
    try {
      await llmApi.clearKey(provider);
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      setMessage(`${provider} API key cleared`);
    } catch (error: any) {
      setMessage(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="p-4 bg-gray-800 rounded-lg">
      <h2 class="text-xl font-bold text-white mb-4">LLM Configuration</h2>

      <Show when={config()} fallback={<div class="text-gray-400">Loading...</div>}>
        {(cfg) => (
          <>
            {/* Provider Selection */}
            <div class="mb-6">
              <h3 class="text-lg font-semibold text-white mb-2">Primary Provider</h3>
              <div class="flex gap-2">
                <button
                  onClick={() => handleSetProvider('claude')}
                  disabled={loading()}
                  class={`px-4 py-2 rounded ${
                    cfg().primary_provider === 'Claude'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  } disabled:opacity-50`}
                >
                  Claude Sonnet 4
                </button>
                <button
                  onClick={() => handleSetProvider('openai')}
                  disabled={loading()}
                  class={`px-4 py-2 rounded ${
                    cfg().primary_provider === 'OpenAI'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  } disabled:opacity-50`}
                >
                  GPT-4 Turbo
                </button>
              </div>
            </div>

            {/* Claude API Key */}
            <div class="mb-6">
              <h3 class="text-lg font-semibold text-white mb-2">
                Claude API Key
                <span class="ml-2 text-sm">
                  {cfg().has_claude_key ? (
                    <span class="text-green-400">✓ Configured</span>
                  ) : (
                    <span class="text-red-400">Not configured</span>
                  )}
                </span>
              </h3>
              <div class="flex gap-2">
                <input
                  type="password"
                  value={claudeKey()}
                  onInput={(e) => setClaudeKey(e.currentTarget.value)}
                  placeholder="sk-ant-..."
                  class="flex-1 px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 outline-none"
                />
                <button
                  onClick={handleSaveClaudeKey}
                  disabled={loading()}
                  class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  Save
                </button>
                <Show when={cfg().has_claude_key}>
                  <button
                    onClick={() => handleClearKey('claude')}
                    disabled={loading()}
                    class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
                  >
                    Clear
                  </button>
                </Show>
              </div>
            </div>

            {/* OpenAI API Key */}
            <div class="mb-6">
              <h3 class="text-lg font-semibold text-white mb-2">
                OpenAI API Key
                <span class="ml-2 text-sm">
                  {cfg().has_openai_key ? (
                    <span class="text-green-400">✓ Configured</span>
                  ) : (
                    <span class="text-red-400">Not configured</span>
                  )}
                </span>
              </h3>
              <div class="flex gap-2">
                <input
                  type="password"
                  value={openaiKey()}
                  onInput={(e) => setOpenaiKey(e.currentTarget.value)}
                  placeholder="sk-..."
                  class="flex-1 px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 outline-none"
                />
                <button
                  onClick={handleSaveOpenAIKey}
                  disabled={loading()}
                  class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  Save
                </button>
                <Show when={cfg().has_openai_key}>
                  <button
                    onClick={() => handleClearKey('openai')}
                    disabled={loading()}
                    class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
                  >
                    Clear
                  </button>
                </Show>
              </div>
            </div>

            {/* Settings */}
            <div class="mb-4">
              <h3 class="text-lg font-semibold text-white mb-2">Advanced Settings</h3>
              <div class="grid grid-cols-2 gap-4 text-sm text-gray-300">
                <div>
                  <span class="font-medium">Max Retries:</span> {cfg().max_retries}
                </div>
                <div>
                  <span class="font-medium">Timeout:</span> {cfg().timeout_seconds}s
                </div>
              </div>
            </div>

            {/* Message */}
            <Show when={message()}>
              <div class="mt-4 p-3 bg-gray-700 text-white rounded">
                {message()}
              </div>
            </Show>
          </>
        )}
      </Show>
    </div>
  );
};

export default LLMSettings;
