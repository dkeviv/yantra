// File: src-ui/components/LLMSettings.tsx
// Purpose: Minimal inline LLM configuration UI - optimized for space
// Design Philosophy: Minimal UX, maximize content space, minimize control space
// Last Updated: November 28, 2025

import { Component, createSignal, onMount } from 'solid-js';
import { llmApi, LLMConfig } from '../api/llm';

type ProviderType = 'claude' | 'openai';

const LLMSettings: Component = () => {
  const [config, setConfig] = createSignal<LLMConfig | null>(null);
  const [selectedProvider, setSelectedProvider] = createSignal<ProviderType>('claude');
  const [apiKey, setApiKey] = createSignal('');
  const [saving, setSaving] = createSignal(false);

  // Load config on mount
  onMount(async () => {
    try {
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      // Set initial provider based on config
      if (cfg.primary_provider === 'Claude') setSelectedProvider('claude');
      else if (cfg.primary_provider === 'OpenAI') setSelectedProvider('openai');
    } catch (error) {
      console.error('Failed to load LLM config:', error);
    }
  });

  const getProviderStatus = (): boolean => {
    const cfg = config();
    if (!cfg) return false;
    
    const provider = selectedProvider();
    switch (provider) {
      case 'claude':
        return cfg.has_claude_key;
      case 'openai':
        return cfg.has_openai_key;
      default:
        return false;
    }
  };

  const handleProviderChange = (e: Event) => {
    const target = e.currentTarget as HTMLSelectElement;
    setSelectedProvider(target.value as ProviderType);
    setApiKey(''); // Clear API key when switching providers
  };

  const handleApiKeyChange = async (e: Event) => {
    const target = e.currentTarget as HTMLInputElement;
    const key = target.value.trim();
    setApiKey(key);
  };

  const handleBlur = async () => {
    const key = apiKey().trim();
    if (!key || saving()) return;
    
    setSaving(true);
    
    try {
      const provider = selectedProvider();
      
      // Save API key based on provider
      switch (provider) {
        case 'claude':
          await llmApi.setClaudeKey(key);
          await llmApi.setProvider('claude');
          break;
        case 'openai':
          await llmApi.setOpenAIKey(key);
          await llmApi.setProvider('openai');
          break;
      }
      
      // Refresh config
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      setApiKey(''); // Clear input for security after successful save
    } catch (error: unknown) {
      console.error('Failed to save API key:', error);
    } finally {
      setSaving(false);
    }
  };

  const providerStatus = () => getProviderStatus();

  return (
    <div class="flex items-center gap-2">
      {/* Provider Dropdown */}
      <select
        value={selectedProvider()}
        onChange={handleProviderChange}
        class="bg-gray-700 text-white text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary-500 w-28"
        title="Select LLM Provider"
      >
        <option value="claude">Claude</option>
        <option value="openai">OpenAI</option>
      </select>

      {/* API Key Input */}
      <input
        type="password"
        value={apiKey()}
        onInput={handleApiKeyChange}
        onBlur={handleBlur}
        placeholder={providerStatus() ? '••••••••' : 'Enter API Key'}
        class="flex-1 bg-gray-700 text-white text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary-500 min-w-0"
        disabled={saving()}
      />

      {/* Status Indicator */}
      <div
        class={`w-2 h-2 rounded-full flex-shrink-0 ${
          saving() 
            ? 'bg-yellow-500 animate-pulse' 
            : providerStatus() 
            ? 'bg-green-500' 
            : 'bg-red-500'
        }`}
        title={
          saving() 
            ? 'Saving...' 
            : providerStatus() 
            ? `${selectedProvider().charAt(0).toUpperCase() + selectedProvider().slice(1)} configured` 
            : `No ${selectedProvider()} API key`
        }
      />
    </div>
  );
};

export default LLMSettings;
