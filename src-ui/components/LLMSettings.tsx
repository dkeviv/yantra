// File: src-ui/components/LLMSettings.tsx
// Purpose: Minimal inline LLM configuration UI - optimized for space
// Design Philosophy: Minimal UX, maximize content space, minimize control space
// Last Updated: November 28, 2025

import { Component, createSignal, onMount, Show, For, createEffect } from 'solid-js';
import { llmApi, LLMConfig, ModelInfo } from '../api/llm';

type ProviderType = 'claude' | 'openai' | 'openrouter' | 'groq' | 'gemini';

const LLMSettings: Component = () => {
  const [config, setConfig] = createSignal<LLMConfig | null>(null);
  const [selectedProvider, setSelectedProvider] = createSignal<ProviderType>('claude');
  const [apiKey, setApiKey] = createSignal('');
  const [saving, setSaving] = createSignal(false);
  const [availableModels, setAvailableModels] = createSignal<ModelInfo[]>([]);
  const [selectedModelIds, setSelectedModelIds] = createSignal<string[]>([]);
  const [savingModels, setSavingModels] = createSignal(false);

  // Load config on mount
  onMount(async () => {
    try {
      const cfg = await llmApi.getConfig();
      setConfig(cfg);
      // Set initial provider based on config
      if (cfg.primary_provider === 'Claude') setSelectedProvider('claude');
      else if (cfg.primary_provider === 'OpenAI') setSelectedProvider('openai');
      else if (cfg.primary_provider === 'OpenRouter') setSelectedProvider('openrouter');
      else if (cfg.primary_provider === 'Groq') setSelectedProvider('groq');
      else if (cfg.primary_provider === 'Gemini') setSelectedProvider('gemini');
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
      case 'openrouter':
        return cfg.has_openrouter_key;
      case 'groq':
        return cfg.has_groq_key;
      case 'gemini':
        return cfg.has_gemini_key;
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
        case 'openrouter':
          await llmApi.setOpenRouterKey(key);
          await llmApi.setProvider('openrouter');
          break;
        case 'groq':
          await llmApi.setGroqKey(key);
          await llmApi.setProvider('groq');
          break;
        case 'gemini':
          await llmApi.setGeminiKey(key);
          await llmApi.setProvider('gemini');
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

  // Load models when provider is configured
  createEffect(async () => {
    const hasKey = providerStatus();
    const provider = selectedProvider();
    
    if (hasKey) {
      try {
        const models = await llmApi.getAvailableModels(provider);
        setAvailableModels(models);
        const selected = await llmApi.getSelectedModels();
        setSelectedModelIds(selected);
      } catch (error) {
        console.error('Failed to load models:', error);
        setAvailableModels([]);
      }
    } else {
      setAvailableModels([]);
    }
  });

  const handleModelToggle = (modelId: string, checked: boolean) => {
    if (checked) {
      setSelectedModelIds([...selectedModelIds(), modelId]);
    } else {
      setSelectedModelIds(selectedModelIds().filter(id => id !== modelId));
    }
  };

  const saveModelSelection = async () => {
    setSavingModels(true);
    try {
      await llmApi.setSelectedModels(selectedModelIds());
      // Success feedback could be added here
    } catch (error) {
      console.error('Failed to save model selection:', error);
    } finally {
      setSavingModels(false);
    }
  };

  const providerStatus = () => getProviderStatus();

  return (
    <div class="flex flex-col gap-2">
      <div class="flex items-center gap-2">
        {/* Provider Dropdown */}
        <select
          value={selectedProvider()}
          onChange={handleProviderChange}
          class="text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 w-28"
          style={{
            "background-color": "var(--bg-secondary)",
            "color": "var(--text-primary)",
            "border": "1px solid var(--border-primary)",
          }}
          title="Select LLM Provider"
        >
          <option value="claude">Claude</option>
          <option value="openai">OpenAI</option>
          <option value="openrouter">OpenRouter</option>
          <option value="groq">Groq</option>
          <option value="gemini">Gemini</option>
        </select>

        {/* API Key Input */}
        <input
          type="password"
          value={apiKey()}
          onInput={handleApiKeyChange}
          onBlur={handleBlur}
          placeholder={providerStatus() ? '••••••••' : 'Enter API Key'}
          class="flex-1 text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 min-w-0"
          style={{
            "background-color": "var(--bg-secondary)",
            "color": "var(--text-primary)",
            "border": "1px solid var(--border-primary)",
          }}
          disabled={saving()}
        />

        {/* Status Indicator */}
        <div
          class={`w-2 h-2 rounded-full flex-shrink-0 ${
            saving() 
              ? 'animate-pulse' 
              : ''
          }`}
          style={{
            "background-color": saving() 
              ? "var(--status-warning)" 
              : providerStatus() 
              ? "var(--status-success)" 
              : "var(--status-error)"
          }}
          title={
            saving() 
              ? 'Saving...' 
              : providerStatus() 
              ? `${selectedProvider().charAt(0).toUpperCase() + selectedProvider().slice(1)} configured` 
              : `No ${selectedProvider()} API key`
          }
        />
      </div>

      {/* Model Selection Section - Always visible when provider configured */}
      <Show when={providerStatus()}>
        <div class="mt-2 pt-2" style={{ "border-top": "1px solid var(--border-primary)" }}>
          <div class="flex justify-between items-center mb-2">
            <h3 class="text-xs font-medium" style={{ "color": "var(--text-secondary)" }}>
              Available Models ({availableModels().length})
            </h3>
            <div class="text-xs" style={{ "color": "var(--text-tertiary)" }}>
              {selectedModelIds().length} selected
            </div>
          </div>
          
          <div class="max-h-60 overflow-y-auto space-y-1 mb-2">
            <Show when={availableModels().length === 0}>
              <div class="text-xs py-2" style={{ "color": "var(--text-tertiary)" }}>Loading models...</div>
            </Show>
            <For each={availableModels()}>
              {(model) => (
                <label 
                  class="flex items-start gap-2 p-2 rounded cursor-pointer"
                  style={{
                    "transition": "background-color 0.15s ease"
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "var(--bg-tertiary)"}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "transparent"}
                >
                  <input
                    type="checkbox"
                    checked={selectedModelIds().includes(model.id)}
                    onChange={(e) => handleModelToggle(model.id, e.target.checked)}
                    class="mt-1 rounded focus:ring-1"
                    style={{
                      "border-color": "var(--border-secondary)",
                      "color": "var(--accent-primary)",
                    }}
                  />
                  <div class="flex-1 min-w-0">
                    <div class="text-xs font-medium truncate" style={{ "color": "var(--text-primary)" }}>{model.name}</div>
                    <div class="text-xs line-clamp-2" style={{ "color": "var(--text-secondary)" }}>{model.description}</div>
                    <div class="text-xs mt-0.5" style={{ "color": "var(--text-tertiary)" }}>
                      Context: {model.context_window.toLocaleString()} tokens
                      {model.supports_code && ' • Code optimized'}
                    </div>
                  </div>
                </label>
              )}
            </For>
          </div>
          
          <div class="flex justify-between items-center">
            <button
              onClick={saveModelSelection}
              class="text-xs px-3 py-1 rounded disabled:opacity-50"
              style={{
                "background-color": "var(--accent-primary)",
                "color": "var(--text-inverse)",
                "transition": "background-color 0.15s ease"
              }}
              onMouseEnter={(e) => !savingModels() && (e.currentTarget.style.backgroundColor = "var(--accent-hover)")}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "var(--accent-primary)"}
              disabled={savingModels()}
            >
              {savingModels() ? 'Saving...' : 'Save Selection'}
            </button>
            <div class="text-xs" style={{ "color": "var(--text-tertiary)" }}>
              {selectedModelIds().length === 0 
                ? 'No selection = all models shown' 
                : `Only selected models in chat`}
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default LLMSettings;