// File: src-ui/api/llm.ts
// Purpose: TypeScript bindings for LLM configuration commands
// Last Updated: November 20, 2025

import { invoke } from '@tauri-apps/api/tauri';

export interface LLMConfig {
  has_claude_key: boolean;
  has_openai_key: boolean;
  primary_provider: 'Claude' | 'OpenAI';
  max_retries: number;
  timeout_seconds: number;
}

export const llmApi = {
  /**
   * Get current LLM configuration (without API keys)
   */
  async getConfig(): Promise<LLMConfig> {
    return await invoke<LLMConfig>('get_llm_config');
  },

  /**
   * Set primary LLM provider
   * @param provider - 'claude' or 'openai'
   */
  async setProvider(provider: 'claude' | 'openai'): Promise<void> {
    await invoke('set_llm_provider', { provider });
  },

  /**
   * Set Claude API key
   * @param apiKey - Your Claude API key
   */
  async setClaudeKey(apiKey: string): Promise<void> {
    await invoke('set_claude_key', { apiKey });
  },

  /**
   * Set OpenAI API key
   * @param apiKey - Your OpenAI API key
   */
  async setOpenAIKey(apiKey: string): Promise<void> {
    await invoke('set_openai_key', { apiKey });
  },

  /**
   * Clear API key for a provider
   * @param provider - 'claude' or 'openai'
   */
  async clearKey(provider: 'claude' | 'openai'): Promise<void> {
    await invoke('clear_llm_key', { provider });
  },

  /**
   * Update retry configuration
   * @param maxRetries - Maximum number of retries per request
   * @param timeoutSeconds - Request timeout in seconds
   */
  async setRetryConfig(maxRetries: number, timeoutSeconds: number): Promise<void> {
    await invoke('set_llm_retry_config', { maxRetries, timeoutSeconds });
  },
};
