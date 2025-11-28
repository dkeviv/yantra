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

export interface ChatMessage {
  role: string;  // "user" or "assistant"
  content: string;
}

export type Intent = 
  | 'code_generation'
  | 'code_modification'
  | 'terminal_command'
  | 'ui_control'
  | 'question'
  | 'general';

export interface DetectedAction {
  action_type: string;
  parameters: Record<string, string>;
}

export interface ChatResponse {
  response: string;
  intent: Intent;
  action?: DetectedAction;
}

export interface FileToGenerate {
  path: string;
  purpose: string;
  dependencies: string[];
  is_test: boolean;
  priority: number;
}

export interface TestSummary {
  total: number;
  passed: number;
  failed: number;
  coverage_percent?: number;
}

export interface ProjectResult {
  success: boolean;
  project_dir: string;
  generated_files: string[];
  test_results?: TestSummary;
  errors: string[];
  attempts: number;
  session_id: string;
}

export const llmApi = {
  /**
   * Get current LLM configuration (without API keys)
   */
  async getConfig(): Promise<LLMConfig> {
    return await invoke<LLMConfig>('get_llm_config');
  },

  /**
   * Natural language chat with agent
   * @param message - User's message
   * @param conversationHistory - Previous messages for context
   */
  async chat(message: string, conversationHistory: ChatMessage[] = []): Promise<ChatResponse> {
    return await invoke<ChatResponse>('chat_with_agent', { 
      message, 
      conversationHistory 
    });
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

  /**
   * Create entire project from high-level intent (E2E agentic workflow)
   * @param intent - High-level project description (e.g., "Create a REST API with authentication")
   * @param projectDir - Directory where project should be created
   * @param template - Optional project template (e.g., "express-api", "react-app", "fastapi")
   */
  async createProjectAutonomous(
    intent: string,
    projectDir: string,
    template?: string
  ): Promise<ProjectResult> {
    return await invoke<ProjectResult>('create_project_autonomous', {
      intent,
      projectDir,
      template,
    });
  },
};
