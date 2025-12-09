// File: src-ui/api/conversationMemory.ts
// Purpose: TypeScript API for conversation memory system
// Last Updated: December 9, 2025

import { invoke } from '@tauri-apps/api/tauri';

export type MessageRole = 'User' | 'Assistant' | 'System';
export type SessionType = 'CodeGeneration' | 'Testing' | 'Deployment' | 'Documentation';
export type ExportFormat = 'Markdown' | 'Json' | 'PlainText';

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  total_tokens: number;
  tags: string[];
  archived: boolean;
}

export interface Message {
  id: string;
  conversation_id: string;
  parent_message_id: string | null;
  role: MessageRole;
  content: string;
  timestamp: string;
  tokens: number;
  metadata: string | null;
}

export interface SessionLink {
  id: string;
  conversation_id: string;
  message_id: string;
  session_type: SessionType;
  session_id: string;
  created_at: string;
  metadata: string | null;
}

export interface SearchFilter {
  keyword?: string;
  start_date?: string;
  end_date?: string;
  tags?: string[];
  session_type?: SessionType;
}

/**
 * Create a new conversation
 * @param initialTitle Optional title for the conversation
 * @returns Created conversation
 */
export async function createConversation(initialTitle?: string): Promise<Conversation> {
  return invoke('create_conversation', { initialTitle });
}

/**
 * Save a message to a conversation
 * @param conversationId ID of the conversation
 * @param role Message role (User, Assistant, System)
 * @param content Message content
 * @param parentMessageId Optional parent message ID for threading
 * @param tokens Optional token count
 * @param metadata Optional metadata JSON string
 * @returns Saved message
 */
export async function saveMessage(
  conversationId: string,
  role: MessageRole,
  content: string,
  parentMessageId?: string,
  tokens?: number,
  metadata?: string
): Promise<Message> {
  return invoke('save_message', {
    conversationId,
    role,
    content,
    parentMessageId,
    tokens,
    metadata,
  });
}

/**
 * Load a conversation with its messages
 * @param conversationId ID of the conversation
 * @param limit Optional limit on number of messages (default: all)
 * @param offset Optional offset for pagination (default: 0)
 * @returns Tuple of conversation and messages
 */
export async function loadConversation(
  conversationId: string,
  limit?: number,
  offset?: number
): Promise<[Conversation, Message[]]> {
  return invoke('load_conversation', { conversationId, limit, offset });
}

/**
 * Get the last active conversation
 * @returns Most recent non-archived conversation, or null if none exists
 */
export async function getLastActiveConversation(): Promise<Conversation | null> {
  return invoke('get_last_active_conversation');
}

/**
 * Load recent messages from a conversation
 * @param conversationId ID of the conversation
 * @param count Number of recent messages to load
 * @returns Recent messages in chronological order
 */
export async function loadRecentMessages(
  conversationId: string,
  count: number
): Promise<Message[]> {
  return invoke('load_recent_messages', { conversationId, count });
}

/**
 * Search conversations by keyword, date range, tags, or session type
 * @param filter Search filter criteria
 * @returns Matching conversations
 */
export async function searchConversations(filter: SearchFilter): Promise<Conversation[]> {
  return invoke('search_conversations', {
    keyword: filter.keyword,
    startDate: filter.start_date,
    endDate: filter.end_date,
    tags: filter.tags,
    sessionType: filter.session_type,
  });
}

/**
 * Link a conversation to a session (code generation, testing, etc.)
 * @param conversationId ID of the conversation
 * @param messageId ID of the message
 * @param sessionType Type of session (CodeGeneration, Testing, etc.)
 * @param sessionId ID of the session
 * @param metadata Optional metadata JSON string
 * @returns Created session link
 */
export async function linkToSession(
  conversationId: string,
  messageId: string,
  sessionType: SessionType,
  sessionId: string,
  metadata?: string
): Promise<SessionLink> {
  return invoke('link_to_session', {
    conversationId,
    messageId,
    sessionType,
    sessionId,
    metadata,
  });
}

/**
 * Get session links for a conversation
 * @param conversationId ID of the conversation
 * @returns All session links for the conversation
 */
export async function getSessionLinks(conversationId: string): Promise<SessionLink[]> {
  return invoke('get_session_links', { conversationId });
}

/**
 * Export conversation to file format
 * @param conversationId ID of the conversation
 * @param format Export format (Markdown, Json, PlainText)
 * @param outputPath Path to save the exported file
 * @returns Success message with file path
 */
export async function exportConversation(
  conversationId: string,
  format: ExportFormat,
  outputPath: string
): Promise<string> {
  return invoke('export_conversation', { conversationId, format, outputPath });
}

/**
 * Build semantic search index for all conversations
 * Must be called before semantic search can be used
 * @returns Number of messages indexed
 */
export async function buildSemanticSearchIndex(): Promise<number> {
  return invoke('build_semantic_search_index');
}

/**
 * Search conversations using semantic similarity
 * Uses vector embeddings to find semantically similar messages
 * @param query Search query
 * @param topK Number of results to return
 * @returns Array of [messageId, similarityScore] tuples
 */
export async function semanticSearchConversations(
  query: string,
  topK: number
): Promise<Array<[string, number]>> {
  return invoke('semantic_search_conversations', { query, topK });
}

/**
 * Hybrid search combining keyword and semantic matching
 * Balances exact keyword matches with semantic similarity
 * @param query Search query
 * @param keywordWeight Weight for keyword matching (0-1)
 * @param semanticWeight Weight for semantic matching (0-1)
 * @param topK Number of results to return
 * @returns Array of [conversationId, relevanceScore] tuples
 */
export async function hybridSearchConversations(
  query: string,
  keywordWeight: number,
  semanticWeight: number,
  topK: number
): Promise<Array<[string, number]>> {
  return invoke('hybrid_search_conversations', {
    query,
    keywordWeight,
    semanticWeight,
    topK,
  });
}
