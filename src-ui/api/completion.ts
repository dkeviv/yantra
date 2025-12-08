// File: src-ui/api/completion.ts
// Purpose: API wrapper for code completion using Tree-sitter and GNN
// Dependencies: @tauri-apps/api
// Last Updated: December 7, 2025

import { invoke } from '@tauri-apps/api/tauri';

export interface CompletionRequest {
  filePath: string;
  content: string;
  line: number;
  column: number;
  projectPath: string;
  language: string;
}

export interface CompletionItem {
  label: string;
  kind: CompletionKind;
  detail?: string;
  documentation?: string;
  insertText: string;
  insertTextAsSnippet: boolean;
  sortText?: string;
  filterText?: string;
}

export enum CompletionKind {
  Method = 0,
  Function = 1,
  Constructor = 2,
  Field = 3,
  Variable = 4,
  Class = 5,
  Struct = 6,
  Interface = 7,
  Module = 8,
  Property = 9,
  Event = 10,
  Operator = 11,
  Unit = 12,
  Value = 13,
  Constant = 14,
  Enum = 15,
  EnumMember = 16,
  Keyword = 17,
  Text = 18,
  Color = 19,
  File = 20,
  Reference = 21,
  Customcolor = 22,
  Folder = 23,
  TypeParameter = 24,
  User = 25,
  Issue = 26,
  Snippet = 27,
}

/**
 * Get code completions using Tree-sitter AST and GNN context
 */
export async function getCodeCompletions(request: CompletionRequest): Promise<CompletionItem[]> {
  try {
    const completions = await invoke<CompletionItem[]>('get_code_completions', {
      request,
    });
    return completions;
  } catch (error) {
    console.error('Failed to get code completions:', error);
    return [];
  }
}
