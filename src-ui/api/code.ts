// File: src-ui/api/code.ts
// Purpose: TypeScript API for code generation
// Last Updated: November 21, 2025

import { invoke } from "@tauri-apps/api/tauri";

export interface CodeGenerationRequest {
  intent: string;
  file_path?: string;
  target_node?: string;
}

export interface CodeGenerationResponse {
  code: string;
  language: string;
  explanation: string;
  tests: string;
  provider: "Claude" | "OpenAI";
  tokens_used: number;
}

/**
 * Generate code using LLM with GNN context
 * @param request Code generation request with intent and optional context
 * @returns Generated code, tests, and metadata
 */
export async function generateCode(
  request: CodeGenerationRequest
): Promise<CodeGenerationResponse> {
  return invoke("generate_code", {
    intent: request.intent,
    filePath: request.file_path,
    targetNode: request.target_node,
  });
}
