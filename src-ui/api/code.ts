// File: src-ui/api/code.ts
// Purpose: TypeScript API for code generation and test generation
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
  tests: string | null;
  provider: "Claude" | "OpenAI";
  tokens_used: number;
}

export interface TestGenerationRequest {
  code: string;
  language: string;
  file_path: string;
  coverage_target?: number; // 0.0 to 1.0, default 0.9
}

export interface TestGenerationResponse {
  tests: string;
  test_count: number;
  estimated_coverage: number;
  fixtures: string[];
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

/**
 * Generate pytest tests for given code
 * @param request Test generation request with code and coverage target
 * @returns Generated tests with metadata
 */
export async function generateTests(
  request: TestGenerationRequest
): Promise<TestGenerationResponse> {
  return invoke("generate_tests", {
    code: request.code,
    language: request.language,
    filePath: request.file_path,
    coverageTarget: request.coverage_target,
  });
}
