// File: src-ui/api/codeValidation.ts
// Purpose: TypeScript API for code validation functionality
// Last Updated: December 9, 2025

import { invoke } from '@tauri-apps/api/tauri';

export type ErrorType = 'Syntax' | 'Type' | 'Import' | 'Lint' | 'Compilation' | 'Other';
export type Severity = 'Error' | 'Warning' | 'Info';

export interface ValidationError {
  file: string;
  line: number;
  column: number;
  error_type: ErrorType;
  message: string;
  severity: Severity;
}

export interface CodeValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  language: string;
}

/**
 * Validate code file for syntax, type, and import errors
 * Fast failure detection before expensive test execution
 * @param filePath Path to the file to validate
 * @param workspacePath Workspace path for context
 * @returns Validation result with errors and warnings
 */
export async function validateCodeFile(
  filePath: string,
  workspacePath: string
): Promise<CodeValidationResult> {
  return invoke('validate_code_file', { filePath, workspacePath });
}
