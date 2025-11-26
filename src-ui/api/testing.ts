// File: src-ui/api/testing.ts
// Purpose: TypeScript API bindings for test execution
// Last Updated: November 25, 2025

import { invoke } from '@tauri-apps/api/tauri';

export interface TestExecutionResult {
  success: boolean;
  passed: number;
  failed: number;
  skipped: number;
  errors: number;
  total: number;
  duration_seconds: number;
  pass_rate: number;
  failures: TestFailureInfo[];
  coverage_percent?: number;
}

export interface TestFailureInfo {
  test_name: string;
  error_type: string;
  error_message: string;
}

export interface TestGenerationRequest {
  code: string;
  language: string;
  file_path: string;
  coverage_target?: number;
}

export interface TestGenerationResponse {
  tests: string;
  test_count: number;
  estimated_coverage: number;
  fixtures: string[];
}

/**
 * Execute pytest tests and return results
 * @param workspacePath - Path to workspace root
 * @param testFile - Relative path to test file
 * @param timeoutSeconds - Optional timeout in seconds (default: 300)
 */
export async function executeTests(
  workspacePath: string,
  testFile: string,
  timeoutSeconds?: number
): Promise<TestExecutionResult> {
  return await invoke<TestExecutionResult>('execute_tests', {
    workspacePath,
    testFile,
    timeoutSeconds,
  });
}

/**
 * Execute pytest tests with coverage analysis
 * @param workspacePath - Path to workspace root
 * @param testFile - Relative path to test file
 * @param timeoutSeconds - Optional timeout in seconds (default: 300)
 */
export async function executeTestsWithCoverage(
  workspacePath: string,
  testFile: string,
  timeoutSeconds?: number
): Promise<TestExecutionResult> {
  return await invoke<TestExecutionResult>('execute_tests_with_coverage', {
    workspacePath,
    testFile,
    timeoutSeconds,
  });
}

/**
 * Generate pytest tests for code
 * @param code - Source code to generate tests for
 * @param language - Programming language (e.g., "python")
 * @param filePath - File path for context
 * @param coverageTarget - Target coverage percentage (0.0-1.0, default: 0.9)
 */
export async function generateTests(
  code: string,
  language: string,
  filePath: string,
  coverageTarget?: number
): Promise<TestGenerationResponse> {
  return await invoke<TestGenerationResponse>('generate_tests', {
    code,
    language,
    filePath,
    coverageTarget,
  });
}

/**
 * Check if test result is good enough for learning
 * @param result - Test execution result
 * @returns true if pass rate >= 90%
 */
export function isLearnable(result: TestExecutionResult): boolean {
  return result.pass_rate >= 0.9;
}

/**
 * Get quality score from test results (0.0 to 1.0)
 * @param result - Test execution result
 * @returns Quality score based on pass rate
 */
export function getQualityScore(result: TestExecutionResult): number {
  return result.total === 0 ? 0.0 : result.pass_rate;
}

/**
 * Format test result for display
 * @param result - Test execution result
 * @returns Formatted string
 */
export function formatTestResult(result: TestExecutionResult): string {
  const { passed, failed, skipped, errors, duration_seconds } = result;
  const parts: string[] = [];

  if (passed > 0) parts.push(`${passed} passed`);
  if (failed > 0) parts.push(`${failed} failed`);
  if (skipped > 0) parts.push(`${skipped} skipped`);
  if (errors > 0) parts.push(`${errors} errors`);

  const summary = parts.length > 0 ? parts.join(', ') : 'no tests';
  const duration = duration_seconds.toFixed(2);

  return `${summary} in ${duration}s (${(result.pass_rate * 100).toFixed(1)}% pass rate)`;
}

/**
 * Get status emoji for test result
 * @param result - Test execution result
 * @returns Emoji representing status
 */
export function getTestStatusEmoji(result: TestExecutionResult): string {
  if (result.success) return '✅';
  if (result.pass_rate >= 0.9) return '⚠️';
  if (result.pass_rate >= 0.5) return '⚠️';
  return '❌';
}
