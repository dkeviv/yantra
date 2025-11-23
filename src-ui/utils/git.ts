// File: src-ui/utils/git.ts
// Purpose: Git operations via Tauri commands
// Dependencies: @tauri-apps/api
// Last Updated: November 23, 2025

import { invoke } from '@tauri-apps/api/tauri';

export interface GitStatus {
  status: string;
}

export interface GitCommitResult {
  message: string;
}

/**
 * Get git status for a workspace
 */
export async function gitStatus(workspacePath: string): Promise<string> {
  return await invoke<string>('git_status', { workspacePath });
}

/**
 * Add files to git staging area
 */
export async function gitAdd(workspacePath: string, files: string[]): Promise<void> {
  return await invoke<void>('git_add', { workspacePath, files });
}

/**
 * Commit staged changes
 */
export async function gitCommit(workspacePath: string, message: string): Promise<string> {
  return await invoke<string>('git_commit', { workspacePath, message });
}

/**
 * Get git diff
 */
export async function gitDiff(workspacePath: string, file?: string): Promise<string> {
  return await invoke<string>('git_diff', { workspacePath, file });
}

/**
 * Get git log
 */
export async function gitLog(workspacePath: string, maxCount: number = 10): Promise<string> {
  return await invoke<string>('git_log', { workspacePath, maxCount });
}

/**
 * List all branches
 */
export async function gitBranchList(workspacePath: string): Promise<string> {
  return await invoke<string>('git_branch_list', { workspacePath });
}

/**
 * Get current branch name
 */
export async function gitCurrentBranch(workspacePath: string): Promise<string> {
  return await invoke<string>('git_current_branch', { workspacePath });
}

/**
 * Checkout a branch
 */
export async function gitCheckout(workspacePath: string, branch: string): Promise<string> {
  return await invoke<string>('git_checkout', { workspacePath, branch });
}

/**
 * Pull from remote
 */
export async function gitPull(workspacePath: string): Promise<string> {
  return await invoke<string>('git_pull', { workspacePath });
}

/**
 * Push to remote
 */
export async function gitPush(workspacePath: string): Promise<string> {
  return await invoke<string>('git_push', { workspacePath });
}
