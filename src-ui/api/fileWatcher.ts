// File: src-ui/api/fileWatcher.ts
// Purpose: TypeScript API for file system watcher functionality
// Last Updated: December 9, 2025

import { invoke } from '@tauri-apps/api/tauri';

/**
 * Start file system watcher for automatic graph synchronization
 * Monitors workspace for file changes and automatically updates GNN
 * @param workspacePath Path to the workspace to watch
 * @returns Success message
 */
export async function startFileWatcher(workspacePath: string): Promise<string> {
  return invoke('start_file_watcher', { workspacePath });
}

/**
 * Stop file system watcher
 * @returns Success message
 */
export async function stopFileWatcher(): Promise<string> {
  return invoke('stop_file_watcher');
}
