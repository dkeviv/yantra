// File: src-ui/utils/tauri.ts
// Purpose: Tauri API wrapper functions for file system operations
// Dependencies: @tauri-apps/api
// Last Updated: November 20, 2025

import { invoke } from '@tauri-apps/api/tauri';
import { open } from '@tauri-apps/api/dialog';

export interface FileEntry {
  name: string;
  path: string;
  is_directory: boolean;
  size?: number;
}

export interface FileOperationResult {
  success: boolean;
  message: string;
}

/**
 * Read file contents as string
 */
export async function readFile(path: string): Promise<string> {
  return await invoke<string>('read_file', { path });
}

/**
 * Write content to file
 */
export async function writeFile(path: string, content: string): Promise<FileOperationResult> {
  return await invoke<FileOperationResult>('write_file', { path, content });
}

/**
 * List directory contents
 */
export async function readDir(path: string): Promise<FileEntry[]> {
  return await invoke<FileEntry[]>('read_dir', { path });
}

/**
 * Check if path exists
 */
export async function pathExists(path: string): Promise<boolean> {
  return await invoke<boolean>('path_exists', { path });
}

/**
 * Get file metadata
 */
export async function getFileInfo(path: string): Promise<FileEntry> {
  return await invoke<FileEntry>('get_file_info', { path });
}

/**
 * Open folder selection dialog
 */
export async function selectFolder(): Promise<string | null> {
  const selected = await open({
    directory: true,
    multiple: false,
  });
  
  return typeof selected === 'string' ? selected : null;
}

/**
 * Open file selection dialog
 */
export async function selectFile(filters?: { name: string; extensions: string[] }[]): Promise<string | null> {
  const selected = await open({
    directory: false,
    multiple: false,
    filters: filters || [
      { name: 'Python Files', extensions: ['py'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  
  return typeof selected === 'string' ? selected : null;
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes?: number): string {
  if (!bytes) return '-';
  
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}
