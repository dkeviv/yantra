// File: src-ui/stores/documentationStore.ts
// Purpose: Frontend store for documentation data (Features, Decisions, Changes, Tasks)
// Dependencies: solid-js, @tauri-apps/api
// Last Updated: November 23, 2025

import { createSignal } from 'solid-js';
import { invoke } from '@tauri-apps/api/tauri';
import { appStore } from './appStore';

export interface Feature {
  id: string;
  title: string;
  description: string;
  status: 'planned' | 'in-progress' | 'completed';
  extractedFrom: string;
  timestamp: string;
}

export interface Decision {
  id: string;
  title: string;
  context: string;
  decision: string;
  rationale: string;
  timestamp: string;
}

export interface Change {
  id: string;
  changeType: 'file-added' | 'file-modified' | 'file-deleted' | 'function-added' | 'function-removed';
  description: string;
  files: string[];
  timestamp: string;
}

export interface Task {
  id: string;
  title: string;
  status: 'completed' | 'in-progress' | 'pending';
  milestone: string;
  dependencies: string[];
  requiresUserAction: boolean;
  userActionInstructions?: string;
}

// Signals for each documentation type
const [features, setFeatures] = createSignal<Feature[]>([]);
const [decisions, setDecisions] = createSignal<Decision[]>([]);
const [changes, setChanges] = createSignal<Change[]>([]);
const [tasks, setTasks] = createSignal<Task[]>([]);
const [loading, setLoading] = createSignal(false);
const [error, setError] = createSignal<string | null>(null);

/**
 * Load all documentation data from backend
 */
export async function loadDocumentation() {
  const workspacePath = appStore.projectPath();
  if (!workspacePath) {
    setError('No workspace open');
    return;
  }

  setLoading(true);
  setError(null);

  try {
    // Load all documentation types in parallel
    const [featuresData, decisionsData, changesData, tasksData] = await Promise.all([
      invoke<Feature[]>('get_features', { workspacePath }),
      invoke<Decision[]>('get_decisions', { workspacePath }),
      invoke<Change[]>('get_changes', { workspacePath }),
      invoke<Task[]>('get_tasks', { workspacePath }),
    ]);

    setFeatures(featuresData);
    setDecisions(decisionsData);
    setChanges(changesData);
    setTasks(tasksData);
  } catch (err) {
    console.error('Failed to load documentation:', err);
    setError(err as string);
  } finally {
    setLoading(false);
  }
}

/**
 * Add a new feature extracted from chat
 */
export async function addFeature(
  title: string,
  description: string,
  extractedFrom: string
): Promise<void> {
  const workspacePath = appStore.projectPath();
  if (!workspacePath) {
    throw new Error('No workspace open');
  }

  await invoke('add_feature', {
    workspacePath,
    title,
    description,
    extractedFrom,
  });

  // Reload features
  await loadDocumentation();
}

/**
 * Add a new decision
 */
export async function addDecision(
  title: string,
  context: string,
  decision: string,
  rationale: string
): Promise<void> {
  const workspacePath = appStore.projectPath();
  if (!workspacePath) {
    throw new Error('No workspace open');
  }

  await invoke('add_decision', {
    workspacePath,
    title,
    context,
    decision,
    rationale,
  });

  // Reload decisions
  await loadDocumentation();
}

/**
 * Add a change log entry
 */
export async function addChange(
  changeType: Change['changeType'],
  description: string,
  files: string[]
): Promise<void> {
  const workspacePath = appStore.projectPath();
  if (!workspacePath) {
    throw new Error('No workspace open');
  }

  await invoke('add_change', {
    workspacePath,
    changeType,
    description,
    files,
  });

  // Reload changes
  await loadDocumentation();
}

/**
 * Get tasks that require user action
 */
export function getUserActionTasks(): Task[] {
  return tasks().filter((task) => task.requiresUserAction && task.status === 'pending');
}

/**
 * Get tasks by milestone
 */
export function getTasksByMilestone(milestone: string): Task[] {
  return tasks().filter((task) => task.milestone === milestone);
}

/**
 * Get completed features count
 */
export function getCompletedFeaturesCount(): number {
  return features().filter((f) => f.status === 'completed').length;
}

/**
 * Get in-progress tasks count
 */
export function getInProgressTasksCount(): number {
  return tasks().filter((t) => t.status === 'in-progress').length;
}

export const documentationStore = {
  // Data accessors
  features,
  decisions,
  changes,
  tasks,
  loading,
  error,

  // Actions
  loadDocumentation,
  addFeature,
  addDecision,
  addChange,

  // Computed helpers
  getUserActionTasks,
  getTasksByMilestone,
  getCompletedFeaturesCount,
  getInProgressTasksCount,
};
