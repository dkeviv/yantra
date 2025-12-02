/**
 * Architecture API - TypeScript wrapper for Tauri architecture commands
 * 
 * Provides type-safe access to all architecture Tauri commands
 * Types match Rust backend: src-tauri/src/architecture/types.rs
 * 
 * Updated: November 29, 2025 - Fixed types to match Rust backend exactly
 */

import { invoke } from '@tauri-apps/api/tauri';

// ============================================================================
// TYPES (Match Rust types.rs exactly)
// ============================================================================

/**
 * Component types with progress tracking
 */
export type ComponentType =
  | { type: 'Planned' }
  | { type: 'InProgress'; completed: number; total: number }
  | { type: 'Implemented'; total: number }
  | { type: 'Misaligned'; reason: string };

export type ComponentCategory = 'frontend' | 'backend' | 'database' | 'external' | 'utility';

export type ConnectionType = 'DataFlow' | 'ApiCall' | 'Event' | 'Dependency' | 'Bidirectional';

export interface Position {
  x: number;
  y: number;
}

export interface Component {
  id: string;
  name: string;
  description: string;
  component_type: ComponentType;
  category: ComponentCategory;
  position: Position;
  files: string[];
  metadata: Record<string, string>;
  created_at: number;
  updated_at: number;
}

export interface Connection {
  id: string;
  source_id: string;
  target_id: string;
  connection_type: ConnectionType;
  description: string;
  metadata: Record<string, string>;
  created_at: number;
  updated_at: number;
}

export interface Architecture {
  id: string;
  name: string;
  description: string;
  components: Component[];
  connections: Connection[];
  metadata: Record<string, string>;
  created_at: number;
  updated_at: number;
}

export interface ArchitectureVersion {
  id: string;
  architecture_id: string;
  version: number;
  commit_message: string;
  snapshot: Architecture;
  created_at: number;
}

export type ExportFormat = 'markdown' | 'mermaid' | 'json';

// ============================================================================
// COMMAND RESPONSE WRAPPER
// ============================================================================

interface CommandResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Create a new architecture
 */
export async function createArchitecture(
  name: string,
  description?: string
): Promise<Architecture> {
  const response = await invoke<CommandResponse<Architecture>>('create_architecture', {
    name,
    description,
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to create architecture');
  }

  return response.data;
}

/**
 * Get architecture by ID
 */
export async function getArchitecture(architectureId: string): Promise<Architecture> {
  const response = await invoke<CommandResponse<Architecture>>('get_architecture', {
    architectureId,
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to get architecture');
  }

  return response.data;
}

/**
 * Create a new component in an architecture
 */
export async function createComponent(
  architectureId: string,
  name: string,
  description: string,
  category: ComponentCategory,
  position: Position
): Promise<Component> {
  const response = await invoke<CommandResponse<Component>>('create_component', {
    request: {
      architecture_id: architectureId,
      name,
      description,
      category,
      position,
    },
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to create component');
  }

  return response.data;
}

/**
 * Update an existing component
 */
export async function updateComponent(component: Component): Promise<void> {
  const response = await invoke<CommandResponse<void>>('update_component', {
    request: {
      component,
    },
  });

  if (!response.success) {
    throw new Error(response.error || 'Failed to update component');
  }
}

/**
 * Delete a component from an architecture
 */
export async function deleteComponent(componentId: string): Promise<void> {
  const response = await invoke<CommandResponse<null>>('delete_component', {
    component_id: componentId,
  });

  if (!response.success) {
    throw new Error(response.error || 'Failed to delete component');
  }
}

/**
 * Create a connection between two components
 */
export async function createConnection(
  architectureId: string,
  sourceId: string,
  targetId: string,
  connectionType: ConnectionType,
  description: string
): Promise<Connection> {
  const response = await invoke<CommandResponse<Connection>>('create_connection', {
    request: {
      architecture_id: architectureId,
      source_id: sourceId,
      target_id: targetId,
      connection_type: connectionType,
      description,
    },
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to create connection');
  }

  return response.data;
}

/**
 * Delete a connection from an architecture
 */
export async function deleteConnection(connectionId: string): Promise<void> {
  const response = await invoke<CommandResponse<null>>('delete_connection', {
    connection_id: connectionId,
  });

  if (!response.success) {
    throw new Error(response.error || 'Failed to delete connection');
  }
}

/**
 * Save a version snapshot of the current architecture
 */
export async function saveArchitectureVersion(
  architectureId: string,
  commitMessage: string
): Promise<ArchitectureVersion> {
  const response = await invoke<CommandResponse<ArchitectureVersion>>(
    'save_architecture_version',
    {
      request: {
        architecture_id: architectureId,
        commit_message: commitMessage,
      },
    }
  );

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to save architecture version');
  }

  return response.data;
}

/**
 * List all versions of an architecture
 */
export async function listArchitectureVersions(
  architectureId: string
): Promise<ArchitectureVersion[]> {
  const response = await invoke<CommandResponse<ArchitectureVersion[]>>(
    'list_architecture_versions',
    {
      architecture_id: architectureId,
    }
  );

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to list architecture versions');
  }

  return response.data;
}

/**
 * Restore architecture to a previous version
 */
export async function restoreArchitectureVersion(
  versionId: string
): Promise<Architecture> {
  const response = await invoke<CommandResponse<Architecture>>(
    'restore_architecture_version',
    {
      version_id: versionId,
    }
  );

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to restore architecture version');
  }

  return response.data;
}

/**
 * Export architecture to various formats
 */
export async function exportArchitecture(
  architectureId: string,
  format: ExportFormat
): Promise<string> {
  const response = await invoke<CommandResponse<string>>('export_architecture', {
    request: {
      architecture_id: architectureId,
      format,
    },
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to export architecture');
  }

  return response.data;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get status indicator emoji for a component
 */
export function getStatusIndicator(componentType: ComponentType): string {
  if (componentType.type === 'Planned') {
    return '📋';
  } else if (componentType.type === 'InProgress') {
    return '🔄';
  } else if (componentType.type === 'Implemented') {
    return '✅';
  } else if (componentType.type === 'Misaligned') {
    return '⚠️';
  }
  return '📋'; // default
}

/**
 * Get status text for UI display
 */
export function getStatusText(componentType: ComponentType): string {
  if (componentType.type === 'Planned') {
    return 'Planned';
  } else if (componentType.type === 'InProgress') {
    return `In Progress (${componentType.completed}/${componentType.total})`;
  } else if (componentType.type === 'Implemented') {
    return `Implemented (${componentType.total})`;
  } else if (componentType.type === 'Misaligned') {
    return 'Misaligned';
  }
  return 'Unknown';
}

/**
 * Get status color for UI styling
 */
export function getStatusColor(componentType: ComponentType): string {
  if (componentType.type === 'Planned') {
    return '#9ca3af'; // gray
  } else if (componentType.type === 'InProgress') {
    return '#fbbf24'; // yellow
  } else if (componentType.type === 'Implemented') {
    return '#10b981'; // green
  } else if (componentType.type === 'Misaligned') {
    return '#ef4444'; // red
  }
  return '#9ca3af'; // default gray
}

/**
 * Get arrow type for connection (React Flow marker)
 */
export function getConnectionArrow(type: ConnectionType): string {
  switch (type) {
    case 'DataFlow':
      return '→'; // solid arrow
    case 'ApiCall':
      return '⇢'; // dashed arrow
    case 'Event':
      return '⤳'; // curved arrow
    case 'Dependency':
      return '⋯>'; // dotted arrow
    case 'Bidirectional':
      return '⇄'; // double arrow
  }
}

/**
 * Get connection color for UI styling
 */
export function getConnectionColor(type: ConnectionType): string {
  switch (type) {
    case 'DataFlow':
      return '#3b82f6'; // blue
    case 'ApiCall':
      return '#8b5cf6'; // purple
    case 'Event':
      return '#ec4899'; // pink
    case 'Dependency':
      return '#6b7280'; // gray
    case 'Bidirectional':
      return '#14b8a6'; // teal
  }
}

/**
 * Get connection stroke style for React Flow
 */
export function getConnectionStrokeStyle(type: ConnectionType): {
  strokeDasharray?: string;
  strokeWidth: number;
} {
  switch (type) {
    case 'DataFlow':
      return { strokeWidth: 2 };
    case 'ApiCall':
      return { strokeDasharray: '5,5', strokeWidth: 2 };
    case 'Event':
      return { strokeWidth: 2 };
    case 'Dependency':
      return { strokeDasharray: '2,2', strokeWidth: 1 };
    case 'Bidirectional':
      return { strokeWidth: 3 };
  }
}
