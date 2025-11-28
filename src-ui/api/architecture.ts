/**
 * Architecture API - TypeScript wrapper for Tauri architecture commands
 * 
 * Provides type-safe access to all 11 architecture Tauri commands:
 * - create_architecture, get_architecture
 * - create_component, update_component, delete_component
 * - create_connection, delete_connection
 * - save_architecture_version, list_architecture_versions, restore_architecture_version
 * - export_architecture
 * 
 * Types match Rust backend: src-tauri/src/architecture/types.rs
 */

import { invoke } from '@tauri-apps/api/tauri';

// ============================================================================
// TYPES (Match Rust types.rs exactly)
// ============================================================================

export type ComponentType = 'Planned' | 'InProgress' | 'Implemented' | 'Misaligned';

export type ComponentCategory = 'Backend' | 'Frontend' | 'Database' | 'External' | 'Utility';

export type ConnectionType = 'DataFlow' | 'ApiCall' | 'Event' | 'Dependency' | 'Bidirectional';

export interface Position {
  x: number;
  y: number;
}

export interface Component {
  id: string;
  name: string;
  component_type: ComponentCategory;
  status: ComponentType;
  position: Position;
  description?: string;
  files: string[];
  created_at: string;
  updated_at: string;
}

export interface Connection {
  id: string;
  source_id: string;
  target_id: string;
  connection_type: ConnectionType;
  label?: string;
  created_at: string;
}

export interface Architecture {
  id: string;
  name: string;
  description?: string;
  components: Component[];
  connections: Connection[];
  created_at: string;
  updated_at: string;
}

export interface ArchitectureVersion {
  id: string;
  architecture_id: string;
  version_number: number;
  description?: string;
  data_snapshot: string; // JSON string of Architecture
  created_at: string;
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
  componentType: ComponentCategory,
  position: Position,
  description?: string,
  files: string[] = []
): Promise<Component> {
  const response = await invoke<CommandResponse<Component>>('create_component', {
    architectureId,
    name,
    componentType,
    position,
    description,
    files,
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to create component');
  }

  return response.data;
}

/**
 * Update an existing component
 */
export async function updateComponent(
  architectureId: string,
  componentId: string,
  name?: string,
  componentType?: ComponentCategory,
  status?: ComponentType,
  position?: Position,
  description?: string,
  files?: string[]
): Promise<Component> {
  const response = await invoke<CommandResponse<Component>>('update_component', {
    architectureId,
    componentId,
    name,
    componentType,
    status,
    position,
    description,
    files,
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to update component');
  }

  return response.data;
}

/**
 * Delete a component from an architecture
 */
export async function deleteComponent(
  architectureId: string,
  componentId: string
): Promise<void> {
  const response = await invoke<CommandResponse<null>>('delete_component', {
    architectureId,
    componentId,
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
  label?: string
): Promise<Connection> {
  const response = await invoke<CommandResponse<Connection>>('create_connection', {
    architectureId,
    sourceId,
    targetId,
    connectionType,
    label,
  });

  if (!response.success || !response.data) {
    throw new Error(response.error || 'Failed to create connection');
  }

  return response.data;
}

/**
 * Delete a connection from an architecture
 */
export async function deleteConnection(
  architectureId: string,
  connectionId: string
): Promise<void> {
  const response = await invoke<CommandResponse<null>>('delete_connection', {
    architectureId,
    connectionId,
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
  description?: string
): Promise<ArchitectureVersion> {
  const response = await invoke<CommandResponse<ArchitectureVersion>>(
    'save_architecture_version',
    {
      architectureId,
      description,
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
      architectureId,
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
  architectureId: string,
  versionId: string
): Promise<Architecture> {
  const response = await invoke<CommandResponse<Architecture>>(
    'restore_architecture_version',
    {
      architectureId,
      versionId,
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
    architectureId,
    format,
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
export function getStatusIndicator(status: ComponentType): string {
  switch (status) {
    case 'Planned':
      return '📋';
    case 'InProgress':
      return '🔄';
    case 'Implemented':
      return '✅';
    case 'Misaligned':
      return '⚠️';
  }
}

/**
 * Get status color for UI styling
 */
export function getStatusColor(status: ComponentType): string {
  switch (status) {
    case 'Planned':
      return '#9ca3af'; // gray
    case 'InProgress':
      return '#fbbf24'; // yellow
    case 'Implemented':
      return '#10b981'; // green
    case 'Misaligned':
      return '#ef4444'; // red
  }
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
