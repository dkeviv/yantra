/**
 * Architecture Store - SolidJS reactive state management
 * 
 * Manages architecture state, component CRUD, connection management,
 * undo/redo stack, and real-time UI updates.
 */

import { createStore } from 'solid-js/store';
import { createSignal } from 'solid-js';
import * as archAPI from '../api/architecture';
import type {
  Architecture,
  Component,
  Connection,
  ComponentType,
  ComponentCategory,
  ConnectionType,
  Position,
  ArchitectureVersion,
} from '../api/architecture';

// ============================================================================
// STORE STATE
// ============================================================================

interface ArchitectureState {
  current: Architecture | null;
  selectedComponentId: string | null;
  selectedConnectionId: string | null;
  isLoading: boolean;
  error: string | null;
  versions: ArchitectureVersion[];
  undoStack: Architecture[];
  redoStack: Architecture[];
}

const [architectureState, setArchitectureState] = createStore<ArchitectureState>({
  current: null,
  selectedComponentId: null,
  selectedConnectionId: null,
  isLoading: false,
  error: null,
  versions: [],
  undoStack: [],
  redoStack: [],
});

// Export mode for filtering components by category
const [filterMode, setFilterMode] = createSignal<ComponentCategory | 'Complete'>('Complete');

// ============================================================================
// ARCHITECTURE OPERATIONS
// ============================================================================

/**
 * Create a new architecture
 */
export async function createArchitecture(name: string, description?: string) {
  setArchitectureState('isLoading', true);
  setArchitectureState('error', null);

  try {
    const architecture = await archAPI.createArchitecture(name, description);
    setArchitectureState('current', architecture);
    setArchitectureState('undoStack', []);
    setArchitectureState('redoStack', []);
    return architecture;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to create architecture';
    setArchitectureState('error', errorMessage);
    throw error;
  } finally {
    setArchitectureState('isLoading', false);
  }
}

/**
 * Load an existing architecture by ID
 */
export async function loadArchitecture(architectureId: string) {
  setArchitectureState('isLoading', true);
  setArchitectureState('error', null);

  try {
    const architecture = await archAPI.getArchitecture(architectureId);
    setArchitectureState('current', architecture);
    setArchitectureState('undoStack', []);
    setArchitectureState('redoStack', []);

    // Load versions
    const versions = await archAPI.listArchitectureVersions(architectureId);
    setArchitectureState('versions', versions);

    return architecture;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to load architecture';
    setArchitectureState('error', errorMessage);
    throw error;
  } finally {
    setArchitectureState('isLoading', false);
  }
}

// ============================================================================
// COMPONENT OPERATIONS
// ============================================================================

/**
 * Add a new component to the current architecture
 */
export async function addComponent(
  name: string,
  description: string,
  category: ComponentCategory,
  position: Position
) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  // Save current state for undo
  pushUndo();

  try {
    const component = await archAPI.createComponent(
      architectureState.current.id,
      name,
      description,
      category,
      position
    );

    // Update local state
    setArchitectureState('current', 'components', (components) => [...components, component]);
    setArchitectureState('selectedComponentId', component.id);

    return component;
  } catch (error) {
    // Revert undo stack on error
    popUndo();
    throw error;
  }
}

/**
 * Update an existing component
 */
export async function updateComponent(componentId: string, updates: Partial<Component>) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  // Save current state for undo
  pushUndo();

  try {
    // Find the current component
    const currentComponent = architectureState.current.components.find(c => c.id === componentId);
    if (!currentComponent) {
      throw new Error('Component not found');
    }

    // Merge updates with current component
    const updatedComponent = { ...currentComponent, ...updates };
    
    await archAPI.updateComponent(updatedComponent);

    // Update local state
    setArchitectureState(
      'current',
      'components',
      (component) => component.id === componentId,
      updatedComponent
    );

    return updatedComponent;
  } catch (error) {
    // Revert undo stack on error
    popUndo();
    throw error;
  }
}

/**
 * Delete a component
 */
export async function deleteComponent(componentId: string) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  // Save current state for undo
  pushUndo();

  try {
    await archAPI.deleteComponent(componentId);

    // Update local state - remove component and associated connections
    setArchitectureState('current', 'components', (components) =>
      components.filter((c) => c.id !== componentId)
    );
    setArchitectureState('current', 'connections', (connections) =>
      connections.filter((conn) => conn.source_id !== componentId && conn.target_id !== componentId)
    );

    // Clear selection if deleted component was selected
    if (architectureState.selectedComponentId === componentId) {
      setArchitectureState('selectedComponentId', null);
    }
  } catch (error) {
    // Revert undo stack on error
    popUndo();
    throw error;
  }
}

/**
 * Select a component
 */
export function selectComponent(componentId: string | null) {
  setArchitectureState('selectedComponentId', componentId);
  setArchitectureState('selectedConnectionId', null);
}

// ============================================================================
// CONNECTION OPERATIONS
// ============================================================================

/**
 * Add a new connection between components
 */
export async function addConnection(
  sourceId: string,
  targetId: string,
  connectionType: ConnectionType,
  description: string = ''
) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  // Save current state for undo
  pushUndo();

  try {
    const connection = await archAPI.createConnection(
      architectureState.current.id,
      sourceId,
      targetId,
      connectionType,
      description
    );

    // Update local state
    setArchitectureState('current', 'connections', (connections) => [...connections, connection]);
    setArchitectureState('selectedConnectionId', connection.id);

    return connection;
  } catch (error) {
    // Revert undo stack on error
    popUndo();
    throw error;
  }
}

/**
 * Delete a connection
 */
export async function deleteConnection(connectionId: string) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  // Save current state for undo
  pushUndo();

  try {
    await archAPI.deleteConnection(connectionId);

    // Update local state
    setArchitectureState('current', 'connections', (connections) =>
      connections.filter((c) => c.id !== connectionId)
    );

    // Clear selection if deleted connection was selected
    if (architectureState.selectedConnectionId === connectionId) {
      setArchitectureState('selectedConnectionId', null);
    }
  } catch (error) {
    // Revert undo stack on error
    popUndo();
    throw error;
  }
}

/**
 * Select a connection
 */
export function selectConnection(connectionId: string | null) {
  setArchitectureState('selectedConnectionId', connectionId);
  setArchitectureState('selectedComponentId', null);
}

// ============================================================================
// VERSION MANAGEMENT
// ============================================================================

/**
 * Save current architecture as a new version
 */
export async function saveVersion(description: string) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  try {
    const version = await archAPI.saveArchitectureVersion(
      architectureState.current.id,
      description
    );

    setArchitectureState('versions', (versions) => [...versions, version]);

    return version;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to save version';
    setArchitectureState('error', errorMessage);
    throw error;
  }
}

/**
 * Restore architecture to a previous version
 */
export async function restoreVersion(versionId: string) {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  // Save current state for undo
  pushUndo();

  try {
    const architecture = await archAPI.restoreArchitectureVersion(versionId);

    setArchitectureState('current', architecture);
  } catch (error) {
    // Revert undo stack on error
    popUndo();
    throw error;
  }
}

// ============================================================================
// UNDO/REDO
// ============================================================================

/**
 * Push current state to undo stack
 */
function pushUndo() {
  if (!architectureState.current) return;

  setArchitectureState('undoStack', (stack) => [...stack, architectureState.current!]);
  setArchitectureState('redoStack', []); // Clear redo stack on new action
}

/**
 * Pop state from undo stack (used when action fails)
 */
function popUndo() {
  setArchitectureState('undoStack', (stack) => stack.slice(0, -1));
}

/**
 * Undo last action
 */
export function undo() {
  if (architectureState.undoStack.length === 0) return;

  const previousState = architectureState.undoStack[architectureState.undoStack.length - 1];

  // Push current state to redo stack
  if (architectureState.current) {
    setArchitectureState('redoStack', (stack) => [...stack, architectureState.current!]);
  }

  // Restore previous state
  setArchitectureState('current', previousState);
  setArchitectureState('undoStack', (stack) => stack.slice(0, -1));
}

/**
 * Redo last undone action
 */
export function redo() {
  if (architectureState.redoStack.length === 0) return;

  const nextState = architectureState.redoStack[architectureState.redoStack.length - 1];

  // Push current state to undo stack
  if (architectureState.current) {
    setArchitectureState('undoStack', (stack) => [...stack, architectureState.current!]);
  }

  // Restore next state
  setArchitectureState('current', nextState);
  setArchitectureState('redoStack', (stack) => stack.slice(0, -1));
}

/**
 * Check if undo is available
 */
export function canUndo(): boolean {
  return architectureState.undoStack.length > 0;
}

/**
 * Check if redo is available
 */
export function canRedo(): boolean {
  return architectureState.redoStack.length > 0;
}

// ============================================================================
// EXPORT
// ============================================================================

/**
 * Export architecture to specified format
 */
export async function exportArchitecture(format: 'markdown' | 'mermaid' | 'json') {
  if (!architectureState.current) {
    throw new Error('No architecture loaded');
  }

  try {
    const exportedData = await archAPI.exportArchitecture(architectureState.current.id, format);
    return exportedData;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to export architecture';
    setArchitectureState('error', errorMessage);
    throw error;
  }
}

// ============================================================================
// FILTERING
// ============================================================================

/**
 * Get filtered components based on current filter mode
 */
export function getFilteredComponents(): Component[] {
  if (!architectureState.current) return [];

  const mode = filterMode();

  if (mode === 'Complete') {
    return architectureState.current.components;
  }

  return architectureState.current.components.filter((comp) => comp.category === mode);
}

/**
 * Set filter mode
 */
export function setFilter(mode: ComponentCategory | 'Complete') {
  setFilterMode(mode);
}

/**
 * Get current filter mode
 */
export function getFilter(): ComponentCategory | 'Complete' {
  return filterMode();
}

// ============================================================================
// EXPORTS
// ============================================================================

export { architectureState, filterMode };

// Export types for convenience
export type {
  Architecture,
  Component,
  Connection,
  ComponentType,
  ComponentCategory,
  ConnectionType,
  Position,
  ArchitectureVersion,
};
