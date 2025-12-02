// File: src-ui/stores/layoutStore.ts
// Purpose: Shared state for panel expansion logic (File Explorer, Agent, Editor)
// Design Philosophy: Only one panel can be expanded at a time
// Last Updated: November 29, 2025

import { createSignal } from 'solid-js';

export type PanelType = 'fileExplorer' | 'agent' | 'editor' | null;

// State
const [expandedPanel, setExpandedPanel] = createSignal<PanelType>(null);
const [fileExplorerWidth, setFileExplorerWidth] = createSignal(250); // Default 250px

// Actions
const togglePanelExpansion = (panel: PanelType) => {
  if (expandedPanel() === panel) {
    // Collapse if already expanded
    setExpandedPanel(null);
  } else {
    // Expand this panel, collapse others
    setExpandedPanel(panel);
  }
};

const collapseAll = () => {
  setExpandedPanel(null);
};

const isExpanded = (panel: PanelType): boolean => {
  return expandedPanel() === panel;
};

// File Explorer width management
const updateFileExplorerWidth = (width: number) => {
  // Clamp width between 200px and 500px
  const clampedWidth = Math.max(200, Math.min(500, width));
  setFileExplorerWidth(clampedWidth);
  
  // Persist to localStorage
  try {
    localStorage.setItem('yantra-fileexplorer-width', clampedWidth.toString());
  } catch (error) {
    console.error('Failed to save file explorer width:', error);
  }
};

const loadFileExplorerWidth = () => {
  try {
    const saved = localStorage.getItem('yantra-fileexplorer-width');
    if (saved) {
      const width = parseInt(saved, 10);
      if (!isNaN(width)) {
        setFileExplorerWidth(Math.max(200, Math.min(500, width)));
      }
    }
  } catch (error) {
    console.error('Failed to load file explorer width:', error);
  }
};

// Load width on import
loadFileExplorerWidth();

export const layoutStore = {
  // State getters
  expandedPanel,
  fileExplorerWidth,
  
  // Actions
  togglePanelExpansion,
  collapseAll,
  isExpanded,
  updateFileExplorerWidth,
};
