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
  const windowWidth = window.innerWidth;
  const tenPercent = windowWidth * 0.1;

  // If width becomes less than 10% of window, auto-close the panel
  if (width < tenPercent) {
    // Signal to close the panel - will be handled by App.tsx
    window.dispatchEvent(new CustomEvent('close-file-explorer'));
    return;
  }

  // Clamp width between 10% and 60% of window width, with max 600px
  const minWidth = Math.max(tenPercent, 150); // At least 150px even if 10% is smaller
  const maxWidth = Math.min(windowWidth * 0.6, 600);
  const clampedWidth = Math.max(minWidth, Math.min(maxWidth, width));
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
      const windowWidth = window.innerWidth;
      const minWidth = Math.max(windowWidth * 0.1, 150);
      const maxWidth = Math.min(windowWidth * 0.6, 600);
      if (!isNaN(width)) {
        setFileExplorerWidth(Math.max(minWidth, Math.min(maxWidth, width)));
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
