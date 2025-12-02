/**
 * layoutStore.test.ts
 * 
 * Unit tests for layoutStore
 * Tests panel expansion state management and localStorage persistence
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { layoutStore } from '../../stores/layoutStore';

describe('layoutStore', () => {
  beforeEach(() => {
    // Clear localStorage
    localStorage.clear();
    
    // Reset store state
    layoutStore.collapseAll();
  });

  describe('Panel Expansion', () => {
    it('expands a panel when togglePanelExpansion is called', () => {
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      
      layoutStore.togglePanelExpansion('fileExplorer');
      
      expect(layoutStore.isExpanded('fileExplorer')).toBe(true);
    });

    it('collapses an expanded panel when togglePanelExpansion is called again', () => {
      layoutStore.togglePanelExpansion('fileExplorer');
      expect(layoutStore.isExpanded('fileExplorer')).toBe(true);
      
      layoutStore.togglePanelExpansion('fileExplorer');
      
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
    });

    it('only allows one panel expanded at a time', () => {
      layoutStore.togglePanelExpansion('fileExplorer');
      expect(layoutStore.isExpanded('fileExplorer')).toBe(true);
      
      layoutStore.togglePanelExpansion('agent');
      
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      expect(layoutStore.isExpanded('agent')).toBe(true);
    });

    it('can expand editor panel', () => {
      layoutStore.togglePanelExpansion('editor');
      
      expect(layoutStore.isExpanded('editor')).toBe(true);
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      expect(layoutStore.isExpanded('agent')).toBe(false);
    });

    it('can cycle through all three panels', () => {
      layoutStore.togglePanelExpansion('fileExplorer');
      expect(layoutStore.isExpanded('fileExplorer')).toBe(true);
      
      layoutStore.togglePanelExpansion('agent');
      expect(layoutStore.isExpanded('agent')).toBe(true);
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      
      layoutStore.togglePanelExpansion('editor');
      expect(layoutStore.isExpanded('editor')).toBe(true);
      expect(layoutStore.isExpanded('agent')).toBe(false);
    });
  });

  describe('isExpanded Method', () => {
    it('returns false for all panels initially', () => {
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      expect(layoutStore.isExpanded('agent')).toBe(false);
      expect(layoutStore.isExpanded('editor')).toBe(false);
    });

    it('returns true for expanded panel', () => {
      layoutStore.togglePanelExpansion('agent');
      
      expect(layoutStore.isExpanded('agent')).toBe(true);
    });

    it('returns false for non-expanded panels', () => {
      layoutStore.togglePanelExpansion('agent');
      
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      expect(layoutStore.isExpanded('editor')).toBe(false);
    });
  });

  describe('collapseAll Method', () => {
    it('collapses all panels', () => {
      layoutStore.togglePanelExpansion('fileExplorer');
      expect(layoutStore.isExpanded('fileExplorer')).toBe(true);
      
      layoutStore.collapseAll();
      
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
      expect(layoutStore.isExpanded('agent')).toBe(false);
      expect(layoutStore.isExpanded('editor')).toBe(false);
    });

    it('does nothing if no panels are expanded', () => {
      layoutStore.collapseAll();
      
      // Should not throw error
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
    });
  });

  describe('File Explorer Width', () => {
    it('has default width of 250px', () => {
      expect(layoutStore.fileExplorerWidth()).toBe(250);
    });

    it('updates width when updateFileExplorerWidth is called', () => {
      layoutStore.updateFileExplorerWidth(350);
      
      expect(layoutStore.fileExplorerWidth()).toBe(350);
    });

    it('clamps width to minimum 200px', () => {
      layoutStore.updateFileExplorerWidth(150);
      
      expect(layoutStore.fileExplorerWidth()).toBe(200);
    });

    it('clamps width to maximum 500px', () => {
      layoutStore.updateFileExplorerWidth(600);
      
      expect(layoutStore.fileExplorerWidth()).toBe(500);
    });

    it('accepts width exactly at minimum', () => {
      layoutStore.updateFileExplorerWidth(200);
      
      expect(layoutStore.fileExplorerWidth()).toBe(200);
    });

    it('accepts width exactly at maximum', () => {
      layoutStore.updateFileExplorerWidth(500);
      
      expect(layoutStore.fileExplorerWidth()).toBe(500);
    });

    it('accepts width in valid range', () => {
      const validWidths = [250, 300, 350, 400, 450];
      
      validWidths.forEach((width) => {
        layoutStore.updateFileExplorerWidth(width);
        expect(layoutStore.fileExplorerWidth()).toBe(width);
      });
    });
  });

  describe('localStorage Persistence', () => {
    describe('File Explorer Width', () => {
      it('saves width to localStorage', () => {
        layoutStore.updateFileExplorerWidth(350);
        
        expect(localStorage.getItem('yantra-fileexplorer-width')).toBe('350');
      });

      it('persists width across updates', () => {
        layoutStore.updateFileExplorerWidth(300);
        expect(localStorage.getItem('yantra-fileexplorer-width')).toBe('300');
        
        layoutStore.updateFileExplorerWidth(400);
        expect(localStorage.getItem('yantra-fileexplorer-width')).toBe('400');
      });

      it('reads initial width from store', () => {
        // The store loads width from localStorage on init
        // Just verify we can read the current width
        const width = layoutStore.fileExplorerWidth();
        expect(width).toBeGreaterThanOrEqual(200);
        expect(width).toBeLessThanOrEqual(500);
      });

      it('updates width and persists to localStorage', () => {
        const newWidth = 450;
        layoutStore.updateFileExplorerWidth(newWidth);
        
        expect(layoutStore.fileExplorerWidth()).toBe(newWidth);
        expect(localStorage.getItem('yantra-fileexplorer-width')).toBe(String(newWidth));
      });
    });
  });

  describe('Performance', () => {
    it('updates state quickly (<5ms)', () => {
      const start = performance.now();
      layoutStore.togglePanelExpansion('fileExplorer');
      const duration = performance.now() - start;
      
      expect(duration).toBeLessThan(5);
    });

    it('handles rapid updates', () => {
      for (let i = 0; i < 100; i++) {
        layoutStore.togglePanelExpansion('fileExplorer');
      }
      
      // Should end with panel collapsed (even number of toggles)
      expect(layoutStore.isExpanded('fileExplorer')).toBe(false);
    });

    it('clamps width efficiently', () => {
      const start = performance.now();
      
      for (let i = 100; i < 600; i += 10) {
        layoutStore.updateFileExplorerWidth(i);
      }
      
      const duration = performance.now() - start;
      
      expect(duration).toBeLessThan(10);
    });
  });

  describe('Edge Cases', () => {
    it('handles localStorage unavailable', () => {
      const originalLocalStorage = window.localStorage;
      Object.defineProperty(window, 'localStorage', { value: undefined });
      
      // Should not crash
      expect(() => layoutStore.togglePanelExpansion('fileExplorer')).not.toThrow();
      expect(() => layoutStore.updateFileExplorerWidth(300)).not.toThrow();
      
      Object.defineProperty(window, 'localStorage', { value: originalLocalStorage });
    });

    it('handles negative width', () => {
      layoutStore.updateFileExplorerWidth(-100);
      
      expect(layoutStore.fileExplorerWidth()).toBe(200); // Clamped to min
    });

    it('handles zero width', () => {
      layoutStore.updateFileExplorerWidth(0);
      
      expect(layoutStore.fileExplorerWidth()).toBe(200); // Clamped to min
    });

    it('handles very large width', () => {
      layoutStore.updateFileExplorerWidth(10000);
      
      expect(layoutStore.fileExplorerWidth()).toBe(500); // Clamped to max
    });

    it('handles floating point width', () => {
      layoutStore.updateFileExplorerWidth(325.7);
      
      // Should round or accept as-is
      expect(layoutStore.fileExplorerWidth()).toBeGreaterThanOrEqual(325);
      expect(layoutStore.fileExplorerWidth()).toBeLessThanOrEqual(326);
    });
  });
});
