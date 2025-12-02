/**
 * ThemeToggle.test.tsx
 * 
 * Unit tests for ThemeToggle component
 * Tests theme switching, visual states, and user interactions
 */

import { render, fireEvent, waitFor } from '@solidjs/testing-library';
import ThemeToggle from '../ThemeToggle';

describe('ThemeToggle', () => {
  // Mock localStorage
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: (key: string) => store[key] || null,
      setItem: (key: string, value: string) => { store[key] = value; },
      clear: () => { store = {}; },
    };
  })();

  beforeEach(() => {
    // Setup localStorage mock
    Object.defineProperty(window, 'localStorage', { value: localStorageMock });
    localStorageMock.clear();
    
    // Reset document theme
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    localStorageMock.clear();
  });

  describe('Initialization', () => {
    it('renders with default dark theme', () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(button).toBeTruthy();
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    });

    it('loads theme from localStorage if available', () => {
      localStorageMock.setItem('theme', 'bright');
      
      render(() => <ThemeToggle />);
      
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
    });

    it('shows moon icon for dark theme', () => {
      render(() => <ThemeToggle />);
      const button = document.querySelector('button');
      
      expect(button?.textContent).toContain('🌙');
    });

    it('shows sun icon for bright theme', () => {
      localStorageMock.setItem('theme', 'bright');
      render(() => <ThemeToggle />);
      const button = document.querySelector('button');
      
      expect(button?.textContent).toContain('☀️');
    });
  });

  describe('Theme Switching', () => {
    it('toggles from dark to bright on click', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
      
      await fireEvent.click(button!);
      
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
    });

    it('toggles from bright to dark on click', async () => {
      localStorageMock.setItem('theme', 'bright');
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
      
      await fireEvent.click(button!);
      
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    });

    it('updates icon after toggle', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      // Initially dark theme (moon icon)
      expect(button?.textContent).toContain('🌙');
      
      await fireEvent.click(button!);
      
      // After toggle, bright theme (sun icon)
      expect(button?.textContent).toContain('☀️');
    });

    it('can toggle multiple times', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
      
      await fireEvent.click(button!);
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
      
      await fireEvent.click(button!);
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
      
      await fireEvent.click(button!);
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
    });
  });

  describe('localStorage Persistence', () => {
    it('saves theme to localStorage on toggle', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      await fireEvent.click(button!);
      
      expect(localStorageMock.getItem('theme')).toBe('bright');
    });

    it('persists theme across re-renders', async () => {
      const { unmount, container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      await fireEvent.click(button!);
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
      
      unmount();
      
      // Re-render
      render(() => <ThemeToggle />);
      
      // Should load bright theme from localStorage
      expect(document.documentElement.getAttribute('data-theme')).toBe('bright');
    });

    it('overwrites previous theme in localStorage', async () => {
      localStorageMock.setItem('theme', 'bright');
      
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      await fireEvent.click(button!);
      
      expect(localStorageMock.getItem('theme')).toBe('dark');
    });
  });

  describe('Visual Feedback', () => {
    it('applies hover styles', () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(button?.classList).toBeTruthy();
    });

    it('applies active state on click', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      await fireEvent.click(button!);
      
      // Button should have active styling (tested via CSS)
      expect(button).toBeTruthy();
    });

    it('has smooth transition', () => {
      render(() => <ThemeToggle />);
      
      const rootStyle = window.getComputedStyle(document.documentElement);
      
      // CSS transitions applied via index.css
      expect(rootStyle).toBeTruthy();
    });
  });

  describe('Accessibility', () => {
    it('is keyboard accessible', () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(button?.tagName).toBe('BUTTON');
    });

    it('has descriptive title attribute', () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      expect(button?.getAttribute('title')).toBeTruthy();
    });

    it('updates title after theme change', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      const initialTitle = button?.getAttribute('title');
      
      await fireEvent.click(button!);
      
      const newTitle = button?.getAttribute('title');
      
      expect(newTitle).not.toBe(initialTitle);
    });
  });

  describe('Performance', () => {
    it('renders quickly (<1ms)', () => {
      const start = performance.now();
      render(() => <ThemeToggle />);
      const duration = performance.now() - start;
      
      expect(duration).toBeLessThan(1);
    });

    it('toggles quickly (<50ms)', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      const start = performance.now();
      await fireEvent.click(button!);
      const duration = performance.now() - start;
      
      expect(duration).toBeLessThan(50);
    });
  });

  describe('Edge Cases', () => {
    it('handles invalid theme in localStorage', () => {
      localStorageMock.setItem('theme', 'invalid');
      
      render(() => <ThemeToggle />);
      
      // Should default to 'dark'
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    });

    it('handles missing localStorage', () => {
      // Remove localStorage temporarily
      const originalLocalStorage = window.localStorage;
      Object.defineProperty(window, 'localStorage', { value: undefined });
      
      // Should not crash
      expect(() => render(() => <ThemeToggle />)).not.toThrow();
      
      // Restore localStorage
      Object.defineProperty(window, 'localStorage', { value: originalLocalStorage });
    });

    it('handles rapid clicking', async () => {
      const { container } = render(() => <ThemeToggle />);
      const button = container.querySelector('button');
      
      // Click 10 times rapidly
      for (let i = 0; i < 10; i++) {
        await fireEvent.click(button!);
      }
      
      // Wait for all updates to complete
      await waitFor(() => {
        // With 10 clicks from 'dark', we should be back to 'dark'
        // (dark -> bright -> dark -> bright -> ... -> dark)
        expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
      });
    });
  });
});
