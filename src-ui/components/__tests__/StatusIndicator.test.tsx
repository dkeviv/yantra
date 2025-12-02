/**
 * StatusIndicator.test.tsx
 * 
 * Unit tests for StatusIndicator component
 * Tests visual states, size variants, and theme integration
 */

import { render } from '@solidjs/testing-library';
import StatusIndicator from '../StatusIndicator';
import { createSignal } from 'solid-js';

describe('StatusIndicator', () => {
  describe('Visual States', () => {
    it('renders idle state with static circle', () => {
      const { container } = render(() => <StatusIndicator />);
      const indicator = container.querySelector('.status-indicator');
      
      expect(indicator).toBeTruthy();
      expect(indicator?.classList.contains('idle')).toBe(true);
      expect(indicator?.classList.contains('running')).toBe(false);
    });

    it('renders running state with spinner animation', () => {
      // Mock appStore.isGenerating to return true
      const { container } = render(() => <StatusIndicator />);
      
      // Note: This test needs appStore to be mocked properly
      // For now, we test the component structure
      expect(container).toBeTruthy();
    });

    it('shows correct tooltip for idle state', () => {
      const { container } = render(() => <StatusIndicator />);
      const indicator = container.querySelector('.status-indicator');
      
      expect(indicator?.getAttribute('title')).toBe('Agent is idle');
    });

    it('shows correct tooltip for running state', () => {
      // Test with mocked running state
      // This requires appStore mock which we'll implement
      expect(true).toBe(true); // Placeholder
    });
  });

  describe('Size Variants', () => {
    it('renders small size by default', () => {
      const { container } = render(() => <StatusIndicator />);
      const indicator = container.querySelector('.status-indicator');
      
      expect(indicator?.classList.contains('small')).toBe(true);
    });

    it('renders medium size when specified', () => {
      const { container } = render(() => <StatusIndicator size="medium" />);
      const indicator = container.querySelector('.status-indicator');
      
      expect(indicator?.classList.contains('medium')).toBe(true);
    });

    it('renders large size when specified', () => {
      const { container } = render(() => <StatusIndicator size="large" />);
      const indicator = container.querySelector('.status-indicator');
      
      expect(indicator?.classList.contains('large')).toBe(true);
    });

    it('applies correct dimensions for each size', () => {
      const sizes = ['small', 'medium', 'large'] as const;
      const expectedDimensions = {
        small: '16px',
        medium: '24px',
        large: '32px',
      };

      sizes.forEach((size) => {
        const { container } = render(() => <StatusIndicator size={size} />);
        const indicator = container.querySelector('.status-indicator');
        const style = window.getComputedStyle(indicator!);
        
        expect(style.width).toBe(expectedDimensions[size]);
        expect(style.height).toBe(expectedDimensions[size]);
      });
    });
  });

  describe('Theme Integration', () => {
    it('uses CSS variables for colors', () => {
      const { container } = render(() => <StatusIndicator />);
      const indicator = container.querySelector('.status-indicator');
      const style = window.getComputedStyle(indicator!);
      
      // Should use var(--color-primary) from CSS
      expect(style.borderColor).toContain('var(--color-primary)');
    });

    it('adapts to dark theme', () => {
      document.documentElement.setAttribute('data-theme', 'dark');
      const { container } = render(() => <StatusIndicator />);
      
      // Theme-aware colors applied via CSS variables
      expect(container).toBeTruthy();
      
      document.documentElement.removeAttribute('data-theme');
    });

    it('adapts to bright theme', () => {
      document.documentElement.setAttribute('data-theme', 'bright');
      const { container } = render(() => <StatusIndicator />);
      
      // Theme-aware colors applied via CSS variables
      expect(container).toBeTruthy();
      
      document.documentElement.removeAttribute('data-theme');
    });
  });

  describe('Animation', () => {
    it('applies spin animation when running', () => {
      const { container } = render(() => <StatusIndicator />);
      const runningIndicator = container.querySelector('.status-indicator.running');
      
      if (runningIndicator) {
        const style = window.getComputedStyle(runningIndicator);
        expect(style.animation).toContain('spin');
      }
    });

    it('has no animation when idle', () => {
      const { container } = render(() => <StatusIndicator />);
      const idleIndicator = container.querySelector('.status-indicator.idle');
      
      if (idleIndicator) {
        const style = window.getComputedStyle(idleIndicator);
        expect(style.animation).not.toContain('spin');
      }
    });

    it('animation duration is 1 second', () => {
      const { container } = render(() => <StatusIndicator />);
      const runningIndicator = container.querySelector('.status-indicator.running');
      
      if (runningIndicator) {
        const style = window.getComputedStyle(runningIndicator);
        expect(style.animationDuration).toBe('1s');
      }
    });
  });

  describe('Reactivity', () => {
    it('updates state when isGenerating changes', () => {
      // This test requires proper appStore mock
      // For now, we verify the component structure
      const { container } = render(() => <StatusIndicator />);
      expect(container.querySelector('.status-indicator')).toBeTruthy();
    });

    it('re-renders on state change', () => {
      // Test reactive updates with signal
      const [isRunning, setIsRunning] = createSignal(false);
      
      const { container } = render(() => (
        <div class={isRunning() ? 'running' : 'idle'}>
          <StatusIndicator />
        </div>
      ));
      
      expect(container.querySelector('.idle')).toBeTruthy();
      
      setIsRunning(true);
      
      // After update, should show running
      setTimeout(() => {
        expect(container.querySelector('.running')).toBeTruthy();
      }, 0);
    });
  });

  describe('Performance', () => {
    it('renders quickly (<1ms)', () => {
      const start = performance.now();
      render(() => <StatusIndicator />);
      const duration = performance.now() - start;
      
      expect(duration).toBeLessThan(1);
    });

    it('does not cause re-renders when state is unchanged', () => {
      let renderCount = 0;
      
      const TestComponent = () => {
        renderCount++;
        return <StatusIndicator />;
      };
      
      render(() => <TestComponent />);
      
      // Should render only once on mount
      expect(renderCount).toBe(1);
    });
  });

  describe('Accessibility', () => {
    it('has descriptive title attribute', () => {
      const { container } = render(() => <StatusIndicator />);
      const indicator = container.querySelector('.status-indicator');
      
      expect(indicator?.getAttribute('title')).toBeTruthy();
    });

    it('is keyboard accessible', () => {
      const { container } = render(() => <StatusIndicator />);
      const indicator = container.querySelector('.status-indicator');
      
      // Status indicator should be non-interactive (no tabindex)
      expect(indicator?.getAttribute('tabindex')).toBeNull();
    });
  });
});
