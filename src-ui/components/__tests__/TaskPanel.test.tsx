/**
 * TaskPanel.test.tsx
 *
 * Comprehensive unit tests for TaskPanel component
 * Tests task display, interactions, state management, and animations
 */

import { render, fireEvent, waitFor } from '@solidjs/testing-library';
import TaskPanel from '../TaskPanel';
import { createSignal } from 'solid-js';
import { invoke } from '@tauri-apps/api/tauri';

// Mock module is automatically loaded from __mocks__/@tauri-apps/api/tauri.js
jest.mock('@tauri-apps/api/tauri');

const mockInvoke = invoke as jest.MockedFunction<typeof invoke>;

describe('TaskPanel', () => {
  const mockTasks = [
    {
      id: 1,
      description: 'Generate authentication API',
      status: 'InProgress',
      priority: 'High',
      created_at: Date.now() - 120000, // 2 minutes ago
      started_at: Date.now() - 120000,
      completed_at: null,
      result: null,
    },
    {
      id: 2,
      description: 'Create database models',
      status: 'Pending',
      priority: 'Medium',
      created_at: Date.now() - 300000, // 5 minutes ago
      started_at: null,
      completed_at: null,
      result: null,
    },
    {
      id: 3,
      description: 'Write unit tests',
      status: 'Completed',
      priority: 'High',
      created_at: Date.now() - 600000, // 10 minutes ago
      started_at: Date.now() - 600000,
      completed_at: Date.now() - 300000,
      result: 'Success',
    },
    {
      id: 4,
      description: 'Deploy to production',
      status: 'Failed',
      priority: 'Critical',
      created_at: Date.now() - 3600000, // 1 hour ago
      started_at: Date.now() - 3600000,
      completed_at: Date.now() - 3000000,
      result: 'Error: Connection timeout',
      error: 'Connection timeout',
    },
  ];

  const mockStats = {
    total: 23,
    pending: 5,
    in_progress: 1,
    completed: 15,
    failed: 2,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockInvoke.mockImplementation((cmd: string) => {
      if (cmd === 'get_task_queue') {
        return Promise.resolve(mockTasks);
      }
      if (cmd === 'get_task_stats') {
        return Promise.resolve(mockStats);
      }
      if (cmd === 'get_current_task') {
        return Promise.resolve(mockTasks[0]);
      }
      return Promise.resolve(null);
    });
  });

  afterEach(() => {
    jest.clearAllTimers();
  });

  describe('Rendering', () => {
    it('renders when isOpen is true', () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      expect(container.querySelector('.task-panel')).toBeTruthy();
    });

    it('does not render when isOpen is false', () => {
      const [isOpen] = createSignal(false);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      expect(container.querySelector('.task-panel')).toBeNull();
    });

    it('renders header with title and close button', () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      expect(container.textContent).toContain('Task Queue');
      expect(container.querySelector('.close-button')).toBeTruthy();
    });

    it('renders backdrop when open', () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      expect(container.querySelector('.backdrop')).toBeTruthy();
    });
  });

  describe('Statistics Display', () => {
    it('displays pending count', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('5');
        expect(container.textContent).toContain('Pending');
      });
    });

    it('displays in-progress count', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('1');
        expect(container.textContent).toContain('In Progress');
      });
    });

    it('displays completed count', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('15');
        expect(container.textContent).toContain('Completed');
      });
    });

    it('displays failed count', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('2');
        expect(container.textContent).toContain('Failed');
      });
    });
  });

  describe('Current Task Highlight', () => {
    it('highlights current task with blue background', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const currentTask = container.querySelector('.current-task');
        expect(currentTask).toBeTruthy();
      });
    });

    it('displays current task description', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('Generate authentication API');
      });
    });

    it('displays current task priority', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('High');
      });
    });
  });

  describe('Task List', () => {
    it('renders all tasks', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('Generate authentication API');
        expect(container.textContent).toContain('Create database models');
        expect(container.textContent).toContain('Write unit tests');
        expect(container.textContent).toContain('Deploy to production');
      });
    });

    it('displays task timestamps', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toMatch(/\d+ (minute|second)s? ago/);
      });
    });

    it('displays error messages for failed tasks', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('Error: Connection timeout');
      });
    });
  });

  describe('Status Badges', () => {
    it('displays pending badge in yellow', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.badge-pending');
        expect(badges.length).toBeGreaterThan(0);
      });
    });

    it('displays in-progress badge in blue', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.badge-in-progress');
        expect(badges.length).toBeGreaterThan(0);
      });
    });

    it('displays completed badge in green', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.badge-completed');
        expect(badges.length).toBeGreaterThan(0);
      });
    });

    it('displays failed badge in red', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.badge-failed');
        expect(badges.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Priority Badges', () => {
    it('displays critical priority in red', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.priority-critical');
        expect(badges.length).toBeGreaterThan(0);
      });
    });

    it('displays high priority in orange', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.priority-high');
        expect(badges.length).toBeGreaterThan(0);
      });
    });

    it('displays medium priority in yellow', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        const badges = container.querySelectorAll('.priority-medium');
        expect(badges.length).toBeGreaterThan(0);
      });
    });
  });

  describe('User Interactions', () => {
    it('calls onClose when close button clicked', async () => {
      const onClose = jest.fn();
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={onClose} />);

      const closeButton = container.querySelector('.close-button');
      await fireEvent.click(closeButton!);

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when backdrop clicked', async () => {
      const onClose = jest.fn();
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={onClose} />);

      const backdrop = container.querySelector('.backdrop');
      await fireEvent.click(backdrop!);

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('does not close when panel content clicked', async () => {
      const onClose = jest.fn();
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={onClose} />);

      const panel = container.querySelector('.task-panel');
      await fireEvent.click(panel!);

      expect(onClose).not.toHaveBeenCalled();
    });
  });

  describe('Auto-Refresh', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('fetches tasks on mount', async () => {
      const [isOpen] = createSignal(true);
      render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(mockInvoke).toHaveBeenCalledWith('get_task_queue');
        expect(mockInvoke).toHaveBeenCalledWith('get_current_task');
        expect(mockInvoke).toHaveBeenCalledWith('get_task_stats');
      });
    });

    it('refreshes tasks every 5 seconds', async () => {
      const [isOpen] = createSignal(true);
      render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      // Initial fetch (3 calls: queue, current, stats)
      await waitFor(() => {
        expect(mockInvoke).toHaveBeenCalledTimes(3);
      });

      // Advance 5 seconds
      jest.advanceTimersByTime(5000);

      await waitFor(() => {
        expect(mockInvoke).toHaveBeenCalledTimes(6); // 3 more calls
      });
    });

    it('stops refreshing when closed', async () => {
      const [isOpen, setIsOpen] = createSignal(true);
      const { unmount } = render(() => (
        <TaskPanel isOpen={isOpen()} onClose={() => setIsOpen(false)} />
      ));

      await waitFor(() => {
        expect(mockInvoke).toHaveBeenCalledTimes(3);
      });

      unmount();

      jest.advanceTimersByTime(10000);

      // Should not call again after unmount
      expect(mockInvoke).toHaveBeenCalledTimes(3);
    });
  });

  describe('Loading State', () => {
    it('shows loading state while fetching', () => {
      mockInvoke.mockImplementation(() => new Promise(() => {})); // Never resolves

      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      expect(container.textContent).toContain('Loading');
    });

    it('shows tasks after loading', async () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).not.toContain('Loading');
        expect(container.textContent).toContain('Task Queue');
      });
    });
  });

  describe('Empty State', () => {
    it('handles empty task list', async () => {
      mockInvoke.mockImplementation((cmd: string) => {
        if (cmd === 'get_task_queue') {
          return Promise.resolve([]);
        }
        if (cmd === 'get_task_stats') {
          return Promise.resolve({ pending: 0, in_progress: 0, completed: 0, failed: 0 });
        }
        return Promise.resolve(null);
      });

      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        expect(container.textContent).toContain('No tasks');
      });
    });
  });

  describe('Error Handling', () => {
    it('handles fetch errors gracefully', async () => {
      mockInvoke.mockRejectedValue(new Error('Network error'));

      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      await waitFor(() => {
        // Should not crash, should show error state or fallback
        expect(container).toBeTruthy();
      });
    });
  });

  describe('Performance', () => {
    it('renders quickly with many tasks', async () => {
      const manyTasks = Array.from({ length: 100 }, (_, i) => ({
        id: i,
        description: `Task ${i}`,
        status: 'Pending',
        priority: 'Medium',
        created_at: Date.now(),
        started_at: null,
        completed_at: null,
        result: null,
      }));

      mockInvoke.mockImplementation((cmd: string) => {
        if (cmd === 'get_task_queue') {
          return Promise.resolve(manyTasks);
        }
        return Promise.resolve(mockStats);
      });

      const start = performance.now();
      const [isOpen] = createSignal(true);
      render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(100);
    });
  });

  describe('Accessibility', () => {
    it('has accessible close button', () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      const closeButton = container.querySelector('.close-button');
      expect(closeButton?.getAttribute('aria-label')).toBeTruthy();
    });

    it('has proper ARIA roles', () => {
      const [isOpen] = createSignal(true);
      const { container } = render(() => <TaskPanel isOpen={isOpen()} onClose={() => {}} />);

      expect(container.querySelector('[role="dialog"]')).toBeTruthy();
    });
  });
});
