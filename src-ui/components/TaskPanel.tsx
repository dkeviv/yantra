// File: src-ui/components/TaskPanel.tsx
// Purpose: Collapsible overlay showing current and upcoming agent tasks
// Design Philosophy: Minimal, slide-in panel, task list with status badges
// Last Updated: November 29, 2025

import { Component, createSignal, Show, For, onMount, onCleanup } from 'solid-js';
import { invoke } from '@tauri-apps/api/tauri';

interface Task {
  id: string;
  description: string;
  status: 'Pending' | 'InProgress' | 'Completed' | 'Failed';
  priority: 'Low' | 'Medium' | 'High' | 'Critical';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

interface TaskStats {
  total: number;
  pending: number;
  in_progress: number;
  completed: number;
  failed: number;
}

interface TaskPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const TaskPanel: Component<TaskPanelProps> = (props) => {
  const [tasks, setTasks] = createSignal<Task[]>([]);
  const [currentTask, setCurrentTask] = createSignal<Task | null>(null);
  const [stats, setStats] = createSignal<TaskStats | null>(null);
  const [loading, setLoading] = createSignal(true);

  // Load tasks on mount and periodically
  onMount(() => {
    loadTasks();
    const interval = setInterval(loadTasks, 5000); // Refresh every 5 seconds
    onCleanup(() => clearInterval(interval));
  });

  const loadTasks = async () => {
    try {
      const [allTasks, current, taskStats] = await Promise.all([
        invoke<Task[]>('get_task_queue'),
        invoke<Task | null>('get_current_task'),
        invoke<TaskStats>('get_task_stats'),
      ]);
      
      setTasks(allTasks);
      setCurrentTask(current);
      setStats(taskStats);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load tasks:', error);
      setLoading(false);
    }
  };

  const getStatusBadgeColor = (status: Task['status']) => {
    switch (status) {
      case 'Pending':
        return 'var(--text-tertiary)';
      case 'InProgress':
        return 'var(--accent-primary)';
      case 'Completed':
        return 'var(--status-success)';
      case 'Failed':
        return 'var(--status-error)';
      default:
        return 'var(--text-tertiary)';
    }
  };

  const getPriorityBadgeColor = (priority: Task['priority']) => {
    switch (priority) {
      case 'Critical':
        return 'var(--status-error)';
      case 'High':
        return 'var(--status-warning)';
      case 'Medium':
        return 'var(--accent-primary)';
      case 'Low':
        return 'var(--text-tertiary)';
      default:
        return 'var(--text-tertiary)';
    }
  };

  const formatDate = (dateString: string | number) => {
    try {
      const date = typeof dateString === 'string' ? new Date(dateString) : new Date(dateString);
      const now = Date.now();
      const diff = now - date.getTime();
      
      // Convert to seconds
      const seconds = Math.floor(diff / 1000);
      
      if (seconds < 60) {
        return `${seconds} ${seconds === 1 ? 'second' : 'seconds'} ago`;
      }
      
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) {
        return `${minutes} ${minutes === 1 ? 'minute' : 'minutes'} ago`;
      }
      
      const hours = Math.floor(minutes / 60);
      if (hours < 24) {
        return `${hours} ${hours === 1 ? 'hour' : 'hours'} ago`;
      }
      
      const days = Math.floor(hours / 24);
      return `${days} ${days === 1 ? 'day' : 'days'} ago`;
    } catch {
      return String(dateString);
    }
  };

  // Click-away listener
  const handleBackdropClick = (e: MouseEvent) => {
    if (e.target === e.currentTarget) {
      props.onClose();
    }
  };

  return (
    <Show when={props.isOpen}>
      {/* Backdrop */}
      <div
        class="backdrop fixed inset-0 z-40"
        style={{
          'background-color': 'rgba(0, 0, 0, 0.5)',
          transition: 'opacity 0.3s ease',
        }}
        onClick={handleBackdropClick}
      >
        {/* Panel */}
        <div
          class="task-panel fixed right-0 top-0 h-full w-80 shadow-2xl flex flex-col"
          style={{
            'background-color': 'var(--bg-primary)',
            transition: 'transform 0.3s ease',
            transform: props.isOpen ? 'translateX(0)' : 'translateX(100%)',
          }}
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-label="Task Queue Panel"
        >
          {/* Header */}
          <div
            class="px-4 py-3 border-b flex items-center justify-between"
            style={{
              'border-bottom-color': 'var(--border-primary)',
              'background-color': 'var(--bg-secondary)',
            }}
          >
            <h2 class="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
              Task Queue
            </h2>
            <button
              onClick={props.onClose}
              class="close-button text-2xl leading-none hover:opacity-70 transition-opacity"
              style={{ color: 'var(--text-secondary)' }}
              title="Close task panel"
              aria-label="Close task panel"
            >
              ×
            </button>
          </div>

          {/* Stats */}
          <Show when={stats()}>
            <div
              class="px-4 py-2 border-b grid grid-cols-5 gap-1 text-center"
              style={{ 'border-bottom-color': 'var(--border-primary)' }}
            >
              <div>
                <div class="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
                  {stats()!.total}
                </div>
                <div class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  Total
                </div>
              </div>
              <div>
                <div class="text-lg font-bold" style={{ color: 'var(--accent-primary)' }}>
                  {stats()!.in_progress}
                </div>
                <div class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  In Progress
                </div>
              </div>
              <div>
                <div class="text-lg font-bold" style={{ color: 'var(--text-tertiary)' }}>
                  {stats()!.pending}
                </div>
                <div class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  Pending
                </div>
              </div>
              <div>
                <div class="text-lg font-bold" style={{ color: 'var(--status-success)' }}>
                  {stats()!.completed}
                </div>
                <div class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  Completed
                </div>
              </div>
              <div>
                <div class="text-lg font-bold" style={{ color: 'var(--status-error)' }}>
                  {stats()!.failed}
                </div>
                <div class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  Failed
                </div>
              </div>
            </div>
          </Show>

          {/* Current Task */}
          <Show when={currentTask()}>
            <div
              class="current-task px-4 py-3 border-b"
              style={{
                'border-bottom-color': 'var(--border-primary)',
                'background-color': 'var(--bg-tertiary)',
              }}
            >
              <div class="flex items-center gap-2 mb-2">
                <div
                  class="w-2 h-2 rounded-full animate-pulse"
                  style={{ 'background-color': 'var(--accent-primary)' }}
                />
                <span class="text-xs font-semibold" style={{ color: 'var(--accent-primary)' }}>
                  CURRENT TASK
                </span>
              </div>
              <p class="text-sm" style={{ color: 'var(--text-primary)' }}>
                {currentTask()!.description}
              </p>
              <div class="flex items-center gap-2 mt-2">
                <span
                  class={`priority-${currentTask()!.priority.toLowerCase()} text-xs px-2 py-0.5 rounded`}
                  style={{
                    'background-color': getPriorityBadgeColor(currentTask()!.priority),
                    color: 'var(--text-inverse)',
                  }}
                >
                  {currentTask()!.priority}
                </span>
                <span class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  Started {formatDate(currentTask()!.started_at || currentTask()!.created_at)}
                </span>
              </div>
            </div>
          </Show>

          {/* Task List */}
          <div class="flex-1 overflow-y-auto px-4 py-3">
            <Show when={loading()}>
              <div class="text-center py-8" style={{ color: 'var(--text-tertiary)' }}>
                Loading tasks...
              </div>
            </Show>

            <Show when={!loading() && tasks().length === 0}>
              <div class="text-center py-8" style={{ color: 'var(--text-tertiary)' }}>
                No tasks in queue
              </div>
            </Show>

            <Show when={!loading() && tasks().length > 0}>
              <div class="space-y-2">
                <For each={tasks()}>
                  {(task) => (
                    <div
                      class="p-3 rounded border"
                      style={{
                        'border-color': 'var(--border-primary)',
                        'background-color': task.status === 'InProgress' ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
                      }}
                    >
                      <div class="flex items-start justify-between gap-2 mb-2">
                        <p class="text-sm flex-1" style={{ color: 'var(--text-primary)' }}>
                          {task.description}
                        </p>
                        <span
                          class={`badge-${task.status.toLowerCase().replace('inprogress', 'in-progress')} text-xs px-2 py-0.5 rounded whitespace-nowrap`}
                          style={{
                            'background-color': getStatusBadgeColor(task.status),
                            color: 'var(--text-inverse)',
                          }}
                        >
                          {task.status}
                        </span>
                      </div>
                      <div class="flex items-center gap-2">
                        <span
                          class={`priority-${task.priority.toLowerCase()} text-xs px-2 py-0.5 rounded`}
                          style={{
                            'background-color': getPriorityBadgeColor(task.priority),
                            color: 'var(--text-inverse)',
                          }}
                        >
                          {task.priority}
                        </span>
                        <span class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                          {formatDate(task.created_at)}
                        </span>
                      </div>
                      <Show when={task.error}>
                        <div
                          class="mt-2 text-xs p-2 rounded"
                          style={{
                            'background-color': 'rgba(239, 68, 68, 0.1)',
                            color: 'var(--status-error)',
                          }}
                        >
                          Error: {task.error}
                        </div>
                      </Show>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </div>
        </div>
      </div>
    </Show>
  );
};

export default TaskPanel;
