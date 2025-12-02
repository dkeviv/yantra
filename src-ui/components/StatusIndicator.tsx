// File: src-ui/components/StatusIndicator.tsx
// Purpose: Visual indicator showing agent state (Running with spinner or Idle with static circle)
// Design Philosophy: Minimal, unobtrusive, clear visual feedback
// Last Updated: November 29, 2025

import { Component, Show } from 'solid-js';

export type AgentStatus = 'idle' | 'running';

interface StatusIndicatorProps {
  status?: AgentStatus;
  label?: string;
  size?: 'small' | 'medium' | 'large';
}

const StatusIndicator: Component<StatusIndicatorProps> = (props) => {
  const status = () => props.status || 'idle';
  const label = () => props.label;
  const size = () => props.size || 'small';  // Default to small as per tests

  // Size mappings
  const sizeClasses = {
    small: 'w-2 h-2',
    medium: 'w-3 h-3',
    large: 'w-4 h-4',
  };

  const sizePixels = {
    small: '16px',
    medium: '24px',
    large: '32px',
  };

  const containerClasses = {
    small: 'gap-1',
    medium: 'gap-1.5',
    large: 'gap-2',
  };

  const textClasses = {
    small: 'text-xs',
    medium: 'text-sm',
    large: 'text-base',
  };

  return (
    <div 
      class={`status-indicator flex items-center ${containerClasses[size()]} ${status()} ${size()}`}
      title={status() === 'running' ? 'Agent is working...' : 'Agent is idle'}
    >
      {/* Status Indicator Circle */}
      <div class="relative">
        <Show when={status() === 'running'}>
          {/* Animated Spinner for Running State */}
          <div
            class={`${sizeClasses[size()]} rounded-full animate-spin`}
            style={{
              width: sizePixels[size()],
              height: sizePixels[size()],
              border: '2px solid var(--bg-tertiary)',
              'border-top-color': 'var(--color-primary)',
            }}
          />
        </Show>
        
        <Show when={status() === 'idle'}>
          {/* Static Circle for Idle State */}
          <div
            class={`${sizeClasses[size()]} rounded-full`}
            style={{
              width: sizePixels[size()],
              height: sizePixels[size()],
              'background-color': 'var(--color-primary)',
            }}
          />
        </Show>
      </div>

      {/* Optional Label */}
      <Show when={label()}>
        <span 
          class={`${textClasses[size()]} font-medium`}
          style={{ color: 'var(--text-secondary)' }}
        >
          {label()}
        </span>
      </Show>
    </div>
  );
};

export default StatusIndicator;
