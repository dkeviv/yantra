import { createSignal, onCleanup } from 'solid-js';
import { listen } from '@tauri-apps/api/event';

export interface AgentState {
  phase: string;
  progress: number;
  status: 'idle' | 'running' | 'success' | 'error';
  currentTask: string;
  timestamp: string;
}

export function AgentStatus() {
  const [agentState, setAgentState] = createSignal<AgentState>({
    phase: 'Idle',
    progress: 0,
    status: 'idle',
    currentTask: 'Waiting for task...',
    timestamp: new Date().toISOString(),
  });

  // Listen for agent status updates from Tauri backend
  const unlisten = listen<AgentState>('agent-status-update', (event) => {
    setAgentState(event.payload);
  });

  onCleanup(async () => {
    (await unlisten)();
  });

  const getStatusColor = () => {
    switch (agentState().status) {
      case 'running':
        return 'text-blue-500';
      case 'success':
        return 'text-green-500';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = () => {
    switch (agentState().status) {
      case 'running':
        return '🔄';
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      default:
        return '⏸️';
    }
  };

  return (
    <div
      class="p-2"
      style={{
        'background-color': 'var(--bg-secondary)',
        'border-top': '1px solid var(--border-primary)',
      }}
    >
      {/* Minimal Status Display */}
      <div class="flex items-center justify-between mb-1">
        <span class={`text-xs ${getStatusColor()}`}>
          {getStatusIcon()} {agentState().phase}
        </span>
        <span class="text-xs" style={{ color: 'var(--text-tertiary)' }}>
          {Math.round(agentState().progress)}%
        </span>
      </div>

      {/* Progress Bar */}
      <div
        class="w-full rounded-full h-1 mb-1"
        style={{ 'background-color': 'var(--bg-tertiary)' }}
      >
        <div
          class="h-1 rounded-full transition-all duration-300"
          style={{
            width: `${agentState().progress}%`,
            'background-color': 'var(--accent-primary)',
          }}
        />
      </div>

      {/* Current Task */}
      <div class="text-xs truncate" style={{ color: 'var(--text-tertiary)' }}>
        {agentState().currentTask}
      </div>
    </div>
  );
}

export default AgentStatus;
