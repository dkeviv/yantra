import { createSignal, onCleanup, Show } from 'solid-js';
import { listen } from '@tauri-apps/api/event';

export interface AgentState {
  phase: string;
  confidence: number;
  progress: number;
  status: 'idle' | 'running' | 'success' | 'error';
  message: string;
  timestamp: string;
}

export function AgentStatus() {
  const [agentState, setAgentState] = createSignal<AgentState>({
    phase: 'Idle',
    confidence: 0,
    progress: 0,
    status: 'idle',
    message: 'Waiting for task...',
    timestamp: new Date().toISOString()
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
      case 'running': return 'text-blue-500';
      case 'success': return 'text-green-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = () => {
    switch (agentState().status) {
      case 'running': return '🔄';
      case 'success': return '✅';
      case 'error': return '❌';
      default: return '⏸️';
    }
  };

  const getPhaseDescription = (phase: string) => {
    const descriptions: Record<string, string> = {
      'Idle': 'Agent is ready and waiting',
      'ContextAssembly': 'Gathering code context and dependencies',
      'CodeGeneration': 'Generating code with LLM',
      'DependencyValidation': 'Validating dependencies with GNN',
      'UnitTesting': 'Running pytest tests',
      'SecurityScanning': 'Scanning for vulnerabilities',
      'BrowserValidation': 'Validating in browser',
      'EnvironmentSetup': 'Setting up execution environment',
      'DependencyInstallation': 'Installing missing packages',
      'ScriptExecution': 'Running generated code',
      'RuntimeValidation': 'Validating runtime behavior',
      'PerformanceProfiling': 'Profiling performance',
      'Packaging': 'Building distributable package',
      'Deployment': 'Deploying to cloud',
      'Monitoring': 'Monitoring production',
      'Complete': 'Task completed successfully'
    };
    return descriptions[phase] || phase;
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return { label: 'High', color: 'text-green-500' };
    if (confidence >= 0.5) return { label: 'Medium', color: 'text-yellow-500' };
    return { label: 'Low', color: 'text-red-500' };
  };

  return (
    <div class="bg-gray-800 border border-gray-700 rounded-lg p-4 mb-4">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-lg font-semibold text-white">Agent Status</h3>
        <span class={`text-2xl ${getStatusColor()}`}>
          {getStatusIcon()}
        </span>
      </div>

      {/* Current Phase */}
      <div class="mb-3">
        <div class="flex items-center justify-between mb-1">
          <span class="text-sm text-gray-400">Current Phase</span>
          <span class="text-xs text-gray-500">
            {new Date(agentState().timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div class="text-white font-medium">{agentState().phase}</div>
        <div class="text-sm text-gray-400 mt-1">
          {getPhaseDescription(agentState().phase)}
        </div>
      </div>

      {/* Progress Bar */}
      <Show when={agentState().status === 'running'}>
        <div class="mb-3">
          <div class="flex items-center justify-between mb-1">
            <span class="text-sm text-gray-400">Progress</span>
            <span class="text-sm text-gray-400">{Math.round(agentState().progress)}%</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-2">
            <div 
              class="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${agentState().progress}%` }}
            />
          </div>
        </div>
      </Show>

      {/* Confidence Score */}
      <div class="mb-3">
        <div class="flex items-center justify-between mb-1">
          <span class="text-sm text-gray-400">Confidence</span>
          <span class={`text-sm font-medium ${getConfidenceLevel(agentState().confidence).color}`}>
            {getConfidenceLevel(agentState().confidence).label}
          </span>
        </div>
        <div class="flex items-center">
          <div class="flex-1 bg-gray-700 rounded-full h-2 mr-2">
            <div 
              class={`h-2 rounded-full transition-all duration-300 ${
                agentState().confidence >= 0.8 ? 'bg-green-500' :
                agentState().confidence >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${agentState().confidence * 100}%` }}
            />
          </div>
          <span class="text-sm text-white font-medium">
            {(agentState().confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Status Message */}
      <div class="mt-3 pt-3 border-t border-gray-700">
        <div class={`text-sm ${getStatusColor()}`}>
          {agentState().message}
        </div>
      </div>

      {/* Spinning indicator for running state */}
      <Show when={agentState().status === 'running'}>
        <div class="flex items-center justify-center mt-3">
          <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
        </div>
      </Show>
    </div>
  );
}
