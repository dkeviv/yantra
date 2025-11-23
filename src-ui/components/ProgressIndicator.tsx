import { createSignal, For, Show } from 'solid-js';

export interface ProgressStep {
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'failed';
  duration?: number;
  error?: string;
}

interface ProgressIndicatorProps {
  steps: ProgressStep[];
  currentStep: number;
}

export function ProgressIndicator(props: ProgressIndicatorProps) {
  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓';
      case 'in-progress': return '◉';
      case 'failed': return '✗';
      default: return '○';
    }
  };

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-500 border-green-500';
      case 'in-progress': return 'text-blue-500 border-blue-500';
      case 'failed': return 'text-red-500 border-red-500';
      default: return 'text-gray-500 border-gray-600';
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return '';
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <div class="bg-gray-800 rounded-lg p-4">
      <h4 class="text-white font-medium mb-4">Pipeline Progress</h4>
      
      <div class="space-y-3">
        <For each={props.steps}>
          {(step, index) => (
            <div class="flex items-start">
              {/* Step Icon */}
              <div class={`flex-shrink-0 w-8 h-8 rounded-full border-2 flex items-center justify-center ${getStepColor(step.status)}`}>
                <span class="text-sm font-bold">
                  {getStepIcon(step.status)}
                </span>
              </div>

              {/* Step Content */}
              <div class="ml-3 flex-1">
                <div class="flex items-center justify-between">
                  <span class={`font-medium ${
                    step.status === 'completed' ? 'text-gray-300' :
                    step.status === 'in-progress' ? 'text-white' :
                    step.status === 'failed' ? 'text-red-400' :
                    'text-gray-500'
                  }`}>
                    {step.name}
                  </span>
                  
                  <Show when={step.duration}>
                    <span class="text-xs text-gray-500">
                      {formatDuration(step.duration)}
                    </span>
                  </Show>
                </div>

                {/* In-progress animation */}
                <Show when={step.status === 'in-progress'}>
                  <div class="mt-2 w-full bg-gray-700 rounded-full h-1">
                    <div class="bg-blue-500 h-1 rounded-full animate-pulse" style={{ width: '60%' }} />
                  </div>
                </Show>

                {/* Error message */}
                <Show when={step.status === 'failed' && step.error}>
                  <div class="mt-1 text-xs text-red-400">
                    {step.error}
                  </div>
                </Show>
              </div>

              {/* Connecting Line */}
              <Show when={index() < props.steps.length - 1}>
                <div class="absolute left-4 top-8 w-0.5 h-6 bg-gray-700" style={{ 'margin-left': '14px' }} />
              </Show>
            </div>
          )}
        </For>
      </div>

      {/* Overall Progress */}
      <div class="mt-4 pt-4 border-t border-gray-700">
        <div class="flex items-center justify-between text-sm mb-2">
          <span class="text-gray-400">Overall Progress</span>
          <span class="text-white font-medium">
            {props.steps.filter(s => s.status === 'completed').length} / {props.steps.length}
          </span>
        </div>
        <div class="w-full bg-gray-700 rounded-full h-2">
          <div 
            class="bg-blue-500 h-2 rounded-full transition-all duration-500"
            style={{ 
              width: `${(props.steps.filter(s => s.status === 'completed').length / props.steps.length) * 100}%` 
            }}
          />
        </div>
      </div>
    </div>
  );
}

// Example usage component
export function OrchestrationProgress() {
  const [steps] = createSignal<ProgressStep[]>([
    { name: 'Context Assembly', status: 'completed', duration: 150 },
    { name: 'Code Generation', status: 'completed', duration: 2500 },
    { name: 'Dependency Validation', status: 'completed', duration: 45 },
    { name: 'Unit Testing', status: 'in-progress' },
    { name: 'Security Scanning', status: 'pending' },
    { name: 'Browser Validation', status: 'pending' },
    { name: 'Script Execution', status: 'pending' },
    { name: 'Deployment', status: 'pending' },
  ]);

  const [currentStep] = createSignal(3);

  return <ProgressIndicator steps={steps()} currentStep={currentStep()} />;
}
