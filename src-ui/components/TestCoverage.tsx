// File: src-ui/components/TestCoverage.tsx
// Purpose: Display test coverage metrics from GNN
// Shows tested/untested files, coverage percentage, test-to-code ratio

import { Component, createSignal, onMount, For, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/tauri";

interface TestCoverageMetrics {
  total_source_files: number;
  total_test_files: number;
  tested_source_files: number;
  untested_source_files: number;
  coverage_percentage: number;
  untested_files: string[];
}

interface TestCoverageProps {
  workspacePath?: string;
}

export const TestCoverage: Component<TestCoverageProps> = (props) => {
  const [metrics, setMetrics] = createSignal<TestCoverageMetrics | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [expanded, setExpanded] = createSignal(false);

  const loadCoverageMetrics = async () => {
    if (!props.workspacePath) {
      setError("No workspace loaded");
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Call Rust backend to get test coverage metrics
      const result = await invoke<TestCoverageMetrics>("get_test_coverage", {
        workspacePath: props.workspacePath,
      });

      setMetrics(result);
    } catch (err) {
      console.error("Failed to load test coverage:", err);
      setError(err as string);
    } finally {
      setLoading(false);
    }
  };

  onMount(() => {
    loadCoverageMetrics();
  });

  const getCoverageColor = (percentage: number): string => {
    if (percentage >= 80) return "text-green-500";
    if (percentage >= 60) return "text-yellow-500";
    if (percentage >= 40) return "text-orange-500";
    return "text-red-500";
  };

  const getCoverageBackgroundColor = (percentage: number): string => {
    if (percentage >= 80) return "bg-green-500";
    if (percentage >= 60) return "bg-yellow-500";
    if (percentage >= 40) return "bg-orange-500";
    return "bg-red-500";
  };

  return (
    <div class="flex flex-col h-full bg-[var(--background)] border-l border-[var(--border)]">
      {/* Header */}
      <div class="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
        <div class="flex items-center gap-2">
          <span class="text-lg">🧪</span>
          <h3 class="text-sm font-semibold">Test Coverage</h3>
        </div>
        <button
          onClick={loadCoverageMetrics}
          class="p-1 hover:bg-[var(--hover)] rounded"
          title="Refresh coverage"
        >
          <svg
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div class="flex-1 overflow-y-auto p-4">
        <Show
          when={!loading()}
          fallback={
            <div class="flex items-center justify-center h-32">
              <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
          }
        >
          <Show
            when={!error()}
            fallback={
              <div class="text-sm text-red-500 bg-red-50 dark:bg-red-900/20 p-3 rounded">
                {error()}
              </div>
            }
          >
            <Show when={metrics()}>
              {(m) => (
                <div class="space-y-4">
                  {/* Coverage Percentage - Large Display */}
                  <div class="bg-[var(--surface)] rounded-lg p-6 text-center">
                    <div class={`text-5xl font-bold ${getCoverageColor(m().coverage_percentage)}`}>
                      {m().coverage_percentage.toFixed(1)}%
                    </div>
                    <div class="text-sm text-[var(--text-secondary)] mt-2">
                      Test Coverage
                    </div>

                    {/* Progress Bar */}
                    <div class="mt-4 w-full bg-[var(--border)] rounded-full h-3 overflow-hidden">
                      <div
                        class={`h-full ${getCoverageBackgroundColor(m().coverage_percentage)} transition-all duration-300`}
                        style={{ width: `${m().coverage_percentage}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Stats Grid */}
                  <div class="grid grid-cols-2 gap-3">
                    <div class="bg-[var(--surface)] rounded-lg p-4">
                      <div class="text-2xl font-bold text-blue-500">
                        {m().tested_source_files}
                      </div>
                      <div class="text-xs text-[var(--text-secondary)] mt-1">
                        Tested Files
                      </div>
                    </div>

                    <div class="bg-[var(--surface)] rounded-lg p-4">
                      <div class="text-2xl font-bold text-gray-500">
                        {m().untested_source_files}
                      </div>
                      <div class="text-xs text-[var(--text-secondary)] mt-1">
                        Untested Files
                      </div>
                    </div>

                    <div class="bg-[var(--surface)] rounded-lg p-4">
                      <div class="text-2xl font-bold text-purple-500">
                        {m().total_test_files}
                      </div>
                      <div class="text-xs text-[var(--text-secondary)] mt-1">
                        Test Files
                      </div>
                    </div>

                    <div class="bg-[var(--surface)] rounded-lg p-4">
                      <div class="text-2xl font-bold text-teal-500">
                        {m().total_source_files}
                      </div>
                      <div class="text-xs text-[var(--text-secondary)] mt-1">
                        Source Files
                      </div>
                    </div>
                  </div>

                  {/* Ratio */}
                  <div class="bg-[var(--surface)] rounded-lg p-4">
                    <div class="flex items-center justify-between">
                      <span class="text-sm text-[var(--text-secondary)]">
                        Test-to-Code Ratio
                      </span>
                      <span class="text-sm font-semibold">
                        {m().total_test_files} : {m().total_source_files}
                      </span>
                    </div>
                  </div>

                  {/* Untested Files */}
                  <Show when={m().untested_files.length > 0}>
                    <div class="bg-[var(--surface)] rounded-lg overflow-hidden">
                      <button
                        onClick={() => setExpanded(!expanded())}
                        class="w-full flex items-center justify-between p-4 hover:bg-[var(--hover)] transition-colors"
                      >
                        <div class="flex items-center gap-2">
                          <span class="text-sm font-semibold">
                            Untested Files ({m().untested_files.length})
                          </span>
                        </div>
                        <svg
                          class={`w-4 h-4 transition-transform ${expanded() ? "rotate-180" : ""}`}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M19 9l-7 7-7-7"
                          />
                        </svg>
                      </button>

                      <Show when={expanded()}>
                        <div class="border-t border-[var(--border)] max-h-64 overflow-y-auto">
                          <For each={m().untested_files}>
                            {(file) => (
                              <div class="px-4 py-2 border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--hover)]">
                                <div class="flex items-center gap-2">
                                  <span class="text-red-500">⚠️</span>
                                  <span class="text-xs font-mono truncate">
                                    {file.split("/").pop()}
                                  </span>
                                </div>
                                <div class="text-xs text-[var(--text-secondary)] mt-1 truncate">
                                  {file}
                                </div>
                              </div>
                            )}
                          </For>
                        </div>
                      </Show>
                    </div>
                  </Show>

                  {/* Coverage Guide */}
                  <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                    <div class="text-xs font-semibold mb-2">Coverage Guide</div>
                    <div class="space-y-1 text-xs text-[var(--text-secondary)]">
                      <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span>≥80% - Excellent</span>
                      </div>
                      <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-yellow-500 rounded-full"></div>
                        <span>60-79% - Good</span>
                      </div>
                      <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-orange-500 rounded-full"></div>
                        <span>40-59% - Fair</span>
                      </div>
                      <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-red-500 rounded-full"></div>
                        <span>&lt;40% - Needs Work</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </Show>
          </Show>
        </Show>
      </div>
    </div>
  );
};
