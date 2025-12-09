/**
 * HierarchicalTabs - Sliding tab navigation for filtering components
 *
 * Tabs:
 * - Complete (show all)
 * - Frontend
 * - Backend
 * - Database
 * - External
 * - Utility
 */

import { For } from 'solid-js';
import { setFilter, getFilter } from '../../stores/architectureStore';
import type { ComponentCategory } from '../../api/architecture';

type TabMode = ComponentCategory | 'Complete';

const tabs: { mode: TabMode; label: string; icon: string }[] = [
  { mode: 'Complete', label: 'Complete', icon: '🎯' },
  { mode: 'frontend', label: 'Frontend', icon: '🎨' },
  { mode: 'backend', label: 'Backend', icon: '⚙️' },
  { mode: 'database', label: 'Database', icon: '🗄️' },
  { mode: 'external', label: 'External', icon: '🔌' },
  { mode: 'utility', label: 'Utility', icon: '🛠️' },
];

export default function HierarchicalTabs() {
  const currentMode = getFilter;

  const handleTabClick = (mode: TabMode) => {
    setFilter(mode);
  };

  return (
    <div
      class="border-b shadow-lg"
      style={{ 'background-color': 'var(--bg-secondary)', 'border-color': 'var(--border-primary)' }}
    >
      <div class="flex items-center gap-2 px-4 py-2 overflow-x-auto">
        <span class="text-sm mr-2" style={{ color: 'var(--text-secondary)' }}>
          View:
        </span>
        <For each={tabs}>
          {(tab) => (
            <button
              onClick={() => handleTabClick(tab.mode)}
              class={`
                px-4 py-2 rounded-lg text-sm font-medium
                transition-all duration-200
                flex items-center gap-2 whitespace-nowrap
              `}
              style={{
                'background-color':
                  currentMode() === tab.mode ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                color:
                  currentMode() === tab.mode ? 'var(--text-on-accent)' : 'var(--text-secondary)',
                'box-shadow':
                  currentMode() === tab.mode ? '0 4px 6px -1px rgb(0 0 0 / 0.1)' : 'none',
                transform: currentMode() === tab.mode ? 'scale(1.05)' : 'scale(1)',
              }}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          )}
        </For>
      </div>
    </div>
  );
}
