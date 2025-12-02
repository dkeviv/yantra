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
    <div class="bg-gray-800 border-b border-gray-700 shadow-lg">
      <div class="flex items-center gap-2 px-4 py-2 overflow-x-auto">
        <span class="text-gray-400 text-sm mr-2">View:</span>
        <For each={tabs}>
          {(tab) => (
            <button
              onClick={() => handleTabClick(tab.mode)}
              class={`
                px-4 py-2 rounded-lg text-sm font-medium
                transition-all duration-200
                flex items-center gap-2 whitespace-nowrap
                ${
                  currentMode() === tab.mode
                    ? 'bg-blue-600 text-white shadow-lg scale-105'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
                }
              `}
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
