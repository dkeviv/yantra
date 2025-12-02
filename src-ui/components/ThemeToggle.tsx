// File: src-ui/components/ThemeToggle.tsx
// Purpose: Theme toggle button for switching between dark blue and bright white themes
// Created: November 29, 2025

import { Component, createSignal, onMount } from 'solid-js';

type Theme = 'dark' | 'bright';

const ThemeToggle: Component = () => {
  const [theme, setTheme] = createSignal<Theme>('dark');

  // Load theme from localStorage on mount
  onMount(() => {
    try {
      const savedTheme = localStorage?.getItem('theme') as Theme;
      if (savedTheme && (savedTheme === 'dark' || savedTheme === 'bright')) {
        setTheme(savedTheme);
        applyTheme(savedTheme);
      } else {
        // Set default theme
        applyTheme('dark');
      }
    } catch {
      // Fallback if localStorage is not available
      applyTheme('dark');
    }
  });

  // Apply theme to document
  const applyTheme = (newTheme: Theme) => {
    try {
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage?.setItem('theme', newTheme);
    } catch {
      // Ignore localStorage errors
      document.documentElement.setAttribute('data-theme', newTheme);
    }
  };

  // Toggle between themes
  const toggleTheme = () => {
    const newTheme: Theme = theme() === 'dark' ? 'bright' : 'dark';
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  return (
    <button
      onClick={toggleTheme}
      class="theme-toggle-btn"
      title={`Switch to ${theme() === 'dark' ? 'Bright' : 'Dark'} theme`}
      aria-label="Toggle theme"
    >
      {theme() === 'dark' ? '🌙' : '☀️'}
      <style>{`
        .theme-toggle-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 36px;
          height: 36px;
          border-radius: 8px;
          border: 1px solid var(--border-primary);
          background-color: var(--bg-secondary);
          color: var(--text-primary);
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .theme-toggle-btn:hover {
          background-color: var(--bg-tertiary);
          border-color: var(--border-secondary);
        }

        .theme-toggle-btn:active {
          transform: scale(0.95);
        }

        .theme-toggle-btn svg {
          transition: transform 0.3s ease;
        }

        .theme-toggle-btn:hover svg {
          transform: rotate(15deg);
        }
      `}</style>
    </button>
  );
};

export default ThemeToggle;
