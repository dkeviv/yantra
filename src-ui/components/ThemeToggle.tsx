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
      // Check what's actually on the HTML element first
      const currentDataTheme = document.documentElement.getAttribute('data-theme');
      console.log('Current data-theme on HTML:', currentDataTheme);

      // Determine theme from data-theme attribute
      let initialTheme: Theme = 'dark';
      if (currentDataTheme === 'bright-white') {
        initialTheme = 'bright';
      } else if (currentDataTheme === 'dark-blue') {
        initialTheme = 'dark';
      }

      // Check localStorage
      const savedTheme = localStorage?.getItem('theme') as Theme;
      console.log('Saved theme from localStorage:', savedTheme);

      if (savedTheme && (savedTheme === 'dark' || savedTheme === 'bright')) {
        setTheme(savedTheme);
        applyTheme(savedTheme);
      } else {
        // Use what's on HTML or default to dark
        setTheme(initialTheme);
        applyTheme(initialTheme);
      }
    } catch {
      // Fallback if localStorage is not available
      setTheme('dark');
      applyTheme('dark');
    }
  });

  // Apply theme to document
  const applyTheme = (newTheme: Theme) => {
    try {
      const themeValue = newTheme === 'dark' ? 'dark-blue' : 'bright-white';
      document.documentElement.setAttribute('data-theme', themeValue);
      localStorage?.setItem('theme', newTheme);
    } catch {
      // Ignore localStorage errors
      const themeValue = newTheme === 'dark' ? 'dark-blue' : 'bright-white';
      document.documentElement.setAttribute('data-theme', themeValue);
    }
  };

  // Toggle between themes
  const toggleTheme = () => {
    const newTheme: Theme = theme() === 'dark' ? 'bright' : 'dark';
    console.log('Toggling theme from', theme(), 'to', newTheme);
    setTheme(newTheme);
    applyTheme(newTheme);
    console.log('Theme after toggle:', document.documentElement.getAttribute('data-theme'));
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
