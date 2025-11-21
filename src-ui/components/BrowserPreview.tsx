// File: src-ui/components/BrowserPreview.tsx
// Purpose: Browser preview component for live UI validation
// Dependencies: solid-js
// Last Updated: November 20, 2025

import { Component } from 'solid-js';

const BrowserPreview: Component = () => {
  return (
    <div class="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div class="px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold text-white">Preview</h2>
        <p class="text-sm text-gray-400 mt-1">Live browser validation</p>
      </div>

      {/* Preview Area */}
      <div class="flex-1 overflow-auto bg-white">
        <div class="h-full flex items-center justify-center text-gray-400">
          <div class="text-center">
            <svg
              class="mx-auto h-16 w-16 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              />
            </svg>
            <p class="mt-4 text-sm">No preview available</p>
            <p class="text-xs mt-2">UI code will be validated here</p>
          </div>
        </div>
      </div>

      {/* Footer - Console Output */}
      <div class="px-6 py-3 border-t border-gray-700">
        <div class="text-sm text-gray-400">
          <span class="font-medium">Console:</span> No errors
        </div>
      </div>
    </div>
  );
};

export default BrowserPreview;
