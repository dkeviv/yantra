import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src-ui/test-setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src-tauri/',
        'dist/',
        '**/*.config.*',
        '**/*.test.*',
        '**/__tests__/**',
      ],
    },
    // Exclude component tests due to SolidJS+vitest JSX compatibility issues
    // Store tests work fine with the browser build aliases below
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/src-ui/components/__tests__/**',
    ],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src-ui'),
      // Force SolidJS to use browser/client builds for store tests
      'solid-js/web': path.resolve(__dirname, './node_modules/solid-js/web/dist/web.js'),
      'solid-js': path.resolve(__dirname, './node_modules/solid-js/dist/solid.js'),
    },
    conditions: ['browser'],
  },
});
