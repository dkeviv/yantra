// Jest setup file for component tests
require('@testing-library/jest-dom');

// Mock Tauri API
// Jest setup file for component tests
require('@testing-library/jest-dom');

// Mock Tauri API with proper Promise returns
const mockInvoke = jest.fn((cmd) => {
  // Return mock data based on command
  switch (cmd) {
    case 'get_task_queue':
      return Promise.resolve([
        {
          id: '1',
          description: 'Generate authentication API',
          status: 'InProgress',
          priority: 'High',
          created_at: '2025-11-30T10:00:00Z',
        },
        {
          id: '2',
          description: 'Write unit tests',
          status: 'Pending',
          priority: 'Medium',
          created_at: '2025-11-30T10:05:00Z',
        },
      ]);
    case 'get_current_task':
      return Promise.resolve({
        id: '1',
        description: 'Generate authentication API',
        status: 'InProgress',
        priority: 'High',
        created_at: '2025-11-30T10:00:00Z',
        started_at: '2025-11-30T10:01:00Z',
      });
    case 'get_task_stats':
      return Promise.resolve({
        total: 23,
        pending: 5,
        in_progress: 1,
        completed: 15,
        failed: 2,
      });
    default:
      return Promise.resolve(null);
  }
});

global.window.__TAURI__ = {
  tauri: { invoke: mockInvoke },
  event: { listen: jest.fn(), emit: jest.fn() },
};

// Mock localStorage
const localStorageMock = (() => {
  let store = {};
  return {
    getItem: (key) => store[key] || null,
    setItem: (key, value) => { store[key] = value.toString(); },
    removeItem: (key) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});
