// Mock for @tauri-apps/api/tauri
// This ensures the invoke function is properly mocked in tests

export const invoke = jest.fn((cmd) => {
  // Return mock data based on command
  switch (cmd) {
    case 'get_task_queue':
      return Promise.resolve([
        {
          id: 1,
          description: 'Generate authentication API',
          status: 'InProgress',
          priority: 'High',
          created_at: Date.now() - 120000, // 2 minutes ago
          started_at: Date.now() - 120000,
          completed_at: null,
          result: null,
        },
        {
          id: 2,
          description: 'Create database models',
          status: 'Pending',
          priority: 'Medium',
          created_at: Date.now() - 300000, // 5 minutes ago
          started_at: null,
          completed_at: null,
          result: null,
        },
        {
          id: 3,
          description: 'Write unit tests',
          status: 'Completed',
          priority: 'High',
          created_at: Date.now() - 600000, // 10 minutes ago
          started_at: Date.now() - 600000,
          completed_at: Date.now() - 300000,
          result: 'Success',
        },
        {
          id: 4,
          description: 'Deploy to production',
          status: 'Failed',
          priority: 'Critical',
          created_at: Date.now() - 3600000, // 1 hour ago
          started_at: Date.now() - 3600000,
          completed_at: Date.now() - 3000000,
          result: 'Error: Connection timeout',
          error: 'Connection timeout',
        },
      ]);
    case 'get_current_task':
      return Promise.resolve({
        id: 1,
        description: 'Generate authentication API',
        status: 'InProgress',
        priority: 'High',
        created_at: Date.now() - 120000,
        started_at: Date.now() - 120000,
        completed_at: null,
        result: null,
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
