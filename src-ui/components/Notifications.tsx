import { createSignal, For, Show, onCleanup } from 'solid-js';
import { listen } from '@tauri-apps/api/event';

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: number;
  duration?: number; // Auto-dismiss after ms (0 = no auto-dismiss)
  action?: {
    label: string;
    callback: () => void;
  };
}

export function Notifications() {
  const [notifications, setNotifications] = createSignal<Notification[]>([]);

  // Listen for notification events from Tauri backend
  const unlisten = listen<Notification>('notification', (event) => {
    addNotification(event.payload);
  });

  onCleanup(async () => {
    (await unlisten)();
  });

  const addNotification = (notification: Notification) => {
    setNotifications([...notifications(), notification]);

    // Auto-dismiss if duration specified
    if (notification.duration && notification.duration > 0) {
      setTimeout(() => {
        removeNotification(notification.id);
      }, notification.duration);
    }
  };

  const removeNotification = (id: string) => {
    setNotifications(notifications().filter((n) => n.id !== id));
  };

  const getNotificationStyles = (type: string) => {
    switch (type) {
      case 'success':
        return {
          bg: 'bg-green-900 border-green-500',
          icon: '✓',
          iconColor: 'text-green-400',
        };
      case 'error':
        return {
          bg: 'bg-red-900 border-red-500',
          icon: '✗',
          iconColor: 'text-red-400',
        };
      case 'warning':
        return {
          bg: 'bg-yellow-900 border-yellow-500',
          icon: '⚠',
          iconColor: 'text-yellow-400',
        };
      default:
        return {
          bg: 'bg-blue-900 border-blue-500',
          icon: 'ℹ',
          iconColor: 'text-blue-400',
        };
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <div class="fixed top-4 right-4 z-50 space-y-2 max-w-md">
      <For each={notifications()}>
        {(notification) => {
          const styles = getNotificationStyles(notification.type);

          return (
            <div class={`${styles.bg} border-l-4 rounded-lg shadow-lg p-4 animate-slide-in`}>
              <div class="flex items-start">
                {/* Icon */}
                <div class={`flex-shrink-0 ${styles.iconColor} text-xl font-bold mr-3`}>
                  {styles.icon}
                </div>

                {/* Content */}
                <div class="flex-1 mr-3">
                  <div class="flex items-center justify-between mb-1">
                    <h4 class="text-white font-semibold">{notification.title}</h4>
                    <span class="text-xs text-gray-400">{formatTime(notification.timestamp)}</span>
                  </div>
                  <p class="text-sm text-gray-300">{notification.message}</p>

                  {/* Action Button */}
                  <Show when={notification.action}>
                    <button
                      class="mt-2 px-3 py-1 bg-white bg-opacity-20 hover:bg-opacity-30 rounded text-sm text-white transition-colors"
                      onClick={notification.action!.callback}
                    >
                      {notification.action!.label}
                    </button>
                  </Show>
                </div>

                {/* Close Button */}
                <button
                  class="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
                  onClick={() => removeNotification(notification.id)}
                  title="Close notification"
                  aria-label="Close notification"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            </div>
          );
        }}
      </For>
    </div>
  );
}

// Helper function to create notifications from anywhere in the app
export function createNotification(
  type: Notification['type'],
  title: string,
  message: string,
  duration: number = 5000,
  action?: Notification['action']
): Notification {
  return {
    id: `notification-${Date.now()}-${Math.random()}`,
    type,
    title,
    message,
    timestamp: Date.now(),
    duration,
    action,
  };
}

// Pre-built notification creators
export const NotificationHelpers = {
  success: (title: string, message: string, duration = 5000) =>
    createNotification('success', title, message, duration),

  error: (title: string, message: string, duration = 0) =>
    createNotification('error', title, message, duration),

  warning: (title: string, message: string, duration = 7000) =>
    createNotification('warning', title, message, duration),

  info: (title: string, message: string, duration = 5000) =>
    createNotification('info', title, message, duration),
};

// Example notifications for common scenarios
export const AgentNotifications = {
  codeGenerated: () =>
    NotificationHelpers.success(
      'Code Generated',
      'Successfully generated code with 95% confidence'
    ),

  testsPassed: (count: number) =>
    NotificationHelpers.success('Tests Passed', `All ${count} tests passed successfully`),

  testsFailed: (count: number, total: number) =>
    NotificationHelpers.error(
      'Tests Failed',
      `${count} of ${total} tests failed. Retrying with fixes...`
    ),

  dependencyInstalled: (pkg: string) =>
    NotificationHelpers.info('Dependency Installed', `Successfully installed ${pkg}`),

  securityIssue: (severity: string, count: number) =>
    NotificationHelpers.warning(
      'Security Issues Found',
      `Found ${count} ${severity} security issues. Applying fixes...`
    ),

  deploymentStarted: (environment: string) =>
    NotificationHelpers.info('Deployment Started', `Deploying to ${environment}...`),

  deploymentSuccess: (url: string) =>
    NotificationHelpers.success(
      'Deployment Complete',
      `Application deployed successfully to ${url}`
    ),

  deploymentFailed: (error: string) => NotificationHelpers.error('Deployment Failed', error),

  lowConfidence: (confidence: number) =>
    NotificationHelpers.warning(
      'Low Confidence',
      `Agent confidence is ${(confidence * 100).toFixed(0)}%. Review recommended.`
    ),
};
