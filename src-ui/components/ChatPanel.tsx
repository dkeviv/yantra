// File: src-ui/components/ChatPanel.tsx
// Purpose: Chat interface component for user interaction
// Dependencies: solid-js, appStore
// Last Updated: November 20, 2025

import { Component, For, createSignal } from 'solid-js';
import { appStore } from '../stores/appStore';

const ChatPanel: Component = () => {
  const [input, setInput] = createSignal('');

  const handleSend = () => {
    const message = input().trim();
    if (!message) return;

    // Add user message
    appStore.addMessage('user', message);
    setInput('');

    // TODO: Send to backend for processing
    appStore.setIsGenerating(true);

    // Mock response with sample Python code for testing Monaco Editor
    setTimeout(() => {
      appStore.addMessage(
        'assistant',
        'I\'ve generated a Python function for you. Check the Code panel to see it with syntax highlighting!'
      );
      
      // Update the code viewer with sample Python code
      const sampleCode = `# Generated Python Code
# Request: ${message}

def fibonacci(n: int) -> list[int]:
    """
    Generate Fibonacci sequence up to n terms.
    
    Args:
        n: Number of terms to generate
        
    Returns:
        List of Fibonacci numbers
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    return fib_sequence


def main():
    """Main function to demonstrate Fibonacci sequence."""
    print("Fibonacci Sequence Generator")
    print("-" * 40)
    
    n = 10
    sequence = fibonacci(n)
    print(f"First {n} Fibonacci numbers:")
    print(sequence)
    
    # Calculate sum
    total = sum(sequence)
    print(f"\\nSum of sequence: {total}")


if __name__ == "__main__":
    main()
`;
      appStore.updateCode(sampleCode);
      appStore.setIsGenerating(false);
    }, 1500);
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div class="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div class="px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold text-white">Chat</h2>
        <p class="text-sm text-gray-400 mt-1">Describe what you want to build</p>
      </div>

      {/* Messages */}
      <div class="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        <For each={appStore.messages()}>
          {(message) => (
            <div
              class={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                class={`max-w-[80%] rounded-lg px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-primary-600 text-white'
                    : message.role === 'system'
                    ? 'bg-gray-700 text-gray-200'
                    : 'bg-gray-800 text-gray-100'
                }`}
              >
                <div class="text-sm font-medium mb-1">
                  {message.role === 'user' ? 'You' : message.role === 'system' ? 'System' : 'Yantra'}
                </div>
                <div class="whitespace-pre-wrap">{message.content}</div>
                <div class="text-xs opacity-60 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          )}
        </For>

        {appStore.isGenerating() && (
          <div class="flex justify-start">
            <div class="bg-gray-800 text-gray-100 rounded-lg px-4 py-3">
              <div class="flex items-center space-x-2">
                <div class="animate-pulse">●</div>
                <span>Generating...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div class="px-6 py-4 border-t border-gray-700">
        <div class="flex space-x-2">
          <textarea
            value={input()}
            onInput={(e) => setInput(e.currentTarget.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your request... (Shift+Enter for new line)"
            class="flex-1 bg-gray-800 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            rows="3"
            disabled={appStore.isGenerating()}
          />
          <button
            onClick={handleSend}
            disabled={!input().trim() || appStore.isGenerating()}
            class="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;
