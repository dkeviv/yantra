// File: src-ui/stores/agentStore.ts
// Purpose: Agent-first command execution and UI control
// Dependencies: solid-js, appStore, terminalStore
// Last Updated: November 23, 2025

import { createSignal } from 'solid-js';
import { appStore } from './appStore';
import { terminalStore } from './terminalStore';

// Agent action types
export type AgentAction = {
  id: string;
  timestamp: Date;
  command: string;
  action: string;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  result?: string;
  error?: string;
};

// Agent state
const [actions, setActions] = createSignal<AgentAction[]>([]);
const [isProcessing, setIsProcessing] = createSignal(false);

// Command patterns - natural language to action mapping
const COMMAND_PATTERNS = [
  // Terminal commands
  { pattern: /^(?:open|create|new)\s+terminal$/i, action: 'create_terminal' },
  { pattern: /^close\s+terminal(?:\s+(\d+))?$/i, action: 'close_terminal' },
  { pattern: /^run\s+(.+)$/i, action: 'run_command' },
  { pattern: /^execute\s+(.+)$/i, action: 'run_command' },
  
  // View commands
  { pattern: /^(?:show|open|display)\s+(?:dependency|dependencies)$/i, action: 'show_dependencies' },
  { pattern: /^(?:show|open|display)\s+(?:code|editor)$/i, action: 'show_editor' },
  { pattern: /^(?:hide|close)\s+(?:file\s+)?tree$/i, action: 'hide_file_tree' },
  { pattern: /^(?:show|open|display)\s+(?:file\s+)?tree$/i, action: 'show_file_tree' },
  
  // File commands
  { pattern: /^(?:open|show)\s+(?:file\s+)?(.+)$/i, action: 'open_file' },
  { pattern: /^close\s+(?:file\s+)?(.+)?$/i, action: 'close_file' },
  { pattern: /^(?:list|show)\s+files$/i, action: 'list_files' },
  
  // Layout commands
  { pattern: /^reset\s+layout$/i, action: 'reset_layout' },
  { pattern: /^(?:maximize|focus)\s+chat$/i, action: 'maximize_chat' },
  { pattern: /^(?:maximize|focus)\s+code$/i, action: 'maximize_code' },
  
  // Project commands
  { pattern: /^(?:open|load)\s+project(?:\s+(.+))?$/i, action: 'open_project' },
  { pattern: /^analyze\s+project$/i, action: 'analyze_project' },
];

// Agent store with command execution
export const agentStore = {
  // Get all actions
  getActions: () => actions(),
  
  // Check if agent is processing
  isProcessing: () => isProcessing(),
  
  // Parse and execute natural language command
  executeCommand: async (input: string): Promise<string> => {
    const actionId = `action-${Date.now()}`;
    const timestamp = new Date();
    
    // Create action record
    const action: AgentAction = {
      id: actionId,
      timestamp,
      command: input,
      action: 'unknown',
      status: 'pending',
    };
    
    setActions([...actions(), action]);
    setIsProcessing(true);
    
    try {
      // Match command pattern
      let matched = false;
      for (const { pattern, action: actionType } of COMMAND_PATTERNS) {
        const match = input.match(pattern);
        if (match) {
          matched = true;
          action.action = actionType;
          action.status = 'executing';
          setActions([...actions().filter(a => a.id !== actionId), action]);
          
          // Execute the action
          const result = await executeAction(actionType, match);
          
          // Update action status
          action.status = 'completed';
          action.result = result;
          setActions([...actions().filter(a => a.id !== actionId), action]);
          setIsProcessing(false);
          
          return result;
        }
      }
      
      if (!matched) {
        action.status = 'failed';
        action.error = 'Command not recognized. Try: "open new terminal", "show dependencies", "run npm test"';
        setActions([...actions().filter(a => a.id !== actionId), action]);
        setIsProcessing(false);
        return action.error;
      }
    } catch (error) {
      action.status = 'failed';
      action.error = error instanceof Error ? error.message : String(error);
      setActions([...actions().filter(a => a.id !== actionId), action]);
      setIsProcessing(false);
      return `Error: ${action.error}`;
    }
    
    return 'Command executed';
  },
  
  // Clear action history
  clearActions: () => {
    setActions([]);
  },
};

// Execute specific action
async function executeAction(actionType: string, match: RegExpMatchArray): Promise<string> {
  switch (actionType) {
    // Terminal actions
    case 'create_terminal': {
      const terminalId = terminalStore.createTerminal();
      terminalStore.setActiveTerminal(terminalId);
      return `✅ Created new terminal: ${terminalId}`;
    }
    
    case 'close_terminal': {
      const terminalNum = match[1];
      if (terminalNum) {
        const terminalId = `terminal-${terminalNum}`;
        terminalStore.closeTerminal(terminalId);
        return `✅ Closed terminal ${terminalNum}`;
      } else {
        const activeTerminal = terminalStore.getActiveTerminal();
        if (activeTerminal) {
          terminalStore.closeTerminal(activeTerminal.id);
          return `✅ Closed active terminal`;
        }
        return '❌ No terminal to close';
      }
    }
    
    case 'run_command': {
      const command = match[1];
      const terminalId = await terminalStore.executeCommand(command);
      if (terminalId) {
        return `✅ Executing: ${command} in ${terminalId}`;
      }
      return `❌ Failed to execute command`;
    }
    
    // View actions
    case 'show_dependencies': {
      appStore.setActiveView('dependencies');
      return '✅ Showing dependency graph';
    }
    
    case 'show_editor': {
      appStore.setActiveView('editor');
      return '✅ Showing code editor';
    }
    
    case 'hide_file_tree': {
      appStore.setShowFileTree(false);
      return '✅ File tree hidden';
    }
    
    case 'show_file_tree': {
      appStore.setShowFileTree(true);
      return '✅ File tree visible';
    }
    
    // File actions
    case 'open_file': {
      const fileName = match[1];
      // TODO: Implement file search and open
      return `✅ Opening file: ${fileName} (search implementation pending)`;
    }
    
    case 'close_file': {
      const fileName = match[1];
      if (fileName) {
        // Find and close specific file
        const files = appStore.openFiles();
        const fileIndex = files.findIndex(f => f.path.includes(fileName) || f.name.includes(fileName));
        if (fileIndex >= 0) {
          appStore.closeFile(fileIndex);
          return `✅ Closed file: ${fileName}`;
        }
        return `❌ File not found: ${fileName}`;
      } else {
        // Close active file
        const activeIndex = appStore.activeFileIndex();
        if (activeIndex >= 0) {
          appStore.closeFile(activeIndex);
          return `✅ Closed active file`;
        }
        return `❌ No file to close`;
      }
    }
    
    case 'list_files': {
      const files = appStore.openFiles();
      if (files.length === 0) {
        return '📝 No files currently open';
      }
      return `📝 Open files:\n${files.map((f, i) => `  ${i + 1}. ${f.name}`).join('\n')}`;
    }
    
    // Layout actions
    case 'reset_layout': {
      appStore.setShowFileTree(true);
      appStore.setShowCode(true);
      appStore.setChatWidth(45);
      appStore.setCodeWidth(35);
      return '✅ Layout reset to default';
    }
    
    case 'maximize_chat': {
      appStore.setChatWidth(70);
      appStore.setCodeWidth(20);
      return '✅ Chat panel maximized';
    }
    
    case 'maximize_code': {
      appStore.setChatWidth(30);
      appStore.setCodeWidth(55);
      return '✅ Code panel maximized';
    }
    
    // Project actions
    case 'open_project': {
      const path = match[1];
      // TODO: Implement project opening
      return `✅ Opening project${path ? `: ${path}` : ''} (implementation pending)`;
    }
    
    case 'analyze_project': {
      // TODO: Implement project analysis
      return `✅ Analyzing project... (implementation pending)`;
    }
    
    default:
      return `❌ Action not implemented: ${actionType}`;
  }
}

// Quick command suggestions based on current context
export function getCommandSuggestions(): string[] {
  const suggestions: string[] = [];
  
  // Terminal suggestions
  if (terminalStore.canExecuteCommand()) {
    suggestions.push('open new terminal');
    suggestions.push('run npm test');
  }
  
  // View suggestions
  if (appStore.activeView() === 'editor') {
    suggestions.push('show dependencies');
  } else {
    suggestions.push('show code');
  }
  
  // File suggestions
  const openFiles = appStore.openFiles();
  if (openFiles.length > 0) {
    suggestions.push('close file');
    suggestions.push('list files');
  }
  
  return suggestions;
}
