# Agent Commands Reference

**Last Updated:** November 23, 2025  
**Version:** 0.3.0-agent-first

## Overview

Yantra implements an **agent-first architecture** where natural language commands in the chat interface control all UI actions. Instead of clicking buttons and menus, you simply type what you want to do in the chat.

### Philosophy: "Yantra is Agent and Agent is Yantra"

- The chat interface is the primary control mechanism
- Natural language commands replace button clicks
- The UI becomes a passive reflection of agent actions
- All controllable actions are accessible via chat commands

## Command Categories

### 1. Terminal Commands

Create, manage, and execute commands in integrated terminals.

| Command | Action | Example |
|---------|--------|---------|
| `open new terminal` | Creates a new terminal tab | "open new terminal" |
| `create terminal` | Creates a new terminal tab | "create terminal" |
| `run [command]` | Executes command in active terminal | "run npm test" |
| `execute [command]` | Executes command in active terminal | "execute cargo build" |
| `close terminal` | Closes active terminal | "close terminal" |
| `close terminal [number]` | Closes specific terminal by number | "close terminal 2" |

**Examples:**
```
User: open new terminal
Agent: ✅ Created new terminal

User: run npm install
Agent: ✅ Executing: npm install

User: execute cargo test
Agent: ✅ Executing: cargo test

User: close terminal 1
Agent: ✅ Closed terminal 1
```

### 2. View Commands

Switch between different views and panels in the UI.

| Command | Action | Example |
|---------|--------|---------|
| `show dependencies` | Switches to dependency graph view | "show dependencies" |
| `show dependency` | Switches to dependency graph view | "show dependency" |
| `display dependencies` | Switches to dependency graph view | "display dependencies" |
| `open dependencies` | Switches to dependency graph view | "open dependencies" |
| `show editor` | Switches to code editor view | "show editor" |
| `show code` | Switches to code editor view | "show code" |
| `display editor` | Switches to code editor view | "display editor" |
| `open editor` | Switches to code editor view | "open editor" |

**Examples:**
```
User: show dependencies
Agent: ✅ Switched to dependencies view

User: show editor
Agent: ✅ Switched to editor view
```

### 3. File Tree Commands

Control the visibility of the file tree panel.

| Command | Action | Example |
|---------|--------|---------|
| `hide file tree` | Collapses the file tree panel | "hide file tree" |
| `hide sidebar` | Collapses the file tree panel | "hide sidebar" |
| `collapse file tree` | Collapses the file tree panel | "collapse file tree" |
| `show file tree` | Expands the file tree panel | "show file tree" |
| `show sidebar` | Expands the file tree panel | "show sidebar" |
| `expand file tree` | Expands the file tree panel | "expand file tree" |

**Examples:**
```
User: hide file tree
Agent: ✅ File tree hidden

User: show sidebar
Agent: ✅ File tree shown
```

### 4. File Management Commands

Open, close, and list files in the editor.

| Command | Action | Example |
|---------|--------|---------|
| `open file [name]` | Opens a file by name/path | "open file main.rs" |
| `show file [name]` | Opens a file by name/path | "show file app.py" |
| `close file` | Closes the currently active file | "close file" |
| `list files` | Lists all currently open files | "list files" |
| `show files` | Lists all currently open files | "show files" |

**Examples:**
```
User: open file src/main.rs
Agent: ✅ Opened file: main.rs

User: list files
Agent: Open files:
1. main.rs
2. app.py
3. config.json

User: close file
Agent: ✅ Closed active file
```

### 5. Layout Commands

Control the layout and panel sizes.

| Command | Action | Example |
|---------|--------|---------|
| `reset layout` | Restores default panel sizes | "reset layout" |
| `restore layout` | Restores default panel sizes | "restore layout" |
| `default layout` | Restores default panel sizes | "default layout" |
| `maximize chat` | Expands chat panel to maximum | "maximize chat" |
| `maximize code` | Expands code panel to maximum | "maximize code" |
| `maximize editor` | Expands code panel to maximum | "maximize editor" |

**Examples:**
```
User: reset layout
Agent: ✅ Layout reset to default

User: maximize chat
Agent: ✅ Chat panel maximized

User: maximize code
Agent: ✅ Code panel maximized
```

### 6. Project Commands

Project-level operations and analysis.

| Command | Action | Example |
|---------|--------|---------|
| `open project` | Opens a project directory | "open project" |
| `open project [path]` | Opens specific project path | "open project ~/my-app" |
| `analyze project` | Analyzes the current project | "analyze project" |

**Examples:**
```
User: open project ~/Projects/my-app
Agent: ✅ Opened project: my-app

User: analyze project
Agent: ✅ Analyzing project...
```

## Command Features

### 1. Natural Language Processing

The agent uses pattern matching to understand natural variations:

- **Case insensitive:** "Open Terminal" = "open terminal" = "OPEN TERMINAL"
- **Flexible phrasing:** "show dependencies" = "display dependencies" = "open dependencies"
- **Short forms:** "deps" = "dependencies" (coming soon)

### 2. Context-Aware Suggestions

The agent provides intelligent command suggestions based on:

- Current view (editor vs dependencies)
- Open files
- Active terminal
- Recent actions
- Common workflows

**Example:**
```
When viewing dependencies, suggestions include:
- "show editor"
- "run npm test"
- "list files"

When in editor, suggestions include:
- "show dependencies"
- "open new terminal"
- "analyze project"
```

### 3. Command Feedback

Every command execution provides clear feedback:

- ✅ **Success:** Action completed successfully
- ❌ **Error:** Command failed with reason
- ⚠️ **Warning:** Command executed with warnings
- ℹ️ **Info:** Additional information

### 4. Command History

The agent tracks all executed commands:

- Stored in action history
- Timestamps for each action
- Success/failure status
- Error messages for debugging

## Advanced Usage

### Chaining Commands (Coming Soon)

Execute multiple commands in sequence:

```
User: open new terminal, run npm install, then run npm test
Agent: ✅ Created new terminal
       ✅ Executing: npm install
       ⏳ Waiting for completion...
       ✅ Executing: npm test
```

### Conditional Commands (Coming Soon)

Execute commands based on conditions:

```
User: if tests pass, then commit changes
Agent: ✅ Running tests...
       ✅ Tests passed
       ✅ Committing changes
```

### Variables and Parameters (Coming Soon)

Use variables in commands:

```
User: set $file = main.rs
User: open file $file
Agent: ✅ Opened file: main.rs
```

## Error Handling

### Unrecognized Commands

If a command isn't recognized, the agent provides helpful suggestions:

```
User: shut down terminal
Agent: ❌ Command not recognized. Did you mean:
       • "close terminal"
       • "close terminal [number]"
       
       Type "help" to see all available commands.
```

### Invalid Parameters

If a command has invalid parameters, the agent explains:

```
User: close terminal 99
Agent: ❌ Terminal 99 not found. Available terminals:
       • Terminal 1
       • Terminal 2
       
       Use "close terminal [1-2]"
```

### Missing Context

If a command requires context that's not available:

```
User: close file
Agent: ⚠️ No active file to close. Open files:
       (none)
       
       Try "open file [name]" first.
```

## Help System

### Getting Help

| Command | Action |
|---------|--------|
| `help` | Shows all available commands |
| `what can you do` | Shows all available commands |
| `show commands` | Shows all available commands |
| `help [category]` | Shows commands for specific category |

**Examples:**
```
User: help
Agent: Available command categories:
       • Terminal: open, run, close terminals
       • Views: show dependencies, show editor
       • Files: open, close, list files
       • Layout: reset, maximize panels
       • Project: open, analyze projects
       
       Type "help [category]" for details.

User: help terminal
Agent: Terminal Commands:
       • "open new terminal" - Create terminal
       • "run [command]" - Execute command
       • "close terminal" - Close terminal
```

## Tips and Best Practices

### 1. Use Descriptive Commands

**Good:**
```
run npm test
open file src/main.rs
show dependencies
```

**Less Clear:**
```
npm test
main.rs
deps
```

### 2. Start with Action Verbs

Commands that start with action verbs are recognized faster:

- ✅ "open new terminal"
- ✅ "show dependencies"
- ✅ "run npm test"

### 3. Use Exact File Names

When opening files, use complete names or paths:

- ✅ "open file main.rs"
- ✅ "open file src/stores/appStore.ts"
- ⚠️ "open file main" (might not find it)

### 4. Check Suggestions

The agent shows context-aware suggestions - use them!

- Faster than typing full commands
- Always valid for current context
- Reduce typing errors

### 5. Regular Chat Still Works

Not everything needs to be a command. You can still:

- Ask questions: "How do I implement authentication?"
- Request features: "Create a user login component"
- Get explanations: "Explain this code"

The agent knows the difference between commands and chat.

## Implementation Details

### Command Parser

The agent uses regex pattern matching in `agentStore.ts`:

```typescript
const COMMAND_PATTERNS = [
  {
    pattern: /^(open|create|new)\s+terminal$/i,
    action: 'create_terminal'
  },
  {
    pattern: /^run\s+(.+)$/i,
    action: 'run_command',
    extract: (match) => ({ command: match[1] })
  },
  // ... more patterns
];
```

### Action Execution

Commands are executed through the `executeAction()` function:

```typescript
switch (actionType) {
  case 'create_terminal':
    terminalStore.createTerminal();
    return '✅ Created new terminal';
    
  case 'run_command':
    terminalStore.executeCommand(params.command);
    return `✅ Executing: ${params.command}`;
    
  // ... more actions
}
```

### State Integration

Agent commands interact with existing stores:

- **terminalStore:** Terminal management
- **appStore:** View routing, file management
- **agentStore:** Command parsing and execution

## Roadmap

### Near Term (Week 1-2)

- ✅ Basic command parsing
- ✅ Terminal commands
- ✅ View switching
- ✅ File operations
- ✅ Layout control
- ⏳ Git commands
- ⏳ Search commands
- ⏳ Settings commands

### Medium Term (Week 3-4)

- ⏳ Command chaining
- ⏳ Conditional execution
- ⏳ Variables and parameters
- ⏳ Keyboard shortcuts
- ⏳ Voice control

### Long Term (Month 2+)

- ⏳ Custom command macros
- ⏳ Command aliases
- ⏳ Natural language refinement
- ⏳ Multi-step workflows
- ⏳ Command history search

## Troubleshooting

### Commands Not Working

1. **Check if command is recognized:**
   - Type "help" to see available commands
   - Look for typos in command
   - Check command syntax

2. **Verify context:**
   - Some commands need active files/terminals
   - Check if prerequisites are met
   - Read error message for clues

3. **Restart if needed:**
   - Reload application
   - Check browser console for errors
   - Report bugs on GitHub

### Suggestions Not Showing

1. **Enable suggestions:**
   - Type in chat input
   - Suggestions appear after 1-2 characters
   - Click or hover to use them

2. **Context matters:**
   - Suggestions change based on current view
   - More suggestions when files are open
   - Try different contexts

### Performance Issues

1. **Too many actions:**
   - Agent tracks all actions
   - Clear history periodically (coming soon)
   - Restart app if sluggish

2. **Long-running commands:**
   - Background processes handled automatically
   - Check terminal for output
   - Use "close terminal" if stuck

## Contributing

Want to add new commands?

1. Add pattern to `COMMAND_PATTERNS` in `agentStore.ts`
2. Implement action in `executeAction()` switch
3. Update this documentation
4. Submit PR with tests

**Example:**
```typescript
// Add to COMMAND_PATTERNS:
{
  pattern: /^commit\s+changes$/i,
  action: 'git_commit'
}

// Add to executeAction():
case 'git_commit':
  await gitStore.commitChanges();
  return '✅ Changes committed';
```

## Feedback

We're constantly improving the agent command system. Please share:

- Commands you wish existed
- Confusing error messages
- Better phrasing suggestions
- Use cases we missed

Open issues on GitHub or discuss in community channels.

---

**Remember:** Yantra is Agent and Agent is Yantra. The chat interface is your primary control mechanism. Embrace natural language commands for a more fluid development experience! 🚀
