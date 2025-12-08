// File: src-ui/monaco-setup.ts
// Purpose: Monaco Editor worker configuration and completion provider for Vite
// Dependencies: monaco-editor
// Last Updated: December 7, 2025

import * as monaco from 'monaco-editor';
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';
import { getCodeCompletions, CompletionKind } from './api/completion';

// Configure Monaco Editor web workers
self.MonacoEnvironment = {
  getWorker(_: string, label: string) {
    if (label === 'json') {
      return new jsonWorker();
    }
    if (label === 'css' || label === 'scss' || label === 'less') {
      return new cssWorker();
    }
    if (label === 'html' || label === 'handlebars' || label === 'razor') {
      return new htmlWorker();
    }
    if (label === 'typescript' || label === 'javascript') {
      return new tsWorker();
    }
    return new editorWorker();
  },
};

/**
 * Register Yantra completion provider for a language
 * Integrates Tree-sitter AST parsing with GNN-based context completions
 */
export function registerCompletionProvider(language: string, projectPath: string) {
  monaco.languages.registerCompletionItemProvider(language, {
    triggerCharacters: ['.', '(', ' '],

    async provideCompletionItems(model, position, _context, _token) {
      try {
        const content = model.getValue();
        const filePath = model.uri.path;

        // Get completions from Rust backend (Tree-sitter + GNN)
        const completions = await getCodeCompletions({
          filePath,
          content,
          line: position.lineNumber,
          column: position.column - 1,
          projectPath,
          language,
        });

        // Convert to Monaco completion items
        const suggestions: monaco.languages.CompletionItem[] = completions.map((item) => ({
          label: item.label,
          kind: mapCompletionKind(item.kind),
          detail: item.detail,
          documentation: item.documentation
            ? {
                value: item.documentation,
                isTrusted: true,
              }
            : undefined,
          insertText: item.insertText,
          insertTextRules: item.insertTextAsSnippet
            ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
            : undefined,
          sortText: item.sortText,
          filterText: item.filterText,
          range: {
            startLineNumber: position.lineNumber,
            startColumn: getWordStart(model, position),
            endLineNumber: position.lineNumber,
            endColumn: position.column,
          },
        }));

        return { suggestions };
      } catch (error) {
        console.error('Completion provider error:', error);
        return { suggestions: [] };
      }
    },
  });
}

/**
 * Get the start column of the current word being typed
 */
function getWordStart(model: monaco.editor.ITextModel, position: monaco.Position): number {
  const lineContent = model.getLineContent(position.lineNumber);
  let column = position.column - 1;

  while (column > 0 && /[a-zA-Z0-9_]/.test(lineContent[column - 1])) {
    column--;
  }

  return column + 1;
}

/**
 * Map our CompletionKind enum to Monaco's CompletionItemKind
 */
function mapCompletionKind(kind: CompletionKind): monaco.languages.CompletionItemKind {
  switch (kind) {
    case CompletionKind.Method:
      return monaco.languages.CompletionItemKind.Method;
    case CompletionKind.Function:
      return monaco.languages.CompletionItemKind.Function;
    case CompletionKind.Constructor:
      return monaco.languages.CompletionItemKind.Constructor;
    case CompletionKind.Field:
      return monaco.languages.CompletionItemKind.Field;
    case CompletionKind.Variable:
      return monaco.languages.CompletionItemKind.Variable;
    case CompletionKind.Class:
      return monaco.languages.CompletionItemKind.Class;
    case CompletionKind.Struct:
      return monaco.languages.CompletionItemKind.Struct;
    case CompletionKind.Interface:
      return monaco.languages.CompletionItemKind.Interface;
    case CompletionKind.Module:
      return monaco.languages.CompletionItemKind.Module;
    case CompletionKind.Property:
      return monaco.languages.CompletionItemKind.Property;
    case CompletionKind.Keyword:
      return monaco.languages.CompletionItemKind.Keyword;
    case CompletionKind.Snippet:
      return monaco.languages.CompletionItemKind.Snippet;
    default:
      return monaco.languages.CompletionItemKind.Text;
  }
}

export { monaco };
