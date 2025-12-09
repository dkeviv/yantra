// File: src-ui/monaco-ydoc-language.ts
// Purpose: Monaco Editor language definition for .ydoc files (Yantra Documentation Format)
// Dependencies: monaco-editor
// Last Updated: December 8, 2025

import * as monaco from 'monaco-editor';

/**
 * YDoc Language ID
 */
export const YDOC_LANGUAGE_ID = 'ydoc';

/**
 * YDoc file extension
 */
export const YDOC_EXTENSION = '.ydoc';

/**
 * Yantra metadata types for IntelliSense
 */
export const YANTRA_DOC_TYPES = [
  'ProductRequirement',
  'FeatureSpec',
  'TechnicalDesign',
  'APIContract',
  'TestPlan',
  'UserGuide',
  'DeveloperGuide',
  'ReleaseNotes',
  'ArchitectureDoc',
  'SecuritySpec',
  'PerformanceSpec',
  'Glossary',
];

export const YANTRA_BLOCK_TYPES = [
  'Overview',
  'Context',
  'Requirement',
  'Design',
  'Implementation',
  'Test',
  'Example',
  'Note',
  'Warning',
  'API',
  'Diagram',
  'Reference',
];

export const YANTRA_EDGE_TYPES = [
  'traces_to',
  'implements',
  'realized_in',
  'tested_by',
  'documents',
  'depends_on',
  'related_to',
  'has_issue',
];

/**
 * Register YDoc language with Monaco
 */
export function registerYDocLanguage() {
  // Register the language
  monaco.languages.register({
    id: YDOC_LANGUAGE_ID,
    extensions: [YDOC_EXTENSION],
    aliases: ['YDoc', 'ydoc', 'Yantra Documentation'],
    mimetypes: ['text/x-ydoc'],
  });

  // Set language configuration
  monaco.languages.setLanguageConfiguration(YDOC_LANGUAGE_ID, {
    comments: {
      lineComment: '//',
      blockComment: ['/*', '*/'],
    },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')'],
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
      { open: '`', close: '`' },
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
      { open: '`', close: '`' },
    ],
    folding: {
      markers: {
        start: new RegExp('^\\s*<!--\\s*#region\\b.*-->'),
        end: new RegExp('^\\s*<!--\\s*#endregion\\b.*-->'),
      },
    },
  });

  // Set monarch tokenizer for syntax highlighting
  monaco.languages.setMonarchTokensProvider(YDOC_LANGUAGE_ID, {
    defaultToken: '',
    tokenPostfix: '.ydoc',

    // Yantra metadata keywords
    keywords: [
      'yantra_id',
      'yantra_doc_id',
      'yantra_doc_type',
      'yantra_type',
      'yantra_title',
      'yantra_version',
      'created_at',
      'modified_at',
      'created_by',
      'modified_by',
      'modifier_id',
      'status',
      'graph_edges',
      'target_id',
      'target_type',
      'edge_type',
      'metadata',
    ],

    // Document types
    docTypes: YANTRA_DOC_TYPES,

    // Block types
    blockTypes: YANTRA_BLOCK_TYPES,

    // Edge types
    edgeTypes: YANTRA_EDGE_TYPES,

    // Markdown-like patterns
    escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
      root: [
        // YAML-like metadata block at start
        [/^---\s*$/, 'meta.separator', '@yamlMetadata'],

        // Markdown headers
        [/^(#{1,6}\s)(.*)$/, ['keyword', 'string']],

        // Code blocks
        [/^```(\w+)?/, 'string.code', '@codeblock'],

        // Lists
        [/^\s*[-*+]\s/, 'keyword'],
        [/^\s*\d+\.\s/, 'keyword'],

        // Bold, italic
        [/\*\*[^*]+\*\*/, 'emphasis'],
        [/\*[^*]+\*/, 'emphasis'],
        [/__[^_]+__/, 'emphasis'],
        [/_[^_]+_/, 'emphasis'],

        // Links
        [/\[([^\]]+)\]\(([^)]+)\)/, 'link'],

        // Inline code
        [/`[^`]+`/, 'string.code'],

        // Default text
        [/./, 'text'],
      ],

      yamlMetadata: [
        // End of metadata block
        [/^---\s*$/, 'meta.separator', '@pop'],

        // Yantra keywords
        [
          /(\s*)([a-zA-Z_]\w*)(\s*)(:)/,
          {
            cases: {
              '$2@keywords': ['white', 'keyword.ydoc', 'white', 'delimiter'],
              '@default': ['white', 'identifier', 'white', 'delimiter'],
            },
          },
        ],

        // Document types
        [
          /:\s*(\w+)/,
          {
            cases: {
              '$1@docTypes': 'type.doctype',
              '$1@blockTypes': 'type.blocktype',
              '$1@edgeTypes': 'type.edgetype',
              '@default': 'string',
            },
          },
        ],

        // Strings
        [/"([^"\\]|\\.)*$/, 'string.invalid'],
        [/"/, 'string', '@string_double'],
        [/'([^'\\]|\\.)*$/, 'string.invalid'],
        [/'/, 'string', '@string_single'],

        // Numbers
        [/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/, 'number.datetime'],
        [/\d+\.\d+/, 'number.float'],
        [/\d+/, 'number'],

        // Comments
        [/#.*$/, 'comment'],

        // Whitespace
        [/\s+/, 'white'],
      ],

      string_double: [
        [/[^\\"]+/, 'string'],
        [/@escapes/, 'string.escape'],
        [/\\./, 'string.escape.invalid'],
        [/"/, 'string', '@pop'],
      ],

      string_single: [
        [/[^\\']+/, 'string'],
        [/@escapes/, 'string.escape'],
        [/\\./, 'string.escape.invalid'],
        [/'/, 'string', '@pop'],
      ],

      codeblock: [
        [/^```/, 'string.code', '@pop'],
        [/.*/, 'string.code'],
      ],
    },
  });

  // Define theme colors for YDoc
  monaco.editor.defineTheme('ydoc-theme', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'keyword.ydoc', foreground: '569CD6', fontStyle: 'bold' },
      { token: 'type.doctype', foreground: '4EC9B0', fontStyle: 'bold' },
      { token: 'type.blocktype', foreground: '4EC9B0' },
      { token: 'type.edgetype', foreground: 'DCDCAA' },
      { token: 'string.code', foreground: 'CE9178', background: '1E1E1E' },
      { token: 'number.datetime', foreground: 'B5CEA8' },
      { token: 'meta.separator', foreground: '808080', fontStyle: 'bold' },
      { token: 'emphasis', fontStyle: 'italic' },
      { token: 'link', foreground: '3794FF', fontStyle: 'underline' },
    ],
    colors: {
      'editor.background': '#1E1E1E',
      'editor.foreground': '#D4D4D4',
      'editorLineNumber.foreground': '#858585',
      'editorLineNumber.activeForeground': '#C6C6C6',
    },
  });
}

/**
 * Register IntelliSense completion provider for YDoc files
 */
export function registerYDocCompletionProvider() {
  monaco.languages.registerCompletionItemProvider(YDOC_LANGUAGE_ID, {
    triggerCharacters: [':', ' ', '\n'],

    provideCompletionItems(model, position) {
      const lineContent = model.getLineContent(position.lineNumber);
      const linePrefix = lineContent.substring(0, position.column - 1);

      const suggestions: monaco.languages.CompletionItem[] = [];

      // Suggest yantra metadata fields
      if (linePrefix.trim() === '' || linePrefix.match(/^\s*$/)) {
        suggestions.push(...createMetadataFieldCompletions(position));
      }

      // Suggest doc types after yantra_doc_type:
      if (linePrefix.match(/yantra_doc_type:\s*$/)) {
        suggestions.push(
          ...YANTRA_DOC_TYPES.map((type) => ({
            label: type,
            kind: monaco.languages.CompletionItemKind.EnumMember,
            detail: `Document type: ${type}`,
            insertText: type,
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            },
          }))
        );
      }

      // Suggest block types after yantra_type:
      if (linePrefix.match(/yantra_type:\s*$/)) {
        suggestions.push(
          ...YANTRA_BLOCK_TYPES.map((type) => ({
            label: type,
            kind: monaco.languages.CompletionItemKind.EnumMember,
            detail: `Block type: ${type}`,
            insertText: type,
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            },
          }))
        );
      }

      // Suggest edge types after edge_type:
      if (linePrefix.match(/edge_type:\s*$/)) {
        suggestions.push(
          ...YANTRA_EDGE_TYPES.map((type) => ({
            label: type,
            kind: monaco.languages.CompletionItemKind.EnumMember,
            detail: `Edge type: ${type}`,
            insertText: type,
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            },
          }))
        );
      }

      return { suggestions };
    },
  });
}

/**
 * Create completion items for Yantra metadata fields
 */
function createMetadataFieldCompletions(
  position: monaco.Position
): monaco.languages.CompletionItem[] {
  const metadataFields = [
    {
      label: 'yantra_id',
      detail: 'Unique identifier for this block',
      insertText: 'yantra_id: "${1:block-uuid}"',
    },
    {
      label: 'yantra_doc_id',
      detail: 'Document identifier',
      insertText: 'yantra_doc_id: "${1:doc-uuid}"',
    },
    {
      label: 'yantra_doc_type',
      detail: 'Type of document',
      insertText:
        'yantra_doc_type: ${1|ProductRequirement,FeatureSpec,TechnicalDesign,APIContract,TestPlan|}',
    },
    {
      label: 'yantra_type',
      detail: 'Type of content block',
      insertText: 'yantra_type: ${1|Overview,Context,Requirement,Design,Implementation,Test|}',
    },
    {
      label: 'yantra_title',
      detail: 'Document title',
      insertText: 'yantra_title: "${1:Title}"',
    },
    {
      label: 'yantra_version',
      detail: 'Document version',
      insertText: 'yantra_version: "${1:1.0.0}"',
    },
    {
      label: 'created_at',
      detail: 'Creation timestamp (ISO-8601)',
      insertText: 'created_at: "${1:' + new Date().toISOString() + '}"',
    },
    {
      label: 'modified_at',
      detail: 'Last modification timestamp (ISO-8601)',
      insertText: 'modified_at: "${1:' + new Date().toISOString() + '}"',
    },
    {
      label: 'created_by',
      detail: 'Creator (user or agent)',
      insertText: 'created_by: "${1:user}"',
    },
    {
      label: 'modified_by',
      detail: 'Last modifier',
      insertText: 'modified_by: "${1:user}"',
    },
    {
      label: 'modifier_id',
      detail: 'Modifier identifier',
      insertText: 'modifier_id: "${1:user-123}"',
    },
    {
      label: 'status',
      detail: 'Document status',
      insertText: 'status: ${1|draft,review,approved,deprecated|}',
    },
    {
      label: 'graph_edges',
      detail: 'Traceability edges to other entities',
      insertText: [
        'graph_edges:',
        '  - target_id: "${1:target-id}"',
        '    target_type: "${2:code_file}"',
        '    edge_type: ${3|traces_to,implements,tested_by|}',
        '    metadata: "${4:optional metadata}"',
      ].join('\n  '),
    },
  ];

  return metadataFields.map((field) => ({
    label: field.label,
    kind: monaco.languages.CompletionItemKind.Field,
    detail: field.detail,
    insertText: field.insertText,
    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
    range: {
      startLineNumber: position.lineNumber,
      startColumn: position.column,
      endLineNumber: position.lineNumber,
      endColumn: position.column,
    },
  }));
}

/**
 * Register hover provider for YDoc files
 */
export function registerYDocHoverProvider() {
  monaco.languages.registerHoverProvider(YDOC_LANGUAGE_ID, {
    provideHover(model, position) {
      const word = model.getWordAtPosition(position);
      if (!word) return null;

      const hoverText = getHoverText(word.word);
      if (!hoverText) return null;

      return {
        range: new monaco.Range(
          position.lineNumber,
          word.startColumn,
          position.lineNumber,
          word.endColumn
        ),
        contents: [{ value: hoverText }],
      };
    },
  });
}

/**
 * Get hover text for a word
 */
function getHoverText(word: string): string | null {
  const hoverInfo: Record<string, string> = {
    yantra_id: '**Unique Identifier**: UUID for this content block. Used for traceability links.',
    yantra_doc_id:
      '**Document ID**: UUID identifying this document. Must be unique across the project.',
    yantra_doc_type:
      '**Document Type**: Category of documentation (ProductRequirement, FeatureSpec, TechnicalDesign, etc.)',
    yantra_type:
      '**Block Type**: Category of content block (Overview, Context, Requirement, Design, Implementation, Test, etc.)',
    yantra_title: '**Document Title**: Human-readable title for this document.',
    yantra_version: '**Version**: Semantic version number (e.g., 1.0.0).',
    created_at: '**Created At**: ISO-8601 timestamp when this was created.',
    modified_at: '**Modified At**: ISO-8601 timestamp of last modification.',
    created_by: '**Created By**: User or agent who created this (e.g., "user", "agent").',
    modified_by: '**Modified By**: User or agent who last modified this.',
    modifier_id:
      '**Modifier ID**: Unique identifier for the modifier (e.g., "user-123", "agent-task-456").',
    status: '**Status**: Current state of the document (draft, review, approved, deprecated).',
    graph_edges:
      '**Traceability Edges**: Links to code, tests, or other documentation. Enables impact analysis and coverage tracking.',
    target_id:
      '**Target ID**: ID of the entity this edge points to (file path, function name, etc.).',
    target_type:
      '**Target Type**: Type of target entity (code_file, function, class, doc_block, test_file, etc.).',
    edge_type:
      '**Edge Type**: Relationship type (traces_to, implements, realized_in, tested_by, documents, depends_on, etc.).',
    traces_to: '**Traces To**: This documentation requirement traces to implementation.',
    implements: '**Implements**: This code implements a specification.',
    realized_in: '**Realized In**: This design is realized in code.',
    tested_by: '**Tested By**: This requirement/code is tested by test files.',
    documents: '**Documents**: This documentation describes the target.',
    depends_on: '**Depends On**: This entity depends on another.',
    related_to: '**Related To**: Generic relationship between entities.',
    has_issue: '**Has Issue**: Links to a bug/issue tracking system.',
  };

  return hoverInfo[word] || null;
}

/**
 * Initialize all YDoc Monaco features
 */
export function initializeYDocMonaco() {
  registerYDocLanguage();
  registerYDocCompletionProvider();
  registerYDocHoverProvider();

  console.log('✅ YDoc Monaco language support initialized');
}
