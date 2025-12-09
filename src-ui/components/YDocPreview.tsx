// File: src-ui/components/YDocPreview.tsx
// Purpose: Live preview of YDoc document structure and metadata
// Dependencies: react, @monaco-editor/react
// Last Updated: December 8, 2025

import React, { useEffect, useState } from 'react';
import './YDocPreview.css';

interface YantraMetadata {
  yantra_id?: string;
  yantra_type?: string;
  created_at?: string;
  modified_at?: string;
  created_by?: string;
  modified_by?: string;
  modifier_id?: string;
  status?: string;
  graph_edges?: Array<{
    target_id: string;
    target_type: string;
    edge_type: string;
    metadata?: string;
  }>;
}

interface DocumentMetadata {
  yantra_doc_id?: string;
  yantra_doc_type?: string;
  yantra_title?: string;
  yantra_version?: string;
  created_at?: string;
  modified_at?: string;
  created_by?: string;
  status?: string;
}

interface YDocBlock {
  metadata: YantraMetadata;
  content: string;
  lineNumber: number;
}

interface YDocPreviewProps {
  content: string;
  currentLine?: number;
}

export const YDocPreview: React.FC<YDocPreviewProps> = ({ content, currentLine }) => {
  const [docMetadata, setDocMetadata] = useState<DocumentMetadata | null>(null);
  const [blocks, setBlocks] = useState<YDocBlock[]>([]);
  const [activeBlock, setActiveBlock] = useState<number>(-1);

  useEffect(() => {
    parseYDocContent(content);
  }, [content]);

  useEffect(() => {
    if (currentLine !== undefined) {
      // Find which block the current line belongs to
      const blockIndex = blocks.findIndex((block, idx) => {
        const nextBlock = blocks[idx + 1];
        return (
          currentLine >= block.lineNumber && (!nextBlock || currentLine < nextBlock.lineNumber)
        );
      });
      setActiveBlock(blockIndex);
    }
  }, [currentLine, blocks]);

  const parseYDocContent = (text: string) => {
    try {
      const lines = text.split('\n');
      let inMetadata = false;
      let metadataLines: string[] = [];
      const parsedBlocks: YDocBlock[] = [];
      let currentBlockMeta: YantraMetadata | null = null;
      let currentBlockContent: string[] = [];
      let currentBlockLine = 0;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Document metadata block
        if (line.trim() === '---') {
          if (!inMetadata) {
            inMetadata = true;
            metadataLines = [];
          } else {
            inMetadata = false;
            // Parse document metadata
            const metadata = parseYamlMetadata(metadataLines.join('\n'));
            setDocMetadata(metadata as DocumentMetadata);
          }
          continue;
        }

        if (inMetadata) {
          metadataLines.push(line);
          continue;
        }

        // Cell metadata block
        if (line.trim().startsWith('<!-- yantra:')) {
          // Save previous block if exists
          if (currentBlockMeta && currentBlockContent.length > 0) {
            parsedBlocks.push({
              metadata: currentBlockMeta,
              content: currentBlockContent.join('\n'),
              lineNumber: currentBlockLine,
            });
          }

          // Parse new block metadata
          const metadataMatch = line.match(/<!-- yantra:\s*({[\s\S]*?})\s*-->/);
          if (metadataMatch) {
            try {
              currentBlockMeta = JSON.parse(metadataMatch[1]);
              currentBlockContent = [];
              currentBlockLine = i + 1;
            } catch (e) {
              console.error('Failed to parse block metadata:', e);
            }
          }
        } else if (currentBlockMeta) {
          currentBlockContent.push(line);
        }
      }

      // Add last block
      if (currentBlockMeta && currentBlockContent.length > 0) {
        parsedBlocks.push({
          metadata: currentBlockMeta,
          content: currentBlockContent.join('\n'),
          lineNumber: currentBlockLine,
        });
      }

      setBlocks(parsedBlocks);
    } catch (error) {
      console.error('Error parsing YDoc content:', error);
    }
  };

  const parseYamlMetadata = (yaml: string): Record<string, any> => {
    const metadata: Record<string, any> = {};
    const lines = yaml.split('\n');

    for (const line of lines) {
      const match = line.match(/^\s*([a-zA-Z_]\w*):\s*(.+)$/);
      if (match) {
        const [, key, value] = match;
        metadata[key] = value.replace(/^["']|["']$/g, '');
      }
    }

    return metadata;
  };

  const getBlockTypeColor = (type?: string): string => {
    const colors: Record<string, string> = {
      Overview: '#4EC9B0',
      Context: '#569CD6',
      Requirement: '#C586C0',
      Design: '#DCDCAA',
      Implementation: '#CE9178',
      Test: '#9CDCFE',
      Example: '#4FC1FF',
      Note: '#B5CEA8',
      Warning: '#F48771',
      API: '#DCDCAA',
      Diagram: '#C586C0',
      Reference: '#9CDCFE',
    };
    return colors[type || ''] || '#808080';
  };

  const getEdgeTypeIcon = (type: string): string => {
    const icons: Record<string, string> = {
      traces_to: '→',
      implements: '✓',
      realized_in: '⚙',
      tested_by: '🧪',
      documents: '📄',
      depends_on: '⚡',
      related_to: '🔗',
      has_issue: '⚠',
    };
    return icons[type] || '•';
  };

  return (
    <div className="ydoc-preview">
      <div className="preview-header">
        <h3>📄 YDoc Preview</h3>
      </div>

      {/* Document Metadata */}
      {docMetadata && (
        <div className="doc-metadata">
          <div className="metadata-section">
            <h4>Document</h4>
            <div className="metadata-grid">
              {docMetadata.yantra_doc_id && (
                <div className="metadata-item">
                  <span className="meta-key">ID:</span>
                  <span className="meta-value">{docMetadata.yantra_doc_id}</span>
                </div>
              )}
              {docMetadata.yantra_doc_type && (
                <div className="metadata-item">
                  <span className="meta-key">Type:</span>
                  <span className="meta-value doc-type">{docMetadata.yantra_doc_type}</span>
                </div>
              )}
              {docMetadata.yantra_title && (
                <div className="metadata-item">
                  <span className="meta-key">Title:</span>
                  <span className="meta-value">{docMetadata.yantra_title}</span>
                </div>
              )}
              {docMetadata.yantra_version && (
                <div className="metadata-item">
                  <span className="meta-key">Version:</span>
                  <span className="meta-value">{docMetadata.yantra_version}</span>
                </div>
              )}
              {docMetadata.status && (
                <div className="metadata-item">
                  <span className="meta-key">Status:</span>
                  <span className={`meta-value status-${docMetadata.status}`}>
                    {docMetadata.status}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Document Structure */}
      {blocks.length > 0 && (
        <div className="doc-structure">
          <h4>Structure ({blocks.length} blocks)</h4>
          <div className="blocks-list">
            {blocks.map((block, index) => (
              <div key={index} className={`block-item ${activeBlock === index ? 'active' : ''}`}>
                <div className="block-header">
                  <span
                    className="block-type-indicator"
                    style={{ backgroundColor: getBlockTypeColor(block.metadata.yantra_type) }}
                  />
                  <span className="block-type">{block.metadata.yantra_type || 'Unknown'}</span>
                  <span className="block-line">Line {block.lineNumber}</span>
                </div>

                {block.metadata.yantra_id && (
                  <div className="block-id">
                    <span className="id-label">ID:</span>
                    <code>{block.metadata.yantra_id.substring(0, 8)}...</code>
                  </div>
                )}

                {block.metadata.graph_edges && block.metadata.graph_edges.length > 0 && (
                  <div className="block-edges">
                    <span className="edges-label">Edges:</span>
                    <div className="edges-list">
                      {block.metadata.graph_edges.map((edge, edgeIdx) => (
                        <div key={edgeIdx} className="edge-item">
                          <span className="edge-icon">{getEdgeTypeIcon(edge.edge_type)}</span>
                          <span className="edge-type">{edge.edge_type}</span>
                          <span className="edge-arrow">→</span>
                          <span className="edge-target">{edge.target_type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="block-preview">
                  {block.content.substring(0, 100)}
                  {block.content.length > 100 ? '...' : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {blocks.length === 0 && !docMetadata && (
        <div className="empty-state">
          <p>Start typing to see document structure...</p>
          <div className="example-hint">
            <code>---</code>
            <code>yantra_doc_type: FeatureSpec</code>
            <code>yantra_title: "My Document"</code>
            <code>---</code>
          </div>
        </div>
      )}
    </div>
  );
};
