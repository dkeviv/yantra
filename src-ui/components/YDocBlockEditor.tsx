// File: src-ui/components/YDocBlockEditor.tsx
// Purpose: Block editor with metadata panel for YDoc documents
// Dependencies: react, @monaco-editor/react, api/ydoc
// Last Updated: December 8, 2025

import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { createBlock, BlockType, EdgeType } from '../api/ydoc';
import './YDocBlockEditor.css';

interface YantraMetadata {
  yantra_id: string;
  yantra_type: string;
  created_at: string;
  modified_at: string;
  created_by: string;
  modified_by: string;
  modifier_id: string;
  status: string;
  graph_edges: Array<{
    target_id: string;
    target_type: string;
    edge_type: string;
    metadata?: string;
  }>;
}

interface YDocBlockEditorProps {
  docId: string;
  blockId?: string;
  initialContent?: string;
  initialMetadata?: YantraMetadata;
  onSave: (content: string, metadata: YantraMetadata) => void;
  onCancel: () => void;
}

export const YDocBlockEditor: React.FC<YDocBlockEditorProps> = ({
  docId,
  blockId,
  initialContent = '',
  initialMetadata,
  onSave,
  onCancel,
}) => {
  const [content, setContent] = useState(initialContent);
  const [metadata, setMetadata] = useState<YantraMetadata>(
    initialMetadata || {
      yantra_id: blockId || generateUUID(),
      yantra_type: BlockType.Overview,
      created_at: new Date().toISOString(),
      modified_at: new Date().toISOString(),
      created_by: 'user',
      modified_by: 'user',
      modifier_id: 'user-1',
      status: 'draft',
      graph_edges: [],
    }
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'content' | 'metadata'>('content');
  const [newEdge, setNewEdge] = useState({
    target_id: '',
    target_type: 'code_file',
    edge_type: EdgeType.TracesTo,
    metadata: '',
  });

  function generateUUID(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  const handleSave = async () => {
    if (!content.trim()) {
      setError('Content cannot be empty');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      // Update modified timestamp
      const updatedMetadata = {
        ...metadata,
        modified_at: new Date().toISOString(),
      };

      // If creating new block, call backend API
      if (!blockId) {
        await createBlock({
          doc_id: docId,
          yantra_type: metadata.yantra_type,
          content: content,
        });
      }

      onSave(content, updatedMetadata);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save block');
      console.error('Save error:', err);
    } finally {
      setSaving(false);
    }
  };

  const updateMetadata = (field: keyof YantraMetadata, value: any) => {
    setMetadata((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const addEdge = () => {
    if (!newEdge.target_id.trim()) {
      setError('Target ID is required');
      return;
    }

    setMetadata((prev) => ({
      ...prev,
      graph_edges: [
        ...prev.graph_edges,
        {
          target_id: newEdge.target_id,
          target_type: newEdge.target_type,
          edge_type: newEdge.edge_type,
          metadata: newEdge.metadata || undefined,
        },
      ],
    }));

    // Reset form
    setNewEdge({
      target_id: '',
      target_type: 'code_file',
      edge_type: EdgeType.TracesTo,
      metadata: '',
    });
    setError(null);
  };

  const removeEdge = (index: number) => {
    setMetadata((prev) => ({
      ...prev,
      graph_edges: prev.graph_edges.filter((_, i) => i !== index),
    }));
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
    <div className="ydoc-block-editor">
      <div className="editor-header">
        <h3>{blockId ? '✏️ Edit Block' : '➕ New Block'}</h3>
        <div className="editor-tabs">
          <button
            className={`tab-btn ${activeTab === 'content' ? 'active' : ''}`}
            onClick={() => setActiveTab('content')}
          >
            Content
          </button>
          <button
            className={`tab-btn ${activeTab === 'metadata' ? 'active' : ''}`}
            onClick={() => setActiveTab('metadata')}
          >
            Metadata
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="editor-body">
        {activeTab === 'content' ? (
          <div className="content-editor">
            <Editor
              height="100%"
              defaultLanguage="markdown"
              theme="vs-dark"
              value={content}
              onChange={(value) => setContent(value || '')}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: 'on',
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                padding: { top: 16, bottom: 16 },
              }}
            />
          </div>
        ) : (
          <div className="metadata-panel">
            <div className="metadata-section">
              <h4>Block Information</h4>

              <div className="form-group">
                <label>Block ID</label>
                <input type="text" value={metadata.yantra_id} readOnly className="input-readonly" />
              </div>

              <div className="form-group">
                <label>Block Type</label>
                <select
                  value={metadata.yantra_type}
                  onChange={(e) => updateMetadata('yantra_type', e.target.value)}
                  className="select-input"
                >
                  {Object.values(BlockType).map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Status</label>
                <select
                  value={metadata.status}
                  onChange={(e) => updateMetadata('status', e.target.value)}
                  className="select-input"
                >
                  <option value="draft">Draft</option>
                  <option value="review">Review</option>
                  <option value="approved">Approved</option>
                  <option value="deprecated">Deprecated</option>
                </select>
              </div>

              <div className="form-group">
                <label>Created By</label>
                <input
                  type="text"
                  value={metadata.created_by}
                  onChange={(e) => updateMetadata('created_by', e.target.value)}
                  className="text-input"
                />
              </div>

              <div className="form-group">
                <label>Modified By</label>
                <input
                  type="text"
                  value={metadata.modified_by}
                  onChange={(e) => updateMetadata('modified_by', e.target.value)}
                  className="text-input"
                />
              </div>

              <div className="form-group">
                <label>Modifier ID</label>
                <input
                  type="text"
                  value={metadata.modifier_id}
                  onChange={(e) => updateMetadata('modifier_id', e.target.value)}
                  className="text-input"
                />
              </div>
            </div>

            <div className="metadata-section">
              <h4>Traceability Edges ({metadata.graph_edges.length})</h4>

              {metadata.graph_edges.length > 0 && (
                <div className="edges-list">
                  {metadata.graph_edges.map((edge, idx) => (
                    <div key={idx} className="edge-item">
                      <div className="edge-info">
                        <span className="edge-icon">{getEdgeTypeIcon(edge.edge_type)}</span>
                        <div className="edge-details">
                          <div className="edge-type">{edge.edge_type}</div>
                          <div className="edge-target">
                            {edge.target_type}: {edge.target_id}
                          </div>
                          {edge.metadata && <div className="edge-metadata">{edge.metadata}</div>}
                        </div>
                      </div>
                      <button
                        className="btn-remove-edge"
                        onClick={() => removeEdge(idx)}
                        title="Remove edge"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="add-edge-form">
                <h5>Add New Edge</h5>

                <div className="form-group">
                  <label>Edge Type</label>
                  <select
                    value={newEdge.edge_type}
                    onChange={(e) => setNewEdge((prev) => ({ ...prev, edge_type: e.target.value }))}
                    className="select-input"
                  >
                    {Object.values(EdgeType).map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Target Type</label>
                  <select
                    value={newEdge.target_type}
                    onChange={(e) =>
                      setNewEdge((prev) => ({ ...prev, target_type: e.target.value }))
                    }
                    className="select-input"
                  >
                    <option value="code_file">Code File</option>
                    <option value="function">Function</option>
                    <option value="class">Class</option>
                    <option value="doc_block">Doc Block</option>
                    <option value="test_file">Test File</option>
                    <option value="api_endpoint">API Endpoint</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Target ID</label>
                  <input
                    type="text"
                    value={newEdge.target_id}
                    onChange={(e) => setNewEdge((prev) => ({ ...prev, target_id: e.target.value }))}
                    placeholder="e.g., src/auth/login.ts"
                    className="text-input"
                  />
                </div>

                <div className="form-group">
                  <label>Metadata (optional)</label>
                  <input
                    type="text"
                    value={newEdge.metadata}
                    onChange={(e) => setNewEdge((prev) => ({ ...prev, metadata: e.target.value }))}
                    placeholder="e.g., line:42-56"
                    className="text-input"
                  />
                </div>

                <button className="btn-add-edge" onClick={addEdge}>
                  + Add Edge
                </button>
              </div>
            </div>

            <div className="metadata-section timestamps">
              <div className="timestamp-item">
                <span className="timestamp-label">Created:</span>
                <span className="timestamp-value">
                  {new Date(metadata.created_at).toLocaleString()}
                </span>
              </div>
              <div className="timestamp-item">
                <span className="timestamp-label">Modified:</span>
                <span className="timestamp-value">
                  {new Date(metadata.modified_at).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="editor-footer">
        <button className="btn-cancel" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
        <button className="btn-save" onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save Block'}
        </button>
      </div>
    </div>
  );
};
