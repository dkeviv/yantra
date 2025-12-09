// File: src-ui/components/YDocBrowser.tsx
// Purpose: Document browser/tree view for YDoc documents
// Dependencies: react, api/ydoc
// Last Updated: December 9, 2025

import React, { useEffect, useState } from 'react';
import { listDocuments, getDocumentMetadata, DocumentMetadata } from '../api/ydoc';
import { YDocArchivePanel } from './YDocArchivePanel';
import './YDocBrowser.css';

interface DocumentListItem {
  id: string;
  type: string;
  title: string;
  metadata?: DocumentMetadata;
}

interface YDocBrowserProps {
  onDocumentSelect: (docId: string) => void;
  onNewDocument: () => void;
  selectedDocId?: string;
}

export const YDocBrowser: React.FC<YDocBrowserProps> = ({
  onDocumentSelect,
  onNewDocument,
  selectedDocId,
}) => {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set());
  const [filterType, setFilterType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showArchive, setShowArchive] = useState(false);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    setLoading(true);
    setError(null);

    try {
      const docs = await listDocuments();
      const documentItems: DocumentListItem[] = docs.map(([id, type, title]) => ({
        id,
        type,
        title,
      }));

      setDocuments(documentItems);

      // Load metadata for each document
      for (const doc of documentItems) {
        try {
          const metadata = await getDocumentMetadata(doc.id);
          setDocuments((prev) => prev.map((d) => (d.id === doc.id ? { ...d, metadata } : d)));
        } catch (err) {
          console.error(`Failed to load metadata for ${doc.id}:`, err);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
      console.error('Failed to load documents:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleDocumentExpanded = (docId: string) => {
    setExpandedDocs((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  const getDocTypeIcon = (type: string): string => {
    const icons: Record<string, string> = {
      ProductRequirement: '📋',
      FeatureSpec: '✨',
      TechnicalDesign: '🏗️',
      APIContract: '🔌',
      TestPlan: '🧪',
      UserGuide: '📖',
      DeveloperGuide: '💻',
      ReleaseNotes: '📝',
      ArchitectureDoc: '🏛️',
      SecuritySpec: '🔒',
      PerformanceSpec: '⚡',
      Glossary: '📚',
    };
    return icons[type] || '📄';
  };

  const getDocTypeColor = (type: string): string => {
    const colors: Record<string, string> = {
      ProductRequirement: '#C586C0',
      FeatureSpec: '#4EC9B0',
      TechnicalDesign: '#DCDCAA',
      APIContract: '#569CD6',
      TestPlan: '#9CDCFE',
      UserGuide: '#4FC1FF',
      DeveloperGuide: '#CE9178',
      ReleaseNotes: '#B5CEA8',
      ArchitectureDoc: '#C586C0',
      SecuritySpec: '#F48771',
      PerformanceSpec: '#DCDCAA',
      Glossary: '#9CDCFE',
    };
    return colors[type] || '#808080';
  };

  const filteredDocuments = documents.filter((doc) => {
    if (filterType !== 'all' && doc.type !== filterType) return false;
    if (searchQuery && !doc.title.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    return true;
  });

  const documentTypes = [...new Set(documents.map((d) => d.type))];

  return (
    <div className="ydoc-browser">
      <div className="browser-header">
        <h3>📚 YDoc Documents</h3>
        <button className="btn-new-doc" onClick={onNewDocument} title="Create new document">
          + New
        </button>
      </div>

      <div className="browser-controls">
        <input
          type="text"
          className="search-input"
          placeholder="Search documents..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <select
          className="filter-select"
          value={filterType}
          onChange={(e) => setFilterType(e.target.value)}
        >
          <option value="all">All Types</option>
          {documentTypes.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>

        <button className="btn-refresh" onClick={loadDocuments} title="Refresh">
          🔄
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={loadDocuments}>Retry</button>
        </div>
      )}

      {loading && documents.length === 0 ? (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading documents...</p>
        </div>
      ) : filteredDocuments.length === 0 ? (
        <div className="empty-state">
          <p>No documents found</p>
          <button className="btn-create-first" onClick={onNewDocument}>
            Create your first document
          </button>
        </div>
      ) : (
        <div className="documents-list">
          {filteredDocuments.map((doc) => {
            const isExpanded = expandedDocs.has(doc.id);
            const isSelected = selectedDocId === doc.id;

            return (
              <div key={doc.id} className={`document-item ${isSelected ? 'selected' : ''}`}>
                <div className="document-header" onClick={() => onDocumentSelect(doc.id)}>
                  <button
                    className="expand-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleDocumentExpanded(doc.id);
                    }}
                  >
                    {isExpanded ? '▼' : '▶'}
                  </button>

                  <span className="doc-icon">{getDocTypeIcon(doc.type)}</span>

                  <div className="doc-info">
                    <div className="doc-title">{doc.title}</div>
                    <div className="doc-type" style={{ color: getDocTypeColor(doc.type) }}>
                      {doc.type}
                    </div>
                  </div>

                  {doc.metadata && (
                    <div className="doc-stats">
                      <span className="stat-badge" title="Blocks">
                        {doc.metadata.block_count} 📝
                      </span>
                      <span className="stat-badge" title="Edges">
                        {doc.metadata.edge_count} 🔗
                      </span>
                    </div>
                  )}
                </div>

                {isExpanded && doc.metadata && (
                  <div className="document-details">
                    <div className="detail-row">
                      <span className="detail-label">Version:</span>
                      <span className="detail-value">{doc.metadata.version}</span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Modified:</span>
                      <span className="detail-value">
                        {new Date(doc.metadata.modified_at).toLocaleString()}
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Blocks:</span>
                      <span className="detail-value">{doc.metadata.block_count}</span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Edges:</span>
                      <span className="detail-value">{doc.metadata.edge_count}</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Archive Panel */}
      <YDocArchivePanel isExpanded={showArchive} onToggle={() => setShowArchive(!showArchive)} />

      <div className="browser-footer">
        <span className="doc-count">
          {filteredDocuments.length} document{filteredDocuments.length !== 1 ? 's' : ''}
        </span>
        <button
          className="btn-archive"
          onClick={() => setShowArchive(!showArchive)}
          title={showArchive ? 'Hide Archive' : 'Show Archive'}
        >
          {showArchive ? '📦 Hide Archive' : '📦 Archive'}
        </button>
      </div>
    </div>
  );
};
