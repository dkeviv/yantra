// File: src-ui/components/YDocArchivePanel.tsx
// Purpose: Test result archive management UI
// Dependencies: react, api/ydoc
// Last Updated: December 9, 2025

import React, { useState, useEffect } from 'react';
import { archiveOldTestResults, getArchivedTestResults, cleanupArchive } from '../api/ydoc';
import './YDocArchivePanel.css';

interface YDocArchivePanelProps {
  isExpanded?: boolean;
  onToggle?: () => void;
}

export const YDocArchivePanel: React.FC<YDocArchivePanelProps> = ({
  isExpanded = false,
  onToggle,
}) => {
  const [summaries, setSummaries] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [daysThreshold, setDaysThreshold] = useState(30);
  const [daysToKeep, setDaysToKeep] = useState(365);
  const [lastArchived, setLastArchived] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [expandedSummaries, setExpandedSummaries] = useState<Set<number>>(new Set());

  // Load summaries on component mount and when expanded
  useEffect(() => {
    if (isExpanded) {
      loadSummaries();
    }
  }, [isExpanded]);

  const loadSummaries = async () => {
    setLoading(true);
    setMessage(null);

    try {
      const results = await getArchivedTestResults();
      setSummaries(results);
    } catch (err) {
      console.error('Failed to load archived summaries:', err);
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to load summaries',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleArchive = async () => {
    setArchiving(true);
    setMessage(null);

    try {
      const count = await archiveOldTestResults(daysThreshold);
      setLastArchived(new Date().toLocaleString());
      setMessage({
        type: 'success',
        text: `Archived ${count} test result${count !== 1 ? 's' : ''}`,
      });

      // Reload summaries to show newly archived items
      await loadSummaries();
    } catch (err) {
      console.error('Failed to archive test results:', err);
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to archive results',
      });
    } finally {
      setArchiving(false);
    }
  };

  const handleCleanup = async () => {
    if (
      !window.confirm(
        `Delete archives older than ${daysToKeep} days?\n\nThis action cannot be undone.`
      )
    ) {
      return;
    }

    setCleaning(true);
    setMessage(null);

    try {
      const count = await cleanupArchive(daysToKeep);
      setMessage({
        type: 'success',
        text: `Deleted ${count} old archive${count !== 1 ? 's' : ''}`,
      });

      // Reload summaries to show updated list
      await loadSummaries();
    } catch (err) {
      console.error('Failed to cleanup archives:', err);
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to cleanup archives',
      });
    } finally {
      setCleaning(false);
    }
  };

  const toggleSummary = (index: number) => {
    const newExpanded = new Set(expandedSummaries);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSummaries(newExpanded);
  };

  const parseSummary = (summary: string) => {
    const lines = summary.split('\n');
    const title = lines[0]?.replace('ARCHIVED: ', '') || 'Unknown';
    const details: Record<string, string> = {};

    lines.slice(1).forEach((line) => {
      const [key, ...valueParts] = line.split(':');
      if (key && valueParts.length > 0) {
        details[key.trim()] = valueParts.join(':').trim();
      }
    });

    return { title, details };
  };

  if (!isExpanded) {
    return null;
  }

  return (
    <div className="ydoc-archive-panel">
      <div className="archive-header">
        <h4>📦 Test Result Archive</h4>
        {onToggle && (
          <button className="collapse-button" onClick={onToggle} title="Collapse">
            ▼
          </button>
        )}
      </div>

      <div className="archive-controls">
        <div className="control-row">
          <label>
            Archive results older than:
            <select
              value={daysThreshold}
              onChange={(e) => setDaysThreshold(Number(e.target.value))}
              disabled={archiving}
            >
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
              <option value={90}>90 days</option>
            </select>
          </label>

          <button
            onClick={handleArchive}
            disabled={archiving || loading}
            className="archive-button"
          >
            {archiving ? 'Archiving...' : 'Archive Now'}
          </button>
        </div>

        {lastArchived && <div className="last-archived">Last archived: {lastArchived}</div>}

        {message && <div className={`message message-${message.type}`}>{message.text}</div>}
      </div>

      <div className="archive-list">
        <div className="list-header">
          <h5>📊 Archived Summaries ({summaries.length})</h5>
          <button
            onClick={loadSummaries}
            disabled={loading}
            className="refresh-button"
            title="Refresh"
          >
            ↻
          </button>
        </div>

        {loading ? (
          <div className="loading">Loading summaries...</div>
        ) : summaries.length === 0 ? (
          <div className="empty-state">
            No archived test results yet.
            <br />
            <small>Click "Archive Now" to archive old test results.</small>
          </div>
        ) : (
          <div className="summaries">
            {summaries.map((summary, index) => {
              const { title, details } = parseSummary(summary);
              const isExpanded = expandedSummaries.has(index);

              return (
                <div key={index} className="summary-item">
                  <div className="summary-header" onClick={() => toggleSummary(index)}>
                    <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                    <span className="summary-title">{title}</span>
                    <span className="summary-meta">
                      {details['Modified at']?.split('T')[0] || 'Unknown date'}
                    </span>
                  </div>

                  {isExpanded && (
                    <div className="summary-details">
                      {Object.entries(details).map(([key, value]) => (
                        <div key={key} className="detail-row">
                          <span className="detail-key">{key}:</span>
                          <span className="detail-value">{value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="cleanup-controls">
        <h5>🧹 Cleanup Settings</h5>
        <div className="control-row">
          <label>
            Keep archives for:
            <select
              value={daysToKeep}
              onChange={(e) => setDaysToKeep(Number(e.target.value))}
              disabled={cleaning}
            >
              <option value={90}>90 days</option>
              <option value={180}>180 days</option>
              <option value={365}>1 year</option>
              <option value={730}>2 years</option>
            </select>
          </label>

          <button onClick={handleCleanup} disabled={cleaning || loading} className="cleanup-button">
            {cleaning ? 'Cleaning...' : 'Cleanup Old Archives'}
          </button>
        </div>
      </div>
    </div>
  );
};
