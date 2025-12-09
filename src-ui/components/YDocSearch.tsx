// File: src-ui/components/YDocSearch.tsx
// Purpose: Full-text search interface for YDoc with FTS5 integration
// Dependencies: react, api/ydoc
// Last Updated: December 8, 2025

import React, { useState, useEffect } from 'react';
import { searchBlocks } from '../api/ydoc';
import './YDocSearch.css';

interface SearchResult {
  blockId: string;
  docId: string;
  yantraType: string;
  content: string;
}

interface YDocSearchProps {
  onResultClick: (docId: string, blockId: string) => void;
}

export const YDocSearch: React.FC<YDocSearchProps> = ({ onResultClick }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  useEffect(() => {
    // Load search history from localStorage
    const history = localStorage.getItem('ydoc_search_history');
    if (history) {
      try {
        setSearchHistory(JSON.parse(history));
      } catch (e) {
        console.error('Failed to parse search history:', e);
      }
    }
  }, []);

  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const searchResults = await searchBlocks(searchQuery);
      const formattedResults: SearchResult[] = searchResults.map(
        ([blockId, docId, yantraType, content]) => ({
          blockId,
          docId,
          yantraType,
          content,
        })
      );

      setResults(formattedResults);

      // Add to search history
      const newHistory = [searchQuery, ...searchHistory.filter((h) => h !== searchQuery)].slice(
        0,
        10
      );
      setSearchHistory(newHistory);
      localStorage.setItem('ydoc_search_history', JSON.stringify(newHistory));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    performSearch(query);
  };

  const handleHistoryClick = (historyQuery: string) => {
    setQuery(historyQuery);
    performSearch(historyQuery);
  };

  const clearHistory = () => {
    setSearchHistory([]);
    localStorage.removeItem('ydoc_search_history');
  };

  const highlightQuery = (text: string, searchQuery: string): React.ReactNode => {
    if (!searchQuery.trim()) return text;

    const parts = text.split(new RegExp(`(${searchQuery})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === searchQuery.toLowerCase() ? <mark key={i}>{part}</mark> : part
    );
  };

  const getBlockTypeColor = (type: string): string => {
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
    return colors[type] || '#808080';
  };

  return (
    <div className="ydoc-search">
      <div className="search-header">
        <h3>🔍 Search YDoc</h3>
      </div>

      <form className="search-form" onSubmit={handleSearch}>
        <input
          type="text"
          className="search-input"
          placeholder="Search documentation (e.g., 'authentication flow')..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
        <button type="submit" className="btn-search" disabled={!query.trim() || loading}>
          {loading ? '⏳' : '🔍'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
        </div>
      )}

      {searchHistory.length > 0 && results.length === 0 && !loading && (
        <div className="search-history">
          <div className="history-header">
            <span>Recent Searches</span>
            <button className="btn-clear-history" onClick={clearHistory}>
              Clear
            </button>
          </div>
          <div className="history-list">
            {searchHistory.map((historyItem, idx) => (
              <button
                key={idx}
                className="history-item"
                onClick={() => handleHistoryClick(historyItem)}
              >
                <span className="history-icon">🕐</span>
                {historyItem}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="search-loading">
          <div className="spinner"></div>
          <p>Searching...</p>
        </div>
      )}

      {!loading && results.length > 0 && (
        <div className="search-results">
          <div className="results-header">
            <span>
              {results.length} result{results.length !== 1 ? 's' : ''} for "{query}"
            </span>
          </div>

          <div className="results-list">
            {results.map((result, idx) => (
              <div
                key={`${result.docId}-${result.blockId}-${idx}`}
                className="result-item"
                onClick={() => onResultClick(result.docId, result.blockId)}
              >
                <div className="result-header">
                  <span
                    className="result-type"
                    style={{ color: getBlockTypeColor(result.yantraType) }}
                  >
                    {result.yantraType}
                  </span>
                  <span className="result-doc-id">{result.docId.substring(0, 8)}...</span>
                </div>

                <div className="result-content">
                  {highlightQuery(result.content.substring(0, 200), query)}
                  {result.content.length > 200 && '...'}
                </div>

                <div className="result-footer">
                  <span className="result-block-id">
                    Block: {result.blockId.substring(0, 8)}...
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!loading && query && results.length === 0 && !error && (
        <div className="no-results">
          <p>No results found for "{query}"</p>
          <div className="search-tips">
            <h4>Search Tips:</h4>
            <ul>
              <li>Try different keywords or phrases</li>
              <li>Use simpler terms</li>
              <li>Check your spelling</li>
              <li>Search for partial matches</li>
            </ul>
          </div>
        </div>
      )}

      {!loading && !query && results.length === 0 && (
        <div className="search-placeholder">
          <p>Enter a search query to find content across all YDoc documents</p>
          <div className="example-searches">
            <h4>Example searches:</h4>
            <button className="example-chip" onClick={() => handleHistoryClick('authentication')}>
              authentication
            </button>
            <button className="example-chip" onClick={() => handleHistoryClick('API')}>
              API
            </button>
            <button className="example-chip" onClick={() => handleHistoryClick('test')}>
              test
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
