// File: src-ui/api/ydoc.ts
// Purpose: API interface for YDoc Tauri commands
// Dependencies: @tauri-apps/api
// Last Updated: December 8, 2025

import { invoke } from '@tauri-apps/api/tauri';

/**
 * YDoc Document Types
 */
export enum DocumentType {
  ProductRequirement = 'ProductRequirement',
  FeatureSpec = 'FeatureSpec',
  TechnicalDesign = 'TechnicalDesign',
  APIContract = 'APIContract',
  TestPlan = 'TestPlan',
  UserGuide = 'UserGuide',
  DeveloperGuide = 'DeveloperGuide',
  ReleaseNotes = 'ReleaseNotes',
  ArchitectureDoc = 'ArchitectureDoc',
  SecuritySpec = 'SecuritySpec',
  PerformanceSpec = 'PerformanceSpec',
  Glossary = 'Glossary',
}

/**
 * YDoc Block Types
 */
export enum BlockType {
  Overview = 'Overview',
  Context = 'Context',
  Requirement = 'Requirement',
  Design = 'Design',
  Implementation = 'Implementation',
  Test = 'Test',
  Example = 'Example',
  Note = 'Note',
  Warning = 'Warning',
  API = 'API',
  Diagram = 'Diagram',
  Reference = 'Reference',
}

/**
 * Traceability Edge Types
 */
export enum EdgeType {
  TracesTo = 'traces_to',
  Implements = 'implements',
  RealizedIn = 'realized_in',
  TestedBy = 'tested_by',
  Documents = 'documents',
  DependsOn = 'depends_on',
  RelatedTo = 'related_to',
  HasIssue = 'has_issue',
}

/**
 * Request to create a new document
 */
export interface CreateDocumentRequest {
  doc_type: string;
  title: string;
  created_by: string;
}

/**
 * Request to create a new block
 */
export interface CreateBlockRequest {
  doc_id: string;
  yantra_type: string;
  content: string;
}

/**
 * Request to create a traceability edge
 */
export interface CreateEdgeRequest {
  source_block_id: string;
  target_block_id: string;
  edge_type: string;
  metadata?: string;
}

/**
 * Document metadata response
 */
export interface DocumentMetadata {
  doc_type: string;
  title: string;
  version: string;
  modified_at: string;
  block_count: number;
  edge_count: number;
}

/**
 * Traceability entity data
 */
export interface TraceabilityEntity {
  entity_id: string;
  entity_type: string;
  label?: string;
}

/**
 * Traceability chain response
 */
export interface TraceabilityChain {
  root: TraceabilityEntity;
  forward_chain: TraceabilityEntity[];
  backward_chain: TraceabilityEntity[];
}

/**
 * Coverage statistics
 */
export interface CoverageStats {
  total_requirements: number;
  implemented_requirements: number;
  tested_requirements: number;
  documented_code: number;
  coverage_percentage: number;
}

/**
 * Initialize YDoc system with project root
 */
export async function initializeYDoc(projectRoot: string): Promise<string> {
  return await invoke<string>('ydoc_initialize', { projectRoot });
}

/**
 * Create a new YDoc document
 */
export async function createDocument(request: CreateDocumentRequest): Promise<string> {
  return await invoke<string>('ydoc_create_document', { request });
}

/**
 * Create a new block in a document
 */
export async function createBlock(request: CreateBlockRequest): Promise<string> {
  return await invoke<string>('ydoc_create_block', { request });
}

/**
 * Create a traceability edge
 */
export async function createEdge(request: CreateEdgeRequest): Promise<string> {
  return await invoke<string>('ydoc_create_edge', { request });
}

/**
 * Load a document from disk
 */
export async function loadDocument(docId: string): Promise<string> {
  return await invoke<string>('ydoc_load_document', { docId });
}

/**
 * List all documents
 */
export async function listDocuments(): Promise<Array<[string, string, string]>> {
  return await invoke<Array<[string, string, string]>>('ydoc_list_documents');
}

/**
 * Get document metadata
 */
export async function getDocumentMetadata(docId: string): Promise<DocumentMetadata> {
  return await invoke<DocumentMetadata>('ydoc_get_document_metadata', {
    docId,
  });
}

/**
 * Search blocks using full-text search
 */
export async function searchBlocks(
  query: string
): Promise<Array<[string, string, string, string]>> {
  return await invoke<Array<[string, string, string, string]>>('ydoc_search_blocks', { query });
}

/**
 * Get traceability chain for a block
 */
export async function getTraceabilityChain(blockId: string): Promise<TraceabilityChain> {
  return await invoke<TraceabilityChain>('ydoc_get_traceability_chain', {
    blockId,
  });
}

/**
 * Get coverage statistics
 */
export async function getCoverageStats(): Promise<Record<string, number>> {
  return await invoke<Record<string, number>>('ydoc_get_coverage_stats');
}

/**
 * Export document to Markdown
 */
export async function exportToMarkdown(docId: string): Promise<string> {
  return await invoke<string>('ydoc_export_to_markdown', { docId });
}

/**
 * Export document to HTML
 */
export async function exportToHtml(docId: string): Promise<string> {
  return await invoke<string>('ydoc_export_to_html', { docId });
}

/**
 * Delete a document
 */
export async function deleteDocument(docId: string, deleteFile: boolean = true): Promise<string> {
  return await invoke<string>('ydoc_delete_document', { docId, deleteFile });
}

/**
 * Archive old test results (>30 days by default)
 * Returns the number of test results archived
 */
export async function archiveOldTestResults(daysThreshold: number = 30): Promise<number> {
  return await invoke<number>('ydoc_archive_old_test_results', { daysThreshold });
}

/**
 * Get archived test results summaries
 * Returns array of summary strings with statistics
 */
export async function getArchivedTestResults(): Promise<string[]> {
  return await invoke<string[]>('ydoc_get_archived_test_results');
}

/**
 * Clean up old archive entries
 * Removes archives older than daysToKeep (default 365 days)
 * Returns the number of archive entries deleted
 */
export async function cleanupArchive(daysToKeep: number = 365): Promise<number> {
  return await invoke<number>('ydoc_cleanup_archive', { daysToKeep });
}
