// File: src-ui/stores/appStore.test.ts
// Purpose: Unit tests for appStore state management
// Dependencies: vitest, appStore
// Last Updated: November 20, 2025

import { describe, it, expect } from 'vitest';
import { appStore } from './appStore';

describe('appStore', () => {
  describe('messages', () => {
    it('should have initial welcome message', () => {
      const messages = appStore.messages();
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('system');
      expect(messages[0].content).toContain('Welcome to Yantra');
    });

    it('should add user message', () => {
      const initialCount = appStore.messages().length;
      appStore.addMessage('user', 'Test message');
      expect(appStore.messages()).toHaveLength(initialCount + 1);
      expect(appStore.messages()[initialCount].content).toBe('Test message');
    });

    it('should add assistant message', () => {
      const initialCount = appStore.messages().length;
      appStore.addMessage('assistant', 'Assistant reply');
      expect(appStore.messages()).toHaveLength(initialCount + 1);
      expect(appStore.messages()[initialCount].role).toBe('assistant');
    });

    it('should clear all messages', () => {
      appStore.addMessage('user', 'Test 1');
      appStore.addMessage('user', 'Test 2');
      appStore.clearMessages();
      expect(appStore.messages()).toHaveLength(0);
    });
  });

  describe('currentCode', () => {
    it('should have initial placeholder code', () => {
      const code = appStore.currentCode();
      expect(code).toContain('Your generated code will appear here');
    });

    it('should update code', () => {
      const newCode = 'def hello():\n    print("Hello")';
      appStore.updateCode(newCode);
      expect(appStore.currentCode()).toBe(newCode);
    });
  });

  describe('projectPath', () => {
    it('should be null initially', () => {
      expect(appStore.projectPath()).toBeNull();
    });

    it('should load project path', () => {
      const path = '/Users/test/project';
      appStore.loadProject(path);
      expect(appStore.projectPath()).toBe(path);
    });
  });

  describe('isGenerating', () => {
    it('should be false initially', () => {
      expect(appStore.isGenerating()).toBe(false);
    });

    it('should toggle generating state', () => {
      appStore.setIsGenerating(true);
      expect(appStore.isGenerating()).toBe(true);
      appStore.setIsGenerating(false);
      expect(appStore.isGenerating()).toBe(false);
    });
  });

  describe('panel widths', () => {
    it('should have initial widths', () => {
      expect(appStore.chatWidth()).toBe(45);
      expect(appStore.codeWidth()).toBe(25);
      expect(appStore.previewWidth()).toBe(15);
    });

    it('should update panel widths', () => {
      appStore.setChatWidth(40);
      appStore.setCodeWidth(30);
      appStore.setPreviewWidth(15);
      
      expect(appStore.chatWidth()).toBe(40);
      expect(appStore.codeWidth()).toBe(30);
      expect(appStore.previewWidth()).toBe(15);
    });
  });
});
