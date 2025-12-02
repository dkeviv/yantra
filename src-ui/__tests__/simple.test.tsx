/**
 * simple.test.tsx
 * 
 * Simple test to verify test setup is working
 */

import { describe, it, expect } from 'vitest';

describe('Simple Tests', () => {
  it('basic math works', () => {
    expect(1 + 1).toBe(2);
  });

  it('localStorage mock is available', () => {
    localStorage.setItem('test', 'value');
    expect(localStorage.getItem('test')).toBe('value');
  });

  it('document is available', () => {
    expect(document).toBeDefined();
    expect(document.documentElement).toBeDefined();
  });
});
