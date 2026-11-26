import { describe, expect, it } from 'vitest';
import { isChunkEvalError, isChunkFetchError } from './chunkError';

describe('chunkError', () => {
  it.each([
    ['Chromium dynamic import network failure', new TypeError('Failed to fetch dynamically imported module: https://studio/assets/mock.js')],
    ['Firefox dynamic import network failure', new TypeError('error loading dynamically imported module: https://studio/assets/Runs.js')],
    ['Safari dynamic import network failure', new TypeError('Importing a module script failed.')],
    ['Safari generic fetch failure', new TypeError('Load failed')],
    ['Generic network error', new TypeError('NetworkError when attempting to fetch resource.')],
    ['Vite CSS preload failure', new Error('Unable to preload CSS for /assets/Overview-abc.css')],
    ['ChunkLoadError by error name', Object.assign(new Error('Loading chunk 42 failed'), { name: 'ChunkLoadError' })],
    ['Wrapped fetch error via cause', new Error('wrapper', { cause: new TypeError('Failed to fetch dynamically imported module') })],
    ['AbortError', new DOMException('Aborted', 'AbortError')],
  ])('treats %s as a retryable fetch error', (_, err) => {
    expect(isChunkFetchError(err)).toBe(true);
    expect(isChunkEvalError(err)).toBe(false);
  });

  it.each([
    ['SyntaxError during module evaluation', new SyntaxError('Unexpected token <')],
    ['ReferenceError during module evaluation', new ReferenceError('foo is not defined')],
    ['Runtime TypeError inside module body', new TypeError("Cannot read properties of undefined (reading 'x')")],
    ['Plain Error thrown at module top level', new Error('mock init exploded')],
    ['Non-Error thrown value', 'bad module'],
  ])('treats %s as a non-retryable evaluation error', (_, err) => {
    expect(isChunkFetchError(err)).toBe(false);
    expect(isChunkEvalError(err)).toBe(true);
  });
});
