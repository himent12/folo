import { describe, expect, it } from 'vitest';
import type { SystemInfo } from '$lib/api/types';
import { canonicalDevice, missingDevice } from './device';

const system = { gpus: [{ ordinal: 0, name: 'GPU', backend: 'vulkan', memory_bytes: 0, util: null }] } as unknown as SystemInfo;

describe('canonicalDevice', () => {
  it('names every spelling the trainer accepts as the select does', () => {
    expect(['gpu:00', 'GPU:0', ' cuda:0 ', 'gpu', 'CUDA', 'gpu:+0'].map(canonicalDevice)).toEqual(Array(6).fill('gpu:0'));
    expect(canonicalDevice('cuda:12')).toBe('gpu:12');
    expect(canonicalDevice(' CPU ')).toBe('cpu');
    expect(canonicalDevice('')).toBe('auto');
    expect(canonicalDevice('Auto')).toBe('auto');
    // Not a device the trainer reads: kept, for the server to say what is wrong.
    expect(canonicalDevice('gpu:x')).toBe('gpu:x');
  });
});

describe('missingDevice', () => {
  it('compares GPUs by ordinal, not by spelling', () => {
    expect(missingDevice('gpu:00', system)).toBeNull();
    expect(missingDevice('GPU:0', system)).toBeNull();
    expect(missingDevice('gpu:1', system)).toEqual({ device: 'gpu:1', gone: true, detail: '' });
    expect(missingDevice('gpu:1', undefined)).toBeNull();
  });
});
