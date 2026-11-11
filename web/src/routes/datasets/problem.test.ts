import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import { datasetProblem, isUnlabeled } from './problem';

describe('datasetProblem', () => {
  it('tells a missing folder from one that cannot be read, and blames neither on the connection', () => {
    expect(datasetProblem(new ApiError(404, 'path_not_found', "the folder of dataset 'a' is gone"))).toBe('missing');
    expect(datasetProblem(new ApiError(400, 'invalid_dataset', 'data.yaml is malformed'))).toBe('unreadable');
    expect(datasetProblem(new ApiError(404, 'dataset_not_found', 'no dataset'))).toBeUndefined();
    expect(datasetProblem(new TypeError('Failed to fetch'))).toBeUndefined();
  });
});

describe('isUnlabeled', () => {
  it('knows a folder of images without labels from one with nothing in it', () => {
    expect(isUnlabeled(new ApiError(400, 'unknown_format', "'D:/x/images-only' has 2 images but no labels"))).toBe(true);
    expect(isUnlabeled(new ApiError(400, 'unknown_format', 'could not detect a dataset in D:/x: expected a data.yaml, …'))).toBe(false);
  });
});
