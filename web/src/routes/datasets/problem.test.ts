import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import { datasetProblem } from './problem';

describe('datasetProblem', () => {
  it('tells a missing folder from one that cannot be read, and blames neither on the connection', () => {
    expect(datasetProblem(new ApiError(404, 'path_not_found', "the folder of dataset 'a' is gone"))).toBe('missing');
    expect(datasetProblem(new ApiError(400, 'invalid_dataset', 'data.yaml is malformed'))).toBe('unreadable');
    expect(datasetProblem(new ApiError(404, 'dataset_not_found', 'no dataset'))).toBeUndefined();
    expect(datasetProblem(new TypeError('Failed to fetch'))).toBeUndefined();
  });
});
