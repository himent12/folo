import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import { errorDetail, humanHint } from './toast';

// The server's own hints (crates/folo-server/src): what a user can do stays, what is written for agents goes.
describe('humanHint (PR #287 review)', () => {
  it('keeps the remedy of a hint that also names an endpoint', () => {
    expect(humanHint('label images first: POST /api/label/jobs/{id}/start, or draw boxes by hand')).toBe('label images first, or draw boxes by hand');
    expect(humanHint('stop it first: POST /api/runs/r1/stop')).toBe('stop it first');
    expect(humanHint('stop it first (POST /api/runs/r1/stop)')).toBe('stop it first');
    expect(humanHint('nothing to stop: the run has finished; run_status (GET /api/runs/{id}) has its result')).toBe('nothing to stop: the run has finished');
    expect(humanHint('it may have expired (an hour without a part) or been completed; start a new one with POST /api/uploads')).toBe(
      'it may have expired (an hour without a part) or been completed',
    );
    expect(humanHint('cloud compute needs a server with the `cloud` feature (GET /api/system lists its features)')).toBe(
      'cloud compute needs a server with the `cloud` feature',
    );
  });

  it('drops a hint that is only for agents, and keeps one that is not', () => {
    expect(humanHint('list_models (MCP) or GET /api/models lists them')).toBeNull();
    expect(humanHint('GET /api/cloud/connections (`folo cloud connections`, MCP cloud_connections) lists them')).toBeNull();
    expect(humanHint('see docs/API.md for the endpoints')).toBeNull();
    expect(humanHint('stop it first')).toBe('stop it first');
  });

  it('gives a toast the remedy in a sentence', () => {
    const e = new ApiError(400, 'invalid_dataset', "labeling job 'lb_1' has no labeled image yet", 'label images first: POST /api/label/jobs/lb_1/start, or draw boxes by hand');
    expect(errorDetail(e)).toBe("Labeling job 'lb_1' has no labeled image yet. Label images first, or draw boxes by hand.");
  });

  // PR #287 re-review 1: Settings → Cloud, disconnecting a connection in use lost the whole remedy.
  it('keeps the remedy of a hint whose endpoint is the subject of its instruction', () => {
    const hint = 'stop its runs and wait until GET /api/cloud/instances shows its instances terminated'; // cloud/mod.rs in_use
    const e = new ApiError(409, 'connection_in_use', "connection 'c1' has active run(s) r1", hint);
    expect(errorDetail(e)).toBe("Connection 'c1' has active run(s) r1. Stop its runs and wait until its instances are terminated.");
  });
});
