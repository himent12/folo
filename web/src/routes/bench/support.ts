// Whether the server has vision benchmarks (docs/API.md, Vision benchmarks): it lists `vision_bench` in `/api/system`.
// An older server lists no such feature (and has no such routes); `folo serve --mock` neither, and answers the
// endpoints `501 not_implemented`.

import { ApiError } from '$lib/api/client';
import { hasFeature } from '$lib/api/queue';
import type { SystemInfo } from '$lib/api/types';
import { BENCH_FEATURE } from '$lib/bench/rules';

/** True when the server says it has no benchmarks: no feature in a system answer, or a benchmark call's 501. */
export function cannotBench(system: SystemInfo | undefined, error?: unknown): boolean {
  if (error instanceof ApiError && error.status === 501 && error.code === 'not_implemented') return true;
  return !!system && !hasFeature(system, BENCH_FEATURE);
}
