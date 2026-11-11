// Whether a run was resumed after it ended, from the server's own log lines (folo-server runs/resume.rs `restart`):
// what makes its duration its training time rather than the wall time from its first start to its end.

import type { LogLine } from '$lib/api/types';

const RESUME_LINES = ['resuming from weights/last.folo', 'no epoch finished before; starting over'];

export const wasResumed = (lines: readonly LogLine[]) => lines.some((l) => RESUME_LINES.includes(l.message));
