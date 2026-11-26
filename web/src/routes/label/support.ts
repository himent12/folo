// Whether the server labels images (docs/API.md, Agent labeling): it lists `agent_labeling` in `/api/system`. An
// older server lists no such feature; `folo serve --mock` neither, and answers the endpoints `501 not_implemented`.

import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { hasFeature } from '$lib/api/queue';
import type { SystemInfo } from '$lib/api/types';
import { LABEL_FEATURE } from '$lib/label';

/** True when the server says it cannot label: no feature in a system answer, or a label call's 501. */
export function cannotLabel(system: SystemInfo | undefined, error?: unknown): boolean {
  if (error instanceof ApiError && error.status === 501 && error.code === 'not_implemented') return true;
  return !!system && !hasFeature(system, LABEL_FEATURE);
}

/** True once health features (`hub.lacks`), `/api/system`, or a 501 response shows agent labeling is unavailable. */
export function labelUnavailable(system: SystemInfo | undefined, error?: unknown): boolean {
  return hub.lacks(LABEL_FEATURE) || cannotLabel(system, error);
}
