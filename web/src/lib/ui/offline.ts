// The server is unreachable: a request failed because nothing answered (not an error the server reported).
// Screens use it to say so once (ErrorState) and to disable the actions that need the server, with OFFLINE_REASON.

import { ApiError } from '$lib/api/client';

/** True when `error` means nothing answered at the server URL. */
export const isOffline = (error: unknown): boolean => error instanceof ApiError && error.code === 'network';

/** Why an action that needs the server is disabled: its tooltip and accessible description (ui/Gate). */
export const OFFLINE_REASON = 'Server offline';
