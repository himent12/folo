// The server is unreachable: a request failed because nothing answered (not an error the server reported).
// Screens use it to say so once (ErrorState) and to disable the actions that need the server, with OFFLINE_REASON.

import { ApiError } from '$lib/api/client';

/** True when `error` means nothing answered at the server URL. */
export const isOffline = (error: unknown): boolean => error instanceof ApiError && error.code === 'network';

/** True when `error` means the server answered but refused Studio's connection handshake (e.g. `--token`). */
export const isConnectRefused = (error: unknown): boolean => error instanceof ApiError && error.code === 'connect_refused';

/** True when Studio has no usable server connection (unreachable or refused). */
export const isDisconnected = (error: unknown): boolean => isOffline(error) || isConnectRefused(error);

/** Why an action that needs the server is disabled: its tooltip and accessible description (ui/Gate). */
export const OFFLINE_REASON = 'Server offline';
