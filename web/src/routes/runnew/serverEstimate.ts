// The server's estimate of a run (POST /api/runs/estimate), asked again 300 ms after the settings stop changing.
import { effect, signal, untrack, type ReadSignal, type Signal } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import type { RunEstimate, TrainConfig } from '$lib/api/types';
import { withoutInexactSeed } from '$lib/seed';
import { quote } from '$lib/trainCli';

/** Serializes the estimate-relevant fields of `config` (omitting `name`, which never affects time or memory). */
export function estimateKey(config: TrainConfig): string {
  const { name: _name, ...rest } = config;
  return JSON.stringify(rest);
}

/** Formats a `--name` argument the same way `trainCommand` does (attaching `-`-prefixed values). */
function formatNameFlag(name: string): string {
  if (name.startsWith('-')) return quote(`--name=${name}`);
  return `--name ${quote(name)}`;
}

/**
 * Updates the `--name` flag in a server-built CLI command when the user edits the run name after the estimate arrived,
 * avoiding a redundant `POST /api/runs/estimate` roundtrip.
 */
export function withServerCliName(cli: string, sentName: string, currentName: string): string {
  const want = currentName.trim();
  const sent = sentName.trim();
  if (want === sent) return cli;
  const oldFlag = sent ? formatNameFlag(sent) : null;
  const newFlag = want ? formatNameFlag(want) : '';
  if (oldFlag) {
    const idx = cli.lastIndexOf(oldFlag);
    if (idx !== -1 && (idx === 0 || cli[idx - 1] === ' ') && (idx + oldFlag.length === cli.length || cli[idx + oldFlag.length] === ' ')) {
      const before = cli.slice(0, idx).trimEnd();
      const after = cli.slice(idx + oldFlag.length).trimStart();
      return [before, newFlag, after].filter(Boolean).join(' ');
    }
  }
  return newFlag ? `${cli} ${newFlag}` : cli;
}

/**
 * The server's estimate (POST /api/runs/estimate, feature `run_estimate`): the first asked at once, every later one
 * 300 ms after the settings stop changing, whether or not the last one was answered (a slow or failing server is not
 * asked once per keystroke); the last answer stays up meanwhile. Its errors are the form's own (a missing device,
 * weights of another size), already said at their fields: the reference figures stand in. `possible` says whether
 * these settings can be asked about at all. The one error no field has said yet is mixed precision on a device the
 * capabilities do not describe (a second GPU, a restarted server): `ampRefusal` holds it until the next estimate ends any other way.
 */
export function serverEstimate(
  config: ReadSignal<TrainConfig>,
  estKey: ReadSignal<string>,
  possible: () => boolean,
  ampRefusal: Signal<{ device: string; message: string } | null>,
) {
  const served = signal<{ key: string; est: RunEstimate; name: string } | null>(null);
  /** Whether the latest request is still out: set by the run of the effect that owns it, cleared by every other. */
  const estimating = signal(false);
  /** Whether an estimate has been asked for yet (not state: it only sets the next one's delay). */
  let asked = false;
  /** The latest request's number: an answer to an older (cancelled) one never touches the state. */
  let estSeq = 0;
  // Track only `estKey` (which excludes `name`) and `possible()`, so typing in the Run name field never triggers a
  // redundant `/api/runs/estimate` request.
  effect(() => {
    const key = estKey.get();
    const canAsk = possible();
    const c = untrack(() => config.get());
    // A Seed field that holds no seed is left out (lib/seed, rule B): the estimate does not depend on it.
    const body = withoutInexactSeed({ ...c, name: c.name || 'estimate' });
    const seq = ++estSeq;
    // No request for these settings (none possible, or the answer is already in): none is pending either, even where
    // the one this replaced (since cancelled) was.
    if (!canAsk) {
      served.set(null);
      estimating.set(false);
      return;
    }
    if (untrack(() => served.get()?.key) === key) {
      estimating.set(false);
      return;
    }
    const ctrl = new AbortController();
    estimating.set(true);
    const t = setTimeout(
      () => {
        asked = true;
        hub.api.estimateRun(body, ctrl.signal).then(
          (e) => {
            if (ctrl.signal.aborted || seq !== estSeq) return;
            served.set({ key, est: e, name: body.name });
            estimating.set(false);
            ampRefusal.set(null);
          },
          (err) => {
            if (ctrl.signal.aborted || seq !== estSeq) return;
            served.set(null);
            estimating.set(false);
            // A cloud run's mixed precision is the worker's to check, so this is never about it. Any other failure (the
            // server restarting) says nothing about mixed precision: Start asks the server itself.
            const refused = err instanceof ApiError && err.code === 'not_supported' && body.amp && !body.compute;
            ampRefusal.set(refused ? { device: body.device, message: err.message } : null);
          },
        );
      },
      asked ? 300 : 0,
    );
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  });
  return { served, estimating };
}
