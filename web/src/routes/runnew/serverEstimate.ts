// The server's estimate of a run (POST /api/runs/estimate), asked again 300 ms after the settings stop changing.
import { effect, signal, untrack, type ReadSignal } from 'fw';
import { hub } from '$lib/api/index';
import type { RunEstimate, TrainConfig } from '$lib/api/types';
import { withoutInexactSeed } from '$lib/seed';

/**
 * The server's estimate (POST /api/runs/estimate, feature `run_estimate`): the first asked at once, every later one
 * 300 ms after the settings stop changing, whether or not the last one was answered (a slow or failing server is not
 * asked once per keystroke); the last answer stays up meanwhile. Its errors are the form's own (a missing device,
 * weights of another size), already said at their fields: the reference figures stand in. `possible` says whether
 * these settings can be asked about at all.
 */
export function serverEstimate(config: ReadSignal<TrainConfig>, estKey: ReadSignal<string>, possible: () => boolean) {
  const served = signal<{ key: string; est: RunEstimate } | null>(null);
  /** Whether the latest request is still out: set by the run of the effect that owns it, cleared by every other. */
  const estimating = signal(false);
  /** Whether an estimate has been asked for yet (not state: it only sets the next one's delay). */
  let asked = false;
  /** The latest request's number: an answer to an older (cancelled) one never touches the state. */
  let estSeq = 0;
  // The name is part of the request: the server's command (`cli`) names the run.
  effect(() => {
    const key = estKey.get();
    const c = config.get();
    // A Seed field that holds no seed is left out (lib/seed, rule B): the estimate does not depend on it.
    const body = withoutInexactSeed({ ...c, name: c.name || 'estimate' });
    const seq = ++estSeq;
    // No request for these settings (none possible, or the answer is already in): none is pending either, even where
    // the one this replaced (since cancelled) was.
    if (!possible()) {
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
            served.set({ key, est: e });
            estimating.set(false);
          },
          () => {
            if (ctrl.signal.aborted || seq !== estSeq) return;
            served.set(null);
            estimating.set(false);
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
