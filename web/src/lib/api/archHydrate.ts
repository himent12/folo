// Bounded, one-shot canonical `folo2-*` architecture hydration for `HttpApi` list and detail endpoints.
// Decouples shared in-flight probes from any single caller's `AbortSignal`, enforces a hydration deadline so
// stalled probes never block primary responses, marks deterministic 4xx/501 errors as checked so 2 s polls never
// loop, and skips all probes once `health()` or `system()` reports a server without `folo2_arch`.
import type { Arch, CliCommand, Model, Run } from './types';
import { ApiError } from './client';
import { archFromCli, isFolo2Arch, supportsFolo2 } from '$lib/arch';

const MAX_ARCH_HYDRATE = 24;
const ARCH_HYDRATE_CONCURRENCY = 6;
export const ARCH_HYDRATE_DEADLINE_MS = 1500;
const ARCH_PROBE_TIMEOUT_MS = 2500;

function isTerminalProbeError(err: unknown): boolean {
  return err instanceof ApiError && ((err.status >= 400 && err.status < 500) || err.status === 501);
}

/** Races hydration work against `deadlineMs` and a caller `AbortSignal` (rethrows `AbortError` if caller aborts). */
export function awaitHydration(work: Promise<unknown>, deadlineMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
  return new Promise((resolve, reject) => {
    let settled = false;
    const done = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      fn();
    };
    const onAbort = () => done(() => reject(new DOMException('Aborted', 'AbortError')));
    const timer = setTimeout(() => done(resolve), deadlineMs);
    signal?.addEventListener('abort', onAbort, { once: true });
    work.then(() => done(resolve), () => done(resolve));
  });
}

export interface ArchProbeClient {
  model(id: string, signal?: AbortSignal): Promise<Model>;
  runCli(id: string, signal?: AbortSignal): Promise<CliCommand>;
}

export class ArchHydrator {
  /** Whether the server advertises `folo2_arch` (`null` until `health()` or `system()` resolves). */
  #folo2Supported: boolean | null = null;
  #runArchs = new Map<string, Arch>();
  #modelArchs = new Map<string, Arch>();
  #checkedModelProbes = new Set<string>();
  #checkedRunCli = new Set<string>();
  #hydratedModels = new Set<string>();
  #pendingModelProbes = new Map<string, Promise<boolean>>();
  #pendingRunCli = new Map<string, Promise<boolean>>();
  #pendingModels = new Map<string, Promise<void>>();
  #pendingRuns = new Map<string, Promise<void>>();
  #client: ArchProbeClient;

  constructor(client: ArchProbeClient) {
    this.#client = client;
  }

  noteCapabilities(info: { features?: string[] } | null | undefined) {
    if (info && typeof info === 'object') this.#folo2Supported = supportsFolo2(info);
  }

  recordRun(id: string, arch: Arch) {
    if (isFolo2Arch(arch)) this.#runArchs.set(id, arch);
  }

  recordModelProbe(m: Model, headerArch: Arch | null): Model {
    if (m?.id) this.#checkedModelProbes.add(m.id);
    if (headerArch && isFolo2Arch(headerArch)) {
      this.#modelArchs.set(m.id, headerArch);
      this.#hydratedModels.add(m.id);
      if (m.run_id) this.#runArchs.set(m.run_id, headerArch);
      return headerArch !== m.arch ? { ...m, arch: headerArch } : m;
    }
    return this.promoteModel(m);
  }

  recordRunCli(id: string, cmd: CliCommand) {
    this.#checkedRunCli.add(id);
    const cliArch = archFromCli(cmd);
    if (cliArch && isFolo2Arch(cliArch)) this.#runArchs.set(id, cliArch);
  }

  promoteRun(r: Run): Run {
    if (!r?.config) return r;
    const known = this.#runArchs.get(r.id);
    if (known && known !== r.config.arch) return { ...r, config: { ...r.config, arch: known } };
    if (isFolo2Arch(r.config.arch)) this.#runArchs.set(r.id, r.config.arch);
    return r;
  }

  promoteModel(m: Model): Model {
    if (!m) return m;
    const known = this.#modelArchs.get(m.id) ?? (m.run_id ? this.#runArchs.get(m.run_id) : undefined);
    if (known && known !== m.arch) return { ...m, arch: known };
    if (isFolo2Arch(m.arch)) this.#modelArchs.set(m.id, m.arch);
    return m;
  }

  #probeModel(id: string): Promise<boolean> {
    if (this.#checkedModelProbes.has(id)) return Promise.resolve(true);
    const existing = this.#pendingModelProbes.get(id);
    if (existing) return existing;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), ARCH_PROBE_TIMEOUT_MS);
    const p = this.#client
      .model(id, ctrl.signal)
      .then(
        () => true,
        (err) => {
          if (isTerminalProbeError(err)) {
            this.#checkedModelProbes.add(id);
            return true;
          }
          return false;
        },
      )
      .finally(() => {
        clearTimeout(timer);
        this.#pendingModelProbes.delete(id);
      });
    this.#pendingModelProbes.set(id, p);
    return p;
  }

  #probeRunCli(id: string): Promise<boolean> {
    if (this.#checkedRunCli.has(id)) return Promise.resolve(true);
    const existing = this.#pendingRunCli.get(id);
    if (existing) return existing;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), ARCH_PROBE_TIMEOUT_MS);
    const p = this.#client
      .runCli(id, ctrl.signal)
      .then(
        () => true,
        (err) => {
          if (isTerminalProbeError(err)) {
            this.#checkedRunCli.add(id);
            return true;
          }
          return false;
        },
      )
      .finally(() => {
        clearTimeout(timer);
        this.#pendingRunCli.delete(id);
      });
    this.#pendingRunCli.set(id, p);
    return p;
  }

  #hydrateModel(m: Model): Promise<void> {
    if (!m?.id || isFolo2Arch(m.arch) || this.#modelArchs.has(m.id) || (m.run_id && this.#runArchs.has(m.run_id)) || this.#hydratedModels.has(m.id)) {
      return Promise.resolve();
    }
    const existing = this.#pendingModels.get(m.id);
    if (existing) return existing;
    const p = (async () => {
      const mOk = await this.#probeModel(m.id);
      if (!this.#modelArchs.has(m.id) && m.run_id && !this.#runArchs.has(m.run_id)) {
        const cliOk = await this.#probeRunCli(m.run_id);
        if (mOk && cliOk) this.#hydratedModels.add(m.id);
      } else if (mOk) {
        this.#hydratedModels.add(m.id);
      }
    })().finally(() => {
      this.#pendingModels.delete(m.id);
    });
    this.#pendingModels.set(m.id, p);
    return p;
  }

  #hydrateRun(r: Run): Promise<void> {
    if (!r?.id || !r?.config || isFolo2Arch(r.config.arch) || this.#runArchs.has(r.id)) return Promise.resolve();
    const existing = this.#pendingRuns.get(r.id);
    if (existing) return existing;
    const needModel = !!r.model_id && !this.#checkedModelProbes.has(r.model_id);
    const needCli = !this.#checkedRunCli.has(r.id);
    if (!needModel && !needCli) return Promise.resolve();
    const p = (async () => {
      if (needModel && r.model_id) {
        await this.#probeModel(r.model_id);
        const mArch = this.#modelArchs.get(r.model_id);
        if (mArch) this.#runArchs.set(r.id, mArch);
      }
      if (!this.#runArchs.has(r.id) && needCli) {
        await this.#probeRunCli(r.id);
      }
    })().finally(() => {
      this.#pendingRuns.delete(r.id);
    });
    this.#pendingRuns.set(r.id, p);
    return p;
  }

  async #hydrateList<T>(items: readonly T[], task: (item: T) => Promise<void>, signal?: AbortSignal) {
    if (this.#folo2Supported === false || !items.length) return;
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const slice = items.slice(0, MAX_ARCH_HYDRATE);
    let idx = 0;
    const work = Promise.all(
      Array.from({ length: Math.min(ARCH_HYDRATE_CONCURRENCY, slice.length) }, async () => {
        while (idx < slice.length && !signal?.aborted) {
          await task(slice[idx++]!);
        }
      }),
    );
    await awaitHydration(work, ARCH_HYDRATE_DEADLINE_MS, signal);
  }

  async hydrateModels(list: Model[], signal?: AbortSignal): Promise<Model[]> {
    const unchecked = list.filter(
      (m) =>
        m?.id &&
        !isFolo2Arch(m.arch) &&
        !this.#modelArchs.has(m.id) &&
        !(m.run_id && this.#runArchs.has(m.run_id)) &&
        (!this.#hydratedModels.has(m.id) || this.#pendingModels.has(m.id)),
    );
    await this.#hydrateList(unchecked, (m) => this.#hydrateModel(m), signal);
    return list.map((m) => this.promoteModel(m));
  }

  async hydrateRuns(list: Run[], signal?: AbortSignal): Promise<Run[]> {
    const unchecked = list.filter(
      (r) =>
        r?.id &&
        r?.config &&
        !isFolo2Arch(r.config.arch) &&
        !this.#runArchs.has(r.id) &&
        (!this.#checkedRunCli.has(r.id) || (!!r.model_id && !this.#checkedModelProbes.has(r.model_id)) || this.#pendingRuns.has(r.id)),
    );
    await this.#hydrateList(unchecked, (r) => this.#hydrateRun(r), signal);
    return list.map((r) => this.promoteRun(r));
  }

  async hydrateRun(r: Run, signal?: AbortSignal): Promise<Run> {
    if (this.#folo2Supported !== false) await awaitHydration(this.#hydrateRun(r), ARCH_HYDRATE_DEADLINE_MS, signal);
    return this.promoteRun(r);
  }
}
