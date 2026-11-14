/**
 * The live panel's foot (design/UNIFIED.md §4 Overview): the machine at a glance in one 36 px row of meters, the GPU
 * the run trains on (its load, where the driver reports it), the CPU and the memory, from `/api/system` (polled). A
 * reading the server no longer answers for is dropped, never shown as current; with none, there is no foot.
 */
import { computed, css } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Run } from '$lib/api/types';
import { bytes, deviceName, pct } from '$lib/format';
import { prefs } from '$lib/prefs';
import Progress from '$lib/ui/Progress';
import { runGpu } from './device';

export default function MachineFoot(props: { run: Run }) {
  const sys = new Resource((s) => hub.api.system(s), { poll: 2000 });
  /** Live readings only: once the server stops answering the foot goes, rather than show a stale load. */
  const info = computed(() => (sys.stale === undefined ? sys.data : undefined));
  // The run's own GPU (./device), or the first one when it cannot be told; none for a CPU run.
  const gpu = computed(() => {
    const s = info.get();
    if (!s || props.run.config.device === 'cpu') return null;
    return runGpu(props.run, s.gpus) ?? s.gpus[0] ?? null;
  });
  const mem = computed(() => {
    const m = info.get()?.memory;
    return m && m.total_bytes > 0 ? { frac: m.used_bytes / m.total_bytes, text: `${bytes(m.used_bytes, prefs.units)} / ${bytes(m.total_bytes, prefs.units)}` } : null;
  });

  return (
    <>
      {info.get() ? (
        <footer class="foot" aria-label="Machine">
          {gpu.get() ? (
            <span class="meter">
              {/* Its memory named, so the meter reads as load, not memory use; no meter where the driver has no load. */}
              <span class="m-label truncate">
                {deviceName(gpu.get()!.name)} · {bytes(gpu.get()!.memory_bytes, prefs.units)}
              </span>
              {gpu.get()!.util !== null ? (
                <>
                  <span class="m-bar">
                    <Progress value={gpu.get()!.util!} size="xs" tone={gpu.get()!.util! > 0.95 ? 'warning' : 'neutral'} label="GPU load" />
                  </span>
                  <span class="m-val num">{pct(gpu.get()!.util, 0)} load</span>
                </>
              ) : null}
            </span>
          ) : null}
          <span class="meter">
            <span class="m-label">CPU · {info.get()!.cpu.threads} threads</span>
            <span class="m-bar">
              <Progress value={info.get()!.cpu.util} size="xs" tone="neutral" label="CPU load" />
            </span>
            <span class="m-val num">{pct(info.get()!.cpu.util, 0)}</span>
          </span>
          {mem.get() ? (
            <span class="meter">
              <span class="m-label">Memory</span>
              <span class="m-bar">
                <Progress value={mem.get()!.frac} size="xs" tone="neutral" label="Memory in use" />
              </span>
              <span class="m-val num">{mem.get()!.text}</span>
            </span>
          ) : null}
        </footer>
      ) : null}
    </>
  );
}

css`
  .foot {
    display: flex;
    align-items: center;
    gap: var(--space-6);
    height: 36px;
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    overflow: hidden;
    white-space: nowrap;
  }
  .meter {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .m-label {
    color: var(--text-tertiary);
    min-width: 0;
  }
  .m-bar {
    flex: none;
    width: 48px;
  }
  .m-val {
    color: var(--text-secondary);
  }
  .meter {
    flex: none;
  }
  /* Below 1280 px the meters wrap to a second line rather than run past the panel (the 36 px row is the floor). */
  @media (max-width: 1279px) {
    .foot {
      flex-wrap: wrap;
      height: auto;
      min-height: 36px;
      gap: var(--space-1) var(--space-5);
      padding: var(--space-2) var(--space-4);
      white-space: normal;
    }
    .meter {
      white-space: nowrap;
    }
  }
  /* Phones: the values without their bars, the GPU's name cut first. */
  @media (max-width: 767px) {
    .foot {
      gap: var(--space-1) var(--space-4);
      padding: var(--space-2) var(--space-3);
    }
    .meter:first-child {
      flex: 0 1 auto;
    }
    .m-bar {
      display: none;
    }
  }
`;
