/** mAP50-95 per epoch for one run, as a sparkline. */
import { computed, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import { mapHistory } from '$lib/api/metricsCache';
import type { Run } from '$lib/api/types';
import Sparkline from './Sparkline';

export interface RunSparkProps {
  run: Run;
  width?: number;
  height?: number;
}

export default function RunSpark(props: RunSparkProps) {
  const values = signal<number[]>([]);

  effect(() => {
    let live = true;
    mapHistory(hub.api, props.run).then(
      (v) => {
        if (live) values.set(v);
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });

  // Neutral ink, a live run's end dot lime ("now", design/UNIFIED.md §4 Runs). Failure is stated by the status mark,
  // not repeated here.
  const end = computed(() => (props.run.status === 'running' ? 'var(--accent-line)' : undefined));
  const domain = computed<[number, number]>(() => [0, Math.max(0.1, ...values.get())]);

  return (
    <Sparkline
      values={values.get()}
      width={props.width ?? 96}
      height={props.height ?? 20}
      color="var(--text-tertiary)"
      endColor={end.get()}
      domain={domain.get()}
      length={props.run.epochs}
      fill={false}
    />
  );
}
