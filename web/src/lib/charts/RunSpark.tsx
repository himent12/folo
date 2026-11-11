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

  // Neutral ink; only a live run's line takes the accent. Failure is stated by the status dot, not repeated here.
  const color = computed(() => (props.run.status === 'running' ? 'var(--accent-line)' : 'var(--text-tertiary)'));
  const domain = computed<[number, number]>(() => [0, Math.max(0.1, ...values.get())]);

  return (
    <Sparkline
      values={values.get()}
      width={props.width ?? 88}
      height={props.height ?? 22}
      color={color.get()}
      domain={domain.get()}
      length={props.run.epochs}
      fill={props.run.status === 'running'}
    />
  );
}
