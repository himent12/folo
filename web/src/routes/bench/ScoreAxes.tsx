import { computed, css, For } from 'fw';
import type { BenchAxisScore } from '$lib/api/types';
import { AXIS_NAME } from '$lib/bench/rules';
import { tooltip } from '$lib/ui/tooltip';

/** The five axes in words, for a tooltip and assistive tech: "Accuracy 0.64 · Robustness not measured · …". */
export function axesText(axes: readonly BenchAxisScore[]): string {
  return axes.map((a) => `${AXIS_NAME[a.axis]} ${a.value === null ? 'not measured' : a.value.toFixed(2)}`).join(' · ');
}

/**
 * An entry's five axes at a glance: one small bar per axis in the score's order (accuracy, robustness, self-labeling,
 * calibration, localization), each filled to its value on 0–1 in neutral ink. An axis that was not measured is an
 * empty dashed slot, never a zero bar (a zero is a solid track with nothing in it). Each bar names its axis and value
 * on hover (the leaderboard's header letters and foot key name them at rest); all five are its accessible name.
 */
export default function ScoreAxes(props: { axes: readonly BenchAxisScore[] }) {
  const text = computed(() => axesText(props.axes));

  return (
    <span class="axes" role="img" aria-label={`Axes: ${text.get()}`}>
      <For each={props.axes} key={(a) => a.axis}>
        {(a) => (
          <>
            {a().value === null ? (
              <span class="bar none" data-axis={a().axis} use:tooltip={`${AXIS_NAME[a().axis]}: not measured`}></span>
            ) : (
              <span class="bar" data-axis={a().axis} use:tooltip={`${AXIS_NAME[a().axis]} ${a().value!.toFixed(2)}`}>
                <span class="fill" style:height={`${Math.max(0, Math.min(1, a().value!)) * 100}%`}></span>
              </span>
            )}
          </>
        )}
      </For>
    </span>
  );
}

css`
  .axes {
    display: inline-flex;
    align-items: flex-end;
    gap: 4px;
    height: 20px;
    vertical-align: middle;
  }
  .bar {
    position: relative;
    width: 10px;
    height: 20px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--text-secondary);
  }
  /* Not measured: no track to fill, a dashed outline of the slot. */
  .bar.none {
    background: none;
    box-shadow: none;
    border: 1px dashed var(--line-strong);
  }
`;
