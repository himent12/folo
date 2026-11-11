/**
 * The in-place definition of a chart's ⓘ on phones (./infoSlot): rendered by the frame under the chart's heading,
 * empty (and taking no room) until a `chart` InfoButton opens it.
 */
import { computed, css } from 'fw';
import type { InfoSlot } from './infoSlot';

export interface InfoRegionProps {
  slot: InfoSlot;
}

export default function InfoRegion(props: InfoRegionProps) {
  // The open button's snippet, by identity: the region re-renders when another snippet comes, not on every write.
  const children = computed(() => props.slot.content?.children);
  return (
    <div class="info-region" id={props.slot.id} hidden={!props.slot.content}>
      {children.get() ? children.get() : props.slot.content ? props.slot.content.text : null}
    </div>
  );
}

css`
  /* A raised well at the frame's inset: the definition reads as part of the chart, pushing the readout down. */
  .info-region {
    margin: 0 var(--info-inset, var(--space-4)) var(--info-after, var(--space-3));
    padding: var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-raised);
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-secondary);
  }
  .info-region[hidden] {
    display: none;
  }
  .info-region :global(b),
  .info-region :global(dt) {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
`;
