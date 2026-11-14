/**
 * A sortable column header for a `.table`: the label is a button (Enter / Space sort), the mark beside it shows the
 * direction, and the cell carries `aria-sort` while it is the sorted column. A sortable head always shows a chevron
 * (design/UNIFIED.md §2, Tables): a faint up-down pair while another column is sorted, the direction's chevron in ink on
 * the sorted one, so labels never shift when the sort moves. Right-aligned (number) columns put the mark before the
 * label, so the label stays flush with the numbers under it.
 */
import ChevronDown from '$icons/chevron-down';
import ChevronUp from '$icons/chevron-up';
import ChevronsUpDown from '$icons/chevrons-up-down';
import { css, defaults } from 'fw';

export interface SortHeaderProps {
  label: string;
  /** This column's sort direction, or null when another column is sorted. */
  dir?: 'asc' | 'desc' | null;
  onsort: () => void;
  align?: 'left' | 'right';
  disabled?: boolean;
  /** The column's fixed width (DESIGN_DIRECTION 7: one geometry in every state); the name column leaves it out. */
  width?: string;
  class?: string;
}

export default function SortHeader(props: SortHeaderProps) {
  const p = defaults(props, { dir: null, align: 'left', disabled: false, class: '' });
  return (
    <th
      scope="col"
      class={`sortable ${p.class}`}
      class:r={p.align === 'right'}
      style:width={p.width}
      aria-sort={p.dir ? (p.dir === 'asc' ? 'ascending' : 'descending') : undefined}
    >
      <button type="button" class="sort" class:on={!!p.dir} disabled={p.disabled} onclick={() => p.onsort()}>
        <span>{p.label}</span>
        <span class="mark" aria-hidden="true">
          {p.dir === 'asc' ? <ChevronUp size={12} strokeWidth={2} /> : p.dir === 'desc' ? <ChevronDown size={12} strokeWidth={2} /> : <ChevronsUpDown size={12} strokeWidth={2} />}
        </span>
      </button>
    </th>
  );
}

css`
  th.sortable {
    padding: 0;
  }
  .sort {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    width: 100%;
    height: var(--size-row);
    padding: 0 var(--space-3);
    color: inherit;
    font-weight: inherit;
    white-space: nowrap;
    transition: color var(--dur-fast) var(--ease-standard);
  }
  th.r .sort {
    flex-direction: row-reverse;
    justify-content: flex-start;
  }
  th:first-child .sort {
    padding-left: var(--space-4);
  }
  th:last-child .sort {
    padding-right: var(--space-4);
  }
  .mark {
    display: flex;
    color: var(--text-tertiary);
    transition: color var(--dur-fast) var(--ease-standard);
  }
  .sort:not(:disabled):hover,
  .sort.on {
    color: var(--text-primary);
  }
  .sort:not(:disabled):hover .mark,
  .sort:focus-visible .mark,
  .sort.on .mark {
    color: inherit;
  }
  .sort:disabled {
    cursor: default;
  }
  .sort:focus-visible {
    outline-offset: calc(-1 * var(--size-focus-ring));
  }
`;
