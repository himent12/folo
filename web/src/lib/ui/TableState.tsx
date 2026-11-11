/**
 * An empty or error state in the frame the data would fill (DESIGN_DIRECTION §2.8): one row of a `.table`'s body
 * spanning every column. The table renders no column heads with it (nothing to head) and the list's toolbar stays
 * above, disabled, so the frame starts where the loaded list's does. The block is compact (the message and its
 * actions with the panel's 16 px padding), never held at the loaded list's height; on phones the frame is the
 * content plus 24 px. The message stays inline at the content's left edge (EmptyState / ErrorState).
 */
import { css } from 'fw';

export interface TableStateProps {
  cols: number;
  children: JSX.Element;
}

export default function TableState(props: TableStateProps) {
  return (
    <tr class="state">
      <td colspan={props.cols}>
        <div class="frame">{props.children}</div>
      </td>
    </tr>
  );
}

css`
  .state td {
    height: auto;
    padding: 0;
    white-space: normal;
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
    box-shadow: none;
  }
  .state:hover {
    background: none;
  }
  /* One padding for every message in the frame, whatever its own default: 16 px on the panel's edges. */
  .frame {
    padding: var(--space-4);
  }
  .state .frame > :global(*) {
    padding: 0;
  }
  @media (max-width: 767px) {
    .frame {
      padding: var(--space-3) var(--space-4);
    }
  }
`;
