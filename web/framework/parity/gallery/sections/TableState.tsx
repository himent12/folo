// Gallery twin of TableState.svelte.
import TableState from '$lib/ui/TableState';

export default function TableStateSection() {
  return (
    <div class="g-stack" data-parity-phone>
      <table class="table">
        <tbody>
          <TableState cols={4}>
            <p>No runs yet. Start one from New run.</p>
          </TableState>
        </tbody>
      </table>
      <table class="table">
        <tbody>
          <TableState cols={2}>
            <div class="g-row">
              <strong>The server did not answer.</strong>
              <span>Retry in a moment.</span>
            </div>
          </TableState>
        </tbody>
      </table>
    </div>
  );
}
