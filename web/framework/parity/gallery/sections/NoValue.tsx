// Gallery twin of NoValue.svelte.
import NoValue from '$lib/ui/NoValue';

export default function NoValueSection() {
  return (
    <div class="g-row">
      <span>mAP50-95 <NoValue /></span>
      <span class="g-title">
        <NoValue label="no loss yet" />
      </span>
      <span class="mono">0.535 / <NoValue /></span>
    </div>
  );
}
