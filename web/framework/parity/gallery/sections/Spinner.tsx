// Gallery twin of Spinner.svelte.
import Spinner from '$lib/ui/Spinner';

export default function SpinnerSection() {
  return (
    <div class="g-row">
      <Spinner />
      <Spinner size={16} />
      <Spinner size={24} label="Loading runs" />
      <span style="color: var(--accent-text)"><Spinner size={32} /></span>
    </div>
  );
}
