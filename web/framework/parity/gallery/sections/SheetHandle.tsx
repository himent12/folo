// Gallery twin of SheetHandle.svelte.
import SheetHandle from '$lib/ui/SheetHandle';

export default function SheetHandleSection() {
  return (
    <div class="g-narrow" style="position: relative; height: 48px; background: var(--bg-overlay); border-radius: 12px 12px 0 0">
      <SheetHandle />
    </div>
  );
}
