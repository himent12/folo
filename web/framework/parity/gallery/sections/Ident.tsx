// Gallery twin of Ident.svelte.
import Ident from '$lib/ui/Ident';

export default function IdentSection() {
  return (
    <div class="g-row">
      <Ident value="folo-s" />
      <Ident value="r_7d1e9a" title="Run r_7d1e9a" />
      <Ident value="folo-s-coco80" href="#/models/m_1" />
      <Ident value="best.folo" class="extra" title="C:/runs/best.folo" />
      <span>
        Started from <Ident value="folo-n" href="#/models/folo-n" class="lineage-link" /> today
      </span>
    </div>
  );
}
