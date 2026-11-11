// Gallery twin of ChartConfusion.svelte: ConfusionMatrix, fitting and paging (charts/ConfusionMatrix).
import ConfusionMatrix from '$lib/charts/ConfusionMatrix';
import { cm3, cm9 } from '../chartData';

export default function ChartConfusionSection() {
  return (
    <div class="g-stack" data-parity-phone>
      <div class="g-medium" data-parity-step="cell" data-parity-do="focus" data-parity-target="[data-r='1'][data-c='2']">
        <ConfusionMatrix classes={cm3.classes} matrix={cm3.matrix} scale />
      </div>
      <div class="g-narrow" data-parity-step="paged" data-parity-do="focus" data-parity-target="[data-r='2'][data-c='0']">
        <ConfusionMatrix classes={cm9.classes} matrix={cm9.matrix} label="Nine classes" minCell={32} maxCell={48} />
      </div>
    </div>
  );
}
