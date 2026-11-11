// Gallery twin of SortHeader.svelte.
import { signal } from 'fw';
import SortHeader from '$lib/ui/SortHeader';

export default function SortHeaderSection() {
  const key = signal<'name' | 'map' | 'epochs'>('map');
  const dir = signal<'asc' | 'desc'>('desc');
  function sort(k: 'name' | 'map' | 'epochs') {
    if (key.get() === k) dir.set(dir.get() === 'asc' ? 'desc' : 'asc');
    else {
      key.set(k);
      dir.set('asc');
    }
  }
  return (
    <div class="g-stack g-wide">
      <div hidden>
        <span data-parity-step="sort" data-parity-target=":document .g-sort-name button"></span>
        <span data-parity-step="sort2" data-parity-target=":document .g-sort-map button"></span>
        <span data-parity-step="hover" data-parity-do="hover" data-parity-target=":document .g-hover-me button"></span>
        <span data-parity-step="focus" data-parity-do="focus" data-parity-target=":document .g-focus-me button"></span>
      </div>
      <table class="table">
        <thead>
          <tr>
            <SortHeader label="Model" dir={key.get() === 'name' ? dir.get() : null} onsort={() => sort('name')} class="g-sort-name" />
            <SortHeader label="mAP50-95" dir={key.get() === 'map' ? dir.get() : null} onsort={() => sort('map')} align="right" width="120px" class="g-sort-map" />
            <SortHeader label="Epochs" dir={key.get() === 'epochs' ? dir.get() : null} onsort={() => sort('epochs')} align="right" width="96px" />
            <SortHeader label="Size" onsort={() => {}} disabled />
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>folo-s coco</td>
            <td class="r">0.535</td>
            <td class="r">60</td>
            <td>18 MB</td>
          </tr>
        </tbody>
      </table>
      <table class="table">
        <thead>
          <tr>
            <SortHeader label="Ascending" dir="asc" onsort={() => {}} />
            <SortHeader label="Descending" dir="desc" onsort={() => {}} align="right" />
            <SortHeader label="Hover" onsort={() => {}} class="g-hover-me" />
            <SortHeader label="Focus" onsort={() => {}} class="g-focus-me" align="right" />
          </tr>
        </thead>
      </table>
    </div>
  );
}
