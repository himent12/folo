<!-- Gallery twin of SortHeader.tsx. -->
<script lang="ts">
  import SortHeader from '$lib/ui/SortHeader.svelte';

  let key = $state<'name' | 'map' | 'epochs'>('map');
  let dir = $state<'asc' | 'desc'>('desc');
  function sort(k: 'name' | 'map' | 'epochs') {
    if (key === k) dir = dir === 'asc' ? 'desc' : 'asc';
    else {
      key = k;
      dir = 'asc';
    }
  }
</script>

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
        <SortHeader label="Model" dir={key === 'name' ? dir : null} onsort={() => sort('name')} class="g-sort-name" />
        <SortHeader label="mAP50-95" dir={key === 'map' ? dir : null} onsort={() => sort('map')} align="right" width="120px" class="g-sort-map" />
        <SortHeader label="Epochs" dir={key === 'epochs' ? dir : null} onsort={() => sort('epochs')} align="right" width="96px" />
        <SortHeader label="Size" onsort={() => {}} disabled />
      </tr>
    </thead>
    <tbody>
      <tr><td>folo-s coco</td><td class="r">0.535</td><td class="r">60</td><td>18 MB</td></tr>
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
