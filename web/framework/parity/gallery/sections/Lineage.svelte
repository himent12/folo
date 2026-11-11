<!-- Gallery twin of Lineage.tsx. -->
<script lang="ts">
  import { origin } from '$lib/lineage';
  import Lineage from '$lib/ui/Lineage.svelte';

  const models = [{ id: 'm_shapes_n', name: 'shapes-n' }];
</script>

<div class="g-stack">
  <p>From: <Lineage origin={origin(null)} /></p>
  <p>From: <Lineage origin={origin(null)} scratch="scratch weights" /></p>
  <p>From: <Lineage origin={origin('m_shapes_n', models)} /></p>
  <p>From: <Lineage origin={origin('m_shapes_n', models)} link={false} /></p>
  <p>From: <Lineage origin={origin('folo-s-coco80')} /></p>
  <p>From: <Lineage origin={origin('C:/runs/r_7/weights/best.folo')} /></p>
  <p data-parity-step="hover" data-parity-do="hover" data-parity-target="a">Hover: <Lineage origin={origin('m_shapes_n', models)} /></p>
</div>
