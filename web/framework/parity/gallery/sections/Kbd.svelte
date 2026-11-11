<!-- Gallery twin of Kbd.tsx. -->
<script lang="ts">
  import Kbd from '$lib/ui/Kbd.svelte';
</script>

<div class="g-stack">
  <div class="g-row">
    <Kbd keys="Mod+K" />
    <Kbd keys="G R" />
    <Kbd keys="?" />
    <Kbd keys="Shift+Enter" />
    <Kbd keys="Mod+Shift+P" />
    <Kbd keys="ArrowUp ArrowDown" />
    <Kbd keys="Escape" />
  </div>
  <div class="g-row">
    <Kbd keys="Mod+K" subtle />
    <Kbd keys="J" subtle />
    <Kbd keys="N" off />
    <Kbd keys="G D" off />
  </div>
  <p>Press <Kbd keys="Mod+V" /> to paste, then <span class="nowrap"><Kbd keys="Enter" />.</span></p>
</div>
