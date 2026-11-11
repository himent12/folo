<!-- Gallery twin of CodeText.tsx. -->
<script lang="ts">
  import CodeText from '$lib/ui/CodeText.svelte';
</script>

<div class="g-stack g-narrow" data-parity-phone>
  <p><CodeText text="Plain prose, no code at all." /></p>
  <p><CodeText text="Run `folo train --amp --device cuda:0` to turn on mixed precision." /></p>
  <p><CodeText text="`folo serve` then open `http://127.0.0.1:7878/` in a browser, or `npx folo doctor --json --verbose`." /></p>
  <p><CodeText text="`only code`" /></p>
  <p><CodeText text="" /></p>
</div>
