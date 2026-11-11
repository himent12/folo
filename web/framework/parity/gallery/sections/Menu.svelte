<!-- Gallery twin of Menu.tsx. -->
<script lang="ts">
  import Plus from '@lucide/svelte/icons/plus';
  import Copy from '@lucide/svelte/icons/copy';
  import Menu from '$lib/ui/Menu.svelte';
</script>

<div class="g-stack" data-parity-phone>
  <div class="g-row" data-parity-step="open" data-parity-target="button[aria-haspopup='menu']">
    <Menu
      label="More run actions"
      items={[
        { label: 'Delete run', danger: true, onclick: () => {} },
        { label: 'Open model', href: '#/models', icon: Copy },
        { label: 'Duplicate', icon: Plus, onclick: () => {} },
      ]}
    />
  </div>
  <div class="g-row" data-parity-step="ghost" data-parity-target="button[aria-haspopup='menu']">
    <Menu label="Row actions" variant="ghost" items={[{ label: 'Compare', onclick: () => {} }, { label: 'Remove from queue', danger: true, onclick: () => {} }]} />
  </div>
  <div class="g-row g-end" data-parity-step="fixed" data-parity-target="button[aria-haspopup='menu']">
    <Menu label="Fixed actions" fixed items={[{ label: 'Open', href: '#/x' }, { label: 'Rename', onclick: () => {} }]} />
  </div>
  <div class="g-row" data-parity-step="keys" data-parity-do="press:ArrowUp" data-parity-target="button[aria-haspopup='menu']">
    <Menu label="Keyboard actions" items={[{ label: 'First', onclick: () => {} }, { label: 'Second', onclick: () => {} }, { label: 'Last', onclick: () => {} }]} />
  </div>
</div>
