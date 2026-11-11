<!-- Gallery twin of CommandLine.tsx. -->
<script lang="ts">
  import CommandLine from '$lib/ui/CommandLine.svelte';
</script>

<div class="g-stack g-medium" data-parity-phone>
  <CommandLine command="folo serve" />
  <CommandLine
    command="$env:WGPU_BACKEND='vulkan'; folo train --data datasets/coco/data.yaml --arch folo-s --epochs 300 --imgsz 640 --amp --device cuda:0"
    label="PowerShell command"
    copied="Paste it in PowerShell."
  />
  <div data-parity-step="focus" data-parity-do="focus" data-parity-target="pre">
    <CommandLine command="folo train --data datasets/shapes/data.yaml --arch folo-n --epochs 30 --batch 16 --workers 4 --seed 7 --project runs" />
  </div>
</div>
