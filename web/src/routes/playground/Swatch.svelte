<script lang="ts">
  /** A class's colour mark: the detections table, the picked box's line and the class list. */
  let { color }: { color: string } = $props();
</script>

<span class="sw" style:background={color} aria-hidden="true"></span>

<style>
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* Class marks keep the palette hue in both themes; a hairline edge holds pale hues on light surfaces. */
  :global([data-theme='light']) .sw {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
</style>
