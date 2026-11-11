<!-- Gallery twin of ClassChips.tsx. -->
<script lang="ts">
  import ClassChips from '$lib/ui/ClassChips.svelte';
  import Field from '$lib/ui/Field.svelte';

  let a = $state(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']);
  let b = $state<string[]>([]);
  let c = $state(['cat']);
  let d = $state(['dog']);
  let draft = $state('half typed');
</script>

<div class="g-stack g-medium" data-parity-coarse>
  <ClassChips id="cc-a" bind:classes={a} max={80} placeholder="person, car, dog" />
  <ClassChips id="cc-b" bind:classes={b} max={80} placeholder="person, car, dog" />
  <ClassChips id="cc-c" bind:classes={c} bind:draft max={80} placeholder="person, car, dog" invalid />
  <Field label="Classes" for="cc-d" hint="Type a name, then Enter.">
    <span data-parity-step="type" data-parity-do="type:bird, fish," data-parity-target="input">
      <ClassChips id="cc-d" bind:classes={d} max={3} placeholder="person, car, dog" />
    </span>
  </Field>
  <span data-parity-step="remove" data-parity-target=":document button[aria-label='Remove bicycle']"></span>
</div>
