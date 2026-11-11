<!-- Gallery twin of Field.tsx. -->
<script lang="ts">
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';

  let fixed = $state('');
</script>

{#snippet aside()}<span>Optional</span>{/snippet}

<div class="g-stack g-medium" data-parity-coarse>
  <Field label="Dataset name" for="gf-plain"><Input id="gf-plain" value="coco-small" /></Field>
  <Field label="Epochs" for="gf-hint" hint="How many passes over the training images."><Input id="gf-hint" value="60" /></Field>
  <Field label="Server" for="gf-code" hint="Start it with `folo serve --port 7878` on the machine with the GPU."
    ><Input id="gf-code" value="http://127.0.0.1:7878" /></Field
  >
  <Field label="Batch size" for="gf-aside" hint="Images per step." {aside}><Input id="gf-aside" value="16" /></Field>
  <Field label="Learning rate" for="gf-error" error="Use a number between 0 and 1."><Input id="gf-error" value="2" /></Field>
  <div data-parity-step="fix" data-parity-target="button.fix">
    <Field
      label="Name"
      for="gf-fix"
      error="Names use letters, digits and dashes."
      fix={{ label: 'Try my-dataset', apply: () => (fixed = 'my-dataset') }}><Input id="gf-fix" value={fixed || 'my dataset'} /></Field
    >
  </div>
  <Field
    label="A long label that runs on and on to show how the top row wraps beside its aside"
    hint="A long hint that keeps going past the readable measure so that it has to wrap onto a second line, maybe a third one too, with `code-spans` inside."
    {aside}><Input value="" placeholder="Nothing yet" /></Field
  >
  <Field label="Without a for" hint="The hint still gets an id of its own."><Input /></Field>
  <div data-parity-step="hover" data-parity-do="hover" data-parity-target="button.fix">
    <Field label="Hover the fix" for="gf-hover" error="Too short." fix={{ label: 'Use the default', apply: () => {} }}
      ><Input id="gf-hover" /></Field
    >
  </div>
</div>
