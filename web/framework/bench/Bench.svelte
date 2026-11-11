<!-- The Svelte 5 side of the benchmark: the js-framework-benchmark keyed Svelte 5 implementation's shape. -->
<script lang="ts" module>
  import { labels } from './data';

  let nextId = 1;
  export class Row {
    id = nextId++;
    label = $state('');
    constructor(label: string) {
      this.label = label;
    }
  }
  export const build = (n: number) => labels(n).map((l) => new Row(l));
</script>

<script lang="ts">
  let { api }: { api: (ops: Record<string, (i: number) => void>) => void } = $props();
  let data = $state.raw<Row[]>([]);
  let selected = $state<number | null>(null);

  api({
    run: () => (data = build(1000)),
    runLots: () => (data = build(10000)),
    add: () => (data = [...data, ...build(1000)]),
    update: () => {
      for (let i = 0; i < data.length; i += 10) data[i]!.label += ' !!!';
    },
    clear: () => (data = []),
    swapRows: () => {
      if (data.length > 998) {
        const d = data.slice();
        const t = d[1]!;
        d[1] = d[998]!;
        d[998] = t;
        data = d;
      }
    },
    select: (i: number) => (selected = data[i]?.id ?? null),
    remove: (i: number) => (data = data.toSpliced(i, 1)),
  });
</script>

<table class="table">
  <tbody>
    {#each data as row (row.id)}
      <tr class={selected === row.id ? 'danger' : ''}>
        <td class="col-md-1">{row.id}</td>
        <td class="col-md-4"><a onclick={() => (selected = row.id)}>{row.label}</a></td>
        <td class="col-md-1"><a onclick={() => (data = data.filter((r) => r !== row))}><span class="glyphicon glyphicon-remove" aria-hidden="true"></span></a></td>
        <td class="col-md-6"></td>
      </tr>
    {/each}
  </tbody>
</table>
