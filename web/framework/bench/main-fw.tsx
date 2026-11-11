// The fw side of the benchmark: the js-framework-benchmark table, written idiomatically.
import { flush, For, render, signal, type Signal } from 'fw';
import { labels, type Ops } from './data';

interface Row {
  id: number;
  label: Signal<string>;
}

let nextId = 1;
const build = (n: number): Row[] => labels(n).map((l) => ({ id: nextId++, label: signal(l) }));

const data = signal<Row[]>([]);
const selected = signal<number | null>(null);

const ops: Ops = {
  run: () => data.set(build(1000)),
  runLots: () => data.set(build(10000)),
  add: () => data.set((d) => [...d, ...build(1000)]),
  update: () => {
    const d = data.peek();
    for (let i = 0; i < d.length; i += 10) d[i]!.label.set((l) => l + ' !!!');
  },
  clear: () => data.set([]),
  swapRows: () => {
    const d = data.peek().slice();
    if (d.length > 998) {
      const t = d[1]!;
      d[1] = d[998]!;
      d[998] = t;
      data.set(d);
    }
  },
  select: (i) => selected.set(data.peek()[i]?.id ?? null),
  remove: (i) => data.set((d) => d.toSpliced(i, 1)),
};

function App() {
  return (
    <table class="table">
      <tbody>
        <For each={data.get()}>
          {(row) => {
            const { id, label } = row();
            return (
              <tr class={selected.get() === id ? 'danger' : ''}>
                <td class="col-md-1">{id}</td>
                <td class="col-md-4">
                  <a onclick={() => selected.set(id)}>{label.get()}</a>
                </td>
                <td class="col-md-1">
                  <a onclick={() => ops.remove(data.peek().findIndex((r) => r.id === id))}>
                    <span class="glyphicon glyphicon-remove" aria-hidden="true" />
                  </a>
                </td>
                <td class="col-md-6" />
              </tr>
            );
          }}
        </For>
      </tbody>
    </table>
  );
}

render(() => <App />, document.getElementById('app')!);
(window as unknown as { bench: Ops }).bench = Object.fromEntries(
  Object.entries(ops).map(([k, fn]) => [k, (...a: [number]) => (fn(...a), flush())]),
) as unknown as Ops;
