<script lang="ts">
  /**
   * A benchmark's self-improvement loops (docs/BENCH.md, Self-improvement loops): each with its teacher and student,
   * where it stands (its phase while it runs), its generations' scores as a small line from the first teacher to the
   * last student, and its verdict. A row opens its detail under it (./LoopDetail, the seam the loop's own view
   * replaces), kept in the URL (`?tab=loops&loop=…`) so it can be linked.
   */
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import Repeat from '@lucide/svelte/icons/repeat';
  import type { Bench, BenchEntry, BenchLoop, Dataset, LabelJob } from '$lib/api/types';
  import Sparkline from '$lib/charts/Sparkline.svelte';
  import { ago, date } from '$lib/format';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import LoopDetail from './LoopDetail.svelte';
  import { loopStatus, scoreText, VERDICT } from './words';

  let {
    bench,
    loops,
    entries,
    datasets,
    jobs,
    expanded,
    ontoggle,
  }: {
    bench: Bench;
    loops: readonly BenchLoop[];
    entries: readonly BenchEntry[];
    datasets: readonly Dataset[] | undefined;
    jobs: readonly LabelJob[] | undefined;
    /** The loop opened under its row, or null. */
    expanded: string | null;
    ontoggle: (id: string | null) => void;
  } = $props();

  const pool = (l: BenchLoop) => {
    const p = l.config.pool;
    if (p.kind === 'dataset') return `${datasets?.find((d) => d.id === p.dataset_id)?.name ?? p.dataset_id} · ${p.split ?? 'train'}`;
    return jobs?.find((j) => j.id === p.job_id)?.name ?? 'a labeling job';
  };
  const teacherArch = (l: BenchLoop) => {
    const e = entries.find((x) => x.id === l.generations[0]?.entry_id);
    return e?.contestant.kind === 'model' ? e.contestant.arch : null;
  };
  const scores = (l: BenchLoop) => l.generations.map((g) => g.score).filter((s): s is number => s !== null);
</script>

<Panel title="Self-improvement loops" meta={loops.length ? String(loops.length) : undefined} pad="none">
  {#if !loops.length}
    <EmptyState
      icon={Repeat}
      title="No loops yet"
      description="A loop tests whether a model's own labels can train a better one: it labels a pool of images by itself, a student trains on those labels alone, and this benchmark grades the student."
    />
  {:else}
    <div class="scroll">
      <table class="table tall loops" aria-label="Self-improvement loops of {bench.name}">
        <thead>
          <tr>
            <th class="lead">Loop</th>
            <th style:width="136px">Status</th>
            <th style:width="232px">Score by generation</th>
            <th style:width="120px">Verdict</th>
            <th class="r hide-md" style:width="104px">Started</th>
            <th style:width="40px"><span class="sr-only">Open</span></th>
          </tr>
        </thead>
        <tbody>
          {#each loops as l (l.id)}
            {@const open = expanded === l.id}
            {@const s = scores(l)}
            {@const arch = teacherArch(l)}
            <tr class="link" class:on={open} aria-expanded={open} aria-label={l.name} onclick={() => ontoggle(open ? null : l.id)}>
              <td class="lead">
                <span class="name">{l.name}</span>
                <span class="sub"
                  ><Meta line>
                    {#if arch}<span>teacher <Ident value={arch} /></span>{/if}
                    {#if l.config.student?.arch}<span>student <Ident value={l.config.student.arch} /></span>{/if}
                    <span>pool {pool(l)}</span>
                  </Meta></span
                >
              </td>
              <td><StatusBadge plain state={loopStatus(l)} /></td>
              <td>
                <span class="gens">
                  {#if s.length >= 2}<Sparkline values={s} width={56} height={20} color="var(--text-secondary)" fill={false} />{/if}
                  <span class="num g-text">
                    {#if s.length}{s.map(scoreText).join(' → ')}{:else}<NoValue label="no score yet" />{/if}
                  </span>
                </span>
              </td>
              <td>{#if l.verdict}<span class="verdict" title={VERDICT[l.verdict].means}>{VERDICT[l.verdict].word}</span>{:else}<NoValue label="no verdict yet" />{/if}</td>
              <td class="r faint hide-md" title={date(l.created)}>{ago(l.created)}</td>
              <td class="chev-cell"><span class="chev" class:open><ChevronDown size={16} strokeWidth={1.75} aria-hidden="true" /></span></td>
            </tr>
            {#if open}
              <tr class="detail-row">
                <td colspan="6"><LoopDetail {bench} loop={l} {entries} /></td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
    <!-- Phones: the name over its status and scores, the verdict on the right; a tap opens the detail under it. -->
    <ul class="rows" aria-label="Self-improvement loops of {bench.name}">
      {#each loops as l (l.id)}
        {@const open = expanded === l.id}
        {@const s = scores(l)}
        <li class:on={open}>
          <button type="button" class="m-row" aria-expanded={open} onclick={() => ontoggle(open ? null : l.id)}>
            <span class="m-main">
              <span class="name truncate">{l.name}</span>
              <span class="m-meta num"><Meta><span>{loopStatus(l).label}</span>{#if s.length}<span>{scoreText(s[0]!)} → {scoreText(s.at(-1)!)}</span>{/if}<span>{l.generations.length - 1} {l.generations.length === 2 ? 'generation' : 'generations'}</span></Meta></span>
            </span>
            <span class="m-side">{#if l.verdict}<span class="verdict">{VERDICT[l.verdict].word}</span>{:else}<StatusBadge plain dotOnly state={loopStatus(l)} />{/if}</span>
          </button>
          {#if open}<LoopDetail {bench} loop={l} {entries} />{/if}
        </li>
      {/each}
    </ul>
  {/if}
</Panel>

<style>
  .scroll {
    overflow-x: auto;
  }
  .name {
    display: block;
  }
  .gens {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .g-text {
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
  }
  .verdict {
    font-weight: var(--weight-medium);
  }
  .chev-cell {
    padding-right: var(--space-3);
    color: var(--text-tertiary);
  }
  .chev {
    display: inline-flex;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .chev.open {
    transform: rotate(180deg);
  }
  .detail-row td {
    padding: 0;
    background: var(--bg-raised);
  }
  .rows {
    display: none;
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows li + li {
      border-top: 1px solid var(--line-subtle);
    }
    .rows li.on {
      background: var(--bg-raised);
    }
    .m-row {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      width: 100%;
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
      text-align: left;
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
    }
    .rows .name {
      font-weight: var(--weight-medium);
    }
    .m-meta {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .m-side {
      flex-shrink: 0;
    }
  }
</style>
