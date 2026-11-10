<script lang="ts">
  /**
   * A benchmark's self-improvement loops (docs/BENCH.md, Self-improvement loops): each with its teacher and student,
   * where it stands (what it does for which generation while it runs; a failed one's cause under its name, so a loop
   * that starved for labels says so in the list), its generations' scores as a small line from the first teacher to the
   * last student, and its verdict. A row opens the loop's own page (`#/benchmarks/{id}/loops/{loopId}`, ../BenchLoop);
   * "New loop" starts one.
   */
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Plus from '@lucide/svelte/icons/plus';
  import Repeat from '@lucide/svelte/icons/repeat';
  import type { Bench, BenchEntry, BenchLoop, Dataset, LabelJob } from '$lib/api/types';
  import Sparkline from '$lib/charts/Sparkline.svelte';
  import { ago, date, sentence } from '$lib/format';
  import CodeText from '$lib/ui/CodeText.svelte';
  import { href, router } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import { OFFLINE_REASON } from '$lib/ui/offline';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { loopStatus, scoreText, VERDICT } from './words';

  let {
    bench,
    loops,
    entries,
    datasets,
    jobs,
    onnew,
    offline = false,
  }: {
    bench: Bench;
    loops: readonly BenchLoop[];
    entries: readonly BenchEntry[];
    datasets: readonly Dataset[] | undefined;
    jobs: readonly LabelJob[] | undefined;
    /** Opens "New loop". */
    onnew: () => void;
    /** The server does not answer: a loop cannot start. */
    offline?: boolean;
  } = $props();
  const path = (l: BenchLoop) => `/benchmarks/${bench.id}/loops/${l.id}`;

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
  {#snippet actions()}
    {#if loops.length}
      <Gate reason={offline ? OFFLINE_REASON : null}>
        {#snippet children(why)}<Button size="sm" icon={Plus} disabled={!!why} aria-describedby={why} onclick={onnew}>New loop</Button>{/snippet}
      </Gate>
    {/if}
  {/snippet}
  {#if !loops.length}
    <EmptyState
      icon={Repeat}
      title="No loops yet"
      description="A loop tests whether a model's own labels can train a better one: it labels a pool of images by itself, a student trains on those labels alone, and this benchmark grades the student."
    >
      {#snippet actions()}
        <Gate reason={offline ? OFFLINE_REASON : null}>
          {#snippet children(why)}<Button icon={Plus} disabled={!!why} aria-describedby={why} onclick={onnew}>New loop</Button>{/snippet}
        </Gate>
      {/snippet}
    </EmptyState>
  {:else}
    <div class="scroll">
      <table class="table tall loops" aria-label="Self-improvement loops of {bench.name}">
        <thead>
          <tr>
            <th class="lead">Loop</th>
            <th class="w-st">Status</th>
            <th class="w-sc">Score by generation</th>
            <th class="w-v">Verdict</th>
            <th class="r hide-md" style:width="104px">Started</th>
            <th style:width="40px"><span class="sr-only">Open</span></th>
          </tr>
        </thead>
        <tbody>
          {#each loops as l (l.id)}
            {@const s = scores(l)}
            {@const arch = teacherArch(l)}
            <tr class="link" aria-label={l.name} onclick={() => router.go(path(l))}>
              <td class="lead">
                <a class="name" href={href(path(l))} onclick={(e) => e.stopPropagation()}>{l.name}</a>
                <span class="sub"
                  ><Meta>
                    {#if arch}<span>teacher <Ident value={arch} /></span>{/if}
                    {#if l.config.student?.arch}<span>student <Ident value={l.config.student.arch} /></span>{/if}
                    <span>pool {pool(l)}</span>
                  </Meta></span
                >
                {#if l.status === 'failed'}<span class="cause" data-fact><CodeText text={sentence(l.error ?? 'The loop failed.')} /></span>{/if}
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
              <td class="chev-cell"><ChevronRight size={16} strokeWidth={1.75} aria-hidden="true" /></td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <!-- Phones: the name over its status and scores, the verdict on the right; a tap opens the loop. -->
    <ul class="rows" aria-label="Self-improvement loops of {bench.name}">
      {#each loops as l (l.id)}
        {@const s = scores(l)}
        <li>
          <a class="m-row" href={href(path(l))}>
            <span class="m-main">
              <span class="name truncate">{l.name}</span>
              <span class="m-meta num"><Meta><span>{loopStatus(l).label}</span>{#if s.length}<span>{scoreText(s[0]!)} → {scoreText(s.at(-1)!)}</span>{/if}<span>{(l.config.generations ?? 1)} {(l.config.generations ?? 1) === 1 ? 'generation' : 'generations'}</span></Meta></span>
              {#if l.status === 'failed'}<span class="m-cause" data-fact><CodeText text={sentence(l.error ?? 'The loop failed.')} /></span>{/if}
            </span>
            <span class="m-side">{#if l.verdict}<span class="verdict">{VERDICT[l.verdict].word}</span>{:else}<StatusBadge plain dotOnly state={loopStatus(l)} />{/if}</span>
          </a>
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
    color: var(--text-primary);
  }
  /* The loop's cell takes what the fixed columns leave, so a failed loop's cause wraps there instead of widening it. */
  .loops td.lead {
    max-width: 0;
    white-space: normal;
  }
  .cause {
    display: block;
    margin-top: 2px;
    white-space: normal;
    max-inline-size: var(--measure-dialog);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .w-st {
    width: 176px;
  }
  .w-sc {
    width: 232px;
  }
  .w-v {
    width: 120px;
  }
  /* Tablets: the fixed columns give the loop's own cell the room its facts need. */
  @media (max-width: 1023px) {
    .w-st {
      width: 160px;
    }
    .w-sc {
      width: 200px;
    }
    .w-v {
      width: 96px;
    }
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
    .m-row:active {
      background: var(--bg-hover);
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
    .m-cause {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
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
