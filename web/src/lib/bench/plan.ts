// A benchmark's plan (docs/API.md, Vision benchmarks: "A plan is …"): whether one is valid, editing it a step at a time
// in the screens' editor, and what a proposed plan changes from the current one.

/** A weight field as typed holds no number (emptied, or text): the plan keeps the last number typed, which it does not show. */
export function unreadableWeight(v: string | undefined): boolean {
  return v !== undefined && (!v.trim() || !Number.isFinite(Number(v)));
}

/**
 * What is wrong with a plan being edited, or null: a weight field that holds no number (`typed`, per condition, the
 * fields as typed), else `planError`. Every save of the edits checks it (the plan's Save and the leave dialog's), so
 * none stores a weight the field does not show.
 */
export function editedPlanError(
  plan: BenchPlan,
  typed: Readonly<Record<string, string>>,
  catalog?: readonly Pick<BenchCondition, 'id' | 'name'>[],
): string | null {
  const empty = plan.conditions.find((c) => unreadableWeight(typed[c.condition]));
  if (!empty) return planError(plan, catalog);
  const name = catalog?.find((c) => c.id === empty.condition)?.name ?? empty.condition;
  return `${name}: the weight is empty. ${weightError(Number.NaN)}.`;
}


import type { BenchCatalog, BenchCondition, BenchPlan, BenchPlanCondition } from '../api/types';
import { MAX_SEVERITY, WEIGHT_MAX } from './rules';

/** What is wrong with a plan, or null: every condition in the catalog and named once, severities 1–5 ascending, `0 < weight <= 10`. */
export function planError(plan: BenchPlan, catalog?: readonly Pick<BenchCondition, 'id' | 'name'>[]): string | null {
  const seen = new Set<string>();
  for (const c of plan.conditions) {
    const name = catalog?.find((x) => x.id === c.condition)?.name ?? c.condition;
    if (catalog && !catalog.some((x) => x.id === c.condition)) return `Unknown condition "${c.condition}".`;
    if (seen.has(c.condition)) return `${name} is in the plan twice.`;
    seen.add(c.condition);
    if (!c.severities.length) return `${name} needs at least one severity.`;
    if (c.severities.length > MAX_SEVERITY) return `${name} has more than ${MAX_SEVERITY} severities.`;
    for (let i = 0; i < c.severities.length; i++) {
      const s = c.severities[i]!;
      if (!Number.isInteger(s) || s < 1 || s > MAX_SEVERITY) return `${name}: severities run from 1 to ${MAX_SEVERITY}.`;
      if (i > 0 && s <= c.severities[i - 1]!) return `${name}: severities must be distinct and ascending.`;
    }
    if (!Number.isFinite(c.weight) || c.weight <= 0 || c.weight > WEIGHT_MAX) return `${name}: the weight is more than 0 and at most ${WEIGHT_MAX}.`;
  }
  return null;
}

/** What is wrong with a weight typed in, or null. */
export function weightError(w: number): string | null {
  return !Number.isFinite(w) || w <= 0 || w > WEIGHT_MAX ? `More than 0, at most ${WEIGHT_MAX}` : null;
}

/** Image passes a plan adds to the unchanged images: one per condition and severity. */
export const planPasses = (plan: BenchPlan) => plan.conditions.reduce((a, c) => a + c.severities.length, 0);

/** The plan's condition for `id`, if it runs. */
export const planCondition = (plan: BenchPlan, id: string) => plan.conditions.find((c) => c.condition === id);

/**
 * Two plans run the same conditions at the same severities and weights, with the same reasons, as the server keeps
 * them: a weight as a 32-bit float (`1.00000001` is `1`), a reason trimmed and an empty one none. So a plan the server
 * answers with is the one sent, though it rounded or trimmed it.
 */
export function samePlan(a: BenchPlan, b: BenchPlan): boolean {
  if (kept(a.rationale) !== kept(b.rationale)) return false;
  if ((a.designed_by?.agent ?? null) !== (b.designed_by?.agent ?? null) || (a.designed_by?.model ?? null) !== (b.designed_by?.model ?? null)) return false;
  if (a.conditions.length !== b.conditions.length) return false;
  return a.conditions.every((c, i) => {
    const d = b.conditions[i];
    return !!d && d.condition === c.condition && Math.fround(d.weight) === Math.fround(c.weight) && kept(d.why) === kept(c.why) && d.severities.join() === c.severities.join();
  });
}

/** A reason as the server keeps it: trimmed, an empty one none (crates/folo-bench `check_plan`). */
const kept = (s: string | null | undefined) => s?.trim() || null;

/** The conditions in catalog order, so a plan edited a step at a time reads in the order the editor lists them. */
function ordered(conditions: BenchPlanCondition[], catalog: readonly Pick<BenchCondition, 'id'>[]): BenchPlanCondition[] {
  const at = (id: string) => {
    const i = catalog.findIndex((c) => c.id === id);
    return i < 0 ? catalog.length : i;
  };
  return [...conditions].sort((a, b) => at(a.condition) - at(b.condition));
}

/**
 * The plan with `id` turned on or off. On, it takes what it had in `fallback` (the plan as loaded, so turning a
 * condition off and on again gives back its severities, weight and reason), else the catalog's default severities.
 */
export function toggleCondition(plan: BenchPlan, id: string, on: boolean, catalog: BenchCatalog, fallback?: BenchPlan): BenchPlan {
  const has = !!planCondition(plan, id);
  if (on === has) return plan;
  if (!on) return { ...plan, conditions: plan.conditions.filter((c) => c.condition !== id) };
  const back = fallback && planCondition(fallback, id);
  const def = planCondition(catalog.default_plan, id);
  const added: BenchPlanCondition = back ? { ...back, severities: [...back.severities] } : { condition: id, severities: [...(def?.severities ?? [1, 3, 5])], weight: 1, why: null };
  return { ...plan, conditions: ordered([...plan.conditions, added], catalog.conditions) };
}

/** The plan with severity `s` of `id` toggled. The last severity of a condition cannot go: turn the condition off instead. */
export function toggleSeverity(plan: BenchPlan, id: string, s: number): BenchPlan {
  return {
    ...plan,
    conditions: plan.conditions.map((c) => {
      if (c.condition !== id) return c;
      const has = c.severities.includes(s);
      if (has && c.severities.length === 1) return c;
      const severities = has ? c.severities.filter((x) => x !== s) : [...c.severities, s].sort((a, b) => a - b);
      return { ...c, severities };
    }),
  };
}

/** The plan with `id`'s weight set (kept as typed; `planError` says when it is out of range). */
export function setWeight(plan: BenchPlan, id: string, weight: number): BenchPlan {
  return { ...plan, conditions: plan.conditions.map((c) => (c.condition === id ? { ...c, weight } : c)) };
}

/** What changed for one condition between two plans. */
export interface PlanChange {
  condition: string;
  /** In the current plan; null when the proposal adds it. */
  from: BenchPlanCondition | null;
  /** In the proposal; null when it removes it. */
  to: BenchPlanCondition | null;
  /** For a change: which parts differ. */
  parts: ('severities' | 'weight')[];
}

/** What a proposed plan adds, removes and changes from the current one (and what it keeps), in catalog order. */
export function diffPlans(from: BenchPlan, to: BenchPlan, catalog: readonly Pick<BenchCondition, 'id'>[]): { added: PlanChange[]; removed: PlanChange[]; changed: PlanChange[]; kept: PlanChange[] } {
  const out = { added: [] as PlanChange[], removed: [] as PlanChange[], changed: [] as PlanChange[], kept: [] as PlanChange[] };
  const ids = [...new Set([...catalog.map((c) => c.id), ...from.conditions.map((c) => c.condition), ...to.conditions.map((c) => c.condition)])];
  for (const id of ids) {
    const a = planCondition(from, id) ?? null;
    const b = planCondition(to, id) ?? null;
    if (!a && !b) continue;
    if (!a) out.added.push({ condition: id, from: null, to: b, parts: [] });
    else if (!b) out.removed.push({ condition: id, from: a, to: null, parts: [] });
    else {
      const parts: PlanChange['parts'] = [];
      if (a.severities.join() !== b.severities.join()) parts.push('severities');
      if (a.weight !== b.weight) parts.push('weight');
      (parts.length ? out.changed : out.kept).push({ condition: id, from: a, to: b, parts });
    }
  }
  return out;
}

/** A plan from the catalog's default: every condition at severities 1, 3 and 5, weight 1, no reasons. */
export function defaultPlan(catalog: BenchCatalog): BenchPlan {
  return { conditions: catalog.default_plan.conditions.map((c) => ({ ...c, severities: [...c.severities] })), rationale: null, designed_by: null };
}

/** A plan as a request sends it: severities ascending, the reasons and the designer as they are. */
export function clonePlan(plan: BenchPlan): BenchPlan {
  return {
    conditions: plan.conditions.map((c) => ({ condition: c.condition, severities: [...c.severities].sort((a, b) => a - b), weight: c.weight, why: c.why ?? null })),
    rationale: plan.rationale ?? null,
    designed_by: plan.designed_by ? { ...plan.designed_by } : null,
  };
}
