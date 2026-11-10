import { describe, expect, it } from 'vitest';
import type { BenchPlan } from '../api/types';
import { catalog } from '../api/mock/benchCatalog';
import { clonePlan, defaultPlan, diffPlans, planError, planPasses, samePlan, setWeight, toggleCondition, toggleSeverity, weightError } from './plan';

const cat = catalog();
const plan = (conditions: BenchPlan['conditions']): BenchPlan => ({ conditions, rationale: null, designed_by: null });

describe('planError', () => {
  it('accepts the default plan and an empty one (the unchanged images only)', () => {
    expect(planError(cat.default_plan, cat.conditions)).toBeNull();
    expect(planError(plan([]), cat.conditions)).toBeNull();
  });

  it('names what is wrong: an unknown or repeated condition, severities, the weight', () => {
    expect(planError(plan([{ condition: 'rain', severities: [1], weight: 1 }]), cat.conditions)).toMatch(/unknown condition "rain"/i);
    expect(planError(plan([{ condition: 'haze', severities: [1], weight: 1 }, { condition: 'haze', severities: [2], weight: 1 }]), cat.conditions)).toMatch(/Haze is in the plan twice/);
    expect(planError(plan([{ condition: 'haze', severities: [], weight: 1 }]), cat.conditions)).toMatch(/at least one severity/);
    expect(planError(plan([{ condition: 'haze', severities: [3, 1], weight: 1 }]), cat.conditions)).toMatch(/ascending/);
    expect(planError(plan([{ condition: 'haze', severities: [6], weight: 1 }]), cat.conditions)).toMatch(/1 to 5/);
    expect(planError(plan([{ condition: 'haze', severities: [1], weight: 0 }]), cat.conditions)).toMatch(/weight/);
    expect(planError(plan([{ condition: 'haze', severities: [1], weight: 10.5 }]), cat.conditions)).toMatch(/at most 10/);
    expect(weightError(0)).toBeTruthy();
    expect(weightError(10)).toBeNull();
  });
});

describe('editing a plan a step at a time', () => {
  it('turns a condition off and back on with what it had, else with the default severities, in catalog order', () => {
    const base = defaultPlan(cat);
    const tuned = setWeight(toggleSeverity(base, 'haze', 2), 'haze', 3);
    const off = toggleCondition(tuned, 'haze', false, cat);
    expect(off.conditions.some((c) => c.condition === 'haze')).toBe(false);
    // Back on: what it had in the plan as loaded.
    const back = toggleCondition(off, 'haze', true, cat, tuned);
    expect(back.conditions.find((c) => c.condition === 'haze')).toMatchObject({ severities: [1, 2, 3, 5], weight: 3 });
    expect(back.conditions.map((c) => c.condition)).toEqual(cat.conditions.map((c) => c.id));
    // Without a fallback: the catalog's default severities, weight 1.
    const fresh = toggleCondition(plan([]), 'jpeg', true, cat);
    expect(fresh.conditions).toEqual([{ condition: 'jpeg', severities: [1, 3, 5], weight: 1, why: null }]);
  });

  it('keeps the last severity of a condition (turning it off is the condition switch)', () => {
    const one = plan([{ condition: 'haze', severities: [3], weight: 1 }]);
    expect(toggleSeverity(one, 'haze', 3)).toEqual(one);
    expect(toggleSeverity(one, 'haze', 1).conditions[0]!.severities).toEqual([1, 3]);
  });

  it('counts the image passes a plan adds, and compares plans', () => {
    expect(planPasses(cat.default_plan)).toBe(36);
    expect(samePlan(cat.default_plan, clonePlan(cat.default_plan))).toBe(true);
    expect(samePlan(cat.default_plan, setWeight(cat.default_plan, 'haze', 2))).toBe(false);
    expect(samePlan(cat.default_plan, { ...cat.default_plan, rationale: 'why' })).toBe(false);
  });
});

describe('diffPlans', () => {
  it('says what a proposal adds, removes and changes, in catalog order, and what it keeps', () => {
    const from = plan([
      { condition: 'low_light', severities: [1, 3, 5], weight: 1 },
      { condition: 'haze', severities: [1, 3, 5], weight: 1 },
      { condition: 'jpeg', severities: [1, 3, 5], weight: 1 },
    ]);
    const to = plan([
      { condition: 'low_light', severities: [1, 3, 5], weight: 1 },
      { condition: 'haze', severities: [2, 3, 4, 5], weight: 3, why: 'murky' },
      { condition: 'color_cast', severities: [1, 2, 3, 4, 5], weight: 3, why: 'green' },
    ]);
    const d = diffPlans(from, to, cat.conditions);
    expect(d.added.map((c) => c.condition)).toEqual(['color_cast']);
    expect(d.removed.map((c) => c.condition)).toEqual(['jpeg']);
    expect(d.changed.map((c) => [c.condition, c.parts])).toEqual([['haze', ['severities', 'weight']]]);
    expect(d.kept.map((c) => c.condition)).toEqual(['low_light']);
  });
});
