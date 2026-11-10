// The mock's "Design with an agent" (docs/BENCH.md, Designing a plan with an agent): a plan drawn from the catalog that
// follows what the benchmark's description (and the request's note) says about the camera, with a reason per
// condition and one for the whole, as Claude Code or Codex would answer it. Pure: the same words give the same plan.

import type { BenchPlan, BenchPlanCondition, LabelAgentId, LabelUsage } from '../types';
import { CONDITIONS, conditionName } from './benchCatalog';
import { hash } from './rng';

interface Rule {
  /** What in the description calls for it. */
  words: RegExp;
  /** The scene in a few words, for the rationale ("murky water"). */
  scene: string;
  conditions: { id: string; weight: number; severities: number[]; why: string }[];
}

const RULES: Rule[] = [
  {
    words: /\b(water|pond|lake|river|fish|koi|aquarium|underwater|murky|tank)\b/i,
    scene: 'looks through water',
    conditions: [
      { id: 'haze', weight: 3, severities: [2, 3, 4, 5], why: 'Murky water veils everything a little way from the lens: the camera’s everyday condition, at every severity.' },
      { id: 'color_cast', weight: 3, severities: [1, 2, 3, 4, 5], why: 'Water and algae turn the image green, more on some days than others; white balance does not undo it.' },
      { id: 'low_contrast', weight: 2, severities: [1, 3, 5], why: 'Fish over a muddy bottom differ little in tone from it.' },
      { id: 'occlusion', weight: 2, severities: [1, 3, 5], why: 'Lily pads and plants hide parts of the fish.' },
      { id: 'motion_blur', weight: 1, severities: [1, 3], why: 'Fish move, and in dim light the shutter stays open longer; mild severities only.' },
    ],
  },
  {
    words: /\b(night|dark|dim|dusk|dawn|evening|shift|unlit)\b/i,
    scene: 'works in little light',
    conditions: [
      { id: 'low_light', weight: 3, severities: [2, 3, 4, 5], why: 'Much of the footage is taken in little light: weighted most, at the higher severities.' },
      { id: 'sensor_noise', weight: 2, severities: [1, 3, 5], why: 'A camera pushing its gain in the dark adds grain.' },
      { id: 'motion_blur', weight: 2, severities: [1, 3, 5], why: 'Longer exposures in the dark smear whatever moves.' },
    ],
  },
  {
    words: /\b(drone|aerial|far|distance|distant|wide|high up)\b/i,
    scene: 'sees its objects from far away',
    conditions: [
      { id: 'distance', weight: 3, severities: [1, 2, 3, 4, 5], why: 'Objects are small and get smaller as the camera is farther: the main question here.' },
      { id: 'low_resolution', weight: 2, severities: [1, 3, 5], why: 'Far objects are few pixels wide, as on a cheaper sensor.' },
    ],
  },
  {
    words: /\b(conveyor|belt|moving|fast|vehicle|traffic|car|cars|sport|running)\b/i,
    scene: 'watches things that move',
    conditions: [{ id: 'motion_blur', weight: 3, severities: [1, 2, 3, 4, 5], why: 'Objects pass the camera fast: blur along their movement is the rule, not the exception.' }],
  },
  {
    words: /\b(outdoor|outdoors|sun|sunny|glare|window|snow|beach)\b/i,
    scene: 'faces bright light',
    conditions: [
      { id: 'overexposure', weight: 2, severities: [1, 3, 5], why: 'Direct sun and glare clip the highlights.' },
      { id: 'low_contrast', weight: 1, severities: [1, 3], why: 'Bright haze flattens the tones.' },
    ],
  },
  {
    words: /\b(stream|streams|webcam|cctv|security|rtsp|bandwidth|compressed|wifi)\b/i,
    scene: 'sends a compressed stream',
    conditions: [
      { id: 'jpeg', weight: 2, severities: [2, 3, 4, 5], why: 'The stream is compressed hard when bandwidth drops.' },
      { id: 'low_resolution', weight: 2, severities: [1, 3, 5], why: 'A stream falls back to a lower resolution under load.' },
    ],
  },
  {
    words: /\b(fog|foggy|smoke|mist|rain|steam|dust)\b/i,
    scene: 'sees through fog or smoke',
    conditions: [
      { id: 'haze', weight: 3, severities: [1, 2, 3, 4, 5], why: 'Fog and smoke veil the scene.' },
      { id: 'low_contrast', weight: 2, severities: [1, 3, 5], why: 'Under a veil, tones crowd towards grey.' },
    ],
  },
  {
    words: /\b(crowd|crowded|cluttered|pile|heap|bin|bins|shelf|shelves|hidden|leaves|overlap\w*)\b/i,
    scene: 'sees objects in clutter',
    conditions: [{ id: 'occlusion', weight: 2, severities: [1, 3, 5], why: 'Objects lie over each other and are seldom seen whole.' }],
  },
  {
    words: /\b(handheld|phone|close|macro|autofocus|focus)\b/i,
    scene: 'is held or focused by hand',
    conditions: [{ id: 'defocus', weight: 2, severities: [1, 3, 5], why: 'A handheld camera often misses its focus.' }],
  },
];

/** A plan with nothing in the description to go on: the usual troubles of a fixed indoor camera, evenly weighted. */
const FALLBACK: Rule['conditions'] = [
  { id: 'low_light', weight: 1, severities: [1, 3, 5], why: 'Most cameras see dim light some of the time.' },
  { id: 'motion_blur', weight: 1, severities: [1, 3, 5], why: 'Whatever moves blurs a little.' },
  { id: 'sensor_noise', weight: 1, severities: [1, 3, 5], why: 'Every sensor adds some grain.' },
  { id: 'jpeg', weight: 1, severities: [1, 3, 5], why: 'Images are usually saved or sent as JPEG.' },
  { id: 'defocus', weight: 1, severities: [1, 3, 5], why: 'A lens is not always in focus.' },
];

const list = (xs: string[]) => (xs.length <= 1 ? (xs[0] ?? '') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);

/** The agent's proposal for a benchmark described by `text` (its description and the request's note). */
export function designPlan(text: string, agent: LabelAgentId, model: string | null): BenchPlan {
  const hits = RULES.filter((r) => r.words.test(text));
  const by = new Map<string, BenchPlanCondition>();
  for (const c of hits.length ? hits.flatMap((r) => r.conditions) : FALLBACK) {
    const had = by.get(c.id);
    if (!had) by.set(c.id, { condition: c.id, severities: [...c.severities], weight: c.weight, why: c.why });
    else if (c.weight > had.weight) by.set(c.id, { condition: c.id, severities: [...new Set([...had.severities, ...c.severities])].sort((a, b) => a - b), weight: c.weight, why: c.why });
    else had.severities = [...new Set([...had.severities, ...c.severities])].sort((a, b) => a - b);
  }
  const conditions = CONDITIONS.map((c) => by.get(c.id)).filter((c): c is BenchPlanCondition => !!c);
  const top = [...conditions].sort((a, b) => b.weight - a.weight).filter((c, _, all) => c.weight === all[0]!.weight);
  // A name in a sentence: lower case, but an acronym keeps its capitals ("JPEG compression").
  const inText = (name: string) => name.replace(/^[A-Z][a-z]/, (m) => m.toLowerCase());
  const left = CONDITIONS.filter((c) => !by.has(c.id)).map((c) => inText(c.name));
  const rationale = hits.length
    ? `This camera ${list(hits.map((h) => h.scene))}. The plan weights ${list(top.map((c) => inText(conditionName(c.condition))))} most and runs ${conditions.length} of the catalog's ${CONDITIONS.length} conditions. It leaves out ${left.length <= 5 ? list(left) : `${list(left.slice(0, 4))} and ${left.length - 4} more`}: not what this camera meets.`
    : 'The description says little about where the camera is, so the plan takes the troubles most cameras meet, at equal weight. Describe the scene (light, distance, what moves) for a sharper plan.';
  return { conditions, rationale, designed_by: { agent, model } };
}

/** What the design call reports it used: one request with the description, the classes and four images. */
export function designUsage(text: string, agent: LabelAgentId, model: string | null): LabelUsage {
  const input = (agent === 'claude' ? 1_840 : 10_900) + 4 * 1_012 + Math.round(text.length / 4) + (hash(text) % 90);
  const output = 610 + (hash(`${text}:out`) % 240);
  const price: Record<string, [number, number]> = { sonnet: [3, 15], haiku: [1, 5], opus: [15, 75] };
  const p = agent === 'claude' ? price[model ?? 'haiku'] : undefined;
  return { calls: 1, input_tokens: input, cached_input_tokens: 0, output_tokens: output, cost_usd: p ? Math.round(((input * p[0] + output * p[1]) / 1e6) * 1e6) / 1e6 : null };
}
