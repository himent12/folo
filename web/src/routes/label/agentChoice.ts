// Which agent, model, effort and service tier a labeling run asks for: picked in the job page's Agent panel and used by every run
// started there (the panel's, and the editor's "Ask the agent again"). The agent is remembered per browser (prefs).

import { signal } from 'fw';
import type { LabelAgent, LabelAgentId, LabelEffort, StartLabeling } from '$lib/api/types';
import { prefs } from '$lib/prefs';

export class AgentChoice {
  #picked = signal<LabelAgentId | null>(prefs.labelAgent);
  /** The model picked; '' until one is, which means the agent's default. */
  #model = signal('');
  /** The effort picked for each agent; one not picked is that agent's default. */
  #effort = signal<Partial<Record<LabelAgentId, LabelEffort>>>({});
  /**
   * The faster service tier, switched on for each agent: one switched on for one agent is not carried to another,
   * and it counts only with a model that offers it (`fastOffered`).
   */
  #fast = signal<Partial<Record<LabelAgentId, boolean>>>({});

  constructor(readonly agents: () => readonly LabelAgent[] | undefined) {}

  /** The agent a run asks: the one picked (or remembered) while it is installed, else the first that is. */
  get agent(): LabelAgent | undefined {
    const list = this.agents() ?? [];
    const picked = this.#picked.get();
    return list.find((a) => a.id === picked && a.available) ?? list.find((a) => a.available);
  }

  pick(id: LabelAgentId) {
    this.#picked.set(id);
    prefs.labelAgent = id;
    this.#model.set('');
  }

  /** The model shown: the one picked, else the agent's default ('' stands for the CLI's own). */
  get model(): string {
    return this.#model.get() || this.agent?.default_model || '';
  }
  set model(id: string) {
    this.#model.set(id);
  }

  /** The effort shown: the one picked for this agent, else its default (what labels best for its cost). */
  get effort(): LabelEffort {
    const a = this.agent;
    return (a && this.#effort.get()[a.id]) ?? a?.default_effort ?? 'low';
  }
  set effort(e: LabelEffort) {
    const a = this.agent;
    if (a) this.#effort.set({ ...this.#effort.peek(), [a.id]: e });
  }

  /**
   * Whether the model shown offers the faster service tier: a listed model says so itself (`LabelModel.fast`), the
   * agent's default model as `LabelAgent.fast` says. A model the agent does not list is not offered it.
   */
  get fastOffered(): boolean {
    const a = this.agent;
    if (!a) return false;
    const model = this.model;
    return model ? !!a.models.find((m) => m.id === model)?.fast : !!a.fast;
  }

  /** The faster service tier: switched on for this agent, and offered with the model shown. */
  get fast(): boolean {
    const a = this.agent;
    return !!a && this.fastOffered && !!this.#fast.get()[a.id];
  }
  set fast(on: boolean) {
    const a = this.agent;
    if (a) this.#fast.set({ ...this.#fast.peek(), [a.id]: on });
  }

  /** The request's agent part, or null when no agent is installed. */
  request(): Pick<StartLabeling, 'agent' | 'model' | 'effort' | 'fast'> | null {
    const a = this.agent;
    return a ? { agent: a.id, model: this.model || null, effort: this.effort, ...(this.fast ? { fast: true } : {}) } : null;
  }
}
