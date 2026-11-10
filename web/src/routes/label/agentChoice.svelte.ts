// Which agent, model and effort a labeling run asks for: picked in the job page's Agent panel and used by every run
// started there (the panel's, and the editor's "Ask the agent again"). The agent is remembered per browser (prefs).

import type { LabelAgent, LabelAgentId, LabelEffort, StartLabeling } from '$lib/api/types';
import { prefs } from '$lib/prefs.svelte';

export class AgentChoice {
  #picked = $state<LabelAgentId | null>(prefs.labelAgent);
  /** The model picked; '' until one is, which means the agent's default. */
  #model = $state('');
  /** The effort picked for each agent; one not picked is that agent's default. */
  #effort = $state<Partial<Record<LabelAgentId, LabelEffort>>>({});

  constructor(readonly agents: () => readonly LabelAgent[] | undefined) {}

  /** The agent a run asks: the one picked (or remembered) while it is installed, else the first that is. */
  get agent(): LabelAgent | undefined {
    const list = this.agents() ?? [];
    return list.find((a) => a.id === this.#picked && a.available) ?? list.find((a) => a.available);
  }

  pick(id: LabelAgentId) {
    this.#picked = id;
    prefs.labelAgent = id;
    this.#model = '';
  }

  /** The model shown: the one picked, else the agent's default ('' stands for the CLI's own). */
  get model(): string {
    return this.#model || this.agent?.default_model || '';
  }
  set model(id: string) {
    this.#model = id;
  }

  /** The effort shown: the one picked for this agent, else its default (what labels best for its cost). */
  get effort(): LabelEffort {
    const a = this.agent;
    return (a && this.#effort[a.id]) ?? a?.default_effort ?? 'low';
  }
  set effort(e: LabelEffort) {
    const a = this.agent;
    if (a) this.#effort = { ...this.#effort, [a.id]: e };
  }

  /** The request's agent part, or null when no agent is installed. */
  request(): Pick<StartLabeling, 'agent' | 'model' | 'effort'> | null {
    const a = this.agent;
    return a ? { agent: a.id, model: this.model || null, effort: this.effort } : null;
  }
}
