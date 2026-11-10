import { describe, expect, it } from 'vitest';
import { mockAgents } from '$lib/api/mock/label';
import { prefs } from '$lib/prefs.svelte';
import { AgentChoice } from './agentChoice.svelte';

describe('AgentChoice', () => {
  it("starts on each agent's default model and effort, and remembers the agent", () => {
    prefs.labelAgent = null;
    const c = new AgentChoice(() => mockAgents(true));
    expect(c.request()).toEqual({ agent: 'claude', model: 'haiku', effort: 'medium' });
    c.pick('codex');
    expect(prefs.labelAgent).toBe('codex');
    // Codex names no default model: the request leaves it to the CLI.
    expect(c.request()).toEqual({ agent: 'codex', model: null, effort: 'low' });
  });

  it('keeps an effort picked by hand for its agent only', () => {
    prefs.labelAgent = null;
    const c = new AgentChoice(() => mockAgents(true));
    c.effort = 'high';
    c.pick('codex');
    expect(c.effort).toBe('low');
    c.pick('claude');
    expect(c.effort).toBe('high');
  });

  it('falls back to an installed agent when the remembered one is not', () => {
    prefs.labelAgent = 'codex';
    const c = new AgentChoice(() => mockAgents(false));
    expect(c.agent?.id).toBe('claude');
    expect(new AgentChoice(() => []).request()).toBeNull();
  });
});
