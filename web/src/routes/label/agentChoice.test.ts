import { describe, expect, it } from 'vitest';
import { mockAgents } from '$lib/api/mock/label';
import { prefs } from '$lib/prefs';
import { AgentChoice } from './agentChoice';

describe('AgentChoice', () => {
  it("starts on each agent's default model and effort, and remembers the agent", () => {
    prefs.labelAgent = null;
    const c = new AgentChoice(() => mockAgents(true));
    expect(c.request()).toEqual({ agent: 'claude', model: 'haiku', effort: 'low' });
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

  it('asks for the faster tier only of an agent that offers one, and keeps it per agent', () => {
    prefs.labelAgent = null;
    const c = new AgentChoice(() => mockAgents(true));
    // Claude Code offers none: switched on there, it is not asked for, nor carried over to Codex.
    expect(c.fastOffered).toBe(false);
    c.fast = true;
    expect(c.fast).toBe(false);
    expect(c.request()).toEqual({ agent: 'claude', model: 'haiku', effort: 'low' });
    c.pick('codex');
    expect([c.fastOffered, c.fast]).toEqual([true, false]);
    expect(c.request()).toEqual({ agent: 'codex', model: null, effort: 'low' });
    c.fast = true;
    expect(c.request()).toEqual({ agent: 'codex', model: null, effort: 'low', fast: true });
    // Back and forth: Codex keeps it, Claude Code still does not ask for it.
    c.pick('claude');
    expect(c.request()).toEqual({ agent: 'claude', model: 'haiku', effort: 'low' });
    c.pick('codex');
    expect(c.fast).toBe(true);
    c.fast = false;
    expect(c.request()).toEqual({ agent: 'codex', model: null, effort: 'low' });
  });

  it('offers the faster tier only with a model that has it', () => {
    prefs.labelAgent = 'codex';
    const agents = mockAgents(true);
    agents[1]!.models = [{ id: 'quick', label: 'Quick', fast: true }, { id: 'plain', label: 'Plain' }];
    const c = new AgentChoice(() => agents);
    c.fast = true;
    c.model = 'quick';
    expect(c.request()).toEqual({ agent: 'codex', model: 'quick', effort: 'low', fast: true });
    c.model = 'plain';
    expect(c.request()).toEqual({ agent: 'codex', model: 'plain', effort: 'low' });
    expect(c.fastOffered).toBe(false);
    // A model the agent does not list: not offered (the server would refuse it).
    c.model = 'typed-by-hand';
    expect(c.fast).toBe(false);
  });

  it('falls back to an installed agent when the remembered one is not', () => {
    prefs.labelAgent = 'codex';
    const c = new AgentChoice(() => mockAgents(false));
    expect(c.agent?.id).toBe('claude');
    expect(new AgentChoice(() => []).request()).toBeNull();
  });
});
