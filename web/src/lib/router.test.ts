import { describe, expect, it } from 'vitest';
import { parse, section } from './router.svelte';

describe('router.parse', () => {
  it('maps the root and empty hash to overview', () => {
    expect(parse('').name).toBe('overview');
    expect(parse('#/').name).toBe('overview');
  });

  it('extracts params and decodes them', () => {
    const r = parse('#/runs/run%207d1e');
    expect(r.name).toBe('run');
    expect(r.params.id).toBe('run 7d1e');
  });

  it('prefers static segments over params', () => {
    expect(parse('#/runs/new').name).toBe('run-new');
    expect(parse('#/runs/compare?ids=a,b').name).toBe('run-compare');
  });

  it('parses the query and ignores trailing slashes', () => {
    const r = parse('#/datasets/ds_1/?split=val&class=2');
    expect(r.name).toBe('dataset');
    expect(r.query.get('split')).toBe('val');
    expect(r.query.get('class')).toBe('2');
  });

  it('reports unknown paths', () => {
    expect(parse('#/nope/nope').name).toBe('not-found');
  });

  it('groups detail routes under their section', () => {
    expect(section('run')).toBe('runs');
    expect(section('run-new')).toBe('runs');
    expect(section('dataset')).toBe('datasets');
    expect(section('model')).toBe('models');
  });
});
