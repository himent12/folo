import { describe, expect, it } from 'vitest';
import { hashUrl, parse, section } from './router.svelte';

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

  it('routes labeling jobs under Datasets, ahead of a dataset id', () => {
    expect(parse('#/datasets/label').name).toBe('label-jobs');
    expect(parse('#/datasets/label?new=1').query.get('new')).toBe('1');
    const job = parse('#/datasets/label/lb_3f9a?image=17&show=failed');
    expect(job.name).toBe('label-job');
    expect(job.params.id).toBe('lb_3f9a');
    expect(job.query.get('image')).toBe('17');
    expect(parse('#/datasets/labels').name).toBe('dataset');
    expect(parse('#/datasets/label/lb_1/x').name).toBe('not-found');
  });

  it('treats malformed percent-encoding as not found', () => {
    expect(parse('#/datasets/%ZZ').name).toBe('not-found');
  });

  it('reports unknown paths', () => {
    expect(parse('#/nope/nope').name).toBe('not-found');
  });

  it('groups detail routes under their section', () => {
    expect(section('run')).toBe('runs');
    expect(section('run-new')).toBe('runs');
    expect(section('dataset')).toBe('datasets');
    expect(section('model')).toBe('models');
    expect(section('label-jobs')).toBe('datasets');
    expect(section('label-job')).toBe('datasets');
  });
});

describe('hashUrl (a path URL opened directly, DOGFOOD-3 S5)', () => {
  const at = (pathname: string, search = '', hash = '') => hashUrl({ pathname, search, hash });

  it('moves a known path into its hash route, the page query kept', () => {
    expect(at('/datasets')).toBe('/#/datasets');
    expect(at('/runs/run_7d1e')).toBe('/#/runs/run_7d1e');
    expect(at('/runs/new')).toBe('/#/runs/new');
    expect(at('/models/m_31aa/')).toBe('/#/models/m_31aa');
    expect(at('/playground', '?mock=1')).toBe('/?mock=1#/playground');
    expect(at('/settings')).toBe('/#/settings');
    expect(at('/datasets/label/lb_1')).toBe('/#/datasets/label/lb_1');
  });

  it("moves the route's query into the hash with it, and keeps the page's own keys in the page's query", () => {
    const route = (url: string | null) => parse(url!.slice(url!.indexOf('#')));
    const retry = at('/runs/new', '?from=run_a&batch=8');
    expect(retry).toBe('/#/runs/new?from=run_a&batch=8');
    expect(Object.fromEntries(route(retry).query)).toEqual({ from: 'run_a', batch: '8' });
    const compare = at('/runs/compare', '?ids=a,b');
    expect(route(compare).name).toBe('run-compare');
    expect(route(compare).query.get('ids')).toBe('a,b');
    // Both kinds at once, encoded values kept as they were meant.
    const both = at('/runs/new', '?mock=1&from=run%20a%26b&scenario=gpu-unplugged&name=caf%C3%A9+r6');
    expect(both).toMatch(/^\/\?mock=1&scenario=gpu-unplugged#\/runs\/new\?/);
    expect(new URLSearchParams(both!.slice(0, both!.indexOf('#')).replace(/^\/\?/, '')).get('mock')).toBe('1');
    expect(Object.fromEntries(route(both).query)).toEqual({ from: 'run a&b', name: 'café r6' });
  });

  it('keeps a hash route that is already there', () => {
    expect(at('/datasets', '', '#/datasets')).toBe('/#/datasets');
    expect(at('/datasets', '', '#/runs?ids=a')).toBe('/#/runs?ids=a');
  });

  it('leaves the root, index.html and paths that are no route alone', () => {
    expect(at('/')).toBeNull();
    expect(at('/', '', '#/runs')).toBeNull();
    expect(at('/index.html')).toBeNull();
    expect(at('/favicon.svg')).toBeNull();
    expect(at('/nope/nope')).toBeNull();
  });

  it('works under a base path', () => {
    expect(hashUrl({ pathname: '/studio/runs', search: '', hash: '' }, '/studio/')).toBe('/studio/#/runs');
    expect(hashUrl({ pathname: '/other/runs', search: '', hash: '' }, '/studio/')).toBeNull();
  });
});
