import { describe, expect, it } from 'vitest';
import { SourceMapConsumer } from 'source-map-js';
import { transformJsx } from './jsx';

const compile = (src: string, scope?: string) => transformJsx(src, 'x.tsx', { scope }).code;
const templates = (code: string) => [...code.matchAll(/_\$template\(("(?:[^"\\]|\\.)*")/g)].map((m) => JSON.parse(m[1]!) as string);

describe('templates', () => {
  it('hoists static HTML with the scope class and leaves dynamic parts out', () => {
    const code = compile("import { css } from 'fw'; css`.a{}`; const x = <div class=\"a\" id={id()}><span>hi</span>{n()}</div>;", 'fw-s');
    expect(templates(code)).toEqual(['<div class="a fw-s"><span class="fw-s">hi</span></div>']);
    expect(code).toContain('_$attr(');
    expect(code).toContain('_$insert(');
    expect(code).not.toContain('css`');
  });

  it('no scope class without css``', () => {
    expect(templates(compile('const x = <div class="a"><i /></div>;'))).toEqual(['<div class="a"><i></i></div>']);
  });

  it('separates text around a dynamic part with a comment marker', () => {
    expect(templates(compile('const x = <p>a{b()}c</p>;'))).toEqual(['<p>a<!>c</p>']);
    expect(templates(compile('const x = <p>{a()}{b()}</p>;'))).toEqual(['<p><!></p>']);
    expect(templates(compile('const x = <p><b />{a()}</p>;'))).toEqual(['<p><b></b></p>']);
  });

  it('follows JSX whitespace rules and inlines literal children', () => {
    const t = templates(
      compile(`const x = <p>
      a   b
      <b>c</b> <i>d</i>
      {' '}e{"&"}{1}
    </p>;`),
    );
    expect(t).toEqual(['<p>a   b<b>c</b> <i>d</i> e&amp;1</p>']);
  });

  it('decodes entities and escapes for HTML', () => {
    expect(templates(compile('const x = <p title="a &amp; &quot;b&quot;">&lt;x&gt; &nbsp;</p>;'))).toEqual(['<p title="a &amp; &quot;b&quot;">&lt;x&gt;  </p>']);
    expect(() => compile('const x = <p>&bogus;</p>;')).toThrow(/unknown HTML entity/);
  });

  it('parses SVG fragments in the SVG namespace', () => {
    const code = compile('const x = <svg>{a() ? <path d="M0" /> : <line />}</svg>; const y = <g><circle /></g>;');
    expect(code).toContain('_$template("<path d=\\"M0\\"></path>", 1)');
    expect(code).toContain('_$template("<g><circle></circle></g>", 1)');
    expect(code).toContain('_$template("<svg></svg>")');
  });

  it('rejects markup the HTML parser would rewrite', () => {
    expect(() => compile('const x = <table><tr /></table>;')).toThrow(/tbody/);
    expect(() => compile('const x = <p><div /></p>;')).toThrow(/inside <p>/);
  });
});

describe('expressions', () => {
  it('memoizes conditional children, keeps static ones static', () => {
    const code = compile('const x = <div>{a() ? <b /> : c() ? "x" : null}{flag ? 1 : 2}{ok() && <i />}</div>;');
    expect(code.match(/_\$cond\(/g)?.length).toBe(3);
    expect(code).toContain('(flag ? 1 : 2)');
  });

  it('delegates common events and listens to others', () => {
    const code = compile('const x = <div onclick={f} onscroll={g} onkeydowncapture={h} />;');
    expect(code).toContain('.$$click = (f)');
    expect(code).toContain('"scroll", (g)');
    expect(code).toContain('"keydown", (h), true');
    expect(code).toContain('_$delegateEvents(["click"])');
  });

  it('components: getters for dynamic props, children as a function, render props passed through', () => {
    const code = compile('const x = <C a="s" b={1} c={d()} e={<i />}>{(v) => v}</C>; const y = <C>text</C>; const z = <C><b /></C>;');
    expect(code).toContain('a: "s"');
    expect(code).toContain('b: (1)');
    expect(code).toContain('get c() { return (d()); }');
    expect(code).toContain('get e() { return (_tmpl$');
    expect(code).toContain('children: ((v) => v)');
    expect(code).toContain('children: "text"');
    expect(code).toMatch(/children: \(\) => \(_tmpl\$\d\(\)\)/);
  });

  it('bind: on components and spreads merge in order', () => {
    const code = compile('const x = <C a={1} {...rest} bind:open={o} b={b()} />;');
    expect(code).toContain('_$mergeProps({ a: (1) }, (rest), _$bound("open", (o)), { get b() { return (b()); } })');
  });

  it('directives', () => {
    const code = compile('const x = <div class:on={a()} style:top={t()} use:tip={p()} transition:fly={{ y: 8 }} ref={r} bind:value={v} />;');
    expect(code).toContain('_$toggleClass(_el$1, "on", () => (a()))');
    expect(code).toContain('_$styleProp(_el$1, "top", () => (t()))');
    expect(code).toContain('_$use(_el$1, tip, () => (p()))');
    expect(code).toContain('_$transition(_el$1, fly, () => ({ y: 8 }), 3)');
    expect(code).toContain('_$ref(_el$1, (r))');
    expect(code).toContain('_$bindValue(_el$1, (v))');
  });

  it('css`` escapes are undone', () => {
    const src = "import { css } from 'fw'; css`/* \\`x\\` \\${y} */ .a{content:'\\\\201C'}`;";
    expect(transformJsx(src, 'x.tsx', { scope: 's' }).css).toBe("/* `x` ${y} */ .a{content:'\\201C'}");
  });
});

describe('source map', () => {
  it('maps an expression inside JSX to its original position', () => {
    const src = 'const a = 1;\nconst x = <div title={value()}>\n  {other()}\n</div>;\n';
    const out = transformJsx(src, 'x.tsx', {});
    const consumer = new SourceMapConsumer(out.map as never);
    const lines = out.code.split('\n');
    const line = lines.findIndex((l) => l.includes('other()'));
    const col = lines[line]!.indexOf('other()');
    const pos = consumer.originalPositionFor({ line: line + 1, column: col });
    expect(pos.line).toBe(3);
    expect(pos.column).toBe(3);
  });
});
