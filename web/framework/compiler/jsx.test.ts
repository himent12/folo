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

  it('merges static text and expressions into one text node, as Svelte does', () => {
    const code = compile('const x = <p>Epoch {a()}/{b}</p>;');
    expect(templates(code)).toEqual(['<p> </p>']);
    expect(code).toContain('_$run(_el$2, ["Epoch ", () => (a()), "/", () => (b)]);');
    expect(templates(compile('const x = <p>{a()}{b()}</p>;'))).toEqual(['<p> </p>']);
    expect(templates(compile('const x = <p><b />{a()}</p>;'))).toEqual(['<p><b></b></p>']);
    // An expression with JSX in it is a block, never part of a run; the text around it keeps its own nodes.
    expect(templates(compile('const x = <p>a{c() ? <b /> : null}c</p>;'))).toEqual(['<p>a<!>c</p>', '<b></b>']);
    expect(compile('const x = <>Save {n()}</>;')).toContain('_$runNode(["Save ", () => (n())])');
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

  it('memoizes nested conditional and logical branches and avoids leaking falsy static && operands', () => {
    const nested = compile('const x = <div>{a() && (b() ? <b /> : <i />)}{c() ? (d() && <span />) : null}</div>;');
    expect(nested.match(/_\$cond\(/g)?.length).toBe(4);
    const staticAnd = compile('const x = <div>{n && <i />}{m && label()}</div>;');
    expect(staticAnd).toMatch(/\(\(n\) \? _tmpl\$\d+\(\) : null\)/);
    expect(staticAnd).toContain('((m) ? (label()) : null)');
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
    expect(code).toContain('get c() { return _$propMemo(this, "c", () => (d())); }');
    expect(code).toContain('get e() { return _$propJsx(this, "e", () => (_tmpl$');
    expect(code).toContain('children: ((v) => v)');
    expect(code).toContain('children: "text"');
    expect(code).toMatch(/children: \(\) => \(_tmpl\$\d\(\)\)/);
  });

  it('bind: on components and spreads merge in order', () => {
    const code = compile('const x = <C a={1} {...rest} bind:open={o} b={b()} {...more()} />;');
    expect(code).toContain('_$mergeProps({ a: (1) }, (rest), _$bound("open", (o)), { get b() { return _$propMemo(this, "b", () => (b())); } }, _$derive(() => (more())))');
  });

  it('memoizes props as Svelte does: calls and non-trivial reads, not identifiers or member reads', () => {
    const code = compile('const x = <C a={p.x} b={p.y ? null : p.z} c={`${p.w}px`} d={`${n}`} e={f(1)} f={<i />} />;');
    expect(code).toContain('get a() { return (p.x); }');
    expect(code).toContain('get b() { return _$propMemo(this, "b"');
    expect(code).toContain('get c() { return (`${p.w}px`); }');
    expect(code).toContain('d: (`${n}`)');
    expect(code).toContain('get e() { return _$propMemo(this, "e"');
    expect(code).toMatch(/get f\(\) \{ return _\$propJsx\(this, "f", \(\) => \(_tmpl/);

    const condJsx = compile('const x = <C a={flag() ? <b /> : null} b={() => <b />} />;');
    expect(condJsx).toContain('get a() { return _$propJsx(this, "a"');
    expect(condJsx).toContain('b: (() =>');
    expect(condJsx).not.toContain('propJsx(this, "b"');
  });

  it('booleans on non-boolean attributes stringify, as Svelte; boolean attributes drop on false', () => {
    const code = compile('const x = <input spellcheck={false} aria-hidden={true} disabled={false} readonly={true} data-x />;');
    // A DOM property given as an expression is set as the property, not written in the template (Svelte).
    expect(templates(code)).toEqual(['<input data-x>']);
    expect(code).toContain('_$setAttribute(_el$1, "spellcheck", false)');
    expect(code).toContain('_$setAttribute(_el$1, "aria-hidden", true)');
    expect(code).toContain('_el$1.readOnly = (true);');
    expect(code).toContain('_el$1.disabled = (false);');
  });

  it('static values only a property holds are set from JS (indeterminate, muted, defaultValue, <option value={1}>)', () => {
    const code = compile('const x = <div><input indeterminate={true} /><video muted /><input defaultValue="d" /><option value={1} /><option value="a" /></div>;');
    expect(templates(code)).toEqual(['<div><input><video></video><input><option></option><option value="a"></option></div>']);
    expect(code).toMatch(/_el\$\d+\.indeterminate = \(true\);/);
    expect(code).toMatch(/_el\$\d+\.muted = true;/);
    expect(code).toMatch(/_el\$\d+\.defaultValue = "d";/);
    expect(code).toMatch(/_\$setValue\(_el\$\d+, \(1\)\);/);
  });

  it('style={...} with style: directives on one element', () => {
    const code = compile('const x = <div style:top={t()} style={s()} />;');
    expect(code).toMatch(/_\$bindStyle\(_el\$1, _v\$\d+, \(\) => \(\{ "top": _v\$\d+\(\) \}\)\)/);
  });

  it('a static style with directives is set with them, not in the template (Svelte set_style)', () => {
    const code = compile('const x = <div style="color: red" style:top={t()} style:left="1px" />;');
    expect(templates(code)).toEqual(['<div></div>']);
    expect(code).toMatch(/_\$bindStyle\(_el\$1, "color: red", \(\) => \(\{ "top": _v\$\d+\(\), "left": "1px" \}\)\)/);
    expect(templates(compile('const x = <div style="color: red" />;'))).toEqual(['<div style="color: red"></div>']);
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

  it('deduplicates identical templates per module while keeping HTML and SVG templates distinct', () => {
    const code = compile('const a = <span>{x()}</span>; const b = <span>{y()}</span>; const c = <a href="#">x</a>; const d = <svg>{ok() ? <a href="#">x</a> : null}</svg>;');
    expect(templates(code)).toEqual(['<span></span>', '<a href="#">x</a>', '<svg></svg>', '<a href="#">x</a>']);
    expect(code.match(/const _tmpl\$\d+ = /g)?.length).toBe(4);
    expect(code.match(/_tmpl\$1\(\)/g)?.length).toBe(2);
    expect(code).toContain('_$template("<a href=\\"#\\">x</a>");');
    expect(code).toContain('_$template("<a href=\\"#\\">x</a>", 1);');
  });

  it('inlines simple class: and style: bindings and falls back to bindStyle on style= or duplicate style: props', () => {
    // Static class: and style: avoid renderEffect closures.
    const staticCode = compile('const x = <div class:on={flag} style:color="red" style:opacity={alpha} style:margin="" style:padding={null} />;');
    expect(staticCode).toContain('_el$1.classList.toggle("on", !!(flag));');
    expect(staticCode).toContain('_el$1.style.setProperty("color", "red");');
    expect(staticCode).toContain('_$setStyleProp(_el$1, "opacity", (alpha));');
    expect(staticCode).not.toContain('toggleClass');
    expect(staticCode).not.toContain('bindStyle');
    expect(staticCode).not.toContain('styleProp(');
    expect(staticCode).not.toContain('"margin"');
    expect(staticCode).not.toContain('"padding"');

    // Dynamic class= with static class: avoids wrapping static directives in closures.
    const mixedClass = compile('const x = <div class={cls()} class:a class:b={flag} class:c={dyn()} />;');
    expect(mixedClass).toMatch(/const _v\$\d+ = \(flag\);/);
    expect(mixedClass).toMatch(/\{ "a": true, "b": _v\$\d+, "c": _v\$\d+\(\) \}/);

    // Distinct dynamic style: directives inline independently without bindStyle or intermediate wrappers.
    const dynStyle = compile('const x = <div style:top={t()} style:left={l()} />;');
    expect(dynStyle).toContain('_$styleProp(_el$1, "top", () => (t()));');
    expect(dynStyle).toContain('_$styleProp(_el$1, "left", () => (l()));');
    expect(dynStyle).not.toContain('bindStyle');

    // Duplicate style: property (case-insensitive) falls back to bindStyle to preserve precedence.
    const dupStyle = compile('const x = <div style:Color={a()} style:color={b()} />;');
    expect(dupStyle).toContain('_$bindStyle(');
    expect(dupStyle).not.toContain('_$styleProp(');
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
