// The landing page, built from README.md: the pitch, the headline numbers, the quickstart, Studio and Desktop, agent
// setup and project status. The README's sections are found by heading, so a renamed heading fails the build instead
// of silently dropping content.
import type { MarkdownIt, Token } from 'markdown-it';
import { NAV, REPO_URL, slugFor } from './config.ts';
import { ICONS } from './layout.ts';
import { relUrl } from './links.ts';
import { attr, escapeHtml, inlineText, type ParsedPage } from './markdown.ts';

interface Section {
  id: string;
  title: string;
  tokens: Token[];
}

function splitSections(tokens: Token[], level: string): { intro: Token[]; sections: Section[] } {
  const sections: Section[] = [];
  let intro: Token[] = [];
  let cur: Token[] = intro;
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === 'heading_open' && t.tag === level) {
      cur = [];
      sections.push({ id: attr(t, 'id'), title: inlineText(tokens[i + 1]).trim(), tokens: cur });
      i += 2;
      continue;
    }
    if (t.type === 'heading_open' && t.tag === 'h1') {
      i += 2;
      continue;
    }
    cur.push(t);
  }
  return { intro, sections };
}

/** The top-level blocks of a token list, each as its own token slice. */
function blocks(tokens: Token[]): Token[][] {
  const out: Token[][] = [];
  let depth = 0;
  let cur: Token[] = [];
  for (const t of tokens) {
    cur.push(t);
    depth += t.nesting;
    if (depth === 0) {
      out.push(cur);
      cur = [];
    }
  }
  return out;
}

const isType = (b: Token[], type: string) => b[0]?.type === type;
const dims = (img: Token) => {
  const w = attr(img, 'width');
  const h = attr(img, 'height');
  return w && h ? ` width="${w}" height="${h}"` : '';
};
/** A paragraph that holds nothing but one image. */
function imageOf(b: Token[]): Token | null {
  if (!isType(b, 'paragraph_open')) return null;
  const kids = b[1]?.children?.filter((c) => !(c.type === 'text' && !c.content.trim())) ?? [];
  return kids.length === 1 && kids[0].type === 'image' ? kids[0] : null;
}

export interface Landing {
  main: string;
  anchors: Set<string>;
  description: string;
}

export function renderLanding(md: MarkdownIt, page: ParsedPage, slug: string): Landing {
  const render = (tokens: Token[]) => md.renderer.render(tokens, md.options, {});
  const { intro, sections } = splitSections(page.tokens, 'h2');
  const need = (title: string) => {
    const s = sections.find((x) => x.title === title);
    if (!s) throw new Error(`README.md: the landing page needs a "## ${title}" section`);
    return s;
  };
  const anchors = new Set<string>();
  const section = (s: Section, cls: string, body: string, eyebrow = '') => {
    anchors.add(s.id);
    return `<section class="l-section ${cls}" id="${s.id}" aria-labelledby="${s.id}-h">
  <div class="l-section-head">${eyebrow ? `<p class="eyebrow">${escapeHtml(eyebrow)}</p>` : ''}<h2 id="${s.id}-h">${escapeHtml(s.title)}</h2></div>
  ${body}
</section>`;
  };

  // Hero: "**FOLO Only Looks Once.** A from-scratch ... in Rust: a YOLO-style ..." -> eyebrow, headline, lead.
  const introBlocks = blocks(intro);
  const lead = introBlocks.find((b) => isType(b, 'paragraph_open'));
  if (!lead) throw new Error('README.md: the landing page needs an opening paragraph');
  const kids = (lead[1].children ?? []).filter((c) => !(c.type === 'text' && c.content === ''));
  let eyebrow = '';
  let rest = kids;
  if (kids[0]?.type === 'strong_open' && kids[2]?.type === 'strong_close') {
    eyebrow = kids[1].content.replace(/\.$/, '');
    rest = kids.slice(3);
  }
  const plain = rest
    .map((c) => (c.type === 'text' || c.type === 'code_inline' ? c.content : c.type === 'softbreak' ? ' ' : ''))
    .join('')
    .trim();
  const colon = plain.indexOf(':');
  const headline = colon > 0 ? plain.slice(0, colon) : plain;
  const description = plain;
  const leadRest = colon > 0 ? plain.slice(colon + 1).trim() : '';
  const features = introBlocks.find((b) => isType(b, 'bullet_list_open'));
  const status = introBlocks.find((b) => isType(b, 'blockquote_open'));

  const numbers = need('Why FOLO: the numbers');
  const quick = need('Quickstart');
  const studio = need('FOLO Studio');
  const desktop = need('FOLO Desktop');
  const agent = need('Use FOLO from an agent');
  const projectStatus = need('Project status');
  const docsTable = need('Docs');

  const studioShot = blocks(studio.tokens).map(imageOf).find(Boolean);
  if (!studioShot) throw new Error('README.md: "## FOLO Studio" needs a screenshot');
  const heroShot = `<figure class="hero-shot"><img src="${attr(studioShot, 'src')}" alt="${escapeHtml(inlineText(studioShot))}"${dims(studioShot)} fetchpriority="high"></figure>`;

  const docs = relUrl(slug, slugFor('docs/QUICKSTART.md'));
  const hero = `<section class="hero" aria-labelledby="hero-title">
  <div class="hero-text">
    ${eyebrow ? `<p class="eyebrow eyebrow-accent">${escapeHtml(eyebrow)}</p>` : ''}
    <h1 id="hero-title">${escapeHtml(headline)}</h1>
    ${leadRest ? `<p class="hero-lead">${escapeHtml(leadRest.charAt(0).toUpperCase() + leadRest.slice(1))}</p>` : ''}
    <div class="hero-actions">
      <a class="btn btn-primary" href="${docs}">Get started ${ICONS.arrowRight}</a>
      <a class="btn" href="${relUrl(slug, slugFor('docs/CLI.md'))}">CLI reference</a>
      <a class="btn btn-ghost" href="${REPO_URL}">${ICONS.github} GitHub</a>
    </div>
    ${status ? `<div class="hero-status">${render(status.slice(1, -1))}</div>` : ''}
  </div>
  ${heroShot}
</section>`;

  const featureCards = features
    ? `<ul class="features" aria-label="Highlights">${blocks(features.slice(1, -1))
        .map((li) => `<li class="feature">${render(li.slice(1, -1))}</li>`)
        .join('')}</ul>`
    : '';

  // The numbers: the whole section (the comparison table and its caveats).
  const numbersHtml = section(numbers, 'l-numbers', `<div class="l-prose prose">${render(numbers.tokens)}</div>`, 'Benchmarks');

  // Quickstart: everything before its first subsection.
  const h3 = quick.tokens.findIndex((t) => t.type === 'heading_open');
  const quickTokens = h3 < 0 ? quick.tokens : quick.tokens.slice(0, h3);
  const quickHtml = section(quick, 'l-quick', `<div class="l-prose prose">${render(quickTokens)}</div>`, 'Get started');

  // Studio and Desktop: the first paragraph and a screenshot each (Studio's second one; the first is the hero's).
  const app = (s: Section, name: string, href: string, shotIndex: number) => {
    const bs = blocks(s.tokens);
    const shots = bs.map(imageOf).filter((x): x is Token => !!x);
    const shot = shots[Math.min(shotIndex, shots.length - 1)];
    const at = bs.findIndex((b) => isType(b, 'paragraph_open') && !imageOf(b));
    if (!shot || at < 0) throw new Error(`README.md: "## ${s.title}" needs a screenshot and a paragraph`);
    // A paragraph that introduces a list ("It has these screens:") keeps its list.
    const para = [...bs[at], ...(/:\s*$/.test(inlineText(bs[at][1])) && bs[at + 1] && isType(bs[at + 1], 'bullet_list_open') ? bs[at + 1] : [])];
    anchors.add(s.id);
    return `<article class="app-card" id="${s.id}" aria-labelledby="${s.id}-h">
  <img src="${attr(shot, 'src')}" alt="${escapeHtml(inlineText(shot))}"${dims(shot)} loading="lazy" decoding="async">
  <div class="app-card-body">
    <h3 id="${s.id}-h">${escapeHtml(s.title)}</h3>
    <div class="prose">${render(para)}</div>
    <a class="more" href="${href}">${escapeHtml(name)} README ${ICONS.external}</a>
  </div>
</article>`;
  };
  const appsHtml = `<section class="l-section l-apps" aria-labelledby="apps-h">
  <div class="l-section-head"><p class="eyebrow">Apps</p><h2 id="apps-h">Studio and Desktop</h2></div>
  <div class="apps">${app(studio, 'Studio', `${REPO_URL}/blob/main/web/README.md`, 1)}${app(desktop, 'Desktop', `${REPO_URL}/blob/main/apps/desktop/README.md`, 0)}</div>
</section>`;

  // Agents: the MCP paragraph and the first code block.
  const ab = blocks(agent.tokens);
  const agentPara = ab.find((b) => isType(b, 'paragraph_open'));
  const agentCode = ab.find((b) => isType(b, 'fence'));
  const agentHtml = section(
    agent,
    'l-agent',
    `<div class="l-prose prose">${agentPara ? render(agentPara) : ''}${agentCode ? render(agentCode) : ''}<p><a class="more" href="${relUrl(slug, slugFor('docs/AGENT_COOKBOOK.md'))}">Agent cookbook ${ICONS.arrowRight}</a> <a class="more" href="${relUrl(slug, slugFor('docs/MCP.md'))}">MCP reference ${ICONS.arrowRight}</a></p></div>`,
    'Agents',
  );

  // Every page, grouped as in the sidebar, described by the README's docs table where it has a row.
  const blurbs = new Map<string, string>();
  for (const t of docsTable.tokens) {
    if (t.type !== 'inline') continue;
    const link = t.children?.find((c) => c.type === 'link_open');
    if (link) blurbs.set(attr(link, 'href'), '');
  }
  const rows = blocks(docsTable.tokens).find((b) => isType(b, 'table_open')) ?? [];
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].type !== 'tr_open') continue;
    const cells = rows.slice(i).filter((t) => t.type === 'inline').slice(0, 2);
    const link = cells[0]?.children?.find((c) => c.type === 'link_open');
    const href = link && attr(link, 'href');
    if (href && cells[1]) blurbs.set(href, inlineText(cells[1]));
  }
  const docsHtml = `<section class="l-section l-docs" id="${docsTable.id}" aria-labelledby="${docsTable.id}-h">
  <div class="l-section-head"><p class="eyebrow">Documentation</p><h2 id="${docsTable.id}-h">Explore the docs</h2></div>
  <div class="doc-groups">${NAV.map(
    (g) => `<div class="doc-group"><h3>${escapeHtml(g.title)}</h3><ul>${g.pages
      .map((p) => {
        const href = relUrl(slug, slugFor(p.src));
        const blurb = blurbs.get(href);
        return `<li><a href="${href}"><span class="doc-link-label">${escapeHtml(p.label)}</span>${blurb ? `<span class="doc-link-blurb">${escapeHtml(blurb)}</span>` : ''}</a></li>`;
      })
      .join('')}</ul></div>`,
  ).join('')}</div>
</section>`;
  anchors.add(docsTable.id);

  const statusHtml = section(projectStatus, 'l-status', `<div class="l-prose prose">${render(projectStatus.tokens)}</div>`, 'Status');

  const main = `<div class="landing">
${hero}
${featureCards}
${numbersHtml}
${quickHtml}
${appsHtml}
${agentHtml}
${docsHtml}
${statusHtml}
</div>`;
  return { main, anchors, description };
}
