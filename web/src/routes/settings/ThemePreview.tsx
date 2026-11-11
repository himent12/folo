/**
 * A tiny Studio window drawn in one theme's colours, whatever the current theme is. The app's CSS variables only
 * describe the active theme, so the colours come straight from the design tokens (design/tokens.json). Drawn in the
 * theme's ink only: a preview shows surfaces and text, not data colours.
 */
import { css } from 'fw';
import tokens from '../../../../design/tokens.json';

/** `half`: half a card (System's two themes side by side), with tighter insets so the window still fits a phone's card. */
export default function ThemePreview(props: { theme: 'dark' | 'light'; half?: boolean }) {
  const c = () => tokens.color[props.theme];
  return (
    <span
      class="win"
      class:half={props.half ?? false}
      aria-hidden="true"
      style={{ '--app': c().bg.app, '--panel': c().bg.panel, '--line': c().line.default, '--t1': c().text.primary, '--t3': c().text.tertiary }}
    >
      <span class="side">
        <span class="logo"></span>
        <span class="nav on"></span>
        <span class="nav"></span>
        <span class="nav"></span>
      </span>
      <span class="main">
        <span class="title"></span>
        <span class="card">
          <svg viewBox="0 0 60 24" preserveAspectRatio="none">
            <polyline points="0,22 10,15 20,11 30,8 40,6 50,5 60,4" fill="none" stroke="var(--t1)" stroke-width="1.5" vector-effect="non-scaling-stroke" />
            <polyline points="0,23 10,19 20,16 30,14 40,13 50,12 60,11" fill="none" stroke="var(--t3)" stroke-width="1.5" vector-effect="non-scaling-stroke" />
          </svg>
        </span>
      </span>
    </span>
  );
}

css`
  .win {
    display: flex;
    width: 100%;
    height: 100%;
    background: var(--app);
    overflow: hidden;
  }
  .side {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 30%;
    padding: 8px 6px;
    background: var(--panel);
    border-right: 1px solid var(--line);
  }
  .logo {
    width: 10px;
    height: 10px;
    margin-bottom: 4px;
    border-radius: 2px;
    border: 1.5px solid var(--t1);
  }
  .nav {
    height: 4px;
    width: 80%;
    border-radius: 2px;
    background: var(--t3);
    opacity: 0.5;
  }
  .nav.on {
    background: var(--t1);
    opacity: 0.9;
  }
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px;
  }
  .title {
    width: 40%;
    height: 5px;
    border-radius: 2px;
    background: var(--t1);
  }
  .card {
    flex: 1;
    display: flex;
    padding: 6px;
    border-radius: 3px;
    background: var(--panel);
    border: 1px solid var(--line);
  }
  .half .side {
    padding: 6px 4px;
  }
  .half .logo {
    width: 8px;
    height: 8px;
  }
  .half .main {
    padding: 6px;
  }
  .side,
  .main {
    min-width: 0;
  }
  .logo {
    max-width: 100%;
  }
  svg {
    width: 100%;
    height: 100%;
    max-width: none;
  }
`;
