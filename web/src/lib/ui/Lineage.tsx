// fw port of Lineage.svelte.
/**
 * The checkpoint a run started from, named one way on every screen (header, Config, Model's Details, Compare): the
 * model's name (an identifier, so mono through ui/Ident) linked to it, with its server id in the tooltip; a file by
 * its file name, the full path in the tooltip; "From scratch" in sans. Resolve the origin with `$lib/lineage`.
 */
import { css } from 'fw';
import { FROM_SCRATCH, originTip, type Origin } from '$lib/lineage';
import { href } from '$lib/router';
import Ident from './Ident';

export interface LineageProps {
  origin: Origin;
  link?: boolean;
  scratch?: string;
}

export default function Lineage(props: LineageProps) {
  const link = () => props.link ?? true;
  return (
    <>
      {props.origin.kind === 'scratch' ? (
        (props.scratch ?? FROM_SCRATCH)
      ) : props.origin.kind === 'model' ? (
        <Ident
          value={props.origin.label}
          title={originTip(props.origin)}
          class={link() ? 'lineage-link' : ''}
          href={link() ? href(`/models/${encodeURIComponent((props.origin as { id: string }).id)}`) : undefined}
        />
      ) : (
        <Ident value={props.origin.label} title={originTip(props.origin)} />
      )}
    </>
  );
}

css`
  :global(a.ident.lineage-link) {
    color: inherit;
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  :global(a.ident.lineage-link:hover) {
    text-decoration-color: var(--text-secondary);
  }
`;
