// fw port of Ident.svelte.
/**
 * A model, checkpoint or run identifier: an architecture (`folo-s`), the checkpoint a run started from
 * (`folo-s-coco80`, `best.folo`) or a server id (`r_…`). Always mono, wherever it appears (DESIGN_DIRECTION §2.3);
 * a run's, model's or dataset's own name, as its row, title or chip label, stays in sans, and so does a named optimizer
 * ("AdamW", through `run/resolved` `optimizerLabel`): a name, not an identifier. With `href` it is a link.
 */

export interface IdentProps {
  value: string;
  href?: string;
  title?: string;
  class?: string;
  onclick?: (e: MouseEvent) => void;
}

export default function Ident(props: IdentProps) {
  return (
    <>
      {props.href ? (
        <a class={`ident mono ${props.class ?? ''}`} href={props.href} title={props.title} onclick={props.onclick}>
          {props.value}
        </a>
      ) : (
        <span class={`ident mono ${props.class ?? ''}`} title={props.title}>
          {props.value}
        </span>
      )}
    </>
  );
}
