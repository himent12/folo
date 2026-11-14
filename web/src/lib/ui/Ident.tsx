/**
 * A model, checkpoint or run identifier: an architecture (`folo-s`), the checkpoint a run started from
 * (`folo-s-coco80`, `best.folo`) or a server id (`r_…`). With `href` it is a link.
 *
 * Names and identifiers (design/UNIFIED.md §2): an architecture, and a model or checkpoint named for one, is a name, set
 * in sans like a run's or a model's own name (mono put a second typeface into every table). Mono is for paths, file
 * names, server ids, fingerprints, commands and logs. `kind` says which; without it the value decides
 * (`identKind`): a FOLO or YOLO architecture or a model named after one is a name, anything else stays mono.
 */

export interface IdentProps {
  value: string;
  href?: string;
  title?: string;
  class?: string;
  onclick?: (e: MouseEvent) => void;
  /** A name (sans) or an identifier (mono); default: from the value (`identKind`). */
  kind?: 'name' | 'id';
}

/**
 * Whether a value is set as a name or an identifier: `folo-s`, `yolo26n` and `folo-s-coco80` are names; a path or a
 * file (`/data/best.folo`, `best.folo`), a server id (`run_7d1e`) or a fingerprint (`9f2c…`) is an identifier.
 */
export function identKind(value: string): 'name' | 'id' {
  return /^(folo|yolo)[a-z0-9]*(-[a-z0-9]+)*$/i.test(value) ? 'name' : 'id';
}

export default function Ident(props: IdentProps) {
  const cls = () => `ident ${(props.kind ?? identKind(props.value)) === 'id' ? 'mono' : 'name'} ${props.class ?? ''}`;
  return (
    <>
      {props.href ? (
        <a class={cls()} href={props.href} title={props.title} onclick={props.onclick}>
          {props.value}
        </a>
      ) : (
        <span class={cls()} title={props.title}>
          {props.value}
        </span>
      )}
    </>
  );
}
