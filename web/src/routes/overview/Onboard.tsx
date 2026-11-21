/** A fresh install's Overview (no datasets, no runs, no models of the user's own): one path forward in three steps. */
import Plus from '$icons/plus';
import { css } from 'fw';
import { hasFeature } from '$lib/api/queue';
import { LABEL_FEATURE } from '$lib/label';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';

export default function Onboard(props: { system: { features?: readonly string[] } | undefined; systemError: unknown }) {
  return (
    <section class="onboard" aria-labelledby="onboard-title">
      <div class="ob-head">
        <h2 id="onboard-title">Train your first detector</h2> <p>Three steps from a folder of images to boxes on new ones.</p>
      </div>{' '}
      <ol class="steps">
        <li class="on">
          <span class="n">1</span>{' '}
          <div>
            <b>Add a dataset</b>
            <span>Register a YOLO or COCO folder, or generate a sample set of shapes in seconds.</span>
          </div>{' '}
          <div class="ob-actions">
            <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>
              Add dataset
            </Button>{' '}
            <Button href={href('/datasets?add=synthetic')}>Generate sample set</Button>
          </div>
          {hasFeature(props.system, LABEL_FEATURE) ? (
            // Label images lives under Datasets: the first step is where people with unlabeled images look for it.
            <p class="ob-alt">
              Have images but no labels?{' '}
              <a class="ob-link" href={href('/datasets/label')}>
                Label them with an agent
              </a>
            </p>
          ) : !props.system && !props.systemError ? (
            // Until the server says what it has (a server just started takes seconds to), the line keeps its place,
            // so it never pops in under the buttons (v-shell N11); a real server always has labeling.
            <p class="ob-alt held" aria-hidden="true">
              Have images but no labels? <span class="ob-link">Label them with an agent</span>
            </p>
          ) : null}
        </li>{' '}
        <li>
          <span class="n">2</span>{' '}
          <div>
            <b>Train</b>
            <span>Pick a model size; loss, mAP and validation samples stream in live.</span>
          </div>
        </li>{' '}
        <li>
          <span class="n">3</span>{' '}
          <div>
            <b>Inspect predictions</b>
            <span>Drop any image into the Playground and tune confidence and IoU instantly.</span>
          </div>
        </li>
      </ol>
    </section>
  );
}

css`
  /* The content width, the three steps across it where they fit (a stack below that): each step's actions sit under its
     sentence, never at the far edge of a wide row, and the card never sits alone in a corner of an empty page. */
  .onboard {
    container-type: inline-size;
    display: grid;
    gap: var(--space-5);
    padding: var(--space-4);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
  }
  .ob-head h2 {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
  }
  .ob-head p {
    color: var(--text-secondary);
  }
  .steps {
    display: grid;
    gap: var(--space-2);
  }
  /* The number sits on the title's line, top-aligned however tall the step's sentence and actions run. */
  .steps li {
    display: grid;
    grid-template-columns: 28px 1fr;
    align-items: start;
    align-content: start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4) var(--space-3) var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-subtle);
    color: var(--text-tertiary);
  }
  .steps li.on {
    border-color: var(--line-strong);
    background: var(--bg-raised);
    color: var(--text-secondary);
  }
  /* The title's first line on the number's centre (28 px circle, 20 px line). */
  .steps li div:not(.ob-actions) {
    display: grid;
    gap: 2px;
    padding-top: var(--space-1);
  }
  .steps b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .n {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid var(--line-strong);
    font-variant-numeric: tabular-nums;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .on .n {
    border-color: var(--text-primary);
    background: var(--text-primary);
    color: var(--bg-panel);
  }
  .ob-actions {
    grid-column: 2;
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    margin-top: var(--space-1);
  }
  .ob-alt {
    grid-column: 2;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .ob-alt.held {
    visibility: hidden;
  }
  .ob-link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .ob-link:hover {
    text-decoration-color: var(--text-secondary);
  }
  /* A container query, not a tier: the onboarding card sits in the content column, whose width depends on the sidebar
     as much as on the viewport; three steps need 840 px of it. */
  @container (min-width: 840px) {
    .steps {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--space-3);
    }
    .steps li {
      padding: var(--space-4);
    }
    .ob-actions {
      margin-top: var(--space-2);
    }
  }
`;
