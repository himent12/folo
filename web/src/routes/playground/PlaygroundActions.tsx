/**
 * Playground page header actions: Compare A/B toggle, one-click Export PNG and Export JSON (when results exist), and
 * Open image on non-phone viewports. Labels shorten cleanly on narrow viewports so header actions stay on one row.
 */
import Download from '$icons/download';
import FileJson from '$icons/file-json';
import GitCompareArrows from '$icons/git-compare-arrows';
import ImagePlus from '$icons/image-plus';
import Button from '$lib/ui/Button';
import { tooltip } from '$lib/ui/tooltip';

export interface PlaygroundActionsProps {
  hasModels: boolean;
  phone: boolean;
  narrow: boolean;
  compare: boolean;
  canExport: boolean;
  exportingPng: boolean;
  onToggleCompare: () => void;
  onExportPng: () => void;
  onExportJson: () => void;
  onOpenImage: () => void;
}

export default function PlaygroundActions(props: PlaygroundActionsProps) {
  return (
    <>
      {props.hasModels ? (
        <Button
          icon={GitCompareArrows}
          variant={props.compare ? 'primary' : 'secondary'}
          aria-pressed={props.compare}
          onclick={props.onToggleCompare}
        >
          Compare A/B
        </Button>
      ) : null}
      {props.canExport ? (
        <>
          <span use:tooltip={'Export annotated PNG'}>
            <Button
              icon={Download}
              disabled={props.exportingPng}
              aria-label="Export annotated PNG"
              onclick={props.onExportPng}
            >
              {props.narrow ? 'PNG' : 'Export PNG'}
            </Button>
          </span>
          <span use:tooltip={'Export detections JSON'}>
            <Button icon={FileJson} aria-label="Export detections JSON" onclick={props.onExportJson}>
              {props.narrow ? 'JSON' : 'Export JSON'}
            </Button>
          </span>
        </>
      ) : null}
      {props.hasModels && !props.phone ? (
        <Button icon={ImagePlus} onclick={props.onOpenImage}>
          Open image
        </Button>
      ) : null}
    </>
  );
}
