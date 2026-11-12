import { describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import { deleteWords } from './deleteWords';

const model = (id: string, name: string, run_id: string | null, bytes: number, source?: Model['source']) =>
  ({ id, name, run_id, bytes, source }) as Model;

describe('deleteWords: the Delete run confirm tells what goes', () => {
  it('names the models saved from the run, their size on disk, and says so on the button', () => {
    const models = [
      model('m_f6f1', 'shapes-n-best', 'run_f6f1', 6 * 2 ** 20, 'run'),
      model('m_f6f1-last', 'shapes-n-last', 'run_f6f1', 6 * 2 ** 20, 'run'),
      model('m_other', 'other-best', 'run_0000', 2 ** 20, 'run'),
    ];
    const w = deleteWords({ id: 'run_f6f1' }, models);
    expect(w.description).toBe("Deletes the run, its metric history and its weights on disk: 2 models (shapes-n-best, shapes-n-last), 12.0 MiB. This can't be undone.");
    expect(w.button).toBe('Delete run and 2 models');
    expect(w.description).not.toMatch(/not touched/);
  });

  it('leaves out copies that are files of their own, and says when there are no weights', () => {
    const w = deleteWords({ id: 'run_f6f1' }, [model('copy', 'copy', 'run_f6f1', 10, 'file')]);
    expect(w).toEqual({ description: "Deletes the run and its metric history. It has no saved weights. This can't be undone.", button: 'Delete run' });
    expect(deleteWords({ id: 'run_f6f1' }, [model('m_f6f1', 'n-best', 'run_f6f1', 1000, 'run')], 'metric').button).toBe('Delete run and 1 model');
  });

  it('still says the weights and models go while the models list loads, never that there are none', () => {
    expect(deleteWords({ id: 'run_f6f1' }, undefined)).toEqual({
      description: "Deletes the run and its metric history. Its weights and models are deleted too. This can't be undone.",
      button: 'Delete run',
    });
  });
});
