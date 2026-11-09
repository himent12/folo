// When a live run sets a new best, its "best" ring on the chart pulses once (LineChart's Marker.pulse). This counts the
// new bests: the first value seen is where the page started, not news, and a finished run's best never pulses.

export class BestPulse {
  /** Bumped once per new best; a LineChart marker replays its pulse whenever it changes. */
  count = 0;
  #last: number | null | undefined = undefined;

  /** Feeds the run's current best epoch (null: none yet) and returns the pulse count. */
  see(best: number | null, running: boolean): number {
    if (this.#last !== undefined && best !== null && best !== this.#last && running) this.count++;
    this.#last = best;
    return this.count;
  }
}

/**
 * When a run completes while its chart is open, the best epoch's ring is focused once (LineChart's Marker.focus: lime
 * viewfinder corners close in on it). A run that was already complete when the page opened is not news.
 */
export class CompleteFocus {
  count = 0;
  #last: string | undefined = undefined;

  /** Feeds the run's status and returns the focus count, bumped when a run seen running or stopping completes. */
  see(status: string): number {
    if (status === 'completed' && (this.#last === 'running' || this.#last === 'stopping')) this.count++;
    this.#last = status;
    return this.count;
  }
}
