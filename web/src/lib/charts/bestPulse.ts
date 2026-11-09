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
