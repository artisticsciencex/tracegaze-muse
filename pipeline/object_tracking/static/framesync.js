"use strict";
/**
 * FrameSync — presentation-truth state machine for the object-tracking GUI.
 *
 * Root cause it exists to fix: the GUI used to derive the mask frame from
 * video.currentTime and painted masks on timeupdate/seeked/timer ticks —
 * none of which prove the browser has PRESENTED the expected VFR frame, so
 * frame-N masks could be painted over frame N-1 pixels.
 *
 * Rules enforced here:
 *  - requestedFrame changes immediately; displayedFrame changes ONLY when
 *    the matching image has actually been obtained (exact still) or a
 *    presented-frame callback reported it (playback)
 *  - masks/points/readout/reference-frame operations key off displayedFrame
 *  - every seek bumps seekGeneration; any image/mask response from an older
 *    generation is discarded, so background + masks + readout always commit
 *    atomically for the same frame
 *  - annotation is disabled while presentation is unresolved
 *
 * Dependency-injected (fetchImage/fetchMasks/callbacks) so the sequencing
 * logic runs unmodified under node for regression tests.
 */
class FrameSync {
  /**
   * @param opts {
   *   ptsTable: number[] — authoritative decoded-frame PTS (file domain)
   *   fetchImage(frame, gen) -> Promise<any>   exact still for frame
   *   fetchMasks(frame, gen) -> Promise<any>   masks for frame
   *   onSeeking(frame, gen)                    entering unresolved state
   *   onCommit({frame, image, masks, gen, mediaTime})  atomic commit
   *   onPlaybackFrame({frame, masks, mediaTime})       presented frame
   *   onError(err, gen)
   * }
   */
  constructor(opts) {
    this.o = opts;
    this.pts = opts.ptsTable || [];
    const gaps = [];
    for (let i = 1; i < this.pts.length; i++)
      gaps.push(this.pts[i] - this.pts[i - 1]);
    this.snapTol = gaps.length ? Math.min.apply(null, gaps) / 2 : 1 / 60;
    this.requestedFrame = null;
    this.displayedFrame = null;
    this.displayedMediaTime = null;
    this.seekGeneration = 0;
    this.framePresentationReady = false;
    this.mode = "paused";              // "paused" | "playing"
  }

  nFrames() { return this.pts.length; }

  /** nearest-PTS mapping with half-frame tolerance; null = unmappable */
  frameForMediaTime(t) {
    if (!this.pts.length) return null;
    let lo = 0, hi = this.pts.length - 1;
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1;
      if (this.pts[m] <= t) lo = m; else hi = m - 1;
    }
    let best = lo;
    if (lo + 1 < this.pts.length
        && Math.abs(this.pts[lo + 1] - t) < Math.abs(t - this.pts[lo]))
      best = lo + 1;
    if (Math.abs(this.pts[best] - t) > this.snapTol + 1e-9) return null;
    return best;
  }

  /** floor mapping used for coarse scrubbing targets (always resolves) */
  floorFrameForTime(t) {
    if (t <= 0) return 0;
    let lo = 0, hi = this.pts.length - 1;
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1;
      if (this.pts[m] <= t) lo = m; else hi = m - 1;
    }
    return lo;
  }

  /**
   * Exact paused-frame request. Resolves to:
   *   {committed: true, frame}  — background+masks committed atomically
   *   {stale: true}             — superseded by a newer request
   * Throws only through onError (also returns {error}).
   */
  async requestFrame(n) {
    n = Math.max(0, Math.min(this.nFrames() - 1, n | 0));
    this.requestedFrame = n;
    const gen = ++this.seekGeneration;
    this.mode = "paused";
    this.framePresentationReady = false;
    this.o.onSeeking && this.o.onSeeking(n, gen);
    let image, masks;
    try {
      image = await this.o.fetchImage(n, gen);
      if (gen !== this.seekGeneration) return { stale: true, gen };
      masks = await this.o.fetchMasks(n, gen);
      if (gen !== this.seekGeneration) return { stale: true, gen };
    } catch (err) {
      if (gen !== this.seekGeneration) return { stale: true, gen };
      this.o.onError && this.o.onError(err, gen);
      return { error: String(err), gen };
    }
    // atomic commit: one generation, one frame, image + masks together
    this.displayedFrame = n;
    this.displayedMediaTime = this.pts[n];
    this.framePresentationReady = true;
    this.o.onCommit && this.o.onCommit(
      { frame: n, image, masks, gen, mediaTime: this.pts[n] });
    return { committed: true, frame: n, gen };
  }

  /** playback: called ONLY from a presented-frame callback (rVFC) */
  presentedFrame(mediaTime, masksLookup) {
    if (this.mode !== "playing") return null;
    const f = this.frameForMediaTime(mediaTime);
    if (f === null) return null;       // clock skew beyond tolerance: skip
    this.displayedFrame = f;
    this.displayedMediaTime = mediaTime;
    this.requestedFrame = f;
    const masks = masksLookup ? masksLookup(f) : null;
    this.o.onPlaybackFrame
      && this.o.onPlaybackFrame({ frame: f, masks, mediaTime });
    return f;
  }

  enterPlayback() {
    this.mode = "playing";
    this.framePresentationReady = true;  // playback truth comes from rVFC
    ++this.seekGeneration;               // void in-flight paused requests
  }

  /** pausing hands the last presented frame to the exact-still path */
  async exitPlayback() {
    this.mode = "paused";
    const target = this.displayedFrame !== null ? this.displayedFrame : 0;
    return this.requestFrame(target);
  }

  annotationAllowed() {
    return this.mode === "paused" && this.framePresentationReady
      && this.displayedFrame === this.requestedFrame;
  }

  /** true only when a screenshot would show one consistent frame */
  screenshotReady() {
    return this.annotationAllowed();
  }

  debugState() {
    return {
      requestedFrame: this.requestedFrame,
      displayedFrame: this.displayedFrame,
      displayedMediaTime: this.displayedMediaTime,
      seekGeneration: this.seekGeneration,
      framePresentationReady: this.framePresentationReady,
      mode: this.mode,
    };
  }
}

if (typeof module !== "undefined" && module.exports)
  module.exports = { FrameSync };
