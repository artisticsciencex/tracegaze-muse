// FrameSync race regressions (run under node via test_framesync_js.py).
// Each scenario reproduces a way the old currentTime-driven GUI could paint
// frame-N masks over a different visible frame.
"use strict";
const assert = require("assert");
const path = require("path");
const { FrameSync } = require(path.join(__dirname, "..", "static",
                                        "framesync.js"));

// VFR pts table WITH the fixture's real 16 ms container offset
const PTS = [];
let t = 0.016;
const GAPS = [0.050, 0.050, 0.033, 0.033, 0.014, 0.040, 0.033, 0.033, 0.034];
PTS.push(t);
for (const g of GAPS) PTS.push(t += g);

const sleep = ms => new Promise(r => setTimeout(r, ms));

function harness(imageDelays = {}, maskDelays = {}) {
  const log = [];
  const fs = new FrameSync({
    ptsTable: PTS,
    fetchImage: async (n) => {
      await sleep(imageDelays[n] || 1);
      return { img: n };
    },
    fetchMasks: async (n) => {
      await sleep(maskDelays[n] || 1);
      return [{ frame: n }];
    },
    onSeeking: (n) => log.push(["seeking", n]),
    onCommit: ({ frame, image, masks }) => {
      assert.strictEqual(image.img, frame,
        "committed image must belong to the committed frame");
      assert.strictEqual(masks[0].frame, frame,
        "committed masks must belong to the committed frame");
      log.push(["commit", frame]);
    },
    onPlaybackFrame: ({ frame }) => log.push(["played", frame]),
    onError: (e) => log.push(["error", String(e)]),
  });
  return { fs, log };
}

(async () => {
  // 1) delayed seek completion: slow frame still commits atomically
  {
    const { fs, log } = harness({ 3: 40 });
    const r = await fs.requestFrame(3);
    assert.ok(r.committed && fs.displayedFrame === 3);
    assert.deepStrictEqual(log.pop(), ["commit", 3]);
    assert.ok(fs.annotationAllowed());
  }

  // 2) rapid 35->36->20 pattern (here 5->6->2): only the LAST commits;
  //    stale image AND stale mask responses are discarded
  {
    const { fs, log } = harness({ 5: 60, 6: 30 }, { 5: 60, 6: 30 });
    const p5 = fs.requestFrame(5);
    const p6 = fs.requestFrame(6);
    const p2 = fs.requestFrame(2);
    const [r5, r6, r2] = await Promise.all([p5, p6, p2]);
    assert.ok(r5.stale, "frame-5 response must be discarded");
    assert.ok(r6.stale, "frame-6 response must be discarded");
    assert.ok(r2.committed && fs.displayedFrame === 2);
    const commits = log.filter(e => e[0] === "commit");
    assert.deepStrictEqual(commits, [["commit", 2]],
      "exactly one commit, for the final request");
    await sleep(80);   // even after slow responses land: no late commits
    assert.deepStrictEqual(log.filter(e => e[0] === "commit"),
      [["commit", 2]]);
    assert.strictEqual(fs.displayedFrame, 2);
  }

  // 3) stale image arriving late never overwrites a newer displayed frame
  {
    const { fs, log } = harness({ 7: 50, 1: 1 });
    const p7 = fs.requestFrame(7);
    await sleep(5);
    await fs.requestFrame(1);
    assert.strictEqual(fs.displayedFrame, 1);
    const r7 = await p7;
    assert.ok(r7.stale);
    assert.strictEqual(fs.displayedFrame, 1);
  }

  // 4) annotation disabled while a seek is unresolved
  {
    const { fs } = harness({ 4: 50 });
    const p = fs.requestFrame(4);
    assert.strictEqual(fs.framePresentationReady, false);
    assert.strictEqual(fs.annotationAllowed(), false,
      "no annotation while presentation unresolved");
    await p;
    assert.ok(fs.annotationAllowed());
  }

  // 5) playback: displayedFrame comes ONLY from presented mediaTime, with
  //    nearest-PTS mapping across VFR gaps and the 16 ms offset preserved
  {
    const { fs, log } = harness();
    await fs.requestFrame(0);
    fs.enterPlayback();
    assert.strictEqual(fs.presentedFrame(PTS[4] + 0.004, f => [{ frame: f }]),
      4, "4 ms skew inside the 7 ms half-gap tolerance maps to frame 4");
    assert.strictEqual(fs.displayedFrame, 4);
    // skew beyond the smallest half-gap (14 ms gap -> 7 ms tol) is refused
    assert.strictEqual(fs.presentedFrame(PTS[5] + 0.009, () => []), null,
      "unmappable mediaTime must not change displayedFrame");
    assert.strictEqual(fs.displayedFrame, 4);
  }

  // 6) entering playback voids in-flight paused seeks
  {
    const { fs, log } = harness({ 8: 40 });
    const p = fs.requestFrame(8);
    fs.enterPlayback();
    const r = await p;
    assert.ok(r.stale, "paused seek superseded by playback must be stale");
    assert.deepStrictEqual(log.filter(e => e[0] === "commit"), []);
  }

  // 7) pause -> exact-still path for the resolved displayed frame
  {
    const { fs, log } = harness();
    await fs.requestFrame(0);
    fs.enterPlayback();
    fs.presentedFrame(PTS[6], () => []);
    const r = await fs.exitPlayback();
    assert.ok(r.committed && r.frame === 6);
    assert.ok(fs.screenshotReady(),
      "screenshot-ready only after the exact-frame commit");
  }

  // 8) new-object semantics: displayedFrame (not requestedFrame) is the
  //    only frame annotation may bind to while a seek is pending
  {
    const { fs } = harness({ 9: 50 });
    await fs.requestFrame(2);
    const pending = fs.requestFrame(9);      // not resolved yet
    assert.strictEqual(fs.requestedFrame, 9);
    assert.strictEqual(fs.displayedFrame, 2);
    assert.strictEqual(fs.annotationAllowed(), false);
    await pending;
    assert.strictEqual(fs.displayedFrame, 9);
  }

  // 9) recording switch mid-flight: a NEW FrameSync (new ptsTable) never
  //    receives old commits; old instance's late responses are stale
  {
    const a = harness({ 5: 40 });
    const pOld = a.fs.requestFrame(5);
    const b = harness();                     // fresh instance = new recording
    await b.fs.requestFrame(1);
    a.fs.seekGeneration++;                   // loadRec voids old instance
    const rOld = await pOld;
    assert.ok(rOld.stale);
    assert.strictEqual(b.fs.displayedFrame, 1);
    assert.deepStrictEqual(a.log.filter(e => e[0] === "commit"), []);
  }

  // 10) image fetch failure surfaces as error, never a mismatched commit
  {
    const fs = new FrameSync({
      ptsTable: PTS,
      fetchImage: async () => { throw new Error("boom"); },
      fetchMasks: async () => [],
      onError: () => {},
    });
    const r = await fs.requestFrame(3);
    assert.ok(r.error);
    assert.strictEqual(fs.displayedFrame, null);
    assert.strictEqual(fs.annotationAllowed(), false);
  }

  console.log("FRAMESYNC_RACES_OK (10 scenarios)");
})().catch(e => { console.error(e); process.exit(1); });
