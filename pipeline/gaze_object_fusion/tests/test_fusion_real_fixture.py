"""Real-fixture fusion test against the user-verified milestone artifact.

Skipped when the private fixture is absent (it is never in git). The full
publish + demo render happens in the real-verification run, not here — this
test proves the assignment math on real data with an INDEPENDENT
brute-force re-implementation of the rule.
"""

import json
import random
from pathlib import Path

import pytest

REAL = Path("data/neon-data/"
            "lab_study/participant_01/"
            "2024-01-01_00-00-00-0000abcd_participant_a_trim_f3000-f3299"
            "_tracked_objects")

pytestmark = pytest.mark.skipif(not REAL.is_dir(),
                                reason="real milestone fixture not present")


def test_real_assignment_full_coverage_and_brute_force_agreement():
    from gaze_object_fusion.fusion import FusionInputs, fuse

    inputs = FusionInputs(REAL)
    fused, cells, validation, extras = fuse(inputs)
    assert all(v["ok"] for v in validation), \
        [v for v in validation if not v["ok"]]

    # the trim retained exactly the in-interval samples -> all assigned
    assert len(fused) == 1998
    assert (fused["assignment"] == "assigned").all()
    fi = fused["frame_index"].astype("int64")
    assert fi.min() >= 0 and fi.max() <= 299

    # every offset lies inside its (half-open) cell
    for _, r in fused.iterrows():
        lo, hi = cells.cell_bounds(int(r["frame_index"]))
        assert lo <= int(r["gaze_ts_ns"]) < hi

    # INDEPENDENT brute-force: linear scan over midpoint boundaries,
    # re-derived here from the raw CSVs + trim manifest (not from cells)
    world = []
    import csv
    with open(inputs.world_path) as f:
        for row in csv.DictReader(f):
            world.append(int(row["timestamp [ns]"]))
    sel = json.loads((inputs.parent_dir / "trim_manifest.json")
                     .read_text())["selection"]
    bounds = [sel["absolute_start_ns"]]
    for a, b in zip(world, world[1:]):
        bounds.append((a + b) // 2)
    bounds.append(sel["absolute_end_boundary_ns"])

    def brute(ts):
        for i in range(len(world)):
            if bounds[i] <= ts < bounds[i + 1]:
                return i
        return None

    rng = random.Random(7)
    rows = fused.sample(n=200, random_state=7)
    for _, r in rows.iterrows():
        assert brute(int(r["gaze_ts_ns"])) == int(r["frame_index"])

    # ~200 Hz gaze on ~30 Hz frames: every frame should carry samples on
    # this clip, and the demo rule selects exactly one row per such frame
    assert len(extras["demo_rows"]) == 300
    demo = fused[fused["demo_selected"]]
    assert len(demo) == 300
    assert demo["frame_index"].is_unique

    # milestone objects appear as hit targets
    names = set()
    for h in fused["hit_object_names"]:
        names.update(h)
    assert names <= {"left-hand", "right-hand", "tool-1"}

    # no anomalous world gaps on the verified clip
    assert cells.anomalous_gaps() == []
