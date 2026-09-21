"""The published site must not outrun the measurements.

The previously deployed page was the archived v1 frontend. It advertised a
"TOP-DOWN RADAR", a MiDaS depth toggle and a `BRAKE` system command, and every
control called a Flask API at `127.0.0.1:5000` that this repository does not
contain. Three of those phrases are the exact ones CLAUDE.md hard rule 13 names
as disqualifying, and the page was the most public artefact in the project.

Nothing caught it because the site lived on a detached `gh-pages` branch, where
no test, lint or review ever looked at it. The source now lives in `site/` and
these tests run over it on every push:

  * no scope-language violations;
  * no calls to a backend that does not exist;
  * every headline number traceable to `docs/benchmarks.md`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
BENCHMARKS = REPO / "docs" / "benchmarks.md"


def site_text() -> str:
    return "\n".join(p.read_text() for p in sorted(SITE.glob("*.html")))


def site_code() -> str:
    return "\n".join(p.read_text() for p in sorted(SITE.glob("*.js")))


def strip_comments(js: str) -> str:
    """Drop comments before scanning.

    app.js documents the endpoints it removed and why. The record of a deleted
    call is not a live call to it, and a test that cannot tell the difference
    would punish the explanation.
    """
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"^\s*//.*$", "", js, flags=re.M)


def test_site_sources_exist():
    """Guard the suite: an empty site/ would make everything below vacuous."""
    assert (SITE / "index.html").is_file()
    assert (SITE / "app.js").is_file()
    assert (SITE / "styles.css").is_file()


def headings_and_labels() -> str:
    """Where the page describes ITSELF: title, headings, and UI labels.

    Body prose legitimately contains the forbidden words -- the page has a
    "what this is not" table that must say "sensor fusion", and the near-field
    section correctly notes that a production stack would use radar. Banning
    them everywhere would forbid the disclaimers along with the overclaims. The
    v1 violations were all self-description: `<title>Autonomous Perception Web
    HUD`, `<h2>PERCEPTION RADAR HUD`, `<h3>TOP-DOWN RADAR`, a MiDaS toggle label.
    """
    html = re.sub(r"<!--.*?-->", "", site_text(), flags=re.S)
    parts = re.findall(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    parts += re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", html, re.S | re.I)
    parts += re.findall(
        r'class="[^"]*(?:control-label|telemetry-label)[^"]*"[^>]*>(.*?)<', html, re.S
    )
    return " ".join(parts).lower()


# The phrases hard rule 13 forbids, checked where the page describes itself.
@pytest.mark.parametrize(
    "phrase",
    ["sensor fusion", "radar", "midas", "real-time", "brake", "braking", "depth map"],
)
def test_no_scope_language_violation_in_self_description(phrase):
    assert phrase not in headings_and_labels(), (
        f"a heading, title or label says {phrase!r} -- CLAUDE.md hard rule 13. "
        "This project has no radar, no fusion, no MiDaS, and actuates nothing."
    )


def test_the_disclaimer_table_is_present():
    """The forbidden phrases must appear somewhere -- as disclaimers.

    A page that simply avoided the words would be quieter but no more honest.
    The point is to name the overclaim and say why it is false here.
    """
    visible = site_text().lower()
    assert "what this is not" in visible
    for phrase in ("sensor fusion", "autonomous emergency braking", "real-time"):
        assert phrase in visible, f"the disclaimer table no longer names {phrase!r}"


def test_the_page_never_hardcodes_a_backend_address():
    """The v1 defect, pinned.

    The old app.js fetched /api/control, /api/telemetry, /api/obstacles and
    /video_feed from a hardcoded `http://127.0.0.1:5000`. That page was public,
    so every visitor's browser tried to reach a service on *their own* machine,
    every control was dead, and nothing on the page admitted it.

    Live mode is allowed to exist -- `tools/serve.py` serves this same page --
    but only over relative, same-origin paths, so the request is meaningful
    exactly when something is there to answer it.
    """
    code = strip_comments(site_code())
    # The LOCAL_HOSTS gate necessarily names these hosts -- it exists to compare
    # location.hostname against them. What matters is that no host appears in a
    # URL the page actually requests, so the declaration is removed before the
    # scan and checked on its own in the next test.
    # Matched to end of line, not to the closing bracket: the list contains the
    # IPv6 literal "[::1]", so a `[^\]]*` scan stops inside it and matches nothing.
    code = re.sub(r"^.*LOCAL_HOSTS\s*=.*$", "", code, flags=re.M)
    for dead in ("127.0.0.1", "localhost", "http://", "https://"):
        assert dead not in code, (
            f"site/app.js hardcodes {dead!r}. Live endpoints must be relative so "
            "the published page cannot point at a machine that is not serving it."
        )
    for call in re.findall(r"fetch\(\s*[\"`']([^\"`'$]*)", code):
        assert not call.startswith("/"), (
            f"fetch target {call!r} is root-relative; it must be relative to the "
            "page so the site works from a subpath on GitHub Pages"
        )


def test_live_mode_is_gated_on_being_served_locally():
    """The published page must never probe for an API.

    Without the hostname gate, every visitor to github.io would fire a request
    at a path that does not exist there. It would fail silently, which is how
    the v1 page looked fine to its author and broken to everyone else.
    """
    code = strip_comments(site_code())
    assert "LOCAL_HOSTS" in code, "the hostname gate is gone"
    assert "location.hostname" in code
    gate = re.search(r"LOCAL_HOSTS\s*=\s*\[([^\]]*)\]", code)
    assert gate, "LOCAL_HOSTS is no longer a literal list"
    for host in ("localhost", "127.0.0.1"):
        assert host in gate.group(1), f"{host} missing from the local-host gate"
    # the gate must run before any live fetch
    gate_at = code.index("LOCAL_HOSTS.includes")
    first_api = code.index('fetch("api/')
    assert gate_at < first_api, "a live fetch is issued before the hostname check"


def test_the_page_states_the_scope():
    """Hard rule 13 fixes the scope language. It must be present, not implied."""
    visible = site_text().lower()
    for required in ("monocular", "open-loop", "ground truth"):
        assert required in visible, f"the page never says {required!r}"
    assert "does not brake" in visible or "nothing is actuated" in visible


# Headline figures, each with the benchmark row that must also contain it.
# A number on the page that benchmarks.md does not carry is exactly the drift
# this project exists to prevent.
@pytest.mark.parametrize(
    "number",
    [
        "103.56",  # sensor budget, Row 0.1
        "59.67",  # pipeline p99, Row 8.1
        "5.91",  # D_min, Row 9.8
        "6.03",  # measured near-field saturation, val
        "6.02",  # measured near-field saturation, test
        "0.737",  # held-out detection AP, Row 9.2
        "17.2",  # held-out 10-20 m MAPE, Row 9.3
        "15.6",  # held-out 20-30 m MAPE, Row 9.3
    ],
)
def test_every_headline_number_appears_in_benchmarks(number):
    assert number in site_text(), f"{number} is no longer on the page -- update this test"
    assert number in BENCHMARKS.read_text(), (
        f"the site publishes {number}, which docs/benchmarks.md does not contain. "
        "benchmarks.md is the only source of numbers (CLAUDE.md)."
    )


def test_the_site_leads_with_the_negative_result():
    """The held-out failure is the headline and must not drift below the fold.

    A page that opens with the flattering validation figure and mentions the
    refutation later would be the exact presentation this project refuses.
    """
    html = site_text()
    failure = html.lower().find("did not survive")
    assert failure != -1, "the page no longer states that the claim failed on held-out data"
    # it must appear before the latency section, i.e. in the first third
    assert failure < len(html) // 2, "the held-out failure has drifted down the page"


def test_images_referenced_by_the_page_are_present():
    for src in re.findall(r'<img[^>]+src="([^"]+)"', site_text()):
        if src.startswith("http"):
            continue
        assert (SITE / src).is_file(), f"page references {src}, which is not in site/"


def test_images_carry_alt_text():
    """A figure without alt text is unreadable to anyone using a screen reader."""
    for tag in re.findall(r"<img[^>]*>", site_text()):
        assert 'alt="' in tag and 'alt=""' not in tag, f"missing alt text: {tag[:70]}"


# ── the exported replay ──────────────────────────────────────────────────────

REPLAY = sorted((SITE / "data").glob("*/telemetry.json"))


def test_a_replay_is_exported():
    assert REPLAY, "site/data/<drive>/telemetry.json is missing; run tools/export_replay.py"


@pytest.mark.parametrize("path", REPLAY, ids=lambda p: p.parent.name)
def test_replay_frames_match_the_telemetry(path):
    """One image per telemetry record, or the player drifts out of sync.

    A missing frame would silently shift every subsequent reading onto the
    wrong picture -- the viewer would see correct numbers against the wrong
    moment, which is worse than a gap.
    """
    import json

    meta = json.loads((path.parent / "meta.json").read_text())
    telemetry = json.loads(path.read_text())
    images = sorted(path.parent.glob("*.jpg"))
    assert len(telemetry) == meta["frames"] == len(images)
    for i in range(len(images)):
        assert (path.parent / f"{i:04d}.jpg").is_file(), f"frame {i} missing"


@pytest.mark.parametrize("path", REPLAY, ids=lambda p: p.parent.name)
def test_abstentions_survive_serialisation_as_null(path):
    """`--` must reach the browser as null, never as 0.

    json.dumps writes bare NaN, which is invalid JSON and which JSON.parse
    rejects; substituting 0.0 would turn "the estimator declined" into "the
    estimator said zero", which is the fabrication hard rule 14 forbids,
    laundered through a serialiser. The exported clip must actually contain
    some abstentions, or this proves nothing.
    """
    import json

    raw = path.read_text()
    assert "NaN" not in raw, "invalid JSON: NaN leaked into the export"
    assert "Infinity" not in raw

    telemetry = json.loads(raw)
    declined = sum(1 for f in telemetry for o in f["objects"] if o["range_m"] is None)
    assert declined > 0, "no abstentions in the clip -- it cannot demonstrate the rule"

    for f in telemetry:
        for o in f["objects"]:
            for key in ("range_m", "ttc_s", "lateral_m"):
                assert o[key] is None or isinstance(o[key], (int, float))


@pytest.mark.parametrize("path", REPLAY, ids=lambda p: p.parent.name)
def test_the_exported_envelope_matches_the_shipped_config(path):
    """The clip records the band it was exported under; it must be current."""
    import json

    from aps.geometry import CredibleEnvelope

    meta = json.loads((path.parent / "meta.json").read_text())
    env = CredibleEnvelope.from_config()
    assert meta["envelope_m"] == [env.min_range_m, env.max_range_m], (
        "the exported replay was made under a different operating envelope; "
        "re-run tools/export_replay.py"
    )
    assert meta["ttc_gated_by_envelope"] is False


@pytest.mark.parametrize("path", REPLAY, ids=lambda p: p.parent.name)
def test_in_envelope_flag_agrees_with_the_envelope(path):
    """The flag the player styles on must not contradict the recorded band."""
    import json

    meta = json.loads((path.parent / "meta.json").read_text())
    lo, hi = meta["envelope_m"]
    for f in json.loads(path.read_text()):
        for o in f["objects"]:
            expected = o["range_m"] is not None and lo <= o["range_m"] <= hi
            assert o["in_envelope"] is expected, (
                f"frame {f['frame']} track {o['id']}: range {o['range_m']} "
                f"flagged in_envelope={o['in_envelope']} against band {lo}-{hi}"
            )


@pytest.mark.parametrize("path", REPLAY, ids=lambda p: p.parent.name)
def test_the_replay_is_small_enough_to_serve(path):
    """GitHub Pages is a static host, not a CDN for a video archive."""
    mb = sum(f.stat().st_size for f in path.parent.iterdir()) / 1e6
    assert mb < 40, f"{path.parent.name} is {mb:.1f} MB; reduce --frames or --width"


def test_the_player_points_at_a_drive_that_was_actually_exported():
    """`app.js` hardcodes a drive; `export_replay.py` takes it as a flag.

    Exporting a different clip without editing the player leaves a page whose
    every frame request 404s, and the only symptom is a spinner that never
    clears. Cheap to check, invisible to catch by eye.
    """
    js = (SITE / "app.js").read_text()
    m = re.search(r'const DRIVE = "([^"]+)"', js)
    assert m, "site/app.js no longer declares DRIVE"
    data_dir = SITE / "data"
    assert data_dir.is_dir(), (
        "site/data/ is missing. It holds the exported replay the page depends on; "
        "if it is absent from a checkout, check that .gitignore's dataset rule is "
        "anchored to /data/ and has not swallowed it."
    )
    exported = {p.name for p in data_dir.iterdir() if p.is_dir()}
    assert m.group(1) in exported, (
        f"the player expects {m.group(1)!r} but site/data holds {sorted(exported)}. "
        "Re-run tools/export_replay.py for that drive, or update DRIVE."
    )
