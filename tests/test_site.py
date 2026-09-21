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


def test_the_page_does_not_call_a_backend_that_does_not_exist():
    """A public page pointing at localhost is broken for every visitor.

    The old app.js fetched /api/control, /api/telemetry, /api/obstacles and
    /video_feed from 127.0.0.1:5000. APS is offline and open-loop: there is no
    service to call, so every control was permanently dead.
    """
    # Strip comments first: app.js documents what was removed and why, and the
    # record of a deleted endpoint is not a live call to it.
    code = re.sub(r"/\*.*?\*/", "", site_code(), flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    for dead in ("127.0.0.1", "localhost", "/api/", "/video_feed"):
        assert dead not in code, f"site/app.js still references {dead!r}"
    assert "fetch(" not in code, "the evidence page is static; it fetches nothing"


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
