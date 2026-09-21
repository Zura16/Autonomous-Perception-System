"""Every config key is either read by the code or declared inert.

The charter says config over constants, and a magic number in a `.py` file is a
bug. Both halves of that have failed here before:

  * [D-017] `min_rows_from_bottom` sat in `camera.yaml` looking operative while
    nothing read it, so the clipped-box rule it described was never applied;
  * `ttc.report_below_s` sat in `motion.yaml` while `eval_ttc.py` hardcoded
    `(3.0, 6.0, 10.0)`. Reading either file alone, both looked correct.

A key that nothing reads is indistinguishable from a key that is *supposed* to
do nothing, so the inert ones are listed explicitly below. That list is the
point of the test: adding to it is a deliberate act, and leaving a key off it is
a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))

# Keys that are documentation, not inputs. Each must say why in the config file
# itself, and each is here because it is recorded for humans and provably never
# reaches an estimator.
DELIBERATELY_INERT = {
    # camera.yaml -- the measured pitch distribution. Recorded so the
    # sensitivity analysis quotes real numbers; the estimator uses pitch_deg: 0
    # by decision (D-016), and reading these would silently start correcting.
    "measured_pitch",
    "mean_deg",
    "std_deg",
    "abs_p95_deg",
    "abs_max_deg",
    "n_frames",
    # camera.yaml -- prior transfer error dev->val, recorded for the error
    # budget (D-014). Applying it would be fitting the prior to val.
    "transfer_error_pct",
    # free-text provenance, on several keys
    "source",
    "caveat",
    "notes",
    "note",  # dataset.yaml: per-drive human description of what the sequence holds
}


def all_keys(obj: object) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found |= all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= all_keys(v)
    return found


def python_source() -> str:
    parts = []
    for p in REPO.rglob("*.py"):
        s = str(p)
        if "legacy" in s or "venv" in s or "/.git/" in s:
            continue  # legacy/ is a museum piece, excluded by the charter
        parts.append(p.read_text())
    return "\n".join(parts)


SOURCE = python_source()


def test_configs_exist():
    """Guard the test: an empty glob would make everything below vacuous."""
    assert len(CONFIGS) >= 4, f"expected several configs, found {[c.name for c in CONFIGS]}"


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.name)
def test_every_config_key_is_read_or_declared_inert(config):
    unread = sorted(
        k
        for k in all_keys(yaml.safe_load(config.read_text()))
        if k not in DELIBERATELY_INERT
        and f'"{k}"' not in SOURCE
        and f"'{k}'" not in SOURCE
        and f".{k}" not in SOURCE
    )
    assert not unread, (
        f"{config.name} declares {unread}, which no code reads. Either wire it up, "
        f"or add it to DELIBERATELY_INERT with a reason -- a key that looks operative "
        f"and is not is the D-017 bug."
    )


def test_the_ttc_reporting_horizons_are_actually_used():
    """The specific regression: the horizons must come from config, not source.

    `eval/eval_ttc.py` hardcoded (3.0, 6.0, 10.0) while the config declared a
    lone 3.0 that nothing read.
    """
    from aps.motion import MotionConfig

    cfg = MotionConfig.from_config()
    declared = yaml.safe_load((REPO / "configs" / "motion.yaml").read_text())
    assert tuple(cfg.report_below_s) == tuple(declared["ttc"]["report_below_s"])
    assert 3.0 in cfg.report_below_s, "the headline horizon must still be reported"

    src = (REPO / "eval" / "eval_ttc.py").read_text()
    assert "(3.0, 6.0, 10.0)" not in src, "horizons are hardcoded again"


def test_inert_keys_are_explained_where_they_live():
    """An inert key must justify itself in its own config file.

    Otherwise the allowlist becomes a place to hide dead configuration, which
    is the failure it exists to prevent.
    """
    blob = "\n".join(c.read_text() for c in CONFIGS)
    for key in ("measured_pitch", "transfer_error_pct"):
        assert key in blob
        # the file must say, in words, that it is not consumed
        assert (
            "NOT read" in blob or "not read" in blob
        ), f"{key} is allowlisted as inert but no config explains that it is unused"
