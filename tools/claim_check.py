"""Audit every external claim against a benchmark row and a commit hash.

    python tools/claim_check.py            # audit; non-zero exit if anything is unbacked
    python tools/claim_check.py --pin      # fill in *pending* commit hashes

Hard rule 12: claims are generated *from* `benchmarks.md`, never authored, and
each is pinned to the commit that produced its measurement. This makes that
auditable instead of aspirational. Three failures are possible and all are
reported:

  UNBACKED   the claim cites no benchmark row
  MISSING    it cites a row that does not exist in benchmarks.md
  UNPINNED   it has no commit hash, so the measurement behind it is not located

`--pin` resolves a hash by asking git which commit first introduced that row
heading into `docs/benchmarks.md`. It is a lookup, not an assertion: a row that
was never committed cannot be pinned, which is the point.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAIMS = REPO / "docs" / "claims.md"
BENCHMARKS = REPO / "docs" / "benchmarks.md"
ROW_ANY = re.compile(r"\d+\.\d+")
CLAIM_ID = re.compile(r"^\|\s*(C-\d+)\s*\|")


def cited_rows(backing: str) -> list[str]:
    """Every row a backing cell cites, including ranges and lists.

    `Row 3.1-3.2, 9.3` cites three rows. Matching only `Row\\s+(\\d+\\.\\d+)`
    saw one of them, so the rest went unchecked -- a claim could cite a row that
    does not exist and still pass the audit, which is the single thing this
    script is for. Requires the literal word "Row" so that a stray number in
    prose cannot masquerade as a citation.
    """
    if "Row" not in backing:
        return []
    return list(dict.fromkeys(ROW_ANY.findall(backing)))


def benchmark_rows() -> set[str]:
    return set(re.findall(r"^###\s+Row\s+(\d+\.\d+)", BENCHMARKS.read_text(), re.M))


def commit_for_row(row: str) -> str | None:
    """The commit that first introduced this row heading into benchmarks.md."""
    out = subprocess.run(
        ["git", "log", "--reverse", "--format=%h", "-S", f"### Row {row}", "--", str(BENCHMARKS)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    hashes = [h for h in out.stdout.split() if h]
    return hashes[0] if hashes else None


def _commit_time(sha: str) -> int:
    out = subprocess.run(
        ["git", "show", "-s", "--format=%ct", sha],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(out.stdout.strip() or 0)


def commit_for_claim(refs: list[str]) -> tuple[str | None, list[str]]:
    """The commit at which a claim citing all of `refs` first became provable.

    That is the LATEST of its rows' commits, not the earliest. A claim revised to
    rest on new evidence keeps citing its original row, and pinning to the first
    would date the claim to before the measurement that justifies it -- which is
    precisely the backdating this file exists to prevent. Returns the chosen hash
    and the rows that are not committed yet.
    """
    resolved = {r: commit_for_row(r) for r in refs}
    missing = [r for r, h in resolved.items() if not h]
    found = [h for h in resolved.values() if h]
    if missing or not found:
        return None, missing
    return max(found, key=_commit_time), []


def audit(pin: bool) -> int:
    rows = benchmark_rows()
    lines = CLAIMS.read_text().split("\n")
    problems: list[str] = []
    checked = 0

    for i, line in enumerate(lines):
        m = CLAIM_ID.match(line)
        if not m:
            continue
        claim = m.group(1)
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            problems.append(
                f"{claim}: malformed row (expected id | claim | backing | commit | caveat)"
            )
            continue
        _, _, backing, commit, caveat = cells[0], cells[1], cells[2], cells[3], cells[4]
        checked += 1

        refs = cited_rows(backing)
        if not refs:
            problems.append(f"{claim}: UNBACKED -- cites no benchmark row")
            continue
        for ref in refs:
            if ref not in rows:
                problems.append(f"{claim}: MISSING -- cites Row {ref}, absent from benchmarks.md")

        if not caveat:
            problems.append(f"{claim}: no caveat recorded")

        if commit.strip("*") in ("pending", "") or commit == "*pending*":
            resolved, missing = commit_for_claim(refs)
            if pin and resolved:
                lines[i] = line.replace("*pending*", f"`{resolved}`", 1)
                print(f"{claim}: pinned to {resolved} (latest of Rows {', '.join(refs)})")
            else:
                problems.append(
                    f"{claim}: UNPINNED -- "
                    + (
                        f"Row(s) {', '.join(missing)} not in any commit yet"
                        if missing
                        else f"Rows {', '.join(refs)} resolve to {resolved}"
                    )
                )

    if pin:
        CLAIMS.write_text("\n".join(lines))

    print(f"\nchecked {checked} claims against {len(rows)} benchmark rows")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("all claims backed by a benchmark row, pinned to a commit, and carrying a caveat")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pin", action="store_true", help="fill in pending commit hashes")
    return audit(ap.parse_args().pin)


if __name__ == "__main__":
    raise SystemExit(main())
