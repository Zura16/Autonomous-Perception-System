"""Every command in CLAUDE.md must actually run.

The charter's command table is the first thing a reader copies, and it had
drifted from the code three ways at once: `eval_fcw.py --ttc-threshold`,
`eval_range.py --bins` and `run_bench.py --iters/--report` were all documented
and none existed. Two of those flags had never been implemented; the third was
removed. Nothing caught it, because documentation is not executed.

These tests execute it. They parse the command block out of CLAUDE.md and check
each invocation against the script's own argparse, so a flag that is renamed or
dropped fails the suite instead of wasting the next reader's afternoon.

Deliberately static: the commands are checked for *validity*, not run. Running
them needs the KITTI data and minutes of GPU time, and a test suite that cannot
pass on a clean checkout is a test suite people learn to skip.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = REPO / "CLAUDE.md"

# `python path/to/script.py --flags...`
COMMAND = re.compile(r"^python\s+((?:tools|eval|app|bench)/[a-z_0-9]+\.py)([^\n]*)", re.M)


def documented_commands() -> list[tuple[str, str]]:
    """(script path, argument string) for every python command in CLAUDE.md.

    Shell line-continuations are joined BEFORE matching. Doing it after does not
    work: a greedy `[^\n]*` eats the trailing backslash, so the continuation
    never joins and the second line's flags are silently dropped -- which made
    this very test report a false failure on `render_frame.py`, whose flags were
    fine, while the real drift was elsewhere.
    """
    text = CLAUDE_MD.read_text().replace("\\\n", " ")
    out = []
    for m in COMMAND.finditer(text):
        args = m.group(2).split("#")[0]  # strip trailing comment
        out.append((m.group(1), args.strip()))
    return out


def declared_flags(script: Path) -> set[str]:
    """Every option string the script's argparse accepts.

    Read from the source rather than by importing: these modules execute heavy
    imports (torch, ultralytics) at module scope, and a lint test should not
    need a GPU stack.
    """
    tree = ast.parse(script.read_text())
    flags: set[str] = {"-h", "--help"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                flags.add(arg.value)
        # argparse also accepts dest= for positionals; store= keywords are not flags
        for kw in node.keywords:
            if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                flags.add("--" + str(kw.value.value).replace("_", "-"))
    return flags


def test_claude_md_lists_some_commands():
    """Guard the parser itself: a regex that matches nothing would pass silently."""
    cmds = documented_commands()
    assert len(cmds) >= 10, f"only found {len(cmds)} commands -- has the format changed?"


@pytest.mark.parametrize("script,args", documented_commands(), ids=lambda v: str(v)[:40])
def test_documented_command_script_exists(script, args):
    assert (REPO / script).is_file(), f"CLAUDE.md documents {script}, which does not exist"


@pytest.mark.parametrize("script,args", documented_commands(), ids=lambda v: str(v)[:40])
def test_documented_command_flags_exist(script, args):
    """Every --flag in a documented command must be declared by that script."""
    path = REPO / script
    if not path.is_file():
        pytest.skip("covered by test_documented_command_script_exists")
    declared = declared_flags(path)
    used = [tok for tok in shlex.split(args) if tok.startswith("--")]
    # `--flag=value` spellings
    used = [tok.split("=", 1)[0] for tok in used]
    unknown = sorted(set(used) - declared)
    assert not unknown, (
        f"CLAUDE.md runs `{script} {args}` but {script} does not accept {unknown}. "
        f"It accepts: {sorted(declared)}"
    )


@pytest.mark.parametrize("negative_flag", ["--ttc-threshold", "--iters", "--report", "--bins"])
def test_the_flags_that_had_drifted_stay_gone_or_get_implemented(negative_flag):
    """The three drifted commands, pinned as a regression.

    Each of these was documented in CLAUDE.md against a script that never
    accepted it. The fix was to correct the documentation, so the invariant is
    simply that no documented command uses a flag its script lacks -- already
    enforced above. This test states the specific history so a future edit that
    reintroduces one of these spellings has to do it deliberately.
    """
    for script, args in documented_commands():
        if negative_flag in args:
            declared = declared_flags(REPO / script)
            assert negative_flag in declared, (
                f"`{negative_flag}` is back in a CLAUDE.md command for {script} "
                f"but {script} still does not implement it"
            )
