"""Every `flexsweep <cmd> --opt ...` in the published docs must name a real command/option.

Oracle: the Click command objects themselves (names and declared options).
Covers only the pages in docs/index.rst's toctree; it checks names, not values.
"""

import os
import re
import shlex

import pytest

from flexsweep.main import cli

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def _published_pages():
    """Pages listed in every toctree of docs/index.rst."""
    index = open(os.path.join(DOCS, "index.rst")).read()
    pages = []
    for block in index.split(".. toctree::")[1:]:
        names = [ln.strip() for ln in block.splitlines()[1:] if ln.strip() and not ln.strip().startswith(":")]
        for name in names:
            if not re.match(r"^[\w-]+$", name):
                break
            pages.append(os.path.join(DOCS, f"{name}.rst"))
    return pages


def test_all_toctrees_collected():
    names = {os.path.basename(p) for p in _published_pages()}
    assert {"quickstart.rst", "scan.rst", "dann.rst", "api.rst"} <= names


def _commands(path):
    """Yield (page, line_no, argv) for each flexsweep invocation in code blocks."""
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("flexsweep "):
            start = i
            cmd = line
            while cmd.endswith("\\") and i + 1 < len(lines):
                i += 1
                cmd = cmd[:-1] + " " + lines[i].strip()
            cmd = cmd.split(" #", 1)[0]
            yield os.path.basename(path), start + 1, shlex.split(cmd)[1:]
        i += 1


CASES = [c for p in _published_pages() for c in _commands(p)]


@pytest.mark.parametrize("page,line,argv", CASES, ids=[f"{p}:{n}" for p, n, _ in CASES])
def test_documented_command_exists(page, line, argv):
    name = argv[0]
    if name.startswith("--"):
        assert name in ("--help",), f"{page}:{line}: unknown top-level option {name}"
        return
    assert name in cli.commands, f"{page}:{line}: unknown command 'flexsweep {name}'"
    cmd = cli.commands[name]
    known = {opt for p in cmd.params for opt in p.opts + p.secondary_opts}
    known.add("--help")
    used = [a.split("=", 1)[0] for a in argv[1:] if a.startswith("--")]
    unknown = [u for u in used if u not in known]
    assert not unknown, f"{page}:{line}: 'flexsweep {name}' has no option(s) {unknown}"
