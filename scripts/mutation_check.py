"""Disable each guard on purpose and check that the test suite notices.

    python scripts/mutation_check.py

Each mutant rewrites one line of the source, runs the test suite and restores
the file, so a guard that no test depends on shows up as SURVIVED. It edits
files in place: if it is interrupted, ``git checkout -- src`` restores them.

pytest runs from a throwaway working directory: with a guard disabled, an attack
in the tests may really succeed (``COPY orders TO 'leak.csv'`` writes a file), and
that file must land there, not in the repository. Takes a few minutes; not part
of CI.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "nl_sql_agent"

# name, file, exact source text, replacement
MUTANTS = [
    ("lexer: stacked statements", "validator.py",
     "found.append(violation(\"multiple_statements\", \"code after the first ';'\"))", "pass"),
    ("lexer: nested comments", "validator.py",
     'if "/*" in sql[i + 2 : close]:', "if False:"),
    ("lexer: E-strings", "validator.py",
     "if prefix and prefix.kind is TokenKind.WORD and len(prefix.text) == 1:", "if False:"),
    ("lexer: denied keywords", "validator.py",
     'found.extend(violation("denied_keyword", keyword) for keyword in keywords)', "pass"),
    ("lexer: hidden table names", "validator.py",
     "            or name in denied_names\n        )", "            or False\n        )"),
    ("engine: statement count", "validator.py",
     "if len(statements) > 1:", "if False:"),
    ("engine: CTE sibling scope", "validator.py",
     "visible = set(names[:k])", "visible = set(names)"),
    ("engine: CTE names are not tables", "validator.py",
     "if qualified not in scope:", "if True:"),
    ("engine: metadata refs", "validator.py",
     "elif kind not in _ALLOWED_REF_TYPES:", "elif False:"),
    ("engine: allowlist", "validator.py",
     "if not self._is_allowed(table)", "if False"),
    ("executor: refuse failed reports", "executor.py",
     "if not report.ok:", "if False:"),
    ("executor: one statement only", "executor.py",
     "if len(statements) != 1:", "if False:"),
    ("executor: read-only flag", "executor.py",
     "duckdb.connect(str(path), read_only=True,", "duckdb.connect(str(path), read_only=False,"),
    ("executor: no external access", "executor.py",
     '"enable_external_access": False,', '"enable_external_access": True,'),
    ("agent: no retry on policy", "agent.py",
     "if validation.violations and validation.rules <= _REPAIRABLE_RULES:",
     "if validation.violations:"),
    ("envelope: rows need validation", "envelopes.py",
     "if self.result is not None and (validation is None or not validation.ok):", "if False:"),
    ("eval: grader can fail", "eval.py",
     "    if len(gold) != len(model):\n        return False", "    return True"),
]  # fmt: skip


def main() -> int:
    survivors = []
    for name, file, old, new in MUTANTS:
        path = SRC / file
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            print(f"BROKEN   {name}: the pattern matches {original.count(old)} times")
            survivors.append(name)
            continue
        path.write_text(original.replace(old, new), encoding="utf-8", newline="\n")
        try:
            with tempfile.TemporaryDirectory() as scratch:
                run = subprocess.run(
                    [sys.executable, "-m", "pytest", str(REPO / "tests"), "-x", "-q",
                     "-p", "no:cacheprovider"],
                    cwd=scratch,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )  # fmt: skip
        finally:
            path.write_text(original, encoding="utf-8", newline="\n")
        caught = run.returncode != 0
        first = next((line for line in run.stdout.splitlines() if line.startswith("FAILED")), "")
        print(f"{'caught  ' if caught else 'SURVIVED'} {name:<34} {first[7:80]}")
        if not caught:
            survivors.append(name)
    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
