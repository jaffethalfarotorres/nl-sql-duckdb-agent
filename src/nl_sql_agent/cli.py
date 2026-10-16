"""Command line: ask a question, or run SQL through the guard, and print the AgentResult as JSON.

nl-sql-agent "Which 3 products brought in the most revenue?" --backend groq
nl-sql-agent --sql "SELECT 1; DROP TABLE orders"      # no model: guard + read-only executor
nl-sql-agent --schema                                  # the schema text the model is shown
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from nl_sql_agent.agent import build_agent
from nl_sql_agent.dataset import PUBLIC_TABLES, SEED, build_database, ensure_database
from nl_sql_agent.executor import DEFAULT_MAX_ROWS, DEFAULT_TIMEOUT_S
from nl_sql_agent.providers import BACKENDS, BackendUnavailable, LLMBackend, backend_from_env

BACKEND_CHOICES = ["none", *BACKENDS]
DEFAULT_DB = Path("shop.duckdb")


def resolve_backend(name: str, env: Mapping[str, str] | None = None) -> LLMBackend | None:
    """``none`` means no LLM: no credentials, no cost. Only ``--sql`` works then."""
    return None if name == "none" else backend_from_env(name, os.environ if env is None else env)


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl-sql-agent", description=__doc__.splitlines()[0])
    parser.add_argument(
        "question", nargs="?", help="question to answer; read from stdin if omitted"
    )
    parser.add_argument("--sql", help="run this SQL through the guard and executor, no model")
    parser.add_argument("--schema", action="store_true", help="print the schema the model sees")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="none")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="built on first use")
    parser.add_argument("--rebuild-db", action="store_true", help=f"rebuild it (seed {SEED})")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    args = parser.parse_args(argv)

    try:
        backend = resolve_backend(args.backend, env)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    asking = not args.schema and args.sql is None
    if asking and backend is None:
        print(
            "error: a question needs an LLM backend (--backend groq|watsonx|huggingface|ollama); "
            "use --sql to try the guard without one",
            file=sys.stderr,
        )
        return 2

    db = build_database(args.db) if args.rebuild_db else ensure_database(args.db)
    with build_agent(db, backend, max_rows=args.max_rows, timeout_s=args.timeout) as agent:
        if args.schema:
            print(agent.executor.describe(PUBLIC_TABLES))
            return 0
        if args.sql is not None:
            result = agent.run_sql(args.sql)
        else:
            result = agent.ask(args.question if args.question is not None else sys.stdin.read())
    print(result.model_dump_json(indent=2))
    return 0
