# nl-sql-duckdb-agent

[![CI](https://github.com/jaffethalfarotorres/nl-sql-duckdb-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jaffethalfarotorres/nl-sql-duckdb-agent/actions/workflows/ci.yml)
![Python 3.11 to 3.13](https://img.shields.io/badge/python-3.11%20to%203.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Ask questions about a database in plain English. An LLM proposes the SQL;
**code decides whether it runs**. Every query goes through a two-layer adversarial
validator and runs on a locked, read-only DuckDB connection, and the whole thing is
pinned by tests and an eval suite that run in CI **without any API key**.

## The problem it solves

Text-to-SQL agents are one prompt injection away from `DROP TABLE`. The common
pattern hands the model a "run SQL" tool and lets it loop; the common defense is a
keyword blocklist that a `;` hidden inside a string literal or a comment walks
straight past. And even a perfectly behaved model can read a table it was never
meant to see.

This project treats the model as **compromised by default**. It may only propose a
query. A validator decides whether that query is a single, read-only `SELECT` over
allowlisted tables, and an executor that does not trust the validator runs it on a
connection that cannot write, read files, load extensions or change its own
settings.

The scenario is an office-supplies seller with business customers: 40 customers,
15 products and 400 orders in 2025, generated in code from a fixed seed (no
download, no personal data). The same file holds a confidential `payroll` table the
agent is never told about and must never read.

## Architecture

```mermaid
flowchart TD
    Q["Question"] --> DEF["Prompt defense scan"]
    DEF -- injection --> BLK["blocked: no model call"]
    DEF -- clean --> PLAN["Planner: the LLM proposes one query as JSON"]
    PROV[("LLMBackend: watsonx, Groq, HF, Ollama, Mock")] <--> PLAN
    PLAN -- declines --> UNA["unanswerable"]
    PLAN -- SQL --> LEX["Layer 1: lexer, pure Python"]
    LEX -- violation --> REJ["rejected: never retried"]
    LEX -- pass --> ENG["Layer 2: DuckDB's parser + AST allowlist"]
    ENG -- violation --> REJ
    ENG -- pass --> GATE["Executor gate: exactly one parsed SELECT"]
    GATE --> RO[("Read-only DuckDB: no files, no extensions, locked config, time and row limits")]
    RO -- rows --> OK["completed"]
    RO -- "fixable engine error" --> REP["One informed repair on the large tier"]
    REP --> LEX
    RO -- "timeout or second failure" --> ERR["sql_error"]
```

1. **Prompt defense** (`defense.py`, reused from the Phase 1 project) blocks known
   injection shapes in the question and fences it as untrusted data in the prompt.
2. **Planner** (`planner.py`) asks the model for `{"sql": ..., "explanation": ...}`.
   The schema in the prompt is generated from the database catalog and lists only
   the allowlisted tables. A `null` query is a valid answer ("I can't answer that").
3. **Validator, layer 1** (`validator.py`): a small tokenizer that knows where
   strings, quoted identifiers and comments begin and end. On the tokens: one
   statement, a leading `SELECT`/`WITH`, no denied keyword, function or identifier.
4. **Validator, layer 2**: DuckDB's own parser on an empty in-memory connection.
   Exactly one `SELECT`; then the JSON syntax tree is walked to find every table
   reference, with CTE scoping, and each must be an allowlisted base table.
5. **Executor** (`executor.py`) refuses unvalidated reports, re-parses, executes the
   one parsed statement object on a `read_only=True` connection with file access,
   extension loading and configuration changes all disabled, under a time limit.
6. **Agent** (`agent.py`): policy rejections are final; an engine error the model can
   fix (unknown column, syntax) gets one retry that includes the error.

## Defense in depth

Each attack class against each layer **in isolation**. The table is asserted by
[`tests/test_defense_in_depth.py`](tests/test_defense_in_depth.py), so it cannot
drift from the code. ✅ = this layer stops it on its own.

| Attack | Lexer | DuckDB parser + AST | Executor gate | Read-only connection |
|---|:-:|:-:|:-:|:-:|
| `SELECT 1; DROP TABLE orders` (stacked write) | ✅ | ✅ | ✅ | ✅ |
| `SELECT 1; SELECT * FROM payroll` (stacked read) | ✅ | ✅ | ✅ | |
| `DROP` / `UPDATE` / `CREATE TEMP TABLE` | ✅ | ✅ | ✅ | ✅ except temp tables |
| `SELECT E'\''; DROP TABLE orders; --'` (literal smuggling) | ✅ | ✅ | ✅ | ✅ |
| `read_text('/etc/passwd')`, `COPY ... TO`, `INSTALL`, `SET` | ✅ | ✅ | ✅ except file reads | ✅ |
| `SELECT * FROM payroll` (hidden table) | ✅ | ✅ | | |
| ...via `query('SELECT * FROM payroll')` or CTE scoping tricks | ✅ | ✅ | | |
| `information_schema`, `(SUMMARIZE payroll)`, `PRAGMA database_list` | ✅ | ✅ | | |

Every one of the 16 attack classes in the test is stopped by at least two
independent layers, and the database file is byte-identical afterwards. The empty
cells are the honest part: a read-only connection happily runs a read of a hidden
table, and DuckDB has no `GRANT` to fall back on. In Postgres, Snowflake or Db2 the
extra layer for the allowlist would be a role with `SELECT` granted only on those
tables.

## Quickstart (3 commands, no API key)

```bash
git clone https://github.com/jaffethalfarotorres/nl-sql-duckdb-agent && cd nl-sql-duckdb-agent
pip install -e ".[dev]"
pytest && python -m nl_sql_agent.eval
```

Try the guard without a model (`--sql` skips the LLM; the database is built on first
use):

```bash
nl-sql-agent --sql "SELECT 1; DROP TABLE orders"
```

```json
{
  "status": "rejected",
  "validation": {
    "violations": [
      { "layer": "lexer", "rule": "multiple_statements", "detail": "code after the first ';'" },
      { "layer": "lexer", "rule": "denied_keyword", "detail": "DROP" }
    ],
    "ok": false
  },
  "result": null,
  "trace": [ "defense: skipped (direct SQL, no model in the loop)",
             "validate: rejected (denied_keyword, multiple_statements)" ]
}
```

A legitimate query goes through and reports what it read:

```bash
nl-sql-agent --sql "SELECT p.category, round(sum(i.quantity * i.unit_price), 2) AS revenue FROM order_items i JOIN orders o USING (order_id) JOIN products p USING (product_id) WHERE o.status = 'paid' GROUP BY p.category ORDER BY revenue DESC"
# -> "status": "completed", "tables": ["order_items", "orders", "products"],
#    rows: furniture 219307.9, electronics 152044.8, accessories 43706.7
```

With a model (any of the four providers; see [`.env.example`](.env.example)):

```bash
pip install -e ".[groq]"
export GROQ_API_KEY=...
nl-sql-agent "Which 5 products brought in the most revenue?" --backend groq
nl-sql-agent --schema    # exactly what the model is shown
```

## Design decisions

| Decision | Where | Why |
|---|---|---|
| **The model proposes, code disposes** | `agent.py` | No tool loop where the model executes SQL itself. The model returns one query as typed JSON; deterministic code validates and runs it. A successful injection can at worst produce a query that gets rejected. |
| **Assume the model is compromised** | `evals/guard.yaml` | The guard suite scripts the model to reply with attack SQL and runs the whole agent. Safety does not depend on the model behaving. |
| **Two independent validator layers** | `validator.py` | A Python lexer and DuckDB's own parser fail in different ways. Each layer is tested on its own; either one rejects. |
| **Validate with the parser that executes** | `validator.py` | A home-made parser and the engine can disagree about where a statement ends (a parser differential). Layer 2 asks DuckDB itself, so for statement boundaries and table references there is nothing to disagree about. |
| **Fail closed on ambiguous syntax** | `validator.py` | DuckDB honors `E'...'` backslash escapes, `$$` quoting and nested `/* */` comments, so a naive scanner would misjudge where a literal ends. The lexer refuses all three instead of guessing; analytics SQL never needs them. |
| **Walk the syntax tree for tables** | `validator.py` | `get_table_names()` looked like the easy answer, but testing showed it binds the query (fails on `USING` joins without the real catalog), omits system views such as `sqlite_master` and `information_schema`, and returns clause text for `TABLESAMPLE`. The JSON AST from `json_serialize_sql` is the single source of truth, with CTE scope resolved conservatively (a CTE cannot hide the real table outside its scope). |
| **Executor does not trust the validator** | `executor.py` | It refuses failed reports, re-parses, and executes the one parsed statement *object*, never the raw string, on a `read_only=True` connection with external access, extension autoloading and configuration changes disabled. |
| **Policy rejections are never retried** | `agent.py` | Letting the model retry against the validator turns the guard into an oracle an attacker can probe. Only engine errors the model can actually fix get one informed retry (failed query and error in the prompt, both fenced as untrusted), on the larger model tier. No blind retries. |
| **Typed envelopes with invariants** | `envelopes.py` | `AgentResult` cannot be constructed with rows unless the exact SQL that produced them passed validation. Computed fields like `ok` are recomputed on input, so a tampered `"ok": true` does not survive. |
| **Schema from the catalog** | `dataset.py`, `executor.py` | Column documentation and business rules ("revenue counts paid orders only") live in the database as `COMMENT ON`. The prompt is generated from them, so it cannot drift, and hidden tables never appear in it. |
| **Execution accuracy, not string match** | `eval.py` | The model's query and the gold query both run; every gold row must appear in the model's result. Extra columns and rounding are tolerated, wrong values are not. Declining is the correct answer for four cases (hidden salaries, delete, update, off-topic). |
| **Provider-agnostic** | `providers/` | The `LLMBackend` contract and its five backends are copied from the Phase 1 project. Switching provider is configuration. |
| **DuckDB, synthetic data** | `dataset.py` | Embedded, free, sub-second to build, and runs identically on a laptop and in CI. The pattern moves to Postgres or Snowflake by swapping the connection and the parser layer (the engine's own parser where available, or sqlglot) and adding role grants. |

## Results

Measured locally and on every push by [CI](.github/workflows/ci.yml) on Python 3.11,
3.12 and 3.13:

- **283 tests, 99% line coverage**, no network and no credentials, ~15 s.
- **Eval gates** (`python -m nl_sql_agent.eval`):

```text
backend: none (oracle replays gold SQL for nl2sql: measures the harness, not a model)  k=1
suite       ran  pass@k  pass^k  gate
guard        54    1.00    1.00  pass^k >= 1.00  PASS
nl2sql       24    1.00    1.00  pass@k >= 0.80  PASS
database unchanged: yes
```

- **Differential fuzzing against DuckDB's parser**: 20,000 seeded cases built from
  quotes, comments, `E'...'`, `$$` and separators. DuckDB parsed 1,737 of them, and
  748 of those as more than one statement or a non-`SELECT`. The lexer rejected
  **all 748** on its own.
- **Mutation check** ([`scripts/mutation_check.py`](scripts/mutation_check.py)): each
  of 17 guards (lexer rules, CTE scoping, allowlist, read-only flag, external access,
  no-retry-on-policy, the envelope invariant, the grader itself) was disabled on
  purpose. The suite failed every time: **17/17 caught**.

### What the tests found along the way

- The defense matrix showed hidden-table reads and catalog discovery were stopped by
  **one** layer only. The lexer now also refuses identifiers that name system
  catalogs or a known table outside the allowlist: blunt, but independent.
- DuckDB reports `PRAGMA database_list` as a `SELECT` statement (it is rewritten into
  a pragma call), so statement type alone is not enough; `json_serialize_sql` is what
  refuses it.
- `SELECT * FROM (SHOW TABLES)` lists every table without a single base-table
  reference in the AST, so the walk also allowlists *reference types*.
- The first dataset build took ~8 s: the DuckDB Python client retries a failed
  `numpy`/`pandas` import for every bound parameter. Inserting each table as one
  JSON parameter takes under 0.1 s.

## Project layout

```text
src/nl_sql_agent/
  envelopes.py     typed, self-validating Pydantic models (the contract)
  defense.py       prompt-defense scan and untrusted-input fencing (from Phase 1)
  planner.py       asks the model for one query; diagnoses replies, no blind retry
  validator.py     two-layer read-only SQL validator
  executor.py      locked read-only DuckDB executor with time and row limits
  agent.py         the pipeline, its no-retry policy and the single repair
  dataset.py       seeded synthetic database with catalog documentation
  providers/       LLMBackend base + watsonx, Groq, Hugging Face, Ollama, Mock
  eval.py          YAML eval runner with pass@k / pass^k gates
  cli.py           `nl-sql-agent` command
evals/             guard.yaml (54 attack and near-miss cases), nl2sql.yaml (24 questions)
tests/             unit tests, defense matrix, differential fuzz, provider fakes
scripts/           mutation_check.py
```

## Known limitations

- **No live-model numbers yet.** CI has no provider key, so the `nl2sql` suite runs
  against the oracle. It proves the harness works and that the validator lets 20 real
  analytics queries through; it does not measure a model. Run
  `python -m nl_sql_agent.eval --backend groq --k 3` with a key for real numbers.
- The lexer's identifier denylist cannot tell a table from a column: a column named
  like a hidden table, or an unquoted column named like a denied keyword (`set`,
  `show`), is rejected. Quoting it works.
- The allowlist is per table, not per column or row.
- The parser layer is DuckDB-specific, and `json_serialize_sql`'s output format can
  change between DuckDB versions; the dependency is pinned to `duckdb>=1.5,<2` and CI
  would catch a change.
- The executor caps time and rows, not memory.
- Relaxed execution accuracy can, in rare cases, accept a result with the right values
  under the wrong labels.

## Related

Phase 1 of the same portfolio: [pydantic-router-agent](https://github.com/jaffethalfarotorres/pydantic-router-agent),
an intent router with prompt-injection defense and code-gated tools. This project
reuses its `LLMBackend` providers and prompt-defense module.

## License

[MIT](LICENSE) © 2026 Jaffeth Alfaro
