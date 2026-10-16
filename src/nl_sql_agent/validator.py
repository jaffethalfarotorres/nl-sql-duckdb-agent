"""Adversarial read-only SQL validator: the boundary between model output and the database.

The validator assumes the model that wrote the SQL may be compromised (by a
prompt injection, a poisoned schema comment or plain error) and checks the SQL
with two independent layers. Either one rejects on its own.

1. **Lexer** (pure Python, no engine involved). A small tokenizer that knows where
   string literals, quoted identifiers and comments begin and end, so a ``;`` or a
   ``DROP`` inside them is data, not code. On the token stream it enforces: exactly
   one statement, a leading ``SELECT`` or ``WITH``, no denied keyword (DDL, DML,
   files, extensions, session state) and no denied function (file readers,
   ``query()``, catalog functions). Lexical forms whose boundaries depend on the
   engine dialect (``E'...'`` strings, ``$$`` quoting, nested comments) are rejected
   outright: when the guard cannot be sure how the engine will tokenize something,
   it refuses it.

2. **Engine** (DuckDB's own parser, on an empty in-memory connection). Using the
   parser that will run the query removes any parser differential: the engine must
   see exactly one SELECT statement, and DuckDB's ``json_serialize_sql`` must agree
   (it refuses PRAGMA, which ``extract_statements`` reports as a SELECT). The JSON
   syntax tree is then walked to find every table reference, with CTE scoping
   resolved conservatively, and every reference must be an allowlisted base table.
   Table functions and metadata references (``SHOW``, ``DESCRIBE``, ``SUMMARIZE``)
   are refused.

Input that fails the lexer never reaches the engine's parser.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

import duckdb

from nl_sql_agent.envelopes import Layer, ValidationReport, Violation

DEFAULT_MAX_CHARS = 10_000
_DETAIL_CHARS = 200

DENIED_KEYWORDS = frozenset(
    {
        # data changes
        "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "TRUNCATE",
        # schema changes
        "CREATE", "ALTER", "DROP", "COMMENT",
        # files, other databases and extensions
        "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "INSTALL", "LOAD", "USE",
        # session and engine state
        "SET", "RESET", "PRAGMA", "CALL", "CHECKPOINT", "VACUUM",
        # metadata statements that DuckDB also accepts as subqueries
        "DESCRIBE", "SUMMARIZE", "SHOW",
        # transactions, prepared statements and privileges
        "BEGIN", "COMMIT", "ROLLBACK", "TRANSACTION", "PREPARE", "EXECUTE", "DEALLOCATE",
        "GRANT", "REVOKE",
    }
)  # fmt: skip
DENIED_FUNCTIONS = frozenset({"query", "query_table", "glob", "sniff_csv", "getenv"})
DENIED_FUNCTION_PREFIXES = (
    "read_", "parquet_", "pragma_", "duckdb_", "sqlite_", "postgres_", "mysql_",
    "iceberg_", "delta_",
)  # fmt: skip
_LEADING_KEYWORDS = {"SELECT", "WITH"}
_ALLOWED_SCHEMAS = {"", "main"}
_ALLOWED_REF_TYPES = {"BASE_TABLE", "SUBQUERY", "JOIN", "EMPTY", "EXPRESSION_LIST", "PIVOT"}
_PARSER_CONFIG = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "lock_configuration": True,
}


def _short(text: str) -> str:
    return text.strip().splitlines()[0][:_DETAIL_CHARS] if text.strip() else ""


# --- lexer ---------------------------------------------------------------------------


class TokenKind(StrEnum):
    WORD = "word"
    QUOTED = "quoted_identifier"
    STRING = "string"
    NUMBER = "number"
    SEMICOLON = "semicolon"
    PUNCT = "punct"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    start: int
    end: int


class LexError(ValueError):
    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule, self.detail = rule, detail


def _read_quoted(sql: str, start: int, quote: str, what: str) -> tuple[str, int]:
    """Read a quoted run where a doubled quote is an escaped quote. Return (text, end)."""
    i, parts = start + 1, []
    while True:
        close = sql.find(quote, i)
        if close == -1:
            raise LexError("unterminated_literal", f"{what} starting at offset {start}")
        parts.append(sql[i:close])
        if sql.startswith(quote * 2, close):
            parts.append(quote)
            i = close + 2
            continue
        return "".join(parts), close + 1


def tokenize(sql: str) -> list[Token]:
    """Split SQL into tokens, dropping whitespace and comments. Raises ``LexError``."""
    tokens: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
        elif sql.startswith("--", i):
            newline = sql.find("\n", i)
            i = n if newline == -1 else newline + 1
        elif sql.startswith("/*", i):
            close = sql.find("*/", i + 2)
            if close == -1:
                raise LexError("unterminated_literal", f"block comment starting at offset {i}")
            if "/*" in sql[i + 2 : close]:
                # DuckDB nests block comments; a non-nesting reader would disagree on
                # where this comment ends, so the guard refuses to guess.
                raise LexError("nested_comment", f"block comment inside a block comment at {i}")
            i = close + 2
        elif c == "'":
            prefix = tokens[-1] if tokens and tokens[-1].end == i else None
            if prefix and prefix.kind is TokenKind.WORD and len(prefix.text) == 1:
                # E'...' honors backslash escapes, so it ends somewhere else; other
                # one-letter prefixes are refused for the same reason: unknown rules.
                raise LexError("exotic_literal", f"{prefix.text}'...' string literal")
            text, end = _read_quoted(sql, i, "'", "string literal")
            tokens.append(Token(TokenKind.STRING, text, i, end))
            i = end
        elif c == '"':
            text, end = _read_quoted(sql, i, '"', "quoted identifier")
            tokens.append(Token(TokenKind.QUOTED, text, i, end))
            i = end
        elif c == "$":
            raise LexError("exotic_literal", "dollar quoting or a $ parameter")
        elif c == ";":
            tokens.append(Token(TokenKind.SEMICOLON, c, i, i + 1))
            i += 1
        elif c.isdigit():
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] in "._"):
                j += 1
            tokens.append(Token(TokenKind.NUMBER, sql[i:j], i, j))
            i = j
        elif c.isalpha() or c == "_":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            tokens.append(Token(TokenKind.WORD, sql[i:j], i, j))
            i = j
        else:
            tokens.append(Token(TokenKind.PUNCT, c, i, i + 1))
            i += 1
    return tokens


def _invisible_characters(sql: str) -> list[str]:
    return [
        f"U+{ord(ch):04X}"
        for ch in sql
        if unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\t\n\r"
    ]


def _is_denied_function(name: str) -> bool:
    name = name.lower()
    return name in DENIED_FUNCTIONS or name.startswith(DENIED_FUNCTION_PREFIXES)


def _lexical_violations(sql: str) -> list[Violation]:
    def violation(rule: str, detail: str) -> Violation:
        return Violation(layer=Layer.LEXER, rule=rule, detail=detail)

    if invisible := _invisible_characters(sql):
        return [violation("control_character", ", ".join(sorted(set(invisible))))]
    try:
        tokens = tokenize(sql)
    except LexError as exc:
        return [violation(exc.rule, exc.detail)]
    if not tokens:
        return [violation("empty", "no statement")]

    found: list[Violation] = []
    first_semicolon = next(
        (k for k, t in enumerate(tokens) if t.kind is TokenKind.SEMICOLON), len(tokens)
    )
    if any(t.kind is not TokenKind.SEMICOLON for t in tokens[first_semicolon:]):
        found.append(violation("multiple_statements", "code after the first ';'"))

    lead = next((t for t in tokens if t.text != "("), tokens[0])
    if lead.kind is not TokenKind.WORD or lead.text.upper() not in _LEADING_KEYWORDS:
        found.append(violation("not_a_select", f"statement starts with {lead.text[:40]!r}"))

    keywords = dict.fromkeys(
        t.text.upper()
        for t in tokens
        if t.kind is TokenKind.WORD and t.text.upper() in DENIED_KEYWORDS
    )
    found.extend(violation("denied_keyword", keyword) for keyword in keywords)

    functions = dict.fromkeys(
        t.text.lower()
        for t, after in pairwise(tokens)
        if t.kind in (TokenKind.WORD, TokenKind.QUOTED)
        and after.text == "("
        and _is_denied_function(t.text)
    )
    found.extend(violation("denied_function", name) for name in functions)
    return found


# --- engine --------------------------------------------------------------------------


def _is_table_ref(node: dict[str, Any]) -> bool:
    return "alias" in node and "sample" in node and "class" not in node


def _is_recursive(cte_entry: dict[str, Any]) -> bool:
    query = cte_entry.get("value", {}).get("query", {})
    return query.get("node", {}).get("type") == "RECURSIVE_CTE_NODE"


def _walk(node: Any, scope: frozenset[str], refs: list[str], others: list[str]) -> None:
    """Collect base-table references (minus CTE names in scope) and any other ref types.

    CTE scope is resolved conservatively: a CTE body sees the CTEs defined before it
    (and itself only when recursive); the statement body sees all of them. When the
    engine is more permissive than this, the guard treats the name as a real table,
    which can only make it stricter.
    """
    if isinstance(node, list):
        for item in node:
            _walk(item, scope, refs, others)
        return
    if not isinstance(node, dict):
        return

    entries = (node.get("cte_map") or {}).get("map") or []
    if entries:
        names = [entry["key"].lower() for entry in entries]
        for k, entry in enumerate(entries):
            visible = set(names[:k]) | ({names[k]} if _is_recursive(entry) else set())
            _walk(entry["value"], scope | visible, refs, others)
        scope = scope | set(names)

    if _is_table_ref(node):
        kind = node.get("type")
        if kind == "BASE_TABLE":
            parts = [node.get("catalog_name", ""), node.get("schema_name", ""), node["table_name"]]
            qualified = ".".join(part for part in parts if part).lower()
            if qualified not in scope:
                refs.append(qualified)
        elif kind == "TABLE_FUNCTION":
            others.append(f"TABLE_FUNCTION {node.get('function', {}).get('function_name', '?')}")
        elif kind not in _ALLOWED_REF_TYPES:
            others.append(f"{kind} {node.get('show_type', '')}".strip())

    for key, value in node.items():
        if key != "cte_map":
            _walk(value, scope, refs, others)


class SQLValidator:
    """Two-layer read-only validator. ``allowed_tables=None`` turns the allowlist off."""

    def __init__(
        self,
        allowed_tables: Iterable[str] | None = None,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.allowed = (
            None if allowed_tables is None else frozenset(t.lower() for t in allowed_tables)
        )
        self.max_chars = max_chars
        self._parser = duckdb.connect(":memory:", config=_PARSER_CONFIG)

    def close(self) -> None:
        self._parser.close()

    def validate(self, sql: str) -> ValidationReport:
        if len(sql) > self.max_chars:
            oversized = Violation(
                layer=Layer.LEXER, rule="oversized", detail=f"{len(sql)} > {self.max_chars} chars"
            )
            return ValidationReport(sql=sql, violations=(oversized,))
        if lexical := self.lexical_check(sql):
            return ValidationReport(sql=sql, violations=tuple(lexical))
        violations, tables = self.engine_check(sql)
        return ValidationReport(sql=sql, violations=tuple(violations), tables=tables)

    def lexical_check(self, sql: str) -> list[Violation]:
        """Layer 1 on its own."""
        return _lexical_violations(sql)

    def engine_check(self, sql: str) -> tuple[list[Violation], tuple[str, ...]]:
        """Layer 2 on its own. Returns the violations and the base tables read."""

        def violation(rule: str, detail: str) -> Violation:
            return Violation(layer=Layer.ENGINE, rule=rule, detail=detail)

        try:
            statements = self._parser.extract_statements(sql)
        except duckdb.Error as exc:
            return [violation("parse_error", _short(str(exc)))], ()
        if not statements:
            return [violation("empty", "no statement")], ()
        if len(statements) > 1:
            return [violation("multiple_statements", f"{len(statements)} statements")], ()
        if statements[0].type != duckdb.StatementType.SELECT:
            return [violation("not_a_select", statements[0].type.name)], ()

        tree = json.loads(self._parser.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
        if tree.get("error"):
            message = _short(tree.get("error_message", ""))
            # DuckDB reports PRAGMA as StatementType.SELECT (it rewrites it into a
            # pragma function call); the serializer is what refuses it.
            rule = "not_a_select" if "Only SELECT statements" in message else "parse_error"
            return [violation(rule, message)], ()

        refs: list[str] = []
        others: list[str] = []
        _walk(tree["statements"], frozenset(), refs, others)
        found = [
            violation("table_function" if other.startswith("TABLE_FUNCTION") else "unsupported_ref",
                      other.removeprefix("TABLE_FUNCTION "))
            for other in dict.fromkeys(others)
        ]  # fmt: skip
        if found:
            return found, ()

        # The syntax tree is the single source of truth for table references.
        # get_table_names() was evaluated and rejected: it binds the query (USING joins
        # fail without the real catalog), omits system views such as sqlite_master and
        # information_schema, and returns clause text for TABLESAMPLE references.
        tables = set(refs)
        if self.allowed is not None:
            found.extend(
                violation("table_not_allowed", table)
                for table in sorted(tables)
                if not self._is_allowed(table)
            )
        return found, tuple(sorted(tables))

    def _is_allowed(self, qualified: str) -> bool:
        *schema, table = qualified.split(".")
        return len(schema) <= 1 and "".join(schema) in _ALLOWED_SCHEMAS and table in self.allowed
