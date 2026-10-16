from datetime import date

import duckdb
import pytest

from nl_sql_agent.dataset import PUBLIC_TABLES, build_database, ensure_database


def query(path, sql):
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute(sql).fetchall()


def snapshot(path):
    tables = [*PUBLIC_TABLES, "payroll"]
    return {t: query(path, f"SELECT * FROM {t} ORDER BY ALL") for t in tables}


def test_builds_the_public_tables_and_one_confidential_table(db_path):
    names = {row[0] for row in query(db_path, "SELECT table_name FROM duckdb_tables()")}
    assert names == {*PUBLIC_TABLES, "payroll"}


def test_sizes_are_small_enough_for_ci_and_big_enough_to_aggregate(db_path):
    counts = dict(
        query(
            db_path,
            """
        SELECT 'customers', count(*) FROM customers UNION ALL
        SELECT 'products', count(*) FROM products UNION ALL
        SELECT 'orders', count(*) FROM orders UNION ALL
        SELECT 'order_items', count(*) FROM order_items
    """,
        )
    )
    assert counts["customers"] == 40
    assert counts["products"] == 15
    assert counts["orders"] == 400
    assert 400 < counts["order_items"] < 1600


def test_same_seed_builds_identical_data(db_path, tmp_path):
    again = build_database(tmp_path / "again.duckdb")
    assert snapshot(again) == snapshot(db_path)


def test_a_different_seed_builds_different_data(db_path, tmp_path):
    other = build_database(tmp_path / "other.duckdb", seed=8)
    assert snapshot(other)["orders"] != snapshot(db_path)["orders"]


def test_foreign_keys_hold(db_path):
    orphans = query(
        db_path,
        """
        SELECT count(*) FROM orders o LEFT JOIN customers c USING (customer_id)
        WHERE c.customer_id IS NULL
        UNION ALL
        SELECT count(*) FROM order_items i LEFT JOIN orders o USING (order_id)
        WHERE o.order_id IS NULL
        UNION ALL
        SELECT count(*) FROM order_items i LEFT JOIN products p USING (product_id)
        WHERE p.product_id IS NULL
    """,
    )
    assert orphans == [(0,), (0,), (0,)]


def test_values_stay_inside_their_documented_domains(db_path):
    assert {r[0] for r in query(db_path, "SELECT DISTINCT status FROM orders")} == {
        "paid",
        "refunded",
        "pending",
    }
    assert {r[0] for r in query(db_path, "SELECT DISTINCT channel FROM orders")} == {
        "web",
        "partner",
        "sales",
    }
    first, last = query(db_path, "SELECT min(order_date), max(order_date) FROM orders")[0]
    assert date(2025, 1, 1) <= first <= last <= date(2025, 12, 31)


def test_order_ids_follow_order_dates(db_path):
    rows = query(db_path, "SELECT order_date FROM orders ORDER BY order_id")
    dates = [row[0] for row in rows]
    assert dates == sorted(dates)


def test_partner_orders_are_charged_ten_percent_below_list(db_path):
    rows = query(
        db_path,
        """
        SELECT DISTINCT o.channel = 'partner', i.unit_price = round(p.unit_price * 0.9, 2)
        FROM order_items i JOIN orders o USING (order_id) JOIN products p USING (product_id)
    """,
    )
    assert sorted(rows) == [(False, False), (True, True)]


def test_every_public_column_is_documented_in_the_catalog(db_path):
    undocumented = query(
        db_path,
        f"""
        SELECT table_name, column_name FROM duckdb_columns()
        WHERE table_name IN {PUBLIC_TABLES} AND (comment IS NULL OR comment = '')
    """,
    )
    assert undocumented == []


def test_ensure_database_builds_once_and_then_reuses_the_file(tmp_path):
    path = tmp_path / "shop.duckdb"
    assert ensure_database(path) == path
    stamp = path.stat().st_mtime_ns
    ensure_database(path)
    assert path.stat().st_mtime_ns == stamp


def test_build_replaces_an_existing_file(tmp_path):
    path = tmp_path / "shop.duckdb"
    path.write_bytes(b"not a database")
    build_database(path)
    assert query(path, "SELECT count(*) FROM products") == [(15,)]


def test_build_refuses_a_path_inside_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_database(tmp_path / "missing" / "shop.duckdb")
