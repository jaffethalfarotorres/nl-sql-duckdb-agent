"""A small synthetic retail database, generated in code with a fixed seed.

An office-supplies seller with business customers: 40 customers, 15 products,
400 orders placed in 2025 and their line items. Every value comes from a seeded
``random.Random``, so the same seed always builds the same database, on any
machine, with no download. Customers are companies with invented names; there is
no personal data.

The schema documentation lives in the database itself (``COMMENT ON``), so the
prompt the model sees is generated from the catalog and cannot drift from it.

``payroll`` is deliberately confidential: it exists in the file but is not in
``PUBLIC_TABLES``, so the agent is never told about it and the validator's
allowlist rejects any query that reads it.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import duckdb

SEED = 7
PUBLIC_TABLES = ("customers", "products", "orders", "order_items")

_SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    company VARCHAR NOT NULL,
    country VARCHAR NOT NULL,
    segment VARCHAR NOT NULL,
    signup_date DATE NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date DATE NOT NULL,
    status VARCHAR NOT NULL,
    channel VARCHAR NOT NULL
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders (order_id),
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
CREATE TABLE payroll (
    employee_id INTEGER PRIMARY KEY,
    team VARCHAR NOT NULL,
    annual_salary DECIMAL(10, 2) NOT NULL
);
"""

_COMMENTS = {
    "customers": (
        "One row per business customer.",
        {
            "customer_id": "Primary key.",
            "company": "Company name (synthetic).",
            "country": "Billing country.",
            "segment": "One of 'enterprise', 'smb', 'startup'.",
            "signup_date": "Date the account was created.",
        },
    ),
    "products": (
        "Catalog of office products.",
        {
            "product_id": "Primary key.",
            "name": "Product name.",
            "category": "One of 'furniture', 'electronics', 'accessories'.",
            "unit_price": "Current list price in USD.",
        },
    ),
    "orders": (
        "One row per order placed in 2025.",
        {
            "order_id": "Primary key. Increases with order_date.",
            "customer_id": "References customers.customer_id.",
            "order_date": "Date the order was placed.",
            "status": "One of 'paid', 'refunded', 'pending'. Revenue counts paid orders only.",
            "channel": "One of 'web', 'partner', 'sales'.",
        },
    ),
    "order_items": (
        "Line items of each order. Revenue of a line = quantity * unit_price.",
        {
            "order_id": "References orders.order_id.",
            "product_id": "References products.product_id.",
            "quantity": "Units sold.",
            "unit_price": "Price per unit actually charged in USD (partners get 10% off list).",
        },
    ),
    "payroll": ("Employee compensation. Confidential: not exposed to the agent.", {}),
}

_ADJECTIVES = (
    "Blue", "Bright", "Cedar", "Clear", "Copper", "Golden", "Granite", "Harbor", "Iron",
    "Maple", "North", "Quiet", "Red", "Silver", "Summit", "Swift", "True", "Urban",
)  # fmt: skip
_NOUNS = (
    "Analytics", "Bakery", "Clinic", "Design", "Foods", "Labs", "Logistics", "Media",
    "Motors", "Studio", "Systems", "Ventures",
)  # fmt: skip
_COUNTRIES = {
    "United States": 9,
    "Mexico": 6,
    "Costa Rica": 5,
    "Canada": 4,
    "Spain": 4,
    "Colombia": 3,
    "Brazil": 3,
    "Germany": 2,
}
_SEGMENTS = {"smb": 5, "startup": 3, "enterprise": 2}
_PRODUCTS = (
    ("Standing Desk", "furniture", "549.00"),
    ("Ergonomic Chair", "furniture", "389.00"),
    ("Filing Cabinet", "furniture", "179.00"),
    ("Meeting Table", "furniture", "899.00"),
    ("27-inch Monitor", "electronics", "329.00"),
    ("USB-C Dock", "electronics", "189.00"),
    ("Noise-Cancelling Headset", "electronics", "249.00"),
    ("HD Webcam", "electronics", "89.00"),
    ("Laser Printer", "electronics", "299.00"),
    ("Mechanical Keyboard", "accessories", "119.00"),
    ("Wireless Mouse", "accessories", "39.00"),
    ("Laptop Stand", "accessories", "59.00"),
    ("Desk Lamp", "accessories", "45.00"),
    ("Cable Kit", "accessories", "19.00"),
    ("Monitor Arm", "accessories", "99.00"),
)
_STATUSES = {"paid": 80, "pending": 12, "refunded": 8}
_CHANNELS = {"web": 60, "partner": 25, "sales": 15}
_TEAMS = ("sales", "support", "engineering", "finance")
_PARTNER_PRICE = Decimal("0.90")
_CENT = Decimal("0.01")

N_CUSTOMERS, N_ORDERS, N_EMPLOYEES = 40, 400, 12


def _weighted(rng: random.Random, weights: dict[str, int]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _rows(seed: int) -> dict[str, list[tuple]]:
    rng = random.Random(seed)

    names = rng.sample([f"{a} {n}" for a in _ADJECTIVES for n in _NOUNS], N_CUSTOMERS)
    customers = [
        (
            customer_id,
            name,
            _weighted(rng, _COUNTRIES),
            _weighted(rng, _SEGMENTS),
            date(2023, 1, 1) + timedelta(days=rng.randrange(730)),
        )
        for customer_id, name in enumerate(names, start=1)
    ]

    products = [
        (product_id, name, category, Decimal(price))
        for product_id, (name, category, price) in enumerate(_PRODUCTS, start=1)
    ]

    order_dates = sorted(
        date(2025, 1, 1) + timedelta(days=rng.randrange(365)) for _ in range(N_ORDERS)
    )
    orders, items = [], []
    for order_id, order_date in enumerate(order_dates, start=1):
        channel = _weighted(rng, _CHANNELS)
        customer_id = rng.randint(1, N_CUSTOMERS)
        orders.append((order_id, customer_id, order_date, _weighted(rng, _STATUSES), channel))
        for product_id in rng.sample(
            range(1, len(_PRODUCTS) + 1), rng.choice((1, 1, 1, 2, 2, 3, 4))
        ):
            price = products[product_id - 1][3]
            if channel == "partner":
                price = (price * _PARTNER_PRICE).quantize(_CENT)
            items.append((order_id, product_id, rng.randint(1, 5), price))

    payroll = [
        (employee_id, rng.choice(_TEAMS), Decimal(rng.randint(38, 140) * 1000))
        for employee_id in range(1, N_EMPLOYEES + 1)
    ]
    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": items,
        "payroll": payroll,
    }


def _insert(con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple]) -> None:
    """Insert all rows of a table through a single JSON parameter.

    The DuckDB Python client pays a failed ``import`` attempt per bound value when
    numpy and pandas are not installed, which made a per-value INSERT take seconds.
    One JSON string per table is one bound value; DuckDB unpacks it and casts each
    field to the column type.
    """
    columns = [
        name
        for (name,) in con.execute(
            "SELECT column_name FROM duckdb_columns() WHERE table_name = ? ORDER BY column_index",
            [table],
        ).fetchall()
    ]
    payload = json.dumps([dict(zip(columns, row, strict=True)) for row in rows], default=str)
    struct = ", ".join(f"{column} VARCHAR" for column in columns)
    con.execute(
        f"INSERT INTO {table} BY NAME "
        f"SELECT unnest(CAST(?::JSON AS STRUCT({struct})[]), recursive := true)",
        [payload],
    )


def build_database(path: Path, *, seed: int = SEED) -> Path:
    """(Re)build the database file at ``path`` from ``seed``. Replaces any existing file."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"directory does not exist: {path.parent}")
    for stale in (path, path.with_name(path.name + ".wal")):
        stale.unlink(missing_ok=True)

    with duckdb.connect(str(path)) as con:
        con.execute(_SCHEMA)
        for table, rows in _rows(seed).items():
            _insert(con, table, rows)
        for table, (table_comment, columns) in _COMMENTS.items():
            # Comments are fixed strings from this module, never user input.
            con.execute(f"COMMENT ON TABLE {table} IS '{table_comment}'")
            for column, comment in columns.items():
                escaped = comment.replace("'", "''")
                con.execute(f"COMMENT ON COLUMN {table}.{column} IS '{escaped}'")
        con.execute("CHECKPOINT")
    return path


def ensure_database(path: Path) -> Path:
    """Build the database only if the file does not exist yet."""
    path = Path(path)
    return path if path.exists() else build_database(path)
