import pytest

from nl_sql_agent.dataset import build_database


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    """One synthetic database for the whole session. Tests only ever read it."""
    return build_database(tmp_path_factory.mktemp("db") / "shop.duckdb")
