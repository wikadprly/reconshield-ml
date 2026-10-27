"""DetectionStore must survive a database that goes away (audit finding T1)."""

import psycopg2
import pytest

from pipeline.detect import DetectionStore


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, _params=None):
        self.conn.attempts.append(" ".join(sql.split()[:2]).lower())
        if self.conn.broken:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

    def fetchone(self):
        return ("INC-1057",)


class FakeConn:
    """Minimal psycopg2 connection: autocommit, commit/rollback, cursor."""

    def __init__(self, broken=False):
        self.closed = 0
        self.broken = broken
        self.attempts = []
        self.autocommit = True
        self.committed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        if self.broken:
            raise psycopg2.OperationalError("connection lost during commit")
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = 1


def _flow():
    return {"src": "10.10.30.5", "sport": 40000, "dst": "10.10.20.254", "dport": 22}


def test_dry_run_never_touches_the_database():
    def connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not open a connection")

    store = DetectionStore("", dry_run=True, connect=connect)

    assert store.insert(_flow(), "Reconnaissance", 0.97) is None
    assert store.conn is None


def test_startup_fails_fast_when_the_database_is_unreachable():
    def connect(*_args, **_kwargs):
        raise psycopg2.OperationalError("no route to host")

    with pytest.raises(psycopg2.OperationalError):
        DetectionStore("postgresql://unreachable/db", dry_run=False, connect=connect)


def test_insert_reconnects_when_the_connection_dies():
    conns = []

    def connect(*_args, **_kwargs):
        # The first connection arrives dead; the one opened afterwards works.
        conn = FakeConn(broken=not conns)
        conns.append(conn)
        return conn

    store = DetectionStore("postgresql://db", dry_run=False, connect=connect)
    incident = store.insert(_flow(), "Reconnaissance", 0.97)

    assert len(conns) == 2, "the store should have opened a second connection"
    assert incident == "INC-1058"
    assert store.errors == 1


def test_id_allocation_holds_one_lock_then_commits_once():
    """T3: max-id read + INSERT must share one locked transaction."""

    def connect(*_args, **_kwargs):
        return FakeConn()

    store = DetectionStore("postgresql://db", dry_run=False, connect=connect)
    incident = store.insert(_flow(), "Reconnaissance", 0.97)
    conn = store.conn

    assert incident == "INC-1058"
    assert conn.attempts[0] == "select pg_advisory_xact_lock(%s)", "lock must be taken first"
    assert conn.attempts[1].startswith("select id"), "max id read inside the lock"
    assert conn.committed, "id allocation + insert must commit together"
    assert conn.autocommit is True, "autocommit must be restored afterwards"


def test_total_outage_is_counted_instead_of_raising():
    def connect(*_args, **_kwargs):
        return FakeConn(broken=True)

    store = DetectionStore("postgresql://db", dry_run=False, connect=connect)

    assert store.insert(_flow(), "Reconnaissance", 0.97) is None
    assert store.errors >= 1
    assert not store._healthy()

    store.close()
