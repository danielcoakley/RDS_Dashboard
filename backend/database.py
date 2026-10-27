from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any
from urllib.parse import unquote, urlparse


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
DATABASE_URL_ENV = "SAAS_DATABASE_URL"
DEFAULT_DATABASE_URL = "sqlite:///:memory:"
SQLITE_BACKEND = "sqlite"
POSTGRES_BACKEND = "postgresql"
SUPPORTED_DATABASE_BACKENDS = {SQLITE_BACKEND, POSTGRES_BACKEND}


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
    backend: str
    sqlite_path: str | None = None

    @property
    def is_local(self) -> bool:
        return self.backend == SQLITE_BACKEND


class PostgresConnectionAdapter:
    def __init__(self, conn: Any):
        self.conn = conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        return self.conn.execute(_sqlite_placeholders_to_postgres(sql), params)

    def executescript(self, script: str) -> None:
        for statement in split_sql_statements(script):
            self.conn.execute(statement)

    def __enter__(self) -> "PostgresConnectionAdapter":
        self.conn.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self.conn.__exit__(exc_type, exc, traceback)


class _LockingCursor:
    """A sqlite3.Cursor wrapper that keeps fetch calls under the connection's lock.

    Python's sqlite3 module requires external synchronization for concurrent use
    of one connection from multiple threads. Locking only around `execute` is not
    enough: `execute(...).fetchall()` is two separate calls, and without also
    locking the fetch, another thread's `execute` could interleave between them.
    """

    __slots__ = ("_cursor", "_lock")

    def __init__(self, cursor: sqlite3.Cursor, lock: threading.RLock):
        self._cursor = cursor
        self._lock = lock

    def fetchone(self) -> Any:
        with self._lock:
            return self._cursor.fetchone()

    def fetchall(self) -> Any:
        with self._lock:
            return self._cursor.fetchall()

    def fetchmany(self, size: int | None = None) -> Any:
        with self._lock:
            if size is None:
                return self._cursor.fetchmany()
            return self._cursor.fetchmany(size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class SQLiteConnectionAdapter:
    """Serializes access to a shared sqlite3.Connection across request threads.

    FastAPI runs sync endpoints via a thread pool, so multiple requests can
    execute concurrently against the single shared SQLite connection created at
    startup. sqlite3 connections are not safe for concurrent multi-threaded use
    even with check_same_thread=False, so every operation that touches the
    connection is serialized with an RLock. RLock (not Lock) is required because
    `with self.conn:` transaction blocks nest further `self.conn.execute(...)`
    calls from the same thread.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = threading.RLock()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _LockingCursor:
        with self._lock:
            cursor = self.conn.execute(sql, params)
            return _LockingCursor(cursor, self._lock)

    def executemany(self, sql: str, params_seq: Any) -> _LockingCursor:
        with self._lock:
            cursor = self.conn.executemany(sql, params_seq)
            return _LockingCursor(cursor, self._lock)

    def executescript(self, script: str) -> None:
        with self._lock:
            self.conn.executescript(script)

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def __enter__(self) -> "SQLiteConnectionAdapter":
        self._lock.acquire()
        try:
            self.conn.__enter__()
        except BaseException:
            self._lock.release()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        try:
            return self.conn.__exit__(exc_type, exc, traceback)
        finally:
            self._lock.release()


def _sqlite_placeholders_to_postgres(sql: str) -> str:
    return sql.replace("?", "%s")


def split_sql_statements(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def _sqlite_path_from_url(url: str) -> str:
    parsed = urlparse(url)
    raw_path = unquote(parsed.path)
    if raw_path in {"", "/", "/:memory:"}:
        return ":memory:"
    if raw_path.startswith("//"):
        return raw_path[1:]
    if raw_path.startswith("/"):
        return raw_path[1:]
    return raw_path


def database_config_from_url(url: str) -> DatabaseConfig:
    value = url.strip()
    if not value:
        raise ValueError(f"{DATABASE_URL_ENV} cannot be empty")

    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme == "postgres":
        scheme = POSTGRES_BACKEND

    if scheme == SQLITE_BACKEND:
        if parsed.netloc:
            raise ValueError("SQLite database URLs must not include a host")
        return DatabaseConfig(
            url=value,
            backend=SQLITE_BACKEND,
            sqlite_path=_sqlite_path_from_url(value),
        )

    if scheme == POSTGRES_BACKEND:
        if not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(
                f"{DATABASE_URL_ENV} must include a PostgreSQL host and database name"
            )
        return DatabaseConfig(url=value, backend=POSTGRES_BACKEND)

    raise ValueError(
        f"{DATABASE_URL_ENV} must use one of: {', '.join(sorted(SUPPORTED_DATABASE_BACKENDS))}"
    )


def database_config_from_env() -> DatabaseConfig:
    return database_config_from_url(os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL))


def connect_database(path: str | Path = ":memory:") -> SQLiteConnectionAdapter:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return SQLiteConnectionAdapter(conn)


def connect_database_from_config(config: DatabaseConfig) -> Any:
    if config.backend == SQLITE_BACKEND:
        return connect_database(config.sqlite_path or ":memory:")
    if config.backend == POSTGRES_BACKEND:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL metadata storage") from exc
        return PostgresConnectionAdapter(psycopg.connect(config.url, row_factory=dict_row))
    raise ValueError(f"Unsupported database backend: {config.backend}")


def connect_database_from_env() -> Any:
    return connect_database_from_config(database_config_from_env())


def apply_migrations(
    conn: Any,
    migrations_dir: str | Path = MIGRATIONS_DIR,
) -> list[str]:
    migration_path = Path(migrations_dir)
    if not migration_path.exists():
        raise FileNotFoundError(f"Migration directory not found: {migration_path}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {
        row["version"]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    applied_now: list[str] = []

    for path in sorted(migration_path.glob("*.sql")):
        version = path.stem
        if version in applied:
            continue
        with conn:
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
        applied_now.append(version)

    return applied_now


def initialize_database(path: str | Path = ":memory:") -> SQLiteConnectionAdapter:
    conn = connect_database(path)
    apply_migrations(conn)
    return conn


def initialize_database_from_config(config: DatabaseConfig) -> Any:
    conn = connect_database_from_config(config)
    apply_migrations(conn)
    return conn


def initialize_database_from_env() -> Any:
    return initialize_database_from_config(database_config_from_env())
