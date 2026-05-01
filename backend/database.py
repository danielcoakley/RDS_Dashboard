from __future__ import annotations

from pathlib import Path
import sqlite3


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def connect_database(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def apply_migrations(
    conn: sqlite3.Connection,
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


def initialize_database(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = connect_database(path)
    apply_migrations(conn)
    return conn
