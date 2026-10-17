"""SQLite adapter for MemoryRepository; no JSON, embeddings or Agent imports."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from memory_repository import MemoryFact, MutationResult, MutationStatus


class SQLiteMemoryRepository:
    """Own one SQLite connection at an explicitly supplied database path.

    Each instance is confined to its creating thread (sqlite3's default).
    Independent instances may share a database; BEGIN IMMEDIATE serializes
    writers, and expected_version rejects stale snapshots. Lock timeouts and
    other SQLite errors propagate. Opening initializes only this adapter's
    table/index; it does not import or migrate any existing JSON data.
    """

    def __init__(self, database_path: str | Path, *, timeout: float = 5.0):
        self._connection = sqlite3.connect(
            database_path, timeout=timeout, isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._closed = False
        self._transaction_depth = 0
        try:
            with self.transaction():
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS memory_facts (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE CHECK (length(trim(id)) > 0),
                        sender_id TEXT NOT NULL CHECK (length(trim(sender_id)) > 0),
                        content TEXT NOT NULL CHECK (length(trim(content)) > 0),
                        version INTEGER NOT NULL DEFAULT 1
                            CHECK (typeof(version) = 'integer' AND version >= 1)
                    )
                """)
                self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_facts_sender_sequence
                    ON memory_facts (sender_id, sequence)
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS memory_senders (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        sender_id TEXT NOT NULL UNIQUE
                            CHECK (length(trim(sender_id)) > 0)
                    )
                """)
                # Backward compatibility for databases created before the
                # sender registry existed. Facts and their IDs never change.
                self._connection.execute("""
                    INSERT INTO memory_senders (sender_id)
                    SELECT sender_id FROM memory_facts
                    WHERE sender_id NOT IN (SELECT sender_id FROM memory_senders)
                    GROUP BY sender_id ORDER BY MIN(sequence)
                """)
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    @staticmethod
    def _validate_text(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    @staticmethod
    def _validate_version(value: int) -> None:
        if type(value) is not int or value < 1:
            raise ValueError("expected_version must be a positive integer.")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Memory repository is closed.")
        if self._transaction_depth and not self._connection.in_transaction:
            raise RuntimeError("Memory repository transaction was aborted by SQLite.")

    @staticmethod
    def _fact(row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            id=row["id"], sender_id=row["sender_id"],
            content=row["content"], version=row["version"],
        )

    def list(self, sender_id: str) -> list[MemoryFact]:
        self._ensure_open()
        self._validate_text(sender_id, "sender_id")
        rows = self._connection.execute(
            "SELECT id, sender_id, content, version FROM memory_facts "
            "WHERE sender_id = ? ORDER BY sequence",
            (sender_id,),
        ).fetchall()
        return [self._fact(row) for row in rows]

    def list_senders(self) -> list[str]:
        """Include empty senders, in registration order."""
        self._ensure_open()
        return [row["sender_id"] for row in self._connection.execute(
            "SELECT sender_id FROM memory_senders ORDER BY sequence",
        ).fetchall()]

    def list_facts(self, sender_id: str) -> list[MemoryFact]:
        """Explicit fact-reader name; retain list() for existing callers."""
        return self.list(sender_id)

    def ensure_sender(self, sender_id: str) -> None:
        """Register a namespace without inventing a placeholder fact."""
        self._ensure_open()
        self._validate_text(sender_id, "sender_id")
        with self.transaction():
            self._connection.execute(
                "INSERT INTO memory_senders (sender_id) VALUES (?) "
                "ON CONFLICT(sender_id) DO NOTHING",
                (sender_id,),
            )

    def add(self, sender_id: str, content: str) -> MemoryFact:
        self._ensure_open()
        self._validate_text(sender_id, "sender_id")
        self._validate_text(content, "content")
        fact = MemoryFact(id=str(uuid4()), sender_id=sender_id, content=content, version=1)
        with self.transaction():
            self.ensure_sender(sender_id)
            self._connection.execute(
                "INSERT INTO memory_facts (id, sender_id, content, version) "
                "VALUES (?, ?, ?, ?)",
                (fact.id, fact.sender_id, fact.content, fact.version),
            )
        return fact

    def _failed_mutation(self, sender_id: str, fact_id: str) -> MutationResult:
        # Called while holding the same write transaction as the failed CAS.
        exists = self._connection.execute(
            "SELECT 1 FROM memory_facts WHERE sender_id = ? AND id = ?",
            (sender_id, fact_id),
        ).fetchone()
        status = MutationStatus.VERSION_CONFLICT if exists else MutationStatus.NOT_FOUND
        return MutationResult(status=status)

    def update(
        self, sender_id: str, fact_id: str, content: str, *, expected_version: int,
    ) -> MutationResult:
        self._ensure_open()
        self._validate_text(sender_id, "sender_id")
        self._validate_text(fact_id, "fact_id")
        self._validate_text(content, "content")
        self._validate_version(expected_version)
        with self.transaction():
            cursor = self._connection.execute(
                "UPDATE memory_facts SET content = ?, version = version + 1 "
                "WHERE sender_id = ? AND id = ? AND version = ?",
                (content, sender_id, fact_id, expected_version),
            )
            if cursor.rowcount == 0:
                return self._failed_mutation(sender_id, fact_id)
            row = self._connection.execute(
                "SELECT id, sender_id, content, version FROM memory_facts "
                "WHERE sender_id = ? AND id = ?",
                (sender_id, fact_id),
            ).fetchone()
            return MutationResult(status=MutationStatus.SUCCESS, fact=self._fact(row))

    def delete(
        self, sender_id: str, fact_id: str, *, expected_version: int,
    ) -> MutationResult:
        self._ensure_open()
        self._validate_text(sender_id, "sender_id")
        self._validate_text(fact_id, "fact_id")
        self._validate_version(expected_version)
        with self.transaction():
            row = self._connection.execute(
                "SELECT id, sender_id, content, version FROM memory_facts "
                "WHERE sender_id = ? AND id = ? AND version = ?",
                (sender_id, fact_id, expected_version),
            ).fetchone()
            if row is None:
                return self._failed_mutation(sender_id, fact_id)
            self._connection.execute(
                "DELETE FROM memory_facts WHERE sender_id = ? AND id = ? AND version = ?",
                (sender_id, fact_id, expected_version),
            )
            return MutationResult(status=MutationStatus.SUCCESS, fact=self._fact(row))

    @contextmanager
    def transaction(self) -> Iterator["SQLiteMemoryRepository"]:
        self._ensure_open()
        outermost = self._transaction_depth == 0
        if outermost:
            self._connection.execute("BEGIN IMMEDIATE")
        else:
            # A fixed internal name is safe: SQLite targets the innermost
            # matching savepoint, including recursively nested scopes.
            self._connection.execute("SAVEPOINT memory_repository_nested")
        self._transaction_depth += 1
        try:
            yield self
            if not self._connection.in_transaction:
                raise RuntimeError("Memory repository transaction was aborted by SQLite.")
            if outermost:
                self._connection.commit()
            else:
                self._connection.execute("RELEASE SAVEPOINT memory_repository_nested")
        except BaseException:
            # SQLite can roll back the entire transaction itself (e.g. a
            # RAISE(ROLLBACK) trigger). Its savepoints then no longer exist.
            if self._connection.in_transaction:
                if outermost:
                    self._connection.rollback()
                else:
                    self._connection.execute("ROLLBACK TO SAVEPOINT memory_repository_nested")
                    self._connection.execute("RELEASE SAVEPOINT memory_repository_nested")
            raise
        finally:
            self._transaction_depth -= 1

    def close(self) -> None:
        if self._closed:
            return
        if self._transaction_depth:
            raise RuntimeError("Cannot close a memory repository inside a transaction.")
        self._connection.close()
        self._closed = True

    def __enter__(self) -> "SQLiteMemoryRepository":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
