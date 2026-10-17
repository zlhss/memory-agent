"""Repository contract tests using only disposable SQLite databases."""

import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID

from memory_repository import MemoryFact, MemoryRepository, MutationStatus
from sqlite_memory_repository import SQLiteMemoryRepository


class SQLiteMemoryRepositoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="memory-repository-test-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "facts.sqlite3"
        self.repository: MemoryRepository = SQLiteMemoryRepository(self.path)
        self.addCleanup(self.repository.close)

    def test_empty_and_sender_scoped_lists_preserve_insertion_order(self):
        self.assertEqual(self.repository.list("alpha"), [])
        first = self.repository.add("alpha", "联系人住在苏州")
        other = self.repository.add("beta", "联系人住在厦门")
        second = self.repository.add("alpha", "联系人喜欢摄影")
        self.assertEqual(self.repository.list("alpha"), [first, second])
        self.assertEqual(self.repository.list("beta"), [other])
        self.assertEqual(self.repository.list("missing"), [])

    def test_add_returns_immutable_unique_facts_and_preserves_exact_content(self):
        text = "  联系人喜欢摄影\n和音乐 ' quoted \" text  "
        first = self.repository.add("alpha", text)
        second = self.repository.add("alpha", text)
        self.assertIsInstance(first, MemoryFact)
        self.assertEqual(UUID(first.id).version, 4)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual((first.sender_id, first.content, first.version), ("alpha", text, 1))
        self.assertEqual(self.repository.list("alpha"), [first, second])
        with self.assertRaises(FrozenInstanceError):
            first.content = "changed"

    def test_standalone_mutations_survive_close_and_reopen(self):
        fact = self.repository.add("alpha", "联系人住在苏州")
        result = self.repository.update("alpha", fact.id, "联系人住在南京", expected_version=1)
        self.repository.close()
        with SQLiteMemoryRepository(self.path) as reopened:
            self.assertEqual(reopened.list("alpha"), [result.fact])
            self.assertEqual(reopened.delete("alpha", fact.id, expected_version=2).status, MutationStatus.SUCCESS)
        with SQLiteMemoryRepository(self.path) as reopened:
            self.assertEqual(reopened.list("alpha"), [])

    def test_update_keeps_id_increments_version_and_preserves_list_position(self):
        first = self.repository.add("alpha", "联系人住在苏州")
        second = self.repository.add("alpha", "联系人喜欢摄影")
        result = self.repository.update("alpha", first.id, "联系人住在南京", expected_version=1)
        self.assertEqual(result.status, MutationStatus.SUCCESS)
        self.assertEqual(result.fact, MemoryFact(first.id, "alpha", "联系人住在南京", 2))
        self.assertEqual(self.repository.list("alpha"), [result.fact, second])
        self.assertEqual(first.version, 1)  # Earlier retrieval snapshots remain immutable.

    def test_same_content_update_still_increments_version(self):
        fact = self.repository.add("alpha", "联系人喜欢摄影")
        result = self.repository.update("alpha", fact.id, fact.content, expected_version=1)
        self.assertEqual(result.status, MutationStatus.SUCCESS)
        self.assertEqual(result.fact.version, 2)
        self.assertEqual(result.fact.id, fact.id)

    def test_delete_returns_deleted_snapshot_and_removes_only_selected_id(self):
        first = self.repository.add("alpha", "same content")
        second = self.repository.add("alpha", "same content")
        other = self.repository.add("beta", "same content")
        result = self.repository.delete("alpha", second.id, expected_version=1)
        self.assertEqual(result.status, MutationStatus.SUCCESS)
        self.assertEqual(result.fact, second)
        self.assertEqual(self.repository.list("alpha"), [first])
        self.assertEqual(self.repository.list("beta"), [other])
        replacement = self.repository.add("alpha", "same content")
        self.assertNotIn(replacement.id, (first.id, second.id, other.id))

    def test_update_rejects_stale_version_without_changing_state(self):
        fact = self.repository.add("alpha", "old")
        current = self.repository.update("alpha", fact.id, "new", expected_version=1).fact
        result = self.repository.update("alpha", fact.id, "stale overwrite", expected_version=1)
        self.assertEqual(result.status, MutationStatus.VERSION_CONFLICT)
        self.assertIsNone(result.fact)
        self.assertEqual(self.repository.list("alpha"), [current])

    def test_delete_rejects_stale_version_without_changing_state(self):
        fact = self.repository.add("alpha", "old")
        current = self.repository.update("alpha", fact.id, "new", expected_version=1).fact
        result = self.repository.delete("alpha", fact.id, expected_version=1)
        self.assertEqual(result.status, MutationStatus.VERSION_CONFLICT)
        self.assertIsNone(result.fact)
        self.assertEqual(self.repository.list("alpha"), [current])

    def test_future_version_is_also_a_conflict(self):
        fact = self.repository.add("alpha", "original")
        self.assertEqual(self.repository.update("alpha", fact.id, "changed", expected_version=2).status, MutationStatus.VERSION_CONFLICT)
        self.assertEqual(self.repository.delete("alpha", fact.id, expected_version=2).status, MutationStatus.VERSION_CONFLICT)
        self.assertEqual(self.repository.list("alpha"), [fact])

    def test_missing_and_previously_deleted_ids_return_not_found(self):
        for fact_id in ("unknown", self.repository.add("alpha", "removed").id):
            self.repository.delete("alpha", fact_id, expected_version=1)
            with self.subTest(fact_id=fact_id):
                for result in (
                    self.repository.update("alpha", fact_id, "replacement", expected_version=1),
                    self.repository.delete("alpha", fact_id, expected_version=1),
                ):
                    self.assertEqual(result.status, MutationStatus.NOT_FOUND)
                    self.assertIsNone(result.fact)

    def test_wrong_sender_cannot_update_or_delete_or_observe_version(self):
        fact = self.repository.add("alpha", "private")
        for version in (1, 2):
            with self.subTest(version=version):
                for result in (
                    self.repository.update("beta", fact.id, "pollution", expected_version=version),
                    self.repository.delete("beta", fact.id, expected_version=version),
                ):
                    self.assertEqual(result.status, MutationStatus.NOT_FOUND)
                    self.assertIsNone(result.fact)
        self.assertEqual(self.repository.list("alpha"), [fact])
        self.assertEqual(self.repository.list("beta"), [])

    def test_two_connections_detect_stale_retrieval_snapshots(self):
        fact = self.repository.add("alpha", "original")
        with SQLiteMemoryRepository(self.path) as second:
            snapshot = second.list("alpha")[0]
            current = self.repository.update("alpha", fact.id, "winner", expected_version=1).fact
            self.assertEqual(second.update("alpha", snapshot.id, "stale", expected_version=snapshot.version).status, MutationStatus.VERSION_CONFLICT)
            self.assertEqual(second.delete("alpha", snapshot.id, expected_version=snapshot.version).status, MutationStatus.VERSION_CONFLICT)
            self.assertEqual(second.list("alpha"), [current])

    def test_simultaneous_writers_have_exactly_one_success(self):
        fact = self.repository.add("alpha", "original")
        barrier = threading.Barrier(2)

        def attempt(content):
            with SQLiteMemoryRepository(self.path) as repository:
                barrier.wait(timeout=5)
                return repository.update("alpha", fact.id, content, expected_version=1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(attempt, content) for content in ("first", "second")]
            results = [future.result(timeout=10) for future in futures]
        self.assertCountEqual([result.status for result in results], [MutationStatus.SUCCESS, MutationStatus.VERSION_CONFLICT])
        winner = next(result.fact for result in results if result.status == MutationStatus.SUCCESS)
        self.assertEqual(self.repository.list("alpha"), [winner])
        self.assertEqual(winner.version, 2)

    def test_sql_metacharacters_are_literal_values_in_all_operations(self):
        sender = "alpha' OR 1=1 --"
        content = "'); DROP TABLE memory_facts; --"
        first = self.repository.add(sender, content)
        other = self.repository.add("beta", "unrelated")
        self.assertEqual(self.repository.list(sender), [first])
        self.assertEqual(self.repository.list("' OR 1=1 --"), [])
        self.assertEqual(self.repository.update(sender, "' OR 1=1 --", "bad", expected_version=1).status, MutationStatus.NOT_FOUND)
        self.assertEqual(self.repository.delete("' OR 1=1 --", first.id, expected_version=1).status, MutationStatus.NOT_FOUND)
        changed = self.repository.update(sender, first.id, content + " 'UPDATE'", expected_version=1)
        self.assertEqual(changed.fact.content, content + " 'UPDATE'")
        self.assertEqual(self.repository.delete(sender, first.id, expected_version=2).status, MutationStatus.SUCCESS)
        self.assertEqual(self.repository.list("beta"), [other])

    def test_transaction_commits_batch_and_hides_uncommitted_rows(self):
        with SQLiteMemoryRepository(self.path) as observer:
            with self.repository.transaction() as transaction:
                first = transaction.add("alpha", "one")
                second = transaction.add("beta", "two")
                self.assertEqual(transaction.list("alpha"), [first])
                self.assertEqual(observer.list("alpha"), [])
                self.assertEqual(observer.list("beta"), [])
            self.assertEqual(observer.list("alpha"), [first])
            self.assertEqual(observer.list("beta"), [second])

    def test_exception_rolls_back_add_update_and_delete_across_senders(self):
        original = self.repository.add("alpha", "original")
        other = self.repository.add("beta", "keep")
        with self.assertRaisesRegex(RuntimeError, "abort batch"):
            with self.repository.transaction():
                self.repository.update("alpha", original.id, "changed", expected_version=1)
                self.repository.delete("beta", other.id, expected_version=1)
                self.repository.add("alpha", "temporary")
                raise RuntimeError("abort batch")
        self.assertEqual(self.repository.list("alpha"), [original])
        self.assertEqual(self.repository.list("beta"), [other])
        with SQLiteMemoryRepository(self.path) as reopened:
            self.assertEqual(reopened.list("alpha"), [original])
            self.assertEqual(reopened.list("beta"), [other])
        # The connection remains usable after rollback.
        self.assertEqual(self.repository.update("alpha", original.id, "recovered", expected_version=1).status, MutationStatus.SUCCESS)

    def test_real_sqlite_error_rolls_back_entire_batch(self):
        original = self.repository.add("alpha", "original")
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("""
                CREATE TRIGGER reject_test_content BEFORE INSERT ON memory_facts
                WHEN NEW.content = 'reject this test insert'
                BEGIN SELECT RAISE(ABORT, 'test storage failure'); END
            """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "test storage failure"):
            with self.repository.transaction():
                self.repository.update("alpha", original.id, "changed", expected_version=1)
                self.repository.add("beta", "temporary")
                self.repository.add("alpha", "reject this test insert")
        self.assertEqual(self.repository.list("alpha"), [original])
        self.assertEqual(self.repository.list("beta"), [])

    def _install_automatic_rollback_trigger(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("""
                CREATE TRIGGER force_test_rollback BEFORE INSERT ON memory_facts
                WHEN NEW.content = 'force automatic rollback'
                BEGIN SELECT RAISE(ROLLBACK, 'test automatic rollback'); END
            """)

    def test_sqlite_automatic_rollback_preserves_original_error(self):
        self._install_automatic_rollback_trigger()
        with self.assertRaisesRegex(sqlite3.IntegrityError, "test automatic rollback"):
            with self.repository.transaction():
                self.repository.add("alpha", "must roll back")
                self.repository.add("beta", "force automatic rollback")
        self.assertEqual(self.repository.list("alpha"), [])
        self.assertEqual(self.repository.list("beta"), [])

    def test_caught_sqlite_automatic_rollback_cannot_silently_commit_later_writes(self):
        self._install_automatic_rollback_trigger()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            with self.repository.transaction():
                self.repository.add("alpha", "must roll back")
                with self.assertRaisesRegex(sqlite3.IntegrityError, "test automatic rollback"):
                    self.repository.add("beta", "force automatic rollback")
                with self.assertRaisesRegex(RuntimeError, "aborted"):
                    self.repository.add("alpha", "must not start a new transaction")
        self.assertEqual(self.repository.list("alpha"), [])
        self.assertEqual(self.repository.list("beta"), [])
        self.assertEqual(self.repository.add("alpha", "after scope exits").version, 1)

    def test_nested_savepoint_rollback_can_be_caught_by_outer_scope(self):
        with self.repository.transaction():
            first = self.repository.add("alpha", "outer first")
            with self.assertRaisesRegex(ValueError, "inner failure"):
                with self.repository.transaction():
                    self.repository.add("alpha", "inner removed")
                    self.repository.update("alpha", first.id, "inner change", expected_version=1)
                    raise ValueError("inner failure")
            self.assertEqual(self.repository.list("alpha"), [first])
            second = self.repository.add("alpha", "outer second")
        self.assertEqual(self.repository.list("alpha"), [first, second])

    def test_nested_success_is_still_rolled_back_by_outer_failure(self):
        with self.assertRaisesRegex(RuntimeError, "outer failure"):
            with self.repository.transaction():
                self.repository.add("alpha", "outer")
                with self.repository.transaction():
                    with self.repository.transaction():
                        self.repository.add("beta", "deep nested")
                raise RuntimeError("outer failure")
        self.assertEqual(self.repository.list("alpha"), [])
        self.assertEqual(self.repository.list("beta"), [])

    def test_failed_result_is_explicit_and_caller_can_abort_batch(self):
        fact = self.repository.add("alpha", "original")
        with self.assertRaisesRegex(RuntimeError, "version conflict"):
            with self.repository.transaction():
                self.repository.add("beta", "must roll back")
                result = self.repository.update("alpha", fact.id, "stale", expected_version=2)
                self.assertEqual(result.status, MutationStatus.VERSION_CONFLICT)
                raise RuntimeError("version conflict")
        self.assertEqual(self.repository.list("alpha"), [fact])
        self.assertEqual(self.repository.list("beta"), [])

    def test_invalid_strings_are_rejected_without_writes(self):
        fact = self.repository.add("alpha", "original")
        for value in (None, 1, "", " \t\n"):
            with self.subTest(value=value):
                calls = (
                    lambda: self.repository.list(value),
                    lambda: self.repository.add(value, "content"),
                    lambda: self.repository.add("alpha", value),
                    lambda: self.repository.update(value, fact.id, "new", expected_version=1),
                    lambda: self.repository.update("alpha", value, "new", expected_version=1),
                    lambda: self.repository.update("alpha", fact.id, value, expected_version=1),
                    lambda: self.repository.delete(value, fact.id, expected_version=1),
                    lambda: self.repository.delete("alpha", value, expected_version=1),
                )
                for invoke in calls:
                    with self.assertRaises(ValueError):
                        invoke()
        self.assertEqual(self.repository.list("alpha"), [fact])

    def test_expected_version_is_required_and_must_be_a_positive_integer(self):
        fact = self.repository.add("alpha", "original")
        with self.assertRaises(TypeError):
            self.repository.update("alpha", fact.id, "new")
        with self.assertRaises(TypeError):
            self.repository.delete("alpha", fact.id)
        for version in (None, 0, -1, True, False, 1.0, "1"):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    self.repository.update("alpha", fact.id, "new", expected_version=version)
                with self.assertRaises(ValueError):
                    self.repository.delete("alpha", fact.id, expected_version=version)
        self.assertEqual(self.repository.list("alpha"), [fact])

    def test_close_is_idempotent_and_closed_repository_rejects_operations(self):
        self.repository.close()
        self.repository.close()
        for invoke in (
            lambda: self.repository.list("alpha"),
            lambda: self.repository.add("alpha", "new"),
            lambda: self.repository.update("alpha", "id", "new", expected_version=1),
            lambda: self.repository.delete("alpha", "id", expected_version=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                invoke()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with self.repository.transaction():
                self.fail("Closed transaction must not start")

    def test_close_inside_transaction_is_rejected_and_rolls_back(self):
        with self.assertRaisesRegex(RuntimeError, "inside a transaction"):
            with self.repository.transaction():
                self.repository.add("alpha", "temporary")
                self.repository.close()
        self.assertEqual(self.repository.list("alpha"), [])


if __name__ == "__main__":
    unittest.main()
