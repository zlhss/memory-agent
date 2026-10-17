"""Safe migration and sender-registry tests; all inputs/DBs are temporary."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import migrate_json_to_sqlite as migration
from sqlite_memory_repository import SQLiteMemoryRepository


class MigrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="json-sqlite-migration-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "memories.json"
        self.target = self.directory / "facts.sqlite3"

    def write_source(self, data):
        self.source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def leftovers(self):
        return list(self.directory.glob(f".{self.target.name}.migration-*"))

    def assert_no_publication(self, report):
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse(report["published"])
        self.assertFalse(self.target.exists())
        self.assertEqual(self.leftovers(), [])

    def assert_success(self, data, report):
        self.assertEqual(report["status"], "SUCCESS", report)
        self.assertTrue(report["published"])
        self.assertTrue(report["transaction"]["committed"])
        self.assertTrue(report["source_unchanged"])
        self.assertEqual(report["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report["source_sha256_after"], report["source_sha256"])
        self.assertEqual(report["sender_count"], len(data))
        self.assertEqual(report["fact_count"], sum(map(len, data.values())))
        self.assertEqual(report["per_sender_counts"], {sender: len(values) for sender, values in data.items()})
        self.assertEqual(report["integrity_check"], {"results": ["ok"], "passed": True})
        self.assertTrue(report["verification"]["in_transaction"]["passed"])
        self.assertTrue(report["verification"]["reopened"]["passed"])
        self.assertTrue(report["connections_closed"])
        self.assertTrue(report["staging_cleanup_passed"])
        self.assertEqual(self.leftovers(), [])
        self.assertFalse(any(Path(str(self.target) + suffix).exists() for suffix in migration.SIDECARS))
        with SQLiteMemoryRepository(self.target) as repository:
            self.assertEqual(repository.list_senders(), list(data))
            ids = []
            for sender, contents in data.items():
                facts = repository.list(sender)
                self.assertEqual([fact.content for fact in facts], contents)
                self.assertTrue(all(fact.version == 1 and UUID(fact.id).version == 4 for fact in facts))
                ids.extend(fact.id for fact in facts)
        self.assertEqual(len(ids), len(set(ids)))

    def test_empty_json_publishes_valid_empty_database(self):
        self.write_source({})
        self.assert_success({}, migration.migrate_json_to_sqlite(self.source, self.target))

    def test_empty_contacts_survive_without_placeholder_facts(self):
        data = {"zhangsan": [], "empty second": []}
        self.write_source(data)
        report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_success(data, report)
        self.assertEqual(report["empty_sender_count"], 2)

    def test_multiple_contacts_order_and_duplicates_are_preserved(self):
        data = {"empty": [], "lisi": ["喜欢摄影", "喜欢摄影", "第三条", "第一条"], "张三": ["同文", "另一条", "同文"]}
        self.write_source(data)
        before = self.source.read_bytes()
        report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_success(data, report)
        self.assertEqual(report["duplicate_fact_occurrences"], 2)
        self.assertEqual(self.source.read_bytes(), before)

    def test_unicode_quotes_newlines_whitespace_sql_and_embedded_nul_round_trip(self):
        data = {"联系人'😀": ["  喜欢摄影\n和音乐\t  ", "'); DROP TABLE memory_facts; --", '双引号"和\\路径', "记忆\0尾部字符"]}
        self.write_source(data)
        self.assert_success(data, migration.migrate_json_to_sqlite(self.source, self.target))

    def test_dry_run_does_not_open_or_create_database(self):
        data = {"empty": [], "lisi": ["重复", "重复"]}
        self.write_source(data)
        before = self.source.read_bytes()
        with patch.object(migration, "SQLiteMemoryRepository", side_effect=AssertionError("No database in dry-run")):
            report = migration.migrate_json_to_sqlite(self.source, self.target, dry_run=True)
        self.assertEqual(report["status"], "DRY_RUN_PASSED")
        self.assertEqual(report["sender_count"], 2)
        self.assertEqual(report["fact_count"], 2)
        self.assertTrue(report["source_unchanged"])
        self.assertIsNone(report["integrity_check"]["passed"])
        self.assertFalse(report["transaction"]["committed"])
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.leftovers(), [])

    def test_malformed_json_is_explicit_failure_and_source_is_unchanged(self):
        self.source.write_bytes(b'{"sender": ["unfinished"')
        before = self.source.read_bytes()
        report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "JSONDecodeError")
        self.assertEqual(report["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.source.read_bytes(), before)

    def test_invalid_structures_are_rejected_without_silent_skips(self):
        invalid = [[], None, 1, "text", {"sender": {}}, {"sender": "fact"}, {"": []}, {" \t": []},
                   {"sender": [1]}, {"sender": [None]}, {"sender": [{}]}, {"sender": [""]},
                   {"sender": [" \t\n"]}, {"sender": [float("nan")]}, {"sender": ["\0unrepresentable"]}, {"\0sender": []}]
        for data in invalid:
            with self.subTest(data=data):
                self.write_source(data)
                before = self.source.read_bytes()
                report = migration.migrate_json_to_sqlite(self.source, self.target)
                self.assert_no_publication(report)
                self.assertEqual(report["error"]["type"], "MigrationValidationError")
                self.assertEqual(self.source.read_bytes(), before)

    def test_duplicate_sender_keys_are_rejected_instead_of_last_value_winning(self):
        self.source.write_text('{"lisi": ["first"], "lisi": ["second"]}', encoding="utf-8")
        report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertIn("Duplicate JSON object key", report["error"]["message"])

    def test_invalid_utf8_is_rejected_with_raw_byte_hash(self):
        self.source.write_bytes(b'{"a":["\xff"]}')
        report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "UnicodeDecodeError")
        self.assertIsNotNone(report["source_sha256"])

    def test_missing_input_or_output_directory_is_explicit_failure(self):
        self.assert_no_publication(migration.migrate_json_to_sqlite(self.source, self.target))
        self.write_source({})
        report = migration.migrate_json_to_sqlite(self.source, self.directory / "missing" / "target.sqlite3")
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse((self.directory / "missing").exists())

    def test_existing_target_is_never_overwritten(self):
        self.write_source({"a": ["new"]})
        self.target.write_bytes(b"existing valuable file")
        before = self.target.read_bytes()
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                report = migration.migrate_json_to_sqlite(self.source, self.target, dry_run=dry_run)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(report["error"]["type"], "FileExistsError")
                self.assertFalse(report["published"])
                self.assertEqual(self.target.read_bytes(), before)
                self.assertEqual(self.leftovers(), [])

    def test_same_input_and_output_path_is_refused(self):
        self.write_source({"a": ["preserve"]})
        before = self.source.read_bytes()
        report = migration.migrate_json_to_sqlite(self.source, self.source)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(self.source.read_bytes(), before)

    def test_existing_target_sidecars_are_not_touched(self):
        self.write_source({})
        sidecar = Path(str(self.target) + "-wal")
        sidecar.write_bytes(b"existing wal")
        report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(sidecar.read_bytes(), b"existing wal")

    def test_repeat_migration_refuses_target_and_preserves_ids_and_bytes(self):
        data = {"empty": [], "a": ["one", "one"]}
        self.write_source(data)
        first = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_success(data, first)
        before = self.target.read_bytes()
        second = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assertEqual(second["status"], "FAILED")
        self.assertEqual(second["error"]["type"], "FileExistsError")
        self.assertEqual(self.target.read_bytes(), before)

    def test_independent_migrations_may_assign_different_ids(self):
        data = {"a": ["one", "one"]}
        self.write_source(data)
        self.assert_success(data, migration.migrate_json_to_sqlite(self.source, self.target))
        other_path = self.directory / "second.sqlite3"
        report = migration.migrate_json_to_sqlite(self.source, other_path)
        self.assertEqual(report["status"], "SUCCESS")
        with SQLiteMemoryRepository(self.target) as first, SQLiteMemoryRepository(other_path) as second:
            self.assertEqual([fact.content for fact in first.list("a")], [fact.content for fact in second.list("a")])
            self.assertTrue(set(fact.id for fact in first.list("a")).isdisjoint(fact.id for fact in second.list("a")))

    def test_nth_insert_failure_rolls_back_all_preceding_facts_and_senders(self):
        self.write_source({"empty": [], "a": ["one", "two", "three"], "b": ["four"]})
        source_before = self.source.read_bytes()
        original_add, cleanup = SQLiteMemoryRepository.add, migration._cleanup_staging
        observations = {}

        def fail_third(repository, sender, content):
            observations["inserts"] = observations.get("inserts", 0) + 1
            if observations["inserts"] == 3:
                observations["inside_count"] = len(repository.list("a"))
                staging = next(path for path in self.leftovers() if path.suffix == ".sqlite3")
                with closing(sqlite3.connect(staging)) as observer:
                    observations["outside_count_before_failure"] = observer.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
                raise RuntimeError("injected third insert failure")
            return original_add(repository, sender, content)

        def inspect_rollback(staging):
            with closing(sqlite3.connect(staging)) as observer:
                observations["facts_after_rollback"] = observer.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
                observations["senders_after_rollback"] = observer.execute("SELECT COUNT(*) FROM memory_senders").fetchone()[0]
            cleanup(staging)

        with patch.object(SQLiteMemoryRepository, "add", new=fail_third), patch.object(migration, "_cleanup_staging", side_effect=inspect_rollback):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(observations, {"inserts": 3, "inside_count": 2, "outside_count_before_failure": 0, "facts_after_rollback": 0, "senders_after_rollback": 0})
        self.assertTrue(report["transaction"]["rollback_verified"])
        self.assertFalse(report["transaction"]["committed"])
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_source_change_during_import_rolls_back_and_refuses_publication(self):
        self.write_source({"a": ["one", "two"]})
        original_add = SQLiteMemoryRepository.add

        def change_source(repository, sender, content):
            fact = original_add(repository, sender, content)
            self.write_source({"a": ["external writer changed source"]})
            return fact

        with patch.object(SQLiteMemoryRepository, "add", new=change_source):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "SourceChangedError")
        self.assertFalse(report["source_unchanged"])
        self.assertTrue(report["transaction"]["rollback_verified"])

    def test_source_change_after_commit_is_caught_before_publication(self):
        self.write_source({"a": ["one"]})
        prepare = migration._prepare_closed_database

        def change_after_close(staging, report):
            prepare(staging, report)
            self.write_source({"a": ["changed after commit"]})

        with patch.object(migration, "_prepare_closed_database", side_effect=change_after_close):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "SourceChangedError")
        self.assertTrue(report["transaction"]["committed"])
        self.assertFalse(report["source_unchanged"])

    def test_in_transaction_content_mismatch_rolls_back(self):
        self.write_source({"a": ["one"]})
        original_add = SQLiteMemoryRepository.add

        def corrupt_content(repository, sender, content):
            return original_add(repository, sender, content + " corrupted")

        with patch.object(SQLiteMemoryRepository, "add", new=corrupt_content):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "MigrationValidationError")
        self.assertFalse(report["verification"]["in_transaction"]["passed"])
        self.assertTrue(report["transaction"]["rollback_verified"])

    def test_reopened_content_mismatch_refuses_publication(self):
        self.write_source({"a": ["one"]})
        original_verify = migration._verify_snapshot
        calls = []

        def corrupt_on_reopen(repository, expected, expected_ids=None):
            calls.append(1)
            if len(calls) == 2:
                fact = repository.list("a")[0]
                repository.update("a", fact.id, "external staging corruption", expected_version=1)
            return original_verify(repository, expected, expected_ids)

        with patch.object(migration, "_verify_snapshot", side_effect=corrupt_on_reopen):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertFalse(report["verification"]["reopened"]["passed"])

    def test_integrity_check_failure_refuses_publication(self):
        self.write_source({"a": ["one"]})
        with patch.object(migration, "_integrity_results", return_value=["simulated corrupt index"]):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["integrity_check"], {"results": ["simulated corrupt index"], "passed": False})
        self.assertEqual(report["error"]["type"], "MigrationValidationError")

    def test_wal_database_is_checkpointed_and_published_without_sidecars(self):
        self.write_source({"a": ["one"]})
        prepare = migration._prepare_closed_database

        def enable_wal_then_prepare(staging, report):
            with closing(sqlite3.connect(staging, isolation_level=None)) as connection:
                self.assertEqual(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
                connection.execute("PRAGMA user_version=1")
            prepare(staging, report)

        with patch.object(migration, "_prepare_closed_database", side_effect=enable_wal_then_prepare):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_success({"a": ["one"]}, report)
        self.assertEqual(report["journal_mode"], "delete")
        self.assertEqual(report["wal_checkpoint"][0], 0)

    def test_remaining_staging_sidecar_refuses_publication(self):
        self.write_source({"a": ["one"]})
        check = migration._integrity_results

        def leave_sidecar(connection):
            staging = self.leftovers()[0]
            Path(str(staging) + "-shm").write_bytes(b"unexpected sidecar")
            return check(connection)

        with patch.object(migration, "_integrity_results", side_effect=leave_sidecar):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertIn("sidecars remain", report["error"]["message"])

    def test_source_disappears_after_commit_refuses_publication(self):
        self.write_source({"a": ["one"]})
        prepare = migration._prepare_closed_database

        def remove_source(staging, report):
            prepare(staging, report)
            self.source.unlink()

        with patch.object(migration, "_prepare_closed_database", side_effect=remove_source):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "SourceChangedError")
        self.assertFalse(report["source_unchanged"])
        self.assertIsNone(report["source_sha256_after"])

    def test_unsupported_hard_links_fail_without_replace_fallback(self):
        self.write_source({"a": ["one"]})
        with patch.object(migration.os, "link", side_effect=OSError("hard links unavailable")), patch.object(migration.os, "replace", side_effect=AssertionError("No replace fallback")):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_no_publication(report)
        self.assertEqual(report["error"]["type"], "OSError")

    def test_destination_created_at_link_time_is_never_overwritten(self):
        self.write_source({"a": ["one"]})
        original_link = os.link

        def race_at_link(staging, target):
            target.write_bytes(b"concurrent owner's data")
            original_link(staging, target)

        with patch.object(migration.os, "link", side_effect=race_at_link):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error"]["type"], "FileExistsError")
        self.assertFalse(report["published"])
        self.assertEqual(self.target.read_bytes(), b"concurrent owner's data")
        self.assertEqual(self.leftovers(), [])

    def test_two_simultaneous_migrations_publish_exactly_one_database(self):
        self.write_source({"empty": [], "a": ["one", "one"]})
        original_link = os.link
        barrier = threading.Barrier(2)

        def competing_links(staging, target):
            barrier.wait(timeout=10)
            original_link(staging, target)

        with patch.object(migration.os, "link", side_effect=competing_links):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(migration.migrate_json_to_sqlite, self.source, self.target) for _ in range(2)]
                reports = [future.result(timeout=20) for future in futures]
        self.assertCountEqual([report["status"] for report in reports], ["SUCCESS", "FAILED"])
        self.assertEqual(sum(report["published"] for report in reports), 1)
        self.assertEqual(self.leftovers(), [])
        with SQLiteMemoryRepository(self.target) as repository:
            self.assertEqual(repository.list_senders(), ["empty", "a"])
            self.assertEqual([fact.content for fact in repository.list("a")], ["one", "one"])

    def test_publication_occurs_after_close_without_sidecars_on_same_filesystem(self):
        self.write_source({"a": ["one"]})
        publish = migration._publish_no_replace
        observed = []

        def inspect_publish(staging, target):
            self.assertEqual(staging.parent, target.parent)
            self.assertFalse(any(Path(str(staging) + suffix).exists() for suffix in migration.SIDECARS))
            with closing(sqlite3.connect(staging, timeout=0)) as connection:
                connection.execute("BEGIN EXCLUSIVE")
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                connection.rollback()
            observed.append(True)
            publish(staging, target)

        with patch.object(migration, "_publish_no_replace", side_effect=inspect_publish):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assert_success({"a": ["one"]}, report)
        self.assertEqual(observed, [True])

    def test_post_publication_cleanup_warning_does_not_delete_published_target(self):
        self.write_source({"a": ["one"]})
        cleanup = migration._cleanup_staging
        with patch.object(migration, "_cleanup_staging", side_effect=PermissionError("alias cleanup denied")):
            report = migration.migrate_json_to_sqlite(self.source, self.target)
        self.assertEqual(report["status"], "SUCCESS")
        self.assertTrue(report["published"])
        self.assertFalse(report["staging_cleanup_passed"])
        self.assertIn("alias cleanup denied", report["cleanup_error"])
        with SQLiteMemoryRepository(self.target) as repository:
            self.assertEqual([fact.content for fact in repository.list("a")], ["one"])
        for staging in self.leftovers():
            cleanup(staging)

    def test_cli_json_reports_and_exit_codes(self):
        self.write_source({"empty": [], "a": ["one", "one"]})
        command = [sys.executable, "-B", str(Path(migration.__file__)), "--input", str(self.source), "--output", str(self.target)]
        for options, status, code in ((["--dry-run"], "DRY_RUN_PASSED", 0), ([], "SUCCESS", 0), ([], "FAILED", 1)):
            completed = subprocess.run(command + options, capture_output=True, text=True, encoding="utf-8", check=False)
            self.assertEqual(completed.returncode, code, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], status)
            self.assertEqual(report["per_sender_counts"], {"empty": 0, "a": 2})

    def test_import_has_no_model_or_agent_side_effects(self):
        root = str(Path(migration.__file__).parent)
        code = f"import sys; sys.path.insert(0, {root!r}); import migrate_json_to_sqlite; assert all(n not in sys.modules for n in ('main','memory_v2','memory_search','ollama_client','sentence_transformers'))"
        completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=self.directory, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_emits_utf8_report_even_with_ascii_stdio_default(self):
        self.write_source({"😀联系人": ["喜欢摄影"]})
        command = [sys.executable, "-B", str(Path(migration.__file__)), "--input", str(self.source), "--output", str(self.target)]
        completed = subprocess.run(command, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "ascii"}, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("ascii", errors="replace"))
        report = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["per_sender_counts"], {"😀联系人": 1})


class SenderRegistryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="memory-sender-registry-test-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "facts.sqlite3"

    def test_empty_senders_are_persistent_ordered_and_idempotent(self):
        with SQLiteMemoryRepository(self.path) as repository:
            repository.ensure_sender("empty")
            repository.ensure_sender("second")
            repository.ensure_sender("empty")
            self.assertEqual(repository.list_senders(), ["empty", "second"])
            self.assertEqual(repository.list("empty"), [])
        with SQLiteMemoryRepository(self.path) as reopened:
            self.assertEqual(reopened.list_senders(), ["empty", "second"])

    def test_fact_add_registers_sender_and_last_delete_preserves_empty_namespace(self):
        with SQLiteMemoryRepository(self.path) as repository:
            fact = repository.add("a", "one")
            repository.delete("a", fact.id, expected_version=1)
            self.assertEqual(repository.list_senders(), ["a"])
            self.assertEqual(repository.list("a"), [])

    def test_sender_registration_and_fact_add_roll_back_together(self):
        with SQLiteMemoryRepository(self.path) as repository:
            with self.assertRaises(RuntimeError):
                with repository.transaction():
                    repository.ensure_sender("empty")
                    repository.add("a", "one")
                    raise RuntimeError("abort")
            self.assertEqual(repository.list_senders(), [])
            self.assertEqual(repository.list("a"), [])

    def test_legacy_fact_only_database_backfills_registry_without_changing_facts(self):
        with SQLiteMemoryRepository(self.path) as repository:
            first = repository.add("b", "one")
            second = repository.add("a", "two")
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TABLE memory_senders")
        with SQLiteMemoryRepository(self.path) as reopened:
            self.assertEqual(reopened.list_senders(), ["b", "a"])
            self.assertEqual(reopened.list("b"), [first])
            self.assertEqual(reopened.list("a"), [second])

    def test_invalid_sender_and_closed_registry_operations_fail(self):
        with SQLiteMemoryRepository(self.path) as repository:
            for sender in (None, "", " \t", 1):
                with self.assertRaises(ValueError):
                    repository.ensure_sender(sender)
            self.assertEqual(repository.list_senders(), [])
        with self.assertRaises(RuntimeError):
            repository.list_senders()
        with self.assertRaises(RuntimeError):
            repository.ensure_sender("a")


if __name__ == "__main__":
    unittest.main()
