"""Validate a JSON snapshot and publish a new, verified SQLite database.

No Agent/model imports, no implicit input/output paths, no overwrite option.
Pause legacy writers and drain in-flight writes before a real migration: hash
checks detect observed changes, but cannot coordinate a live cutover.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from sqlite_memory_repository import SQLiteMemoryRepository


SIDECARS = ("-journal", "-wal", "-shm")


class SourceChangedError(RuntimeError):
    pass


class MigrationValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MigrationValidationError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise MigrationValidationError(f"Non-standard JSON constant: {value}")


def _read_snapshot(source: Path, report: dict) -> dict[str, list[str]]:
    raw = source.read_bytes()
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise MigrationValidationError("Memory JSON must be an object mapping sender IDs to lists.")
    for sender, contents in data.items():
        if not isinstance(sender, str) or not sender.strip():
            raise MigrationValidationError("Every sender ID must be a non-empty string.")
        if not isinstance(contents, list):
            raise MigrationValidationError(f"Memory for sender {sender!r} must be a list.")
        for index, content in enumerate(contents):
            if not isinstance(content, str) or not content.strip():
                raise MigrationValidationError(f"Fact {index} for sender {sender!r} must be a non-empty string.")
            # The existing SQLite CHECK uses length(), which stops at NUL.
            # Reject its unrepresentable leading-NUL case during preflight.
            if not content.lstrip(" ").split("\0", 1)[0]:
                raise MigrationValidationError(f"Fact {index} for sender {sender!r} violates SQLite's content constraint.")
        if not sender.lstrip(" ").split("\0", 1)[0]:
            raise MigrationValidationError("Sender ID violates SQLite's non-empty constraint.")
    report.update({
        "sender_count": len(data), "fact_count": sum(map(len, data.values())),
        "per_sender_counts": {sender: len(contents) for sender, contents in data.items()},
        "empty_sender_count": sum(not contents for contents in data.values()),
        "duplicate_fact_occurrences": sum(len(contents) - len(set(contents)) for contents in data.values()),
    })
    return data


def _check_source(source: Path, report: dict) -> None:
    try:
        current = _sha256(source)
    except OSError as error:
        report["source_sha256_after"] = None
        report["source_unchanged"] = False
        raise SourceChangedError("Source JSON could not be re-read; publication refused.") from error
    report["source_sha256_after"] = current
    report["source_unchanged"] = current == report["source_sha256"]
    if not report["source_unchanged"]:
        raise SourceChangedError("Source JSON changed during migration; publication refused.")


def _check_destination(target: Path) -> None:
    # lexists also rejects a dangling symlink; exists alone would miss it.
    for path in (target, *(Path(str(target) + suffix) for suffix in SIDECARS)):
        if os.path.lexists(path):
            raise FileExistsError(f"Destination or SQLite sidecar already exists: {path}")


def _verify_snapshot(repository, expected, expected_ids=None):
    actual_senders = repository.list_senders()
    details, ids = {}, {}
    for sender, contents in expected.items():
        facts = repository.list(sender)
        actual_contents = [fact.content for fact in facts]
        ids[sender] = [fact.id for fact in facts]
        details[sender] = {
            "expected_count": len(contents), "actual_count": len(facts),
            "content_and_order_match": actual_contents == contents,
            "duplicates_match": Counter(actual_contents) == Counter(contents),
            "versions_are_one": all(fact.version == 1 for fact in facts),
            "ids_unchanged": expected_ids is None or ids[sender] == expected_ids[sender],
        }
    flat_ids = [fact_id for values in ids.values() for fact_id in values]
    checks = {
        "senders_and_order_match": actual_senders == list(expected),
        "fact_count_match": sum(len(repository.list(sender)) for sender in actual_senders) == sum(map(len, expected.values())),
        "ids_unique": len(flat_ids) == len(set(flat_ids)), "per_sender": details,
    }
    checks["passed"] = (
        checks["senders_and_order_match"] and checks["fact_count_match"] and checks["ids_unique"]
        and all(item["expected_count"] == item["actual_count"] and all(
            item[field] for field in ("content_and_order_match", "duplicates_match", "versions_are_one", "ids_unchanged")
        ) for item in details.values())
    )
    return checks, ids


def _integrity_results(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute("PRAGMA integrity_check").fetchall()]


def _prepare_closed_database(staging: Path, report: dict) -> None:
    # Storage controls only: all fact/sender CRUD is delegated to Repository.
    with closing(sqlite3.connect(staging, isolation_level=None)) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        if mode == "wal":
            checkpoint = tuple(connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
            report["wal_checkpoint"] = checkpoint
            if checkpoint[0] != 0:
                raise MigrationValidationError("WAL checkpoint was busy; publication refused.")
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if mode != "delete":
            raise MigrationValidationError("Could not switch staging database to DELETE journal mode.")
        results = _integrity_results(connection)
        report["integrity_check"] = {"results": results, "passed": results == ["ok"]}
        if results != ["ok"]:
            raise MigrationValidationError("SQLite integrity_check failed.")
    report["connections_closed"] = True
    report["journal_mode"] = mode
    if any(os.path.lexists(str(staging) + suffix) for suffix in SIDECARS):
        raise MigrationValidationError("Staging SQLite sidecars remain after closing; publication refused.")
    with staging.open("r+b") as stream:
        os.fsync(stream.fileno())
    report["database_sha256"] = _sha256(staging)


def _publish_no_replace(staging: Path, target: Path) -> None:
    _check_destination(target)
    # Hard-link creation is atomic and fails if the target exists, including a
    # target created after the preceding check. No rename/replace fallback.
    os.link(staging, target)


def _cleanup_staging(staging: Path) -> None:
    errors = []
    for path in (*(Path(str(staging) + suffix) for suffix in SIDECARS), staging):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise OSError("; ".join(errors))


def migrate_json_to_sqlite(source_path: str | Path, target_path: str | Path, *, dry_run: bool = False) -> dict:
    """Return a structured SUCCESS / DRY_RUN_PASSED / FAILED report.

    Failure reports include the exception type/message; CLI failures exit 1.
    A successful hard link is the publication commit point. If removing its
    private staging alias subsequently fails, report a cleanup warning while
    retaining SUCCESS: the fully verified target is already published safely.
    Never attempt to unlink a published target that another process may own.
    """
    started = time.perf_counter()
    source = Path(source_path).expanduser().absolute()
    requested_target = Path(target_path).expanduser().absolute()
    staging = None
    report = {
        "status": "FAILED", "dry_run": dry_run,
        "source_path": str(source), "target_path": str(requested_target),
        "source_sha256": None, "source_sha256_after": None, "source_unchanged": None,
        "sender_count": None, "fact_count": None, "per_sender_counts": {},
        "preflight_passed": False, "verification": {},
        "integrity_check": {"results": [], "passed": None},
        "transaction": {"committed": False, "rollback_verified": None},
        "published": False, "publication_method": "atomic_hard_link_no_replace",
        "staging_cleanup_passed": None,
    }
    try:
        data = _read_snapshot(source, report)
        parent = requested_target.parent.resolve(strict=True)
        if not parent.is_dir():
            raise NotADirectoryError(str(parent))
        target = parent / requested_target.name
        report["target_path"] = str(target)
        _check_destination(target)
        report["preflight_passed"] = True
        if dry_run:
            _check_source(source, report)
            report["status"] = "DRY_RUN_PASSED"
        else:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.migration-", suffix=".sqlite3", dir=parent,
            )
            os.close(descriptor)
            staging = Path(name)
            with SQLiteMemoryRepository(staging) as repository:
                try:
                    with repository.transaction():
                        for sender, contents in data.items():
                            repository.ensure_sender(sender)
                            for content in contents:
                                repository.add(sender, content)
                        checks, ids = _verify_snapshot(repository, data)
                        report["verification"]["in_transaction"] = checks
                        if not checks["passed"]:
                            raise MigrationValidationError("Imported facts do not match the source snapshot.")
                        _check_source(source, report)
                    report["transaction"]["committed"] = True
                except Exception:
                    report["transaction"]["rollback_verified"] = (
                        repository.list_senders() == [] and all(repository.list(sender) == [] for sender in data)
                    )
                    raise
            # Verify durable state and IDs through a freshly opened Repository.
            with SQLiteMemoryRepository(staging) as reopened:
                checks, _ = _verify_snapshot(reopened, data, ids)
                report["verification"]["reopened"] = checks
                if not checks["passed"]:
                    raise MigrationValidationError("Reopened database does not match the imported snapshot.")
            _prepare_closed_database(staging, report)
            _check_source(source, report)  # Last read after commit, validation and close.
            _publish_no_replace(staging, target)
            report["published"] = True
            report["status"] = "SUCCESS"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if report["source_sha256"] is not None:
            try:
                _check_source(source, report)
            except Exception as source_error:
                report["source_check_error"] = {"type": type(source_error).__name__, "message": str(source_error)}
    finally:
        if staging is not None:
            try:
                _cleanup_staging(staging)
                report["staging_cleanup_passed"] = True
            except OSError as error:
                report["staging_cleanup_passed"] = False
                report["cleanup_error"] = str(error)
                # A published target is already a complete migration; cleanup
                # cannot safely undo that atomic commit point.
        report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main(argv=None) -> int:
    # Reports are UTF-8 even when redirected on Windows or when the caller's
    # default stdio encoding cannot represent a sender ID.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source UTF-8 memory JSON (read-only)")
    parser.add_argument("--output", required=True, type=Path, help="New SQLite path; existing destinations are refused")
    parser.add_argument("--dry-run", action="store_true", help="Validate without creating a database")
    args = parser.parse_args(argv)
    report = migrate_json_to_sqlite(args.input, args.output, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"SUCCESS", "DRY_RUN_PASSED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
