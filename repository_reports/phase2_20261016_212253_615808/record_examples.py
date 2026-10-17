"""Record read-only real preflight and disposable synthetic migration evidence."""

import hashlib
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import migrate_json_to_sqlite as migration
from sqlite_memory_repository import SQLiteMemoryRepository


def save(name, value):
    (REPORT_DIR / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


dry_target = REPORT_DIR / "real_dry_run_NOT_CREATED.sqlite3"
real = migration.migrate_json_to_sqlite(ROOT / "memories.json", dry_target, dry_run=True)
assert real["status"] == "DRY_RUN_PASSED" and not dry_target.exists()
save("real_json_dry_run.json", real)

with tempfile.TemporaryDirectory(prefix="migration-report-example-") as directory:
    temporary = Path(directory)
    source = temporary / "memories.json"
    data = {"zhangsan": [], "lisi": ["喜欢摄影", "喜欢摄影", "第二条含引号\"和换行\n"], "王老师": ["联系人住在厦门", "联系人喜欢茶"]}
    source.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "synthetic_source.json").write_bytes(source.read_bytes())
    successful = migration.migrate_json_to_sqlite(source, temporary / "success.sqlite3")
    assert successful["status"] == "SUCCESS"
    save("synthetic_success.json", successful)

    original_add, cleanup = SQLiteMemoryRepository.add, migration._cleanup_staging
    observations = {"insert_attempts": 0}

    def fail_third(repository, sender, content):
        observations["insert_attempts"] += 1
        if observations["insert_attempts"] == 3:
            observations["facts_inside_transaction_before_error"] = sum(len(repository.list(item)) for item in repository.list_senders())
            staging = next(temporary.glob(".failed.sqlite3.migration-*.sqlite3"))
            with closing(sqlite3.connect(staging)) as observer:
                observations["facts_visible_to_other_connection_before_error"] = observer.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
            raise RuntimeError("Injected third-insert failure for atomicity evidence")
        return original_add(repository, sender, content)

    def observe_cleanup(staging):
        with closing(sqlite3.connect(staging)) as observer:
            observations["facts_after_rollback"] = observer.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
            observations["senders_after_rollback"] = observer.execute("SELECT COUNT(*) FROM memory_senders").fetchone()[0]
        cleanup(staging)

    with patch.object(SQLiteMemoryRepository, "add", new=fail_third), patch.object(migration, "_cleanup_staging", side_effect=observe_cleanup):
        failed = migration.migrate_json_to_sqlite(source, temporary / "failed.sqlite3")
    failed["atomicity_observations"] = observations
    assert failed["status"] == "FAILED" and failed["transaction"]["rollback_verified"]
    assert observations["facts_inside_transaction_before_error"] == 2
    assert observations["facts_visible_to_other_connection_before_error"] == observations["facts_after_rollback"] == observations["senders_after_rollback"] == 0
    assert not (temporary / "failed.sqlite3").exists()
    save("synthetic_failure_atomicity.json", failed)

save("example_cleanup.json", {
    "temporary_directory_removed": not temporary.exists(),
    "real_dry_run_created_database": dry_target.exists(),
    "real_source_sha256_after": hashlib.sha256((ROOT / "memories.json").read_bytes()).hexdigest(),
    "loaded_agent_or_model_modules": [name for name in ("main", "memory_v2", "memory_search", "ollama_client", "sentence_transformers") if name in sys.modules],
})
print(json.dumps({"real": real["status"], "synthetic": successful["status"], "injected_failure": failed["status"], "atomicity": observations}, ensure_ascii=False))
