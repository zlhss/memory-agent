"""Run SQLite adapter tests and the unchanged 40 baseline tests in isolation."""

import argparse
import functools
import hashlib
import io
import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_MODULES = [
    "test_contextual_memory_flow", "test_ollama_timing",
    "test_response_pipeline_mock", "test_memory_v2_interface_mock",
    "test_agent_memory_versions",
]


def run(*, include_migration=False, include_retriever=False):
    include_migration = include_migration or include_retriever
    sys.dont_write_bytecode = True
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    started_at = datetime.now(timezone(timedelta(hours=8)))
    phase = "phase3" if include_retriever else ("phase2" if include_migration else "phase1")
    report_dir = ROOT / "repository_reports" / (phase + started_at.strftime("_%Y%m%d_%H%M%S_%f"))
    report_dir.mkdir(parents=True)
    originals = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and ".git" not in path.parts and report_dir not in path.parents
    }
    protected_data = {ROOT / name for name in ("memories.json", "feedback.json", "style_samples.json")}
    temp_root = Path(tempfile.gettempdir()).resolve()
    report = {
        "started_at": started_at.isoformat(), "python": sys.executable,
        "python_version": platform.python_version(), "sqlite_version": sqlite3.sqlite_version,
        "blocked_io": [], "database_paths": [], "source_sha256_before": originals,
        "real_llm_calls": 0,
    }
    guard_state = {"active": True}

    def guard(event, args):
        if not guard_state["active"]:
            return
        if event == "sqlite3.connect":
            database = os.fsdecode(args[0])
            report["database_paths"].append(database)
            if database != ":memory:" and temp_root not in Path(database).resolve().parents:
                report["blocked_io"].append(database)
                raise PermissionError("SQLite tests may only open temporary databases.")
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if path in protected_data or (writing and ROOT in path.parents and report_dir not in path.parents):
            report["blocked_io"].append(str(path))
            raise PermissionError(f"Isolated tests refused production/source access: {path}")

    def suite(name, modules):
        started = time.perf_counter()
        output = io.StringIO()
        with redirect_stdout(output), (report_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromNames(modules),
            )
        (report_dir / f"{name}_console.log").write_text(output.getvalue(), encoding="utf-8")
        report[name] = {
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "PASS" if result.wasSuccessful() else "FAIL", "modules": modules,
        }
        print(f"{name}: {report[name]}", flush=True)

    sys.addaudithook(guard)
    started = time.perf_counter()
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="memory-repository-baseline-") as directory:
            data_dir = Path(directory)
            os.chdir(data_dir)
            (data_dir / "memories.json").write_text("{}", encoding="utf-8")
            (data_dir / "feedback.json").write_text("[]", encoding="utf-8")
            (data_dir / "style_samples.json").write_text("[]", encoding="utf-8")
            try:
                suite("repository", ["test_sqlite_memory_repository"])
                if include_migration:
                    suite("migration", ["test_json_to_sqlite_migration"])
                if include_retriever:
                    suite("retriever", ["test_sqlite_memory_retriever"])
                setup_started = time.perf_counter()
                # The old storage loads relative JSON at import, so import only
                # after chdir. Style/feedback defaults must be bound explicitly.
                import main
                import style_memory
                import feedback_store
                import requests

                style_memory.load_style_samples = functools.partial(
                    style_memory.load_style_samples, path=data_dir / "style_samples.json",
                )
                main.record_feedback = functools.partial(
                    feedback_store.record_feedback, path=data_dir / "feedback.json",
                )

                def refuse_http(session, method, url, *args, **kwargs):
                    raise AssertionError(f"Unexpected real HTTP in regression tests: {method} {url}")

                requests.sessions.Session.request = refuse_http
                report["baseline_setup_seconds"] = time.perf_counter() - setup_started
                suite("baseline", BASELINE_MODULES)
            finally:
                os.chdir(previous_cwd)
        report["baseline_temporary_data_removed"] = not data_dir.exists()
    except Exception:
        report["suite_exception"] = traceback.format_exc()
        print(report["suite_exception"], flush=True)
    finally:
        os.chdir(previous_cwd)
        guard_state["active"] = False
        report["source_sha256_after"] = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in originals
        }
        report["original_files_unchanged"] = originals == report["source_sha256_after"]
        report["total_elapsed_seconds"] = time.perf_counter() - started
        report["database_paths"] = sorted(set(report["database_paths"]))
        report["temporary_databases_removed"] = all(
            path == ":memory:" or not Path(path).exists() for path in report["database_paths"]
        )
        report["status"] = "PASS" if (
            report.get("repository", {}).get("status") == "PASS"
            and report.get("baseline", {}).get("status") == "PASS"
            and (not include_migration or report.get("migration", {}).get("status") == "PASS")
            and (not include_retriever or report.get("retriever", {}).get("status") == "PASS")
            and report["original_files_unchanged"] and not report["blocked_io"]
            and report["temporary_databases_removed"]
        ) else "FAIL"
        (report_dir / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"RESULT: {report['status']} | {report_dir / 'results.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-migration", action="store_true", help="Also run phase 2 migration tests")
    parser.add_argument("--include-retriever", action="store_true", help="Run phase 3 retriever tests and all earlier suites")
    args = parser.parse_args()
    raise SystemExit(run(include_migration=args.include_migration, include_retriever=args.include_retriever))
