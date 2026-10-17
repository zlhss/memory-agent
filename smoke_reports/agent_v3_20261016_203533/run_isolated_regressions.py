"""Run the unchanged 40 baseline tests without production-data access."""
import functools
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[name] = "1"
originals = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.iterdir() if path.is_file()}
protected = {ROOT / name for name in ("memories.json", "feedback.json", "style_samples.json")}
blocked = []
state = {"active": True}


def guard(event, args):
    if not state["active"] or event != "open" or isinstance(args[0], int):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if path in protected or (writing and ROOT in path.parents and REPORT_DIR not in path.parents):
        blocked.append(str(path))
        raise PermissionError(f"Isolated regression refused access: {path}")


sys.addaudithook(guard)
started = time.perf_counter()
exit_code = 1
summary = {}
with tempfile.TemporaryDirectory(prefix="agent-v3-regressions-") as directory:
    temporary = Path(directory)
    previous_cwd = Path.cwd()
    os.chdir(temporary)
    try:
        (temporary / "memories.json").write_text("{}", encoding="utf-8")
        (temporary / "feedback.json").write_text("[]", encoding="utf-8")
        (temporary / "style_samples.json").write_text("[]", encoding="utf-8")
        import main
        import style_memory
        import feedback_store
        import requests

        style_memory.load_style_samples = functools.partial(style_memory.load_style_samples, path=temporary / "style_samples.json")
        main.record_feedback = functools.partial(feedback_store.record_feedback, path=temporary / "feedback.json")
        # Existing tests provide their own scoped mocks. An unexpected real HTTP
        # request is an error, never an uncounted inference during regressions.
        original_request = requests.sessions.Session.request

        def refuse_http(session, method, url, *args, **kwargs):
            raise AssertionError(f"Unexpected real HTTP request in baseline tests: {method} {url}")

        requests.sessions.Session.request = refuse_http
        modules = ["test_contextual_memory_flow", "test_ollama_timing", "test_response_pipeline_mock", "test_memory_v2_interface_mock", "test_agent_memory_versions"]
        with (REPORT_DIR / "regression.log").open("w", encoding="utf-8") as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
        summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "modules": modules, "real_llm_calls": 0}
        exit_code = 0 if result.wasSuccessful() else 1
    finally:
        os.chdir(previous_cwd)
        state["active"] = False
        summary["original_files_unchanged"] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in originals.items())
        summary["blocked_io"] = blocked
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["status"] = "PASS" if exit_code == 0 and summary["original_files_unchanged"] and not blocked else "FAIL"
        (REPORT_DIR / "regression_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
raise SystemExit(0 if summary["status"] == "PASS" else 1)
