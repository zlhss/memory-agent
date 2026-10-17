"""Real Agent V3/V2 smoke; all mutable data lives in a temporary directory.

Run with the existing memory_agent interpreter. No inference or CRUD is mocked.
The audit hook refuses production-data access and writes to existing repo files.
Reports contain only synthetic contacts, real model results and measurements.
"""

import copy
import functools
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_NAMES = {"memories.json", "feedback.json", "style_samples.json"}
CONTACT_A = "agent_v3_smoke_alpha"
CONTACT_B = "agent_v3_smoke_beta"
SENTINEL = "agent_v3_smoke_sentinel"
SEED = {CONTACT_A: [], CONTACT_B: ["联系人住在厦门"], SENTINEL: ["联系人喜欢天文观测"]}
STYLES = [
    {"sender_id": CONTACT_A, "incoming": "我最近换了住的城市。", "reply": "好呀，记住啦。"},
    {"sender_id": CONTACT_B, "incoming": "你记得我住在哪吗？", "reply": "记得呀。"},
]
CASES = [
    ("new", CONTACT_A, "我住在苏州。"),
    ("update", CONTACT_A, "我现在已经搬到南京居住了。"),
    ("retrieve", CONTACT_A, "我现在住在哪个城市？"),
    ("duplicate", CONTACT_A, "我住在南京。"),
    ("isolation", CONTACT_B, "我现在住在哪个城市？"),
    ("delete", CONTACT_A, "我不再住在南京了。"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def within(path, directory):
    return path == directory or directory in path.parents


def run():
    sys.dont_write_bytecode = True
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "smoke_reports" / f"agent_v3_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    # Includes frozen Python case definitions as well as JSON evaluation artifacts.
    original_files = {str(p): digest(p) for p in ROOT.iterdir() if p.is_file()}
    report = {
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "source_sha256_before": original_files,
        "cases": [],
        "blocked_io": [],
        "real_http_calls": [],
        "embedding_calls": [],
    }
    suite_started = time.perf_counter()
    previous_cwd = Path.cwd()
    # Hold this object until the finally block; retain snapshots, remove temp data.
    temporary = tempfile.TemporaryDirectory(prefix="agent-v3-real-smoke-")
    data_dir = Path(temporary.name).resolve()
    paths = {name: data_dir / name for name in DATA_NAMES}
    write_json(paths["memories.json"], SEED)
    write_json(paths["feedback.json"], [])
    write_json(paths["style_samples.json"], STYLES)
    audit_state = {"active": True}

    def guard(event, args):
        if not audit_state["active"] or event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (
            isinstance(mode, str) and any(char in mode for char in "wax+")
        ) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        reason = None
        if path.name in DATA_NAMES and path != paths[path.name]:
            reason = "data path is outside the isolated temporary directory"
        elif writing and within(path, ROOT) and not within(path, report_dir):
            reason = "write to source, production data or frozen evaluation file"
        if reason:
            report["blocked_io"].append({"path": str(path), "reason": reason})
            raise PermissionError(f"Smoke isolation guard: {reason}: {path}")

    sys.addaudithook(guard)
    exit_code = 1
    try:
        os.chdir(data_dir)
        assert Path.cwd() == data_dir and data_dir != ROOT
        assert all(path.parent == data_dir for path in paths.values())
        assert not any(name in sys.modules for name in ("main", "memory_search", "memory_v2"))
        # No project imports before chdir: memory_search loads memories at import.
        import memory_search
        import feedback_store
        import style_memory
        import main
        import memory_v2
        import ollama_client
        import requests
        import sentence_transformers

        # Bind the actual load/record functions, not inference or storage doubles.
        style_memory.load_style_samples = functools.partial(
            style_memory.load_style_samples, path=paths["style_samples.json"],
        )
        main.record_feedback = functools.partial(
            feedback_store.record_feedback, path=paths["feedback.json"],
        )
        assert style_memory.load_style_samples.keywords["path"] == paths["style_samples.json"]
        assert main.record_feedback.keywords["path"] == paths["feedback.json"]
        assert memory_search.load_memories() == SEED == memory_search.memories_by_sender
        assert style_memory.load_style_samples() == STYLES
        assert feedback_store.load_feedback(paths["feedback.json"]) == []
        main.reset_working_memories()
        report["isolation"] = {
            "confirmed_before_requests": True,
            "cwd": str(Path.cwd()),
            "data_paths": {name: str(path) for name, path in paths.items()},
            "guard": "block all other paths with production data names; block repo writes outside this report directory",
            "seed": SEED,
            "styles": STYLES,
        }
        report["model"] = {
            "ollama_url": ollama_client.OLLAMA_URL,
            "ollama_model": ollama_client.OLLAMA_MODEL,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_class": type(memory_search.model).__module__ + "." + type(memory_search.model).__name__,
            "embedding_device": str(memory_search.model.device),
            "sentence_transformers_version": sentence_transformers.__version__,
            "requests_version": requests.__version__,
            "offline_embedding_cache": True,
        }
        version = requests.get("http://localhost:11434/api/version", timeout=10)
        version.raise_for_status()
        report["model"]["ollama_version"] = version.json()
        tags = requests.get("http://localhost:11434/api/tags", timeout=10)
        tags.raise_for_status()
        report["model"]["ollama_model_metadata"] = [
            item for item in tags.json()["models"] if item["name"] == ollama_client.OLLAMA_MODEL
        ]
        assert report["model"]["ollama_model_metadata"], "Configured Ollama model is unavailable"

        current_case = {"name": "preflight"}
        original_request = requests.sessions.Session.request

        def measured_request(session, method, url, *args, **kwargs):
            if method.upper() != "POST" or url != ollama_client.OLLAMA_URL:
                return original_request(session, method, url, *args, **kwargs)
            payload = kwargs["json"]
            record = {"case": current_case["name"], "model": payload["model"],
                      "prompt_sha256": hashlib.sha256(payload["prompt"].encode()).hexdigest(),
                      "prompt": payload["prompt"], "start": time.perf_counter()}
            report["real_http_calls"].append(record)
            try:
                response = original_request(session, method, url, *args, **kwargs)
                record["http_status"] = response.status_code
                record["response"] = response.json()
                return response
            except Exception:
                record["error"] = traceback.format_exc()
                raise
            finally:
                record["elapsed_seconds"] = time.perf_counter() - record["start"]
                record["end"] = time.perf_counter()

        requests.sessions.Session.request = measured_request
        original_encode = memory_search.model.encode

        def measured_encode(sentences, *args, **kwargs):
            started = time.perf_counter()
            vectors = original_encode(sentences, *args, **kwargs)
            report["embedding_calls"].append({
                "case": current_case["name"], "sentences": list(sentences),
                "shape": list(vectors.shape), "elapsed_seconds": time.perf_counter() - started,
            })
            return vectors

        memory_search.model.encode = measured_encode
        setup_elapsed = time.perf_counter() - suite_started
        report["setup_elapsed_seconds"] = setup_elapsed
        print(f"ISOLATION_CONFIRMED: {data_dir}", flush=True)
        print(f"MODEL: {report['model']['ollama_model']}; embedding={report['model']['embedding_model']}", flush=True)

        for case_name, sender, incoming in CASES:
            current_case["name"] = case_name
            # Reconstruct from disk and discard A's working context to prove LTM recall.
            if case_name == "retrieve":
                memory_search.memories_by_sender = memory_search.load_memories()
                main.working_memories.pop(CONTACT_A, None)
            before = copy.deepcopy(memory_search.memories_by_sender)
            wm_before = copy.deepcopy(main.get_working_memory(sender).get_messages())
            http_start = len(report["real_http_calls"])
            embedding_start = len(report["embedding_calls"])
            result = None
            case = {"name": case_name, "sender_id": sender, "input": incoming,
                    "memory_before": before, "working_messages_before": wm_before, "checks": {}}
            started = time.perf_counter()
            print(f"CASE_START: {case_name} | {incoming}", flush=True)
            try:
                result = main.process_user_message(sender, incoming, memory_version="v2")
            except Exception:
                case["exception"] = traceback.format_exc()
            ended = time.perf_counter()
            case["request_elapsed_seconds"] = ended - started
            case["result"] = result
            case["memory_after"] = copy.deepcopy(memory_search.memories_by_sender)
            case["disk_after"] = memory_search.load_memories()
            http = report["real_http_calls"][http_start:]
            embeddings = report["embedding_calls"][embedding_start:]
            case["real_llm_call_count"] = len(http)
            case["real_llm_elapsed_seconds"] = sum(item["elapsed_seconds"] for item in http)
            case["embedding_call_count"] = len(embeddings)
            checks = case["checks"]
            checks["request_succeeded"] = result is not None
            checks["disk_matches_live_state"] = case["disk_after"] == case["memory_after"]
            checks["other_senders_unchanged"] = all(
                case["memory_after"].get(key) == value for key, value in before.items() if key != sender
            ) and set(case["memory_after"]) == set(before)
            checks["no_exact_duplicates"] = all(len(values) == len(set(values)) for values in case["disk_after"].values())
            checks["http_success"] = bool(http) and all(item.get("http_status") == 200 and "error" not in item for item in http)
            if result:
                details = result["memory_v2_details"]
                profile = result["timing_profile"]
                stages = [item["stage"] for item in profile["calls"]]
                actions = details["action_results"]
                decisions = [item["alignment"]["decision"] for item in details["fact_results"]]
                final = case["disk_after"][sender]
                checks["personalized_reply_nonempty"] = bool(result["personalized_reply"].strip())
                checks["v2_details_complete"] = details["pipeline_version"] == "v2" and details["sender_id"] == sender
                checks["final_memories_match_disk"] = details["final_memories"] == final
                checks["working_snapshot_matches"] = details["working_messages"] == wm_before
                checks["actions_successful"] = all(item["success"] for item in actions)
                checks["change_coverage_complete"] = details["coverage"]["uncovered_fact_indices"] == []
                checks["profile_counts_all_real_calls"] = profile["total_llm_calls"] == len(profile["calls"]) == len(http)
                checks["profile_covers_reply_and_memory"] = all(
                    stage in stages for stage in ("Planner", "Draft", "Style Rewriter", "Atomic Fact Extractor")
                ) and stages.index("Draft") < stages.index("Atomic Fact Extractor")
                checks["profile_full_request_duration"] = -0.002 <= case["request_elapsed_seconds"] - profile["total_request_elapsed_seconds"] <= 0.25
                checks["profile_http_duration_matches"] = abs(profile["total_llm_elapsed_seconds"] - case["real_llm_elapsed_seconds"]) < 0.1
                checks["profile_bounds_all_http_calls"] = all(started <= item["start"] < item["end"] <= ended for item in http)
                checks["real_embedding_executed"] = bool(embeddings) and all(item["shape"][-1] == 384 for item in embeddings)
                checks["style_is_sender_scoped"] = all(
                    item["reply"] in {style["reply"] for style in STYLES if style["sender_id"] == sender}
                    for item in result["style_samples"]
                ) and bool(result["style_samples"])
                if case_name == "new":
                    checks["new_fact_add_once"] = decisions == ["NEW"] and len(actions) == 1 and actions[0]["action"]["action"] == "ADD" and len(final) == 1 and "苏州" in final[0]
                    checks["evidence_and_worthiness"] = all(
                        item["validation"].get("evidence", {}).get("supported") is True
                        and item["validation"].get("worthiness", {}).get("should_save") is True
                        for item in details["fact_results"]
                    )
                elif case_name == "update":
                    checks["update_replaces_old_fact_once"] = decisions == ["UPDATE"] and len(actions) == 1 and actions[0]["action"]["action"] == "UPDATE" and len(final) == 1 and "南京" in final[0] and "苏州" not in final[0]
                    checks["restricted_consistency_no_secondary_crud"] = len(details["consistency_results"]) == 1 and details["consistency_results"][0]["actions"] == []
                elif case_name in {"retrieve", "isolation"}:
                    expected_city = "南京" if case_name == "retrieve" else "厦门"
                    checks["recall_from_ltm"] = any(expected_city in item for item in result["relevant_memories"]) and expected_city in result["personalized_reply"]
                    checks["query_does_not_write"] = not actions and case["disk_after"] == before
                    checks["no_working_context_needed"] = wm_before == []
                    if case_name == "isolation":
                        # Includes retrieval/embedding inputs, all prompts and the entire returned result.
                        surface = json.dumps([result, [item["prompt"] for item in http], embeddings], ensure_ascii=False)
                        checks["no_alpha_memory_in_beta_inputs_or_output"] = all(token not in surface for token in ("苏州", "南京", CONTACT_A))
                elif case_name == "duplicate":
                    checks["semantic_duplicate_no_write"] = decisions == ["DUPLICATE"] and not actions and case["disk_after"] == before
                elif case_name == "delete":
                    checks["delete_removes_only_target"] = decisions == ["DELETE"] and len(actions) == 1 and actions[0]["action"]["action"] == "DELETE" and final == []
                # Exercise the real feedback and Working Memory closure, only in temp files.
                # "accepted" here is simulated test feedback, not a human quality score.
                feedback_before = feedback_store.load_feedback(paths["feedback.json"])
                record = main.finalize_response(sender, result, "accepted")
                feedback_after = feedback_store.load_feedback(paths["feedback.json"])
                checks["feedback_persisted_once"] = feedback_after == feedback_before + [record]
                checks["working_memory_finalized"] = main.get_working_memory(sender).get_messages()[-2:] == [
                    {"role": "user", "content": incoming},
                    {"role": "assistant", "content": result["personalized_reply"].strip()},
                ]
            case["status"] = "PASS" if all(checks.values()) else "FAIL"
            case["failed_checks"] = [key for key, value in checks.items() if not value]
            report["cases"].append(case)
            write_json(report_dir / "results.json", report)
            print(f"CASE_RESULT: {case_name} | {case['status']} | calls={len(http)} | request={case['request_elapsed_seconds']:.3f}s | failed={case['failed_checks']}", flush=True)

        report["final_memory"] = memory_search.load_memories()
        report["final_feedback"] = feedback_store.load_feedback(paths["feedback.json"])
        report["final_working_memory"] = {key: value.get_messages() for key, value in main.working_memories.items()}
        report["passed"] = sum(item["status"] == "PASS" for item in report["cases"])
        exit_code = 0 if report["passed"] == len(CASES) else 1
    except Exception:
        report["suite_exception"] = traceback.format_exc()
        print(report["suite_exception"], flush=True)
    finally:
        os.chdir(previous_cwd)
        # Read-only original-file hashing after runtime is over.
        audit_state["active"] = False
        report["source_sha256_after"] = {path: digest(Path(path)) for path in original_files}
        report["protected_files_unchanged"] = original_files == report["source_sha256_after"]
        report["total_elapsed_seconds"] = time.perf_counter() - suite_started
        report["total_request_elapsed_seconds"] = sum(item["request_elapsed_seconds"] for item in report["cases"])
        report["total_real_llm_calls"] = len(report["real_http_calls"])
        report["total_real_llm_elapsed_seconds"] = sum(item["elapsed_seconds"] for item in report["real_http_calls"])
        report["status"] = "PASS" if exit_code == 0 and report["protected_files_unchanged"] and not report["blocked_io"] else "FAIL"
        temporary.cleanup()
        report["temporary_data_removed"] = not data_dir.exists()
        write_json(report_dir / "results.json", report)
        print(f"REPORT: {report_dir / 'results.json'}", flush=True)
        print(f"SUITE_RESULT: {report['status']} | {report.get('passed', 0)}/{len(CASES)} | real_calls={report['total_real_llm_calls']} | original_files_unchanged={report['protected_files_unchanged']}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
