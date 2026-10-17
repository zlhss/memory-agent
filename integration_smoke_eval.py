import json
import os
import sys
import time
import traceback
from pathlib import Path

from regression_test_support import is_environment_error


os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_FILE = PROJECT_DIR / "memories.json"

SMOKE_CASES = [
    ("ADD", "add_korean_learning"),
    ("UPDATE", "update_city"),
    ("DELETE", "delete_guitar_learning"),
    ("Semantic Duplicate", "duplicate_swimming_habit"),
    ("Worthiness", "worthiness_exam_plus_queue"),
    ("Multi-action", "multi_switch_language"),
    ("Consistency", "consistency_exam_friend"),
    (
        "Working Memory Reference",
        "working_memory_reference_photography"
    )
]


def public_status(result):
    if result["status"] == "PASS":
        return "PASS"

    if result["status"] == "FAIL":
        return "BEHAVIOR_FAIL"

    if result["environment_error"]:
        return "ENVIRONMENT_ERROR"

    return "PROGRAM_ERROR"


def print_failure_log(result):
    print("  FAILURE_LOG:")

    for error in result["errors"]:
        print(f"    - {error}")

    if result["exception"]:
        print("  EXCEPTION:")
        print(result["exception"].rstrip())

    if result["log"]:
        print("  INTERNAL_LOG:")
        print(result["log"].rstrip())


def run_smoke_cases():
    original_memories = json.loads(
        MEMORY_FILE.read_text(encoding="utf-8")
    )

    try:
        import main
        import memory_search
        import memory_heldout_eval
        import ollama_client
    except Exception as error:
        status = (
            "ENVIRONMENT_ERROR"
            if is_environment_error(error)
            else "PROGRAM_ERROR"
        )
        print(f"SUITE_STATUS: {status}")
        print(traceback.format_exc().rstrip())
        return 2

    cases_by_name = {
        case["name"]: case
        for case in memory_heldout_eval.HELDOUT_CASES
    }

    results = []

    try:
        for category, case_name in SMOKE_CASES:
            case = cases_by_name[case_name]
            started_at = time.perf_counter()
            result = memory_heldout_eval.evaluate_case(case)
            elapsed_seconds = time.perf_counter() - started_at
            status = public_status(result)
            llm_timings = ollama_client.get_ollama_timings()
            llm_call_count = len(llm_timings)
            llm_elapsed_seconds = sum(
                timing["elapsed_seconds"]
                for timing in llm_timings
            )

            result["smoke_category"] = category
            result["elapsed_seconds"] = elapsed_seconds
            result["public_status"] = status
            result["llm_call_count"] = llm_call_count
            result["llm_elapsed_seconds"] = llm_elapsed_seconds
            results.append(result)

            print(
                f"{category} | {case_name} | {status} | "
                f"{elapsed_seconds:.2f}s | "
                f"llm_calls={llm_call_count} | "
                f"llm_elapsed={llm_elapsed_seconds:.2f}s | "
                f"cross_sender_pollution="
                f"{'NO' if result['sentinel_unchanged'] else 'YES'}",
                flush=True
            )

            if status != "PASS":
                print_failure_log(result)

    finally:
        memory_search.reset_memories(original_memories)
        main.reset_working_memories()

    counts = {
        status: sum(
            result["public_status"] == status
            for result in results
        )
        for status in (
            "PASS",
            "BEHAVIOR_FAIL",
            "PROGRAM_ERROR",
            "ENVIRONMENT_ERROR"
        )
    }

    print("SUMMARY")
    for status, count in counts.items():
        print(f"{status}: {count}")

    total_elapsed_seconds = sum(
        result["elapsed_seconds"]
        for result in results
    )
    average_case_latency = (
        total_elapsed_seconds / len(results)
        if results
        else 0.0
    )
    average_llm_calls = (
        sum(
            result["llm_call_count"]
            for result in results
        ) / len(results)
        if results
        else 0.0
    )

    print(
        "ERROR: ",
        counts["PROGRAM_ERROR"] + counts["ENVIRONMENT_ERROR"],
        sep=""
    )
    print(f"total_elapsed_seconds={total_elapsed_seconds:.2f}")
    print(f"average_case_latency_seconds={average_case_latency:.2f}")
    print(f"average_llm_calls={average_llm_calls:.2f}")

    print(
        "cross_sender_pollution=",
        "YES"
        if any(
            not result["sentinel_unchanged"]
            for result in results
        )
        else "NO",
        sep=""
    )

    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_cases())
