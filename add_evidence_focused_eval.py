import io
import json
import os
import traceback
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import main
import memory_search
from ollama_client import check_add_evidence
from regression_test_support import (
    get_sentinel_memories,
    get_target_memories,
    is_environment_error,
    reset_case_state,
    sentinel_is_unchanged,
    TARGET_SENDER_ID,
)


MEMORY_FILE = Path(__file__).resolve().parent / "memories.json"
BASE_MEMORIES = ["联系人喜欢羽毛球"]


def run_agent(user_text):
    output = io.StringIO()

    with redirect_stdout(output):
        result = main.process_user_message(
            TARGET_SENDER_ID,
            user_text,
        )

    return result, output.getvalue()


def run_gate(user_text, new_memory):
    output = io.StringIO()

    with redirect_stdout(output):
        result = check_add_evidence(
            user_text,
            new_memory,
            [],
        )

    return result, output.getvalue()


def run_case(name, callback):
    try:
        details = callback()
        passed = details.pop("passed")
        return {
            "name": name,
            "status": "PASS" if passed else "BEHAVIOR_FAIL",
            "details": details,
        }
    except Exception as error:
        return {
            "name": name,
            "status": (
                "ENVIRONMENT_ERROR"
                if is_environment_error(error)
                else "PROGRAM_ERROR"
            ),
            "details": {
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        }


def run_full_case(
    user_text,
    candidate_expectations,
    memory_check,
):
    gate_results = []
    gate_logs = []

    for new_memory, expected_supported in candidate_expectations:
        gate_result, gate_log = run_gate(user_text, new_memory)
        gate_results.append({
            "new_memory": new_memory,
            "expected_supported": expected_supported,
            "actual": gate_result,
        })
        gate_logs.append(gate_log)

    reset_case_state(
        memory_search,
        main,
        BASE_MEMORIES,
    )
    result, agent_log = run_agent(user_text)
    memories = get_target_memories(memory_search)
    gates_match = all(
        item["actual"] == {
            "supported": item["expected_supported"]
        }
        for item in gate_results
    )

    return {
        "passed": (
            gates_match
            and memory_check(memories, result)
            and sentinel_is_unchanged(memory_search)
        ),
        "gate_results": gate_results,
        "memories": memories,
        "sentinel_memories": get_sentinel_memories(memory_search),
        "gate_logs": gate_logs,
        "agent_log": agent_log,
    }


def knowledge_question():
    return run_full_case(
        "为什么天空看起来是蓝色的？",
        [
            (
                "联系人询问天空为什么是蓝色的原因",
                False,
            )
        ],
        lambda memories, result: (
            memories == BASE_MEMORIES
            and result["processed_actions"] == []
        ),
    )


def one_time_request():
    return run_full_case(
        "帮我写一个冒泡排序",
        [
            (
                "联系人想学习冒泡排序",
                False,
            )
        ],
        lambda memories, result: (
            memories == BASE_MEMORIES
            and result["processed_actions"] == []
        ),
    )


def explicit_photography_learning():
    return run_full_case(
        "我最近开始学摄影了",
        [
            (
                "联系人最近开始学习摄影",
                True,
            )
        ],
        lambda memories, result: any(
            "摄影" in memory and "学" in memory
            for memory in memories
        ),
    )


def fact_plus_knowledge_question():
    return run_full_case(
        "我最近开始学摄影了，为什么光圈会影响景深？",
        [
            (
                "联系人最近开始学习摄影",
                True,
            ),
            (
                "联系人询问光圈影响景深的原因",
                False,
            ),
        ],
        lambda memories, result: (
            any(
                "摄影" in memory and "学" in memory
                for memory in memories
            )
            and not any(
                "光圈" in memory
                or "景深" in memory
                or "询问" in memory
                for memory in memories
            )
        ),
    )


def sentinel_result(results):
    polluted = any(
        result["details"].get("sentinel_memories")
        and result["details"]["sentinel_memories"]
        != [
            "联系人住在哨兵城市",
            "联系人喜欢哨兵音乐",
        ]
        for result in results
    )
    return {
        "name": "sender_sentinel",
        "status": "BEHAVIOR_FAIL" if polluted else "PASS",
        "details": {"cross_sender_pollution": polluted},
    }


def main_entry():
    original_memories = json.loads(
        MEMORY_FILE.read_text(encoding="utf-8")
    )

    try:
        results = [
            run_case("knowledge_question", knowledge_question),
            run_case("one_time_request", one_time_request),
            run_case(
                "explicit_photography_learning",
                explicit_photography_learning,
            ),
            run_case(
                "fact_plus_knowledge_question",
                fact_plus_knowledge_question,
            ),
        ]
        results.append(sentinel_result(results))

        for result in results:
            print(f"{result['name']}: {result['status']}")
            if result["status"] != "PASS":
                print(json.dumps(
                    result["details"],
                    ensure_ascii=False,
                    indent=2,
                ))

        return 0 if all(
            result["status"] == "PASS"
            for result in results
        ) else 1
    finally:
        memory_search.reset_memories(original_memories)
        main.reset_working_memories()


if __name__ == "__main__":
    raise SystemExit(main_entry())
