import io
import json
import os
import traceback
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import main
import memory_search
from ollama_client import check_action_coverage
from regression_test_support import (
    get_sentinel_memories,
    get_target_memories,
    is_environment_error,
    reset_case_state,
    sentinel_is_unchanged,
    TARGET_SENDER_ID,
)


MEMORY_FILE = Path(__file__).resolve().parent / "memories.json"


def run_agent(user_text):
    output = io.StringIO()

    with redirect_stdout(output):
        result = main.process_user_message(
            TARGET_SENDER_ID,
            user_text,
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


def mixed_delete_add_transient():
    user_text = (
        "吉他先不学了，我最近开始学钢琴，刚好现在在地铁站等车"
    )
    relevant_memories = [
        "联系人正在学习吉他",
        "联系人喜欢爵士乐",
    ]
    direct_coverage = check_action_coverage(
        user_text,
        relevant_memories,
        [
            {
                "action": "ADD",
                "target_index": None,
                "memory": "联系人正在学习钢琴",
            }
        ],
    )
    direct_guitar_change = any(
        action["action"] in {"UPDATE", "DELETE"}
        and action["target_index"] == 0
        for action in direct_coverage["missing_actions"]
    )

    reset_case_state(
        memory_search,
        main,
        relevant_memories,
    )
    result, log = run_agent(user_text)
    memories = get_target_memories(memory_search)
    processed_actions = result["processed_actions"]
    guitar_change_count = sum(
        action.get("action") in {"UPDATE", "DELETE"}
        and action.get("target") == "联系人正在学习吉他"
        for action in processed_actions
    )
    piano_added = any(
        action.get("action") == "ADD"
        and "钢琴" in action.get("memory", "")
        for action in processed_actions
    )
    transient_saved = any(
        "地铁站" in memory or "等车" in memory
        for memory in memories
    )

    return {
        "passed": (
            direct_guitar_change
            and guitar_change_count == 1
            and piano_added
            and not transient_saved
            and sentinel_is_unchanged(memory_search)
        ),
        "direct_coverage": direct_coverage,
        "guitar_change_count": guitar_change_count,
        "piano_added": piano_added,
        "transient_saved": transient_saved,
        "memories": memories,
        "sentinel_memories": get_sentinel_memories(memory_search),
        "log": log,
    }


def no_op_knowledge_question():
    base_memories = [
        "联系人住在西安",
        "联系人喜欢羽毛球",
    ]
    reset_case_state(
        memory_search,
        main,
        base_memories,
    )
    coverage = check_action_coverage(
        "为什么天空看起来是蓝色的？",
        base_memories,
        [],
    )
    result, log = run_agent(
        "为什么天空看起来是蓝色的？"
    )
    memories = get_target_memories(memory_search)

    return {
        "passed": (
            coverage == {"missing_actions": []}
            and memories == base_memories
            and result["processed_actions"] == []
            and sentinel_is_unchanged(memory_search)
        ),
        "coverage": coverage,
        "processed_actions": result["processed_actions"],
        "memories": memories,
        "sentinel_memories": get_sentinel_memories(memory_search),
        "log": log,
    }


def existing_update_is_not_duplicated():
    reset_case_state(
        memory_search,
        main,
        [
            "联系人住在南京",
            "联系人喜欢羽毛球",
        ],
    )
    result, log = run_agent(
        "我已经搬去苏州长期住了"
    )
    processed_actions = result["processed_actions"]
    location_target_count = sum(
        action.get("action") in {"UPDATE", "DELETE"}
        and action.get("target") == "联系人住在南京"
        for action in processed_actions
    )
    memories = get_target_memories(memory_search)

    return {
        "passed": (
            location_target_count == 1
            and "联系人住在南京" not in memories
            and any("苏州" in memory for memory in memories)
            and sentinel_is_unchanged(memory_search)
        ),
        "location_target_count": location_target_count,
        "processed_actions": processed_actions,
        "memories": memories,
        "sentinel_memories": get_sentinel_memories(memory_search),
        "log": log,
    }


def sender_sentinel(results):
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
            run_case(
                "mixed_delete_add_transient",
                mixed_delete_add_transient,
            ),
            run_case(
                "no_op_knowledge_question",
                no_op_knowledge_question,
            ),
            run_case(
                "existing_update_is_not_duplicated",
                existing_update_is_not_duplicated,
            ),
        ]
        results.append(sender_sentinel(results))

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
