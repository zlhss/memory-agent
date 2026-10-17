"""Development evaluation for the independent Memory V2 pipeline.

This suite deliberately exercises only public V2 behavior:
``process_user_message_v2`` plus the persisted sender memory state.
It is not a final or held-out evaluation.
"""

import copy
import io
import json
import time
from contextlib import redirect_stdout

import main
import memory_search
from memory_v2 import process_user_message_v2
from ollama_client import get_ollama_timings
from regression_test_support import is_environment_error


TARGET_SENDER_ID = "memory_v2_dev_target"
SENTINEL_SENDER_ID = "memory_v2_dev_sentinel"
SENTINEL_MEMORIES = [
    "联系人在哨兵单位负责档案管理",
    "联系人习惯周日整理纸质账单",
]


with open("memories.json", "r", encoding="utf-8") as file:
    ORIGINAL_MEMORIES = json.load(file)


def _reset_case(case):
    memory_search.reset_memories({
        TARGET_SENDER_ID: copy.deepcopy(case["initial_memories"]),
        SENTINEL_SENDER_ID: copy.deepcopy(case["sentinel_sender_memories"]),
    })
    main.reset_working_memories()
    working_memory = main.get_working_memory(TARGET_SENDER_ID)
    for message in case.get("working_memory", []):
        working_memory.add_message(message["role"], message["content"])


def _target_memories():
    return memory_search.get_sender_memories(TARGET_SENDER_ID).copy()


def _sentinel_memories():
    return memory_search.get_sender_memories(SENTINEL_SENDER_ID).copy()


def _contains(memory, *terms):
    return all(term in memory for term in terms)


def _has_memory(memories, *terms):
    return any(_contains(memory, *terms) for memory in memories)


def _lacks_concepts(memories, *terms):
    return all(
        all(term not in memory for term in terms)
        for memory in memories
    )


CASES = [
    {
        "case_id": "v2_constraint_fidelity_schedule_deadline",
        "category": "Constraint Fidelity",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": [],
        "working_memory": [],
        "user_text": "我从十月起每周二和周四晚上练习手语，一直坚持到明年十二月。",
        "expected_final_memories": [
            "一条以‘联系人’开头、同时保留手语、每周二、每周四、晚上、明年十二月约束的长期记忆"
        ],
        "forbidden_concepts": [],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 1
            and _contains(
                memories[0],
                "联系人",
                "手语",
                "每周二",
                "周四",
                "晚上",
                "明年十二月",
            )
        ),
    },
    {
        "case_id": "v2_same_class_distinct_preference",
        "category": "Distinct Object",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人早餐固定喝燕麦奶"],
        "working_memory": [],
        "user_text": "我晚饭时更喜欢喝无糖豆浆。",
        "expected_final_memories": [
            "联系人早餐固定喝燕麦奶",
            "一条关于晚饭偏好无糖豆浆的独立联系人记忆",
        ],
        "forbidden_concepts": ["把燕麦奶与豆浆判为重复"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 2
            and "联系人早餐固定喝燕麦奶" in memories
            and _has_memory(memories, "晚饭", "无糖豆浆")
        ),
    },
    {
        "case_id": "v2_two_independent_persistent_adds",
        "category": "Multi ADD",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": [],
        "working_memory": [],
        "user_text": "我在长期自学速记法，而且每个月参加一次社区河道清洁。",
        "expected_final_memories": [
            "一条关于长期自学速记法的记忆",
            "一条保留每个月一次频率的社区河道清洁记忆",
        ],
        "forbidden_concepts": ["把两个事实合并成一条"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 2
            and _has_memory(memories, "速记法")
            and _has_memory(memories, "每个月", "一次", "社区河道清洁")
        ),
    },
    {
        "case_id": "v2_update_plus_add",
        "category": "UPDATE + ADD",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人目前做夜班客服"],
        "working_memory": [],
        "user_text": "我下个月转成白班客服，还报名了为期半年的咖啡烘焙课程。",
        "expected_final_memories": [
            "一条取代夜班客服、保留下个月与白班限定的状态记忆",
            "一条保留为期半年限定的咖啡烘焙课程记忆",
        ],
        "forbidden_concepts": ["夜班客服仍是当前状态"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 2
            and _has_memory(memories, "下个月", "白班客服")
            and _has_memory(memories, "为期半年", "咖啡烘焙课程")
            and _lacks_concepts(memories, "夜班")
        ),
    },
    {
        "case_id": "v2_delete_plus_add",
        "category": "DELETE + ADD",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人每周参加线上读书会"],
        "working_memory": [],
        "user_text": "线上读书会我不参加了，之后每周六去做社区花园志愿者。",
        "expected_final_memories": [
            "一条保留每周六限定的社区花园志愿者记忆"
        ],
        "forbidden_concepts": ["仍参加线上读书会"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 1
            and _has_memory(memories, "每周六", "社区花园", "志愿者")
            and _lacks_concepts(memories, "线上读书会")
        ),
    },
    {
        "case_id": "v2_persistent_plus_transient",
        "category": "Persistent + Transient",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": [],
        "working_memory": [],
        "user_text": "我从今年起一直采用无麸质饮食，现在正在电梯口等同事。",
        "expected_final_memories": [
            "一条保留从今年起限定的无麸质饮食记忆"
        ],
        "forbidden_concepts": ["电梯口", "等同事"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 1
            and _has_memory(memories, "从今年起", "无麸质饮食")
            and _lacks_concepts(memories, "电梯口", "等同事")
        ),
    },
    {
        "case_id": "v2_change_add_transient",
        "category": "Change + ADD + Transient",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人正在参加城市定向训练"],
        "working_memory": [],
        "user_text": "城市定向训练我已经退出了，我开始准备潜水证考试，刚刚在便利店买水。",
        "expected_final_memories": [
            "一条关于开始准备潜水证考试的长期记忆"
        ],
        "forbidden_concepts": ["仍参加城市定向训练", "便利店", "买水"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 1
            and _has_memory(memories, "潜水证", "考试")
            and _lacks_concepts(memories, "城市定向训练", "便利店", "买水")
        ),
    },
    {
        "case_id": "v2_unique_working_memory_referent",
        "category": "Working Memory Unique",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人正在整理家族老照片数字化项目"],
        "working_memory": [
            {"role": "user", "content": "家族老照片数字化项目最近占用了很多周末。"},
            {"role": "assistant", "content": "听起来这个项目投入不小。"},
        ],
        "user_text": "这个项目我决定先暂停。",
        "expected_final_memories": [
            "原进行中状态被删除，或被更新成明确暂停状态"
        ],
        "forbidden_concepts": ["继续进行老照片数字化项目"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            "联系人正在整理家族老照片数字化项目" not in memories
            and (
                len(memories) == 0
                or (
                    len(memories) == 1
                    and _has_memory(memories, "家族老照片", "暂停")
                )
            )
        ),
    },
    {
        "case_id": "v2_multiple_referents_abstain",
        "category": "Working Memory Ambiguous",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": [
            "联系人正在制作社区口述史地图",
            "联系人正在整理阳台植物图鉴",
        ],
        "working_memory": [
            {"role": "user", "content": "社区口述史地图和阳台植物图鉴最近都卡住了。"},
            {"role": "assistant", "content": "两个项目同时推进确实不容易。"},
        ],
        "user_text": "这个我不做了。",
        "expected_final_memories": [
            "联系人正在制作社区口述史地图",
            "联系人正在整理阳台植物图鉴",
        ],
        "forbidden_concepts": ["在两个合理 referent 中猜一个删除"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: memories == [
            "联系人正在制作社区口述史地图",
            "联系人正在整理阳台植物图鉴",
        ],
    },
    {
        "case_id": "v2_knowledge_question_noop",
        "category": "Knowledge Question",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人习惯睡前阅读纸质书"],
        "working_memory": [],
        "user_text": "潮汐为什么每天会有变化？",
        "expected_final_memories": ["联系人习惯睡前阅读纸质书"],
        "forbidden_concepts": ["询问潮汐", "想了解潮汐"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: memories == [
            "联系人习惯睡前阅读纸质书"
        ],
    },
    {
        "case_id": "v2_one_time_request_noop",
        "category": "One-time Request",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人养了一只玄凤鹦鹉"],
        "working_memory": [],
        "user_text": "把这段通知改成更正式的语气。",
        "expected_final_memories": ["联系人养了一只玄凤鹦鹉"],
        "forbidden_concepts": ["修改通知", "正式语气偏好"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: memories == [
            "联系人养了一只玄凤鹦鹉"
        ],
    },
    {
        "case_id": "v2_delete_paraphrase_preserves_unrelated",
        "category": "Unrelated Preservation",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": [
            "联系人每周三晚上参加陶艺课",
            "联系人对贝类食物严重过敏",
        ],
        "working_memory": [],
        "user_text": "这学期结束后我就不再去上陶艺课了。",
        "expected_final_memories": ["联系人对贝类食物严重过敏"],
        "forbidden_concepts": ["仍参加陶艺课", "删除贝类过敏"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: memories == [
            "联系人对贝类食物严重过敏"
        ],
    },
    {
        "case_id": "v2_restricted_consistency_no_overdelete",
        "category": "Restricted Consistency",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": [
            "联系人目前在做社区播客剪辑项目",
            "联系人的姐姐喜欢收听历史播客",
            "联系人周末常去图书馆",
        ],
        "working_memory": [],
        "user_text": "社区播客剪辑项目已经交付结束了。",
        "expected_final_memories": [
            "联系人的姐姐喜欢收听历史播客",
            "联系人周末常去图书馆",
        ],
        "forbidden_concepts": ["删除姐姐的播客偏好", "删除图书馆习惯"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            "联系人目前在做社区播客剪辑项目" not in memories
            and "联系人的姐姐喜欢收听历史播客" in memories
            and "联系人周末常去图书馆" in memories
            and len(memories) == 2
        ),
    },
    {
        "case_id": "v2_sender_isolation_with_similar_topics",
        "category": "Sender Isolation",
        "sender_id": TARGET_SENDER_ID,
        "initial_memories": ["联系人每周进行两次室内攀岩"],
        "working_memory": [],
        "user_text": "我把室内攀岩调整为每周四次。",
        "expected_final_memories": [
            "一条保留每周四次限定、取代每周两次的室内攀岩记忆"
        ],
        "forbidden_concepts": ["每周两次仍是当前频率", "修改 sentinel"],
        "sentinel_sender_memories": SENTINEL_MEMORIES,
        "assertion": lambda memories: (
            len(memories) == 1
            and _has_memory(memories, "室内攀岩", "每周四次")
            and _lacks_concepts(memories, "每周进行两次")
        ),
    },
]


def run_case(case):
    _reset_case(case)
    output = io.StringIO()
    started = time.perf_counter()
    result = None

    try:
        with redirect_stdout(output):
            result = process_user_message_v2(
                case["sender_id"],
                case["user_text"],
            )
        elapsed = time.perf_counter() - started
        memories = _target_memories()
        sentinel = _sentinel_memories()
        sender_clean = sentinel == case["sentinel_sender_memories"]
        passed = bool(case["assertion"](memories)) and sender_clean
        profile = result.get("timing_profile", {})
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "status": "PASS" if passed else "BEHAVIOR_FAIL",
            "elapsed": elapsed,
            "llm_calls": profile.get("total_llm_calls", 0),
            "llm_elapsed": profile.get("total_llm_elapsed_seconds", 0.0),
            "sender_pollution": not sender_clean,
            "final_memories": memories,
            "sentinel_memories": sentinel,
            "result": result,
            "log": output.getvalue(),
        }
    except Exception as error:
        elapsed = time.perf_counter() - started
        sentinel = _sentinel_memories()
        calls = get_ollama_timings()
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "status": (
                "ENVIRONMENT_ERROR"
                if is_environment_error(error)
                else "PROGRAM_ERROR"
            ),
            "elapsed": elapsed,
            "llm_calls": len(calls),
            "llm_elapsed": sum(
                call.get("elapsed_seconds", 0.0) for call in calls
            ),
            "sender_pollution": sentinel != case["sentinel_sender_memories"],
            "final_memories": _target_memories(),
            "sentinel_memories": sentinel,
            "error": repr(error),
            "log": output.getvalue(),
        }


def run_evaluation():
    results = []
    for case in CASES:
        print(f"Running {case['case_id']}...")
        results.append(run_case(case))
    return results


def print_results(results):
    print("\nMEMORY V2 DEVELOPMENT EVALUATION")
    for result in results:
        print(
            f"{result['case_id']}: {result['status']} "
            f"({result['elapsed']:.3f}s, {result['llm_calls']} LLM calls, "
            f"sender pollution={'YES' if result['sender_pollution'] else 'NO'})"
        )
        if result["status"] != "PASS":
            print("  Final memories:", result["final_memories"])
            print("  Sentinel memories:", result["sentinel_memories"])
            if result.get("error"):
                print("  Error:", result["error"])
            if result.get("log"):
                print("  Log:\n", result["log"])
            if result.get("result"):
                print(
                    "  V2 Debug:\n",
                    json.dumps(result["result"], ensure_ascii=False, indent=2),
                )

    total = len(results)
    status_counts = {
        status: sum(result["status"] == status for result in results)
        for status in (
            "PASS",
            "BEHAVIOR_FAIL",
            "PROGRAM_ERROR",
            "ENVIRONMENT_ERROR",
        )
    }
    average_latency = (
        sum(result["elapsed"] for result in results) / total
        if total
        else 0.0
    )
    average_calls = (
        sum(result["llm_calls"] for result in results) / total
        if total
        else 0.0
    )
    print("\nSUMMARY")
    for status, count in status_counts.items():
        print(f"{status}: {count}")
    print(f"Average latency: {average_latency:.3f}s")
    print(f"Average LLM calls: {average_calls:.2f}")
    print(
        "Cross-sender pollution:",
        "YES" if any(result["sender_pollution"] for result in results) else "NO",
    )


if __name__ == "__main__":
    try:
        evaluation_results = run_evaluation()
        print_results(evaluation_results)
    finally:
        memory_search.reset_memories(ORIGINAL_MEMORIES)
        main.reset_working_memories()
        print("\nOriginal memories restored; Working Memory cleared.")
