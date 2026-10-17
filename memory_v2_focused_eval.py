"""Focused safety checks for the final Memory V2 development repair."""

import copy
import json

import main
import memory_search
from memory_v2 import process_user_message_v2, retrieve_fact_candidates


TARGET_SENDER_ID = "memory_v2_focused_target"
SENTINEL_SENDER_ID = "memory_v2_focused_sentinel"
SENTINEL_MEMORIES = ["联系人长期负责哨兵资料校对"]


with open("memories.json", "r", encoding="utf-8") as file:
    ORIGINAL_MEMORIES = json.load(file)


def _reset(target_memories):
    memory_search.reset_memories({
        TARGET_SENDER_ID: copy.deepcopy(target_memories),
        SENTINEL_SENDER_ID: copy.deepcopy(SENTINEL_MEMORIES),
    })
    main.reset_working_memories()


def test_change_retrieval_keeps_low_score_paraphrase():
    old_memory = "联系人每周三晚上参加陶艺课"
    _reset([old_memory, "联系人对贝类食物严重过敏"])
    atomic_fact = {
        "source_span": "这学期结束后我就不再去上陶艺课了",
        "kind": "persistent_change",
        "operation_hint": "end",
        "fact": "联系人不再去上陶艺课",
        "constraints": ["这学期结束后"],
    }

    retrieval = retrieve_fact_candidates(
        TARGET_SENDER_ID,
        atomic_fact,
        top_k=5,
    )
    candidates = retrieval["candidates"]
    memories = [item["memory"] for item in candidates]
    score = next(
        (item["score"] for item in candidates if item["memory"] == old_memory),
        None,
    )

    return {
        "name": "change_retrieval_keeps_low_score_paraphrase",
        "passed": old_memory in memories and score is not None and score < 0.40,
        "details": retrieval,
        "sentinel_unchanged": (
            memory_search.get_sender_memories(SENTINEL_SENDER_ID)
            == SENTINEL_MEMORIES
        ),
    }


def test_multiple_referents_force_abstain():
    initial = [
        "联系人正在整理供应商税务归档项目",
        "联系人正在执行仓库年度盘点项目",
    ]
    _reset(initial)
    working_memory = main.get_working_memory(TARGET_SENDER_ID)
    working_memory.add_message(
        "user",
        "供应商税务归档项目和仓库年度盘点项目都遇到阻碍了。",
    )
    working_memory.add_message(
        "assistant",
        "两个项目一起处理确实会有压力。",
    )

    result = process_user_message_v2(
        TARGET_SENDER_ID,
        "这个我不继续做了。",
    )
    final_memories = memory_search.get_sender_memories(TARGET_SENDER_ID).copy()
    sentinel_unchanged = (
        memory_search.get_sender_memories(SENTINEL_SENDER_ID)
        == SENTINEL_MEMORIES
    )
    return {
        "name": "multiple_referents_force_abstain",
        "passed": final_memories == initial and not result["processed_actions"],
        "details": {
            "atomic_facts": result["atomic_facts"],
            "fact_results": result["fact_results"],
            "final_memories": final_memories,
        },
        "sentinel_unchanged": sentinel_unchanged,
    }


def test_future_deadline_is_new_persistent_fact():
    _reset([])
    result = process_user_message_v2(
        TARGET_SENDER_ID,
        "我计划从十一月开始每周末学习盲文，坚持到后年三月。",
    )
    final_memories = memory_search.get_sender_memories(TARGET_SENDER_ID).copy()
    persistent_add = any(
        fact["kind"] == "persistent"
        and fact["operation_hint"] == "add"
        for fact in result["atomic_facts"]
    )
    saved_with_constraints = any(
        "盲文" in memory
        and "十一月" in memory
        and "每周末" in memory
        and "后年三月" in memory
        for memory in final_memories
    )
    return {
        "name": "future_deadline_is_new_persistent_fact",
        "passed": persistent_add and saved_with_constraints,
        "details": {
            "atomic_facts": result["atomic_facts"],
            "final_memories": final_memories,
        },
        "sentinel_unchanged": (
            memory_search.get_sender_memories(SENTINEL_SENDER_ID)
            == SENTINEL_MEMORIES
        ),
    }


def test_paraphrased_delete_applies_and_preserves_unrelated():
    old_memory = "联系人每周三晚上参加陶艺课"
    unrelated = "联系人对贝类食物严重过敏"
    _reset([old_memory, unrelated])
    result = process_user_message_v2(
        TARGET_SENDER_ID,
        "这学期结束后我就不再去上陶艺课了。",
    )
    final_memories = memory_search.get_sender_memories(TARGET_SENDER_ID).copy()
    return {
        "name": "paraphrased_delete_applies_and_preserves_unrelated",
        "passed": (
            old_memory not in final_memories
            and unrelated in final_memories
            and len(final_memories) == 1
        ),
        "details": {
            "fact_results": result["fact_results"],
            "final_memories": final_memories,
        },
        "sentinel_unchanged": (
            memory_search.get_sender_memories(SENTINEL_SENDER_ID)
            == SENTINEL_MEMORIES
        ),
    }


def test_high_recall_does_not_delete_unrelated_candidates():
    initial = [
        "联系人每周参加晨间瑜伽",
        "联系人在银行负责合规审查",
    ]
    _reset(initial)
    result = process_user_message_v2(
        TARGET_SENDER_ID,
        "我不再参加周末帆船俱乐部了。",
    )
    final_memories = memory_search.get_sender_memories(TARGET_SENDER_ID).copy()
    return {
        "name": "high_recall_does_not_delete_unrelated_candidates",
        "passed": final_memories == initial and not result["processed_actions"],
        "details": {
            "fact_results": result["fact_results"],
            "final_memories": final_memories,
        },
        "sentinel_unchanged": (
            memory_search.get_sender_memories(SENTINEL_SENDER_ID)
            == SENTINEL_MEMORIES
        ),
    }


def test_unique_working_memory_referent_can_change():
    old_memory = "联系人正在维护家庭食谱索引项目"
    _reset([old_memory])
    working_memory = main.get_working_memory(TARGET_SENDER_ID)
    working_memory.add_message(
        "user",
        "家庭食谱索引项目最近需要重新整理分类。",
    )
    working_memory.add_message(
        "assistant",
        "这个项目确实需要不少整理工作。",
    )
    result = process_user_message_v2(
        TARGET_SENDER_ID,
        "这个项目我决定先暂停。",
    )
    final_memories = memory_search.get_sender_memories(TARGET_SENDER_ID).copy()
    changed = old_memory not in final_memories and bool(result["processed_actions"])
    return {
        "name": "unique_working_memory_referent_can_change",
        "passed": changed,
        "details": {
            "fact_results": result["fact_results"],
            "final_memories": final_memories,
        },
        "sentinel_unchanged": (
            memory_search.get_sender_memories(SENTINEL_SENDER_ID)
            == SENTINEL_MEMORIES
        ),
    }


def test_termination_without_target_is_noop():
    _reset([])
    result = process_user_message_v2(
        TARGET_SENDER_ID,
        "我不再参加夜间观星小组了。",
    )
    final_memories = memory_search.get_sender_memories(TARGET_SENDER_ID).copy()
    return {
        "name": "termination_without_target_is_noop",
        "passed": final_memories == [] and not result["processed_actions"],
        "details": {
            "fact_results": result["fact_results"],
            "final_memories": final_memories,
        },
        "sentinel_unchanged": (
            memory_search.get_sender_memories(SENTINEL_SENDER_ID)
            == SENTINEL_MEMORIES
        ),
    }


TESTS = [
    test_change_retrieval_keeps_low_score_paraphrase,
    test_multiple_referents_force_abstain,
    test_future_deadline_is_new_persistent_fact,
    test_paraphrased_delete_applies_and_preserves_unrelated,
    test_high_recall_does_not_delete_unrelated_candidates,
    test_unique_working_memory_referent_can_change,
    test_termination_without_target_is_noop,
]


def run_focused_tests():
    results = []
    for test in TESTS:
        try:
            result = test()
            if not result["sentinel_unchanged"]:
                result["passed"] = False
            result["status"] = "PASS" if result["passed"] else "FAIL"
        except Exception as error:
            result = {
                "name": test.__name__,
                "passed": False,
                "status": "ERROR",
                "error": repr(error),
                "sentinel_unchanged": (
                    memory_search.get_sender_memories(SENTINEL_SENDER_ID)
                    == SENTINEL_MEMORIES
                ),
            }
        results.append(result)
        print(
            f"{result['name']}: {result['status']} "
            f"sentinel={'UNCHANGED' if result['sentinel_unchanged'] else 'POLLUTED'}"
        )
        if result["status"] != "PASS":
            print(json.dumps(result, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    try:
        focused_results = run_focused_tests()
        passed = sum(result["passed"] for result in focused_results)
        print(f"Focused: {passed}/{len(focused_results)} PASS")
    finally:
        memory_search.reset_memories(ORIGINAL_MEMORIES)
        main.reset_working_memories()
        print("Original memories restored; Working Memory cleared.")
