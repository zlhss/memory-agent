import json
import io
from contextlib import redirect_stdout

import main
import memory_search
from regression_test_support import (
    TARGET_SENDER_ID,
    SENTINEL_MEMORIES,
    get_sentinel_memories,
    get_target_memories,
    is_environment_error,
    reset_case_state as reset_sender_case_state,
    sentinel_is_unchanged
)


# =========================================================
# 1. 保存原来的长期记忆
# =========================================================

with open(
    "memories.json",
    "r",
    encoding="utf-8"
) as f:
    original_memories = json.load(f)


# =========================================================
# 2. Benchmark 固定初始状态
# =========================================================

BASE_MEMORIES = [
    "联系人住在上海",
    "联系人正在学习日语",
    "联系人喜欢吃苹果",
    "联系人的朋友小明在教联系人日语，内容与动漫相关"
]


def reset_test_state(base_memories=None):
    """
    每个测试开始前：
    1. 重置长期记忆
    2. 清空 Working Memory
    """

    if base_memories is None:
        base_memories = BASE_MEMORIES

    reset_sender_case_state(
        memory_search,
        main,
        base_memories
    )


def target_memories():
    return get_target_memories(memory_search)


def run_message(user_text):
    """
    执行一次 Agent，同时抓取日志。

    测试成功时不打印大量内部日志；
    测试失败时再显示，方便调试。
    """

    output = io.StringIO()

    with redirect_stdout(output):
        result = main.process_user_message(
            TARGET_SENDER_ID,
            user_text
        )

    return result, output.getvalue()


# =========================================================
# 3. 基础 ADD
# =========================================================

def test_add_new_memory():

    reset_test_state()

    _, log = run_message(
        "我最近开始学法语"
    )

    passed = any(
        "法语" in memory
        and "学" in memory
        for memory in target_memories()
    )

    return {
        "name": "add_new_memory",
        "category": "ADD",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 4. Semantic Duplicate
# =========================================================

def test_duplicate_rejection():

    reset_test_state()

    _, log = run_message(
        "我最近开始学日语"
    )

    japanese_learning_memories = [
        memory
        for memory in target_memories()
        if "日语" in memory
        and "小明" not in memory
    ]

    passed = (
        len(japanese_learning_memories) == 1
    )

    return {
        "name": "duplicate_rejection",
        "category": "Duplicate",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 5. Worthiness：长期 + 瞬时事实
# =========================================================

def test_worthiness_filter():

    reset_test_state()

    _, log = run_message(
        "我最近开始学法语，我现在正在食堂吃饭"
    )

    has_french = any(
        "法语" in memory
        for memory in target_memories()
    )

    has_temporary_eating = any(
        "食堂" in memory
        or "吃饭" in memory
        for memory in target_memories()
    )

    passed = (
        has_french
        and not has_temporary_eating
    )

    return {
        "name": "worthiness_filter",
        "category": "Worthiness",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 6. UPDATE + Semantic Target Resolver
# =========================================================

def test_update_location():

    reset_test_state()

    _, log = run_message(
        "我已经搬到北京住了"
    )

    old_memory_exists = (
        "联系人住在上海"
        in target_memories()
    )

    new_memory_exists = any(
        "北京" in memory
        and "住" in memory
        for memory in target_memories()
    )

    passed = (
        not old_memory_exists
        and new_memory_exists
    )

    return {
        "name": "update_location",
        "category": "UPDATE",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 7. DELETE
# =========================================================

def test_delete_memory():

    reset_test_state()

    _, log = run_message(
        "我现在不学日语了"
    )

    still_learning = (
        "联系人正在学习日语"
        in target_memories()
    )

    passed = not still_learning

    return {
        "name": "delete_memory",
        "category": "DELETE",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 8. Consistency / Invalidation
# =========================================================

def test_consistency():

    reset_test_state()

    _, log = run_message(
        "我现在不学日语了"
    )

    still_learning = (
        "联系人正在学习日语"
        in target_memories()
    )

    still_currently_teaching = any(
        "小明在教联系人日语" in memory
        for memory in target_memories()
    )

    # 不强制要求必须使用“曾教”这个确切措辞，
    # 只要求仍然保留一个与小明 + 日语有关的更新后事实。
    updated_friend_memory = any(
        "小明" in memory
        and "日语" in memory
        and "在教联系人日语" not in memory
        for memory in target_memories()
    )

    passed = (
        not still_learning
        and not still_currently_teaching
        and updated_friend_memory
    )

    return {
        "name": "consistency",
        "category": "Consistency",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 9. Multi-action
# =========================================================

def test_multi_action():

    reset_test_state()

    _, log = run_message(
        "我现在不学日语了，小明也不用再教我了"
    )

    still_learning = (
        "联系人正在学习日语"
        in target_memories()
    )

    still_currently_teaching = any(
        "小明在教联系人日语" in memory
        for memory in target_memories()
    )

    updated_friend_memory = any(
        "小明" in memory
        and "日语" in memory
        and "在教联系人日语" not in memory
        for memory in target_memories()
    )

    passed = (
        not still_learning
        and not still_currently_teaching
        and updated_friend_memory
    )

    return {
        "name": "multi_action",
        "category": "Multi-action",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 10. Multi-ADD / 原子事实拆分
# =========================================================

def test_multi_add():

    reset_test_state()

    _, log = run_message(
        "我最近开始学法语，也开始每周跑步三次"
    )

    french_memories = [
        memory
        for memory in target_memories()
        if "法语" in memory
    ]

    running_memories = [
        memory
        for memory in target_memories()
        if "跑步" in memory
        and "Alice" not in memory
    ]

    # 希望被拆成两个独立长期事实
    passed = (
        len(french_memories) >= 1
        and len(running_memories) >= 1
    )

    return {
        "name": "multi_add",
        "category": "Multi-action",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 11. No-op / 不应该写长期记忆
# =========================================================

def test_no_op():

    reset_test_state()

    before = target_memories()

    _, log = run_message(
        "帮我写一个冒泡排序"
    )

    after = target_memories()

    passed = (
        before == after
    )

    return {
        "name": "no_op",
        "category": "No-op",
        "passed": passed,
        "memories": after,
        "log": log
    }


# =========================================================
# 12. 瞬时行为不能进入长期记忆
# =========================================================

def test_transient_action():

    reset_test_state()

    _, log = run_message(
        "我现在正在操场跑步"
    )

    polluted = any(
        "操场" in memory
        or "正在操场跑步" in memory
        for memory in target_memories()
    )

    passed = not polluted

    return {
        "name": "transient_action",
        "category": "Worthiness",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 13. Working Memory 指代消解
# =========================================================

def test_working_memory_reference():

    custom_memories = [
        "联系人住在上海",
        "联系人正在学习法语",
        "联系人喜欢吃苹果"
    ]

    reset_test_state(
        custom_memories
    )

    # 模拟之前的近期对话
    working_memory = main.get_working_memory(
        TARGET_SENDER_ID
    )

    working_memory.add_message(
        "user",
        "我最近法语学得有点累"
    )

    working_memory.add_message(
        "assistant",
        "你最近确实一直在学法语。"
    )

    _, log = run_message(
        "这个我暂时不学了"
    )

    still_learning_french = any(
        "法语" in memory
        and (
            "正在学习" in memory
            or "正在学" in memory
        )
        for memory in target_memories()
    )

    passed = not still_learning_french

    return {
        "name": "working_memory_reference",
        "category": "Working Memory",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 14. Held-out DELETE 表达
#     不使用我们开发时一直测试的“我现在不学日语了”
# =========================================================

def test_delete_paraphrase():

    reset_test_state()

    _, log = run_message(
        "日语我暂时不打算继续学了"
    )

    still_learning = (
        "联系人正在学习日语"
        in target_memories()
    )

    passed = not still_learning

    return {
        "name": "delete_paraphrase",
        "category": "Generalization",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 15. Held-out UPDATE 表达
# =========================================================

def test_update_paraphrase():

    reset_test_state()

    _, log = run_message(
        "以后常住地改成杭州了"
    )

    old_memory_exists = (
        "联系人住在上海"
        in target_memories()
    )

    new_memory_exists = any(
        "杭州" in memory
        for memory in target_memories()
    )

    passed = (
        not old_memory_exists
        and new_memory_exists
    )

    return {
        "name": "update_paraphrase",
        "category": "Generalization",
        "passed": passed,
        "memories": target_memories(),
        "log": log
    }


# =========================================================
# 16. 单个测试的安全运行包装
# =========================================================

def safe_run_test(test_func, category):

    try:
        result = test_func()

        sentinel_clean = sentinel_is_unchanged(
            memory_search
        )
        result["sentinel_unchanged"] = sentinel_clean
        result["sentinel_memories"] = get_sentinel_memories(
            memory_search
        )
        result["environment_error"] = False

        if not sentinel_clean:
            result["passed"] = False
            result.setdefault("errors", []).append(
                "检测到跨 sender 长期记忆污染"
            )

        result["status"] = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        return result

    except Exception as error:

        sentinel_clean = sentinel_is_unchanged(
            memory_search
        )

        return {
            "name": test_func.__name__,
            "category": category,
            "passed": False,
            "status": "ERROR",
            "memories": target_memories(),
            "sentinel_unchanged": sentinel_clean,
            "sentinel_memories": get_sentinel_memories(
                memory_search
            ),
            "log": "",
            "error": repr(error),
            "environment_error": is_environment_error(error)
        }


# =========================================================
# 17. Benchmark
# =========================================================

def run_evaluation():

    tests = [
        ("ADD", test_add_new_memory),
        ("Duplicate", test_duplicate_rejection),
        ("Worthiness", test_worthiness_filter),
        ("UPDATE", test_update_location),
        ("DELETE", test_delete_memory),
        ("Consistency", test_consistency),
        ("Multi-action", test_multi_action),
        ("Multi-action", test_multi_add),
        ("No-op", test_no_op),
        ("Worthiness", test_transient_action),
        ("Working Memory", test_working_memory_reference),
        ("Generalization", test_delete_paraphrase),
        ("Generalization", test_update_paraphrase),
    ]

    results = []

    for category, test in tests:

        print(
            f"Running {test.__name__}..."
        )

        result = safe_run_test(
            test,
            category
        )

        results.append(
            result
        )

    return results


# =========================================================
# 18. 输出结果
# =========================================================

def print_results(results):

    print("\n")
    print("=" * 60)
    print("MEMORY AGENT EVALUATION")
    print("=" * 60)

    passed_count = 0
    fail_count = 0
    error_count = 0
    environment_error_count = 0

    category_stats = {}

    for result in results:

        passed = result["passed"]
        status = result["status"]

        if passed:
            passed_count += 1
        elif status == "ERROR":
            error_count += 1
        else:
            fail_count += 1

        if result.get("environment_error"):
            environment_error_count += 1

        name = result["name"]
        category = result["category"]

        print(
            f"{name:<30} "
            f"{status:<7} "
            f"[{category}]"
        )

        if category not in category_stats:
            category_stats[category] = {
                "passed": 0,
                "total": 0
            }

        category_stats[category]["total"] += 1

        if passed:
            category_stats[category]["passed"] += 1

        # 只在失败的时候打印详细信息
        if not passed:

            if "error" in result:
                print(
                    "  Integration Error:"
                    if result.get("environment_error")
                    else "  Program Error:",
                    result["error"]
                )

            if not result["sentinel_unchanged"]:
                print(
                    "  Sentinel Pollution:",
                    result["sentinel_memories"]
                )

            print(
                "  Final Memories:"
            )

            for memory in result["memories"]:
                print(
                    "   -",
                    memory
                )

            if result.get("log"):
                print(
                    "\n  Internal Log:"
                )
                print(
                    result["log"]
                )

            print("-" * 60)

    total = len(results)

    accuracy = (
        passed_count / total
        if total > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Passed: {passed_count}/{total}"
    )

    print(f"FAIL: {fail_count}")
    print(f"ERROR: {error_count}")
    print(
        f"Environment/Integration Errors: "
        f"{environment_error_count}"
    )
    print(
        "Cross-sender Pollution: ",
        "YES"
        if any(
            not result["sentinel_unchanged"]
            for result in results
        )
        else "NO",
        sep=""
    )

    print(
        f"Pass Rate: {accuracy:.2%}"
    )

    print("\nBy Category:")

    for category, stats in category_stats.items():

        category_accuracy = (
            stats["passed"]
            / stats["total"]
        )

        print(
            f"  {category:<20} "
            f"{stats['passed']}/{stats['total']} "
            f"({category_accuracy:.2%})"
        )


# =========================================================
# 19. Main
# =========================================================

if __name__ == "__main__":

    try:

        results = run_evaluation()

        print_results(
            results
        )

    finally:

        # 无论测试成功、失败还是中途异常，
        # 尽量恢复测试前的长期记忆。
        memory_search.reset_memories(
            original_memories
        )
        main.reset_working_memories()

        print(
            "\nOriginal memories restored."
        )
