import io
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import main
import memory_search
from regression_test_support import (
    TARGET_SENDER_ID,
    get_sentinel_memories,
    get_target_memories,
    is_environment_error,
    reset_case_state as reset_sender_case_state,
    sentinel_is_unchanged
)


PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_FILE = PROJECT_DIR / "memories.json"


# =========================================================
# 保存原始长期记忆
# =========================================================

with open(MEMORY_FILE, "r", encoding="utf-8") as f:
    ORIGINAL_MEMORIES = json.load(f)


# =========================================================
# 测试基础工具
# =========================================================

def reset_case_state(base_memories, working_messages=None):
    reset_sender_case_state(
        memory_search,
        main,
        base_memories,
        working_messages
    )


def run_message(user_text):
    """
    执行一次 Agent。
    正常时隐藏内部日志；
    FAIL 时再打印。
    """

    buffer = io.StringIO()

    try:
        with redirect_stdout(buffer):
            result = main.process_user_message(
                TARGET_SENDER_ID,
                user_text
            )

        return {
            "result": result,
            "log": buffer.getvalue(),
            "error": None
        }

    except Exception as error:
        return {
            "result": None,
            "log": buffer.getvalue(),
            "error": error,
            "traceback": traceback.format_exc()
        }


def contains_all(memory, keywords):
    return all(
        keyword in memory
        for keyword in keywords
    )


def count_matching(memories, keywords):
    return sum(
        1
        for memory in memories
        if contains_all(
            memory,
            keywords
        )
    )


def exists_all(memories, keywords):
    return (
        count_matching(
            memories,
            keywords
        )
        > 0
    )


def exists_any_group(memories, groups):
    return any(
        exists_all(
            memories,
            keywords
        )
        for keywords in groups
    )


# =========================================================
# 通用检查器
# =========================================================

def check_condition(
    check,
    memories,
    before_memories
):

    check_type = check["type"]

    # 至少有一条 memory 同时包含全部关键词
    if check_type == "exists_all":

        keywords = check["keywords"]

        if exists_all(
            memories,
            keywords
        ):
            return None

        return (
            f"缺少预期记忆，关键词: "
            f"{keywords}"
        )


    # 不允许出现同时包含这些关键词的 memory
    if check_type == "not_exists_all":

        keywords = check["keywords"]

        if not exists_all(
            memories,
            keywords
        ):
            return None

        return (
            f"出现了不应存在的记忆，关键词: "
            f"{keywords}"
        )


    # 多种合理表达，命中任意一种即可
    if check_type == "exists_any_group":

        groups = check["groups"]

        if exists_any_group(
            memories,
            groups
        ):
            return None

        return (
            f"没有命中任何允许的预期表达: "
            f"{groups}"
        )


    # 多种不允许的表达，任何一种都不能出现
    if check_type == "not_exists_any_group":

        groups = check["groups"]

        if not exists_any_group(
            memories,
            groups
        ):
            return None

        return (
            f"命中了不应存在的表达之一: "
            f"{groups}"
        )


    # 精确存在
    if check_type == "exact_present":

        memory = check["memory"]

        if memory in memories:
            return None

        return (
            f"缺少精确记忆: "
            f"{memory}"
        )


    # 精确不存在
    if check_type == "exact_absent":

        memory = check["memory"]

        if memory not in memories:
            return None

        return (
            f"旧记忆仍然存在: "
            f"{memory}"
        )


    # 某类记忆最多允许出现 N 条
    if check_type == "count_max":

        keywords = check["keywords"]
        maximum = check["max"]

        actual = count_matching(
            memories,
            keywords
        )

        if actual <= maximum:
            return None

        return (
            f"匹配 {keywords} 的记忆数量过多："
            f"{actual} > {maximum}"
        )


    # 某类记忆必须精确出现 N 条
    if check_type == "count_exact":

        keywords = check["keywords"]
        expected = check["count"]

        actual = count_matching(
            memories,
            keywords
        )

        if actual == expected:
            return None

        return (
            f"匹配 {keywords} 的记忆数量错误："
            f"实际 {actual}，预期 {expected}"
        )


    # 长期记忆完全不能发生变化
    if check_type == "unchanged":

        if memories == before_memories:
            return None

        return (
            "长期记忆发生了不应发生的变化"
        )


    return (
        f"未知检查类型: "
        f"{check_type}"
    )


# =========================================================
# Held-out Test Cases
#
# 注意：
# 这些表达不要拿去继续改 Prompt。
# 一旦根据某条 FAIL 修改了系统，
# 那条 case 就应该移到 regression/dev set。
# =========================================================

HELDOUT_CASES = [

    # -----------------------------------------------------
    # ADD
    # -----------------------------------------------------

    {
        "name": "add_korean_learning",
        "category": "ADD",

        "base_memories": [
            "联系人住在南京",
            "联系人喜欢看科幻电影"
        ],

        "working_messages": [],

        "input":
            "这阵子我开始学韩语了",

        "checks": [
            {
                "type": "exists_all",
                "keywords": ["韩语"]
            }
        ]
    },


    {
        "name": "add_exchange_goal",
        "category": "ADD",

        "base_memories": [
            "联系人住在成都"
        ],

        "working_messages": [],

        "input":
            "我准备明年申请学校的交换项目",

        "checks": [
            {
                "type": "exists_any_group",
                "groups": [
                    ["交换", "申请"],
                    ["交换", "项目"],
                    ["交换生"]
                ]
            }
        ]
    },


    {
        "name": "add_swimming_habit",
        "category": "ADD",

        "base_memories": [
            "联系人喜欢摄影"
        ],

        "working_messages": [],

        "input":
            "最近固定每周去游两次泳",

        "checks": [
            {
                "type": "exists_all",
                "keywords": ["游", "泳"]
            }
        ]
    },


    # -----------------------------------------------------
    # Duplicate
    # -----------------------------------------------------

    {
        "name": "duplicate_swimming_habit",
        "category": "Duplicate",

        "base_memories": [
            "联系人每周游泳三次",
            "联系人喜欢摄影"
        ],

        "working_messages": [],

        "input":
            "我基本一周会去游三次泳",

        "checks": [
            {
                "type": "count_max",
                "keywords": ["游", "泳"],
                "max": 1
            }
        ]
    },


    {
        "name": "duplicate_korean_learning",
        "category": "Duplicate",

        "base_memories": [
            "联系人正在学习韩语",
            "联系人住在苏州"
        ],

        "working_messages": [],

        "input":
            "韩语我最近一直在学",

        "checks": [
            {
                "type": "count_max",
                "keywords": ["韩语"],
                "max": 1
            }
        ]
    },


    # -----------------------------------------------------
    # UPDATE
    # -----------------------------------------------------

    {
        "name": "update_city",
        "category": "UPDATE",

        "base_memories": [
            "联系人住在南京",
            "联系人喜欢羽毛球"
        ],

        "working_messages": [],

        "input":
            "我已经搬去苏州长期住了",

        "checks": [
            {
                "type": "exact_absent",
                "memory": "联系人住在南京"
            },
            {
                "type": "exists_all",
                "keywords": ["苏州"]
            }
        ]
    },


    {
        "name": "update_swimming_frequency",
        "category": "UPDATE",

        "base_memories": [
            "联系人每周游泳三次",
            "联系人喜欢悬疑小说"
        ],

        "working_messages": [],

        "input":
            "最近没那么勤了，游泳现在改成一周一次",

        "checks": [
            {
                "type": "exact_absent",
                "memory": "联系人每周游泳三次"
            },

            {
                "type": "exists_any_group",
                "groups": [
                    ["游", "泳", "一次"],
                    ["游泳", "每周", "1"]
                ]
            }
        ]
    },


    # -----------------------------------------------------
    # DELETE
    # -----------------------------------------------------

    {
        "name": "delete_guitar_learning",
        "category": "DELETE",

        "base_memories": [
            "联系人正在学习吉他",
            "联系人喜欢爵士乐"
        ],

        "working_messages": [],

        "input":
            "吉他我决定先不学了",

        "checks": [
            {
                "type": "exact_absent",
                "memory": "联系人正在学习吉他"
            }
        ]
    },


    {
        "name": "delete_exam_plan",
        "category": "DELETE",

        "base_memories": [
            "联系人正在准备教师资格证考试",
            "联系人住在武汉"
        ],

        "working_messages": [],

        "input":
            "教资我不准备了",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在准备教师资格证考试"
            }
        ]
    },


    # -----------------------------------------------------
    # Worthiness
    # -----------------------------------------------------

    {
        "name": "worthiness_exam_plus_queue",
        "category": "Worthiness",

        "base_memories": [
            "联系人喜欢羽毛球"
        ],

        "working_messages": [],

        "input":
            "最近开始准备教资考试，我刚才正在便利店排队结账",

        "checks": [
            {
                "type": "exists_any_group",
                "groups": [
                    ["教资"],
                    ["教师资格证"]
                ]
            },

            {
                "type": "not_exists_any_group",
                "groups": [
                    ["便利店"],
                    ["排队"],
                    ["结账"]
                ]
            }
        ]
    },


    {
        "name": "worthiness_transient_only",
        "category": "Worthiness",

        "base_memories": [
            "联系人喜欢科幻电影",
            "联系人住在重庆"
        ],

        "working_messages": [],

        "input":
            "我现在下楼取个快递",

        "checks": [
            {
                "type": "unchanged"
            }
        ]
    },


    # -----------------------------------------------------
    # Multi-action
    # -----------------------------------------------------

    {
        "name": "multi_switch_language",
        "category": "Multi-action",

        "base_memories": [
            "联系人正在学习韩语",
            "联系人喜欢摄影"
        ],

        "working_messages": [],

        "input":
            "韩语我先停了，最近改学西班牙语",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在学习韩语"
            },

            {
                "type": "exists_all",
                "keywords": [
                    "西班牙语"
                ]
            }
        ]
    },


    {
        "name": "multi_two_new_persistent_facts",
        "category": "Multi-action",

        "base_memories": [
            "联系人住在青岛"
        ],

        "working_messages": [],

        "input":
            "最近开始学法语，也养成了每周游两次泳的习惯",

        "checks": [
            {
                "type": "exists_all",
                "keywords": ["法语"]
            },

            {
                "type": "exists_all",
                "keywords": ["游", "泳"]
            }
        ]
    },


    {
        "name":
            "multi_persistent_plus_transient",

        "category": "Multi-action",

        "base_memories": [
            "联系人喜欢纪录片"
        ],

        "working_messages": [],

        "input":
            "我最近开始系统学摄影了，现在正在咖啡店等朋友",

        "checks": [
            {
                "type": "exists_all",
                "keywords": ["摄影"]
            },

            {
                "type": "not_exists_any_group",
                "groups": [
                    ["咖啡店"],
                    ["等朋友"]
                ]
            }
        ]
    },


    # -----------------------------------------------------
    # Consistency
    # -----------------------------------------------------

    {
        "name": "consistency_exam_friend",
        "category": "Consistency",

        "base_memories": [
            "联系人正在准备考研",
            "联系人的朋友阿杰每天陪联系人复习考研英语",
            "联系人喜欢篮球"
        ],

        "working_messages": [],

        "input":
            "我决定不考研了",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在准备考研"
            },

            {
                "type": "exact_absent",
                "memory":
                    "联系人的朋友阿杰每天陪联系人复习考研英语"
            }
        ]
    },


    {
        "name": "consistency_language_teacher",
        "category": "Consistency",

        "base_memories": [
            "联系人正在学习韩语",
            "联系人的朋友小林正在教联系人韩语口语",
            "联系人喜欢跑步"
        ],

        "working_messages": [],

        "input":
            "韩语我不继续学了",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在学习韩语"
            },

            {
                "type": "exact_absent",
                "memory":
                    "联系人的朋友小林正在教联系人韩语口语"
            }
        ]
    },


    # -----------------------------------------------------
    # Working Memory
    # -----------------------------------------------------

    {
        "name":
            "working_memory_reference_photography",

        "category":
            "Working Memory",

        "base_memories": [
            "联系人正在学习摄影",
            "联系人住在厦门"
        ],

        "working_messages": [
            {
                "role": "user",
                "content":
                    "最近摄影课作业有点多"
            },

            {
                "role": "assistant",
                "content":
                    "你最近确实花了不少时间在摄影上。"
            }
        ],

        "input":
            "这个我准备先停一阵",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在学习摄影"
            }
        ]
    },


    {
        "name":
            "working_memory_reference_rag_project",

        "category":
            "Working Memory",

        "base_memories": [
            "联系人正在做一个 RAG 项目",
            "联系人喜欢听摇滚乐"
        ],

        "working_messages": [
            {
                "role": "user",
                "content":
                    "RAG 那个项目这两天调检索调得很烦"
            },

            {
                "role": "assistant",
                "content":
                    "那个项目目前主要卡在检索阶段。"
            }
        ],

        "input":
            "这个项目我准备先搁置",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在做一个 RAG 项目"
            }
        ]
    },


    # -----------------------------------------------------
    # No-op
    # -----------------------------------------------------

    {
        "name": "no_op_knowledge_question",
        "category": "No-op",

        "base_memories": [
            "联系人住在西安",
            "联系人喜欢羽毛球"
        ],

        "working_messages": [],

        "input":
            "为什么天空看起来是蓝色的？",

        "checks": [
            {
                "type": "unchanged"
            }
        ]
    },


    # -----------------------------------------------------
    # 综合泛化
    # -----------------------------------------------------

    {
        "name":
            "mixed_delete_add_transient",

        "category":
            "Generalization",

        "base_memories": [
            "联系人正在学习吉他",
            "联系人喜欢爵士乐"
        ],

        "working_messages": [],

        "input":
            "吉他先不学了，我最近开始学钢琴，刚好现在在地铁站等车",

        "checks": [
            {
                "type": "exact_absent",
                "memory":
                    "联系人正在学习吉他"
            },

            {
                "type": "exists_all",
                "keywords": [
                    "钢琴"
                ]
            },

            {
                "type": "not_exists_any_group",
                "groups": [
                    ["地铁站"],
                    ["等车"]
                ]
            }
        ]
    }
]


# =========================================================
# 单个 case 执行
# =========================================================

def evaluate_case(case):

    reset_case_state(
        case["base_memories"],
        case.get(
            "working_messages",
            []
        )
    )

    before_memories = get_target_memories(
        memory_search
    )

    execution = run_message(
        case["input"]
    )

    if execution["error"] is not None:

        sentinel_clean = sentinel_is_unchanged(
            memory_search
        )

        return {
            "name":
                case["name"],

            "category":
                case["category"],

            "passed":
                False,

            "status":
                "ERROR",

            "errors":
                ["运行时异常"],

            "before_memories":
                before_memories,

            "final_memories":
                get_target_memories(memory_search),

            "sentinel_unchanged":
                sentinel_clean,

            "sentinel_memories":
                get_sentinel_memories(memory_search),

            "log":
                execution["log"],

            "exception":
                execution["traceback"],

            "environment_error":
                is_environment_error(
                    execution["error"]
                )
        }

    final_memories = get_target_memories(
        memory_search
    )

    errors = []

    for check in case["checks"]:

        error = check_condition(
            check,
            final_memories,
            before_memories
        )

        if error is not None:
            errors.append(
                error
            )

    sentinel_clean = sentinel_is_unchanged(
        memory_search
    )

    if not sentinel_clean:
        errors.append(
            "检测到跨 sender 长期记忆污染"
        )

    passed = len(errors) == 0

    return {
        "name":
            case["name"],

        "category":
            case["category"],

        "passed":
            passed,

        "status":
            "PASS" if passed else "FAIL",

        "errors":
            errors,

        "before_memories":
            before_memories,

        "final_memories":
            final_memories,

        "sentinel_unchanged":
            sentinel_clean,

        "sentinel_memories":
            get_sentinel_memories(memory_search),

        "log":
            execution["log"],

        "exception":
            None,

        "environment_error":
            False
    }


# =========================================================
# 跑全部 held-out
# =========================================================

def run_heldout_evaluation():

    results = []

    total = len(
        HELDOUT_CASES
    )

    for index, case in enumerate(
        HELDOUT_CASES,
        start=1
    ):

        print(
            f"[{index:02d}/{total}] "
            f"Running "
            f"{case['name']}..."
        )

        result = evaluate_case(
            case
        )

        results.append(
            result
        )

    return results


# =========================================================
# 输出结果
# =========================================================

def print_results(results):

    print()

    print("=" * 72)
    print(
        "MEMORY AGENT HELD-OUT EVALUATION"
    )
    print("=" * 72)

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

        if result["environment_error"]:
            environment_error_count += 1

        print(
            f"{result['name']:<42}"
            f"{status:<7}"
            f"[{result['category']}]"
        )

        category = (
            result["category"]
        )

        if category not in category_stats:

            category_stats[category] = {
                "passed": 0,
                "total": 0
            }

        category_stats[
            category
        ]["total"] += 1

        if passed:

            category_stats[
                category
            ]["passed"] += 1


        # 只打印 FAIL 细节
        if not passed:

            print(
                "  Errors:"
            )

            for error in result["errors"]:

                print(
                    "   -",
                    error
                )

            print(
                "  Before:"
            )

            for memory in result[
                "before_memories"
            ]:

                print(
                    "   -",
                    memory
                )

            print(
                "  Final:"
            )

            for memory in result[
                "final_memories"
            ]:

                print(
                    "   -",
                    memory
                )


            if result["exception"]:

                print(
                    "\n  Environment/Integration Error:"
                    if result["environment_error"]
                    else "\n  Program Error:"
                )

                print(
                    result["exception"]
                )

            if not result["sentinel_unchanged"]:
                print(
                    "\n  Sentinel Pollution:"
                )
                print(
                    result["sentinel_memories"]
                )


            if result["log"]:

                print(
                    "\n  Internal Log:"
                )

                print(
                    result["log"]
                )

            print(
                "-" * 72
            )


    total = len(results)

    pass_rate = (
        passed_count / total
        if total
        else 0.0
    )


    print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Passed: "
        f"{passed_count}/{total}"
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
        f"Held-out Pass Rate: "
        f"{pass_rate:.2%}"
    )


    print(
        "\nBy Category:"
    )

    for category, stats in (
        category_stats.items()
    ):

        rate = (
            stats["passed"]
            / stats["total"]
        )

        print(
            f"  {category:<20}"
            f"{stats['passed']}/"
            f"{stats['total']} "
            f"({rate:.2%})"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    try:

        results = (
            run_heldout_evaluation()
        )

        print_results(
            results
        )

    finally:

        memory_search.reset_memories(
            ORIGINAL_MEMORIES
        )
        main.reset_working_memories()

        print(
            "\nOriginal memories restored."
        )
