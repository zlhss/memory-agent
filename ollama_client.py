import json
import time
from datetime import datetime

import requests
from memory_search import retrieve

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

_ollama_timing_records = []
_ollama_prompt_records = []


def reset_ollama_timings():
    _ollama_timing_records.clear()
    _ollama_prompt_records.clear()


def get_ollama_timings():
    return [record.copy() for record in _ollama_timing_records]


def get_ollama_profiled_prompts():
    return [record.copy() for record in _ollama_prompt_records]


def get_ollama_timing_summary(total_request_elapsed_seconds):
    stage_totals = {}

    for record in _ollama_timing_records:
        stage = record["stage"]
        totals = stage_totals.setdefault(
            stage,
            {
                "stage": stage,
                "call_count": 0,
                "elapsed_seconds": 0.0,
            }
        )
        totals["call_count"] += 1
        totals["elapsed_seconds"] += record["elapsed_seconds"]

    stage_timings = []

    for totals in stage_totals.values():
        stage_timings.append({
            "stage": totals["stage"],
            "call_count": totals["call_count"],
            "elapsed_seconds": round(
                totals["elapsed_seconds"],
                3
            )
        })

    total_llm_elapsed = sum(
        record["elapsed_seconds"]
        for record in _ollama_timing_records
    )

    return {
        "total_llm_calls": len(_ollama_timing_records),
        "stage_timings": stage_timings,
        "total_llm_elapsed_seconds": round(
            total_llm_elapsed,
            3
        ),
        "total_request_elapsed_seconds": round(
            total_request_elapsed_seconds,
            3
        )
    }


def _record_ollama_timing(
    stage,
    prompt,
    started_at,
    started_perf,
    status,
    response_chars,
    thinking_chars,
    ollama_metrics
):
    record = {
        "stage": stage,
        "prompt_chars": len(prompt),
        "started_at": started_at,
        "elapsed_seconds": round(
            time.perf_counter() - started_perf,
            3
        ),
        "status": status,
        "thinking_chars": thinking_chars,
        "response_chars": response_chars,
        "load_duration": ollama_metrics.get("load_duration"),
        "prompt_eval_count": ollama_metrics.get(
            "prompt_eval_count"
        ),
        "prompt_eval_duration": ollama_metrics.get(
            "prompt_eval_duration"
        ),
        "eval_count": ollama_metrics.get("eval_count"),
        "eval_duration": ollama_metrics.get("eval_duration"),
        "total_duration": ollama_metrics.get("total_duration")
    }
    _ollama_timing_records.append(record)
    print(
        "OLLAMA_TIMING:",
        json.dumps(record, ensure_ascii=False)
    )


def call_ollama(prompt, stage="Unspecified"):
    started_at = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )
    started_perf = time.perf_counter()
    status = "error"
    response_chars = 0
    thinking_chars = 0
    ollama_metrics = {}

    _ollama_prompt_records.append({
        "stage": stage,
        "prompt": prompt
    })

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        response_text = data.get("response", "").strip()
        thinking_text = data.get("thinking", "").strip()
        response_chars = len(response_text)
        thinking_chars = len(thinking_text)
        ollama_metrics = {
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get(
                "prompt_eval_duration"
            ),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration")
        }

        raw_text = response_text

        # 某些 thinking 模型可能把结果放进 thinking
        if not raw_text:
            raw_text = thinking_text

        if not raw_text:
            print("\nOLLAMA RAW RESPONSE:")
            print(data)
            raise RuntimeError("Ollama 没有返回可解析内容")

        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Ollama 返回的内容不是有效 JSON。"
            ) from error

        if not isinstance(decision, dict):
            raise ValueError(
                "Ollama JSON 输出必须是 object。"
            )

        status = "success"
        return decision
    except requests.exceptions.Timeout:
        status = "timeout"
        raise
    finally:
        _record_ollama_timing(
            stage=stage,
            prompt=prompt,
            started_at=started_at,
            started_perf=started_perf,
            status=status,
            response_chars=response_chars,
            thinking_chars=thinking_chars,
            ollama_metrics=ollama_metrics
        )


def resolve_contextual_query(user_text, working_messages):
    """Resolve conversational references before long-term retrieval."""

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    if not isinstance(working_messages, list):
        raise ValueError("working_messages must be a list.")

    if not working_messages:
        return user_text.strip()

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in working_messages
    )

    prompt = f"""
你是一个上下文检索查询解析器。

【最近对话】
{history_text}

【当前联系人发来的消息】
{user_text}

你的任务是生成一条用于检索该联系人长期记忆的 query。

规则：
1. 当前消息中的“我”指当前联系人。
2. 结合最近对话解析“这个”“那个”“它”“这件事”“这个项目”
   等指代、省略和承接表达。
3. 如果最近对话中只有一个明确 referent，query 必须用明确主题替换指代，
   同时保留当前消息表达的变化、停止、继续、替代等意图。
4. 不得引入最近对话和当前消息中从未出现的新主题、实体或活动。
5. user 消息是联系人说的话；assistant 消息只用于理解对话承接，
   不能单独作为联系人事实。
6. 如果没有需要解析的指代，保持当前消息的原始语义，不要扩写事实。
7. 如果存在多个合理 referent 且无法可靠区分，不要猜测或选择其中一个；
   保留原指代表达。
8. 这里只生成检索 query，不决定 ADD、UPDATE 或 DELETE，
   也不生成长期记忆。

示例：

最近对话：
user: 最近摄影课作业有点多
assistant: 你最近确实花了不少时间在摄影上。

当前消息：
这个我准备先停一阵

正确输出：

{{
    "query": "联系人准备暂停学习摄影"
}}

只输出 JSON：

{{
    "query": "用于长期记忆检索的明确查询"
}}
"""

    result = call_ollama(
        prompt,
        stage="Contextual Query Resolver"
    )
    query = result.get("query")

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Contextual Query Resolver requires a non-empty query."
        )

    return query.strip()


def build_memory_prompt(
    sender_id,
    user_text,
    relevant_memories,
    working_messages
):
    """
    构造当前联系人专属的 Memory Manager Prompt。
    """

    memories_text = "\n".join(
        f"{index}. {memory}"
        for index, memory in enumerate(relevant_memories)
    )
    history_text = ""

    for message in working_messages:
        history_text += (
            f"{message['role']}: {message['content']}\n"
        )

    prompt = f"""
你是一个 Personalized Conversational Agent 的长期记忆管理器。

【当前联系人 sender_id】
{sender_id}

重要语义：
- 当前消息来自上述联系人，不是 Agent 拥有者说的话。
- 当前消息里的“我”默认指当前联系人。
- 现有长期记忆和新生成的长期记忆都只描述当前联系人。
- 每条新 memory 必须统一使用“联系人……”表述，不能使用“用户……”。
- sender_id 已经负责区分联系人，因此 memory 中不要重复写 sender_id 或联系人名字。
- 最近对话中的 user 表示当前联系人，assistant 表示 Agent 给该联系人的回复。

【最近对话】
{history_text if history_text else "无最近对话"}

【现有相关长期记忆】
{memories_text if memories_text else "无相关长期记忆"}

【当前联系人发来的消息】
{user_text}

你的任务是判断：
当前联系人发来的消息是否需要对该联系人的长期记忆执行 ADD、UPDATE 或 DELETE。

一条联系人消息可能同时影响多个独立的长期事实。
如果存在多个变化，必须分别生成多个 action。

可选操作：

ADD：
联系人提供了新的、值得长期保存的信息。

UPDATE：
已有长期记忆仍有保留价值，
但联系人的新信息使这条记忆需要修改。

DELETE：
已有长期记忆已经不应该继续作为有效长期记忆保存。

如果没有任何需要执行的长期记忆操作，
返回空 actions。

只允许输出 JSON。

输出格式：

{{
    "actions": [
        {{
            "action": "ADD|UPDATE|DELETE",
            "target_index": "UPDATE/DELETE 对应的相关长期记忆 index；ADD 时为 null",
            "memory": "ADD 或 UPDATE 后的新记忆；DELETE 时为 null"
        }}
    ]
}}

规则：

1. 每个 action 只能处理一个原子化事实。

2. 一条联系人消息可能产生 0 个、1 个或多个 action。
   不要因为已经生成一个 action，
   就忽略联系人消息中其他明确的长期记忆变化。

3. ADD 时：
   - target_index 必须为 null。
   - memory 只包含联系人当前消息中新出现的长期信息。
   - 不要把已有记忆复制、拼接到新的 memory 中。

4. UPDATE 时：
   - target_index 必须是现有相关长期记忆列表中的有效 index。
   - memory 只表达该 index 对应记忆更新后的那个事实。
   - 不要把多个事实合并进同一条 memory。

5. DELETE 时：
   - target_index 必须是需要删除的相关长期记忆 index。
   - memory 必须为 null。

6. 如果联系人消息没有任何值得长期保存或修改的信息：

{{
    "actions": []
}}

7. 每条 memory 必须是独立、简洁、原子化的长期事实。

8. 联系人新增一个信息，不代表已有信息自动失效。

例如：

已有：
- 联系人正在学习日语

联系人：
“我最近也开始学法语了”

应该 ADD：

{{
    "actions": [
        {{
            "action": "ADD",
            "target_index": null,
            "memory": "联系人最近开始学习法语"
        }}
    ]
}}

不要 UPDATE 掉“联系人正在学习日语”。

9. 只有联系人明确表示已有状态发生改变、替代、暂停、停止、
   取消或不再成立时，才考虑 UPDATE 或 DELETE。

   特别重要：

   如果【现有相关长期记忆】中已经存在某个持续状态，
   而联系人当前消息明确表示这个同一状态发生了变化，
   必须优先修改已有记忆，
   不能把新的状态作为一个独立事实 ADD。

   判断原则：

   旧状态
   +
   对同一状态的停止 / 暂停 / 替换 / 改变
   =
   UPDATE 或 DELETE

   而不是 ADD。

   例如：

   已有：
   0. 联系人正在学习摄影

   联系人：
   “这个我准备先停一阵”

   如果最近对话已经明确“这个”指摄影，
   合理输出：

   {{
       "actions": [
           {{
               "action": "UPDATE",
               "target_index": 0,
               "memory": "联系人暂时停止学习摄影"
           }}
       ]
   }}

   不应该输出：

   {{
       "actions": [
           {{
               "action": "ADD",
               "target_index": null,
               "memory": "联系人暂时停止学习摄影"
           }}
       ]
   }}

   因为“正在学习摄影”和“暂时停止学习摄影”
   描述的是同一个持续状态在不同时间的当前值，
   两者不能同时作为两个有效当前状态存在。

   同样适用于：
   - 当前居住地发生变化
   - 当前工作发生变化
   - 当前学习内容发生变化
   - 当前项目停止或暂停
   - 当前计划取消
   - 当前长期活动停止

   如果新消息明确只是结束已有状态，
   并且结束后的状态本身没有未来保存价值，
   可以 DELETE。

   如果结束后的新状态本身仍有未来价值，
   优先 UPDATE 为新的当前状态。

   Duplicate Gate 不能用来代替 UPDATE / DELETE。
   如果新事实与旧事实属于同一个状态的变化，
   应在 Memory Manager 阶段直接修改旧状态。

10. 如果一个当前状态结束了，
    但其中包含的历史事实未来仍可能有价值，
    可以优先 UPDATE 成历史事实，而不是直接删除。

例如：

已有：
- 联系人的朋友小明在教联系人日语，内容与动漫相关

联系人：
“小明以后不用再教我日语了”

可以 UPDATE 为：

{{
    "actions": [
        {{
            "action": "UPDATE",
            "target_index": 0,
            "memory": "联系人的朋友小明曾教联系人日语，内容与动漫相关"
        }}
    ]
}}

11. 联系人提出普通问题时，
    如果问题本身没有提供新的长期联系人事实，
    不要修改长期记忆。

例如：

已有：
- 联系人喜欢听丸之内虐待狂

联系人：
“我喜欢听什么？”

正确：

{{
    "actions": []
}}

不要重新 ADD 已有记忆。

12. 现有相关长期记忆只用于：
    - 判断新信息是否与旧信息有关
    - 判断是否需要 UPDATE
    - 判断是否需要 DELETE

不要因为某条旧记忆被检索出来，
就重新把它 ADD 一次。

13. 最近对话只用于理解联系人当前消息中的上下文和指代，
    例如：
    “他”
    “这个”
    “之前那个”
    “算了”
    “不学了”

    当当前消息使用“这个”“那个”“它”“这件事”“那件事”
    “这个项目”等省略或指代表达时：
    - 必须优先从最近对话和现有相关长期记忆中寻找 referent。
    - 不允许生成上下文中从未出现过的新主题、新实体或新活动。
    - 如果只有一个明确 referent，必须沿用该 referent 所指的事实。
    - 如果存在多个合理 referent 且无法可靠区分，返回空 actions，不要猜测。
    - UPDATE / DELETE 的 target_index 必须直接指向现有相关长期记忆。
    - 如果最近对话中的 referent 没有对应到任何相关长期记忆，
      返回空 actions，不允许自由生成 target。

14. 不要仅仅因为某个事实出现在最近对话中，
    就把它写入长期记忆。

15. assistant 在最近对话中的回复不能被当作联系人事实。

16. 最终产生哪些 action，
    必须以联系人当前消息所表达的信息和意图为依据。

17. 一次性任务、普通知识问题、临时命令和当前操作请求，
    默认不写入长期记忆。

例如：

联系人：
“帮我写一个冒泡排序”

正确：

{{
    "actions": []
}}

错误行为：
ADD “联系人请求编写冒泡排序算法”。

18. 长期记忆主要保存未来对话中可能持续有用的信息，例如：
    - 稳定个人事实
    - 联系人偏好
    - 长期目标
    - 持续中的学习或工作状态
    - 人际关系
    - 联系人明确要求长期记住的信息

19. 如果联系人当前消息同时修改多个长期事实，
    必须分别输出多个 action。

例如：

已有：
- 联系人正在学习日语
- 联系人的朋友小明在教联系人日语，内容与动漫相关

联系人：
“我现在不学日语了，小明也不用再教我了”

合理输出：

{{
    "actions": [
        {{
            "action": "DELETE",
            "target_index": 0,
            "memory": null
        }},
        {{
            "action": "UPDATE",
            "target_index": 1,
            "memory": "联系人的朋友小明曾教联系人日语，内容与动漫相关"
        }}
    ]
}}

20. 不要输出 actions 之外的解释、Markdown 或其他文字。

最终只能输出 JSON。

21. 联系人当前消息可能同时包含多个彼此独立的新事实。
    必须先将它们拆成原子事实，并分别判断是否需要长期保存。

22. 同一句话中的不同事实可以产生不同结果：
    某些事实可以 ADD，
    某些事实可以 UPDATE / DELETE，
    某些临时事实可以完全忽略。
联系人：
“我最近开始学日语，同时也开始坚持跑步了”

合理输出：

{{
    "actions": [
        {{
            "action": "ADD",
            "target_index": null,
            "memory": "联系人正在学习日语"
        }},
        {{
            "action": "ADD",
            "target_index": null,
            "memory": "联系人开始坚持跑步"
        }}
    ]
}}
23. 必须区分“长期/持续状态”和“瞬时状态”。

适合长期保存：
- 联系人正在学习日语
- 联系人每周跑步三次
- 联系人最近开始健身
- 联系人正在准备研究生申请

默认不保存的瞬时状态：
- 联系人现在正在吃饭
- 联系人正在操场跑步
- 联系人正在坐公交
- 联系人现在很困
- 联系人当前正在写一段代码

仅仅出现“正在”不代表值得长期保存。
关键是判断这个事实在未来对话中是否仍可能持续有用。

24. 一条消息包含多个事实时，必须分别判断每个事实是否值得长期保存。
不能因为其中一个事实值得保存，就把其他临时事实也一起存入长期记忆。

联系人：
“我在学日语，我现在正在操场跑步”

合理输出：

{{
    "actions": [
        {{
            "action": "ADD",
            "target_index": null,
            "memory": "联系人正在学习日语"
        }}
    ]
}}

“联系人正在操场跑步”属于当前瞬时状态，不应写入长期记忆。

"""

    return prompt


def decide_memory_actions(
    sender_id,
    user_text,
    relevant_memories,
    working_messages
):
    """
    根据当前联系人的新消息和相关旧记忆，
    让 LLM 决定 ADD / UPDATE / DELETE。
    """

    prompt = build_memory_prompt(
        sender_id,
        user_text,
        relevant_memories,
        working_messages
    )

    result = call_ollama(
        prompt,
        stage="Memory Manager"
    )

    actions = result.get("actions")

    if not isinstance(actions, list):
        raise ValueError(
            "Memory Manager requires an actions list."
        )

    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(
                f"Memory action {index} must be an object."
            )

        action_type = action.get("action")
        target_index = action.get("target_index")
        memory = action.get("memory")

        if action_type not in {"ADD", "UPDATE", "DELETE"}:
            raise ValueError(
                f"Memory action {index} has an invalid action."
            )

        if action_type == "ADD":
            if target_index is not None:
                raise ValueError(
                    f"Memory ADD action {index} requires null "
                    "target_index."
                )
            if not isinstance(memory, str) or not memory.strip():
                raise ValueError(
                    f"Memory ADD action {index} requires memory."
                )

        if action_type == "UPDATE":
            if type(target_index) is not int:
                raise ValueError(
                    f"Memory UPDATE action {index} requires an integer "
                    "target_index."
                )
            if not 0 <= target_index < len(relevant_memories):
                raise ValueError(
                    f"Memory UPDATE action {index} has an out-of-range "
                    "target_index."
                )
            if not isinstance(memory, str) or not memory.strip():
                raise ValueError(
                    f"Memory UPDATE action {index} requires memory."
                )

        if action_type == "DELETE":
            if type(target_index) is not int:
                raise ValueError(
                    f"Memory DELETE action {index} requires an integer "
                    "target_index."
                )
            if not 0 <= target_index < len(relevant_memories):
                raise ValueError(
                    f"Memory DELETE action {index} has an out-of-range "
                    "target_index."
                )
            if memory is not None:
                raise ValueError(
                    f"Memory DELETE action {index} requires null memory."
                )

    return actions


def build_action_coverage_prompt(
    user_text,
    relevant_memories,
    existing_actions
):
    """Build the narrowly scoped Action Coverage Checker prompt."""

    memories_text = "\n".join(
        f"{index}. {memory}"
        for index, memory in enumerate(relevant_memories)
    )
    actions_text = json.dumps(
        existing_actions,
        ensure_ascii=False,
        indent=2
    )

    return f"""
你是长期记忆 Action Coverage Checker。

你的唯一职责是检查：
当前消息是否明确让某条【相关长期记忆】失效、停止、取消、暂停、
被替换或发生变化，但【已有 actions】没有覆盖这条记忆。

【当前消息】
{user_text}

【相关长期记忆】
{memories_text}

【已有 actions】
{actions_text}

只允许输出 JSON：

{{
    "missing_actions": [
        {{
            "action": "UPDATE|DELETE",
            "target_index": 0,
            "memory": "UPDATE 时的新记忆；DELETE 时为 null"
        }}
    ]
}}

如果没有遗漏：

{{
    "missing_actions": []
}}

严格规则：
1. 只能针对【相关长期记忆】中已有的 index。
2. 只能补充 UPDATE 或 DELETE，绝不允许生成 ADD。
3. 如果【已有 actions】已经对某个 target_index 执行 UPDATE 或 DELETE，
   不得再为该 index 生成 action。
4. 只有当前消息明确改变、停止、取消、暂停、替换或否定某条现有记忆时，
   才能补充 action。
5. 如果无法明确判断某条已有记忆是否被改变，返回空 missing_actions。
6. 不能仅根据关键词、主题相似或常识推测记忆发生变化。
7. 普通知识问题、临时行为和一次性请求不产生 missing_actions。
8. UPDATE 的 memory 只能表达该 index 更新后的同一事实；
   DELETE 的 memory 必须为 null。
9. target_index 必须直接映射到【相关长期记忆】中的真实条目。
10. 不允许自由生成 target 字符串，也不负责判断或补充任何新事实。
11. 不要输出解释、Markdown 或其他字段。
"""


def check_action_coverage(
    user_text,
    relevant_memories,
    existing_actions
):
    """Return uncovered UPDATE/DELETE actions for relevant memories."""

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    if not isinstance(relevant_memories, list):
        raise ValueError("relevant_memories must be a list.")

    if not isinstance(existing_actions, list):
        raise ValueError("existing_actions must be a list.")

    prompt = build_action_coverage_prompt(
        user_text,
        relevant_memories,
        existing_actions
    )
    result = call_ollama(
        prompt,
        stage="Action Coverage Checker"
    )
    missing_actions = result.get("missing_actions")

    if not isinstance(missing_actions, list):
        raise ValueError(
            "Action Coverage Checker requires a missing_actions list."
        )

    covered_indices = {
        action.get("target_index")
        for action in existing_actions
        if isinstance(action, dict)
        and action.get("action") in {"UPDATE", "DELETE"}
        and type(action.get("target_index")) is int
    }
    valid_actions = []
    used_indices = set(covered_indices)

    for index, action in enumerate(missing_actions):
        if not isinstance(action, dict):
            raise ValueError(
                f"Coverage action {index} must be an object."
            )

        action_type = action.get("action")
        target_index = action.get("target_index")
        memory = action.get("memory")

        if action_type not in {"UPDATE", "DELETE"}:
            raise ValueError(
                f"Coverage action {index} must be UPDATE or DELETE."
            )

        if type(target_index) is not int:
            raise ValueError(
                f"Coverage action {index} requires an integer "
                "target_index."
            )

        if not 0 <= target_index < len(relevant_memories):
            raise ValueError(
                f"Coverage action {index} has an out-of-range "
                "target_index."
            )

        if action_type == "UPDATE":
            if not isinstance(memory, str) or not memory.strip():
                raise ValueError(
                    f"Coverage UPDATE action {index} requires memory."
                )

        if action_type == "DELETE" and memory is not None:
            raise ValueError(
                f"Coverage DELETE action {index} requires null memory."
            )

        if target_index in used_indices:
            continue

        valid_actions.append({
            "action": action_type,
            "target_index": target_index,
            "memory": memory.strip()
            if action_type == "UPDATE"
            else None
        })
        used_indices.add(target_index)

    return {
        "missing_actions": valid_actions
    }


def build_add_evidence_prompt(
    user_text,
    new_memory,
    working_messages
):
    """Build the narrowly scoped ADD Evidence Gate prompt."""

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in working_messages
    )

    return f"""
你是长期记忆 ADD Evidence Gate。

你的唯一职责是判断：
准备 ADD 的候选记忆，是否被联系人当前消息明确支持为
“关于联系人自身的长期事实”。

【最近对话】
{history_text if history_text else "无最近对话"}

【联系人当前消息】
{user_text}

【准备 ADD 的候选记忆】
{new_memory}

只允许输出 JSON：

{{
    "supported": true
}}

或：

{{
    "supported": false
}}

严格规则：
1. supported=true 必须同时满足：
   - 候选记忆描述联系人自身的事实、偏好、目标、关系、习惯，
     或持续中的学习、工作、项目等长期状态；
   - 这个事实能从联系人当前消息中直接得到。
2. 不允许依靠推测、常识、改写问题意图或 assistant 回复提供证据。
3. 普通知识问题、信息查询和一次性请求本身，
   不能被转写成长久联系人事实。
4. 问句如果同时明确陈述了联系人长期事实，
   该明确事实仍然可以 supported=true。
5. “联系人询问 X”“联系人想知道 X”“联系人请求 X”
   不能自动视为长期目标、偏好或 ongoing learning。
6. 最近对话中的 assistant 内容不能作为联系人事实来源。
7. 候选记忆必须由当前消息明确支持；证据不明确时返回 false。
8. 候选记忆本身必须以联系人为事实主体。
   如果候选只是复述问题中的世界知识命题、待解释现象或答案内容，
   而不是描述联系人自身，必须返回 false。
9. “为什么 X”“如何 X”“什么是 X”“是否 X”等问句只表达提问，
   不表示联系人确认 X 是自己的事实，也不表示联系人正在长期学习 X。
10. 只判断给定候选记忆，不生成、改写或补充其他记忆。
11. 不要输出解释、Markdown 或其他字段。

示例一：
当前消息：“为什么天空看起来是蓝色的？”
候选记忆：“联系人询问天空为什么是蓝色的原因”
输出：{{"supported": false}}

示例二：
当前消息：“帮我写一个冒泡排序”
候选记忆：“联系人想学习冒泡排序”
输出：{{"supported": false}}

示例三：
当前消息：“我最近开始学摄影了，为什么光圈会影响景深？”
候选记忆：“联系人最近开始学习摄影”
输出：{{"supported": true}}

同一条消息中，如果候选记忆只是“联系人询问光圈影响景深的原因”
或“光圈会影响景深”，输出：{{"supported": false}}
"""


def check_add_evidence(
    user_text,
    new_memory,
    working_messages
):
    """Check whether an ADD is grounded in an explicit contact fact."""

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    if not isinstance(new_memory, str) or not new_memory.strip():
        raise ValueError("new_memory must be a non-empty string.")

    if not isinstance(working_messages, list):
        raise ValueError("working_messages must be a list.")

    prompt = build_add_evidence_prompt(
        user_text,
        new_memory,
        working_messages
    )
    result = call_ollama(
        prompt,
        stage="ADD Evidence Gate"
    )
    supported = result.get("supported")

    if type(supported) is not bool:
        raise ValueError(
            "ADD Evidence Gate requires a boolean supported field."
        )

    return {
        "supported": supported
    }


def _format_memories(relevant_memories):
    return "\n".join(
        f"- {memory}"
        for memory in relevant_memories
    )


def _format_working_messages(working_messages):
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in working_messages
    )


def _validate_response_plan(result):
    if not isinstance(result, dict):
        raise ValueError("Response plan must be a JSON object.")

    intent = result.get("intent")
    points = result.get("points")
    tone = result.get("tone")

    if not isinstance(intent, str) or not intent.strip():
        raise ValueError(
            "Response plan requires a non-empty intent."
        )

    if not isinstance(points, list) or not points:
        raise ValueError(
            "Response plan requires a non-empty points list."
        )

    if not all(
        isinstance(point, str) and point.strip()
        for point in points
    ):
        raise ValueError(
            "Every response plan point must be a non-empty string."
        )

    if not isinstance(tone, str) or not tone.strip():
        raise ValueError(
            "Response plan requires a non-empty tone."
        )

    return {
        "intent": intent.strip(),
        "points": [
            point.strip()
            for point in points
        ],
        "tone": tone.strip()
    }


def plan_response(
    sender_id,
    user_text,
    relevant_memories,
    working_messages
):
    """Plan reply content without imitating the contact's style."""

    memories_text = _format_memories(
        relevant_memories
    )
    history_text = _format_working_messages(
        working_messages
    )

    prompt = f"""
你是 Personalized Conversational Agent 的 Response Planner。

【当前联系人 sender_id】
{sender_id}

【当前联系人发来的消息】
{user_text}

【该联系人的相关长期记忆】
{memories_text if memories_text else "无相关长期记忆"}

【与该联系人的最近对话】
{history_text if history_text else "无最近对话"}

只规划这次回复应该表达什么内容，不要撰写最终回复，
也不要模仿联系人的措辞、口头禅、标点或聊天风格。

要求：
1. intent 是本次回复的主要沟通目的。
2. points 是回复中需要表达的事实、回应或问题。
3. tone 只描述合适的基本沟通态度，例如关心、平静、明确；
   不描述或模仿联系人的个人语言风格。
4. 不得编造长期记忆中不存在的事实。
5. 当前消息里的“我”指当前联系人，不是 Agent 拥有者。

只输出 JSON：

{{
    "intent": "回复意图",
    "points": ["要点 1", "要点 2"],
    "tone": "基本沟通态度"
}}
"""

    result = call_ollama(
        prompt,
        stage="Planner"
    )

    return _validate_response_plan(result)


def generate_draft_reply(
    sender_id,
    user_text,
    relevant_memories,
    working_messages,
    plan
):
    """Generate a content-correct neutral draft from a validated plan."""

    plan = _validate_response_plan(plan)
    memories_text = _format_memories(
        relevant_memories
    )
    history_text = _format_working_messages(
        working_messages
    )
    plan_text = json.dumps(
        plan,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
你是 Personalized Conversational Agent 的 Draft Generator。

【当前联系人 sender_id】
{sender_id}

【当前联系人发来的消息】
{user_text}

【与该联系人的最近对话】
{history_text if history_text else "无最近对话"}

【该联系人的相关长期记忆】
{memories_text if memories_text else "无相关长期记忆"}

【Response Plan】
{plan_text}

请根据 Response Plan 写一条内容正确、自然、可以直接发送的普通回复。

要求：
1. 覆盖 plan 的 intent 和 points。
2. 不得编造长期记忆、最近对话或当前消息中没有的事实。
3. 不要刻意模仿联系人的语言风格。
4. 不要提及 Planner、长期记忆、Style Memory 或内部流程。
5. 当前消息里的“我”指当前联系人。

只输出 JSON：

{{
    "draft": "普通回复"
}}
"""

    result = call_ollama(
        prompt,
        stage="Draft"
    )
    draft = result.get("draft")

    if not isinstance(draft, str) or not draft.strip():
        raise ValueError(
            "Draft Generator requires a non-empty draft field."
        )

    return draft.strip()


def rewrite_reply_style(
    draft_reply,
    user_text,
    style_samples
):
    """Rewrite expression only, preserving the draft's core content."""

    if not isinstance(draft_reply, str) or not draft_reply.strip():
        raise ValueError("draft_reply must be a non-empty string.")

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    if not isinstance(style_samples, list):
        raise ValueError("style_samples must be a list.")

    if not style_samples:
        return draft_reply.strip()

    validated_samples = []

    for index, sample in enumerate(style_samples):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Style sample {index} must be an object."
            )

        incoming = sample.get("incoming")
        reply = sample.get("reply")
        score = sample.get("score")

        if not isinstance(incoming, str) or not incoming.strip():
            raise ValueError(
                f"Style sample {index} has invalid incoming."
            )

        if not isinstance(reply, str) or not reply.strip():
            raise ValueError(
                f"Style sample {index} has invalid reply."
            )

        if not isinstance(score, (int, float)):
            raise ValueError(
                f"Style sample {index} has invalid score."
            )

        validated_samples.append({
            "incoming": incoming.strip(),
            "reply": reply.strip(),
            "score": float(score)
        })

    samples_text = json.dumps(
        validated_samples,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
你是 Personalized Conversational Agent 的 Style Rewriter。

【当前联系人发来的消息】
{user_text}

【内容已经确定的 Draft Reply】
{draft_reply}

【同一联系人的相似历史 Style Samples】
{samples_text}

你的任务只是在不改变内容的前提下，参考历史 reply 的表达习惯，
改写 Draft Reply 的措辞、句长、语气词、标点和简洁程度。

严格要求：
1. 不得增加、删除或改变 Draft Reply 的核心事实。
2. 不得改变 Draft Reply 的沟通意图。
3. Style Sample 的 incoming/reply 只用于学习表达风格，
   不能把历史样本中的人物、事件或事实带入本次回复。
4. 不要照抄与当前内容无关的历史句子。
5. 不要提及 Style Memory、样本或内部流程。

只输出 JSON：

{{
    "reply": "个性化后的最终回复"
}}
"""

    result = call_ollama(
        prompt,
        stage="Style Rewriter"
    )
    reply = result.get("reply")

    if not isinstance(reply, str) or not reply.strip():
        raise ValueError(
            "Style Rewriter requires a non-empty reply field."
        )

    return reply.strip()


def generate_answer(user_text, relevant_memories,working_messages):
    memories_text = "\n".join(
        f"- {memory}"
        for memory in relevant_memories
    )
    
    history_text = ""

    for message in working_messages:
        history_text += (
            f"{message['role']}: {message['content']}\n"
        )

    prompt = f"""
你是一个带长期记忆的聊天助手。

【长期记忆】
{memories_text if memories_text else "无相关长期记忆"}

【最近对话】
{history_text if history_text else "无最近对话"}

【用户当前消息】
{user_text}

注意：
长期记忆与当前消息必须严格区分。
如果长期记忆为空，直接正常回答，不要声称用户之前说过当前消息。

请回答用户。
"""

    started_at = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )
    started_perf = time.perf_counter()
    status = "error"
    response_chars = 0
    thinking_chars = 0
    ollama_metrics = {}

    _ollama_prompt_records.append({
        "stage": "Legacy Answer",
        "prompt": prompt
    })

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        response_text = data.get("response", "").strip()
        thinking_text = data.get("thinking", "").strip()
        response_chars = len(response_text)
        thinking_chars = len(thinking_text)
        ollama_metrics = {
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get(
                "prompt_eval_duration"
            ),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration")
        }

        answer = response_text

        if not answer:
            answer = thinking_text

        status = "success"
        return answer
    except requests.exceptions.Timeout:
        status = "timeout"
        raise
    finally:
        _record_ollama_timing(
            stage="Legacy Answer",
            prompt=prompt,
            started_at=started_at,
            started_perf=started_perf,
            status=status,
            response_chars=response_chars,
            thinking_chars=thinking_chars,
            ollama_metrics=ollama_metrics
        )

def filter_relevant_memories(user_text, candidate_memories):
    if not candidate_memories:
        return []

    memories_text = "\n".join(
        f"{i}. {memory}"
        for i, memory in enumerate(candidate_memories)
    )

    prompt = f"""
你是一个长期记忆检索的相关性判断器。

用户当前消息：
{user_text}

候选记忆：
{memories_text}

判断哪些候选记忆与用户当前消息真正相关。

“相关”包括：
1. 可以帮助回答用户当前的问题。
2. 可以帮助理解用户当前提供的新信息。
3. 与用户当前信息存在更新、冲突或删除关系。
4. 当前消息明确替换某个属性的旧“当前状态”时，
   描述该属性旧值的记忆属于相关记忆。
   即使新旧值的词面不同、当前消息没有重复旧值，仍应保留。
   这类属性包括当前居住地、当前学习内容、当前工作、
   当前项目、当前计划以及其他互斥或被替代的持续状态。

判断更新关系时应比较“事实属性/状态槽位”是否相同，
不能只依赖新旧文本是否共享地点名、实体名或关键词。

不要因为某条记忆是所有候选中最接近的一条，就认为它相关。
如果所有候选记忆都与当前消息无关，应返回空列表。

只输出 JSON：

{{
    "relevant_indices": [0, 1]
}}

如果没有相关记忆：

{{
    "relevant_indices": []
}}
"""

    result = call_ollama(
        prompt,
        stage="Relevance Gate"
    )

    indices = result.get("relevant_indices")

    if not isinstance(indices, list):
        raise ValueError(
            "Relevance Gate requires a relevant_indices list."
        )

    if not all(type(index) is int for index in indices):
        raise ValueError(
            "Every relevant memory index must be an integer."
        )

    relevant_memories = []

    for index in indices:
        if (
            isinstance(index, int)
            and 0 <= index < len(candidate_memories)
        ):
            relevant_memories.append(
                candidate_memories[index]
            )

    return relevant_memories


def check_memory_duplicate(new_memory, candidate_memories):
    if not candidate_memories:
        return {
            "is_duplicate": False,
            "duplicate_index": None
        }

    memories_text = "\n".join(
        f"{i}. {memory}"
        for i, memory in enumerate(candidate_memories)
    )

    prompt = f"""
你是一个长期记忆去重判断器。

【准备新增的记忆】
{new_memory}

【已有候选记忆】
{memories_text}

判断准备新增的记忆是否与某条已有记忆表达了基本相同的长期事实。

注意：

1. duplicate 的标准非常严格：
   两条记忆必须表达同一个长期事实，并且在长期记忆库中保留其中任意一条，
   都不会损失另一条所包含的独立信息。

2. 可以把 duplicate 理解为“信息等价”：
   如果把已有记忆替换成新记忆，或把新记忆替换成已有记忆，
   都不会改变核心事实，才算 duplicate。

3. 仅仅满足以下情况绝对不能判定 duplicate：
   - 属于同一个主题
   - 属于同一种行为
   - 使用相同谓词
   - embedding 很相似
   - 都属于同一个 memory_type

4. 必须比较关键事实槽位：
   - 主体
   - 关系 / 属性
   - 对象 / 属性值
   - 必要的状态信息

   只要对象、属性值或关键状态存在实质区别，
   就必须判定为非 duplicate。

5. 如果不能确定两条记忆是否真正信息等价，
   宁可返回 false，也不要误删独立信息。
6. 不要自己改写已有记忆，只返回候选的 index。

例如：

准备新增：
用户最近开始学习日语

已有：
0. 用户正在学习日语
1. 用户喜欢吃苹果

应该返回：

{{
    "is_duplicate": true,
    "duplicate_index": 0
}}

例如：

准备新增：
用户正在学习法语

已有：
0. 用户正在学习日语

应该返回：

{{
    "is_duplicate": false,
    "duplicate_index": null
}}
语义相关、属于同一类别、使用相似表达，不代表是重复记忆。

只有两条记忆表达“同一个事实”时才能判定 duplicate。

如果核心实体、对象、属性值不同，则不能判定为 duplicate。

例如：

“联系人正在学习日语”
和
“联系人正在学习法语”
不是 duplicate。

“联系人正在学习吉他”
和
“联系人正在学习钢琴”
不是 duplicate。

“联系人每周跑步三次”
和
“联系人每周跑步五次”
也不是简单 duplicate；
它们可能是同一属性发生变化，应由 UPDATE 逻辑处理。

判断 duplicate 时必须比较事实的关键槽位：
- 主体
- 关系 / 属性
- 对象或属性值

对象或属性值发生实质变化时，
不能仅因主题相似而判定重复。
判断 duplicate 时采用保守原则：

只有当新记忆和已有候选在核心事实层面信息等价，
保留任意一条都不会损失另一条的独立信息，
才能判定为 duplicate。

如果对象、属性值、目标或关键状态不同，
必须返回 false。

如果无法确定是否属于同一个事实，
也返回 false。

只输出 JSON：

{{
    "is_duplicate": false,
    "duplicate_index": null
}}
"""

    result = call_ollama(
        prompt,
        stage="Duplicate Gate"
    )

    is_duplicate = result.get("is_duplicate")
    duplicate_index = result.get("duplicate_index")

    if not isinstance(is_duplicate, bool):
        raise ValueError(
            "Duplicate Check requires a boolean is_duplicate."
        )

    if duplicate_index is not None and type(duplicate_index) is not int:
        raise ValueError(
            "duplicate_index must be an integer or null."
        )

    if is_duplicate:
        if (
            duplicate_index is None
            or not 0 <= duplicate_index < len(candidate_memories)
        ):
            raise ValueError(
                "Duplicate Check returned an invalid duplicate_index."
            )
    elif duplicate_index is not None:
        raise ValueError(
            "Non-duplicate result requires null duplicate_index."
        )

    return {
        "is_duplicate": is_duplicate,
        "duplicate_index": duplicate_index
    }


    
def resolve_memory_target(target_memory, candidate_memories):
    """
    从候选长期记忆中，找到与 target_memory
    实际对应的那一条真实记忆。

    返回：
    - 匹配成功：候选 memory 的原文
    - 匹配失败：None
    """

    if not candidate_memories:
        return None

    memories_text = "\n".join(
        f"{i}. {memory}"
        for i, memory in enumerate(candidate_memories)
    )

    prompt = f"""
你是一个长期记忆目标匹配器。

【模型想操作的目标记忆】
{target_memory}

【数据库中的候选记忆】
{memories_text}

你的任务是判断：
目标记忆实际对应候选中的哪一条。

注意：

1. 不要求字符串完全相同。
2. 如果只是措辞不同，但表达的是同一个事实，可以认为匹配。
3. 主题相似不代表是同一条记忆。
4. 如果没有任何候选真正对应目标，返回 null。
5. 不要自己生成或改写记忆。
6. 只能返回候选记忆的 index。
7. 匹配时同时判断是否描述同一个实体、同一主题和同一当前状态，
   不要只比较表面动词或固定短语。
8. 对同一持续状态的合理释义差异允许匹配，例如“正在准备某件事”、
   “正在为某件事学习”和“正在为某件事做准备”可以表达同一状态。
9. 仅仅共享关键词或主题仍不足以匹配；事实主体或状态不同则不能匹配。
10. 如果多个候选都可能对应目标且无法可靠区分，返回 null。

例如：

目标：
用户最近开始学日语

候选：
0. 用户最近开始学习日语
1. 用户喜欢吃苹果

正确：

{{
    "matched_index": 0
}}

如果没有匹配：

{{
    "matched_index": null
}}

只输出 JSON。
"""

    result = call_ollama(
        prompt,
        stage="Target Resolver"
    )

    if "matched_index" not in result:
        raise ValueError(
            "Target Resolver requires matched_index."
        )

    index = result["matched_index"]

    if index is not None and type(index) is not int:
        raise ValueError(
            "matched_index must be an integer or null."
        )

    if index is None:
        return None

    if (
        0 <= index < len(candidate_memories)
    ):
        return candidate_memories[index]

    raise ValueError(
        "Target Resolver returned an out-of-range matched_index."
    )

def check_memory_consistency(
    user_text,
    primary_action,
    old_memory,
    new_memory,
    candidate_memories,
    working_messages
):
    """
    检查一次 UPDATE / DELETE 后，
    其他长期记忆是否需要连带 UPDATE / DELETE。
    """

    if not candidate_memories:
        return {"actions": []}

    memories_text = "\n".join(
        f"{i}. {memory}"
        for i, memory in enumerate(candidate_memories)
    )

    history_text = ""

    for message in working_messages:
        history_text += (
            f"{message['role']}: {message['content']}\n"
        )

    prompt = f"""
你是一个长期记忆一致性检查器。

系统刚刚执行了一次主要长期记忆修改。

【用户当前消息】
{user_text}

【最近对话】
{history_text if history_text else "无最近对话"}

【主要操作】
{primary_action}

【原记忆】
{old_memory if old_memory else "无"}

【更新后的记忆】
{new_memory if new_memory else "无"}

【其他候选长期记忆】
{memories_text}

你的任务是判断：
由于主要记忆发生了 UPDATE 或 DELETE，
这些其他候选记忆是否也已经变得不准确、过时或需要调整。

原则：

1. 只有当前用户消息和主要记忆变化明确影响某条候选记忆时，
   才修改它。

2. 只是主题相似，不代表需要修改。

3. 如果一条“当前状态”已经不再成立，
   可以 UPDATE 或 DELETE。

4. 如果某件事情曾经真实发生过，但现在不再持续，
   优先考虑把当前状态改成历史事实，
   而不是直接删除。

例如：

主要操作：
DELETE

原记忆：
用户正在学习日语

用户当前消息：
“他教我教的我都烦了，我现在不学日语了”

候选：
0. 用户的朋友小明在教用户日语，内容与动漫相关
1. 用户喜欢吃苹果

合理结果：

{{
    "actions": [
        {{
            "index": 0,
            "action": "UPDATE",
            "memory": "用户的朋友小明曾教用户日语，内容与动漫相关"
        }}
    ]
}}

5. 如果候选记忆仍然完全成立，不要返回它。

6. action 只允许：
UPDATE
DELETE

7. UPDATE 时必须提供新的 memory。

8. DELETE 时 memory 必须为 null。

9. index 必须来自候选列表。

10. 不要凭空添加用户没有表达过的新事实。

11. 不要修改主要操作本身。

12. 每个 index 最多返回一次。

13. 不要仅仅因为候选记忆与主要记忆拥有相同人物、主题或关键词，
    就认为它受到了影响。

14. 只有主要变化会使候选记忆的事实状态变得不正确、过时或矛盾时，
    才可以 UPDATE 或 DELETE。

15. 不允许为了改写措辞而 UPDATE。
    如果更新后的事实与原候选事实含义没有变化，则不要返回该候选。

例如：

主要操作：
DELETE

原记忆：
用户正在学习日语

用户当前消息：
“我现在不学日语了”

候选：
0. 用户的朋友小明最近开始学 Python

这条候选仍然成立，与用户停止学习日语没有逻辑依赖关系。

正确：

{{
    "actions": []
}}

如果所有候选都不需要修改：

{{
    "actions": []
}}

只输出 JSON：

{{
    "actions": [
        {{
            "index": 0,
            "action": "UPDATE",
            "memory": "更新后的原子化记忆"
        }},
        {{
            "index": 1,
            "action": "DELETE",
            "memory": null
        }}
    ]
}}
"""

    result = call_ollama(
        prompt,
        stage="Consistency"
    )

    raw_actions = result.get("actions")

    if not isinstance(raw_actions, list):
        raise ValueError(
            "Consistency Check requires an actions list."
        )

    valid_actions = []
    used_indices = set()

    for item in raw_actions:
        if not isinstance(item, dict):
            continue

        index = item.get("index")
        action = item.get("action")
        memory = item.get("memory")

        if not isinstance(index, int):
            continue

        if not 0 <= index < len(candidate_memories):
            continue

        if index in used_indices:
            continue

        if action not in ["UPDATE", "DELETE"]:
            continue

        if action == "UPDATE":
            if not isinstance(memory, str) or not memory.strip():
                continue

            # 新旧记忆完全相同，说明实际上没有发生更新
            if memory.strip() == candidate_memories[index].strip():
                continue

        if action == "DELETE":
            memory = None

        valid_actions.append({
            "index": index,
            "action": action,
            "memory": memory
        })

        used_indices.add(index)

    return {
        "actions": valid_actions
    }

def check_memory_worthiness(
    new_memory,
    user_text,
    working_messages
):
    history_text = ""

    for message in working_messages:
        history_text += (
            f"{message['role']}: {message['content']}\n"
        )

    prompt = f"""
你是一个长期记忆类型分类器。

【用户当前消息】
{user_text}

【准备写入的记忆】
{new_memory}
【分类对象规则】

你现在唯一需要分类的对象是：

{new_memory}

必须只判断这一个原子事实属于什么类型。

【用户当前消息】和【最近对话】仅用于帮助理解歧义、指代和上下文，
不能把其中其他独立事实的类型套到当前这条记忆上。

特别是：

如果用户当前消息同时包含多个事实，
这些事实必须彼此独立判断。

例如用户说：

“我最近开始准备教资考试，我刚才正在便利店排队结账”

如果当前【准备写入的记忆】是：

“联系人最近开始准备教资考试”

则应分类为：
ongoing_learning 或 goal。

但如果当前【准备写入的记忆】是：

“联系人最近在便利店排队结账”

则必须分类为：
transient_action。

不能因为同一句原始消息中还出现了“准备教资考试”，
就把“排队结账”错误分类为 ongoing_learning。

判断优先级：

1. 首先只看【准备写入的记忆】本身描述的事实。
2. 只有该事实存在歧义时，才参考原始消息和最近对话。
3. 原始消息中的其他独立事实不得影响当前候选记忆的类型。
4. “最近”“正在”“开始”等时间词本身不能决定长期或短期，
   必须看这个具体活动本身是否会持续、是否对未来对话有价值。

【最近对话】
{history_text if history_text else "无最近对话"}

请判断这条记忆属于哪一种类型。

只允许从以下类型中选择一个：

profile
稳定个人事实，例如居住地、身份等。

preference
稳定偏好，例如喜欢某种食物、音乐。

relationship
人际关系，例如妹妹、朋友、伴侣。

goal
长期目标，例如申请研究生。

ongoing_learning
持续中的学习状态。
例如：
“用户正在学习日语”
“用户最近开始学习 Python”

ongoing_work
持续中的工作或长期任务状态。

habit
持续习惯。
例如：
“用户每周跑步三次”

project
持续中的项目或长期活动。

transient_action
当前瞬时行为。
例如：
“用户正在操场跑步”
“用户正在吃饭”
“用户正在坐公交”

temporary_location
当前临时所在位置。

momentary_state
短暂的情绪、疲劳等状态。

one_off_task
一次性的请求、命令或操作。

other
不属于以上类别。

特别注意：

“用户正在学习日语”
必须分类为 ongoing_learning。

“用户正在操场跑步”
必须分类为 transient_action。

判断时必须关注这个事实的持续时间和未来价值：

- 如果一个事实预计会持续数天、数周或更久，
  并且未来对话中仍可能有用，
  可以属于 goal、ongoing_learning、ongoing_work、
  habit、project 等长期类型。

- 如果一个事实只是当前几分钟或几小时内的现场状态，
  会自然很快结束，
  并且通常不会影响未来对话，
  必须分类为 transient_action、temporary_location、
  momentary_state 或 one_off_task，
  不能因为它使用了“正在”“准备”“现在”等词，
  就误判为持续状态。

例如：

“用户开始准备教师资格考试”
属于 goal 或 ongoing_learning，
因为这是会持续一段时间的目标或学习状态。

“用户正在便利店排队结账”
属于 transient_action，
因为排队结账是很快结束的当前行为。

“用户正在等公交”
属于 transient_action。

“用户正在吃晚饭”
属于 transient_action。

“用户最近开始每天跑步”
属于 habit。

“用户最近开始准备研究生考试”
属于 goal 或 ongoing_learning。

如果当前准备写入的记忆只描述一个短暂现场动作，
即使联系人同一句原始消息中还包含其他值得长期保存的信息，
也必须把当前这条记忆单独分类为 transient_action 等临时类型。

只输出 JSON：

{{
    "memory_type": "ongoing_learning"
}}
"""

    result = call_ollama(
        prompt,
        stage="Worthiness Gate"
    )

    memory_type = result.get("memory_type")

    if not isinstance(memory_type, str) or not memory_type.strip():
        raise ValueError(
            "Worthiness Check requires a memory_type string."
        )

    save_types = {
        "profile",
        "preference",
        "relationship",
        "goal",
        "ongoing_learning",
        "ongoing_work",
        "habit",
        "project"
    }

    should_save = memory_type in save_types

    return {
        "memory_type": memory_type,
        "should_save": should_save
    }
