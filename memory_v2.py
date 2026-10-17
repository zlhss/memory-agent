"""Independent Memory V2 pipeline.

Memory V1 remains implemented by ``main.process_user_message``.  This module
owns the V2 orchestration and reuses only the frozen shared infrastructure:
sender-scoped Working Memory, embedding retrieval, schema validation, gates,
CRUD, and Ollama profiling.
"""

import json
import re
import time

import agent_context
import memory_search
from sklearn.metrics.pairwise import cosine_similarity
from ollama_client import (
    call_ollama,
    check_add_evidence,
    check_memory_worthiness,
    get_ollama_timing_summary,
    get_ollama_timings,
    reset_ollama_timings,
)


ATOMIC_FACT_KINDS = {
    "persistent",
    "persistent_change",
    "transient",
    "query",
    "request",
    "ambiguous",
}
OPERATION_HINTS = {"add", "update", "end", "none"}
ALIGNMENT_DECISIONS = {
    "NEW",
    "DUPLICATE",
    "UPDATE",
    "DELETE",
    "AMBIGUOUS",
}
V2_MIN_FACT_RELEVANCE_SCORE = 0.40
V2_AMBIGUOUS_REFERENT_SCORE = 0.30
V2_DESTRUCTIVE_SUPPORT_SCORE = 0.25
REFERENCE_MARKERS = (
    "这个",
    "那个",
    "它",
    "这件事",
    "那件事",
    "这项",
    "那项",
)
EXPLICIT_CHANGE_MARKERS = (
    "停止",
    "取消",
    "不再",
    "不继续",
    "不打算继续",
    "改成",
    "改为",
    "变成",
    "转成",
    "调整为",
    "替换",
    "换成",
    "搬到",
    "搬去",
    "退出",
    "结束",
    "暂停",
    "放弃",
    "终止",
)


def has_explicit_change_evidence(source_span):
    """Check for explicit language that authorizes change classification."""

    if not isinstance(source_span, str):
        return False
    if any(marker in source_span for marker in EXPLICIT_CHANGE_MARKERS):
        return True
    return bool(re.search(r"不.{1,12}了", source_span))


def _format_working_messages(working_messages):
    if not working_messages:
        return "无最近对话"
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in working_messages
    )


def build_atomic_fact_prompt(sender_id, user_text, working_messages):
    """Build the narrow, atomizing extraction prompt used only by V2."""

    return f"""
你是 Memory V2 的原子事实提取器。当前消息来自联系人 {sender_id}，
不是 Agent 拥有者；消息里的“我”默认指当前联系人。

【最近 Working Memory】
{_format_working_messages(working_messages)}

【联系人当前消息】
{user_text}

你的唯一职责是把当前消息拆成互相独立、可验证的原子项并分类。

只输出以下 JSON：
{{
  "facts": [
    {{
      "source_span": "当前消息中直接支持该项的原文片段",
      "kind": "persistent|persistent_change|transient|query|request|ambiguous",
      "operation_hint": "add|update|end|none",
      "fact": "联系人……，或 null",
      "constraints": ["必须逐字保留的原文限定短语"]
    }}
  ]
}}

规则：
1. 一项只表达一个事实；同一句里的多个长期事实、变化、临时状态、问题和请求必须拆开。
2. persistent 是明确陈述的稳定事实、偏好、关系、长期目标、习惯或持续项目。
3. persistent_change 只用于明确表示已有持续状态被替换、改变、暂停、取消、停止或结束。
   单纯给新事实说明开始时间或未来起点，或说“开始/报名/新增”一项活动，不代表它替换了旧状态，
   除非原文明确说它取代另一个事实；否则必须归为 persistent/add。
4. transient 是很快结束的现场动作、临时位置或短暂状态；“现在、刚刚、正在”描述的现场购物、
   等待、通勤、吃饭等一次性动作必须归为 transient。query 是知识/信息问题；request 是一次性请求。
5. transient、query、request 不得伪装成关于联系人的长期事实，其 fact 必须为 null，operation_hint 必须为 none。
6. 对“这个、那个、它、这件事、这个项目”等指代，只能使用最近联系人消息中明确出现的 referent；assistant 的话不能成为联系人事实来源。
7. 若 Working Memory 中只有一个可靠 referent，必须把 fact 中的指代词替换成最近 user 消息里的具体 referent；
   不能在 fact 中留下“这个项目/那个/它”。若存在多个合理 referent，必须输出 ambiguous、none、null，不能猜。
8. persistent/persistent_change 的 fact 必须以“联系人”开头，不重复 sender 名字。
9. operation_hint：新增或新陈述用 add；改变/替换用 update；停止/取消/结束用 end；其余用 none。
10. 必须保留所有明确限定：期限、次数、频率、时间、数量、否定、程度、范围、偏好强度和状态。
11. constraints 中每一项必须是联系人消息（当前或最近 user 消息）里逐字出现的最短关键短语；fact 中也必须逐字包含每一项。
12. 不能补充常识、推测动机、扩写目标，也不能把提问或请求本身改写成长期事实。
13. 如果当前消息没有任何项目，返回 {{"facts": []}}。

指代示例：
- 最近 user 消息只提到“我在维护家庭账本自动化项目”，当前说“这个项目先暂停”，
  则 persistent_change.fact 必须写成“联系人暂停家庭账本自动化项目”，不能写“联系人暂停这个项目”。
- 最近 user 消息同时提到两个项目，当前只说“这个不做了”，则输出 ambiguous/none/null，不能任选一个。
"""


def _validate_atomic_facts(result):
    if not isinstance(result, dict):
        raise ValueError("Atomic Fact Extractor output must be an object.")

    facts = result.get("facts")
    if not isinstance(facts, list):
        raise ValueError("Atomic Fact Extractor requires a facts list.")

    validated = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Every atomic fact must be an object.")

        source_span = fact.get("source_span")
        kind = fact.get("kind")
        operation_hint = fact.get("operation_hint")
        memory_fact = fact.get("fact")
        constraints = fact.get("constraints")

        if not isinstance(source_span, str) or not source_span.strip():
            raise ValueError("Atomic facts require a non-empty source_span.")
        if kind not in ATOMIC_FACT_KINDS:
            raise ValueError("Atomic fact kind is invalid.")
        if operation_hint not in OPERATION_HINTS:
            raise ValueError("Atomic fact operation_hint is invalid.")
        if not isinstance(constraints, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in constraints
        ):
            raise ValueError("Atomic fact constraints must be strings.")

        if kind in {"persistent", "persistent_change"}:
            if isinstance(memory_fact, str) and memory_fact.strip():
                memory_fact = memory_fact.strip()
                if not memory_fact.startswith("联系人"):
                    memory_fact = f"联系人{memory_fact}"
            if not agent_context.is_valid_contact_memory(memory_fact):
                raise ValueError(
                    "Persistent atomic facts must follow the contact schema."
                )
            if kind == "persistent" and operation_hint not in {"add", "none"}:
                raise ValueError("Persistent facts require add/none hint.")
            if kind == "persistent_change" and operation_hint == "add":
                operation_hint = "update"
            if kind == "persistent_change" and operation_hint not in {"update", "end"}:
                raise ValueError("Persistent changes require update/end hint.")
            if (
                kind == "persistent_change"
                and not has_explicit_change_evidence(source_span)
            ):
                kind = "persistent"
                operation_hint = "add"
        else:
            # Ignored kinds are deterministically canonicalized to no-write.
            # Extra model fields never gain write authority.
            memory_fact = None
            operation_hint = "none"

        validated.append({
            "source_span": source_span.strip(),
            "kind": kind,
            "operation_hint": operation_hint,
            "fact": memory_fact.strip() if isinstance(memory_fact, str) else None,
            "constraints": [item.strip() for item in constraints],
        })

    return validated


def extract_atomic_facts(sender_id, user_text, working_messages):
    """Extract and validate V2 atomic items from one contact message."""

    if not isinstance(sender_id, str) or not sender_id.strip():
        raise ValueError("sender_id must be a non-empty string.")
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")
    if not isinstance(working_messages, list):
        raise ValueError("working_messages must be a list.")

    result = call_ollama(
        build_atomic_fact_prompt(sender_id, user_text, working_messages),
        stage="Atomic Fact Extractor",
    )
    return _validate_atomic_facts(result)


def retrieve_fact_candidates(sender_id, atomic_fact, top_k=5):
    """Run embedding retrieval independently for one persistent fact."""

    query = atomic_fact["fact"]
    started = time.perf_counter()
    high_recall = (
        atomic_fact.get("kind") == "persistent_change"
        or atomic_fact.get("operation_hint") in {"update", "end"}
    )

    if high_recall:
        sender_memories = memory_search.get_sender_memories(sender_id)
        if sender_memories:
            memory_embeddings = memory_search.model.encode(sender_memories)
            query_embedding = memory_search.model.encode([query])
            similarities = cosine_similarity(
                query_embedding,
                memory_embeddings,
            )[0]
            retrieved = sorted(
                zip(similarities, sender_memories),
                key=lambda pair: pair[0],
                reverse=True,
            )[:top_k]
        else:
            retrieved = []
    else:
        retrieved = memory_search.retrieve(sender_id, query, top_k=top_k)
        retrieved = [
            (score, memory)
            for score, memory in retrieved
            if float(score) >= V2_MIN_FACT_RELEVANCE_SCORE
        ]
    return {
        "stage": "Fact Relevance",
        "query": query,
        "mode": "high_recall" if high_recall else "standard",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "candidates": [
            {"index": index, "memory": memory, "score": float(score)}
            for index, (score, memory) in enumerate(retrieved)
        ],
    }


def build_fact_alignment_prompt(atomic_fact, candidates):
    candidates_text = (
        "\n".join(
            f"{index}. {memory}"
            for index, memory in enumerate(candidates)
        )
        if candidates
        else "无候选记忆"
    )
    return f"""
你是 Memory V2 的单事实对齐器。你只对齐下面这一条原子事实，
不提取其他事实，不处理瞬时信息，也不创建额外动作。

【原子事实】
{json.dumps(atomic_fact, ensure_ascii=False)}

【该事实专属候选长期记忆】
{candidates_text}

只输出：
{{
  "decision": "NEW|DUPLICATE|UPDATE|DELETE|AMBIGUOUS",
  "target_index": null,
  "memory": "联系人……，或 null"
}}

定义与约束：
1. NEW：事实是新的长期事实，没有信息等价候选；target_index=null，memory 是完整新记忆。
2. DUPLICATE：某候选已表达同一事实且没有信息损失；target_index 是该候选 index，memory=null。
3. UPDATE：原子事实明确替换或改变同一属性/持续状态；target_index 是旧候选 index，memory 是变化后的完整记忆。
4. DELETE：原子事实明确停止、取消或结束候选中的持续状态，且结束后不应保存成另一条当前事实；target_index 是旧候选 index，memory=null。
5. AMBIGUOUS：无法可靠确定匹配对象，或多个候选同样可能；target_index=null，memory=null。
6. 仅共享主题、类别、谓词或 embedding 相似度不能算 DUPLICATE/UPDATE/DELETE。
7. target_index 只能引用上面真实候选，不能生成 target 字符串，不能改写候选。
8. persistent_change 必须有可确认的同一旧事实才能 UPDATE/DELETE；找不到时返回 AMBIGUOUS，不能 NEW。
9. memory 必须以“联系人”开头，不写姓名，并逐字包含原子事实 constraints 的每一项。
10. 不得丢失期限、次数、频率、时间、数量、否定、程度、范围或状态；不得增加原文未明确支持的信息。
11. 不确定时选择 AMBIGUOUS。
"""


def _validate_alignment(result, atomic_fact, candidates):
    if not isinstance(result, dict):
        raise ValueError("Fact Alignment output must be an object.")

    decision = result.get("decision")
    target_index = result.get("target_index")
    memory = result.get("memory")

    if decision not in ALIGNMENT_DECISIONS:
        raise ValueError("Fact Alignment decision is invalid.")

    raw_memory = memory

    if decision in {"NEW", "AMBIGUOUS"}:
        if target_index is not None:
            return {
                "decision": "AMBIGUOUS",
                "target_index": None,
                "memory": None,
                "raw_memory": raw_memory,
                "safety_reason": f"{decision} returned a non-null target_index",
            }
    else:
        if type(target_index) is not int or not 0 <= target_index < len(candidates):
            return {
                "decision": "AMBIGUOUS",
                "target_index": None,
                "memory": None,
                "raw_memory": raw_memory,
                "safety_reason": "invalid target_index",
            }

    if decision in {"NEW", "UPDATE"}:
        # Atomic Fact Extractor owns wording and constraint fidelity. Alignment
        # owns only relation/target selection, so it cannot re-summarize facts.
        memory = atomic_fact["fact"]
        if not agent_context.is_valid_contact_memory(memory):
            raise ValueError("Canonical atomic memory violates contact schema.")
    else:
        memory = None

    if atomic_fact["kind"] == "persistent_change" and decision == "NEW":
        return {
            "decision": "AMBIGUOUS",
            "target_index": None,
            "memory": None,
            "raw_memory": raw_memory,
            "safety_reason": "persistent_change cannot be NEW",
        }

    if atomic_fact["kind"] == "persistent" and decision == "DELETE":
        return {
            "decision": "AMBIGUOUS",
            "target_index": None,
            "memory": None,
            "raw_memory": raw_memory,
            "safety_reason": "persistent add cannot authorize DELETE",
        }

    return {
        "decision": decision,
        "target_index": target_index,
        "memory": memory.strip() if isinstance(memory, str) else None,
        "raw_memory": raw_memory,
    }


def align_fact_to_memory(atomic_fact, candidates):
    """Align one persistent fact to only its fact-specific candidates."""

    if atomic_fact.get("kind") not in {"persistent", "persistent_change"}:
        raise ValueError("Only persistent facts can be aligned.")
    if not isinstance(candidates, list) or any(
        not isinstance(memory, str) for memory in candidates
    ):
        raise ValueError("candidates must be a list of memory strings.")

    result = call_ollama(
        build_fact_alignment_prompt(atomic_fact, candidates),
        stage="Fact Alignment",
    )
    return _validate_alignment(result, atomic_fact, candidates)


def has_ambiguous_reference(atomic_fact, retrieval):
    """Return whether a vague reference has multiple plausible targets."""

    source_span = atomic_fact.get("source_span", "")
    if not any(marker in source_span for marker in REFERENCE_MARKERS):
        return False

    plausible_candidates = [
        candidate
        for candidate in retrieval.get("candidates", [])
        if candidate.get("score", 0.0) >= V2_AMBIGUOUS_REFERENT_SCORE
    ]
    return len(plausible_candidates) >= 2


def enforce_destructive_target_support(alignment, retrieval):
    """Abstain when an aligned destructive target lacks semantic support."""

    if alignment.get("decision") not in {"UPDATE", "DELETE"}:
        return alignment

    target_index = alignment.get("target_index")
    candidates = retrieval.get("candidates", [])
    if type(target_index) is not int or not 0 <= target_index < len(candidates):
        return alignment

    if candidates[target_index].get("score", 0.0) >= V2_DESTRUCTIVE_SUPPORT_SCORE:
        return alignment

    return {
        "decision": "AMBIGUOUS",
        "target_index": None,
        "memory": None,
        "raw_memory": alignment.get("raw_memory"),
        "safety_reason": "destructive target lacks minimum semantic support",
    }


def check_v2_change_coverage(facts, fact_results):
    """Report uncovered changes without inventing corrective actions."""

    covered_fact_indices = {
        item["fact_index"]
        for item in fact_results
        if item.get("alignment", {}).get("decision") in {"UPDATE", "DELETE"}
        and item.get("applied") is True
    }
    return {
        "uncovered_fact_indices": [
            index
            for index, fact in enumerate(facts)
            if fact["kind"] == "persistent_change"
            and index not in covered_fact_indices
        ]
    }


def check_v2_consistency(primary_action, sender_memories):
    """Conservatively abstain without an explicit dependency relation.

    V2's string-only store has no provenance/dependency metadata.  Therefore a
    secondary mutation cannot be *proved* from mere semantic relatedness.  The
    restricted policy intentionally returns no action instead of over-deleting.
    """

    return {
        "stage": "V2 Consistency",
        "primary_action": primary_action,
        "candidate_memories": sender_memories.copy(),
        "actions": [],
        "reason": "no explicit dependency relation available",
    }


def _apply_alignment(sender_id, alignment, candidates):
    decision = alignment["decision"]
    if decision == "NEW":
        action = {"action": "ADD", "memory": alignment["memory"]}
    elif decision == "UPDATE":
        action = {
            "action": "UPDATE",
            "target": candidates[alignment["target_index"]],
            "memory": alignment["memory"],
        }
    elif decision == "DELETE":
        action = {
            "action": "DELETE",
            "target": candidates[alignment["target_index"]],
            "memory": None,
        }
    else:
        return None, None

    success = memory_search.apply_memory_action(sender_id, action)
    return action, success


def run_memory_lifecycle_v2(sender_id, user_text, working_messages):
    """Run V2 with a supplied snapshot, without resetting or summarizing timing."""

    if not isinstance(sender_id, str) or not sender_id.strip():
        raise ValueError("sender_id must be a non-empty string.")
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    facts = extract_atomic_facts(sender_id, user_text, working_messages)
    fact_results = []
    processed_actions = []
    action_results = []
    consistency_results = []
    used_targets = set()

    for fact_index, atomic_fact in enumerate(facts):
        fact_result = {
            "fact_index": fact_index,
            "atomic_fact": atomic_fact,
            "retrieval": None,
            "alignment": None,
            "validation": {},
            "action": None,
            "applied": False,
        }

        if atomic_fact["kind"] not in {"persistent", "persistent_change"}:
            fact_result["alignment"] = {
                "decision": "IGNORED_NON_PERSISTENT",
                "target_index": None,
                "memory": None,
            }
            fact_results.append(fact_result)
            continue

        retrieval = retrieve_fact_candidates(sender_id, atomic_fact)
        candidates = [item["memory"] for item in retrieval["candidates"]]
        if has_ambiguous_reference(atomic_fact, retrieval):
            alignment = {
                "decision": "AMBIGUOUS",
                "target_index": None,
                "memory": None,
                "raw_memory": None,
                "safety_reason": "vague reference has multiple plausible targets",
            }
        elif atomic_fact["kind"] == "persistent_change" and not candidates:
            alignment = {
                "decision": "AMBIGUOUS",
                "target_index": None,
                "memory": None,
                "raw_memory": None,
                "safety_reason": "persistent change has no candidate target",
            }
        else:
            alignment = align_fact_to_memory(atomic_fact, candidates)
        alignment = enforce_destructive_target_support(alignment, retrieval)
        fact_result["retrieval"] = retrieval
        fact_result["alignment"] = alignment

        decision = alignment["decision"]
        if decision in {"DUPLICATE", "AMBIGUOUS"}:
            fact_results.append(fact_result)
            continue

        if decision == "NEW":
            new_memory = alignment["memory"]
            if not agent_context.is_valid_contact_memory(new_memory):
                fact_result["validation"]["schema"] = False
                fact_results.append(fact_result)
                continue
            fact_result["validation"]["schema"] = True

            evidence = check_add_evidence(
                user_text,
                new_memory,
                working_messages,
            )
            fact_result["validation"]["evidence"] = evidence
            if not evidence["supported"]:
                fact_results.append(fact_result)
                continue

            worthiness = check_memory_worthiness(
                new_memory,
                user_text,
                working_messages,
            )
            fact_result["validation"]["worthiness"] = worthiness
            if not worthiness["should_save"]:
                fact_results.append(fact_result)
                continue

        target_memory = None
        if decision in {"UPDATE", "DELETE"}:
            target_memory = candidates[alignment["target_index"]]
            if target_memory in used_targets:
                fact_result["validation"]["target_conflict"] = True
                fact_results.append(fact_result)
                continue
            used_targets.add(target_memory)

        action, success = _apply_alignment(sender_id, alignment, candidates)
        fact_result["action"] = action
        fact_result["applied"] = bool(success)

        if action is not None:
            processed_actions.append(action)
            action_results.append({"action": action, "success": bool(success)})
            if success and decision in {"UPDATE", "DELETE"}:
                consistency_results.append(
                    check_v2_consistency(
                        action,
                        memory_search.get_sender_memories(sender_id),
                    )
                )

        fact_results.append(fact_result)

    coverage = check_v2_change_coverage(facts, fact_results)
    return {
        "pipeline_version": "v2",
        "sender_id": sender_id,
        "incoming": user_text,
        "working_messages": working_messages,
        "atomic_facts": facts,
        "fact_results": fact_results,
        "coverage": coverage,
        "processed_actions": processed_actions,
        "action_results": action_results,
        "consistency_results": consistency_results,
        "final_memories": memory_search.get_sender_memories(sender_id).copy(),
    }


def _process_user_message_v2_impl(sender_id, user_text):
    if not isinstance(sender_id, str) or not sender_id.strip():
        raise ValueError("sender_id must be a non-empty string.")
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    working_messages = agent_context.get_working_memory(
        sender_id
    ).get_messages().copy()
    return run_memory_lifecycle_v2(sender_id, user_text, working_messages)


def process_user_message_v2(sender_id, user_text):
    """Run the independent Memory V2 pipeline with timing profiling."""

    reset_ollama_timings()
    request_started = time.perf_counter()
    result = None
    outcome = "success"
    try:
        result = _process_user_message_v2_impl(sender_id, user_text)
        return result
    except Exception:
        outcome = "error"
        raise
    finally:
        elapsed = time.perf_counter() - request_started
        profile = {
            "outcome": outcome,
            "calls": get_ollama_timings(),
            **get_ollama_timing_summary(elapsed),
        }
        if isinstance(result, dict):
            result["timing_profile"] = profile
        print("MEMORY_V2_TIMING_SUMMARY:", json.dumps(profile, ensure_ascii=False))
