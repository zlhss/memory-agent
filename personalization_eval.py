"""Development evaluation for personalized conversational replies.

This harness is intentionally separate from the frozen Memory V1/V2
benchmarks.  It compares three response-generation modes without invoking
Memory CRUD:

* generic: current message plus only visible conversation context;
* memory_only: sender-scoped retrieval, planner, and neutral draft;
* full: the same memory-only draft plus sender-scoped style retrieval/rewrite.
"""

import copy
import json
import random
import re
import statistics
import sys
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

import main
import memory_search
from ollama_client import (
    call_ollama,
    filter_relevant_memories,
    generate_draft_reply,
    get_ollama_timing_summary,
    get_ollama_timings,
    plan_response,
    reset_ollama_timings,
    resolve_contextual_query,
    rewrite_reply_style,
)
from regression_test_support import is_environment_error
from style_memory import retrieve_style_samples


PROJECT_DIR = Path(__file__).resolve().parent
CASES_FILE = PROJECT_DIR / "personalization_dev_cases.json"
MEMORY_FILE = PROJECT_DIR / "memories.json"
RESULTS_FILE = PROJECT_DIR / "personalization_dev_results.json"
BLIND_REVIEW_FILE = PROJECT_DIR / "personalization_blind_review.json"
BLIND_KEY_FILE = PROJECT_DIR / "personalization_blind_key.json"

VALID_FACTUAL_ERRORS = {
    "missing_relevant_fact",
    "stale_fact",
    "unsupported_personal_fact",
    "cross_sender_leakage",
    "factual_distortion",
}
VALID_RELEVANCE = {"PASS", "FAIL", "NOT_REQUIRED"}
VALID_GENERIC_FULL = {"GENERIC", "FULL", "TIE"}
VALID_MEMORY_FULL = {"MEMORY_ONLY", "FULL", "TIE"}


REQUIRED_CASE_FIELDS = {
    "case_id",
    "category",
    "sender_id",
    "long_term_memories",
    "sentinel_sender_id",
    "sentinel_memories",
    "working_memory",
    "style_examples",
    "current_message",
    "required_facts",
    "forbidden_facts",
    "should_personalize",
    "max_personal_facts",
    "style_expectations",
    "notes",
}


def load_cases(path=CASES_FILE):
    path = Path(path)
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Personalization cases must be a JSON list.")
    return cases


def validate_cases(cases):
    errors = []
    if not 12 <= len(cases) <= 15:
        errors.append(f"case count must be 12-15, got {len(cases)}")

    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases):
        errors.append("every case must be a JSON object")
        return errors
    if len(set(case_ids)) != len(case_ids):
        errors.append("case_id values must be unique")

    all_sender_ids = []
    for case in cases:
        case_id = case.get("case_id", "<missing>")
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            errors.append(f"{case_id}: missing fields {sorted(missing)}")
            continue

        sender_id = case["sender_id"]
        sentinel_id = case["sentinel_sender_id"]
        if not isinstance(sender_id, str) or not sender_id.startswith(
            "personal_dev_sender_"
        ):
            errors.append(f"{case_id}: invalid synthetic sender_id")
        if not isinstance(sentinel_id, str) or not sentinel_id.startswith(
            "personal_dev_sentinel_"
        ):
            errors.append(f"{case_id}: invalid synthetic sentinel_sender_id")
        if sender_id == sentinel_id:
            errors.append(f"{case_id}: target and sentinel sender collide")
        all_sender_ids.extend([sender_id, sentinel_id])

        for field in (
            "long_term_memories",
            "sentinel_memories",
            "style_examples",
        ):
            if not isinstance(case[field], list) or not case[field]:
                errors.append(f"{case_id}: {field} must be a non-empty list")

        for field in ("required_facts", "forbidden_facts"):
            if not isinstance(case[field], list):
                errors.append(f"{case_id}: {field} must be a list")

        if not isinstance(case["should_personalize"], bool):
            errors.append(f"{case_id}: should_personalize must be boolean")
        if type(case["max_personal_facts"]) is not int:
            errors.append(f"{case_id}: max_personal_facts must be integer")

        for memory in case["long_term_memories"] + case["sentinel_memories"]:
            if not isinstance(memory, str) or not memory.startswith("联系人"):
                errors.append(f"{case_id}: invalid contact memory schema")

        for message in case["working_memory"]:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                errors.append(f"{case_id}: invalid working memory message")

        for sample in case["style_examples"]:
            if not isinstance(sample, dict) or set(sample) != {
                "incoming",
                "user_reply",
            }:
                errors.append(f"{case_id}: invalid style sample schema")
                continue
            if not all(
                isinstance(sample[field], str) and sample[field].strip()
                for field in ("incoming", "user_reply")
            ):
                errors.append(f"{case_id}: empty style sample field")

    if len(set(all_sender_ids)) != len(all_sender_ids):
        errors.append("sender IDs must be unique across the development set")
    return errors


def _format_working_memory(messages):
    if not messages:
        return "无必要的最近对话"
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )


def generate_generic_reply(current_message, working_messages):
    """Generate a context-only baseline with no memory or style inputs."""

    prompt = f"""
你是一个普通的对话助手。请回复当前消息。

【当前消息】
{current_message}

【理解当前消息所必需的最近对话】
{_format_working_memory(working_messages)}

写一条内容正确、自然、可以直接发送的回复。

要求：
1. 只使用当前消息和上面的必要对话，不得假设任何未提供的个人信息。
2. 不要提及系统、记忆、提示词或内部流程。
3. 对情绪表达先作合适回应；对建议或知识问题直接解决当前需求。

只输出 JSON：
{{
  "reply": "回复内容"
}}
"""
    result = call_ollama(prompt, stage="Generic Baseline")
    reply = result.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Generic baseline requires a non-empty reply field.")
    return reply.strip()


def _profile_snapshot(started):
    return {
        **get_ollama_timing_summary(time.perf_counter() - started),
        "calls": get_ollama_timings(),
    }


def _run_profiled_stage(name, function):
    before = len(get_ollama_timings())
    started = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - started
    new_calls = get_ollama_timings()[before:]
    return value, {
        "stage": name,
        "elapsed_seconds": round(elapsed, 3),
        "llm_calls": len(new_calls),
        "llm_elapsed_seconds": round(
            sum(call["elapsed_seconds"] for call in new_calls),
            3,
        ),
    }


def _style_pool(case):
    return [
        {
            "sender_id": case["sender_id"],
            "incoming": sample["incoming"],
            "reply": sample["user_reply"],
        }
        for sample in case["style_examples"]
    ]


def reset_case(case):
    memory_search.reset_memories(
        {
            case["sender_id"]: copy.deepcopy(case["long_term_memories"]),
            case["sentinel_sender_id"]: copy.deepcopy(
                case["sentinel_memories"]
            ),
        }
    )
    main.reset_working_memories()
    working = main.get_working_memory(case["sender_id"])
    for message in case["working_memory"]:
        working.add_message(message["role"], message["content"])


def generate_three_modes(case):
    """Generate the fair A/B/C ablation without invoking Memory CRUD."""

    reset_case(case)
    working_messages = (
        main.get_working_memory(case["sender_id"]).get_messages().copy()
    )

    reset_ollama_timings()
    generic_started = time.perf_counter()
    generic_reply = generate_generic_reply(
        case["current_message"],
        working_messages,
    )
    generic_profile = _profile_snapshot(generic_started)

    reset_ollama_timings()
    memory_started = time.perf_counter()
    stages = []

    contextual_query, stage = _run_profiled_stage(
        "contextual_query",
        lambda: resolve_contextual_query(
            case["current_message"], working_messages
        ),
    )
    stages.append(stage)

    retrieval_results, stage = _run_profiled_stage(
        "long_term_retrieval",
        lambda: memory_search.retrieve(
            case["sender_id"], contextual_query, top_k=3
        ),
    )
    stages.append(stage)
    candidates = [memory for _, memory in retrieval_results]

    relevant_memories, stage = _run_profiled_stage(
        "relevance_gate",
        lambda: filter_relevant_memories(contextual_query, candidates),
    )
    stages.append(stage)

    response_plan, stage = _run_profiled_stage(
        "planner",
        lambda: plan_response(
            case["sender_id"],
            case["current_message"],
            relevant_memories,
            working_messages,
        ),
    )
    stages.append(stage)

    draft_reply, stage = _run_profiled_stage(
        "draft",
        lambda: generate_draft_reply(
            case["sender_id"],
            case["current_message"],
            relevant_memories,
            working_messages,
            response_plan,
        ),
    )
    stages.append(stage)
    memory_profile = _profile_snapshot(memory_started)
    memory_profile["stage_breakdown"] = copy.deepcopy(stages)

    style_samples, stage = _run_profiled_stage(
        "style_retrieval",
        lambda: retrieve_style_samples(
            case["sender_id"],
            case["current_message"],
            top_k=3,
            samples=_style_pool(case),
        ),
    )
    stages.append(stage)

    full_reply, stage = _run_profiled_stage(
        "style_rewrite",
        lambda: rewrite_reply_style(
            draft_reply,
            case["current_message"],
            style_samples,
        ),
    )
    stages.append(stage)

    full_elapsed = sum(item["elapsed_seconds"] for item in stages)
    full_profile = {
        **get_ollama_timing_summary(full_elapsed),
        "calls": get_ollama_timings(),
        "stage_breakdown": copy.deepcopy(stages),
    }

    sentinel_after = memory_search.get_sender_memories(
        case["sentinel_sender_id"]
    ).copy()
    target_after = memory_search.get_sender_memories(case["sender_id"]).copy()

    return {
        "case_id": case["case_id"],
        "generic": {"reply": generic_reply, "profile": generic_profile},
        "memory_only": {"reply": draft_reply, "profile": memory_profile},
        "full": {"reply": full_reply, "profile": full_profile},
        "retrieval": {
            "contextual_query": contextual_query,
            "embedding_candidates": [
                {"score": float(score), "memory": memory}
                for score, memory in retrieval_results
            ],
            "relevant_memories": relevant_memories,
        },
        "plan": response_plan,
        "style_retrieval": style_samples,
        "namespace_integrity": {
            "target_unchanged": target_after == case["long_term_memories"],
            "sentinel_unchanged": sentinel_after == case["sentinel_memories"],
            "target_after": target_after,
            "sentinel_after": sentinel_after,
        },
    }


def _judge_prompt(case, generated):
    rubric = {
        "current_message": case["current_message"],
        "working_memory": case["working_memory"],
        "target_sender_memories": case["long_term_memories"],
        "sentinel_sender_memories_never_allowed": case["sentinel_memories"],
        "required_facts_when_relevant": case["required_facts"],
        "forbidden_facts": case["forbidden_facts"],
        "should_personalize": case["should_personalize"],
        "max_personal_facts": case["max_personal_facts"],
        "style_expectations": case["style_expectations"],
        "notes": case["notes"],
    }
    replies = {
        "generic": generated["generic"]["reply"],
        "memory_only": generated["memory_only"]["reply"],
        "full": generated["full"]["reply"],
    }
    style_replies = [sample["user_reply"] for sample in case["style_examples"]]
    return f"""
你是 Personalized Response Development Evaluation 的严格分类 Judge。
你只做逐项事实和语义判定，不生成新回复，也不输出总分。

【Case rubric】
{json.dumps(rubric, ensure_ascii=False, indent=2)}

【三个待评回复】
{json.dumps(replies, ensure_ascii=False, indent=2)}

【目标表达风格样本】
{json.dumps(style_replies, ensure_ascii=False, indent=2)}

判定规则：
1. factual_pass 只有在 required facts（若有）被正确用于当前回复、没有 forbidden/stale/sentinel/unsupported 个人事实、也没有事实扭曲时才为 true。
2. 不需要个性化的 case，不提个人记忆是正确行为；不能因没有提记忆而判失败。
3. personalization_relevance 只判断个人信息使用是否自然且必要。无需个性化时，无强行提及则为 NOT_REQUIRED；若强行提及则为 FAIL。
4. relevant_memory_used 只表示回复确实使用了与当前需求相关的 target memory，不能把复述当前消息算成使用长期记忆。
5. unsupported_personal_detail 指关于联系人的具体事实无法从当前消息、Working Memory 或 target memories 得到。
6. cross_sender_leakage 指使用了 sentinel sender 的任何事实。
7. semantic_preservation 比较 memory_only 与 full：Full 必须保留 Draft 的核心事实、建议、限制和意图；仅措辞变化才 PASS。
8. style_match 只判断哪条更接近 style replies 的句长、口语、标点和语气；不能因事实更多而获胜。
9. 对否定语义要准确理解，不得只凭关键词判断。
10. errors 只能从以下值选择：missing_relevant_fact、stale_fact、unsupported_personal_fact、cross_sender_leakage、factual_distortion；没有则为空数组。

只输出 JSON：
{{
  "generic": {{
    "factual_pass": true,
    "personalization_relevance": "PASS|FAIL|NOT_REQUIRED",
    "relevant_memory_used": false,
    "errors": [],
    "unsupported_personal_detail": false,
    "cross_sender_leakage": false,
    "reason": "简短证据"
  }},
  "memory_only": {{
    "factual_pass": true,
    "personalization_relevance": "PASS|FAIL|NOT_REQUIRED",
    "relevant_memory_used": true,
    "errors": [],
    "unsupported_personal_detail": false,
    "cross_sender_leakage": false,
    "reason": "简短证据"
  }},
  "full": {{
    "factual_pass": true,
    "personalization_relevance": "PASS|FAIL|NOT_REQUIRED",
    "relevant_memory_used": true,
    "errors": [],
    "unsupported_personal_detail": false,
    "cross_sender_leakage": false,
    "reason": "简短证据"
  }},
  "semantic_preservation": {{
    "pass": true,
    "lost_or_changed": [],
    "reason": "简短证据"
  }},
  "generic_vs_full": {{
    "more_personally_relevant": "GENERIC|FULL|TIE",
    "better_context_use": "GENERIC|FULL|TIE",
    "factual_preservation": "GENERIC|FULL|TIE",
    "reason": "简短证据"
  }},
  "memory_only_vs_full": {{
    "style_match": "MEMORY_ONLY|FULL|TIE",
    "semantic_preservation": true,
    "reason": "简短证据"
  }}
}}
"""


def _validate_mode_judgment(value, mode):
    if not isinstance(value, dict):
        raise ValueError(f"Judge {mode} result must be an object.")
    required = {
        "factual_pass",
        "personalization_relevance",
        "relevant_memory_used",
        "errors",
        "unsupported_personal_detail",
        "cross_sender_leakage",
        "reason",
    }
    if not required <= set(value):
        raise ValueError(f"Judge {mode} result is missing fields.")
    for field in (
        "factual_pass",
        "relevant_memory_used",
        "unsupported_personal_detail",
        "cross_sender_leakage",
    ):
        if not isinstance(value[field], bool):
            raise ValueError(f"Judge {mode}.{field} must be boolean.")
    if value["personalization_relevance"] not in VALID_RELEVANCE:
        raise ValueError(f"Judge {mode} relevance is invalid.")
    if not isinstance(value["errors"], list) or not set(value["errors"]) <= (
        VALID_FACTUAL_ERRORS
    ):
        raise ValueError(f"Judge {mode} errors are invalid.")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError(f"Judge {mode} reason must be non-empty.")
    return value


def validate_judgment(result):
    if not isinstance(result, dict):
        raise ValueError("Personalization judge must return an object.")
    validated = {
        mode: _validate_mode_judgment(result.get(mode), mode)
        for mode in ("generic", "memory_only", "full")
    }

    preservation = result.get("semantic_preservation")
    if (
        not isinstance(preservation, dict)
        or not isinstance(preservation.get("pass"), bool)
        or not isinstance(preservation.get("lost_or_changed"), list)
        or not isinstance(preservation.get("reason"), str)
    ):
        raise ValueError("Judge semantic_preservation is invalid.")
    validated["semantic_preservation"] = preservation

    generic_full = result.get("generic_vs_full")
    if not isinstance(generic_full, dict) or any(
        generic_full.get(field) not in VALID_GENERIC_FULL
        for field in (
            "more_personally_relevant",
            "better_context_use",
            "factual_preservation",
        )
    ):
        raise ValueError("Judge generic_vs_full is invalid.")
    validated["generic_vs_full"] = generic_full

    memory_full = result.get("memory_only_vs_full")
    if (
        not isinstance(memory_full, dict)
        or memory_full.get("style_match") not in VALID_MEMORY_FULL
        or not isinstance(memory_full.get("semantic_preservation"), bool)
    ):
        raise ValueError("Judge memory_only_vs_full is invalid.")
    validated["memory_only_vs_full"] = memory_full
    return validated


def judge_outputs(case, generated):
    reset_ollama_timings()
    started = time.perf_counter()
    result = call_ollama(
        _judge_prompt(case, generated),
        stage="Personalization Eval Judge",
    )
    judgment = validate_judgment(result)
    return judgment, _profile_snapshot(started)


def _reply_structure(reply):
    clauses = [part for part in re.split(r"[。！？!?；;\n]+", reply) if part]
    markers = [
        marker
        for marker in ("哈哈", "先", "稳住", "别急", "可以啊", "～", "吧")
        if marker in reply
    ]
    emoji_count = len(
        re.findall(r"[\U0001F300-\U0001FAFF]", reply)
    )
    return {
        "characters": len(reply),
        "clause_count": len(clauses),
        "average_clause_characters": round(
            sum(len(clause) for clause in clauses) / len(clauses), 3
        ) if clauses else 0.0,
        "exclamation_marks": reply.count("!") + reply.count("！"),
        "question_marks": reply.count("?") + reply.count("？"),
        "tildes": reply.count("~") + reply.count("～"),
        "emoji_count": emoji_count,
        "colloquial_markers": markers,
    }


def _longest_verbatim_copy(reply, style_replies):
    best = {"characters": 0, "text": "", "sample": ""}
    for sample in style_replies:
        match = SequenceMatcher(None, reply, sample).find_longest_match(
            0, len(reply), 0, len(sample)
        )
        if match.size > best["characters"]:
            best = {
                "characters": match.size,
                "text": reply[match.a:match.a + match.size],
                "sample": sample,
            }
    best["flagged"] = best["characters"] >= 12
    return best


def compute_proxy_metrics(case, generated):
    style_replies = [sample["user_reply"] for sample in case["style_examples"]]
    texts = [
        generated["memory_only"]["reply"],
        generated["full"]["reply"],
        *style_replies,
    ]
    embeddings = memory_search.model.encode(texts)
    memory_scores = cosine_similarity(
        embeddings[0:1], embeddings[2:]
    )[0]
    full_scores = cosine_similarity(
        embeddings[1:2], embeddings[2:]
    )[0]
    semantic_similarity = float(
        cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    )
    return {
        "style_similarity_proxy": {
            "description": (
                "Embedding similarity to style replies; proxy only, not "
                "human preference."
            ),
            "memory_only_mean": float(memory_scores.mean()),
            "memory_only_max": float(memory_scores.max()),
            "full_mean": float(full_scores.mean()),
            "full_max": float(full_scores.max()),
        },
        "draft_full_semantic_embedding_similarity": semantic_similarity,
        "structure": {
            "memory_only": _reply_structure(generated["memory_only"]["reply"]),
            "full": _reply_structure(generated["full"]["reply"]),
            "style_examples": [
                _reply_structure(reply) for reply in style_replies
            ],
        },
        "verbatim_copy": _longest_verbatim_copy(
            generated["full"]["reply"], style_replies
        ),
    }


def evaluate_case(case):
    started = time.perf_counter()
    try:
        generated = generate_three_modes(case)
        proxy_metrics = compute_proxy_metrics(case, generated)
        judgment, evaluation_profile = judge_outputs(case, generated)
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "status": "PASS",
            "case_elapsed_seconds": round(time.perf_counter() - started, 3),
            "inputs": {
                "current_message": case["current_message"],
                "working_memory": case["working_memory"],
                "required_facts": case["required_facts"],
                "forbidden_facts": case["forbidden_facts"],
                "should_personalize": case["should_personalize"],
            },
            "outputs": generated,
            "automatic_evaluation": judgment,
            "proxy_metrics": proxy_metrics,
            "evaluation_profile": evaluation_profile,
        }
    except Exception as error:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "status": (
                "ENVIRONMENT_ERROR" if is_environment_error(error)
                else "PROGRAM_ERROR"
            ),
            "case_elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _average(values):
    return round(statistics.mean(values), 3) if values else 0.0


def summarize_results(results):
    completed = [result for result in results if result["status"] == "PASS"]
    summary = {
        "case_count": len(results),
        "completed_cases": len(completed),
        "program_errors": sum(
            result["status"] == "PROGRAM_ERROR" for result in results
        ),
        "environment_errors": sum(
            result["status"] == "ENVIRONMENT_ERROR" for result in results
        ),
        "human_preference": "PENDING HUMAN REVIEW",
    }

    for mode in ("generic", "memory_only", "full"):
        judgments = [
            result["automatic_evaluation"][mode] for result in completed
        ]
        profiles = [result["outputs"][mode]["profile"] for result in completed]
        summary[mode] = {
            "factual_pass_count": sum(
                judgment["factual_pass"] for judgment in judgments
            ),
            "factual_pass_rate": (
                sum(judgment["factual_pass"] for judgment in judgments)
                / len(judgments)
                if judgments else 0.0
            ),
            "relevant_memory_use_count": sum(
                judgment["relevant_memory_used"] for judgment in judgments
            ),
            "unsupported_personalization_count": sum(
                judgment["unsupported_personal_detail"]
                for judgment in judgments
            ),
            "cross_sender_leakage_count": sum(
                judgment["cross_sender_leakage"] for judgment in judgments
            ),
            "average_agent_latency_seconds": _average(
                profile["total_request_elapsed_seconds"] for profile in profiles
            ),
            "average_llm_calls": _average(
                profile["total_llm_calls"] for profile in profiles
            ),
        }

    summary["full"]["semantic_preservation_count"] = sum(
        result["automatic_evaluation"]["semantic_preservation"]["pass"]
        for result in completed
    )
    summary["full"]["semantic_preservation_rate"] = (
        summary["full"]["semantic_preservation_count"] / len(completed)
        if completed else 0.0
    )
    summary["full"]["verbatim_copy_flag_count"] = sum(
        result["proxy_metrics"]["verbatim_copy"]["flagged"]
        for result in completed
    )
    summary["namespace_pollution_count"] = sum(
        not result["outputs"]["namespace_integrity"]["sentinel_unchanged"]
        or not result["outputs"]["namespace_integrity"]["target_unchanged"]
        for result in completed
    )
    summary["generic_vs_full"] = {
        field: {
            value: sum(
                result["automatic_evaluation"]["generic_vs_full"][field]
                == value
                for result in completed
            )
            for value in VALID_GENERIC_FULL
        }
        for field in (
            "more_personally_relevant",
            "better_context_use",
            "factual_preservation",
        )
    }
    summary["memory_only_vs_full_style_match"] = {
        value: sum(
            result["automatic_evaluation"]["memory_only_vs_full"][
                "style_match"
            ] == value
            for result in completed
        )
        for value in VALID_MEMORY_FULL
    }
    summary["personalization_factual_advantage_count"] = sum(
        not result["automatic_evaluation"]["generic"]["factual_pass"]
        and result["automatic_evaluation"]["full"]["factual_pass"]
        for result in completed
    )
    summary["personalization_factual_regression_count"] = sum(
        result["automatic_evaluation"]["generic"]["factual_pass"]
        and not result["automatic_evaluation"]["full"]["factual_pass"]
        for result in completed
    )
    summary["average_evaluation_latency_seconds"] = _average(
        result["evaluation_profile"]["total_request_elapsed_seconds"]
        for result in completed
    )
    summary["average_evaluation_llm_calls"] = _average(
        result["evaluation_profile"]["total_llm_calls"]
        for result in completed
    )
    return summary


def write_blind_review(cases, results):
    result_by_id = {result["case_id"]: result for result in results}
    rng = random.Random("personalization-dev-blind-v1")
    review_cases = []
    key_cases = []
    for case in cases:
        result = result_by_id.get(case["case_id"])
        if not result or result["status"] != "PASS":
            continue
        labels = ["GENERIC", "FULL"]
        rng.shuffle(labels)
        replies = {
            "GENERIC": result["outputs"]["generic"]["reply"],
            "FULL": result["outputs"]["full"]["reply"],
        }
        review_cases.append(
            {
                "case_id": case["case_id"],
                "current_message": case["current_message"],
                "conversation_context": case["working_memory"],
                "reply_a": replies[labels[0]],
                "reply_b": replies[labels[1]],
                "review": {
                    "q1_would_send": None,
                    "q2_matches_my_style": None,
                    "q3_fits_contact_context": None,
                    "q4_contains_wrong_personal_fact": None,
                    "q5_edit_needed_for_better_reply": None,
                },
            }
        )
        key_cases.append(
            {
                "case_id": case["case_id"],
                "reply_a": labels[0],
                "reply_b": labels[1],
            }
        )

    BLIND_REVIEW_FILE.write_text(
        json.dumps(
            {
                "human_preference": "PENDING HUMAN REVIEW",
                "instructions": {
                    "q1_would_send": "A|B|TIE",
                    "q2_matches_my_style": "A|B|TIE",
                    "q3_fits_contact_context": "A|B|TIE",
                    "q4_contains_wrong_personal_fact": "A|B|BOTH|NEITHER",
                    "q5_edit_needed_for_better_reply": (
                        "ACCEPT|MINOR_EDIT|MAJOR_EDIT|REJECT"
                    ),
                },
                "cases": review_cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    BLIND_KEY_FILE.write_text(
        json.dumps(
            {
                "warning": "Internal key; do not show during blind review.",
                "cases": key_cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_development_evaluation(cases):
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case['case_id']}", flush=True)
        result = evaluate_case(case)
        results.append(result)
        if result["status"] == "PASS":
            auto = result["automatic_evaluation"]
            print(
                "  completed | factual="
                f"G:{auto['generic']['factual_pass']} "
                f"M:{auto['memory_only']['factual_pass']} "
                f"F:{auto['full']['factual_pass']} | "
                f"semantic_preservation={auto['semantic_preservation']['pass']}",
                flush=True,
            )
        else:
            print(
                f"  {result['status']}: {result['error']}",
                flush=True,
            )
    return results


def print_summary(summary):
    print("\nPERSONALIZATION DEVELOPMENT SUMMARY")
    print(
        f"COMPLETED: {summary['completed_cases']}/{summary['case_count']} | "
        f"PROGRAM_ERROR: {summary['program_errors']} | "
        f"ENVIRONMENT_ERROR: {summary['environment_errors']}"
    )
    for mode in ("generic", "memory_only", "full"):
        metrics = summary[mode]
        print(
            f"{mode.upper()}: factual="
            f"{metrics['factual_pass_count']}/{summary['completed_cases']} "
            f"({metrics['factual_pass_rate']:.1%}) | "
            f"avg_latency={metrics['average_agent_latency_seconds']:.3f}s | "
            f"avg_calls={metrics['average_llm_calls']:.2f} | "
            f"unsupported={metrics['unsupported_personalization_count']} | "
            f"leakage={metrics['cross_sender_leakage_count']}"
        )
    print(
        "FULL_SEMANTIC_PRESERVATION: "
        f"{summary['full']['semantic_preservation_count']}/"
        f"{summary['completed_cases']} "
        f"({summary['full']['semantic_preservation_rate']:.1%})"
    )
    print(f"NAMESPACE_POLLUTION: {summary['namespace_pollution_count']}")
    print("HUMAN_PREFERENCE: PENDING HUMAN REVIEW")


def main_entry():
    cases = load_cases()
    errors = validate_cases(cases)
    if errors:
        for error in errors:
            print(f"VALIDATION_ERROR: {error}")
        return 2
    print(f"PERSONALIZATION_DEV_CASES_VALID: {len(cases)}")
    if "--validate" in sys.argv:
        return 0

    original_memories = json.loads(
        MEMORY_FILE.read_text(encoding="utf-8")
    )
    try:
        results = run_development_evaluation(cases)
        summary = summarize_results(results)
        payload = {
            "evaluation": "personalization_development",
            "model": "qwen2.5:7b-instruct",
            "case_source": CASES_FILE.name,
            "human_preference": "PENDING HUMAN REVIEW",
            "summary": summary,
            "cases": results,
        }
        RESULTS_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_blind_review(cases, results)
        print_summary(summary)
        print(f"RESULTS_FILE: {RESULTS_FILE}")
        print(f"BLIND_REVIEW_FILE: {BLIND_REVIEW_FILE}")
        print(f"BLIND_KEY_FILE: {BLIND_KEY_FILE}")
        return 0
    finally:
        memory_search.reset_memories(original_memories)
        main.reset_working_memories()
        print("Original memories restored; Working Memory cleared.")


if __name__ == "__main__":
    raise SystemExit(main_entry())
