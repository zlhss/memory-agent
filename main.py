import copy
import json
import time

from memory_search import (
    retrieve,
    apply_memory_action
)

from feedback_store import record_feedback
from style_memory import retrieve_style_samples
from agent_context import (
    working_memories,
    get_working_memory,
    reset_working_memories,
    is_valid_contact_memory
)

from ollama_client import (
    get_ollama_timing_summary,
    get_ollama_timings,
    reset_ollama_timings,
    resolve_contextual_query,
    decide_memory_actions,
    check_action_coverage,
    check_add_evidence,
    filter_relevant_memories,
    check_memory_duplicate,
    check_memory_consistency,
    check_memory_worthiness,
    plan_response,
    generate_draft_reply,
    rewrite_reply_style
)


def build_personalized_reply(
    sender_id,
    user_text,
    relevant_memories,
    working_messages
):
    """Build reply artifacts without managing the memory lifecycle."""

    style_samples = retrieve_style_samples(
        sender_id,
        user_text,
        top_k=3
    )

    response_plan = plan_response(
        sender_id,
        user_text,
        relevant_memories,
        working_messages
    )

    draft_reply = generate_draft_reply(
        sender_id,
        user_text,
        relevant_memories,
        working_messages,
        response_plan
    )

    personalized_reply = rewrite_reply_style(
        draft_reply,
        user_text,
        style_samples
    )

    return {
        "style_samples": style_samples,
        "plan": response_plan,
        "draft": draft_reply,
        "personalized_reply": personalized_reply
    }


def run_memory_lifecycle_v1(
    sender_id,
    user_text,
    relevant_memories,
    working_messages
):
    """Run the existing V1 lifecycle and return its operation results."""


     
    # 5. Memory Manager
    # 一句话可能产生多个 action
     

    actions = decide_memory_actions(
        sender_id,
        user_text,
        relevant_memories,
        working_messages
    )

    print(
        "Raw Actions:",
        actions
    )

    missing_actions = []

    if relevant_memories:
        coverage_result = check_action_coverage(
            user_text,
            relevant_memories,
            actions
        )
        missing_actions = coverage_result["missing_actions"]

    print(
        "Missing Coverage Actions:",
        missing_actions
    )

    merged_actions = []
    covered_target_indices = set()

    for action in actions + missing_actions:
        if action.get("action") in {"UPDATE", "DELETE"}:
            target_index = action.get("target_index")

            if target_index in covered_target_indices:
                continue

            covered_target_indices.add(target_index)

        merged_actions.append(action)

    actions = merged_actions

    print(
        "Merged Actions:",
        actions
    )


     
    # 6. Write Validation
    #
    # ADD：
    #   Semantic Duplicate Check
    #
    # UPDATE / DELETE：
    #   Semantic Target Resolution
     

    processed_actions = []

    for action in actions:

        action_type = action.get("action")


        # ----------------------------------------------
        # ADD：检查是否已经存在语义重复记忆
        # ----------------------------------------------

        if action_type == "ADD":

            new_memory = action.get("memory")

            if not is_valid_contact_memory(new_memory):
                print(
                    "Contact Memory Schema Rejected:",
                    new_memory
                )
                continue

            # 1. ADD Evidence Gate
            evidence = check_add_evidence(
                user_text,
                new_memory,
                working_messages
            )

            print(
                "ADD Evidence:",
                new_memory,
                evidence
            )

            if not evidence["supported"]:
                continue

            # 2. Worthiness Gate
            worthiness = check_memory_worthiness(
                new_memory,
                user_text,
                working_messages
            )

            print(
                "Worthiness:",
                new_memory,
                worthiness
            )

            # 不值得长期保存
            if not worthiness["should_save"]:
                continue


            # 3. Duplicate Gate
            duplicate_results = retrieve(
                sender_id,
                new_memory,
                top_k=3
            )

            duplicate_candidates = [
                memory
                for score, memory in duplicate_results
            ]

            duplicate_result = check_memory_duplicate(
                new_memory,
                duplicate_candidates
            )

            print(
                "Duplicate Candidates:",
                duplicate_candidates
            )

            print(
                "Duplicate Check:",
                duplicate_result
            )

            # 已经存在同义长期记忆
            if duplicate_result.get("is_duplicate"):
                continue
        # ----------------------------------------------
        # UPDATE / DELETE：
        # 使用 Memory Manager 选择的 relevant memory index，
        # 映射到数据库中的真实原文。
        # ----------------------------------------------

        elif action_type in ["UPDATE", "DELETE"]:

            if (
                action_type == "UPDATE"
                and not is_valid_contact_memory(
                    action.get("memory")
                )
            ):
                print(
                    "Contact Memory Schema Rejected:",
                    action.get("memory")
                )
                continue

            target_index = action.get("target_index")
            resolved_target = relevant_memories[target_index]

            print(
                "Selected Target:",
                resolved_target
            )

            # 使用数据库中的真实原文
            action["target"] = resolved_target

            # UPDATE 前后完全相同，没有任何实际变化
            if (
                action_type == "UPDATE"
                and action.get("memory")
                and action["memory"].strip() == resolved_target.strip()
            ):
                continue


        # ----------------------------------------------
        # 非法 action
        # ----------------------------------------------

        else:
            continue


        # 只有通过验证的 action 才进入执行阶段
        processed_actions.append(action)


     
    # 7. 真正执行长期记忆修改
     

    action_results = []

    for action in processed_actions:

        error_message = None

        try:
            success = apply_memory_action(
                sender_id,
                action
            )
        except Exception as error:
            success = False
            error_message = str(error)
            print(
                "Memory Action Error:",
                error_message
            )

        action_results.append({
            "action": action,
            "success": success,
            "error": error_message
        })

    # ==================================================
    # 8. Consistency Check
    #
    # 所有显式 action 执行完成后，
    # 检查其他长期记忆是否因此变得过时
    # ==================================================

    # 用户已经明确处理过的 memory
    # Consistency 不允许再次修改它们
    explicitly_touched_memories = set()

    for result in action_results:

        if not result["success"]:
            continue

        action = result["action"]

        target = action.get("target")
        new_memory = action.get("memory")

        if target:
            explicitly_touched_memories.add(target)

        if new_memory:
            explicitly_touched_memories.add(new_memory)


    # 记录 Consistency 已经修改过的 memory
    # 防止多个 primary action 重复修改同一条
    consistency_touched_memories = set()

    consistency_results_all = []
    consistency_errors = []


    for result in action_results:

        # 主 action 没成功，就不做一致性检查
        if not result["success"]:
            continue

        primary = result["action"]
        primary_action = primary["action"]

        # ADD 暂时不触发 consistency
        if primary_action not in ["UPDATE", "DELETE"]:
            continue

        primary_target = primary.get("target")
        primary_new_memory = primary.get("memory")


        # 用用户原话 + 主 memory 变化
        # 找可能受到影响的其他长期记忆
        consistency_query = " ".join(
            part
            for part in [
                user_text,
                primary_target,
                primary_new_memory
            ]
            if part
        )

        try:
            consistency_retrieval = retrieve(
                sender_id,
                consistency_query,
                top_k=5
            )
        except Exception as error:
            error_record = {
                "primary_action": primary,
                "stage": "retrieval",
                "error": str(error)
            }
            consistency_errors.append(error_record)
            print("Consistency Error:", error_record)
            continue

        consistency_candidates = [
            memory
            for score, memory in consistency_retrieval
        ]


        # 排除：
        # 1. 用户已经显式操作过的 memory
        # 2. 前面的 consistency 已经处理过的 memory
        consistency_candidates = [
            memory
            for memory in consistency_candidates
            if memory not in explicitly_touched_memories
            and memory not in consistency_touched_memories
        ]


        if not consistency_candidates:
            continue


        try:
            consistency_result = check_memory_consistency(
                user_text=user_text,
                primary_action=primary_action,
                old_memory=primary_target,
                new_memory=primary_new_memory,
                candidate_memories=consistency_candidates,
                working_messages=working_messages
            )
        except Exception as error:
            error_record = {
                "primary_action": primary,
                "stage": "llm_check",
                "error": str(error)
            }
            consistency_errors.append(error_record)
            print("Consistency Error:", error_record)
            continue

        print(
            "Consistency Candidates:",
            consistency_candidates
        )

        print(
            "Consistency Result:",
            consistency_result
        )


        # 执行 Consistency 产生的 UPDATE / DELETE
        for consistency_action in consistency_result["actions"]:

            index = consistency_action["index"]

            target_memory = consistency_candidates[index]

            action_type = consistency_action["action"]

            if action_type == "UPDATE":

                if not is_valid_contact_memory(
                    consistency_action["memory"]
                ):
                    print(
                        "Contact Memory Schema Rejected:",
                        consistency_action["memory"]
                    )
                    continue

                side_action = {
                    "action": "UPDATE",
                    "target": target_memory,
                    "memory": consistency_action["memory"]
                }

            elif action_type == "DELETE":

                side_action = {
                    "action": "DELETE",
                    "target": target_memory,
                    "memory": None
                }

            else:
                continue


            error_message = None

            try:
                success = apply_memory_action(
                    sender_id,
                    side_action
                )
            except Exception as error:
                success = False
                error_message = str(error)
                print(
                    "Consistency Action Error:",
                    error_message
                )

            consistency_results_all.append({
                "action": side_action,
                "success": success,
                "error": error_message
            })


            if success:
                # 旧 memory 已经处理
                consistency_touched_memories.add(
                    target_memory
                )

                # UPDATE 后的新 memory 也标记
                if side_action["memory"]:
                    consistency_touched_memories.add(
                        side_action["memory"]
                    )


    print(
        "Consistency Action Results:",
        consistency_results_all
    )


     
    return {
        "processed_actions": processed_actions,
        "action_results": action_results,
        "consistency_results": consistency_results_all,
        "consistency_errors": consistency_errors
    }


def _process_user_message_impl(sender_id, user_text, *, memory_version="v1"):

    if not isinstance(sender_id, str) or not sender_id.strip():
        raise ValueError("sender_id must be a non-empty string.")

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string.")

    working_memory = get_working_memory(
    sender_id
)    
        
     
    # 1. Working Memory + Contextual Query Resolution
    # 获取当前消息之前的最近对话，并解析指代后再做长期记忆检索。
     
    working_messages = working_memory.get_messages()
    if memory_version == "v2":
        working_messages = copy.deepcopy(working_messages)

    print(
        "Working Memory:",
        working_messages
    )

    contextual_query = resolve_contextual_query(
        user_text,
        working_messages
    )

    print(
        "Contextual Query:",
        contextual_query
    )


    # 2. Long-Term Memory：Embedding 粗召回

    results = retrieve(
        sender_id,
        contextual_query,
        top_k=3
    )

    candidate_memories = [
        memory
        for score, memory in results
    ]


     
    # 3. LLM Relevance Gate
    # 过滤掉 embedding 找到的无关候选
     

    relevant_memories = filter_relevant_memories(
        contextual_query,
        candidate_memories
    )
    print("Relevant Memories:", relevant_memories)

    # 4. Personalized Response Pipeline
    # 先完成回复生成；任何阶段失败时，本轮尚未写入长期记忆。

    reply_result = build_personalized_reply(
        sender_id,
        user_text,
        relevant_memories,
        working_messages
    )

    style_samples = reply_result["style_samples"]
    response_plan = reply_result["plan"]
    draft_reply = reply_result["draft"]
    personalized_reply = reply_result["personalized_reply"]


     
    if memory_version == "v1":
        memory_result = run_memory_lifecycle_v1(
            sender_id,
            user_text,
            relevant_memories,
            working_messages
        )
    else:
        from memory_v2 import run_memory_lifecycle_v2

        memory_result = {
            "memory_version": "v2",
            "memory_v2_details": run_memory_lifecycle_v2(
                sender_id,
                user_text,
                working_messages
            )
        }

    # 9. 返回待人工确认的回复及调试信息。
    # Working Memory 必须等 accept/edit/reject 后再记录真实结果。

    return {
        "sender_id": sender_id,
        "incoming": user_text,
        "contextual_query": contextual_query,
        "candidate_memories": candidate_memories,
        "relevant_memories": relevant_memories,
        "style_samples": style_samples,
        "plan": response_plan,
        "draft": draft_reply,
        "personalized_reply": personalized_reply,
        **memory_result
    }


def process_user_message(sender_id, user_text, *, memory_version="v1"):
    """Run one request with an explicit memory lifecycle and timing profile.

    V1 preserves the original result contract. V2 returns its native lifecycle
    result under memory_v2_details; its indices and consistency records do not
    use the V1 contract. V2 errors propagate, and earlier writes remain applied.
    """

    if not isinstance(memory_version, str) or memory_version not in ("v1", "v2"):
        raise ValueError("memory_version must be 'v1' or 'v2'.")

    reset_ollama_timings()
    request_started = time.perf_counter()
    result = None
    outcome = "success"

    try:
        result = _process_user_message_impl(
            sender_id,
            user_text,
            memory_version=memory_version
        )
        return result
    except Exception:
        outcome = "error"
        raise
    finally:
        total_request_elapsed = (
            time.perf_counter() - request_started
        )
        calls = get_ollama_timings()
        summary = get_ollama_timing_summary(
            total_request_elapsed
        )
        profile = {
            "outcome": outcome,
            "calls": calls,
            **summary
        }

        if isinstance(result, dict):
            result["timing_profile"] = profile

        print(
            "OLLAMA_TIMING_SUMMARY:",
            json.dumps(profile, ensure_ascii=False)
        )


def finalize_response(
    sender_id,
    response_result,
    feedback,
    edited_reply=None
):
    """Persist feedback, then record the actual conversation outcome."""

    if not isinstance(response_result, dict):
        raise ValueError("response_result must be an object.")

    if response_result.get("sender_id") != sender_id:
        raise ValueError(
            "sender_id does not match the generated response."
        )

    record = record_feedback(
        sender_id=sender_id,
        incoming=response_result.get("incoming"),
        generated=response_result.get("personalized_reply"),
        feedback=feedback,
        final=edited_reply
    )

    working_memory = get_working_memory(sender_id)
    working_memory.add_message(
        "user",
        record["incoming"]
    )

    if record["final"]:
        working_memory.add_message(
            "assistant",
            record["final"]
        )

    return record


 
# CLI Chat

if __name__ == "__main__":

    while True:

        sender_id = input("Sender: ").strip()

        if sender_id.lower() in [
            "exit",
            "quit"
        ]:
            break

        if not sender_id:
            print("Sender cannot be empty.")
            continue

        user_text = input("Message: ").strip()
        
        
        if user_text.lower() in [
            "exit",
            "quit"
        ]:
            break

        try:
            response_result = process_user_message(
                sender_id,
                user_text
            )
        except Exception as error:
            print(f"Generation failed: {error}")
            continue

        print("\nRelevant Long-Term Memories")
        print(json.dumps(
            response_result["relevant_memories"],
            ensure_ascii=False,
            indent=2
        ))

        print("\nRetrieved Style Samples")
        print(json.dumps(
            response_result["style_samples"],
            ensure_ascii=False,
            indent=2
        ))

        print("\nPlan")
        print(json.dumps(
            response_result["plan"],
            ensure_ascii=False,
            indent=2
        ))

        print("\nDraft")
        print(response_result["draft"])

        print("\nPersonalized Reply")
        print(response_result["personalized_reply"])

        while True:
            choice = input(
                "\n[a] accept  [e] edit  [r] reject: "
            ).strip().lower()

            try:
                if choice == "a":
                    feedback_record = finalize_response(
                        sender_id,
                        response_result,
                        "accepted"
                    )
                    break

                if choice == "e":
                    edited_reply = input(
                        "Edited Reply: "
                    ).strip()
                    feedback_record = finalize_response(
                        sender_id,
                        response_result,
                        "edited",
                        edited_reply=edited_reply
                    )
                    break

                if choice == "r":
                    feedback_record = finalize_response(
                        sender_id,
                        response_result,
                        "rejected"
                    )
                    break

                print("Please choose a, e, or r.")
            except (ValueError, OSError) as error:
                print(f"Feedback failed: {error}")

        print(
            "Feedback saved:",
            feedback_record["feedback"]
        )
