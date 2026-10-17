import hashlib
import json
import statistics
import time
from pathlib import Path

import requests

from ollama_client import (
    OLLAMA_URL,
    decide_memory_actions,
    filter_relevant_memories,
    get_ollama_profiled_prompts,
    plan_response,
    reset_ollama_timings,
)


PROJECT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = PROJECT_DIR / "ollama_ab_prompts.json"
RESULTS_FILE = PROJECT_DIR / "ollama_ab_results.json"

MODELS = (
    "qwen3-vl:8b",
    "qwen3-vl:8b-instruct",
)

STAGES = (
    "Relevance Gate",
    "Planner",
    "Memory Manager",
)

USER_TEXT = "我已经搬去苏州长期住了"
CANDIDATE_MEMORIES = [
    "联系人住在南京",
    "联系人喜欢羽毛球",
]
RELEVANT_MEMORIES = ["联系人住在南京"]
SENDER_ID = "regression_target"


def capture_profiled_prompts():
    reset_ollama_timings()

    calls = (
        lambda: filter_relevant_memories(
            USER_TEXT,
            CANDIDATE_MEMORIES,
        ),
        lambda: plan_response(
            SENDER_ID,
            USER_TEXT,
            RELEVANT_MEMORIES,
            [],
        ),
        lambda: decide_memory_actions(
            SENDER_ID,
            USER_TEXT,
            RELEVANT_MEMORIES,
            [],
        ),
    )

    capture_errors = []

    for call in calls:
        try:
            call()
        except Exception as error:
            capture_errors.append(repr(error))

    captured = {}

    for item in get_ollama_profiled_prompts():
        if item["stage"] in STAGES:
            captured[item["stage"]] = item["prompt"]

    missing = [stage for stage in STAGES if stage not in captured]

    if missing:
        raise RuntimeError(
            f"Failed to capture prompts for stages: {missing}"
        )

    saved = {
        "source": "real stage-level profiling calls",
        "capture_errors": capture_errors,
        "prompts": [
            {
                "stage": stage,
                "prompt_chars": len(captured[stage]),
                "sha256": hashlib.sha256(
                    captured[stage].encode("utf-8")
                ).hexdigest(),
                "prompt": captured[stage],
            }
            for stage in STAGES
        ],
    }
    PROMPTS_FILE.write_text(
        json.dumps(saved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return captured, saved


def is_valid_stage_json(stage, raw_text):
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        return False

    if not isinstance(data, dict):
        return False

    if stage == "Relevance Gate":
        indices = data.get("relevant_indices")
        return (
            isinstance(indices, list)
            and all(type(index) is int for index in indices)
        )

    if stage == "Planner":
        intent = data.get("intent")
        points = data.get("points")
        tone = data.get("tone")
        return (
            isinstance(intent, str)
            and bool(intent.strip())
            and isinstance(points, list)
            and bool(points)
            and all(
                isinstance(point, str) and bool(point.strip())
                for point in points
            )
            and isinstance(tone, str)
            and bool(tone.strip())
        )

    if stage == "Memory Manager":
        actions = data.get("actions")

        if not isinstance(actions, list):
            return False

        for action in actions:
            if not isinstance(action, dict):
                return False

            action_type = action.get("action")
            target_index = action.get("target_index")
            memory = action.get("memory")

            if action_type == "ADD":
                if target_index is not None:
                    return False
                if not isinstance(memory, str) or not memory.strip():
                    return False
            elif action_type == "UPDATE":
                if type(target_index) is not int:
                    return False
                if not isinstance(memory, str) or not memory.strip():
                    return False
            elif action_type == "DELETE":
                if type(target_index) is not int:
                    return False
                if memory is not None:
                    return False
            else:
                return False

        return True

    return False


def run_once(model, stage, prompt, run_number):
    started = time.perf_counter()
    status = "error"
    response_text = ""
    thinking_text = ""
    metrics = {}
    error_text = None
    json_valid = False

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        response_text = data.get("response", "").strip()
        thinking_text = data.get("thinking", "").strip()
        raw_text = response_text or thinking_text
        json_valid = is_valid_stage_json(stage, raw_text)
        metrics = {
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get(
                "prompt_eval_duration"
            ),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
        }
        status = "success"
    except requests.exceptions.Timeout as error:
        status = "timeout"
        error_text = repr(error)
    except Exception as error:
        status = "error"
        error_text = repr(error)

    return {
        "model": model,
        "stage": stage,
        "run": run_number,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "status": status,
        "thinking_chars": len(thinking_text),
        "response_chars": len(response_text),
        "prompt_eval_count": metrics.get("prompt_eval_count"),
        "prompt_eval_duration": metrics.get("prompt_eval_duration"),
        "eval_count": metrics.get("eval_count"),
        "eval_duration": metrics.get("eval_duration"),
        "load_duration": metrics.get("load_duration"),
        "total_duration": metrics.get("total_duration"),
        "json_valid": json_valid,
        "error": error_text,
    }


def summarize(results):
    summaries = []

    for model in MODELS:
        model_results = [
            result for result in results
            if result["model"] == model
        ]
        latencies = [
            result["elapsed_seconds"]
            for result in model_results
        ]
        summaries.append({
            "model": model,
            "attempts": len(model_results),
            "average_latency_seconds": round(
                statistics.mean(latencies), 3
            ),
            "median_latency_seconds": round(
                statistics.median(latencies), 3
            ),
            "max_latency_seconds": round(max(latencies), 3),
            "timeout_count": sum(
                result["status"] == "timeout"
                for result in model_results
            ),
            "error_count": sum(
                result["status"] == "error"
                for result in model_results
            ),
            "json_valid_count": sum(
                result["json_valid"]
                for result in model_results
            ),
            "thinking_chars_total": sum(
                result["thinking_chars"]
                for result in model_results
            ),
        })

    return summaries


def main():
    prompts, prompt_artifact = capture_profiled_prompts()
    print(
        "PROMPTS_SAVED "
        + json.dumps(
            [
                {
                    "stage": item["stage"],
                    "prompt_chars": item["prompt_chars"],
                    "sha256": item["sha256"],
                }
                for item in prompt_artifact["prompts"]
            ],
            ensure_ascii=False,
        ),
        flush=True,
    )

    results = []

    for model in MODELS:
        for stage in STAGES:
            for run_number in range(1, 4):
                result = run_once(
                    model,
                    stage,
                    prompts[stage],
                    run_number,
                )
                results.append(result)
                print(
                    "BENCHMARK_RESULT "
                    + json.dumps(result, ensure_ascii=False),
                    flush=True,
                )

    summaries = summarize(results)
    artifact = {
        "models": list(MODELS),
        "runs_per_model_per_prompt": 3,
        "timeout_seconds": 300,
        "results": results,
        "summaries": summaries,
    }
    RESULTS_FILE.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "BENCHMARK_SUMMARY "
        + json.dumps(summaries, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
