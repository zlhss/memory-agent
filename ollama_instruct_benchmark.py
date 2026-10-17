import hashlib
import json
import multiprocessing
import queue
import statistics
import subprocess
import time
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = PROJECT_DIR / "ollama_ab_prompts.json"
RESULTS_FILE = PROJECT_DIR / "ollama_instruct_results.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-vl:8b-instruct"
HTTP_TIMEOUT_SECONDS = 300
HARD_STOP_SECONDS = 60
RUNS_PER_PROMPT = 5
STAGES = (
    "Relevance Gate",
    "Planner",
    "Memory Manager",
)
EXPECTED_HASHES = {
    "Relevance Gate": (
        "d02a72fdb2d04ee93b63d33efc9fd7d9fedf0e9479fcd3100dbb2e24c4e582a6"
    ),
    "Planner": (
        "6ec2929150dbba7f4018703876ab46718ae88cb49582bb06e1f21323f6302e63"
    ),
    "Memory Manager": (
        "c89487fe5133e726016439dccd8fd6b7888d225e35cf67611c9e9ff6ed910567"
    ),
}


def load_prompts():
    saved = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    prompts = {item["stage"]: item["prompt"] for item in saved["prompts"]}
    for stage in STAGES:
        digest = hashlib.sha256(prompts[stage].encode("utf-8")).hexdigest()
        if digest != EXPECTED_HASHES[stage]:
            raise RuntimeError(f"Prompt hash mismatch for {stage}: {digest}")
    return prompts


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
        return (
            isinstance(data.get("intent"), str)
            and bool(data["intent"].strip())
            and isinstance(data.get("points"), list)
            and bool(data["points"])
            and all(
                isinstance(point, str) and bool(point.strip())
                for point in data["points"]
            )
            and isinstance(data.get("tone"), str)
            and bool(data["tone"].strip())
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
                if type(target_index) is not int or memory is not None:
                    return False
            else:
                return False
        return True

    return False


def request_worker(result_queue, stage, prompt):
    started = time.perf_counter()
    result = {
        "status": "error",
        "http_status": None,
        "thinking_chars": 0,
        "response_chars": 0,
        "load_duration": None,
        "prompt_eval_count": None,
        "prompt_eval_duration": None,
        "eval_count": None,
        "eval_duration": None,
        "total_duration": None,
        "json_valid": False,
        "error": None,
    }
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        result["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        response_text = (data.get("response", "") or "").strip()
        thinking_text = (data.get("thinking", "") or "").strip()
        raw_text = response_text or thinking_text
        result.update({
            "status": "success",
            "thinking_chars": len(thinking_text),
            "response_chars": len(response_text),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
            "json_valid": is_valid_stage_json(stage, raw_text),
        })
    except requests.exceptions.Timeout as error:
        result["status"] = "timeout"
        result["error"] = repr(error)
    except Exception as error:
        result["status"] = "error"
        result["error"] = repr(error)

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result_queue.put(result)


def gpu_utilization():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(completed.stdout.splitlines()[0].strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def wait_until_gpu_quiet():
    quiet_samples = 0
    started = time.perf_counter()
    while time.perf_counter() - started < 30:
        quiet_samples = quiet_samples + 1 if gpu_utilization() < 50 else 0
        if quiet_samples >= 2:
            return
        time.sleep(1)


def run_once(stage, prompt, run_number, sequence):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=request_worker,
        args=(result_queue, stage, prompt),
    )
    started = time.perf_counter()
    process.start()
    process.join(HARD_STOP_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(5)
        result = {
            "status": "hard_stop_over_60s",
            "http_status": None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "thinking_chars": 0,
            "response_chars": 0,
            "load_duration": None,
            "prompt_eval_count": None,
            "prompt_eval_duration": None,
            "eval_count": None,
            "eval_duration": None,
            "total_duration": None,
            "json_valid": False,
            "error": "Request exceeded 60-second benchmark hard stop",
        }
    else:
        try:
            result = result_queue.get(timeout=2)
        except queue.Empty:
            result = {
                "status": "error",
                "http_status": None,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "thinking_chars": 0,
                "response_chars": 0,
                "load_duration": None,
                "prompt_eval_count": None,
                "prompt_eval_duration": None,
                "eval_count": None,
                "eval_duration": None,
                "total_duration": None,
                "json_valid": False,
                "error": f"Worker exited without result (exit code {process.exitcode})",
            }

    result_queue.close()
    result.update({
        "sequence": sequence,
        "stage": stage,
        "run": run_number,
    })
    if result["status"] == "hard_stop_over_60s":
        wait_until_gpu_quiet()
    return result


def summarize(results):
    latencies = [item["elapsed_seconds"] for item in results]
    return {
        "attempted": len(results),
        "completed": sum(item["status"] == "success" for item in results),
        "average_latency_seconds": round(statistics.mean(latencies), 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "max_latency_seconds": round(max(latencies), 3),
        "over_10_seconds": sum(value > 10 for value in latencies),
        "over_60_seconds": sum(
            item["status"] == "hard_stop_over_60s" for item in results
        ),
        "json_valid": sum(item["json_valid"] for item in results),
        "timeout_count": sum(item["status"] == "timeout" for item in results),
        "error_count": sum(item["status"] == "error" for item in results),
        "thinking_chars": sum(item["thinking_chars"] for item in results),
        "response_chars": sum(item["response_chars"] for item in results),
        "thinking_present_count": sum(
            item["thinking_chars"] > 0 for item in results
        ),
        "thinking_only_count": sum(
            item["thinking_chars"] > 0 and item["response_chars"] == 0
            for item in results
        ),
    }


def save_artifact(artifact):
    RESULTS_FILE.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    prompts = load_prompts()
    artifact = {
        "model": MODEL,
        "request_parameters": {
            "stream": False,
            "format": "json",
            "think": False,
            "temperature": 0,
        },
        "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "hard_stop_seconds": HARD_STOP_SECONDS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "prompt_hashes": EXPECTED_HASHES,
        "results": [],
    }
    save_artifact(artifact)
    sequence = 0

    for stage in STAGES:
        for run_number in range(1, RUNS_PER_PROMPT + 1):
            sequence += 1
            result = run_once(stage, prompts[stage], run_number, sequence)
            artifact["results"].append(result)
            save_artifact(artifact)
            print(
                "INSTRUCT_RESULT " + json.dumps(result, ensure_ascii=False),
                flush=True,
            )

    artifact["summary"] = summarize(artifact["results"])
    save_artifact(artifact)
    print(
        "INSTRUCT_SUMMARY "
        + json.dumps(artifact["summary"], ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
