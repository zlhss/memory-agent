import hashlib
import json
import multiprocessing
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests


PROJECT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = PROJECT_DIR / "ollama_ab_prompts.json"
RESULTS_FILE = PROJECT_DIR / "ollama_soak_results.json"

MODEL = "qwen3-vl:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
HTTP_TIMEOUT_SECONDS = 300
HARD_STOP_SECONDS = 60
CYCLES = 10
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
                if type(target_index) is not int or memory is not None:
                    return False
            else:
                return False
        return True

    return False


def utc_now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_prompts():
    saved = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    prompts = {item["stage"]: item["prompt"] for item in saved["prompts"]}

    for stage in STAGES:
        digest = hashlib.sha256(prompts[stage].encode("utf-8")).hexdigest()
        if digest != EXPECTED_HASHES[stage]:
            raise RuntimeError(
                f"Prompt hash mismatch for {stage}: {digest}"
            )

    return prompts


def process_snapshot():
    processes = []
    total_rss = 0

    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = process.info["name"] or ""
            normalized = name.lower()
            if "ollama" not in normalized and "llama-server" not in normalized:
                continue
            rss = process.info["memory_info"].rss
            processes.append({
                "pid": process.info["pid"],
                "name": name,
                "rss_bytes": rss,
            })
            total_rss += rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    gpu = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 6:
                gpu.append({
                    "index": int(fields[0]),
                    "name": fields[1],
                    "utilization_percent": int(fields[2]),
                    "vram_used_mib": int(fields[3]),
                    "vram_total_mib": int(fields[4]),
                    "temperature_c": int(fields[5]),
                })
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        gpu = []

    return {
        "timestamp": utc_now(),
        "system_available_ram_bytes": psutil.virtual_memory().available,
        "ollama_process_count": len(processes),
        "ollama_process_ram_bytes": total_rss,
        "ollama_processes": sorted(processes, key=lambda item: item["pid"]),
        "gpu": gpu,
    }


def get_ollama_ps():
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return completed.stdout.rstrip()
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return f"ERROR: {error!r}"


def request_worker(queue, stage, prompt):
    started = time.perf_counter()
    result = {
        "elapsed_seconds": None,
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
                "options": {
                    "temperature": 0,
                },
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        result["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        response_text = data.get("response", "").strip()
        thinking_text = data.get("thinking", "").strip()
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
    queue.put(result)


def run_call(stage, prompt):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=request_worker,
        args=(queue, stage, prompt),
    )
    outer_started = time.perf_counter()
    process.start()
    process.join(HARD_STOP_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(5)
        queue.close()
        return {
            "elapsed_seconds": round(time.perf_counter() - outer_started, 3),
            "status": "hard_stop_over_60s",
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
            "error": "Request exceeded 60-second soak hard stop",
        }

    if queue.empty():
        queue.close()
        return {
            "elapsed_seconds": round(time.perf_counter() - outer_started, 3),
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
            "error": f"Worker exited without result (exit code {process.exitcode})",
        }

    result = queue.get()
    queue.close()
    return result


def percentile_95(values):
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * 0.95 + 0.999999999))
    return ordered[rank - 1]


def build_summary(results):
    latencies = [item["elapsed_seconds"] for item in results]
    first_ten = latencies[:10]
    last_ten = latencies[-10:]
    initial_pids = {
        item["pid"] for item in results[0]["before"]["ollama_processes"]
        if item["name"].lower() == "llama-server.exe"
    } if results else set()
    observed_pids = {
        item["pid"]
        for result in results
        for snapshot in (result["before"], result["after"])
        for item in snapshot["ollama_processes"]
        if item["name"].lower() == "llama-server.exe"
    }
    reload_sequences = [
        item["sequence"]
        for item in results
        if isinstance(item.get("load_duration"), int)
        and item["load_duration"] >= 1_000_000_000
    ]

    return {
        "completed_calls": len(results),
        "average_latency_seconds": round(statistics.mean(latencies), 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "p95_latency_seconds": round(percentile_95(latencies), 3),
        "max_latency_seconds": round(max(latencies), 3),
        "first_10_average_latency_seconds": round(
            statistics.mean(first_ten), 3
        ),
        "last_10_average_latency_seconds": round(
            statistics.mean(last_ten), 3
        ),
        "over_10_seconds_sequences": [
            item["sequence"] for item in results
            if item["elapsed_seconds"] > 10
        ],
        "hard_stop_count": sum(
            item["status"] == "hard_stop_over_60s" for item in results
        ),
        "timeout_count": sum(
            item["status"] == "timeout" for item in results
        ),
        "error_count": sum(
            item["status"] == "error" for item in results
        ),
        "json_valid_count": sum(item["json_valid"] for item in results),
        "llama_server_initial_pids": sorted(initial_pids),
        "llama_server_observed_pids": sorted(observed_pids),
        "load_duration_reload_sequences": reload_sequences,
        "model_reload_detected": (
            len(observed_pids) > 1 or bool(reload_sequences)
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
        "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "hard_stop_seconds": HARD_STOP_SECONDS,
        "cycles_requested": CYCLES,
        "stage_order": list(STAGES),
        "prompt_hashes": EXPECTED_HASHES,
        "started_at": utc_now(),
        "initial_ollama_ps": get_ollama_ps(),
        "initial_state": process_snapshot(),
        "results": [],
        "stopped_early": False,
        "stop_reason": None,
    }
    save_artifact(artifact)
    print(
        "SOAK_INITIAL "
        + json.dumps({
            "ollama_ps": artifact["initial_ollama_ps"],
            "state": artifact["initial_state"],
        }, ensure_ascii=False),
        flush=True,
    )

    sequence = 0
    should_stop = False

    for cycle in range(1, CYCLES + 1):
        for stage in STAGES:
            sequence += 1
            before = process_snapshot()
            result = run_call(stage, prompts[stage])
            after = process_snapshot()
            result.update({
                "sequence": sequence,
                "cycle": cycle,
                "stage": stage,
                "before": before,
                "after": after,
            })
            artifact["results"].append(result)
            save_artifact(artifact)
            print(
                "SOAK_RESULT " + json.dumps(result, ensure_ascii=False),
                flush=True,
            )

            if result["status"] == "hard_stop_over_60s":
                artifact["stopped_early"] = True
                artifact["stop_reason"] = result["error"]
                should_stop = True
                break

        if should_stop:
            break

    artifact["finished_at"] = utc_now()
    artifact["final_ollama_ps"] = get_ollama_ps()
    artifact["final_state"] = process_snapshot()
    artifact["summary"] = build_summary(artifact["results"])
    save_artifact(artifact)
    print(
        "SOAK_SUMMARY "
        + json.dumps(artifact["summary"], ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
