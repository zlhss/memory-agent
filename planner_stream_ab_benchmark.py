import hashlib
import json
import multiprocessing
import queue
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = PROJECT_DIR / "ollama_ab_prompts.json"
RESULTS_FILE = PROJECT_DIR / "planner_stream_ab_results.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-vl:8b"
PLANNER_HASH = (
    "6ec2929150dbba7f4018703876ab46718ae88cb49582bb06e1f21323f6302e63"
)
RUNS_PER_EXPERIMENT = 5
HTTP_TIMEOUT_SECONDS = 300
HARD_STOP_SECONDS = 60
TELEMETRY_INTERVAL_SECONDS = 5


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_planner_prompt():
    saved = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    item = next(
        value for value in saved["prompts"]
        if value["stage"] == "Planner"
    )
    prompt = item["prompt"]
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if digest != PLANNER_HASH:
        raise RuntimeError(f"Planner prompt hash mismatch: {digest}")
    return prompt


def gpu_snapshot():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,clocks.current.sm,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        devices = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 8:
                continue
            devices.append({
                "index": int(fields[0]),
                "name": fields[1],
                "utilization_percent": int(fields[2]),
                "vram_used_mib": int(fields[3]),
                "vram_total_mib": int(fields[4]),
                "sm_clock_mhz": int(fields[5]),
                "power_draw_watts": float(fields[6]),
                "temperature_c": int(fields[7]),
            })
        return devices
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


def planner_json_valid(raw_text):
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
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


def stream_worker(event_queue, prompt, think_value):
    started = time.perf_counter()
    response_text = ""
    thinking_text = ""
    chunk_count = 0
    http_status = None

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "format": "json",
        "options": {"temperature": 0},
    }
    if think_value is not None:
        payload["think"] = think_value

    try:
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            http_status = response.status_code
            event_queue.put({
                "type": "headers",
                "http_status": http_status,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            response.raise_for_status()

            final_metrics = {}
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                received_elapsed = time.perf_counter() - started
                data = json.loads(raw_line)
                thinking_delta = data.get("thinking", "") or ""
                response_delta = data.get("response", "") or ""
                thinking_text += thinking_delta
                response_text += response_delta
                chunk_count += 1
                event_queue.put({
                    "type": "chunk",
                    "chunk_number": chunk_count,
                    "received_at": now_iso(),
                    "elapsed_seconds": round(received_elapsed, 6),
                    "thinking_delta_chars": len(thinking_delta),
                    "response_delta_chars": len(response_delta),
                })
                if data.get("done"):
                    final_metrics = {
                        "load_duration": data.get("load_duration"),
                        "prompt_eval_count": data.get("prompt_eval_count"),
                        "prompt_eval_duration": data.get("prompt_eval_duration"),
                        "eval_count": data.get("eval_count"),
                        "eval_duration": data.get("eval_duration"),
                        "total_duration": data.get("total_duration"),
                    }

        raw_text = response_text.strip() or thinking_text.strip()
        event_queue.put({
            "type": "complete",
            "status": "success",
            "http_status": http_status,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "thinking_chars": len(thinking_text),
            "response_chars": len(response_text),
            "chunk_count": chunk_count,
            "json_valid": planner_json_valid(raw_text),
            "error": None,
            **final_metrics,
        })
    except requests.exceptions.Timeout as error:
        event_queue.put({
            "type": "complete",
            "status": "timeout",
            "http_status": http_status,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "thinking_chars": len(thinking_text),
            "response_chars": len(response_text),
            "chunk_count": chunk_count,
            "json_valid": False,
            "error": repr(error),
        })
    except Exception as error:
        event_queue.put({
            "type": "complete",
            "status": "error",
            "http_status": http_status,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "thinking_chars": len(thinking_text),
            "response_chars": len(response_text),
            "chunk_count": chunk_count,
            "json_valid": False,
            "error": repr(error),
        })


def drain_events(event_queue, chunks, counters, result_holder):
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            return
        if event["type"] == "headers":
            counters["http_status"] = event["http_status"]
        elif event["type"] == "chunk":
            chunks.append(event)
            counters["thinking_chars"] += event["thinking_delta_chars"]
            counters["response_chars"] += event["response_delta_chars"]
            counters["chunk_count"] += 1
        elif event["type"] == "complete":
            result_holder.append(event)


def telemetry_sample(elapsed, counters):
    return {
        "sampled_at": now_iso(),
        "elapsed_seconds": round(elapsed, 3),
        "accumulated_thinking_chars": counters["thinking_chars"],
        "accumulated_response_chars": counters["response_chars"],
        "gpu": gpu_snapshot(),
    }


def run_stream(prompt, think_value, experiment, run_number):
    context = multiprocessing.get_context("spawn")
    event_queue = context.Queue()
    process = context.Process(
        target=stream_worker,
        args=(event_queue, prompt, think_value),
    )
    chunks = []
    telemetry = []
    counters = {
        "thinking_chars": 0,
        "response_chars": 0,
        "chunk_count": 0,
        "http_status": None,
    }
    complete_events = []
    started = time.perf_counter()
    next_telemetry = 0
    process.start()

    while True:
        elapsed = time.perf_counter() - started
        drain_events(event_queue, chunks, counters, complete_events)

        if elapsed >= next_telemetry:
            sample = telemetry_sample(elapsed, counters)
            telemetry.append(sample)
            print(
                "STREAM_TELEMETRY "
                + json.dumps({
                    "experiment": experiment,
                    "run": run_number,
                    **sample,
                }, ensure_ascii=False),
                flush=True,
            )
            next_telemetry += TELEMETRY_INTERVAL_SECONDS

        if complete_events:
            process.join(5)
            drain_events(event_queue, chunks, counters, complete_events)
            result = complete_events[-1]
            break

        if elapsed >= HARD_STOP_SECONDS:
            process.terminate()
            process.join(5)
            drain_events(event_queue, chunks, counters, complete_events)
            result = {
                "type": "complete",
                "status": "hard_stop_over_60s",
                "http_status": counters["http_status"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "thinking_chars": counters["thinking_chars"],
                "response_chars": counters["response_chars"],
                "chunk_count": counters["chunk_count"],
                "json_valid": False,
                "error": "Streaming request exceeded 60-second hard stop",
            }
            break

        time.sleep(0.05)

    event_queue.close()
    result.pop("type", None)
    result.update({
        "experiment": experiment,
        "run": run_number,
        "think_parameter": think_value,
        "chunks": chunks,
        "telemetry": telemetry,
    })
    return result


def wait_until_gpu_quiet():
    samples = []
    consecutive_quiet = 0
    started = time.perf_counter()
    while time.perf_counter() - started < 30:
        gpu = gpu_snapshot()
        samples.append({"sampled_at": now_iso(), "gpu": gpu})
        utilization = gpu[0]["utilization_percent"] if gpu else 0
        consecutive_quiet = consecutive_quiet + 1 if utilization < 50 else 0
        if consecutive_quiet >= 2:
            break
        time.sleep(1)
    return samples


def percentile_95(values):
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * 0.95 + 0.999999999))
    return ordered[rank - 1]


def summarize(experiment, results):
    selected = [item for item in results if item["experiment"] == experiment]
    latencies = [item["elapsed_seconds"] for item in selected]
    return {
        "experiment": experiment,
        "attempted": len(selected),
        "completed": sum(item["status"] == "success" for item in selected),
        "over_60s": sum(
            item["status"] == "hard_stop_over_60s" for item in selected
        ),
        "average_latency_seconds": (
            round(statistics.mean(latencies), 3) if latencies else None
        ),
        "median_latency_seconds": (
            round(statistics.median(latencies), 3) if latencies else None
        ),
        "p95_latency_seconds": (
            round(percentile_95(latencies), 3) if latencies else None
        ),
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
        "thinking_chars": sum(item["thinking_chars"] for item in selected),
        "response_chars": sum(item["response_chars"] for item in selected),
        "json_valid": sum(item["json_valid"] for item in selected),
        "timeout_count": sum(item["status"] == "timeout" for item in selected),
        "error_count": sum(item["status"] == "error" for item in selected),
    }


def save_artifact(artifact):
    RESULTS_FILE.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    prompt = load_planner_prompt()
    artifact = {
        "model": MODEL,
        "planner_prompt_sha256": PLANNER_HASH,
        "planner_prompt_chars": len(prompt),
        "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "hard_stop_seconds": HARD_STOP_SECONDS,
        "telemetry_interval_seconds": TELEMETRY_INTERVAL_SECONDS,
        "started_at": now_iso(),
        "experiments": {
            "A": {"think_parameter": "omitted"},
            "B": {"think_parameter": False},
        },
        "results": [],
        "between_experiment_cooldown": [],
    }
    save_artifact(artifact)

    for experiment, think_value in (("A", None), ("B", False)):
        for run_number in range(1, RUNS_PER_EXPERIMENT + 1):
            result = run_stream(
                prompt,
                think_value,
                experiment,
                run_number,
            )
            artifact["results"].append(result)
            save_artifact(artifact)
            print(
                "STREAM_RESULT "
                + json.dumps({
                    key: value for key, value in result.items()
                    if key not in {"chunks", "telemetry"}
                }, ensure_ascii=False),
                flush=True,
            )
            if result["status"] == "hard_stop_over_60s":
                break

        if experiment == "A" and any(
            item["experiment"] == "A"
            and item["status"] == "hard_stop_over_60s"
            for item in artifact["results"]
        ):
            artifact["between_experiment_cooldown"] = wait_until_gpu_quiet()
            save_artifact(artifact)

    artifact["finished_at"] = now_iso()
    artifact["summaries"] = [
        summarize("A", artifact["results"]),
        summarize("B", artifact["results"]),
    ]
    save_artifact(artifact)
    print(
        "STREAM_SUMMARY "
        + json.dumps(artifact["summaries"], ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
