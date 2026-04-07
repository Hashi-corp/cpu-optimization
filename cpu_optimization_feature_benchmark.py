from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

from cpu_utils import configure_cpu_threads

ROOT = Path(__file__).resolve().parent
WORKLOAD_PATH = ROOT / "semantic_cache_workload.json"
SUMMARY_CSV = ROOT / "feature_benchmark_summary.csv"
TRACE_CSV = ROOT / "feature_benchmark_trace.csv"
DASHBOARD_PNG = ROOT / "feature_benchmark_dashboard.png"

MODEL_ID = "gpt2"
ASSISTANT_MODEL_ID = "sshleifer/tiny-gpt2"
MAX_NEW_TOKENS = 36
CONCURRENT_WORKERS = 4

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


@dataclass
class QueryItem:
    group_id: str
    variant: str
    prompt: str


def load_workload(path: Path = WORKLOAD_PATH) -> List[QueryItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: List[QueryItem] = []
    for group in raw["prompt_groups"]:
        canonical = group["canonical"]
        rows.append(QueryItem(group_id=group["id"], variant="canonical", prompt=canonical))
        for idx in range(group.get("exact_repeats", 0)):
            rows.append(QueryItem(group_id=group["id"], variant=f"exact_repeat_{idx + 1}", prompt=canonical))
        for idx, paraphrase in enumerate(group.get("paraphrases", []), start=1):
            rows.append(QueryItem(group_id=group["id"], variant=f"paraphrase_{idx}", prompt=paraphrase))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU optimization features benchmark for issues #3-#9.",
    )
    parser.add_argument(
        "--issue",
        action="append",
        type=int,
        choices=[3, 4, 5, 6, 7, 8, 9],
        help="Run only a specific issue (repeat flag for multiple issues).",
    )
    parser.add_argument(
        "--list-issues",
        action="store_true",
        help="List supported issue numbers and exit.",
    )
    return parser.parse_args()


def resolve_output_paths(selected_issues: List[int]) -> Tuple[Path, Path, Path]:
    if selected_issues == [3, 4, 5, 6, 7, 8, 9]:
        return SUMMARY_CSV, TRACE_CSV, DASHBOARD_PNG
    suffix = "_".join(str(issue) for issue in selected_issues)
    return (
        ROOT / f"feature_benchmark_summary_issue_{suffix}.csv",
        ROOT / f"feature_benchmark_trace_issue_{suffix}.csv",
        ROOT / f"feature_benchmark_dashboard_issue_{suffix}.png",
    )


class InferenceBackend:
    mode: str = "unknown"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        sampling: Optional[Dict[str, Any]] = None,
        speculative: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class HuggingFaceBackend(InferenceBackend):
    mode = "huggingface"

    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        configure_cpu_threads(0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.eval()

        self.assistant_model = None
        try:
            self.assistant_model = AutoModelForCausalLM.from_pretrained(ASSISTANT_MODEL_ID, torch_dtype=torch.float32)
            self.assistant_model.eval()
        except Exception:
            self.assistant_model = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        sampling: Optional[Dict[str, Any]] = None,
        speculative: bool = False,
    ) -> Dict[str, Any]:
        sampling = sampling or {}
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        gen_kwargs.update(sampling)
        if speculative and self.assistant_model is not None:
            gen_kwargs["assistant_model"] = self.assistant_model
            gen_kwargs["num_assistant_tokens"] = 8

        proc = psutil.Process()
        gc.collect()
        mem_before = proc.memory_info().rss / 1e6
        t0 = time.perf_counter()
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        mem_after = proc.memory_info().rss / 1e6

        tokens_generated = int(output_ids.shape[1] - input_len)
        text = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return {
            "response": text,
            "latency_ms": elapsed_ms,
            "tokens_generated": max(tokens_generated, 1),
            "memory_mb": max(mem_after - mem_before, 0.0),
        }


class SimulatedBackend(InferenceBackend):
    mode = "simulated"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        sampling: Optional[Dict[str, Any]] = None,
        speculative: bool = False,
    ) -> Dict[str, Any]:
        sampling = sampling or {}
        seed = abs(hash((prompt, json.dumps(sampling, sort_keys=True), speculative))) % (2**32)
        rng = np.random.default_rng(seed)
        prompt_len = len(prompt.split())
        base_ms = 105 + prompt_len * 4.1 + rng.uniform(0, 14)

        if sampling.get("do_sample"):
            base_ms *= 1.06
        if speculative:
            base_ms *= 0.8

        tokens_generated = int(max_new_tokens * 0.55 + (seed % 9))
        vocabulary = ["cpu", "cache", "token", "latency", "thread", "model", "batch", "decode", "serve", "quant"]
        response_tokens = [vocabulary[int(rng.integers(0, len(vocabulary)))] for _ in range(max(tokens_generated, 10))]
        return {
            "response": " ".join(response_tokens),
            "latency_ms": float(base_ms),
            "tokens_generated": max(tokens_generated, 1),
            "memory_mb": float(14 + rng.uniform(0, 7)),
        }


def build_backend() -> Tuple[InferenceBackend, Optional[str]]:
    try:
        return HuggingFaceBackend(MODEL_ID), None
    except Exception as exc:
        raise RuntimeError(f"Native backend initialization failed: {exc}") from exc


def distinct1(text: str) -> float:
    tokens = [token for token in text.lower().split() if token.strip()]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def profile_single_call(backend: InferenceBackend, prompt: str) -> Dict[str, float]:
    proc = psutil.Process()
    _ = psutil.cpu_percent(interval=None)
    result = backend.generate(prompt)
    cpu_now = psutil.cpu_percent(interval=0.05)
    mem_rss = proc.memory_info().rss / 1e6
    return {
        "latency_ms": result["latency_ms"],
        "cpu_percent": cpu_now,
        "memory_mb": mem_rss,
        "distinct1": distinct1(result["response"]),
    }


def run_threading_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cores = max(psutil.cpu_count(logical=False) or 2, 2)
    thread_options = sorted(set([1, max(2, cores // 2), cores]))
    for n_threads in thread_options:
        try:
            configure_cpu_threads(n_threads)
        except Exception:
            pass

        latencies = [backend.generate(q.prompt)["latency_ms"] for q in queries[:8]]
        rows.append({
            "issue": "#5 multithreading implementation",
            "strategy": f"Threads={n_threads}",
            "latency_ms": statistics.mean(latencies),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "throughput_qps": 1000.0 / max(statistics.mean(latencies), 1e-6),
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "memory_mb": psutil.Process().memory_info().rss / 1e6,
            "distinct1": np.nan,
            "notes": "Thread tuning with torch intra/inter-op settings",
        })
    return pd.DataFrame(rows)


def run_profiling_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = [profile_single_call(backend, q.prompt) for q in queries[:6]]
    rows.append({
        "issue": "#6 memory and cpu profiling",
        "strategy": "Process CPU/RSS profiling",
        "latency_ms": statistics.mean([m["latency_ms"] for m in metrics]),
        "p95_latency_ms": float(np.percentile([m["latency_ms"] for m in metrics], 95)),
        "throughput_qps": 1000.0 / max(statistics.mean([m["latency_ms"] for m in metrics]), 1e-6),
        "cpu_percent": statistics.mean([m["cpu_percent"] for m in metrics]),
        "memory_mb": statistics.mean([m["memory_mb"] for m in metrics]),
        "distinct1": statistics.mean([m["distinct1"] for m in metrics]),
        "notes": "Per-query CPU% and RSS memory collection",
    })
    return pd.DataFrame(rows)


def run_diversity_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    presets = [
        ("Greedy", {"do_sample": False}),
        ("Diverse", {"do_sample": True, "temperature": 0.9, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.08}),
    ]
    for name, sampling in presets:
        samples = [backend.generate(q.prompt, sampling=sampling) for q in queries[:6]]
        rows.append({
            "issue": "#7 generation layer for improved output diversity",
            "strategy": f"{name} decoding",
            "latency_ms": statistics.mean([s["latency_ms"] for s in samples]),
            "p95_latency_ms": float(np.percentile([s["latency_ms"] for s in samples], 95)),
            "throughput_qps": 1000.0 / max(statistics.mean([s["latency_ms"] for s in samples]), 1e-6),
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "memory_mb": statistics.mean([s["memory_mb"] for s in samples]),
            "distinct1": statistics.mean([distinct1(s["response"]) for s in samples]),
            "notes": "Adds temperature/top-p/top-k/repetition-penalty layer",
        })
    return pd.DataFrame(rows)


def run_speculative_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = [backend.generate(q.prompt, speculative=False) for q in queries[:6]]
    spec = [backend.generate(q.prompt, speculative=True) for q in queries[:6]]
    rows.append({
        "issue": "#4 speculative decoding",
        "strategy": "Standard decode",
        "latency_ms": statistics.mean([s["latency_ms"] for s in base]),
        "p95_latency_ms": float(np.percentile([s["latency_ms"] for s in base], 95)),
        "throughput_qps": 1000.0 / max(statistics.mean([s["latency_ms"] for s in base]), 1e-6),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "memory_mb": statistics.mean([s["memory_mb"] for s in base]),
        "distinct1": statistics.mean([distinct1(s["response"]) for s in base]),
        "notes": "Reference path",
    })
    rows.append({
        "issue": "#4 speculative decoding",
        "strategy": "Speculative decode",
        "latency_ms": statistics.mean([s["latency_ms"] for s in spec]),
        "p95_latency_ms": float(np.percentile([s["latency_ms"] for s in spec], 95)),
        "throughput_qps": 1000.0 / max(statistics.mean([s["latency_ms"] for s in spec]), 1e-6),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "memory_mb": statistics.mean([s["memory_mb"] for s in spec]),
        "distinct1": statistics.mean([distinct1(s["response"]) for s in spec]),
        "notes": "Assistant-model speculative generation",
    })
    return pd.DataFrame(rows)


def run_openvino_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    mode = "native"
    latencies: List[float] = []
    memories: List[float] = []
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ov_model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True)

    proc = psutil.Process()
    for q in queries[:6]:
        inputs = tokenizer(q.prompt, return_tensors="pt")
        mem_before = proc.memory_info().rss / 1e6
        t0 = time.perf_counter()
        output_ids = ov_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        mem_after = proc.memory_info().rss / 1e6
        _ = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        latencies.append(elapsed_ms)
        memories.append(max(mem_after - mem_before, 0.0))

    rows.append({
        "issue": "#3 openvino implementation feature",
        "strategy": "OpenVINO backend",
        "latency_ms": statistics.mean(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_qps": 1000.0 / max(statistics.mean(latencies), 1e-6),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "memory_mb": statistics.mean(memories),
        "distinct1": np.nan,
        "notes": f"OpenVINO path ({mode})",
    })
    return pd.DataFrame(rows)


def run_onnx_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    memories: List[float] = []
    onnx_mode = "native"
    from onnx_optimizer import ORTInferenceSession, export_to_onnx, quantize_onnx_dynamic

    candidate_dirs = [ROOT / "onnx_int8", ROOT / "onnx_base"]
    model_dir = next((path for path in candidate_dirs if path.exists()), None)
    if model_dir is None:
        onnx_base = export_to_onnx(MODEL_ID, str(ROOT))
        try:
            onnx_int8 = quantize_onnx_dynamic(onnx_base, str(ROOT))
            model_dir = Path(onnx_int8)
        except Exception:
            model_dir = Path(onnx_base)

    session = ORTInferenceSession(str(model_dir), label="ORT")
    for q in queries[:6]:
        result = session.generate(q.prompt, max_new_tokens=MAX_NEW_TOKENS)
        latencies.append(result["latency_ms"])
        memories.append(result["memory_mb"])

    rows.append({
        "issue": "#8 onnx imp on the main dashboard and summary.df",
        "strategy": "ONNX Runtime",
        "latency_ms": statistics.mean(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_qps": 1000.0 / max(statistics.mean(latencies), 1e-6),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "memory_mb": statistics.mean(memories),
        "distinct1": np.nan,
        "notes": f"ONNX included in dashboard ({onnx_mode})",
    })
    return pd.DataFrame(rows)


def run_concurrent_sampling_benchmark(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prompts = [q.prompt for q in queries[:12]]
    t0 = time.perf_counter()
    latencies: List[float] = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
        futures = [pool.submit(backend.generate, prompt) for prompt in prompts]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result["latency_ms"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    rows.append({
        "issue": "#9 concurrent cpu sampling",
        "strategy": f"Concurrent sampling (workers={CONCURRENT_WORKERS})",
        "latency_ms": statistics.mean(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_qps": len(prompts) * 1000.0 / max(elapsed_ms, 1e-6),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "memory_mb": psutil.Process().memory_info().rss / 1e6,
        "distinct1": np.nan,
        "notes": "ThreadPool-based parallel request execution",
    })
    return pd.DataFrame(rows)


def build_dashboard(summary_df: pd.DataFrame, dashboard_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("CPU Optimization Feature Dashboard (#3-#9)", fontsize=14, fontweight="bold")

    latency_df = summary_df.sort_values("latency_ms", ascending=True)
    sns.barplot(data=latency_df, x="latency_ms", y="strategy", hue="issue", dodge=False, ax=axes[0, 0])
    axes[0, 0].set_title("Average Latency by Strategy")
    axes[0, 0].set_xlabel("Latency (ms)")
    axes[0, 0].set_ylabel("")
    axes[0, 0].legend_.remove()

    throughput_df = summary_df.sort_values("throughput_qps", ascending=False)
    sns.barplot(data=throughput_df, x="throughput_qps", y="strategy", hue="issue", dodge=False, ax=axes[0, 1])
    axes[0, 1].set_title("Throughput by Strategy")
    axes[0, 1].set_xlabel("QPS")
    axes[0, 1].set_ylabel("")
    axes[0, 1].legend_.remove()

    axes[1, 0].scatter(summary_df["cpu_percent"], summary_df["memory_mb"], c="#2E86AB", s=70)
    for _, row in summary_df.iterrows():
        axes[1, 0].annotate(row["strategy"], (row["cpu_percent"], row["memory_mb"]), fontsize=8, alpha=0.85)
    axes[1, 0].set_title("CPU vs Memory Footprint")
    axes[1, 0].set_xlabel("CPU %")
    axes[1, 0].set_ylabel("Memory (MB)")

    diversity_df = summary_df.dropna(subset=["distinct1"]).copy()
    if diversity_df.empty:
        axes[1, 1].text(0.5, 0.5, "No diversity metrics available", ha="center", va="center")
        axes[1, 1].set_axis_off()
    else:
        sns.barplot(data=diversity_df, x="distinct1", y="strategy", hue="issue", dodge=False, ax=axes[1, 1])
        axes[1, 1].set_title("Output Diversity (Distinct-1)")
        axes[1, 1].set_xlabel("Distinct-1")
        axes[1, 1].set_ylabel("")
        axes[1, 1].legend_.remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    issue_runners = {
        3: run_openvino_benchmark,
        4: run_speculative_benchmark,
        5: run_threading_benchmark,
        6: run_profiling_benchmark,
        7: run_diversity_benchmark,
        8: run_onnx_benchmark,
        9: run_concurrent_sampling_benchmark,
    }

    if args.list_issues:
        print("Available issues: 3, 4, 5, 6, 7, 8, 9")
        print("Example: python cpu_optimization_feature_benchmark.py --issue 3")
        print("Example: python cpu_optimization_feature_benchmark.py --issue 4 --issue 7")
        return

    selected_issues = sorted(set(args.issue or issue_runners.keys()))
    summary_path, trace_path, dashboard_path = resolve_output_paths(selected_issues)

    queries = load_workload()
    backend, backend_error = build_backend()

    frames = [issue_runners[issue](backend, queries) for issue in selected_issues]
    summary_df = pd.concat(frames, ignore_index=True)
    summary_df.insert(1, "backend_mode", backend.mode)
    summary_df.insert(2, "timestamp", pd.Timestamp.utcnow().isoformat())

    trace_rows = []
    for q in queries[:12]:
        result = backend.generate(q.prompt)
        trace_rows.append({
            "issues_executed": ",".join(str(issue) for issue in selected_issues),
            "group_id": q.group_id,
            "variant": q.variant,
            "latency_ms": result["latency_ms"],
            "tokens_generated": result["tokens_generated"],
            "response_preview": result["response"][:90],
        })
    trace_df = pd.DataFrame(trace_rows)

    summary_df.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    build_dashboard(summary_df, dashboard_path)

    print("CPU optimization feature benchmark complete.")
    print(f"Backend mode      : {backend.mode}")
    print(f"Issues executed   : {selected_issues}")
    if backend_error:
        print(f"Backend fallback  : {backend_error}")
    print()
    print(summary_df[["issue", "strategy", "latency_ms", "throughput_qps", "cpu_percent", "memory_mb", "distinct1"]].to_string(index=False))
    print()
    print(f"Saved -> {summary_path.name}")
    print(f"Saved -> {trace_path.name}")
    print(f"Saved -> {dashboard_path.name}")


if __name__ == "__main__":
    main()
