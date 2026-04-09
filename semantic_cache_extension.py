from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import statistics
import time
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
SUMMARY_CSV = ROOT / "semantic_cache_summary.csv"
TRACE_CSV = ROOT / "semantic_cache_trace.csv"
LATENCY_PNG = ROOT / "semantic_cache_latency_comparison.png"
BREAKDOWN_PNG = ROOT / "semantic_cache_hit_breakdown.png"
SEQUENCE_PNG = ROOT / "semantic_cache_query_savings.png"

MODEL_ID = "gpt2"
FAST_DEMO = os.getenv("CPU_OPT_FAST_DEMO", "0") == "1"
MAX_NEW_TOKENS = 18 if FAST_DEMO else 36
WARMUP_RUNS = 0 if FAST_DEMO else 1
SEMANTIC_THRESHOLD = 0.35
WORKLOAD_LIMIT = 8 if FAST_DEMO else None

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


@dataclass
class QueryItem:
    group_id: str
    variant: str
    prompt: str


class InferenceBackend:
    mode: str = "unknown"

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, use_cache: bool = True) -> Dict[str, Any]:
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

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, use_cache: bool = True) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])

        for _ in range(WARMUP_RUNS):
            with self.torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )

        gc.collect()
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1e6
        t0 = time.perf_counter()
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        mem_after = proc.memory_info().rss / 1e6

        tokens_generated = int(output_ids.shape[1] - input_len)
        text = self.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )
        return {
            "response": text,
            "latency_ms": elapsed_ms,
            "tokens_generated": max(tokens_generated, 1),
            "throughput_qps": 1000.0 / max(elapsed_ms, 1e-6),
            "memory_mb": max(mem_after - mem_before, 0.0),
        }


class SimulatedBackend(InferenceBackend):
    mode = "simulated"

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, use_cache: bool = True) -> Dict[str, Any]:
        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        prompt_len = len(prompt.split())
        base_ms = 110 + prompt_len * 4.2 + rng.uniform(0, 18)
        if not use_cache:
            base_ms *= 1.22
        tokens_generated = int(max_new_tokens * 0.55 + (seed % 7))
        return {
            "response": f"Simulated response for: {prompt[:48]}",
            "latency_ms": float(base_ms),
            "tokens_generated": max(tokens_generated, 1),
            "throughput_qps": 1000.0 / max(base_ms, 1e-6),
            "memory_mb": float(18 + rng.uniform(0, 8)),
        }


class LexicalSemanticCache:
    def __init__(self, threshold: float = SEMANTIC_THRESHOLD):
        self.threshold = threshold
        self._entries: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _score(self, left: str, right: str) -> float:
        a = set(self._normalize(left))
        b = set(self._normalize(right))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def lookup(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self._entries:
            return None
        best = max(self._entries, key=lambda item: self._score(prompt, item["query"]))
        score = self._score(prompt, best["query"])
        if score >= self.threshold:
            return {**best, "score": score}
        return None

    def store(self, query: str, response: str, latency_ms: float) -> None:
        self._entries.append({
            "query": query,
            "response": response,
            "latency_ms": latency_ms,
        })


def load_workload(path: Path = WORKLOAD_PATH) -> List[QueryItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: List[QueryItem] = []
    for group in raw["prompt_groups"]:
        canonical = group["canonical"]
        queries.append(QueryItem(group_id=group["id"], variant="canonical", prompt=canonical))
        for repeat_idx in range(group.get("exact_repeats", 0)):
            queries.append(QueryItem(group_id=group["id"], variant=f"exact_repeat_{repeat_idx + 1}", prompt=canonical))
        for para_idx, paraphrase in enumerate(group.get("paraphrases", []), start=1):
            queries.append(QueryItem(group_id=group["id"], variant=f"paraphrase_{para_idx}", prompt=paraphrase))
    return queries[:WORKLOAD_LIMIT] if WORKLOAD_LIMIT else queries


def build_backend() -> Tuple[InferenceBackend, Optional[str]]:
    if FAST_DEMO:
        return SimulatedBackend(), "Forced simulated backend for fast demo mode (CPU_OPT_FAST_DEMO=1)"
    try:
        return HuggingFaceBackend(MODEL_ID), None
    except Exception as exc:  # pragma: no cover - fallback is intentional for quick verification
        return SimulatedBackend(), str(exc)


def build_semantic_cache() -> Tuple[Any, str]:
    try:
        from kv_cache import SemanticCache

        cache = SemanticCache(similarity_threshold=0.80, max_entries=128)
        return cache, "embedding"
    except Exception:
        return LexicalSemanticCache(), "lexical"


def exact_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def run_baseline(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    rows = []
    for index, query in enumerate(queries, start=1):
        result = backend.generate(query.prompt)
        rows.append({
            "strategy": "Baseline",
            "query_index": index,
            "group_id": query.group_id,
            "variant": query.variant,
            "cache_event": "miss",
            "latency_ms": result["latency_ms"],
            "tokens_generated": result["tokens_generated"],
            "memory_mb": result["memory_mb"],
            "response_preview": result["response"][:90],
        })
    return pd.DataFrame(rows)


def run_exact_cache(backend: InferenceBackend, queries: List[QueryItem]) -> pd.DataFrame:
    cache: Dict[str, Dict[str, Any]] = {}
    rows = []
    for index, query in enumerate(queries, start=1):
        t0 = time.perf_counter()
        key = exact_key(query.prompt)
        hit = cache.get(key)
        lookup_ms = (time.perf_counter() - t0) * 1000
        if hit is not None:
            rows.append({
                "strategy": "Exact Cache",
                "query_index": index,
                "group_id": query.group_id,
                "variant": query.variant,
                "cache_event": "exact_hit",
                "latency_ms": max(lookup_ms, 0.02),
                "tokens_generated": 0,
                "memory_mb": 0.0,
                "response_preview": hit["response"][:90],
            })
            continue

        result = backend.generate(query.prompt)
        cache[key] = {"response": result["response"], "latency_ms": result["latency_ms"]}
        rows.append({
            "strategy": "Exact Cache",
            "query_index": index,
            "group_id": query.group_id,
            "variant": query.variant,
            "cache_event": "miss",
            "latency_ms": result["latency_ms"],
            "tokens_generated": result["tokens_generated"],
            "memory_mb": result["memory_mb"],
            "response_preview": result["response"][:90],
        })
    return pd.DataFrame(rows)


def run_semantic_cache(backend: InferenceBackend, queries: List[QueryItem]) -> Tuple[pd.DataFrame, str]:
    exact_cache: Dict[str, Dict[str, Any]] = {}
    semantic_cache, semantic_mode = build_semantic_cache()
    rows = []

    for index, query in enumerate(queries, start=1):
        key = exact_key(query.prompt)
        exact_hit = exact_cache.get(key)
        if exact_hit is not None:
            rows.append({
                "strategy": "Semantic Cache",
                "query_index": index,
                "group_id": query.group_id,
                "variant": query.variant,
                "cache_event": "exact_hit",
                "latency_ms": 0.02,
                "tokens_generated": 0,
                "memory_mb": 0.0,
                "response_preview": exact_hit["response"][:90],
            })
            continue

        semantic_t0 = time.perf_counter()
        semantic_hit = semantic_cache.lookup(query.prompt)
        semantic_lookup_ms = (time.perf_counter() - semantic_t0) * 1000
        if semantic_hit is not None:
            response = semantic_hit.response if hasattr(semantic_hit, "response") else semantic_hit["response"]
            rows.append({
                "strategy": "Semantic Cache",
                "query_index": index,
                "group_id": query.group_id,
                "variant": query.variant,
                "cache_event": "semantic_hit",
                "latency_ms": max(semantic_lookup_ms, 0.25),
                "tokens_generated": 0,
                "memory_mb": 0.0,
                "response_preview": response[:90],
            })
            exact_cache[key] = {"response": response, "latency_ms": max(semantic_lookup_ms, 0.25)}
            continue

        result = backend.generate(query.prompt)
        exact_cache[key] = {"response": result["response"], "latency_ms": result["latency_ms"]}
        semantic_cache.store(query.prompt, result["response"], result["latency_ms"])
        rows.append({
            "strategy": "Semantic Cache",
            "query_index": index,
            "group_id": query.group_id,
            "variant": query.variant,
            "cache_event": "miss",
            "latency_ms": result["latency_ms"],
            "tokens_generated": result["tokens_generated"],
            "memory_mb": result["memory_mb"],
            "response_preview": result["response"][:90],
        })

    return pd.DataFrame(rows), semantic_mode


def summarize(trace_df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    baseline_mean = float(trace_df.loc[trace_df["strategy"] == "Baseline", "latency_ms"].mean())
    for strategy, group in trace_df.groupby("strategy", sort=False):
        avg_latency = float(group["latency_ms"].mean())
        p95_latency = float(group["latency_ms"].quantile(0.95))
        hit_rate = float((group["cache_event"] != "miss").mean())
        summary.append({
            "strategy": strategy,
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "throughput_qps": round(1000.0 / max(avg_latency, 1e-6), 2),
            "hit_rate_pct": round(hit_rate * 100.0, 1),
            "misses": int((group["cache_event"] == "miss").sum()),
            "exact_hits": int((group["cache_event"] == "exact_hit").sum()),
            "semantic_hits": int((group["cache_event"] == "semantic_hit").sum()),
            "speedup_vs_baseline": round(baseline_mean / max(avg_latency, 1e-6), 2),
            "avg_memory_mb": round(float(group["memory_mb"].mean()), 2),
        })
    return pd.DataFrame(summary)


def save_latency_plot(summary_df: pd.DataFrame) -> None:
    order = summary_df.sort_values("avg_latency_ms", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(order["strategy"], order["avg_latency_ms"], color=["#5B8FF9", "#61DDAA", "#65789B"])
    baseline_value = float(summary_df.loc[summary_df["strategy"] == "Baseline", "avg_latency_ms"].iloc[0])
    ax.axvline(baseline_value, color="gray", linestyle="--", linewidth=1.2, label="Existing baseline")
    for bar, speedup in zip(bars, order["speedup_vs_baseline"]):
        ax.text(bar.get_width() + baseline_value * 0.015, bar.get_y() + bar.get_height() / 2, f"×{speedup:.2f}", va="center")
    ax.set_title("Semantic Cache Extension — Latency vs Existing Baseline")
    ax.set_xlabel("Average query latency (ms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(LATENCY_PNG, bbox_inches="tight")
    plt.close(fig)


def save_breakdown_plot(summary_df: pd.DataFrame) -> None:
    relevant = summary_df[summary_df["strategy"].isin(["Exact Cache", "Semantic Cache"])].copy()
    x = np.arange(len(relevant))
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(x, relevant["misses"], label="Misses", color="#5B8FF9")
    ax1.bar(x, relevant["exact_hits"], bottom=relevant["misses"], label="Exact hits", color="#61DDAA")
    ax1.bar(x, relevant["semantic_hits"], bottom=relevant["misses"] + relevant["exact_hits"], label="Semantic hits", color="#F6BD16")
    ax1.set_xticks(x)
    ax1.set_xticklabels(relevant["strategy"])
    ax1.set_ylabel("Query count")
    ax1.set_title("Hit Breakdown for New Cache Strategies")

    ax2 = ax1.twinx()
    ax2.plot(x, relevant["avg_latency_ms"], color="#E8684A", marker="o", linewidth=2, label="Avg latency")
    ax2.set_ylabel("Average latency (ms)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(BREAKDOWN_PNG, bbox_inches="tight")
    plt.close(fig)


def save_sequence_plot(trace_df: pd.DataFrame) -> None:
    baseline = trace_df[trace_df["strategy"] == "Baseline"][ ["query_index", "latency_ms"] ].rename(columns={"latency_ms": "baseline_latency_ms"})
    semantic = trace_df[trace_df["strategy"] == "Semantic Cache"][ ["query_index", "latency_ms", "cache_event"] ].rename(columns={"latency_ms": "semantic_latency_ms"})
    merged = baseline.merge(semantic, on="query_index", how="inner")
    merged["saved_ms"] = merged["baseline_latency_ms"] - merged["semantic_latency_ms"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(merged["query_index"], merged["baseline_latency_ms"], color="#9CA3AF", marker="o", linewidth=1.8, label="Existing baseline")
    ax.plot(merged["query_index"], merged["semantic_latency_ms"], color="#61DDAA", marker="o", linewidth=2.2, label="Semantic cache")
    miss_mask = merged["cache_event"] == "miss"
    ax.scatter(merged.loc[miss_mask, "query_index"], merged.loc[miss_mask, "semantic_latency_ms"], color="#E8684A", label="Semantic cache miss", zorder=4)
    ax.set_title("Per-Query Savings After Adding Semantic Cache")
    ax.set_xlabel("Query sequence")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SEQUENCE_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    queries = load_workload()
    backend, backend_error = build_backend()

    baseline_df = run_baseline(backend, queries)
    exact_df = run_exact_cache(backend, queries)
    semantic_df, semantic_mode = run_semantic_cache(backend, queries)

    trace_df = pd.concat([baseline_df, exact_df, semantic_df], ignore_index=True)
    summary_df = summarize(trace_df)
    summary_df.insert(1, "backend_mode", backend.mode)
    summary_df.insert(2, "semantic_mode", semantic_mode)

    trace_df.to_csv(TRACE_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    save_latency_plot(summary_df)
    save_breakdown_plot(summary_df)
    save_sequence_plot(trace_df)

    print("Semantic cache extension benchmark complete.")
    print(f"Backend mode      : {backend.mode}")
    print(f"Semantic mode     : {semantic_mode}")
    if backend_error:
        print(f"Backend fallback  : {backend_error}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved -> {SUMMARY_CSV.name}")
    print(f"Saved -> {TRACE_CSV.name}")
    print(f"Saved -> {LATENCY_PNG.name}")
    print(f"Saved -> {BREAKDOWN_PNG.name}")
    print(f"Saved -> {SEQUENCE_PNG.name}")


if __name__ == "__main__":
    main()
