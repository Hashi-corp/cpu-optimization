"""
CPU Optimization Utilities for LLM Inference
=============================================
Baseline profiling, threading, and quantization helpers.
"""

import time
import psutil
import os
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ─── Benchmark Result Container ──────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    method:           str
    latency_ms:       float          # per-token
    throughput_tps:   float          # tokens/sec
    first_token_ms:   float          # time-to-first-token
    total_time_s:     float
    memory_mb:        float
    cpu_percent:      float
    tokens_generated: int
    perplexity:       Optional[float] = None
    extra:            Dict[str, Any]  = field(default_factory=dict)


# ─── System Info ─────────────────────────────────────────────────────────────

def get_system_info() -> Dict[str, Any]:
    cpu      = psutil.cpu_freq()
    mem      = psutil.virtual_memory()
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores":  psutil.cpu_count(logical=True),
        "cpu_freq_mhz":   round(cpu.current, 1) if cpu else "N/A",
        "total_ram_gb":   round(mem.total / 1e9, 2),
        "available_ram_gb": round(mem.available / 1e9, 2),
        "python_threads": threading.active_count(),
    }
    try:
        import torch
        cpu_info["torch_threads"] = torch.get_num_threads()
        cpu_info["torch_interop"] = torch.get_num_interop_threads()
        cpu_info["mkl_enabled"]   = torch.backends.mkl.is_available()
        cpu_info["onednn_enabled"]= torch.backends.mkldnn.is_available()
    except ImportError:
        pass
    return cpu_info


# ─── Memory Tracker ──────────────────────────────────────────────────────────

class MemoryTracker:
    """Lightweight RSS-based memory tracker."""

    def __init__(self):
        self._proc = psutil.Process(os.getpid())
        self._baseline = self._sample()

    def _sample(self) -> float:
        return self._proc.memory_info().rss / 1e6   # MB

    def current_mb(self) -> float:
        return self._sample()

    def delta_mb(self) -> float:
        return self._sample() - self._baseline

    def reset(self):
        self._baseline = self._sample()


# ─── Timer ───────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start: Optional[float] = None
        self.checkpoints: List[float] = []

    def start(self):
        self._start = time.perf_counter()
        self.checkpoints = [self._start]

    def checkpoint(self) -> float:
        t = time.perf_counter()
        self.checkpoints.append(t)
        return (t - self._start) * 1000   # ms

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start


# ─── CPU Thread Configuration ────────────────────────────────────────────────

def configure_cpu_threads(n_threads: int = 0):
    """
    Set PyTorch intra/inter-op threads for optimal CPU performance.
    n_threads=0  →  auto (physical cores).
    """
    import torch
    physical = psutil.cpu_count(logical=False) or 4
    n = n_threads if n_threads > 0 else physical
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))
    os.environ["OMP_NUM_THREADS"]       = str(n)
    os.environ["MKL_NUM_THREADS"]       = str(n)
    os.environ["OPENBLAS_NUM_THREADS"]  = str(n)
    return n


# ─── Perplexity Helper ───────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, text: str, max_length: int = 512) -> float:
    import torch, math
    encodings = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss    = outputs.loss
    return math.exp(loss.item())
