"""
ONNX Runtime and Quantization Optimizations for CPU Inference
=============================================================
• export_to_onnx        – export HF model → ONNX
• optimize_onnx_graph   – ORT graph-level optimization
• quantize_onnx_dynamic – INT8 dynamic quantization via ORT
• ORTInferenceSession   – unified generation wrapper around ONNX Runtime
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Export ──────────────────────────────────────────────────────────────────

def export_to_onnx(
    model_name_or_path: str,
    output_dir: str,
    task: str = "text-generation",
) -> str:
    """
    Export a HuggingFace model to ONNX using Optimum.
    Returns the path to the exported ONNX model directory.
    """
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    out = Path(output_dir) / "onnx_base"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting %s → %s", model_name_or_path, out)
    t0 = time.perf_counter()

    model = ORTModelForCausalLM.from_pretrained(
        model_name_or_path,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    elapsed = time.perf_counter() - t0
    logger.info("Export done in %.1fs", elapsed)
    return str(out)


def quantize_onnx_dynamic(
    onnx_model_dir: str,
    output_dir: str,
) -> str:
    """
    Apply ORT dynamic INT8 quantization to an already-exported ONNX model.
    """
    from optimum.onnxruntime import ORTModelForCausalLM
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    out = Path(output_dir) / "onnx_int8"
    out.mkdir(parents=True, exist_ok=True)

    qconfig  = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(onnx_model_dir)
    quantizer.quantize(save_dir=str(out), quantization_config=qconfig)

    logger.info("INT8 quantized model saved → %s", out)
    return str(out)


# ─── ORT Inference Session ────────────────────────────────────────────────────

class ORTInferenceSession:
    """
    Generation wrapper that uses ORT-backed ORTModelForCausalLM.
    Handles timing, token counting, and memory tracking.
    """

    def __init__(self, model_dir: str, label: str = "ORT"):
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
        import onnxruntime as ort
        import psutil

        self.label     = label
        self._proc     = psutil.Process(os.getpid())
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        n = psutil.cpu_count(logical=False) or 4
        sess_opts.intra_op_num_threads = n
        sess_opts.inter_op_num_threads = max(1, n // 2)

        self.model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            session_options=sess_opts,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> Dict:
        mem_before = self._proc.memory_info().rss / 1e6
        t0         = time.perf_counter()

        inputs  = self.tokenizer(prompt, return_tensors="pt")
        t_first = None

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        t_total = time.perf_counter() - t0
        mem_after = self._proc.memory_info().rss / 1e6

        n_new = output_ids.shape[1] - inputs["input_ids"].shape[1]
        text  = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return {
            "text":             text,
            "tokens_generated": n_new,
            "total_time_s":     t_total,
            "latency_ms":       t_total * 1000 / max(n_new, 1),
            "throughput_tps":   n_new / max(t_total, 1e-9),
            "memory_mb":        mem_after - mem_before,
        }


# ─── PyTorch Dynamic Quantization (no ONNX) ───────────────────────────────────

def apply_pytorch_dynamic_quantization(model):
    """
    Apply PyTorch's built-in dynamic INT8 quantization to Linear layers.
    Returns a new quantized model (CPU only).
    """
    import torch

    quantized = torch.quantization.quantize_dynamic(
        model.cpu(),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized
