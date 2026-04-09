# CPU Optimization Project

Benchmarking and evaluation toolkit for CPU-focused language model optimization.

This repository compares practical strategies for improving inference efficiency on CPU hardware. It includes the original notebook-based benchmark suite, extracted helper modules, semantic caching experiments, ONNX and OpenVINO paths, and a separate issue-by-issue feature runner for the remaining project work.

## What This Project Covers

The main benchmark suite evaluates:

- Baseline HuggingFace generation
- Intel Extension for PyTorch (IPEX)
- INT8 dynamic quantization
- BF16 autocast on CPU
- KV-cache based generation
- Semantic caching for repeated or similar prompts
- ONNX Runtime export and inference
- Final comparison dashboards and strategy summaries

The issue runner extends the project with:

- OpenVINO inference
- Speculative decoding
- CPU thread tuning and multithreading
- CPU and memory profiling
- Output diversity controls for generation
- ONNX Runtime dashboard integration
- Concurrent request sampling

## Repository Layout

- `llm_cpu_optimization.ipynb` - full benchmark notebook for the core optimization suite
- `cpu_utils.py` - shared timing, memory, thread, and profiling helpers
- `kv_cache.py` - manual KV cache, prefix cache, and semantic cache implementations
- `onnx_optimizer.py` - ONNX export, quantization, and runtime inference utilities
- `semantic_cache_extension.py` - semantic cache benchmark and reporting workflow
- `cpu_optimization_feature_benchmark.py` - issue-by-issue benchmark runner for features #3 through #9

## Setup

Use the included virtual environment and install the project dependencies before running benchmarks.

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Benchmarks

Run the notebook top to bottom for the original full-suite evaluation.

For the issue-specific feature runner, execute one issue at a time from the `cpu-optimization` directory:

```powershell
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 3
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 4
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 5
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 6
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 7
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 8
.\.venv\Scripts\python .\cpu_optimization_feature_benchmark.py --issue 9
```

### One-Command Judge Demo (Fast)

Use this command to generate all core benchmark outputs in one run and archive each run under a new numbered folder.

```powershell
.\.venv\Scripts\python .\run_all_outputs_demo.py
```

What it does:

- Runs both `semantic_cache_extension.py` and `cpu_optimization_feature_benchmark.py`
- Uses fast demo mode by default for reliable sub-minute output generation
- Creates `outputs/outputs(1)`, `outputs/outputs(2)`, ... on every run
- Copies generated CSV and PNG artifacts into that run folder
- Generates `final_demo_summary.txt` and `run_manifest.json` in the run folder

If you want the full native path instead of fast demo mode:

```powershell
.\.venv\Scripts\python .\run_all_outputs_demo.py --full
```

## Outputs

Core benchmark outputs:

- `semantic_cache_summary.csv`
- `semantic_cache_trace.csv`
- `semantic_cache_latency_comparison.png`
- `semantic_cache_hit_breakdown.png`
- `semantic_cache_query_savings.png`

Issue-specific outputs:

- `feature_benchmark_summary_issue_3.csv` to `feature_benchmark_summary_issue_9.csv`
- `feature_benchmark_trace_issue_3.csv` to `feature_benchmark_trace_issue_9.csv`
- `feature_benchmark_dashboard_issue_3.png` to `feature_benchmark_dashboard_issue_9.png`

## Notes

- The feature runner is designed to use native model paths when the environment supports them.
- The saved outputs are intended for comparison, reporting, and presentation.